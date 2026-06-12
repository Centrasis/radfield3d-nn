from lightning.pytorch.callbacks import Callback
from lightning.pytorch.trainer import Trainer
from loggers.logger import LoggerBase
from radfield3dnn.models.base import BaseNeuralRadFieldModel
import torch
from radfield3dnn.rftypes import RadiationField, TrainingInputData, RadiationFieldChannel, AirKermaField
from radfield3dnn.metrics.airkerma_accuracy import AirkermaAccuracy
from radfield3dnn.preprocessing.normalizations.base import Normalizer
from radfield3dnn.rfhelpers import InferenceHelper
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from radfield3dnn.datasets.channel_join import ChannelsJoin
from radfield3dnn.metrics import MetricBase
import warnings


class MetricsPlotter(Callback):
    def __init__(self, metrics: dict[str, MetricBase], spectra_bins: int = 32, voxel_resolution: tuple[int, int, int] = (50, 50, 50)):
        self.metrics = metrics
        self.voxel_resolution = voxel_resolution
        self.spectra_bins = spectra_bins
        for k, v in self.metrics.items():
            self.metrics[k] = v.eval()
        self.channels_join = ChannelsJoin().eval()

        # Per-metric sum AND count: a metric that can't score *some* batches must not
        # corrupt the others' mean, and a metric that scores *no* batch must surface as
        # NaN (visibly failing) rather than silently vanishing from the dashboard.
        self.metrics_accumulator = {name: torch.tensor(0.0) for name in self.metrics.keys()}
        self.metrics_count = {name: 0 for name in self.metrics.keys()}
        self._warned_none = set()

    def _reset(self, device):
        self.channels_join = self.channels_join.to(device)
        for name, metric in self.metrics.items():
            self.metrics[name] = metric.to(device)
            self.metrics_accumulator[name] = torch.tensor(0.0, device=device)
            self.metrics_count[name] = 0

    def on_validation_epoch_start(self, trainer, pl_module: BaseNeuralRadFieldModel):
        self._reset(pl_module.device)

    def _accumulate(self, pl_module, batch):
        gt, pred_field = InferenceHelper.generate_gt_and_pred_for_validation(
            batch, pl_module, voxel_resolution=self.voxel_resolution, spectra_bins=32
        )
        with torch.no_grad():
            for name, metric in self.metrics.items():
                val = metric.forward(gt, pred_field, input=batch)
                if val is None:
                    # Never silently drop a metric: warn once and leave it to surface as
                    # NaN at epoch end. A None here means the metric could not score this
                    # model's output — a bug to fix, not to hide.
                    if name not in self._warned_none:
                        warnings.warn(f"[MetricsPlotter] metric '{name}' returned None for "
                                      f"{type(pl_module).__name__} — logged as NaN.", RuntimeWarning)
                        self._warned_none.add(name)
                    continue
                if val.ndim >= 1:
                    # Metrics return an EMPTY tensor (torch.zeros(0), see smape.py) when their
                    # mask selects no voxels in this batch — the batch carries no information
                    # for that metric, so skip it (don't count it into the epoch mean). Adding
                    # an empty tensor to the scalar accumulator crashed with "output with shape
                    # [] doesn't match the broadcast shape [0]" (killed e0/a2/b1/c3 mid-sweep).
                    if val.numel() == 0:
                        continue
                    val = val.mean()
                self.metrics_accumulator[name] += val
                self.metrics_count[name] += 1

    def _log(self, pl_module, prefix):
        for name, acc in self.metrics_accumulator.items():
            n = self.metrics_count[name]
            value = (acc / n) if n > 0 else float("nan")   # NaN = visibly failed, not absent
            pl_module.log(f"{prefix}_{name}", value, on_step=False, on_epoch=True, prog_bar=False, logger=True)

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        self._accumulate(pl_module, batch)

    def on_validation_epoch_end(self, trainer, pl_module):
        self._log(pl_module, "val")

    def on_test_epoch_start(self, trainer, pl_module):
        return self.on_validation_epoch_start(trainer, pl_module)

    def on_test_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx = 0):
        self._accumulate(pl_module, batch)

    def on_test_epoch_end(self, trainer, pl_module):
        self._log(pl_module, "test")


class ErrorFieldPlotter(Callback):
    def __init__(self, logger: LoggerBase, mu_tr_file: str, spectra_bins: int, max_energy_eV: float, voxel_resolution: tuple[int, int, int] = (50, 50, 50)):
        self.logger = logger
        self.airkerma = AirkermaAccuracy(mu_tr_file, spectra_bins, max_energy_eV, keep_dim=True)
        self.airkerma.eval()
        self.voxel_resolution = voxel_resolution
        self.spectra_bins = spectra_bins
        self.channels_join = ChannelsJoin().eval()
    
    def on_validation_epoch_start(self, trainer, pl_module: BaseNeuralRadFieldModel):
        self.channels_join = self.channels_join.to(pl_module.device)
        self.airkerma = self.airkerma.to(pl_module.device)
        self.airkerma.normalizer = pl_module._normalizer.clone().to(pl_module.device).eval()

    def on_validation_batch_end(
        self,
        trainer: Trainer,
        pl_module: BaseNeuralRadFieldModel,
        outputs,
        batch: TrainingInputData,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        if batch_idx == 0:
            gt, pred_field = InferenceHelper.generate_gt_and_pred_for_validation(
                batch,
                pl_module,
                voxel_resolution=self.voxel_resolution,
                spectra_bins=32
            )

            aikerma_gt = self.airkerma.airkerma.forward(gt.spectrum, gt.flux)
            if isinstance(pred_field, RadiationFieldChannel):
                aikerma_pred = self.airkerma.airkerma.forward(pred_field.spectrum, pred_field.flux)
            elif isinstance(pred_field, AirKermaField):                
                aikerma_pred = pred_field.air_kerma
            else:
                raise ValueError("Prediction must be RadiationFieldChannel or AirKermaField")

            metric_field: torch.Tensor = 1.0 - self.airkerma._calc_metric(aikerma_gt, aikerma_pred)
            metric_field = metric_field.view_as(aikerma_gt)
            mask = ~torch.isfinite(metric_field)
            if mask.any():
                metric_field[mask] = 0.0

            # Normalize each batch independently to [0, 1]
            for i in range(metric_field.shape[0]):
                maxv = torch.max(metric_field[i])
                metric_field[i] = metric_field[i] / (maxv if maxv > 0 else 1.0)

            # Squeeze optional channel dim -> (B, X, Y, Z)
            if len(metric_field.shape) == 5:
                metric_field = metric_field.squeeze(1)

            # Build a subplots grid with 3 rows (XZ, XY, YZ) and one column per batch item
            batch_count = metric_field.shape[0]
            fig = make_subplots(
                rows=3,
                cols=batch_count,
                subplot_titles=[f"XZ {i}" for i in range(batch_count)]
                               + [f"XY {i}" for i in range(batch_count)]
                               + [f"YZ {i}" for i in range(batch_count)],
                vertical_spacing=0.03,
                horizontal_spacing=0.01,
                row_heights=[0.33, 0.33, 0.33]
            )
            # Single shared colorbar inside the figure
            fig.update_layout(
                coloraxis=dict(
                    colorscale='Viridis',
                    cmin=0.0,
                    cmax=1.0,
                    colorbar=dict(
                        title='RelErr',
                        x=1.02,            # outside the plotting area on the right
                        xanchor='left',
                        y=0.5,
                        len=0.9,
                        thickness=12,
                        bgcolor='rgba(255,255,255,0.6)'
                    )
                )
            )

            # Add one column per batch item with the three mid-plane slices
            for i in range(batch_count):
                # Shapes: (X, Y, Z) per batch after slicing
                # XZ plane: y = mid
                err_xz = metric_field[i, :, metric_field.shape[2] // 2, :].detach().cpu().numpy()
                # XY plane: z = mid
                err_xy = metric_field[i, :, :, metric_field.shape[3] // 2].detach().cpu().numpy()
                # YZ plane: x = mid
                err_yz = metric_field[i, metric_field.shape[1] // 2, :, :].detach().cpu().numpy()

                fig.add_trace(
                    go.Heatmap(
                        z=err_xz,
                        coloraxis='coloraxis',
                        showscale=False,
                        name=f'Batch {i} - XZ'
                    ),
                    row=1, col=i + 1
                )
                fig.add_trace(
                    go.Heatmap(
                        z=err_xy,
                        coloraxis='coloraxis',
                        showscale=False,
                        name=f'Batch {i} - XY'
                    ),
                    row=2, col=i + 1
                )
                fig.add_trace(
                    go.Heatmap(
                        z=err_yz,
                        coloraxis='coloraxis',
                        showscale=False,
                        name=f'Batch {i} - YZ'
                    ),
                    row=3, col=i + 1
                )

            fig.update_layout(
                title='Relative Error of Air Kerma - Batch',
                width=max(400, 250 * batch_count),
                height=900,
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                margin=dict(l=10, r=80, t=40, b=10),  # add right margin for the external colorbar
                showlegend=False
            )
            # Hide axis ticks for a clean grid
            fig.update_xaxes(showticklabels=False)
            fig.update_yaxes(showticklabels=False)

            self.logger.log_plot("Air Kerma Relative Error", fig)


class SpectrumErrorFieldPlotter(Callback):
    """Three-plane sliced view of the per-voxel spectrum error.

    Computes the histogram-intersection complement
    ``err = 1 - sum_b min(pred_b, target_b)`` per voxel (bounded in
    ``[0, 1]``: 0 = identical spectra, 1 = disjoint support) and renders
    the mid-plane slice through XZ, XY and YZ for each batch element,
    matching ``ErrorFieldPlotter``'s grid so the wandb dashboard stays
    visually consistent.

    Like the airkerma plotter, this fires once per validation epoch on
    batch 0 only — the wandb upload cost would otherwise dominate for
    long runs.
    """

    def __init__(self, logger: LoggerBase, spectra_bins: int = 32,
                 voxel_resolution: tuple[int, int, int] = (50, 50, 50)):
        self.logger = logger
        self.spectra_bins = spectra_bins
        self.voxel_resolution = voxel_resolution
        self.channels_join = ChannelsJoin().eval()

    def on_validation_epoch_start(self, trainer, pl_module: BaseNeuralRadFieldModel):
        self.channels_join = self.channels_join.to(pl_module.device)

    def on_validation_batch_end(
        self,
        trainer: Trainer,
        pl_module: BaseNeuralRadFieldModel,
        outputs,
        batch: TrainingInputData,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        if batch_idx != 0:
            return
        gt, pred_field = InferenceHelper.generate_gt_and_pred_for_validation(
            batch,
            pl_module,
            voxel_resolution=self.voxel_resolution,
            spectra_bins=self.spectra_bins,
        )

        # Spectrum lives on the field returned by the inference helper,
        # which has already been joined into a single channel.
        gt_spec = getattr(gt, "spectrum", None)
        pr_spec = getattr(pred_field, "spectrum", None)
        if gt_spec is None or pr_spec is None:
            return  # AirKermaField path; nothing to plot for spectrum.

        # Locate the histogram (bin) axis: it's the axis with `spectra_bins`
        # entries. Volumes from the inference helper have shape
        # (B, C, D, H, W) or (B, D, H, W, C) depending on layout. We
        # normalise both to (B, C, D, H, W) before computing the per-voxel
        # error so the spatial slicing below is uniform.
        def _to_bchwd(t: torch.Tensor) -> torch.Tensor:
            if t.ndim == 4:
                # (B, C, …) — add a trivial spatial dim 1 so the slicing
                # below stays robust; later code handles all three slices.
                t = t.unsqueeze(2)
            return t

        gt_spec = _to_bchwd(gt_spec)
        pr_spec = _to_bchwd(pr_spec)

        # Find the bin dim — first matching size after batch.
        bin_axis = None
        for ax in range(1, gt_spec.ndim):
            if gt_spec.size(ax) == self.spectra_bins:
                bin_axis = ax
                break
        if bin_axis is None:
            return

        # 1 - sum_b min(p_b, t_b) per voxel; the result has the bin axis
        # collapsed, leaving the spatial layout intact.
        err = 1.0 - torch.minimum(gt_spec, pr_spec).sum(dim=bin_axis)
        err = err.clamp(min=0.0, max=1.0)

        mask = ~torch.isfinite(err)
        if mask.any():
            err[mask] = 0.0

        if err.ndim != 4:
            # Expected (B, X, Y, Z). Bail out if downstream changed the
            # convention; we'd rather skip a plot than emit a confusing one.
            return

        batch_count = err.shape[0]
        fig = make_subplots(
            rows=3,
            cols=batch_count,
            subplot_titles=[f"XZ {i}" for i in range(batch_count)]
                           + [f"XY {i}" for i in range(batch_count)]
                           + [f"YZ {i}" for i in range(batch_count)],
            vertical_spacing=0.03,
            horizontal_spacing=0.01,
            row_heights=[0.33, 0.33, 0.33],
        )
        # Shared colorbar — Magma is visually distinct from the airkerma
        # plotter's Viridis so the two error maps don't blur together in
        # wandb's media panel.
        fig.update_layout(
            coloraxis=dict(
                colorscale='Magma',
                cmin=0.0,
                cmax=1.0,
                colorbar=dict(
                    title='SpecErr',
                    x=1.02,
                    xanchor='left',
                    y=0.5,
                    len=0.9,
                    thickness=12,
                    bgcolor='rgba(255,255,255,0.6)',
                ),
            )
        )

        for i in range(batch_count):
            # XZ: y = mid; XY: z = mid; YZ: x = mid.
            err_xz = err[i, :, err.shape[2] // 2, :].detach().cpu().numpy()
            err_xy = err[i, :, :, err.shape[3] // 2].detach().cpu().numpy()
            err_yz = err[i, err.shape[1] // 2, :, :].detach().cpu().numpy()
            fig.add_trace(
                go.Heatmap(z=err_xz, coloraxis='coloraxis', showscale=False,
                           name=f'Batch {i} - XZ'),
                row=1, col=i + 1,
            )
            fig.add_trace(
                go.Heatmap(z=err_xy, coloraxis='coloraxis', showscale=False,
                           name=f'Batch {i} - XY'),
                row=2, col=i + 1,
            )
            fig.add_trace(
                go.Heatmap(z=err_yz, coloraxis='coloraxis', showscale=False,
                           name=f'Batch {i} - YZ'),
                row=3, col=i + 1,
            )

        fig.update_layout(
            title='Per-Voxel Spectrum Error (1 - histogram intersection)',
            width=max(400, 250 * batch_count),
            height=900,
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            margin=dict(l=10, r=80, t=40, b=10),
            showlegend=False,
        )
        fig.update_xaxes(showticklabels=False)
        fig.update_yaxes(showticklabels=False)

        self.logger.log_plot("Spectrum Per-Voxel Error", fig)
