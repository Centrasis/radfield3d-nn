from lightning.pytorch.callbacks import Callback
from lightning.pytorch.trainer import Trainer
from loggers.logger import LoggerBase
from models.base import BaseNeuralRadFieldModel
import torch
from rftypes import RadiationField, TrainingInputData, RadiationFieldChannel, AirKermaField
from metrics.airkerma_accuracy import AirkermaAccuracy
from normalizations.base import Normalizer
from rfhelpers import InferenceHelper
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datasets.channel_join import ChannelsJoin
from metrics import MetricBase


class MetricsPlotter(Callback):
    def __init__(self, metrics: dict[str, MetricBase], spectra_bins: int = 32, voxel_resolution: tuple[int, int, int] = (50, 50, 50)):
        self.metrics = metrics
        self.voxel_resolution = voxel_resolution
        self.spectra_bins = spectra_bins
        for k, v in self.metrics.items():
            self.metrics[k] = v.eval()
        self.channels_join = ChannelsJoin().eval()

        self.metrics_accumulator = {
            name: torch.tensor(0.0, requires_grad=False) for name in self.metrics.keys()
        }
        self.batches_count = 0

    def on_validation_epoch_start(self, trainer, pl_module: BaseNeuralRadFieldModel):
        self.channels_join = self.channels_join.to(pl_module.device)
        for name, metric in self.metrics.items():
            self.metrics[name] = metric.to(pl_module.device)
            self.metrics_accumulator[name] = torch.tensor(0.0, requires_grad=False).to(pl_module.device)
        self.batches_count = 0

    def on_validation_batch_end(
        self,
        trainer: Trainer,
        pl_module: BaseNeuralRadFieldModel,
        outputs,
        batch: TrainingInputData,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        gt, pred_field = InferenceHelper.generate_gt_and_pred_for_validation(
            batch,
            pl_module,
            voxel_resolution=self.voxel_resolution,
            spectra_bins=32
        )
        with torch.no_grad():
            for name, metric in self.metrics.items():
                val = metric.forward(gt, pred_field, input=batch)
                if val is None:
                    continue
                if len(val.shape) == 1:
                    val = val.squeeze(0)
                self.metrics_accumulator[name] += val

        self.batches_count += 1

    def on_validation_epoch_end(self, trainer, pl_module):
        for name, accumulator in self.metrics_accumulator.items():
            pl_module.log(
                f"val_{name}",
                accumulator / self.batches_count if self.batches_count > 0 else 0,
                on_step=False,
                on_epoch=True,
                prog_bar=False,
                logger=True
            )

    def on_test_epoch_start(self, trainer, pl_module):
        return self.on_validation_epoch_start(trainer, pl_module)

    def on_test_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx = 0):
        return self.on_validation_batch_end(trainer, pl_module, outputs, batch, batch_idx, dataloader_idx)
    
    def on_test_epoch_end(self, trainer, pl_module):
        for name, accumulator in self.metrics_accumulator.items():
            pl_module.log(
                f"test_{name}",
                accumulator / self.batches_count if self.batches_count > 0 else 0,
                on_step=False,
                on_epoch=True,
                prog_bar=False,
                logger=True
            )


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
