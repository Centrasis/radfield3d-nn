from lightning.pytorch.callbacks import Callback
from lightning.pytorch.trainer import Trainer
from loggers.logger import LoggerBase
import torch
import random
from radfield3dnn.rftypes import AirKermaField, TrainingInputData, RadiationField, PositionalInput
from torch import Tensor
import plotly.graph_objects as go
from rich import print
from radfield3dnn.datasets.channel_join import ChannelsJoin
from radfield3dnn.models.base import BaseNeuralRadFieldModel
from radfield3dnn.rfhelpers import InferenceHelper
from visualizers.spectrum_plotter import SpectrumPlotter, SpectrumDescriptor
from visualizers.volumetric_plotter import AirkermaPlotter
from visualizers.sliced_plotter import SlicedAirkermaPlotter


class ValidationPlotter(Callback):
    def __init__(self, logger: LoggerBase, mu_tr_path: str, max_energy_eV: float = 1.5e+5, voxel_size: float = 0.025, voxel_resolution: tuple = (50, 50, 50), plot_spectra: bool = True, plot_volume: bool = True):
        super().__init__()
        self.logger = logger
        self.voxel_resolution = torch.tensor(voxel_resolution, dtype=torch.int32)
        self.join_channels: ChannelsJoin = ChannelsJoin()
        self.plot_spectra = plot_spectra
        self.plot_volume = plot_volume
        self.spectra_plotter = SpectrumPlotter(bins=32, max_energy_eV=max_energy_eV, used_unit="keV", bar_opacity=0.6)
        self.airkerma_plotter = AirkermaPlotter(
            mutr_path=mu_tr_path,
            max_energy_eV=max_energy_eV,
            voxel_size=voxel_size,
            normalize_airkerma=True
        )
        self.sliced_airkerma_plotter = SlicedAirkermaPlotter(
            mutr_path=mu_tr_path,
            max_energy_eV=max_energy_eV,
            voxel_size=voxel_size,
            normalize_airkerma=True
        )

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

            with torch.no_grad():
                gt_flux = InferenceHelper.extract_flux_or_airkerma(gt)
                gt_spectra = InferenceHelper.try_extract_spectrum(gt)
                self.voxel_resolution = gt_flux.shape[-3:]  # robust for 4D/5D
                self.spectra_bins = gt_spectra.shape[1] if gt_spectra is not None else 32
                batch_size = gt_flux.shape[0] if len(gt_flux.shape) >= 4 else 1
                is_complete_volume = len(gt_flux.shape) >= 4
                
                pred_field_flux = InferenceHelper.extract_flux_or_airkerma(pred_field).detach()
                pred_field_flux: Tensor = torch.clamp(pred_field_flux, min=0.0)
                pred_field_spectra: Tensor | None = InferenceHelper.try_extract_spectrum(pred_field)
                if pred_field_spectra is not None:
                    pred_field_spectra = pred_field_spectra.detach()

            # Build a single figure with 3 rows (XZ, XY, YZ) and n columns (per batch item)
            fig = self.sliced_airkerma_plotter.create_airkerma_subplot_figure(
                rows=3,
                cols=batch_size,
                subplot_titles=[f"XZ {i}" for i in range(batch_size)]
                               + [f"XY {i}" for i in range(batch_size)]
                               + [f"YZ {i}" for i in range(batch_size)],
            )


            if self.plot_volume:
                fig_volume = self.airkerma_plotter.create_airkerma_figure(title="Predicted Airkerma Volume")

            invalid_mask = torch.isneginf(pred_field_flux)

            # Iterate over batch
            for idx in range(batch_size):
                pred_flux = pred_field_flux[idx] if batch_size > 1 else pred_field_flux
 
                if (pred_flux == 0.0).all():
                    print(f"[yellow]Warning: Predicted flux is all zeros for item {idx} in validation batch.[/yellow]")

                if self.plot_volume and idx == 0:
                    vidx = int(torch.randint(0, batch_size - 1, size=(1,)).item()) if batch_size > 1 else 0
                    self.airkerma_plotter.add_airkerma_to_figure(
                        fig=fig_volume,
                        field=pred_field,
                        batch_idx=vidx,
                        name=f"Predicted Flux {vidx}"
                    )

                # Optional target for blending (normalize independently to ensure visibility)
                if is_complete_volume:
                    gt_field = gt_flux[idx] if batch_size > 1 else gt_flux
                    mask = invalid_mask[idx] if batch_size > 1 else invalid_mask
                    if mask.any():
                        pred_flux = pred_flux.clone()
                        pred_flux[mask] = 0.0
                    
                    self.sliced_airkerma_plotter.add_blended_slices(
                        fig=fig,
                        pred_field=pred_flux,
                        gt_field=gt_field,
                        X=0.5,
                        name1=f"Pred YZ{idx}",
                        name2=f"GT YZ{idx}",
                        row=1,
                        col=idx + 1
                    )
                    self.sliced_airkerma_plotter.add_blended_slices(
                        fig=fig,
                        pred_field=pred_flux,
                        gt_field=gt_field,
                        Y=0.5,
                        name1=f"Pred XZ{idx}",
                        name2=f"GT XZ{idx}",
                        row=2,
                        col=idx + 1
                    )
                    self.sliced_airkerma_plotter.add_blended_slices(
                        fig=fig,
                        pred_field=pred_flux,
                        gt_field=gt_field,
                        Z=0.5,
                        name1=f"Pred XY{idx}",
                        name2=f"GT XY{idx}",
                        row=3,
                        col=idx + 1
                    )
                else:
                    self.sliced_airkerma_plotter.add_airkerma_slice_to_figure(
                        fig=fig,
                        field=pred_flux,
                        X=0.5,
                        name=f"Pred XZ {idx}",
                        row=1,
                        col=idx + 1
                    )
                    self.sliced_airkerma_plotter.add_airkerma_slice_to_figure(
                        fig=fig,
                        field=pred_flux,
                        Y=0.5,
                        name=f"Pred XY {idx}",
                        row=2,
                        col=idx + 1
                    )
                    self.sliced_airkerma_plotter.add_airkerma_slice_to_figure(
                        fig=fig,
                        field=pred_flux,
                        Z=0.5,
                        name=f"Pred YZ {idx}",
                        row=3,
                        col=idx + 1
                    )

            rendered_measurand = "Air Kerma" if isinstance(pred_field, AirKermaField) else "Flux"
            fig.update_layout(
                title_text=f"Predicted vs Target {rendered_measurand} (Red=Pred, Blue=Target, Purple=Overlap) - Batch",
                showlegend=False,
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                margin=dict(l=10, r=10, t=40, b=10)
            )
            # Hide ticks for compact grid
            fig.update_xaxes(showticklabels=False)
            fig.update_yaxes(showticklabels=False)
            self.logger.log_plot("Predicted vs Target (Batch)", fig)

            if self.plot_volume:
                self.logger.log_plot(f"Predicted Volume", fig_volume)

            if pred_field_spectra is not None and self.plot_spectra:
                if invalid_mask.any():
                    mask = invalid_mask.expand_as(pred_field_spectra)
                    allowed_spectra_gt = gt_spectra[~mask] if gt_spectra is not None else None
                    allowed_spectra_gt = allowed_spectra_gt.view(-1, pred_field_spectra.shape[1]) if allowed_spectra_gt is not None else None
                    allowed_spectra_pd = pred_field_spectra[~mask] if pred_field_spectra is not None else None
                    allowed_spectra_pd = allowed_spectra_pd.view(-1, pred_field_spectra.shape[1])
                    voxel_count = allowed_spectra_gt.shape[0]
                else:
                    voxel_count = pred_field_spectra.shape[2] * pred_field_spectra.shape[3] * pred_field_spectra.shape[4]
                voxel_idxs = [random.randint(0, voxel_count - 1) for _ in range(4)]
                voxel_xyzs = []
                for vx_idx in voxel_idxs:
                    flat_idx = vx_idx
                    z = flat_idx % self.voxel_resolution[2]
                    flat_idx = flat_idx // self.voxel_resolution[2]
                    y = flat_idx % self.voxel_resolution[1]
                    x = flat_idx // self.voxel_resolution[1]
                    voxel_xyzs.append((x, y, z))
                subplot_titles = [f"Spectrum {idx}" for idx in voxel_xyzs]

                fig = self.spectra_plotter.create_spectrum_subplots(rows=round(len(voxel_idxs) ** (1/2)), cols=round(len(voxel_idxs) ** (1/2)), subtitles=subplot_titles)
                vidx = 0 if batch_size == 1 else random.randint(0, batch_size - 1)
                for i, _ in enumerate(voxel_idxs):
                    row = (i // 2) + 1
                    col = (i % 2)  + 1

                    self.spectra_plotter.add_spectrum_to_figure(
                        fig=fig,
                        field=pred_field,
                        descriptor=SpectrumDescriptor(
                            batch_idx=vidx,
                            xyz=voxel_xyzs[i],
                            trace_name=f"Predicted {i}"
                        ),
                        row=row,
                        col=col
                    )
                    if gt_spectra is not None:
                        self.spectra_plotter.add_spectrum_to_figure(
                            fig=fig,
                            field=batch.original_ground_truth if batch.original_ground_truth is not None else batch.ground_truth,
                            descriptor=SpectrumDescriptor(
                                batch_idx=vidx,
                                xyz=voxel_xyzs[i],
                                trace_name=f"Ground Truth {i}"
                            ),
                            row=row,
                            col=col
                        )
                fig.update_layout(title_text="Spectra at Random Voxels", barmode='overlay', height=1200, width=1000, showlegend=True, xaxis_title='Energy in keV', yaxis_title='Intensity', plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
                self.logger.log_plot("Spectra at Random Voxels", fig)


class ValidationDepthIntesityPlotter(Callback):
    def __init__(self, logger: LoggerBase, voxel_resolution: tuple = (50, 50, 50)):
        super().__init__()
        self.logger = logger
        self.voxel_resolution = torch.tensor(voxel_resolution, dtype=torch.int32)
        self.first_call = True
        self.join_channels: ChannelsJoin = ChannelsJoin()

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
            if self.first_call:
                self.first_call = False
                self.join_channels = self.join_channels.to(pl_module.device)

            targets = batch.ground_truth
            inputs = batch.input
            idx = random.randint(0, targets.scatter_field.flux.shape[0] - 1) if isinstance(targets, RadiationField) else random.randint(0, targets.flux.shape[0] - 1)
            
            direction = inputs.direction[idx].unsqueeze(0)
            spectrum = inputs.spectrum[idx].unsqueeze(0)
            origin_point = -direction
            target_point = direction

            # determine origin_point as the intersection of the ray with the bounding box in negative direction and target_point as the intersection of the ray with the bounding box in positive direction
            min_t = torch.tensor(float('inf'), device=direction.device)
            max_t = torch.tensor(float('-inf'), device=direction.device)
            for axis in range(3):  # x, y, z axes
                for face_value in [-1.0, 1.0]:  # negative and positive faces
                    non_parallel = torch.abs((-direction)[0, axis]) > 1e-6
                    if non_parallel.any():
                        t = face_value / (-direction)[0, axis]
                        valid = t != 0
                        if valid.all():
                            point = -direction * t
                    
                            # Check if point is within the face bounds
                            in_bounds = True
                            for j in range(3):
                                if j != axis and abs(point[0, j]) > 1.0:
                                    in_bounds = False
                                    break
                    
                            # If within bounds and closer than current minimum, update
                            if in_bounds and t < min_t:
                                min_t = t
                                origin_point = point
                            if in_bounds and t > max_t:
                                max_t = t
                                target_point = point            


            length = (target_point - origin_point).norm()
            step = length / self.voxel_resolution[0]
            steps = torch.arange(0, length, step).unsqueeze(1).to(direction.device)
            steps: Tensor = steps * direction + origin_point

            net_in = PositionalInput(
                direction=direction.repeat(steps.shape[0], 1),
                spectrum=spectrum.repeat(steps.shape[0], 1),
                position=steps,
                origin=origin_point.repeat(steps.shape[0], 1),
                geometry=inputs.geometry[idx].unsqueeze(0).repeat(steps.shape[0], 1) if inputs.geometry is not None else None,
                beam_shape_parameters=inputs.beam_shape_parameters[idx].unsqueeze(0).repeat(steps.shape[0], 1) if inputs.beam_shape_parameters is not None else None,
                beam_shape_type=inputs.beam_shape_type[idx].unsqueeze(0).repeat(steps.shape[0], 1) if inputs.beam_shape_type is not None else None
            )

            field: RadiationField = pl_module.forward(net_in)
            flux = field.scatter_field.flux
            if field.direct_beam is not None:
                flux += field.direct_beam.flux
            flux = flux / (flux.max() if flux.max() > 0 else 1.0)
            flux = flux.squeeze(0)
            if len(flux.shape) >= 4:
                return

            data = [
                [x - flux.shape[0] / 2.0, float(flux[x].cpu().numpy())]
                for x in range(flux.shape[0])
            ]

            fig = go.Figure()
            fig.add_trace(go.Scatter(x=[d[0] for d in data], y=[d[1] for d in data], mode='lines', name='Flux'))
            fig.update_layout(title='Flux vs Depth', xaxis_title='Depth', yaxis_title='Flux', plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
            self.logger.log_plot("Flux vs Depth", fig)
