import argparse
import os
from rich import print
import json
import plotly.graph_objects as go
from rich.table import Table
import torch
from datasets.channel_join import ChannelsJoin
from datasets.dataloader import RadiationFieldDataModule
from models import ModelConstructor, BaseNeuralRadFieldModel
from datasets import OriginalGroundTruthPreservation, CropDataset, construct_datamodule, get_dataset_dimensions_and_voxel_size
from rftypes import TrainingInputData, RadiationField, RadiationFieldChannel, DirectionalInput, rf3RadiationField
from rich.progress import track
from rich.console import Console
from rfhelpers import InferenceHelper
import numpy as np
from models.encoders.sinusoidal_encoding import SinusoidalFrequencyEncoding
from models.encoders.spherical_hamonics import SphericalHarmonics


def batch2cuda(batch: TrainingInputData) -> TrainingInputData:
    return TrainingInputData(
            input=DirectionalInput(
                direction=batch.input.direction.cuda(),
                spectrum=batch.input.spectrum.cuda(),
                geometry=batch.input.geometry.cuda() if batch.input.geometry is not None else None,
                origin=batch.input.origin.cuda() if batch.input.origin is not None else None,
                beam_shape_parameters=batch.input.beam_shape_parameters.cuda() if batch.input.beam_shape_parameters is not None else None,
                beam_shape_type=batch.input.beam_shape_type.cuda() if batch.input.beam_shape_type is not None else None
            ),
            ground_truth=RadiationField(
                scatter_field=RadiationFieldChannel(
                    flux=batch.ground_truth.scatter_field.flux.cuda(),
                    spectrum=batch.ground_truth.scatter_field.spectrum.cuda(),
                    error=batch.ground_truth.scatter_field.error.cuda() if batch.ground_truth.scatter_field.error is not None else None
                ),
                direct_beam=RadiationFieldChannel(
                    flux=batch.ground_truth.direct_beam.flux.cuda(),
                    spectrum=batch.ground_truth.direct_beam.spectrum.cuda(),
                    error=batch.ground_truth.direct_beam.error.cuda() if batch.ground_truth.direct_beam.error is not None else None
                )
            ) if isinstance(batch.ground_truth, (rf3RadiationField, RadiationField)) else RadiationFieldChannel(
                flux=batch.ground_truth.flux.cuda(),
                spectrum=batch.ground_truth.spectrum.cuda(),
                error=batch.ground_truth.error.cuda() if batch.ground_truth.error is not None else None
            ),
            original_ground_truth=RadiationField(
                    scatter_field=RadiationFieldChannel(
                        flux=batch.original_ground_truth.scatter_field.flux.cuda(),
                        spectrum=batch.original_ground_truth.scatter_field.spectrum.cuda(),
                        error=batch.original_ground_truth.scatter_field.error.cuda() if batch.original_ground_truth.scatter_field.error is not None else None
                    ),
                    direct_beam=RadiationFieldChannel(
                        flux=batch.original_ground_truth.direct_beam.flux.cuda(),
                        spectrum=batch.original_ground_truth.direct_beam.spectrum.cuda(),
                        error=batch.original_ground_truth.direct_beam.error.cuda() if batch.original_ground_truth.direct_beam.error is not None else None
                ) if isinstance(batch.original_ground_truth, (RadiationField, rf3RadiationField)) else RadiationFieldChannel(
                    flux=batch.original_ground_truth.flux.cuda(),
                    spectrum=batch.original_ground_truth.spectrum.cuda(),
                    error=batch.original_ground_truth.error.cuda() if batch.original_ground_truth.error is not None else None
                )
            ) if isinstance(batch, TrainingInputData) and batch.original_ground_truth is not None else None
        )



class Result(object):
    hyper_parameters: dict
    metrics: dict

    def __init__(self, hyper_parameters: dict, metrics: dict, model_path: str = None):
        self.hyper_parameters = hyper_parameters
        self.metrics = metrics
        self.model_path = model_path
        self._inference_duration_per_field_s = None
        self._inference_duration_per_field_stddev_s = None

    @property
    def d_model(self) -> int:
        return self.hyper_parameters["d_model"]
    
    @property
    def randomize_voxel_location_in_training(self) -> int:
        return self.hyper_parameters["randomize_voxel_location_in_training"]
    
    @property
    def voxels_centered_around_origin(self) -> int:
        return self.hyper_parameters["voxels_centered_around_origin"]
    
    @property
    def use_spectra_encoding(self) -> int:
        return self.hyper_parameters["use_spectra_encoding"] if "use_spectra_encoding" in self.hyper_parameters else "--"

    @property
    def in_spectra_dim(self) -> int:
        return self.hyper_parameters["in_spectra_dim"] if "in_spectra_dim" in self.hyper_parameters else "--"

    @property
    def encoded_spectra_dims(self) -> int:
        return self.hyper_parameters["encoded_spectra_dims"] if "encoded_spectra_dims" in self.hyper_parameters else "--"

    @property
    def location_encoding_dims(self) -> int:
        return self.hyper_parameters["location_encoding_dims"] if "location_encoding_dims" in self.hyper_parameters else "--"

    @property
    def direction_encoding_dims(self) -> int:
        return self.hyper_parameters["direction_encoding_dims"] if "direction_encoding_dims" in self.hyper_parameters else "--"
    
    @property
    def spectrum_loss(self) -> int:
        return self.hyper_parameters["spectrum_loss"] if "spectrum_loss" in self.hyper_parameters else "--"
    
    @property
    def conditioning_method(self) -> str:
        return self.hyper_parameters["conditioning"] if "conditioning" in self.hyper_parameters else "--"
    
    @property
    def normalizer(self) -> str:
        return self.hyper_parameters["normalizer"] if "normalizer" in self.hyper_parameters else "--"

    @property
    def top90_airkerma_accuracy(self) -> float:
        return self.metrics["test_top90_airkerma_accuracy"]
    
    @property
    def airkerma_ssim(self) -> float:
        return self.metrics["test_airkerma_ssim"]
    
    @property
    def spectrum_accuracy(self) -> float:
        return self.metrics["test_spectrum_accuracy"]
    
    @property
    def airkerma_accuracy_scatter(self) -> float:
        return self.metrics["test_airkerma_accuracy_scatter"]
    
    @property
    def airkerma_gpr_6cm_3percent(self) -> float:
        return self.metrics["test_global_airkerma_gamma_index_3percent_per_6cm"]
    
    @property
    def airkerma_gpr_4cm_10percent(self) -> float:
        return self.metrics["test_global_airkerma_gamma_index_10percent_per_4cm"]

    @property
    def ranking_score(self) -> float:
        rank = self.top90_airkerma_accuracy
        rank += self.airkerma_ssim
        rank += self.spectrum_accuracy
        rank += self.airkerma_accuracy_scatter
        return rank / 4.0
    
    @property
    def inference_duration_per_field_s(self) -> float:
        if self._inference_duration_per_field_s is None:
            return "Undef"
        else:
            return self._inference_duration_per_field_s
        
    @property
    def inference_duration_per_field_stddev_s(self) -> float:
        if self._inference_duration_per_field_stddev_s is None:
            return "Undef"
        else:
            return self._inference_duration_per_field_stddev_s
    
    def __repr__(self):
        return f"Result<rank={self.ranking_score}, name={os.path.splitext(os.path.basename(self.model_path))[0] if self.model_path is not None else '--'}>"
    
    def construct_model(self) -> BaseNeuralRadFieldModel:
        model_config_path = os.path.splitext(self.model_path)[0] + ".json"
        if not os.path.exists(model_config_path):
            raise FileNotFoundError(f"Model config file not found at {model_config_path}")
        model_config = json.load(open(model_config_path, "r"))
        model_cls = ModelConstructor.create_model_from_dict(model_config if "parameters" in model_config else {
            "parameters": model_config["hyper_parameters"],
            "model_name": model_config["model_name"]
        })
        try:
            model = model_cls.load_from_checkpoint(self.model_path)
        except Exception as e:
            print(f"[yellow]Weigths file was probably not a pytorch-lightning checkpoint: {e} [/yellow]")
            model = model_cls()
            try:
                model.load_state_dict(torch.load(self.model_path))
            except:
                pen = model.positional_location_encoding
                model.positional_location_encoding = SinusoidalFrequencyEncoding(
                    pos_enc_dim=pen.pos_enc_dim,
                    d_input=pen.d_input,
                    append_input=pen.append_input,
                    dim=-1,
                    use_tcnn=False
                )
                model.load_state_dict(torch.load(self.model_path))
                model.positional_location_encoding = pen
        return model.cuda().eval()
    
    def measure_inference_time(self, datamodule: RadiationFieldDataModule, warmup_runs: int = 5, measured_runs: int = 25, voxel_counts: tuple[int, int, int] = (50, 50, 50), spectra_bins: int = 32):
        try:
            model = self.construct_model()
            model.max_inner_batch_size = 65536 * 4
        except Exception as e:
            print(f"[red]Failed to construct model for inference time measurement: {e}[/red]")
            return
        # Warmup
        with torch.no_grad():
            # Measured runs
            measured_durations = []
            for i, batch in track(enumerate(datamodule.test_dataloader()), description="Measuring inference time...", total=warmup_runs + measured_runs):
                batch = batch2cuda(batch)
                if i >= measured_runs + warmup_runs:
                    break
                for dp in datamodule.data_processings:
                    batch: TrainingInputData = dp(batch)

                _, duration_ms = InferenceHelper.timed_inference_step(
                    batch,
                    model,
                    voxel_resolution=voxel_counts,
                    spectra_bins=spectra_bins
                )
                if i < warmup_runs:
                    continue  # skip warmup runs
                measured_durations.append(duration_ms)
        
        self._inference_duration_per_field_s = np.array(measured_durations).mean() / 1000.0  # convert ms to s
        self._inference_duration_per_field_stddev_s = np.array(measured_durations).std() / 1000.0  # convert ms to s


if __name__ == "__main__":
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.enabled = True

    parser = argparse.ArgumentParser(add_help=True, description="Evaluate hyperparameter optimization results.")
    parser.add_argument("--results_path", type=str, required=True, help="Path to the hyperparameter optimization results file.")
    parser.add_argument("--dataset_path", type=str, required=True, help="The dataset to use.")
    parser.add_argument("--join_channels", action="store_true", help="Join the channels of the radiation field.", required=False, default=False)
    parser.add_argument("--enforce_voxel_resolution", type=int, nargs=3, required=False, default=None, help="Enforce a specific voxel resolution for the dataset. Format: x y z")
    parser.add_argument("--use_geometry", action="store_true", help="Use geometry dataset (only for Layerwise datasets).", required=False, default=False)
    parser.add_argument("--use_beam_parameters", action="store_true", help="Use beam parameters normalization.", required=False, default=False)
    parser.add_argument("--no_timing", action="store_true", help="Skip inference timing measurement.", required=False, default=False)
    parser.add_argument("--max_d_model", type=int, required=False, default=None, help="Maximum d_model size to consider.")

    args = parser.parse_args()
    results_path = os.path.join(args.results_path, "optuna_studies")
    print(f"Evaluating hyperparameter optimization results from: {results_path}")

    result_files = [f for f in os.listdir(results_path) if f.endswith(".json")]

    raw_results = [(json.load(open(os.path.join(results_path, f), "r")), os.path.join(results_path, f)) for f in result_files]

    results = [
        Result(
            raw_result[0]["parameters"],
            raw_result[0]["metrics"],
            model_path=os.path.splitext(raw_result[1])[0] + ".pt" if os.path.exists(os.path.splitext(raw_result[1])[0] + ".pt") else None,
        ) for raw_result in raw_results
    ]

    if args.max_d_model is not None:
        results = [r for r in results if r.d_model <= args.max_d_model]

    datamodule = construct_datamodule(
        dataset_path=args.dataset_path,
        batch_size=1,
        num_workers=1,
        use_geometry=args.use_geometry,
        use_beam_parameters=args.use_beam_parameters,
        dataprocessings=[
            OriginalGroundTruthPreservation()
        ] + [ChannelsJoin()] if args.join_channels else [],
        voxel_resolution=args.enforce_voxel_resolution
    )
    datamodule.data_processings = [dp.cuda().eval() for dp in datamodule.data_processings]
    field_dim, vx_size = get_dataset_dimensions_and_voxel_size(datamodule)
    vx_counts = (int(field_dim[0] / vx_size), int(field_dim[1] / vx_size), int(field_dim[2] / vx_size))

    top_10_results = sorted(results, key=lambda r: r.ranking_score, reverse=True)#[:min(10, len(results))]
    print("\n[bold underline]Top 10 Results:[/bold underline]")
    tbl = Table(show_header=True, header_style="bold magenta", width=None)
    tbl.add_column("Rank", style="dim", width=6, no_wrap=True)
    tbl.add_column("D Model", justify="right", no_wrap=True)
    tbl.add_column("Loc Enc Dims", justify="right", no_wrap=True)
    tbl.add_column("Dir Enc Dims", justify="right", no_wrap=True)
    tbl.add_column("Spectrum Loss", justify="right", no_wrap=True)
    tbl.add_column("Conditioning", justify="right", no_wrap=True)
    tbl.add_column("Normalizer", justify="right", no_wrap=True)
    tbl.add_column("spectra enc", justify="right", no_wrap=True)
    tbl.add_column("enc spec dim", justify="right", no_wrap=True)
    tbl.add_column("rand vx", justify="right", no_wrap=True)
    tbl.add_column("Top90 Airkerma Acc", justify="right", no_wrap=True)
    tbl.add_column("Airkerma SSIM", justify="right", no_wrap=True)
    tbl.add_column("Spectrum Acc", justify="right", no_wrap=True)
    tbl.add_column("Airkerma Acc Scatter", justify="right", no_wrap=True)
    tbl.add_column("Airkerma GPR 6cm 3%", justify="right", no_wrap=True)
    tbl.add_column("Airkerma GPR 4cm 10%", justify="right", no_wrap=True)
    tbl.add_column("Model Size (MB)", justify="right", no_wrap=True)
    if not args.no_timing:
        tbl.add_column("Inference duration (ms)", justify="right", no_wrap=True)
    tbl.add_column("Trial name", justify="right", no_wrap=True)
    for i, result in enumerate(top_10_results):
        model_size_in_mb = os.path.getsize(result.model_path) / (1024 ** 2) if result.model_path is not None else 0
        if not args.no_timing:
            result.measure_inference_time(datamodule, voxel_counts=vx_counts, spectra_bins=32)
        row = [
            str(i + 1),
            str(result.d_model),
            str(result.location_encoding_dims),
            str(result.direction_encoding_dims),
            str(result.spectrum_loss),
            result.conditioning_method,
            f"{result.normalizer}",
            f"{result.use_spectra_encoding}",
            f"{result.encoded_spectra_dims}",
            f"{result.randomize_voxel_location_in_training}", 
            f"{result.top90_airkerma_accuracy:.4f}",
            f"{result.airkerma_ssim:.4f}",
            f"{result.spectrum_accuracy:.4f}",
            f"{result.airkerma_accuracy_scatter:.4f}",
            f"{result.airkerma_gpr_6cm_3percent:.4f}",
            f"{result.airkerma_gpr_4cm_10percent:.4f}",
            f"{model_size_in_mb:.2f}"
        ]
        if not args.no_timing:
            row.append(f"{result.inference_duration_per_field_s*1000:.2f} ± {result.inference_duration_per_field_stddev_s*1000:.2f}" if isinstance(result.inference_duration_per_field_s, float) else "Undef")
        tbl.add_row(
            *row,
            f"{os.path.splitext(os.path.basename(result.model_path))[0] if result.model_path is not None else '--'}",
        )
    print(tbl)

    # store tbl to file
    with open(os.path.join(results_path, "top_results.txt"), "w") as f:
        console = Console(force_terminal=False, file=f)
        console.print(tbl)
