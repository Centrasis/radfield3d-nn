import json
import torch
from models import ModelConstructor
import os
import argparse
from RadFiled3D.RadFiled3D import CartesianRadiationField, HistogramVoxel
from RadFiled3D.pytorch.radiationfieldloader import DataLoaderBuilder
from RadFiled3D.metadata.v1 import Metadata
from models.encoders.sinusoidal_encoding import SinusoidalFrequencyEncoding
from rftypes import TrainingInputData, RadiationField, RadiationFieldChannel, DirectionalInput, rf3RadiationField, rf3TrainingInputData
from rich.progress import track
from normalizations.beam_parameters import BeamParametersNormalization
import numpy as np
from normalizations.linear import LinearNormalizer
from normalizations.lognormalizer import LogNormalizer
from normalizations.asinh import LearnableAsinhNormalizer
from torch import Tensor
from RadFiled3D.pytorch.datasets.radfield3d import RadField3DDataset, RadField3DDatasetWithGeometry
from datasets.dataloader import RadiationFieldDataModule
from metrics import SMAPEAccuracy
from datasets.channel_join import ChannelsJoin
from plotly import graph_objects as go
from processings.airkerma import Airkerma
from metrics import HistogramOverlapAccuracy
from datasets.crop_dataset import CropDataset
from rich import print
from plotly.subplots import make_subplots
from RadFiled3D.utils import FieldStore
from datasets import OriginalGroundTruthPreservation
from rfhelpers import InferenceHelper


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


if __name__ == "__main__":
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.enabled = True

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True, help="Path to the model weights file.")
    parser.add_argument("--dataset_path", type=str, required=True, help="Path to the dataset directory.")
    parser.add_argument("--mu_tr_path", type=str, required=True, help="Path to the mu_tr file.")
    parser.add_argument("--enforce_voxel_resolution", type=int, nargs=3, default=None, help="Enforce voxel resolution for the model.")
    parser.add_argument("--use_beam_parameters", action="store_true", help="Use beam parameters normalization.")
    args = parser.parse_args()

    USE_BEAM_PARAMETERS = args.use_beam_parameters
    VOXEL_SIZE_M = 0.0
    FIELD_DIMENSIONS = (1.0, 1.0, 1.0)

    model_weights_path = args.model_path
    model_config_path = os.path.splitext(model_weights_path)[0] + ".config"
    if not os.path.exists(model_config_path):
        model_config_path = os.path.splitext(model_weights_path)[0] + ".json"
        if not os.path.exists(model_config_path):
            raise FileNotFoundError(f"Model config file not found for weights at {model_weights_path}")
    dataset_path = args.dataset_path
    voxel_counts = args.enforce_voxel_resolution if args.enforce_voxel_resolution is not None else [50, 50, 50]

    check_dataset = DataLoaderBuilder(dataset_path)
    template_field: tuple[CartesianRadiationField, Metadata] = check_dataset.build_train_dataset()[0]
    spectrum_vx_template: HistogramVoxel = template_field[0].get_channel("scatter_field").get_voxel_flat("spectrum", 0)
    spectra_bins = spectrum_vx_template.get_bins()
    spectra_bin_width = spectrum_vx_template.get_histogram_bin_width()

    DATASET_STATISTICS = {}
    if os.path.exists(os.path.join(dataset_path, "statistics.json")):
        statistics_path = os.path.join(dataset_path, "statistics.json")
        with open(statistics_path, "r") as f:
            DATASET_STATISTICS = json.load(f)
        print(f"[green]Loaded dataset statistics from {statistics_path}[/green]")
    else:
        print(f"[yellow]No dataset statistics found at {os.path.join(dataset_path, 'statistics.json')}[/yellow]")

    # load mu_tr file
    mu_tr_path = args.mu_tr_path
    if mu_tr_path is not None and not os.path.isabs(mu_tr_path):
        mu_tr_path = os.path.abspath(mu_tr_path)
    try:
        mu_tr = torch.tensor(np.loadtxt(mu_tr_path, skiprows=0), dtype=torch.float32)
    except:
        mu_tr = torch.tensor(np.loadtxt(mu_tr_path, skiprows=1), dtype=torch.float32)
    bin_edges = torch.arange(0, spectra_bins * spectra_bin_width, spectra_bin_width)

    model_config = json.load(open(model_config_path, "r"))
    model_cls = ModelConstructor.create_model_from_dict(model_config if "parameters" in model_config else {
        "parameters": model_config["hyper_parameters"],
        "model_name": model_config["model_name"]
    })
    if not os.path.exists(model_weights_path):
        try:
            model = model_cls.load_from_checkpoint(model_weights_path)
        except Exception as e:
            print(f"[yellow]Weigths file was probably not a pytorch-lightning checkpoint: {e} [/yellow]")
            model = model_cls()
            try:
                model.load_state_dict(torch.load(model_weights_path))
            except:
                try:
                    print(f"[yellow]Try load pure pytorch encoder, but use tccn encoder...[/yellow]")
                    pen = model.positional_location_encoding
                    model.positional_location_encoding = SinusoidalFrequencyEncoding(
                        pos_enc_dim=pen.pos_enc_dim,
                        d_input=pen.d_input,
                        append_input=pen.append_input,
                        dim=-1,
                        use_tcnn=False
                    )
                    model.load_state_dict(torch.load(model_weights_path))
                    model.positional_location_encoding = pen
                except:
                    print("[yellow]Could not load weight! Testing untrained![/yellow]")
    else:
        model = model_cls()
        print("[yellow]There is no weights file! Testing untrained![/yellow]")
    model.eval()
    model.cuda()

    #model._search_optimal_batch_size()
    #model.max_inner_batch_size = 65536  # to avoid OOM errors
    model.max_inner_batch_size = 65536 * 4

    airkerma = Airkerma(mu_tr.cuda(), spectra_bins).cuda().eval()
    model_train_normalizer = model._normalizer.clone().to(model.device).eval()
    normalizer_01 = LinearNormalizer((0.0, 1.0)).cuda().eval()
    radfield_join = ChannelsJoin().cuda().eval()

    if isinstance(model_train_normalizer, LearnableAsinhNormalizer):
        print(f"[yellow]Using LearnableAsinhNormalizer with alpha={model_train_normalizer.input_scale}[/yellow]")

    dataset_cls = RadField3DDataset
    if os.path.exists(os.path.join(dataset_path, "geom_desc")):
        print("[yellow]Using geometry dataset with voxelized geometries!")
        dataset_cls = RadField3DDatasetWithGeometry

    dataprocessings = [
        OriginalGroundTruthPreservation(),
        radfield_join,
        CropDataset(voxel_counts).cuda(),
        model_train_normalizer
    ]
    if USE_BEAM_PARAMETERS:
        print("[yellow]Using beam parameters normalization!")
        beam_normalizer = BeamParametersNormalization(
            opening_angle_range_deg=(
                DATASET_STATISTICS["tube_opening_angles_deg"]["Min"],
                DATASET_STATISTICS["tube_opening_angles_deg"]["Max"]
            ),
            size_per_voxel_m=VOXEL_SIZE_M if VOXEL_SIZE_M > 0.0 else 0.01,
            is_origin_centered=False,
            distance_range_m=(
                DATASET_STATISTICS["tube_distances_m"]["Min"],
                DATASET_STATISTICS["tube_distances_m"]["Max"]
            ),
            half_field_size=(FIELD_DIMENSIONS[0]/2, FIELD_DIMENSIONS[1]/2, FIELD_DIMENSIONS[2]/2)
        ).cuda()
        dataprocessings.append(beam_normalizer)

    datamodule = RadiationFieldDataModule(
        zip_directory=dataset_path,
        dataset_cls=dataset_cls,
        batch_size=1,
        num_workers=1,
        data_processings=[]
    )

    if voxel_counts is not None:
        print(f"[blue]Testing dataset with voxel resolution {voxel_counts}")
        test_dl = datamodule.test_dataloader()
        test_ds = test_dl.dataset
        test_files = test_ds.file_paths
        test_field: CartesianRadiationField = FieldStore.load(test_files[0])
        field_dim = test_field.get_field_dimensions()
        FIELD_DIMENSIONS = (field_dim.x, field_dim.y, field_dim.z)
        assert test_field.get_voxel_counts().x >= voxel_counts[0] and test_field.get_voxel_counts().y >= voxel_counts[1] and test_field.get_voxel_counts().z >= voxel_counts[2], f"Voxel resolution of dataset {test_field.get_voxel_counts()} does not match enforced resolution {voxel_counts}"
        vx_size_x = field_dim.x / test_field.get_voxel_counts().x
        vx_size_y = field_dim.y / test_field.get_voxel_counts().y
        vx_size_z = field_dim.z / test_field.get_voxel_counts().z
        assert vx_size_x == vx_size_y and vx_size_x == vx_size_z, f"Voxels are not isotropic! Voxel sizes: {vx_size_x}, {vx_size_y}, {vx_size_z}"
        assert abs(test_field.get_voxel_dimensions().x - vx_size_x) < 1e-5 and abs(test_field.get_voxel_dimensions().y - vx_size_y) < 1e-5 and abs(test_field.get_voxel_dimensions().z - vx_size_z) < 1e-5, f"Voxel dimensions do not match calculated voxel size! {test_field.get_voxel_dimensions()} vs. {vx_size_x}, {vx_size_y}, {vx_size_z}"
        VOXEL_SIZE_M = vx_size_x
        if USE_BEAM_PARAMETERS:
            beam_normalizer.set_voxel_size(VOXEL_SIZE_M)
        print(f"[green]Voxel resolution of dataset matches enforced resolution {voxel_counts}!")

    print(f"Voxel size: {VOXEL_SIZE_M} m")
    datamodule.prepare_data()

    # Render a single fields
    tested_count = 0
    inference_durations = []

    print(f"Model width: {model.d_model}")

    print("=== Starting Evaluation ===")
    dl = datamodule.test_dataloader()
    #meas_steps = len(dl) - 20
    warmup_steps = 20
    meas_steps = 50
    model.assert_model_on_gpu()
    with torch.no_grad():
        for batch in track(dl, description="Testing...",total=warmup_steps + meas_steps):
            batch = batch2cuda(batch)
            tested_count += 1
            if tested_count > meas_steps + warmup_steps:
                break
            for dp in dataprocessings:
                batch: TrainingInputData = dp(batch)

            out_batch, duration_ms = InferenceHelper.timed_inference_step(
                batch,
                model,
                voxel_resolution=voxel_counts,
                spectra_bins=spectra_bins
            )
            inference_durations.append(duration_ms / batch.input.direction.shape[0])  # per field
            out_batch: RadiationFieldChannel = normalizer_01.forward(out_batch)

            gt_target: RadiationField = batch.ground_truth
            gt_target = model_train_normalizer.inverse(gt_target)
            gt_target: RadiationFieldChannel = radfield_join.forward(gt_target) if isinstance(gt_target, RadiationField) else gt_target
            gt_target: RadiationFieldChannel = normalizer_01.forward(gt_target)

            gt_airkerma: Tensor = airkerma(batch.ground_truth.spectrum, batch.ground_truth.flux)
            pred_airkerma: Tensor = airkerma(out_batch.spectrum, out_batch.flux) 

            spatial_dim = pred_airkerma.shape[-3:]
            model.assert_model_on_gpu()

    print("=== Evaluation Results ===")
    inference_durations = inference_durations[warmup_steps:meas_steps + warmup_steps]  # skip first for warm-up
    print(f"{inference_durations}")
    inference_durations = np.array(inference_durations)
    mean_duration_ms = np.mean(inference_durations)
    std_duration_ms = np.std(inference_durations)
    print(f"Inference duration per field: {mean_duration_ms:.2f} ms ± {std_duration_ms:.2f} ms")
