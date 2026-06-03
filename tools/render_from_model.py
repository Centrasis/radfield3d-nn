import torch
from models import ModelConstructor
import os
import argparse
from RadFiled3D.RadFiled3D import CartesianRadiationField, FieldStore, DType, vec3
from RadFiled3D.metadata.v1 import Metadata
from rich.progress import track
import numpy as np
from scipy.spatial.transform import Rotation as R
from RadFiled3D.pytorch.types import DirectionalInput, RadiationField
from normalizations.linear import LinearNormalizer
from normalizations.lognormalizer import LogNormalizer
from normalizations.base import Normalizer
from torch import Tensor
from utils.samplings.mean_sampling import resample_histogram_means
import pandas as pd


def read_spectrum(file_path, bins_count: int = 150, bin_width_eV: int = 1000) -> Tensor:
    if file_path.endswith(".csv"):
        data = pd.read_csv(file_path, comment='#', delimiter=" ")
        data: Tensor = torch.tensor(data.to_numpy(), dtype=torch.float32)
    else:
        data: Tensor = torch.load(file_path)
        if data.shape[0] == 2:
            data = data.permute(1, 0)
    data = data[:, ~torch.isnan(data).any(dim=0)]  # Remove columns with NaN values
    data = data[:, [0, -1]]
    bins = torch.arange(0, bins_count * bin_width_eV, bin_width_eV, dtype=torch.float32)
    data = resample_histogram_means(data, bins, False)
    data[:, 1] = torch.where(~torch.isnan(data[:, 1]), data[:, 1], 0.0)
    data = data[:, 1] / data[:, 1].sum()
    data = data.unsqueeze(0)
    return data


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="model.ckpt")
    parser.add_argument("--output_path", type=str, default="./output/")
    parser.add_argument("--tube_spectrum", type=str, default=None, help="Path to the tube spectrum *.csv or *.spectrum file.")
    args = parser.parse_args()

    #if torch.cuda.is_available():
    #    torch.set_default_dtype(torch.FloatTensor)
    #else:
    #    torch.set_default_dtype(torch.FloatTensor)
    model_weights_path = args.model_path
    output_path = args.output_path
    tube_spectrum: Tensor = args.tube_spectrum
    if tube_spectrum is not None:
        tube_spectrum = os.path.abspath(tube_spectrum)
        if not os.path.exists(tube_spectrum):
            raise FileNotFoundError(f"Tube spectrum file {tube_spectrum} does not exist.")
        tube_spectrum = read_spectrum(tube_spectrum)

    if not os.path.isabs(output_path):
        output_path = os.path.join(os.getcwd(), output_path)
    model_config_path = os.path.splitext(model_weights_path)[0] + ".config"

    model_cls = ModelConstructor.create_model_from_config(model_config_path)
    model = model_cls.load_from_checkpoint(model_weights_path)
    model.eval()
    model.cuda()

    model_train_normalizer = model._normalizer.clone().to(model.device).eval()

    field = CartesianRadiationField(vec3(1.0, 1.0, 1.0), vec3(0.02, 0.02, 0.02))
    voxel_counts = field.get_voxel_counts()
    
    scatter_channel_spec_array: np.ndarray = None # scatter_channel.get_layer_as_ndarray("spectrum")
    scatter_channel_hits_array: np.ndarray = None # scatter_channel.get_layer_as_ndarray("hits")
    xray_channel_spec_array: np.ndarray = None # xray_channel.get_layer_as_ndarray("spectrum")
    xray_channel_hits_array: np.ndarray = None # xray_channel.get_layer_as_ndarray("hits")

    # Render a single image
    with torch.no_grad():
        for alpha in range(-90, 270, 10):
        #for alpha in [0.0, 45.0, 90.0, 135.0, 180.0, 225.0, 270.0, 315.0]:
            for beta in [0.0]:
            #for beta in [0.0, 45.0, 90.0, 135.0, 180.0, 225.0, 270.0, 315.0]:
                direction = np.array([0.0, 0.0, -1.0], dtype=np.float32)
                alpha_radians = np.radians(alpha)
                beta_radians = np.radians(beta)

                r_y = R.from_euler('y', alpha_radians)
                r_x = R.from_euler('x', beta_radians)
                rotation = r_x * r_y

                rotated_direction = rotation.apply(direction)
                direction = torch.tensor(rotated_direction, device="cpu", dtype=torch.float32)
                direction = direction / torch.norm(direction)

                input = DirectionalInput(
                    direction=direction.clone().to(model.device).unsqueeze(0),
                    spectrum=tube_spectrum.clone().to(model.device) if tube_spectrum is not None else None,
                    geometry=None,
                    origin=torch.tensor([0.0, -2.5, 0.0], device=model.device).unsqueeze(0),
                    beam_shape_parameters=torch.tensor([5.0], device=model.device, dtype=torch.float32).unsqueeze(0), # 5deg opening angle
                    beam_shape_type=torch.tensor([0], device=model.device, dtype=torch.int32).unsqueeze(0)  # cone beam
                )
                pred_field = model.forward2volume(input, torch.tensor([voxel_counts.x, voxel_counts.y, voxel_counts.z]).to(model.device))

                if scatter_channel_spec_array is None and pred_field.scatter_field is not None:
                    scatter_channel = field.add_channel("scatter_field")
                    scatter_channel.add_histogram_layer("spectrum", 32, 1.5e+5 / 32, "Gy/s")
                    scatter_channel.add_layer("hits", "hits", DType.FLOAT32)
                    scatter_channel.get_layer_as_ndarray("hits")[:, :, :] = 1.0
                    scatter_channel.add_layer("energy", "eV", DType.FLOAT32)
                    scatter_channel.get_layer_as_ndarray("energy")[:, :, :] = 100000.0
                    scatter_channel.add_layer("error", "", DType.FLOAT32)
                    scatter_channel.get_layer_as_ndarray("error")[:, :, :] = 0.0
                    scatter_channel_spec_array = scatter_channel.get_layer_as_ndarray("spectrum")
                    scatter_channel_hits_array = scatter_channel.get_layer_as_ndarray("hits")
                
                if xray_channel_spec_array is None and pred_field.direct_beam is not None:
                    xray_beam_channel = field.add_channel("direct_beam")
                    xray_beam_channel.add_histogram_layer("spectrum", 32, 1.5e+5 / 32, "Gy/s")
                    xray_beam_channel.add_layer("hits", "hits", DType.FLOAT32)
                    xray_beam_channel.get_layer_as_ndarray("hits")[:, :, :] = 1.0
                    xray_beam_channel.add_layer("energy", "eV", DType.FLOAT32)
                    xray_beam_channel.get_layer_as_ndarray("energy")[:, :, :] = 100000.0
                    xray_beam_channel.add_layer("error", "", DType.FLOAT32)
                    xray_beam_channel.get_layer_as_ndarray("error")[:, :, :] = 0.0
                    xray_channel_spec_array = xray_beam_channel.get_layer_as_ndarray("spectrum")
                    xray_channel_hits_array = xray_beam_channel.get_layer_as_ndarray("hits")

                total_fluence = torch.zeros_like(pred_field.scatter_field.flux).cpu().numpy()
                if pred_field.scatter_field is not None:
                    if pred_field.scatter_field.spectrum is not None:
                        spec = pred_field.scatter_field.spectrum.squeeze(0)
                    else:
                        spec = torch.ones(32, voxel_counts.x, voxel_counts.y, voxel_counts.z, device=model.device)
                        spec /= spec.sum(0)

                    scatter_channel_spec_array[:] = spec.permute(1, 2, 3, 0).cpu().numpy()
                    flux = model_train_normalizer.inverse(pred_field.scatter_field).flux.cpu()
                    total_fluence += flux.numpy()
                    scatter_channel_hits_array[:] = flux.squeeze(0).numpy()
                
                if pred_field.direct_beam is not None:
                    if pred_field.direct_beam.spectrum is not None:
                        spec = pred_field.direct_beam.spectrum.squeeze(0)
                    else:
                        spec = torch.ones(32, voxel_counts.x, voxel_counts.y, voxel_counts.z, device=model.device)
                        spec /= spec.sum(0)

                    xray_channel_spec_array[:] = spec.permute(1, 2, 3, 0).cpu().numpy()
                    flux = model_train_normalizer.inverse(pred_field.direct_beam).flux.cpu()
                    total_fluence += flux.numpy()
                    xray_channel_hits_array[:] = flux.squeeze(0).numpy()

                max_fluence = np.max(total_fluence)
                if max_fluence > 0.0:
                    if pred_field.scatter_field is not None:
                        scatter_channel_hits_array[:] /= max_fluence
                    if pred_field.direct_beam is not None:
                        xray_channel_hits_array[:] /= max_fluence
                else:
                    print("Max flux is zero, skipping normalization.")

                if not os.path.exists(output_path):
                    os.makedirs(output_path)

                FieldStore.store(field, Metadata(
                        Metadata.Header.Simulation(
                            0,
                            "Alderson-H-100",
                            model_cls.__model_name__,
                            Metadata.Header.XRayTube(
                                vec3(direction[0], direction[1], direction[2]),
                                vec3(0.0, -2.5, 0.0),
                                1.5e+4,
                                "XRayTube"
                            )
                        ),
                        Metadata.Header.Software(
                            "RenderFromModel",
                            "0.0.1",
                            "",
                            ""
                        )
                    ),
                    #os.path.join(output_path, f"rendered_field_{alpha}_{beta}.rf3")
                    os.path.join(output_path, f"RF_100.0keV_{alpha}_{beta}_Alderson_Headless_wLung.rf3")
                )
