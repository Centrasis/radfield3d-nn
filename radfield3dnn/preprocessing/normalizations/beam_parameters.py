from RadFiled3D.pytorch.datasets.processing import DataProcessing
from radfield3dnn.rftypes import TrainingInputData, DirectionalInput, PositionalInput
from RadFiled3D.RadFiled3D import FieldShape
import torch


class BeamParametersNormalization(DataProcessing):
    def __init__(self, opening_angle_range_deg: tuple[float, float], size_per_voxel_m: float, is_origin_centered: bool, distance_range_m: tuple[float, float] | None = None, half_field_size: tuple[float, float, float] | None = None):
        super().__init__()
        self.opening_angle_range = (float(torch.deg2rad(torch.tensor(opening_angle_range_deg[0], dtype=torch.float32))), float(torch.deg2rad(torch.tensor(opening_angle_range_deg[1], dtype=torch.float32))))  # in radians
        assert self.opening_angle_range[0] > 0 and self.opening_angle_range[1] > 0, f"Expected opening_angle_range_deg to be > 0, but got {opening_angle_range_deg}"
        assert size_per_voxel_m > 0, f"Expected size_per_voxel_m to be > 0, but got {size_per_voxel_m}"
        self.size_per_voxel = size_per_voxel_m  # in meters
        self.is_origin_centered = is_origin_centered
        with torch.no_grad():
            if distance_range_m is not None:
                self.distance_range = (float(torch.tensor(distance_range_m[0], dtype=torch.float32)), float(torch.tensor(distance_range_m[1], dtype=torch.float32)))  # in meters
                assert self.distance_range[0] > 0 and self.distance_range[1] > 0, f"Expected distance_range_m to be > 0, but got {distance_range_m}"
            else:
                self.distance_range = None
            if half_field_size is not None:
                assert len(half_field_size) == 3, f"Expected half_field_size to be a tuple of 3 floats, but got {half_field_size}"
                self.half_field_size = half_field_size
            else:
                self.half_field_size = None

    def forward(self, x: TrainingInputData) -> TrainingInputData:
        """
        Apply data processing to the input data.
        """
        with torch.no_grad():
            if self.half_field_size is not None:
                half_field_size = torch.tensor(self.half_field_size, dtype=torch.float32, device=x.input.origin.device)
                half_field_size = half_field_size.unsqueeze(0).expand(x.input.origin.shape[0], -1)  # expand to batch size

            if self.distance_range is not None:
                distance_range = torch.tensor(self.distance_range, dtype=torch.float32, device=x.input.origin.device)
            else:
                size_per_voxel = self.size_per_voxel
                distance_range = torch.tensor(
                    [
                        0.0,
                        torch.linalg.norm(torch.tensor(x.ground_truth.flux.shape[-3:], dtype=torch.float32, device=x.input.origin.device) * size_per_voxel, dim=-1, ord=2)  # in meters
                    ],
                    dtype=torch.float32,
                    device=x.input.origin.device
                )
            if self.is_origin_centered:
                distance_range = distance_range / 2.0

            # reverse field-relative normalization of origin from dataloader
            origin = (x.input.origin * 2 * half_field_size) - half_field_size if self.half_field_size is not None else x.input.origin
            origin_distance = torch.linalg.norm(origin, dim=1)  # in meters
            origin_distance = (origin_distance - distance_range[0]) / (distance_range[1] - distance_range[0])  # normalize to [0, 1] depending on is_origin_centered
            origin_distance[(origin_distance > 1.0) & torch.isclose(origin_distance, torch.tensor(1.0, device=origin_distance.device))] = 1.0  # ensure max value is exactly 1.0 avoiding floating point issues
            origin_distance[(origin_distance < 0.0) & torch.isclose(origin_distance, torch.tensor(0.0, device=origin_distance.device))] = 0.0  # ensure min value is exactly 0.0 avoiding floating point issues
            assert (origin_distance >= 0).all() and (origin_distance <= 1.0).all(), f"Expected origin distances to be in [{distance_range[0].item()}, {distance_range[1].item()}] meters, but got min { (origin_distance).min().item()} and max {(origin_distance).max().item()} meters."

            x = TrainingInputData(
                # _replace keeps the concrete input type (Directional/Positional/TranslationalInput)
                # and every field this normalizer does not touch.
                input=x.input._replace(
                    origin=origin_distance.unsqueeze(-1),
                    beam_shape_parameters=x.input.beam_shape_parameters.clone(),  # clone: the CONE block normalizes this in place; never mutate the shared/cached source
                ),
                ground_truth=x.ground_truth,
                original_ground_truth=x.original_ground_truth if hasattr(x, 'original_ground_truth') else None
            )
            

            if FieldShape(int(x.input.beam_shape_type[0, 0].item())) == FieldShape.CONE and (x.input.beam_shape_type[:, 0] == float(int(FieldShape.CONE))).all():
                opening_angle_range = torch.tensor(self.opening_angle_range, dtype=torch.float32, device=x.input.beam_shape_parameters.device)
                opening_angle_range = opening_angle_range.unsqueeze(0).expand(x.input.beam_shape_parameters.shape[0], -1)  # expand to batch size
                assert len(x.input.beam_shape_parameters.shape) == 2 and x.input.beam_shape_parameters.shape[1] == 1, f"Expected beam_shape_parameters to have shape (N, 1) for CONE shape, but got {x.input.beam_shape_parameters.shape}"
                # batched-safe: per-sample range, zero out degenerate (min==max) samples.
                denom = opening_angle_range[:, 1] - opening_angle_range[:, 0]   # (B,)
                normed = (torch.deg2rad(x.input.beam_shape_parameters[:, 0]) - opening_angle_range[:, 0]) / denom.clamp_min(1e-8)
                x.input.beam_shape_parameters[:, 0] = torch.where(denom > 1e-8, normed, torch.zeros_like(normed))  # normalize to [0, 1]
                x.input.beam_shape_parameters[:, 0][(x.input.beam_shape_parameters[:, 0] > 1.0) & torch.isclose(x.input.beam_shape_parameters[:, 0], torch.tensor(1.0, device=x.input.beam_shape_parameters.device))] = 1.0  # ensure max value is exactly 1.0 avoiding floating point issues
                x.input.beam_shape_parameters[:, 0][(x.input.beam_shape_parameters[:, 0] < 0.0) & torch.isclose(x.input.beam_shape_parameters[:, 0], torch.tensor(0.0, device=x.input.beam_shape_parameters.device))] = 0.0  # ensure min value is exactly 0.0 avoiding floating point issues
                assert (x.input.beam_shape_parameters[:, 0] >= 0).all() and (x.input.beam_shape_parameters[:, 0] <= 1.0).all(), f"Expected beam opening angles to be in [0.0, 1.0], but got min {x.input.beam_shape_parameters[:, 0].min().item()} and max {x.input.beam_shape_parameters[:, 0].max().item()} after normalization."
            elif (x.input.beam_shape_type[:, 0] == float(int(FieldShape.RECTANGLE))).all():
                # RECTANGLE collimation: beam_shape_parameters is the field rectangle (x, y) in
                # metres. Normalize both dimensions by the full field extent so the collimation
                # size reaches the network in [0, 1], the same scale as the other beam parameters.
                assert x.input.beam_shape_parameters.shape[1] == 2, f"Expected beam_shape_parameters to have shape (N, 2) for RECTANGLE shape, but got {x.input.beam_shape_parameters.shape}"
                if self.half_field_size is not None:
                    field_extent = float(max(2.0 * s for s in self.half_field_size))
                else:
                    field_extent = float(torch.linalg.norm(torch.tensor(x.ground_truth.flux.shape[-3:], dtype=torch.float32, device=x.input.beam_shape_parameters.device) * self.size_per_voxel))
                x.input.beam_shape_parameters[:, :2] = (x.input.beam_shape_parameters[:, :2] / field_extent).clamp(0.0, 1.0)
                assert (x.input.beam_shape_parameters[:, :2] >= 0).all() and (x.input.beam_shape_parameters[:, :2] <= 1.0).all(), f"Expected rectangle dimensions to be in [0.0, 1.0], but got min {x.input.beam_shape_parameters[:, :2].min().item()} and max {x.input.beam_shape_parameters[:, :2].max().item()} after normalization."
            else:
                pass
            return x

    @classmethod
    def from_dataset_definition(cls, definition, size_per_voxel_m: float | None = None, is_origin_centered: bool = False) -> 'BeamParametersNormalization':
        """Build from the dataset definition JSON (path or already-parsed dict) the dataset was
        GENERATED with — the authoritative source for the parameter ranges, available even when no
        statistics.json was computed. Reads ``Parameters`` (source_distance, source_opening_angle)
        and ``Metaparameters`` (WorldDim, VoxelSize).

        source_opening_angle is a per-axis rectangle range ([[w_min, h_min], [w_max, h_max]], in
        metres) for rectangle collimation — unused by the RECTANGLE normalization branch, which
        scales by field extent — or a scalar [deg_min, deg_max] range for cone beams."""
        import json as _json
        if not isinstance(definition, dict):
            with open(definition, "r") as f:
                definition = _json.load(f)
        params = {p["name"]: p for p in definition.get("Parameters", [])}
        meta = definition.get("Metaparameters", {})

        assert "source_distance" in params and "range" in params["source_distance"], \
            "Dataset definition lacks a source_distance range"
        distance_range_m = tuple(float(v) for v in params["source_distance"]["range"])

        opening_angle_range_deg = (1.0, 1.0)  # harmless placeholder; only the CONE branch reads it
        oa = params.get("source_opening_angle", {}).get("range")
        if oa is not None and not isinstance(oa[0], (list, tuple)):
            opening_angle_range_deg = (float(oa[0]), float(oa[1]))  # cone: scalar degree range

        world_dim = meta.get("WorldDim")
        half_field_size = tuple(float(d) / 2.0 for d in world_dim) if world_dim is not None else None
        if size_per_voxel_m is None:
            size_per_voxel_m = float(meta.get("VoxelSize", 0.0))
        assert size_per_voxel_m > 0, "VoxelSize missing from definition and size_per_voxel_m not given"

        return cls(
            opening_angle_range_deg=opening_angle_range_deg,
            size_per_voxel_m=size_per_voxel_m,
            is_origin_centered=is_origin_centered,
            distance_range_m=distance_range_m,
            half_field_size=half_field_size,
        )

    @staticmethod
    def create_from_config(config: dict) -> 'BeamParametersNormalization':
        """
        Create a BeamParametersNormalization instance from a configuration dictionary as provided by self.get_parameters().
        """
        return BeamParametersNormalization(
            opening_angle_range_deg=config["opening_angle_range_deg"],
            size_per_voxel_m=config["size_per_voxel_m"],
            is_origin_centered=config["is_origin_centered"],
            distance_range_m=config["distance_range_m"],
            half_field_size=config["half_field_size_m"]
        )

    def set_voxel_size(self, size_per_voxel_m: float):
        """
        Set the size per voxel in meters.
        """
        #with torch.no_grad():
            #self.size_per_voxel.fill_(float(size_per_voxel_m))
        self.size_per_voxel = size_per_voxel_m

    def get_parameters(self):
        """
        Get the parameters for the current processing module as a dictionary
        such that each key is the name of a parameter and each value is the value of the parameter.
        This way, the parameters can be easily logged.
        """
        return {
            "opening_angle_range_deg": (float(torch.rad2deg(torch.tensor(self.opening_angle_range[0], dtype=torch.float32))), float(torch.rad2deg(torch.tensor(self.opening_angle_range[1], dtype=torch.float32)))),
            "size_per_voxel_m": self.size_per_voxel,
            "is_origin_centered": self.is_origin_centered,
            "distance_range_m": self.distance_range,
            "half_field_size_m": self.half_field_size
        }
