from RadFiled3D.pytorch.datasets.processing import DataProcessing
from radfield3dnn.rftypes import TrainingInputData, RadiationField, RadiationFieldChannel, DirectionalInput
import torch
from torch.nn import functional as F


class UniformRotation(DataProcessing):
    """
    Augmentation that applies a uniform random rotation to the input data.
    """
    def __init__(self, max_rotation_angle_degrees: float = 2.5, probability: float = 0.5, rotate_input: bool = True):
        """
        Initializes the UniformRotation augmentation.

        :param max_rotation_angle_degrees: Maximum rotation angle in degrees (+/-).
        :param probability: Probability of applying the rotation.
        """
        super().__init__()
        self.max_rotation_angle_degrees = torch.tensor(max_rotation_angle_degrees, dtype=torch.float32, requires_grad=False)
        self.probability = torch.tensor(probability, dtype=torch.float32, requires_grad=False)
        self.rotate_input = rotate_input

    def _should_apply_augmentation(self) -> bool:
        """
        Determines whether the augmentation should be applied based on the defined probability.

        :return: True if the augmentation should be applied, False otherwise.
        """
        return torch.rand(1).item() < self.probability.item()

    def dataset_multiplier(self) -> float:
        return 1.0 + self.probability.item()

    def get_parameters(self) -> dict[str, float]:
        return {
            "max_rotation_angle_degrees": self.max_rotation_angle_degrees.item(),
            "probability": self.probability.item(),
            "rotate_input": self.rotate_input
        }

    def forward(self, input_data: TrainingInputData) -> TrainingInputData:
        """
        Applies the uniform random rotation to the input data.

        :param input_data: The input data to augment.
        :return: The augmented input data.
        """
        if self._should_apply_augmentation():
            if self.max_rotation_angle_degrees.device != input_data.ground_truth.flux.device:
                self.max_rotation_angle_degrees = self.max_rotation_angle_degrees.to(input_data.ground_truth.flux.device)
                self.probability = self.probability.to(input_data.ground_truth.flux.device)

            rotation_angles_xyz = (torch.rand((1, 3), dtype=torch.float32) - 0.5) * 2 * torch.deg2rad(self.max_rotation_angle_degrees)
            cos_x, sin_x = torch.cos(rotation_angles_xyz[0, 0]), torch.sin(rotation_angles_xyz[0, 0])
            cos_y, sin_y = torch.cos(rotation_angles_xyz[0, 1]), torch.sin(rotation_angles_xyz[0, 1])
            cos_z, sin_z = torch.cos(rotation_angles_xyz[0, 2]), torch.sin(rotation_angles_xyz[0, 2])
            
            R = torch.mm(
                torch.mm(
                    torch.tensor([
                        [cos_z, -sin_z, 0],
                        [sin_z, cos_z, 0],
                        [0, 0, 1]
                    ], dtype=torch.float32),
                    torch.tensor([
                        [cos_y, 0, sin_y],
                        [0, 1, 0],
                        [-sin_y, 0, cos_y]
                    ], dtype=torch.float32)
                ),
                torch.tensor([
                    [1, 0, 0],
                    [0, cos_x, -sin_x],
                    [0, sin_x, cos_x]
                ], dtype=torch.float32)
            )
            R = torch.cat([R, torch.zeros(3, 1)], dim=1).unsqueeze(0)

            grid_fluence = F.affine_grid(
                R,
                (1, *(input_data.ground_truth.scatter_field.flux.shape if isinstance(input_data.ground_truth, RadiationField) else input_data.ground_truth.flux.shape)),
                align_corners=False
            )
            grid_spectrum = F.affine_grid(
                R,
                (1, *(input_data.ground_truth.scatter_field.spectrum.shape if isinstance(input_data.ground_truth, RadiationField) else input_data.ground_truth.spectrum.shape)),
                align_corners=False
            )

            if not self.rotate_input:
                input = input_data.input
            else:
                dir = input_data.input.direction.unsqueeze(0)
                input = DirectionalInput(
                    direction=F.normalize(
                        torch.matmul(
                            dir,
                            R[:, :3, :3]  # Use only the 3x3 rotation part
                        ),
                        p=2, dim=-1
                    ).squeeze(0),
                    spectrum=input_data.input.spectrum if input_data.input.spectrum is not None else None,
                    origin=input_data.input.origin if input_data.input.origin is not None else None,
                    geometry=input_data.input.geometry if input_data.input.geometry is not None else None,
                    beam_shape_parameters=input_data.input.beam_shape_parameters if input_data.input.beam_shape_parameters is not None else None,
                    beam_shape_type=input_data.input.beam_shape_type if input_data.input.beam_shape_type is not None else None
                )

            input_data = TrainingInputData(
                input=input,
                ground_truth=RadiationField(
                    scatter_field=RadiationFieldChannel(
                        spectrum=F.grid_sample(
                            input_data.ground_truth.scatter_field.spectrum.unsqueeze(0),
                            grid_spectrum,
                            mode='bilinear',
                            padding_mode='border',
                            align_corners=False
                        ).squeeze(0),
                        flux=F.grid_sample(
                            input_data.ground_truth.scatter_field.flux.unsqueeze(0),
                            grid_fluence,
                            mode='bilinear',
                            padding_mode='border',
                            align_corners=False
                        ).squeeze(0),
                        error=F.grid_sample(
                            input_data.ground_truth.scatter_field.error.unsqueeze(0),
                            grid_fluence,
                            mode='bilinear',
                            padding_mode='border',
                            align_corners=False
                        ).squeeze(0) if input_data.ground_truth.scatter_field.error is not None else None
                    ),
                    direct_beam=RadiationFieldChannel(
                        spectrum=F.grid_sample(
                            input_data.ground_truth.direct_beam.spectrum.unsqueeze(0),
                            grid_spectrum,
                            mode='bilinear',
                            padding_mode='border',
                            align_corners=False
                        ).squeeze(0),
                        flux=F.grid_sample(
                            input_data.ground_truth.direct_beam.flux.unsqueeze(0),
                            grid_fluence,
                            mode='bilinear',
                            padding_mode='border',
                            align_corners=False
                        ).squeeze(0),
                        error=F.grid_sample(
                            input_data.ground_truth.direct_beam.error.unsqueeze(0),
                            grid_fluence,
                            mode='bilinear',
                            padding_mode='border',
                            align_corners=False
                        ).squeeze(0) if input_data.ground_truth.direct_beam.error is not None else None
                    ) if input_data.ground_truth.direct_beam is not None else None
                ) if isinstance(input_data.ground_truth, RadiationField) else RadiationFieldChannel(
                    spectrum=F.grid_sample(
                        input_data.ground_truth.spectrum,
                        grid_spectrum,
                        mode='bilinear',
                        padding_mode='border',
                        align_corners=False
                    ),
                    flux=F.grid_sample(
                        input_data.ground_truth.flux,
                        grid_fluence,
                        mode='bilinear',
                        padding_mode='border',
                        align_corners=False
                    ),
                    error=F.grid_sample(
                        input_data.ground_truth.error,
                        grid_fluence,
                        mode='bilinear',
                        padding_mode='border',
                        align_corners=False
                    ) if input_data.ground_truth.error is not None else None
                ),
                original_ground_truth=input_data.original_ground_truth
            )
            
        return input_data
