from radfield3dnn.rftypes import RadiationFieldChannel, RadiationField, TrainingInputData, DirectionalInput, rf3TrainingInputData, rf3RadiationField
from typing import Union
from RadFiled3D.pytorch.datasets.processing import DataProcessing
from torch import Tensor


class CropDataset(DataProcessing):
    """
    CropDataset is an augmentation that crops the input tensor and radiation field to a specified size.
    It is useful for reducing the size of the input data and focusing on a specific region of interest.
    Args:
        crop_size (tuple[int, int, int]): The size to crop the input tensor and radiation field to.
    """

    def __init__(self, crop_size: tuple[int, int, int] = (50, 50, 50)):
        super().__init__()
        self.crop_size = crop_size

    def crop_tensor(self, tensor: Tensor) -> Tensor:
        """
        Crop the tensor to the specified crop size.
        Args:
            tensor (Tensor): Input tensor of shape (C, D, H, W).
        Returns:
            Tensor: Cropped tensor of shape (C, crop_size[0], crop_size[1], crop_size[2]).
        """
        batch_size = tensor.shape[0] if tensor.ndim == 5 else 0
        field_shape = tensor.shape[1:] if tensor.ndim == 5 else tensor.shape
        C, D, H, W = field_shape
        diff_D = D - self.crop_size[0]
        diff_H = H - self.crop_size[1]
        diff_W = W - self.crop_size[2]
        start_D = diff_D // 2
        start_H = diff_H // 2
        start_W = diff_W // 2
        if batch_size > 0:
            return tensor[:, :, start_D:start_D + self.crop_size[0], start_H:start_H + self.crop_size[1], start_W:start_W + self.crop_size[2]]
        else:
            return tensor[:, start_D:start_D + self.crop_size[0], start_H:start_H + self.crop_size[1], start_W:start_W + self.crop_size[2]]

    def crop_channel(self, field: RadiationFieldChannel) -> RadiationFieldChannel:
        """
        Crop the RadiationFieldChannel to the specified crop size.
        Args:
            field (RadiationFieldChannel): Input RadiationFieldChannel.
        Returns:
            RadiationFieldChannel: Cropped RadiationFieldChannel.
        """
        cropped_spectrum = self.crop_tensor(field.spectrum)
        cropped_flux = self.crop_tensor(field.flux)
        cropped_error = self.crop_tensor(field.error) if field.error is not None else None
        return RadiationFieldChannel(
            spectrum=cropped_spectrum,
            flux=cropped_flux,
            error=cropped_error
        )
    
    def crop_radiation_field(self, field: RadiationField) -> RadiationField:
        """
        Crop the RadiationField to the specified crop size.
        Args:
            field (RadiationField): Input RadiationField.
        Returns:
            RadiationField: Cropped RadiationField.
        """
        direct_beam_cropped = self.crop_channel(field.direct_beam) if field.direct_beam is not None else None
        scatter_field_cropped = self.crop_channel(field.scatter_field) if field.scatter_field is not None else None
        return RadiationField(
            direct_beam=direct_beam_cropped,
            scatter_field=scatter_field_cropped,
            geometry=self.crop_tensor(field.geometry) if "geometry" in field._fields and field.geometry is not None else None
        )
    
    def crop_input_data(self, input_data: DirectionalInput) -> DirectionalInput:
        if input_data.geometry is not None:
            return DirectionalInput(
                direction=input_data.direction,
                spectrum=input_data.spectrum,
                geometry=self.crop_tensor(input_data.geometry) if input_data.geometry is not None else None,
                origin=input_data.origin if input_data.origin is not None else None,
                beam_shape_parameters=input_data.beam_shape_parameters if input_data.beam_shape_parameters is not None else None,
                beam_shape_type=input_data.beam_shape_type if input_data.beam_shape_type is not None else None
            )
        return input_data
    
    def forward(self, x: Union[TrainingInputData, RadiationField]) -> Union[TrainingInputData, RadiationFieldChannel]:
        if isinstance(x, TrainingInputData) or isinstance(x, rf3TrainingInputData):
            ogt = None
            if hasattr(x, "original_ground_truth"):
                ogt = self.forward(x.original_ground_truth) if x.original_ground_truth is not None else None
            return TrainingInputData(
                input=self.crop_input_data(x.input),
                ground_truth=self.forward(x.ground_truth),
                original_ground_truth=ogt
            )
        elif isinstance(x, RadiationField) or isinstance(x, rf3RadiationField):
            return self.crop_radiation_field(x)
        elif isinstance(x, RadiationFieldChannel):
            return self.crop_channel(x)
        else:
            raise TypeError(f"Unsupported type: {type(x)}. Expected TrainingInputData, RadiationField, or RadiationFieldChannel.")

    @classmethod
    def create_from_config(cls, config: dict) -> "CropDataset":
        return CropDataset(
            crop_size=tuple(config.get("crop_size", (50, 50, 50)))
        )

    def get_parameters(self) -> dict[str, Union[tuple[int, int, int]]]:
        return {
            "crop_size": self.crop_size
        }