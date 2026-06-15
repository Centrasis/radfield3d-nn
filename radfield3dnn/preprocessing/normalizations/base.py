from typing import Union
from radfield3dnn.rftypes import RadiationField, TrainingInputData, RadiationFieldChannel, AirKermaField
from RadFiled3D.pytorch.types import TrainingInputData as rf3TrainingInputData, RadiationField as rf3RadiationField
from RadFiled3D.pytorch.datasets.processing import DataProcessing
from torch import Tensor


class Normalizer(DataProcessing):
    def __init__(self):
        super().__init__()
        self.device = None

    def to(self, *args, **kwargs):
        self.device = args[0] if len(args) > 0 else kwargs.get('device', self.device)
        return super().to(*args, **kwargs)

    def forward(self, x: Union[TrainingInputData, RadiationField, RadiationFieldChannel, AirKermaField, Tensor], respect_to: Union[TrainingInputData, RadiationField, RadiationFieldChannel, AirKermaField, Tensor, None] = None) -> Union[TrainingInputData, RadiationField, RadiationFieldChannel, Tensor, AirKermaField]:
        assert respect_to is None or (x.__class__ == respect_to.__class__), "respect_to must be of the same type as x or None."
        
        if isinstance(x, (TrainingInputData, rf3TrainingInputData)):
            return TrainingInputData(
                input=x.input,
                ground_truth=self.forward(x.ground_truth, respect_to.ground_truth if respect_to is not None else None),
                original_ground_truth=x.original_ground_truth if hasattr(x, "original_ground_truth") else None
            )
        elif isinstance(x, (RadiationField, rf3RadiationField)):
            return RadiationField(
                scatter_field=self.forward(x.scatter_field, respect_to=respect_to.scatter_field if respect_to is not None else None) if x.scatter_field is not None else None,
                direct_beam=self.forward(x.direct_beam, respect_to=respect_to.direct_beam if respect_to is not None else None) if x.direct_beam is not None else None,
            )
        elif isinstance(x, RadiationFieldChannel):
            return RadiationFieldChannel(
                flux=self.forward(x.flux, respect_to=respect_to.flux if respect_to is not None else None),
                spectrum=x.spectrum,
                error=x.error
            )
        elif isinstance(x, Tensor):
            return self.apply_transformation(x, respect_to=respect_to)
        elif isinstance(x, AirKermaField):
            return AirKermaField(
                air_kerma=self.forward(x.air_kerma, respect_to=respect_to.air_kerma if respect_to is not None else None),
                geometry=x.geometry
            )
        else:
            raise TypeError(f"Unsupported type for normalization: {type(x)}")

    def apply_transformation(self, x: Tensor, respect_to: Union[Tensor, None]) -> Tensor:
        raise NotImplementedError("This method must be implemented in a subclass.")

    def inverse(self, x: Union[TrainingInputData, RadiationField, RadiationFieldChannel, AirKermaField, Tensor], respect_to: Union[TrainingInputData, RadiationField, RadiationFieldChannel, AirKermaField, Tensor, None] = None) -> Union[TrainingInputData, RadiationField, RadiationFieldChannel, AirKermaField, Tensor]:
        assert respect_to is None or (x.__class__ == respect_to.__class__), "respect_to must be of the same type as x or None."
        
        if isinstance(x, TrainingInputData):
            return TrainingInputData(
                input=x.input,
                ground_truth=self.inverse(x.ground_truth, respect_to.ground_truth if respect_to is not None else None),
                original_ground_truth=x.original_ground_truth if x.original_ground_truth is not None else None
            )
        elif isinstance(x, RadiationField):
            return RadiationField(
                scatter_field=self.inverse(x.scatter_field, respect_to=respect_to.scatter_field if respect_to is not None else None) if x.scatter_field is not None else None,
                direct_beam=self.inverse(x.direct_beam, respect_to=respect_to.direct_beam if respect_to is not None else None) if x.direct_beam is not None else None,
            )
        elif isinstance(x, RadiationFieldChannel):   
            return RadiationFieldChannel(
                flux=self.inverse(x.flux, respect_to=respect_to.flux if respect_to is not None else None),
                spectrum=x.spectrum,
                error=x.error
            )
        elif isinstance(x, Tensor):
            return self.apply_inverse_transformation(x, respect_to=respect_to)
        elif isinstance(x, AirKermaField):
            return AirKermaField(
                air_kerma=self.inverse(x.air_kerma, respect_to=respect_to.air_kerma if respect_to is not None else None),
                geometry=x.geometry
            )
        else:
            raise TypeError(f"Unsupported type for inverse normalization: {type(x)}")
        
    def apply_inverse_transformation(self, x: Tensor, respect_to: Union[Tensor, None]) -> Tensor:
        raise NotImplementedError("This method must be implemented in a subclass.")

    def validate_range(self, x: Tensor):
        """
        Validate that the input tensor x is suitable for normalization.
        This method should be overridden in subclasses to implement specific validation logic.
        :param x: Input tensor to validate.
        :raises ValueError: If the input tensor does not meet the validation criteria.
        """
        raise NotImplementedError("This method must be implemented in a subclass.")
    
    def get_type(self) -> str:
        return self.__class__.__name__

    def clone(self) -> "Normalizer":
        """
        Safe clone:
        Returns a new instance of the same class, but with standard initialization.
        Must be overridden in subclasses if they have specific parameters.
        Not the intended solution, but pytorch is otherwise interfering with registered buffers when using copy.deepcopy or load_state_dict.
        """
        new_inst = type(self)()
        return new_inst
