from radfield3dnn.normalizations.base import Normalizer
from .base import BaseNeuralRadFieldModel
from .cnn import *
from .nerf import *
from .feedforward import *
from typing import Literal, Union, Type, List
import json
from radfield3dnn.rftypes import PositionalInput
import os
from rich import print
from enum import Enum


class ModelConstructor:
    @staticmethod
    def get_subclasses(cls) -> List[Type[BaseNeuralRadFieldModel]]:
        list_of_subclasses = []
        for subclass in cls.__subclasses__():
            list_of_subclasses.append(subclass)
            list_of_subclasses += ModelConstructor.get_subclasses(subclass)
        return list_of_subclasses

    @staticmethod
    def construct(name: str, **kwargs):
        model = None
        
        for cls in ModelConstructor.get_subclasses(BaseNeuralRadFieldModel):
            if "__model_name__" in cls.__dict__ and cls.__model_name__ == name:
                model = cls(**kwargs)
                break

        if model is None:
            raise ValueError(f"Model {name} not found.")
        return model
        
    @staticmethod
    def get_model_names():
        return ["NeuralRadiationField", "NeDF", "DeConvFluence", "FluenceNeRF"]
    
    @staticmethod
    def get_dataset_type_for_model(name: str) -> Union[Literal["Voxelwise"], Literal["Layerwise"]]:
        # currently always use Layerwise
        return "Layerwise"

    @staticmethod
    def create_model_with_defaults(name: str, **kwargs) -> Type[BaseNeuralRadFieldModel]:
        base_model = ModelConstructor.construct(name, **kwargs)
        
        class DerivedModel(type(base_model)):
            def __init__(self, **init_kwargs):
                defaults = {**kwargs, **init_kwargs}
                super().__init__(**defaults)
                self.__class__.__name__ = base_model.__class__.__name__
        
        return DerivedModel

    @staticmethod
    def create_model_from_config(config_path: str, normalizer: Normalizer) -> Type[BaseNeuralRadFieldModel]:
        with open(config_path, "r") as f:
            config = json.load(f)
        config["parameters"]["normalizer"] = normalizer
        return ModelConstructor.create_model_from_dict(config)
    
    @staticmethod
    def create_model_from_dict(config: dict, normalizer: Normalizer = None) -> Type[BaseNeuralRadFieldModel]:
        if normalizer is not None and "normalizer" not in config["parameters"]:
            config["parameters"]["normalizer"] = normalizer
        return ModelConstructor.create_model_with_defaults(config["model_name"], **config["parameters"])

    @staticmethod
    def load_model_from(path: str) -> BaseNeuralRadFieldModel:
        if not os.path.exists(path):
            raise ValueError(f"Could not find model file: {path}")

        ext = os.path.splitext(path)[1]
        if ext.lower() == ".json":
            config_path = path
            weight_path = os.path.splitext(path)[0] + ".pt"
            if not os.path.exists(weight_path):
                weight_path = os.path.splitext(path)[0] + ".ckpt"
            if not os.path.exists(weight_path):
                raise ValueError(f"Could not find weights file corresponding to given config file: {path}")
        elif ext.lower() in [".pt", ".ckpt"]:
            weight_path = path
            config_path = os.path.splitext(path)[0] + ".json"
            if not os.path.exists(config_path):
                raise ValueError(f"Could not find config file corresponding to given weights file: {path}")
        else:
            raise ValueError(f"Unknown extension ({ext}) for loading model from: {path}")
        
        model_config = json.load(open(config_path, "r"))
        model_cls = ModelConstructor.create_model_from_dict(
            model_config if "parameters" in model_config else {
                "parameters": model_config["hyper_parameters"],
                "model_name": model_config["model_name"]
            }
        )
        try:
            model = model_cls.load_from_checkpoint(weight_path)
        except Exception as e:
            print(f"[yellow]Weigths file was probably not a pytorch-lightning checkpoint: {e} [/yellow]")
            model = model_cls()
            try:
                model.load_state_dict(torch.load(weight_path))
            except:
                # Try loading with a known pure python encoding and restore intended encoding afterwards
                from radfield3dnn.encodings.sinusoidal_encoding import SinusoidalFrequencyEncoding
                pen = model.positional_location_encoding
                model.positional_location_encoding = SinusoidalFrequencyEncoding(
                    pos_enc_dim=pen.pos_enc_dim,
                    d_input=pen.d_input,
                    append_input=pen.append_input,
                    dim=-1,
                    use_tcnn=False
                )
                model.load_state_dict(torch.load(weight_path))
                model.positional_location_encoding = pen
        return model


class ModelFormat(Enum):
    ONNX = 0
    TORCH_SCRIPT = 1
    TENSOR_RT = 2


class ModelExporter:
    @staticmethod
    def export(model: BaseNeuralRadFieldModel, path: str, format: ModelFormat = ModelFormat.ONNX):
        if format == ModelFormat.ONNX:
            ModelExporter.onnx_export(model, path)
        elif format == ModelFormat.TORCH_SCRIPT:
            ModelExporter.ts_export(model, path)
        elif format == ModelFormat.TENSOR_RT:
            ModelExporter.rt_export(model, path)
        else:
            raise ValueError(f"Unknown format to exprot model: {format}")

    @staticmethod
    def rt_export(model: BaseNeuralRadFieldModel, path: str):
        raise NotImplementedError()

    @staticmethod
    def ts_export(model: BaseNeuralRadFieldModel, path: str):
        scripted = torch.jit.script(model.get_core_model())
        scripted.save(path)

    @staticmethod
    def onnx_export(model: BaseNeuralRadFieldModel, path: str):
        class ModelONNXDectorator(BaseNeuralRadFieldModel):
            def __init__(self, decoratee: BaseNeuralRadFieldModel):
                super().__init__()
                self._decoratee = decoratee

            def forward(self, direction: Tensor, position: Tensor, spectrum: Tensor, origin: Tensor, beam_shape_parameters: Tensor, beam_shape_type: Tensor, geometry: Tensor):
                return self._decoratee.forward(
                    PositionalInput(
                        direction=direction,
                        beam_shape_parameters=beam_shape_parameters,
                        beam_shape_type=beam_shape_type,
                        position=position,
                        origin=origin,
                        geometry=geometry,
                        spectrum=spectrum
                    )
                )
            
            def forward2volume(self, x, voxel_counts, spectra_bins = 32, mask = None):
                return self._decoratee.forward2volume(x, voxel_counts, spectra_bins, mask)
            
            def forward2volume_from_training_input(self, batch, voxel_counts = None, spectra_bins = 32):
                return self._decoratee.forward2volume_from_training_input(batch, voxel_counts, spectra_bins)
            
            def evaluate_forward(self, batch: TrainingInputData):
                return self._decoratee.evaluate_forward(batch)
            
            def get_model_config(self) -> dict:
                return self._decoratee.get_model_config()

            def get_custom_parameters(self) -> dict:
                return self._decoratee.get_custom_parameters()
            
            lr = property(lambda x: x._decoratee.get_lr(), lambda x, v: x._decoratee.set_lr(v))
            learning_rate = property(lambda x: x._decoratee.get_lr(), lambda x, v: x._decoratee.set_lr(v))

        input_tuple: PositionalInput = model._generate_random_input(model.device)

        model = ModelONNXDectorator(model.get_core_model())
        torch.onnx.export(
            model=model,
            args=(
                input_tuple.direction,
                input_tuple.position,
                input_tuple.spectrum,
                input_tuple.origin,
                input_tuple.beam_shape_parameters,
                input_tuple.beam_shape_type,
                input_tuple.geometry
            ),
            input_names=[
                "direction",
                "position",
                "spectrum",
                "origin",
                "beam_shape_parameters",
                "beam_shape_type",
                "geometry"
            ],
            f=path,
            dynamo=True
        )
