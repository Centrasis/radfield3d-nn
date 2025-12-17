from radfield3dnn.normalizations.base import Normalizer
from .base import BaseNeuralRadFieldModel
from .cnn import *
from .nerf import *
from .feedforward import *
from typing import Literal, Union, Type, List
import json
import os


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
