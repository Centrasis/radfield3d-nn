from normalizations.base import Normalizer
from .base import BaseNeuralRadFieldModel
from .cnn import *
from .nerf import *
from .feedforward import *
from typing import Literal, Union, Type, List
import json


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
