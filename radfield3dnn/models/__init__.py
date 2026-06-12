import json
import os
from typing import Literal, Union, Type, List

import torch

from radfield3dnn.preprocessing.normalizations import NormalizerConstructor
from .base import BaseNeuralRadFieldModel
from .cnn import *
from .nerf import *
from .hdr_net import *
from .twohead_hdr import *
from .nerf_cpp import *
from .feedforward import *
from .xattn_siren import *
from .field_unet import *
from .mlp import *
from radfield3dnn.rftypes import PositionalInput
from enum import Enum


class ModelConstructor:
    @staticmethod
    def get_subclasses(cls) -> List[Type[BaseNeuralRadFieldModel]]:
        result = []
        for sub in cls.__subclasses__():
            result.append(sub)
            result += ModelConstructor.get_subclasses(sub)
        return result

    @staticmethod
    def construct(name: str, **kwargs):
        for cls in ModelConstructor.get_subclasses(BaseNeuralRadFieldModel):
            if "__model_name__" in cls.__dict__ and cls.__model_name__ == name:
                return cls(**kwargs)
        raise ValueError(f"Model '{name}' not found.")

    @staticmethod
    def get_dataset_type_for_model(name: str) -> Union[Literal["Voxelwise"], Literal["Layerwise"]]:
        return "Layerwise"

    @staticmethod
    def create_model_with_defaults(name: str, **kwargs) -> Type[BaseNeuralRadFieldModel]:
        base_model = ModelConstructor.construct(name, **kwargs)

        class DerivedModel(type(base_model)):
            def __init__(self, **init_kwargs):
                super().__init__(**{**kwargs, **init_kwargs})
                self.__class__.__name__ = base_model.__class__.__name__

        return DerivedModel

    @staticmethod
    def _resolve_normalizer(params: dict) -> dict:
        """Replace a string normalizer key in params with a Normalizer instance."""
        norm = params.get("normalizer")
        if isinstance(norm, str):
            auto_tune = norm == "asinh_auto"
            params = dict(params)
            params["normalizer"] = NormalizerConstructor.construct_by_name(norm)
            params["_asinh_auto"] = auto_tune
        return params

    @staticmethod
    def create_model_from_config(config_path: str) -> Type[BaseNeuralRadFieldModel]:
        with open(config_path) as f:
            config = json.load(f)
        return ModelConstructor.create_model_from_dict(config)

    @staticmethod
    def create_model_from_dict(config: dict) -> Type[BaseNeuralRadFieldModel]:
        params = ModelConstructor._resolve_normalizer(dict(config.get("parameters", {})))
        params.pop("_asinh_auto", None)
        return ModelConstructor.create_model_with_defaults(config["model_name"], **params)

    @staticmethod
    def load_model_from(path: str) -> BaseNeuralRadFieldModel:
        if not os.path.exists(path):
            raise ValueError(f"Model file not found: {path}")

        ext = os.path.splitext(path)[1].lower()
        if ext == ".json":
            config_path = path
            weight_path = os.path.splitext(path)[0] + ".pt"
            if not os.path.exists(weight_path):
                weight_path = os.path.splitext(path)[0] + ".ckpt"
        elif ext in (".pt", ".ckpt"):
            weight_path = path
            config_path = os.path.splitext(path)[0] + ".json"
        else:
            raise ValueError(f"Unknown extension '{ext}' for: {path}")

        if not os.path.exists(config_path):
            raise ValueError(f"Config not found: {config_path}")
        if not os.path.exists(weight_path):
            raise ValueError(f"Weights not found: {weight_path}")

        config = json.load(open(config_path))
        if "parameters" not in config and "hyper_parameters" in config:
            config = {"parameters": config["hyper_parameters"], "model_name": config["model_name"]}

        model_cls = ModelConstructor.create_model_from_dict(config)
        model = model_cls()
        core = model.get_core_model()
        try:
            core.load_state_dict(torch.load(weight_path))
        except Exception:
            from radfield3dnn.models.encoders.sinusoidal_encoding import SinusoidalFrequencyEncoding
            pen = core.positional_location_encoding
            core.positional_location_encoding = SinusoidalFrequencyEncoding(
                pos_enc_dim=pen.pos_enc_dim, d_input=pen.d_input,
                append_input=pen.append_input, dim=-1, use_tcnn=False,
            )
            core.load_state_dict(torch.load(weight_path))
            core.positional_location_encoding = pen
        return model


class ModelFormat(Enum):
    ONNX = 0
    TORCH_SCRIPT = 1
    TENSOR_RT = 2


class ModelExporter:
    @staticmethod
    def export(model: BaseNeuralRadFieldModel, path: str, format: ModelFormat = ModelFormat.ONNX):
        {
            ModelFormat.ONNX: ModelExporter.onnx_export,
            ModelFormat.TORCH_SCRIPT: ModelExporter.ts_export,
            ModelFormat.TENSOR_RT: ModelExporter.rt_export,
        }[format](model, path)

    @staticmethod
    def rt_export(model, path):
        raise NotImplementedError()

    @staticmethod
    def ts_export(model, path):
        torch.jit.script(model.get_core_model()).save(path)

    @staticmethod
    def onnx_export(model: BaseNeuralRadFieldModel, path: str):
        class _OnnxWrapper(BaseNeuralRadFieldModel):
            def __init__(self, decoratee):
                super().__init__()
                self._d = decoratee

            def forward(self, direction, position, spectrum, origin, beam_shape_parameters, beam_shape_type, geometry):
                return self._d.forward(PositionalInput(
                    direction=direction, beam_shape_parameters=beam_shape_parameters,
                    beam_shape_type=beam_shape_type, position=position,
                    origin=origin, geometry=geometry, spectrum=spectrum,
                ))

        inp = model._generate_random_input(model.device)
        inp = getattr(inp, "input", inp)   # _generate_random_input may return a PositionalInput directly
        wrapped = _OnnxWrapper(model.get_core_model())
        torch.onnx.export(
            model=wrapped,
            args=(inp.direction, inp.position, inp.spectrum,
                  inp.origin, inp.beam_shape_parameters,
                  inp.beam_shape_type, inp.geometry),
            input_names=["direction", "position", "spectrum", "origin",
                         "beam_shape_parameters", "beam_shape_type", "geometry"],
            dynamo=True,
        ).save(path)

    # ── Two-graph export (beam-encoder + trunk) ───────────────────────────────
    # Per-voxel NeRF models (RFBackboneModel subclasses) factor into a beam encoder
    # (beam parameters -> a d_model latent) and a position trunk that consumes that latent
    # via forward(batch, global_parameters=latent). Exporting them as two ONNX graphs lets the
    # deploy runtime encode the beam ONCE and reuse the latent across every voxel query.
    @staticmethod
    def supports_two_graph_split(model: BaseNeuralRadFieldModel) -> bool:
        core = model.get_core_model()
        return hasattr(core, "encode_additional_parameters")

    @staticmethod
    def onnx_export_beam_encoder(model: BaseNeuralRadFieldModel, path: str):
        """Export the beam encoder: (direction, distance, spectrum [, opening_angle]) -> latent."""
        core = model.get_core_model()
        use_beam_shape = bool(getattr(core, "use_beam_shape", False))
        inp = model._generate_random_input(model.device)
        inp = getattr(inp, "input", inp)

        class _BeamEnc(BaseNeuralRadFieldModel):
            def __init__(self, d): super().__init__(); self._d = d
            def forward(self, direction, distance, spectrum, beam_shape_parameters=None):
                return self._d.encode_additional_parameters(PositionalInput(
                    direction=direction, origin=distance, spectrum=spectrum,
                    position=torch.zeros_like(direction),  # unused by the beam encoder
                    beam_shape_parameters=beam_shape_parameters))

        args = [inp.direction, inp.origin, inp.spectrum]
        names = ["direction", "distance", "spectrum"]
        if use_beam_shape:
            args.append(inp.beam_shape_parameters)
            names.append("beam_shape_parameters")
        # Dynamic batch axis: without it dynamo freezes the traced batch (=2) into the graph and
        # the deployed ONNX rejects any other batch size (observed via the rfnn_deploy bindings).
        batch = torch.export.Dim("batch")
        dyn = tuple({0: batch} for _ in args)
        torch.onnx.export(model=_BeamEnc(core), args=tuple(args), input_names=names,
                          dynamic_shapes=dyn, dynamo=True).save(path)

    @staticmethod
    def onnx_export_trunk(model: BaseNeuralRadFieldModel, path: str):
        """Export the trunk: (position, latent) -> flux + spectrum (beam encoder bypassed)."""
        core = model.get_core_model()
        inp = model._generate_random_input(model.device)
        inp = getattr(inp, "input", inp)
        with torch.no_grad():
            latent = core.encode_additional_parameters(inp)  # example [B, d_model]

        class _Trunk(BaseNeuralRadFieldModel):
            def __init__(self, d): super().__init__(); self._d = d
            def forward(self, position, latent):
                return self._d.forward(PositionalInput(
                    direction=torch.zeros_like(position), origin=position[..., :1] * 0,
                    spectrum=position[..., :1] * 0, position=position),
                    global_parameters=latent)

        # Dynamic batch on BOTH inputs: position rows vary per inner batch at deploy time, and the
        # latent is broadcast to the same row count by the caller.
        batch = torch.export.Dim("batch")
        torch.onnx.export(model=_Trunk(core), args=(inp.position, latent),
                          input_names=["position", "latent"],
                          dynamic_shapes=({0: batch}, {0: batch}), dynamo=True).save(path)
