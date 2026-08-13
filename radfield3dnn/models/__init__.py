import json
import os
from typing import Literal, Union, Type, List

import torch

from radfield3dnn.preprocessing.normalizations import NormalizerConstructor
from .base import BaseNeuralRadFieldModel
from .nerf import *
from .nerf_cpp import *
from .nerf_translation import *
from .feedforward import *
from .field_unet import *
from radfield3dnn.rftypes import PositionalInput, positional_like


def _export_batch_dim():
    """Dynamic batch dim for the ONNX exports, with an EXPLICIT upper bound.

    An unbounded Dim declares the range [0, int64_max]; some traced ops then emit the guard
    `batch != int64_max` (broadcast/expand sentinel), which the declared range cannot satisfy —
    torch.export aborts with "Constraints violated (batch) ... != 9223372036854775807" (seen on
    the trained-model package export). Bounding the dim to a realistic maximum removes the
    conflict without freezing the batch size.
    """
    return torch.export.Dim("batch", min=1, max=2**31 - 1)


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
            params = dict(params)
            params["normalizer"] = NormalizerConstructor.construct_by_name(norm)
        return params

    @staticmethod
    def create_model_from_config(config_path: str) -> Type[BaseNeuralRadFieldModel]:
        with open(config_path) as f:
            config = json.load(f)
        return ModelConstructor.create_model_from_dict(config)

    @staticmethod
    def create_model_from_dict(config: dict) -> Type[BaseNeuralRadFieldModel]:
        params = ModelConstructor._resolve_normalizer(dict(config.get("parameters", {})))
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


class ModelExporter:
    @staticmethod
    def onnx_export(model: BaseNeuralRadFieldModel, path: str):
        inp = model._generate_random_input(model.device)
        inp = getattr(inp, "input", inp)   # _generate_random_input may return a PositionalInput directly
        # Extra beam parameters of the concrete input type (TPBRFNet's patient translation) become
        # extra graph inputs; "translation" is the name the deploy runtime binds by intent.
        _translation = getattr(inp, "translation", None)

        class _OnnxWrapper(BaseNeuralRadFieldModel):
            def __init__(self, decoratee):
                super().__init__()
                self._d = decoratee

            def forward(self, direction, position, spectrum, origin, beam_shape_parameters, beam_shape_type, geometry, translation=None):
                return self._d.forward(positional_like(
                    inp,
                    direction=direction, beam_shape_parameters=beam_shape_parameters,
                    beam_shape_type=beam_shape_type, position=position,
                    origin=origin, geometry=geometry, spectrum=spectrum,
                    translation=translation,
                ))

        wrapped = _OnnxWrapper(model.get_core_model())
        args = (inp.direction, inp.position, inp.spectrum,
                inp.origin, inp.beam_shape_parameters,
                inp.beam_shape_type, inp.geometry)
        arg_names = ["direction", "position", "spectrum", "origin",
                     "beam_shape_parameters", "beam_shape_type", "geometry"]
        if _translation is not None:
            # trailing keyword-with-default -> safe to append positionally
            args = args + (_translation,)
            arg_names = arg_names + ["translation"]
        # A None arg produces NO graph input, and input_names is applied POSITIONALLY to the ones
        # that exist — so listing a name for a skipped arg shifts every later name onto the wrong
        # tensor (with geometry=None that silently labelled the translation "geometry", which the
        # deploy runtime binds by name).
        input_names = [n for n, a in zip(arg_names, args) if a is not None]
        # Dynamic batch axis on every (non-None) input: a stored model must accept any batch size
        # at inference. Without this dynamo freezes the traced example batch into the graph and the
        # deployed ONNX rejects every other batch size.
        batch = _export_batch_dim()
        dynamic_shapes = tuple({0: batch} if a is not None else None for a in args)
        torch.onnx.export(
            model=wrapped,
            args=args,
            input_names=input_names,
            dynamic_shapes=dynamic_shapes,
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

        # A beam parameter the model's input type adds beyond PositionalInput becomes an extra
        # traced graph input. The name must match what the deploy runtime binds by intent:
        # "translation" -> ModelInput::PatientTranslation3D (see field_predictors.cpp).
        translation = getattr(inp, "translation", None)

        class _BeamEnc(BaseNeuralRadFieldModel):
            def __init__(self, d):
                super().__init__()
                self._d = d
            def forward(self, direction, distance, spectrum, beam_shape_parameters=None, translation=None):
                return self._d.encode_additional_parameters(positional_like(
                    inp,
                    direction=direction, origin=distance, spectrum=spectrum,
                    position=torch.zeros_like(direction),  # unused by the beam encoder
                    beam_shape_parameters=beam_shape_parameters,
                    translation=translation))

        # The graph's input SET is data-dependent (beam shape and/or translation may be absent), so
        # the example inputs are passed BY NAME: positional args would bind e.g. the translation to
        # the beam_shape_parameters slot whenever an earlier optional input is skipped.
        graph_inputs = {"direction": inp.direction, "distance": inp.origin, "spectrum": inp.spectrum}
        if use_beam_shape:
            graph_inputs["beam_shape_parameters"] = inp.beam_shape_parameters
        if translation is not None:
            graph_inputs["translation"] = translation
        names = list(graph_inputs)
        # Dynamic batch axis: without it dynamo freezes the traced batch (=2) into the graph and
        # the deployed ONNX rejects any other batch size (observed via the rfnn_deploy bindings).
        batch = _export_batch_dim()
        dyn = {name: {0: batch} for name in names}
        torch.onnx.export(model=_BeamEnc(core).eval(), args=(), kwargs=graph_inputs,
                          input_names=names, dynamic_shapes=dyn, dynamo=True).save(path)

    @staticmethod
    def onnx_export_encoding_config(model: BaseNeuralRadFieldModel, path: str):
        """Export the region-configuration graph: (region_width) -> region_state.

        Only meaningful for grid-resolution-aware location encoders (region_state_dims > 0). The
        deploy-side runs this ONCE whenever the queried grid resolution changes (LOD) and feeds the
        cached vector to every trunk call for that grid, so the resolution can change per inference
        without re-exporting and without the consumer knowing the encoder's maths. ``region_width``
        is one voxel of the queried grid in the model's coordinate frame (see
        ``FeedforwardPointwiseModel.voxel_width_in_encoder_frame``).
        """
        enc = model.get_core_model().positional_location_encoding

        class _RegionConfig(BaseNeuralRadFieldModel):
            def __init__(self, e):
                super().__init__()
                self._e = e
            def forward(self, region_width):
                return self._e.compute_region_state(region_width)

        example = torch.tensor(2.0 / 64, device=model.device, dtype=torch.float32)
        torch.onnx.export(model=_RegionConfig(enc), args=(example,),
                          input_names=["region_width"], output_names=["region_state"],
                          dynamo=True).save(path)

    @staticmethod
    def onnx_export_trunk(model: BaseNeuralRadFieldModel, path: str):
        """Export the trunk: (position, latent [, region_state]) -> flux + spectrum.

        ``region_state`` is present only for grid-resolution-aware location encoders; it comes from
        the encoding-config graph and stays constant for as long as the queried grid does.
        """
        core = model.get_core_model()
        inp = model._generate_random_input(model.device)
        inp = getattr(inp, "input", inp)
        with torch.no_grad():
            latent = core.encode_additional_parameters(inp)  # example [B, d_model]
        state_dims = model.deploy_interface().region_state_dims

        class _Trunk(BaseNeuralRadFieldModel):
            def __init__(self, d):
                super().__init__()
                self._d = d
            def forward(self, position, latent, region_state=None):
                return self._d.forward(PositionalInput(
                    direction=torch.zeros_like(position), origin=position[..., :1] * 0,
                    spectrum=position[..., :1] * 0, position=position),
                    global_parameters=latent, region_state=region_state)

        # Dynamic batch on the per-voxel inputs: position rows vary per inner batch at deploy time,
        # and the latent is broadcast to the same row count by the caller. region_state is per-GRID
        # (not per-row), so it carries no batch axis.
        batch = _export_batch_dim()
        args = (inp.position, latent)
        names = ["position", "latent"]
        dyn = ({0: batch}, {0: batch})
        if state_dims > 0:
            with torch.no_grad():
                state = core.positional_location_encoding.compute_region_state(
                    torch.tensor(2.0 / 64, device=model.device, dtype=torch.float32))
            args = args + (state,)
            names = names + ["region_state"]
            dyn = dyn + (None,)
        torch.onnx.export(model=_Trunk(core), args=args, input_names=names,
                          dynamic_shapes=dyn, dynamo=True).save(path)
