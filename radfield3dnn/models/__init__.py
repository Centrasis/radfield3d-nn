import json
import os
from typing import Literal, NamedTuple, Union, Type, List

import torch

from radfield3dnn.preprocessing.normalizations import NormalizerConstructor
from .base import BaseNeuralRadFieldModel
from .nerf import *
from .nerf_translation import *
from .nerf_translation_direction import *
from .nerf_physics import *
from .feedforward import *
from .field_unet import *
from radfield3dnn.models.encoders.base import EncodingBase
from radfield3dnn.rftypes import PositionalInput, positional_like
from radfiled3d.nn.torch_export import export_to_onnx


def _field_output_tensors(out):
    """Flatten a model output into a plain tuple of TENSORS for the ONNX export.

    The models return NamedTuples whose unused members are None
    (``RadiationField(scatter_field=RadiationFieldChannel(spectrum, flux, error=None),
    direct_beam=None, geometry=None)`` flattens to ``(spectrum, flux, None, None, None)``).
    torch.onnx's converter walks the returned pytree and asks every leaf for its ``.name`` —
    a None leaf aborts the conversion with "'NoneType' object has no attribute 'name'"
    (step 3/3). Returning only the real tensors also fixes the output ORDER contract: the deploy
    runtime identifies them structurally (the one whose trailing dim is the bin count is the
    spectrum), so (flux, spectrum) is unambiguous.
    """
    from radfield3dnn.rftypes import AirKermaField, RadiationField, rf3RadiationField
    from radfiled3d.pytorch.types import RadiationFieldChannel

    def channel(ch):
        return tuple(t for t in (ch.flux, ch.spectrum) if t is not None)

    if isinstance(out, (RadiationField, rf3RadiationField)):
        tensors = channel(out.scatter_field) if out.scatter_field is not None else ()
        if out.direct_beam is not None:
            tensors = tensors + channel(out.direct_beam)
        return tensors
    if isinstance(out, RadiationFieldChannel):
        return channel(out)
    if isinstance(out, AirKermaField):
        return (out.air_kerma,)
    if isinstance(out, torch.Tensor):
        return (out,)
    # unknown structure: keep every tensor leaf, drop the Nones
    return tuple(t for t in out if isinstance(t, torch.Tensor))


def in_final_units(model, tensors: tuple) -> tuple:
    """The exported outputs, with every flux taken back to the dataset's own units INSIDE the graph.

    A deployed model emits final units itself; nothing outside its graph converts them. That is
    possible only when the target normalizer has ONE constant for the whole dataset (a
    `dataset_scale`): its inverse needs no reference field, so it is traced into the graph here.
    A normalizer fitted per field has no such constant — each field was divided by its own maximum
    — so its flux stays relative, and the package says so in the output's unit (ModelPackager).
    The spectrum is a per-voxel distribution the normalizer never touches; it passes unchanged.
    """
    normalizer = getattr(model, "_normalizer", None)
    if getattr(normalizer, "dataset_scale", None) is None:
        return tensors
    names = _output_names_from_count(len(tensors))
    return tuple(normalizer.apply_inverse_transformation(t, None) if n.endswith("flux") else t
                 for t, n in zip(tensors, names))


def _output_names_from_count(n: int) -> list:
    base = ["flux", "spectrum", "direct_flux", "direct_spectrum"]
    return base[:n] if n <= len(base) else [f"output_{i}" for i in range(n)]


def _output_names_for(out) -> list:
    """Names matching _field_output_tensors' order (informational: the deploy runtime binds
    outputs structurally, not by name)."""
    return _output_names_from_count(len(_field_output_tensors(out)))


def _export_batch_dim():
    """Dynamic batch dim for the ONNX exports — the deploy module's own, never a second copy."""
    from radfiled3d.nn.torch_export import batch_dim
    return batch_dim()


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
        """Replace a normalizer name (or {"type": ..., knobs} dict) in params with an instance."""
        norm = params.get("normalizer")
        if isinstance(norm, (str, dict)):
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
                append_input=pen.append_input, dim=-1,
            )
            core.load_state_dict(torch.load(weight_path))
            core.positional_location_encoding = pen
        return model


class ExportedGraph(NamedTuple):
    """One ONNX graph plus the tensor names it actually carries.

    The names travel WITH the bytes because the RF3M runtime binds by name: a package that
    declares `beam_dir` while the graph declares `beam_direction` loads and then cannot be driven.
    Returning them together is what lets the packager declare exactly what it shipped.
    """
    onnx: bytes
    input_names: list
    output_names: list


# The beam conditioning is a property of the FIELD, not of the voxel: one row describes the whole
# query grid. The graph therefore takes these at `[1, k]` and expands inside itself, so a caller
# binds one row instead of materialising N identical ones — 168 MB of them at 64³, 1.3 GB at 128³,
# on a device whose free memory is what decides the grid size in the first place.
PER_FIELD_INPUTS = frozenset({
    "beam_direction", "tube_spectrum", "source_distance",
    "beam_collimation", "beam_collimation_kind", "patient_translation",
})


def per_field_inputs(model) -> frozenset:
    """Which of this model's graph inputs travel as ONE row rather than one per query.

    A tensor a stage-source graph consumes is excluded: it reaches the trunk through a buffer, and
    a buffer is sized by the row count of the stage that writes it — which runs per query. Such a
    tensor stays per-query on both sides, so the saving applies to everything else (on a hash-grid
    model that is still `tube_spectrum`, 150 of the 163 floats a query would otherwise carry).
    """
    core = model.get_core_model()
    relayed = set()
    for entry in staged_encoders(model):
        source = core.deploy_stage_source(entry.encoder)
        if source is not None:
            relayed.update(source[1])
    relayed.discard("position")
    return PER_FIELD_INPUTS - relayed


def encoder_frame_position(position, extent: float):
    """Map the deployed `[0, 1]` position onto the frame the location encoder was trained in.

    THE GRAPH DOES THIS, not the consumer. A model trained with
    `voxels_centered_around_origin` queries `[-1, 1]` with the isocentre at 0, but that is an
    internal convention: a package that asked a consumer to know it would have every renderer
    branching on which model it loaded, and one that forgot would sample an octant of the domain and
    put the beam against a face. One interface, `[0, 1]`, for every model; the conversion travels
    inside the graph where it cannot be forgotten.
    """
    return position * float(extent) - (float(extent) / 2.0 - 0.5) * 2.0 if extent != 1.0 else position


def _dynamic_shapes(names, per_field):
    """A dynamic batch axis for the per-query inputs; none for the per-field ones."""
    from radfiled3d.nn.torch_export import batch_dim
    batch = batch_dim()
    return tuple({} if n in per_field else {0: batch} for n in names)


def _per_field_example(name, tensor, per_field):
    """The example the trace sees: one row for a per-field input, unchanged otherwise."""
    return tensor[:1] if name in per_field and tensor.dim() > 1 else tensor


def _broadcast_per_field(kw: dict, per_field) -> dict:
    """Expand each per-field row to the query count, INSIDE the graph."""
    if "position" not in kw:
        return dict(kw)
    rows = kw["position"].shape[0]
    return {n: (t.expand(rows, *t.shape[1:]) if n in per_field and t.dim() > 1 else t)
            for n, t in kw.items()}


def relay_name(tensor: str) -> str:
    """The name a relayed tensor leaves its producing graph under.

    One ONNX namespace covers a graph's inputs and outputs, so a tensor passed straight through
    cannot keep its name. The buffer it lands in carries the same suffix, so the wiring reads the
    same on both sides.
    """
    return f"{tensor}_relay"


class StagedEncoder(NamedTuple):
    """An encoder that ships as its own RF3M stage, and where it lives on the model."""
    stage: str            # the stage name, and the name its blocks are added under (unique per encoder)
    owner: object         # the module holding it
    attribute: str        # the attribute on that owner
    encoder: object       # the encoder itself
    tensor: str           # the trunk graph input its features arrive on


def staged_encoders(model) -> list:
    """Every encoder in `model` that announces a deploy stage.

    Found by asking each encoder (`EncodingBase.deploy_stage`), never by testing its type, so a
    second kernel-backed encoding needs no change here.
    """
    found = []
    for owner in model.modules():
        for attribute, child in vars(owner).get("_modules", {}).items():
            stage = getattr(child, "deploy_stage", lambda: None)()
            if stage is not None:
                # Named after the ATTRIBUTE, not the stage: a model may hold two grids
                # (`beam_frame_encoding` and `patient_frame_encoding` are both encoder sites) and
                # one name for both would wire the second one's features to the first one's buffer.
                found.append(StagedEncoder(f"{stage}_{attribute}", owner, attribute, child,
                                           f"{attribute}_features"))
    return found


class _StagedEncodingShim(EncodingBase):
    """Stands in for a staged encoder while the trunk is traced.

    The features arrive as a graph INPUT instead of being computed: the encoder is a CUDA custom op
    the exporter cannot lower, and in the package a separate stage computes it into a buffer that
    feeds this tensor. Holding it on the instance rather than threading it through `forward` leaves
    the model's call site untouched — it still calls an encoder with a coordinate — so one
    `encode_location` serves both training and export.
    """

    def __init__(self, dims: int):
        super().__init__()
        self._dims = dims
        self.features = None

    def calc_encoded_dim(self) -> int:
        return self._dims

    def forward(self, x, region_state=None):
        if self.features is None:
            raise RuntimeError("the staged encoder's features were never supplied to the trace")
        return self.features


def _traced_module(names, call):
    """A traceable module whose forward takes exactly ``names`` as positional parameters.

    The signature is BUILT from the name list rather than fixed, for two reasons that each bite on
    their own. Which beam parameters a model consumes is configuration (collimation, patient
    translation), so a fixed signature binds the translation to the beam-shape slot whenever an
    earlier optional input is absent — silent, and fatal at deployment, because the runtime binds
    by name. And a ``*args`` forward cannot be traced with a dynamic batch axis: torch.export
    matches ``dynamic_shapes`` against the signature and sees the whole pack as one entry
    ("inputs has 1 elements, but dynamic_shapes has 5").

    ``call`` receives the traced tensors as a {name: tensor} mapping and returns the module output.
    """
    src = ("def forward(self, %s):\n    return self._call({%s})\n"
           % (", ".join(names), ", ".join("%r: %s" % (n, n) for n in names)))
    ns = {}
    exec(src, ns)
    cls = type("_Traced", (BaseNeuralRadFieldModel,),
               {"forward": ns["forward"], "_call": staticmethod(call)})
    return cls()


class ModelExporter:
    @staticmethod
    def onnx_export(model: BaseNeuralRadFieldModel) -> ExportedGraph:
        """Export the whole model as one self-contained graph: beam parameters + position -> field."""
        inp = model._generate_random_input(model.device)
        inp = getattr(inp, "input", inp)   # _generate_random_input may return a PositionalInput directly
        core = model.get_core_model()
        # An encoder that ships as its own stage is replaced, for the trace, by a shim taking its
        # features as a graph input — the package computes them in a separate stage and a buffer
        # carries them here. Swapped in BEFORE the model is run for its output names: the export
        # runs on CPU and the encoder being replaced is exactly the one that cannot.
        staged = staged_encoders(model)
        for entry in staged:
            # Refused BEFORE the shim goes in. Without a source graph the package could not feed
            # the stage, and tracing anyway would emit a graph in which the encoder is dead code —
            # complete-looking, silently missing the encoder, which is the worst outcome available.
            if core.deploy_stage_source(entry.encoder) is None:
                raise ValueError(
                    f"{type(core).__name__} holds a `{entry.stage}` stage encoder at "
                    f"`{entry.attribute}` but does not implement deploy_stage_source(), so nothing "
                    f"can compute that stage's input. The package would carry a stage that cannot "
                    f"run.")
        shims, examples = {}, {}

        per_field = per_field_inputs(model)

        extent = float(getattr(model, "encoder_frame_extent", 1.0))

        def _call(kw):
            kw = _broadcast_per_field(kw, per_field)
            kw = dict(kw, position=encoder_frame_position(kw["position"], extent))
            for tensor, shim in shims.items():
                shim.features = kw[tensor]
            # tensors only — a None in the returned pytree kills the ONNX conversion
            return in_final_units(model, _field_output_tensors(core.forward(positional_like(
                inp,
                direction=kw["beam_direction"], position=kw["position"],
                spectrum=kw["tube_spectrum"], origin=kw["source_distance"],
                beam_shape_parameters=kw.get("beam_collimation"),
                beam_shape_type=kw.get("beam_collimation_kind"),
                geometry=kw.get("geometry_map"),
                translation=kw.get("patient_translation"),
            ))))

        # Everything from the first swap to the last trace sits in ONE try/finally: a failure in
        # between would otherwise leave the model's encoder permanently replaced by a shim, and the
        # export callback swallows exceptions, so nothing would say so.
        try:
            for entry in staged:
                shim = _StagedEncodingShim(entry.encoder.encoded_dims).to(model.device)
                examples[entry.tensor] = torch.zeros(
                    inp.position.shape[0], entry.encoder.encoded_dims,
                    device=model.device, dtype=inp.position.dtype)
                shim.features = examples[entry.tensor]
                shims[entry.tensor] = shim
                setattr(entry.owner, entry.attribute, shim)
            with torch.no_grad():
                output_names = _output_names_for(core.forward(inp))
            # The tube spectrum is named `tube_spectrum`, never `spectrum`: an ONNX graph has ONE
            # namespace for inputs and outputs, and this export puts the tube spectrum and the
            # PREDICTED spectrum in the same graph — a shared name writes fine and is then rejected
            # by every consumer with "Duplicate definition of name (spectrum)".
            # Tensor names are the deploy module's own semantic vocabulary (`Semantic::get_name`),
            # not this model's python argument names: the runtime binds by name, and every package
            # in the corpus — including the migrated historical ones — speaks that vocabulary.
            candidates = [("beam_direction", inp.direction), ("position", inp.position),
                          ("tube_spectrum", inp.spectrum), ("source_distance", inp.origin),
                          ("beam_collimation", inp.beam_shape_parameters),
                          ("beam_collimation_kind", inp.beam_shape_type),
                          ("geometry_map", inp.geometry),
                          ("patient_translation", getattr(inp, "translation", None))]
            for entry in staged:
                candidates.append((entry.tensor, examples[entry.tensor]))
            named = [(n, _per_field_example(n, t, per_field)) for n, t in candidates if t is not None]
            names = [n for n, _ in named]
            # Only the per-QUERY inputs carry the dynamic batch axis. A per-field input is traced at
            # a static 1, which is what makes the runtime bind it as one row: the ONNX backend reads
            # each bound shape off the graph and substitutes the query count for dynamic dims only.
            onnx = export_to_onnx(_traced_module(names, _call).eval(), tuple(t for _, t in named),
                                  input_names=names, output_names=output_names,
                                  dynamic_shapes=_dynamic_shapes(names, per_field))
            return ExportedGraph(onnx, names, output_names)
        finally:
            for entry in staged:
                setattr(entry.owner, entry.attribute, entry.encoder)

    @staticmethod
    def onnx_export_stage_source(model: BaseNeuralRadFieldModel, staged,
                                 relay: list = ()) -> ExportedGraph:
        """Export the graph that computes a staged encoder's input.

        A kernel stage has no caller-bound tensors: what it reads comes from a buffer an earlier
        stage wrote. This is that earlier stage — it takes the caller's beam parameters and writes
        the coordinate the encoder is queried at.

        It also RELAYS `relay`: the tensors the trunk needs and this graph already takes. A tensor
        cannot be a caller-bound input of two stages — the runtime refuses the bind as ambiguous,
        because a caller naming it could mean either — so the shared ones are bound here once and
        reach the trunk through buffers. A relayed tensor leaves under a different name: an ONNX
        graph has one namespace for inputs and outputs, so passing `position` straight through as
        `position` would collide.
        """
        core = model.get_core_model()
        source = core.deploy_stage_source(staged.encoder)
        if source is None:
            raise ValueError(
                f"{type(core).__name__} holds a `{staged.stage}` stage encoder at "
                f"`{staged.attribute}` but does not implement deploy_stage_source(), so nothing "
                f"can compute that stage's input. The package would carry a stage that cannot run.")
        module, names = source
        names = list(names)
        relay = [n for n in relay if n in names]
        inp = model._generate_random_input(model.device)
        inp = getattr(inp, "input", inp)
        available = {
            "position": inp.position, "beam_direction": inp.direction,
            "source_distance": inp.origin, "beam_collimation": inp.beam_shape_parameters,
            "beam_collimation_kind": inp.beam_shape_type,
            "patient_translation": getattr(inp, "translation", None),
        }
        per_field = per_field_inputs(model)
        extent = float(getattr(model, "encoder_frame_extent", 1.0))
        coordinate = f"{staged.stage}_coordinate"
        out_names = [coordinate] + [relay_name(n) for n in relay]

        def _call(kw):
            # The coordinate is per query, so the beam row is expanded for the module — but the
            # RELAY passes the row through unexpanded, because the trunk takes it per field too.
            # Relaying the expanded copy would push N rows through the buffer and defeat the point.
            dense = _broadcast_per_field(kw, per_field)
            dense["position"] = encoder_frame_position(dense["position"], extent)
            tensors = [module(*(dense[n] for n in names))]
            # `+ 0` so a relayed tensor is a distinct graph value: returning an input unchanged
            # makes the exporter alias it to the input name and the output vanishes.
            tensors += [kw[n] + 0 for n in relay]
            return tuple(tensors)

        onnx = export_to_onnx(_traced_module(names, _call).eval(),
                              tuple(_per_field_example(n, available[n], per_field) for n in names),
                              input_names=names, output_names=out_names,
                              dynamic_shapes=_dynamic_shapes(names, per_field))
        return ExportedGraph(onnx, names, out_names)

    # ── Two-graph export (beam-encoder + trunk) ───────────────────────────────
    # Per-voxel NeRF models (RFBackboneModel subclasses) factor into a beam encoder
    # (beam parameters -> a d_model latent) and a position trunk that consumes that latent
    # via forward(batch, global_parameters=latent). Exporting them as two ONNX graphs lets the
    # deploy runtime encode the beam ONCE and reuse the latent across every voxel query.
    @staticmethod
    def supports_two_graph_split(model: BaseNeuralRadFieldModel) -> bool:
        core = model.get_core_model()
        # The split factors the model into beam->latent + (position, latent)->output. A model whose
        # TRUNK consumes per-point inputs beyond the position (TPBRFNet's relative/dual-frame
        # translation coordinate, the beam_geometry block's beam/patient-frame coordinates) does not
        # fit that factorization — the trunk graph would need the beam inputs as further inputs,
        # which the deploy runtime does not bind. Such models export as a single self-contained
        # graph instead (every beam parameter is a named input there).
        if getattr(core, "trunk_consumes_beam_inputs", False):
            return False
        return hasattr(core, "encode_additional_parameters")

    @staticmethod
    def onnx_export_beam_encoder(model: BaseNeuralRadFieldModel) -> ExportedGraph:
        """Export the beam encoder: (beam_direction, source_distance, tube_spectrum [, ...]) -> latent."""
        core = model.get_core_model()
        inp = model._generate_random_input(model.device)
        inp = getattr(inp, "input", inp)

        def _call(kw):
            return core.encode_additional_parameters(positional_like(
                inp,
                direction=kw["beam_direction"], origin=kw["source_distance"],
                spectrum=kw["tube_spectrum"],
                position=torch.zeros_like(kw["beam_direction"]),   # unused by the beam encoder
                beam_shape_parameters=kw.get("beam_collimation"),
                translation=kw.get("patient_translation")))

        candidates = [("beam_direction", inp.direction), ("source_distance", inp.origin),
                      ("tube_spectrum", inp.spectrum)]
        if bool(getattr(core, "use_beam_shape", False)):
            candidates.append(("beam_collimation", inp.beam_shape_parameters))
        translation = getattr(inp, "translation", None)
        if translation is not None:
            candidates.append(("patient_translation", translation))
        names = [n for n, _ in candidates]
        onnx = export_to_onnx(_traced_module(names, _call).eval(), tuple(t for _, t in candidates),
                              input_names=names, output_names=["latent"])
        return ExportedGraph(onnx, names, ["latent"])

    @staticmethod
    def onnx_export_encoding_config(model: BaseNeuralRadFieldModel) -> ExportedGraph:
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
        # Per-GRID, not per-query: the one graph here carries no batch axis.
        onnx = export_to_onnx(_RegionConfig(enc).eval(), (example,),
                              input_names=["region_width"], output_names=["region_state"],
                              dynamic_batch=False)
        return ExportedGraph(onnx, ["region_width"], ["region_state"])

    @staticmethod
    def onnx_export_trunk(model: BaseNeuralRadFieldModel) -> ExportedGraph:
        """Export the trunk: (position, latent [, region_state]) -> flux + spectrum.

        ``region_state`` is present only for grid-resolution-aware location encoders; it comes from
        the encoding-config graph and stays constant for as long as the queried grid does.
        """
        core = model.get_core_model()
        inp = model._generate_random_input(model.device)
        inp = getattr(inp, "input", inp)
        with torch.no_grad():
            latent = core.encode_additional_parameters(inp)  # example [B, d_model]
        enc = core.positional_location_encoding
        state_dims = int(getattr(enc, "region_state_dims", 0) or 0)

        def _call(kw):
            position = kw["position"]
            # tensors only — see _field_output_tensors
            return in_final_units(model, _field_output_tensors(core.forward(PositionalInput(
                direction=torch.zeros_like(position), origin=position[..., :1] * 0,
                spectrum=position[..., :1] * 0, position=position),
                global_parameters=kw["latent"], region_state=kw.get("region_state"))))

        # Dynamic batch on the per-voxel inputs: position rows vary per inner batch at deploy time,
        # and the latent is broadcast to the same row count by the caller. region_state is per-GRID
        # (not per-row), so it carries no batch axis.
        batch = _export_batch_dim()
        args = (inp.position, latent)
        names = ["position", "latent"]
        dyn = ({0: batch}, {0: batch})
        if state_dims > 0:
            with torch.no_grad():
                state = enc.compute_region_state(
                    torch.tensor(2.0 / 64, device=model.device, dtype=torch.float32))
            args = args + (state,)
            names = names + ["region_state"]
            dyn = dyn + (None,)
        with torch.no_grad():
            out_names = _output_names_for(core.forward(inp, global_parameters=latent))
        onnx = export_to_onnx(_traced_module(names, _call).eval(), args, input_names=names,
                              output_names=out_names, dynamic_shapes=dyn)
        return ExportedGraph(onnx, names, out_names)
