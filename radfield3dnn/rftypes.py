from RadFiled3D.pytorch.types import RadiationField as rf3RadiationField, RadiationFieldChannel, TrainingInputData as rf3TrainingInputData, DirectionalInput as DirectionalInput, PositionalInput as PositionalInput
from typing import NamedTuple, Union
from torch import Tensor


class RadiationField(NamedTuple):
    scatter_field: RadiationFieldChannel
    direct_beam: RadiationFieldChannel
    geometry: Union[Tensor, None] = None  # Optional geometry tensor associated with the radiation field


class AirKermaField(NamedTuple):
    air_kerma: Tensor
    geometry: Union[Tensor, None] = None  # Optional geometry tensor associated with the air kerma field


class FieldInput(NamedTuple):
    """The ONE field-level network input: every member any dataset feature can provide.

    The base dataset initializes the core members (direction, origin, spectrum, beam shape from
    the tube metadata); each OPTIONAL member is initialized by its input DECORATOR
    (radfield3dnn.datasets.decorators): GeometryInputDecorator -> ``geometry``,
    TranslationInputDecorator -> ``translation``. Absent features stay None, so
    use_geometry / use_translation / use_beam_parameters combine freely — consumers already
    read optional members via getattr(..., None) and rebuild via _replace, both of which are
    structural, not class-bound.
    """
    direction: Tensor
    origin: Tensor
    spectrum: Tensor
    geometry: Union[Tensor, None] = None
    beam_shape_type: Union[Tensor, None] = None
    beam_shape_parameters: Union[Tensor, None] = None
    translation: Union[Tensor, None] = None


def as_field_input(x) -> FieldInput:
    """Lift any DirectionalInput-shaped NamedTuple into a FieldInput, carrying over every member
    the source provides and leaving the rest None. A FieldInput passes through unchanged."""
    if isinstance(x, FieldInput):
        return x
    return FieldInput(
        direction=x.direction, origin=x.origin, spectrum=x.spectrum,
        geometry=getattr(x, "geometry", None),
        beam_shape_type=getattr(x, "beam_shape_type", None),
        beam_shape_parameters=getattr(x, "beam_shape_parameters", None),
        translation=getattr(x, "translation", None),
    )


class TranslationalPositionalInput(NamedTuple):
    """A PER-VOXEL query input that also carries the patient translation.

    RadFiled3D's TranslationalInput is the FIELD-level input (direction, origin, spectrum,
    translation) — it has no ``position``, because a whole-field sample has no single query point.
    The per-voxel pipeline therefore rebuilds each chunk as a PositionalInput, which drops the
    translation; any consumer that then calls ``encode_additional_parameters`` on such a chunk
    (the batch-size search, the plot callbacks, the ONNX exporters — everything that does NOT
    precompute ``global_parameters``) hits TPBRFNet's missing-translation assertion.

    This type is that missing combination: every PositionalInput field plus ``translation``, so a
    per-voxel input can round-trip the translation to the beam encoder. Build it through
    ``positional_like`` rather than directly.
    """
    direction: Tensor
    origin: Tensor
    spectrum: Tensor
    position: Tensor
    geometry: Union[Tensor, None] = None
    beam_shape_type: Union[Tensor, None] = None
    beam_shape_parameters: Union[Tensor, None] = None
    translation: Union[Tensor, None] = None


def positional_like(template, **fields):
    """Build a per-voxel input of the type matching ``template``.

    Returns a plain PositionalInput for ordinary inputs, and a TranslationalPositionalInput when
    ``template`` carries a patient translation — so the extra beam parameter survives the
    per-voxel rebuild instead of being silently dropped. ``fields`` are the PositionalInput
    fields; pass ``translation=`` explicitly when the rows need re-indexing (chunking, repeats),
    otherwise the template's translation is carried over as-is.
    """
    translation = fields.pop("translation", _MISSING)
    if translation is _MISSING:
        translation = getattr(template, "translation", None)
    if translation is None:
        return PositionalInput(**fields)
    return TranslationalPositionalInput(**fields, translation=translation)


_MISSING = object()


class TrainingInputData(NamedTuple):
    input: Union[DirectionalInput, PositionalInput, TranslationalPositionalInput, Tensor]
    ground_truth: Union[rf3RadiationField, RadiationFieldChannel, RadiationField, AirKermaField]
    original_ground_truth: Union[rf3RadiationField, RadiationFieldChannel, RadiationField, AirKermaField, None] = None  # Optional original ground truth for reference
