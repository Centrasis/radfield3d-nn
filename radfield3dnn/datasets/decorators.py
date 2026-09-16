"""Input decorators: compose dataset features instead of multiplying dataset subclasses.

use_geometry / use_translation / use_beam_parameters were mutually exclusive-ish because each
lived in its own RadField3DDataset subclass (with an explicit "not combinable yet" assert). Here
each feature is a DECORATOR around the plain dataset: it intercepts __getitem__, lifts the input
to the all-members FieldInput (radfield3dnn.rftypes) and initializes EXACTLY its own member.
Stacking decorators in any order yields any combination:

    ds = GeometryInputDecorator(TranslationInputDecorator(RadField3DDataset(...)))

Contract: a decorator NEVER overwrites a member — if its member is already non-None (another
decorator or a custom dataset filled it), it warns once and leaves the existing value.

use_beam_parameters needs no decorator: its members (origin, beam_shape_type,
beam_shape_parameters) are initialized by the base dataset from the tube metadata; the flag only
adds the BeamParametersNormalization processing, which composes with everything by construction.

Everything else about the wrapped dataset is transparent: attribute reads AND writes are
delegated to the inner dataset (so DataLoaderBuilder's on_dataset_created hook, data_processings,
file_paths, the __len__ dataset multiplier and the metadata/field accessors all behave as if the
decorator were not there). Instances are picklable (module-level classes, plain __dict__ state)
for spawn-based DataLoader workers.
"""
import warnings

import torch

from radfield3dnn.rftypes import FieldInput, as_field_input


def decorator_chain(dataset):
    """Yield every layer of a (possibly) decorated dataset, outermost first, innermost last."""
    layer = dataset
    while True:
        yield layer
        inner = layer.__dict__.get("_inner") if hasattr(layer, "__dict__") else None
        if inner is None:
            return
        layer = inner


def unwrap_dataset(dataset, cls=None):
    """Return the innermost dataset, or — with ``cls`` — the first chain layer whose REAL type
    is (a subclass of) ``cls``, or None if no layer matches. Matches by ``type(layer)`` on
    purpose: the decorators spoof ``__class__`` for isinstance-transparency, and unwrap must
    return the actual instance of the requested class, not a look-alike."""
    last = None
    for layer in decorator_chain(dataset):
        if cls is not None and issubclass(type(layer), cls):
            return layer
        last = layer
    return last if cls is None else None


def has_dataset_type(dataset, cls) -> bool:
    """True when ``cls`` occurs anywhere in the decorator chain — the check to use when testing
    dataset types explicitly (e.g. has_dataset_type(ds, TranslationInputDecorator) or
    has_dataset_type(ds, RadField3DDataset))."""
    return unwrap_dataset(dataset, cls) is not None


def _rebuild_decorator(cls, state):
    """Pickle reconstructor: bypass __init__/__setattr__ and restore the instance dict raw."""
    obj = object.__new__(cls)
    for key, value in state.items():
        object.__setattr__(obj, key, value)
    return obj


class FieldInputDecorator:
    """Base decorator: wraps a dataset, fills ``member_name`` on each item's input.

    TRANSPARENT TO isinstance: ``__class__`` reports the wrapped dataset's class, so
    ``isinstance(decorated, RadiationFieldDataset)`` (RadFiled3D's DataLoaderBuilder asserts
    exactly this) and any check against the inner dataset's type pass; checks against the
    decorator's own type still pass too (CPython consults type() first). For explicit
    chain-membership tests use ``has_dataset_type`` / ``unwrap_dataset`` above.
    """

    member_name: str = None   # set by subclasses

    def __init__(self, inner):
        object.__setattr__(self, "_inner", inner)
        object.__setattr__(self, "_warned_existing", False)

    @property
    def __class__(self):  # noqa: F811 — deliberate isinstance transparency (wrapt-style)
        return object.__getattribute__(self, "_inner").__class__

    def __reduce__(self):
        # The __class__ spoof breaks default pickling (protocol-2 __newobj__ reads __class__);
        # reconstruct explicitly from the REAL type + raw instance dict instead. Spawn-based
        # DataLoader workers depend on this.
        return (_rebuild_decorator,
                (type(self), dict(object.__getattribute__(self, "__dict__"))))

    # ── own state vs delegation ───────────────────────────────────────────────
    def _own(self, name, value):
        """Set decorator-own state (bypasses the delegation of __setattr__)."""
        object.__setattr__(self, name, value)

    def __getattr__(self, name):
        # only reached when the attribute is not on the decorator itself
        inner = object.__getattribute__(self, "_inner")
        return getattr(inner, name)

    def __setattr__(self, name, value):
        # every plain assignment is meant for the wrapped dataset (data_processings etc.);
        # decorator-own state goes through _own().
        setattr(object.__getattribute__(self, "_inner"), name, value)

    # ── dataset protocol ──────────────────────────────────────────────────────
    def __len__(self):
        return len(object.__getattribute__(self, "_inner"))

    def __getitem__(self, idx):
        inner = object.__getattribute__(self, "_inner")
        item = inner[idx]
        inp = as_field_input(item.input)
        existing = getattr(inp, self.member_name)
        if existing is not None:
            if not object.__getattribute__(self, "_warned_existing"):
                warnings.warn(
                    f"{type(self).__name__}: input member {self.member_name!r} is already set "
                    f"(by another decorator or the dataset); keeping the existing value and NOT "
                    f"overwriting it.", stacklevel=2)
                self._own("_warned_existing", True)
            return item._replace(input=inp)
        value = self.compute_member(idx % len(inner.file_paths))
        return item._replace(input=inp._replace(**{self.member_name: value}))

    def __getitems__(self, indices):
        # torch's fetcher calls this ATTRIBUTE explicitly when present; without an override the
        # delegation would resolve it on the INNER dataset and bypass the decorator entirely.
        return [self[i] for i in indices]

    def compute_member(self, file_idx: int):
        """Return this decorator's member value for the field at ``file_idx`` (or None)."""
        raise NotImplementedError


class TranslationInputDecorator(FieldInputDecorator):
    """Initializes ``translation``: the patient translation (vec3, metres) sampled at dataset
    generation, read from the field's dynamic metadata — same source and same failure contract as
    RadField3DTranslationDataset (the translation-metadata preflight reports offenders up front).
    """

    member_name = "translation"

    def compute_member(self, file_idx: int):
        metadata = object.__getattribute__(self, "_inner")._get_metadata(file_idx)
        translation = getattr(metadata.simulation, "patient_translation", None)
        if translation is None:
            raise ValueError(
                "TranslationInputDecorator requires a 'patient_translation' entry in the field's "
                "dynamic metadata (generate the dataset with patient translation enabled)."
            )
        return torch.tensor([translation.x, translation.y, translation.z], dtype=torch.float32)


class GeometryInputDecorator(FieldInputDecorator):
    """Initializes ``geometry``: the phantom geometry channel's density layer, (1, D, W, H).

    binary=True (default) turns it into the occupancy mask (density > 0) — what both the
    geometry-voxel exclusion and the old use_geometry pipeline consumed. normalize applies
    z-score normalization to the raw density instead (mirrors RadField3DDatasetWithGeometry).
    Fields without a geometry channel yield None (member stays unset).
    """

    member_name = "geometry"

    def __init__(self, inner, binary: bool = True, normalize: bool = False):
        super().__init__(inner)
        assert not (binary and normalize), "binary mask and z-score normalization are exclusive"
        self._own("_binary", bool(binary))
        self._own("_normalize", bool(normalize))

    def compute_member(self, file_idx: int):
        inner = object.__getattribute__(self, "_inner")
        if not inner.has_geometry:
            return None
        arrays = inner._access_field_arrays(file_idx, ["geometry"], ["density"])
        geometry = torch.from_numpy(arrays["geometry"]["density"]).to(torch.float32)
        if object.__getattribute__(self, "_binary"):
            geometry = (geometry > 0.0).to(torch.float32)
        elif object.__getattribute__(self, "_normalize"):
            geometry = (geometry - geometry.mean()) / (geometry.std() + 1e-6)
        return geometry
