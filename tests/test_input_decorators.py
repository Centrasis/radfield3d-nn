"""Input decorators: freely composable dataset features over the all-members FieldInput.

Each decorator initializes exactly one optional member; existing (non-None) members are never
overwritten — a warning is emitted and the value kept. The inner dataset is stubbed at the same
surface the real decorators use (_get_metadata, _access_field_arrays, has_geometry, file_paths).
"""
import pickle
import types
import warnings

import numpy as np
import pytest
import torch

try:
    import RadFiled3D  # noqa: F401
except ImportError:  # pragma: no cover
    pytest.skip("RadFiled3D not available", allow_module_level=True)

from radfield3dnn.datasets.decorators import (FieldInputDecorator, GeometryInputDecorator,
                                              TranslationInputDecorator)
from radfield3dnn.rftypes import FieldInput, TrainingInputData, as_field_input

VC = (4, 4, 4)


class StubFieldDataset:
    """Module-level (picklable) stand-in for RadField3DDataset's decorator-facing surface."""

    def __init__(self, n=3, with_translation=True, with_geometry=True, preset_member=None):
        self.file_paths = [f"f{i}.rf3" for i in range(n)]
        self.data_processings = []
        self.with_translation = with_translation
        self.with_geometry = with_geometry
        self.preset_member = preset_member   # (name, value) preset on the base input

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        inp = FieldInput(direction=torch.tensor([0.0, 0.0, 1.0]), origin=torch.tensor([0.5]),
                         spectrum=torch.full((32,), 1 / 32),
                         beam_shape_type=torch.tensor([3.0]),
                         beam_shape_parameters=torch.tensor([0.1, 0.1]))
        if self.preset_member:
            inp = inp._replace(**{self.preset_member[0]: self.preset_member[1]})
        return TrainingInputData(input=inp, ground_truth=torch.zeros(1))

    def __getitems__(self, indices):
        return [self[i] for i in indices]

    # decorator-facing accessors
    def _get_metadata(self, idx):
        t = types.SimpleNamespace(x=0.1 * idx, y=-0.2, z=0.0) if self.with_translation else None
        return types.SimpleNamespace(simulation=types.SimpleNamespace(patient_translation=t))

    @property
    def has_geometry(self):
        return self.with_geometry

    def _access_field_arrays(self, idx, channels, layers):
        assert channels == ["geometry"] and layers == ["density"]
        density = np.zeros((1, *VC), dtype=np.float32)
        density[0, 1, 2, 3] = 1.85
        return {"geometry": {"density": density}}


def test_translation_decorator_initializes_member():
    ds = TranslationInputDecorator(StubFieldDataset())
    item = ds[1]
    assert isinstance(item.input, FieldInput)
    assert torch.allclose(item.input.translation, torch.tensor([0.1, -0.2, 0.0]))
    assert item.input.geometry is None                       # untouched members stay None


def test_geometry_decorator_initializes_binary_mask():
    ds = GeometryInputDecorator(StubFieldDataset())
    g = ds[0].input.geometry
    assert g.shape == (1, *VC) and g[0, 1, 2, 3] == 1.0 and g.sum() == 1.0   # binary, not density


def test_all_three_features_combine():
    # use_beam_parameters members come from the base input; the two decorators stack on top —
    # the combination the old subclass scheme asserted away.
    ds = GeometryInputDecorator(TranslationInputDecorator(StubFieldDataset()))
    inp = ds[2].input
    assert inp.translation is not None and inp.geometry is not None
    assert inp.beam_shape_parameters is not None and inp.origin is not None
    # order-independence
    ds2 = TranslationInputDecorator(GeometryInputDecorator(StubFieldDataset()))
    inp2 = ds2[2].input
    assert torch.equal(inp2.translation, inp.translation) and torch.equal(inp2.geometry, inp.geometry)


def test_existing_member_is_kept_and_warned_not_overwritten():
    preset = torch.tensor([9.0, 9.0, 9.0])
    ds = TranslationInputDecorator(StubFieldDataset(preset_member=("translation", preset)))
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        item = ds[0]
        ds[1]                                            # second access: warning only ONCE
    assert torch.equal(item.input.translation, preset)   # kept, not overwritten
    hits = [w for w in caught if "translation" in str(w.message)]
    assert len(hits) == 1


def test_missing_translation_metadata_fails_loudly():
    ds = TranslationInputDecorator(StubFieldDataset(with_translation=False))
    with pytest.raises(ValueError, match="patient_translation"):
        ds[0]


def test_field_without_geometry_channel_yields_none():
    ds = GeometryInputDecorator(StubFieldDataset(with_geometry=False))
    assert ds[0].input.geometry is None


def test_getitems_batched_path_goes_through_decorator():
    # torch's fetcher calls dataset.__getitems__ explicitly; without the override the delegation
    # would resolve it on the inner dataset and silently skip the decorator.
    ds = TranslationInputDecorator(StubFieldDataset())
    items = ds.__getitems__([0, 1])
    assert all(it.input.translation is not None for it in items)


def test_attribute_reads_and_writes_are_delegated():
    inner = StubFieldDataset()
    ds = GeometryInputDecorator(TranslationInputDecorator(inner))
    assert ds.file_paths is inner.file_paths             # read-through
    marker = ["proc"]
    ds.data_processings = marker                         # write-through (on_dataset_created hook)
    assert inner.data_processings is marker
    assert len(ds) == len(inner)


def test_decorator_stack_is_picklable():
    ds = GeometryInputDecorator(TranslationInputDecorator(StubFieldDataset()))
    clone = pickle.loads(pickle.dumps(ds))
    assert clone[1].input.translation is not None and clone[0].input.geometry is not None
    assert isinstance(clone, GeometryInputDecorator)


def test_as_field_input_lifts_foreign_named_tuples():
    from RadFiled3D.pytorch.types import DirectionalInput
    d = DirectionalInput(direction=torch.zeros(3), origin=torch.zeros(1), spectrum=torch.zeros(32))
    lifted = as_field_input(d)
    assert isinstance(lifted, FieldInput)
    assert lifted.translation is None and lifted.geometry is None
    assert as_field_input(lifted) is lifted


def test_base_decorator_requires_member_hook():
    with pytest.raises(NotImplementedError):
        FieldInputDecorator(StubFieldDataset()).compute_member(0)
