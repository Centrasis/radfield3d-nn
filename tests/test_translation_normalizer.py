"""TranslationNormalization: patient translation -> [0, 1] from the dataset definition file.

CPU-only. The definition dict below mirrors the real generator input (dataset_ds_04.json): the
patient translation was sampled X in [-0.25, 0.25], Y in [-0.5, 0.5], Z never varied.
"""
import pytest
import torch

try:
    import RadFiled3D  # noqa: F401
except ImportError:  # pragma: no cover
    pytest.skip("RadFiled3D not available", allow_module_level=True)

from RadFiled3D.pytorch.types import TranslationalInput
from radfield3dnn.rftypes import TrainingInputData
from radfield3dnn.preprocessing.normalizations.translation import TranslationNormalization
from radfield3dnn.preprocessing.normalizations.beam_parameters import BeamParametersNormalization

DEFINITION = {
    "Metaparameters": {"MaxEnergy": 1.5e5, "BinCount": 32, "WorldDim": [1.28, 1.28, 1.28],
                       "VoxelSize": 0.02, "WorldMaterial": "Air"},
    "Parameters": [
        {"name": "source_distance", "range": [0.35, 0.75]},
        {"name": "source_shape", "value": "rectangle"},
        {"name": "source_opening_angle", "range": [[0.05, 0.05], [0.2, 0.2]]},
    ],
    "GeometryTransformations": {
        "patient": {"Translation": {"X": [-0.25, 0.25], "Y": [-0.5, 0.5]}}
    },
}


def _batch(translation):
    x = TranslationalInput(
        direction=torch.tensor([[0.0, 0.0, 1.0]]).expand(translation.shape[0], -1),
        origin=torch.full((translation.shape[0], 1), 0.5),
        spectrum=torch.full((translation.shape[0], 32), 1.0 / 32),
        translation=translation,
    )
    return TrainingInputData(input=x, ground_truth=torch.zeros(1))


def test_from_dataset_definition_ranges():
    n = TranslationNormalization.from_dataset_definition(DEFINITION)
    assert n.get_parameters()["translation_ranges_m"] == {"X": (-0.25, 0.25), "Y": (-0.5, 0.5)}


def test_normalizes_to_unit_range_and_zeroes_constant_axis():
    n = TranslationNormalization.from_dataset_definition(DEFINITION)
    t = torch.tensor([[-0.25, -0.5, 0.123],   # both minima; z arbitrary constant
                      [0.25, 0.5, 0.123],     # both maxima
                      [0.0, 0.0, 0.123]])     # centres
    out = n.forward(_batch(t)).input.translation
    expected = torch.tensor([[0.0, 0.0, 0.0],
                             [1.0, 1.0, 0.0],
                             [0.5, 0.5, 0.0]])
    assert torch.allclose(out, expected)
    assert isinstance(n.forward(_batch(t)).input, TranslationalInput)  # concrete type survives


def test_border_float_noise_is_forgiven_but_true_outliers_fail():
    n = TranslationNormalization.from_dataset_definition(DEFINITION)
    eps = torch.tensor([[0.25 + 1e-7, 0.5, 0.0]])
    out = n.forward(_batch(eps)).input.translation
    assert out[0, 0] == 1.0 and out[0, 1] == 1.0
    with pytest.raises(AssertionError, match="definition file"):
        n.forward(_batch(torch.tensor([[0.4, 0.0, 0.0]])))  # 0.4 m > X max of 0.25 m


def test_config_roundtrip():
    n = TranslationNormalization.from_dataset_definition(DEFINITION)
    n2 = TranslationNormalization.create_from_config(n.get_parameters())
    assert n2.get_parameters() == n.get_parameters()


def test_missing_transform_block_fails_loudly():
    with pytest.raises(AssertionError, match="GeometryTransformations"):
        TranslationNormalization.from_dataset_definition({"GeometryTransformations": {}})


def test_beam_normalizer_from_dataset_definition():
    b = BeamParametersNormalization.from_dataset_definition(DEFINITION)
    # the ctor stores ranges as fp32 -> compare approximately
    assert b.distance_range == pytest.approx((0.35, 0.75))
    assert b.half_field_size == pytest.approx((0.64, 0.64, 0.64))
    assert b.size_per_voxel == pytest.approx(0.02)
    # rectangle definition: the per-axis (w, h) ranges are metres, not a cone angle — the cone
    # branch placeholder must be present but the rect branch (field-extent scaling) is what runs.
    assert b.get_parameters()["distance_range_m"] == pytest.approx((0.35, 0.75))


def test_beam_normalizer_cone_definition():
    cone_def = {
        "Metaparameters": {"WorldDim": [1.0, 1.0, 1.0], "VoxelSize": 0.02},
        "Parameters": [
            {"name": "source_distance", "range": [0.2, 1.5]},
            {"name": "source_shape", "value": "cone"},
            {"name": "source_opening_angle", "range": [5.0, 40.0]},
        ],
    }
    b = BeamParametersNormalization.from_dataset_definition(cone_def)
    deg = b.get_parameters()["opening_angle_range_deg"]
    assert abs(deg[0] - 5.0) < 1e-4 and abs(deg[1] - 40.0) < 1e-4
