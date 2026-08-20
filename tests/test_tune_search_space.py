"""Optuna search-space handling: dict-valued hyperparameter choices round-trip as JSON strings.

Optuna's sqlite storage and resume-time distribution checks want primitive categorical choices;
suggest_parameters encodes structured blocks (conditioning_params, location_encoding_params)
to JSON for the draw and decodes the winner.
"""
import json

import pytest

try:
    import RadFiled3D  # noqa: F401
    from tasks.tune import suggest_parameters
except ImportError:  # pragma: no cover
    pytest.skip("tune stack not available", allow_module_level=True)


class _FakeTrial:
    def __init__(self, picks=None):
        self.picks = picks or {}
        self.seen = {}

    def suggest_categorical(self, name, choices):
        # the contract that keeps the sqlite study clean on resume
        assert all(isinstance(c, (str, int, float, bool, type(None))) for c in choices), \
            f"non-primitive Optuna choice for {name}: {choices!r}"
        self.seen[name] = choices
        return choices[self.picks.get(name, 0)]


SPACE = {
    "d_model": [192, 256, 384],
    "conditioning_params": [
        {"type": "Concat", "use_beam_shape": True},
        {"type": "FiLM", "use_beam_shape": True},
    ],
}


def test_plain_values_pass_through():
    s = suggest_parameters(_FakeTrial({"d_model": 1}), SPACE)
    assert s["d_model"] == 256 and isinstance(s["d_model"], int)


def test_dict_choices_json_roundtrip():
    trial = _FakeTrial({"conditioning_params": 1})
    s = suggest_parameters(trial, SPACE)
    assert s["conditioning_params"] == {"type": "FiLM", "use_beam_shape": True}
    # what Optuna saw were JSON strings, decodable and stable-keyed
    assert all(isinstance(c, str) for c in trial.seen["conditioning_params"])
    assert json.loads(trial.seen["conditioning_params"][0])["type"] == "Concat"


def test_non_list_space_rejected():
    with pytest.raises(ValueError, match="list/tuple"):
        suggest_parameters(_FakeTrial(), {"d_model": 192})


def test_shipped_search_space_is_valid():
    cfg = json.load(open("configs/tpbrfnet_ds04.model.json"))
    space = cfg["hyperparameter_space"]
    assert set(space) == {"d_model", "location_encoding_params",
                          "conditioning_params.translation_encoding"}
    assert space["d_model"] == [128, 192, 256, 384]
    assert {e["type"] for e in space["location_encoding_params"]} == {"sinusoidal", "hash"}
    assert {t["type"] for t in space["conditioning_params.translation_encoding"]} == {"fourier", "relative"}
    s = suggest_parameters(_FakeTrial(), space)
    assert set(s) == set(space)


def test_apply_suggestions_flat_and_dotted():
    from tasks.tune import apply_suggestions
    base = {"d_model": 192,
            "conditioning_params": {"type": "Concat", "use_beam_shape": True,
                                    "translation_encoding": {"type": "fourier"}}}
    out = apply_suggestions(base, {"d_model": 384,
                                   "conditioning_params.translation_encoding": {"type": "relative"}})
    assert out["d_model"] == 384
    assert out["conditioning_params"]["translation_encoding"] == {"type": "relative"}
    assert out["conditioning_params"]["type"] == "Concat"          # siblings untouched
    # the base config is never mutated (it is reused across trials)
    assert base["d_model"] == 192
    assert base["conditioning_params"]["translation_encoding"]["type"] == "fourier"


def test_apply_suggestions_rejects_bad_paths():
    from tasks.tune import apply_suggestions
    with pytest.raises(KeyError, match="typo_params"):
        apply_suggestions({"conditioning_params": {}}, {"typo_params.translation_encoding": {}})


def test_dotted_axis_builds_the_right_models():
    # end to end: the exact shipped space drives ModelConstructor to the intended variants
    import copy, torch
    from tasks.tune import apply_suggestions
    from radfield3dnn.models import ModelConstructor
    cfg = json.load(open("configs/tpbrfnet_ds04.model.json"))
    space = cfg["hyperparameter_space"]
    for pick, expect_relative in ((0, False), (1, True)):
        sugg = suggest_parameters(_FakeTrial({"conditioning_params.translation_encoding": pick,
                                              "d_model": 0}), space)
        params = apply_suggestions(cfg["parameters"], sugg)
        assert params["d_model"] == 128
        mc = {"model_name": "TPBRFNet", "parameters": {**params, "trunk_depth": 2}}
        if sugg["location_encoding_params"]["type"] == "hash":
            mc["parameters"]["location_encoding_params"] = space["location_encoding_params"][0]  # no tcnn in CI
        m = ModelConstructor.create_model_from_dict(mc)()
        assert getattr(m.get_core_model(), "_relative_translation", False) is expect_relative
