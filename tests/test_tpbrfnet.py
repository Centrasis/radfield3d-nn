"""TPBRFNet: PBRFNet + vec3 patient-translation beam parameter (TranslationalInput; x, y used).

CPU-only; verifies the subclass wires the extra encoder into the beam vector without touching
PBRFNet, that the translation actually conditions the output, and that TranslationalInput
survives the shared input-rebuild sites (forward2volume path).
"""
import pytest
import torch
import torch.nn.functional as F

try:
    import RadFiled3D  # noqa: F401
except ImportError:  # pragma: no cover
    pytest.skip("RadFiled3D not available", allow_module_level=True)

from radfield3dnn.models.nerf import PBRFNet
from radfield3dnn.models.nerf_translation import TPBRFNet, TranslationalInput
from radfield3dnn.preprocessing.normalizations.linear import LinearNormalizer


def _model_kwargs():
    return dict(
        d_model=64,
        location_encoding_params={"type": "sinusoidal", "pos_enc_dim": 6, "append_input": True},
        direction_encoding_params={"type": "spherical_harmonics", "degree": 4, "append_input": True},
        spectra_encoding_params={"type": "simple", "in_spectra_dim": 32, "encoded_spectra_dims": 32},
        out_spectra_dim=32, conditioning_params={"type": "Concat", "use_beam_shape": False},
        normalizer=LinearNormalizer((0.0, 1.0)), flux_loss="SMAPEBalanced",
        flux_activation="sigmoid", trunk_depth=2,
        training_params={"voxel_supersampling": 1, "voxels_centered_around_origin": False},
    )


def _build_model():
    torch.manual_seed(0)
    m = TPBRFNet(**_model_kwargs())
    m.max_inner_batch_size = 4096
    return m


def _field_input(B=2, seed=1, translation=None):
    g = torch.Generator().manual_seed(seed)
    direction = F.normalize(torch.randn(B, 3, generator=g), dim=-1)
    origin = torch.rand(B, 1, generator=g) * 0.4 + 0.35
    spectrum = torch.rand(B, 32, generator=g)
    spectrum = spectrum / spectrum.sum(-1, keepdim=True)
    if translation is None:
        translation = torch.rand(B, 3, generator=g) - 0.5
    return TranslationalInput(direction=direction, origin=origin, spectrum=spectrum,
                              translation=translation)


def test_registered_and_extends_pbrfnet():
    from radfield3dnn.models import ModelConstructor
    names = [c.__model_name__ for c in ModelConstructor.get_subclasses(__import__("radfield3dnn.models.base", fromlist=["BaseNeuralRadFieldModel"]).BaseNeuralRadFieldModel) if "__model_name__" in c.__dict__]
    assert "TPBRFNet" in names
    torch.manual_seed(0)
    base = PBRFNet(**_model_kwargs())
    trans = _build_model()
    n_base = sum(p.numel() for p in base.parameters())
    n_trans = sum(p.numel() for p in trans.parameters())
    # extra params = translation_encoder + the widened beam_encoder input Linear, nothing else
    assert n_trans > n_base
    d = trans.get_core_model().scalar_encoding_dims
    enc_params = sum(p.numel() for p in trans.get_core_model().translation_encoder.parameters())
    widened = d * trans.get_core_model().d_model
    assert n_trans - n_base == enc_params + widened


def _activate_beam_conditioning(m):
    # ConcatLinear identity-starts with the cond-half of its weight at ZERO (no beam influence at
    # init, by design) — randomize it so translation sensitivity/gradients are observable.
    for c in (m.get_core_model().beam_conditioner1, m.get_core_model().beam_conditioner2):
        torch.nn.init.xavier_uniform_(c.proj.weight)


def test_beam_encoding_reacts_to_translation():
    m = _build_model()
    x_a = _field_input(B=2, translation=torch.full((2, 3), -0.4))
    x_b = x_a._replace(translation=torch.full((2, 3), 0.4))
    core = m.get_core_model()
    with torch.no_grad():
        assert not torch.allclose(core.encode_additional_parameters(x_a),
                                  core.encode_additional_parameters(x_b))


def test_forward2volume_and_translation_sensitivity():
    m = _build_model()
    _activate_beam_conditioning(m)
    vc = torch.tensor([6, 6, 6])
    x_a = _field_input(B=2, translation=torch.full((2, 3), -0.4))
    x_b = x_a._replace(translation=torch.full((2, 3), 0.4))
    with torch.no_grad():
        out_a = m.forward2volume(x_a, vc, spectra_bins=32)
        out_b = m.forward2volume(x_b, vc, spectra_bins=32)
    flux_a, flux_b = out_a.scatter_field.flux, out_b.scatter_field.flux
    assert torch.isfinite(flux_a).all() and flux_a.min() >= 0.0 and flux_a.max() <= 1.0
    # only the translation differs -> the predicted field must differ
    assert not torch.allclose(flux_a, flux_b)


def test_translation_encoder_receives_gradient():
    m = _build_model()
    _activate_beam_conditioning(m)
    x = _field_input(B=2)
    out = m.forward2volume(x, torch.tensor([4, 4, 4]), spectra_bins=32)
    out.scatter_field.flux.sum().backward()
    grads = [p.grad for p in m.get_core_model().translation_encoder.parameters()]
    assert all(g is not None for g in grads)
    assert any(g.abs().sum() > 0 for g in grads)


def test_input_type_survives_rebuild_sites():
    # the shared rebuild sites use _replace now — the concrete input type and its translation
    # field must survive them (geometry drop, unbatch->batch unsqueeze, per-batch extraction)
    x = _field_input(B=2)
    dropped = x._replace(geometry=None)
    assert isinstance(dropped, TranslationalInput) and dropped.translation is not None
    single = x._replace(**{f: v[0] for f, v in zip(x._fields, x) if isinstance(v, torch.Tensor)})
    rebatched = single._replace(**{f: v.unsqueeze(0) for f, v in zip(single._fields, single) if isinstance(v, torch.Tensor)})
    assert isinstance(rebatched, TranslationalInput) and rebatched.translation.shape == (1, 3)
    from radfield3dnn.models.base import BaseNeuralRadFieldModel
    one = BaseNeuralRadFieldModel.extract_input_from_batch(None, x, 0)
    assert isinstance(one, TranslationalInput) and one.translation.shape == (3,)


def test_missing_translation_fails_loudly():
    from radfield3dnn.rftypes import DirectionalInput
    m = _build_model()
    x = _field_input(B=2)
    plain = DirectionalInput(direction=x.direction, origin=x.origin, spectrum=x.spectrum)
    with pytest.raises(AssertionError, match="translation"):
        m.forward2volume(plain, torch.tensor([4, 4, 4]), spectra_bins=32)


def _rect_model():
    torch.manual_seed(0)
    kwargs = _model_kwargs()
    kwargs["conditioning_params"] = {"type": "Concat", "use_beam_shape": True, "beam_shape_param_dims": 2}
    m = TPBRFNet(**kwargs)
    m.max_inner_batch_size = 4096
    return m


def _rect_input(rect):
    x = _field_input(B=2)
    return x._replace(beam_shape_type=torch.full((2, 1), 3.0),  # FieldShape.RECTANGLE
                      beam_shape_parameters=rect)


def test_rectangle_collimation_both_dims_condition_output():
    m = _rect_model()
    _activate_beam_conditioning(m)
    core = m.get_core_model()
    assert core.opening_angle_encoder[0].in_features == 4  # (w, h, area, ratio)
    x_a = _rect_input(torch.tensor([[0.3, 0.2]] * 2))
    x_b = _rect_input(torch.tensor([[0.3, 0.5]] * 2))  # only h differs
    with torch.no_grad():
        enc_a = core.encode_additional_parameters(x_a)
        enc_b = core.encode_additional_parameters(x_b)
        assert not torch.allclose(enc_a, enc_b)
        out_a = m.forward2volume(x_a, torch.tensor([4, 4, 4]), spectra_bins=32)
        out_b = m.forward2volume(x_b, torch.tensor([4, 4, 4]), spectra_bins=32)
    assert not torch.allclose(out_a.scatter_field.flux, out_b.scatter_field.flux)


def test_beam_shape_features_cone_and_rect():
    from radfield3dnn.models.nerf import PBRFNet as P
    m = _rect_model()
    core = m.get_core_model()
    params = torch.tensor([[0.4, 0.2]])
    feats = core._beam_shape_features(params)
    w, h = 0.4, 0.2
    expected = torch.tensor([[w, h, w * h, (w - h) / (w + h + 1e-6)]])
    assert torch.allclose(feats, expected)
    # cone path (default dims=1) is byte-identical to the old behavior
    torch.manual_seed(0)
    cone = TPBRFNet(**_model_kwargs() | {"conditioning_params": {"type": "Concat", "use_beam_shape": True}})
    cone_core = cone.get_core_model()
    assert cone_core.opening_angle_encoder[0].in_features == 1
    angle = torch.tensor([[0.7, 999.0]])  # trailing columns must be ignored for cone
    assert torch.equal(cone_core._beam_shape_features(angle), torch.tensor([[0.7]]))


def test_translation_fourier_encoding_is_default():
    from radfield3dnn.models.encoders.sinusoidal_encoding import SinusoidalFrequencyEncoding
    m = _build_model()
    enc = m.get_core_model().translation_encoder
    assert isinstance(enc[0], SinusoidalFrequencyEncoding)
    assert enc[0].pos_enc_dim == 6 and enc[0].d_input == 2 and enc[0].append_input
    # projection consumes the full Fourier width: 2 dims * 6 freqs * (sin, cos) + appended input
    assert enc[1].in_features == 2 * 6 * 2 + 2
    # resolved encoding is persisted into the stored config (checkpoint reload stability)
    stored = m.get_custom_parameters()["conditioning_params"]["translation_encoding"]
    assert stored["type"] == "fourier" and stored["n_frequencies"] == 6


def test_translation_mlp_fallback_matches_pre_fourier_structure():
    # {"type": "mlp"} must rebuild the ORIGINAL Linear(2, d) stack so checkpoints trained before
    # the Fourier default keep loading.
    torch.manual_seed(0)
    kwargs = _model_kwargs()
    kwargs["conditioning_params"] = {"type": "Concat", "use_beam_shape": False,
                                     "translation_encoding": {"type": "mlp"}}
    m = TPBRFNet(**kwargs)
    enc = m.get_core_model().translation_encoder
    assert isinstance(enc[0], torch.nn.Linear) and enc[0].in_features == 2
    d = m.get_core_model().scalar_encoding_dims
    assert sum(p.numel() for p in enc.parameters()) == (2 * d + d) + (d * d + d)


def test_fourier_translation_conditions_output():
    # normalized-domain inputs ([0,1], as TranslationNormalization delivers): the Fourier encoder
    # must separate two translations and drive a different predicted field.
    m = _build_model()
    _activate_beam_conditioning(m)
    x_a = _field_input(B=2, translation=torch.full((2, 3), 0.1))
    x_b = x_a._replace(translation=torch.full((2, 3), 0.9))
    with torch.no_grad():
        out_a = m.forward2volume(x_a, torch.tensor([4, 4, 4]), spectra_bins=32)
        out_b = m.forward2volume(x_b, torch.tensor([4, 4, 4]), spectra_bins=32)
    assert not torch.allclose(out_a.scatter_field.flux, out_b.scatter_field.flux)


def test_unknown_translation_encoding_rejected():
    kwargs = _model_kwargs()
    kwargs["conditioning_params"] = {"type": "Concat", "use_beam_shape": False,
                                     "translation_encoding": {"type": "hash"}}
    with pytest.raises(ValueError, match="translation_encoding"):
        TPBRFNet(**kwargs)


def test_random_input_carries_translation_and_forwards():
    # The batch-size search (on_fit_start) feeds _generate_random_input straight into forward()
    # with global_parameters=None -> the beam encoder runs on it. A plain PositionalInput here
    # crashed every TPBRFNet run at startup with "requires a TranslationalInput".
    from radfield3dnn.rftypes import TranslationalPositionalInput
    m = _build_model()
    x = m._generate_random_input(device="cpu")
    assert isinstance(x, TranslationalPositionalInput)
    assert x.translation.shape == (2, 3)
    assert x.position is not None          # still a per-voxel input
    with torch.no_grad():
        out = m.forward(x)
    assert torch.isfinite(out.scatter_field.flux).all()


def test_random_input_shapes_follow_the_model_config():
    # Widths must come from the model's own config: a rectangle model (beam_shape_param_dims=2)
    # indexes beam_shape_parameters[:, 1], so a hardcoded [B, 1] raised IndexError.
    rect = _rect_model()
    x = rect._generate_random_input(device="cpu")
    assert x.beam_shape_parameters.shape == (2, 2)
    with torch.no_grad():
        rect.forward(x)   # must not raise
    cone = _build_model()
    assert cone._generate_random_input(device="cpu").beam_shape_parameters.shape == (2, 1)


def test_random_input_spectrum_is_the_dataset_width_not_the_rebin_width():
    # in_spectra_dim (32) is what the encoder REBINS to; the model's input contract is the
    # dataset's tube-spectrum width (150 by default, configurable) — and that is what the ONNX
    # export traces, so the deployed graph must not declare the internal width.
    m = _build_model()   # spectra_encoding_params: in_spectra_dim=32
    assert m._generate_random_input(device="cpu").spectrum.shape == (2, 150)
    torch.manual_seed(0)
    kwargs = _model_kwargs()
    kwargs["spectra_encoding_params"] = {"type": "simple", "in_spectra_dim": 32,
                                         "encoded_spectra_dims": 32, "input_spectra_dim": 64}
    assert TPBRFNet(**kwargs)._generate_random_input(device="cpu").spectrum.shape == (2, 64)


def test_positional_like_preserves_translation_across_rebuilds():
    from radfield3dnn.rftypes import positional_like, TranslationalPositionalInput, PositionalInput
    x = _field_input(B=2)
    rebuilt = positional_like(x, direction=x.direction, origin=x.origin, spectrum=x.spectrum,
                              position=torch.rand(2, 3))
    assert isinstance(rebuilt, TranslationalPositionalInput)
    assert torch.equal(rebuilt.translation, x.translation)
    # per-chunk re-indexing: an explicit translation overrides the template's
    idx = torch.tensor([0, 0, 1])
    chunked = positional_like(x, direction=x.direction[idx], origin=x.origin[idx],
                              spectrum=x.spectrum[idx], position=torch.rand(3, 3),
                              translation=x.translation[idx])
    assert chunked.translation.shape == (3, 3)
    # a template WITHOUT translation still yields a plain PositionalInput
    from radfield3dnn.rftypes import DirectionalInput
    plain = DirectionalInput(direction=x.direction, origin=x.origin, spectrum=x.spectrum)
    assert type(positional_like(plain, direction=x.direction, origin=x.origin,
                                spectrum=x.spectrum, position=torch.rand(2, 3))) is PositionalInput


@pytest.mark.slow
def test_beam_encoder_export_declares_the_translation_input():
    # The deploy runtime binds ONNX inputs BY NAME ("translation" -> PatientTranslation3D), so a
    # missing/misnamed input silently breaks the exported package.
    onnx = pytest.importorskip("onnx")
    pytest.importorskip("onnxscript")
    import tempfile, os
    from radfield3dnn.models import ModelExporter
    m = _rect_model()
    path = os.path.join(tempfile.mkdtemp(), "beam.onnx")
    ModelExporter.onnx_export_beam_encoder(m, path)
    names = [i.name for i in onnx.load(path).graph.input]
    assert names == ["direction", "distance", "spectrum", "beam_shape_parameters", "translation"]


def test_pbrfnet_unaffected():
    from radfield3dnn.rftypes import DirectionalInput
    torch.manual_seed(0)
    m = PBRFNet(**_model_kwargs())
    m.max_inner_batch_size = 4096
    x = _field_input(B=2)
    plain = DirectionalInput(direction=x.direction, origin=x.origin, spectrum=x.spectrum)
    with torch.no_grad():
        out = m.forward2volume(plain, torch.tensor([4, 4, 4]), spectra_bins=32)
    assert torch.isfinite(out.scatter_field.flux).all()
