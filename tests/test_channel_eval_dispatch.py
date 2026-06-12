"""Tests for the channel-eval dispatch of BaseNeuralRadFieldModel:

evaluate_forward runs a one-time TEST RUN of the model's forward (a tiny probe) to determine
whether the model predicts one channel (single/joined) or two channels (scatter + direct), then
binds the matching plain eval method (_calculate_metrics_single_channel / _join_gt / _join_pred /
_two_channel / _airkerma) for the lifetime of the instance. calculate_metrics is the public seam
(used by subclasses with their own evaluate_forward) and wires lazily from the first real
prediction. Covered here under BOTH conditions (single- and two-channel) on both the GT and the
prediction side.
"""
import torch
import pytest

from radfield3dnn.models.nerf import PBRFNet
from radfield3dnn.preprocessing.normalizations import LinearNormalizer
from radfield3dnn.rftypes import (TrainingInputData, RadiationField, RadiationFieldChannel,
                                  AirKermaField, DirectionalInput)

B, S, D = 2, 32, 8


@pytest.fixture()
def model():
    m = PBRFNet(d_model=32, out_spectra_dim=S, flux_loss="L1Plain", spectrum_loss="HistogramLoss",
                normalizer=LinearNormalizer((0.0, 1.0)), trunk_depth=2,
                location_encoding_params={"type": "sinusoidal", "pos_enc_dim": 4, "append_input": True},
                conditioning_params={"type": "Concat", "use_beam_shape": False})
    m.max_inner_batch_size = 2048
    return m


def _input():
    return DirectionalInput(direction=torch.randn(B, 3), origin=torch.rand(B, 1),
                            spectrum=torch.rand(B, S), geometry=None,
                            beam_shape_type=None, beam_shape_parameters=torch.rand(B, 2))


def _channel():
    return RadiationFieldChannel(flux=torch.rand(B, 1, D, D, D) * 1e-3,
                                 spectrum=torch.softmax(torch.rand(B, S, D, D, D), 1), error=None)


def _gt_single():
    return RadiationField(scatter_field=_channel(), direct_beam=None)


def _gt_two():
    return RadiationField(scatter_field=_channel(), direct_beam=_channel())


def _batch(gt):
    return TrainingInputData(input=_input(), ground_truth=gt, original_ground_truth=None)


# ── evaluate_forward: probe + wiring under both GT conditions ───────────────────────────────

def test_evaluate_forward_single_gt_wires_single_channel(model):
    metrics = model.evaluate_forward(_batch(_gt_single()))
    assert model._calc_metrics_impl.__func__ is PBRFNet._calculate_metrics_single_channel
    assert torch.isfinite(metrics.scatter_field.flux_loss).all()
    assert metrics.direct_beam is None


def test_evaluate_forward_split_gt_wires_join_gt(model):
    metrics = model.evaluate_forward(_batch(_gt_two()))
    # single-channel predictor + split GT -> the GT is physically joined, scored single-channel
    assert model._calc_metrics_impl.__func__ is PBRFNet._calculate_metrics_join_gt
    assert torch.isfinite(metrics.scatter_field.flux_loss).all()
    assert metrics.direct_beam is None


def test_wiring_is_bound_once(model):
    model.evaluate_forward(_batch(_gt_single()))
    bound = model._calc_metrics_impl
    model.evaluate_forward(_batch(_gt_single()))
    assert model._calc_metrics_impl is bound          # no re-probe / re-bind


# ── calculate_metrics (public seam): wiring from the given prediction ───────────────────────

def test_two_channel_prediction_with_split_gt(model):
    gt = _gt_two()
    pred = RadiationField(scatter_field=_channel(), direct_beam=_channel())
    metrics = model.calculate_metrics(pred, gt, _batch(gt))
    assert model._calc_metrics_impl.__func__ is PBRFNet._calculate_metrics_two_channel
    assert torch.isfinite(metrics.scatter_field.flux_loss).all()
    assert metrics.direct_beam is not None and torch.isfinite(metrics.direct_beam.flux_loss).all()


def test_two_channel_prediction_with_single_gt_joins_prediction(model):
    gt = _gt_single()
    pred = RadiationField(scatter_field=_channel(), direct_beam=_channel())
    metrics = model.calculate_metrics(pred, gt, _batch(gt))
    assert model._calc_metrics_impl.__func__ is PBRFNet._calculate_metrics_join_pred
    assert torch.isfinite(metrics.scatter_field.flux_loss).all()
    assert metrics.direct_beam is None                # joined before scoring


def test_airkerma_prediction(model):
    gt = AirKermaField(air_kerma=torch.rand(B, 1, D, D, D), geometry=None)
    pred = AirKermaField(air_kerma=torch.rand(B, 1, D, D, D), geometry=None)
    metrics = model.calculate_metrics(pred, gt, TrainingInputData(input=_input(), ground_truth=gt, original_ground_truth=None))
    assert model._calc_metrics_impl.__func__ is PBRFNet._calculate_metrics_airkerma
    assert torch.isfinite(metrics.airkerma_field).all()


# ── correctness of the join paths ────────────────────────────────────────────────────────────

def test_join_gt_perfect_prediction_gives_zero_loss(model):
    """End-to-end correctness of the join-gt path: a prediction equal to the physically joined
    (and re-normalised) target must score ~zero flux loss — i.e. the path joins the GT exactly the
    way the model's normalizer/ChannelsJoin pipeline defines the joined field."""
    batch = model._normalizer.forward(_batch(_gt_two()))     # mirror evaluate_forward's entry
    joined_y = model._join_field(batch.ground_truth)         # joined single channel (the target)
    joined_ch = joined_y.scatter_field if isinstance(joined_y, RadiationField) else joined_y
    pred = RadiationField(scatter_field=RadiationFieldChannel(
        flux=joined_ch.flux.clone(),
        spectrum=joined_ch.spectrum.clone(), error=None), direct_beam=None)
    metrics = model.calculate_metrics(pred, batch.ground_truth, batch)
    assert model._calc_metrics_impl.__func__ is PBRFNet._calculate_metrics_join_gt
    assert float(metrics.scatter_field.flux_loss.abs().max()) < 1e-6
