"""Regression tests for the field-wise CNN models and the metric factory after the code audit."""
import os
import pytest
import torch

from radfield3dnn.models.cnn import Beam2ScatterUNet
from radfield3dnn.models.field_unet import FieldScatterUNet
from radfield3dnn.preprocessing.normalizations.linear import LinearNormalizer

MU_TR = "/mnt/data/const/mu_tr/emuen_rho_air_1keV-1MeV.txt"


def test_beam2scatterunet_constructs():
    m = Beam2ScatterUNet(d_model=16, out_spectra_bins=8, in_spectra_bins=8, out_dims=(8, 8, 8),
                         normalizer=LinearNormalizer((0, 1)))
    assert sum(p.numel() for p in m.parameters()) > 0


def test_fieldscatterunet_infers_voxel_size_from_resolution():
    m = FieldScatterUNet(d_model=8, depth=2, in_spectra_dim=8, out_spectra_bins=8, out_dims=(50, 50, 50))
    assert abs(m.voxel_size_m - 1.0 / 50.0) < 1e-9
    m2 = FieldScatterUNet(d_model=8, depth=2, in_spectra_dim=8, out_spectra_bins=8, out_dims=(25, 25, 25))
    assert abs(m2.voxel_size_m - 1.0 / 25.0) < 1e-9
    m3 = FieldScatterUNet(d_model=8, depth=2, in_spectra_dim=8, out_spectra_bins=8, out_dims=(50, 50, 50), voxel_size_m=0.02)
    assert abs(m3.voxel_size_m - 0.02) < 1e-9


@pytest.mark.skipif(not os.path.exists(MU_TR), reason="mu_tr table not present")
def test_metric_factory_single_source():
    from radfield3dnn.metrics.factory import build_airkerma_metrics
    metrics = build_airkerma_metrics(MU_TR, voxel_size_m=0.02)
    for key in ("global_airkerma_accuracy", "top90_airkerma_accuracy", "airkerma_accuracy_scatter",
                "airkerma_accuracy_beam", "spectrum_accuracy"):
        assert key in metrics
