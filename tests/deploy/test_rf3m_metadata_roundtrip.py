"""Deploy round-trip: store an RF3M package, load it, and feed it a beam whose spectrum is sized
from the package metadata — the model must run without an ONNX shape error.

This is the deployment-side guard for the input-spectrum mismatch that broke the UE5 plugin: the
model's INPUT tube-spectrum length (the "spectrum" beam-parameter / ONNX input) is distinct from the
OUTPUT per-voxel histogram bins (ModelDomain.spectrum_bins). Feeding a spectrum of the wrong length
makes ONNX Runtime reject the input ("Got invalid dimensions for input: spectrum"). The fix exposes
the true input length via VolumeFieldPredictor.input_spectrum_bins (read straight from the graph); this
test asserts that a metadata/graph-sized spectrum runs and a wrong-sized one fails.

Pure ONNX-Runtime deploy path — no CUDA / torch / tcnn. Skips if the deploy bindings or onnx aren't
available (e.g. bindings not compiled in this environment).
"""

import pytest

np = pytest.importorskip("numpy")
onnx = pytest.importorskip("onnx")

# Prefer the packaged entry point (it pre-loads ONNX Runtime), but fall back to importing the compiled
# `rfnn_deploy` module directly so CI can run this without the torch-coupled `radfield3dnn` package.
rd = None
try:
    from radfield3dnn.deploy.onnx_runtime import rfnn_deploy as rd  # noqa: F401
except Exception:
    try:
        import rfnn_deploy as rd  # CI: PYTHONPATH -> built .so, LD_LIBRARY_PATH -> ONNX Runtime
    except Exception:
        rd = None
if rd is None:
    pytest.skip("rfnn_deploy bindings not available", allow_module_level=True)

from onnx import helper, numpy_helper, TensorProto

IN_BINS = 150        # the model's INPUT tube-spectrum length
OUT_BINS = 32        # the model's OUTPUT per-voxel histogram bins (distinct from the input)
MAX_ENERGY_EV = 1.5e5


def _make_trunk_onnx() -> bytes:
    """A minimal single-graph per-voxel trunk: inputs `position` [N,3] + `spectrum` [N,IN_BINS],
    outputs `flux` [N,1] + `out_spectrum` [N,OUT_BINS]. The `position` input makes it a voxel model;
    the `spectrum` input's trailing dim (IN_BINS) is what input_spectrum_bins must report."""
    Wf = np.zeros((3, 1), dtype=np.float32); Wf[0, 0] = 1.0
    Ws = np.full((IN_BINS, OUT_BINS), 1.0 / IN_BINS, dtype=np.float32)

    position = helper.make_tensor_value_info("position", TensorProto.FLOAT, ["N", 3])
    spectrum = helper.make_tensor_value_info("spectrum", TensorProto.FLOAT, ["N", IN_BINS])
    flux = helper.make_tensor_value_info("flux", TensorProto.FLOAT, ["N", 1])
    ospec = helper.make_tensor_value_info("out_spectrum", TensorProto.FLOAT, ["N", OUT_BINS])

    graph = helper.make_graph(
        [helper.make_node("MatMul", ["position", "Wf"], ["flux"]),
         helper.make_node("MatMul", ["spectrum", "Ws"], ["out_spectrum"])],
        "trunk", [position, spectrum], [flux, ospec],
        [numpy_helper.from_array(Wf, "Wf"), numpy_helper.from_array(Ws, "Ws")],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 9   # within the bundled ONNX Runtime's supported IR range
    return model.SerializeToString()


def _domain():
    # spectrum_bins is the OUTPUT histogram; the "spectrum" beam-parameter count is the INPUT length.
    return rd.ModelDomain(
        spectrum_bins=OUT_BINS,
        spectrum_max_energy_ev=MAX_ENERGY_EV,
        beam_parameters=[
            rd.BeamParameterSpec("direction", 3, rd.ParameterRange(-1.0, 1.0, "")),
            rd.BeamParameterSpec("spectrum", IN_BINS, rd.ParameterRange(0.0, MAX_ENERGY_EV, "eV")),
        ],
    )


def _beam(spectrum):
    return rd.BeamParameters(direction=[0.0, 0.0, -1.0], origin=[0.5, 0.5, 0.5],
                             spectrum=list(spectrum), rect=[0.0, 0.0])


def _load_roundtrip():
    data = rd.save_to_memory(
        {"trunk": _make_trunk_onnx()}, _domain(),
        rd.ModelProvenance(dataset_name="unit-test", software_version="", physics=""), {})
    assert isinstance(data, (bytes, bytearray)) and len(data) > 0
    return rd.ModelStore.load_from_memory(data)


def test_rf3m_roundtrip_reports_input_and_output_spectrum_layout():
    pred = _load_roundtrip()
    # INPUT length is read from the ONNX graph; OUTPUT bins from the spectrum output.
    assert pred.input_spectrum_bins == IN_BINS
    assert pred.spectrum_bins == OUT_BINS
    # The package metadata also carries the input layout (the "spectrum" beam parameter).
    spec_bp = next(bp for bp in pred.domain.beam_parameters if bp.name == "spectrum")
    assert spec_bp.count == IN_BINS
    assert spec_bp.range.unit == "eV" and spec_bp.range.max == pytest.approx(MAX_ENERGY_EV)


def test_rf3m_predicts_with_metadata_sized_spectrum():
    pred = _load_roundtrip()
    # Build the spectrum at the length the package declares — this must run cleanly.
    spectrum = np.full(pred.input_spectrum_bins, 1.0 / pred.input_spectrum_bins, dtype=np.float32)
    out = pred.predict_volume(_beam(spectrum), (4, 4, 4))
    flux = np.asarray(out["flux"])
    assert flux.size == 4 * 4 * 4
    assert np.isfinite(flux).all()


def test_rf3m_wrong_spectrum_length_is_rejected():
    pred = _load_roundtrip()
    # Half the required length -> ONNX Runtime must reject the input shape (the original bug).
    bad = np.ones(pred.input_spectrum_bins // 2, dtype=np.float32)
    with pytest.raises(Exception):
        pred.predict_volume(_beam(bad), (2, 2, 2))
