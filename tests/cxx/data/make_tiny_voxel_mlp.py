"""Regenerate tiny_voxel_mlp.onnx, the checked-in fixture for tests/cxx/test_model_io.cpp.

A minimal position-only per-voxel model: input `position` [N,3] (the name is what makes
field_predictors.cpp classify it as a voxel model), outputs `flux` [N,1] and `spectrum` [N,32].
The predictor identifies outputs structurally — the trailing dim 32 is read back as the
spectrum bin count, matching ModelDomain.spectrum_bins = 32 in the C++ test. Weights are
deterministic so the fixture is reproducible byte-for-byte.

Run from the repo root:  python tests/cxx/data/make_tiny_voxel_mlp.py
"""
import os

import numpy as np
from onnx import TensorProto, helper, numpy_helper

OUT_BINS = 32  # must match ModelDomain.spectrum_bins in test_model_io.cpp

Wf = np.zeros((3, 1), dtype=np.float32)
Wf[0, 0] = 1.0
Ws = np.linspace(0.0, 1.0, num=3 * OUT_BINS, dtype=np.float32).reshape(3, OUT_BINS)

position = helper.make_tensor_value_info("position", TensorProto.FLOAT, ["N", 3])
flux = helper.make_tensor_value_info("flux", TensorProto.FLOAT, ["N", 1])
spectrum = helper.make_tensor_value_info("spectrum", TensorProto.FLOAT, ["N", OUT_BINS])

graph = helper.make_graph(
    [helper.make_node("MatMul", ["position", "Wf"], ["flux"]),
     helper.make_node("MatMul", ["position", "Ws"], ["spectrum"])],
    "tiny_voxel_mlp", [position], [flux, spectrum],
    [numpy_helper.from_array(Wf, "Wf"), numpy_helper.from_array(Ws, "Ws")],
)
model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
model.ir_version = 9  # within the bundled ONNX Runtime's supported IR range

out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tiny_voxel_mlp.onnx")
with open(out, "wb") as f:
    f.write(model.SerializeToString())
print(f"wrote {out} ({os.path.getsize(out)} bytes)")
