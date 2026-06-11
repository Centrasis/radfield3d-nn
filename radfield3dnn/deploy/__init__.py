from typing import TYPE_CHECKING

from radfield3dnn.deploy.model_packager import ModelPackager

if TYPE_CHECKING:
    # resolved via typings/rfnn_deploy.pyi — gives load_rf3m a fully-typed return in the IDE
    from rfnn_deploy import VolumeFieldPredictor, VoxelFieldPredictor


def load_rf3m(path: str, use_cuda: bool = False) -> "VoxelFieldPredictor | VolumeFieldPredictor":
    """Load an RF3M package via the C++ ONNX deployment bindings (rfnn_deploy) straight to the
    runnable predictor. Imported lazily so importing :mod:`radfield3dnn.deploy` never requires the
    compiled deploy module to be built."""
    from radfield3dnn.deploy.onnx_runtime import load_rf3m as _load
    return _load(path, use_cuda=use_cuda)


__all__ = ["ModelPackager", "load_rf3m"]
