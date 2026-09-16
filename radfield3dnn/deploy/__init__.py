"""Export a trained model as a deployable RF3M package.

The container, the inference runtime and the graphics interop live in the `radfiled3d-nn` module
(`from RadFiled3D import nn`), which this package depends on — nothing about the byte layout or the
execution backends is re-implemented here. What stays is the part that needs the training stack:
turning a trained LightningModule into ONNX graphs plus the metadata that says what its tensors
mean.

To read a package back (metadata, or a session to run it), use the module directly::

    from RadFiled3D.nn import deploy, inference
    meta = deploy.read_metadata("model.rf3m")
    session = inference.load_rf3m("model.rf3m", backend="cuda", device=0)
"""

from .model_packager import ModelPackager

__all__ = ["ModelPackager"]
