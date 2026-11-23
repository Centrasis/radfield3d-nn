"""Export a trained model as a deployable RF3M package.

The container, the inference runtime and the graphics interop live in the `radfiled3d-nn` module
(`from radfiled3d import nn`), which this package depends on — nothing about the byte layout or the
execution backends is re-implemented here. What stays is the part that needs the training stack:
turning a trained LightningModule into ONNX graphs plus the metadata that says what its tensors
mean.

To read a package back (metadata, or a session to run it), use the module directly::

    from radfiled3d.nn import deploy, inference
    meta = deploy.ModelStore.load_metadata("model.rf3m")
    session = inference.Session(deploy.ModelStore.load("model.rf3m"), backend="cuda", device=0)
"""

from .model_packager import ModelPackager

__all__ = ["ModelPackager"]
