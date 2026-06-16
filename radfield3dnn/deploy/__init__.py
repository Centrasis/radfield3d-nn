from radfield3dnn.deploy.model_packager import ModelPackager

# The full static types for everything re-exported here live in the shipped inline stub
# `radfield3dnn/deploy/__init__.pyi` (PEP 561) — it is the authoritative typed surface for the
# deploy inference API and is what linters read. This module stays runtime-only: the native
# inference types are served lazily by `__getattr__` so importing the package never requires the
# compiled `rfnn_deploy` bindings to be built.


def load_rf3m(path: str, use_cuda: bool = False):
    """Load an RF3M package via the C++ ONNX deployment bindings (rfnn_deploy) straight to the
    runnable predictor. Imported lazily so importing :mod:`radfield3dnn.deploy` never requires the
    compiled deploy module to be built."""
    from radfield3dnn.deploy.onnx_runtime import load_rf3m as _load
    return _load(path, use_cuda=use_cuda)


def __getattr__(name: str):
    # Lazily re-export the deploy binding surface (ModelStore, BeamParameters, …) so they are
    # importable from `radfield3dnn.deploy` without forcing the compiled module to load on import.
    import importlib
    onnx_runtime = importlib.import_module("radfield3dnn.deploy.onnx_runtime")
    try:
        return getattr(onnx_runtime, name)
    except AttributeError:
        raise AttributeError(f"module 'radfield3dnn.deploy' has no attribute {name!r}") from None


# Only the names safe to expose without the compiled bindings go in the runtime __all__: the
# inference types are served lazily by __getattr__, so listing them here would make `import *`
# eagerly load the native module (and defeats the lazy design). They are still importable by name
# (`from radfield3dnn.deploy import BeamParameters`) and the full public surface — with types — is
# declared in the inline stub __init__.pyi.
__all__ = ["ModelPackager", "load_rf3m"]
