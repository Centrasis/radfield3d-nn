"""Encapsulated optimizer/scheduler behaviours (one class per behaviour, shared interface).

``OptimizerBehaviour`` is the interface (and owns the fp16 fp32-master-weight setup); concrete
behaviours like ``CosineWithWarmup`` implement ``build()`` to return a model's
``configure_optimizers`` result. A model holds a behaviour and delegates to it.
"""
from .base import OptimizerBehaviour
from .cosine_warmup import CosineWithWarmup

__all__ = ["OptimizerBehaviour", "CosineWithWarmup"]
