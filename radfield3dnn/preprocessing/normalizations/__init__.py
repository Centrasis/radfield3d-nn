from .base import Normalizer
from .linear import LinearNormalizer
from .asinh import AsinhTonemapNormalizer
from .lognormalizer import LogNormalizer
from .translation import TranslationNormalization
from typing import Union


class NormalizerConstructor:
    _LINEAR_KNOBS = frozenset({"dataset_scale"})

    @staticmethod
    def construct_by_name(name: Union[str, dict]) -> Normalizer:
        """`name` is a normalizer name, or a dict {"type": <name>, ...knobs} for the ones that take
        a parameter. `dataset_scale` (linear only) replaces each field's own maximum with one
        constant for the whole dataset — `statistics.json:min_max_fluences.Max`."""
        if isinstance(name, dict):
            spec = dict(name)
            kind = spec.pop("type", None)
            if kind not in ("linear0_1", "linear-1_1"):
                raise ValueError(f"Only linear normalizers take parameters, got {kind!r}.")
            unknown = set(spec) - NormalizerConstructor._LINEAR_KNOBS
            if unknown:
                raise ValueError(f"Unknown normalizer knobs {sorted(unknown)}; available: "
                                 f"{sorted(NormalizerConstructor._LINEAR_KNOBS)}.")
            rng = (0.0, 1.0) if kind == "linear0_1" else (-1.0, 1.0)
            return LinearNormalizer(rng, **spec)
        name = name.lower()
        if name == "linear0_1":
            return LinearNormalizer((0.0, 1.0))
        elif name == "linear-1_1":
            return LinearNormalizer((-1.0, 1.0))
        elif name == "asinh":
            return AsinhTonemapNormalizer(sigma=3e-3)
        elif name == "log_scale":
            return LogNormalizer()
        else:
            raise ValueError(
                f"Unknown normalizer: {name!r}. Available: {NormalizerConstructor.get_available_normalizers()}"
            )

    @staticmethod
    def instance2str(instance: Normalizer) -> str:
        # LogNormalizer subclasses LinearNormalizer, so it must be matched FIRST.
        if isinstance(instance, LogNormalizer):
            return "log_scale"
        elif isinstance(instance, AsinhTonemapNormalizer):
            return instance.get_type()
        elif isinstance(instance, LinearNormalizer):
            if (instance.range[0], instance.range[1]) in ((0.0, 1.0), (-1.0, 1.0)):
                return instance.get_type()   # "linear0_1" / "linear-1_1"
        raise ValueError(f"Non-standard normalizer cannot be serialized to simple string! ({instance})")

    @staticmethod
    def get_available_normalizers() -> list[str]:
        return ["linear0_1", "linear-1_1", "asinh", "log_scale"]
