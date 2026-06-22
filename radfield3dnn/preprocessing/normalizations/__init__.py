from .base import Normalizer
from .linear import LinearNormalizer
from .asinh import AsinhTonemapNormalizer
from .lognormalizer import LogNormalizer


class NormalizerConstructor:
    @staticmethod
    def construct_by_name(name: str) -> Normalizer:
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
