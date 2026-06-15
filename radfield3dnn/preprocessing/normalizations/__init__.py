from .base import Normalizer
from .linear import LinearNormalizer
from .asinh import AsinhTonemapNormalizer


class NormalizerConstructor:
    @staticmethod
    def construct_by_name(name: str) -> Normalizer:
        name = name.lower()
        if name == "linear0_1":
            return LinearNormalizer((0.0, 1.0))
        elif name == "asinh":
            return AsinhTonemapNormalizer(sigma=3e-3)
        else:
            raise ValueError(f"Unknown normalizer: {name}")

    @staticmethod
    def instance2str(instance: Normalizer) -> str:
        if isinstance(instance, AsinhTonemapNormalizer):
            return instance.get_type()
        elif isinstance(instance, LinearNormalizer):
            if instance.range[0] == 0.0 and instance.range[1] == 1.0:
                return "linear0_1"
        raise ValueError(f"Non-standard normalizer cannot be serialized to simple string! ({instance})")

    @staticmethod
    def get_available_normalizers() -> list[str]:
        return ["linear0_1", "asinh"]
