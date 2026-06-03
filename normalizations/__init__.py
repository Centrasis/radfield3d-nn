from .base import Normalizer
from .linear import LinearNormalizer
from .lognormalizer import LogNormalizer
from .asinh import AsinhNormalizer, LearnableAsinhNormalizer


class NormalizerConstructor:
    @staticmethod
    def construct_by_name(name: str) -> Normalizer:
        name = name.lower()
        if name == "linear0_1":
            return LinearNormalizer((0.0, 1.0))
        elif name == "linear-1_1":
            return LinearNormalizer((-1.0, 1.0))
        elif name == "log_1e+3":
            return LogNormalizer((0.0, 1.0), input_scale=1e+3)
        elif name == "log_1e+5":
            return LogNormalizer((0.0, 1.0), input_scale=1e+5)
        elif name == "asinh_1e-3":
            return AsinhNormalizer((0.0, 1.0), input_scale=1e-3)
        elif name == "learnable_asinh":
            return LearnableAsinhNormalizer((0.0, 1.0))
        else:
            raise ValueError(f"Unknown normalizer: {name}")

    @staticmethod
    def instance2str(instance: Normalizer) -> str:
        if isinstance(instance, LogNormalizer):
            return "log"
        elif isinstance(instance, AsinhNormalizer):
            return "asinh"
        elif isinstance(instance, LinearNormalizer):
            if instance.range[0] == 0.0 and instance.range[1] == 1.0:
                return "linear0_1"
            elif instance.range[0] == -1.0 and instance.range[1] == 1.0:
                return "linear-1_1"
        raise ValueError(f"Non-standard normalizer cannot be serialized to simple string! ({instance})")
    
    @staticmethod
    def get_available_normalizers() -> list[str]:
        return [
            "linear0_1",
            "linear-1_1",
            "log_1e+3",
            "log_1e+5",
            "asinh_1e-3",
            "learnable_asinh"
        ]
