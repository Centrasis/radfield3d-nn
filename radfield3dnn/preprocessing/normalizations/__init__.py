from .base import Normalizer
from .linear import LinearNormalizer, JointLinearNormalizer
from .logscale import LogScaleNormalizer
from .asinh import AsinhTonemapNormalizer, SplitChannelAsinhNormalizer


class NormalizerConstructor:
    @staticmethod
    def construct_by_name(name: str) -> Normalizer:
        name = name.lower()
        if name == "linear0_1":
            return LinearNormalizer((0.0, 1.0))
        elif name == "linear-1_1":
            return LinearNormalizer((-1.0, 1.0))
        elif name == "linear_joint":
            # Shared-scale linear for two-head models: both flux channels ÷ the SAME per-field max,
            # preserving the scatter:direct relation (pairs with ChannelMaxBalancedLoss + raw split).
            return JointLinearNormalizer((0.0, 1.0))
        elif name == "log_scale":
            return LogScaleNormalizer(x_min=1e-8, x_max=1.0, zero_floor=-9.0)
        elif name == "log_scale_1e6":
            # MC-noise-floored log scale: the HDR analysis showed ~1e-6 is the lowest real signal
            # decade (below it is Monte-Carlo noise), so clamp the log domain to [-6, 0] — flux below
            # 1e-6 folds onto the floor and true-zero/occluded voxels use the -7 sentinel. Pairs with
            # the data-adaptive flux-bias landing at the ~-3 log signal mean.
            return LogScaleNormalizer(x_min=1e-6, x_max=1.0, zero_floor=-7.0)
        elif name.startswith("logscale_1e"):
            # Round-trip parse of LogScaleNormalizer.get_type() -> "logscale_1e{lo}_1e{hi}_zf{zf}".
            import re
            mobj = re.fullmatch(r"logscale_1e(-?\d+)_1e(-?\d+)_zf(-?\d+)", name)
            if not mobj:
                raise ValueError(f"Malformed logscale normalizer string: {name}")
            lo, hi, zf = (int(g) for g in mobj.groups())
            return LogScaleNormalizer(x_min=10.0 ** lo, x_max=10.0 ** hi, zero_floor=float(zf))
        elif name == "asinh":
            return AsinhTonemapNormalizer(sigma=3e-3)
        elif name == "asinh_split":
            return SplitChannelAsinhNormalizer()
        elif name == "asinh_auto":
            # Sentinel — training pipeline rebuilds via SplitChannelAsinhNormalizer.from_dataset()
            return SplitChannelAsinhNormalizer()
        else:
            raise ValueError(f"Unknown normalizer: {name}")

    @staticmethod
    def instance2str(instance: Normalizer) -> str:
        if isinstance(instance, SplitChannelAsinhNormalizer):
            return instance.get_type()
        elif isinstance(instance, AsinhTonemapNormalizer):
            return instance.get_type()
        elif isinstance(instance, LogScaleNormalizer):
            return instance.get_type()
        elif isinstance(instance, JointLinearNormalizer):
            return "linear_joint"
        elif isinstance(instance, LinearNormalizer):
            if instance.range[0] == 0.0 and instance.range[1] == 1.0:
                return "linear0_1"
            elif instance.range[0] == -1.0 and instance.range[1] == 1.0:
                return "linear-1_1"
        raise ValueError(f"Non-standard normalizer cannot be serialized to simple string! ({instance})")

    @staticmethod
    def get_available_normalizers() -> list[str]:
        return ["linear0_1", "linear-1_1", "log_scale", "log_scale_1e6", "asinh", "asinh_split", "asinh_auto"]
