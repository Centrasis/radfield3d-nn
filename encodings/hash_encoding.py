import torch
from .base import EncodingBase
from torch import Tensor
from torch.amp import autocast
from .sinusoidal_encoding import SinusoidalFrequencyEncoding
try:
    import tinycudann as tcnn
except ImportError as e:
    tcnn = None


class HashGridEncoding(EncodingBase):
    """
    Hash Encoding for 3D coordinates using a hash table using Tiny CUDA Neural Networks (tcnn).
    This encoding maps 3D coordinates to a higher-dimensional space using multiple levels of hash grids.
    Each level has a different resolution, allowing the model to capture both coarse and fine details.
    """

    def __init__(self,
                 in_dim=3,                  # Eingabedimension (z.B. 3 für 3D)
                 n_levels=16,
                 features_per_level=2,
                 base_resolution=16,
                 log2_hashmap_size=19,
                 per_level_scale=2.0):
        super().__init__()

        if tcnn is None:
            raise ImportError("tinycudann (tcnn) is required for HashGridEncoding.")

        self.n_levels = n_levels
        self.features_per_level = features_per_level
        self.in_dim = in_dim
        self.base_resolution = base_resolution
        self.log2_hashmap_size = log2_hashmap_size
        self.per_level_scale = per_level_scale
        self.encoding = tcnn.Encoding(
            n_input_dims=in_dim,
            encoding_config={
                "otype": "HashGrid",
                "n_levels": n_levels,
                "n_features_per_level": features_per_level,
                "log2_hashmap_size": log2_hashmap_size,
                "base_resolution": base_resolution,
                "per_level_scale": per_level_scale,
                "interpolation": "Linear"
            }
        )
    
    def to(self, *args, **kwargs):
        super().to(*args, **kwargs)
        self.encoding.to(*args, **kwargs)
        return self

    def calc_encoded_dim(self) -> int:
        return self.encoding.n_output_dims
    
    def ensure_device(self, device):
        if self.encoding is not None and self.encoding.params.device != device:
            self.encoding.to(device)
    
    def forward(self, x: Tensor) -> Tensor:
        assert x.shape[-1] == self.in_dim, f"Input tensor last dim should be {self.in_dim}, got {x.shape[-1]}"
        self.ensure_device(x.device)
        orig_shape = x.shape[:-1]
        x = x.contiguous()  # tcnn requires contiguous input
        x = x.view(-1, self.in_dim)

        with autocast(device_type=x.device.type, enabled=False):
            out: Tensor = self.encoding(x)
        if out.dtype != x.dtype:
            out = out.to(x.dtype)
        return out.contiguous().view(*orig_shape, -1)


class VoxelHashGridEncoding(HashGridEncoding):
    def __init__(self, voxel_count: int, n_levels=16, features_per_level=2, base_resolution=16, log2_hashmap_size=19):
        per_level_scale = (voxel_count / base_resolution) ** (1 / (n_levels - 1))

        super().__init__(
            in_dim=3,
            n_levels=n_levels,
            features_per_level=features_per_level,
            base_resolution=base_resolution,
            log2_hashmap_size=log2_hashmap_size,
            per_level_scale=per_level_scale
        )
