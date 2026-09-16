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

    It has no ONNX lowering, so for deployment it ships as an RF3M stage of compiled CUDA rather
    than as part of a graph — see `deploy_stage` and `radfield3dnn/deploy/hashgrid_kernel.py`.
    """

    DEPLOY_ENTRY_POINT = "hashgrid_encode"

    def __init__(self,
                 in_dim=3,                  # Eingabedimension (z.B. 3 für 3D)
                 n_levels=16,
                 features_per_level=2,
                 base_resolution=16,
                 log2_hashmap_size=19,
                 per_level_scale=2.0,
                 input_domain=(0.0, 1.0)):
        super().__init__()

        if tcnn is None:
            raise ImportError("tinycudann (tcnn) is required for HashGridEncoding.")

        # tcnn's grid indexes the UNIT cube: a coordinate outside [0,1] wraps modulo the table and
        # aliases onto unrelated cells. `input_domain` (lo, hi) is the range the caller's
        # coordinates actually span; it is mapped affinely onto [0,1] and clamped there.
        lo, hi = (float(v) for v in input_domain)
        if not hi > lo:
            raise ValueError(f"input_domain must be (lo, hi) with hi > lo, got {input_domain!r}")
        self.input_domain = (lo, hi)
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

    def table_parameters(self):
        return self.encoding.parameters()

    def calc_encoded_dim(self) -> int:
        return self.encoding.n_output_dims

    # ── deployment: this encoder ships as compiled code, not as part of a graph ───────────────
    def deploy_stage(self) -> str:
        """tcnn's grid is a CUDA custom op with no ONNX lowering, so it travels as its own stage."""
        return "hashgrid"

    def deploy_stage_blocks(self) -> dict:
        from radfield3dnn.deploy.hashgrid_kernel import compile_ptx, cuda_source
        # PTX only: a virtual architecture the driver JITs for whatever card is present. A cubin
        # would load faster and would have to be built and shipped per architecture.
        return {"cuda_ptx": compile_ptx(cuda_source(self, self.DEPLOY_ENTRY_POINT))}

    def deploy_stage_weights(self):
        from radfield3dnn.deploy.hashgrid_kernel import table_weights
        table = table_weights(self)
        # f16, because that is the precision tcnn computes in and therefore the precision the
        # trained trunk was fed from — see hashgrid_kernel.table_weights.
        return table.tobytes(), int(table.size), "f16"

    def deploy_stage_launch(self) -> dict:
        # One thread per query row, so the derived grid (ceil(rows / block)) is exactly right and
        # one descriptor serves every resolution a consumer chooses.
        return {"cuda_ptx": dict(entry_point=self.DEPLOY_ENTRY_POINT,
                                 block_size=[256, 1, 1], grid_size=[0, 0, 0],
                                 shared_memory_bytes=0)}
    
    def ensure_device(self, device):
        if self.encoding is not None and self.encoding.params.device != device:
            self.encoding.to(device)
    
    def forward(self, x: Tensor, region_state: Tensor | None = None) -> Tensor:
        # region_state is unused by a point encoding (region_state_dims == 0), but the location
        # encoder is always CALLED with it (nerf.py encode_location) — dropping it from the
        # signature makes this encoder unusable as a location encoder.
        assert x.shape[-1] == self.in_dim, f"Input tensor last dim should be {self.in_dim}, got {x.shape[-1]}"
        self.ensure_device(x.device)
        orig_shape = x.shape[:-1]
        lo, hi = self.input_domain
        if (lo, hi) != (0.0, 1.0):
            x = ((x - lo) / (hi - lo)).clamp(0.0, 1.0)
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
