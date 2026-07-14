import torch
from torch import Tensor
from .sinusoidal_encoding import SinusoidalFrequencyEncoding


class IntegratedSinusoidalEncoding(SinusoidalFrequencyEncoding):
    """Sinusoidal encoding of a UNIFORM BOX region instead of a point (mip-NeRF-style IPE).

    mip-NeRF's integrated positional encoding damps each frequency by the closed-form expectation
    of the sinusoid over the query region (google/mipnerf ``expected_sin``:
    ``E[sin(x)] = exp(-0.5*var)*sin(mean)`` for a Gaussian region). The regions here are axis-aligned
    voxel CUBES of side ``region_width``, for which the expectation is exact:

        E[sin(f*x)] over x ~ U[mu - w/2, mu + w/2]  =  sin(f*mu) * sinc(f*w/2)

    so each frequency level's (sin, cos) pair is scaled by ``sinc(2^l*pi*w/2)`` -- the REGION STATE
    (one float per frequency level). It is produced by ``compute_region_state`` and handed to
    ``forward``, which keeps the region an ordinary graph input: a deployed consumer recomputes it
    only when the queried grid resolution changes (LOD) and reuses it across every query of that
    grid.

    Widths are in the encoder's own input frame. With no region state and no configured
    ``region_width`` this degrades to the parent's exact point encoding, so "sinusoidal" and "ipe"
    checkpoints stay interchangeable. The appended raw input (``append_input``) is never damped --
    E[x] is the region mean itself.
    """

    def __init__(self, pos_enc_dim: int, d_input: int, append_input: bool = False, dim: int = -1,
                 region_width: float = None, auto_region_width: bool = False):
        # pure-PyTorch path only: the fused tcnn Frequency encoding cannot be damped internally.
        super().__init__(pos_enc_dim, d_input, append_input=append_input, dim=dim, use_tcnn=False)
        self.region_width = region_width
        # auto_region_width: the region follows whatever grid the caller announces via
        # set_query_region / forward2volume. Off by default so "ipe" with no region configured is
        # numerically identical to "sinusoidal".
        self.auto_region_width = bool(auto_region_width)
        self._cached_width = None
        self._cached_state = None

    @property
    def region_state_dims(self) -> int:
        return self.pos_enc_dim

    def compute_region_state(self, region_width: Tensor) -> Tensor:
        """Voxel width -> per-frequency damping sinc(2^l*pi*w/2). Traceable; exported as a graph."""
        half = self._freqs.to(region_width.dtype) * (region_width.reshape(()) * 0.5)
        return torch.where(half.abs() < 1e-12, torch.ones_like(half), torch.sin(half) / half)

    def set_query_region(self, region_width: float | None) -> None:
        # An explicitly configured region_width is authoritative and is never overridden by the
        # caller's grid; auto mode tracks whatever grid is being queried.
        if self.auto_region_width:
            self.region_width = region_width

    def _state_for_width(self, width: float, device, dtype) -> Tensor:
        if self._cached_state is None or self._cached_width != width or self._cached_state.device != device:
            w = torch.tensor(float(width), device=device, dtype=torch.float32)
            self._cached_state = self.compute_region_state(w)
            self._cached_width = width
        return self._cached_state.to(dtype)

    def forward(self, x: Tensor, region_state: Tensor | None = None) -> Tensor:
        assert x.shape[-1] == self.d_input, f"Input tensor last dim should be {self.d_input}, got {x.shape[-1]}"
        if region_state is None:
            if self.auto_region_width and self.region_width is None:
                raise RuntimeError(
                    "IntegratedSinusoidalEncoding(auto_region_width=True) was queried without a "
                    "region. Query through the model's volume assembly, call "
                    "model.set_query_region(voxel_width_in_encoder_frame), or pass a region_state "
                    "from compute_region_state().")
            if self.region_width is not None:
                region_state = self._state_for_width(float(self.region_width), x.device, x.dtype)

        ang = x.unsqueeze(-1) * self._freqs.to(x.dtype)                     # (..., d_input, P)
        enc = torch.stack([torch.sin(ang), torch.cos(ang)], dim=-1)         # (..., d_input, P, 2)
        if region_state is not None:
            enc = enc * region_state.to(enc.dtype).view(1, -1, 1)
        enc = enc.reshape(*x.shape[:-1], 2 * self.pos_enc_dim * self.d_input)
        if self.append_input:
            enc = torch.cat([enc, x], dim=-1)
        return enc
