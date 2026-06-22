from torch import nn, Tensor
import torch
import math

from .base import FusionBase


class CrossAttentionFusion(FusionBase):
    """Cross-attention fusion: the voxel feature ``x`` (a single query) attends over ``n_tokens``
    tokens projected from the beam latent ``cond``, and the attended value is residually added to x.

    ``y = x + Wo · MHA(q=x, kv=tokens(cond))``  →  dim(y) == dim(x).

    Content-adaptive conditioning as a drop-in same-dim fusion. Heaviest
    of the lightweight set: q/k/v/out projections (4·d²) + a cond→tokens projection (d·n_tokens·d). The
    fusion-influence benchmark and the 90-fps budget judge whether its extra authority pays for the cost.

    Identity-startable: the output projection ``Wo`` is zero-initialised, so ``y = x`` at step 0 and the
    attention is learned in — matching FiLM/ConcatLinear/GatedFusion stability.
    """

    def __init__(self, condition_channels: int, out_channels: int,
                 n_heads: int = 4, n_tokens: int = 4, non_linearity: type[nn.Module] | None = None):
        super().__init__(condition_channels, out_channels)
        assert out_channels % n_heads == 0, "out_channels must be divisible by n_heads."
        self.n_heads = int(n_heads)
        self.n_tokens = int(n_tokens)
        self.head_dim = out_channels // n_heads
        self.norm_x = nn.LayerNorm(out_channels)
        self.to_tokens = nn.Linear(condition_channels, n_tokens * out_channels)  # cond -> tokens
        self.q = nn.Linear(out_channels, out_channels)
        self.k = nn.Linear(out_channels, out_channels)
        self.v = nn.Linear(out_channels, out_channels)
        self.out = nn.Linear(out_channels, out_channels)
        self.non_linearity = non_linearity(inplace=True) if non_linearity is not None else None
        self.initialize()

    def initialize(self) -> None:
        with torch.no_grad():
            nn.init.zeros_(self.out.weight)  # residual identity at init
            nn.init.zeros_(self.out.bias)

    def forward(self, x: Tensor, cond: Tensor) -> Tensor:
        self._check(x, cond)
        B = x.shape[0]
        H, Dh, T = self.n_heads, self.head_dim, self.n_tokens

        h = self.norm_x(x)
        q = self.q(h).view(B, H, 1, Dh)                                   # one query per voxel
        tok = self.to_tokens(cond).view(B, T, self.out_channels)         # beam -> T tokens
        k = self.k(tok).view(B, T, H, Dh).transpose(1, 2)                # B,H,T,Dh
        v = self.v(tok).view(B, T, H, Dh).transpose(1, 2)                # B,H,T,Dh

        attn = (q @ k.transpose(-1, -2)) / math.sqrt(Dh)                  # B,H,1,T
        attn = attn.softmax(dim=-1)
        ctx = (attn @ v).reshape(B, self.out_channels)                   # B,out

        y = x + self.out(ctx)                                            # residual (zero-init -> x)
        return self.non_linearity(y) if self.non_linearity is not None else y


class TokenCrossAttentionFusion(FusionBase):
    """Cross-attention over a SET of *meaningful* conditioning tokens — FusionBase-compatible.

    Same ``forward(x, cond)`` contract as every other fusion: ``cond`` is the per-parameter beam
    encodings **concatenated** into one vector of width ``condition_channels = n_tokens · token_dim``.
    The fusion **reshapes/splits the last dim** to ``[B, n_tokens, token_dim]`` and lets the voxel query
    attend/route over those tokens — "for this location, how much does direction vs distance vs spectrum
    matter?". This is the set-conditioning regime where attention conditioning is meant to win (Rebain
    et al. 2022, "Attention Beats Concatenation for Conditioning Neural Fields"); unlike
    :class:`CrossAttentionFusion` the tokens are the real per-parameter encodings, not a reshape of one
    pooled vector.

    Residual + zero-init output ⇒ identity at init (starts as x, learns to attend).
    """

    def __init__(self, condition_channels: "int | list[int] | tuple[int, ...]", out_channels: int,
                 n_heads: int = 4, non_linearity: type[nn.Module] | None = None):
        """``condition_channels`` may be either:

        * an **int** — the total length of the (concatenated) conditioning vector; treated as a single
          token spanning the whole vector, or
        * a **list/tuple of ints** — the length of each individual encoded parameter (must sum to the
          conditioning vector length, checked by ``torch.split`` in forward); one token per parameter.

        Either way the per-token projections live inside this fusion.
        """
        if isinstance(condition_channels, (list, tuple)):
            component_dims = [int(d) for d in condition_channels]
        else:
            component_dims = [int(condition_channels)]   # one token over the whole cond vector
        super().__init__(int(sum(component_dims)), out_channels)
        assert out_channels % n_heads == 0, "out_channels must be divisible by n_heads."
        self.component_dims = component_dims              # per-parameter encoded widths
        self.n_tokens = len(self.component_dims)
        self.n_heads = int(n_heads)
        self.head_dim = out_channels // n_heads
        self.norm_x = nn.LayerNorm(out_channels)
        # one projection per beam parameter (direction / distance / spectrum / [opening-angle]) into the
        # common token space — these live inside the fusion, not the backbone.
        self.token_projs = nn.ModuleList([nn.Linear(d, out_channels) for d in self.component_dims])
        self.q = nn.Linear(out_channels, out_channels)
        self.k = nn.Linear(out_channels, out_channels)
        self.v = nn.Linear(out_channels, out_channels)
        self.out = nn.Linear(out_channels, out_channels)
        self.non_linearity = non_linearity(inplace=True) if non_linearity is not None else None
        self.initialize()

    def initialize(self) -> None:
        with torch.no_grad():
            nn.init.zeros_(self.out.weight)  # residual identity at init
            nn.init.zeros_(self.out.bias)

    def forward(self, x: Tensor, cond: Tensor) -> Tensor:
        self._check(x, cond)
        B, H, Dh, T = x.shape[0], self.n_heads, self.head_dim, self.n_tokens
        parts = torch.split(cond, self.component_dims, dim=-1)            # split the concatenated encodings
        tokens = torch.stack([proj(p) for proj, p in zip(self.token_projs, parts)], dim=1)  # [B, T, out]
        h = self.norm_x(x)
        q = self.q(h).view(B, H, 1, Dh)
        k = self.k(tokens).view(B, T, H, Dh).transpose(1, 2)
        v = self.v(tokens).view(B, T, H, Dh).transpose(1, 2)
        attn = (q @ k.transpose(-1, -2)) / math.sqrt(Dh)
        attn = attn.softmax(dim=-1)
        ctx = (attn @ v).reshape(B, self.out_channels)
        y = x + self.out(ctx)
        return self.non_linearity(y) if self.non_linearity is not None else y
