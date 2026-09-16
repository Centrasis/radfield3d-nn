from torch import Tensor
from torch import nn
import torch

from .base import FusionBase


class ModulativeSigmoidGate(nn.Module):
    """
    A modulative sigmoid gate that controls the flow of information through a neural network.
    """

    def __init__(self, control_features_dim: int, main_path_dim: int):
        super().__init__()
        self.sigmoid = nn.Sigmoid()
        self.linear = nn.Linear(control_features_dim, main_path_dim)

    def forward(self, x: Tensor, y: Tensor) -> Tensor:
        """
        Applies the modulative sigmoid gate to the main input tensor.
        Args:
            x (Tensor): The main input tensor.
            y (Tensor): The control input tensor.
        Returns:
            Tensor: The output tensor after applying the modulative sigmoid gate to x.
        """
        gate = self.sigmoid(self.linear(y))
        return x * gate
    
    def set_gate_permeability(self, permeability: float):
        """
        Sets the permeability of the gate.
        A permeability of zero means the gate filters most aggressively.
        A permeability of one means the gate is fully open.
        Args:
            permeability (float): The permeability value to set. (0..1)
        """
        # gate = sigmoid(bias) at init (weight is zeroed), and x*gate flows. permeability=1 must give
        # gate=1 (fully open) => bias=logit(1)=+inf. The previous `1-permeability` flip inverted this
        # (permeability=1 -> logit(0)=-inf -> gate=0 = fully closed). Use permeability directly and
        # clamp away from the 0/1 singularities of logit.
        p = float(min(max(permeability, 1e-4), 1.0 - 1e-4))
        nn.init.constant_(self.linear.bias, torch.logit(torch.tensor(p)).item())
        nn.init.zeros_(self.linear.weight)


class ResidualAdditiveTanhGate(nn.Module):
    """
    A residual additive tanh gate that controls the flow of information through a neural network.
    """

    def __init__(self, control_features_dim: int, main_path_dim: int):
        super().__init__()
        self.tanh = nn.Tanh()
        self.control_feat_proj = nn.Linear(control_features_dim, main_path_dim) if control_features_dim != main_path_dim else nn.Identity()
        self.linear = nn.Linear(main_path_dim, main_path_dim)

    def forward(self, x: Tensor, y: Tensor) -> Tensor:
        """
        Applies the residual additive tanh gate to the main input tensor.
        Args:
            x (Tensor): The main input tensor.
            y (Tensor): The control input tensor.
        Returns:
            Tensor: The output tensor after applying the residual additive tanh gate to x.
        """
        proj = self.control_feat_proj(y)
        gate = self.tanh(self.linear(proj))
        return x + proj * gate


class GatedFusion(FusionBase):
    """GRU/GMU-style gated fusion: a learned per-channel convex blend between transformed x and cond.

    Fusion contract (:class:`FusionBase`): ``forward(x, cond) -> y``, dim(y)==dim(x)==d_model.

    Identity-at-init: the x-branch ``h_x = proj_x(x)`` is init to identity (no tanh squashing it) and
    the gate ``z`` is opened fully toward x (``z → 1``), so ``y ≈ x`` at step 0 — matching the stability
    of FiLM/ResidualFiLM/ConcatLinear. The beam (cond) branch is bounded by tanh and mixed in only as
    ``z`` learns to drop below 1.
    """
    def __init__(self, d_model: int, d_cond: int, hidden: int = 128, non_linearity: type[nn.Module] = nn.SiLU):
        """
        d_model: dimension of the main (location) feature per voxel and of the output.
        d_cond: dimension of the conditioning (beam) vector.
        """
        super().__init__(condition_channels=d_cond, out_channels=d_model)
        self.proj_cond  = nn.Linear(d_cond, d_model)
        self.proj_x     = nn.Linear(d_model, d_model)
        self.z_x        = nn.Linear(d_model, d_model, bias=True)
        self.z_cond     = nn.Linear(d_cond, d_model, bias=True)
        self.gate = nn.Sequential(
            nn.Linear(d_model + d_cond, hidden),
            non_linearity(inplace=True),
            nn.Linear(hidden, d_model)
        )
        self.d_model = d_model
        self.initialize()

    def forward(self, x: torch.Tensor, cond: torch.Tensor):
        """
        xyz_feat: (N, d_model) or (B, N, d_model)
        cond: either (d_cond,), (B, d_cond), or (B, T, d_cond) or (N, d_cond)
        returns (same batch/voxel dims, d_model)
        """
        assert x.shape[-1] == self.d_model, f"Invalid feature dims: ({x.shape[-1]} x {self.d_model})"

        hx = self.proj_x(x)                     # (B,d_model) — identity-init -> x (no tanh squash)
        hc = torch.tanh(self.proj_cond(cond))   # (B,d_model) — bounded beam branch

        z = torch.sigmoid(self.z_x(x) + self.z_cond(cond) + self.gate(torch.cat([x, cond], dim=-1)))

        return z * hx + (1 - z) * hc

    def initialize(self):
        # Identity-at-init: hx = identity(x); gate z -> 1 so y = z*hx + (1-z)*hc ≈ x.
        nn.init.eye_(self.proj_x.weight)        # h_x = x
        nn.init.zeros_(self.proj_x.bias)
        nn.init.xavier_uniform_(self.proj_cond.weight)
        nn.init.zeros_(self.proj_cond.bias)
        # z-logits: zero the data-dependent paths, drive z -> 1 via a large +bias on the gate head.
        for lin in (self.z_x, self.z_cond):
            nn.init.zeros_(lin.weight)
            nn.init.zeros_(lin.bias)
        for m in self.gate:
            if isinstance(m, nn.Linear):
                nn.init.zeros_(m.weight)
                nn.init.zeros_(m.bias)
        nn.init.constant_(self.gate[-1].bias, 8.0)  # z = σ(8) ≈ 0.99966 ≈ 1 -> y ≈ x
