from torch import Tensor
from torch import nn
import torch


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
        permeability = 1 - permeability
        nn.init.constant_(self.linear.bias, torch.logit(torch.tensor(permeability)))
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


class GatedFusion(nn.Module):
    def __init__(self, d_model: int, d_cond: int, hidden: int = 128, non_linearity: type[nn.Module] = nn.SiLU):
        """
        d_xyz: dimension of xyz-encoding per voxel
        d_cond: dimension of conditioning token (can be mean-pooled if tokens)
        out_dim: output dim (defaults to d_xyz)
        """
        super().__init__()
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

        hx = torch.tanh(self.proj_x(x))         # (B,d_model)
        hc = torch.tanh(self.proj_cond(cond))   # (B,d_model)

        z = torch.sigmoid(self.z_x(x) + self.z_cond(cond) + self.gate(torch.cat([x, cond], dim=-1)))

        return z * hx + (1 - z) * hc
    
    def initialize(self):
        nn.init.xavier_uniform_(self.proj_x.weight)
        nn.init.xavier_uniform_(self.proj_cond.weight)
        nn.init.zeros_(self.proj_x.bias)
        nn.init.zeros_(self.proj_cond.bias)
        nn.init.xavier_uniform_(self.z_x.weight)
        nn.init.xavier_uniform_(self.z_cond.weight)
