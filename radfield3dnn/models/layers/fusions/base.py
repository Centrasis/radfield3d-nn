from torch import nn, Tensor


class FusionBase(nn.Module):
    """Common interface for two-vector feature fusions.

    A fusion merges a *main* vector ``x`` (here: the per-voxel location feature) with a
    *conditioning* vector ``cond`` (here: the per-field beam latent) and returns a vector of the
    **same dimensionality as x**::

        forward(x: Tensor[..., out_channels], cond: Tensor[..., condition_channels])
            -> Tensor[..., out_channels]

    Convention (shared by every subclass): the constructor is
    ``__init__(condition_channels, out_channels, ...)`` and a fusion is treated as **one logical
    trunk layer** when counting model depth (PBRF uses trunk_depth=6 + 2 fusions = logical depth 8).

    Subclasses implement :meth:`forward`. :meth:`_check` validates the contract.
    """

    def __init__(self, condition_channels: int, out_channels: int):
        super().__init__()
        self.condition_channels = int(condition_channels)
        self.out_channels = int(out_channels)

    def _check(self, x: Tensor, cond: Tensor) -> None:
        assert x.size(0) == cond.size(0), \
            f"Batch of x ({x.size(0)}) and cond ({cond.size(0)}) must match."
        assert x.shape[-1] == self.out_channels, \
            f"Expected x last-dim {self.out_channels}, got {x.shape[-1]}."
        assert cond.shape[-1] == self.condition_channels, \
            f"Expected cond last-dim {self.condition_channels}, got {cond.shape[-1]}."

    def forward(self, x: Tensor, cond: Tensor) -> Tensor:  # pragma: no cover - interface
        raise NotImplementedError("FusionBase subclasses must implement forward(x, cond).")
