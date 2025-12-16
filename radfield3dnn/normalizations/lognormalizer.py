from torch import Tensor
import torch
from .linear import LinearNormalizer
from typing import Union


class LogNormalizer(LinearNormalizer):
    def __init__(self, range: tuple[float, float] = (-1.0, 1.0), epsilon: float = 1e-9, input_scale: float = 1e+3):
        super().__init__(range=range)
        self.epsilon = torch.tensor(epsilon, dtype=torch.float32, requires_grad=False)
        self.input_scale = torch.tensor(input_scale, dtype=torch.float32, requires_grad=False)

    def validate_range(self, x: Tensor):
        """
        Validate that the input tensor x is suitable for log normalization.
        Ensures that x is non-negative and has a sufficient range of values.
        Raises an error if the conditions are not met.
        """
        min_x = x.min()
        min_2nd = x[x > min_x].min() if (x > min_x).any() else min_x
        if min_2nd < self._eps_like(x).item() * 10:
            raise ValueError(f"Input to LogNormalizer has too small values. Minimum: {min_x.item()}, 2nd minimum: {min_2nd.item()}. Consider using a different normalizer.")

    def _eps_like(self, ref: Tensor) -> Tensor:
        # Ensure epsilon is representable in ref dtype/device
        eps = self.epsilon.to(device=ref.device, dtype=ref.dtype)
        # Use machine tiny (min positive normal) to avoid fp16 underflow to zero
        min_pos = torch.finfo(ref.dtype).tiny
        return torch.clamp(eps, min=min_pos)

    def _default_log_max(self, dtype: torch.dtype, device: torch.device) -> Tensor:
        value = torch.tensor(1.0, dtype=dtype, device=device)
        log_max = self.log(value)
        return torch.clamp(log_max, min=self._eps_like(log_max))

    def _prepare_log_reference(self, ref: Tensor, *, dtype: torch.dtype, device: torch.device) -> tuple[Tensor, Tensor]:
        ref = ref.to(device=device, dtype=dtype)
        self.validate_range(ref)
        ref_log = self.log(ref)
        log_max = torch.clamp(ref_log.max(), min=self._eps_like(ref_log))
        ref_norm = ref_log / log_max
        return log_max, ref_norm

    def apply_transformation(self, x: Tensor, respect_to: Union[Tensor, None]) -> Tensor:
        with torch.no_grad():
            if self.epsilon.device != x.device:
                self.epsilon = self.epsilon.to(x.device)
            if respect_to is not None and not isinstance(respect_to, Tensor):
                raise TypeError("respect_to must be a Tensor when normalizing a Tensor.")
            self.validate_range(x)
            if respect_to is not None:
                log_max, ref_for_linear = self._prepare_log_reference(
                    respect_to, dtype=x.dtype, device=x.device
                )
            else:
                log_max = self._default_log_max(dtype=x.dtype, device=x.device)
                ref_for_linear = None
            x = self.log(x)
            x = x / log_max
            assert torch.isfinite(x).all(), "Normalization resulted in non-finite values."
            return super().apply_transformation(x, respect_to=ref_for_linear)
        
    def log(self, x: Tensor) -> Tensor:
        x = x * self.input_scale
        x = torch.log1p(x)
        return x

    def inv_log(self, x: Tensor) -> Tensor:
        x = torch.expm1(x)
        x = x / self.input_scale
        return x

    def apply_inverse_transformation(self, x: Tensor, respect_to: Union[Tensor, None]) -> Tensor:
        if self.epsilon.device != x.device:
            self.epsilon = self.epsilon.to(x.device)
        if respect_to is not None and not isinstance(respect_to, Tensor):
            raise TypeError("respect_to must be a Tensor when inverting a Tensor.")
        if respect_to is not None:
            log_max, ref_for_linear = self._prepare_log_reference(
                respect_to, dtype=x.dtype, device=x.device
            )
        else:
            log_max = self._default_log_max(dtype=x.dtype, device=x.device)
            ref_for_linear = None
        log = super().apply_inverse_transformation(x, respect_to=ref_for_linear)
        log = log * log_max
        x = self.inv_log(log)
        assert torch.isfinite(x).all(), "Inverse normalization resulted in non-finite values."
        return x

    def get_type(self) -> str:
        exponent = int(torch.log10(self.input_scale).item())
        sign_prefix = "+" if exponent > 0 else ""
        scale_str = f"_1e{sign_prefix}{exponent}"
        return "log" + scale_str
    
    def __repr__(self):
        return f"LogNormalizer(range={self.range}, input_scale={self.input_scale.item()})"

    def clone(self) -> "LogNormalizer":
        new_int: LogNormalizer = super().clone()
        new_int.epsilon = self.epsilon.clone()
        new_int.range = (
            self.range[0],
            self.range[1]
        )
        #new_int.input_scale = self.input_scale.clone()
        return new_int
