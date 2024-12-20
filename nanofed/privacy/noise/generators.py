from functools import wraps
from typing import Callable, ParamSpec, TypeVar

import torch

from ..exceptions import NoiseGenerationError
from ..types import Shape, Tensor
from .base import BaseNoiseGenerator

P = ParamSpec("P")
T = TypeVar("T", bound=torch.Tensor)


def validate_noise_input(func: Callable[P, T]) -> Callable[P, T]:
    """Decorator to validate noise generation inputs."""

    @wraps(func)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
        shape = args[1] if len(args) > 1 else kwargs.get("shape")
        scale = args[2] if len(args) > 2 else kwargs.get("scale")

        if not shape:
            raise ValueError("Shape must be provided")

        if not isinstance(shape, tuple):
            raise ValueError("Shape must be a tuple")

        if not all(isinstance(d, int) and d > 0 for d in shape):
            raise ValueError(
                "Invalid shape: must be a tuple of positive integers"
            )

        if not isinstance(scale, (int, float)):
            raise ValueError("Scale must be a number")

        if scale <= 0:
            raise ValueError("Scale must be positive")

        try:
            return func(*args, **kwargs)
        except Exception as e:
            raise NoiseGenerationError(
                f"Noise generation failed: {str(e)}"
            ) from e

    return wrapper


class GaussianNoiseGenerator(BaseNoiseGenerator):
    """Gaussian noise generator implementation."""

    @validate_noise_input
    def generate(self, shape: Shape, scale: float) -> Tensor:
        return torch.randn(shape, generator=self._rng) * scale


class LaplacianNoiseGenerator(BaseNoiseGenerator):
    """Laplacian noise generator implementation."""

    @validate_noise_input
    def generate(self, shape: Shape, scale: float) -> Tensor:
        uniform = torch.rand(shape, generator=self._rng)
        return (
            torch.sign(uniform - 0.5)
            * scale
            * torch.log1p(-2 * torch.abs(uniform - 0.5))
        )
