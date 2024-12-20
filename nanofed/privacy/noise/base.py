from abc import ABC, abstractmethod
from typing import Protocol

import torch

from ..types import Shape, Tensor


class NoiseGenerator(Protocol):
    """Protocol for noise generation."""

    def generate(self, shape: Shape, scale: float) -> Tensor: ...
    def set_seed(self, seed: int) -> None: ...


class BaseNoiseGenerator(ABC):
    """Abstract base class for noise generator."""

    def __init__(self, seed: int | None = None) -> None:
        self._rng = torch.Generator()
        if seed is not None:
            self.set_seed(seed)

    def set_seed(self, seed: int) -> None:
        """Set the random seed for reproducibility."""
        self._rng.manual_seed(seed)

    @abstractmethod
    def generate(self, shape: Shape, scale: float) -> Tensor:
        """Generate noise tensor."""
        pass
