from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum, auto
from typing import Any, Protocol, TypeAlias, TypedDict

import torch

from nanofed.utils.logger import Logger

from .accountant import GaussianAccountant, PrivacySpent
from .config import PrivacyConfig
from .noise import GaussianNoiseGenerator

ModelState: TypeAlias = dict[str, torch.Tensor]


class PrivacyType(Enum):
    """Type of privacy mechanism."""

    CENTRAL = auto()
    LOCAL = auto()


class PrivacyMetrics(TypedDict):
    """Privacy-related metrics."""

    epsilon_spent: float
    delta_spent: float
    noise_scale: float
    clip_ratio: float  # Ratio of updates clipped


class PrivacyMechanism(Protocol):
    """PRotocol defining privacy mechanism interface."""

    def add_noise(
        self, parameters: ModelState, batch_size: int
    ) -> ModelState: ...

    def get_privacy_spent(self) -> PrivacySpent: ...

    @property
    @abstractmethod
    def privacy_type(self) -> PrivacyType: ...


@dataclass(slots=True, frozen=True)
class UpdateMetadata:
    """Metadata about model update."""

    total_norm: float
    clipped_norm: float
    num_parameters: int
    noise_scale: float


class BasePrivacyMechanism(ABC):
    """Base class for privacy mechanism."""

    def __init__(
        self,
        config: PrivacyConfig,
        accountant: GaussianAccountant | None = None,
        noise_generator: GaussianNoiseGenerator | None = None,
    ) -> None:
        self._config = config
        self._accountant = accountant or GaussianAccountant(config)
        self._noise_gen = noise_generator or GaussianNoiseGenerator()
        self._logger = Logger()

    @property
    @abstractmethod
    def privacy_type(self) -> PrivacyType:
        """Get type of privacy mechanism."""
        pass

    def _compute_noise_scale(self, batch_size: int) -> float:
        """Compute noise scale based on privacy parameters."""
        return (
            self._config.noise_multiplier
            * self._config.max_gradient_norm
            / batch_size
        )

    def _clip_update(
        self, parameters: ModelState, max_norm: float
    ) -> tuple[ModelState, UpdateMetadata]:
        """Clip parameters to specified maximum norm."""
        total_norm = torch.norm(
            torch.cat([p.flatten() for p in parameters.values()])
        )

        clip_coef = min(max_norm / (total_norm + 1e-6), 1.0)

        clipped = {k: p * clip_coef for k, p in parameters.items()}

        metadata = UpdateMetadata(
            total_norm=total_norm.item(),
            clipped_norm=(total_norm * clip_coef).item(),
            num_parameters=sum(p.numel() for p in parameters.values()),
            noise_scale=self._config.noise_multiplier,
        )

        return clipped, metadata

    def add_noise(self, parameters: ModelState, batch_size: int) -> ModelState:
        """Add privacy-preserving noise to parameters."""
        clipped, metadata = self._clip_update(
            parameters, self._config.max_gradient_norm
        )

        # Add calibrarted noise
        noise_scale = self._compute_noise_scale(batch_size)
        noised = {}
        for key, param in clipped.items():
            noise = self._noise_gen.generate(param.shape, scale=noise_scale)
            noised[key] = param + noise

        self._accountant.add_noise_event(
            sigma=self._config.noise_multiplier, samples=batch_size
        )

        self._logger.debug(
            f"Applied privacy mechanism: "
            f"norm={metadata.total_norm:.3f}->{metadata.clipped_norm:.3f}, "
            f"noise={noise_scale:.3f}"
        )

        return noised

    def get_privacy_spent(self) -> PrivacySpent:
        """Get current privacy budget consumption."""
        return self._accountant.get_privacy_spent()

    def validate_budget(self) -> bool:
        """Check if the privacy budget has been exceeded."""
        return self._accountant.validate_budget()


class CentralPrivacyMechanism(BasePrivacyMechanism):
    """Central differential privacy mechanism."""

    @property
    def privacy_type(self) -> PrivacyType:
        return PrivacyType.CENTRAL


class LocalPrivacyMechanism(BasePrivacyMechanism):
    """Local differential privacy mechanism."""

    @property
    def privacy_type(self) -> PrivacyType:
        return PrivacyType.LOCAL

    def add_noise(self, parameters: ModelState, batch_size: int) -> ModelState:
        """Add noise with local privacy guarantees."""
        # Local DP uses batch_size=1 since each update is individual
        return super().add_noise(parameters, batch_size=1)


class PrivacyMechanismFactory:
    """Factory for creating privacy mechanisms."""

    @staticmethod
    def create(
        privacy_type: PrivacyType, config: PrivacyConfig, **kwargs: Any
    ) -> BasePrivacyMechanism:
        """Create privacy mechanism of specified type."""
        if privacy_type == PrivacyType.CENTRAL:
            return CentralPrivacyMechanism(config, **kwargs)
        elif privacy_type == PrivacyType.LOCAL:
            return LocalPrivacyMechanism(config, **kwargs)
        else:
            raise ValueError(f"Unknown privacy type: {privacy_type}")
