from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Protocol

from ..config import PrivacyConfig


@dataclass(frozen=True)
class PrivacySpent:
    """Privacy budget consumption tracking."""

    epsilon_spent: float
    delta_spent: float

    def validate(self, config: PrivacyConfig) -> bool:
        """Validate against privacy budget."""
        return (
            self.epsilon_spent <= config.epsilon
            and self.delta_spent <= config.delta
        )


class PrivacyAccountant(Protocol):
    """Protocol for privacy budget accounting."""

    def get_privacy_spent(self) -> PrivacySpent: ...
    def add_noise_event(self, sigma: float, samples: int) -> None: ...
    def validate_budget(self, config: PrivacyConfig) -> bool: ...


class BasePrivacyAccountant(ABC):
    """Base class for privacy accountants."""

    def __init__(self, config: PrivacyConfig) -> None:
        """Initialize privacy accountant."""
        self._config = config
        self._privacy_spent = PrivacySpent(0.0, 0.0)
        self._event_count = 0

    @abstractmethod
    def _compute_privacy_spent(self) -> PrivacySpent:
        """Compute current privacy consumption."""
        pass

    def get_privacy_spent(self) -> PrivacySpent:
        """Get current privacy budget consumption."""
        return self._compute_privacy_spent()

    def validate_budget(self, config: PrivacyConfig | None = None) -> bool:
        """Validate current privacy consumption against budget."""
        config = config or self._config
        spent = self.get_privacy_spent()
        return bool(spent.validate(config))
