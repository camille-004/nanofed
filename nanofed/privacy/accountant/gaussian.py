import math

from ..config import PrivacyConfig
from .base import BasePrivacyAccountant, PrivacySpent


class GaussianAccountant(BasePrivacyAccountant):
    """Privacy accountant for Gaussian mechanism."""

    def __init__(self, config: PrivacyConfig) -> None:
        super().__init__(config)
        self._noise_multipliers: list[float] = []
        self._sample_rates: list[float] = []
        self._c = math.sqrt(2 * math.log(1.25 / self._config.delta))
        self._q = 0.0  # Sampling rate

    def add_noise_event(self, sigma: float, samples: int) -> None:
        if samples <= 0:
            raise ValueError("Number of samples must be positive")
        if sigma <= 0:
            raise ValueError("Noise multiplier must be positive")

        self._q = min(
            float(samples) / float(self._config.max_gradient_norm), 1.0
        )

        self._noise_multipliers.append(sigma)
        self._sample_rates.append(self._q)
        self._event_count += 1

        self._compute_privacy_spent()

    def _compute_privacy_spent(self) -> PrivacySpent:
        if not self._noise_multipliers:
            self._privacy_spent = PrivacySpent(0.0, 0.0)
            return self._privacy_spent

        epsilons = [
            self._c * q / sigma
            for sigma, q in zip(self._noise_multipliers, self._sample_rates)
        ]
        total_epsilon = sum(epsilons)

        self._privacy_spent = PrivacySpent(
            epsilon_spent=total_epsilon, delta_spent=self._config.delta
        )

        return self._privacy_spent
