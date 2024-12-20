import math
from typing import Sequence

import numpy as np

from ..config import PrivacyConfig
from ..exceptions import PrivacyError
from .base import BasePrivacyAccountant, PrivacySpent


class RDPAccountant(BasePrivacyAccountant):
    """Privacy accountant using Rényi Differential Privacy.

    Implements tighter privacy composition using RDP accounting.
    Based on: "Rényi Differential Privacy" (Mironov, 2017)
    """

    def __init__(
        self, config: PrivacyConfig, orders: Sequence[float] | None = None
    ) -> None:
        """Initialize RDP accountant.

        Parameters
        ----------
        config : PrivacyConfig
            Privacy configuration
        orders : Sequence[float] | None, optional
            RDP orders to track, by default None
        """
        super().__init__(config)
        self._orders = np.array(
            orders or [1.5, 2.0, 2.5, 3.0, 4.0, 8.0, 16.0, 32.0, 64.0]
        )
        if len(self._orders) == 0:
            raise PrivacyError("Must specify at least one RDP order")
        if not np.all(self._orders > 1.0):
            raise PrivacyError("All RDP orders must be > 1.0")

        self._rdp_budget = {alpha: 0.0 for alpha in self._orders}

    def _compute_rdp_gaussian(
        self, sigma: float, sampling_rate: float
    ) -> dict[float, float]:
        """Compute RDP values for Gaussian mechanism.

        Parameters
        ----------
        sigma : float
            Noise scale (standard deviation)
        sampling_rate : float
            Sampling probability

        Returns
        -------
        dict[float, float]
            RDP values for each order
        """
        rdp_values = {}
        for alpha in self._orders:
            rdp = (sampling_rate**2) * alpha / (2 * sigma**2)
            rdp_values[alpha] = rdp
        return rdp_values

    def add_noise_event(self, sigma: float, samples: int) -> None:
        """Record a noise addition event.

        Parameters
        ----------
        sigma : float
            Noise scale (standard deviation)
        samples : int
            Number of samples in the batch
        """
        if samples <= 0:
            raise ValueError("Number of samples must be positive")
        if sigma <= 0:
            raise ValueError("Noise multiplier must be positive")

        sampling_rate = min(
            float(samples) / float(self._config.max_gradient_norm), 1.0
        )

        rdp_values = self._compute_rdp_gaussian(sigma, sampling_rate)
        for alpha in self._orders:
            self._rdp_budget[alpha] += rdp_values[alpha]

        self._event_count += 1
        self._compute_privacy_spent()

    def _compute_privacy_spent(self) -> PrivacySpent:
        """Convert RDP to (ε,δ)-DP guarantee.

        Uses the optimal conversion from RDP to approximate DP.

        Returns
        -------
        PrivacySpent
            Current privacy budget consumption
        """
        if not self._rdp_budget:
            self._privacy_spent = PrivacySpent(0.0, 0.0)
            return self._privacy_spent

        epsilon = float("inf")
        delta = self._config.delta

        for alpha in self._orders:
            rdp = self._rdp_budget[alpha]
            cur_eps = rdp + (math.log(1 / delta) / (alpha - 1))
            epsilon = min(epsilon, cur_eps)

        self._privacy_spent = PrivacySpent(
            epsilon_spent=epsilon, delta_spent=delta
        )
        return self._privacy_spent
