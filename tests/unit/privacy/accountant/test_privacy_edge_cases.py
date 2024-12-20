import math

import numpy as np
import pytest

from nanofed.privacy.accountant import GaussianAccountant
from nanofed.privacy.config import PrivacyConfig


@pytest.fixture
def default_config():
    """Default privacy configuration."""
    return PrivacyConfig(
        epsilon=1.0,
        delta=1e-5,
        noise_multiplier=8.0,
        max_gradient_norm=1000.0,
    )


class TestNumericalEdgeCases:
    """Test numerical stability at extreme values."""

    def test_very_small_noise(self, default_config):
        """Test behavior with very small noise values."""
        accountant = GaussianAccountant(default_config)
        small_noise = 1e-5

        accountant.add_noise_event(sigma=small_noise, samples=4)
        spent = accountant.get_privacy_spent()
        assert np.isfinite(spent.epsilon_spent)
        assert spent.epsilon_spent > 0

    def test_very_large_noise(self, default_config):
        """Test behavior with very large noise values."""
        accountant = GaussianAccountant(default_config)
        large_noise = 1e5

        accountant.add_noise_event(sigma=large_noise, samples=4)
        spent = accountant.get_privacy_spent()
        assert np.isfinite(spent.epsilon_spent)
        assert spent.epsilon_spent > 0

    def test_tiny_sample_fraction(self, default_config):
        """Test with extremely small sampling fraction."""
        accountant = GaussianAccountant(default_config)
        samples = 1

        accountant.add_noise_event(
            sigma=default_config.noise_multiplier, samples=samples
        )
        spent = accountant.get_privacy_spent()
        assert np.isfinite(spent.epsilon_spent)
        assert spent.epsilon_spent > 0


class TestSequentialComposition:
    """Test sequential composition scenarios."""

    def test_alternating_noise_levels(self, default_config):
        """Test privacy accounting with alternating noise levels."""
        accountant = GaussianAccountant(default_config)
        base_noise = default_config.noise_multiplier

        noises = [base_noise, base_noise * 2, base_noise, base_noise * 2]
        expected_eps = 0
        c = math.sqrt(2 * math.log(1.25 / default_config.delta))

        for noise in noises:
            accountant.add_noise_event(sigma=noise, samples=4)
            q = 4 / default_config.max_gradient_norm
            expected_eps += c * q / noise

        spent = accountant.get_privacy_spent()
        assert math.isclose(spent.epsilon_spent, expected_eps, rel_tol=1e-9)

    def test_varying_sample_sizes(self, default_config):
        """Test privacy accounting with varying sample sizes."""
        accountant = GaussianAccountant(default_config)
        noise = default_config.noise_multiplier

        samples = [2, 4, 8, 4, 2]
        expected_eps = 0
        c = math.sqrt(2 * math.log(1.25 / default_config.delta))

        for n in samples:
            accountant.add_noise_event(sigma=noise, samples=n)
            q = n / default_config.max_gradient_norm
            expected_eps += c * q / noise

        spent = accountant.get_privacy_spent()
        assert math.isclose(spent.epsilon_spent, expected_eps, rel_tol=1e-9)


class TestBudgetBoundaries:
    """Test privacy budget boundary conditions."""

    def test_exact_budget(self, default_config):
        """Test behavior when hitting exact privacy budget."""
        accountant = GaussianAccountant(default_config)
        noise = default_config.noise_multiplier
        samples = 4

        c = math.sqrt(2 * math.log(1.25 / default_config.delta))
        eps_per_event = (
            c * (samples / default_config.max_gradient_norm) / noise
        )
        events_needed = int(default_config.epsilon / eps_per_event)

        for _ in range(events_needed - 1):
            accountant.add_noise_event(sigma=noise, samples=samples)
            assert accountant.validate_budget()

        # Add final event
        accountant.add_noise_event(sigma=noise, samples=samples)
        spent = accountant.get_privacy_spent()
        assert math.isclose(
            spent.epsilon_spent, default_config.epsilon, rel_tol=1e-2
        )

    def test_budget_overflow(self, default_config):
        """Test graceful handling of budget overflow."""
        accountant = GaussianAccountant(default_config)
        noise = default_config.noise_multiplier

        prev_eps = 0
        while accountant.validate_budget():
            prev_eps = accountant.get_privacy_spent().epsilon_spent
            accountant.add_noise_event(sigma=noise, samples=4)

        final_eps = accountant.get_privacy_spent().epsilon_spent
        assert final_eps > prev_eps
        assert np.isfinite(final_eps)


class TestAdvancedScenarios:
    """Test more complex privacy accounting scenarios."""

    def test_noise_sample_tradeoff(self, default_config):
        """Tets noise vs sample size trade-off."""
        accountant1 = GaussianAccountant(default_config)
        accountant2 = GaussianAccountant(default_config)

        # Same privacy loss with different noise/sample combinations
        accountant1.add_noise_event(sigma=8.0, samples=4)
        accountant2.add_noise_event(sigma=4.0, samples=2)

        eps1 = accountant1.get_privacy_spent().epsilon_spent
        eps2 = accountant2.get_privacy_spent().epsilon_spent
        assert math.isclose(eps1, eps2, rel_tol=1e-9)

    def test_repeated_small_updates(self, default_config):
        """Test many small updates vs fewer large updates."""
        accountant1 = GaussianAccountant(default_config)
        accountant2 = GaussianAccountant(default_config)

        # Many small updates
        for _ in range(10):
            accountant1.add_noise_event(
                sigma=default_config.noise_multiplier, samples=1
            )

        # Fewer large updates
        for _ in range(2):
            accountant2.add_noise_event(
                sigma=default_config.noise_multiplier, samples=5
            )

        eps1 = accountant1.get_privacy_spent().epsilon_spent
        eps2 = accountant2.get_privacy_spent().epsilon_spent
        assert math.isclose(eps1, eps2, rel_tol=1e-9)
