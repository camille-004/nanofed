import math

import pytest

from nanofed.privacy.accountant import GaussianAccountant
from nanofed.privacy.config import PrivacyConfig


@pytest.fixture
def default_config():
    """Default privacy configuration."""
    return PrivacyConfig(
        epsilon=2.5, delta=1e-5, noise_multiplier=8.0, max_gradient_norm=1000.0
    )


class TestPrivacyParameterScaling:
    """Test suite for privacy parameter scaling properties."""

    def test_noise_to_epsilon_scaling(self, default_config):
        """Test inverse relationship between noise and epsilon."""
        samples = 4
        multipliers = [4.0, 8.0, 16.0]
        epsilons = []

        for noise in multipliers:
            accountant = GaussianAccountant(default_config)
            accountant.add_noise_event(sigma=noise, samples=samples)
            epsilons.append(accountant.get_privacy_spent().epsilon_spent)

        # Check inverse relationship: doubling noise should halve epsilon
        for i in range(len(multipliers) - 1):
            ratio = epsilons[i] / epsilons[i + 1]
            assert math.isclose(ratio, 2.0, rel_tol=1e-9)

    def test_sample_to_epsilon_scaling(self, default_config):
        """Test linear relationship between sample size and epsilon."""
        noise = default_config.noise_multiplier
        samples = [4, 8, 16]
        epsilons = []

        for n in samples:
            accountant = GaussianAccountant(default_config)
            accountant.add_noise_event(sigma=noise, samples=n)
            epsilons.append(accountant.get_privacy_spent().epsilon_spent)

        # Check linear relationship: doubling samples should double epsilon
        for i in range(len(samples) - 1):
            ratio = epsilons[i + 1] / epsilons[i]
            assert math.isclose(ratio, 2.0, rel_tol=1e-9)


class TestCompositionTheorems:
    """Test suite for privacy composition properties."""

    def test_basic_composition(self, default_config):
        """Test basic composition of privacy losses."""
        accountant = GaussianAccountant(default_config)

        accountant.add_noise_event(
            sigma=default_config.noise_multiplier, samples=4
        )
        single_eps = accountant.get_privacy_spent().epsilon_spent

        accountant = GaussianAccountant(default_config)
        num_events = 5
        for _ in range(num_events):
            accountant.add_noise_event(
                sigma=default_config.noise_multiplier, samples=4
            )

        multi_eps = accountant.get_privacy_spent().epsilon_spent
        assert math.isclose(multi_eps, num_events * single_eps, rel_tol=1e-9)

    def test_heterogeneous_composition(self, default_config):
        """Test composition with different noise levels."""
        accountant = GaussianAccountant(default_config)
        expected_eps = 0
        c = math.sqrt(2 * math.log(1.25 / default_config.delta))

        # Add events with different noise levels
        events = [(8.0, 4), (16.0, 4), (32.0, 4)]  # (sigma, samples)
        for sigma, samples in events:
            accountant.add_noise_event(sigma=sigma, samples=samples)
            q = samples / default_config.max_gradient_norm
            expected_eps += c * q / sigma

        actual_eps = accountant.get_privacy_spent().epsilon_spent
        assert math.isclose(actual_eps, expected_eps, rel_tol=1e-9)


class TestPrivacyAmplification:
    """Test suite for privacy amplification properties."""

    def test_sampling_amplification(self, default_config):
        """Test privacy amplification by sampling."""
        noise = default_config.noise_multiplier
        epsilons = []

        for samples in [2, 4, 8, 16]:
            accountant = GaussianAccountant(default_config)
            accountant.add_noise_event(sigma=noise, samples=samples)
            epsilons.append(accountant.get_privacy_spent().epsilon_spent)

        # Privacy loss should scale linearly with sampling rate
        for i in range(len(epsilons) - 1):
            ratio = epsilons[i + 1] / epsilons[i]
            assert math.isclose(ratio, 2.0, rel_tol=1e-9)


class TestBoundaryConditions:
    """Test suite for boundary conditions."""

    def test_minimal_sampling(self, default_config):
        """Test behavior with minimal sampling."""
        accountant = GaussianAccountant(default_config)
        accountant.add_noise_event(
            sigma=default_config.noise_multiplier, samples=1
        )
        eps = accountant.get_privacy_spent().epsilon_spent
        assert eps > 0

        c = math.sqrt(2 * math.log(1.25 / default_config.delta))
        expected_eps = c / (
            default_config.max_gradient_norm * default_config.noise_multiplier
        )
        assert math.isclose(eps, expected_eps, rel_tol=1e-9)

    def test_maximal_sampling(self, default_config):
        """Test behavior with sampling rate capped at 1.0."""
        accountant = GaussianAccountant(default_config)
        samples = int(
            default_config.max_gradient_norm * 2
        )  # Should get clipped to 1.0
        accountant.add_noise_event(
            sigma=default_config.noise_multiplier, samples=samples
        )
        eps = accountant.get_privacy_spent().epsilon_spent

        c = math.sqrt(2 * math.log(1.25 / default_config.delta))
        expected_eps = c / default_config.noise_multiplier
        assert math.isclose(eps, expected_eps, rel_tol=1e-9)


class TestMonotonicityProperties:
    """Test suite for monotonicity properties."""

    def test_privacy_loss_monotonicity(self, default_config):
        """Test that privacy loss is monotonically increasing."""
        accountant = GaussianAccountant(default_config)
        prev_eps = 0.0

        for _ in range(5):
            accountant.add_noise_event(
                sigma=default_config.noise_multiplier, samples=4
            )
            curr_eps = accountant.get_privacy_spent().epsilon_spent
            assert curr_eps > prev_eps
            prev_eps = curr_eps

    def test_noise_monotonicity(self, default_config):
        """Test that more noise means better privacy."""
        samples = 4
        noise_levels = [4.0, 8.0, 16.0]
        epsilons = []

        for noise in noise_levels:
            accountant = GaussianAccountant(default_config)
            accountant.add_noise_event(sigma=noise, samples=samples)
            epsilons.append(accountant.get_privacy_spent().epsilon_spent)

        # More noise should strictly decrease privacy loss
        for i in range(len(epsilons) - 1):
            assert epsilons[i] > epsilons[i + 1]
