import pytest

from nanofed.privacy.accountant import RDPAccountant
from nanofed.privacy.config import PrivacyConfig
from nanofed.privacy.exceptions import PrivacyError


@pytest.fixture
def privacy_config():
    return PrivacyConfig(
        epsilon=2.5, delta=1e-5, noise_multiplier=1.1, max_gradient_norm=1.0
    )


class TestRDPAccountant:
    """Test RDP accountant functionality."""

    def test_initialization(self, privacy_config):
        """Test accountant initialization."""
        accountant = RDPAccountant(privacy_config)
        assert len(accountant._orders) > 0
        assert all(alpha > 1.0 for alpha in accountant._orders)

        with pytest.raises(PrivacyError):
            RDPAccountant(privacy_config, orders=[0.5])

    def test_rdp_computation(self, privacy_config):
        """Test RDP value computation."""
        accountant = RDPAccountant(privacy_config)

        # Test single noise event
        sigma = 3.0
        samples = 100
        accountant.add_noise_event(sigma, samples)
        spent = accountant.get_privacy_spent()

        assert spent.epsilon_spent > 0
        assert spent.epsilon_spent < privacy_config.epsilon
        assert spent.delta_spent == privacy_config.delta

    def test_composition(self, privacy_config):
        """Test RDP composition."""
        accountant = RDPAccountant(privacy_config)

        # Add multiple noise events
        events = [
            (1.0, 100),  # (sigma, samples)
            (1.5, 50),
            (2.0, 75),
        ]

        prev_epsilon = 0
        for sigma, samples in events:
            accountant.add_noise_event(sigma, samples)
            spent = accountant.get_privacy_spent()
            # Privacy loss should be monotonically increasing
            assert spent.epsilon_spent > prev_epsilon
            prev_epsilon = spent.epsilon_spent

    def test_sampling_amplification(self, privacy_config):
        """Test privacy amplification by sampling."""
        privacy_config_sampling = PrivacyConfig(
            epsilon=2.5,
            delta=1e-5,
            noise_multiplier=1.1,
            max_gradient_norm=1000.0,
        )
        accountant = RDPAccountant(privacy_config_sampling)

        sigma = 3.0
        samples_large = 1000
        samples_small = 100

        accountant.add_noise_event(sigma, samples_large)
        eps_large = accountant.get_privacy_spent().epsilon_spent

        accountant = RDPAccountant(privacy_config_sampling)

        accountant.add_noise_event(sigma, samples_small)
        eps_small = accountant.get_privacy_spent().epsilon_spent

        # Smaller sampling should give better privacy
        assert (
            eps_small < eps_large
        ), f"Expected eps_small ({eps_small}) < eps_large ({eps_large})"

    def test_monotonicity(self, privacy_config):
        """Test monotonicity properties."""
        accountant = RDPAccountant(privacy_config)
        sigma = 1.0
        samples = 100

        epsilons = []
        for _ in range(5):
            accountant.add_noise_event(sigma, samples)
            spent = accountant.get_privacy_spent()
            epsilons.append(spent.epsilon_spent)

        assert all(x < y for x, y in zip(epsilons[:-1], epsilons[1:]))

    def test_privacy_budget_validation(self, privacy_config):
        """Test privacy budget validation."""
        accountant = RDPAccountant(privacy_config)
        sigma = 0.1
        samples = 1000

        count = 0
        while accountant.validate_budget():
            accountant.add_noise_event(sigma, samples)
            count += 1
            if count > 100:  # Safety limit
                break

        assert not accountant.validate_budget()
        spent = accountant.get_privacy_spent()
        assert spent.epsilon_spent > privacy_config.epsilon

    def test_invalid_inputs(self, privacy_config):
        """Test handling of invalid inputs."""
        accountant = RDPAccountant(privacy_config)

        with pytest.raises(ValueError):
            accountant.add_noise_event(-1.0, 100)  # negative sigma

        with pytest.raises(ValueError):
            accountant.add_noise_event(1.0, -100)

        with pytest.raises(ValueError):
            accountant.add_noise_event(1.0, 0)  # Zero samples

    def test_fixed_order_behavior(self, privacy_config):
        """Test behavior with fixed RDP orders"""
        fixed_orders = [2.0, 4.0, 8.0]
        accountant = RDPAccountant(privacy_config, orders=fixed_orders)

        assert len(accountant._orders) == len(fixed_orders)
        assert all(a == b for a, b in zip(accountant._orders, fixed_orders))

        accountant.add_noise_event(1.0, 100)
        spent = accountant.get_privacy_spent()
        assert spent.epsilon_spent > 0
