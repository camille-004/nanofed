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


class TestGaussianAccount:
    def test_initial_state(self, default_config):
        """Test initial accountant state."""
        accountant = GaussianAccountant(default_config)
        spent = accountant.get_privacy_spent()
        assert spent.epsilon_spent == 0.0
        assert spent.delta_spent == 0.0
        assert accountant.validate_budget()

    def test_single_noise_event(self, default_config):
        """Test privacy accounting for single noise event."""
        accountant = GaussianAccountant(default_config)

        accountant.add_noise_event(
            sigma=float(default_config.noise_multiplier), samples=1
        )

        spent = accountant.get_privacy_spent()
        print(
            f"Privacy spent: ε={spent.epsilon_spent:.4f}, δ={spent.delta_spent}"  # noqa
        )
        assert spent.epsilon_spent > 0
        assert spent.epsilon_spent < default_config.epsilon
        assert spent.delta_spent == default_config.delta
        assert accountant.validate_budget()

    def test_multiple_noise_events(self, default_config):
        """Test privacy accounting for multiple noise events."""
        accountant = GaussianAccountant(default_config)

        for _ in range(5):
            accountant.add_noise_event(
                sigma=float(default_config.noise_multiplier), samples=1
            )

        spent = accountant.get_privacy_spent()
        prev_epsilon = spent.epsilon_spent

        accountant.add_noise_event(
            sigma=float(default_config.noise_multiplier) * 0.25, samples=2
        )

        new_spent = accountant.get_privacy_spent()
        print(
            f"Multiple events - final privacy: ε={new_spent.epsilon_spent:.4f}, δ={new_spent.delta_spent}"  # noqa
        )
        assert new_spent.epsilon_spent > prev_epsilon

    def test_invalid_inputs(self, default_config):
        """Test handling of invalid inputs."""
        accountant = GaussianAccountant(default_config)

        with pytest.raises(
            ValueError, match="Number of samples must be positive"
        ):
            accountant.add_noise_event(sigma=1.0, samples=0)

        with pytest.raises(
            ValueError, match="Noise multiplier must be positive"
        ):
            accountant.add_noise_event(sigma=-1.0, samples=100)

    def test_privacy_exhaustion(self, default_config):
        """Test detection of privacy budget exhaustion."""
        tight_config = PrivacyConfig(
            epsilon=0.1,
            delta=1e-6,
            noise_multiplier=2.0,
            max_gradient_norm=100.0,
        )

        accountant = GaussianAccountant(tight_config)
        event_count = 0
        max_events = 100

        while accountant.validate_budget() and event_count < max_events:
            accountant.add_noise_event(
                sigma=float(tight_config.noise_multiplier), samples=4
            )
            event_count += 1

        spent = accountant.get_privacy_spent()
        assert spent.epsilon_spent > tight_config.epsilon
        assert not accountant.validate_budget()
