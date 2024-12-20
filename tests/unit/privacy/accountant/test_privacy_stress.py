import numpy as np

from nanofed.privacy.accountant import GaussianAccountant
from nanofed.privacy.config import PrivacyConfig


def test_long_training_run():
    """Test privacy accounting over long training run."""
    config = PrivacyConfig(
        epsilon=10.0,  # Larger budget for long run
        delta=1e-5,
        noise_multiplier=16.0,
        max_gradient_norm=1000.0,
    )

    accountant = GaussianAccountant(config)

    num_epochs = 100
    batches_per_epoch = 50

    for _ in range(num_epochs):
        for _ in range(batches_per_epoch):
            accountant.add_noise_event(
                sigma=config.noise_multiplier, samples=4
            )
            assert np.isfinite(accountant.get_privacy_spent().epsilon_spent)


def test_rapid_updates():
    """Test rapid sequence of privacy updates."""
    config = PrivacyConfig(
        epsilon=5.0,
        delta=1e-5,
        noise_multiplier=8.0,
        max_gradient_norm=1000.0,
    )

    accountant = GaussianAccountant(config)

    for i in range(1000):
        noise = config.noise_multiplier * (1 + 0.1 * np.sin(i))
        samples = max(1, int(4 + 2 * np.cos(i)))

        accountant.add_noise_event(sigma=noise, samples=samples)
        spent = accountant.get_privacy_spent()
        assert np.isfinite(spent.epsilon_spent)
        assert spent.epsilon_spent >= 0


def test_large_batch_sequence():
    """Test sequence of large batch updates."""
    config = PrivacyConfig(
        epsilon=5.0,
        delta=1e-5,
        noise_multiplier=8.0,
        max_gradient_norm=10000.0,  # Larger to allow bigger batches
    )

    accountant = GaussianAccountant(config)

    batch_sizes = [100, 200, 500, 1000, 2000]

    for batch_size in batch_sizes:
        accountant.add_noise_event(
            sigma=config.noise_multiplier, samples=batch_size
        )
        spent = accountant.get_privacy_spent()
        assert np.isfinite(spent.epsilon_spent)
        assert spent.epsilon_spent > 0


def test_minimal_updates_sequence():
    """Test long sequence of minimal updates."""
    config = PrivacyConfig(
        epsilon=5.0, delta=1e-5, noise_multiplier=8.0, max_gradient_norm=1000.0
    )

    accountant = GaussianAccountant(config)

    for _ in range(1000):
        accountant.add_noise_event(sigma=config.noise_multiplier, samples=1)
        spent = accountant.get_privacy_spent()
        assert np.isfinite(spent.epsilon_spent)
        assert spent.epsilon_spent > 0
