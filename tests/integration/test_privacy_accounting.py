from nanofed.privacy.accountant import GaussianAccountant
from nanofed.privacy.config import PrivacyConfig
from nanofed.privacy.noise import GaussianNoiseGenerator


def test_accountant_with_noise_generator():
    """Test interaction between accountant and noise generator."""
    config = PrivacyConfig(
        epsilon=2.5,
        delta=1e-5,
        noise_multiplier=8.0,
        max_gradient_norm=1000.0,
    )

    accountant = GaussianAccountant(config)
    noise_gen = GaussianNoiseGenerator(seed=42)

    batch_size = 4
    total_batches = 10

    # Simulate privacy-preserving operations
    for _ in range(total_batches):
        shape = (batch_size, 10)
        _ = noise_gen.generate(shape, scale=config.noise_multiplier)

        accountant.add_noise_event(
            sigma=config.noise_multiplier,
            samples=batch_size,
        )

    spent = accountant.get_privacy_spent()
    print(f"Privacy spent: ε={spent.epsilon_spent:.4f}, δ={spent.delta_spent}")
    assert spent.epsilon_spent > 0
    assert spent.epsilon_spent < config.epsilon
    assert spent.delta_spent == config.delta
    assert accountant.validate_budget()
