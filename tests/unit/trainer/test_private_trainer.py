import pytest
import torch
import torch.nn as nn
import torch.optim as optim

from nanofed.privacy.config import PrivacyConfig
from nanofed.trainer import PrivateTrainer, TrainingConfig


@pytest.fixture
def dummy_model():
    """Create a simple model for testing."""
    model = nn.Sequential(nn.Linear(10, 5), nn.ReLU(), nn.Linear(5, 2))
    return model


@pytest.fixture
def training_config():
    """Default training configuration."""
    return TrainingConfig(
        epochs=1, batch_size=4, learning_rate=0.1, device="cpu", max_batches=10
    )


@pytest.fixture
def privacy_config():
    """Default privacy configuration."""
    return PrivacyConfig(
        epsilon=2.5, delta=1e-5, noise_multiplier=8.0, max_gradient_norm=100.0
    )


class TestPrivateTrainer:
    def test_initialization(self, training_config, privacy_config):
        trainer = PrivateTrainer(training_config, privacy_config)
        assert trainer._accountant is not None
        assert trainer._noise_gen is not None

    def test_gradient_privacy(
        self, training_config, privacy_config, dummy_model
    ):
        """Test that gradients are properly privatized."""
        trainer = PrivateTrainer(training_config, privacy_config)
        optimizer = optim.SGD(
            dummy_model.parameters(), lr=training_config.learning_rate
        )

        # Create dummy batch
        inputs = torch.randn(4, 10)
        targets = torch.randint(0, 2, (4,))

        # Get initial gradients
        optimizer.zero_grad()
        outputs = dummy_model(inputs)
        loss = nn.functional.cross_entropy(outputs, targets)
        loss.backward()

        # Store original gradients
        original_grads = {
            name: param.grad.clone()
            for name, param in dummy_model.named_parameters()
            if param.grad is not None
        }

        trainer._apply_privacy_to_gradients(dummy_model, batch_size=4)

        # Check gradients were modified
        for name, param in dummy_model.named_parameters():
            if param.grad is not None:
                assert not torch.allclose(param.grad, original_grads[name])

    def test_privacy_budget_tracking(
        self, training_config, privacy_config, dummy_model
    ):
        """Test privacy budget is tracked correctly."""
        trainer = PrivateTrainer(training_config, privacy_config)
        optimizer = optim.SGD(
            dummy_model.parameters(), lr=training_config.learning_rate
        )

        for _ in range(5):
            inputs = torch.randn(4, 10)
            targets = torch.randint(0, 2, (4,))
            _ = trainer.train_batch(dummy_model, (inputs, targets), optimizer)

            spent = trainer.get_privacy_spent()
            assert spent.epsilon_spent > 0
            assert spent.epsilon_spent < privacy_config.epsilon
            assert spent.delta_spent == privacy_config.delta

    def test_gradient_clipping(
        self, training_config, privacy_config, dummy_model
    ):
        """Test gradient clipping enforces norm bound."""
        trainer = PrivateTrainer(training_config, privacy_config)
        optimizer = optim.SGD(
            dummy_model.parameters(), lr=training_config.learning_rate
        )

        inputs = torch.randn(4, 10) * 100
        targets = torch.randint(0, 2, (4,))

        # Forward and backward pass
        optimizer.zero_grad()
        outputs = dummy_model(inputs)
        loss = nn.functional.cross_entropy(outputs, targets)
        loss.backward()

        original_total_norm = torch.norm(
            torch.stack(
                [
                    torch.norm(p.grad.detach(), 2)
                    for p in dummy_model.parameters()
                ]
            ),
            2,
        )

        # Apply clipping
        trainer._clip_gradients(dummy_model)

        total_norm_clipped = torch.norm(
            torch.stack(
                [
                    torch.norm(p.grad.detach(), 2)
                    for p in dummy_model.parameters()
                ]
            ),
            2,
        )

        print(f"Original norm: {original_total_norm:.4f}")
        print(f"Clipped norm: {total_norm_clipped:.4f}")
        print(f"Max allowed norm: {privacy_config.max_gradient_norm:.4f}")

        assert total_norm_clipped <= privacy_config.max_gradient_norm

    def test_batch_training(
        self, training_config, privacy_config, dummy_model
    ):
        """Test complete batch training process."""
        trainer = PrivateTrainer(training_config, privacy_config)
        optimizer = optim.SGD(
            dummy_model.parameters(), lr=training_config.learning_rate
        )

        inputs = torch.randn(4, 10)
        targets = torch.randint(0, 2, (4,))
        metrics = trainer.train_batch(
            dummy_model, (inputs, targets), optimizer
        )

        assert isinstance(metrics.loss, float)
        assert isinstance(metrics.accuracy, float)
        assert metrics.samples_processed == 4

        spent = trainer.get_privacy_spent()
        assert spent.epsilon_spent > 0
        assert spent.epsilon_spent < privacy_config.epsilon
        assert spent.delta_spent == privacy_config.delta

    def test_privacy_budget_validation(
        self, training_config, privacy_config, dummy_model
    ):
        """Test privacy budget validation."""
        tight_config = PrivacyConfig(
            epsilon=0.1,  # Very small budget
            delta=1e-5,
            noise_multiplier=1.0,  # Small noise (will exhaust budget quickly)
            max_gradient_norm=1.0,
        )
        trainer = PrivateTrainer(training_config, tight_config)
        optimizer = optim.SGD(
            dummy_model.parameters(), lr=training_config.learning_rate
        )

        batch_count = 0
        max_batches = 100

        while trainer.validate_privacy_budget() and batch_count < max_batches:
            inputs = torch.randn(4, 10)
            targets = torch.randint(0, 2, (4,))
            trainer.train_batch(dummy_model, (inputs, targets), optimizer)
            batch_count += 1

        assert batch_count < max_batches  # Should exhaust budget
        assert (
            not trainer.validate_privacy_budget()
        )  # Budget should be exceeded
        spent = trainer.get_privacy_spent()
        assert spent.epsilon_spent > tight_config.epsilon
