import pytest
import torch
import torch.nn as nn
import torch.optim as optim

from nanofed.models import MNISTModel
from nanofed.orchestration import CoordinatorConfig
from nanofed.privacy.accountant import GaussianAccountant
from nanofed.privacy.config import PrivacyConfig
from nanofed.privacy.noise import GaussianNoiseGenerator
from nanofed.server import FedAvgAggregator
from nanofed.trainer import TrainingConfig
from nanofed.utils import get_current_time


@pytest.fixture
def privacy_config():
    """Default privacy configuration."""
    return PrivacyConfig(
        epsilon=2.5,
        delta=1e-5,
        noise_multiplier=8.0,
        max_gradient_norm=1000.0,
    )


@pytest.fixture
def training_config():
    """Default training configruation."""
    return TrainingConfig(
        epochs=1, batch_size=4, learning_rate=0.1, device="cpu", max_batches=10
    )


@pytest.fixture
def model():
    """Create MNIST model."""
    return MNISTModel()


class TestPrivacyInTraining:
    """Test privacy integration with model training."""

    def test_private_training_loop(
        self, model, privacy_config, training_config
    ):
        """Test privacy accounting during training loop."""
        accountant = GaussianAccountant(privacy_config)
        noise_gen = GaussianNoiseGenerator(seed=42)
        optimizer = optim.SGD(
            model.parameters(), lr=training_config.learning_rate
        )

        # Create dummy data
        batch_size = training_config.batch_size
        input_shape = (batch_size, 1, 28, 28)  # MNIST shape
        dummy_data = torch.randn(*input_shape)
        dummy_labels = torch.randint(0, 10, (batch_size,))

        for _ in range(training_config.max_batches):
            optimizer.zero_grad()

            # Forward pass
            output = model(dummy_data)
            loss = nn.functional.cross_entropy(output, dummy_labels)

            # Backward pass
            loss.backward()

            # Add noise to gradients
            for param in model.parameters():
                if param.grad is not None:
                    noise = noise_gen.generate(
                        param.grad.shape, scale=privacy_config.noise_multiplier
                    )
                    param.grad += noise

            # Record privacy spent
            accountant.add_noise_event(
                sigma=privacy_config.noise_multiplier, samples=batch_size
            )

            optimizer.step()

        # Verify privacy accounting
        spent = accountant.get_privacy_spent()
        assert spent.epsilon_spent > 0
        assert spent.epsilon_spent < privacy_config.epsilon
        assert spent.delta_spent == privacy_config.delta


class TestPrivacyInFederated:
    """Test privacy integration in federated learning."""

    def test_federated_client_update(
        self, model, privacy_config, training_config
    ):
        """Test privacy in federated client update."""
        accountant = GaussianAccountant(privacy_config)
        noise_gen = GaussianNoiseGenerator(seed=42)

        # Create dummy client data
        num_clients = 3
        client_data = []
        for _ in range(num_clients):
            inputs = torch.randn(training_config.batch_size * 5, 1, 28, 28)
            labels = torch.randint(0, 10, (training_config.batch_size * 5,))
            client_data.append((inputs, labels))

        client_updates = []
        for client_id, (inputs, labels) in enumerate(client_data):
            optimizer = optim.SGD(
                model.parameters(), lr=training_config.learning_rate
            )

            for batch_start in range(
                0, len(inputs), training_config.batch_size
            ):
                batch_end = batch_start + training_config.batch_size
                batch_x = inputs[batch_start:batch_end]
                batch_y = labels[batch_start:batch_end]

                optimizer.zero_grad()
                output = model(batch_x)
                loss = nn.functional.cross_entropy(output, batch_y)
                loss.backward()

                # Add noise to gradients
                for param in model.parameters():
                    if param.grad is not None:
                        noise = noise_gen.generate(
                            param.grad.shape,
                            scale=privacy_config.noise_multiplier,
                        )
                        param.grad += noise

                # Record privacy spent
                accountant.add_noise_event(
                    sigma=privacy_config.noise_multiplier, samples=len(batch_x)
                )

                optimizer.step()

            client_updates.append(
                {
                    "client_id": str(client_id),
                    "model_state": model.state_dict(),
                    "privacy_spent": accountant.get_privacy_spent(),
                }
            )

        for update in client_updates:
            spent = update["privacy_spent"]
            assert spent.epsilon_spent > 0
            assert spent.epsilon_spent < privacy_config.epsilon
            assert spent.delta_spent == privacy_config.delta


def test_full_federated_workflow(
    tmp_path, model, privacy_config, training_config
):
    """Test privacy in complete federated workflow."""
    coordinator_config = CoordinatorConfig(
        num_rounds=2,
        min_clients=2,
        min_completion_rate=1.0,
        round_timeout=30,
        base_dir=tmp_path,
    )

    aggregator = FedAvgAggregator()
    noise_gen = GaussianNoiseGenerator(seed=42)
    accountant = GaussianAccountant(privacy_config)

    # Create dummy clients
    num_clients = 2
    clients = []
    for i in range(num_clients):
        client_data = torch.randn(50, 1, 28, 28)
        client_labels = torch.randint(0, 10, (50,))
        clients.append({"id": str(i), "data": (client_data, client_labels)})

    # Simulate FL with privacy
    for round_num in range(coordinator_config.num_rounds):
        client_updates = []

        for client in clients:
            client_model = MNISTModel()
            optimizer = optim.SGD(
                client_model.parameters(), lr=training_config.learning_rate
            )

            data, labels = client["data"]
            for batch_start in range(0, len(data), training_config.batch_size):
                batch_end = batch_start + training_config.batch_size
                batch_x = data[batch_start:batch_end]
                batch_y = labels[batch_start:batch_end]

                optimizer.zero_grad()
                output = model(batch_x)
                loss = nn.functional.cross_entropy(output, batch_y)
                loss.backward()

                # Add noise to gradients
                for param in client_model.parameters():
                    if param.grad is not None:
                        noise = noise_gen.generate(
                            param.grad.shape,
                            scale=privacy_config.noise_multiplier,
                        )
                        param.grad += noise

                accountant.add_noise_event(
                    sigma=privacy_config.noise_multiplier, samples=len(batch_x)
                )

                optimizer.step()

            client_updates.append(
                {
                    "client_id": client["id"],
                    "round_number": round_num,
                    "model_state": client_model.state_dict(),
                    "metrics": {"samples_processed": len(data)},
                    "timestamp": get_current_time().isoformat(),
                }
            )

        aggregator._validate_updates(client_updates)
        model = aggregator.aggregate(model, client_updates).model

    final_privacy = accountant.get_privacy_spent()
    assert final_privacy.epsilon_spent > 0
    assert final_privacy.epsilon_spent < privacy_config.epsilon
    assert final_privacy.delta_spent == privacy_config.delta
