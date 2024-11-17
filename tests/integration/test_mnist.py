import asyncio
from pathlib import Path

import pytest
import torch
from torch.utils.data import DataLoader

from nanofed.communication.http.client import HTTPClient
from nanofed.communication.http.server import HTTPServer
from nanofed.core.types import ModelConfig
from nanofed.data.mnist import load_mnist_data
from nanofed.models.mnist import MNISTModel
from nanofed.orchestration.coordinator import (
    Coordinator,
    CoordinatorConfig,
)
from nanofed.server.aggregator.fedavg import FedAvgAggregator
from nanofed.server.model_manager.manager import ModelManager
from nanofed.trainer.base import TrainingConfig
from nanofed.trainer.torch import TorchTrainer


@pytest.fixture
def mnist_model() -> MNISTModel:
    return MNISTModel()


@pytest.fixture
def mnist_data(tmp_path: Path) -> DataLoader:
    return load_mnist_data(
        tmp_path / "data", batch_size=32, train=True, download=True
    )


@pytest.fixture
def training_config():
    return TrainingConfig(
        epochs=1, batch_size=32, learning_rate=0.01, device="cpu"
    )


@pytest.mark.asyncio
async def test_mnist_e2e(
    tmp_path: Path,
    mnist_model: MNISTModel,
    mnist_data: DataLoader,
    training_config: TrainingConfig,
):
    """Test end-to-end MNIST training."""
    manager = ModelManager(tmp_path, mnist_model)

    # Save initial model state
    initial_config = ModelConfig(
        name="mnist", version="1.0", architecture={"type": "cnn"}
    )
    manager.save_model(initial_config)

    aggregator = FedAvgAggregator()
    server = HTTPServer(
        host="localhost",
        port=8080,
        model_manager=manager,
        max_request_size=100 * 1024 * 1024,
    )

    coordinator_config = CoordinatorConfig(
        num_rounds=1,
        min_clients=2,
        min_completion_rate=1.0,
        round_timeout=300,  # 5 minutes timeout
        checkpoint_dir=tmp_path / "checkpoints",
        metrics_dir=tmp_path / "metrics",
    )

    coordinator = Coordinator(
        mnist_model, aggregator, server, coordinator_config
    )

    trainer = TorchTrainer(training_config)

    # Start server
    await server.start()
    await asyncio.sleep(1)

    try:
        # Start training on coordinator first
        coordinator_task = asyncio.create_task(
            anext(coordinator.start_training())
        )

        # Wait for coordinator to initialize
        await asyncio.sleep(1)

        # Create and train clients
        async with HTTPClient(
            "http://localhost:8080", "client_1", timeout=300
        ) as client1, HTTPClient(
            "http://localhost:8080", "client_2", timeout=300
        ) as client2:
            # Train clients sequentially
            for client_id, client in enumerate([client1, client2], 1):
                # Get model
                model_params, round_num = await client.fetch_global_model()

                # Train locally
                model = MNISTModel()
                model.load_state_dict(model_params)
                model.to(training_config.device)

                optimizer = torch.optim.SGD(
                    model.parameters(), lr=training_config.learning_rate
                )

                metrics = trainer.train_epoch(
                    model, mnist_data, optimizer, round_num
                )

                # Submit update
                success = await client.submit_update(
                    model, {"loss": metrics.loss, "accuracy": metrics.accuracy}
                )
                assert success

            # Wait for coordinator to finish
            round_metrics = await asyncio.wait_for(
                coordinator_task,
                timeout=30.0,  # 30s timeout after clients finish
            )

            # Verify results
            assert round_metrics.round_id == 0
            assert round_metrics.num_clients == 2
            assert "loss" in round_metrics.agg_metrics
            assert "accuracy" in round_metrics.agg_metrics

    finally:
        if "coordinator_task" in locals() and not coordinator_task.done():
            coordinator_task.cancel()
            try:
                await coordinator_task
            except (asyncio.CancelledError, Exception):
                pass
        await server.stop()
