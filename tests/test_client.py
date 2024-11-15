from __future__ import annotations

import asyncio
from typing import AsyncGenerator

import numpy as np
import pytest

from fl.core.client import (
    ClientMetrics,
    ClientMode,
    ClientState,
    TrainingConfig,
)
from fl.core.exceptions import ValidationError
from fl.core.protocols import ModelWeights
from tests.helpers.mock_client import MockClient, MockSecurityProtocol


@pytest.fixture(scope="function")
async def test_client() -> AsyncGenerator[MockClient, None]:
    """Fixture for test client instance."""
    client = await MockClient.create(
        client_id="test-client",
        mode=ClientMode.LOCAL,
        security_protocol=MockSecurityProtocol(),
        training_config=TrainingConfig(batch_size=32, local_epochs=1),
    )
    try:
        yield client
    finally:
        await client.cleanup()


@pytest.fixture
async def mock_weights() -> ModelWeights:
    """Fixture for test model weights."""
    return {
        "layer1": np.zeros((10, 10), dtype=np.float32),
        "layer2": np.zeros((10, 1), dtype=np.float32),
    }


@pytest.mark.asyncio
async def test_client_initialization():
    """Test client initialization."""
    client = await MockClient.create("test-client")
    assert client.client_id == "test-client"
    assert client.state == ClientState.INITIALIZED
    assert isinstance(client.metrics, list)
    await client.cleanup()


@pytest.mark.asyncio
async def test_client_local_mode(test_client):
    """Test client in local mode."""
    async for client in test_client:
        assert client._mode == ClientMode.LOCAL
        reg_file = client._secure_storage_path / "registration.json"
        assert reg_file.exists()


@pytest.mark.asyncio
async def test_client_weight_update(test_client, mock_weights):
    """Test updating client weights."""
    weights = await mock_weights
    async for client in test_client:
        await client.update_global_model(weights)
        await asyncio.sleep(0.1)
        assert client.state == ClientState.INITIALIZED
        assert len(client.local_weights) == len(weights)


@pytest.mark.asyncio
async def test_client_training():
    """Test client training process."""
    client = await MockClient.create(
        "test-client", training_config=TrainingConfig(max_rounds=2)
    )

    try:
        data = np.random.randn(100, 10)
        labels = np.random.randint(0, 2, 100)

        updates = []
        async for update in client.train_local_model(data, labels):
            updates.append(update)

        assert len(updates) == 2
        assert all(u.client_id == "test-client" for u in updates)
        assert len(client.metrics) == 2
    finally:
        await client.cleanup()


@pytest.mark.asyncio
async def test_client_error_handling(test_client):
    """Test client error handling."""
    async for client in test_client:
        invalid_weights = {"invalid": "weights"}

        with pytest.raises(ValidationError):
            await client.update_global_model(invalid_weights)

        await asyncio.sleep(0.1)
        assert client.state == ClientState.ERROR


@pytest.mark.asyncio
async def test_client_metrics(test_client):
    """Test client metrics collection."""
    async for client in test_client:
        metrics = ClientMetrics(
            loss=0.5,
            accuracy=0.8,
            training_time=1.0,
            samples_processed=100,
            round_number=0,
        )
        client._metrics_history.append(metrics)

        assert len(client.metrics) == 1
        assert client.metrics[0].loss == 0.5


@pytest.mark.asyncio
async def test_client_state_persistence(test_client, mock_weights, tmp_path):
    """Test client state saving and loading."""
    weights = await mock_weights
    async for client in test_client:
        client.local_weights = weights
        save_path = tmp_path / "client_state.pkl"

        # Save state
        client.save_state(save_path)
        assert save_path.exists()

        # Load state
        loaded_client = MockClient.load_state(save_path, "test-client")
        assert loaded_client.client_id == client.client_id
        assert len(loaded_client.local_weights) == len(client.local_weights)


@pytest.mark.asyncio
async def test_concurrent_operations(test_client):
    """Test concurrent client operations."""
    async for client in test_client:
        data = np.random.randn(100, 10)
        labels = np.random.randint(0, 2, 100)

        weights: ModelWeights = {  # type: ignore
            "layer1": np.zeros((10, 10)),
            "layer2": np.zeros((10, 1)),
        }

        async def training_task():
            async for _ in client.train_local_model(data, labels):
                pass

        async def update_task():
            await client.update_global_model(weights)

        # Run operations concurrently
        await asyncio.gather(training_task(), update_task())
