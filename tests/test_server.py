import asyncio
import time
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import pytest_asyncio

from fl.config.settings import get_settings
from fl.core.exceptions import AggregationError, ValidationError
from fl.core.protocols import ModelUpdate, ModelWeights
from fl.core.server import Server, ServerState, TrainingRound

settings = get_settings()


class MockAggregator:
    """Mock aggregator for testing purposes."""

    def aggregate(self, updates: list[ModelUpdate]) -> ModelWeights:
        """Aggregate updates by averaging weights."""
        if not updates:
            raise AggregationError("No updates to aggregate")
        aggregated_weights: ModelWeights = {}
        for key in updates[0].weights.keys():
            arrays = [update.weights[key] for update in updates]
            aggregated_weights[key] = np.mean(arrays, axis=0)
        return aggregated_weights

    def validate_update(self, update: ModelUpdate) -> bool:
        """Always return True for validation in testing."""
        return True


@pytest_asyncio.fixture
async def server():
    """Fixture to initialize the server with a MockAggregator."""
    with patch("fl.aggregator.get_aggregator", return_value=MockAggregator()):
        server = Server(aggregator_type="Fed_Avg")
        await server.initialize()
        try:
            yield server
        finally:
            await server.cleanup()


@pytest.fixture
def model_update():
    """Fixture to create a sample model update."""
    weights = {"layer1": np.ones((10, 10)), "layer2": np.ones((10, 1))}
    return ModelUpdate(
        client_id="test-client",
        weights=weights,
        round_metrics={"loss": 0.1, "accuracy": 0.9},
        round_number=0,
    )


@pytest.mark.asyncio
async def test_server_initialization():
    """Test server initialization."""
    with patch("fl.aggregator.get_aggregator", return_value=MockAggregator()):
        server = Server(aggregator_type="Fed_Avg")
        await server.initialize()
        assert server._state == ServerState.READY
        assert server._clients == {}
        await server.cleanup()


@pytest.mark.asyncio
async def test_register_client(server):
    """Test registering a new client."""
    response = await server.register_client(
        client_id="client1",
        signature="signature1",
        capabilities={"version": "1.0"},
    )
    assert response["status"] == "success"
    assert response["client_id"] == "client1"
    assert "server_time" in response
    assert server._clients["client1"].client_id == "client1"
    assert server._clients["client1"].is_active is True


@pytest.mark.asyncio
async def test_register_existing_client(server):
    """Test registering an existing client (should update capabilities)."""
    await server.register_client(
        client_id="client1",
        signature="signature1",
        capabilities={"version": "1.0"},
    )
    response = await server.register_client(
        client_id="client1",
        signature="signature1",
        capabilities={"version": "1.1"},
    )
    assert response["status"] == "success"
    assert server._clients["client1"].capabilities["version"] == "1.1"


@pytest.mark.asyncio
async def test_start_training_with_insufficient_clients(server):
    """Test starting training without enough clients."""
    # No clients registered yet
    with pytest.raises(ValidationError):
        await server.start_training()


@pytest.mark.asyncio
async def test_start_training(server):
    """Test starting the training process."""
    for i in range(server._min_clients):
        await server.register_client(
            client_id=f"client{i}",
            signature=f"signature{i}",
            capabilities={"version": "1.0"},
        )
    await server.start_training()
    assert server._state == ServerState.TRAINING
    server._training_task.cancel()
    try:
        await server._training_task
    except asyncio.CancelledError:
        pass


@pytest.mark.asyncio
async def test_submit_update(server, model_update):
    """Test submitting a model update."""
    await server.register_client(
        client_id="test-client",
        signature="signature",
        capabilities={"version": "1.0"},
    )
    server._current_round = TrainingRound(round_id=0, start_time=time.time())
    response = await server.submit_update(
        client_id="test-client", model_update=model_update
    )
    assert response["status"] == "success"
    assert len(server._current_round.updates) == 1


@pytest.mark.asyncio
async def test_submit_update_invalid_client(server, model_update):
    """Test submitting an update from an unregistered client."""
    # No client registered
    with pytest.raises(ValidationError):
        await server.submit_update(
            client_id="invalid-client", model_update=model_update
        )


@pytest.mark.asyncio
async def test_submit_update_wrong_round(server, model_update):
    """Test submitting an update for the wrong round."""
    # Register client
    await server.register_client(
        client_id="test-client",
        signature="signature",
        capabilities={"version": "1.0"},
    )
    # Start training round with round_id=1
    server._current_round = TrainingRound(round_id=1, start_time=time.time())
    # Update has round_number=0, but current round is 1
    with pytest.raises(ValidationError):
        await server.submit_update(
            client_id="test-client", model_update=model_update
        )


@pytest.mark.asyncio
async def test_aggregate_round(server):
    """Test the aggregation of updates in a training round."""
    server._current_round = TrainingRound(round_id=0, start_time=time.time())
    for i in range(server._min_clients):
        update = ModelUpdate(
            client_id=f"client{i}",
            weights={"layer1": np.ones((10, 10)) * i},
            round_metrics={"loss": 0.1 * i, "accuracy": 0.9 - 0.1 * i},
            round_number=0,
        )
        server._current_round.updates.append(update)
    await server._aggregate_round()
    assert server._global_weights is not None
    assert server._current_round.completed is True
    # Verify aggregated weights
    expected_layer1 = np.mean(
        [np.ones((10, 10)) * i for i in range(server._min_clients)], axis=0
    )
    np.testing.assert_array_almost_equal(
        server._global_weights["layer1"], expected_layer1
    )


@pytest.mark.asyncio
async def test_get_status(server):
    """Test retrieving the server status."""
    status = server.get_status()
    assert status["status"] == "ready"
    assert status["active_clients"] == 0
    assert status["current_round"] is None


@pytest.mark.asyncio
async def test_cleanup_inactive_clients(server):
    """Test cleaning up inactive clients."""
    await server.register_client(
        client_id="client1",
        signature="signature1",
        capabilities={"version": "1.0"},
    )
    # Simulate client inactivity by setting last_seen to an old timestamp
    server._clients["client1"].last_seen -= settings.timeout + 1
    server._remove_inactive_clients()
    assert server._clients["client1"].is_active is False
    assert "client1" not in server._active_clients


@pytest.mark.asyncio
async def test_server_error_handling(server):
    """Test error handling during aggregation."""
    original_aggregate = server._aggregator.aggregate
    server._aggregator.aggregate = MagicMock(
        side_effect=Exception("Aggregation failed")
    )
    server._current_round = TrainingRound(round_id=0, start_time=time.time())
    for i in range(server._min_clients):
        update = ModelUpdate(
            client_id=f"client{i}",
            weights={"layer1": np.ones((10, 10)) * i},
            round_metrics={"loss": 0.1 * i, "accuracy": 0.9 - 0.1 * i},
            round_number=0,
        )
        server._current_round.updates.append(update)
    with pytest.raises(AggregationError):
        await server._aggregate_round()
    # Restore original aggregate method
    server._aggregator.aggregate = original_aggregate
