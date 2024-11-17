from pathlib import Path

import pytest
import torch

from nanofed.communication.http.server import HTTPServer
from nanofed.orchestration.coordinator import (
    Coordinator,
    CoordinatorConfig,
)
from nanofed.orchestration.types import RoundStatus
from nanofed.server.aggregator.fedavg import FedAvgAggregator
from nanofed.server.model_manager.manager import ModelManager
from tests.unit.helpers import SimpleModel


@pytest.fixture
def coordinator(tmp_path: Path):
    model = SimpleModel()
    aggregator = FedAvgAggregator()
    manager = ModelManager(tmp_path, model)
    server = HTTPServer("localhost", 8080, manager)

    config = CoordinatorConfig(
        num_rounds=3,
        min_clients=2,
        min_completion_rate=0.8,
        round_timeout=30,
        checkpoint_dir=tmp_path / "checkpoints",
        metrics_dir=tmp_path / "metrics",
    )

    return Coordinator(model, aggregator, server, config)


@pytest.mark.asyncio
async def test_training_round(coordinator: Coordinator):
    fc_weight = torch.randn(2, 10)
    fc_bias = torch.randn(2)

    client_update = {
        "client_id": "test_client",
        "round_number": 0,
        "model_state": {
            "fc.weight": fc_weight.tolist(),
            "fc.bias": fc_bias.tolist(),
        },
        "metrics": {"loss": 0.5, "accuracy": 0.95},
    }

    coordinator._server._updates = {
        "client1": client_update,
        "client2": client_update,
    }

    metrics = await coordinator.train_round()

    assert metrics.round_id == 0
    assert metrics.status == RoundStatus.COMPLETED
    assert metrics.num_clients == 2
    assert "loss" in metrics.agg_metrics
    assert "accuracy" in metrics.agg_metrics

    state_dict = coordinator._model.state_dict()
    assert tuple(state_dict["fc.weight"].shape) == (2, 10)
    assert tuple(state_dict["fc.bias"].shape) == (2,)


@pytest.mark.asyncio
async def test_training_progress(coordinator: Coordinator):
    progress = coordinator.training_progress

    assert progress["current_round"] == 0
    assert progress["total_rounds"] == 3
    assert progress["active_clients"] == 0
    assert progress["status"] == "INITIALIZED"
