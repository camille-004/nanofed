from datetime import datetime

import pytest
import torch

from nanofed.core.exceptions import AggregationError
from nanofed.core.types import ModelUpdate
from nanofed.server.aggregator.fedavg import FedAvgAggregator
from tests.unit.helpers import SimpleModel


@pytest.fixture
def model() -> SimpleModel:
    return SimpleModel()


@pytest.fixture
def updates() -> list[ModelUpdate]:
    def create_update(client_id: str, round_num: int) -> ModelUpdate:
        return ModelUpdate(
            model_state={
                "fc.weight": torch.randn(2, 10),
                "fc.bias": torch.randn(2),
            },
            client_id=client_id,
            round_number=round_num,
            metrics={"loss": 0.5, "accuracy": 0.95},
            timestamp=datetime.now(),
        )

    return [
        create_update("client1", 1),
        create_update("client2", 1),
        create_update("client3", 1),
    ]


def test_fedavg_aggregation(model, updates):
    aggregator = FedAvgAggregator()
    result = aggregator.aggregate(model, updates)

    assert result.round_number == 1
    assert result.num_clients == 3
    assert "loss" in result.metrics
    assert "accuracy" in result.metrics


def test_fedavg_validation(model):
    aggregator = FedAvgAggregator()

    with pytest.raises(AggregationError):
        aggregator.aggregate(model, [])

    invalid_updates = [
        ModelUpdate(
            model_state={"fc.weight": torch.randn(2, 10)},
            client_id="client1",
            round_number=1,
            metrics={},
            timestamp=datetime.now(),
        ),
        ModelUpdate(
            model_state={"fc.weight": torch.randn(2, 10)},
            client_id="client2",
            round_number=2,
            metrics={},
            timestamp=datetime.now(),
        ),
    ]

    with pytest.raises(AggregationError):
        aggregator.aggregate(model, invalid_updates)
