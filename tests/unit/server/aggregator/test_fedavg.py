import pytest
import torch

from nanofed.core.exceptions import AggregationError
from nanofed.core.interfaces import ModelProtocol
from nanofed.server.aggregator.fedavg import FedAvgAggregator


class DummyModel(torch.nn.Module, ModelProtocol):
    def __init__(self):
        super().__init__()
        self.fc = torch.nn.Linear(2, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)


def test_fedavg_aggregate_success():
    model = DummyModel()
    aggregator = FedAvgAggregator()

    updates = [
        {
            "model_state": {
                "fc.weight": [[1.0, 1.0], [1.0, 1.0]],
                "fc.bias": [0.5, 0.5],
            },
            "round_number": 1,
            "metrics": {"loss": 0.1, "accuracy": 0.9},
        },
        {
            "model_state": {
                "fc.weight": [[2.0, 2.0], [2.0, 2.0]],
                "fc.bias": [1.0, 1.0],
            },
            "round_number": 1,
            "metrics": {"loss": 0.2, "accuracy": 0.8},
        },
    ]

    result = aggregator.aggregate(model, updates)

    assert pytest.approx(result.metrics["loss"], 0.001) == 0.15
    assert pytest.approx(result.metrics["accuracy"], 0.001) == 0.85


def test_fedavg_aggregate_validation_error():
    model = DummyModel()
    aggregator = FedAvgAggregator()

    updates = [
        {"model_state": {}, "round_number": 1, "metrics": {}},
        {"model_state": {}, "round_number": 2, "metrics": {}},
    ]

    with pytest.raises(
        AggregationError, match="Updates from different rounds: {1, 2}"
    ):
        aggregator.aggregate(model, updates)
