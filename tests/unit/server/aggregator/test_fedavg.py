from datetime import UTC, datetime

import pytest
import torch

from nanofed.core.exceptions import AggregationError
from nanofed.core.interfaces import ModelProtocol
from nanofed.core.types import ModelUpdate
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
        ModelUpdate(
            client_id="client1",
            round_number=1,
            model_state={
                "fc.weight": [[1.0, 1.0], [1.0, 1.0]],
                "fc.bias": [0.5, 0.5],
            },
            metrics={
                "loss": 0.1,
                "accuracy": 0.9,
                "samples_processed": 1000,  # First client has 1000 samples
            },
            timestamp=datetime.now(UTC),
        ),
        ModelUpdate(
            client_id="client2",
            round_number=1,
            model_state={
                "fc.weight": [[2.0, 2.0], [2.0, 2.0]],
                "fc.bias": [1.0, 1.0],
            },
            metrics={
                "loss": 0.2,
                "accuracy": 0.8,
                "samples_processed": 2000,  # Second client has 2000 samples
            },
            timestamp=datetime.now(UTC),
        ),
    ]

    result = aggregator.aggregate(model, updates)

    assert pytest.approx(result.metrics["loss"], rel=1e-5) == (
        0.1 * 1 / 3 + 0.2 * 2 / 3
    )
    assert pytest.approx(result.metrics["accuracy"], rel=1e-5) == (
        0.9 * 1 / 3 + 0.8 * 2 / 3
    )

    # Check model parameters are weighted correctly
    state_dict = model.state_dict()
    assert torch.allclose(
        state_dict["fc.weight"],
        torch.tensor([[1.667, 1.667], [1.667, 1.667]], dtype=torch.float32),
        rtol=1e-3,
    )
    assert torch.allclose(
        state_dict["fc.bias"],
        torch.tensor([0.833, 0.833], dtype=torch.float32),
        rtol=1e-3,
    )


def test_fedavg_aggregate_missing_samples():
    """Test FedAvg when sample counts are missing."""
    model = DummyModel()
    aggregator = FedAvgAggregator()

    updates = [
        ModelUpdate(
            client_id="client1",
            round_number=1,
            model_state={
                "fc.weight": [[1.0, 1.0], [1.0, 1.0]],
                "fc.bias": [0.5, 0.5],
            },
            metrics={"loss": 0.1, "accuracy": 0.9},  # No samples_processed
            timestamp=datetime.now(UTC),
        ),
        ModelUpdate(
            client_id="client2",
            round_number=1,
            model_state={
                "fc.weight": [[2.0, 2.0], [2.0, 2.0]],
                "fc.bias": [1.0, 1.0],
            },
            metrics={"loss": 0.2, "accuracy": 0.8},  # No samples_processed
            timestamp=datetime.now(UTC),
        ),
    ]

    result = aggregator.aggregate(model, updates)

    # With missing sample counts, should default to equal weights
    assert pytest.approx(result.metrics["loss"], 0.001) == 0.15
    assert pytest.approx(result.metrics["accuracy"], 0.001) == 0.85


def test_fedavg_aggregate_validation_error():
    """Test FedAvg validation for different round numbers."""
    model = DummyModel()
    aggregator = FedAvgAggregator()

    updates = [
        ModelUpdate(
            client_id="client1",
            round_number=1,
            model_state={},
            metrics={},
            timestamp=datetime.now(UTC),
        ),
        ModelUpdate(
            client_id="client2",
            round_number=2,
            model_state={},
            metrics={},
            timestamp=datetime.now(UTC),
        ),
    ]

    with pytest.raises(
        AggregationError, match="Updates from different rounds: {1, 2}"
    ):
        aggregator.aggregate(model, updates)


def test_fedavg_aggregate_different_architectures():
    """Test FedAvg validation for different model architectures."""
    model = DummyModel()
    aggregator = FedAvgAggregator()

    updates = [
        ModelUpdate(
            client_id="client1",
            round_number=1,
            model_state={"layer1": [1.0]},
            metrics={},
            timestamp=datetime.now(UTC),
        ),
        ModelUpdate(
            client_id="client2",
            round_number=1,
            model_state={"layer2": [1.0]},
            metrics={},
            timestamp=datetime.now(UTC),
        ),
    ]

    with pytest.raises(
        AggregationError, match="Inconsistent model architectures in updates"
    ):
        aggregator.aggregate(model, updates)
