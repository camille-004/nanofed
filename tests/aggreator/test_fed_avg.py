import numpy as np
import pytest

from fl.aggregator.fed_avg import FedAvgAggregator
from fl.core.exceptions import AggregationError, ValidationError
from fl.core.protocols import ModelUpdate


@pytest.fixture
def fed_avg_aggregator() -> FedAvgAggregator:
    return FedAvgAggregator()


@pytest.fixture
def sample_updates() -> list[ModelUpdate]:
    update1 = ModelUpdate(
        client_id="client1",
        weights={"w1": np.array([1.0, 2.0]), "w2": np.array([3.0, 4.0])},
        round_metrics={"loss": 0.5, "accuracy": 0.8},
        round_number=1,
    )
    update2 = ModelUpdate(
        client_id="client2",
        weights={"w1": np.array([5.0, 6.0]), "w2": np.array([7.0, 8.0])},
        round_metrics={"loss": 0.4, "accuracy": 0.85},
        round_number=1,
    )
    return [update1, update2]


def test_fed_avg_aggregate_correctness(fed_avg_aggregator, sample_updates):
    """Test that FedAvgAggregator correctly averages the weights."""
    weights_agg = fed_avg_aggregator.aggregate(sample_updates)

    expected_w1 = np.array([3.0, 4.0])  # (1+5)/2, (2+6)/2
    expected_w2 = np.array([5.0, 6.0])  # (3+7)/2, (4+8)/2

    np.testing.assert_array_almost_equal(weights_agg["w1"], expected_w1)
    np.testing.assert_array_almost_equal(weights_agg["w2"], expected_w2)


def test_fed_avg_empty_updates(fed_avg_aggregator):
    """Test that agggregating empty updates raises AggregationError."""
    with pytest.raises(
        AggregationError, match="No model updates provided for aggregation"
    ):
        fed_avg_aggregator.aggregate([])


def test_fed_avg_inconsistent_keys(fed_avg_aggregator):
    """Test that aggregating updates with inconsistent keys raises error."""
    update1 = ModelUpdate(
        client_id="client1",
        weights={"w1": np.array([1.0, 2.0])},
        round_metrics={"loss": 0.5, "accuracy": 0.8},
        round_number=1,
    )
    update2 = ModelUpdate(
        client_id="client2",
        weights={"w2": np.array([3.0, 4.0])},
        round_metrics={"loss": 0.4, "accuracy": 0.85},
        round_number=1,
    )

    with pytest.raises(AggregationError):
        fed_avg_aggregator.aggregate([update1, update2])


def test_fed_avg_validate_update(fed_avg_aggregator, sample_updates):
    """Test that validate_update always returns True for valid updates."""
    for update in sample_updates:
        assert fed_avg_aggregator.validate_update(update) is True


def test_fed_avg_validate_update_invalid(fed_avg_aggregator):
    """Test that validate_update handles invalid updates."""
    update = ModelUpdate(
        client_id="client1",
        weights={"w1": "invalid_weights"},
        round_metrics={"loss": 0.5, "accuracy": 0.8},
        round_number=1,
    )
    with pytest.raises(ValidationError):
        fed_avg_aggregator.validate_update(update)
