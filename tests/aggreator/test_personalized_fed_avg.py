import numpy as np
import pytest

from fl.aggregator.personalized import PersonalizedFedAvgAggregator
from fl.core.exceptions import AggregationError, ValidationError
from fl.core.protocols import ModelUpdate


@pytest.fixture
def personalized_fed_avg_aggregator() -> PersonalizedFedAvgAggregator:
    return PersonalizedFedAvgAggregator(alpha=0.1)


@pytest.fixture
def sample_updates() -> list[ModelUpdate]:
    update1 = ModelUpdate(
        client_id="client1",
        weights={"w1": np.array([2.0, 4.0]), "w2": np.array([6.0, 8.0])},
        round_metrics={"loss": 0.3, "accuracy": 0.85},
        round_number=1,
    )
    update2 = ModelUpdate(
        client_id="client2",
        weights={"w1": np.array([4.0, 6.0]), "w2": np.array([8.0, 10.0])},
        round_metrics={"loss": 0.2, "accuracy": 0.9},
        round_number=1,
    )
    return [update1, update2]


def test_personalized_fed_avg_aggregate_correctness(
    personalized_fed_avg_aggregator, sample_updates
):
    weights_agg = personalized_fed_avg_aggregator.aggregate(sample_updates)

    avg_w1 = np.array([3.0, 5.0])  # (2+4)/2, (4+6)/2
    avg_w2 = np.array([7.0, 9.0])  # (6+8)/2, (8+10)/2

    expected_w1 = avg_w1 * personalized_fed_avg_aggregator.alpha
    expected_w2 = avg_w2 * personalized_fed_avg_aggregator.alpha

    np.testing.assert_array_almost_equal(weights_agg["w1"], expected_w1)
    np.testing.assert_array_almost_equal(weights_agg["w2"], expected_w2)


def test_personalized_fed_avg_empty_updates(personalized_fed_avg_aggregator):
    """Test that agggregating empty updates raises AggregationError."""
    with pytest.raises(
        AggregationError, match="No model updates provided for aggregation"
    ):
        personalized_fed_avg_aggregator.aggregate([])


def test_personalized_fed_avg_inconsistent_keys(
    personalized_fed_avg_aggregator,
):
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
        personalized_fed_avg_aggregator.aggregate([update1, update2])


def test_personalized_fed_avg_validate_update(
    personalized_fed_avg_aggregator, sample_updates
):
    """Test that validate_update always returns True for valid updates."""
    for update in sample_updates:
        assert personalized_fed_avg_aggregator.validate_update(update) is True


def test_personalized_fed_avg_validate_update_invalid(
    personalized_fed_avg_aggregator,
):
    """Test that validate_update handles invalid updates."""
    update = ModelUpdate(
        client_id="client1",
        weights={"w1": "invalid_weights"},
        round_metrics={"loss": 0.5, "accuracy": 0.8},
        round_number=1,
    )
    with pytest.raises(ValidationError):
        personalized_fed_avg_aggregator.validate_update(update)
