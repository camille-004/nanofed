import pytest
import torch
import torch.nn as nn

from nanofed.privacy.mechanisms import PrivacyType
from nanofed.server.aggregator import (
    PrivacyAwareAggregationConfig,
    PrivacyAwareAggregator,
    SecureAggregationType,
    ThresholdSecureAggregation,
)


class SimpleModel(nn.Module):
    """Simple model for testing."""

    def __init__(self) -> None:
        super().__init__()
        self.fc = nn.Linear(10, 5)


@pytest.fixture
def model():
    return SimpleModel()


@pytest.fixture
def aggregation_config():
    return PrivacyAwareAggregationConfig(
        epsilon=2.5,
        delta=1e-5,
        noise_multiplier=1.1,
        max_gradient_norm=1.0,
        privacy_type=PrivacyType.CENTRAL,
        min_clients=2,
        clip_norm=1.0,
        secure_aggregation=SecureAggregationType.NONE,
    )


@pytest.fixture
def client_updates():
    """Generate mock client updates."""
    updates = []
    for i in range(3):
        state = {
            "fc.weight": torch.randn(5, 10),
            "fc.bias": torch.randn(5),
        }
        updates.append(
            {
                "client_id": f"client_{i}",
                "round_number": 0,
                "model_state": state,
                "metrics": {
                    "samples_processed": 100 * (i + 1),
                    "accuracy": 0.9,
                },
                "privacy_spent": {"epsilon": 0.1 * (i + 1), "delta": 1e-5},
            }
        )
    return updates


class TestPrivacyAwareAggregator:
    """Test privacy-aware aggregation functionality."""

    def test_weight_computation(self, aggregation_config, client_updates):
        """Test weight computation logic."""
        aggregator = PrivacyAwareAggregator(aggregation_config)
        weights = aggregator._compute_weights(client_updates)

        assert len(weights) == len(client_updates)
        assert abs(sum(weights) - 1.0) < 1e-6  # Weights sum to 1
        assert all(w > 0 for w in weights)  # All weights positive

        samples = [u["metrics"]["samples_processed"] for u in client_updates]
        total_samples = sum(samples)
        expected_weights = [s / total_samples for s in samples]

        assert all(
            abs(w1 - w2) < 1e-6 for w1, w2 in zip(weights, expected_weights)
        )

    def test_central_privacy(self, model, aggregation_config, client_updates):
        """Test central privacy mechanism."""
        aggregator = PrivacyAwareAggregator(aggregation_config)
        result = aggregator.aggregate(model, client_updates)

        # Verify result structure
        assert result.model == model
        assert result.num_clients == len(client_updates)
        assert isinstance(result.metrics, dict)

        # Check weights computation
        weights = aggregator._compute_weights(client_updates)
        assert len(weights) == len(client_updates)
        assert abs(sum(weights) - 1.0) < 1e-6

    def test_local_privacy(self, model, aggregation_config, client_updates):
        """Test local privacy mechanism."""
        config = PrivacyAwareAggregationConfig(
            epsilon=1.0,
            delta=1e-5,
            noise_multiplier=1.1,
            max_gradient_norm=1.0,
            privacy_type=PrivacyType.LOCAL,
            min_clients=2,
        )
        aggregator = PrivacyAwareAggregator(config)

        # Check privacy-adjusted weights
        weights = aggregator._compute_weights(client_updates)
        epsilons = [u["privacy_spent"]["epsilon"] for u in client_updates]

        # Higher epsilon should correspond to higher weight
        for i in range(len(weights) - 1):
            if epsilons[i] < epsilons[i + 1]:
                assert weights[i] < weights[i + 1]

        result = aggregator.aggregate(model, client_updates)
        assert result.model == model
        assert result.num_clients == len(client_updates)

    def test_missing_sample_counts(self, aggregation_config):
        """Test handling of missing sample counts."""
        updates = [
            {
                "client_id": "client_1",
                "metrics": {},
                "model_state": {"param": torch.ones(5)},
            },
            {
                "client_id": "client_2",
                "metrics": {"samples_processed": 100},
                "model_state": {"param": torch.ones(5)},
            },
        ]

        aggregator = PrivacyAwareAggregator(aggregation_config)
        weights = aggregator._compute_weights(updates)

        assert len(weights) == 2
        assert abs(sum(weights) - 1.0) < 1e-6

    def test_secure_aggregation(
        self, model, aggregation_config, client_updates
    ):
        """Test secure aggregation protocol."""
        config = PrivacyAwareAggregationConfig(
            epsilon=2.5,
            delta=1e-5,
            noise_multiplier=1.1,
            max_gradient_norm=1.0,
            privacy_type=PrivacyType.CENTRAL,
            secure_aggregation=SecureAggregationType.THRESHOLD,
            min_clients=2,
        )
        aggregator = PrivacyAwareAggregator(config)
        result = aggregator.aggregate(model, client_updates)

        assert result.model == model
        assert result.num_clients == len(client_updates)

    def test_min_clients_validation(self, model, client_updates):
        """Test minimum clients requirement."""
        config = PrivacyAwareAggregationConfig(
            epsilon=1.0,
            delta=1e-5,
            noise_multiplier=1.1,
            max_gradient_norm=1.0,
            privacy_type=PrivacyType.CENTRAL,
            min_clients=5,
            clip_norm=1.0,
            secure_aggregation=SecureAggregationType.NONE,
        )
        aggregator = PrivacyAwareAggregator(config)

        test_updates = client_updates[:2]

        with pytest.raises(ValueError, match="Not enough clients"):
            aggregator.aggregate(model, test_updates)

    def test_privacy_budget_validation(self, model, client_updates):
        """Test privacy budget validation for local DP."""
        config = PrivacyAwareAggregationConfig(
            privacy_type=PrivacyType.LOCAL, min_clients=2
        )
        aggregator = PrivacyAwareAggregator(config)

        # Remove privacy budget from one update
        invalid_updates = client_updates.copy()
        del invalid_updates[0]["privacy_spent"]

        with pytest.raises(ValueError, match="Missing privacy budget"):
            aggregator.aggregate(model, invalid_updates)

    def test_validation_errors(
        self,
        aggregation_config,
        client_updates,
    ):
        """Test validation error cases."""
        aggregator = PrivacyAwareAggregator(aggregation_config)

        invalid_updates = client_updates.copy()
        invalid_updates[0]["model_state"]["extra_param"] = torch.ones(5)
        with pytest.raises(
            ValueError, match="Inconsistent model architectures"
        ):
            aggregator._validate_updates(invalid_updates)

        invalid_updates = client_updates.copy()
        invalid_updates[0]["round_number"] = 1
        with pytest.raises(ValueError, match="Updates from different rounds"):
            aggregator._validate_updates(invalid_updates)

    def test_metrics_aggregation(self, aggregation_config, client_updates):
        """Test metrics aggregation."""
        aggregator = PrivacyAwareAggregator(aggregation_config)
        metrics = aggregator._aggregate_metrics(client_updates)

        assert "samples_processed" in metrics
        assert "accuracy" in metrics
        assert "privacy_epsilon" in metrics
        assert "privacy_delta" in metrics

        # Check weighted averaging
        total_samples = sum(
            u["metrics"]["samples_processed"] for u in client_updates
        )
        expected_accuracy = (
            sum(
                u["metrics"]["accuracy"] * u["metrics"]["samples_processed"]
                for u in client_updates
            )
            / total_samples
        )

        assert abs(metrics["accuracy"] - expected_accuracy) < 1e-6


class TestThresholdSecureAggregation:
    """Test threshold-based secure aggregation."""

    def test_share_aggregation(self):
        """Test aggregation of shares."""
        protocol = ThresholdSecureAggregation(min_clients=2)

        # Create test shares
        shares = [
            {"param": torch.ones(5, 3)},
            {"param": torch.ones(5, 3) * 2},
            {"param": torch.ones(5, 3) * 3},
        ]

        result = protocol.aggregate_shares(shares)
        expected = torch.ones(5, 3) * 6

        assert torch.allclose(result["param"], expected)

    def test_share_verification(self):
        """Test verification of shares."""
        protocol = ThresholdSecureAggregation(min_clients=2)

        valid_shares = [
            {"param": torch.randn(5, 3)},
            {"param": torch.randn(5, 3)},
        ]
        assert protocol.verify_shares(valid_shares)

        invalid_shares = [
            {"param": torch.randn(5, 3)},
            {"param": torch.randn(5, 4)},
        ]
        assert not protocol.verify_shares(invalid_shares)

    def test_min_clients_requirement(self):
        """Test minimum clients requirement."""
        protocol = ThresholdSecureAggregation(min_clients=3)

        shares = [
            {"param": torch.randn(5, 3)},
            {"param": torch.randn(5, 3)},
        ]

        assert not protocol.verify_shares(shares)
        with pytest.raises(ValueError, match="Not enough clients"):
            protocol.aggregate_shares(shares)
