import pytest
import torch

from nanofed.server.validation import (
    DefaultModelValidator,
    SecurityManager,
    ValidationConfig,
    ValidationResult,
)


@pytest.fixture
def validation_config():
    return ValidationConfig(
        max_norm=10.0,
        max_update_size=1024 * 1024,
        min_clients_for_stats=3,
        z_score_threshold=2.0,
        signature_required=True,
    )


@pytest.fixture
def reference_shapes():
    return {
        "layer1.weight": torch.Size([10, 5]),
        "layer1.bias": torch.Size([10]),
    }


@pytest.fixture
def valid_update():
    return {
        "client_id": "client_1",
        "round_number": 1,
        "model_state": {
            "layer1.weight": torch.randn(10, 5),
            "layer1.bias": torch.randn(10),
        },
        "metrics": {"accuracy": 0.9},
    }


@pytest.fixture
def reference_updates():
    updates = []
    for i in range(5):
        updates.append(
            {
                "client_id": f"client_{i}",
                "round_number": 1,
                "model_state": {
                    "layer1.weight": torch.randn(10, 5),
                    "layer1.bias": torch.randn(10),
                },
                "metrics": {"accuracy": 0.9},
            }
        )
    return updates


class TestDefaultModelValidator:
    """Test model update validation."""

    def test_shape_validation(
        self, validation_config, reference_shapes, valid_update
    ):
        """Test shape validation."""
        validator = DefaultModelValidator(validation_config)

        result = validator.validate_shape(valid_update, reference_shapes)
        assert result == ValidationResult.VALID

        invalid_update = valid_update.copy()
        invalid_update["model_state"] = {
            "layer1.weight": torch.randn(5, 5),
            "layer1.bias": torch.randn(10),
        }
        result = validator.validate_shape(invalid_update, reference_shapes)
        assert result == ValidationResult.INVALID_SHAPE

        missing_param = valid_update.copy()
        missing_param["model_state"] = {
            "layer1.weight": torch.randn(10, 5)
            # Missing bias
        }
        result = validator.validate_shape(missing_param, reference_shapes)
        assert result == ValidationResult.INVALID_SHAPE

    def test_range_validation(self, validation_config, valid_update):
        """Test value range validation."""
        validator = DefaultModelValidator(validation_config)

        result = validator.validate_range(valid_update, validation_config)
        assert result == ValidationResult.VALID

        invalid_update = valid_update.copy()
        invalid_update["model_state"] = {
            "layer1.weight": torch.tensor([[float("nan")]]),
            "layer1.bias": torch.randn(10),
        }
        result = validator.validate_range(invalid_update, validation_config)
        assert result == ValidationResult.INVALID_RANGE

        large_update = valid_update.copy()
        large_update["model_state"] = {
            "layer1.weight": torch.ones(10, 5) * 1000,
            "layer1.bias": torch.randn(10),
        }
        result = validator.validate_range(large_update, validation_config)
        assert result == ValidationResult.INVALID_RANGE

    def test_statistical_validation(
        self, validation_config, valid_update, reference_updates
    ):
        """Test statistical validation."""
        validator = DefaultModelValidator(validation_config)

        result = validator.validate_statistics(valid_update, reference_updates)
        assert result == ValidationResult.VALID

        anomalous_update = valid_update.copy()
        anomalous_update["model_state"] = {
            "layer1.weight": torch.ones(10, 5) * 50,
            "layer1.bias": torch.ones(10) * 50,
        }
        result = validator.validate_statistics(
            anomalous_update, reference_updates
        )
        assert result == ValidationResult.ANOMALOUS

        result = validator.validate_statistics(
            valid_update,
            reference_updates[:2],  # Less than min_clients_for_stats
        )
        assert result == ValidationResult.VALID


class TestSecurityManager:
    """Test security management."""

    def test_update_signing(self, valid_update):
        """Test update signing and verification."""
        security = SecurityManager()

        public_key = security.get_public_key()
        assert isinstance(public_key, bytes)

        signature = security.sign_update(valid_update)
        assert isinstance(signature, bytes)

        assert security.verify_signature(valid_update, signature, public_key)

        modified_update = valid_update.copy()
        modified_update["model_state"] = {
            k: v + 0.1 for k, v in valid_update["model_state"].items()
        }
        assert not security.verify_signature(
            modified_update, signature, public_key
        )

        other_security = SecurityManager()
        other_public_key = other_security.get_public_key()
        assert not security.verify_signature(
            valid_update, signature, other_public_key
        )
