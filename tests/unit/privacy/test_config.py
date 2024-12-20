import pytest
from pydantic import ValidationError

from nanofed.privacy import NoiseType, PrivacyConfig
from nanofed.privacy.constants import (
    DEFAULT_DELTA,
    DEFAULT_EPSILON,
    MAX_DELTA,
    MAX_EPSILON,
    MIN_DELTA,
    MIN_EPSILON,
)


def test_privacy_config_defaults():
    """Test default configuration values."""
    config = PrivacyConfig()
    assert config.epsilon == DEFAULT_EPSILON
    assert config.delta == DEFAULT_DELTA
    assert config.noise_type == NoiseType.GAUSSIAN


def test_privacy_config_validation():
    """Test configuration validation."""
    # Test invalid epsilon
    with pytest.raises(ValidationError):
        PrivacyConfig(epsilon=-1.0)

    # Test invalid delta
    with pytest.raises(ValidationError):
        PrivacyConfig(delta=2.0)

    # Test invalid gradient norm
    with pytest.raises(ValidationError):
        PrivacyConfig(max_gradient_norm=-1.0)


def test_privacy_config_bounds():
    """Test configuration bounds."""
    config = PrivacyConfig(epsilon=MIN_EPSILON, delta=MIN_DELTA)
    assert config.epsilon == MIN_EPSILON
    assert config.delta == MIN_DELTA

    config = PrivacyConfig(
        epsilon=MAX_EPSILON,
        delta=MAX_DELTA,
    )
    assert config.epsilon == MAX_EPSILON
    assert config.delta == MAX_DELTA


def test_privacy_config_immutable():
    """Test configuration immutability"""
    config = PrivacyConfig()
    with pytest.raises(ValidationError):
        config.epsilon = 2.0
