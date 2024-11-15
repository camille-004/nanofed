from __future__ import annotations

import asyncio

import numpy as np
import pytest
from cryptography.fernet import Fernet

from fl.core.exceptions import SecurityError
from fl.core.protocols import ModelWeights
from fl.security.encryption import (
    BasicEncryption,
    DifferentialPrivacy,
    SecurityMiddleware,
)


@pytest.fixture
def test_weights() -> ModelWeights:
    """Fixture for test weights."""
    return {
        "layer1": np.random.randn(10, 10),
        "layer2": np.random.randn(10, 1),
    }


@pytest.fixture
def security_middleware() -> SecurityMiddleware:
    """Fixture for security middleware."""
    return SecurityMiddleware(Fernet.generate_key())


def test_basic_encryption(test_weights):
    """Test basic encryption strategy."""
    encryption = BasicEncryption()

    encrypted = encryption.encrypt(test_weights)
    assert isinstance(encrypted, bytes)

    decrypted = encryption.decrypt(encrypted)
    assert isinstance(decrypted, dict)
    assert all(
        np.array_equal(decrypted[k], test_weights[k]) for k in test_weights
    )


def test_differential_privacy(test_weights):
    """Test differential privacy strategy."""
    dp = DifferentialPrivacy(epsilon=1.0, delta=1e-5)

    noisy_weights = dp.decrypt(dp.encrypt(test_weights))

    assert all(
        not np.array_equal(noisy_weights[k], test_weights[k])
        for k in test_weights
    )
    assert all(
        np.allclose(noisy_weights[k], test_weights[k], atol=1.0)
        for k in test_weights
    )


def test_security_middleware(security_middleware):
    """Test security middleware functionality."""
    message = b"test message"

    # Test enryption/decryption
    encrypted = security_middleware.encrypt_message(message)
    decrypted = security_middleware.decrypt_message(encrypted)  # noqa

    # Test signatures
    signature = security_middleware.create_signature(message)
    assert security_middleware.verify_signature(message, signature)


def test_invalid_weights():
    """Test handling of invalid weights."""
    encryption = BasicEncryption()

    with pytest.raises(SecurityError):
        encryption.encrypt({"invalid": "weights"})

    with pytest.raises(SecurityError):
        encryption.encrypt({"layer1": np.array([np.nan, 1.0])})


def test_encryption_validation():
    """Test validation of encrypted data."""
    encryption = BasicEncryption()

    with pytest.raises(SecurityError):
        encryption.decrypt(b"invalid encrypted data")


@pytest.mark.asyncio
async def test_concurrent_encryption(test_weights):
    """Test concurrent encryption operations."""
    encryption = BasicEncryption()

    async def encrypt_decrypt():
        encrypted = encryption.encrypt(test_weights)
        decrypted = encryption.decrypt(encrypted)
        assert all(
            np.array_equal(decrypted[k], test_weights[k]) for k in test_weights
        )

    # Run multiple encryption/decryption operations concurrently
    await asyncio.gather(*[encrypt_decrypt() for _ in range(5)])
