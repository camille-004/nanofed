import pytest
import torch

from nanofed.server.aggregator import (
    HomomorphicSecureAggregator,
    SecureAggregationConfig,
    SecureMaskingAggregator,
)


@pytest.fixture
def secure_config():
    return SecureAggregationConfig(
        min_clients=2,
        key_size=2048,
        threshold=None,
        masking_seed_size=256,
        dropout_tolerance=0.0,
    )


@pytest.fixture
def fixed_key_iv():
    key = b"0123456789abcdef0123456789abcdef"  # 32 bytes for AES-256
    iv = b"abcdef0123456789"  # 16 bytes for AES
    return key, iv


@pytest.fixture
def model_updates():
    """Generate test model updates."""
    updates = []
    for _ in range(3):
        state = {
            "layer1.weight": torch.randn(10, 5),
            "layer1.bias": torch.randn(10),
        }
        updates.append(state)
    return updates


@pytest.fixture
def small_updates():
    """Generate small test updates for encryption."""
    updates = []
    for _ in range(3):
        state = {
            "small.weight": torch.randn(2, 2),
            "small.bias": torch.randn(2),
        }
        updates.append(state)
    return updates


class TestHomomorphicSecureAggregator:
    """Test homomorphic encryption-based secure aggregation."""

    def test_encryption_decryption(self, secure_config, small_updates):
        """Test encryption and decryption of updates."""
        aggregator = HomomorphicSecureAggregator(secure_config)

        update = small_updates[0]
        encrypted = aggregator.encrypt_update(update)

        assert isinstance(encrypted, dict)
        assert set(encrypted.keys()) == set(update.keys())
        for key in update:
            assert isinstance(encrypted[key], list)
            assert all(isinstance(chunk, bytes) for chunk in encrypted[key])

        decrypted = aggregator.decrypt_aggregate(encrypted)
        assert isinstance(decrypted, dict)
        assert set(decrypted.keys()) == set(update.keys())
        for key in update:
            assert isinstance(decrypted[key], torch.Tensor)
            assert decrypted[key].shape == update[key].shape
            assert torch.allclose(
                decrypted[key], update[key], atol=1e-4
            ), f"Decrypted tensor does not match the original for {key}"

    def test_empty_update(self, secure_config):
        """Test encryption and decryption with an empty update."""
        aggregator = HomomorphicSecureAggregator(secure_config)

        empty_update = {}
        encrypted = aggregator.encrypt_update(empty_update)

        assert isinstance(encrypted, dict)
        assert len(encrypted) == 0

        decrypted = aggregator.decrypt_aggregate(encrypted)
        assert isinstance(decrypted, dict)
        assert len(decrypted) == 0

    def test_large_tensor(self, secure_config):
        """Test encryption and decryption of a large tensor."""
        aggregator = HomomorphicSecureAggregator(secure_config)

        large_tensor = torch.randn(100, 100)
        update = {"large.weight": large_tensor}
        encrypted = aggregator.encrypt_update(update)

        assert isinstance(encrypted, dict)
        assert set(encrypted.keys()) == set(update.keys())
        for key in update:
            assert isinstance(encrypted[key], list)
            expected_num_chunks = (
                update[key].numel() * update[key].element_size()
            ) // aggregator._chunk_size
            if (
                update[key].numel() * update[key].element_size()
            ) % aggregator._chunk_size != 0:
                expected_num_chunks += 1
            assert len(encrypted[key]) == expected_num_chunks

        decrypted = aggregator.decrypt_aggregate(encrypted)
        assert isinstance(decrypted, dict)
        assert set(decrypted.keys()) == set(update.keys())
        for key in update:
            assert isinstance(decrypted[key], torch.Tensor)
            assert decrypted[key].shape == update[key].shape
            assert torch.allclose(
                decrypted[key], update[key], atol=1e-4
            ), f"Decrypted large tensor does not match the original for {key}"

    def test_decryption_failure_tampered_data(
        self, secure_config, small_updates
    ):
        """Tests that decryption fails when encrypted data is tampered with."""
        aggregator = HomomorphicSecureAggregator(secure_config)

        update = small_updates[0]
        encrypted = aggregator.encrypt_update(update)

        tampered_encrypted = {}
        for key, chunks in encrypted.items():
            if chunks:
                tampered_chunk = bytearray(chunks[0])
                tampered_chunk[0] ^= 0xFF  # Flip first byte
                tampered_encrypted[key] = [bytes(tampered_chunk)] + chunks[1:]
            else:
                tampered_encrypted[key] = chunks

        with pytest.raises(ValueError, match="Decryption failed for .*"):
            aggregator.decrypt_aggregate(tampered_encrypted)


class TestSecureMaskingAggregator:
    """Test masking-based secure aggregation."""

    def test_masking_unmasking(
        self, secure_config, small_updates, fixed_key_iv
    ):
        """Test masking and unmasking of updates."""
        key, iv = fixed_key_iv
        aggregator = SecureMaskingAggregator(secure_config, key=key)

        encrypted_updates = [
            aggregator.encrypt_update(update) for update in small_updates[:2]
        ]

        # Aggregate encrypted updates
        aggregated_encrypted = aggregator.aggregate_encrypted(
            encrypted_updates
        )

        assert isinstance(aggregated_encrypted, dict)
        assert set(aggregated_encrypted.keys()) == set(small_updates[0].keys())

        # Decrypt aggregated data
        decrypted_aggregated = aggregator.decrypt_aggregate(
            aggregated_encrypted
        )

        assert isinstance(decrypted_aggregated, dict)
        for key in decrypted_aggregated:
            assert isinstance(
                decrypted_aggregated[key], torch.Tensor
            ), f"Decrypted aggregated data for {key} should be torch.Tensor"
            expected_sum = small_updates[0][key] + small_updates[1][key]
            assert torch.allclose(
                decrypted_aggregated[key], expected_sum, atol=1e-4
            ), f"Decrypted sum does not match expected sum for {key}"

    def test_aggregation_with_insufficient_clients(
        self, secure_config, small_updates
    ):
        """Test that agg. fails when # of clients below min. requirement."""
        aggregator = SecureMaskingAggregator(secure_config)

        encrypted_updates = [
            aggregator.encrypt_update(update) for update in small_updates[:1]
        ]

        with pytest.raises(ValueError, match="Need at least .* clients"):
            aggregator.aggregate_encrypted(encrypted_updates)

    def test_decryption_failure_tampered_data(
        self, secure_config, small_updates
    ):
        """Test that decryption fails when encrypted data is tampered with."""
        aggregator = SecureMaskingAggregator(secure_config)

        # Encrypt the first two updates
        encrypted_updates = [
            aggregator.encrypt_update(update) for update in small_updates[:2]
        ]

        tampered_encrypted_updates = []
        for encrypted in encrypted_updates:
            tampered = {}
            for key, value in encrypted.items():
                if isinstance(value, bytes):
                    tampered_value = bytearray(value)
                    tampered_value[0] ^= 0xFF  # Flip the first byte
                    tampered[key] = bytes(tampered_value)
                else:
                    tampered[key] = value
            tampered_encrypted_updates.append(tampered)

        with pytest.raises(ValueError, match="Decryption failed for .*"):
            aggregator.aggregate_encrypted(tampered_encrypted_updates)

    def test_multiple_rounds_of_aggregation(
        self, secure_config, small_updates, fixed_key_iv
    ):
        """Test multiple rounds of encryption, aggregation, and decryption."""
        key, iv = fixed_key_iv
        aggregator = SecureMaskingAggregator(secure_config, key=key)

        # Round 1
        encrypted_updates_round1 = [
            aggregator.encrypt_update(update) for update in small_updates[:2]
        ]
        aggregated_encrypted_round1 = aggregator.aggregate_encrypted(
            encrypted_updates_round1
        )

        decrypted_aggregated_round1 = aggregator.decrypt_aggregate(
            aggregated_encrypted_round1
        )  # dict[str, torch.Tensor]

        for key in decrypted_aggregated_round1:
            assert isinstance(
                decrypted_aggregated_round1[key], torch.Tensor
            ), f"Decrypted aggregated data for {key} should be torch.Tensor"
            expected_sum = small_updates[0][key] + small_updates[1][key]
            assert torch.allclose(
                decrypted_aggregated_round1[key], expected_sum, atol=1e-4
            ), f"Round 1: Decrypted sum does not match expected sum for {key}"

        # Round 2
        encrypted_updates_round2 = [
            aggregator.encrypt_update(update) for update in small_updates[1:3]
        ]
        aggregated_encrypted_round2 = aggregator.aggregate_encrypted(
            encrypted_updates_round2
        )

        # Decrypt aggregated data
        decrypted_aggregated_round2 = aggregator.decrypt_aggregate(
            aggregated_encrypted_round2
        )  # dict[str, torch.Tensor]

        for key in decrypted_aggregated_round2:
            assert isinstance(
                decrypted_aggregated_round2[key], torch.Tensor
            ), f"Decrypted aggregated data for {key} should be torch.Tensor"
            expected_sum = small_updates[1][key] + small_updates[2][key]
            assert torch.allclose(
                decrypted_aggregated_round2[key], expected_sum, atol=1e-4
            ), f"Round 2: Decrypted sum does not match expected sum for {key}"
