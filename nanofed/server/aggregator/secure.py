import os
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Generic, Protocol, Sequence, TypeVar

import numpy as np
import torch
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import padding, rsa
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC

from nanofed.utils import Logger

EncryptedType = TypeVar("EncryptedType")


class SecureAggregationProtocol(Protocol, Generic[EncryptedType]):
    """Protocol for secure aggregation mechanisms."""

    def encrypt_update(
        self, update: dict[str, torch.Tensor]
    ) -> dict[str, EncryptedType]: ...
    def decrypt_aggregate(
        self, encrypted_sum: dict[str, EncryptedType]
    ) -> dict[str, torch.Tensor]: ...
    def aggregate_encrypted(
        self, encrypted_updates: Sequence[dict[str, EncryptedType]]
    ) -> dict[str, EncryptedType]: ...


@dataclass(slots=True, frozen=True)
class SecureAggregationConfig:
    """Configuration for secure aggregation."""

    min_clients: int
    key_size: int = 2048
    threshold: int | None = None  # Number of clients needed for decryption
    masking_seed_size: int = 256
    dropout_tolerance: float = 0.0


class BaseSecureAggregator(ABC, Generic[EncryptedType]):
    """Base class for secure aggregation mechanisms."""

    def __init__(self, config: SecureAggregationConfig) -> None:
        self._config = config
        self._logger = Logger()
        self._setup_crypto()

    @abstractmethod
    def _setup_crypto(self) -> None:
        """Setup cryptographic components."""
        pass

    @abstractmethod
    def encrypt_update(
        self, update: dict[str, torch.Tensor]
    ) -> dict[str, EncryptedType]:
        """Encrypt a model update."""
        pass

    @abstractmethod
    def decrypt_aggregate(
        self, encrypted_sum: dict[str, EncryptedType]
    ) -> dict[str, torch.Tensor]:
        """Decrypt aggregated result."""
        pass

    @abstractmethod
    def aggregate_encrypted(
        self, encrypted_updates: Sequence[dict[str, EncryptedType]]
    ) -> dict[str, EncryptedType]:
        """Aggregate encrypted updates."""
        pass


class HomomorphicSecureAggregator(
    BaseSecureAggregator[list[bytes]], SecureAggregationProtocol[list[bytes]]
):
    """Secure aggregation using homomorphic encryption."""

    def _setup_crypto(self) -> None:
        """Setup homomorphic encryption scheme."""
        self._private_key = rsa.generate_private_key(
            public_exponent=65537, key_size=self._config.key_size
        )
        self._public_key = self._private_key.public_key()
        self._shapes: dict[str, tuple[int, ...]] = {}
        self._chunk_size = (self._config.key_size // 8) - 2 * 32 - 2

    def encrypt_update(
        self, update: dict[str, torch.Tensor]
    ) -> dict[str, list[bytes]]:
        """Encrypt update using chunked RSA."""
        encrypted = {}
        for key, tensor in update.items():
            self._shapes[key] = tensor.shape
            flat_data = tensor.detach().numpy().tobytes()
            chunks = [
                flat_data[i : i + self._chunk_size]
                for i in range(0, len(flat_data), self._chunk_size)
            ]

            if chunks and len(chunks[-1]) < self._chunk_size:
                pad_size = self._chunk_size - len(chunks[-1])
                chunks[-1] = chunks[-1] + bytes([pad_size] * pad_size)

            encrypted[key] = [
                self._public_key.encrypt(
                    chunk,
                    padding.OAEP(
                        mgf=padding.MGF1(algorithm=hashes.SHA256()),
                        algorithm=hashes.SHA256(),
                        label=None,
                    ),
                )
                for chunk in chunks
            ]

        return encrypted

    def aggregate_encrypted(
        self, encrypted_updates: Sequence[dict[str, list[bytes]]]
    ) -> dict[str, list[bytes]]:
        """Aggregate encrypted updates."""
        if len(encrypted_updates) < self._config.min_clients:
            raise ValueError(
                f"Need at least {self._config.min_clients} clients"
            )

        aggregated: dict[str, list[bytes]] = {}
        for key in encrypted_updates[0]:
            num_chunks = len(encrypted_updates[0][key])
            aggregated[key] = [
                self._homomorphic_add(
                    [update[key][i] for update in encrypted_updates]
                )
                for i in range(num_chunks)
            ]
        return aggregated

    def _homomorphic_add(self, encrypted_values: list[bytes]) -> bytes:
        """Add encrypted values."""
        from functools import reduce

        import numpy as np

        arrays = [
            np.frombuffer(val, dtype=np.uint8) for val in encrypted_values
        ]
        result = reduce(np.bitwise_xor, arrays)
        return bytes(result)

    def decrypt_aggregate(
        self, encrypted_sum: dict[str, list[bytes]]
    ) -> dict[str, torch.Tensor]:
        """Decrypt aggregated result."""
        decrypted: dict[str, torch.Tensor] = {}
        for key, encrypted_chunks in encrypted_sum.items():
            try:
                chunks = [
                    self._private_key.decrypt(
                        chunk,
                        padding.OAEP(
                            mgf=padding.MGF1(algorithm=hashes.SHA256()),
                            algorithm=hashes.SHA256(),
                            label=None,
                        ),
                    )
                    for chunk in encrypted_chunks
                ]

                if chunks:
                    last_pad = chunks[-1][-1]
                    if last_pad < self._chunk_size:
                        chunks[-1] = chunks[-1][:-last_pad]

                flat_data = b"".join(chunks)
                array = np.frombuffer(flat_data, dtype=np.float32)
                tensor = torch.tensor(array)
                decrypted[key] = tensor.reshape(self._shapes[key])

            except Exception as e:
                raise ValueError(
                    f"Decryption failed for {key}: {str(e)}"
                ) from e

        return decrypted


class SecureMaskingAggregator(
    BaseSecureAggregator[bytes], SecureAggregationProtocol[bytes]
):
    """Secure aggregation using masking."""

    def __init__(
        self,
        config: SecureAggregationConfig,
        key: bytes | None = None,
    ) -> None:
        if key:
            self._key = key
        super().__init__(config)

    def _setup_crypto(self) -> None:
        """Setup masking scheme."""
        if not hasattr(self, "_key"):
            kdf = PBKDF2HMAC(
                algorithm=hashes.SHA256(),
                length=32,
                salt=os.urandom(16),
                iterations=100000,
            )
            self._key = kdf.derive(os.urandom(32))
            self._iv = os.urandom(16)

        self._shapes: dict[str, tuple[int, ...]] = {}
        self._cumulative_mask: dict[str, torch.Tensor] = {}

    def encrypt_update(
        self, update: dict[str, torch.Tensor]
    ) -> dict[str, bytes]:
        """Encrypt update using masking."""
        encrypted = {}
        for key, tensor in update.items():
            self._shapes[key] = tensor.shape

            # Generate and apply mask
            mask = torch.rand_like(tensor)
            masked = tensor + mask

            # Initialize cumulative mask
            if key not in self._cumulative_mask:
                self._cumulative_mask[key] = torch.zeros_like(tensor)

            self._cumulative_mask[key] += mask

            masked_bytes = masked.detach().numpy().tobytes()

            nonce = os.urandom(12)
            aesgcm = AESGCM(self._key)

            ciphertext = aesgcm.encrypt(nonce, masked_bytes, None)
            encrypted[key] = nonce + ciphertext

        return encrypted

    def decrypt_aggregate(
        self, encrypted_sum: dict[str, bytes]
    ) -> dict[str, torch.Tensor]:
        """Decrypt and unmask aggregated result."""
        decrypted = {}
        for key, encrypted in encrypted_sum.items():
            try:
                nonce = encrypted[:12]
                ciphertext = encrypted[12:]

                aesgcm = AESGCM(self._key)

                data = aesgcm.decrypt(nonce, ciphertext, None)

                # Convert to tensor
                array = np.frombuffer(data, dtype=np.float32)
                tensor = torch.tensor(array)
                decrypted[key] = tensor.reshape(self._shapes[key])

            except Exception as e:
                raise ValueError(
                    f"Decryption failed for {key}: {str(e)}"
                ) from e

        return decrypted

    def aggregate_encrypted(
        self, encrypted_updates: Sequence[dict[str, bytes]]
    ) -> dict[str, bytes]:
        """Aggregate encrypted updates."""
        if len(encrypted_updates) < self._config.min_clients:
            raise ValueError(
                f"Need at least {self._config.min_clients} clients"
            )

        decrypted_sum: dict[str, torch.Tensor] = {}
        for encrypted in encrypted_updates:
            decrypted = self.decrypt_aggregate(encrypted)
            for key, tensor in decrypted.items():
                if key not in decrypted_sum:
                    decrypted_sum[key] = torch.zeros_like(tensor)
                decrypted_sum[key] += tensor

        aggregated: dict[str, bytes] = {}
        for key, tensor_sum in decrypted_sum.items():
            aggregated_sum = tensor_sum - self._cumulative_mask.get(
                key, torch.zeros_like(tensor_sum)
            )
            data = (
                aggregated_sum.detach()
                .cpu()
                .numpy()
                .astype(np.float32)
                .tobytes()
            )

            nonce = os.urandom(12)
            aesgcm = AESGCM(self._key)

            ciphertext = aesgcm.encrypt(nonce, data, None)
            aggregated[key] = nonce + ciphertext

        self._cumulative_mask = {}

        return aggregated
