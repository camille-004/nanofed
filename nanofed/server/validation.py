from dataclasses import dataclass
from enum import Enum, auto
from typing import Protocol, Sequence

import torch
from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding, rsa
from cryptography.hazmat.primitives.asymmetric.rsa import RSAPublicKey

from nanofed.core import ModelUpdate
from nanofed.utils import Logger


class ValidationResult(Enum):
    """Result of update validation."""

    VALID = auto()
    INVALID_SHAPE = auto()
    INVALID_RANGE = auto()
    INVALID_SIGNATURE = auto()
    ANOMALOUS = auto()


@dataclass(frozen=True)
class ValidationConfig:
    """Configuration for update validation."""

    max_norm: float = 10.0
    max_update_size: int = 1024 * 1024 * 100
    min_clients_for_stats: int = 5
    z_score_threshold: float = 2.0
    signature_required: bool = True


class ModelValidator(Protocol):
    """Protocol for model update validation."""

    def validate_shape(
        self, update: ModelUpdate, reference: dict[str, torch.Size]
    ) -> ValidationResult: ...
    def validate_range(
        self, update: ModelUpdate, config: ValidationConfig
    ) -> ValidationResult: ...
    def validate_statistics(
        self, update: ModelUpdate, reference_updates: Sequence[ModelUpdate]
    ) -> ValidationResult: ...
    def validate_signature(
        self, update: ModelUpdate, public_key: bytes
    ) -> ValidationResult: ...


class DefaultModelValidator:
    """Default implementation of model validation."""

    def __init__(self, config: ValidationConfig) -> None:
        self._config = config
        self._logger = Logger()

    def validate_shape(
        self, update: ModelUpdate, reference: dict[str, torch.Size]
    ) -> ValidationResult:
        """Validate tensor shapes match reference."""
        try:
            for key, shape in reference.items():
                if key not in update["model_state"]:
                    self._logger.warning(f"Missing parameter: {key}")
                    return ValidationResult.INVALID_SHAPE

                if update["model_state"][key].shape != shape:
                    self._logger.warning(
                        f"Shape mismatch for {key}: "
                        f"got {update['model_state'][key].shape}, "
                        f"expected {shape}"
                    )
                    return ValidationResult.INVALID_SHAPE

            return ValidationResult.VALID

        except Exception as e:
            self._logger.error(f"Shape validation failed: {str(e)}")
            return ValidationResult.INVALID_SHAPE

    def validate_range(
        self, update: ModelUpdate, config: ValidationConfig
    ) -> ValidationResult:
        """Validation parameter values are within acceptable range."""
        try:
            for tensor in update["model_state"].values():
                if not torch.all(torch.isfinite(tensor)):
                    return ValidationResult.INVALID_RANGE

                norm = torch.norm(tensor)
                if norm > config.max_norm:
                    return ValidationResult.INVALID_RANGE

            return ValidationResult.VALID

        except Exception as e:
            self._logger.error(f"Range validation failed: {str(e)}")
            return ValidationResult.INVALID_RANGE

    def validate_statistics(
        self, update: ModelUpdate, reference_updates: Sequence[ModelUpdate]
    ) -> ValidationResult:
        """Validate update against statistical properties of other updates."""
        if len(reference_updates) < self._config.min_clients_for_stats:
            return ValidationResult.VALID

        try:
            norms = []
            for ref in reference_updates:
                tensors = torch.cat(
                    [t.flatten() for t in ref["model_state"].values()]
                )
                norms.append(torch.norm(tensors).item())

            ref_mean = torch.tensor(norms).mean()
            ref_std = torch.tensor(norms).std()

            update_tensors = torch.cat(
                [t.flatten() for t in update["model_state"].values()]
            )
            update_norm = torch.norm(update_tensors).item()

            # Check if update is within z_score_threshold standard deviations
            z_score = abs(update_norm - ref_mean) / (ref_std + 1e-8)
            if z_score > self._config.z_score_threshold:
                return ValidationResult.ANOMALOUS

            return ValidationResult.VALID

        except Exception as e:
            self._logger.error(f"Statistical validation failed: {str(e)}")
            return ValidationResult.ANOMALOUS


class SecurityManager:
    """Manages security aspects of FL."""

    def __init__(self) -> None:
        self._private_key = rsa.generate_private_key(
            public_exponent=65537, key_size=2048
        )
        self._public_key = self._private_key.public_key()
        self._logger = Logger()

    def get_public_key(self) -> bytes:
        """Get public key for client verification."""
        return self._public_key.public_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PublicFormat.SubjectPublicKeyInfo,
        )

    def sign_update(self, update: ModelUpdate) -> bytes:
        """Sign model update."""
        try:
            message = b""
            for key, tensor in sorted(update["model_state"].items()):
                message += (
                    key.encode("utf-8") + b":" + tensor.numpy().tobytes()
                )

            # Sign
            signature = self._private_key.sign(
                message,
                padding.PSS(
                    mgf=padding.MGF1(hashes.SHA256()),
                    salt_length=padding.PSS.MAX_LENGTH,
                ),
                hashes.SHA256(),
            )
            return signature

        except Exception as e:
            self._logger.error(f"Failed to sign update: {str(e)}")
            raise

    def verify_signature(
        self, update: ModelUpdate, signature: bytes, public_key: bytes
    ) -> bool:
        """Verify update signature."""
        try:
            message_bytes = b""
            for key, tensor in sorted(update["model_state"].items()):
                message_bytes += (
                    key.encode("utf-8") + b":" + tensor.numpy().tobytes()
                )

            # Load public key
            public_key_obj = serialization.load_pem_public_key(public_key)

            if not isinstance(public_key_obj, RSAPublicKey):
                self._logger.error("Unsupported public key type.")
                return False

            public_key_obj.verify(
                signature,
                message_bytes,
                padding.PSS(
                    mgf=padding.MGF1(hashes.SHA256()),
                    salt_length=padding.PSS.MAX_LENGTH,
                ),
                hashes.SHA256(),
            )
            return True

        except InvalidSignature:
            return False
        except Exception as e:
            self._logger.error(f"Signature verification failed: {str(e)}")
            return False
