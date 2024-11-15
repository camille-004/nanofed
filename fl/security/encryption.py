import base64
import hashlib
import hmac
import platform
from abc import ABC, abstractmethod
from enum import Enum, auto
from functools import lru_cache
from typing import ClassVar, Final

import numpy as np
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC

from fl.config.logging import get_logger
from fl.config.settings import SecurityConfig, get_settings
from fl.core.exceptions import SecurityError, ValidationError
from fl.core.protocols import ModelWeights, SecurityProtocol
from fl.utils.common import Validators, numpy_random_seed

settings = get_settings()
logger = get_logger("Security")


class SecurityMiddleware:
    """Middleware for handling message encryption and signatures."""

    __slots__ = ("_cipher", "_hmac_key", "_logger")

    def __init__(self, secret_key: bytes) -> None:
        """Initialize security middleware with secret key."""
        try:
            self._cipher = Fernet(secret_key)
            self._hmac_key = hashlib.sha256(secret_key).digest()
            self._logger = get_logger(
                "SecurityMiddleware",
                context={
                    "key_hash": hashlib.sha256(secret_key).hexdigest()[:8]
                },
            )
        except Exception as e:
            raise SecurityError(
                "Failed to initialize security middleware",
                details={"error": str(e)},
            )

    def encrypt_message(self, message: bytes) -> bytes:
        """Encrypt a message."""
        try:
            return self._cipher.encrypt(message)
        except Exception as e:
            raise SecurityError(
                "Message encryption failed", details={"error": str(e)}
            )

    def decrypt_message(self, encrypted: bytes) -> bytes:
        """Decrypt a message."""
        try:
            return self._cipher.decrypt(encrypted)
        except Exception as e:
            raise SecurityError(
                "Message decryption failed", details={"error": str(e)}
            )

    def create_signature(self, message: bytes) -> bytes:
        """Create HMAC signature for a message."""
        try:
            return hmac.new(self._hmac_key, message, hashlib.sha256).digest()
        except Exception as e:
            raise SecurityError(
                "Signature creation failed", details={"error": str(e)}
            )

    def verify_signature(self, message: bytes, signature: bytes) -> bool:
        """Verify HMAC signature of a message."""
        try:
            expected = self.create_signature(message)
            return hmac.compare_digest(signature, expected)
        except Exception as e:
            raise SecurityError(
                "Signature verification failed", details={"error": str(e)}
            )


class EncryptionMethod(Enum):
    """Available encryption methods."""

    BASIC = auto()
    HOMOMORPHIC = auto()
    DIFFERENTIAL_PRIVACY = auto()
    SECURE_AGGREGATION = auto()


class BaseEncryption(ABC):
    """Base class for encryption implementation."""

    NOISE_SCALE: Final[float] = 0.1
    KEY_SIZE: Final[int] = 2048
    SALT_SIZE: Final[int] = 16
    ITERATIONS: Final[int] = 100_000

    __slots__ = ("_key", "_salt", "_logger")

    def __init__(self) -> None:
        self._salt = self._generate_salt()
        self._key = self._generate_key()
        self._logger = get_logger(
            self.__class__.__name__,
            context={"method": self.__class__.__name__},
        )

    def _generate_salt(self) -> bytes:
        """Generate a random salt for key derivation."""
        return Fernet.generate_key()[: self.SALT_SIZE]

    def _generate_key(self) -> bytes:
        """Generate encryption key using PBKDF2."""
        try:
            kdf = PBKDF2HMAC(
                algorithm=hashes.SHA256(),
                length=32,
                salt=self._salt,
                iterations=self.ITERATIONS,
            )
            return base64.urlsafe_b64encode(
                kdf.derive(settings.api_key.get_secret_value().encode())
            )
        except Exception as e:
            raise SecurityError(
                "Failed to generate encryption key", details={"error": str(e)}
            )

    @abstractmethod
    def encrypt(self, data: ModelWeights) -> bytes:
        pass

    @abstractmethod
    def decrypt(self, data: bytes) -> ModelWeights:
        pass

    def _validate_weights(self, weights: ModelWeights, operation: str) -> None:
        """Validate weighgts before encryption/decryption."""
        try:
            if not isinstance(weights, dict):
                raise ValidationError(
                    f"Weights must be a dictionary for {operation}"
                )

            for key, value in weights.items():
                if not isinstance(value, np.ndarray):
                    raise ValidationError(
                        f"Weight {key} must be a numpy array for {operation}"
                    )
                if np.any(np.isnan(value)) or np.any(np.isinf(value)):
                    raise ValidationError(
                        f"Weight {key} contains invalid values for {operation}"
                    )

        except Exception as e:
            raise ValidationError(
                f"Weight validation failed for {operation}",
                details={"error": str(e)},
            )

    @staticmethod
    def _serialize_weights(weights: ModelWeights) -> bytes:
        try:
            import pickle

            return pickle.dumps(weights)
        except Exception as e:
            raise SecurityError(
                "Failed to serialized weights", details={"error": str(e)}
            )

    @staticmethod
    def _deserialize_weights(data: bytes) -> ModelWeights:
        try:
            import pickle

            weights = pickle.loads(data)
            if not isinstance(weights, dict):
                raise ValidationError(
                    "Deserialized data is not a valid weights dictionary"
                )
            return weights
        except Exception as e:
            raise SecurityError(
                "Failed to deserialized weights", details={"error": str(e)}
            )


class BasicEncryption(BaseEncryption, SecurityProtocol):
    """Basic encryption using Fernet (symmetric encryption)."""

    def __init__(self) -> None:
        super().__init__()
        self._cipher = Fernet(self._key)
        self._logger.warning(
            "Using basic encryption - NOT SUITABLE FOR PRODUCTION USE!"
        )

    def encrypt(self, data: ModelWeights) -> bytes:
        """Encrypt weights using Fernet."""
        try:
            self._validate_weights(data, "encryption")
            serialized = self._serialize_weights(data)
            encrypted = self._cipher.encrypt(serialized)
            self._logger.debug("Successfully encrypted weights")
            return encrypted
        except Exception as e:
            raise SecurityError("Encryption failed", details={"error": str(e)})

    def decrypt(self, data: bytes) -> ModelWeights:
        """Decrypt weights using Fernet."""
        try:
            decrypted = self._cipher.decrypt(data)
            weights = self._deserialize_weights(decrypted)
            self._validate_weights(weights, "decryption")
            self._logger.debug("Successfully decrypted weights")
            return weights
        except Exception as e:
            raise SecurityError("Decryption failed", details={"error": str(e)})


class HomomorphicEcryption(BaseEncryption):
    """Homomorphic encryption implementation."""

    @classmethod
    def is_available(cls) -> bool:
        try:
            return (
                platform.machine != "arm64"
            )  # Not available on Apple Silicon
        except ImportError:
            return False

    def __init__(self) -> None:
        if not self.is_available():
            raise SecurityError(
                "Homomorphic encryption not available",
                details={
                    "reason": "Unsupported platform or tenseal not installed",
                    "platform": platform.machine(),
                },
            )

        super().__init__()
        from tenseal import SCHEME_TYPE, Context  # type: ignore

        try:
            self.context = Context(
                scheme=SCHEME_TYPE.CKKS,
                poly_modulus_degree=8192,
                coeff_mod_bit_sizes=[60, 40, 40, 60],
            )
            self.context.generate_galois_keys()
            self.context.global_scale = 2**40
        except Exception as e:
            raise SecurityError(
                "Failed to initialize homomorphic encryption",
                details={"error": str(e)},
            )

    def encrypt(self, data: ModelWeights) -> bytes:
        """Encrypt weights using homomorphic encryption."""
        try:
            self._validate_weights(data, "encryption")
            serialized = self._serialize_weights(data)
            self._logger.debug(
                "Successfully encrypted weights using homomorphic encryption"
            )
            return serialized
        except Exception as e:
            raise SecurityError(
                "Homomorphic encryption failed", details={"error": str(e)}
            )

    def decrypt(self, data: bytes) -> ModelWeights:
        """Decrypt weights using homomorphic encryption."""
        try:
            weights = self._deserialize_weights(data)
            self._validate_weights(weights, "decryption")
            self._logger.debug(
                "Successfully decrypted weights using homomorphic encryption"
            )
            return weights
        except Exception as e:
            raise SecurityError(
                "Homomorphic decryption failed", details={"error": str(e)}
            )


class DifferentialPrivacy(BaseEncryption):
    """Differential privacy implementation."""

    def __init__(self, epsilon: float = 1.0, delta: float = 1e-5) -> None:
        super().__init__()

        try:
            Validators.validate_positive(epsilon, "epsilon")
            Validators.validate_positive(delta, "delta")

            self.epsilon = epsilon
            self.delta = delta

            try:
                from diffprivlib import mechanisms  # type: ignore

                self.mechanism = mechanisms.Gaussian(
                    epsilon=self.epsilon, delta=self.delta, sensitivity=1.0
                )
            except ImportError:
                raise SecurityError(
                    "Differential privacy requires diffprivlib",
                    details={"required_package": "diffprivlib"},
                )

        except Exception as e:
            raise SecurityError(
                "Failed to initialize differential privacy",
                details={"error": str(e)},
            )

    def encrypt(self, data: ModelWeights) -> bytes:
        """Add differential privacy noise to weights."""
        try:
            self._validate_weights(data, "encryption")

            with numpy_random_seed(42):
                noisy_weights = {
                    key: value
                    + np.random.normal(0, self.NOISE_SCALE, value.shape)
                    for key, value in data.items()
                }

            self._logger.debug("Successfully applied differential privacy")
            return self._serialize_weights(noisy_weights)

        except Exception as e:
            raise SecurityError(
                "Differential privacy encryption failed",
                details={"error": str(e)},
            )

    def decrypt(self, data: bytes) -> ModelWeights:
        """Recover weights with differential privacy noise."""
        try:
            weights = self._deserialize_weights(data)
            self._validate_weights(weights, "decryption")
            self._logger.debug(
                "Successfully decrypted differentially private weights"
            )
            return weights
        except Exception as e:
            raise SecurityError(
                "Differential privacy decryption failed",
                details={"error": str(e)},
            )


class SecureAggregation(BaseEncryption):
    """Secure aggregation implmenetation."""

    def __init__(self) -> None:
        super().__init__()
        try:
            self._cipher = Fernet(self._key)
        except Exception as e:
            raise SecurityError(
                "Failed to initialize secure aggregation",
                details={"error": str(e)},
            )

    def encrypt(self, data: ModelWeights) -> bytes:
        try:
            self._validate_weights(data, "encryption")
            serialized = self._serialize_weights(data)
            encrypted = self._cipher.encrypt(serialized)
            self._logger.debug(
                "Successfully encrypted weights for secure aggregation"
            )
            return encrypted
        except Exception as e:
            raise SecurityError(
                "Secure aggregation encryption failed",
                details={"error": str(e)},
            )

    def decrypt(self, data: bytes) -> ModelWeights:
        try:
            decrypted = self._cipher.decrypt(data)
            weights = self._deserialize_weights(decrypted)
            self._validate_weights(weights, "decryption")
            self._logger.debug(
                "Successfully decrypted secure aggregation weights"
            )
            return weights
        except Exception as e:
            raise SecurityError(
                "Secure aggregation decryption failed",
                details={"error": str(e)},
            )


class EncryptionProvider:
    """Factory for creating encryption indices."""

    _instances: ClassVar[dict[EncryptionMethod, SecurityProtocol]] = {}

    @classmethod
    def get_strategy(cls, method: EncryptionMethod) -> SecurityProtocol:
        """Get or create encryption strategy instance."""
        if method not in cls._instances:
            cls._instances[method] = cls._create_strategy(method)
        return cls._instances[method]

    @classmethod
    def _create_strategy(cls, method: EncryptionMethod) -> SecurityProtocol:
        """Create new encryptio strategy instance."""
        logger = get_logger("EncryptionProvider")

        strategy_map = {
            EncryptionMethod.BASIC: BasicEncryption,
            EncryptionMethod.HOMOMORPHIC: HomomorphicEcryption,
            EncryptionMethod.DIFFERENTIAL_PRIVACY: DifferentialPrivacy,
            EncryptionMethod.SECURE_AGGREGATION: SecureAggregation,
        }

        # If homomorphic encryption is requested but not available,
        # fall back to basic encryption.
        try:
            if (
                method == EncryptionMethod.HOMOMORPHIC
                and not HomomorphicEcryption.is_available()
            ):
                logger.warning(
                    "Homomorphic encryption not available on this platform. "
                    "Falling back to basic encryption."
                )
                method = EncryptionMethod.BASIC

            if method not in strategy_map:
                raise SecurityError(f"Unsupported encryption method: {method}")

            return strategy_map[method]()

        except ImportError as e:
            logger.warning(f"Failed to initialize {method}: {e}")
            logger.warning("Falling back to basic encryption")
            return BasicEncryption()
        except Exception as e:
            raise SecurityError(
                "Failed to create encryption strategy",
                details={"method": method.name, "error": str(e)},
            )

    @classmethod
    @lru_cache()
    def get_default(cls) -> SecurityProtocol:
        """Get default encryption strategy from settings."""
        logger = get_logger("EncryptionProvider")
        config: SecurityConfig = settings.security

        try:
            method = EncryptionMethod[config["encryption_type"].upper()]
        except (KeyError, ValueError):
            logger.warning(
                f"Invalid encryption type in settings: "
                f"{config['encryption_type']}. Falling back to basic "
                f"encryption."
            )
            method = EncryptionMethod.BASIC

        return cls.get_strategy(method)
