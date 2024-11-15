import asyncio
import hashlib
import json
import pickle
import time
from abc import abstractmethod
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import Any, AsyncGenerator, Final, Self, final

import aiofiles
import aiohttp
import numpy as np
from aiohttp import ClientTimeout

from fl.config.settings import get_settings
from fl.core.base import ClientState, Component, Role, validate_weights
from fl.core.exceptions import ClientError, ValidationError
from fl.core.protocols import ModelUpdate, ModelWeights, SecurityProtocol
from fl.security.encryption import EncryptionProvider
from fl.utils.common import retry_async, timed

EncryptedWeights = bytes
ModelWeightsType = ModelWeights | EncryptedWeights

settings = get_settings()


class ClientMode(Enum):
    LOCAL = auto()  # For local testing without server
    REMOTE = auto()  # For production with server connection


@dataclass(frozen=True, slots=True)
class ClientMetrics:
    """Immutable container for client training metrics."""

    loss: float
    accuracy: float
    training_time: float
    samples_processed: int
    round_number: int
    timestamp: float = field(default_factory=lambda: time.time())


@dataclass(slots=True)
class TrainingConfig:
    """Configuration for local training."""

    batch_size: int = 32
    local_epochs: int = 1
    learning_rate: float = 0.01
    momentum: float = 0.9
    validation_split: float = 0.2
    max_rounds: int = 100


class Client(Component):
    """Base client implementation."""

    __slots__ = (
        "client_id",
        "local_weights",
        "_update_queue",
        "_state",
        "_metrics_history",
        "_security_protocol",
        "_training_config",
        "_executor",
        "_data_cache",
        "_secure_storage_path",
        "_server_session",
        "_mode",
        "_last_cleanup",
    )

    MAX_CACHE_SIZE: Final[int] = 1000
    CLEANUP_INTERVAL: Final[int] = 300
    MAX_RETRIES: Final[int] = 3
    RETRY_DELAY: Final[float] = 1.0

    def __init__(
        self,
        client_id: str,
        mode: ClientMode = ClientMode.LOCAL,
        security_protocol: SecurityProtocol | None = None,
        training_config: TrainingConfig | None = None,
    ) -> None:
        """Initialize client with optional security and training configs."""
        super().__init__(Role.CLIENT)
        self.client_id = client_id
        self._mode = mode
        self.local_weights: ModelWeights = {}
        self._secure_storage_path = (
            settings.data_dir / "secure_storage" / client_id
        )
        self._secure_storage_path.mkdir(parents=True, exist_ok=True)

        self._update_queue: deque[ModelUpdate] = deque(
            maxlen=settings.max_clients
        )
        self._state = ClientState.INITIALIZED
        self._metrics_history: list[ClientMetrics] = []
        self._security_protocol = (
            security_protocol or EncryptionProvider.get_default()
        )
        self._training_config = training_config or TrainingConfig()
        self._executor = ThreadPoolExecutor(max_workers=1)

        self._data_cache: dict[str, Any] = {}
        if len(self._data_cache) > self.MAX_CACHE_SIZE:
            self._clean_cache()

        self._server_session = aiohttp.ClientSession()
        self._last_cleanup = time.time()

        self.logger.info(
            "Initialized client",
            extra={
                "mode": mode.name,
                "security": self._security_protocol.__class__.__name__,
            },
        )

    def _clean_cache(self) -> None:
        """Clean old entries from data cache."""
        if len(self._data_cache) > self.MAX_CACHE_SIZE:
            sorted_items = sorted(
                self._data_cache.items(),
                key=lambda x: x[1].get("timestamp", 0),
            )
            for key, _ in sorted_items[
                : len(self._data_cache) - self.MAX_CACHE_SIZE
            ]:
                del self._data_cache[key]

    @classmethod
    async def create(
        cls,
        client_id: str,
        mode: ClientMode = ClientMode.LOCAL,
        security_protocol: SecurityProtocol | None = None,
        training_config: TrainingConfig | None = None,
    ) -> Self:
        """Asynchronous factory method for creating a new client."""
        try:
            self = cls(client_id, mode, security_protocol, training_config)
            await self._initialize()
            return self
        except Exception as e:
            raise ClientError(
                "Failed to create client", details={"error": str(e)}
            )

    async def _initialize(self) -> None:
        """Initialize client resources and verify environment."""
        self.logger.info(
            "Initializing client", extra={"client_id": self.client_id}
        )
        try:
            async with asyncio.TaskGroup() as tg:
                tg.create_task(self._verify_environment())
                tg.create_task(self._setup_secure_storage())

            if self._mode == ClientMode.REMOTE:
                await self._register_with_server()
            else:
                await self._setup_local_mode()

        except* Exception as eg:
            self._state = ClientState.ERROR
            self.logger.error(
                "Initialization failed", extra={"errors": str(eg)}
            )
            raise ClientError(
                "Failed to initialize client", details={"errors": str(eg)}
            )
        except* Exception as e:
            self._state = ClientState.ERROR
            self.logger.error(f"Initialization failed: {str(e)}")
            raise ClientError(
                "Failed to initialize client", details={"error": str(e)}
            )

    async def _verify_environment(self) -> None:
        required_paths = [
            Path("./cache"),
            settings.model_dir,
            Path("./logs"),
            settings.data_dir,
        ]
        try:
            for path in required_paths:
                path.mkdir(exist_ok=True)
                self.logger.debug(f"Verified path: {path}")
        except Exception as e:
            raise ClientError(
                "Failed to verify environment",
                details={"error": str(e), "path": str(path)},
            )

    async def _setup_local_mode(self) -> None:
        try:
            registration = {
                "client_id": self.client_id,
                "mode": "local",
                "timestamp": time.time(),
                "local_token": hashlib.sha256(
                    f"local:{self.client_id}:{time.time()}".encode()
                ).hexdigest(),
            }

            reg_path = self._secure_storage_path / "registration.json"
            async with aiofiles.open(reg_path, "w") as f:
                await f.write(json.dumps(registration, indent=2))

            self.logger.info("Local mode initialized successfully")

        except Exception as e:
            raise ClientError(
                "Failed to setup local mode", details={"error": str(e)}
            )

    async def _setup_secure_storage(self) -> None:
        """Setup secure storage for client data."""
        try:
            self._secure_storage_path.mkdir(parents=True, exist_ok=True)
            cache_path = self._secure_storage_path / "cache"
            cache_path.mkdir(exist_ok=True)

            metadata = {
                "client_id": self.client_id,
                "created_at": time.time(),
                "last_updated": time.time(),
                "encryption_type": self._security_protocol.__class__.__name__,
            }

            async with aiofiles.open(
                self._secure_storage_path / "metadata.json", "w"
            ) as f:
                await f.write(json.dumps(metadata, indent=2))

            self.logger.info("Secure storage initialized successfully")

        except Exception as e:
            raise ClientError(
                "Failed to setup secure storage", details={"error": str(e)}
            )

    async def _register_with_server(self) -> None:
        """Register with server retry logic."""
        if self._mode != ClientMode.REMOTE:
            return

        async def registration_attempt() -> None:
            signature = hashlib.sha256(
                f"{self.client_id}:{time.time()}".encode()
            ).hexdigest()

            payload = {
                "client_id": self.client_id,
                "signature": signature,
                "capabilities": {
                    "batch_size": self._training_config.batch_size,
                    "local_epochs": self._training_config.local_epochs,
                    "encryption_type": (
                        self._security_protocol.__class__.__name__
                    ),
                },
            }

            async with self._server_session.post(
                f"{settings.server_url}/register",
                json=payload,
                timeout=ClientTimeout(total=settings.timeout),
            ) as response:
                if response.status != 200:
                    raise ConnectionError(
                        f"Server registration failed: {response.status}"
                    )

                registration = await response.json()
                async with aiofiles.open(
                    self._secure_storage_path / "registration.json", "w"
                ) as f:
                    await f.write(json.dumps(registration, indent=2))

                self.logger.info("Successfully registered with server")

        try:
            await retry_async(
                registration_attempt,
                max_attempts=self.MAX_RETRIES,
                delay=self.RETRY_DELAY,
                backoff=2.0,
                exceptions=(
                    ConnectionError,
                    TimeoutError,
                    aiohttp.ClientError,
                ),
            )
        except Exception as e:
            self.logger.error(
                f"Failed to register with server after {self.MAX_RETRIES} "
                f"attempts: {str(e)}"
            )
            raise ClientError(
                "Server registration failed", details={"error": str(e)}
            )

    async def cleanup(self) -> None:
        try:
            if self._state != ClientState.ERROR:
                self._secure_storage_path.mkdir(parents=True, exist_ok=True)
                self.save_state(self._secure_storage_path / "final_state.pkl")

            if time.time() - self._last_cleanup > self.CLEANUP_INTERVAL:
                self._clean_cache()
                self._last_cleanup = time.time()

            metadata_path = self._secure_storage_path / "metadata.json"
            try:
                if metadata_path.exists():
                    async with aiofiles.open(metadata_path, "r") as f:
                        metadata = json.loads(await f.read())
                else:
                    metadata = {}

                metadata.update(
                    {
                        "last_updated": time.time(),
                        "final_state": self._state.name,
                    }
                )

                async with aiofiles.open(metadata_path, "w") as f:
                    await f.write(json.dumps(metadata, indent=2))

            except Exception as e:
                self.logger.error(
                    "Error updating metadata", extra={"error": str(e)}
                )

            self.logger.info(f"Client {self.client_id} cleanup completed")

        except Exception as e:
            self.logger.error("Error during cleanup", extra={"error": str(e)})
            raise ClientError("Cleanup failed", details={"error": str(e)})

    @property
    def state(self) -> ClientState:
        return self._state

    @property
    def metrics(self) -> list[ClientMetrics]:
        return self._metrics_history.copy()

    @validate_weights
    async def update_global_model(
        self, new_weights: ModelWeightsType
    ) -> ModelWeights:
        """Update local model with new global weights."""
        try:
            self._state = ClientState.UPDATING

            if isinstance(new_weights, bytes) and self._security_protocol:
                decrypted_weights = await asyncio.to_thread(
                    self._security_protocol.decrypt, new_weights
                )
                if not isinstance(decrypted_weights, dict):
                    raise ValidationError(
                        "Decrypted weights must be a dictionary"
                    )
            elif isinstance(new_weights, dict):
                decrypted_weights = new_weights
            else:
                raise ValidationError("Invalid weights format")

            if not isinstance(decrypted_weights, dict):
                self._state = ClientState.ERROR
                raise ValidationError("Decrypted weights must be a dictionary")

            self.local_weights: dict[str, np.ndarray] = decrypted_weights  # type: ignore
            self.logger.info(
                "Successfully updated local model with global weights"
            )
            self._state = ClientState.INITIALIZED

            return decrypted_weights

        except Exception as e:
            self._state = ClientState.ERROR
            self.logger.error(f"Error updating global model: {str(e)}")
            raise

    @final
    async def train_local_model(
        self,
        data: np.ndarray,
        labels: np.ndarray,
    ) -> AsyncGenerator[ModelUpdate, None]:
        """Async generator for local training updates with validation."""
        if self._state == ClientState.ERROR:
            raise ClientError("Client is in error state")

        if not isinstance(data, np.ndarray) or not isinstance(
            labels, np.ndarray
        ):
            raise ValidationError("Data and labels must be numpy arrays")

        self._state = ClientState.TRAINING
        training_rounds = range(self._training_config.max_rounds)

        try:
            for round_num in training_rounds:
                metrics = await self._train_one_round(data, labels, round_num)
                self._metrics_history.append(metrics)

                if self._security_protocol:
                    encrypted_weights = await asyncio.to_thread(
                        self._security_protocol.encrypt, self.local_weights
                    )
                    weights: ModelWeightsType = encrypted_weights
                else:
                    weights = self.local_weights

                update = ModelUpdate(
                    client_id=self.client_id,
                    weights=weights,  # type: ignore[arg-type]
                    round_metrics={
                        "loss": metrics.loss,
                        "accuracy": metrics.accuracy,
                        "samples_processed": metrics.samples_processed,
                    },
                    round_number=round_num,
                )
                yield update

        except Exception as e:
            self._state = ClientState.ERROR
            self.logger.error("Error during training", extra={"error": str(e)})
            raise ClientError("Training failed", details={"error": str(e)})
        finally:
            self._state = ClientState.STOPPED

    async def _train_one_round(
        self, data: np.ndarray, labels: np.ndarray, round_num: int
    ) -> ClientMetrics:
        """Train the model for one round."""
        result = await timed(self._train_round_impl)(data, labels, round_num)
        metrics, training_time = result
        return ClientMetrics(
            loss=metrics.loss,
            accuracy=metrics.accuracy,
            training_time=training_time,
            samples_processed=metrics.samples_processed,
            round_number=metrics.round_number,
        )

    async def _train_round_impl(
        self, data: np.ndarray, labels: np.ndarray, round_num: int
    ) -> ClientMetrics:
        """Implementation of training round logic."""
        try:
            num_samples = len(data)
            indices = np.random.permutation(num_samples)

            for i in range(0, num_samples, self._training_config.batch_size):
                batch_indices = indices[
                    i : i + self._training_config.batch_size
                ]
                batch_data = data[batch_indices]
                batch_labels = labels[batch_indices]
                await self._train_batch(batch_data, batch_labels)

            loss, accuracy = await asyncio.to_thread(
                self._compute_metrics, data, labels
            )

            metrics = ClientMetrics(
                loss=loss,
                accuracy=accuracy,
                training_time=0.0,
                samples_processed=num_samples,
                round_number=round_num,
            )
            return metrics

        except Exception as e:
            raise ClientError(
                "Round training failed",
                details={"round": round_num, "error": str(e)},
            )

    @abstractmethod
    async def _train_batch(
        self, batch_data: np.ndarray, batch_labels: np.ndarray
    ) -> None:
        """Train model on a single batch.

        This method must be implemented by concrete client classes.
        """
        raise NotImplementedError

    def _compute_metrics(
        self, data: np.ndarray, labels: np.ndarray
    ) -> tuple[float, float]:
        """Compute metrics for model evaluation."""
        try:
            predictions = np.random.rand(len(data))  # Placeholder
            loss = float(np.mean((predictions - labels) ** 2))
            accuracy = float(
                np.mean((predictions > 0.5).astype(int) == labels)
            )
            return loss, accuracy
        except Exception as e:
            self.logger.error(
                "Error computing metrics", extra={"error": str(e)}
            )
            return 0.0, 0.0

    def save_state(self, path: Path) -> None:
        """Save client state to disk."""
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            state = {
                "weights": self.local_weights.copy(),
                "metrics": self._metrics_history,
                "config": self._training_config,
            }
            with path.open("wb") as f:
                pickle.dump(state, f)

            self.logger.debug(f"Saved state to {path}")

        except Exception as e:
            raise ClientError(
                "Failed to save state",
                details={"path": str(path), "error": str(e)},
            )

    @classmethod
    def load_state(cls, path: Path, client_id: str) -> "Client":
        """Load client state from disk."""
        try:
            with path.open("rb") as f:
                state = pickle.load(f)

            client = cls(client_id)
            client.local_weights = state["weights"]
            client._metrics_history = state["metrics"]
            client._training_config = state["config"]
            return client

        except Exception as e:
            raise ClientError(
                "Failed to load state",
                details={"path": str(path), "error": str(e)},
            )

    async def __aenter__(self) -> "Client":
        """Async context manager entry."""
        return self

    def __aiter__(self) -> "Client":
        """Mkae the client itself iterable for training."""
        return self

    async def __anext__(self):
        raise NotImplementedError("Subclasses must implement __anext__")

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: Any,
    ) -> None:
        """Context manager exit."""
        try:
            await self.cleanup()
        except Exception as e:
            self.logger.error(f"Error during cleanup: {str(e)}")
            raise ClientError("Cleanup failed", details={"error": str(e)})
