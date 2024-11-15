import asyncio
import json
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from pathlib import Path
from typing import Any, Final

import numpy as np
from pydantic import BaseModel, ConfigDict

from fl.aggregator import AggregatorType, get_aggregator
from fl.config.logging import get_logger
from fl.config.settings import get_settings
from fl.core.base import Component, Role
from fl.core.exceptions import (
    AggregationError,
    ResourceError,
    ServerError,
    ValidationError,
)
from fl.core.protocols import (
    ModelUpdate,
    ModelWeights,
    SecurityProtocol,
)
from fl.security.encryption import EncryptionProvider
from fl.utils.common import MetricsTracker, Validators, timeout_context

settings = get_settings()


class ServerState(Enum):
    """States that the server can be in."""

    INITIALIZED = auto()
    READY = auto()
    TRAINING = auto()
    AGGREGATING = auto()
    UPDATING = auto()
    ERROR = auto()
    STOPPED = auto()


class ClientInfo(BaseModel):
    """Information about connected clients."""

    client_id: str
    last_seen: float = field(default_factory=time.time)
    round_updates: int = 0
    total_updates: int = 0
    is_active: bool = True
    capabilities: dict = field(default_factory=dict)

    model_config = ConfigDict(arbitrary_types_allowed=True)


@dataclass
class TrainingRound:
    """Information about a training round."""

    round_id: int
    start_time: float
    updates: list[ModelUpdate] = field(default_factory=list)
    completed: bool = False
    aggregated_weights: ModelWeights | None = None
    metrics: dict[str, Any] = field(default_factory=dict)


class Server(Component):
    """Federated learning server component."""

    MIN_CLIENTS: Final[int] = 2
    MAX_ROUNDS: Final[int] = 100
    CLEANUP_INTERVAL: Final[int] = 60
    ROUND_TIMEOUT: Final[int] = 300

    __slots__ = (
        "_state",
        "_aggregator",
        "_security_protocol",
        "_min_clients",
        "_min_updates_ratio",
        "_max_rounds",
        "_round_timeout",
        "_current_round",
        "_round_history",
        "_clients",
        "_active_clients",
        "_global_weights",
        "_storage_path",
        "_model_path",
        "_round_path",
        "_training_task",
        "_cleanup_task",
        "_metrics_tracker",
        "_logger",
    )

    def __init__(
        self,
        aggregator_type: str = "FedAvg",
        security_protocol: SecurityProtocol | None = None,
        min_clients: int = MIN_CLIENTS,
        min_updates_ratio: float = 0.8,
        rounds: int = MAX_ROUNDS,
        timeout: float = ROUND_TIMEOUT,
    ) -> None:
        """Initialize the server."""
        super().__init__(Role.SERVER)

        # Validate inputs
        Validators.validate_positive(min_clients, "min_clients")
        Validators.validate_probability(min_updates_ratio, "min_updates_ratio")
        Validators.validate_positive(rounds, "rounds")
        Validators.validate_positive(timeout, "timeout")

        # Core components
        try:
            agg_type = AggregatorType.from_string(aggregator_type)
        except ValueError:
            raise ValidationError(
                f"Invalid aggregator type: {aggregator_type}"
            )

        self._aggregator = get_aggregator(agg_type)
        self._security_protocol = (
            security_protocol or EncryptionProvider.get_default()
        )

        # Configuration
        self._min_clients = min_clients
        self._min_updates_ratio = min_updates_ratio
        self._max_rounds = rounds
        self._round_timeout = timeout

        # State management
        self._state = ServerState.INITIALIZED
        self._current_round: TrainingRound | None = None
        self._round_history: list[TrainingRound] = []
        self._clients: dict[str, ClientInfo] = {}
        self._active_clients: set[str] = set()
        self._global_weights: ModelWeights | None = None

        # Storage paths
        self._storage_path = settings.data_dir / "server"
        self._model_path = self._storage_path / "models"
        self._round_path = self._storage_path / "rounds"

        # Monitoring
        self._metrics_tracker = MetricsTracker()

        # Background tasks
        self._training_task: asyncio.Task | None = None
        self._cleanup_task: asyncio.Task | None = None

        # Log initialization
        self._logger = get_logger(
            "Server",
            context={"min_clients": min_clients, "max_rounds": rounds},
        )

    async def initialize(self) -> None:
        """Initialize server resources."""
        try:
            self._logger.info("Initializing server...")

            # Create storage directories
            self._storage_path.mkdir(parents=True, exist_ok=True)
            self._model_path.mkdir(exist_ok=True)
            self._round_path.mkdir(exist_ok=True)

            # Start background tasks
            self._cleanup_task = asyncio.create_task(
                self._cleanup_inactive_clients()
            )

            await self._load_or_init_global_weights()

            self._state = ServerState.READY
            self._logger.info("Server initialized and ready")

        except Exception as e:
            self._state = ServerState.ERROR
            self._logger.error(f"Server initialization failed: {str(e)}")
            raise ServerError(
                "Failed to initialize server", details={"error": str(e)}
            )

    async def _load_or_init_global_weights(self) -> None:
        """Load existing weights or initialize new ones."""
        try:
            latest_model = self._model_path / "latest.pkl"
            if latest_model.exists():
                self._global_weights = await self._load_weights(latest_model)
                self._logger.info("Loaded existing global weights")
            else:
                self._logger.info("No existing weights found")
        except Exception as e:
            raise ServerError(
                "Failed to load weights", details={"error": str(e)}
            )

    async def register_client(
        self, client_id: str, signature: str, capabilities: dict
    ) -> dict:
        try:
            if client_id in self._clients:
                self._logger.info(f"Updating existing client: {client_id}")
                self._clients[client_id].is_active = True
                self._clients[client_id].last_seen = time.time()
                self._clients[client_id].capabilities.update(capabilities)
            else:
                self._logger.info(f"Registering new client: {client_id}")
                self._clients[client_id] = ClientInfo(
                    client_id=client_id, capabilities=capabilities
                )
                self._active_clients.add(client_id)

            return {
                "status": "success",
                "client_id": client_id,
                "round": self._current_round.round_id
                if self._current_round
                else 0,
                "server_time": datetime.now().isoformat(),
            }

        except Exception as e:
            raise ServerError(
                "Client registration failed",
                details={"client_id": client_id, "error": str(e)},
            )

    async def start_training(self) -> None:
        """Start the federated learning training process."""
        if self._state != ServerState.READY:
            raise ServerError("Server not ready to start training")

        if len(self._active_clients) < self._min_clients:
            raise ValidationError(
                f"Not enough active clients. Need {self._min_clients}, "
                f"have {len(self._active_clients)}"
            )

        self._state = ServerState.TRAINING
        self._training_task = asyncio.create_task(self._training_loop())
        self._logger.info("Training started")

    async def _collect_updates(self, updates_needed: int) -> None:
        """Collect updates from clients until we have enough."""
        while (
            self._current_round
            and len(self._current_round.updates) < updates_needed
        ):
            await asyncio.sleep(1)

    async def _training_loop(self) -> None:
        """Main training loop."""
        try:
            for round_id in range(self._max_rounds):
                self._logger.info(f"Starting round {round_id}")
                self._current_round = TrainingRound(
                    round_id=round_id, start_time=datetime.now().timestamp()
                )

                updates_needed = max(
                    self._min_clients,
                    int(len(self._active_clients) * self._min_updates_ratio),
                )

                try:
                    async with timeout_context(self._round_timeout):
                        await self._collect_updates(updates_needed)
                except asyncio.TimeoutError:
                    self.logger.warning(f"Round {round_id} timed out")

                if len(self._current_round.updates) >= self._min_clients:
                    await self._aggregate_round()
                else:
                    self._logger.error(
                        f"Insufficient updates for round {round_id}",
                        extra={
                            "received": len(self._current_round.updates),
                            "needed": self._min_clients,
                        },
                    )

                self._round_history.append(self._current_round)
                await self._save_round_state(self._current_round)

        except Exception as e:
            self._state = ServerState.ERROR
            self.logger.error(f"Training loop error: {str(e)}")
            raise ServerError(
                "Training loop failed", details={"error": str(e)}
            )
        finally:
            self._state = ServerState.STOPPED

    async def submit_update(
        self, client_id: str, model_update: ModelUpdate
    ) -> dict[str, Any]:
        """Handle model update from a client."""
        try:
            if client_id not in self._active_clients:
                raise ValidationError(f"Unknown client: {client_id}")

            if not self._current_round:
                raise ValidationError("No active training round")

            if model_update.round_number != self._current_round.round_id:
                raise ValidationError(
                    f"Update for wrong round. Expected "
                    f"{self._current_round.round_id}, "
                    f"got {model_update.round_number}"
                )

            if not self._aggregator.validate_update(model_update):
                raise ValidationError("Invalid model update")

            self._current_round.updates.append(model_update)

            # Update client info
            self._clients[client_id].round_updates += 1
            self._clients[client_id].total_updates += 1
            self._clients[client_id].last_seen = datetime.now().timestamp()

            self._logger.info(
                f"Received update from client {client_id}",
                extra={
                    "round": self._current_round.round_id,
                    "updates_received": len(self._current_round.updates),
                },
            )

            return {
                "status": "success",
                "round": self._current_round.round_id,
                "updates_received": len(self._current_round.updates),
            }

        except ValidationError as e:
            raise ValidationError(str(e))
        except Exception as e:
            raise ServerError(
                "Failed to process update",
                details={"client_id": client_id, "error": str(e)},
            )

    async def _aggregate_round(self) -> None:
        """Aggregate updates for the current round."""
        self._state = ServerState.AGGREGATING
        self._logger.info("Starting round aggregation...")

        try:
            if not self._current_round:
                raise ValidationError("No active round to aggregate")

            decrypted_updates: list[ModelUpdate] = []
            for update in self._current_round.updates:
                try:
                    if self._security_protocol and isinstance(
                        update.weights, bytes
                    ):
                        decrypted_weights = await asyncio.to_thread(
                            self._security_protocol.decrypt, update.weights
                        )
                    else:
                        decrypted_weights = update.weights

                    decrypted_updates.append(
                        ModelUpdate(
                            client_id=update.client_id,
                            weights=decrypted_weights,
                            round_metrics=update.round_metrics,
                            round_number=update.round_number,
                        )
                    )
                except Exception as e:
                    self._logger.error(
                        f"Failed to decrypt update from "
                        f"{update.client_id}: {str(e)}"
                    )

            # Aggregate updates
            self._global_weights = self._aggregator.aggregate(
                decrypted_updates
            )

            # Update round state
            self._current_round.completed = True
            self._current_round.aggregated_weights = self._global_weights

            round_path = (
                self._round_path / f"round_{self._current_round.round_id}"
            )
            round_path.mkdir(exist_ok=True)

            await self._save_weights(
                self._global_weights, round_path / "aggregated_weights.pkl"
            )
            await self._save_weights(
                self._global_weights, self._model_path / "latest_pkl"
            )

            client_metrics = []
            for update in decrypted_updates:
                if isinstance(update.round_metrics, dict):
                    client_metrics.append(update.round_metrics)

            if client_metrics:
                metrics = {
                    "round_id": self._current_round.round_id,
                    "num_updates": len(decrypted_updates),
                    "client_metrics": client_metrics,
                    "timestamp": datetime.now().timestamp(),
                    "round_time": datetime.now().timestamp()
                    - self._current_round.start_time,
                }

                avg_loss = sum(m["loss"] for m in client_metrics) / len(
                    client_metrics
                )
                avg_accuracy = sum(
                    m["accuracy"] for m in client_metrics
                ) / len(client_metrics)

            self._current_round.metrics = metrics

            self._logger.info(
                f"Round {self._current_round.round_id} completed",
                extra={
                    "num_updates": len(decrypted_updates),
                    "avg_loss": avg_loss,
                    "avg_accuracy": avg_accuracy,
                },
            )

        except Exception as e:
            self._logger.error(f"Aggregation error: {str(e)}")
            raise AggregationError(
                "Failed to aggregate updates", details={"error": str(e)}
            )
        finally:
            self._state = ServerState.TRAINING

    def _remove_inactive_clients(self) -> None:
        """Remove inactive clients."""
        current_time = datetime.now().timestamp()
        inactive_threshold = current_time - settings.timeout

        for client_id, info in list(self._clients.items()):
            if info.last_seen < inactive_threshold:
                info.is_active = False
                self._active_clients.discard(client_id)
                self._logger.info(f"Client {client_id} marked as inactive")

    async def _cleanup_inactive_clients(self) -> None:
        """Remove inactive clients periodically."""
        while True:
            try:
                self._remove_inactive_clients()
                await asyncio.sleep(self.CLEANUP_INTERVAL)
            except Exception as e:
                self._logger.error(f"Cleanup error: {str(e)}")
                await asyncio.sleep(self.CLEANUP_INTERVAL)

    async def cleanup(self) -> None:
        """Clean up server resources."""
        try:
            if self._cleanup_task:
                self._cleanup_task.cancel()
                try:
                    await self._cleanup_task
                except asyncio.CancelledError:
                    pass

            if self._training_task:
                self._training_task.cancel()
                try:
                    await self._training_task
                except asyncio.CancelledError:
                    pass

            # Save final state
            if self._current_round:
                await self._save_round_state(self._current_round)

            # Save final model weights
            if self._global_weights is not None:
                await self._save_weights(
                    self._global_weights,
                    self._model_path / "final_weights.pkl",
                )

            metrics_path = self._storage_path / "metrics.json"
            try:
                with metrics_path.open("w") as f:
                    json.dump(self._metrics_tracker.values, f, indent=2)
            except Exception as e:
                self._logger.error(f"Failed to save metrics: {str(e)}")

            self._logger.info("Server cleanup completed")

        except Exception as e:
            self._logger.error(f"Cleanup error: {str(e)}")
            raise ServerError("Cleanup failed", details={"error": str(e)})

    async def _save_weights(self, weights: ModelWeights, path: Path) -> None:
        """Save model weights to file"""
        try:
            import pickle

            path.parent.mkdir(parents=True, exist_ok=True)

            if not isinstance(weights, dict):
                raise ValidationError("Weights must be a dictionary")

            for key, value in weights.items():
                if not isinstance(value, np.ndarray):
                    raise ValidationError(
                        f"Weight {key} must be a numpy array"
                    )
                if np.any(np.isnan(value)) or np.any(np.isinf(value)):
                    raise ValidationError(
                        f"Weight {key} contains invalid values"
                    )

            async with asyncio.Lock():
                with path.open("wb") as f:
                    pickle.dump(weights, f)

            self._logger.debug(f"Saved weights to {path}")

        except Exception as e:
            raise ResourceError(
                f"Failed to save waits to {path}", details={"error": str(e)}
            )

    async def _load_weights(self, path: Path) -> ModelWeights:
        """Load model weights from file."""
        try:
            import pickle

            if not path.exists():
                raise FileNotFoundError(f"Weights file not found: {path}")

            async with asyncio.Lock():
                with path.open("rb") as f:
                    weights = pickle.load(f)

            if not isinstance(weights, dict):
                raise ValidationError("Loaded weights must be a dictionary")

            for key, value in weights.items():
                if not isinstance(value, np.ndarray):
                    raise ValidationError(
                        f"Loaded weight {key} must be a numpy array"
                    )
                if np.any(np.isnan(value)) or np.any(np.isinf(value)):
                    raise ValidationError(
                        f"Loaded weight {key} contains invalid values"
                    )

            self._logger.debug(f"Loaded weights from {path}")
            return weights

        except Exception as e:
            raise ResourceError(
                f"Failed to load weights from {path}",
                details={"error": str(e)},
            )

    async def _save_round_state(self, round_state: TrainingRound) -> None:
        """Save training round state to file."""
        try:
            path = self._round_path / f"round_{round_state.round_id}.json"

            # Prepare round data
            state = {
                "rouond_id": round_state.round_id,
                "start_time": round_state.start_time,
                "completed": round_state.completed,
                "num_updates": len(round_state.updates),
                "metrics": round_state.metrics,
                "timestamp": datetime.now().timestamp(),
            }

            path.parent.mkdir(parents=True, exist_ok=True)

            async with asyncio.Lock():
                with path.open("w") as f:
                    json.dump(state, f, indent=2)

            self._logger.debug(f"Saved round state to {path}")

        except Exception as e:
            raise ResourceError(
                "Failed to save round state",
                details={"round": round_state.round_id, "error": str(e)},
            )

    async def get_round_info(self, round_id: int) -> dict[str, Any]:
        """Get information about a specific training round."""
        try:
            path = self._round_path / f"round_{round_id}.json"
            if not path.exists():
                raise ValidationError(f"No data for round {round_id}")

            async with asyncio.Lock():
                with path.open("r") as f:
                    return json.load(f)

        except Exception as e:
            raise ServerError(
                "Failed to get round info",
                details={"round": round_id, "Error": str(e)},
            )

    def get_status(self) -> dict[str, Any]:
        """Get current server status."""
        return {
            "status": (
                "ready"
                if self._state == ServerState.READY
                else str(self._state)
            ),
            "active_clients": len(self._active_clients),
            "current_round": (
                self._current_round.round_id if self._current_round else None
            ),
        }

    async def __aenter__(self) -> "Server":
        """Async context manager entry."""
        await self.initialize()
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: Any,
    ) -> None:
        """async context manager exit."""
        await self.cleanup()
