import asyncio
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, AsyncGenerator, Callable, Sequence

import torch

from nanofed.core import ModelManagerProtocol, ModelUpdate
from nanofed.orchestration.types import (
    ClientInfo,
    RoundMetrics,
    RoundStatus,
    TrainingProgress,
)
from nanofed.server import BaseAggregator
from nanofed.utils import Logger, get_current_time, log_exec

if TYPE_CHECKING:
    from nanofed.communication import HTTPServer
else:
    HTTPServer = "HTTPServer"


@dataclass(slots=True, frozen=True)
class CoordinatorConfig:
    """Coordinator configuration.

    Parameters
    ----------
    num_rounds : int
        Number of federated training rounds to execute.
    min_clients : int
        Minimum number of clients required for training.
    min_completion_rate : float
        Minimum fraction of clients that must complete each round
    round_timeout : int
        Maximum time in seconds to wait for client updates per round.
    base_dir : Path
        Base directory for all federation data, including models, metrics,
        and data.
    """

    num_rounds: int
    min_clients: int
    min_completion_rate: float
    round_timeout: int
    base_dir: Path


class Coordinator:
    """Coordinated federated training across clients.

    Manages training rounds, client synchronization, model aggregation,
    and training progress tracking.

    Parameters
    ----------
    model_manager : ModelManagerProtocol
        Manager for model versioning and storag.
    aggregator : BaseAggregator
        Strategy for aggregating client updates.
    server : HTTPServer
        Server instance for client communication.
    config : CoordinatorConfig
        Coordinator configuration.

    Attributes
    ----------
    training_progress : TrainingProgress
        Current training progress information.
    _current_round : int
        Current training round number.
    _status : RoundStatus
        Current training status.

    Methods
    -------
    start_training(progress_callback=None)
        Start federated training process.
    train_round()
        Execute one training round.

    Examples
    --------
    >>> model = PyTorchModule()
    >>> model_manager = ModelManager(model)
    >>> coordinator = Coordinator(model_manager, aggregator, server, config)
    >>> async for metrics in coordinator.start_training():
    ...     print(f"Round {metrics.round_id} completed")
    """

    def __init__(
        self,
        model_manager: ModelManagerProtocol,
        aggregator: BaseAggregator,
        server: HTTPServer,
        config: CoordinatorConfig,
    ) -> None:
        self._model_manager = model_manager
        self._aggregator = aggregator
        self._server = server
        self._config = config
        self._logger = Logger()

        # State tracking
        self._current_round: int = 0
        self._clients: dict[str, ClientInfo] = {}
        self._round_metrics: list[RoundMetrics] = []
        self._status = RoundStatus.INITIALIZED
        self._round_lock = asyncio.Lock()

        # Create directories
        self._metrics_dir = self._config.base_dir / "metrics"
        self._data_dir = self._config.base_dir / "data"
        self._models_dir = self._config.base_dir / "models"
        self._model_configs_dir = self._models_dir / "configs"
        self._model_weights_dir = self._models_dir / "models"

        self._setup_directories()

        self._model_manager.set_dirs(
            self._model_weights_dir, self._model_configs_dir
        )
        self._server.set_coordinator(self)

    @property
    def server(self) -> HTTPServer:
        """Get the HTTP server instance.

        Returns
        -------
        HTTPServer
            The HTTP server instance.
        """
        return self._server

    @property
    def data_dir(self) -> Path:
        """Get the data directory path.

        Returns
        -------
        Path
            The data directory path.
        """
        return self._data_dir

    @property
    def model_manager(self) -> ModelManagerProtocol:
        """Get the model manager instance.

        Returns
        -------
        ModelManager
            The model manager instance.
        """
        return self._model_manager

    def _setup_directories(self) -> None:
        """Create all required directories under the base directory."""
        with self._logger.context("coordinator.setup"):
            dirs = [
                self._metrics_dir,
                self._data_dir,
                self._model_configs_dir,
                self._model_weights_dir,
            ]

            for directory in dirs:
                try:
                    directory.mkdir(parents=True, exist_ok=True)
                    self._logger.info(f"Created directory: {directory}")
                except Exception as e:
                    self._logger.error(
                        f"Failed to create directory {directory}: {str(e)}"
                    )
                    raise

    @property
    def training_progress(self) -> TrainingProgress:
        """Get current training progress."""
        return {
            "current_round": self._current_round,
            "total_rounds": self._config.num_rounds,
            "active_clients": len(self._clients),
            "global_metrics": self._get_global_metrics(),
            "status": self._status.name,
        }

    def _get_global_metrics(self) -> dict[str, float]:
        if not self._round_metrics:
            return {}

        metrics: dict[str, list[float]] = {}
        for round_metric in self._round_metrics:
            for key, value in round_metric.agg_metrics.items():
                metrics.setdefault(key, []).append(value)

        return {
            key: sum(values) / len(values) for key, values in metrics.items()
        }

    async def _wait_for_clients(self, timeout: int) -> bool:
        """Wait for minimum number of clients to complete round."""
        with self._logger.context("coordinator"):
            start_time = get_current_time()
            required_clients = int(
                self._config.min_clients * self._config.min_completion_rate
            )

            last_logged_client_count = -1
            log_interval = 10

            while (get_current_time() - start_time).total_seconds() < timeout:
                completed_clients = len(self._server._updates)
                elapsed_time = (
                    get_current_time() - start_time
                ).total_seconds()

                if (
                    completed_clients != last_logged_client_count
                    or elapsed_time % log_interval == 0
                ):
                    last_logged_client_count = completed_clients
                    self._logger.info(
                        f"Client training progress: {completed_clients}/{self._config.min_clients} "  # noqa
                        f"(need {required_clients})"
                    )

                if completed_clients >= required_clients:
                    self._logger.info(
                        f"Sufficient clients completed training: {completed_clients}/{self._config.min_clients}"  # noqa
                    )
                    return True

                await asyncio.sleep(1)

            self._logger.error(
                f"Timeout waiting for clients. "
                f"Got {len(self._server._updates)}/{self._config.min_clients} "
                f"(needed {required_clients})"
            )
            return False

    def _save_metrics(
        self, metrics: RoundMetrics, client_metrics: list[dict]
    ) -> None:
        with self._logger.context(
            "coordinator.metrics", f"round_{metrics.round_id}"
        ):
            metrics_file = (
                self._metrics_dir / f"metrics_round_{metrics.round_id}.json"
            )

            metrics_data = {
                "round_id": metrics.round_id,
                "start_time": metrics.start_time.isoformat()
                if metrics.start_time
                else None,
                "end_time": metrics.end_time.isoformat()
                if metrics.end_time
                else None,
                "num_clients": metrics.num_clients,
                "agg_metrics": metrics.agg_metrics,
                "status": metrics.status.name,
                "client_metrics": client_metrics,
            }

            try:
                with metrics_file.open("w") as f:
                    json.dump(metrics_data, f, indent=4)
                self._logger.info(
                    f"Saved metrics for round {metrics.round_id} to {metrics_file}"  # noqa
                )
            except Exception as e:
                self._logger.error(
                    f"Failed to save metrics for round {metrics.round_id}: {str(e)}"  # noqa
                )

    @log_exec
    async def train_round(self) -> RoundMetrics:
        """Execute one training round."""
        with self._logger.context(
            "coordinator", f"round_{self._current_round}"
        ):
            async with self._round_lock:
                try:
                    self._status = RoundStatus.IN_PROGRESS
                    start_time = get_current_time()

                    self._server._updates.clear()

                    if not await self._wait_for_clients(
                        self._config.round_timeout
                    ):
                        self._logger.warning(
                            f"Skipping round {self._current_round} due to timeout."  # noqa
                        )
                        self._status = RoundStatus.FAILED
                        raise TimeoutError(
                            f"Round {self._current_round} timed out waiting for clients"  # noqa
                        )

                    self._status = RoundStatus.AGGREGATING
                    client_updates: Sequence[ModelUpdate] = [
                        ModelUpdate(
                            client_id=update["client_id"],
                            round_number=update["round_number"],
                            model_state={
                                key: torch.tensor(value)
                                for key, value in update["model_state"].items()
                            },
                            metrics=update["metrics"],
                            timestamp=datetime.fromisoformat(
                                update["timestamp"]
                            ),
                            privacy_spent=update["privacy_spent"],
                        )
                        for update in self._server._updates.values()
                    ]

                    weights = self._aggregator._compute_weights(client_updates)

                    client_weights = {
                        update["client_id"]: weight
                        for update, weight in zip(client_updates, weights)
                    }

                    client_metrics = [
                        {
                            "client_id": update.get("client_id"),
                            "metrics": update.get("metrics", {}),
                            "weight": client_weights[
                                str(update.get("client_id", ""))
                            ],
                        }
                        for update in client_updates
                    ]

                    aggregation_result = self._aggregator.aggregate(
                        self._model_manager.model, client_updates
                    )

                    config = {
                        "round_id": self._current_round,
                        "client_metrics": client_metrics,
                        "client_weights": client_weights,
                        "start_time": start_time.isoformat(),
                        "status": self._status.name,
                        "num_clients": len(client_updates),
                    }

                    self._model_manager.save_model(
                        config=config, metrics=aggregation_result.metrics
                    )

                    self._current_round += 1
                    self._status = RoundStatus.COMPLETED

                    metrics = RoundMetrics(
                        round_id=self._current_round - 1,
                        start_time=start_time,
                        end_time=get_current_time(),
                        num_clients=len(client_updates),
                        agg_metrics=aggregation_result.metrics,
                        status=self._status,
                    )

                    self._round_metrics.append(metrics)
                    self._save_metrics(metrics, client_metrics)
                    self._server._updates.clear()

                    return metrics

                except Exception as e:
                    self._status = RoundStatus.FAILED
                    self._logger.error(
                        f"Error in round {self._current_round}: {str(e)}"
                    )
                    raise

    async def start_training(
        self,
        progress_callback: Callable[[TrainingProgress], None] | None = None,
    ) -> AsyncGenerator[RoundMetrics, None]:
        with self._logger.context("coordinator"):
            try:
                for _ in range(self._config.num_rounds):
                    self._server._updates.clear()

                    metrics = await self.train_round()
                    if progress_callback:
                        progress_callback(self.training_progress)

                    yield metrics

                await self._server.stop_training()

            except Exception as e:
                self._logger.error(f"Training failed: {str(e)}")
                raise
            finally:
                self._logger.info("Training completed")
