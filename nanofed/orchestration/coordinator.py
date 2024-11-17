import asyncio
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import AsyncGenerator, Callable

import torch

from nanofed.communication.http.server import HTTPServer
from nanofed.core.interfaces import ModelProtocol
from nanofed.orchestration.types import (
    ClientInfo,
    RoundMetrics,
    RoundStatus,
    TrainingProgess,
)
from nanofed.server.aggregator.base import BaseAggregator
from nanofed.utils.logger import Logger, log_exec


@dataclass(slots=True, frozen=True)
class CoordinatorConfig:
    """Coordinator configuration."""

    num_rounds: int
    min_clients: int
    min_completion_rate: float
    round_timeout: int
    checkpoint_dir: Path
    metrics_dir: Path


class Coordinator:
    """Coordinated federated learning training."""

    def __init__(
        self,
        model: ModelProtocol,
        aggregator: BaseAggregator,
        server: HTTPServer,
        config: CoordinatorConfig,
    ) -> None:
        self._model = model
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
        self._config.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self._config.metrics_dir.mkdir(parents=True, exist_ok=True)

    @property
    def training_progress(self) -> TrainingProgess:
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
        with self._logger.context(
            "coordinator", f"round_{self._current_round}"
        ):
            start_time = datetime.now()
            required_clients = int(
                self._config.min_clients * self._config.min_completion_rate
            )

            last_logged_client_count = -1
            log_interval = 10

            while (datetime.now() - start_time).total_seconds() < timeout:
                completed_clients = len(self._server._updates)
                elapsed_time = (datetime.now() - start_time).total_seconds()

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
            metrics_dir = self._config.metrics_dir
            metrics_file = (
                metrics_dir / f"metrics_round_{metrics.round_id}.json"
            )

            metrics_data = {
                "round_id": metrics.round_id,
                "start_time": metrics.start_time.isoformat(),
                "end_time": metrics.end_time.isoformat(),
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
                    start_time = datetime.now()

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
                    client_updates = list(self._server._updates.values())

                    client_metrics = [
                        {
                            "client_id": update.get("client_id"),
                            "loss": update.get("metrics", {}).get("loss"),
                            "accuracy": update.get("metrics", {}).get(
                                "accuracy"
                            ),
                            "samples_processed": update.get("metrics", {}).get(
                                "samples_processed"
                            ),
                        }
                        for update in client_updates
                    ]

                    aggregation_result = self._aggregator.aggregate(
                        self._model, client_updates
                    )

                    checkpoint_path = (
                        self._config.checkpoint_dir
                        / f"round_{self._current_round}.pt"
                    )
                    torch.save(
                        {
                            "model_state": self._model.state_dict(),
                            "round": self._current_round,
                            "metrics": aggregation_result.metrics,
                        },
                        checkpoint_path,
                    )

                    self._current_round += 1
                    self._status = RoundStatus.COMPLETED

                    metrics = RoundMetrics(
                        round_id=self._current_round - 1,
                        start_time=start_time,
                        end_time=datetime.now(),
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
        progress_callback: Callable[[TrainingProgess], None] | None = None,
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
