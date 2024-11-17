from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Generic, Protocol, TypeVar

import torch
from torch.utils.data import DataLoader

from nanofed.core.interfaces import ModelProtocol
from nanofed.utils.logger import Logger, log_exec

M = TypeVar("M", bound=ModelProtocol)


@dataclass(slots=True, frozen=True)
class TrainingConfig:
    """Training configuration."""

    epochs: int
    batch_size: int
    learning_rate: float
    device: str = "cpu"
    max_batches: int | None = None
    log_interval: int = 10


@dataclass(slots=True)
class TrainingMetrics:
    """Training metrics."""

    loss: float
    accuracy: float
    epoch: int
    batch: int
    samples_processed: int


class TrainingCallback(Protocol):
    """Protocol for training callbacks."""

    def on_eopch_start(self, epoch: int) -> None: ...
    def on_epoch_end(self, epoch: int, metrics: TrainingMetrics) -> None: ...
    def on_batch_end(self, batch: int, metrics: TrainingMetrics) -> None: ...


class BaseTrainer(ABC, Generic[M]):
    """Base trainer class."""

    def __init__(
        self,
        config: TrainingConfig,
        callbacks: list[TrainingCallback] | None = None,
    ) -> None:
        self._config = config
        self._callbacks = callbacks or []
        self._logger = Logger()
        self._device = torch.device(config.device)

    @abstractmethod
    def compute_loss(
        self, output: torch.Tensor, target: torch.Tensor
    ) -> torch.Tensor:
        """Compute loss for current batch."""
        pass

    @abstractmethod
    def compute_accuracy(
        self, output: torch.Tensor, target: torch.Tensor
    ) -> float:
        """Compute accuracy for current batch."""
        pass

    @log_exec
    def train_epoch(
        self,
        model: ModelProtocol,
        dataloader: DataLoader,
        optimizer: torch.optim.Optimizer,
        epoch: int,
    ) -> TrainingMetrics:
        """Train for one epoch."""
        model.train()
        total_loss = 0.0
        total_accuracy = 0.0
        samples_processed = 0

        for callback in self._callbacks:
            callback.on_eopch_start(epoch)

        for batch_idx, (data, target) in enumerate(dataloader):
            if (
                self._config.max_batches is not None
                and batch_idx >= self._config.max_batches
            ):
                break

            data, target = data.to(self._device), target.to(self._device)
            optimizer.zero_grad()

            # Forward pass
            output = model(data)
            loss = self.compute_loss(output, target)

            # Backward pass
            loss.backward()
            optimizer.step()

            # Metrics
            accuracy = self.compute_accuracy(output, target)
            total_loss += loss.item()
            total_accuracy += accuracy
            samples_processed += len(data)

            metrics = TrainingMetrics(
                loss=loss.item(),
                accuracy=accuracy,
                epoch=epoch,
                batch=batch_idx,
                samples_processed=samples_processed,
            )

            for callback in self._callbacks:
                callback.on_batch_end(batch_idx, metrics)

            if batch_idx % self._config.log_interval == 0:
                self._logger.info(
                    f"Train Epoch: {epoch} "
                    f"[{samples_processed}/{len(dataloader.dataset)} "
                    f"({100. * samples_processed / len(dataloader.dataset):.0f}%)] "  # noqa
                    f"Loss: {loss.item():.6f} "
                    f"Accuracy: {accuracy:.4f}"
                )

        avg_loss = total_loss / (batch_idx + 1)
        avg_accuracy = total_accuracy / (batch_idx + 1)

        metrics = TrainingMetrics(
            loss=avg_loss,
            accuracy=avg_accuracy,
            epoch=epoch,
            batch=batch_idx,
            samples_processed=samples_processed,
        )

        for callback in self._callbacks:
            callback.on_epoch_end(epoch.metrics)

        return metrics
