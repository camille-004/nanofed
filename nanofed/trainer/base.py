from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Generic, Protocol, Sized, TypeVar, cast

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from nanofed.core import ModelProtocol
from nanofed.utils import Logger, log_exec

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

    def to_dict(self) -> dict[str, float | int]:
        """Convert TrainingMetrics to a dictionary"""
        return {
            "loss": self.loss,
            "accuracy": self.accuracy,
            "samples_processed": self.samples_processed,
        }


class Callback(Protocol):
    """Protocol for training callbacks."""

    def on_eopch_start(self, epoch: int) -> None: ...
    def on_epoch_end(self, epoch: int, metrics: TrainingMetrics) -> None: ...
    def on_batch_end(self, batch: int, metrics: TrainingMetrics) -> None: ...


class BaseTrainer(ABC, Generic[M]):
    """Base class for model training implementations.

    Provides abstract interface and common functionality for training
    machine learning models with customizable metrics and callbacks.

    Parameters
    ----------
    config : TrainingConfig
        Training configuration.
    callbacks : list[Callback], optional
        List of training callbacks.

    Attributes
    ----------
    _device : torch.device
        Training device (CPU/GPU).
    _config : TrainingConfig
        Training configuration.
    _callbacks : list[Callback]
        List of training callbacks.

    Methods
    -------
    train_epoch(model, dataloader, optimizer, epoch)
        Train model for one epoch.
    compute_loss(output, target)
        Compute loss for current batch.
    compute_accuracy(output, target)
        Compute accuracy for current batch.

    Examples
    --------
    >>> trainer = TorchTrainer(config)
    >>> metrics = trainer.train_epoch(model, dataloader, optimizer, epoch)
    """

    def __init__(
        self,
        config: TrainingConfig,
        callbacks: list[Callback] | None = None,
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
        model: M,
        dataloader: DataLoader,
        optimizer: torch.optim.Optimizer,
        epoch: int,
    ) -> TrainingMetrics:
        """Train for one epoch."""
        model_module = cast(nn.Module, model)
        model_module.train()

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
            output = model_module(data)
            loss = self.compute_loss(output, target)

            # Backward pass
            loss.backward()
            optimizer.step()

            # Metrics
            accuracy = self.compute_accuracy(output, target)
            total_loss += float(loss.item())
            total_accuracy += float(accuracy)
            samples_processed += int(len(data))

            metrics = TrainingMetrics(
                loss=float(loss.item()),
                accuracy=float(accuracy),
                epoch=epoch,
                batch=batch_idx,
                samples_processed=samples_processed,
            )

            for callback in self._callbacks:
                callback.on_batch_end(batch_idx, metrics)

            if batch_idx % self._config.log_interval == 0:
                dataset = cast(Sized, dataloader.dataset)
                total_samples = len(dataset)
                progress = (
                    100.0 * float(samples_processed) / float(total_samples)
                )  # noqa
                self._logger.info(
                    f"Train Epoch: {epoch} "
                    f"[{samples_processed}/{total_samples} "
                    f"({progress:.0f}%)] "
                    f"Loss: {loss.item():.6f} "
                    f"Accuracy: {accuracy:.4f}"
                )

        batch_count = batch_idx + 1
        avg_loss = total_loss / float(batch_count)
        avg_accuracy = total_accuracy / float(batch_count)

        final_metrics = TrainingMetrics(
            loss=avg_loss,
            accuracy=avg_accuracy,
            epoch=epoch,
            batch=batch_idx,
            samples_processed=samples_processed,
        )

        for callback in self._callbacks:
            callback.on_epoch_end(epoch, final_metrics)

        return metrics
