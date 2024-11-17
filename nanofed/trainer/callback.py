import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from nanofed.trainer import Callback, TrainingMetrics


@dataclass(slots=True)
class MetricsLogger(Callback):
    """Callback for logging metrics to a file."""

    log_dir: Path
    experiment_name: str

    def __post_init__(self) -> None:
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self._log_file = (
            self.log_dir
            / f"{self.experiment_name}_{datetime.now():%Y%m%d_%H%M%S}.json"
        )
        self._metrics: list[dict] = []

    def on_eopch_start(self, epoch: int) -> None:
        pass

    def on_epoch_end(self, epoch: int, metrics: TrainingMetrics) -> None:
        """Log metrics at end of epoch."""
        metrics_dict = {
            "type": "epoch",
            "epoch": epoch,
            "loss": metrics.loss,
            "accuracy": metrics.accuracy,
            "samples_processed": metrics.samples_processed,
            "timestamp": datetime.now().isoformat(),
        }
        self._metrics.append(metrics_dict)

        with open(self._log_file, "w") as f:
            json.dump(self._metrics, f, indent=2)

    def on_batch_end(self, batch: int, metrics: TrainingMetrics) -> None:
        """Log metrics at end of batch."""
        metrics_dict = {
            "type": "batch",
            "epoch": metrics.epoch,
            "batch": batch,
            "loss": metrics.loss,
            "accuracy": metrics.accuracy,
            "samples_processed": metrics.samples_processed,
            "timestamp": datetime.now().isoformat(),
        }
        self._metrics.append(metrics_dict)
