import json
from dataclasses import dataclass
from datetime import datetime
from enum import Enum, auto
from pathlib import Path
from typing import Any, Protocol

import torch

from nanofed.core import ModelUpdate
from nanofed.utils import Logger, get_current_time


class RoundState(Enum):
    """Training round state."""

    INITIALIZED = auto()
    IN_PROGRESS = auto()
    FAILED = auto()
    COMPLETED = auto()


@dataclass(slots=True, frozen=True)
class CheckpointMetadata:
    """Metadata for checkpointed state."""

    round_id: int
    timestamp: datetime
    num_clients: int
    client_updates: dict[str, ModelUpdate]
    global_model_version: str
    state: RoundState

    def to_dict(self) -> dict[str, Any]:
        return {
            "round_id": self.round_id,
            "timestamp": self.timestamp.isoformat(),
            "num_clients": self.num_clients,
            "client_updates": self.client_updates,
            "global_model_version": self.global_model_version,
            "state": self.state.name,
        }

    @staticmethod
    def from_dict(data: dict[str, Any]) -> "CheckpointMetadata":
        for _, update in data["client_updates"].items():
            for key, value in update["model_state"].items():
                update["model_state"][key] = torch.tensor(value)
        return CheckpointMetadata(
            round_id=data["round_id"],
            timestamp=datetime.fromisoformat(data["timestamp"]),
            num_clients=data["num_clients"],
            client_updates=data["client_updates"],
            global_model_version=data["global_model_version"],
            state=RoundState[data["state"]],
        )


class StateStore(Protocol):
    """Protocol for state persistence."""

    def save_checkpoint(
        self, metadata: CheckpointMetadata, state: dict[str, Any]
    ) -> None: ...

    def load_checkpoint(
        self, round_id: int
    ) -> tuple[CheckpointMetadata, dict[str, Any]] | None: ...

    def list_checkpoints(self) -> list[CheckpointMetadata]: ...


class RecoveryStrategy(Protocol):
    """Protocol for recovery strategies."""

    def should_recover(self, failure: Exception) -> bool: ...

    def get_recovery_point(
        self, checkpoints: list[CheckpointMetadata]
    ) -> CheckpointMetadata | None: ...


class FileStateStore(StateStore):
    """File-based state persistence."""

    def __init__(self, base_dir: Path) -> None:
        self._base_dir = base_dir / "checkpoints"
        self._base_dir.mkdir(parents=True, exist_ok=True)
        self._logger = Logger()

    def save_checkpoint(
        self, metadata: CheckpointMetadata, state: dict[str, Any]
    ) -> None:
        checkpoint_dir = self._base_dir / f"round_{metadata.round_id}"
        checkpoint_dir.mkdir(exist_ok=True)

        metadata_path = checkpoint_dir / "metadata.json"
        with open(metadata_path, "w") as f:
            json.dump(metadata.to_dict(), f)

        state_path = checkpoint_dir / "state.pt"
        torch.save(state, state_path)

        self._logger.info(f"Saved checkpoint for round {metadata.round_id}")

    def load_checkpoint(
        self, round_id: int
    ) -> tuple[CheckpointMetadata, dict[str, Any]] | None:
        """Load checkpoint from disk."""
        checkpoint_dir = self._base_dir / f"round_{round_id}"
        if not checkpoint_dir.exists():
            return None

        metadata_path = checkpoint_dir / "metadata.json"
        with open(metadata_path) as f:
            metadata_dict = json.load(f)
            metadata = CheckpointMetadata.from_dict(metadata_dict)

        state_path = checkpoint_dir / "state.pt"
        state = torch.load(state_path, weights_only=True)

        self._logger.info(f"Loaded checkpoint for round {round_id}")
        return metadata, state

    def list_checkpoints(self) -> list[CheckpointMetadata]:
        """List all available checkpoints."""
        checkpoints = []
        for path in sorted(self._base_dir.glob("round_*")):
            metadata_path = path / "metadata.json"
            if metadata_path.exists():
                with open(metadata_path) as f:
                    metadata_dict = json.load(f)
                    checkpoints.append(
                        CheckpointMetadata.from_dict(metadata_dict)
                    )
        return checkpoints


class SimpleRecoveryStrategy(RecoveryStrategy):
    """Simple recovery strategy using latest good checkpoint."""

    def should_recover(self, failure: Exception) -> bool:
        recoverable_errors = (TimeoutError, ConnectionError, RuntimeError)
        return isinstance(failure, recoverable_errors)

    def get_recovery_point(
        self, checkpoints: list[CheckpointMetadata]
    ) -> CheckpointMetadata | None:
        completed = [
            cp for cp in checkpoints if cp.state == RoundState.COMPLETED
        ]
        return max(completed, key=lambda x: x.round_id) if completed else None


class FaultTolerantCoordinator:
    """Fault-tolerance federated coordinator."""

    def __init__(
        self,
        base_dir: Path,
        state_store: StateStore | None = None,
        recovery_strategy: RecoveryStrategy | None = None,
    ) -> None:
        self._state_store = state_store or FileStateStore(base_dir)
        self._recovery = recovery_strategy or SimpleRecoveryStrategy()
        self._logger = Logger()

    def checkpoint_round(
        self,
        round_id: int,
        client_updates: dict[str, ModelUpdate],
        model_version: str,
        state: dict[str, Any],
        round_state: RoundState,
    ) -> None:
        """Checkpoint current round state."""
        metadata = CheckpointMetadata(
            round_id=round_id,
            timestamp=get_current_time(),
            num_clients=len(client_updates),
            client_updates=client_updates,
            global_model_version=model_version,
            state=round_state,
        )

        self._state_store.save_checkpoint(metadata, state)

    def restore_round(
        self, round_id: int
    ) -> tuple[CheckpointMetadata, dict[str, Any]] | None:
        """Restore round from checkpoint."""
        return self._state_store.load_checkpoint(round_id)

    def handle_failure(
        self, failure: Exception, current_round: int
    ) -> tuple[CheckpointMetadata, dict[str, Any]] | None:
        """Handle training failure."""
        if not self._recovery.should_recover(failure):
            self._logger.error(
                f"Unrecoverable failure in round {current_round}: {str(failure)}"  # noqa
            )
            return None

        checkpoints = self._state_store.list_checkpoints()
        recovery_point = self._recovery.get_recovery_point(checkpoints)

        if recovery_point is None:
            self._logger.error("No valid recovery point found")
            return None

        self._logger.info(f"Recovering from round {recovery_point.round_id}")
        return self.restore_round(recovery_point.round_id)
