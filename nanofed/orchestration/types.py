from dataclasses import dataclass
from datetime import datetime
from enum import Enum, auto
from typing import TypedDict


@dataclass(slots=True, frozen=True)
class ClientInfo:
    """Client information."""

    client_id: str
    status: str
    last_update: datetime
    metrics: dict[str, float]


class RoundStatus(Enum):
    """Training round status."""

    INITIALIZED = auto()
    IN_PROGRESS = auto()
    AGGREGATING = auto()
    COMPLETED = auto()
    FAILED = auto()


@dataclass(slots=True, frozen=True)
class RoundMetrics:
    """Metrics for a training round."""

    round_id: int
    start_time: datetime
    end_time: datetime | None
    num_clients: int
    agg_metrics: dict[str, float]
    status: RoundStatus


class TrainingProgress(TypedDict):
    """Training progress information."""

    current_round: int
    total_rounds: int
    active_clients: int
    global_metrics: dict[str, float]
    status: str
