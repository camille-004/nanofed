from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from functools import lru_cache
from typing import Generic, Sequence, TypeVar

from nanofed.core import AggregationError, ModelProtocol, ModelUpdate
from nanofed.utils import Logger

T = TypeVar("T", bound=ModelProtocol)


@dataclass(slots=True, frozen=True)
class AggregationResult(Generic[T]):
    """Results from model aggregation."""

    model: T
    round_number: int
    num_clients: int
    timestamp: datetime
    metrics: dict[str, float]


class BaseAggregator(ABC, Generic[T]):
    """Base class for aggregation strategies."""

    def __init__(self) -> None:
        self._logger = Logger()
        self._current_round: int = 0

    @property
    def current_round(self) -> int:
        return self._current_round

    @lru_cache(maxsize=128)
    def _compute_weights(self, num_clients: int) -> list[float]:
        """Compute aggregation weights for clients."""
        return [1.0 / num_clients] * num_clients

    def _validate_updates(self, updates: Sequence[ModelUpdate]) -> None:
        """Validate model updates before aggregation."""
        if not updates:
            raise AggregationError("No updates provided for aggregation")

        # Check all updates are from the same round
        rounds = {update["round_number"] for update in updates}
        if len(rounds) != 1:
            raise AggregationError(f"Updates from different rounds: {rounds}")

        # Check model architectures match
        first_state = updates[0]["model_state"]
        for update in updates[1:]:
            if update["model_state"].keys() != first_state.keys():
                raise AggregationError(
                    "Inconsistent model architectures in updates."
                )

    @abstractmethod
    def aggregate(
        self, model: T, updates: Sequence[ModelUpdate]
    ) -> AggregationResult[T]:
        """Aggregate model updates."""
        pass
