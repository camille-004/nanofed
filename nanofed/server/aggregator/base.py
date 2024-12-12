from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from typing import Generic, Sequence, TypeVar

from nanofed.core.exceptions import AggregationError
from nanofed.core.interfaces import ModelProtocol
from nanofed.core.types import ModelUpdate
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
        self._weights_cache: dict[int, list[float]] = {}

    @property
    def current_round(self) -> int:
        return self._current_round

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

    @abstractmethod
    def _compute_weights(self, updates: Sequence[ModelUpdate]) -> list[float]:
        """Compute aggregation weights for clients.

        Each aggregation strategy should implement its own weighting scheme.

        Parameters
        ----------
        updates : Sequence[ModelUpdate]
            Sequence of model updates from clients.

        Returns
        -------
        list[float]
            List of weights, one per client update.
        """
        pass
