from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Protocol, TypeAlias, runtime_checkable

import numpy as np

ModelWeights: TypeAlias = dict[str, np.ndarray]
ClientID: TypeAlias = str
ModelMetrics: TypeAlias = dict[str, float]


@dataclass(frozen=True, slots=True)
class ModelUpdate:
    """Immutable container for model updates from clients."""

    client_id: ClientID
    weights: ModelWeights
    round_metrics: dict[str, float]
    round_number: int


@runtime_checkable
class ModelAggregator(Protocol):
    """Protocol defining how model updates should be aggregated."""

    def aggregate(self, updates: list[ModelUpdate]) -> ModelWeights: ...
    def validate_update(self, update: ModelUpdate) -> bool: ...


class SecurityProtocol(ABC):
    """Abstract base class for security protocols."""

    @abstractmethod
    def encrypt(self, data: ModelWeights) -> bytes: ...

    @abstractmethod
    def decrypt(self, data: bytes) -> ModelWeights: ...
