from typing import Iterator, Protocol, TypeVar

import torch
from torch import nn
from torch.utils.data import DataLoader

T = TypeVar("T", bound=nn.Module)


class ModelProtocol(Protocol):
    """Protocol defining required model interface."""

    def forwrd(self, x: torch.Tensor) -> torch.Tensor: ...
    def parameters(self) -> Iterator[torch.nn.Parameter]: ...
    def state_dict(self) -> dict[str, torch.Tensor]: ...
    def load_state_dict(self, state_dict: dict[str, torch.Tensor]) -> None: ...


class AggregatorProtoocol(Protocol[T]):
    """Protocol for model update aggregation strategies."""

    def aggregate(self, updates: list[T]) -> T: ...


class TrainerProtocol(Protocol[T]):
    """Protocol for model training implementations."""

    def train(self, model: T, data: DataLoader) -> T: ...
    def validate(self, model: T, data: DataLoader) -> dict[str, float]: ...
