from pathlib import Path
from typing import Any, Iterator, Protocol, TypeVar

import torch
from torch import nn
from torch.utils.data import DataLoader

from .types import ModelVersion

T = TypeVar("T", bound=nn.Module)


class ModelProtocol(Protocol):
    """Protocol defining required model interface."""

    def forward(self, x: torch.Tensor) -> torch.Tensor: ...
    def parameters(self) -> Iterator[torch.nn.Parameter]: ...
    def state_dict(self) -> dict[str, torch.Tensor]: ...
    def load_state_dict(self, state_dict: dict[str, torch.Tensor]) -> None: ...
    def to(self, device: str | torch.device) -> "ModelProtocol": ...


class AggregatorProtoocol(Protocol[T]):
    """Protocol for model update aggregation strategies."""

    def aggregate(self, updates: list[T]) -> T: ...


class TrainerProtocol(Protocol[T]):
    """Protocol for model training implementations."""

    def train(self, model: T, data: DataLoader) -> T: ...
    def validate(self, model: T, data: DataLoader) -> dict[str, float]: ...


class ModelManagerProtocol(Protocol):
    """Protocol defining required model manager interface."""

    def set_dirs(self, models_dir: Path, configs_dir: Path) -> None: ...
    @property
    def current_version(self) -> Any: ...
    def load_model(self) -> Any: ...
    def save_model(
        self, config: dict[str, Any], metrics: dict[str, float] | None
    ) -> Any: ...
    @property
    def list_versions(self) -> list[ModelVersion]: ...
    @property
    def model(self) -> ModelProtocol: ...


class CoordinatorProtocol(Protocol):
    """Protocol defining required coordinator interface."""

    @property
    def model_manager(self) -> ModelManagerProtocol: ...


class ServerProtocol(Protocol):
    """Protocol defining required server interface."""

    @property
    def host(self) -> str: ...
    @property
    def port(self) -> int: ...
    @property
    def url(self) -> str: ...
