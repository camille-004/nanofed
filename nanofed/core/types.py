from datetime import datetime
from typing import Any, Literal, TypedDict

import torch


class ModelConfig(TypedDict):
    """Type definition for model configuration."""

    name: str
    version: str
    architecture: dict[str, Any]


class TrainingConfig(TypedDict):
    """Type definition for training configuration."""

    epochs: int
    batch_size: int
    learning_rate: float
    optimizer: Literal["adam", "sgd"]
    device: Literal["cpu", "cuda"]


class ModelUpdate(TypedDict):
    """Type definition for model updates."""

    model_state: dict[str, torch.Tensor]
    client_id: str
    round_number: int
    metrics: dict[str, float]
    timestamp: datetime
