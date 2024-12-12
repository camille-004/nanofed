from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, TypedDict

import torch


class ModelUpdate(TypedDict):
    """Type definition for model updates."""

    model_state: dict[str, torch.Tensor]
    client_id: str
    round_number: int
    metrics: dict[str, float]
    timestamp: datetime


@dataclass(slots=True, frozen=True)
class ModelVersion:
    """Model version information."""

    version_id: str
    timestamp: datetime
    config: dict[str, Any]
    path: Path
