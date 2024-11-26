from datetime import datetime
from typing import TypedDict

import torch


class ModelUpdate(TypedDict):
    """Type definition for model updates."""

    model_state: dict[str, torch.Tensor]
    client_id: str
    round_number: int
    metrics: dict[str, float]
    timestamp: datetime
