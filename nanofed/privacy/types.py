from typing import Literal, TypeAlias

import torch

PrivacyBudget: TypeAlias = dict[Literal["epsilon", "delta"], float]
Shape: TypeAlias = tuple[int, ...]
Tensor: TypeAlias = torch.Tensor
NoiseScale: TypeAlias = float | dict[str, float]
