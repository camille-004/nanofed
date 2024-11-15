from abc import ABC

import numpy as np

from fl.core.exceptions import ValidationError
from fl.core.protocols import (
    ModelAggregator as ModelAggregatorProtocol,
)
from fl.core.protocols import (
    ModelUpdate,
)


class BaseAggregator(ModelAggregatorProtocol, ABC):
    """Abstract base class for model aggregators."""

    def validate_update(self, update: ModelUpdate) -> bool:
        """Default validation: Make sure weights are valid."""
        if not isinstance(update.weights, dict):
            raise ValidationError("Weights must be a dictionary.")
        for key, value in update.weights.items():
            if not isinstance(value, np.ndarray):
                raise ValidationError(
                    f"Weights for {key} must be numpy arrays"
                )
            if np.any(np.isnan(value)) or np.any(np.isinf(value)):
                raise ValidationError(
                    f"Weights for {key} contain invalid values."
                )
        return True
