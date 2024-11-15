import numpy as np

from fl.aggregator.base import BaseAggregator
from fl.core.exceptions import AggregationError
from fl.core.protocols import ModelUpdate, ModelWeights


class FedProxAggregator(BaseAggregator):
    """Federate Proximal aggregator implementation."""

    def __init__(self, mu: float = 0.1) -> None:
        """Initialize FedProxAggregator.

        Parameters
        ----------
        mu : float, optional
            Proximal term coefficient, by default 0.1
        """
        self.mu = mu

    def aggregate(self, updates: list[ModelUpdate]) -> ModelWeights:
        """Agregate client updates with FedProx proximal term."""
        if not updates:
            raise AggregationError("No model updates provided for aggregation")

        weights_agg: ModelWeights = {}
        first_update = updates[0].weights

        for key in first_update:
            weights_agg[key] = np.zeros_like(first_update[key])

        for update in updates:
            for key, value in update.weights.items():
                if key not in weights_agg:
                    raise AggregationError(
                        f"Inconsistent weight keys: {key} not found."
                    )
                weights_agg[key] += value + self.mu * value

        num_updates = len(updates)
        for key in weights_agg:
            weights_agg[key] /= num_updates

        return weights_agg
