import numpy as np

from fl.aggregator.base import BaseAggregator
from fl.core.exceptions import AggregationError
from fl.core.protocols import ModelUpdate, ModelWeights


class PersonalizedFedAvgAggregator(BaseAggregator):
    """Personalized Federated Average aggregator implementation."""

    def __init__(self, alpha: float = 0.5) -> None:
        """Initialize PersonalizedFedAvgAggregator.

        Parameters
        ----------
        alpha : float, optional
            Weighting factor between global and local models, by default 0.5
        """
        self.alpha = alpha

    def aggregate(self, updates: list[ModelUpdate]) -> ModelWeights:
        """Aggregate client updates by blending global and personalized models.

        Parameters
        ----------
        updates : list[ModelUpdate]
            List of client model updates.

        Returns
        -------
        ModelWeights
            Aggregated global model weights.
        """
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
                        f"Inconsistent weight keys: {key} not found"
                    )
                weights_agg[key] += self.alpha * value

        num_updates = len(updates)
        for key in weights_agg:
            weights_agg[key] /= num_updates

        return weights_agg
