import numpy as np

from fl.aggregator.base import BaseAggregator
from fl.core.exceptions import AggregationError
from fl.core.protocols import ModelUpdate, ModelWeights


class FedAvgAggregator(BaseAggregator):
    """Federate Averaging aggregator implementation."""

    def aggregate(self, updates: list[ModelUpdate]) -> ModelWeights:
        """Aggregate client updates by averaging their weights."""
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
                weights_agg[key] += value

        num_updates = len(updates)
        for key in weights_agg:
            weights_agg[key] /= num_updates

        return weights_agg
