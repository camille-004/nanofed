from typing import Sequence

import torch

from nanofed.core import ModelProtocol, ModelUpdate
from nanofed.server.aggregator import AggregationResult, BaseAggregator
from nanofed.utils import get_current_time, log_exec


class FedAvgAggregator(BaseAggregator[ModelProtocol]):
    """Federate Averaging (FedAvg) aggregation strategy.

    Implements the FedAvg algorithm for aggregating client model updates into
    a global model. Supports weighted averaging based on client data sizes.

    Methods
    -------
    aggregate(model, updates)
        Aggregate client updates into global model.
    _compute_weights(num_clients)
        Compute aggregation weights for clients.
    _aggregate_metrics(updates)
        Aggregate training metrics from clients.

    Notes
    -----
    The aggregation process:
    1. Validates all client updates
    2. Computes weighted average of model parameters
    3. Updates global model with aggregated parameters
    4. Aggregates client metrics

    Examples
    --------
    >>> aggregator = FedAvgAggregator()
    >>> result = aggregator.aggregate(global_model, client_updates)
    """

    def _to_tensor(
        self, data: list[float] | list[list[float]] | torch.Tensor
    ) -> torch.Tensor:
        if isinstance(data, torch.Tensor):
            return data.clone().detach()
        return torch.tensor(data, dtype=torch.float32)

    @log_exec
    def aggregate(
        self, model: ModelProtocol, updates: Sequence[ModelUpdate]
    ) -> AggregationResult[ModelProtocol]:
        """Aggregate updates using FedAvg algorithm."""
        self._validate_updates(updates)

        weights = self._compute_weights(len(updates))
        state_agg: dict[str, torch.Tensor] = {}

        for key, value in updates[0]["model_state"].items():
            tensor = self._to_tensor(value)
            state_agg[key] = tensor * weights[0]

        for update, weight in zip(updates[1:], weights[1:]):
            for key, value in update["model_state"].items():
                tensor = self._to_tensor(value)
                state_agg[key] += tensor * weight

        # Update global model
        model.load_state_dict(state_agg)

        avg_metrics = self._aggregate_metrics(updates)

        self._current_round += 1

        return AggregationResult(
            model=model,
            round_number=self._current_round,
            num_clients=len(updates),
            timestamp=get_current_time(),
            metrics=avg_metrics,
        )

    def _aggregate_metrics(
        self, updates: Sequence[ModelUpdate]
    ) -> dict[str, float]:
        """Aggregate metrics from all updates."""
        all_metrics: dict[str, list[float]] = {}

        for update in updates:
            for key, value in update["metrics"].items():
                if isinstance(value, (int, float)):
                    all_metrics.setdefault(key, []).append(value)

        return {
            key: sum(values) / len(values)
            for key, values in all_metrics.items()
            if values
        }
