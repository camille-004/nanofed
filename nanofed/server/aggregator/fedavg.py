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

        weights = self._compute_weights(updates)
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

        avg_metrics = self._aggregate_metrics(updates, weights)

        self._current_round += 1

        return AggregationResult(
            model=model,
            round_number=self._current_round,
            num_clients=len(updates),
            timestamp=get_current_time(),
            metrics=avg_metrics,
        )

    def _aggregate_metrics(
        self, updates: Sequence[ModelUpdate], weights: list[float]
    ) -> dict[str, float]:
        """Aggregate metrics from all updates."""
        # (value, weight) pairs
        all_metrics: dict[str, list[tuple[float, float]]] = {}

        for update, weight in zip(updates, weights):
            for key, value in update["metrics"].items():
                if isinstance(value, (int, float)):
                    if key not in all_metrics:
                        all_metrics[key] = []
                    all_metrics[key].append((float(value), weight))

        return {
            key: sum(val * w for val, w in value_pairs)
            / sum(w for _, w in value_pairs)
            for key, value_pairs in all_metrics.items()
            if value_pairs
        }

    def _compute_weights(self, updates: Sequence[ModelUpdate]) -> list[float]:
        # In FedAvg, each client's weight is proportional to its local dataset
        # size:
        # w_k = n_k / n where n_k is client k's dataset size and n is total
        # samples.
        sample_counts = []
        for update in updates:
            num_samples = update["metrics"].get("num_samples") or update[
                "metrics"
            ].get("samples_processed")
            if num_samples is None:
                self._logger.warning(
                    f"Client {update['client_id']} did not report sample "
                    f"count. Using 1.0"
                )
                num_samples = 1.0
            sample_counts.append(num_samples)

        total_samples = sum(sample_counts)
        weights = [count / total_samples for count in sample_counts]

        self._logger.debug(f"Client sample counts: {sample_counts}")
        self._logger.debug(f"Computed weights: {weights}")

        return weights
