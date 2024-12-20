from enum import Enum, auto
from typing import Protocol, Sequence, cast

import torch
from pydantic import ConfigDict, Field

from nanofed.core import ModelProtocol, ModelUpdate
from nanofed.privacy.accountant import PrivacySpent
from nanofed.privacy.config import PrivacyConfig
from nanofed.privacy.mechanisms import (
    BasePrivacyMechanism,
    PrivacyMechanismFactory,
    PrivacyType,
)
from nanofed.utils import Logger

from .base import AggregationResult, BaseAggregator


class SecureAggregationType(Enum):
    """Type of secure aggregation protocol."""

    NONE = auto()
    THRESHOLD = auto()  # Threshold-based secure aggregation
    HOMOMORPHIC = auto()  # Homomorphic encryption-based


class PrivacyAwareAggregationConfig(PrivacyConfig):
    """Configuration for privacy-aware aggregation.

    Inherits privacy parameters from PrivacyConfig and adds
    aggregation-specific settings.
    """

    privacy_type: PrivacyType = Field(
        default=PrivacyType.CENTRAL, description="Type of privacy mechanism"
    )
    secure_aggregation: SecureAggregationType = Field(
        default=SecureAggregationType.NONE,
        description="Type of secure aggreation",
    )
    min_clients: int = Field(
        default=1, description="Minimum number of clients", ge=1
    )
    dropout_tolerance: float = Field(
        default=0.0,
        description="Fraction of clients that can drop out",
        ge=0.0,
        le=1.0,
    )
    clip_norm: float = Field(
        default=1.0,
        description="Global clipping norm for aggregated updates",
        gt=0.0,
    )

    model_config = ConfigDict(arbitrary_types_allowed=True)


class SecureAggregationProtocol(Protocol):
    """Protocol for secure aggregation."""

    def aggregate_shares(
        self, shares: Sequence[dict[str, torch.Tensor]]
    ) -> dict[str, torch.Tensor]: ...

    def verify_shares(
        self, shares: Sequence[dict[str, torch.Tensor]]
    ) -> bool: ...


class ThresholdSecureAggregation:
    """Threshold-based secure aggregation."""

    def __init__(self, min_clients: int) -> None:
        self._min_clients = min_clients
        self._logger = Logger()

    def aggregate_shares(
        self, shares: Sequence[dict[str, torch.Tensor]]
    ) -> dict[str, torch.Tensor]:
        """Aggregate shares using threshold-based protocol."""
        if len(shares) < self._min_clients:
            raise ValueError(
                f"Not enough clients: {len(shares)} < {self._min_clients}"
            )

        aggregated = {}
        for key in shares[0].keys():
            # Sum shares for each parameter
            param_shares = torch.stack([share[key] for share in shares])
            aggregated[key] = param_shares.sum(dim=0)

        return aggregated

    def verify_shares(self, shares: Sequence[dict[str, torch.Tensor]]) -> bool:
        """Verify validity of shares."""
        if len(shares) < self._min_clients:
            return False

        shapes = {
            key: share[key].shape
            for key in shares[0].keys()
            for share in shares
        }

        return all(
            all(share[key].shape == shapes[key] for key in shapes)
            for share in shares
        )


class PrivacyAwareAggregator(BaseAggregator):
    """Aggregator with privacy-preserving mechanisms.

    Supports both central and local DP, with optional secure aggregation
    protocols.
    """

    def __init__(
        self,
        config: PrivacyAwareAggregationConfig,
        privacy_mechanism: BasePrivacyMechanism | None = None,
        secure_aggregation: SecureAggregationProtocol | None = None,
    ) -> None:
        super().__init__()
        self._config = config
        self._privacy_mech = (
            privacy_mechanism
            or PrivacyMechanismFactory.create(
                config.privacy_type, config=config
            )
        )
        self._secure_agg = secure_aggregation
        if (
            config.secure_aggregation == SecureAggregationType.THRESHOLD
            and secure_aggregation is None
        ):
            self._secure_agg = ThresholdSecureAggregation(config.min_clients)

    def _validate_updates(self, updates: Sequence[ModelUpdate]) -> None:
        """Validate model updates."""
        if not updates:
            raise ValueError("No updates provided")

        if len(updates) < self._config.min_clients:
            raise ValueError(
                f"Not enough clients: {len(updates)} < {self._config.min_clients}"  # noqa
            )

        first_round = updates[0].get("round_number")
        if not all(
            update.get("round_number") == first_round for update in updates
        ):
            raise ValueError("Updates from different rounds")

        first_state = updates[0]["model_state"]
        if not all(
            update["model_state"].keys() == first_state.keys()
            for update in updates
        ):
            raise ValueError("Inconsistent model architectures")

        # Validate privacy budgets is using local DP
        if self._config.privacy_type == PrivacyType.LOCAL:
            for update in updates:
                privacy_spent = update.get("privacy_spent")
                if privacy_spent is None:
                    raise ValueError(
                        f"Missing privacy budget for client {update['client_id']}"  # noqa
                    )

    def _process_local_updates(
        self, updates: Sequence[ModelUpdate]
    ) -> Sequence[ModelUpdate]:
        """Process updates with local privacy."""
        return list(updates)

    def _process_central_updates(
        self, updates: Sequence[ModelUpdate]
    ) -> Sequence[ModelUpdate]:
        """Process updates with central privacy."""
        processed = []
        batch_size = len(updates)

        for update in updates:
            # Apply central privacy mechanism
            private_state = self._privacy_mech.add_noise(
                update["model_state"], batch_size=batch_size
            )
            new_update = {**update, "model_state": private_state}
            processed.append(cast(ModelUpdate, new_update))

        return processed

    def _compute_weights(self, updates: Sequence[ModelUpdate]) -> list[float]:
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

        if self._config.privacy_type == PrivacyType.LOCAL:
            privacy_adjustments = []
            for update in updates:
                privacy_spent: dict[str, float] | PrivacySpent = update.get(
                    "privacy_spent", {"epsilon": 1.0, "delta": 1e-5}
                )
                if isinstance(privacy_spent, PrivacySpent):
                    privacy_spent_dict = {
                        "epsilon": privacy_spent.epsilon_spent,
                        "delta": privacy_spent.epsilon_spent,
                    }
                elif isinstance(privacy_spent, dict):
                    privacy_spent_dict = privacy_spent
                else:
                    raise TypeError(
                        f"privacy_spent should be a dict or PrivacySpent "
                        f"instance, got {type(privacy_spent)}"
                    )

                # More privacy spent = less noise = higher weight
                epsilon = privacy_spent_dict.get("epsilon", 1.0)
                adjustment = epsilon
                privacy_adjustments.append(adjustment)

            total_adjustment = sum(privacy_adjustments)
            if total_adjustment > 0:
                privacy_adjustments = [
                    adj / total_adjustment for adj in privacy_adjustments
                ]

                weights = [w * p for w, p in zip(weights, privacy_adjustments)]

                total_weight = sum(weights)
                weights = [w / total_weight for w in weights]

        self._logger.debug(f"Computed weights: {weights}")
        return weights

    def _aggregate_metrics(
        self,
        updates: Sequence[ModelUpdate],
        weights: list[float] | None = None,
    ) -> dict[str, float]:
        """Aggregate metrics from all updates."""
        if not updates:
            return {}

        if weights is None:
            total_samples = sum(
                update["metrics"].get("samples_processed", 1)
                for update in updates
            )
            weights = [
                update["metrics"].get("samples_processed", 1) / total_samples
                for update in updates
            ]

        agg_metrics: dict[str, float] = {}

        all_keys: set[str] = set()
        for update in updates:
            metrics = update.get("metrics", {})
            all_keys.update(
                key
                for key, value in metrics.items()
                if isinstance(value, (int, float))
            )

        for key in all_keys:
            weighted_sum = sum(
                update["metrics"].get(key, 0) * weight
                for update, weight in zip(updates, weights)
            )
            agg_metrics[key] = weighted_sum

        if self._privacy_mech is not None:
            privacy_spent = self._privacy_mech.get_privacy_spent()
            agg_metrics.update(
                {
                    "privacy_epsilon": privacy_spent.epsilon_spent,
                    "privacy_delta": privacy_spent.delta_spent,
                }
            )

        return agg_metrics

    def aggregate(
        self, model: ModelProtocol, updates: Sequence[ModelUpdate]
    ) -> AggregationResult:
        """Aggregate updates with privacy preservation."""
        self._validate_updates(updates)

        if self._config.privacy_type == PrivacyType.LOCAL:
            updates_processed = self._process_local_updates(updates)
        else:
            updates_processed = self._process_central_updates(updates)

        # Apply secure aggregation if enabled
        if self._secure_agg is not None:
            if not self._secure_agg.verify_shares(
                [u["model_state"] for u in updates_processed]
            ):
                raise ValueError("Invalid shares for secure aggregation")

            aggregated_state = self._secure_agg.aggregate_shares(
                [u["model_state"] for u in updates_processed]
            )
        else:
            # Standard weighted averaging
            weights = self._compute_weights(updates_processed)
            aggregated_state = {}

            for key in updates_processed[0]["model_state"]:
                weighted_sum = sum(
                    (
                        update["model_state"][key] * weight
                        for update, weight in zip(updates_processed, weights)
                    ),
                    torch.zeros_like(
                        updates_processed[0]["model_state"][key]
                    ),  # Starting with a tensor
                )
                aggregated_state[key] = weighted_sum

        # Update global model
        model.load_state_dict(aggregated_state)

        return AggregationResult(
            model=model,
            round_number=self._current_round,
            num_clients=len(updates),
            timestamp=self._get_timestamp(),
            metrics=self._aggregate_metrics(updates_processed),
        )
