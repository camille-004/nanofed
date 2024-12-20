from typing import Sequence

import torch
import torch.nn as nn
from torch.nn.utils import clip_grad_norm_

from nanofed.privacy.accountant import GaussianAccountant, PrivacySpent
from nanofed.privacy.config import PrivacyConfig
from nanofed.privacy.noise import GaussianNoiseGenerator

from .base import TrainingConfig, TrainingMetrics
from .callback import Callback
from .torch import TorchTrainer


class PrivateTrainer(TorchTrainer):
    """Trainer implementing DP-SGD for private model training.

    This implements the DP-SGD algorithm from:
    "Deep Learning with Differential Privacy" (Abadi et al., 2016)

    Inherits loss and accuracy computations from TorchTrainer while
    adding privacy mechanisms (gradient clipping and noise addition).
    """

    def __init__(
        self,
        training_config: TrainingConfig,
        privacy_config: PrivacyConfig,
        accountant: GaussianAccountant | None = None,
        noise_generator: GaussianNoiseGenerator | None = None,
        callbacks: list[Callback] | None = None,
    ) -> None:
        """Initialize private trainer.

        Parameters
        ----------
        training_config : TrainingConfig
            Training configuration
        privacy_config : PrivacyConfig
            Privacy configuration
        accountant : GaussianAccountant | None, optional
            Privacy accounting for tracking privacy budget, by default None
        noise_generator : GaussianNoiseGenerator | None, optional
            Noise generator for gradient perturbation, by default None
        callbacks : list[Callback] | None, optional
            Training callbacks, by default None
        """
        super().__init__(training_config)
        self._privacy_config = privacy_config
        self._accountant = accountant or GaussianAccountant(privacy_config)
        self._noise_gen = noise_generator or GaussianNoiseGenerator()

    def _clip_gradients(self, model: nn.Module) -> None:
        """Clip gradients to specified maximum norm.

        Parameters
        ----------
        model : nn.Module
            Model being trained
        """
        max_norm = self._privacy_config.max_gradient_norm
        clip_grad_norm_(model.parameters(), max_norm)

    def _add_noise(self, model: nn.Module, batch_size: int) -> None:
        """Add noise to gradients for privacy.

        Parameters
        ----------
        model : nn.Module
            Model being trained
        batch_size : int
            Current batch size for privacy accounting
        """
        sigma = self._privacy_config.noise_multiplier

        for p in model.parameters():
            if p.grad is not None:
                noise = self._noise_gen.generate(
                    p.grad.shape,
                    scale=sigma * self._privacy_config.max_gradient_norm,
                )
                p.grad.add_(noise)

        # Record privacy spent
        self._accountant.add_noise_event(sigma=sigma, samples=batch_size)

    def _apply_privacy_to_gradients(
        self, model: nn.Module, batch_size: int
    ) -> None:
        """Apply gradient clipping and noise addition for privacy.

        Parameters
        ----------
        model : nn.Module
            Model being trained
        batch_size : int
            Current batch size for privacy accounting
        """
        self._clip_gradients(model)
        self._add_noise(model, batch_size)

    def train_batch(
        self,
        model: nn.Module,
        batch: Sequence[torch.Tensor],
        optimizer: torch.optim.Optimizer,
    ) -> TrainingMetrics:
        """Train a single batch with privacy."""
        optimizer.zero_grad()
        inputs, targets = batch
        batch_size = len(inputs)

        # Forward pass
        outputs = model(inputs)
        loss = self.compute_loss(outputs, targets)

        # Backward pass
        loss.backward()

        # Apply privacy mechanisms
        self._apply_privacy_to_gradients(model, batch_size)

        optimizer.step()

        accuracy = self.compute_accuracy(outputs, targets)

        return TrainingMetrics(
            loss=float(loss.item()),
            accuracy=float(accuracy),
            epoch=0,
            batch=0,
            samples_processed=batch_size,
        )

    def get_privacy_spent(self) -> PrivacySpent:
        """Get current privacy expenditure.

        Returns
        -------
        PrivacySpent
            Current privacy budget consumption
        """
        return self._accountant.get_privacy_spent()

    def validate_privacy_budget(self) -> bool:
        """Check if privacy budget is still valid.

        Returns
        -------
        bool
            True if privacy buget is not exceeded
        """
        return self._accountant.validate_budget()
