import torch
import torch.nn.functional as F

from nanofed.trainer import BaseTrainer


class TorchTrainer(BaseTrainer):
    """PyTorch implementation of a trainer."""

    def compute_loss(
        self, output: torch.Tensor, target: torch.Tensor
    ) -> torch.Tensor:
        """Compute cross-entropy loss."""
        return F.cross_entropy(output, target)

    def compute_accuracy(
        self, output: torch.Tensor, target: torch.Tensor
    ) -> float:
        """Compute classification accuracy."""
        pred = output.argmax(dim=1, keepdim=True)
        correct = pred.eq(target.view_as(pred)).sum().item()
        return correct / len(target)
