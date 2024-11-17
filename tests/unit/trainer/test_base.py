from unittest import mock

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from nanofed.core.interfaces import ModelProtocol
from nanofed.trainer.base import BaseTrainer, TrainingConfig, TrainingMetrics


class DummyTrainer(BaseTrainer[ModelProtocol]):
    def compute_loss(
        self, output: torch.Tensor, target: torch.Tensor
    ) -> torch.Tensor:
        return F.cross_entropy(output, target)

    def compute_accuracy(
        self, output: torch.Tensor, target: torch.Tensor
    ) -> float:
        return float((output.argmax(dim=1) == target).float().mean().item())


@mock.patch("torch.optim.SGD.step", autospec=True)
def test_train_epoch(mock_optimizer_step):
    class DummyModel(torch.nn.Module, ModelProtocol):
        def __init__(self):
            super().__init__()
            self.fc = torch.nn.Linear(3, 4)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.fc(x)

    model = DummyModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    trainer = DummyTrainer(
        config=TrainingConfig(epochs=1, batch_size=4, learning_rate=0.01)
    )

    inputs = torch.randn(10, 3)  # 10 samples, 3 features
    targets = torch.randint(0, 4, (10,))  # 10 samples, 4 classes

    dataset = TensorDataset(inputs, targets)
    dataloader = DataLoader(dataset, batch_size=4)

    metrics = trainer.train_epoch(model, dataloader, optimizer, epoch=1)

    assert isinstance(metrics, TrainingMetrics)
    assert metrics.loss > 0
    assert 0 <= metrics.accuracy <= 1
    mock_optimizer_step.assert_called()
