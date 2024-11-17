import torch
from torch.utils.data import DataLoader, TensorDataset

from nanofed.trainer.base import TrainingConfig, TrainingMetrics
from nanofed.trainer.torch import TorchTrainer


def test_torch_trainer_loss_and_accuracy():
    trainer = TorchTrainer(
        config=TrainingConfig(epochs=1, batch_size=4, learning_rate=0.01)
    )

    output = torch.tensor([[0.1, 0.9], [0.3, 0.7]])
    target = torch.tensor([1, 1])

    loss = trainer.compute_loss(output, target)
    accuracy = trainer.compute_accuracy(output, target)

    assert loss > 0
    assert 0 <= accuracy <= 1
    assert accuracy == 1.0


def test_torch_trainer_training():
    class DummyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = torch.nn.Linear(3, 2)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.fc(x)

    model = DummyModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    inputs = torch.randn(16, 3)
    targets = torch.randint(0, 2, (16,))
    dataset = TensorDataset(inputs, targets)
    dataloader = DataLoader(dataset, batch_size=4)

    trainer = TorchTrainer(
        config=TrainingConfig(epochs=1, batch_size=4, learning_rate=0.01)
    )
    metrics = trainer.train_epoch(model, dataloader, optimizer, epoch=1)

    assert isinstance(metrics, TrainingMetrics)
    assert metrics.loss > 0
    assert 0 <= metrics.accuracy <= 1
