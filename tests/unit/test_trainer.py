import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from nanofed.trainer.base import TrainingConfig
from nanofed.trainer.torch import TorchTrainer
from tests.unit.helpers import SimpleModel


@pytest.fixture
def trainer() -> TorchTrainer:
    config = TrainingConfig(
        epochs=1, batch_size=32, learning_rate=0.01, device="cpu"
    )
    return TorchTrainer(config)


@pytest.fixture
def dummy_data() -> DataLoader:
    x = torch.randn(100, 10)
    y = torch.randint(0, 2, (100,))
    dataset = TensorDataset(x, y)
    return DataLoader(dataset, batch_size=32)


def test_trainer_epoch(trainer, dummy_data):
    model = SimpleModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    metrics = trainer.train_epoch(model, dummy_data, optimizer, epoch=0)

    assert isinstance(metrics.loss, float)
    assert isinstance(metrics.accuracy, float)
    assert metrics.epoch == 0
    assert metrics.samples_processed == 100


def test_compute_accuracy(trainer):
    output = torch.tensor([[0.1, 0.9], [0.8, 0.2]])
    target = torch.tensor([1, 0])

    accuracy = trainer.compute_accuracy(output, target)
    assert accuracy == 1.0
