import json
from pathlib import Path
from unittest import mock

import pytest
import torch

from nanofed.core.interfaces import ModelProtocol
from nanofed.server.model_manager.manager import ModelManager, ModelVersion
from nanofed.utils.dates import get_current_time


class DummyModel(torch.nn.Module, ModelProtocol):
    def __init__(self):
        super().__init__()
        self.fc = torch.nn.Linear(2, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)


@pytest.fixture
def setup_model_manager(tmp_path: Path):
    """Setup model manager with directories."""
    model = DummyModel()
    manager = ModelManager(model=model)

    # Set up directories that would normally be set by Coordinator
    models_dir = tmp_path / "models" / "models"
    configs_dir = tmp_path / "models" / "configs"
    models_dir.mkdir(parents=True)
    configs_dir.mkdir(parents=True)

    manager.set_dirs(models_dir, configs_dir)
    return manager


def test_save_model(setup_model_manager):
    manager = setup_model_manager
    config = {"learning_rate": 0.01}
    metrics = {"loss": 0.1, "accuracy": 0.9}

    with mock.patch("torch.save") as mock_save:
        version = manager.save_model(config=config, metrics=metrics)

        assert isinstance(version, ModelVersion)
        assert version.config == config
        assert version.version_id.startswith("model_v_")
        mock_save.assert_called_once()


def test_load_latest_model(setup_model_manager, tmp_path):
    manager = setup_model_manager
    model_version = "model_v_test_001"

    config_path = tmp_path / "models" / "configs" / f"{model_version}.json"
    model_path = tmp_path / "models" / "models" / f"{model_version}.pt"

    with mock.patch.object(
        Path, "glob", return_value=[config_path]
    ), mock.patch(
        "builtins.open",
        mock.mock_open(
            read_data=json.dumps(
                {
                    "version_id": model_version,
                    "timestamp": get_current_time().isoformat(),
                    "config": {"learning_rate": 0.01},
                }
            )
        ),
    ), mock.patch("pathlib.Path.exists", return_value=True), mock.patch(
        "torch.load",
        return_value={
            "fc.weight": torch.tensor([[0.1, 0.2], [0.3, 0.4]]),
            "fc.bias": torch.tensor([0.5, 0.6]),
        },
    ):
        version = manager.load_model()

        assert version.version_id == model_version
        assert version.path == model_path
        assert isinstance(version, ModelVersion)
