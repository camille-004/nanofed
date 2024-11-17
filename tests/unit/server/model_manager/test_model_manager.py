import json
from datetime import datetime
from pathlib import Path
from unittest import mock

import torch

from nanofed.core.interfaces import ModelProtocol
from nanofed.server.model_manager.manager import ModelManager, ModelVersion


class DummyModel(torch.nn.Module, ModelProtocol):
    def __init__(self):
        super().__init__()
        self.fc = torch.nn.Linear(2, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)


def test_save_model():
    model = DummyModel()
    manager = ModelManager(base_dir=Path("/tmp/models"), model=model)

    config = {"learning_rate": 0.01}
    metrics = {"loss": 0.1, "accuracy": 0.9}

    with mock.patch("torch.save") as mock_save:
        version = manager.save_model(config=config, metrics=metrics)

        assert isinstance(version, ModelVersion)
        assert version.config == config
        assert version.version_id.startswith("model_v_")
        mock_save.assert_called_once()


def test_load_latest_model():
    class DummyModel(torch.nn.Module, ModelProtocol):
        def __init__(self):
            super().__init__()
            self.fc = torch.nn.Linear(2, 2)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.fc(x)

    model = DummyModel()
    manager = ModelManager(base_dir=Path("/tmp/models"), model=model)

    config_path = Path("/tmp/models/configs/model_v_test_001.json")
    model_path = Path("/tmp/models/models/model_v_test_001.pt")

    with mock.patch.object(
        Path, "glob", return_value=[config_path]
    ), mock.patch(
        "builtins.open",
        mock.mock_open(
            read_data=json.dumps(
                {
                    "version_id": "model_v_test_001",
                    "timestamp": datetime.now().isoformat(),
                    "config": {"learning_rate": 0.01},
                    "metrics": {"loss": 0.1},
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

        assert version.version_id == "model_v_test_001"
        assert version.path == model_path
        assert isinstance(version, ModelVersion)
