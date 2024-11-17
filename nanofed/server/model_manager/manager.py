import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import torch

from nanofed.core import ModelConfig, ModelManagerError, ModelProtocol
from nanofed.utils import Logger, log_exec


@dataclass(slots=True, frozen=True)
class ModelVersion:
    """Model version information."""

    version_id: str
    timestamp: datetime
    config: ModelConfig
    path: Path


class ModelManager:
    """Manages model versioning and storage."""

    def __init__(self, base_dir: Path, model: ModelProtocol) -> None:
        self._base_dir = base_dir
        self._model = model
        self._logger = Logger()
        self._current_version: ModelVersion | None = None
        self._version_counter: int = 0

        # Create directories
        self._models_dir = base_dir / "models"
        self._configs_dir = base_dir / "configs"
        self._models_dir.mkdir(parents=True, exist_ok=True)
        self._configs_dir.mkdir(parents=True, exist_ok=True)

    @property
    def current_version(self) -> ModelVersion | None:
        return self._current_version

    def _generate_version_id(self) -> str:
        """Generate a unique version ID."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self._version_counter += 1
        return f"model_v_{timestamp}_{self._version_counter:03d}"

    @log_exec
    def save_model(
        self, config: ModelConfig, metrics: dict[str, float] | None = None
    ) -> ModelVersion:
        """Save current model state with configuration."""
        with self._logger.context("model_manager", "save"):
            version_id = self._generate_version_id()

            model_path = self._models_dir / f"{version_id}.pt"
            torch.save(self._model.state_dict(), model_path)

            config_data = {
                "version_id": version_id,
                "timestamp": datetime.now().isoformat(),
                "config": config,
                "metrics": metrics or {},
            }

            config_path = self._configs_dir / f"{version_id}.json"
            with open(config_path, "w") as f:
                json.dump(config_data, f, indent=2)

            version = ModelVersion(
                version_id=version_id,
                timestamp=datetime.now(),
                config=config,
                path=model_path,
            )

            self._current_version = version
            self._logger.info(f"Saved model version: {version_id}")

            return version

    @log_exec
    def load_model(self, version_id: str | None = None) -> ModelVersion:
        """Load a specific model version or latest."""
        with self._logger.context("model_manager", "load"):
            if version_id is None:
                config_files = sorted(self._configs_dir.glob("*.json"))
                if not config_files:
                    raise ModelManagerError("No model versions ofund")
                config_path = config_files[-1]
            else:
                config_path = self._configs_dir / f"{version_id}.json"
                if not config_path.exists():
                    raise ModelManagerError(f"Version {version_id} not found")

            with open(config_path) as f:
                config_data = json.load(f)

            model_path = self._models_dir / f"{config_data['version_id']}.pt"
            if not model_path.exists():
                raise ModelManagerError(
                    f"Model file not found for version {version_id}"
                )

            try:
                state_dict = torch.load(model_path, weights_only=True)
                self._model.load_state_dict(state_dict)
            except Exception as e:
                raise ModelManagerError(f"Failde to load model: {e}")

            version = ModelVersion(
                version_id=config_data["version_id"],
                timestamp=datetime.fromisoformat(config_data["timestamp"]),
                config=config_data["config"],
                path=model_path,
            )

            self._current_version = version
            self._logger.info(f"Loaded model version: {version.version_id}")

            return version

    def list_versions(self) -> list[ModelVersion]:
        """List all available model versions."""
        versions = []
        for config_path in sorted(self._configs_dir.glob("*.json")):
            with open(config_path) as f:
                config_data = json.load(f)

            version = ModelVersion(
                version_id=config_data["version_id"],
                timestamp=datetime.fromisoformat(config_data["timestamp"]),
                config=config_data["config"],
                path=self._models_dir / f"{config_data['version_id']}.pt",
            )
            versions.append(version)

        return versions
