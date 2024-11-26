import json
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import torch

from nanofed.core import ModelManagerError, ModelProtocol
from nanofed.utils import Logger, get_current_time, log_exec


def make_json_serializable(
    data: Any,
) -> dict[str, Any] | list[Any] | str | int | float | bool | None:
    """Recursively convert data to JSON-serializable types."""
    if isinstance(data, dict):
        return {
            key: make_json_serializable(value) for key, value in data.items()
        }
    elif isinstance(data, list):
        return [make_json_serializable(item) for item in data]
    elif hasattr(data, "__dataclass_fields__"):
        return make_json_serializable(asdict(data))
    elif isinstance(data, (int, float, str, bool, type(None))):
        return data
    else:
        return str(data)


@dataclass(slots=True, frozen=True)
class ModelVersion:
    """Model version information."""

    version_id: str
    timestamp: datetime
    config: dict[str, Any]
    path: Path


class ModelManager:
    """Manages versioning and storage of FL models.

    Handles model versioning, persistence, and loading of model checkpoints
    with associated metadata.

    Parameters
    ----------
    base_dir : Path
        Base directory for model storage.
    model : ModelProtocol
        Initial model instance.

    Attributes
    ----------
    current_version : ModelVersion or None
        Current active model version.
    _version_counter : int
        Counter for generating version IDs.

    Methods
    -------
    save_model(config, metrics=None)
        Save current model state with configuration.
    load_model(version_id=None)
        Load a specific model version or latest.
    list_versions()
        List all avaialble model versions.

    Examples
    --------
    >>> manager = ModelManager(Path("./models"), model)
    >>> version = manager.save_model(config)
    >>> manager.load_model(version.version_id)
    """

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

        # Make sure an initial model exists if no versions are present
        if not self.list_versions():
            self._logger.info("No model versions found. Saving initial model.")
            default_config = {"name": "default", "version": "1.0"}
            self.save_model(config=default_config)

    @property
    def current_version(self) -> ModelVersion | None:
        return self._current_version

    def _generate_version_id(self) -> str:
        """Generate a unique version ID."""
        timestamp = get_current_time().strftime("%Y%m%d_%H%M%S")
        self._version_counter += 1
        return f"model_v_{timestamp}_{self._version_counter:03d}"

    @log_exec
    def save_model(
        self, config: dict[str, Any], metrics: dict[str, float] | None = None
    ) -> ModelVersion:
        """Save current model state with configuration."""
        with self._logger.context("model_manager", "save"):
            version_id = self._generate_version_id()

            model_path = self._models_dir / f"{version_id}.pt"
            torch.save(self._model.state_dict(), model_path)

            config_dict = make_json_serializable(config)

            config_data = {
                "version_id": version_id,
                "timestamp": get_current_time().isoformat(),
                "config": config_dict,
                "metrics": metrics or {},
            }

            config_path = self._configs_dir / f"{version_id}.json"
            try:
                with open(config_path, "w") as f:
                    json.dump(config_data, f, indent=2)
            except TypeError as e:
                raise ModelManagerError(
                    f"Failed to serialize config data: {e}"
                ) from e

            version = ModelVersion(
                version_id=version_id,
                timestamp=get_current_time(),
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
                raise ModelManagerError(f"Failde to load model: {e}") from e

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
