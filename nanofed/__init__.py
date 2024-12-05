from importlib.metadata import PackageNotFoundError, version

from nanofed.communication import HTTPClient, HTTPServer
from nanofed.orchestration import (
    Coordinator,
    CoordinatorConfig,
    coordinate,
)
from nanofed.server import FedAvgAggregator, ModelManager
from nanofed.trainer import TorchTrainer

__all__ = [
    "HTTPClient",
    "HTTPServer",
    "TorchTrainer",
    "Coordinator",
    "CoordinatorConfig",
    "FedAvgAggregator",
    "ModelManager",
    "coordinate",
]


try:
    __version__ = version("nanofed")
except PackageNotFoundError:
    __version__ = "unknown"
