from __future__ import annotations

from typing import Any


class FLError(Exception):
    """Base exception for all federated learning error."""

    def __init__(
        self, message: str, details: dict[str, Any] | None = None
    ) -> None:
        super().__init__(message)
        self.details = details or {}


class CommunicationError(FLError):
    """Raised for network and communication related error."""

    pass


class SecurityError(FLError):
    """Raised for security and encryption related error."""

    pass


class ValidationError(FLError):
    """Raised for data validation errors."""

    pass


class ModelError(FLError):
    """Raised for model-related errors."""

    pass


class TrainingError(FLError):
    """Raised for training-related errors."""


class ClientError(FLError):
    """Raised for client-side errors."""

    pass


class ServerError(FLError):
    """Raised for server-side errors."""


class AggregationError(FLError):
    """Raised for aggregation-related errors."""

    pass


class ConfigurationError(FLError):
    """Raised for configruation-related errors."""

    pass


class ResourceError(FLError):
    """Raised for resource-related errors (memory, CPU, etc)."""

    pass
