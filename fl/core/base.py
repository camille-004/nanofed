from enum import Enum, auto
from functools import wraps
from typing import (
    Awaitable,
    Callable,
    ClassVar,
    Concatenate,
    ParamSpec,
    TypeVar,
)

import numpy as np

from fl.config.logging import get_logger
from fl.core.exceptions import ValidationError
from fl.core.protocols import ModelWeights

P = ParamSpec("P")
T = TypeVar("T")


class Role(Enum):
    """Component roles."""

    CLIENT = auto()
    SERVER = auto()
    EVALUATOR = auto()


class ClientState(Enum):
    """Possible client states."""

    INITIALIZED = auto()
    TRAINING = auto()
    VALIDATING = auto()
    UPDATING = auto()
    ERROR = auto()
    STOPPED = auto()


def _validate_weights_dict(weights: ModelWeights) -> None:
    """Validate that weights is a dictionary."""
    if not isinstance(weights, dict):
        raise ValidationError("Weights must be a dictionary")


def _validate_weight_values(weights: ModelWeights) -> None:
    """Validate individual weight values."""
    for key, value in weights.items():
        if not isinstance(value, np.ndarray):
            raise ValidationError(f"Weight {key} must be a numpy array")
        if np.any(np.isnan(value)) or np.any(np.isinf(value)):
            raise ValidationError(f"Weight {key} contains invalid values")


async def _validate_with_state(
    self: T,
    weights: ModelWeights,
    func: Callable[Concatenate[T, ModelWeights, P], Awaitable[ModelWeights]],
    *args: P.args,
    **kwargs: P.kwargs,
) -> ModelWeights:
    """Validate weights with state handling."""
    try:
        _validate_weights_dict(weights)
        _validate_weight_values(weights)
        return await func(self, weights, *args, **kwargs)
    except Exception:
        setattr(self, "_state", ClientState.ERROR)
        raise


async def _validate_without_state(
    self: T,
    weights: ModelWeights,
    func: Callable[Concatenate[T, ModelWeights, P], Awaitable[ModelWeights]],
    *args: P.args,
    **kwargs: P.kwargs,
) -> ModelWeights:
    """Validate weights without state handling."""
    _validate_weights_dict(weights)
    _validate_weight_values(weights)
    return await func(self, weights, *args, **kwargs)


def validate_weights(
    func: Callable[Concatenate[T, ModelWeights, P], Awaitable[ModelWeights]],
) -> Callable[Concatenate[T, ModelWeights, P], Awaitable[ModelWeights]]:
    """Decorator to validate model weights."""

    @wraps(func)
    async def wrapper(
        self: T, weights: ModelWeights, *args: P.args, **kwargs: P.kwargs
    ) -> ModelWeights:
        validate_func = (
            _validate_with_state
            if hasattr(self, "_state")
            else _validate_without_state
        )
        return await validate_func(self, weights, func, *args, **kwargs)

    return wrapper


class Component:
    """Base class for components."""

    _instance: ClassVar[dict[str, "Component"]] = {}

    def __new__(cls, *args: tuple, **kwargs: dict) -> "Component":
        """Singleton pattern implementation."""
        if cls not in cls._instance:
            cls._instance[cls.__name__] = super().__new__(cls)
        return cls._instance[cls.__name__]

    def __init__(self, role: Role) -> None:
        self.role = role
        self.logger = get_logger(
            self.__class__.__name__, context={"role": role.name}
        )
