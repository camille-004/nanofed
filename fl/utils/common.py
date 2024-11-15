from __future__ import annotations

import asyncio
import functools
import time
from collections import deque
from contextlib import asynccontextmanager, contextmanager
from typing import (
    Any,
    AsyncGenerator,
    Awaitable,
    Callable,
    Deque,
    Generator,
    ParamSpec,
    TypeVar,
    overload,
)

import numpy as np
from pydantic import BaseModel

T = TypeVar("T")
P = ParamSpec("P")


@overload
def timed(func: Callable[P, T]) -> Callable[P, tuple[T, float]]: ...


@overload
def timed(
    func: Callable[P, Awaitable[T]],
) -> Callable[P, Awaitable[tuple[T, float]]]: ...


def timed(func: Any) -> Any:
    """Decorator to measure execution time of a function."""
    if asyncio.iscoroutinefunction(func):

        @functools.wraps(func)
        async def async_wrapper(
            *args: P.args, **kwargs: P.kwargs
        ) -> tuple[T, float]:
            start = time.perf_counter()
            result = await func(*args, **kwargs)
            end = time.perf_counter()
            return result, end - start

        return async_wrapper
    else:

        @functools.wraps(func)
        def sync_wrapper(*args: P.args, **kwargs: P.kwargs) -> tuple[T, float]:
            start = time.perf_counter()
            result = func(*args, **kwargs)
            end = time.perf_counter()
            return result, end - start

        return sync_wrapper


async def retry_async(
    func: Callable[P, Awaitable[T]],
    max_attempts: int = 3,
    delay: float = 1.0,
    backoff: float = 2.0,
    exceptions: tuple[type[Exception], ...] = (Exception,),
    *args: P.args,
    **kwargs: P.kwargs,
) -> T:
    """Retry an async function with exponential backoff."""
    last_error = None

    for attempt in range(max_attempts):
        try:
            return await func(*args, **kwargs)
        except exceptions as e:
            last_error = e
            if attempt == max_attempts - 1:
                raise
            await asyncio.sleep(delay * (backoff**attempt))

    raise last_error  # type: ignore


class MovingAverage:
    """Calculate moving average with a fixed window."""

    def __init__(self, window_size: int) -> None:
        self.window_size = window_size
        self.values: Deque[float] = deque(maxlen=window_size)

    def update(self, value: float) -> float:
        self.values.append(value)
        return self.average

    @property
    def average(self) -> float:
        if self.values:
            return sum(self.values) / len(self.values)
        else:
            return 0.0


@contextmanager
def numpy_random_seed(seed: int) -> Generator[None, None, None]:
    """Contet manager for temporarily settings numpy random seed."""
    state = np.random.get_state()
    np.random.seed(seed)
    try:
        yield
    finally:
        np.random.set_state(state)


@asynccontextmanager
async def timeout_context(seconds: float) -> AsyncGenerator[None, None]:
    """Async context manager for timeout."""
    try:
        yield
    except asyncio.TimeoutError:
        raise TimeoutError(f"Operation timed out after {seconds} seconds.")


class Validators:
    """Common validation functions."""

    @staticmethod
    def validate_positive(value: float, name: str) -> None:
        if value <= 0:
            raise ValueError(f"{name} must be positive")

    @staticmethod
    def validate_probability(value: float, name: str) -> None:
        if not 0 <= value <= 1:
            raise ValueError(f"{name} must be between 0 and 1")

    @staticmethod
    def validate_array_shape(
        array: np.ndarray, expected_shape: tuple[int, ...], name: str
    ) -> None:
        if array.shape != expected_shape:
            raise ValueError(
                f"{name} has wrong shape. Expected {expected_shape}, "
                f"got {array.shape}"
            )


class MetricsTracker(BaseModel):
    """Track and compute various metrics."""

    values: dict[str, list[float]] = {}

    def add_metric(self, name: str, value: float) -> None:
        if name not in self.values:
            self.values[name] = []
        self.values[name].append(value)

    def get_average(self, name: str) -> float:
        if name not in self.values:
            raise KeyError(f"No metric named {name}")
        return sum(self.values[name]) / len(self.values[name])

    def get_std(self, name: str) -> float:
        if name not in self.values:
            raise KeyError(f"No metric named {name}")
        return float(np.std(self.values[name]))

    def reset(self) -> None:
        return self.values.clear()
