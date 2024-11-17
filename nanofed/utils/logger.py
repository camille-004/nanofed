import asyncio
import logging
import sys
from contextlib import AbstractContextManager, contextmanager
from dataclasses import dataclass
from datetime import datetime
from enum import Enum, auto
from functools import wraps
from pathlib import Path
from typing import (
    Any,
    Callable,
    ClassVar,
    Iterator,
    Literal,
    ParamSpec,
    TypeVar,
)

P = ParamSpec("P")
R = TypeVar("R")


class LogLevel(Enum):
    DEBUG = auto()
    INFO = auto()
    WARNING = auto()
    ERROR = auto()


@dataclass(slots=True, frozen=True)
class LogConfig:
    """Configuration for logger."""

    level: LogLevel
    color: bool
    format: str
    output: Literal["console", "file", "both"]
    log_dir: Path | None = None


@dataclass(slots=True)
class LogContext:
    _component: str
    _subcomponent: str | None = None

    def __str__(self) -> str:
        if self._subcomponent:
            return f"{self._component}.{self._subcomponent}"
        return self._component


class Logger:
    """Singleton logger with color support and file output."""

    _instance: ClassVar["Logger | None"] = None
    _initialized: bool = False
    current_context: LogContext | None

    def __new__(cls) -> "Logger":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance.current_context = None
        return cls._instance

    def __init__(self) -> None:
        if self._initialized:
            return

        self._logger = logging.getLogger("nanofed")
        self._logger.setLevel(logging.DEBUG)

        if not self._logger.handlers:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setFormatter(Formatter(use_color=True))
            self._logger.addHandler(console_handler)

        self._initialized = True

    @contextmanager
    def context(
        self, component: str, subcomponent: str | None = None
    ) -> Iterator["Logger"]:
        previous_context = self.current_context
        self.current_context = LogContext(component, subcomponent)
        try:
            yield self
        finally:
            self.current_context = previous_context

    def configure(self, config: LogConfig) -> None:
        """Configure logger with given settings."""
        self._logger.handlers.clear()

        level_map = {
            LogLevel.DEBUG: logging.DEBUG,
            LogLevel.INFO: logging.INFO,
            LogLevel.WARNING: logging.WARNING,
            LogLevel.ERROR: logging.ERROR,
        }
        self._logger.setLevel(level_map[config.level])

        if config.output in ("console", "both"):
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setFormatter(Formatter(config.color))
            self._logger.addHandler(console_handler)

        if config.output in ("file", "both") and config.log_dir:
            log_file = (
                config.log_dir / f"nanofed_{datetime.now():%Y%m%d_%H%M%S}.log"
            )
            log_file.parent.mkdir(parents=True, exist_ok=True)
            file_handler = logging.FileHandler(log_file)
            file_handler.setFormatter(Formatter(use_color=False))
            self._logger.addHandler(file_handler)

    def _log(self, level: int, msg: str) -> None:
        extra = {
            "context": str(self.current_context)
            if self.current_context
            else "nanofed"
        }
        self._logger.log(level, msg, extra=extra)

    def debug(self, msg: str) -> None:
        self._log(logging.DEBUG, msg)

    def info(self, msg: str) -> None:
        self._log(logging.INFO, msg)

    def warning(self, msg: str) -> None:
        self._log(logging.WARNING, msg)

    def error(self, msg: str) -> None:
        self._log(logging.ERROR, msg)


class Formatter(logging.Formatter):
    """Custom formatter with color support."""

    COLORS: ClassVar[dict[str, str]] = {
        "DEBUG": "\033[36m",  # Cyan
        "INFO": "\033[32m",  # Green
        "WARNING": "\033[33m",  # Yellow
        "ERROR": "\033[31m",  # Red
        "RESET": "\033[0m",  # Reset
    }

    def __init__(self, use_color: bool = True) -> None:
        super().__init__(
            "%(asctime)s - %(context)s - %(levelname)s - %(message)s"
        )
        self.use_color = use_color

    def format(self, record: logging.LogRecord) -> str:
        if not hasattr(record, "context"):
            record.context = "nanofed"

        if self.use_color:
            level_name = record.levelname
            if level_name in self.COLORS:
                record.levelname = (
                    f"{self.COLORS[level_name]}"
                    f"{level_name}"
                    f"{self.COLORS['RESET']}"
                )
        return super().format(record)


class LoggerContextManager(AbstractContextManager):
    """Context manager for temporary logging contexts."""

    __slots__ = ("_logger", "_context", "_previous_context")

    def __init__(self, logger: "Logger", context: LogContext) -> None:
        self._logger = logger
        self._context = context
        self._previous_context: LogContext | None = None

    def __enter__(self) -> "Logger":
        self._previous_context = self._logger.current_context
        self._logger.current_context = self._context
        return self._logger

    def __exit__(self, *exc: Any) -> None:
        self._logger.current_context = self._previous_context


def log_exec(func: Callable[P, R]) -> Callable[P, R]:
    """Decorator to log function execution time."""

    @wraps(func)
    def sync_wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
        logger = Logger()
        start_time = datetime.now()
        logger.debug(f"Starting {func.__name__}")
        try:
            result = func(*args, **kwargs)
            logger.debug(
                f"Completed {func.__name__} in "
                f"{(datetime.now() - start_time).total_seconds():.2f}s"
            )
            return result
        except Exception as e:
            logger.error(f"Error in {func.__name__}: {str(e)}")
            raise

    @wraps(func)
    async def async_wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
        logger = Logger()
        start_time = datetime.now()
        logger.debug(f"Starting {func.__name__}")
        try:
            result = await func(*args, **kwargs)  # type: ignore[misc]
            logger.debug(
                f"Completed {func.__name__} in "
                f"{(datetime.now() - start_time).total_seconds():.2f}s"
            )
            return result
        except Exception as e:
            logger.error(f"Error in {func.__name__}: {str(e)}")
            raise

    if asyncio.iscoroutinefunction(func):
        return async_wrapper  # type: ignore[return-value]
    return sync_wrapper
