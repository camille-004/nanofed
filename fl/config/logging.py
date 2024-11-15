from __future__ import annotations

import logging
import sys
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping, MutableMapping

from fl.config.settings import get_settings

settings = get_settings()


class ColoredFormatter(logging.Formatter):
    """Custom formatter with colors for different log levels."""

    COLORS = {
        logging.DEBUG: "\033[36m",
        logging.INFO: "\033[32m",
        logging.WARNING: "\033[33m",
        logging.ERROR: "\033[31m",
        logging.CRITICAL: "\033[35m",
        "reset": "\033[0m",
    }

    def format(self, record: logging.LogRecord) -> str:
        if not record.exc_info:
            level_color = self.COLORS.get(record.levelno, self.COLORS["reset"])
            record.levelname = (
                f"{level_color}{record.levelname}{self.COLORS['reset']}"
            )
            record.msg = f"{level_color}{record.msg}{self.COLORS['reset']}"
        return super().format(record)


class JSONFormatter(logging.Formatter):
    """JSON formatter for structured logging."""

    def format(self, record: logging.LogRecord) -> str:
        import json

        log_data = {
            "timestamp": datetime.fromtimestamp(record.created).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }

        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)

        if hasattr(record, "extra"):
            log_data.update(record.extra)

        return json.dumps(log_data)


@lru_cache()
def setup_logging(
    name: str,
    level: int = logging.INFO,
    json_output: bool = False,
    log_file: Path | None = None,
) -> logging.Logger:
    """Configure loggign with both console and file handlers."""

    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Clear existing handlers
    logger.handlers.clear()

    console_handler = logging.StreamHandler(sys.stdout)
    console_formatter: logging.Formatter
    if json_output:
        console_formatter = JSONFormatter()
    else:
        console_formatter = ColoredFormatter(
            fmt="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)

    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_formatter = JSONFormatter()
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)

    return logger


class LoggerAdapter(logging.LoggerAdapter):
    """Custom logger adapter for adding context to log messages."""

    extra: Mapping[str, Any]

    def __init__(
        self, logger: logging.Logger, context: dict[str, Any]
    ) -> None:
        super().__init__(logger, context)
        self.extra = context

    def process(
        self, msg: str, kwargs: MutableMapping[str, Any]
    ) -> tuple[str, MutableMapping[str, Any]]:
        context_str = " ".join(f"{k}={v}" for k, v in self.extra.items())
        return f"{msg} [{context_str}]", kwargs


def get_logger(
    name: str, context: dict[str, Any] | None = None
) -> logging.Logger | LoggerAdapter:
    """Get a logger with optional context."""
    logger = setup_logging(
        name,
        level=logging.INFO,
        json_output=False,
        log_file=settings.data_dir / "logs" / f"{name}.log",
    )

    if context:
        return LoggerAdapter(logger, context)
    return logger
