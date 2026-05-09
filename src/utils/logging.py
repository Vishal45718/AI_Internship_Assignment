"""
src/utils/logging.py — Simple logging configuration.

Provides colored, human-readable output for CLI usage
and suppresses noisy third-party loggers.
"""

from __future__ import annotations

import logging
import sys
from datetime import datetime, timezone


class ColorFormatter(logging.Formatter):
    """Colored, readable log formatter for terminal output."""

    COLORS = {
        "DEBUG": "\033[36m",     # Cyan
        "INFO": "\033[32m",      # Green
        "WARNING": "\033[33m",   # Yellow
        "ERROR": "\033[31m",     # Red
        "CRITICAL": "\033[35m",  # Magenta
    }
    RESET = "\033[0m"

    def format(self, record: logging.LogRecord) -> str:
        color = self.COLORS.get(record.levelname, "")
        ts = datetime.now(timezone.utc).strftime("%H:%M:%S")
        prefix = f"{color}[{record.levelname}]{self.RESET}"
        return f"{ts} {prefix} {record.name}: {record.getMessage()}"


def setup_logging(level: str = "INFO") -> None:
    """
    Configure root logger. Call once at application startup.

    Args:
        level: Log level string (DEBUG, INFO, WARNING, ERROR).
    """
    numeric_level = getattr(logging, level.upper(), logging.INFO)

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(ColorFormatter())

    root = logging.getLogger()
    root.setLevel(numeric_level)
    root.handlers.clear()
    root.addHandler(handler)

    # Suppress noisy third-party loggers
    for noisy in ("httpx", "httpcore", "urllib3", "chromadb", "sentence_transformers"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
