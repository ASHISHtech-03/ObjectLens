"""Logging configuration (console + optional rotating file)."""
from __future__ import annotations

import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Optional

_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"


def setup_logging(level: str = "INFO", log_file: Optional[str] = None) -> logging.Logger:
    """Configure the root ``objectlens`` logger. Safe to call repeatedly."""
    logger = logging.getLogger("objectlens")
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    logger.handlers.clear()
    logger.propagate = False

    console = logging.StreamHandler()
    console.setFormatter(logging.Formatter(_FORMAT))
    logger.addHandler(console)

    if log_file:
        path = Path(log_file)
        path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = RotatingFileHandler(path, maxBytes=1_000_000, backupCount=2,
                                           encoding="utf-8")
        file_handler.setFormatter(logging.Formatter(_FORMAT))
        logger.addHandler(file_handler)
    return logger
