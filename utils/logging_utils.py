"""Logging utilities for AutoML-Insight."""

import logging
import sys
from datetime import datetime
from logging.handlers import RotatingFileHandler
from pathlib import Path


def setup_logger(
    name: str = "automl_insight", log_dir: str = "results/logs", level: int = logging.INFO, console: bool = True
) -> logging.Logger:
    """
    Setup logger with file and console handlers.

    Args:
        name: Logger name
        log_dir: Directory for log files
        level: Logging level
        console: Whether to add console handler

    Returns:
        Configured logger instance
    """
    # Create logger
    logger = logging.getLogger(name)

    # Idempotent: several call sites construct a new instance (and thus call
    # this) per pipeline run/CV fold. Without this guard, every call cleared
    # existing handlers and opened a fresh timestamped log file, scattering
    # one continuous run's log across dozens of files.
    if logger.handlers:
        return logger

    logger.setLevel(level)

    # Create log directory
    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)

    # File handler with rotation
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_path / f"automl_{timestamp}.log"
    file_handler = RotatingFileHandler(log_file, maxBytes=10 * 1024 * 1024, backupCount=5)  # 10MB
    file_handler.setLevel(level)

    # Console handler
    if console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)

    # Formatter
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    file_handler.setFormatter(formatter)
    if console:
        console_handler.setFormatter(formatter)

    # Add handlers
    logger.addHandler(file_handler)
    if console:
        logger.addHandler(console_handler)

    return logger


def get_logger(name: str = None) -> logging.Logger:
    """
    Get or create a logger instance.

    This is a convenience function that returns a logger with the module name.
    If the logger doesn't have handlers, it sets up basic logging.

    Args:
        name: Logger name (typically __name__ from calling module)

    Returns:
        Logger instance
    """
    if name is None:
        name = "automl_insight"

    logger = logging.getLogger(name)

    # If logger has no handlers, set up basic configuration
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(logging.INFO)
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    return logger
