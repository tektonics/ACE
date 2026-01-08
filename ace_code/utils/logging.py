"""Logging utilities for ACE-Code."""

import sys
from typing import Optional
from loguru import logger


def setup_logging(
    level: str = "INFO",
    log_file: Optional[str] = None,
    format_string: Optional[str] = None,
) -> None:
    """
    Configure logging for ACE-Code.

    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Optional file path for log output
        format_string: Custom format string for log messages
    """
    # Remove default handler
    logger.remove()

    # Default format
    if format_string is None:
        format_string = (
            "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
            "<level>{message}</level>"
        )

    # Add stdout handler
    logger.add(
        sys.stdout,
        format=format_string,
        level=level,
        colorize=True,
    )

    # Add file handler if specified
    if log_file:
        logger.add(
            log_file,
            format=format_string,
            level=level,
            rotation="10 MB",
            retention="1 week",
        )


def get_logger(name: str = "ace_code"):
    """
    Get a logger instance with the given name.

    Args:
        name: Logger name for context

    Returns:
        Logger instance bound with the given name
    """
    return logger.bind(name=name)
