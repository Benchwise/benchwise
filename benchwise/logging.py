"""
Benchwise Logging Configuration

Provides centralized logging setup with sensible defaults.
"""

import logging
import sys
from typing import Optional


def setup_logging(
    level: str = "INFO",
    format: Optional[str] = None,
    filename: Optional[str] = None,
) -> logging.Logger:
    """
    Setup logging for Benchwise.
    
    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        format: Custom log format string
        filename: Optional file to write logs to
    
    Returns:
        Configured logger instance
    """
    
    # Default format
    if format is None:
        format = "[%(asctime)s] %(levelname)s [%(name)s] %(message)s"
    
    # Configure root logger
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format=format,
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            logging.StreamHandler(sys.stdout),
        ]
    )
    
    # Add file handler if filename provided
    if filename:
        file_handler = logging.FileHandler(filename)
        file_handler.setFormatter(logging.Formatter(format))
        logging.getLogger("benchwise").addHandler(file_handler)
    
    # Get benchwise logger
    logger = logging.getLogger("benchwise")
    logger.setLevel(getattr(logging, level.upper()))
    
    logger.debug(f"Logging initialized at {level} level")
    
    return logger


def get_logger(name: str = "benchwise") -> logging.Logger:
    """
    Get a logger instance for Benchwise.
    
    Args:
        name: Logger name (default: "benchwise")
    
    Returns:
        Logger instance
    """
    return logging.getLogger(name)


def set_log_level(level: str):
    """
    Change the log level for all Benchwise loggers.
    
    Args:
        level: New log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    """
    logger = logging.getLogger("benchwise")
    logger.setLevel(getattr(logging, level.upper()))
    logger.info(f"Log level changed to {level}")
