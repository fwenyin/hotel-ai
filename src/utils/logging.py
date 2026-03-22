"""Centralized logging configuration."""

import logging

from src.utils.config import load_config


def setup_logging() -> None:
    """Configure logging from config.yaml settings."""
    config = load_config("config/config.yaml")

    logging_config = config["logging"]
    log_format = logging_config["format"]
    log_level = logging_config["level"]
    logging.basicConfig(level=log_level, format=log_format, force=True)
