"""Logging configuration."""

import logging
from pathlib import Path


def setup_logging(config: dict) -> logging.Logger:
    """Setup structured logging."""
    log_level = getattr(logging, config["logging"]["level"])
    log_format = config["logging"]["format"]

    # Create logs directory
    log_dir = Path.home() / ".emx-mcp" / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / "server.log"

    # Configure logging
    logging.basicConfig(
        level=log_level,
        format=log_format,
        handlers=[logging.FileHandler(log_file), logging.StreamHandler()],
    )

    logger = logging.getLogger("emx_mcp")
    logger.info("Logging initialized")
    return logger
