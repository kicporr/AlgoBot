"""Logging setup using loguru."""

import sys
from pathlib import Path
from loguru import logger


def setup_logger(config: dict):
    """Configure loguru logger with file rotation and formatting."""
    logger.remove()  # Remove default handler
    
    log_level = config.get("bot", {}).get("log_level", "INFO")
    logs_dir = Path(config.get("paths", {}).get("logs_dir", "./logs"))
    logs_dir.mkdir(parents=True, exist_ok=True)
    
    # Console output
    logger.add(
        sys.stderr,
        level=log_level,
        format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>"
    )
    
    # File output with rotation
    logger.add(
        logs_dir / "bot_{time:YYYY-MM-DD}.log",
        level="DEBUG",
        rotation="10 MB",
        retention="30 days",
        format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {name}:{function}:{line} | {message}"
    )
    
    # Trade-specific log
    logger.add(
        logs_dir / "trades_{time:YYYY-MM-DD}.log",
        level="INFO",
        rotation="1 day",
        retention="90 days",
        filter=lambda record: record["extra"].get("trade", False),
        format="{time:YYYY-MM-DD HH:mm:ss} | {message}"
    )
    
    return logger
