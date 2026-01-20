"""
SpeechLab Logging Configuration

Rich-formatted console logging with structured output.
"""

import logging
import sys
from rich.console import Console
from rich.logging import RichHandler

# Create console for rich output
console = Console()


def setup_logging(level: str = "INFO") -> logging.Logger:
    """
    Set up logging with Rich formatting.
    
    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR)
        
    Returns:
        Configured logger instance
    """
    logging.basicConfig(
        level=level,
        format="%(message)s",
        datefmt="[%X]",
        handlers=[
            RichHandler(
                console=console,
                rich_tracebacks=True,
                tracebacks_show_locals=True,
            )
        ],
    )
    
    # Suppress noisy loggers
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
    
    return logging.getLogger("speechlab")


# Create default logger
logger = setup_logging()
