"""
Centralized logging configuration for Violence Detection Backend
"""
import logging
import sys
from pathlib import Path
from datetime import datetime
from loguru import logger as loguru_logger
import colorlog

from config import settings


def setup_logger(name: str = None) -> logging.Logger:
    """
    Setup centralized logging with both file and console output
    
    Args:
        name: Logger name (usually __name__)
        
    Returns:
        Configured logger instance
    """
    
    # Remove default loguru handler
    loguru_logger.remove()
    
    # Console handler with colors
    console_handler = colorlog.StreamHandler()
    console_handler.setLevel(getattr(logging, settings.LOG_LEVEL.upper()))
    
    # Color formatter for console
    color_formatter = colorlog.ColoredFormatter(
        "%(log_color)s%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        log_colors={
            'DEBUG': 'cyan',
            'INFO': 'green',
            'WARNING': 'yellow',
            'ERROR': 'red',
            'CRITICAL': 'red,bg_white',
        }
    )
    console_handler.setFormatter(color_formatter)
    
    # File handler
    log_file = settings.LOGS_DIR / f"violence_detection_{datetime.now().strftime('%Y%m%d')}.log"
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.DEBUG)
    
    # File formatter
    file_formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    file_handler.setFormatter(file_formatter)
    
    # Create logger
    logger = logging.getLogger(name or "violence_detection")
    logger.setLevel(logging.DEBUG)
    
    # Add handlers
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    
    # Prevent duplicate logs
    logger.propagate = False
    
    return logger


def get_logger(name: str = None) -> logging.Logger:
    """Get a logger instance"""
    return logging.getLogger(name or "violence_detection")


# Logging decorators
def log_function_call(func):
    """Decorator to log function calls"""
    def wrapper(*args, **kwargs):
        logger = get_logger(func.__module__)
        logger.debug(f"Calling {func.__name__} with args={args}, kwargs={kwargs}")
        try:
            result = func(*args, **kwargs)
            logger.debug(f"{func.__name__} completed successfully")
            return result
        except Exception as e:
            logger.error(f"{func.__name__} failed with error: {e}")
            raise
    return wrapper


def log_async_function_call(func):
    """Decorator to log async function calls"""
    async def wrapper(*args, **kwargs):
        logger = get_logger(func.__module__)
        logger.debug(f"Calling async {func.__name__} with args={args}, kwargs={kwargs}")
        try:
            result = await func(*args, **kwargs)
            logger.debug(f"async {func.__name__} completed successfully")
            return result
        except Exception as e:
            logger.error(f"async {func.__name__} failed with error: {e}")
            raise
    return wrapper


# Performance logging
class PerformanceLogger:
    """Context manager for performance logging"""
    
    def __init__(self, operation_name: str, logger_name: str = None):
        self.operation_name = operation_name
        self.logger = get_logger(logger_name)
        self.start_time = None
    
    def __enter__(self):
        self.start_time = datetime.now()
        self.logger.info(f"Starting {self.operation_name}")
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        duration = (datetime.now() - self.start_time).total_seconds()
        if exc_type is None:
            self.logger.info(f"Completed {self.operation_name} in {duration:.2f}s")
        else:
            self.logger.error(f"Failed {self.operation_name} after {duration:.2f}s: {exc_val}")


# Example usage:
# with PerformanceLogger("video_processing", "core.video_processor"):
#     # Your code here
#     pass
