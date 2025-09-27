"""
Logging utilities for the methane emissions analysis workflow.
"""

import logging
import os
import sys
from pathlib import Path
from typing import Optional
from datetime import datetime


class Logger:
    """Centralized logging configuration for the analysis workflow."""
    
    _instance = None
    _initialized = False
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        if not self._initialized:
            self.setup_logging()
            Logger._initialized = True
    
    def setup_logging(self, 
                     level: str = "INFO",
                     log_file: Optional[str] = None,
                     log_format: Optional[str] = None):
        """
        Setup logging configuration.
        
        Args:
            level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
            log_file: Path to log file (optional)
            log_format: Custom log format (optional)
        """
        # Create logs directory if it doesn't exist
        log_dir = Path("./logs")
        log_dir.mkdir(exist_ok=True)
        
        # Default log file if not specified
        if log_file is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            log_file = log_dir / f"methane_analysis_{timestamp}.log"
        
        # Default format if not specified
        if log_format is None:
            log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        
        # Configure root logger
        logging.basicConfig(
            level=getattr(logging, level.upper()),
            format=log_format,
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler(sys.stdout)
            ]
        )
        
        # Set specific logger levels for external libraries to reduce noise
        logging.getLogger('rasterio').setLevel(logging.WARNING)
        logging.getLogger('fiona').setLevel(logging.WARNING)
        logging.getLogger('matplotlib').setLevel(logging.WARNING)
        logging.getLogger('urllib3').setLevel(logging.WARNING)
    
    @staticmethod
    def get_logger(name: str) -> logging.Logger:
        """
        Get a logger instance with the specified name.
        
        Args:
            name: Logger name (typically __name__)
            
        Returns:
            Logger instance
        """
        return logging.getLogger(name)


def log_function_call(func):
    """
    Decorator to log function calls with parameters and execution time.
    
    Args:
        func: Function to decorate
        
    Returns:
        Decorated function
    """
    def wrapper(*args, **kwargs):
        logger = Logger.get_logger(func.__module__)
        
        # Log function entry
        logger.info(f"Starting {func.__name__} with args={args}, kwargs={kwargs}")
        
        try:
            start_time = datetime.now()
            result = func(*args, **kwargs)
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            
            logger.info(f"Completed {func.__name__} in {duration:.2f} seconds")
            return result
            
        except Exception as e:
            logger.error(f"Error in {func.__name__}: {str(e)}")
            raise
    
    return wrapper


def log_processing_step(step_name: str, logger: logging.Logger):
    """
    Context manager for logging processing steps.
    
    Args:
        step_name: Name of the processing step
        logger: Logger instance
    """
    class ProcessingStepLogger:
        def __init__(self, step_name: str, logger: logging.Logger):
            self.step_name = step_name
            self.logger = logger
            self.start_time = None
        
        def __enter__(self):
            self.start_time = datetime.now()
            self.logger.info(f"Starting processing step: {self.step_name}")
            return self
        
        def __exit__(self, exc_type, exc_val, exc_tb):
            if exc_type is None:
                duration = (datetime.now() - self.start_time).total_seconds()
                self.logger.info(f"Completed processing step: {self.step_name} in {duration:.2f} seconds")
            else:
                self.logger.error(f"Error in processing step: {self.step_name} - {exc_val}")
    
    return ProcessingStepLogger(step_name, logger)


class ProgressLogger:
    """Logger for tracking progress of long-running operations."""
    
    def __init__(self, total_items: int, step_name: str = "Processing", logger: Optional[logging.Logger] = None):
        """
        Initialize progress logger.
        
        Args:
            total_items: Total number of items to process
            step_name: Name of the processing step
            logger: Logger instance (optional)
        """
        self.total_items = total_items
        self.step_name = step_name
        self.logger = logger or Logger.get_logger(__name__)
        self.processed_items = 0
        self.start_time = datetime.now()
    
    def update(self, increment: int = 1):
        """
        Update progress counter.
        
        Args:
            increment: Number of items processed
        """
        self.processed_items += increment
        percent_complete = (self.processed_items / self.total_items) * 100
        
        # Log progress every 10% or every 1000 items, whichever is more frequent
        log_interval = min(self.total_items // 10, 1000)
        if log_interval == 0 or self.processed_items % log_interval == 0:
            elapsed_time = (datetime.now() - self.start_time).total_seconds()
            items_per_second = self.processed_items / elapsed_time if elapsed_time > 0 else 0
            
            eta_seconds = ((self.total_items - self.processed_items) / items_per_second) if items_per_second > 0 else 0
            eta_minutes = eta_seconds / 60
            
            self.logger.info(
                f"{self.step_name}: {self.processed_items}/{self.total_items} "
                f"({percent_complete:.1f}%) - {items_per_second:.1f} items/sec - "
                f"ETA: {eta_minutes:.1f} minutes"
            )
    
    def complete(self):
        """Mark processing as complete."""
        total_time = (datetime.now() - self.start_time).total_seconds()
        items_per_second = self.total_items / total_time if total_time > 0 else 0
        
        self.logger.info(
            f"{self.step_name} completed: {self.total_items} items in {total_time:.1f} seconds "
            f"({items_per_second:.1f} items/sec)"
        )