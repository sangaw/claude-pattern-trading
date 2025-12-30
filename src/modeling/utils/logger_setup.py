"""
Logging Setup
==============

Centralized logging configuration for the trading system.
"""

import logging
import sys
from pathlib import Path
from datetime import datetime


def setup_logging(log_dir: Path, run_id: str = None, level: str = "INFO"):
    """
    Setup centralized logging
    
    Args:
        log_dir: Directory for log files
        run_id: Run identifier (for log filename)
        level: Logging level (DEBUG, INFO, WARNING, ERROR)
    """
    if run_id is None:
        run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create log directory if it doesn't exist
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # Log file path
    log_file = log_dir / f"run_{run_id}.log"
    
    # Configure logging
    log_level = getattr(logging, level.upper(), logging.INFO)
    
    # Create formatters
    file_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    console_formatter = logging.Formatter(
        '%(levelname)s - %(message)s'
    )
    
    # File handler (detailed)
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(file_formatter)
    
    # Console handler (concise)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(log_level)
    console_handler.setFormatter(console_formatter)
    
    # Root logger configuration
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)
    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)
    
    # Suppress verbose logging from third-party libraries
    logging.getLogger('matplotlib').setLevel(logging.WARNING)
    logging.getLogger('PIL').setLevel(logging.WARNING)
    logging.getLogger('urllib3').setLevel(logging.WARNING)
    
    logging.info(f"Logging initialized - Log file: {log_file}")
    
    return log_file

