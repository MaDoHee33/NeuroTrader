import logging
import logging.handlers
import sys
import os
from pathlib import Path

def setup_logger(name, log_file=None, level=logging.INFO):
    """Function to setup as many loggers as you want"""
    
    formatter = logging.Formatter('%(asctime)s %(levelname)s [%(name)s] %(message)s')
    
    # Check if logs directory exists, if not create it assuming standard structure
    if log_file:
        log_path = Path(log_file).parent
        log_path.mkdir(parents=True, exist_ok=True)

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)

    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.addHandler(handler)
    
    if log_file:
        file_handler = logging.handlers.RotatingFileHandler(
            log_file, maxBytes=10*1024*1024, backupCount=5
        )
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger

def get_logger(name):
    # Default to a safe path if not configured yet, assuming running from root
    log_dir = Path("data/logs")
    log_dir.mkdir(parents=True, exist_ok=True)
    return setup_logger(name, log_file=str(log_dir / "neurotrader.log"))
