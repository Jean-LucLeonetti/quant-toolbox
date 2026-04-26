import logging
import sys
from typing import Optional

def setup_logger(name: Optional[str] = None, level: int = logging.INFO) -> logging.Logger:
    """
    @brief Sets up a logger with a standard format and console handler.
    
    @param name Name of the logger. Defaults to None (root logger).
    @param level Logging level. Defaults to logging.INFO.
        
    @return The configured logger instance.
    """
    logger = logging.getLogger(name)
    
    # If the logger already has handlers, don't add more (prevents duplicate logs)
    if logger.hasHandlers():
        return logger
        
    logger.setLevel(level)
    
    # Create console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    
    # Create formatter
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(formatter)
    
    # Add handler to logger
    logger.addHandler(console_handler)
    
    return logger

# Convenience instance
logger = setup_logger("Quant")
