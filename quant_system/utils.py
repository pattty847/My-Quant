import os
import sys
import logging
import traceback
from datetime import datetime
from pathlib import Path

# Configure logging levels
DEBUG = logging.DEBUG
INFO = logging.INFO
WARNING = logging.WARNING
ERROR = logging.ERROR
CRITICAL = logging.CRITICAL

# Create logs directory if it doesn't exist
logs_dir = Path("logs")
logs_dir.mkdir(exist_ok=True)

# Create a timestamp for log files
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_file = logs_dir / f"quant_system_{timestamp}.log"

# Configure root logger
logger = logging.getLogger("quant_system")
logger.setLevel(logging.DEBUG)

# Create file handler for logging to file
file_handler = logging.FileHandler(log_file)
file_handler.setLevel(logging.DEBUG)

# Create console handler for logging to console
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setLevel(logging.INFO)

# Create formatters
file_formatter = logging.Formatter(
    "[%(asctime)s] %(levelname)s [%(name)s.%(funcName)s:%(lineno)d] %(message)s"
)
console_formatter = logging.Formatter(
    "[%(levelname)s] %(message)s"
)

# Apply formatters to handlers
file_handler.setFormatter(file_formatter)
console_handler.setFormatter(console_formatter)

# Add handlers to logger
logger.addHandler(file_handler)
logger.addHandler(console_handler)

# Function to log exceptions with traceback
def log_exception(exc_info=None):
    """
    Log an exception with full traceback.
    
    Args:
        exc_info: Exception info from sys.exc_info(). If None, current exception is used.
    """
    if exc_info is None:
        exc_info = sys.exc_info()
    
    if exc_info[0] is not None:
        exc_type, exc_value, exc_tb = exc_info
        tb_str = "".join(traceback.format_exception(exc_type, exc_value, exc_tb))
        logger.error(f"Exception occurred:\n{tb_str}")

# Context manager for error handling
class ErrorHandler:
    def __init__(self, context="operation", exit_on_error=False):
        """
        Context manager for handling and logging errors.
        
        Args:
            context: Description of the operation being performed
            exit_on_error: Whether to exit the program on error
        """
        self.context = context
        self.exit_on_error = exit_on_error
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is not None:
            logger.error(f"Error during {self.context}: {exc_val}")
            log_exception((exc_type, exc_val, exc_tb))
            
            if self.exit_on_error:
                logger.critical(f"Exiting due to error in {self.context}")
                sys.exit(1)
            
            return True  # Suppress the exception
        return False

# Function to set logging level
def set_log_level(level):
    """
    Set the logging level for the console handler.
    
    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    """
    console_handler.setLevel(level)
    logger.info(f"Console log level set to {logging.getLevelName(level)}")

# Function to get a named logger (child of root logger)
def get_logger(name):
    """
    Get a named logger that inherits from the root logger.
    
    Args:
        name: Name for the logger, typically the module name
        
    Returns:
        Logger instance
    """
    return logging.getLogger(f"quant_system.{name}") 