import logging
import sys
from logging.handlers import RotatingFileHandler
import colorlog
from .settings import LOG_FILE

# Use colorlog for more readable development logs and add more structure.

# --- Configuration ---
LOG_LEVEL = logging.DEBUG
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s (%(filename)s:%(lineno)d)"
CONSOLE_LOG_FORMAT = "%(log_color)s%(levelname)-8s%(reset)s %(blue)s%(message)s"
MAX_BYTES = 10 * 1024 * 1024  # 10 MB
BACKUP_COUNT = 5

# --- Get Root Logger ---
logger = logging.getLogger("pair_trading_system")
logger.setLevel(LOG_LEVEL)
logger.propagate = False  # Prevent duplicate logs

# --- Console Handler ---
if not any(isinstance(h, colorlog.StreamHandler) for h in logger.handlers):
    console_handler = colorlog.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO) # Keep console output concise
    console_formatter = colorlog.ColoredFormatter(
        CONSOLE_LOG_FORMAT,
        log_colors={
            'DEBUG':    'cyan',
            'INFO':     'green',
            'WARNING':  'yellow',
            'ERROR':    'red',
            'CRITICAL': 'red,bg_white',
        }
    )
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)

# --- File Handler ---
if not any(isinstance(h, RotatingFileHandler) for h in logger.handlers):
    file_handler = RotatingFileHandler(
        LOG_FILE,
        maxBytes=MAX_BYTES,
        backupCount=BACKUP_COUNT
    )
    file_handler.setLevel(logging.DEBUG) # Log everything to the file
    file_formatter = logging.Formatter(LOG_FORMAT)
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)