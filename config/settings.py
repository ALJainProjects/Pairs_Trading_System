import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Data directories
DATA_DIR = os.path.join(BASE_DIR, "..", "data")
RAW_DATA_DIR = os.path.join(DATA_DIR, "raw")
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, "processed")

# Database configurations
DATABASE_URI = "sqlite:///pair_trading.db"

# Models/logging
LOG_DIR = os.path.join(BASE_DIR, "..", "logs")
LOG_FILE = os.path.join(LOG_DIR, "pair_trading.log")
MODEL_DIR = os.path.join(BASE_DIR, "..", "models")
