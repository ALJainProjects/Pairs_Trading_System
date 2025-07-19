import os
from pathlib import Path
from dotenv import load_dotenv

# Use pathlib for robust path management and dotenv for environment variables.
# This makes the configuration more secure and portable.

# Load environment variables from a .env file
load_dotenv()

# Project root directory
# BASE_DIR is the directory containing this 'config' folder
BASE_DIR = Path(__file__).resolve().parent.parent

# --- Data Directories ---
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
os.makedirs(RAW_DATA_DIR, exist_ok=True)
os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)

# --- Database Configuration ---
# Default to a local SQLite DB for historical data
DATABASE_URL = os.getenv("DATABASE_URL", f"sqlite:///{BASE_DIR / 'pair_trading_historical.db'}")

# New: Dedicated URL for real-time data database
# Could be a different file, or even an in-memory database for very short-term storage
# For simplicity, let's use another SQLite file for now.
REALTIME_DATABASE_URL = os.getenv("REALTIME_DATABASE_URL", f"sqlite:///{BASE_DIR / 'pair_trading_realtime.db'}")


# --- Logging Directory ---
LOG_DIR = BASE_DIR / "logs"
LOG_FILE = LOG_DIR / "pair_trading.log"
os.makedirs(LOG_DIR, exist_ok=True)

# --- Model & Results Directories ---
MODEL_DIR = BASE_DIR / "models_data"
OPTIMIZATION_DIR = BASE_DIR / "optimization_results"
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(OPTIMIZATION_DIR, exist_ok=True)

# --- Broker API Configuration (for paper/live trading) ---
ALPACA_API_KEY = os.getenv("ALPACA_API_KEY")
ALPACA_SECRET_KEY = os.getenv("ALPACA_SECRET_KEY")
POLYGON_API_KEY = os.getenv("POLYGON_API_KEY") # Added Polygon Key
IS_PAPER_TRADING = os.getenv("IS_PAPER_TRADING", "True").lower() in ('true', '1', 't')

# --- Quant Research Settings ---
RISK_FREE_RATE = float(os.getenv("RISK_FREE_RATE", "0.02")) # Annual risk-free rate for metrics