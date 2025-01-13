# src/data/__init__.py

"""
Data Handling Module

This package contains modules for data acquisition, storage, preprocessing,
feature engineering, and handling live data feeds for the Equity Pair Trading
Research Project.
"""

# Importing necessary modules for easier access
from .downloader import DataDownloader
from .database import DatabaseManager
from .preprocessor import Preprocessor
from .feature_engineering import FeatureEngineer
from .live_data import LiveDataHandler