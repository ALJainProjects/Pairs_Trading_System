# src/strategy/__init__.py

"""
Strategy Module

This package contains modules for developing, optimizing, and backtesting trading
strategies based on equity pair trading. It includes base classes, pair selection
logic, risk management, parameter optimization, and backtesting frameworks.
"""

# Importing necessary classes and functions for easier access
from .base import BaseStrategy
from .static_pairs_strategy import StaticPairsStrategy
from .risk import PortfolioRiskManager
from .optimization import *
from .backtest import Backtester
from .meta_strategy import *
from .advanced_ml_strategy import  *
from .dynamic_pairs_strategy import *
