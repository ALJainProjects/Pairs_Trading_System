# src/strategy/__init__.py

"""
Strategy Module

This package contains modules for developing, optimizing, and backtesting trading
strategies based on equity pair trading. It includes base classes, pair selection
logic, risk management, parameter optimization, and backtesting frameworks.
"""

# Importing necessary classes and functions for easier access
from .base import BaseStrategy
from .pairs_strategy_basic import PairsTrader
from .risk import PairRiskManager
from .optimization import StrategyOptimizer
from .backtest import MultiPairBacktester
