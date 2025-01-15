"""
Session State Management for Pairs Trading Application

This module provides a robust session state management system for the Streamlit-based
pairs trading application. It handles initialization, validation, and updates of all
session state variables.
"""

import streamlit as st
import pandas as pd
from typing import Dict, Any, Optional
from dataclasses import dataclass
from datetime import datetime


@dataclass
class DefaultConfig:
    """Default configuration values for session state."""
    INITIAL_CAPITAL: float = 100000.0
    MAX_POSITION_SIZE: float = 0.05
    STOP_LOSS: float = 0.10
    MAX_DRAWDOWN: float = 0.20
    MAX_CORRELATION: float = 0.70
    LEVERAGE_LIMIT: float = 2.0
    TRANSACTION_COST_BPS: float = 10.0
    MAX_PAIRS: int = 5


class SessionStateManager:
    """Manages Streamlit session state for the pairs trading application."""

    def __init__(self):
        """Initialize the session state manager."""
        self.config = DefaultConfig()
        self._initialize_session_state()

    def _initialize_session_state(self):
        """Initialize all required session state variables with default values."""
        if 'historical_data' not in st.session_state:
            st.session_state.historical_data = pd.DataFrame()

        if 'selected_pairs' not in st.session_state:
            st.session_state.selected_pairs = pd.DataFrame()

        if 'strategy_params' not in st.session_state:
            st.session_state.strategy_params = {
                'type': 'Statistical',
                'params': self._get_default_strategy_params('Statistical')
            }

        if 'risk_params' not in st.session_state:
            st.session_state.risk_params = {
                'max_position_size': self.config.MAX_POSITION_SIZE,
                'stop_loss': self.config.STOP_LOSS,
                'max_drawdown': self.config.MAX_DRAWDOWN,
                'max_correlation': self.config.MAX_CORRELATION,
                'leverage_limit': self.config.LEVERAGE_LIMIT
            }

        if 'backtest_params' not in st.session_state:
            st.session_state.backtest_params = {
                'initial_capital': self.config.INITIAL_CAPITAL,
                'transaction_cost': self.config.TRANSACTION_COST_BPS / 10000,
                'max_pairs': self.config.MAX_PAIRS
            }

        if 'backtest_results' not in st.session_state:
            st.session_state.backtest_results = None

        if 'show_advanced_options' not in st.session_state:
            st.session_state.show_advanced_options = False

        if 'validation_errors' not in st.session_state:
            st.session_state.validation_errors = []

    def _get_default_strategy_params(self, strategy_type: str) -> Dict[str, Any]:
        """Get default parameters for a given strategy type."""
        if strategy_type == "Statistical":
            return {
                'zscore_threshold': 2.0,
                'lookback_window': 63,
                'zscore_window': 21,
                'min_half_life': 5,
                'max_half_life': 126
            }
        elif strategy_type == "Machine Learning":
            return {
                'lookback_windows': [5, 10, 21, 63],
                'zscore_threshold': 2.0,
                'train_size': 252,
                'validation_size': 63
            }
        elif strategy_type == "Deep Learning":
            return {
                'sequence_length': 20,
                'prediction_horizon': 1,
                'zscore_threshold': 2.0,
                'train_size': 252,
                'validation_size': 63
            }
        return {}

    def update_strategy_params(self, strategy_type: str, params: Dict[str, Any]):
        """Update strategy parameters with validation."""
        if self._validate_strategy_params(strategy_type, params):
            st.session_state.strategy_params = {
                'type': strategy_type,
                'params': params
            }
            self._clear_results()

    def update_risk_params(self, params: Dict[str, float]):
        """Update risk management parameters with validation."""
        if self._validate_risk_params(params):
            st.session_state.risk_params = params
            self._clear_results()

    def update_backtest_params(self, params: Dict[str, Any]):
        """Update backtest parameters with validation."""
        if self._validate_backtest_params(params):
            st.session_state.backtest_params = params
            self._clear_results()

    def _validate_strategy_params(self, strategy_type: str, params: Dict[str, Any]) -> bool:
        """Validate strategy parameters."""
        self._clear_validation_errors()

        if strategy_type not in ["Statistical", "Machine Learning", "Deep Learning"]:
            self._add_validation_error(f"Invalid strategy type: {strategy_type}")
            return False

        required_params = set(self._get_default_strategy_params(strategy_type).keys())
        if not required_params.issubset(params.keys()):
            missing = required_params - set(params.keys())
            self._add_validation_error(f"Missing parameters: {missing}")
            return False

        return True

    def _validate_risk_params(self, params: Dict[str, float]) -> bool:
        """Validate risk management parameters."""
        self._clear_validation_errors()

        if params['max_position_size'] <= 0 or params['max_position_size'] > 1:
            self._add_validation_error("Max position size must be between 0 and 1")
            return False

        if params['max_drawdown'] <= 0 or params['max_drawdown'] > 1:
            self._add_validation_error("Max drawdown must be between 0 and 1")
            return False

        if params['leverage_limit'] < 1:
            self._add_validation_error("Leverage limit must be >= 1")
            return False

        return True

    def _validate_backtest_params(self, params: Dict[str, Any]) -> bool:
        """Validate backtest parameters."""
        self._clear_validation_errors()

        if params['initial_capital'] <= 0:
            self._add_validation_error("Initial capital must be positive")
            return False

        if params['transaction_cost'] < 0:
            self._add_validation_error("Transaction cost cannot be negative")
            return False

        if params['max_pairs'] < 1:
            self._add_validation_error("Max pairs must be >= 1")
            return False

        return True

    def _clear_validation_errors(self):
        """Clear all validation errors."""
        st.session_state.validation_errors = []

    def _add_validation_error(self, error: str):
        """Add a validation error."""
        st.session_state.validation_errors.append(error)

    def _clear_results(self):
        """Clear backtest results when parameters change."""
        st.session_state.backtest_results = None

    def get_validation_errors(self) -> list:
        """Get current validation errors."""
        return st.session_state.validation_errors

    def has_required_data(self) -> bool:
        """Check if all required data is available."""
        return (not st.session_state.historical_data.empty and
                not st.session_state.selected_pairs.empty)

    def get_current_state(self) -> Dict[str, Any]:
        """Get current state of all parameters."""
        return {
            'strategy_params': st.session_state.strategy_params,
            'risk_params': st.session_state.risk_params,
            'backtest_params': st.session_state.backtest_params,
            'has_results': st.session_state.backtest_results is not None
        }

    def save_state(self, filename: str):
        """Save current state to a file."""
        state_to_save = {
            'timestamp': datetime.now().isoformat(),
            'strategy_params': st.session_state.strategy_params,
            'risk_params': st.session_state.risk_params,
            'backtest_params': st.session_state.backtest_params
        }
        pd.to_pickle(state_to_save, filename)

    def load_state(self, filename: str):
        """Load state from a file."""
        try:
            loaded_state = pd.read_pickle(filename)
            st.session_state.strategy_params = loaded_state['strategy_params']
            st.session_state.risk_params = loaded_state['risk_params']
            st.session_state.backtest_params = loaded_state['backtest_params']
            self._clear_results()
            return True
        except Exception as e:
            self._add_validation_error(f"Error loading state: {str(e)}")
            return False