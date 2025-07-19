from abc import ABC, abstractmethod
import pandas as pd
from typing import Dict, List, Tuple, Any

class BaseStrategy(ABC):
    """
    Abstract Base Class for all trading strategies.

    This class defines the essential interface that every strategy must implement to be
    compatible with the backtesting and execution engines.
    """
    def __init__(self, name: str, **kwargs):
        """
        Initializes the strategy with a name and any strategy-specific parameters.
        
        Args:
            name (str): A unique identifier for the strategy.
            **kwargs: Strategy-specific hyperparameters.
        """
        self.name = name
        self.params = kwargs
        self.is_fitted = False

    def fit(self, historical_data: pd.DataFrame):
        """
        Train or initialize the strategy based on historical data.
        This method is called once before the backtest begins.
        """
        self.is_fitted = True
        pass

    @abstractmethod
    def generate_signals(self, current_data_window: pd.DataFrame, portfolio_context: Dict) -> Dict[Tuple[str, str], float]:
        """
        Generate trading signals for the current time step.

        Args:
            current_data_window (pd.DataFrame): A rolling window of historical data.
            portfolio_context (Dict): The current state of the portfolio.

        Returns:
            Dict[Tuple[str, str], float]: A dictionary mapping asset pairs to target positions.
        """
        raise NotImplementedError("The 'generate_signals' method must be implemented by the subclass.")

    def reset(self):
        """
        Resets the internal state of the strategy.
        """
        self.is_fitted = False