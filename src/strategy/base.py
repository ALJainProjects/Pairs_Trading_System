from abc import ABC, abstractmethod
import pandas as pd
from typing import Dict, List, Tuple, Any
import logging  # Import logging for the base class

logger = logging.getLogger(__name__)  # Basic logger for BaseModel itself


class BaseStrategy(ABC):
    """
    Abstract Base Class for all trading strategies.

    This class defines the essential interface that every strategy must implement to be
    compatible with the backtesting and execution engines.
    """

    def __init__(self, name: str, **kwargs: Any):  # More precise type hint for kwargs
        """
        Initializes the strategy with a name and any strategy-specific parameters.

        Args:
            name (str): A unique identifier for the strategy.
            **kwargs: Strategy-specific hyperparameters. These parameters will be stored
                      in a dictionary accessible via `self.params`.
        """
        self.name = name
        self.params: Dict[str, Any] = kwargs  # Explicitly type as Dict[str, Any]
        self.is_fitted: bool = False  # Explicitly type

        logger.info(f"Strategy '{self.name}' initialized with parameters: {self.params}")

    # The fit method is not abstract, allowing strategies that don't need explicit fitting
    # to simply inherit this default behavior.
    def fit(self, historical_data: pd.DataFrame):
        """
        Train or initialize the strategy based on historical data.
        This method is called once before the backtest begins.
        Subclasses should override this method to implement their specific fitting logic.
        If overridden, they should typically call `super().fit(historical_data)`
        to ensure `self.is_fitted` is set to True.

        Args:
            historical_data (pd.DataFrame): The historical data available for fitting.
        """
        self.is_fitted = True
        logger.info(f"Default fit method called for strategy '{self.name}'. is_fitted set to True.")

    @abstractmethod
    def generate_signals(self, current_data_window: pd.DataFrame, portfolio_context: Dict) -> Dict[
        Tuple[str, str], float]:
        """
        Generate trading signals for the current time step.

        This is an abstract method and must be implemented by any concrete subclass.

        Args:
            current_data_window (pd.DataFrame): A rolling window of historical data,
                                                representing the data available up to the current time.
            portfolio_context (Dict): The current state of the portfolio, including holdings, cash, etc.

        Returns:
            Dict[Tuple[str, str], float]: A dictionary where keys are asset pairs (e.g., ('AAPL', 'MSFT'))
                                          and values are target position sizes or directional signals
                                          (e.g., 1.0 for long, -1.0 for short, 0.0 for flat).
        """
        # It's good practice to leave `NotImplementedError` in abstract methods.
        raise NotImplementedError("The 'generate_signals' method must be implemented by the subclass.")

    def reset(self):
        """
        Resets the internal state of the strategy to its initial, unfitted state.
        This method should be called if the strategy needs to be reused or retrained.
        Subclasses should extend this method to reset any specific internal components (e.g., models).
        """
        self.is_fitted = False
        logger.info(f"Strategy '{self.name}' reset to unfitted state.")

    def __repr__(self) -> str:
        """
        Returns a string representation of the strategy.
        """
        return f"{self.__class__.__name__}(name='{self.name}', params={self.params}, is_fitted={self.is_fitted})"