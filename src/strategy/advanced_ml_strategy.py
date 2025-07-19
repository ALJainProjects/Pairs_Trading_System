import pandas as pd
import numpy as np
from typing import Dict, Tuple

from .base import BaseStrategy
from src.models.deep_learning import TimesNetModel
from src.data.feature_engineering import FeatureEngineer
from config.logging_config import logger

try:
    from arch import arch_model
except ImportError:
    logger.error("arch library not installed. Please run 'pip install arch'.")
    arch_model = None


class AdvancedMLStrategy(BaseStrategy):
    """
    An advanced pairs trading strategy using a TimesNet model to predict spread
    movements and a GARCH model for adaptive risk management.
    """

    def __init__(self,
                 name="AdvancedMLStrategy_TimesNet",
                 lookback_window: int = 60,
                 sequence_length: int = 30,
                 max_pairs: int = 5):

        super().__init__(name, lookback_window=lookback_window, sequence_length=sequence_length, max_pairs=max_pairs)

        self.feature_engineer = FeatureEngineer()
        self.pair_models: Dict[Tuple[str, str], Dict] = {}
        self.tradeable_pairs = []

    def fit(self, historical_data: pd.DataFrame):
        """
        Finds pairs and trains a dedicated TimesNet and GARCH model for each pair.
        """
        self.tradeable_pairs = [('AAPL', 'MSFT'), ('GOOGL', 'GOOG')]

        for pair in self.tradeable_pairs:
            asset1, asset2 = pair
            logger.info(f"Fitting TimesNet models for pair: {pair}")

            pair_data = historical_data[[asset1, asset2]].copy()
            pair_data['spread'] = pair_data[asset1] - pair_data[asset2]

            # 1. Train GARCH model for volatility forecasting
            spread_returns = pair_data['spread'].pct_change().dropna() * 100
            if arch_model is not None and not spread_returns.empty:
                garch = arch_model(spread_returns, vol='Garch', p=1, q=1)
                garch_fit = garch.fit(update_freq=10, disp='off')
            else:
                garch_fit = None

            # 2. Train TimesNet model to predict future spread
            features = pd.DataFrame({'Adj_Close': pair_data['spread']}).dropna()
            target = features['Adj_Close'].shift(-1)

            features, target = features.align(target, join='inner', axis=0)

            timesnet = TimesNetModel(f"TimesNet_{asset1}_{asset2}")
            X, y = TimesNetModel.create_sequences(features, target, self.params['sequence_length'])

            if len(X) < 20:
                logger.warning(f"Skipping model training for {pair} due to insufficient data.")
                continue

            split_idx = int(len(X) * 0.8)
            X_train, X_val = X[:split_idx], X[split_idx:]
            y_train, y_val = y[:split_idx], y[split_idx:]

            timesnet.build_model(input_shape=(X_train.shape[1], X_train.shape[2]))
            timesnet.train(X_train, y_train, X_val, y_val, epochs=50, batch_size=32)

            self.pair_models[pair] = {'timesnet': timesnet, 'garch': garch_fit}

        self.is_fitted = True

    def generate_signals(self, current_data_window: pd.DataFrame, portfolio_context: Dict) -> Dict[
        Tuple[str, str], float]:
        signals = {}
        for pair, models in self.pair_models.items():
            asset1, asset2 = pair

            vol_forecast = 1.0
            if models['garch']:
                spread_returns = (current_data_window[asset1] - current_data_window[asset2]).pct_change().dropna() * 100
                if not spread_returns.empty:
                    vol_forecast = models['garch'].forecast(horizon=1).variance.iloc[-1, 0]

            features = pd.DataFrame({'Adj_Close': current_data_window[asset1] - current_data_window[asset2]})
            features.dropna(inplace=True)

            if len(features) < self.params['sequence_length']: continue

            sequence = features.tail(self.params['sequence_length']).values.reshape(1, self.params['sequence_length'],
                                                                                    1)
            predicted_spread = models['timesnet'].predict(sequence)[0][0]

            current_spread = features['Adj_Close'].iloc[-1]
            predicted_change = predicted_spread - current_spread

            entry_threshold = 0.5 * np.sqrt(vol_forecast)

            if predicted_change > entry_threshold:
                signals[pair] = 1.0
            elif predicted_change < -entry_threshold:
                signals[pair] = -1.0
            else:
                signals[pair] = 0.0

        return signals