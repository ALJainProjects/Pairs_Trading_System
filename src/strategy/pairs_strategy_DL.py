"""
Deep Learning Pairs Trading Strategy Module

Implements a pairs trading strategy using deep learning models with full integration
into the backtesting system, proper risk management, and optimization support.
"""

from typing import Dict, List, Tuple, Optional, Any
import pandas as pd
import numpy as np
from dataclasses import dataclass, field
from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm
from statsmodels.tsa.stattools import coint
from itertools import combinations

from config.settings import DATA_DIR
from src.strategy.base import BaseStrategy
from src.models import DeepLearningModel
from config.logging_config import logger
from pathlib import Path
import json
import plotly.graph_objects as go
from plotly.subplots import make_subplots


@dataclass
class DLPairPosition:
    """Track position details for a deep learning pair trade."""
    asset1: str
    asset2: str
    quantity: float
    entry_price1: float
    entry_price2: float
    entry_date: pd.Timestamp
    hedge_ratio: float
    entry_confidence: float
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    entry_spread: float = 0.0
    peak_pnl: float = 0.0
    current_drawdown: float = 0.0
    transaction_costs: float = 0.0
    model_confidence: float = 0.0
    trades: List[Dict] = field(default_factory=list)

    def update_metrics(self,
                      current_price1: float,
                      current_price2: float,
                      current_confidence: float) -> None:
        """Update position metrics including model confidence."""
        current_spread = current_price1 - self.hedge_ratio * current_price2
        spread_pnl = self.quantity * (current_spread - self.entry_spread)

        self.peak_pnl = max(self.peak_pnl, spread_pnl)
        self.current_drawdown = (self.peak_pnl - spread_pnl) / abs(self.peak_pnl) if self.peak_pnl != 0 else 0
        self.model_confidence = current_confidence


class PairsTradingDL(BaseStrategy):
    """Deep learning based pairs trading strategy with full backtesting integration."""

    def __init__(
            self,
            sequence_length: int = 20,
            prediction_horizon: int = 1,
            zscore_threshold: float = 2.0,
            min_confidence: float = 0.6,
            max_position_size: float = 0.1,
            stop_loss: float = 0.02,
            take_profit: float = 0.04,
            max_drawdown: float = 0.2,
            max_pairs: int = 10,
            transaction_cost: float = 0.001,
            model_dir: Optional[Path] = None
    ):
        """Initialize the strategy."""
        super().__init__(
            name="PairsTradingDL",
            max_position_size=max_position_size
        )

        self.pair_models = None
        self.validation_size = 0.2
        self.sequence_length = sequence_length
        self.prediction_horizon = prediction_horizon
        self.zscore_threshold = zscore_threshold
        self.min_confidence = min_confidence
        self.stop_loss = stop_loss
        self.take_profit = take_profit
        self.max_drawdown = max_drawdown
        self.max_pairs = max_pairs
        self.transaction_cost = transaction_cost
        self.model_dir = model_dir or Path("pairs_trading_DL_outputs/dl_pairs")

        self.spread_predictor = DeepLearningModel()
        self.signal_classifier = DeepLearningModel()
        self.feature_scaler = StandardScaler()

        self.positions: Dict[Tuple[str, str], DLPairPosition] = {}
        self.pairs: List[Tuple[str, str]] = []
        self.hedge_ratios: Dict[Tuple[str, str], float] = {}
        self.current_portfolio_value: float = 0.0

        self.validate_parameters()
        self._setup_models()

    @classmethod
    def get_default_parameters(cls) -> Dict[str, Any]:
        """
        Return a dictionary of default parameters for PairsTradingDL.
        """
        return {
            "sequence_length": 20,
            "prediction_horizon": 1,
            "zscore_threshold": 2.0,
            "min_confidence": 0.6,
            "max_position_size": 0.1,
            "stop_loss": 0.02,
            "take_profit": 0.04,
            "max_drawdown": 0.2,
            "max_pairs": 10,
            "transaction_cost": 0.001
        }

    def clear_state(self) -> None:
        """Clear strategy state between runs instead of using reset."""
        self.correlation_analyzer = None
        self.pairs = []
        self.positions = {}
        self.hedge_ratios = {}
        self._portfolio_value = 0.0

    def validate_parameters(self) -> None:
        """Validate strategy parameters."""
        if self.sequence_length < 1:
            raise ValueError("sequence_length must be positive")
        if not 0 < self.zscore_threshold:
            raise ValueError("zscore_threshold must be positive")
        if not 0 < self.min_confidence < 1:
            raise ValueError("min_confidence must be between 0 and 1")
        if not 0 < self.max_position_size <= 1:
            raise ValueError("max_position_size must be between 0 and 1")
        if not 0 < self.stop_loss < 1:
            raise ValueError("stop_loss must be between 0 and 1")
        if not 0 < self.take_profit < 1:
            raise ValueError("take_profit must be between 0 and 1")

    def calculate_position_size(self,
                             pair: Tuple[str, str],
                             prices: pd.DataFrame,
                             confidence: float) -> float:
        """
        Calculate position size based on model confidence and constraints.
        Args:
            pair: Asset pair
            prices: Price data
            confidence: Model confidence score
        Returns:
            float: Position size in units
        """
        asset1, asset2 = pair
        price1 = prices[asset1].iloc[-1]
        price2 = prices[asset2].iloc[-1]

        spread = prices[asset1] - prices[asset2]
        spread_vol = spread.rolling(window=self.sequence_length).std().iloc[-1]

        vol_adjustment = 1.0 / (1.0 + spread_vol) if spread_vol > 0 else 1.0
        position_size = self.max_position_size * confidence * vol_adjustment

        max_trade_value = self.current_portfolio_value * position_size
        pair_value = price1 + price2

        position_size = (max_trade_value / pair_value)

        position_size *= (1 - self.transaction_cost)

        return position_size

    def find_trading_pairs(self, prices: pd.DataFrame) -> List[Tuple[str, str]]:
        """Find suitable trading pairs using correlation and cointegration."""
        valid_pairs = []
        n_assets = len(prices.columns)

        for i, j in combinations(range(n_assets), 2):
            asset1, asset2 = prices.columns[i], prices.columns[j]

            correlation = prices[asset1].corr(prices[asset2])
            if abs(correlation) < 0.5:
                continue

            _, pvalue,_ = coint(prices[asset1], prices[asset2])
            if pvalue > 0.05:
                continue

            model = sm.OLS(prices[asset1], prices[asset2]).fit()
            hedge_ratio = model.params[0]

            valid_pairs.append((asset1, asset2))
            self.hedge_ratios[(asset1, asset2)] = hedge_ratio

            if len(valid_pairs) >= self.max_pairs:
                break

        self.pairs = valid_pairs
        return valid_pairs

    def predict_signals(self, pair_data: pd.DataFrame) -> pd.DataFrame:
        """Generate trading predictions using the trained models."""
        X_spread, _ = self.prepare_sequences(
            pair_data,
            'spread',
            feature_columns=['spread', 'ratio', 'return1', 'return2']
        )

        X_spread_scaled = self.feature_scaler.transform(X_spread.reshape(-1, X_spread.shape[-1]))
        X_spread_scaled = X_spread_scaled.reshape(X_spread.shape)

        spread_pred = self.spread_predictor.predict(X_spread_scaled)
        signal_prob = self.signal_classifier.predict(X_spread_scaled)

        predictions = pd.DataFrame(index=pair_data.index[self.sequence_length:])
        predictions['spread_prediction'] = spread_pred
        predictions['signal_probability'] = signal_prob

        zscore = (pair_data['spread'] - pair_data['spread_ma']) / pair_data['spread_std']
        predictions['predicted_signal'] = 0

        predictions.loc[(zscore < -self.zscore_threshold) &
                       (predictions['signal_probability'] > self.min_confidence),
                       'predicted_signal'] = 1

        predictions.loc[(zscore > self.zscore_threshold) &
                       (predictions['signal_probability'] > self.min_confidence),
                       'predicted_signal'] = -1

        return predictions

    def initialize_models(self, train_data: pd.DataFrame) -> None:
        """
        Initialize and train deep learning models for each pair.

        Args:
            train_data: DataFrame with asset returns for training period
        """
        logger.info("Initializing deep learning models for pairs trading")

        if not self.pairs:
            self.pairs = self.find_trading_pairs(train_data)

        for pair in self.pairs:
            try:
                asset1, asset2 = pair
                logger.info(f"Training models for pair {asset1}-{asset2}")

                pair_data = self.prepare_pair_data(
                    stock1_prices=train_data[asset1],
                    stock2_prices=train_data[asset2],
                    start_idx=self.sequence_length
                )

                X_spread, y_spread = self.spread_predictor.prepare_sequences(
                    data=pair_data,
                    target_column='spread',
                    feature_columns=['spread', 'ratio', 'return1', 'return2',
                                     'spread_ma', 'spread_std', 'spread_zscore',
                                     'momentum1', 'momentum2', 'spread_vol',
                                     'rolling_corr'],
                    sequence_length=self.sequence_length
                )

                train_size = len(X_spread) - int(self.validation_size * len(X_spread))
                X_train_spread = X_spread[:train_size]
                X_val_spread = X_spread[train_size:]
                y_train_spread = y_spread[:train_size]
                y_val_spread = y_spread[train_size:]

                self.spread_predictor.build_lstm_model(
                    input_shape=(self.sequence_length, X_spread.shape[2]),
                    lstm_units=[128, 64],
                    dense_units=[32],
                    dropout_rate=0.2
                )

                self.spread_predictor.train_model(
                    X_train_spread, y_train_spread,
                    X_val_spread, y_val_spread,
                    epochs=100,
                    batch_size=32,
                    patience=10,
                    model_prefix=f"spread_predictor_{asset1}_{asset2}"
                )

                predictions = self.predict_signals(pair_data)

                X_signal, y_signal = self.signal_classifier.prepare_sequences(
                    data=pair_data,
                    target_column='predicted_signal',
                    feature_columns=[
                        'RSI_stock1', 'RSI_stock2',
                        'MACD_stock1', 'MACD_stock2',
                        'BB_Upper_stock1', 'BB_Lower_stock1',
                        'BB_Upper_stock2', 'BB_Lower_stock2',
                        'SIMPLE_MA_20_stock1', 'SIMPLE_MA_20_stock2',
                        'EXP_MA_20_stock1', 'EXP_MA_20_stock2',
                
                        'Close_stock1', 'Close_stock2', 'Volume_stock2', 'Volume_stock1',
                        'Volume_ROC_stock1', 'Volume_ROC_stock2',
                
                        'rolling_corr',
                        'spread_zscore'
                    ], 
                    sequence_length=self.sequence_length
                )

                X_train_signal = X_signal[:train_size]
                X_val_signal = X_signal[train_size:]
                y_train_signal = y_signal[:train_size]
                y_val_signal = y_signal[train_size:]

                self.signal_classifier.build_lstm_model(
                    input_shape=(self.sequence_length, X_signal.shape[2]),
                    lstm_units=[64, 32],
                    dense_units=[16],
                    dropout_rate=0.2
                )

                self.signal_classifier.train_model(
                    X_train_signal, y_train_signal,
                    X_val_signal, y_val_signal,
                    epochs=100,
                    batch_size=32,
                    patience=10,
                    model_prefix=f"signal_classifier_{asset1}_{asset2}"
                )

                self.pair_models[pair] = {
                    'spread_predictor': self.spread_predictor,
                    'signal_classifier': self.signal_classifier,
                    'features': pair_data.columns.tolist(),
                    'metrics': {
                        'spread_predictor': self.spread_predictor.evaluate_model(
                            X_val_spread, y_val_spread, task='regression'
                        ),
                        'signal_classifier': self.signal_classifier.evaluate_model(
                            X_val_signal, y_val_signal, task='classification'
                        )
                    }
                }

                logger.info(f"Successfully trained models for {asset1}-{asset2}")

            except Exception as e:
                logger.error(f"Error training models for pair {pair}: {str(e)}")
                continue

        if not self.pair_models:
            raise ValueError("No models could be trained successfully")

    def _setup_models(self) -> None:
        """Initialize and configure the deep learning models."""
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.spread_model_path = self.model_dir / "spread_predictor"
        self.signal_model_path = self.model_dir / "signal_classifier"

        try:
            self.spread_predictor.load_model(str(self.spread_model_path))
            self.signal_classifier.load_model(str(self.signal_model_path))
            logger.info("Loaded existing models")
        except (FileNotFoundError, ValueError) as e:
            logger.warning(f"Could not load existing models: {e}")
            logger.info("Models will be trained on first data")

    def reset(self) -> None:
        """Reset strategy state between optimization runs."""
        self.clear_state()
        self.positions = {}
        self.pairs = []
        self.hedge_ratios = {}
        self.current_portfolio_value = 0.0

        self.spread_predictor = DeepLearningModel()
        self.signal_classifier = DeepLearningModel()
        self.feature_scaler = StandardScaler()

        self._setup_models()

    @property
    def optimization_parameters(self) -> Dict:
        """Define parameters for optimization."""
        return {
            'sequence_length': (10, 50),
            'zscore_threshold': (1.0, 3.0),
            'min_confidence': (0.6, 0.9),
            'stop_loss': (0.01, 0.05),
            'take_profit': (0.02, 0.08)
        }

    def prepare_pair_data(self,
                          stock1_prices: pd.Series,
                          stock2_prices: pd.Series,
                          stock1_volumes: pd.Series = None,
                          stock2_volumes: pd.Series = None,
                          start_idx: Optional[int] = None) -> pd.DataFrame:
        """
        Prepare pair data using expanding windows to prevent look-ahead bias.
        Includes enhanced volume-based features.

        Args:
            stock1_prices: First stock prices
            stock2_prices: Second stock prices
            stock1_volumes: First stock volumes
            stock2_volumes: Second stock volumes
            start_idx: Starting index for calculations

        Returns:
            DataFrame with features
        """
        if start_idx is None:
            start_idx = self.sequence_length

        spread = stock1_prices - stock2_prices
        ratio = stock1_prices / stock2_prices
        log_ratio = np.log(ratio)

        data = pd.DataFrame({
            'spread': spread,
            'ratio': ratio,
            'log_ratio': log_ratio,
            'return1': stock1_prices.pct_change(),
            'return2': stock2_prices.pct_change()
        })

        if stock1_volumes is not None and stock2_volumes is not None:
            vol1_ma = stock1_volumes.rolling(window=20).mean()
            vol2_ma = stock2_volumes.rolling(window=20).mean()
            norm_vol1 = stock1_volumes / vol1_ma
            norm_vol2 = stock2_volumes / vol2_ma

            data['volume_ratio'] = norm_vol1 / norm_vol2
            data['log_volume_ratio'] = np.log(data['volume_ratio'])

            dollar_vol1 = stock1_prices * stock1_volumes
            dollar_vol2 = stock2_prices * stock2_volumes

            data['dollar_volume_ratio'] = dollar_vol1 / dollar_vol2
            data['log_dollar_volume_ratio'] = np.log(data['dollar_volume_ratio'])

            for window in [5, 10, 20]:
                data[f'vol_mom1_{window}'] = norm_vol1.pct_change(window)
                data[f'vol_mom2_{window}'] = norm_vol2.pct_change(window)

                data[f'vol_std1_{window}'] = norm_vol1.rolling(window).std()
                data[f'vol_std2_{window}'] = norm_vol2.rolling(window).std()

                data[f'vol_ratio_mom_{window}'] = data['volume_ratio'].pct_change(window)

                data[f'dollar_vol_mom1_{window}'] = dollar_vol1.pct_change(window)
                data[f'dollar_vol_mom2_{window}'] = dollar_vol2.pct_change(window)

                vwap1 = (dollar_vol1.rolling(window).sum() /
                         stock1_volumes.rolling(window).sum())
                vwap2 = (dollar_vol2.rolling(window).sum() /
                         stock2_volumes.rolling(window).sum())

                data[f'vwap_ratio_{window}'] = vwap1 / vwap2
                data[f'vwap_spread_{window}'] = vwap1 - vwap2

                data[f'vol_weighted_ret1_{window}'] = (
                        data['return1'] * norm_vol1
                ).rolling(window).mean()
                data[f'vol_weighted_ret2_{window}'] = (
                        data['return2'] * norm_vol2
                ).rolling(window).mean()

                data[f'vol_price_corr1_{window}'] = (
                    pd.DataFrame({'price': stock1_prices, 'volume': norm_vol1})
                    .rolling(window)
                    .corr()
                    .unstack()
                    .iloc[:, 1]
                )
                data[f'vol_price_corr2_{window}'] = (
                    pd.DataFrame({'price': stock2_prices, 'volume': norm_vol2})
                    .rolling(window)
                    .corr()
                    .unstack()
                    .iloc[:, 1]
                )

                data[f'vol_spread_{window}'] = spread * np.sqrt(norm_vol1 * norm_vol2)
                data[f'vol_spread_ma_{window}'] = (
                    data[f'vol_spread_{window}'].rolling(window).mean()
                )
                data[f'vol_spread_std_{window}'] = (
                    data[f'vol_spread_{window}'].rolling(window).std()
                )

                data[f'vol_trend1_{window}'] = (
                    norm_vol1.rolling(window)
                    .apply(lambda x: (np.sum(x > x.mean()) / len(x)))
                )
                data[f'vol_trend2_{window}'] = (
                    norm_vol2.rolling(window)
                    .apply(lambda x: (np.sum(x > x.mean()) / len(x)))
                )

                data[f'vol_dispersion1_{window}'] = (
                        norm_vol1.rolling(window).std() /
                        norm_vol1.rolling(window).mean()
                )
                data[f'vol_dispersion2_{window}'] = (
                        norm_vol2.rolling(window).std() /
                        norm_vol2.rolling(window).mean()
                )

                data[f'abnormal_vol1_{window}'] = (
                                                          norm_vol1 - norm_vol1.rolling(window).mean()
                                                  ) / norm_vol1.rolling(window).std()
                data[f'abnormal_vol2_{window}'] = (
                                                          norm_vol2 - norm_vol2.rolling(window).mean()
                                                  ) / norm_vol2.rolling(window).std()

                data[f'vol_weighted_spread_{window}'] = (
                        spread * (norm_vol1 + norm_vol2) / 2
                ).rolling(window).mean()

                delta_vol1 = norm_vol1.diff()
                delta_vol2 = norm_vol2.diff()

                gain1 = (delta_vol1.where(delta_vol1 > 0, 0)
                         .rolling(window).mean())
                loss1 = (-delta_vol1.where(delta_vol1 < 0, 0)
                         .rolling(window).mean())
                gain2 = (delta_vol2.where(delta_vol2 > 0, 0)
                         .rolling(window).mean())
                loss2 = (-delta_vol2.where(delta_vol2 < 0, 0)
                         .rolling(window).mean())

                rs1 = gain1 / loss1
                rs2 = gain2 / loss2

                data[f'vol_rsi1_{window}'] = 100 - (100 / (1 + rs1))
                data[f'vol_rsi2_{window}'] = 100 - (100 / (1 + rs2))

                data[f'high_vol_regime_{window}'] = (
                        (data[f'abnormal_vol1_{window}'] > 1.5) &
                        (data[f'abnormal_vol2_{window}'] > 1.5)
                ).astype(float)

                data[f'vol_accel1_{window}'] = data[f'vol_mom1_{window}'].diff()
                data[f'vol_accel2_{window}'] = data[f'vol_mom2_{window}'].diff()

        for window in [5, 10, 20, 60]:
            data[f'spread_ma_{window}'] = spread.rolling(window).mean()
            data[f'spread_std_{window}'] = spread.rolling(window).std()
            data[f'spread_zscore_{window}'] = (
                    (spread - data[f'spread_ma_{window}']) /
                    data[f'spread_std_{window}']
            )

            data[f'ratio_ma_{window}'] = ratio.rolling(window).mean()
            data[f'ratio_std_{window}'] = ratio.rolling(window).std()
            data[f'ratio_zscore_{window}'] = (
                    (ratio - data[f'ratio_ma_{window}']) /
                    data[f'ratio_std_{window}']
            )

            data[f'log_ratio_ma_{window}'] = log_ratio.rolling(window).mean()
            data[f'log_ratio_std_{window}'] = log_ratio.rolling(window).std()

            data[f'return_corr_{window}'] = (
                pd.DataFrame({'ret1': data['return1'], 'ret2': data['return2']})
                .rolling(window)
                .corr()
                .unstack()
                .iloc[:, 1]
            )

            data[f'momentum1_{window}'] = stock1_prices.pct_change(window)
            data[f'momentum2_{window}'] = stock2_prices.pct_change(window)

            rolling_cov = (
                data['return1']
                .rolling(window)
                .cov(data['return2'])
            )
            rolling_var = data['return2'].rolling(window).var()
            data[f'beta_{window}'] = rolling_cov / rolling_var

        data = data.dropna()

        if start_idx is not None:
            data = data.iloc[start_idx:]

        return data

    def prepare_sequences(self,
                         data: pd.DataFrame,
                         target_column: str,
                         feature_columns: Optional[List[str]] = None) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare sequences for model training."""
        if feature_columns is None:
            feature_columns = data.columns.tolist()
            feature_columns.remove(target_column)

        sequences = []
        targets = []

        for i in range(self.sequence_length, len(data) - self.prediction_horizon + 1):
            sequence = data[feature_columns].iloc[i - self.sequence_length:i].values
            target = data[target_column].iloc[i + self.prediction_horizon - 1]

            sequences.append(sequence)
            targets.append(target)

        return np.array(sequences), np.array(targets)

    def generate_signals(self, prices: pd.DataFrame) -> pd.DataFrame:
        """
        Generate trading signals compatible with backtester.

        Args:
            prices: Price data for all assets

        Returns:
            DataFrame with signals (-1, 0, 1) for each pair
        """
        signals = pd.DataFrame(index=prices.index)

        if not self.pairs:
            self.find_trading_pairs(prices)

        for pair in self.pairs:
            try:
                asset1, asset2 = pair
                if asset1 not in prices.columns or asset2 not in prices.columns:
                    continue

                pair_data = self.prepare_pair_data(
                    prices[asset1],
                    prices[asset2]
                )

                predictions = self.predict_signals(pair_data)
                pair_signals = pd.Series(0, index=prices.index)

                signal_mask = predictions['signal_probability'] > self.min_confidence
                pair_signals[signal_mask] = predictions.loc[signal_mask, 'predicted_signal']

                if not pair_signals.empty and pair_signals.iloc[-1] != 0:
                    confidence = predictions.loc[signal_mask, 'signal_probability'].iloc[-1]
                    position_size = self.calculate_position_size(
                        pair,
                        prices,
                        confidence
                    )
                    pair_signals *= position_size

                signals[pair] = pair_signals

            except Exception as e:
                logger.error(f"Error generating signals for {pair}: {str(e)}")
                continue

        return signals

    def update_positions(self,
                        current_prices: pd.Series,
                        timestamp: pd.Timestamp) -> None:
        """
        Update position metrics and check risk limits.

        Args:
            current_prices: Current asset prices
            timestamp: Current timestamp
        """
        for pair, position in list(self.positions.items()):
            asset1, asset2 = pair

            if asset1 not in current_prices or asset2 not in current_prices:
                self._close_position(timestamp, pair, "Assets not available")
                continue

            current_price1 = current_prices[asset1]
            current_price2 = current_prices[asset2]

            pair_data = self.prepare_pair_data(
                pd.Series(current_price1),
                pd.Series(current_price2)
            )
            current_predictions = self.predict_signals(pair_data)
            current_confidence = current_predictions['signal_probability'].iloc[-1]

            position.update_metrics(current_price1, current_price2, current_confidence)

            if position.current_drawdown > self.stop_loss:
                self._close_position(timestamp, pair, "Stop loss triggered")
                continue

            current_spread = current_price1 - position.hedge_ratio * current_price2
            profit_pct = abs(current_spread - position.entry_spread) / abs(position.entry_spread)

            if profit_pct > self.take_profit:
                self._close_position(timestamp, pair, "Take profit triggered")
                continue

            if position.model_confidence < self.min_confidence:
                self._close_position(timestamp, pair, "Low model confidence")

    def _close_position(self,
                        date: pd.Timestamp,
                        pair: Tuple[str, str],
                        reason: str = None) -> None:
        """Close a position and record the trade."""
        if pair in self.positions:
            position = self.positions[pair]

            trade_record = {
                'pair': f"{position.asset1}/{position.asset2}",
                'entry_date': position.entry_date,
                'exit_date': date,
                'entry_spread': position.entry_spread,
                'quantity': position.quantity,
                'pnl': position.peak_pnl,
                'max_drawdown': position.current_drawdown,
                'transaction_costs': position.transaction_costs,
                'entry_confidence': position.entry_confidence,
                'exit_confidence': position.model_confidence,
                'close_reason': reason
            }

            self.trades.append(trade_record)
            del self.positions[pair]

    def save_state(self, path: str) -> None:
        """Save strategy state and models."""
        state_path = Path(path)
        state_path.mkdir(parents=True, exist_ok=True)

        self.spread_predictor.save_model(str(state_path / "spread_predictor.h5"))
        self.signal_classifier.save_model(str(state_path / "signal_classifier.h5"))

        np.savez(
            state_path / "feature_scaler.npz",
            mean=self.feature_scaler.mean_,
            scale=self.feature_scaler.scale_
        )

        state = {
            'pairs': self.pairs,
            'hedge_ratios': self.hedge_ratios,
            'current_portfolio_value': self.current_portfolio_value,
            'parameters': {
                'sequence_length': self.sequence_length,
                'zscore_threshold': self.zscore_threshold,
                'min_confidence': self.min_confidence
            }
        }

        with open(state_path / "strategy_state.json", 'w') as f:
            json.dump(state, f)

    def load_state(self, path: str) -> None:
        """Load strategy state and models."""
        state_path = Path(path)

        self.spread_predictor.load_model(str(state_path / "spread_predictor.h5"))
        self.signal_classifier.load_model(str(state_path / "signal_classifier.h5"))

        scaler_state = np.load(state_path / "feature_scaler.npz", allow_pickle=True).item()
        self.feature_scaler.mean_ = scaler_state['mean']
        self.feature_scaler.scale_ = scaler_state['scale']

        with open(state_path / "strategy_state.json", 'r') as f:
            state = json.load(f)

        self.pairs = state['pairs']
        self.hedge_ratios = state['hedge_ratios']
        self.current_portfolio_value = state['current_portfolio_value']

        loaded_params = state['parameters']
        if (loaded_params['sequence_length'] != self.sequence_length or
            loaded_params['zscore_threshold'] != self.zscore_threshold or
            loaded_params['min_confidence'] != self.min_confidence):
            logger.warning("Loaded state has different parameters than current strategy")

    def get_portfolio_stats(self) -> Dict:
        """Calculate comprehensive portfolio statistics."""
        if not self.trades:
            return {}

        trades_df = pd.DataFrame(self.trades)

        stats = {
            'total_trades': len(trades_df),
            'winning_trades': len(trades_df[trades_df['pnl'] > 0]),
            'total_pnl': trades_df['pnl'].sum(),
            'total_costs': trades_df['transaction_costs'].sum(),
            'max_drawdown': trades_df['max_drawdown'].max(),
            'avg_trade_duration': (trades_df['exit_date'] - trades_df['entry_date']).mean().days,
            'pairs_traded': len(trades_df['pair'].unique())
        }

        stats.update({
            'avg_entry_confidence': trades_df['entry_confidence'].mean(),
            'avg_exit_confidence': trades_df['exit_confidence'].mean(),
            'confidence_correlation_pnl': trades_df['entry_confidence'].corr(trades_df['pnl'])
        })

        if len(trades_df) > 0:
            stats['win_rate'] = stats['winning_trades'] / stats['total_trades']
            stats['avg_profit_per_trade'] = stats['total_pnl'] / stats['total_trades']
            stats['profit_factor'] = (
                trades_df[trades_df['pnl'] > 0]['pnl'].sum() /
                abs(trades_df[trades_df['pnl'] < 0]['pnl'].sum())
                if len(trades_df[trades_df['pnl'] < 0]) > 0 else float('inf')
            )

        reason_counts = trades_df['close_reason'].value_counts()
        for reason, count in reason_counts.items():
            stats[f'closes_{reason.lower().replace(" ", "_")}'] = count

        return stats


def main():
    """Test the deep learning pairs trading strategy and generate comprehensive outputs."""

    output_dir = Path("pairs_trading_strategy_DL_outputs")
    output_dir.mkdir(parents=True, exist_ok=True)

    models_dir = output_dir / "models"
    plots_dir = output_dir / "plots"
    data_dir = output_dir / "data"
    results_dir = output_dir / "results"

    for directory in [models_dir, plots_dir, data_dir, results_dir]:
        directory.mkdir(exist_ok=True)

    try:
        logger.info("Reading data from local CSV files...")
        raw_data = DATA_DIR.replace(r'\config', '')
        raw_data_dir = Path(f"{raw_data}/raw")

        selected_symbols = [
            'AAPL',
            'MSFT',
            'NVDA',
            'AMD',
            'INTC',
            'QCOM',
            'AVGO',
            'ASML',
            'AMAT',
            'MU'
        ]

        logger.info(f"Selected stocks for analysis: {', '.join(selected_symbols)}")

        prices = pd.DataFrame()
        for symbol in selected_symbols:
            try:
                csv_path = raw_data_dir / f"{symbol}.csv"
                if not csv_path.exists():
                    logger.warning(f"Data file not found for {symbol}")
                    continue

                df = pd.read_csv(csv_path)
                if 'Date' not in df.columns or 'Adj_Close' not in df.columns:
                    logger.warning(f"Required columns missing in {symbol}")
                    continue

                df['Date'] = pd.to_datetime(df['Date'])
                df.set_index('Date', inplace=True)
                prices[symbol] = df['Adj_Close']
                logger.debug(f"Successfully loaded data for {symbol}")
            except Exception as e:
                logger.error(f"Error reading data for {symbol}: {str(e)}")
                continue

        if prices.empty:
            raise ValueError("No valid price data loaded")

        prices = prices.ffill().bfill()
        prices.to_csv(data_dir / "price_data.csv")
        logger.info(f"Successfully loaded data for {len(prices.columns)} symbols")

        strategy = PairsTradingDL(
            sequence_length=20,
            prediction_horizon=1,
            zscore_threshold=2.0,
            min_confidence=0.6,
            max_position_size=0.1,
            stop_loss=0.02,
            take_profit=0.04,
            max_drawdown=0.2,
            max_pairs=5,
            model_dir=models_dir
        )

        logger.info("Initializing deep learning models and finding pairs...")
        pairs = strategy.find_trading_pairs(prices)
        if not pairs:
            logger.warning("No valid trading pairs found")
            return None

        pair_analysis = []
        for pair in pairs:
            asset1, asset2 = pair
            pair_data = strategy.prepare_pair_data(
                prices[asset1],
                prices[asset2]
            )
            predictions = strategy.predict_signals(pair_data)

            analysis = {
                'pair': f"{asset1}/{asset2}",
                'hedge_ratio': strategy.hedge_ratios[pair],
                'avg_confidence': predictions['signal_probability'].mean(),
                'signal_ratio': (predictions['predicted_signal'] != 0).mean(),
                'avg_spread': pair_data['spread'].mean(),
                'spread_vol': pair_data['spread_vol'].mean(),
                'correlation': pair_data['rolling_corr'].mean()
            }
            pair_analysis.append(analysis)

        pd.DataFrame(pair_analysis).to_csv(results_dir / "pair_analysis.csv")

        logger.info("Generating trading signals...")
        signals = strategy.generate_signals(prices)
        signals.to_csv(results_dir / "trading_signals.csv")

        logger.info("Creating visualizations...")
        for pair in pairs:
            asset1, asset2 = pair
            pair_data = strategy.prepare_pair_data(prices[asset1], prices[asset2])
            predictions = strategy.predict_signals(pair_data)

            fig = make_subplots(
                rows=4, cols=1,
                subplot_titles=[
                    f'Price Movement: {asset1} vs {asset2}',
                    'Spread with Predictions',
                    'Model Confidence',
                    'Trading Signals'
                ],
                vertical_spacing=0.1
            )

            fig.add_trace(
                go.Scatter(x=prices.index, y=prices[asset1], name=asset1),
                row=1, col=1
            )
            fig.add_trace(
                go.Scatter(x=prices.index, y=prices[asset2], name=asset2),
                row=1, col=1
            )

            fig.add_trace(
                go.Scatter(
                    x=pair_data.index,
                    y=pair_data['spread'],
                    name='Actual Spread'
                ),
                row=2, col=1
            )
            fig.add_trace(
                go.Scatter(
                    x=predictions.index,
                    y=predictions['spread_prediction'],
                    name='Predicted Spread',
                    line=dict(dash='dash')
                ),
                row=2, col=1
            )

            fig.add_trace(
                go.Scatter(
                    x=predictions.index,
                    y=predictions['signal_probability'],
                    name='Model Confidence',
                    line=dict(color='purple')
                ),
                row=3, col=1
            )
            fig.add_hline(
                y=strategy.min_confidence,
                line_dash="dash",
                line_color="red",
                annotation_text="Min Confidence",
                row=3, col=1
            )

            long_signals = signals[pair] > 0
            short_signals = signals[pair] < 0

            if long_signals.any():
                fig.add_trace(
                    go.Scatter(
                        x=signals.index[long_signals],
                        y=pair_data.loc[signals.index[long_signals], 'spread'],
                        mode='markers',
                        marker=dict(color='green', size=10),
                        name='Long Signal'
                    ),
                    row=4, col=1
                )

            if short_signals.any():
                fig.add_trace(
                    go.Scatter(
                        x=signals.index[short_signals],
                        y=pair_data.loc[signals.index[short_signals], 'spread'],
                        mode='markers',
                        marker=dict(color='red', size=10),
                        name='Short Signal'
                    ),
                    row=4, col=1
                )

            fig.update_layout(
                height=1600,
                title_text=f"Deep Learning Analysis: {asset1}/{asset2}",
                showlegend=True
            )

            fig.write_html(plots_dir / f"dl_analysis_{asset1}_{asset2}.html")

        strategy_config = {
            'sequence_length': strategy.sequence_length,
            'prediction_horizon': strategy.prediction_horizon,
            'zscore_threshold': strategy.zscore_threshold,
            'min_confidence': strategy.min_confidence,
            'max_position_size': strategy.max_position_size,
            'stop_loss': strategy.stop_loss,
            'take_profit': strategy.take_profit,
            'max_pairs': strategy.max_pairs
        }

        with open(results_dir / "strategy_config.json", 'w') as f:
            json.dump(strategy_config, f, indent=4)

        with open(results_dir / "analysis_summary.txt", 'w') as f:
            f.write("Deep Learning Pairs Trading Analysis Summary\n")
            f.write("=========================================\n\n")

            f.write(f"Analysis Period: {prices.index[0].date()} to {prices.index[-1].date()}\n")
            f.write(f"Number of assets analyzed: {len(prices.columns)}\n")
            f.write(f"Number of trading pairs found: {len(pairs)}\n\n")

            f.write("Trading Pairs:\n")
            for pair_info in pair_analysis:
                f.write(f"\n{pair_info['pair']}:\n")
                f.write(f"  Hedge Ratio: {pair_info['hedge_ratio']:.4f}\n")
                f.write(f"  Average Confidence: {pair_info['avg_confidence']:.4f}\n")
                f.write(f"  Signal Ratio: {pair_info['signal_ratio']:.4f}\n")
                f.write(f"  Correlation: {pair_info['correlation']:.4f}\n")
                f.write(f"  Average Spread: {pair_info['avg_spread']:.4f}\n")
                f.write(f"  Spread Volatility: {pair_info['spread_vol']:.4f}\n")

        strategy.save_state(str(models_dir / "final_state"))

        logger.info(f"Analysis complete. Results saved to {output_dir}")

        return {
            'strategy': strategy,
            'signals': signals,
            'prices': prices,
            'pair_analysis': pair_analysis
        }

    except Exception as e:
        logger.error(f"Error in main execution: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return None


if __name__ == "__main__":
    results = main()
    if results is not None:
        print("\nDeep Learning Pairs Trading analysis completed successfully!")
    else:
        print("\nAnalysis failed. Check logs for details.")