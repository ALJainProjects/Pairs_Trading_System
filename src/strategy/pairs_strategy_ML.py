"""
Enhanced ML Pairs Trading Strategy Module

Implements a complete machine learning based pairs trading strategy with:
- Integrated ML model class
- Portfolio value tracking
- Feature engineering
- Risk management
- No look-ahead bias
"""

from typing import Dict, List, Tuple, Optional
import pandas as pd
import numpy as np
from pathlib import Path
import json
from dataclasses import dataclass, field
import statsmodels.api as sm
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split

from src.strategy.base import BaseStrategy
from src.models.machine_learning import MachineLearningModel, time_series_cross_validation
from config.logging_config import logger
from config.settings import MODEL_DIR, DATA_DIR


@dataclass
class MLPosition:
    """Track ML-enhanced position details."""
    asset1: str
    asset2: str
    quantity: float
    entry_price1: float
    entry_price2: float
    entry_date: pd.Timestamp
    hedge_ratio: float
    model_confidence: float
    entry_features: Dict[str, float]
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    entry_spread: float = 0.0
    peak_pnl: float = 0.0
    current_drawdown: float = 0.0
    transaction_costs: float = 0.0
    trades: List[Dict] = field(default_factory=list)

    def update_metrics(self,
                       current_price1: float,
                       current_price2: float,
                       current_confidence: float) -> None:
        """Update position metrics including ML confidence."""
        current_spread = current_price1 - self.hedge_ratio * current_price2
        spread_pnl = self.quantity * (current_spread - self.entry_spread)

        self.peak_pnl = max(self.peak_pnl, spread_pnl)
        self.current_drawdown = (self.peak_pnl - spread_pnl) / abs(self.peak_pnl) if self.peak_pnl != 0 else 0
        self.model_confidence = current_confidence


class MLPairsStrategy(BaseStrategy):
    """Machine Learning based pairs trading strategy."""

    def __init__(
            self,
            initial_capital: float = 1_000_000.0,
            lookback_window: int = 20,
            model_confidence_threshold: float = 0.6,
            zscore_threshold: float = 2.0,
            max_position_size: float = 0.1,
            stop_loss: float = 0.02,
            take_profit: float = 0.04,
            max_drawdown: float = 0.2,
            max_pairs: int = 10,
            transaction_cost: float = 0.001,
            model_dir: Optional[Path] = None
    ):
        """Initialize the ML pairs trading strategy."""
        super().__init__(
            name="MLPairsStrategy",
            max_position_size=max_position_size
        )

        self.validation_size = 0.2
        self.lookback_window = lookback_window
        self.model_confidence_threshold = model_confidence_threshold
        self.zscore_threshold = zscore_threshold
        self.stop_loss = stop_loss
        self.take_profit = take_profit
        self.max_drawdown = max_drawdown
        self.max_pairs = max_pairs
        self.transaction_cost = transaction_cost
        self.model_dir = model_dir or Path(MODEL_DIR) / "ml_pairs"

        self._initial_capital = initial_capital
        self._portfolio_value = initial_capital
        self._cash = initial_capital
        self._positions_value = 0.0

        self.ml_model = MachineLearningModel()
        self._trained = False

        self.positions: Dict[Tuple[str, str], MLPosition] = {}
        self.pairs: List[Tuple[str, str]] = []
        self.hedge_ratios: Dict[Tuple[str, str], float] = {}

        self.model_dir.mkdir(parents=True, exist_ok=True)

    @property
    def current_portfolio_value(self) -> float:
        """Get current portfolio value including positions."""
        return self._cash + self._positions_value

    def update_portfolio_value(self, prices: pd.DataFrame) -> None:
        """Update portfolio value based on current positions and prices."""
        positions_value = 0.0

        for pair, position in self.positions.items():
            asset1, asset2 = pair
            if asset1 in prices.columns and asset2 in prices.columns:
                current_price1 = prices[asset1].iloc[-1]
                current_price2 = prices[asset2].iloc[-1]

                value = abs(position.quantity) * (
                        current_price1 + position.hedge_ratio * current_price2
                )
                positions_value += value

        self._positions_value = positions_value
        self._portfolio_value = self._cash + positions_value

    def calculate_dynamic_hedge_ratio(self,
                                      price1: pd.Series,
                                      price2: pd.Series) -> pd.Series:
        """Calculate dynamic hedge ratio using Kalman Filter."""
        return self.ml_model.calculate_dynamic_hedge_ratio(price1, price2)

    def prepare_features(self, price1: pd.Series, price2: pd.Series) -> pd.DataFrame:
        """Use comprehensive feature engineering from MachineLearningModel."""
        return self.ml_model.prepare_features(
            price1=price1,
            price2=price2,
            windows=[5, 20, 60],
            lag_windows=[1, 2, 3, 5, 10]
        )

    def initialize_models(self, train_data: pd.DataFrame) -> None:
        """
        Initialize and train ML models for each pair.

        Args:
            train_data: DataFrame with asset returns for training period
        """
        logger.info("Initializing ML models for pairs trading")
        self.ml_model = MachineLearningModel()
        self.pair_models = {}

        if not self.pairs:
            self.pairs = self._find_pairs(train_data)

        for pair in self.pairs:
            try:
                asset1, asset2 = pair
                logger.info(f"Training model for pair {asset1}-{asset2}")

                # Prepare features for the pair
                features = self.ml_model.prepare_features(
                    price1=train_data[asset1],
                    price2=train_data[asset2],
                    windows=[5, 20, 60],  # Multiple timeframes for better signals
                    lag_windows=[1, 2, 3, 5, 10]
                )

                # Create training labels
                labels = self.ml_model.create_advanced_spread_labels(
                    price1=train_data[asset1],
                    price2=train_data[asset2],
                    lookback_window=self.lookback_window,
                    zscore_threshold=self.zscore_threshold
                )

                # Ensure features and labels are aligned
                features = features.loc[labels.index]

                # Split data for training
                X_train, X_val, y_train, y_val = train_test_split(
                    features,
                    labels,
                    test_size=self.validation_size,
                    shuffle=False
                )

                # Train model with cross-validation
                metrics = time_series_cross_validation(
                    model=self.ml_model,
                    features=X_train,
                    target=y_train,
                    model_name="RandomForestClassifier",
                    n_splits=5
                )

                logger.info(f"Cross-validation metrics for {asset1}-{asset2}: {metrics}")

                # Final model training
                model, train_metrics = self.ml_model.train_model(
                    X=features,
                    y=labels,
                    model_name="RandomForestClassifier",
                    test_size=self.validation_size / len(features),
                    cv=5
                )

                self.pair_models[pair] = {
                    'model': model,
                    'metrics': train_metrics,
                    'features': features.columns.tolist()
                }

                logger.info(f"Successfully trained model for {asset1}-{asset2}")

            except Exception as e:
                logger.error(f"Error training model for pair {pair}: {str(e)}")
                continue

        if not self.pair_models:
            raise ValueError("No models could be trained successfully")

    def _calculate_rolling_coint_pvalue(self,
                                        price1: pd.Series,
                                        price2: pd.Series,
                                        window: int = 60) -> pd.Series:
        """Calculate rolling cointegration p-value."""
        p_values = pd.Series(index=price1.index, dtype=float)

        for i in range(window, len(price1)):
            try:
                _, p_value, _ = sm.tsa.stattools.coint(
                    price1.iloc[i - window:i],
                    price2.iloc[i - window:i]
                )
                p_values.iloc[i] = p_value
            except:
                p_values.iloc[i] = np.nan

        return p_values.fillna(1.0)

    def generate_signals(self, prices: pd.DataFrame) -> pd.DataFrame:
        """Generate trading signals using advanced ML models."""
        self.update_portfolio_value(prices)
        signals = pd.DataFrame(index=prices.index)

        if not self.pairs:
            self.pairs = self._find_pairs(prices)

        for pair in self.pairs:
            try:
                asset1, asset2 = pair
                if asset1 not in prices.columns or asset2 not in prices.columns:
                    continue

                features = self.prepare_features(
                    prices[asset1],
                    prices[asset2]
                )

                if not self._trained:
                    labels = self.ml_model.create_advanced_spread_labels(
                        prices[asset1],
                        prices[asset2],
                        lookback_window=self.lookback_window,
                        zscore_threshold=self.zscore_threshold
                    )

                    model_metrics = time_series_cross_validation(
                        model=self.ml_model,
                        features=features,
                        target=labels,
                        model_name="RandomForestClassifier",
                        n_splits=5
                    )
                    print(model_metrics)
                    self._trained = True

                prediction_proba = self.ml_model.default_model.predict_proba(
                    self.ml_model.scaler.transform(features)
                )
                confidence = np.max(prediction_proba, axis=1)
                predictions = self.ml_model.default_model.predict(
                    self.ml_model.scaler.transform(features)
                )

                if predictions[-1] != 0 and confidence[-1] >= self.model_confidence_threshold:
                    position_size = self.calculate_position_size(
                        pair=pair,
                        prices=prices,
                        portfolio_value=self.current_portfolio_value,
                        confidence=confidence[-1]
                    )
                    predictions = predictions * position_size

                signals[pair] = np.where(
                    confidence >= self.model_confidence_threshold,
                    predictions,
                    0
                )

            except Exception as e:
                logger.error(f"Error generating signals for {pair}: {str(e)}")
                continue

        return signals

    def _find_pairs(self, prices: pd.DataFrame) -> List[Tuple[str, str]]:
        """Find suitable trading pairs using correlation and cointegration."""
        pairs = []
        n_assets = len(prices.columns)

        for i in range(n_assets):
            for j in range(i + 1, n_assets):
                asset1, asset2 = prices.columns[i], prices.columns[j]

                correlation = prices[asset1].corr(prices[asset2])
                if abs(correlation) < 0.5:
                    continue

                try:
                    _, pvalue, _ = sm.tsa.stattools.coint(
                        prices[asset1],
                        prices[asset2]
                    )
                    if pvalue <= 0.05:
                        pairs.append((asset1, asset2))

                        hedge_ratio = self.calculate_dynamic_hedge_ratio(
                            prices[asset1],
                            prices[asset2]
                        ).iloc[-1]

                        self.hedge_ratios[(asset1, asset2)] = hedge_ratio

                        if len(pairs) >= self.max_pairs:
                            return pairs

                except Exception as e:
                    logger.warning(f"Error testing cointegration for {asset1}-{asset2}: {str(e)}")
                    continue

        return pairs

    def calculate_position_size(self,
                                pair: Tuple[str, str],
                                prices: pd.DataFrame,
                                portfolio_value: float,
                                confidence: float) -> float:
        """
        Calculate position size based on model confidence and portfolio constraints.

        Args:
            pair: Asset pair
            prices: Price data
            portfolio_value: Current portfolio value
            confidence: Model confidence score
        """
        asset1, asset2 = pair
        price1 = prices[asset1].iloc[-1]
        price2 = prices[asset2].iloc[-1]

        spread = prices[asset1] - self.hedge_ratios[pair] * prices[asset2]
        spread_vol = spread.rolling(window=self.lookback_window).std().iloc[-1]

        vol_adjustment = 1.0 / (1.0 + spread_vol) if spread_vol > 0 else 1.0
        position_size = self.max_position_size * confidence * vol_adjustment

        max_trade_value = portfolio_value * position_size
        pair_value = price1 + price2

        position_size = (max_trade_value / pair_value)

        position_size *= (1 - self.transaction_cost)

        return position_size

    def predict_signals(self, features: pd.DataFrame) -> Tuple[pd.Series, pd.Series]:
        """Generate signals and confidence scores from features."""
        if not self._trained:
            raise ValueError("Model needs to be trained first")

        X_scaled = self.ml_model.scaler.transform(features)

        try:

            if hasattr(self.ml_model, 'model'):
                model = self.ml_model.default_model
            else:

                model = self.ml_model.classification_models['RandomForestClassifier']

            predictions = model.predict(X_scaled)
            probabilities = model.predict_proba(X_scaled)
            confidence = np.max(probabilities, axis=1)

            return pd.Series(predictions, index=features.index), pd.Series(confidence, index=features.index)
        except Exception as e:
            logger.error(f"Error predicting signals: {str(e)}")

            return (
                pd.Series(0, index=features.index),
                pd.Series(0, index=features.index)
            )

    def update_positions(self,
                         current_prices: pd.Series,
                         timestamp: pd.Timestamp) -> None:
        """Update position metrics and check risk limits."""
        for pair, position in list(self.positions.items()):
            asset1, asset2 = pair

            if asset1 not in current_prices or asset2 not in current_prices:
                self._close_position(timestamp, pair, "Asset not available")
                continue

            current_price1 = current_prices[asset1]
            current_price2 = current_prices[asset2]

            features = self.prepare_features(
                pd.Series([current_price1]),
                pd.Series([current_price2])
            )
            _, confidence = self.predict_signals(features)

            position.update_metrics(
                current_price1,
                current_price2,
                confidence.iloc[-1]
            )

            if confidence.iloc[-1] < self.model_confidence_threshold:
                self._close_position(timestamp, pair, "Low model confidence")
                continue

            if position.current_drawdown > self.stop_loss:
                self._close_position(timestamp, pair, "Stop loss triggered")
                continue

            current_spread = current_price1 - position.hedge_ratio * current_price2
            profit_pct = abs(current_spread - position.entry_spread) / abs(position.entry_spread)

            if profit_pct > self.take_profit:
                self._close_position(timestamp, pair, "Take profit triggered")

    def _process_signals(self, signals: pd.DataFrame, prices: pd.DataFrame) -> None:
        """Process signals and update positions."""
        current_time = prices.index[-1]

        self.update_positions(prices.iloc[-1], current_time)

        for pair in signals.columns:
            signal = signals[pair].iloc[-1]

            if pair in self.positions:
                position = self.positions[pair]
                current_signal = np.sign(position.quantity)

                if signal == 0 or (signal * current_signal < 0):
                    self._close_position(pair, current_time, "Signal change")

            if signal != 0 and pair not in self.positions:
                self._open_position(
                    pair=pair,
                    signal=np.sign(signal),
                    quantity=abs(signal),
                    timestamp=current_time,
                    prices=prices
                )

    def _open_position(self,
                       pair: Tuple[str, str],
                       signal: float,
                       quantity: float,
                       timestamp: pd.Timestamp,
                       prices: pd.DataFrame = None) -> None:
        """Open new position with proper portfolio tracking."""
        asset1, asset2 = pair
        price1 = prices[asset1].iloc[-1]
        price2 = prices[asset2].iloc[-1]

        position_value = quantity * (price1 + self.hedge_ratios[pair] * price2)
        transaction_costs = position_value * self.transaction_cost

        total_required = position_value + transaction_costs

        if total_required > self._cash:
            logger.warning(f"Insufficient cash for position in {pair}")
            return

        features = self.prepare_features(
            prices[asset1],
            prices[asset2]
        )

        _, confidence = self.ml_model.default_model.predict_proba(
            self.ml_model.scaler.transform(features.iloc[[-1]])
        )[0].max(), self.ml_model.default_model.predict_proba(
            self.ml_model.scaler.transform(features.iloc[[-1]])
        )

        position = MLPosition(
            asset1=asset1,
            asset2=asset2,
            quantity=quantity * signal,
            entry_price1=price1,
            entry_price2=price2,
            entry_date=timestamp,
            hedge_ratio=self.hedge_ratios[pair],
            model_confidence=confidence,
            entry_features=features.iloc[-1].to_dict(),
            stop_loss=self.stop_loss,
            take_profit=self.take_profit,
            transaction_costs=transaction_costs
        )

        self._cash -= total_required
        self._positions_value += position_value
        self.positions[pair] = position

        self.trades.append({
            'timestamp': timestamp,
            'pair': f"{asset1}/{asset2}",
            'action': 'ENTRY',
            'quantity': quantity * signal,
            'price1': price1,
            'price2': price2,
            'hedge_ratio': self.hedge_ratios[pair],
            'confidence': confidence,
            'transaction_costs': transaction_costs,
            'portfolio_value': self.current_portfolio_value
        })

    def _close_position(self,
                        timestamp: pd.Timestamp,
                        pair: Tuple[str, str],
                        reason: str = None) -> None:
        """Close position with proper portfolio tracking."""
        if pair not in self.positions:
            return

        position = self.positions[pair]
        current_price1 = self.current_prices[position.asset1]
        current_price2 = self.current_prices[position.asset2]

        position_value = abs(position.quantity) * (
                current_price1 + position.hedge_ratio * current_price2
        )
        exit_costs = position_value * self.transaction_cost

        spread_entry = position.entry_price1 - position.hedge_ratio * position.entry_price2
        spread_exit = current_price1 - position.hedge_ratio * current_price2
        pnl = position.quantity * (spread_exit - spread_entry)

        self._cash += position_value - exit_costs
        self._positions_value -= position_value

        self.trades.append({
            'timestamp': timestamp,
            'pair': f"{position.asset1}/{position.asset2}",
            'action': 'EXIT',
            'reason': reason,
            'quantity': -position.quantity,
            'price1': current_price1,
            'price2': current_price2,
            'pnl': pnl,
            'transaction_costs': position.transaction_costs + exit_costs,
            'portfolio_value': self.current_portfolio_value
        })

        del self.positions[pair]

    def calculate_portfolio_stats(self) -> Dict:
        """Calculate enhanced portfolio statistics."""
        if not self.trades:
            return {}

        trades_df = pd.DataFrame(self.trades)

        portfolio_values = trades_df['portfolio_value']
        returns = portfolio_values.pct_change().dropna()

        stats = {
            'total_trades': len(trades_df[trades_df['action'] == 'ENTRY']),
            'winning_trades': len(trades_df[trades_df['pnl'] > 0]),
            'total_pnl': trades_df['pnl'].sum(),
            'total_costs': trades_df['transaction_costs'].sum(),
            'net_pnl': trades_df['pnl'].sum() - trades_df['transaction_costs'].sum(),
            'sharpe_ratio': returns.mean() / returns.std() * np.sqrt(252) if len(returns) > 0 else 0,
            'max_drawdown': (portfolio_values.max() - portfolio_values) / portfolio_values.max(),
            'profit_factor': (
                abs(trades_df[trades_df['pnl'] > 0]['pnl'].sum()) /
                abs(trades_df[trades_df['pnl'] < 0]['pnl'].sum())
                if len(trades_df[trades_df['pnl'] < 0]) > 0 else float('inf')
            ),
            'final_portfolio_value': self.current_portfolio_value,
            'return_pct': (self.current_portfolio_value - self._initial_capital) / self._initial_capital * 100
        }

        return stats

    def save_state(self, path: str) -> None:
        """Save strategy state and models."""
        state_path = Path(path)
        state_path.mkdir(parents=True, exist_ok=True)

        from joblib import dump
        dump(self.ml_model.default_model, state_path / "ml_model.joblib")
        dump(self.ml_model.scaler, state_path / "scaler.joblib")

        state = {
            'pairs': self.pairs,
            'hedge_ratios': self.hedge_ratios,
            'parameters': {
                'lookback_window': self.lookback_window,
                'model_confidence_threshold': self.model_confidence_threshold,
                'zscore_threshold': self.zscore_threshold,
                'portfolio_value': self._portfolio_value,
                'cash': self._cash
            },
            'is_trained': self._trained
        }

        with open(state_path / "strategy_state.json", 'w') as f:
            json.dump(state, f, indent=4)

    def load_state(self, path: str) -> None:
        """Load strategy state and models."""
        state_path = Path(path)

        from joblib import load
        self.ml_model.default_model = load(state_path / "ml_model.joblib")
        self.ml_model.scaler = load(state_path / "scaler.joblib")

        with open(state_path / "strategy_state.json", 'r') as f:
            state = json.load(f)

        self.pairs = state['pairs']
        self.hedge_ratios = state['hedge_ratios']
        self.lookback_window = state['parameters']['lookback_window']
        self.model_confidence_threshold = state['parameters']['model_confidence_threshold']
        self.zscore_threshold = state['parameters']['zscore_threshold']
        self._portfolio_value = state['parameters']['portfolio_value']
        self._cash = state['parameters']['cash']
        self._trained = state['is_trained']

    def plot_feature_importance(self, output_dir: Optional[Path] = None) -> None:
        """Plot feature importance using Plotly."""
        if not hasattr(self.ml_model.default_model, "feature_importances_"):
            logger.warning("Model does not have feature importances")
            return

        importances = pd.Series(
            self.ml_model.default_model.feature_importances_,
            index=self.ml_model.default_model.feature_names_in_
        ).sort_values(ascending=True)

        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=importances.values,
            y=importances.index,
            orientation='h'
        ))

        fig.update_layout(
            title="ML Model Feature Importance",
            xaxis_title="Importance",
            yaxis_title="Features",
            height=max(400, len(importances) * 20),
            width=800,
            template="plotly_white"
        )

        if output_dir:
            fig.write_html(output_dir / "feature_importance.html")
        else:
            fig.show()


def main():
    """Test the ML pairs trading strategy."""
    output_dir = Path("pairs_trading_strategy_ML_outputs")
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

        strategy = MLPairsStrategy(
            initial_capital=1_000_000.0,
            lookback_window=20,
            model_confidence_threshold=0.6,
            zscore_threshold=2.0,
            max_position_size=0.1,
            stop_loss=0.02,
            take_profit=0.04,
            max_drawdown=0.2,
            max_pairs=5,
            model_dir=models_dir
        )

        logger.info("Generating trading signals...")
        signals = strategy.generate_signals(prices)
        signals.to_csv(results_dir / "trading_signals.csv")

        logger.info("Processing signals and managing positions...")
        strategy._process_signals(signals, prices)

        logger.info("Analyzing pairs...")
        pair_analysis = []
        for pair in strategy.pairs:
            asset1, asset2 = pair
            features = strategy.prepare_features(
                prices[asset1],
                prices[asset2]
            )

            prediction_proba = strategy.ml_model.default_model.predict_proba(
                strategy.ml_model.scaler.transform(features)
            )
            confidence = np.max(prediction_proba, axis=1)

            analysis = {
                'pair': f"{asset1}/{asset2}",
                'hedge_ratio': strategy.hedge_ratios[pair],
                'avg_confidence': confidence.mean(),
                'signal_ratio': (signals[pair] != 0).mean(),
                'last_confidence': confidence.iloc[-1]
            }
            pair_analysis.append(analysis)

        pd.DataFrame(pair_analysis).to_csv(results_dir / "pair_analysis.csv")

        portfolio_stats = strategy.calculate_portfolio_stats()
        with open(results_dir / "portfolio_stats.json", 'w') as f:
            json.dump(portfolio_stats, f, indent=4)

        strategy.plot_feature_importance(plots_dir)

        strategy.save_state(str(models_dir / "final_state"))

        with open(results_dir / "analysis_summary.txt", 'w') as f:
            f.write("Machine Learning Pairs Trading Analysis\n")
            f.write("=====================================\n\n")

            f.write(f"Analysis Period: {prices.index[0].date()} to {prices.index[-1].date()}\n")
            f.write(f"Number of assets analyzed: {len(prices.columns)}\n")
            f.write(f"Number of trading pairs found: {len(strategy.pairs)}\n\n")

            f.write("Trading Pairs:\n")
            for pair_info in pair_analysis:
                f.write(f"\n{pair_info['pair']}:\n")
                f.write(f"  Hedge Ratio: {pair_info['hedge_ratio']:.4f}\n")
                f.write(f"  Average Confidence: {pair_info['avg_confidence']:.4f}\n")
                f.write(f"  Signal Ratio: {pair_info['signal_ratio']:.4f}\n")
                f.write(f"  Last Confidence: {pair_info['last_confidence']:.4f}\n")

            f.write("\nPortfolio Statistics:\n")
            for key, value in portfolio_stats.items():
                f.write(f"  {key}: {value}\n")

        logger.info(f"Analysis complete. Results saved to {output_dir}")

        return {
            'strategy': strategy,
            'signals': signals,
            'prices': prices,
            'pair_analysis': pair_analysis,
            'portfolio_stats': portfolio_stats
        }

    except Exception as e:
        logger.error(f"Error in main execution: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return None


if __name__ == "__main__":
    results = main()
    if results is not None:
        print("\nMachine Learning Pairs Trading analysis completed successfully!")

        stats = results['portfolio_stats']
        print("\nPortfolio Performance:")
        print(f"Total Return: {stats['return_pct']:.2f}%")
        print(f"Sharpe Ratio: {stats['sharpe_ratio']:.2f}")
        print(f"Win Rate: {(stats['winning_trades'] / stats['total_trades'] * 100):.1f}%")
        print(f"Profit Factor: {stats['profit_factor']:.2f}")
        print(f"Max Drawdown: {stats['max_drawdown'] * 100:.1f}%")
    else:
        print("\nAnalysis failed. Check logs for details.")
