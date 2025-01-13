"""
Pairs Trading ML Implementation (Corrected for Look-Ahead Bias)

This module implements machine learning models for pairs trading with proper
temporal handling to prevent look-ahead bias in:
1. Feature engineering
2. Model training
3. Signal generation
4. Performance analysis
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
import plotly.graph_objects as go
from config.logging_config import logger
from src.models import MachineLearningModel


class PairsTraderML:
    """Machine learning based pairs trading implementation."""

    def __init__(self,
                 lookback_windows: List[int] = [5, 10, 20, 50],
                 zscore_threshold: float = 2.0,
                 train_size: int = 252,  # One year of daily data
                 validation_size: int = 63):  # Quarter of daily data
        """
        Initialize the pairs trader.

        Args:
            lookback_windows: List of periods for rolling statistics
            zscore_threshold: Z-score threshold for labeling positions
            train_size: Number of periods for training window
            validation_size: Number of periods for validation window
        """
        self.lookback_windows = lookback_windows
        self.zscore_threshold = zscore_threshold
        self.train_size = train_size
        self.validation_size = validation_size
        self.ml_model = MachineLearningModel()

    def prepare_features(self,
                         stock1_prices: pd.Series,
                         stock2_prices: pd.Series,
                         start_idx: int = 0,
                         window: int = 20) -> pd.DataFrame:
        """
        Create features using expanding windows to prevent look-ahead bias.

        Args:
            stock1_prices: Price series for first stock
            stock2_prices: Price series for second stock
            start_idx: Starting index for feature calculation
            window: Minimum window size for calculations

        Returns:
            DataFrame with engineered features
        """
        if len(stock1_prices) != len(stock2_prices):
            raise ValueError("Price series must be same length")

        features = pd.DataFrame(index=stock1_prices.index[start_idx:])

        # Calculate returns without look-ahead
        returns1 = stock1_prices.pct_change()
        returns2 = stock2_prices.pct_change()
        features['return1'] = returns1[start_idx:]
        features['return2'] = returns2[start_idx:]

        # Calculate expanding window features
        for w in self.lookback_windows:
            # Need enough data for the largest window
            if start_idx < w:
                continue

            # Calculate spread features
            spread = stock1_prices - stock2_prices
            roll_mean = spread.rolling(window=w, min_periods=w).mean()
            roll_std = spread.rolling(window=w, min_periods=w).std()

            features[f'spread_{w}'] = spread[start_idx:]
            features[f'spread_ma_{w}'] = roll_mean[start_idx:]
            features[f'spread_std_{w}'] = roll_std[start_idx:]

            # Calculate z-scores using only past data
            features[f'spread_zscore_{w}'] = (
                    (spread[start_idx:] - roll_mean[start_idx:]) /
                    roll_std[start_idx:]
            )

            # Calculate ratio features
            ratio = stock1_prices / stock2_prices
            ratio_mean = ratio.rolling(window=w, min_periods=w).mean()
            ratio_std = ratio.rolling(window=w, min_periods=w).std()

            features[f'ratio_{w}'] = ratio[start_idx:]
            features[f'ratio_ma_{w}'] = ratio_mean[start_idx:]
            features[f'ratio_std_{w}'] = ratio_std[start_idx:]
            features[f'ratio_zscore_{w}'] = (
                    (ratio[start_idx:] - ratio_mean[start_idx:]) /
                    ratio_std[start_idx:]
            )

        # Remove any remaining NaN values
        features = features.dropna()
        return features

    def generate_labels(self,
                        features: pd.DataFrame,
                        window: int,
                        forward_returns: bool = True) -> pd.DataFrame:
        """
        Generate trading signals and labels for training.

        Args:
            features: Feature DataFrame
            window: Window for z-score calculation
            forward_returns: Whether to include forward returns

        Returns:
            DataFrame with added labels
        """
        data = features.copy()
        zscore_col = f'spread_zscore_{window}'

        if zscore_col not in data.columns:
            raise ValueError(f"Z-score column {zscore_col} not found")

        # Current position signals
        data['signal'] = 0  # Neutral
        data.loc[data[zscore_col] > self.zscore_threshold, 'signal'] = -1  # Short
        data.loc[data[zscore_col] < -self.zscore_threshold, 'signal'] = 1  # Long

        # Binary classification target
        data['target'] = (data['signal'] != 0).astype(int)

        if forward_returns:
            # Add forward returns for training (not look-ahead in training)
            spread_col = f'spread_{window}'
            data['forward_return'] = data[spread_col].pct_change().shift(-1)

        return data

    def walk_forward_optimization(self,
                                  features: pd.DataFrame,
                                  window: int = 20) -> List[Dict]:
        """
        Perform walk-forward optimization to prevent look-ahead bias.

        Args:
            features: Feature DataFrame
            window: Window size for feature calculation

        Returns:
            List of results for each validation period
        """
        results = []
        tscv = TimeSeriesSplit(
            n_splits=5,
            test_size=self.validation_size,
            gap=window  # Gap between train and test to prevent leakage
        )

        for train_idx, val_idx in tscv.split(features):
            # Get train/validation sets
            train_data = features.iloc[train_idx]
            val_data = features.iloc[val_idx]

            # Generate labels (only for training data)
            train_data = self.generate_labels(
                train_data,
                window,
                forward_returns=True
            )

            # Select features
            feature_cols = [col for col in train_data.columns
                            if any(x in col for x in ['spread_', 'ratio_', 'return'])]

            X_train = train_data[feature_cols]
            y_class_train = train_data['target']
            y_reg_train = train_data['forward_return']

            # Train models
            models = self.train_period_models(
                X_train, y_class_train, y_reg_train
            )

            # Generate predictions for validation set
            predictions = self.predict_signals(
                models,
                val_data[feature_cols]
            )

            results.append({
                'train_period': (features.index[train_idx[0]],
                                 features.index[train_idx[-1]]),
                'val_period': (features.index[val_idx[0]],
                               features.index[val_idx[-1]]),
                'models': models,
                'predictions': predictions,
                'feature_importance': {
                    'classification': models['classification']['feature_importance'],
                    'regression': models['regression']['feature_importance']
                }
            })

        return results

    def train_period_models(self,
                            X_train: pd.DataFrame,
                            y_class: pd.Series,
                            y_reg: pd.Series) -> Dict:
        """
        Train models for a specific period.

        Args:
            X_train: Training features
            y_class: Classification targets
            y_reg: Regression targets

        Returns:
            Dictionary of trained models and metrics
        """
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)

        # Train classification model
        clf_model, clf_score = self.ml_model.train_classification_model(
            X_train_scaled,
            y_class,
            model_name='RandomForestClassifier',
            cv=5
        )

        # Train regression model
        reg_model, reg_score = self.ml_model.train_regression_model(
            X_train_scaled,
            y_reg,
            model_name='RandomForestRegressor',
            cv=5
        )

        # Get feature importance
        clf_importance = self.ml_model.feature_importance(
            clf_model,
            X_train.columns
        )
        reg_importance = self.ml_model.feature_importance(
            reg_model,
            X_train.columns
        )

        return {
            'classification': {
                'model': clf_model,
                'cv_score': clf_score,
                'feature_importance': clf_importance,
                'scaler': scaler
            },
            'regression': {
                'model': reg_model,
                'cv_score': reg_score,
                'feature_importance': reg_importance
            }
        }

    def predict_signals(self,
                        models: Dict,
                        features: pd.DataFrame) -> pd.DataFrame:
        """
        Generate predictions for new data.

        Args:
            models: Dictionary of trained models
            features: New feature data

        Returns:
            DataFrame with predictions
        """
        predictions = pd.DataFrame(index=features.index)

        # Scale features using stored scaler
        X_scaled = models['classification']['scaler'].transform(features)

        # Generate predictions
        predictions['signal_probability'] = (
            models['classification']['model'].predict_proba(X_scaled)[:, 1]
        )
        predictions['predicted_signal'] = (
                predictions['signal_probability'] > 0.5
        ).astype(int)
        predictions['predicted_return'] = (
            models['regression']['model'].predict(X_scaled)
        )

        return predictions

    def plot_walk_forward_results(self,
                                  results: List[Dict],
                                  actual_spread: pd.Series) -> None:
        """Plot walk-forward validation results."""
        fig = go.Figure()

        # Plot actual spread
        fig.add_trace(go.Scatter(
            x=actual_spread.index,
            y=actual_spread,
            mode='lines',
            name='Actual Spread'
        ))

        # Plot signals for each validation period
        colors = ['green', 'red']
        for period_result in results:
            val_predictions = period_result['predictions']
            val_start, val_end = period_result['val_period']

            # Plot signals
            for signal_val, color in zip([1, 0], colors):
                mask = val_predictions['predicted_signal'] == signal_val
                if mask.any():
                    fig.add_trace(go.Scatter(
                        x=val_predictions.index[mask],
                        y=actual_spread[val_predictions.index[mask]],
                        mode='markers',
                        marker=dict(color=color, size=10),
                        name=f'{"Long" if signal_val else "Short"} Signal'
                    ))

            # Add validation period boundaries
            for x in [val_start, val_end]:
                fig.add_vline(
                    x=x,
                    line_dash="dash",
                    line_color="gray",
                    opacity=0.5
                )

        fig.update_layout(
            title='Walk-Forward Validation Results',
            xaxis_title='Date',
            yaxis_title='Spread',
            template='plotly_white'
        )

        fig.show()


def main():
    """Example usage with proper temporal handling."""
    # Load data
    stock1_prices = pd.Series(...)  # Price series for stock 1
    stock2_prices = pd.Series(...)  # Price series for stock 2

    # Initialize trader
    trader = PairsTraderML(
        lookback_windows=[5, 10, 20, 50],
        zscore_threshold=2.0,
        train_size=252,
        validation_size=63
    )

    # Prepare features (minimum window for initial calculation)
    features = trader.prepare_features(
        stock1_prices,
        stock2_prices,
        start_idx=50  # Largest lookback window
    )

    # Perform walk-forward optimization
    results = trader.walk_forward_optimization(features)

    # Plot results
    trader.plot_walk_forward_results(
        results,
        stock1_prices - stock2_prices
    )

    return results


if __name__ == "__main__":
    main()