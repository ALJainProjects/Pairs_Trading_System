"""
Pairs Trading with Deep Learning (Corrected for Look-Ahead Bias)

This module implements deep learning models for pairs trading with proper
temporal handling to prevent look-ahead bias in:
1. Sequence preparation
2. Feature engineering
3. Model training and validation
4. Signal generation
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from sklearn.preprocessing import StandardScaler
import plotly.graph_objects as go
from sklearn.model_selection import TimeSeriesSplit
import tensorflow as tf
from config.logging_config import logger
from src.models import DeepLearningModel


class PairsTradingDL:
    """Deep learning based pairs trading implementation."""

    def __init__(self,
                 sequence_length: int = 20,
                 prediction_horizon: int = 1,
                 zscore_threshold: float = 2.0,
                 train_size: int = 252,  # One year
                 validation_size: int = 63):  # Quarter
        """
        Initialize the pairs trading model.

        Args:
            sequence_length: Length of input sequences
            prediction_horizon: Steps ahead to predict
            zscore_threshold: Threshold for trading signals
            train_size: Number of periods for training
            validation_size: Number of periods for validation
        """
        self.sequence_length = sequence_length
        self.prediction_horizon = prediction_horizon
        self.zscore_threshold = zscore_threshold
        self.train_size = train_size
        self.validation_size = validation_size
        self.dl_model = DeepLearningModel()
        self.scaler = StandardScaler()

    def prepare_sequences(self,
                          data: pd.DataFrame,
                          target_column: str,
                          start_idx: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare sequences without look-ahead bias.

        Args:
            data: Input DataFrame
            target_column: Column to predict
            start_idx: Starting index for sequence preparation

        Returns:
            Tuple of (sequences, targets)
        """
        if start_idx is None:
            start_idx = self.sequence_length

        sequences = []
        targets = []

        # Create sequences using only past data
        for i in range(start_idx, len(data) - self.prediction_horizon):
            # Input sequence
            seq = data.iloc[i - self.sequence_length:i].values
            # Target (future value)
            target = data.iloc[i + self.prediction_horizon - 1][target_column]

            sequences.append(seq)
            targets.append(target)

        return np.array(sequences), np.array(targets)

    def prepare_pair_data(self,
                          stock1_prices: pd.Series,
                          stock2_prices: pd.Series,
                          start_idx: Optional[int] = None) -> pd.DataFrame:
        """
        Prepare pair data using expanding windows.

        Args:
            stock1_prices: First stock prices
            stock2_prices: Second stock prices
            start_idx: Starting index for calculations

        Returns:
            DataFrame with features
        """
        if start_idx is None:
            start_idx = self.sequence_length

        # Calculate basic features
        spread = stock1_prices - stock2_prices
        ratio = stock1_prices / stock2_prices

        data = pd.DataFrame({
            'spread': spread,
            'ratio': ratio,
            'return1': stock1_prices.pct_change(),
            'return2': stock2_prices.pct_change()
        })

        # Calculate rolling features without look-ahead
        for col in ['spread', 'ratio']:
            roll_mean = data[col].rolling(
                window=self.sequence_length,
                min_periods=self.sequence_length
            ).mean()
            roll_std = data[col].rolling(
                window=self.sequence_length,
                min_periods=self.sequence_length
            ).std()

            data[f'{col}_ma'] = roll_mean
            data[f'{col}_std'] = roll_std
            data[f'{col}_zscore'] = (data[col] - roll_mean) / roll_std

        # Remove NaN values
        data = data.dropna()

        # Ensure we only use data from start_idx onwards
        if start_idx is not None:
            data = data.iloc[start_idx:]

        return data

    def generate_labels(self,
                        data: pd.DataFrame,
                        forward_looking: bool = False) -> pd.DataFrame:
        """
        Generate trading signals and labels.

        Args:
            data: Feature DataFrame
            forward_looking: Whether to include forward returns

        Returns:
            DataFrame with labels
        """
        result = data.copy()

        # Generate trading signals based on z-score
        result['signal'] = 0
        result.loc[result['spread_zscore'] > self.zscore_threshold, 'signal'] = -1
        result.loc[result['spread_zscore'] < -self.zscore_threshold, 'signal'] = 1

        # Binary classification target
        result['target'] = (result['signal'] != 0).astype(int)

        if forward_looking:
            # Add forward returns for training
            result['forward_spread'] = result['spread'].shift(-self.prediction_horizon)
            result['forward_return'] = result['spread'].pct_change(self.prediction_horizon)

        return result.dropna()

    def walk_forward_validation(self,
                                features: pd.DataFrame) -> List[Dict]:
        """
        Perform walk-forward validation.

        Args:
            features: Feature DataFrame

        Returns:
            List of results for each validation period
        """
        results = []
        tscv = TimeSeriesSplit(
            n_splits=5,
            test_size=self.validation_size,
            gap=self.sequence_length
        )

        for train_idx, val_idx in tscv.split(features):
            train_data = features.iloc[train_idx]
            val_data = features.iloc[val_idx]

            # Prepare sequences for LSTM
            X_train, y_train = self.prepare_sequences(
                train_data,
                'spread',
                start_idx=self.sequence_length
            )

            X_val, y_val = self.prepare_sequences(
                val_data,
                'spread',
                start_idx=self.sequence_length
            )

            # Train models
            spread_model = self.train_spread_predictor(
                X_train, y_train,
                X_val, y_val
            )

            signal_model = self.train_signal_classifier(
                train_data,
                val_data
            )

            # Generate predictions
            predictions = self.predict_signals(
                {'spread_predictor': spread_model,
                 'signal_classifier': signal_model},
                val_data
            )

            results.append({
                'train_period': (features.index[train_idx[0]],
                                 features.index[train_idx[-1]]),
                'val_period': (features.index[val_idx[0]],
                               features.index[val_idx[-1]]),
                'predictions': predictions,
                'models': {
                    'spread_predictor': spread_model,
                    'signal_classifier': signal_model
                }
            })

        return results

    def train_spread_predictor(self,
                               X_train: np.ndarray,
                               y_train: np.ndarray,
                               X_val: np.ndarray,
                               y_val: np.ndarray) -> Dict:
        """Train LSTM model for spread prediction."""
        # Scale sequences
        X_train_scaled = np.array([self.scaler.fit_transform(x) for x in X_train])
        X_val_scaled = np.array([self.scaler.transform(x) for x in X_val])

        # Build and train model
        model = self.dl_model.build_lstm_model(
            input_shape=(X_train.shape[1], X_train.shape[2]),
            units=64,
            dropout=0.2
        )

        history = self.dl_model.train_lstm_model(
            model,
            X_train_scaled, y_train,
            X_val_scaled, y_val,
            epochs=100,
            batch_size=32
        )

        # Evaluate
        metrics = self.dl_model.evaluate_lstm_model(
            model,
            X_val_scaled,
            y_val
        )

        return {
            'model': model,
            'scaler': self.scaler,
            'metrics': metrics,
            'history': history
        }

    def train_signal_classifier(self,
                                train_data: pd.DataFrame,
                                val_data: pd.DataFrame) -> Dict:
        """Train dense model for signal classification."""
        # Prepare features
        feature_cols = ['spread_zscore', 'ratio_zscore', 'spread_ma',
                        'return1', 'return2']

        X_train = train_data[feature_cols].values
        y_train = train_data['target'].values
        X_val = val_data[feature_cols].values
        y_val = val_data['target'].values

        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)

        # Build and train model
        model = self.dl_model.build_dense_model(
            input_dim=len(feature_cols),
            layers=[64, 32],
            dropout=0.3
        )

        history = self.dl_model.train_dense_model(
            model,
            X_train_scaled, y_train,
            X_val_scaled, y_val,
            epochs=100,
            batch_size=32
        )

        # Evaluate
        metrics = self.dl_model.evaluate_dense_model(
            model,
            X_val_scaled,
            y_val
        )

        return {
            'model': model,
            'scaler': self.scaler,
            'metrics': metrics,
            'history': history,
            'feature_cols': feature_cols
        }

    def predict_signals(self,
                        models: Dict,
                        new_data: pd.DataFrame) -> pd.DataFrame:
        """Generate trading predictions."""
        predictions = pd.DataFrame(index=new_data.index)

        # Prepare sequences for spread prediction
        X_spread, _ = self.prepare_sequences(
            new_data,
            'spread',
            start_idx=self.sequence_length
        )

        # Scale and predict spread
        X_spread_scaled = np.array([
            models['spread_predictor']['scaler'].transform(x)
            for x in X_spread
        ])
        spread_pred = models['spread_predictor']['model'].predict(X_spread_scaled)

        # Prepare features for signal classification
        X_signal = new_data[models['signal_classifier']['feature_cols']].values
        X_signal_scaled = models['signal_classifier']['scaler'].transform(X_signal)

        # Predict signals
        signal_prob = models['signal_classifier']['model'].predict(X_signal_scaled)

        # Store predictions
        predictions['spread_prediction'] = np.nan
        predictions.iloc[self.sequence_length:]['spread_prediction'] = spread_pred
        predictions['signal_probability'] = signal_prob
        predictions['predicted_signal'] = (signal_prob > 0.5).astype(int)

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

        # Plot predictions for each validation period
        for result in results:
            val_start, val_end = result['val_period']
            predictions = result['predictions']

            # Plot predicted spread
            mask = ~predictions['spread_prediction'].isna()
            if mask.any():
                fig.add_trace(go.Scatter(
                    x=predictions.index[mask],
                    y=predictions.loc[mask, 'spread_prediction'],
                    mode='lines',
                    line=dict(dash='dash'),
                    name=f'Predicted Spread ({val_start.date()})'
                ))

            # Plot signals
            for signal_val, color, name in zip(
                    [1, 0],
                    ['green', 'red'],
                    ['Long', 'Short']
            ):
                signal_mask = predictions['predicted_signal'] == signal_val
                if signal_mask.any():
                    fig.add_trace(go.Scatter(
                        x=predictions.index[signal_mask],
                        y=actual_spread[predictions.index[signal_mask]],
                        mode='markers',
                        marker=dict(color=color, size=10),
                        name=f'{name} Signal ({val_start.date()})'
                    ))

            # Add period boundaries
            for x in [val_start, val_end]:
                fig.add_vline(
                    x=x,
                    line_dash="dash",
                    line_color="gray",
                    opacity=0.5
                )

        fig.update_layout(
            title='Walk-Forward Validation Results (Deep Learning)',
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

    # Initialize model
    model = PairsTradingDL(
        sequence_length=20,
        prediction_horizon=1,
        zscore_threshold=2.0,
        train_size=252,
        validation_size=63
    )

    # Prepare features
    features = model.prepare_pair_data(
        stock1_prices,
        stock2_prices,
        start_idx=20  # sequence_length
    )

    # Perform walk-forward validation
    results = model.walk_forward_validation(features)

    # Plot results
    model.plot_walk_forward_results(
        results,
        stock1_prices - stock2_prices
    )

    return results


if __name__ == "__main__":
    main()