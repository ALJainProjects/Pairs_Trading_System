"""
Deep Learning Models Module

Features:
 - LSTM for time-series forecasting
 - Dense model for classification
 - Model checkpointing and state management
 - Plotly-based visualizations
 - Comprehensive evaluation metrics
"""


import os
from pathlib import Path
# import sys
from typing import Tuple, Dict, List, Optional, Union
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import warnings

from src.data import FeatureEngineer

try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential, load_model
    from tensorflow.keras.layers import Dense, Dropout, LSTM
    from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.layers import Input
except ImportError:
    raise ImportError("Please install tensorflow: pip install tensorflow")

try:
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import mean_squared_error, classification_report, confusion_matrix
    from sklearn.model_selection import train_test_split
    from sklearn.model_selection import TimeSeriesSplit
except ImportError:
    raise ImportError("Please install scikit-learn: pip install scikit-learn")

from config.logging_config import logger
from config.settings import MODEL_DIR


class DeepLearningModel:
    """Deep Learning Model with LSTM and Dense architectures for financial time series."""

    def __init__(self):
        """
        Initialize the model with optional model directory for checkpoints.
        """
        self.feature_scaler = StandardScaler()
        self.target_scaler = StandardScaler()
        self.model_dir = MODEL_DIR
        if self.model_dir:
            os.makedirs(self.model_dir, exist_ok=True)
        self._history = None
        self._current_model = None
        self.is_fitted = False

    def save_model(self, filename: str) -> None:
        """Save the current model."""
        if not self._current_model:
            raise ValueError("No model to save")
        save_path = self.model_dir / filename if self.model_dir else filename
        self._current_model.save(save_path)
        logger.info(f"Model saved to {save_path}")

    def load_model(self, filename: str) -> None:
        """Load a saved model."""
        load_path = self.model_dir / filename if self.model_dir else filename
        if not os.path.exists(load_path):
            raise FileNotFoundError(f"Model file not found: {load_path}")
        self._current_model = load_model(load_path)
        logger.info(f"Model loaded from {load_path}")

    def prepare_sequences(
            self,
            data: pd.DataFrame,
            target_column: str,
            feature_columns: Optional[List[str]] = None,
            sequence_length: int = 10,
            stride: int = 1,
            scale_data: bool = True
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare sequences for LSTM with separate scaling for features and target.

        Args:
            data: Input DataFrame
            target_column: Column to predict
            feature_columns: Columns to use as features
            sequence_length: Length of input sequences
            stride: Steps between sequences
            scale_data: Whether to scale the data

        Returns:
            Tuple of (X, y) arrays
        """
        logger.info("Preparing sequences for LSTM with target scaling")

        if data.isnull().any().any():
            raise ValueError("Data contains NaN values")
        if target_column not in data.columns:
            raise ValueError(f"Target column '{target_column}' not found")

        target_values = data[[target_column]].values
        if scale_data:
            target_values = self.target_scaler.fit_transform(target_values)

        feature_columns = feature_columns or [target_column]
        features_df = data[feature_columns]

        if scale_data:
            features_df = pd.DataFrame(
                self.feature_scaler.fit_transform(features_df),
                columns=features_df.columns,
                index=features_df.index
            )
            self.is_fitted = True

        sequences, targets = [], []
        for i in range(0, len(data) - sequence_length, stride):
            seq = features_df.iloc[i:i + sequence_length].values
            target = target_values[i + sequence_length][0]
            sequences.append(seq)
            targets.append(target)

        X = np.array(sequences)
        y = np.array(targets)

        logger.debug(f"Created sequences with shape {X.shape} and targets {y.shape}")
        return X, y

    def build_lstm_model(
        self,
        input_shape: Tuple[int, int],
        lstm_units: List[int] = (50, 50),
        dense_units: List[int] = (32,),
        dropout_rate: float = 0.2,
        learning_rate: float = 0.001
    ) -> Sequential:
        """
        Build LSTM model with configurable architecture.

        Args:
            input_shape: Shape of input sequences
            lstm_units: List of units for LSTM layers
            dense_units: List of units for Dense layers
            dropout_rate: Dropout rate
            learning_rate: Learning rate for optimizer
        """
        logger.info("Building LSTM model")

        model = Sequential([
            Input(shape=input_shape),
            *[
                layer for i, units in enumerate(lstm_units)
                for layer in [
                    LSTM(
                        units,
                        return_sequences=(i < len(lstm_units) - 1),
                        activation='tanh'
                    ),
                    Dropout(dropout_rate)
                ]
            ],
            *[
                layer for units in dense_units
                for layer in [
                    Dense(units, activation='relu'),
                    Dropout(dropout_rate)
                ]
            ],
            Dense(1, activation='linear')
        ])

        model.compile(
            optimizer=Adam(learning_rate=learning_rate),
            loss='mean_squared_error',
            metrics=['mae']
        )

        self._current_model = model
        return model

    def build_dense_model(
        self,
        input_dim: int,
        units: List[int] = (64, 32),
        dropout_rate: float = 0.3,
        learning_rate: float = 0.001
    ) -> Sequential:
        """Build Dense model for classification."""
        logger.info("Building Dense classification model")

        model = Sequential()
        model.add(Input(shape=(input_dim,)))
        model.add(Dense(units[0], activation='relu'))
        model.add(Dropout(dropout_rate))

        for n_units in units[1:]:
            model.add(Dense(n_units, activation='relu'))
            model.add(Dropout(dropout_rate))

        model.add(Dense(1, activation='sigmoid'))

        model.compile(
            optimizer=Adam(learning_rate=learning_rate),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )

        self._current_model = model
        return model

    def train_model(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        epochs: int = 100,
        batch_size: int = 32,
        patience: int = 10,
        model_prefix: str = "model"
    ) -> Sequential:
        """
        Train model with early stopping and checkpoints.

        Args:
            X_train: Training features
            y_train: Training targets
            X_val: Validation features
            y_val: Validation targets
            epochs: Number of epochs
            batch_size: Batch size
            patience: Early stopping patience
            model_prefix: Prefix for saved model files
        """
        if self._current_model is None:
            raise ValueError("Build model first using build_lstm_model or build_dense_model")

        callbacks = [
            EarlyStopping(
                monitor='val_loss',
                patience=patience,
                restore_best_weights=True,
                verbose=1
            )
        ]

        if self.model_dir:
            callbacks.append(
                ModelCheckpoint(
                    filepath=Path(self.model_dir) / f"{model_prefix}_{{epoch:02d}}_{{val_loss:.4f}}.h5",
                    monitor='val_loss',
                    save_best_only=True,
                    verbose=1
                )
            )

        history = self._current_model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1
        )

        self._history = history
        logger.info("Model training completed")
        return self._current_model

    def evaluate_model(
        self,
        X_test: np.ndarray,
        y_test: np.ndarray,
        task: str = 'regression'
    ) -> Dict:
        """
        Evaluate model performance.

        Args:
            X_test: Test features
            y_test: Test targets
            task: 'regression' or 'classification'

        Returns:
            Dictionary of evaluation metrics
        """
        if self._current_model is None:
            raise ValueError("No model available for evaluation")

        if task == 'regression':
            preds = self._current_model.predict(X_test)
            mse = mean_squared_error(y_test, preds)
            rmse = np.sqrt(mse)
            mae = np.mean(np.abs(y_test - preds.flatten()))
            r2 = 1 - (np.sum((y_test - preds.flatten())**2) /
                     np.sum((y_test - np.mean(y_test))**2))

            return {
                "mse": mse,
                "rmse": rmse,
                "mae": mae,
                "r2": r2
            }

        elif task == 'classification':
            preds = (self._current_model.predict(X_test) > 0.5).astype(int)
            return {
                "classification_report": classification_report(y_test, preds, output_dict=True),
                "confusion_matrix": confusion_matrix(y_test, preds).tolist()
            }

        else:
            raise ValueError(f"Unknown task type: {task}")

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions with current model."""
        if self._current_model is None:
            raise ValueError("No model available for prediction")
        return self._current_model.predict(X)

    def plot_training_history(
        self,
        metrics: Union[str, List[str]] = ('loss', 'mae'),
        title: str = 'Training History',
        save_path: Optional[str] = None
    ) -> None:
        """
        Plot training history with multiple metrics support.

        Args:
            metrics: Metric(s) to plot
            title: Plot title
            save_path: Path to save the plot
        """
        if not self._history:
            logger.warning("No training history available")
            return

        metrics = [metrics] if isinstance(metrics, str) else metrics
        fig = go.Figure()

        for metric in metrics:
            if metric not in self._history.history:
                logger.warning(f"Metric {metric} not found in history")
                continue

            fig.add_trace(go.Scatter(
                x=list(range(len(self._history.history[metric]))),
                y=self._history.history[metric],
                mode='lines',
                name=f'Training {metric}'
            ))

            val_metric = f'val_{metric}'
            if val_metric in self._history.history:
                fig.add_trace(go.Scatter(
                    x=list(range(len(self._history.history[val_metric]))),
                    y=self._history.history[val_metric],
                    mode='lines',
                    name=f'Validation {metric}'
                ))

        fig.update_layout(
            title=title,
            xaxis_title='Epoch',
            yaxis_title='Value',
            template='plotly_white',
            hovermode='x unified'
        )

        if save_path:
            fig.write_html(save_path)

        fig.show()


def time_series_cross_validation(
        model: DeepLearningModel,
        X: np.ndarray,
        y: np.ndarray,
        n_splits: int = 5,
        epochs: int = 100,
        batch_size: int = 16,
        patience: int = 10
) -> List[Dict[str, float]]:
    """
    Perform Time-Series Cross-Validation (TSCV) for the given model.

    Args:
        model: Instance of DeepLearningModel.
        X: Input features (3D array: samples, timesteps, features).
        y: Target values (1D or 2D array).
        n_splits: Number of splits for TSCV.
        epochs: Number of epochs for training.
        batch_size: Batch size for training.
        patience: Patience for early stopping.

    Returns:
        List of dictionaries containing evaluation metrics for each fold.
    """
    tscv = TimeSeriesSplit(n_splits=n_splits)
    fold_metrics = []

    for fold, (train_index, val_index) in enumerate(tscv.split(X), start=1):
        print(f"\n=== Fold {fold}/{n_splits} ===")

        X_train, X_val = X[train_index], X[val_index]
        y_train, y_val = y[train_index], y[val_index]

        print(f"Training data shape: {X_train.shape}, Validation data shape: {X_val.shape}")

        model.build_lstm_model(
            input_shape=(X.shape[1], X.shape[2]),
            lstm_units=[128, 64],
            dense_units=[32],
            dropout_rate=0.2,
            learning_rate=0.0001
        )

        print("Training model...")
        model.train_model(
            X_train, y_train,
            X_val, y_val,
            epochs=epochs,
            batch_size=batch_size,
            patience=patience,
            model_prefix=f'{model.model_dir}/deep_learning_model_tscv/tscv_fold_{fold}'
        )

        print("Evaluating model...")
        metrics = model.evaluate_model(X_val, y_val, task='regression')
        fold_metrics.append(metrics)

        print(f"Fold {fold} Metrics: {metrics}")

    return fold_metrics

def main():
    """Test spread forecasting with the DeepLearningModel."""
    warnings.filterwarnings('ignore')

    engineer = FeatureEngineer(
        min_periods=10,
        fill_method='backfill',
        validate=True
    )

    def prepare_spread_data():
        stock1 = pd.read_csv(r"C:\Users\arnav\Downloads\pairs_trading_system\data\raw\AAPL.csv")
        stock2 = pd.read_csv(r'C:\Users\arnav\Downloads\pairs_trading_system\data\raw\MSFT.csv')

        print("Generating technical indicators for stock 1...")
        stock1_features = engineer.generate_features(
            stock1,
            features=['sma', 'ema', 'rsi', 'macd', 'bbands']
        )

        print("Generating technical indicators for stock 2...")
        stock2_features = engineer.generate_features(
            stock2,
            features=['sma', 'ema', 'rsi', 'macd', 'bbands']
        )

        stock1_features['Date'] = pd.to_datetime(stock1_features['Date'])
        stock2_features['Date'] = pd.to_datetime(stock2_features['Date'])

        df = pd.merge(
            stock1_features,
            stock2_features,
            on='Date',
            suffixes=('_stock1', '_stock2')
        )

        df['spread'] = df['Close_stock1'] - df['Close_stock2']

        df['spread_pct'] = df['spread'].pct_change()
        df['spread_sma'] = df['spread'].rolling(window=20).mean()
        df['spread_std'] = df['spread'].rolling(window=20).std()
        df['spread_zscore'] = (df['spread'] - df['spread_sma']) / df['spread_std']

        df['rolling_corr'] = (
            df['Close_stock1'].rolling(window=20)
            .corr(df['Close_stock2'])
        )
        df['Volume_ROC_stock1'] = df['Volume_stock1'].pct_change(periods=20) * 100
        df['Volume_ROC_stock2'] = df['Volume_stock2'].pct_change(periods=20) * 100

        df.dropna(inplace=True)

        print(f"\nFeatures generated. Shape: {df.shape}")
        print("\nFeature columns:", df.columns.tolist())
        return df

    print("Loading and preparing data...")
    df = prepare_spread_data()
    print(f"Data shape: {df.shape}")

    print(df.head(10))

    model = DeepLearningModel()

    feature_columns = [
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
    ]
    sequence_length = 10

    X, y = model.prepare_sequences(
        data=df,
        target_column='spread',
        feature_columns=feature_columns,
        sequence_length=sequence_length,
        scale_data=True
    )
    print(f"Sequence shape: {X.shape}, Target shape: {y.shape}")

    print("\nPerforming Time-Series Cross-Validation...")
    tscv_results = time_series_cross_validation(
        model,
        X,
        y,
        n_splits=5,
        epochs=100,
        batch_size=16,
        patience=10
    )

    print("\nCross-Validation Results:")
    for i, metrics in enumerate(tscv_results, start=1):
        print(f"Fold {i} Metrics: {metrics}")

    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=False
    )

    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=0.2, shuffle=False
    )

    print("Data split sizes:")
    print(f"Train: {X_train.shape}")
    print(f"Validation: {X_val.shape}")
    print(f"Test: {X_test.shape}")

    model.build_lstm_model(
        input_shape=(X.shape[1], X.shape[2]),
        lstm_units=[128, 64],
        dense_units=[32],
        dropout_rate=0.2,
        learning_rate=0.0001
    )

    print("\nTraining model...")
    model.train_model(
        X_train, y_train,
        X_val, y_val,
        epochs=1000,
        batch_size=16,
        patience=15,
        model_prefix=f'{model.model_dir}/deep_learning_model/spread_forecast'
    )

    print("\nEvaluating model...")
    metrics = model.evaluate_model(X_test, y_test, task='regression')
    print("\nTest Metrics:")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")

    print("\nMaking predictions...")
    predictions = model.predict(X_test)
    predictions = model.target_scaler.inverse_transform(predictions.reshape(-1, 1)).flatten()

    results = pd.DataFrame({
        'Actual': model.target_scaler.inverse_transform(y_test.reshape(-1, 1)).flatten(),
        'Predicted': predictions
    })

    print("\nFirst few predictions vs actual values:")
    print(results.head())

    print("\nPlotting training history...")
    model.plot_training_history(
        metrics=['loss', 'mae'],
        title='Spread Forecasting Model Training History',
        save_path=f'{model.model_dir}/deep_learning_model/spread_forecast_history.html'
    )

    return model, results


if __name__ == "__main__":
    model, results = main()
