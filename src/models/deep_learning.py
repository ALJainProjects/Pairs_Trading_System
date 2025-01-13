"""
Deep Learning Models Module (with Plotly visualizations)

Enhancements for Pair Trading:
 - LSTM for time-series forecasting
 - Dense model for classification
 - Plotly-based training history
"""

import pandas as pd
import numpy as np
from typing import Tuple, Dict, List
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, LSTM, Dropout
from tensorflow.python.keras.callbacks import EarlyStopping
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, classification_report, confusion_matrix
from config.logging_config import logger
import plotly.graph_objects as go


class DeepLearningModel:
    """
    DeepLearningModel with Plotly-based plotting.
    """

    def __init__(self):
        logger.info("Initializing DeepLearningModel.")
        self.scaler = StandardScaler()

    # -----------------------------------------------------------------------
    # 1) LSTM for Time-Series (Spread, Ratio, etc.)
    # -----------------------------------------------------------------------
    def prepare_sequences(
        self, data: pd.DataFrame, target_column: str, sequence_length: int = 10
    ) -> Tuple[np.ndarray, np.ndarray]:
        logger.info("Preparing sequences for LSTM.")
        if data.isnull().any().any():
            raise ValueError("Data contains NaNs.")
        if target_column not in data.columns:
            raise ValueError(f"Target column '{target_column}' not found.")

        sequences, targets = [], []
        for i in range(len(data) - sequence_length):
            seq = data.iloc[i : i + sequence_length].values
            t = data.iloc[i + sequence_length][target_column]
            sequences.append(seq)
            targets.append(t)
        X = np.array(sequences)
        y = np.array(targets)
        logger.debug(f"Sequences shape: {X.shape}, Targets shape: {y.shape}")
        return X, y

    def build_lstm_model(
        self, input_shape: Tuple[int, int], units: int = 50, dropout: float = 0.2
    ) -> Sequential:
        logger.info("Building LSTM model.")
        model = Sequential()
        model.add(LSTM(units, return_sequences=True, input_shape=input_shape))
        model.add(Dropout(dropout))
        model.add(LSTM(units))
        model.add(Dropout(dropout))
        model.add(Dense(1, activation='linear'))
        model.compile(optimizer='adam', loss='mean_squared_error')
        return model

    def train_lstm_model(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        units: int = 50,
        dropout: float = 0.2,
        epochs: int = 100,
        batch_size: int = 32
    ) -> Sequential:
        logger.info("Training LSTM model.")
        model = self.build_lstm_model(
            input_shape=(X_train.shape[1], X_train.shape[2]),
            units=units,
            dropout=dropout
        )
        early_stop = EarlyStopping(
            monitor='val_loss', patience=10, restore_best_weights=True
        )
        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[early_stop],
            verbose=1
        )
        logger.info("LSTM model training completed.")
        self._history = history  # store for optional plotting
        return model

    def evaluate_lstm_model(self, model: Sequential, X_test: np.ndarray, y_test: np.ndarray) -> Dict:
        logger.info("Evaluating LSTM model.")
        preds = model.predict(X_test)
        mse = mean_squared_error(y_test, preds)
        ss_res = np.sum((y_test - preds.flatten()) ** 2)
        ss_tot = np.sum((y_test - np.mean(y_test)) ** 2)
        r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
        return {"Mean Squared Error": mse, "R-squared": r2}

    # -----------------------------------------------------------------------
    # 2) Dense for Classification
    # -----------------------------------------------------------------------
    def build_dense_model(
        self, input_dim: int, layers: List[int] = [64, 32], dropout: float = 0.3
    ) -> Sequential:
        logger.info("Building Dense model for classification.")
        model = Sequential()
        model.add(Dense(layers[0], input_dim=input_dim, activation='relu'))
        model.add(Dropout(dropout))
        for units in layers[1:]:
            model.add(Dense(units, activation='relu'))
            model.add(Dropout(dropout))
        # Binary classification output
        model.add(Dense(1, activation='sigmoid'))
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        return model

    def train_dense_model(
        self,
        model: Sequential,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        epochs: int = 100,
        batch_size: int = 32
    ) -> Sequential:
        logger.info("Training Dense model (classification).")
        early_stop = EarlyStopping(
            monitor='val_loss', patience=10, restore_best_weights=True
        )
        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[early_stop],
            verbose=1
        )
        logger.info("Dense model training completed.")
        self._history = history
        return model

    def evaluate_dense_model(self, model: Sequential, X_test: np.ndarray, y_test: np.ndarray) -> Dict:
        logger.info("Evaluating Dense model (classification).")
        preds = (model.predict(X_test) > 0.5).astype(int)
        report = classification_report(y_test, preds, output_dict=True)
        matrix = confusion_matrix(y_test, preds).tolist()
        return {"classification_report": report, "confusion_matrix": matrix}

    # -----------------------------------------------------------------------
    # 3) Plotly for training history
    # -----------------------------------------------------------------------
    def plot_training_history(self, metric: str = 'loss', title: str = 'Training History') -> None:
        """
        Plot the training and validation metrics over epochs using Plotly.
        """
        if not hasattr(self, '_history'):
            logger.warning("No training history found. Train a model first.")
            return

        history = self._history
        logger.info("Plotting training history with Plotly.")
        metric_vals = history.history.get(metric)
        if metric_vals is None:
            logger.warning(f"No '{metric}' metric in history.")
            return

        epochs_range = list(range(len(metric_vals)))
        fig = go.Figure()
        # Train metric
        fig.add_trace(go.Scatter(
            x=epochs_range,
            y=metric_vals,
            mode='lines',
            name=f"Train {metric}"
        ))

        # Val metric if available
        val_metric = f'val_{metric}'
        if val_metric in history.history:
            fig.add_trace(go.Scatter(
                x=epochs_range,
                y=history.history[val_metric],
                mode='lines',
                name=f"Validation {metric}"
            ))

        fig.update_layout(
            title=title,
            xaxis_title="Epoch",
            yaxis_title=metric.capitalize(),
            template="plotly_white"
        )
        fig.show()
