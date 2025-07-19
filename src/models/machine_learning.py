from pathlib import Path

import pandas as pd
import numpy as np
import joblib  # For saving/loading scikit-learn models
from typing import Tuple, Dict, Any, Optional, Union, List
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler  # Explicitly import StandardScaler for type hints
from .base_model import BaseModel
from config.logging_config import logger


class MachineLearningModel(BaseModel):
    """
    A wrapper for scikit-learn models providing a unified interface for training,
    prediction, and persistence, tailored for financial time series.
    It integrates data scaling via BaseModel's methods and handles both
    classification and regression outputs.
    """

    def __init__(self,
                 model_name: str,
                 model_class: Any = RandomForestClassifier,
                 params: Dict = None,
                 model_type: str = 'classifier',  # 'classifier' or 'regressor'
                 # scaler_class is now handled by BaseModel's instantiation of StandardScaler
                 # It's kept here as a parameter for potential future flexibility or custom scaler injection
                 scaler_class: Any = StandardScaler,
                 model_dir: Optional[Union[str, Path]] = None):

        # BaseModel now manages feature_scaler and target_scaler
        super().__init__(model_name, model_dir)
        self.model_class = model_class
        self.params = params if params is not None else {'n_estimators': 100, 'random_state': 42, 'n_jobs': -1}
        self.model_type = model_type.lower()
        if self.model_type not in ['classifier', 'regressor']:
            raise ValueError("model_type must be 'classifier' or 'regressor'.")

        self.scaler_class = scaler_class

        # feature_names is now managed by BaseModel via fit_scalers
        # self.feature_names: Optional[List[str]] = None # This is now self.feature_names from BaseModel

        self.build_model()  # This initializes self.model
        logger.info(f"MachineLearningModel '{model_name}' initialized as a {self.model_type}.")

    def build_model(self, **kwargs):
        """Builds (initializes) the scikit-learn model instance."""
        try:
            self.model = self.model_class(**self.params)
            # Scaler instantiation is now handled by BaseModel's fit_scalers
            logger.info(f"Model instance '{self.model_class.__name__}' created with params: {self.params}")
        except Exception as e:
            logger.error(f"Error building model {self.model_name}: {e}")
            raise

    # The prepare_data method from the original code has been removed.
    # Its functionality is now provided by BaseModel's fit_scalers,
    # transform_features, and transform_target methods.

    def train(self, X_train: pd.DataFrame, y_train: pd.Series, X_val: Optional[pd.DataFrame] = None,
              y_val: Optional[pd.Series] = None, **kwargs):
        """
        Trains the machine learning model.
        Scales data internally using BaseModel's scalers.

        Args:
            X_train (pd.DataFrame): Training features.
            y_train (pd.Series): Training targets.
            X_val (pd.DataFrame, optional): Validation features (not directly used by all scikit-learn models).
                                            Note: For scikit-learn models, validation data is typically
                                            used for hyperparameter tuning or early stopping, which
                                            isn't generically implemented here.
            y_val (pd.Series, optional): Validation targets.
        """
        if self.model is None:
            self.build_model()

        logger.info(f"Training {self.model_name} (Type: {self.model_type})...")

        # Fit scalers using BaseModel's method on original dataframes
        # This will also set self.feature_names in BaseModel
        self.fit_scalers(features=X_train, target=y_train)

        # Transform data using fitted scalers
        X_train_scaled_df = self.transform_features(X_train)
        y_train_scaled_series = self.transform_target(y_train)

        # Convert to NumPy arrays for scikit-learn
        X_train_scaled_np = X_train_scaled_df.values
        # BaseModel's transform_target already flattens to 1D if target is a single feature
        y_train_scaled_np = y_train_scaled_series.values

        # Train the model
        try:
            self.model.fit(X_train_scaled_np, y_train_scaled_np)
            self.is_fitted = True  # Set BaseModel's is_fitted flag
            logger.info("Training complete.")
        except Exception as e:
            logger.error(f"Error during model training for {self.model_name}: {e}")
            self.is_fitted = False
            raise

    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Makes predictions using the trained model.
        Returns predictions and confidence scores (for classifiers) or just predictions.
        Scales data internally using BaseModel's scalers.

        Args:
            X (pd.DataFrame): Input features for prediction.

        Returns:
            pd.DataFrame: DataFrame containing 'prediction' and 'confidence' (if applicable).
        """
        if not self.is_fitted:
            raise RuntimeError("Model must be trained before prediction.")

        # Ensure scalers are loaded if model was just loaded and not trained in current session
        # This check covers cases where you load a trained model and then predict
        if not self._scalers_fitted:
            try:
                self.load_scalers()
            except FileNotFoundError:
                logger.error("Scalers not found. Cannot transform data for prediction. "
                             "Ensure scalers were saved or the model was trained in this session.")
                raise

        # Ensure the scikit-learn model instance is loaded if it's not in memory
        if self.model is None:
            self.load_model()  # This calls the overridden load_model in this class
            if self.model is None:
                raise RuntimeError("Model instance not available for prediction after attempted loading.")

        # Transform data using fitted feature scaler
        X_scaled_df = self.transform_features(X)
        X_scaled_np = X_scaled_df.values

        # Make predictions
        raw_predictions = self.model.predict(X_scaled_np)

        # Inverse transform predictions if a target scaler was used (for regressors)
        # Note: self.target_scaler is from BaseModel
        if self.model_type == 'regressor' and self.target_scaler is not None:
            predictions = self.inverse_transform_target(raw_predictions)
        else:
            predictions = raw_predictions

        result_df = pd.DataFrame({'prediction': predictions}, index=X.index)

        if self.model_type == 'classifier' and hasattr(self.model, "predict_proba"):
            probabilities = self.model.predict_proba(X_scaled_np)
            # Max probability for the predicted class
            confidence = np.max(probabilities, axis=1)
            result_df['confidence'] = confidence

        logger.info(f"Predictions made using {self.model_name}.")
        return result_df

    def get_feature_importance(self) -> pd.Series:
        """
        Retrieves feature importance if available from the model.

        Returns:
            pd.Series: Series of feature importances, sorted descending.
        """
        if not self.is_fitted:
            logger.warning("Model not fitted. Cannot get feature importance.")
            return pd.Series(dtype=float) # Return empty Series with specified dtype

        # feature_names are now managed by BaseModel's self.feature_names
        if self.feature_names is None or len(self.feature_names) == 0:
            logger.warning("Feature names not set. Cannot map importance to features. Call train first.")
            return pd.Series(dtype=float)

        if hasattr(self.model, "feature_importances_"):
            return pd.Series(self.model.feature_importances_, index=self.feature_names).sort_values(ascending=False)
        elif hasattr(self.model, "coef_") and self.model_type == 'regressor':
            # For linear models, coefficients can be seen as importance
            # For multi-output/multi-class, might need more complex handling
            if self.model.coef_.ndim > 1:
                logger.warning(
                    "Model has multi-dimensional coefficients; returning sum of absolute values per feature.")
                return pd.Series(np.sum(np.abs(self.model.coef_), axis=0), index=self.feature_names).sort_values(
                    ascending=False)
            else:
                return pd.Series(self.model.coef_, index=self.feature_names).sort_values(ascending=False)

        logger.warning(
            f"Model '{self.model_name}' of type '{self.model_class.__name__}' does not support feature importance directly.")
        return pd.Series(dtype=float)

    def save_model(self):
        """
        Saves the trained scikit-learn model using joblib.
        Also calls BaseModel's method to save scalers and metadata.
        This overrides the placeholder in BaseModel.
        """
        if not self.model:
            logger.warning("No scikit-learn model instance to save.")
            return

        model_filepath = self.model_path / "model.joblib"
        try:
            joblib.dump(self.model, model_filepath)
            logger.info(f"{self.model_name} (scikit-learn model) saved to {model_filepath}")
        except Exception as e:
            logger.error(f"Error saving {self.model_name} (scikit-learn model): {e}")
            raise

        # Save scalers and metadata via BaseModel's method
        self.save_scalers()

    def load_model(self):
        """
        Loads a pre-trained scikit-learn model using joblib.
        Also calls BaseModel's method to load scalers and metadata.
        This overrides the placeholder in BaseModel.
        """
        model_filepath = self.model_path / "model.joblib"

        if model_filepath.exists():
            try:
                self.model = joblib.load(model_filepath)
                self.is_fitted = True  # Set BaseModel's is_fitted flag
                logger.info(f"{self.model_name} (scikit-learn model) loaded from {model_filepath}")
            except Exception as e:
                logger.error(f"Error loading {self.model_name} (scikit-learn model): {e}")
                self.model = None
                self.is_fitted = False
        else:
            logger.warning(f"No saved scikit-learn model found at {model_filepath} for {self.model_name}.")

        # Load scalers and metadata via BaseModel's method
        try:
            self.load_scalers()
        except FileNotFoundError as e:
            logger.warning(f"Could not load scalers for {self.model_name}: {e}. "
                           "Model might not be fully operational without correct scaling.")
        except Exception as e:
            logger.error(f"Unexpected error loading scalers for {self.model_name}: {e}")


def triple_barrier_labeling(
        prices: pd.Series,
        events: pd.Series,
        pt_sl: Tuple[float, float],  # (profit_target_multiplier, stop_loss_multiplier)
        max_period: int,  # Max number of bars to look forward for a hit
        min_return: float = 0.0,  # Minimum absolute return to consider a hit, else timeout
        verbose: bool = True  # For logging progress
) -> pd.DataFrame:
    """
    Creates labels for a classification model using the triple-barrier method.
    This method defines labels based on which of three barriers (profit target,
    stop loss, or time limit) is hit first.

    Args:
        prices (pd.Series): A series of asset prices (indexed by date/time).
            Must be sorted by index in ascending order.
        events (pd.Series): A series of event timestamps (e.g., signals for trade entry)
            for which to apply barriers. The index should correspond to entry times.
            Values in the series itself are not used, only the index.
        pt_sl (Tuple[float, float]): Profit target (pt) and stop loss (sl) multipliers.
                                     e.g., (0.01, 0.005) means 1% profit target, 0.5% stop loss.
                                     These are applied symmetrically for long/short, but the
                                     current implementation assumes long positions for labels 1/-1.
        max_period (int): Maximum number of bars (time steps) to look forward for a barrier hit
                          from the event's timestamp.
        min_return (float): Minimum absolute return required to consider a barrier hit.
                            If the actual absolute return at the first barrier hit is less
                            than this, the label is forced to 0 (timeout). This filters out
                            insignificant movements.
        verbose (bool): If True, logs progress during processing.

    Returns:
        pd.DataFrame: DataFrame with 'return' (actual return at hit) and 'label' (-1, 0, or 1).
                      The 'label' column:
                      1: Profit target hit first.
                      -1: Stop loss hit first.
                      0: Time limit (timeout) hit first, or the absolute return was below min_return.
                      The index of the DataFrame matches events.index.
    """
    if prices.empty or events.empty:
        logger.warning("Prices or events Series is empty. Cannot perform triple barrier labeling.")
        return pd.DataFrame(columns=['return', 'label'])

    if not all(idx in prices.index for idx in events.index):
        # Filter events to ensure all indices are in prices.
        original_event_count = len(events)
        events = events[events.index.isin(prices.index)]
        if len(events) < original_event_count:
            logger.warning(f"Filtered out {original_event_count - len(events)} event indices not present in prices.")
        if events.empty:
            logger.warning("All event indices filtered out or no valid events after filtering. Returning empty DataFrame.")
            return pd.DataFrame(columns=['return', 'label'])

    if not prices.index.is_monotonic_increasing:
        logger.warning("Prices index is not monotonic increasing. Sorting prices for correct look-forward behavior.")
        prices = prices.sort_index()

    pt, sl = pt_sl[0], pt_sl[1]
    out = pd.DataFrame(index=events.index, columns=['return', 'label'], dtype=float)

    # Convert prices index to positions for faster iloc access
    # This creates a mapping from datetime index to integer position
    price_indices_map = pd.Series(data=np.arange(len(prices)), index=prices.index)

    total_events = len(events)
    log_interval = max(1, total_events // 10)  # Log every 10%

    for i, event_idx in enumerate(events.index):
        if verbose and (i + 1) % log_interval == 0:
            logger.info(f"Processing event {i + 1}/{total_events} for triple barrier labeling.")

        # Get the starting position in the prices array
        start_pos = price_indices_map.loc[event_idx]

        # Define the end position for the look-forward window
        # +1 to include the last bar in slice for iloc/Python slicing behavior
        end_pos = min(start_pos + max_period + 1, len(prices))

        # Extract the price path for the current event's look-forward window
        path = prices.iloc[start_pos: end_pos]

        if len(path) < 2:  # Need at least entry price and one subsequent price to check barriers
            logger.debug(f"Path too short for event_idx {event_idx}. Skipping. Path length: {len(path)}")
            continue

        entry_price = path.iloc[0]

        # Calculate barrier levels for a long position (assuming standard PT/SL logic)
        profit_target_level = entry_price * (1 + pt)
        stop_loss_target_level = entry_price * (1 - sl)

        # Create boolean masks for when prices cross the barriers within the path
        # Use .iloc[1:] to exclude the entry price itself from barrier checks
        path_after_entry = path.iloc[1:]

        pt_hit_mask = path_after_entry >= profit_target_level
        sl_hit_mask = path_after_entry <= stop_loss_target_level

        # Find the first time index where each barrier is hit
        pt_hit_time = pt_hit_mask.idxmax() if pt_hit_mask.any() else None
        sl_hit_time = sl_hit_mask.idxmax() if sl_hit_mask.any() else None

        # Consolidate hit times and find the earliest hit
        # Include timeout barrier, which is simply the last timestamp in the `path` slice
        hit_times = []
        if pt_hit_time is not None:
            hit_times.append((pt_hit_time, 1))  # (timestamp, label for profit)
        if sl_hit_time is not None:
            hit_times.append((sl_hit_time, -1))  # (timestamp, label for loss)

        # The time-out barrier is the last timestamp in the `path` slice
        timeout_time = path.index[-1]
        hit_times.append((timeout_time, 0))  # (timestamp, label for timeout)

        # Sort by timestamp to find the first hit
        hit_times.sort(key=lambda x: x[0])

        # The first element in the sorted list is the actual first hit
        first_hit_timestamp, label = hit_times[0]

        final_price = prices.loc[first_hit_timestamp]
        actual_return = (final_price / entry_price) - 1

        # Apply minimum return threshold: if actual_return is too small, treat as timeout
        if abs(actual_return) < min_return:
            label = 0  # Force label to 0 if movement is insignificant

        out.loc[event_idx, 'return'] = actual_return
        out.loc[event_idx, 'label'] = label

    logger.info(f"Triple barrier labeling complete. Labeled {len(out)} events.")
    return out