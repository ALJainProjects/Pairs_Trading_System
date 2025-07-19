from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
from typing import Tuple, List, Optional, Any, Dict
from pathlib import Path
import joblib
import json # Explicitly import json
from sklearn.preprocessing import StandardScaler # Explicitly import StandardScaler
from config.logging_config import logger


class BaseModel(ABC):
    """
    Abstract base class for machine learning and deep learning models.
    Encapsulates common functionality like data scaling, model persistence
    (for scalers and metadata), and a standard train/predict interface.
    Deep learning model (e.g., Keras) saving/loading is handled by
    the specific child classes due to their unique requirements.
    """

    def __init__(self, model_name: str, model_dir: str = "models_data"):
        self.model_name = model_name
        self.model_path = Path(model_dir) / self.model_name
        self.model_path.mkdir(parents=True, exist_ok=True)

        self.model: Optional[Any] = None # This will be set by concrete classes (e.g., Keras Model)
        self.feature_scaler: Optional[StandardScaler] = None
        self.target_scaler: Optional[StandardScaler] = None
        self._scalers_fitted: bool = False # Internal flag for scaler state
        self.feature_names: List[str] = []

    @abstractmethod
    def build_model(self, **kwargs):
        """Build the model architecture."""
        pass

    @abstractmethod
    def train(self, X_train: np.ndarray, y_train: np.ndarray, X_val: np.ndarray, y_val: np.ndarray, **kwargs):
        """
        Train the model.
        Expects pre-scaled and pre-sequenced numpy arrays.
        """
        pass

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions on new data.
        Expects pre-scaled and pre-sequenced numpy array.
        """
        pass

    # --- Scaler Management ---
    def fit_scalers(self, features: pd.DataFrame, target: Optional[pd.Series] = None):
        """
        Fits feature and optional target scalers.
        Should be called once on training data before sequence creation.
        """
        logger.info(f"Fitting scalers for model '{self.model_name}'.")
        self.feature_scaler = StandardScaler()
        self.feature_names = features.columns.tolist()
        self.feature_scaler.fit(features)

        if target is not None:
            self.target_scaler = StandardScaler()
            # Reshape for single feature scaling
            self.target_scaler.fit(target.values.reshape(-1, 1))

        self._scalers_fitted = True
        logger.info("Scalers fitted successfully.")

    def transform_features(self, features: pd.DataFrame) -> pd.DataFrame:
        """Transforms features using the fitted feature scaler."""
        if not self._scalers_fitted or self.feature_scaler is None:
            raise RuntimeError("Feature scaler not fitted. Call fit_scalers first.")
        logger.debug("Transforming features.")
        scaled_features = self.feature_scaler.transform(features)
        return pd.DataFrame(scaled_features, index=features.index, columns=features.columns)

    def transform_target(self, target: pd.Series) -> pd.Series:
        """Transforms target using the fitted target scaler."""
        if not self._scalers_fitted or self.target_scaler is None:
            raise RuntimeError("Target scaler not fitted. Call fit_scalers first.")
        logger.debug("Transforming target.")
        scaled_target = self.target_scaler.transform(target.values.reshape(-1, 1)).flatten()
        return pd.Series(scaled_target, index=target.index)

    def inverse_transform_target(self, scaled_target: np.ndarray) -> np.ndarray:
        """Inverse transforms scaled target values to original scale."""
        if not self._scalers_fitted or self.target_scaler is None:
            raise RuntimeError("Target scaler not fitted. Cannot inverse transform.")
        logger.debug("Inverse transforming target.")
        # Ensure scaled_target is 2D for inverse_transform
        return self.target_scaler.inverse_transform(scaled_target.reshape(-1, 1)).flatten()

    def save_scalers(self):
        """Saves the fitted scalers and metadata."""
        if not self._scalers_fitted:
            raise RuntimeError("Scalers must be fitted before they can be saved.")

        logger.info(f"Saving scalers and metadata to {self.model_path}")
        joblib.dump(self.feature_scaler, self.model_path / "feature_scaler.joblib")

        metadata = {"feature_names": self.feature_names}
        if self.target_scaler:
            joblib.dump(self.target_scaler, self.model_path / "target_scaler.joblib")

        with open(self.model_path / "metadata.json", "w") as f:
            json.dump(metadata, f)
        logger.info("Scalers and metadata saved.")

    def load_scalers(self):
        """Loads the scalers and metadata."""
        scaler_path = self.model_path / "feature_scaler.joblib"
        if not scaler_path.exists():
            raise FileNotFoundError(f"No feature scaler found at {self.model_path}. "
                                    f"Ensure scalers were saved or fit_scalers was called.")

        logger.info(f"Loading scalers and metadata from {self.model_path}")
        self.feature_scaler = joblib.load(scaler_path)

        target_scaler_path = self.model_path / "target_scaler.joblib"
        if target_scaler_path.exists():
            self.target_scaler = joblib.load(target_scaler_path)

        with open(self.model_path / "metadata.json", "r") as f:
            metadata = json.load(f)
            self.feature_names = metadata.get("feature_names", [])

        self._scalers_fitted = True
        logger.info("Scalers and metadata loaded.")

    # --- Overridden Model Saving/Loading for DL Models ---
    def save_model(self):
        """
        Placeholder for saving the core model.
        For deep learning models, this method should be overridden in child classes
        to use their specific saving mechanisms (e.g., model.save() for Keras).
        """
        logger.warning(f"save_model not implemented for {self.__class__.__name__}. "
                       "Please implement in child class for specific model saving logic.")
        # You might choose to raise NotImplementedError here instead,
        # forcing child classes to implement it.
        # raise NotImplementedError("save_model method must be implemented by concrete classes.")

    def load_model(self):
        """
        Placeholder for loading the core model.
        For deep learning models, this method should be overridden in child classes
        to use their specific loading mechanisms (e.g., tf.keras.models.load_model()).
        """
        logger.warning(f"load_model not implemented for {self.__class__.__name__}. "
                       "Please implement in child class for specific model loading logic.")
        # raise NotImplementedError("load_model method must be implemented by concrete classes.")

    # --- is_fitted for the overall model state ---
    # This flag should now reflect if the *model itself* (e.g., Keras model) is trained.
    # The _scalers_fitted handles scaler state.
    @property
    def is_fitted(self) -> bool:
        """Returns True if the model (not just scalers) has been trained."""
        return self._is_model_fitted # Internal flag to be set by actual train method

    @is_fitted.setter
    def is_fitted(self, value: bool):
        """Sets the fitted state of the model."""
        self._is_model_fitted = value