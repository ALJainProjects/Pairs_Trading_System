from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
from typing import Tuple, List, Optional, Any, Dict
from pathlib import Path
import joblib
from config.logging_config import logger


class BaseModel(ABC):
    """
    Abstract base class for machine learning and deep learning models.
    Encapsulates common functionality like data scaling, model persistence,
    and a standard train/predict interface.
    """

    def __init__(self, model_name: str, model_dir: str = "models_data"):
        self.model_name = model_name
        self.model_path = Path(model_dir) / self.model_name
        self.model_path.mkdir(parents=True, exist_ok=True)

        self.model: Optional[Any] = None
        self.feature_scaler: Optional[Any] = None
        self.target_scaler: Optional[Any] = None
        self.is_fitted: bool = False
        self.feature_names: List[str] = []

    @abstractmethod
    def build_model(self, **kwargs):
        """Build the model architecture."""
        pass

    @abstractmethod
    def train(self, X_train: pd.DataFrame, y_train: pd.Series, X_val: pd.DataFrame, y_val: pd.Series, **kwargs):
        """Train the model."""
        pass

    @abstractmethod
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions on new data."""
        pass

    def prepare_data(self,
                     features: pd.DataFrame,
                     target: Optional[pd.Series] = None
                     ) -> Tuple[pd.DataFrame, Optional[pd.Series]]:
        """
        Scales features and target. Fits scalers if the model hasn't been fitted yet.
        """
        if not self.is_fitted:
            logger.info(f"Fitting scalers for model '{self.model_name}'.")
            from sklearn.preprocessing import StandardScaler
            self.feature_scaler = StandardScaler()
            self.feature_names = features.columns.tolist()
            scaled_features = self.feature_scaler.fit_transform(features)

            scaled_target = None
            if target is not None:
                self.target_scaler = StandardScaler()
                scaled_target = self.target_scaler.fit_transform(target.values.reshape(-1, 1)).flatten()

            self.is_fitted = True
        else:
            if not self.feature_scaler:
                raise RuntimeError("Model is marked as fitted, but scaler is missing.")
            scaled_features = self.feature_scaler.transform(features)

            scaled_target = None
            if target is not None:
                if not self.target_scaler:
                    raise RuntimeError("Target scaler is missing for transformation.")
                scaled_target = self.target_scaler.transform(target.values.reshape(-1, 1)).flatten()

        return pd.DataFrame(scaled_features, index=features.index, columns=features.columns), \
            pd.Series(scaled_target, index=target.index) if scaled_target is not None else None

    def save_model(self):
        """Saves the model, scalers, and metadata to the model's directory."""
        if not self.is_fitted or self.model is None:
            raise RuntimeError("Model must be trained before it can be saved.")

        logger.info(f"Saving model and scalers to {self.model_path}")
        joblib.dump(self.model, self.model_path / "model.joblib")
        joblib.dump(self.feature_scaler, self.model_path / "feature_scaler.joblib")

        metadata = {"feature_names": self.feature_names}
        if self.target_scaler:
            joblib.dump(self.target_scaler, self.model_path / "target_scaler.joblib")

        with open(self.model_path / "metadata.json", "w") as f:
            import json
            json.dump(metadata, f)

    def load_model(self):
        """Loads the model, scalers, and metadata from the model's directory."""
        if not (self.model_path / "model.joblib").exists():
            raise FileNotFoundError(f"No model file found at {self.model_path}")

        logger.info(f"Loading model and scalers from {self.model_path}")
        self.model = joblib.load(self.model_path / "model.joblib")
        self.feature_scaler = joblib.load(self.model_path / "feature_scaler.joblib")

        if (self.model_path / "target_scaler.joblib").exists():
            self.target_scaler = joblib.load(self.model_path / "target_scaler.joblib")

        with open(self.model_path / "metadata.json", "r") as f:
            import json
            metadata = json.load(f)
            self.feature_names = metadata.get("feature_names", [])

        self.is_fitted = True