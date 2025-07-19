import pandas as pd
import numpy as np
from typing import Tuple, Dict, Any
from sklearn.ensemble import RandomForestClassifier
from .base_model import BaseModel
from config.logging_config import logger

class MachineLearningModel(BaseModel):
    """
    A wrapper for scikit-learn models providing a unified interface for training,
    prediction, and persistence, tailored for financial time series.
    """
    def __init__(self, model_name: str, model_class: Any = RandomForestClassifier, params: Dict = None):
        super().__init__(model_name)
        self.model_class = model_class
        self.params = params if params is not None else {'n_estimators': 100, 'random_state': 42, 'n_jobs': -1}
        self.build_model()

    def build_model(self, **kwargs):
        self.model = self.model_class(**self.params)

    def train(self, X_train: pd.DataFrame, y_train: pd.Series, X_val: pd.DataFrame = None, y_val: pd.Series = None, **kwargs):
        """
        Trains the machine learning model.
        """
        logger.info(f"Training {self.model_name}...")
        
        X_train_scaled, y_train_scaled = self.prepare_data(X_train, y_train)
        
        self.model.fit(X_train_scaled, y_train_scaled if y_train_scaled is not None else y_train)
        self.is_fitted = True
        logger.info("Training complete.")

    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        if not self.is_fitted:
            raise RuntimeError("Model must be trained before prediction.")
        
        X_scaled, _ = self.prepare_data(X)
        predictions = self.model.predict(X_scaled)
        
        if hasattr(self.model, "predict_proba"):
            probabilities = self.model.predict_proba(X_scaled)
            confidence = np.max(probabilities, axis=1)
            return pd.DataFrame({
                'prediction': predictions,
                'confidence': confidence
            }, index=X.index)
        
        return pd.DataFrame({'prediction': predictions}, index=X.index)

    def get_feature_importance(self) -> pd.Series:
        if not self.is_fitted or not hasattr(self.model, "feature_importances_"):
            return pd.Series()
        return pd.Series(self.model.feature_importances_, index=self.feature_names).sort_values(ascending=False)

def triple_barrier_labeling(
    prices: pd.Series,
    events: pd.Series,
    pt_sl: Tuple[float, float],
    max_period: int
) -> pd.DataFrame:
    """
    Creates labels for a classification model using the triple-barrier method.
    """
    pt, sl = pt_sl[0], pt_sl[1]
    out = pd.DataFrame(index=events.index, columns=['return', 'label'])
    
    for event_idx in events.index:
        path = prices.loc[event_idx:].iloc[:max_period]
        entry_price = path.iloc[0]
        
        profit_target = entry_price * (1 + pt)
        stop_loss_target = entry_price * (1 - sl)
        
        pt_hit_time = path[path >= profit_target].index.min()
        sl_hit_time = path[path <= stop_loss_target].index.min()
        
        first_hit_time = pd.Series([pt_hit_time, sl_hit_time, path.index[-1]]).dropna().min()
        
        if first_hit_time == pt_hit_time:
            out.loc[event_idx, 'label'] = 1
        elif first_hit_time == sl_hit_time:
            out.loc[event_idx, 'label'] = -1
        else:
            out.loc[event_idx, 'label'] = 0
            
        out.loc[event_idx, 'return'] = (prices.loc[first_hit_time] / entry_price) - 1

    return out