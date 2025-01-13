"""
Machine Learning Models Module (with Plotly visualizations)

Enhancements for Pair Trading:
 - create_pair_features (spread/ratio)
 - classification & regression
 - evaluate & feature importance
 - Plotly-based feature importance chart
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
from typing import Tuple, Dict, List
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    RandomForestRegressor,
    GradientBoostingRegressor
)
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    mean_squared_error,
    r2_score
)
from config.logging_config import logger


class MachineLearningModel:
    """
    MachineLearningModel provides classification and regression for pair-trading tasks:
      - Build features from a pair (e.g., spread, ratio)
      - Classification for signal generation
      - Regression for spread forecasting
      - Feature importances for tree-based models
      - Plotly-based feature importance charts
    """

    def __init__(self):
        logger.info("Initializing MachineLearningModel.")
        self.scaler = StandardScaler()
        self.classification_models = {
            'LogisticRegression': LogisticRegression(),
            'RandomForestClassifier': RandomForestClassifier(n_estimators=100),
            'GradientBoostingClassifier': GradientBoostingClassifier(n_estimators=100),
        }
        self.regression_models = {
            'LinearRegression': LinearRegression(),
            'RandomForestRegressor': RandomForestRegressor(n_estimators=100),
            'GradientBoostingRegressor': GradientBoostingRegressor(n_estimators=100),
        }

    def create_pair_features(
        self, asset1: pd.Series, asset2: pd.Series, window: int = 20, feature_mode: str = "spread"
    ) -> pd.DataFrame:
        logger.info(f"Creating pair features with mode={feature_mode}, window={window}.")
        if feature_mode not in ["spread", "ratio"]:
            raise ValueError("feature_mode must be 'spread' or 'ratio'.")

        if len(asset1) != len(asset2):
            raise ValueError("Asset series must be the same length.")
        df = pd.DataFrame(index=asset1.index)

        if feature_mode == "spread":
            df["feature"] = asset1 - asset2
        else:  # ratio
            df["feature"] = asset1 / (asset2 + 1e-12)

        df["rolling_mean"] = df["feature"].rolling(window).mean()
        df["rolling_std"] = df["feature"].rolling(window).std()
        df.dropna(inplace=True)
        return df

    def train_classification_model(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        model_name: str = "RandomForestClassifier",
        cv: int = 5,
        scoring: str = "accuracy"
    ) -> Tuple[object, float]:
        logger.info(f"Training classification model: {model_name} with CV={cv}")
        if model_name not in self.classification_models:
            raise ValueError(f"Unknown classification model: {model_name}")

        X_scaled = self.scaler.fit_transform(X)
        model = self.classification_models[model_name]
        model.fit(X_scaled, y)

        scores = cross_val_score(model, X_scaled, y, cv=cv, scoring=scoring)
        mean_score = scores.mean()
        logger.debug(f"Cross-validation {scoring}={mean_score:.4f}")
        return model, mean_score

    def train_regression_model(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        model_name: str = "LinearRegression",
        cv: int = 5
    ) -> Tuple[object, float]:
        logger.info(f"Training regression model: {model_name} with CV={cv}")
        if model_name not in self.regression_models:
            raise ValueError(f"Unknown regression model: {model_name}")

        X_scaled = self.scaler.fit_transform(X)
        model = self.regression_models[model_name]
        model.fit(X_scaled, y)

        scores = cross_val_score(model, X_scaled, y, cv=cv, scoring="neg_mean_squared_error")
        mean_mse = -scores.mean()
        logger.debug(f"Cross-validation MSE={mean_mse:.4f}")
        return model, mean_mse

    def evaluate_classification_model(self, model, X_test: pd.DataFrame, y_test: pd.Series) -> Dict:
        logger.info("Evaluating classification model.")
        X_test_scaled = self.scaler.transform(X_test)
        y_pred = model.predict(X_test_scaled)
        report = classification_report(y_test, y_pred, output_dict=True)
        matrix = confusion_matrix(y_test, y_pred).tolist()
        return {
            "classification_report": report,
            "confusion_matrix": matrix
        }

    def evaluate_regression_model(self, model, X_test: pd.DataFrame, y_test: pd.Series) -> Dict:
        logger.info("Evaluating regression model.")
        X_test_scaled = self.scaler.transform(X_test)
        y_pred = model.predict(X_test_scaled)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        return {
            "Mean Squared Error": mse,
            "R-squared": r2
        }

    def feature_importance(self, model, feature_names: List[str]) -> pd.Series:
        """
        Retrieve feature importances from tree-based models. Return a Series sorted by importance.
        """
        if hasattr(model, "feature_importances_"):
            importances = model.feature_importances_
            s = pd.Series(importances, index=feature_names).sort_values(ascending=False)
            return s
        logger.warning("Model does not have feature_importances_.")
        return pd.Series()

    def plot_feature_importance(self, feature_importances: pd.Series, title: str = "Feature Importances") -> None:
        """
        Use Plotly to show a horizontal bar chart of feature importances.
        """
        if feature_importances.empty:
            logger.warning("No feature importances to plot.")
            return

        logger.info("Plotting feature importances with Plotly.")
        import plotly.graph_objects as go

        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=feature_importances.values[::-1],  # Reverse for top-down
            y=feature_importances.index[::-1],
            orientation='h'
        ))
        fig.update_layout(
            title=title,
            xaxis_title="Importance",
            yaxis_title="Features",
            template="plotly_white",
            margin=dict(l=100)
        )
        fig.show()
