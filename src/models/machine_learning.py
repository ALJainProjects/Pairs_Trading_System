"""
Machine Learning Models Module with Feature Engineering Integration

Features:
 - Integrated feature engineering
 - Multiple classification & regression models_data
 - Kalman filter for dynamic hedge ratio
 - Feature importance analysis
 - Plotly visualizations
"""
from pathlib import Path

import pandas as pd
import numpy as np
from typing import Tuple, Dict, List, Union, Any
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.model_selection import TimeSeriesSplit
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
    r2_score,
    accuracy_score
)
import statsmodels.api as sm
import plotly.graph_objects as go
from filterpy.kalman import KalmanFilter

from config.settings import MODEL_DIR
from src.data.feature_engineering import FeatureEngineer
from config.logging_config import logger
import warnings

warnings.filterwarnings('ignore')


class MachineLearningModel:
    """Machine Learning Model for pairs trading with integrated feature engineering."""

    def __init__(self):
        logger.info("Initializing MachineLearningModel.")
        self.scaler = StandardScaler()
        self.feature_engineer = FeatureEngineer(
            min_periods=10,
            fill_method='backfill',
            validate=True
        )

        self.classification_models = {
            'LogisticRegression': LogisticRegression(random_state=42),
            'RandomForestClassifier': RandomForestClassifier(n_estimators=100, random_state=42),
            'GradientBoostingClassifier': GradientBoostingClassifier(n_estimators=100, random_state=42),
        }

        self.regression_models = {
            'LinearRegression': LinearRegression(),
            'RandomForestRegressor': RandomForestRegressor(n_estimators=100, random_state=42),
            'GradientBoostingRegressor': GradientBoostingRegressor(n_estimators=100, random_state=42),
        }

        self.default_model = RandomForestRegressor(n_estimators=100, random_state=42)

    def calculate_dynamic_hedge_ratio(self, price1: pd.Series, price2: pd.Series) -> pd.Series:
        """Calculate dynamic hedge ratio using Kalman Filter."""
        kf = KalmanFilter(dim_x=1, dim_z=1)
        kf.x = np.array([[1.0]])
        kf.P *= 10.0
        kf.R = 1.0
        kf.Q = 0.001

        hedge_ratios = np.zeros(len(price1))

        for t in range(len(price1)):
            kf.predict()
            if not np.isnan(price1.iloc[t]) and not np.isnan(price2.iloc[t]) and price2.iloc[t] != 0:
                ratio = price1.iloc[t] / price2.iloc[t]
                kf.update(ratio)
            hedge_ratios[t] = kf.x[0, 0]

        return pd.Series(hedge_ratios, index=price1.index)

    def prepare_features(self,
                         price1: pd.Series,
                         price2: pd.Series,
                         windows: List[int] = (5, 20, 60),
                         lag_windows: List[int] = (1, 2, 3, 5, 10)) -> pd.DataFrame:
        """Prepare features without lookahead bias, including lagged spread features."""
        features = pd.DataFrame(index=price1.index)

        hedge_ratio = self.calculate_dynamic_hedge_ratio(price1, price2)
        spread = price1 - hedge_ratio * price2

        for lag in lag_windows:
            features[f'spread_lag_{lag}'] = spread.shift(lag)
            features[f'spread_return_{lag}'] = spread.pct_change(lag)

        for window in windows:
            lagged_spread = spread.shift(1)

            features[f'spread_ma_{window}'] = lagged_spread.rolling(window).mean()
            features[f'spread_std_{window}'] = lagged_spread.rolling(window).std()
            features[f'spread_zscore_{window}'] = (
                    (lagged_spread - features[f'spread_ma_{window}']) /
                    features[f'spread_std_{window}']
            )

            features[f'spread_momentum_{window}'] = lagged_spread.diff(window)
            features[f'spread_roc_{window}'] = lagged_spread.pct_change(window)

            features[f'spread_vol_{window}'] = lagged_spread.rolling(window).std()
            features[f'spread_vol_ratio_{window}'] = (
                    features[f'spread_vol_{window}'] /
                    features[f'spread_vol_{window}'].shift(window)
            )

        df1 = pd.DataFrame({'Close': price1})
        df2 = pd.DataFrame({'Close': price2})

        tech_features1 = self.feature_engineer.generate_features(
            df1, features=['sma', 'rsi', 'bbands', 'macd']
        )
        tech_features2 = self.feature_engineer.generate_features(
            df2, features=['sma', 'rsi', 'bbands', 'macd']
        )

        for col in tech_features1.columns:
            if col != 'Close':
                features[f'{col}_1'] = tech_features1[col]
                features[f'{col}_2'] = tech_features2[col]

        for window in windows:
            returns1 = price1.pct_change()
            returns2 = price2.pct_change()

            features[f'vol_1_{window}'] = returns1.shift(1).rolling(window).std()
            features[f'vol_2_{window}'] = returns2.shift(1).rolling(window).std()

            features[f'momentum_1_{window}'] = returns1.shift(1).rolling(window).mean()
            features[f'momentum_2_{window}'] = returns2.shift(1).rolling(window).mean()

            features[f'corr_{window}'] = (
                returns1.shift(1)
                .rolling(window)
                .corr(returns2.shift(1))
            )

            cov = (
                returns1.shift(1)
                .rolling(window)
                .cov(returns2.shift(1))
            )
            var = returns2.shift(1).rolling(window).var()
            features[f'beta_{window}'] = cov / (var + 1e-8)

            coint_pvalues = []
            for i in range(len(price1)):
                if i < window:
                    coint_pvalues.append(np.nan)
                else:
                    try:
                        _, pvalue, _ = sm.tsa.stattools.coint(
                            price1.iloc[i - window:i - 1],
                            price2.iloc[i - window:i - 1]
                        )
                        coint_pvalues.append(pvalue)
                    except:
                        coint_pvalues.append(np.nan)

            features[f'coint_pvalue_{window}'] = coint_pvalues

        price_ratio = price1 / price2
        features['price_ratio'] = price_ratio.shift(1)
        features['log_ratio'] = np.log(price_ratio).shift(1)
        features['ratio_change'] = price_ratio.pct_change().shift(1)

        features = features.bfill().dropna()

        return features

    def create_spread_labels(self,
                             price1: pd.Series,
                             price2: pd.Series,
                             lookback_window: int = 20,
                             entry_zscore: float = 2.0,
                             exit_zscore: float = 0.5) -> pd.Series:
        """
        Create trading signals based on historical spread z-scores.
        Uses lookback window to calculate z-scores to avoid lookahead bias.
        """
        signals = pd.Series(0, index=price1.index)

        X = sm.add_constant(price2)
        model = sm.OLS(price1, X).fit()
        hedge_ratio = model.params[1]

        spread = price1 - hedge_ratio * price2

        for i in range(lookback_window, len(spread)):
            historical_spread = spread.iloc[i - lookback_window:i]
            current_spread = spread.iloc[i]

            mean = historical_spread.mean()
            std = historical_spread.std()

            if std != 0:
                zscore = (current_spread - mean) / std

                if zscore > entry_zscore:
                    signals.iloc[i] = -1
                elif zscore < -entry_zscore:
                    signals.iloc[i] = 1
                elif abs(zscore) < exit_zscore:
                    signals.iloc[i] = 0
                else:
                    signals.iloc[i] = signals.iloc[i - 1]

        return signals.fillna(0)

    def create_advanced_spread_labels(self,
                                      price1: pd.Series,
                                      price2: pd.Series,
                                      lookback_window: int = 20,
                                      zscore_threshold: float = 2.0,
                                      vol_lookback: int = 60) -> pd.Series:
        """
        Create more sophisticated spread trading signals considering:
        1. Z-score mean reversion
        2. Volatility adjustment
        3. Trend detection
        """
        signals = pd.Series(0, index=price1.index)

        hedge_ratios = []
        for i in range(lookback_window, len(price1)):
            X = sm.add_constant(price2.iloc[i - lookback_window:i])
            y = price1.iloc[i - lookback_window:i]
            try:
                model = sm.OLS(y, X).fit()
                hedge_ratios.append(model.params[1])
            except:
                hedge_ratios.append(np.nan if not hedge_ratios else hedge_ratios[-1])

        hedge_ratios = [hedge_ratios[0]] * lookback_window + hedge_ratios
        hedge_ratio_series = pd.Series(hedge_ratios, index=price1.index)

        spread = price1 - hedge_ratio_series * price2

        for i in range(lookback_window, len(spread)):
            historical_spread = spread.iloc[i - lookback_window:i]
            current_spread = spread.iloc[i]

            mean = historical_spread.mean()
            std = historical_spread.std()

            if std != 0:
                zscore = (current_spread - mean) / std

                if i >= vol_lookback:
                    recent_vol = spread.iloc[i - vol_lookback:i].std()
                    historical_vol = spread.iloc[i - 2 * vol_lookback:i - vol_lookback].std()
                    vol_ratio = recent_vol / historical_vol if historical_vol != 0 else 1
                    adjusted_threshold = zscore_threshold * vol_ratio
                else:
                    adjusted_threshold = zscore_threshold

                short_ma = spread.iloc[i - 10:i].mean()
                long_ma = spread.iloc[i - lookback_window:i].mean()
                trend = short_ma > long_ma

                if zscore > adjusted_threshold and not trend:
                    signals.iloc[i] = -1
                elif zscore < -adjusted_threshold and trend:
                    signals.iloc[i] = 1
                elif (abs(zscore) < 0.5) or (signals.iloc[i - 1] == 1 and not trend) or (
                        signals.iloc[i - 1] == -1 and trend):
                    signals.iloc[i] = 0
                else:
                    signals.iloc[i] = signals.iloc[i - 1]

        return signals.fillna(0)

    def train_model(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        model_name: str = "RandomForestClassifier",
        test_size: float = 0.2,
        cv: int = 5
    ) -> Tuple[Any, Dict]:
        """Train and evaluate model."""
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, shuffle=False
        )

        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        if model_name in self.classification_models:
            model = self.classification_models[model_name]
            scoring = 'accuracy'
        else:
            model = self.regression_models[model_name]
            scoring = 'neg_mean_squared_error'

        model.fit(X_train_scaled, y_train)
        cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=cv, scoring=scoring)

        y_pred = model.predict(X_test_scaled)

        if model_name in self.classification_models:
            metrics = {
                "cv_scores": cv_scores,
                "cv_mean": cv_scores.mean(),
                "cv_std": cv_scores.std(),
                "classification_report": classification_report(y_test, y_pred, output_dict=True),
                "confusion_matrix": confusion_matrix(y_test, y_pred).tolist()
            }
        else:
            metrics = {
                "cv_scores": -cv_scores,
                "cv_mean": -cv_scores.mean(),
                "cv_std": cv_scores.std(),
                "mse": mean_squared_error(y_test, y_pred),
                "r2": r2_score(y_test, y_pred)
            }

        return model, metrics

    def plot_feature_importance(self, model, feature_names: List[str], title: str = "Feature Importances") -> None:
        """Plot feature importances using Plotly."""
        if not hasattr(model, "feature_importances_"):
            logger.warning("Model does not have feature importances.")
            return

        importances = pd.Series(
            model.feature_importances_,
            index=feature_names
        ).sort_values(ascending=True)

        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=importances.values,
            y=importances.index,
            orientation='h'
        ))

        fig.update_layout(
            title=title,
            xaxis_title="Importance",
            yaxis_title="Features",
            height=max(400, len(feature_names) * 20),
            width=800,
            template="plotly_white"
        )

        save_dir = Path(MODEL_DIR) / "machine_learning_models"
        save_dir.mkdir(parents=True, exist_ok=True)
        fig.write_html(save_dir / 'ml_feature_importance.html')

def time_series_cross_validation(
    model: MachineLearningModel,
    features: pd.DataFrame,
    target: pd.Series,
    model_name: str = "RandomForestClassifier",
    n_splits: int = 5,
    test_size: float = 0.2,
    verbose: bool = True
) -> List[Dict[str, Union[float, Dict]]]:
    """
    Perform Time-Series Cross-Validation (TSCV) for the given model and data.

    Args:
        model: Instance of MachineLearningModel.
        features: Input features (DataFrame).
        target: Target values (Series).
        model_name: Name of the model to use from the model registry.
        n_splits: Number of splits for TSCV.
        test_size: Fraction of the data used for testing in the final split.
        verbose: Whether to print fold-specific results.

    Returns:
        List of dictionaries containing evaluation metrics for each fold.
    """
    tscv = TimeSeriesSplit(n_splits=n_splits)
    fold_metrics = []

    for fold, (train_idx, test_idx) in enumerate(tscv.split(features), start=1):
        X_train, X_test = features.iloc[train_idx], features.iloc[test_idx]
        y_train, y_test = target.iloc[train_idx], target.iloc[test_idx]

        if verbose:
            print(f"\n=== Fold {fold}/{n_splits} ===")
            print(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")

        trained_model, metrics = model.train_model(
            X=X_train,
            y=y_train,
            model_name=model_name,
            test_size=test_size,
            cv=5
        )

        y_pred = trained_model.predict(model.scaler.transform(X_test))

        if model_name in model.classification_models:
            accuracy = accuracy_score(y_test, y_pred)
            metrics["fold_accuracy"] = accuracy
            if verbose:
                print(f"Fold {fold} Accuracy: {accuracy:.4f}")
        else:
            mse = mean_squared_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            metrics["fold_mse"] = mse
            metrics["fold_r2"] = r2
            if verbose:
                print(f"Fold {fold} MSE: {mse:.4f}, R2: {r2:.4f}")

        fold_metrics.append(metrics)

    return fold_metrics

def main():
    """Example usage of MachineLearningModel with pairs trading."""
    stock1 = pd.read_csv(r"C:\Users\arnav\Downloads\pairs_trading_system\data\raw\AAPL.csv")
    stock2 = pd.read_csv(r'C:\Users\arnav\Downloads\pairs_trading_system\data\raw\MSFT.csv')

    stock1['Date'] = pd.to_datetime(stock1['Date'])
    stock2['Date'] = pd.to_datetime(stock2['Date'])
    stock1.set_index('Date', inplace=True)
    stock2.set_index('Date', inplace=True)

    ml_model = MachineLearningModel()

    features_df = ml_model.prepare_features(
        stock1['Close'],
        stock2['Close'],
        windows=[5, 20, 60]
    )

    signals = ml_model.create_advanced_spread_labels(
        stock1['Close'],
        stock2['Close'],
        lookback_window=20,
        zscore_threshold=2.0,
        vol_lookback=60
    )

    model, metrics = ml_model.train_model(
        X=features_df,
        y=signals,
        model_name="RandomForestClassifier",
        test_size=0.2,
        cv=5
    )

    print("\nModel Performance Metrics:")
    for metric, value in metrics.items():
        if isinstance(value, dict):
            print(f"\n{metric}:")
            for k, v in value.items():
                print(f"  {k}: {v}")
        else:
            print(f"{metric}: {value}")

    ml_model.plot_feature_importance(
        model,
        feature_names=features_df.columns,
        title="Feature Importance for Spread Trading Signals"
    )

    print("Performing Time-Series Cross-Validation...")
    tscv_results = time_series_cross_validation(
        model=ml_model,
        features=features_df,
        target=signals,
        model_name="RandomForestClassifier",
        n_splits=5,
        verbose=True
    )

    print("\nCross-Validation Results:")
    for i, metrics in enumerate(tscv_results, start=1):
        print(f"\n=== Fold {i} Metrics ===")
        for metric, value in metrics.items():
            if isinstance(value, dict):
                print(f"{metric}:")
                for sub_metric, sub_value in value.items():
                    print(f"  {sub_metric}: {sub_value}")
            else:
                print(f"{metric}: {value}")

    return model, metrics, features_df


if __name__ == "__main__":
    model, metrics, features = main()
