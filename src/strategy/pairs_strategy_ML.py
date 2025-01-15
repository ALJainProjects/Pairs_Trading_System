"""
Pairs Trading ML Implementation (Corrected for Look-Ahead Bias)

This module implements machine learning models_data for pairs trading with proper
temporal handling to prevent look-ahead bias in:
1. Feature engineering
2. Model training
3. Signal generation
4. Performance analysis
"""

import pandas as pd
from typing import Dict, List
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
import plotly.graph_objects as go
from src.models import MachineLearningModel
import numpy as np


class PairsTraderML:
    """Machine learning based pairs trading implementation."""

    def __init__(self,
                 lookback_windows: List[int] = (5, 10, 20, 50),
                 zscore_threshold: float = 2.0,
                 train_size: int = 252,
                 validation_size: int = 63):
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
                         min_window: int = 20) -> pd.DataFrame:
        """
        Create features using expanding windows to prevent look-ahead bias.
        Leverages MachineLearningModel's feature calculation methods.

        Args:
            stock1_prices: Price series for first stock
            stock2_prices: Price series for second stock
            start_idx: Starting index for feature calculation
            min_window: Minimum window size for calculations

        Returns:
            DataFrame with engineered features
        """
        if len(stock1_prices) != len(stock2_prices):
            raise ValueError("Price series must be same length")

        if start_idx < min_window:
            raise ValueError(f"start_idx must be >= min_window ({min_window})")

        features = pd.DataFrame(index=stock1_prices.index[start_idx:])

        returns1 = stock1_prices.pct_change()
        returns2 = stock2_prices.pct_change()
        features['return1'] = returns1[start_idx:]
        features['return2'] = returns2[start_idx:]

        for w in self.lookback_windows:
            if start_idx < w:
                continue

            spread_features = self.ml_model.calculate_spread_features(
                stock1_prices[start_idx - w:],
                stock2_prices[start_idx - w:],
                window=w
            )

            ratio_features = self.ml_model.calculate_pair_ratios(
                stock1_prices[start_idx - w:],
                stock2_prices[start_idx - w:],
                window=w
            )

            window_features = pd.concat([spread_features, ratio_features], axis=1)

            window_features = window_features.loc[:, ~window_features.columns.duplicated()]

            for col in window_features.columns:
                features[f'{col}_{w}' if not col.endswith(f'_{w}') else col] = window_features[col]

        features = features.dropna()
        return features

    def generate_labels(self,
                        features: pd.DataFrame,
                        window: int,
                        forward_returns: bool = True) -> pd.DataFrame:
        """Generate trading signals and labels for training."""
        data = features.copy()
        zscore_col = f'spread_zscore_{window}'

        if zscore_col not in data.columns:
            raise ValueError(f"Z-score column {zscore_col} not found")

        data['signal'] = 0
        data.loc[data[zscore_col] > self.zscore_threshold, 'signal'] = -1
        data.loc[data[zscore_col] < -self.zscore_threshold, 'signal'] = 1

        data['target'] = (data['signal'] != 0).astype(int)

        if forward_returns:
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
            gap=window
        )

        for train_idx, val_idx in tscv.split(features):
            train_data = features.iloc[train_idx]
            val_data = features.iloc[val_idx]

            train_data = self.generate_labels(
                train_data,
                window,
                forward_returns=True
            )

            feature_cols = [col for col in train_data.columns
                            if any(x in col for x in ['spread_', 'ratio_', 'return'])]

            X_train = train_data[feature_cols]
            y_class_train = train_data['target']
            y_reg_train = train_data['forward_return']

            models = self.train_period_models(
                X_train, y_class_train, y_reg_train
            )

            predictions = self.predict_signals(
                models,
                val_data[feature_cols]
            )

            results.append({
                'train_period': (features.index[train_idx[0]],
                                 features.index[train_idx[-1]]),
                'val_period': (features.index[val_idx[0]],
                               features.index[val_idx[-1]]),
                'models_data': models,
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
        Train models_data for a specific period.

        Args:
            X_train: Training feature_engineering
            y_class: Classification targets
            y_reg: Regression targets

        Returns:
            Dictionary of trained models_data and metrics
        """

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)

        clf_model, clf_score = self.ml_model.train_classification_model(
            X_train_scaled,
            y_class,
            model_name='RandomForestClassifier',
            cv=5
        )

        reg_model, reg_score = self.ml_model.train_regression_model(
            X_train_scaled,
            y_reg,
            model_name='RandomForestRegressor',
            cv=5
        )

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
            models: Dictionary of trained models_data
            features: New feature data

        Returns:
            DataFrame with predictions
        """
        predictions = pd.DataFrame(index=features.index)

        X_scaled = models['classification']['scaler'].transform(features)

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

        fig.add_trace(go.Scatter(
            x=actual_spread.index,
            y=actual_spread,
            mode='lines',
            name='Actual Spread'
        ))

        colors = ['green', 'red']
        for period_result in results:
            val_predictions = period_result['predictions']
            val_start, val_end = period_result['val_period']

            for signal_val, color in zip([1, 0], colors):
                mask = np.array(val_predictions['predicted_signal'] == signal_val)
                if mask.any():
                    fig.add_trace(go.Scatter(
                        x=val_predictions.index[mask],
                        y=actual_spread[val_predictions.index[mask]],
                        mode='markers',
                        marker=dict(color=color, size=10),
                        name=f'{"Long" if signal_val else "Short"} Signal'
                    ))

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
    """
    Example usage of ML-based pairs trading with proper temporal handling.
    Downloads real market data and performs comprehensive analysis.
    """
    import yfinance as yf
    from datetime import datetime, timedelta
    import logging
    import numpy as np

    try:
        logging.basicConfig(level=logging.INFO)
        logger = logging.getLogger(__name__)

        logger.info("Downloading historical data...")
        end_date = datetime.now()
        start_date = end_date - timedelta(days=365 * 3)

        stock1_symbol = "GOOGL"
        stock2_symbol = "META"

        stock1_data = yf.download(stock1_symbol, start=start_date, end=end_date)
        stock2_data = yf.download(stock2_symbol, start=start_date, end=end_date)

        stock1_prices = stock1_data['Close']
        stock2_prices = stock2_data['Close']

        logger.info(f"Downloaded {len(stock1_prices)} days of data")

        trader = PairsTraderML(
            lookback_windows=[5, 10, 20, 50],
            zscore_threshold=2.0,
            train_size=252,
            validation_size=63
        )

        logger.info("Preparing feature_engineering...")
        features = trader.prepare_features(
            stock1_prices,
            stock2_prices,
            start_idx=50
        )

        logger.info("Starting walk-forward optimization...")
        results = trader.walk_forward_optimization(features)

        logger.info("\nAnalyzing results...")

        all_predictions = pd.DataFrame()
        all_importances = {
            'classification': pd.DataFrame(),
            'regression': pd.DataFrame()
        }

        for period_idx, period_result in enumerate(results, 1):
            val_start, val_end = period_result['val_period']
            predictions = period_result['predictions']
            all_predictions = pd.concat([all_predictions, predictions])

            for model_type in ['classification', 'regression']:
                importance_df = pd.DataFrame({
                    'feature': period_result['feature_importance'][model_type].index,
                    'importance': period_result['feature_importance'][model_type].values,
                    'period': period_idx
                })
                all_importances[model_type] = pd.concat([
                    all_importances[model_type],
                    importance_df
                ])

            print(f"\nPeriod {period_idx}: {val_start.date()} to {val_end.date()}")

            actual_spread = stock1_prices - stock2_prices
            period_returns = predictions['predicted_return']
            period_accuracy = np.mean(predictions['predicted_signal'])

            print(f"Average Signal Probability: {predictions['signal_probability'].mean():.4f}")
            print(f"Signal Rate: {period_accuracy:.4f}")
            print(f"Average Predicted Return: {period_returns.mean():.4f}")

        for model_type in ['classification', 'regression']:
            avg_importance = all_importances[model_type].groupby('feature')['importance'].mean()
            avg_importance = avg_importance.sort_values(ascending=False)

            print(f"\nTop 5 Important Features ({model_type}):")
            print(avg_importance.head().to_string())

        logger.info("\nPlotting results...")
        trader.plot_walk_forward_results(
            results,
            stock1_prices - stock2_prices
        )

        output_data = {
            'feature_engineering': features,
            'results': results,
            'all_predictions': all_predictions,
            'feature_importance': all_importances,
            'pair_info': {
                'stock1': stock1_symbol,
                'stock2': stock2_symbol,
                'start_date': start_date,
                'end_date': end_date
            }
        }

        return output_data

    except Exception as e:
        import traceback
        logging.basicConfig(level=logging.INFO)
        logger = logging.getLogger(__name__)
        logger.error(traceback.format_exc())
        logger.error(f"Error in main execution: {str(e)}")
        return None


if __name__ == "__main__":
    output = main()
    if output is not None:
        print("\nExecution completed successfully!")
    else:
        print("\nExecution failed. Check logs for details.")