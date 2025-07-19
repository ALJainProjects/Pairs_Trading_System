import pandas as pd
import numpy as np
from typing import Dict, Tuple, Any, List, Type
from sklearn.model_selection import TimeSeriesSplit
import optuna
from tqdm import tqdm
import plotly.graph_objects as go

from config.logging_config import logger
from src.strategy.backtest import Backtester
from src.strategy.base import BaseStrategy


class BaseStrategyEvaluator:
    """Base class containing common evaluation methods."""

    @staticmethod
    def _calculate_sharpe_ratio(returns: pd.Series) -> float:
        if returns.empty or len(returns) < 2 or returns.std() == 0:
            return 0.0  # Return 0.0 if not enough data or no volatility
        return (returns.mean() / returns.std()) * np.sqrt(252)

    @staticmethod
    def _calculate_max_drawdown(equity_curve: pd.Series) -> float:
        if equity_curve.empty or equity_curve.max() <= 0:  # Handle empty or non-positive equity curves
            return 0.0
        peaks = equity_curve.expanding().max()
        drawdowns = (equity_curve - peaks) / peaks
        return abs(drawdowns.min())


class OptunaOptimizer:
    """
    Wrapper for Optuna-based Bayesian optimization of strategy parameters.
    The objective function is designed to run a backtest on a specified data segment
    (e.g., a training or validation fold) and optimize for a given metric.
    """

    def __init__(self, strategy_class: Type[BaseStrategy], param_space: Dict):
        self.strategy_class = strategy_class
        self.param_space = param_space
        # Data will be passed to _objective to ensure it's always the correct segment
        self.data_for_objective: Optional[pd.DataFrame] = None

    def _objective(self, trial: optuna.Trial) -> float:
        if self.data_for_objective is None:
            raise RuntimeError("Optimization data not set. Call run_optimization with data.")

        params = {}
        for name, space in self.param_space.items():
            if space['type'] == 'int':
                params[name] = trial.suggest_int(name, space['low'], space['high'])
            elif space['type'] == 'float':
                params[name] = trial.suggest_float(name, space['low'], space['high'])
            elif space['type'] == 'categorical':  # Added categorical type
                params[name] = trial.suggest_categorical(name, space['choices'])
            # Add other types as needed (e.g., 'log_float')

        try:
            strategy = self.strategy_class(**params)
            # The backtester here runs only on the data segment passed for this objective (train/val fold)
            backtester = Backtester(strategy, self.data_for_objective)
            strategy.fit(self.data_for_objective)  # Fit strategy on the specific data fold
            results = backtester.run()

            # Default to -1.0 for invalid Sharpe Ratios
            sharpe = results['metrics'].get('Sharpe Ratio', -1.0)
            return sharpe if pd.notna(sharpe) and np.isfinite(sharpe) else -1.0
        except Exception as e:
            logger.warning(f"Backtest failed for params {params}: {e}")
            return -1.0  # Penalize failed backtests

    def run_optimization(self, data_segment: pd.DataFrame, n_trials: int = 100) -> Dict:
        """
        Runs the optimization study on a specific data segment.

        Args:
            data_segment (pd.DataFrame): The DataFrame on which to run backtests for optimization.
                                         This is typically a training or validation fold.
            n_trials (int): Number of optimization trials.

        Returns:
            Dict: Best parameters found by Optuna.
        """
        self.data_for_objective = data_segment  # Set the data for the objective function
        study = optuna.create_study(direction='maximize')
        study.optimize(self._objective, n_trials=n_trials, show_progress_bar=True)

        logger.info(f"Best trial Sharpe Ratio: {study.best_trial.value:.4f}")
        logger.info(f"Best params: {study.best_params}")
        self.data_for_objective = None  # Clear data after optimization
        return study.best_params


class WalkForwardOptimizer(BaseStrategyEvaluator):
    """
    Implements Walk-Forward Optimization (WFO), a robust method for time-series backtesting.
    Parameters are optimized on a 'training' fold and then evaluated on an unseen 'test' fold.
    """

    def __init__(self, data: pd.DataFrame, train_size_ratio: float, test_size_ratio: float, step_size_ratio: float):
        """
        Initializes the WalkForwardOptimizer.

        Args:
            data (pd.DataFrame): The full historical DataFrame for the walk-forward analysis.
            train_size_ratio (float): The proportion of data to use for the training window (e.g., 0.6).
            test_size_ratio (float): The proportion of data to use for the test window (e.g., 0.2).
            step_size_ratio (float): The proportion of data to advance for the next fold (e.g., 0.1).
        """
        if not (0 < train_size_ratio < 1 and 0 < test_size_ratio < 1 and 0 < step_size_ratio < 1):
            raise ValueError("All size ratios must be between 0 and 1.")
        if train_size_ratio + test_size_ratio > 1:
            logger.warning(
                "Train and test size ratios sum to more than 1. This might lead to overlapping or very short effective periods.")

        self.data = data.sort_index()  # Ensure data is sorted by index
        self.train_size_ratio = train_size_ratio
        self.test_size_ratio = test_size_ratio
        self.step_size_ratio = step_size_ratio
        self.results: List[Dict] = []
        logger.info(f"WalkForwardOptimizer initialized with train_size_ratio={train_size_ratio}, "
                    f"test_size_ratio={test_size_ratio}, step_size_ratio={step_size_ratio}.")

    def _generate_windows(self) -> List[Tuple[pd.Index, pd.Index]]:
        """Generates walk-forward windows for training and testing based on ratios."""
        windows = []
        total_len = len(self.data)

        # Convert ratios to absolute lengths
        train_len = int(total_len * self.train_size_ratio)
        test_len = int(total_len * self.test_size_ratio)
        step_len = int(total_len * self.step_size_ratio)

        if train_len < 10 or test_len < 10:  # Minimum reasonable length for any financial analysis
            logger.error(
                f"Calculated train_len ({train_len}) or test_len ({test_len}) is too small. Increase data or ratios.")
            return []

        start_idx = 0
        while start_idx + train_len + test_len <= total_len:
            train_end_idx = start_idx + train_len
            test_end_idx = train_end_idx + test_len

            windows.append((
                self.data.index[start_idx:train_end_idx],  # Train index
                self.data.index[train_end_idx:test_end_idx]  # Test index
            ))
            start_idx += step_len

        if not windows:
            logger.warning("No valid walk-forward windows generated. Check data length and window sizes/ratios.")
        else:
            logger.info(
                f"Generated {len(windows)} walk-forward windows. First window: Train {windows[0][0][0]} to {windows[0][0][-1]}, Test {windows[0][1][0]} to {windows[0][1][-1]}.")
            logger.info(
                f"Last window: Train {windows[-1][0][0]} to {windows[-1][0][-1]}, Test {windows[-1][1][0]} to {windows[-1][1][-1]}.")
        return windows

    def run(self, strategy_class: Type[BaseStrategy], param_space: Dict, n_trials_per_fold: int = 50) -> Dict:
        """
        Runs the full walk-forward optimization process.
        For each fold:
        1. Optimizes strategy parameters on the training data segment using Optuna.
        2. Evaluates the best parameters on the unseen test data segment.

        Args:
            strategy_class (Type[BaseStrategy]): The strategy class to optimize.
            param_space (Dict): Dictionary defining the parameter search space for Optuna.
            n_trials_per_fold (int): Number of Optuna trials to run for each fold's optimization.

        Returns:
            Dict: Aggregated results including fold-specific metrics and parameter/performance stability.
        """
        windows = self._generate_windows()
        if not windows:
            logger.error("Walk-forward optimization aborted due to no valid windows.")
            return {}

        self.results = []

        # Initialize OptunaOptimizer once outside the loop for efficiency
        # It will receive the specific data_segment in its run_optimization method
        optimizer = OptunaOptimizer(strategy_class, param_space)

        for i, (train_idx, test_idx) in enumerate(tqdm(windows, desc="Walk-Forward Folds")):
            logger.info(f"Running WFO Fold {i + 1}/{len(windows)}...")

            train_data = self.data.loc[train_idx]
            test_data = self.data.loc[test_idx]

            if train_data.empty or test_data.empty:
                logger.warning(f"Fold {i + 1}: Train or test data is empty. Skipping fold.")
                continue

            # 1. Optimize parameters on the training data for this fold
            logger.info(
                f"Fold {i + 1}: Optimizing parameters on training data from {train_idx[0]} to {train_idx[-1]} (len: {len(train_data)}).")
            best_params = optimizer.run_optimization(train_data, n_trials=n_trials_per_fold)

            # 2. Evaluate the best parameters on the unseen test data
            logger.info(
                f"Fold {i + 1}: Evaluating best parameters on test data from {test_idx[0]} to {test_idx[-1]} (len: {len(test_data)}).")
            try:
                strategy = strategy_class(**best_params)
                # The backtester runs on the test data for this fold
                backtester = Backtester(strategy, test_data)
                # The strategy's fit method should be called on the test data too,
                # if it uses `fit` to initialize state necessary for `generate_signals`
                # on the test data. Some strategies might expect a full historical context.
                # For WFO, the test data is treated as the 'live' period, so fit on it.
                strategy.fit(test_data)  # Fit on test data (as if it's the live historical data for that period)
                test_results = backtester.run()

                self.results.append({
                    'fold': i + 1,
                    'train_start': train_idx[0],
                    'train_end': train_idx[-1],
                    'test_start': test_idx[0],
                    'test_end': test_idx[-1],
                    'best_params': best_params,
                    'test_metrics': test_results['metrics'],
                    'equity_curve': test_results['equity_curve'],  # Store equity curve for plotting
                    'trade_log': test_results.get('trades', [])  # Store trade log for detailed analysis
                })
                logger.info(
                    f"Fold {i + 1} completed. Test Sharpe: {test_results['metrics'].get('Sharpe Ratio', np.nan):.4f}")
            except Exception as e:
                logger.error(
                    f"Error evaluating strategy with best params {best_params} on test fold {i + 1}: {e}. Skipping fold results.")
                # Optionally append partial results or mark as failed
                self.results.append({
                    'fold': i + 1,
                    'train_start': train_idx[0],
                    'train_end': train_idx[-1],
                    'test_start': test_idx[0],
                    'test_end': test_idx[-1],
                    'best_params': best_params,
                    'test_metrics': {'Sharpe Ratio': np.nan, 'Max Drawdown': np.nan, 'Total Return': np.nan},
                    'equity_curve': pd.Series(dtype=float),
                    'trade_log': []
                })

        logger.info("Walk-forward optimization complete.")
        return self.aggregate_results()

    def aggregate_results(self) -> Dict:
        """Aggregates and analyzes the results from all walk-forward folds."""
        if not self.results:
            logger.warning("No results to aggregate from walk-forward optimization.")
            return {}

        all_params = [res['best_params'] for res in self.results if 'best_params' in res]
        all_metrics = [res['test_metrics'] for res in self.results if 'test_metrics' in res]

        param_stability = {}
        if all_params:
            # Ensure all params dictionaries have the same keys for consistent aggregation
            param_keys = all_params[0].keys() if all_params else []
            for key in param_keys:
                # Filter out values for keys that might not exist in all dicts (e.g., if optimization failed partially)
                values = [p.get(key) for p in all_params if key in p]
                # Filter out non-numeric values for aggregation
                numeric_values = [v for v in values if pd.notna(v) and np.isfinite(v)]
                if numeric_values:
                    param_stability[key] = {'mean': np.mean(numeric_values), 'std': np.std(numeric_values)}
                else:
                    param_stability[key] = {'mean': np.nan, 'std': np.nan}

        performance_stability = {}
        if all_metrics:
            metrics_keys = all_metrics[0].keys() if all_metrics else []
            for key in metrics_keys:
                values = [m.get(key) for m in all_metrics if key in m]
                numeric_values = [v for v in values if pd.notna(v) and np.isfinite(v)]
                if numeric_values:
                    performance_stability[key] = {'mean': np.mean(numeric_values), 'std': np.std(numeric_values)}
                else:
                    performance_stability[key] = {'mean': np.nan, 'std': np.nan}

        # Combine equity curves from all test folds
        combined_equity_curve = pd.Series(dtype=float)
        for res in self.results:
            if 'equity_curve' in res and not res['equity_curve'].empty:
                # Ensure equity curves are absolute values or normalized relative to start of *their* test fold
                # For aggregation, it's common to re-normalize each fold's equity to start at 1.0
                # or combine daily returns. Combining absolute equity curves directly is tricky.
                # Let's combine daily returns for a more meaningful combined performance.
                fold_equity = res['equity_curve']
                if len(fold_equity) > 1:
                    fold_returns = fold_equity.pct_change().dropna()
                    combined_equity_curve = pd.concat([combined_equity_curve, fold_returns])
                elif len(fold_equity) == 1:  # Handle single data point, cannot pct_change
                    pass  # Or decide how to represent single point returns

        # If we aggregated returns, convert back to equity curve starting at 1.0
        if not combined_equity_curve.empty:
            # Sort by index (date) and drop duplicates if any
            combined_equity_curve = combined_equity_curve.sort_index().drop_duplicates()
            if not combined_equity_curve.empty:
                # Compound returns to get equity curve. Start at 1.0 for easier comparison.
                combined_equity_curve = (1 + combined_equity_curve).cumprod()
                # Ensure it starts at 1.0 if the first value is not 1.0 after cumprod
                if combined_equity_curve.iloc[0] != 1.0:
                    combined_equity_curve = combined_equity_curve / combined_equity_curve.iloc[0]
                combined_equity_curve.iloc[0] = 1.0  # Ensure first point is 1.0 (or appropriate start)

        # Recalculate overall metrics from combined equity curve
        overall_metrics = {}
        if not combined_equity_curve.empty:
            overall_returns = combined_equity_curve.pct_change().dropna()
            overall_metrics['Total Return'] = (combined_equity_curve.iloc[-1] / combined_equity_curve.iloc[0]) - 1.0 if \
            combined_equity_curve.iloc[0] != 0 else np.nan
            overall_metrics['Sharpe Ratio'] = self._calculate_sharpe_ratio(overall_returns)
            overall_metrics['Max Drawdown'] = self._calculate_max_drawdown(combined_equity_curve)
            logger.info(f"Aggregated WFO Metrics: Total Return={overall_metrics['Total Return']:.4f}, "
                        f"Sharpe Ratio={overall_metrics['Sharpe Ratio']:.4f}, Max Drawdown={overall_metrics['Max Drawdown']:.4f}")
        else:
            logger.warning("Combined equity curve is empty. Overall metrics not calculated.")
            overall_metrics = {'Total Return': np.nan, 'Sharpe Ratio': np.nan, 'Max Drawdown': np.nan}

        return {
            'fold_results': self.results,
            'parameter_stability': param_stability,
            'performance_stability': performance_stability,
            'combined_equity_curve': combined_equity_curve,
            'overall_metrics': overall_metrics
        }

    def plot_wfo_equity_curves(self):
        """Plots individual fold equity curves and the combined equity curve."""
        if not self.results:
            logger.warning("No results available to plot equity curves.")
            return

        fig = go.Figure()

        # Plot individual fold equity curves
        for i, res in enumerate(self.results):
            if 'equity_curve' in res and not res['equity_curve'].empty:
                # Normalize each fold's equity curve to start at 1.0 for better visual comparison
                normalized_fold_equity = res['equity_curve'] / res['equity_curve'].iloc[0]
                fig.add_trace(go.Scatter(x=normalized_fold_equity.index, y=normalized_fold_equity,
                                         mode='lines', name=f'Fold {i + 1} Equity (Normalized)',
                                         opacity=0.7))

        # Plot combined equity curve if available
        aggregated_results = self.aggregate_results()
        if 'combined_equity_curve' in aggregated_results and not aggregated_results['combined_equity_curve'].empty:
            fig.add_trace(go.Scatter(x=aggregated_results['combined_equity_curve'].index,
                                     y=aggregated_results['combined_equity_curve'],
                                     mode='lines', name='Combined WFO Equity',
                                     line=dict(color='black', width=3)))

        fig.update_layout(title='Walk-Forward Optimization: Individual and Combined Equity Curves',
                          xaxis_title='Date', yaxis_title='Equity (Normalized)')
        fig.show()


class ParameterSensitivityAnalyzer(BaseStrategyEvaluator):
    """
    Analyzes how sensitive the strategy's performance is to changes in its parameters.
    """

    def __init__(self, strategy_class: Type[BaseStrategy], historical_data: pd.DataFrame, base_params: Dict):
        self.strategy_class = strategy_class
        self.historical_data = historical_data  # Full data for backtest
        self.base_params = base_params
        self.results: Dict[str, pd.DataFrame] = {}
        logger.info(
            f"ParameterSensitivityAnalyzer initialized with strategy {strategy_class.__name__} and base params: {base_params}")

    def run(self, param_to_test: str, test_range: np.ndarray) -> pd.DataFrame:
        """
        Runs sensitivity analysis for a single parameter over a given range of values.

        Args:
            param_to_test (str): The name of the parameter to vary.
            test_range (np.ndarray): An array of values to test for the parameter.

        Returns:
            pd.DataFrame: A DataFrame containing performance metrics for each tested parameter value.
        """
        performance_data = []
        for value in tqdm(test_range, desc=f"Analyzing {param_to_test}"):
            current_params = self.base_params.copy()
            current_params[param_to_test] = value
            try:
                strategy = self.strategy_class(**current_params)
                # For sensitivity analysis, we typically run on the full historical data
                # to see overall impact of parameter change.
                backtester = Backtester(strategy, self.historical_data)
                strategy.fit(self.historical_data)  # Fit on full data
                results = backtester.run()
                metrics = results['metrics']
                metrics['param_value'] = value
                performance_data.append(metrics)
                logger.debug(
                    f"Sensitivity for {param_to_test}={value}: Sharpe Ratio={metrics.get('Sharpe Ratio', np.nan):.4f}")
            except Exception as e:
                logger.error(f"Backtest failed for {param_to_test}={value}: {e}. Skipping this value.")
                # Append a row with NaN metrics for failed runs to keep results_df consistent
                failed_metrics = {k: np.nan for k in results['metrics'].keys()} if 'metrics' in locals() else {}
                failed_metrics['param_value'] = value
                performance_data.append(failed_metrics)
                continue
        results_df = pd.DataFrame(performance_data)
        self.results[param_to_test] = results_df
        logger.info(f"Sensitivity analysis for {param_to_test} complete.")
        return results_df

    def plot_sensitivity(self, param_name: str, metric_name: str = 'Sharpe Ratio'):
        """
        Plots the performance sensitivity for a given parameter and metric.

        Args:
            param_name (str): The name of the parameter for which to plot sensitivity.
            metric_name (str): The performance metric to plot against the parameter values.
        """
        if param_name not in self.results:
            raise ValueError(f"Analysis for '{param_name}' has not been run yet.")
        results_df = self.results[param_name]

        if metric_name not in results_df.columns:
            logger.warning(
                f"Metric '{metric_name}' not found in results for '{param_name}'. Available metrics: {results_df.columns.tolist()}")
            # Attempt to use Sharpe Ratio if requested metric is not found but Sharpe is.
            if 'Sharpe Ratio' in results_df.columns:
                metric_name = 'Sharpe Ratio'
                logger.warning(f"Defaulting to 'Sharpe Ratio' for plotting.")
            else:
                raise ValueError(f"Cannot plot: '{metric_name}' is not available and no default fallback found.")

        fig = go.Figure()
        fig.add_trace(
            go.Scatter(x=results_df['param_value'], y=results_df[metric_name], mode='lines+markers', name=metric_name))
        fig.update_layout(title=f"Sensitivity of {metric_name} to {param_name}",
                          xaxis_title=param_name, yaxis_title=metric_name,
                          hovermode="x unified")  # Improved hover experience
        fig.show()


class CrossValidatedOptimizer(BaseStrategyEvaluator):
    """
    Performs optimization using time-series cross-validation.
    """

    def __init__(self, data: pd.DataFrame, n_splits: int = 5):
        self.data = data.sort_index()  # Ensure data is sorted
        self.n_splits = n_splits
        self.results: List[Dict] = []
        logger.info(f"CrossValidatedOptimizer initialized with {n_splits} splits.")

    def run(self, strategy_class: Type[BaseStrategy], param_space: Dict, n_trials_per_fold: int = 20) -> Dict:
        """
        Runs the full cross-validation optimization process.
        For each split:
        1. Optimizes strategy parameters on the training data.
        2. Evaluates the best parameters on the validation data.

        Args:
            strategy_class (Type[BaseStrategy]): The strategy class to optimize.
            param_space (Dict): Dictionary defining the parameter search space for Optuna.
            n_trials_per_fold (int): Number of Optuna trials to run for each fold's optimization.

        Returns:
            Dict: Aggregated results including fold-specific metrics and overall performance stability.
        """
        tscv = TimeSeriesSplit(n_splits=self.n_splits)
        self.results = []

        # Initialize OptunaOptimizer once outside the loop
        optimizer = OptunaOptimizer(strategy_class, param_space)

        # total_splits_iter = tscv.split(self.data) # Iterator
        # total_splits_count = sum(1 for _ in tscv.split(self.data)) # To get total count for tqdm

        for i, (train_idx, val_idx) in enumerate(
                tqdm(tscv.split(self.data), desc="Cross-Validation Folds", total=self.n_splits)):
            logger.info(f"Running CV Fold {i + 1}/{self.n_splits}...")

            train_data = self.data.iloc[train_idx]
            val_data = self.data.iloc[val_idx]  # This is the test/validation data for this fold

            if train_data.empty or val_data.empty:
                logger.warning(f"Fold {i + 1}: Train or validation data is empty. Skipping fold.")
                continue

            # 1. Optimize parameters on the training data for this fold
            logger.info(
                f"Fold {i + 1}: Optimizing parameters on training data from {train_idx[0]} to {train_idx[-1]} (len: {len(train_data)}).")
            best_params = optimizer.run_optimization(train_data, n_trials=n_trials_per_fold)

            # 2. Evaluate the best parameters on the validation data
            logger.info(
                f"Fold {i + 1}: Evaluating best parameters on validation data from {val_idx[0]} to {val_idx[-1]} (len: {len(val_data)}).")
            try:
                strategy = strategy_class(**best_params)
                # The backtester runs on the validation data for this fold
                backtester = Backtester(strategy, val_data)
                strategy.fit(val_data)  # Fit strategy on validation data (as if it's the historical context)
                val_results = backtester.run()

                self.results.append({
                    'fold': i + 1,
                    'train_start': train_idx[0],
                    'train_end': train_idx[-1],
                    'validation_start': val_idx[0],
                    'validation_end': val_idx[-1],
                    'best_params': best_params,
                    'validation_metrics': val_results['metrics'],
                    'equity_curve': val_results['equity_curve']  # Store equity curve for plotting
                })
                logger.info(
                    f"Fold {i + 1} completed. Validation Sharpe: {val_results['metrics'].get('Sharpe Ratio', np.nan):.4f}")
            except Exception as e:
                logger.error(
                    f"Error evaluating strategy with best params {best_params} on validation fold {i + 1}: {e}. Skipping fold results.")
                self.results.append({
                    'fold': i + 1,
                    'train_start': train_idx[0],
                    'train_end': train_idx[-1],
                    'validation_start': val_idx[0],
                    'validation_end': val_idx[-1],
                    'best_params': best_params,
                    'validation_metrics': {'Sharpe Ratio': np.nan, 'Max Drawdown': np.nan, 'Total Return': np.nan},
                    'equity_curve': pd.Series(dtype=float)
                })

        logger.info("Cross-validation optimization complete.")
        return self.aggregate_results()

    def aggregate_results(self) -> Dict:
        """Aggregates and analyzes the results from all cross-validation folds."""
        if not self.results:
            logger.warning("No results to aggregate from cross-validation optimization.")
            return {}

        all_metrics = [res['validation_metrics'] for res in self.results if 'validation_metrics' in res]

        performance_stability = {}
        if all_metrics:
            metrics_keys = all_metrics[0].keys() if all_metrics else []
            for key in metrics_keys:
                values = [m.get(key) for m in all_metrics if key in m]
                numeric_values = [v for v in values if pd.notna(v) and np.isfinite(v)]
                if numeric_values:
                    performance_stability[key] = {'mean': np.mean(numeric_values), 'std': np.std(numeric_values)}
                else:
                    performance_stability[key] = {'mean': np.nan, 'std': np.nan}

        # Combine equity curves from all validation folds (similar to WFO)
        combined_equity_curve = pd.Series(dtype=float)
        for res in self.results:
            if 'equity_curve' in res and not res['equity_curve'].empty:
                fold_equity = res['equity_curve']
                if len(fold_equity) > 1:
                    fold_returns = fold_equity.pct_change().dropna()
                    combined_equity_curve = pd.concat([combined_equity_curve, fold_returns])
                elif len(fold_equity) == 1:
                    pass

        overall_metrics = {}
        if not combined_equity_curve.empty:
            combined_equity_curve = combined_equity_curve.sort_index().drop_duplicates()
            if not combined_equity_curve.empty:
                combined_equity_curve = (1 + combined_equity_curve).cumprod()
                if combined_equity_curve.iloc[0] != 1.0:
                    combined_equity_curve = combined_equity_curve / combined_equity_curve.iloc[0]
                combined_equity_curve.iloc[0] = 1.0

            overall_returns = combined_equity_curve.pct_change().dropna()
            overall_metrics['Total Return'] = (combined_equity_curve.iloc[-1] / combined_equity_curve.iloc[0]) - 1.0 if \
            combined_equity_curve.iloc[0] != 0 else np.nan
            overall_metrics['Sharpe Ratio'] = self._calculate_sharpe_ratio(overall_returns)
            overall_metrics['Max Drawdown'] = self._calculate_max_drawdown(combined_equity_curve)
            logger.info(f"Aggregated CV Metrics: Total Return={overall_metrics['Total Return']:.4f}, "
                        f"Sharpe Ratio={overall_metrics['Sharpe Ratio']:.4f}, Max Drawdown={overall_metrics['Max Drawdown']:.4f}")
        else:
            logger.warning("Combined equity curve for CV is empty. Overall metrics not calculated.")
            overall_metrics = {'Total Return': np.nan, 'Sharpe Ratio': np.nan, 'Max Drawdown': np.nan}

        return {
            'fold_results': self.results,
            'performance_stability': performance_stability,
            'combined_equity_curve': combined_equity_curve,
            'overall_metrics': overall_metrics
        }

    def plot_cv_equity_curves(self):
        """Plots individual fold equity curves and the combined equity curve for Cross-Validation."""
        if not self.results:
            logger.warning("No results available to plot equity curves for CV.")
            return

        fig = go.Figure()

        # Plot individual fold equity curves
        for i, res in enumerate(self.results):
            if 'equity_curve' in res and not res['equity_curve'].empty:
                normalized_fold_equity = res['equity_curve'] / res['equity_curve'].iloc[0]
                fig.add_trace(go.Scatter(x=normalized_fold_equity.index, y=normalized_fold_equity,
                                         mode='lines', name=f'Fold {i + 1} Equity (Normalized)',
                                         opacity=0.7))

        # Plot combined equity curve if available
        aggregated_results = self.aggregate_results()
        if 'combined_equity_curve' in aggregated_results and not aggregated_results['combined_equity_curve'].empty:
            fig.add_trace(go.Scatter(x=aggregated_results['combined_equity_curve'].index,
                                     y=aggregated_results['combined_equity_curve'],
                                     mode='lines', name='Combined CV Equity',
                                     line=dict(color='black', width=3)))

        fig.update_layout(title='Cross-Validation Optimization: Individual and Combined Equity Curves',
                          xaxis_title='Date', yaxis_title='Equity (Normalized)')
        fig.show()


class MarketImpactModel:
    """
    Models the price impact of trades based on volatility and volume.
    """

    def __init__(self, historical_data: pd.DataFrame, impact_coefficient: float = 0.1):
        """
        Initializes the MarketImpactModel.

        Args:
            historical_data (pd.DataFrame): DataFrame containing 'Date', 'Symbol', 'Adj_Close', 'Volume'.
            impact_coefficient (float): A tuning parameter for the magnitude of market impact.
        """
        self.historical_data = historical_data.set_index(
            ['Date', 'Symbol']).sort_index()  # Multi-index for faster lookup
        self.impact_coefficient = impact_coefficient

        # Ensure prices and volumes are also multi-indexed for efficient rolling calculations
        prices_pivot = self.historical_data['Adj_Close'].unstack(level='Symbol')
        volume_pivot = self.historical_data['Volume'].unstack(level='Symbol')

        # Calculate Average Daily Volume (ADV) and annualized volatility
        # Use min_periods to handle start of data.
        self.adv = volume_pivot.rolling(window=21, min_periods=10).mean()
        self.volatility = prices_pivot.pct_change().rolling(window=21, min_periods=10).std() * np.sqrt(252)

        logger.info(f"MarketImpactModel initialized with impact_coefficient={impact_coefficient}.")

    def calculate_market_impact(self, date: pd.Timestamp, symbol: str, trade_value: float) -> float:
        """
        Calculates the estimated cost of a trade due to market impact using a simple model.
        Model: Impact = coefficient * Volatility * (Trade Value / ADV Value)

        Args:
            date (pd.Timestamp): The date of the trade.
            symbol (str): The symbol of the traded asset.
            trade_value (float): The absolute dollar value of the trade (quantity * price).

        Returns:
            float: The estimated market impact cost (in currency units).
        """
        try:
            # Look up ADV, Volatility, and Price for the specific date and symbol
            # Use .loc for multi-index lookup
            adv_at_date = self.adv.loc[date, symbol]
            vol_at_date = self.volatility.loc[date, symbol]
            price_at_date = self.historical_data.loc[(date, symbol), 'Adj_Close']

            # Handle NaN values for ADV, Volatility, or Price (e.g., at beginning of data)
            if pd.isna(adv_at_date) or pd.isna(vol_at_date) or pd.isna(price_at_date) or adv_at_date <= 0:
                logger.debug(
                    f"Missing or invalid data for impact calculation for {symbol} on {date}. ADV: {adv_at_date}, Vol: {vol_at_date}, Price: {price_at_date}. Returning 0 impact.")
                return 0.0

            adv_value = adv_at_date * price_at_date

            # Prevent division by zero if adv_value is tiny
            if adv_value <= 0:
                logger.debug(f"ADV value is non-positive for {symbol} on {date}. Returning 0 impact.")
                return 0.0

            # Price impact as a percentage of trade value
            # Note: A common model is (trade_shares / ADV_shares)^exponent, but this uses value.
            # This is a simplified version.
            price_impact_pct = self.impact_coefficient * vol_at_date * (abs(trade_value) / adv_value)

            # The total impact cost is price_impact_pct * trade_value
            cost = price_impact_pct * abs(trade_value)
            logger.debug(f"Calculated impact for {symbol} on {date} (Trade Value: {abs(trade_value):.2f}): {cost:.4f}")
            return cost
        except (KeyError, IndexError) as e:
            logger.warning(f"Data not found for {symbol} on {date} in market impact model: {e}. Returning 0 impact.")
            return 0.0
        except Exception as e:
            logger.error(
                f"An unexpected error occurred during market impact calculation for {symbol} on {date}: {e}. Returning 0 impact.")
            return 0.0


class TransactionCostOptimizer(BaseStrategyEvaluator):
    """
    An optimizer wrapper that incorporates a market impact model into the objective function.
    It penalizes the Sharpe Ratio based on estimated transaction costs,
    encouraging strategies with lower trading activity or smaller impact trades.
    """

    def __init__(self, strategy_class: Type[BaseStrategy], historical_data: pd.DataFrame, param_space: Dict):
        self.strategy_class = strategy_class
        self.historical_data = historical_data  # Full data for MarketImpactModel initialization
        self.param_space = param_space
        self.market_impact_model = MarketImpactModel(historical_data)
        # Data for objective function will be set dynamically via run_optimization
        self.data_for_objective: Optional[pd.DataFrame] = None
        logger.info(f"TransactionCostOptimizer initialized for strategy {strategy_class.__name__}.")

    def _objective(self, trial: optuna.Trial) -> float:
        if self.data_for_objective is None:
            raise RuntimeError("Optimization data not set. Call run_optimization with data.")

        params = {}
        for name, space in self.param_space.items():
            if space['type'] == 'int':
                params[name] = trial.suggest_int(name, space['low'], space['high'])
            elif space['type'] == 'float':
                params[name] = trial.suggest_float(name, space['low'], space['high'])
            elif space['type'] == 'categorical':  # Added categorical type
                params[name] = trial.suggest_categorical(name, space['choices'])
            # Add other types as needed (e.g., 'log_float')

        total_impact_cost = 0.0
        sharpe = -1.0  # Default sharpe
        initial_capital = 100000.0  # Default if backtester doesn't provide it

        try:
            strategy = self.strategy_class(**params)
            # The backtester runs on the data segment passed for this objective (train/val fold)
            backtester = Backtester(strategy, self.data_for_objective)
            # Fit strategy on the data segment for the objective function
            strategy.fit(self.data_for_objective)
            results = backtester.run()

            sharpe = results['metrics'].get('Sharpe Ratio', -1.0)
            initial_capital = backtester.initial_capital  # Get initial capital from backtester

            trades = pd.DataFrame(results.get('trades', []))  # Use .get with default empty list
            if not trades.empty:
                # Calculate impact for each trade
                for _, trade in trades.iterrows():
                    # Ensure trade has 'price' and 'quantity' to calculate value
                    if 'price' in trade and 'quantity' in trade and 'date' in trade and 'symbol' in trade:
                        trade_value = trade['price'] * trade['quantity']
                        # Pass the specific trade date to the market impact model
                        impact = self.market_impact_model.calculate_market_impact(
                            trade['date'],
                            trade['symbol'],
                            trade_value
                        )
                        total_impact_cost += impact
                    else:
                        logger.warning(f"Trade missing essential keys for impact calculation: {trade}. Skipping.")

            cost_penalty = total_impact_cost / initial_capital if initial_capital > 0 else 0.0

            # Combine Sharpe Ratio with cost penalty
            final_objective_value = (sharpe if pd.notna(sharpe) and np.isfinite(sharpe) else -1.0) - cost_penalty

            logger.debug(f"Params: {params}, Sharpe: {sharpe:.4f}, Impact Cost: {total_impact_cost:.2f}, "
                         f"Cost Penalty: {cost_penalty:.4f}, Objective: {final_objective_value:.4f}")

            return final_objective_value
        except Exception as e:
            logger.warning(f"Backtest with transaction cost failed for params {params}: {e}")
            return -2.0  # Return a very low value for failed backtests

    def run(self, data_segment: pd.DataFrame, n_trials: int = 100) -> Dict:
        """
        Runs the optimization study incorporating transaction costs.

        Args:
            data_segment (pd.DataFrame): The DataFrame on which to run backtests for optimization.
                                         This is typically a training or validation fold.
            n_trials (int): Number of optimization trials.

        Returns:
            Dict: Best parameters found by Optuna, adjusted for transaction costs.
        """
        self.data_for_objective = data_segment  # Set the data for the objective function
        study = optuna.create_study(direction='maximize')
        study.optimize(self._objective, n_trials=n_trials, show_progress_bar=True)

        logger.info(f"Best trial (Cost-Adjusted Sharpe): {study.best_trial.value:.4f}")
        logger.info(f"Best params: {study.best_params}")
        self.data_for_objective = None  # Clear data after optimization
        return study.best_params
