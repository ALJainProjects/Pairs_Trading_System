"""
Enhanced multi-strategy optimizer supporting various strategies and advanced optimization techniques.
"""
from typing import Dict, Tuple, Optional, Any, List, Union
import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from scipy.stats import spearmanr
import optuna
from config.logging_config import logger
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from src.strategy.backtest import MultiPairBackTester


class BaseStrategyEvaluator:
    """Base class containing common evaluation methods."""

    @staticmethod
    def _calculate_returns(signals: pd.DataFrame, data: pd.DataFrame) -> pd.Series:
        """Calculate strategy returns from signals."""
        position_returns = signals * data.shift(-1)
        return position_returns.sum(axis=1).fillna(0)

    @staticmethod
    def _calculate_sharpe_ratio(returns: pd.Series) -> float:
        """Calculate annualized Sharpe ratio."""
        if len(returns) < 2:
            return float('-inf')
        return (returns.mean() / returns.std()) * np.sqrt(252)

    @staticmethod
    def _calculate_max_drawdown(equity_curve: pd.Series) -> float:
        """Calculate maximum drawdown."""
        peaks = equity_curve.expanding().max()
        drawdowns = (equity_curve - peaks) / peaks
        return abs(drawdowns.min())

    @staticmethod
    def _analyze_parameter_stability(params_history: List[Dict]) -> Dict:
        """Analyze parameter stability across optimization iterations."""
        stability_metrics = {}
        for param in params_history[0].keys():
            values = [p[param] for p in params_history]
            stability_metrics[param] = {
                'mean': np.mean(values),
                'std': np.std(values),
                'cv': np.std(values) / np.mean(values) if np.mean(values) != 0 else np.inf
            }
        return stability_metrics

    @staticmethod
    def _analyze_performance_consistency(performance_history: List[Dict]) -> Dict:
        """Analyze performance consistency across optimization iterations."""
        consistency_metrics = {}
        for metric in performance_history[0].keys():
            values = [p[metric] for p in performance_history]
            consistency_metrics[metric] = {
                'mean': np.mean(values),
                'std': np.std(values),
                'cv': np.std(values) / np.mean(values) if np.mean(values) != 0 else np.inf
            }
        return consistency_metrics

    @staticmethod
    def _get_robust_parameters(params_history: List[Dict]) -> Dict:
        """Get robust parameters by taking median across iterations."""
        robust_params = {}
        for param in params_history[0].keys():
            values = [p[param] for p in params_history]
            robust_params[param] = np.median(values)
        return robust_params

class MultiStrategyOptimizer:
    """Enhanced optimizer supporting multiple strategy types and objectives."""

    def __init__(
            self,
            strategy: Any,
            prices: pd.DataFrame,  # Changed from returns to prices
            strategy_type: str,
            objective_weights: Optional[Dict[str, float]] = None,
            initial_capital: float = 100000,
            transaction_cost: float = 0.001,
            max_pairs: Optional[int] = None,
            cointegration_threshold: float = 0.05,
            correlation_threshold: float = 0.7,
            min_model_confidence: float = 0.6
    ):
        """Initialize the optimizer with strategy and parameters.

        Args:
            strategy: Trading strategy instance
            returns: DataFrame of asset returns
            strategy_type: Type of strategy ('Statistical', 'ML', or 'DL')
            objective_weights: Weight dict for optimization objectives
            initial_capital: Starting capital for backtesting
            transaction_cost: Cost per trade
            max_pairs: Maximum number of trading pairs
            cointegration_threshold: Threshold for cointegration tests
            correlation_threshold: Threshold for correlation filtering
            min_model_confidence: Minimum required model confidence
        """
        if 'Symbol' not in prices.columns:
            raise ValueError("Input data must contain 'Symbol' column")

        self.strategy = strategy
        self.prices = prices.copy()

        self.returns = prices.copy()
        self.returns['Return'] = self.returns.groupby('Symbol')['Adj_Close'].pct_change()

        self.strategy_type = strategy_type
        self.objective_weights = objective_weights or {
            'sharpe_ratio': 1.0,
            'max_drawdown': -1.0,
            'stability': 0.5,
            'model_confidence': 0.5
        }
        self.initial_capital = initial_capital
        self.transaction_cost = transaction_cost
        self.max_pairs = max_pairs
        self.cointegration_threshold = cointegration_threshold
        self.correlation_threshold = correlation_threshold
        self.min_model_confidence = min_model_confidence

        self.optimization_results = []
        self.feature_importance = pd.DataFrame()
        self.parameter_space = self._define_parameter_space()

    def _define_parameter_space(self) -> Dict:
        """Define strategy-specific parameter spaces."""
        base_params = {
            'max_position_size': (0.05, 0.2),
            'stop_loss': (0.02, 0.1),
            'take_profit': (0.02, 0.15)
        }

        strategy_params = {
            'Statistical': {
                'zscore_entry': (1.5, 3.0),
                'zscore_exit': (0.0, 1.0),
                'lookback_period': (10, 50),
                'min_half_life': (5, 20),
                'max_half_life': (50, 200)
            },
            'ML': {
                'model_confidence_threshold': (0.6, 0.9),
                'feature_window': (10, 50),
                'train_window': (126, 252),
                'n_estimators': (100, 500)
            },
            'DL': {
                'sequence_length': (10, 50),
                'prediction_horizon': (1, 5),
                'hidden_units': (32, 128),
                'learning_rate': (1e-4, 1e-2, 'log')
            }
        }

        if self.strategy_type not in strategy_params:
            raise ValueError(f"Unknown strategy type: {self.strategy_type}")

        return {**base_params, **strategy_params[self.strategy_type]}

    def _evaluate_parameters(self, params: Dict) -> float:
        """Evaluate parameters using multiple objectives."""
        try:
            self.strategy.reset()
            for param, value in params.items():
                setattr(self.strategy, param, value)

            backtester = self._create_backtester()
            equity_curve = backtester.run_backtest()

            returns = equity_curve.pct_change().dropna()
            metrics = {
                'sharpe_ratio': self._calculate_sharpe_ratio(returns),
                'max_drawdown': -self._calculate_max_drawdown(equity_curve),
                'stability': self._calculate_strategy_stability(returns),
                'model_confidence': self._evaluate_model_confidence()
            }

            weighted_score = sum(
                metrics[key] * weight
                for key, weight in self.objective_weights.items()
            )

            self.optimization_results.append({
                'parameters': params,
                'metrics': metrics,
                'score': weighted_score,
                'equity_curve': equity_curve
            })

            return weighted_score

        except Exception as e:
            logger.error(f"Parameter evaluation failed: {str(e)}")
            return float('-inf')

    def _create_backtester(self) -> Any:
        """Create backtester with strategy-specific configurations."""
        return MultiPairBackTester(
            strategy=self.strategy,
            prices=self.prices,
            initial_capital=self.initial_capital,
            transaction_cost=self.transaction_cost,
            max_pairs=self.max_pairs,
            cointegration_threshold=self.cointegration_threshold,
            min_liquidity_threshold=100000
        )

    def _calculate_sharpe_ratio(self, returns: pd.Series) -> float:
        """Calculate annualized Sharpe ratio."""
        if len(returns) < 2:
            return float('-inf')
        return (returns.mean() / returns.std()) * np.sqrt(252)

    def _calculate_max_drawdown(self, equity_curve: pd.Series) -> float:
        """Calculate maximum drawdown."""
        peaks = equity_curve.expanding().max()
        drawdowns = (equity_curve - peaks) / peaks
        return abs(drawdowns.min())

    def _calculate_strategy_stability(self, returns: pd.Series) -> float:
        """Calculate strategy stability score."""
        if len(returns) < 2:
            return 0.0

        rolling_sharpe = returns.rolling(window=63).apply(
            lambda x: self._calculate_sharpe_ratio(x)
        )

        market_returns = self.returns.mean(axis=1)
        rolling_beta = (
                returns.rolling(window=63).cov(market_returns) /
                market_returns.rolling(window=63).var()
        )

        stability_score = 1 - (rolling_sharpe.std() + rolling_beta.std()) / 2
        return max(0, stability_score)

    def _evaluate_model_confidence(self) -> float:
        """Evaluate model confidence based on strategy type."""
        if self.strategy_type in ['ML', 'DL']:
            if hasattr(self.strategy, 'model_confidence'):
                return self.strategy.model_confidence
            elif hasattr(self.strategy, 'predict_proba'):
                test_features = self.strategy.prepare_features(self.returns.tail(100))
                probas = self.strategy.predict_proba(test_features)
                return np.mean(np.max(probas, axis=1))
        return 1.0

    def bayesian_optimize(self, n_trials: int = 100) -> Tuple[Dict, float]:
        """Perform Bayesian optimization with enhanced objective."""
        logger.info("Starting Bayesian Optimization")

        def objective(trial):
            params = {}
            for param, bounds in self.parameter_space.items():
                if isinstance(bounds, tuple):
                    if len(bounds) == 3 and bounds[2] == 'log':
                        params[param] = trial.suggest_float(
                            param, bounds[0], bounds[1], log=True
                        )
                    elif isinstance(bounds[0], int):
                        params[param] = trial.suggest_int(
                            param, bounds[0], bounds[1]
                        )
                    else:
                        params[param] = trial.suggest_float(
                            param, bounds[0], bounds[1]
                        )
                else:
                    params[param] = trial.suggest_categorical(param, bounds)

            return self._evaluate_parameters(params)

        study = optuna.create_study(
            direction='maximize',
            sampler=optuna.samplers.TPESampler(seed=42)
        )
        study.optimize(objective, n_trials=n_trials)

        if self.strategy_type in ['ML', 'DL']:
            self._analyze_feature_importance()

        return study.best_params, study.best_value

    def _analyze_feature_importance(self) -> None:
        """Analyze feature importance for ML/DL strategies."""
        if hasattr(self.strategy, 'get_feature_importance'):
            self.feature_importance = self.strategy.get_feature_importance()
        elif hasattr(self.strategy, 'model') and hasattr(self.strategy.model, 'feature_importances_'):
            feature_names = (
                self.strategy.feature_names
                if hasattr(self.strategy, 'feature_names')
                else [f'feature_{i}' for i in range(len(self.strategy.model.feature_importances_))]
            )
            self.feature_importance = pd.Series(
                self.strategy.model.feature_importances_,
                index=feature_names
            ).sort_values(ascending=False)

    def plot_optimization_results(self) -> None:
        """Plot comprehensive optimization results."""
        results_df = pd.DataFrame([
            {
                **r['parameters'],
                **{f'metric_{k}': v for k, v in r['metrics'].items()},
                'score': r['score']
            }
            for r in self.optimization_results
        ])

        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                'Objective Score Distribution',
                'Parameter Correlations',
                'Metric Correlations',
                'Feature Importance'
            )
        )

        fig.add_trace(
            go.Histogram(x=results_df['score'], name='Score Dist'),
            row=1, col=1
        )

        param_corr = results_df[[k for k in self.parameter_space.keys()]].corr()
        fig.add_trace(
            go.Heatmap(
                z=param_corr.values,
                x=param_corr.columns,
                y=param_corr.columns,
                colorscale='RdBu',
                name='Param Corr'
            ),
            row=1, col=2
        )

        metric_cols = [col for col in results_df.columns if col.startswith('metric_')]
        metric_corr = results_df[metric_cols].corr()
        fig.add_trace(
            go.Heatmap(
                z=metric_corr.values,
                x=metric_corr.columns,
                y=metric_corr.columns,
                colorscale='RdBu',
                name='Metric Corr'
            ),
            row=2, col=1
        )

        if not self.feature_importance.empty:
            fig.add_trace(
                go.Bar(
                    x=self.feature_importance.index,
                    y=self.feature_importance.values,
                    name='Feature Imp'
                ),
                row=2, col=2
            )

        fig.update_layout(height=1000, title_text="Optimization Analysis")
        fig.show()

    def generate_optimization_report(self, output_file: Optional[str] = None) -> Dict:
        """Generate comprehensive optimization report."""
        results_df = pd.DataFrame([
            {
                **r['parameters'],
                **{f'metric_{k}': v for k, v in r['metrics'].items()},
                'score': r['score']
            }
            for r in self.optimization_results
        ]).sort_values('score', ascending=False)

        best_result = max(self.optimization_results, key=lambda x: x['score'])

        report = {
            'best_parameters': best_result['parameters'],
            'best_metrics': best_result['metrics'],
            'best_score': best_result['score'],
            'optimization_summary': {
                'n_trials': len(self.optimization_results),
                'score_mean': results_df['score'].mean(),
                'score_std': results_df['score'].std(),
                'parameter_ranges': {
                    param: {
                        'min': results_df[param].min(),
                        'max': results_df[param].max(),
                        'optimal': best_result['parameters'][param]
                    }
                    for param in self.parameter_space.keys()
                }
            }
        }

        if not self.feature_importance.empty:
            report['feature_importance'] = self.feature_importance.to_dict()

        if output_file:
            with pd.ExcelWriter(output_file) as writer:
                results_df.to_excel(
                    writer,
                    sheet_name='Optimization Results',
                    index=False
                )
                best_result['equity_curve'].to_frame('Value').to_excel(
                    writer,
                    sheet_name='Best Equity Curve'
                )

        return report


class WalkForwardOptimizer(BaseStrategyEvaluator):
    """Walk-forward optimization implementation."""

    def __init__(
            self,
            data: pd.DataFrame,
            train_size: int = 252,
            test_size: int = 126,
            step_size: int = 63
    ):
        """Initialize walk-forward optimizer."""
        super().__init__()
        self.n_splits = None
        self.data = data
        self.train_size = train_size
        self.test_size = test_size
        self.step_size = step_size
        self.optimization_results = []

    def get_fold_data(self, fold: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Get training and testing data for a specific fold."""
        if fold >= self.n_splits:
            raise ValueError(f"Invalid fold {fold}. Max fold is {self.n_splits - 1}")

        train_start = fold * self.step_size
        train_end = train_start + self.train_size
        test_end = train_end + self.test_size

        train_data = self.data.iloc[train_start:train_end]
        test_data = self.data.iloc[train_end:test_end]

        return train_data, test_data

    def get_n_windows(self) -> int:
        """Get total number of optimization windows."""
        total_size = len(self.data)
        return (total_size - self.train_size - self.test_size) // self.step_size + 1

    def _generate_windows(self) -> List[Tuple[range, range]]:
        """Generate walk-forward windows for optimization.

        Returns:
            List of tuples containing (train_indices, test_indices)
        """
        windows = []
        total_size = len(self.data)
        start = 0

        while start + self.train_size + self.test_size <= total_size:
            train_end = start + self.train_size
            test_end = train_end + self.test_size

            windows.append((
                range(start, train_end),
                range(train_end, test_end)
            ))

            start += self.step_size

        return windows

    def optimize(self, strategy: Any, param_grid: Dict) -> Dict:
        """Perform walk-forward optimization."""
        windows = self._generate_windows()

        for train_idx, test_idx in windows:
            train_data = self.data.iloc[train_idx]
            test_data = self.data.iloc[test_idx]

            optimizer = MultiStrategyOptimizer(
                strategy=strategy,
                returns=train_data,
                strategy_type=strategy.type
            )
            best_params, _ = optimizer.bayesian_optimize(n_trials=100)

            strategy.reset()
            for param, value in best_params.items():
                setattr(strategy, param, value)

            test_results = self._evaluate_strategy(strategy, test_data)

            self.optimization_results.append({
                'train_period': (train_data.index[0], train_data.index[-1]),
                'test_period': (test_data.index[0], test_data.index[-1]),
                'parameters': best_params,
                'test_metrics': test_results
            })

        return self._aggregate_results()

    def _evaluate_strategy(self, strategy: Any, data: pd.DataFrame) -> Dict:
        """Evaluate strategy performance."""
        signals = strategy.generate_signals(data)
        returns = self._calculate_returns(signals, data)

        return {
            'sharpe_ratio': self._calculate_sharpe_ratio(returns),
            'max_drawdown': self._calculate_max_drawdown(returns),
            'total_return': (1 + returns).prod() - 1
        }

    def _aggregate_results(self) -> Dict:
        """Aggregate walk-forward optimization results."""
        parameters_history = [r['parameters'] for r in self.optimization_results]
        performance_history = [r['test_metrics'] for r in self.optimization_results]

        return {
            'parameter_stability': self._analyze_parameter_stability(parameters_history),
            'performance_consistency': self._analyze_performance_consistency(performance_history),
            'optimal_parameters': self._get_robust_parameters(parameters_history)
        }


class ParameterSensitivityAnalyzer(BaseStrategyEvaluator):
    """Analyze parameter sensitivity and stability."""

    def __init__(self, strategy: Any, base_params: Dict):
        """Initialize parameter sensitivity analyzer."""
        super().__init__()
        self.strategy = strategy
        self.base_params = base_params
        self.sensitivity_results = {}

    def analyze_sensitivity(
            self,
            param_ranges: Dict,
            data: pd.DataFrame,
            n_points: int = 20
    ) -> Dict:
        """Analyze parameter sensitivity.

        Args:
            param_ranges: Ranges to test for each parameter
            data: Input data for testing
            n_points: Number of test points per parameter

        Returns:
            Dict containing sensitivity metrics for each parameter
        """
        for param, (min_val, max_val) in param_ranges.items():
            param_values = np.linspace(min_val, max_val, n_points)
            param_results = []

            for value in param_values:
                self.strategy.reset()
                test_params = self.base_params.copy()
                test_params[param] = value

                for p, v in test_params.items():
                    setattr(self.strategy, p, v)

                performance = self._evaluate_performance(data)
                param_results.append({
                    'value': value,
                    'performance': performance
                })

            self.sensitivity_results[param] = param_results

        return self._calculate_sensitivity_metrics()

    def _evaluate_performance(self, data: pd.DataFrame) -> Dict:
        """Evaluate strategy performance for sensitivity analysis."""
        signals = self.strategy.generate_signals(data)
        returns = self._calculate_returns(signals, data)

        return {
            'sharpe_ratio': self._calculate_sharpe_ratio(returns),
            'max_drawdown': self._calculate_max_drawdown(returns),
            'total_return': (1 + returns).prod() - 1
        }

    # noinspection PyTypeChecker
    def _calculate_sensitivity_metrics(self) -> Dict:
        """Calculate sensitivity metrics for each parameter."""
        sensitivity_metrics = {}

        for param, results in self.sensitivity_results.items():
            values = [r['value'] for r in results]
            performance = [r['performance']['sharpe_ratio'] for r in results]

            sensitivity_metrics[param] = {
                'correlation': spearmanr(np.array(values), np.array(performance))[0],
                'elasticity': self._calculate_elasticity(values, performance),
                'stability': self._calculate_stability(values, performance),
                'optimal_range': self._find_optimal_range(values, performance)
            }

        return sensitivity_metrics

    def _calculate_elasticity(self, values: List[float], performance: List[float]) -> float:
        """Calculate parameter elasticity."""
        pct_change_values = np.diff(values) / values[:-1]
        pct_change_perf = np.diff(performance) / performance[:-1]
        return np.mean(pct_change_perf / pct_change_values)

    # noinspection PyTypeChecker
    def _calculate_stability(self, values: List[float], performance: List[float]) -> float:
        """Calculate parameter stability."""
        window_size = len(values) // 4
        stability_scores = []

        for i in range(len(values) - window_size):
            window_values = values[i:i + window_size]
            window_perf = performance[i:i + window_size]
            stability_scores.append(spearmanr(np.array(window_values), np.array(window_perf))[0])

        return np.std(stability_scores)

    def _find_optimal_range(
            self,
            values: List[float],
            performance: List[float]
    ) -> Tuple[float, float]:
        """Find optimal parameter range."""
        threshold = np.percentile(performance, 75)
        optimal_mask = np.array(performance) >= threshold

        return (
            np.min(np.array(values)[optimal_mask]),
            np.max(np.array(values)[optimal_mask])
        )


class CrossValidatedOptimizer(BaseStrategyEvaluator):
    """Optimizer with cross-validation for parameter stability."""

    def __init__(
            self,
            data: pd.DataFrame,
            n_splits: int = 5,
            validation_size: float = 0.3
    ):
        """Initialize cross-validated optimizer.

        Args:
            data: Input time series data
            n_splits: Number of cross-validation splits
            validation_size: Size of validation set as fraction
        """
        super().__init__()
        self.data = data
        self.n_splits = n_splits
        self.validation_size = validation_size
        self.cv_results = []

    def optimize(self, strategy: Any, param_grid: Dict) -> Dict:
        """Perform cross-validated optimization."""
        tscv = TimeSeriesSplit(
            n_splits=self.n_splits,
            test_size=int(len(self.data) * self.validation_size)
        )

        for train_idx, val_idx in tscv.split(self.data):
            train_data = self.data.iloc[train_idx]
            val_data = self.data.iloc[val_idx]

            train_optimizer = MultiStrategyOptimizer(
                strategy=strategy,
                returns=train_data,
                strategy_type=strategy.type
            )
            best_params, best_score = train_optimizer.bayesian_optimize(n_trials=100)

            val_performance = self._validate_parameters(
                strategy,
                best_params,
                val_data
            )

            self.cv_results.append({
                'parameters': best_params,
                'train_score': best_score,
                'val_performance': val_performance
            })

        return self._aggregate_cv_results()

    def _validate_parameters(
            self,
            strategy: Any,
            parameters: Dict,
            validation_data: pd.DataFrame
    ) -> Dict:
        """Validate parameters on validation set."""
        strategy.reset()
        for param, value in parameters.items():
            setattr(strategy, param, value)

        signals = strategy.generate_signals(validation_data)
        returns = self._calculate_returns(signals, validation_data)

        return {
            'sharpe_ratio': self._calculate_sharpe_ratio(returns),
            'max_drawdown': self._calculate_max_drawdown(returns),
            'total_return': (1 + returns).prod() - 1,
            'stability_score': self._calculate_stability_score(returns)
        }

    def _calculate_stability_score(self, returns: pd.Series) -> float:
        """Calculate strategy stability score."""
        window_size = len(returns) // 4
        rolling_sharpe = returns.rolling(window=window_size).apply(
            lambda x: self._calculate_sharpe_ratio(x)
        )
        return 1 - rolling_sharpe.std()

    def _aggregate_cv_results(self) -> Dict:
        """Aggregate cross-validation results."""
        param_stability = {}
        performance_stability = {}

        all_params = [r['parameters'] for r in self.cv_results]
        for param in all_params[0].keys():
            values = [p[param] for p in all_params]
            param_stability[param] = {
                'mean': np.mean(values),
                'std': np.std(values),
                'cv': np.std(values) / np.mean(values)
            }

        val_metrics = [r['val_performance'] for r in self.cv_results]
        for metric in val_metrics[0].keys():
            values = [m[metric] for m in val_metrics]
            performance_stability[metric] = {
                'mean': np.mean(values),
                'std': np.std(values),
                'cv': np.std(values) / np.mean(values)
            }

        return {
            'parameter_stability': param_stability,
            'performance_stability': performance_stability,
            'optimal_parameters': self._get_robust_parameters()
        }

    def _get_robust_parameters(self) -> Dict:
        """Get robust parameters across folds."""
        all_params = [r['parameters'] for r in self.cv_results]
        robust_params = {}

        for param in all_params[0].keys():
            values = [p[param] for p in all_params]
            robust_params[param] = np.median(values)

        return robust_params

    def get_cv_fold(self, fold: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Get training and validation data for a specific fold."""
        tscv = TimeSeriesSplit(
            n_splits=self.n_splits,
            test_size=int(len(self.data) * self.validation_size)
        )

        splits = list(tscv.split(self.data))
        if fold >= len(splits):
            raise ValueError(f"Invalid fold {fold}. Max fold is {len(splits) - 1}")

        train_idx, val_idx = splits[fold]
        train_data = self.data.iloc[train_idx]
        val_data = self.data.iloc[val_idx]

        return train_data, val_data


class TransactionCostOptimizer:
    """Optimizer considering transaction costs."""

    def __init__(
            self,
            base_optimizer: Any,
            cost_model: Any,  # Should be MarketImpactModel
            cost_weight: float = 0.5
    ):
        """Initialize transaction cost-aware optimizer.

        Args:
            base_optimizer: Base optimization instance
            cost_model: Market impact cost model
            cost_weight: Weight for transaction costs in objective
        """
        self.base_optimizer = base_optimizer
        self.cost_model = cost_model
        self.cost_weight = cost_weight

    def optimize(self, strategy: Any, param_grid: Dict) -> Dict:
        """Perform optimization with transaction cost consideration."""

        def cost_adjusted_objective(params: Dict) -> float:
            base_score = self.base_optimizer._evaluate_parameters(params)

            strategy.reset()
            for param, value in params.items():
                setattr(strategy, param, value)

            signals = strategy.generate_signals(self.base_optimizer.returns)
            total_cost = self._calculate_total_costs(signals)

            return base_score - self.cost_weight * total_cost

        self.base_optimizer.objective_function = cost_adjusted_objective
        return self.base_optimizer.optimize(strategy, param_grid)

    def _calculate_total_costs(self, signals: pd.DataFrame) -> float:
        """Calculate total transaction costs."""
        total_cost = 0.0
        for pair in signals.columns:
            if isinstance(pair, tuple):
                trades = signals[pair].diff().fillna(0)
                trades = trades[trades != 0]

                for date, size in trades.items():
                    impact = self.cost_model.calculate_market_impact(
                        pair=pair,
                        trade_size=abs(size),
                        direction=np.sign(size)
                    )
                    total_cost += impact

        return total_cost

class MarketImpactModel:
    """
    Model for calculating market impact of trades considering both temporary and permanent impact.
    Implements square-root law for market impact and handles bid-ask spread costs.
    """

    def __init__(
            self,
            volatility_window: int = 63,
            spread_window: int = 21,
            impact_coefficient: float = 0.1,
            min_spread: float = 0.0001,
            decay_factor: float = 0.5
    ):
        """
        Initialize the market impact model.

        Args:
            volatility_window: Window for calculating rolling volatility
            spread_window: Window for calculating average spread
            impact_coefficient: Coefficient for square-root impact formula
            min_spread: Minimum spread to use when actual spread is unavailable
            decay_factor: Decay factor for temporary impact
        """
        self.volatility_window = volatility_window
        self.spread_window = spread_window
        self.impact_coefficient = impact_coefficient
        self.min_spread = min_spread
        self.decay_factor = decay_factor

        self._volatility_cache = {}
        self._spread_cache = {}
        self._adv_cache = {}

    def calculate_market_impact(
            self,
            pair: Union[str, Tuple[str, str]],
            trade_size: float,
            direction: int,
            price_data: Optional[pd.DataFrame] = None,
            volume_data: Optional[pd.DataFrame] = None,
            current_date: Optional[pd.Timestamp] = None
    ) -> Dict[str, float]:
        """
        Calculate market impact for a trade.

        Args:
            pair: Trading pair identifier
            trade_size: Size of the trade in base currency
            direction: Trade direction (1 for buy, -1 for sell)
            price_data: Historical price data for volatility calculation
            volume_data: Historical volume data for ADV calculation
            current_date: Current date for historical calculations

        Returns:
            Dict containing temporary impact, permanent impact, and total cost
        """
        volatility = self._get_volatility(pair, price_data, current_date)
        spread = self._get_spread(pair, price_data, current_date)
        adv = self._get_adv(pair, volume_data, current_date)

        temporary_impact = self._calculate_temporary_impact(
            trade_size=trade_size,
            volatility=volatility,
            adv=adv
        )

        permanent_impact = self._calculate_permanent_impact(
            trade_size=trade_size,
            volatility=volatility,
            adv=adv
        )

        spread_cost = self._calculate_spread_cost(
            trade_size=trade_size,
            spread=spread,
            direction=direction
        )

        total_cost = (temporary_impact + permanent_impact + spread_cost) * abs(trade_size)

        return {
            'temporary_impact': temporary_impact,
            'permanent_impact': permanent_impact,
            'spread_cost': spread_cost,
            'total_cost': total_cost
        }

    def _get_volatility(
            self,
            pair: Union[str, Tuple[str, str]],
            price_data: Optional[pd.DataFrame],
            current_date: Optional[pd.Timestamp]
    ) -> float:
        """Calculate historical volatility."""
        if price_data is not None and current_date is not None:
            if pair not in self._volatility_cache:
                self._volatility_cache[pair] = {}

            if current_date not in self._volatility_cache[pair]:
                hist_data = price_data.loc[:current_date].tail(self.volatility_window)
                returns = np.log(hist_data[pair]).diff().dropna()
                volatility = returns.std() * np.sqrt(252)
                self._volatility_cache[pair][current_date] = volatility

            return self._volatility_cache[pair][current_date]

        return 0.2

    def _get_spread(
            self,
            pair: Union[str, Tuple[str, str]],
            price_data: Optional[pd.DataFrame],
            current_date: Optional[pd.Timestamp]
    ) -> float:
        """Calculate average bid-ask spread."""
        if price_data is not None and current_date is not None:
            if pair not in self._spread_cache:
                self._spread_cache[pair] = {}

            if current_date not in self._spread_cache[pair]:
                hist_data = price_data.loc[:current_date].tail(self.spread_window)
                if 'spread' in hist_data.columns:
                    spread = hist_data['spread'].mean()
                else:
                    spread = (hist_data['High'] - hist_data['Low']).mean() / \
                             hist_data['Close'].mean()
                self._spread_cache[pair][current_date] = max(spread, self.min_spread)

            return self._spread_cache[pair][current_date]

        return self.min_spread

    def _get_adv(
            self,
            pair: Union[str, Tuple[str, str]],
            volume_data: Optional[pd.DataFrame],
            current_date: Optional[pd.Timestamp]
    ) -> float:
        """Calculate average daily volume."""
        if volume_data is not None and current_date is not None:
            if pair not in self._adv_cache:
                self._adv_cache[pair] = {}

            if current_date not in self._adv_cache[pair]:
                hist_data = volume_data.loc[:current_date].tail(self.volatility_window)
                adv = hist_data[pair].mean()
                self._adv_cache[pair][current_date] = adv

            return self._adv_cache[pair][current_date]

        return 1e6

    def _calculate_temporary_impact(
            self,
            trade_size: float,
            volatility: float,
            adv: float
    ) -> float:
        """
        Calculate temporary price impact using square-root formula.

        Impact = σ * C * sqrt(Q/ADV)
        where:
        - σ is volatility
        - C is impact coefficient
        - Q is trade size
        - ADV is average daily volume
        """
        normalized_size = abs(trade_size) / adv
        impact = volatility * self.impact_coefficient * np.sqrt(normalized_size)
        return impact * np.exp(-normalized_size * self.decay_factor)

    def _calculate_permanent_impact(
            self,
            trade_size: float,
            volatility: float,
            adv: float
    ) -> float:
        """
        Calculate permanent price impact.
        Typically smaller than temporary impact and doesn't decay.
        """
        normalized_size = abs(trade_size) / adv
        return volatility * self.impact_coefficient * 0.4 * np.sqrt(normalized_size)

    def _calculate_spread_cost(
            self,
            trade_size: float,
            spread: float,
            direction: int
    ) -> float:
        """Calculate cost due to bid-ask spread."""
        return 0.5 * spread * np.sign(direction)

    def estimate_optimal_trade_size(
            self,
            pair: Union[str, Tuple[str, str]],
            target_position: float,
            current_position: float,
            price_data: Optional[pd.DataFrame] = None,
            volume_data: Optional[pd.DataFrame] = None,
            current_date: Optional[pd.Timestamp] = None,
            max_participation_rate: float = 0.1
    ) -> float:
        """
        Estimate optimal trade size considering market impact.

        Args:
            pair: Trading pair identifier
            target_position: Desired position size
            current_position: Current position size
            price_data: Historical price data
            volume_data: Historical volume data
            current_date: Current date
            max_participation_rate: Maximum allowed participation rate

        Returns:
            Optimal trade size
        """
        total_trade_needed = target_position - current_position
        direction = np.sign(total_trade_needed)
        adv = self._get_adv(pair, volume_data, current_date)

        max_trade_size = adv * max_participation_rate

        if abs(total_trade_needed) <= max_trade_size:
            return total_trade_needed

        return direction * max_trade_size

    def calculate_trade_schedule(
            self,
            pair: Union[str, Tuple[str, str]],
            total_size: float,
            num_periods: int,
            price_data: Optional[pd.DataFrame] = None,
            volume_data: Optional[pd.DataFrame] = None,
            current_date: Optional[pd.Timestamp] = None
    ) -> pd.Series:
        """
        Calculate optimal trade schedule using TWAP/VWAP hybrid approach.

        Args:
            pair: Trading pair identifier
            total_size: Total position to trade
            num_periods: Number of periods to split the trade
            price_data: Historical price data
            volume_data: Historical volume data
            current_date: Current date

        Returns:
            Series of trade sizes for each period
        """
        if volume_data is not None:
            volume_profile = self._calculate_volume_profile(volume_data, num_periods)
            trade_sizes = total_size * volume_profile
        else:
            trade_sizes = pd.Series([total_size / num_periods] * num_periods)

        adv = self._get_adv(pair, volume_data, current_date)
        max_trade_size = adv * 0.1
        trade_sizes = trade_sizes.clip(upper=max_trade_size)

        trade_sizes = trade_sizes * (total_size / trade_sizes.sum())

        return trade_sizes

    def _calculate_volume_profile(
            self,
            volume_data: pd.DataFrame,
            num_periods: int
    ) -> pd.Series:
        """Calculate average volume profile."""
        if isinstance(volume_data.index, pd.DatetimeIndex):
            volume_profile = volume_data.groupby(volume_data.index.time).mean()
        else:
            volume_profile = pd.Series(np.ones(num_periods) / num_periods)

        return volume_profile / volume_profile.sum()