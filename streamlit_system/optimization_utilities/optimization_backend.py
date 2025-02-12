"""
Complete optimization backend implementation with all methodologies.
"""
import json
from typing import Dict, Tuple, Optional, Any, List, Callable
import pandas as pd
import numpy as np
import optuna
import logging
from datetime import datetime
import time
from pathlib import Path
from dataclasses import dataclass, field

from numpy._typing import _64Bit

from src.strategy.backtest import MultiPairBackTester
from src.strategy.optimization import MultiStrategyOptimizer, MarketImpactModel, WalkForwardOptimizer, \
    CrossValidatedOptimizer, ParameterSensitivityAnalyzer, TransactionCostOptimizer


# ====== Data Models ======

@dataclass
class StrategyParameters:
    """Strategy parameters container."""
    values: Dict[str, Any]
    bounds: Dict[str, Tuple[float, float]]
    parameter_types: Dict[str, str] = field(default_factory=dict)
    descriptions: Dict[str, str] = field(default_factory=dict)

    def validate(self) -> bool:
        """Validate parameters against bounds."""
        try:
            for param, value in self.values.items():
                if param not in self.bounds:
                    raise ValueError(f"No bounds defined for parameter {param}")
                lower, upper = self.bounds[param]
                if not lower <= float(value) <= upper:
                    raise ValueError(f"Parameter {param} value {value} outside bounds [{lower}, {upper}]")
            return True
        except Exception as e:
            logging.error(f"Parameter validation error: {str(e)}")
            return False

    @property
    def optimization_space(self) -> Dict[str, Tuple]:
        """Get parameter space for optimization."""
        space = {}
        for param, (lower, upper) in self.bounds.items():
            param_type = self.parameter_types.get(param, 'float')
            if param_type == 'int':
                space[param] = (int(lower), int(upper))
            elif param_type == 'float':
                space[param] = (float(lower), float(upper))
            elif param_type == 'log':
                space[param] = (float(lower), float(upper), 'log')
        return space


@dataclass
class OptimizationResult:
    """Optimization result container."""
    parameters: Dict[str, Any]
    score: float
    trial_number: int
    timestamp: datetime = field(default_factory=datetime.now)
    metrics: Dict[str, float] = field(default_factory=dict)
    equity_curve: Optional[pd.Series] = None
    additional_info: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        result = {
            'parameters': self.parameters,
            'score': self.score,
            'trial_number': self.trial_number,
            'timestamp': self.timestamp.isoformat(),
            'metrics': self.metrics,
            'additional_info': self.additional_info
        }
        if self.equity_curve is not None:
            result['equity_curve'] = self.equity_curve.to_dict()
        return result


# ====== Main Optimizer Class ======

class OptimizationBackend:
    """Complete backend for all optimization methods."""

    def __init__(self, config: Optional[Dict] = None):
        """Initialize optimization backend."""
        self.config = config or {}
        self.strategy = None
        self.data = None
        self.parameters = None
        self.results_history = []
        self.best_result = None

        # Optimization components
        self.walk_forward = None
        self.cross_validated = None
        self.sensitivity_analyzer = None
        self.transaction_optimizer = None
        self.market_impact_model = None

    def initialize(
            self,
            strategy: Any,
            data: pd.DataFrame,
            parameters: StrategyParameters,
            settings: Optional[Dict] = None
    ) -> None:
        """Initialize optimization with all components."""
        self.strategy = strategy
        self.data = data
        self.parameters = parameters
        if settings:
            self.config.update(settings)

        # Initialize market impact model
        self.market_impact_model = MarketImpactModel(
            volatility_window=self.config.get('volatility_window', 63),
            spread_window=self.config.get('spread_window', 21),
            impact_coefficient=self.config.get('impact_coefficient', 0.1),
            min_spread=self.config.get('min_spread', 0.0001),
            decay_factor=self.config.get('decay_factor', 0.5)
        )

        # Initialize walk-forward optimizer
        self.walk_forward = WalkForwardOptimizer(
            data=data,
            train_size=self.config.get('train_size', 252),
            test_size=self.config.get('test_size', 126),
            step_size=self.config.get('step_size', 63)
        )

        # Initialize cross-validated optimizer
        self.cross_validated = CrossValidatedOptimizer(
            data=data,
            n_splits=self.config.get('cv_folds', 5),
            validation_size=self.config.get('validation_size', 0.3)
        )

        # Initialize sensitivity analyzer
        self.sensitivity_analyzer = ParameterSensitivityAnalyzer(
            strategy=strategy,
            base_params=parameters.values
        )

        # Initialize transaction cost optimizer
        self.transaction_optimizer = TransactionCostOptimizer(
            base_optimizer=self,
            cost_model=self.market_impact_model,
            cost_weight=self.config.get('cost_weight', 0.5)
        )

    def optimize(
            self,
            method: str = 'bayesian',
            progress_callback: Optional[Callable[[float], None]] = None,
            status_callback: Optional[Callable[[str], None]] = None
    ) -> Dict[str, Any]:
        """Run optimization with specified method."""
        try:
            start_time = time.time()

            if method == 'walk_forward':
                results = self._run_walk_forward_optimization(
                    progress_callback,
                    status_callback
                )
            elif method == 'cross_validated':
                results = self._run_cross_validated_optimization(
                    progress_callback,
                    status_callback
                )
            elif method == 'sensitivity':
                results = self._run_sensitivity_analysis(
                    progress_callback,
                    status_callback
                )
            elif method == 'transaction_cost':
                results = self._run_transaction_cost_optimization(
                    progress_callback,
                    status_callback
                )
            else:
                results = self._run_bayesian_optimization(
                    progress_callback,
                    status_callback
                )

            # Add common information
            results.update({
                'time_taken': time.time() - start_time,
                'optimization_method': method,
                'strategy_type': self.strategy.__class__.__name__,
                'data_info': {
                    'start_date': self.data.index[0],
                    'end_date': self.data.index[-1],
                    'n_assets': len(self.data.columns)
                }
            })

            return results

        except Exception as e:
            logging.error(f"Optimization error: {str(e)}")
            raise

    def _run_bayesian_optimization(
            self,
            progress_callback: Optional[Callable[[float], None]],
            status_callback: Optional[Callable[[str], None]]
    ) -> Dict[str, Any]:
        """Run Bayesian optimization."""
        try:
            def objective(trial):
                params = {}
                for name, bounds in self.parameters.optimization_space.items():
                    if len(bounds) == 3 and bounds[2] == 'log':
                        params[name] = trial.suggest_float(
                            name, bounds[0], bounds[1], log=True
                        )
                    elif isinstance(bounds[0], int):
                        params[name] = trial.suggest_int(
                            name, bounds[0], bounds[1]
                        )
                    else:
                        params[name] = trial.suggest_float(
                            name, bounds[0], bounds[1]
                        )

                score, _ = self._evaluate_parameters(params)

                if progress_callback:
                    progress_callback(trial.number / self.config.get('n_trials', 100))
                if status_callback:
                    status_callback(f"Trial {trial.number}: Score = {score:.4f}")

                return score

            # Create and run study
            study = optuna.create_study(
                direction='maximize',
                sampler=optuna.samplers.TPESampler(seed=42)
            )

            study.optimize(
                objective,
                n_trials=self.config.get('n_trials', 100),
                callbacks=[self._optuna_callback]
            )

            return {
                'best_parameters': study.best_params,
                'best_score': study.best_value,
                'best_trial': study.best_trial.number,
                'n_trials': len(study.trials),
                'optimization_history': [
                    {
                        'trial': t.number,
                        'score': t.value,
                        'params': t.params
                    }
                    for t in study.trials
                ]
            }

        except Exception as e:
            logging.error(f"Bayesian optimization error: {str(e)}")
            raise

    def _run_walk_forward_optimization(
            self,
            progress_callback: Optional[Callable[[float], None]],
            status_callback: Optional[Callable[[str], None]]
    ) -> Dict[str, Any]:
        """Run walk-forward optimization."""
        try:
            windows = self.walk_forward._generate_windows()
            total_windows = len(windows)
            results = []

            for i, (train_idx, test_idx) in enumerate(windows):
                if status_callback:
                    status_callback(f"Processing window {i + 1}/{total_windows}")

                train_data = self.data.iloc[train_idx]
                test_data = self.data.iloc[test_idx]

                # Optimize on training data
                train_optimizer = MultiStrategyOptimizer(
                    strategy=self.strategy,
                    prices=train_data,
                    strategy_type=self.strategy.__class__.__name__
                )
                best_params, best_score = train_optimizer.bayesian_optimize(
                    n_trials=self.config.get('window_trials', 100)
                )

                # Evaluate on test data
                _, test_metrics = self._evaluate_parameters(best_params, test_data)

                results.append({
                    'window': i + 1,
                    'train_period': (train_data.index[0], train_data.index[-1]),
                    'test_period': (test_data.index[0], test_data.index[-1]),
                    'parameters': best_params,
                    'train_score': best_score,
                    'test_metrics': test_metrics
                })

                if progress_callback:
                    progress_callback((i + 1) / total_windows)

            return {
                'window_results': results,
                'parameter_stability': self._analyze_parameter_stability(
                    [r['parameters'] for r in results]
                ),
                'performance_stability': self._analyze_performance_stability(
                    [r['test_metrics'] for r in results]
                ),
                'optimal_parameters': self._get_robust_parameters(
                    [r['parameters'] for r in results]
                )
            }

        except Exception as e:
            logging.error(f"Walk-forward optimization error: {str(e)}")
            raise

    def _run_cross_validated_optimization(
            self,
            progress_callback: Optional[Callable[[float], None]],
            status_callback: Optional[Callable[[str], None]]
    ) -> Dict[str, Any]:
        """Run cross-validated optimization."""
        try:
            cv_results = self.cross_validated.optimize(
                strategy=self.strategy,
                param_grid=self.parameters.optimization_space
            )

            if status_callback:
                status_callback("Cross-validation completed")

            return {
                'cv_results': cv_results,
                'parameter_stability': cv_results['parameter_stability'],
                'performance_stability': cv_results['performance_stability'],
                'optimal_parameters': cv_results['optimal_parameters']
            }

        except Exception as e:
            logging.error(f"Cross-validated optimization error: {str(e)}")
            raise

    def _run_sensitivity_analysis(
            self,
            progress_callback: Optional[Callable[[float], None]],
            status_callback: Optional[Callable[[str], None]]
    ) -> Dict[str, Any]:
        """Run parameter sensitivity analysis."""
        try:
            n_points = self.config.get('sensitivity_points', 20)

            sensitivity_metrics = {}
            total_params = len(self.parameters.optimization_space)

            for i, (param, bounds) in enumerate(self.parameters.optimization_space.items()):
                if status_callback:
                    status_callback(f"Analyzing parameter: {param}")

                param_results = self.sensitivity_analyzer.analyze_sensitivity(
                    {param: bounds},
                    self.data,
                    n_points
                )

                sensitivity_metrics[param] = param_results[param]

                if progress_callback:
                    progress_callback((i + 1) / total_params)

            # Find optimal parameters
            optimal_params = {}
            for param, metrics in sensitivity_metrics.items():
                optimal_params[param] = np.mean(metrics['optimal_range'])

            return {
                'sensitivity_metrics': sensitivity_metrics,
                'optimal_parameters': optimal_params,
                'parameter_importance': {
                    param: metrics['correlation']
                    for param, metrics in sensitivity_metrics.items()
                }
            }

        except Exception as e:
            logging.error(f"Sensitivity analysis error: {str(e)}")
            raise

    def _run_transaction_cost_optimization(
            self,
            progress_callback: Optional[Callable[[float], None]],
            status_callback: Optional[Callable[[str], None]]
    ) -> Dict[str, Any]:
        """Run transaction cost-aware optimization."""
        try:
            if status_callback:
                status_callback("Starting transaction cost optimization")

            # Run optimization
            results = self.transaction_optimizer.optimize(
                self.strategy,
                self.parameters.optimization_space
            )

            # Calculate detailed costs
            signals = self.strategy.generate_signals(self.data)
            costs = self._calculate_detailed_costs(signals)

            if progress_callback:
                progress_callback(1.0)

            return {
                'optimal_parameters': results['optimal_parameters'],
                'base_metrics': results['metrics'],
                'transaction_costs': costs,
                'net_metrics': {
                    k: v - costs['total_cost'] if k in ['total_return', 'sharpe_ratio'] else v
                    for k, v in results['metrics'].items()
                },
                'cost_breakdown': costs
            }

        except Exception as e:
            logging.error(f"Transaction cost optimization error: {str(e)}")
            raise

    def _evaluate_parameters(
            self,
            params: Dict[str, Any],
            data: Optional[pd.DataFrame] = None
    ) -> tuple[float, dict]:
        """Evaluate parameters and return score."""
        try:
            eval_data = data if data is not None else self.data

            # Reset strategy and set parameters
            self.strategy.reset()
            for param, value in params.items():
                setattr(self.strategy, param, value)

            # Run backtest
            backtester = MultiPairBackTester(
                strategy=self.strategy,
                prices=eval_data,
                initial_capital=self.config.get('initial_capital', 100000),
                transaction_cost=self.config.get('transaction_cost', 0.001),
                max_pairs=self.config.get('max_pairs')
            )

            equity_curve = backtester.run_backtest()
            returns = equity_curve.pct_change().dropna()

            # Calculate metrics
            metrics = {
                'sharpe_ratio': self._calculate_sharpe_ratio(returns),
                'max_drawdown': -self._calculate_max_drawdown(equity_curve),
                'stability': self._calculate_strategy_stability(returns),
                'model_confidence': self._evaluate_model_confidence()
            }

            # Calculate weighted score
            weights = self.config.get('objective_weights', {
                'sharpe_ratio': 1.0,
                'max_drawdown': -1.0,
                'stability': 0.5,
                'model_confidence': 0.5
            })

            score = sum(
                metrics[key] * weights.get(key, 0.0)
                for key in metrics.keys()
            )

            # Store result
            result = OptimizationResult(
                parameters=params,
                score=score,
                trial_number=len(self.results_history) + 1,
                metrics=metrics,
                equity_curve=equity_curve
            )
            self.results_history.append(result)

            # Update best result
            if self.best_result is None or score > self.best_result.score:
                self.best_result = result

            return (score, result.to_dict())

        except Exception as e:
            logging.error(f"Parameter evaluation error: {str(e)}")
            return (float('-inf'), {})

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

        # Calculate rolling Sharpe ratio
        window = min(63, len(returns) // 4)
        rolling_sharpe = returns.rolling(window=window).apply(
            lambda x: self._calculate_sharpe_ratio(x)
        )

        # Calculate rolling beta to market
        market_returns = self.data.mean(axis=1).pct_change()
        rolling_beta = (
                returns.rolling(window=window).cov(market_returns) /
                market_returns.rolling(window=window).var()
        )

        # Calculate stability score
        stability_score = 1 - (rolling_sharpe.std() + rolling_beta.std()) / 2
        return max(0, stability_score)

    def _evaluate_model_confidence(self) -> float:
        """Evaluate model confidence based on strategy type."""
        if hasattr(self.strategy, 'model_confidence'):
            return self.strategy.model_confidence
        elif hasattr(self.strategy, 'predict_proba'):
            test_features = self.strategy.prepare_features(self.data.tail(100))
            probas = self.strategy.predict_proba(test_features)
            return np.mean(np.max(probas, axis=1))
        return 1.0

    def _calculate_detailed_costs(self, signals: pd.DataFrame) -> Dict[str, float]:
        """Calculate detailed transaction costs breakdown."""
        try:
            total_cost = 0.0
            cost_breakdown = {
                'temporary_impact': 0.0,
                'permanent_impact': 0.0,
                'spread_costs': 0.0
            }

            for pair in signals.columns:
                if isinstance(pair, tuple):
                    trades = signals[pair].diff().fillna(0)
                    trades = trades[trades != 0]

                    for date, size in trades.items():
                        impact = self.market_impact_model.calculate_market_impact(
                            pair=pair,
                            trade_size=abs(size),
                            direction=np.sign(size),
                            price_data=self.data,
                            current_date=date
                        )

                        cost_breakdown['temporary_impact'] += impact['temporary_impact']
                        cost_breakdown['permanent_impact'] += impact['permanent_impact']
                        cost_breakdown['spread_costs'] += impact['spread_cost']
                        total_cost += impact['total_cost']

            cost_breakdown['total_cost'] = total_cost

            if total_cost > 0:
                cost_breakdown['cost_breakdown_pct'] = {
                    k: v / total_cost * 100
                    for k, v in cost_breakdown.items()
                    if k != 'total_cost'
                }

            return cost_breakdown

        except Exception as e:
            logging.error(f"Cost calculation error: {str(e)}")
            return {'total_cost': 0.0}

    def _analyze_parameter_stability(self, params_history: List[Dict]) -> Dict:
        """Analyze parameter stability across iterations."""
        stability_metrics = {}

        for param in params_history[0].keys():
            values = [p[param] for p in params_history]
            stability_metrics[param] = {
                'mean': np.mean(values),
                'std': np.std(values),
                'cv': np.std(values) / np.mean(values) if np.mean(values) != 0 else np.inf,
                'min': np.min(values),
                'max': np.max(values),
                'median': np.median(values),
                'q25': np.percentile(values, 25),
                'q75': np.percentile(values, 75)
            }

        return stability_metrics

    def _analyze_performance_stability(self, metrics_history: List[Dict]) -> Dict:
        """Analyze performance stability across iterations."""
        stability_metrics = {}

        for metric in metrics_history[0].keys():
            values = [m[metric] for m in metrics_history]
            stability_metrics[metric] = {
                'mean': np.mean(values),
                'std': np.std(values),
                'cv': np.std(values) / np.mean(values) if np.mean(values) != 0 else np.inf,
                'min': np.min(values),
                'max': np.max(values),
                'median': np.median(values),
                'q25': np.percentile(values, 25),
                'q75': np.percentile(values, 75)
            }

        return stability_metrics

    def _get_robust_parameters(self, params_history: List[Dict]) -> Dict:
        """Get robust parameters by taking median across iterations."""
        robust_params = {}

        for param in params_history[0].keys():
            values = [p[param] for p in params_history]
            robust_params[param] = np.median(values)

        return robust_params

    def _optuna_callback(self, study: optuna.Study, trial: optuna.Trial) -> None:
        """Callback for Optuna optimization."""
        if self.best_result is None or trial.study.best_value > self.best_result.score:
            result = OptimizationResult(
                parameters=trial.params,
                score=trial.study.best_value,
                trial_number=trial.number,
                metrics=trial.user_attrs.get('metrics', {})
            )
            self.best_result = result

        if trial.number % 10 == 0:
            logging.info(
                f"Trial {trial.number}: "
                f"Score = {trial.study.best_value:.4f}, "
                f"Best = {study.best_value:.4f}"
            )

    def save_results(self, path: str) -> None:
        """Save optimization results to file."""
        try:
            save_path = Path(path)
            save_path.parent.mkdir(parents=True, exist_ok=True)

            results = {
                'optimization_method': self.config.get('method', 'bayesian'),
                'best_result': self.best_result.to_dict() if self.best_result else None,
                'history': [r.to_dict() for r in self.results_history],
                'config': self.config,
                'parameter_stability': self._analyze_parameter_stability(
                    [r.parameters for r in self.results_history]
                ),
                'performance_stability': self._analyze_performance_stability(
                    [r.metrics for r in self.results_history]
                )
            }

            with open(save_path, 'w') as f:
                json.dump(results, f, indent=4, default=str)

        except Exception as e:
            logging.error(f"Error saving results: {str(e)}")

    def load_results(self, path: str) -> None:
        """Load optimization results from file."""
        try:
            with open(path, 'r') as f:
                results = json.load(f)

            self.config = results.get('config', {})
            self.results_history = [
                OptimizationResult(**r) for r in results.get('history', [])
            ]

            if results.get('best_result'):
                self.best_result = OptimizationResult(**results['best_result'])

        except Exception as e:
            logging.error(f"Error loading results: {str(e)}")

    def get_optimization_summary(self) -> Dict[str, Any]:
        """Get comprehensive optimization summary."""
        if not self.results_history:
            return {}

        return {
            'optimization_method': self.config.get('method', 'bayesian'),
            'total_trials': len(self.results_history),
            'best_score': self.best_result.score if self.best_result else None,
            'best_parameters': self.best_result.parameters if self.best_result else None,
            'optimization_time': sum(
                r.additional_info.get('time_taken', 0)
                for r in self.results_history
            ),
            'parameter_stability': self._analyze_parameter_stability(
                [r.parameters for r in self.results_history]
            ),
            'performance_stability': self._analyze_performance_stability(
                [r.metrics for r in self.results_history]
            ),
            'convergence_rate': self._calculate_convergence_rate(),
            'optimization_statistics': self._calculate_optimization_statistics()
        }

    def _calculate_convergence_rate(self) -> float | np.ndarray[Any, np.dtype[np.floating[_64Bit] | np.float_]]:
        """Calculate convergence rate based on optimization history."""
        try:
            scores = [r.score for r in self.results_history]
            scores = np.array(scores, dtype=float)
            if len(scores) < 2:
                return 0.0
            slope = np.polyfit(range(len(scores)), scores, 1)
            return slope[0]  # Linear trend slope
        except Exception as e:
            logging.error(f"Error calculating convergence rate: {str(e)}")
            return 0.0

    def _calculate_optimization_statistics(self) -> Dict[str, Any]:
        """Calculate statistics about the optimization process."""
        try:
            scores = [r.score for r in self.results_history]
            return {
                'mean_score': np.mean(scores),
                'std_score': np.std(scores),
                'max_score': np.max(scores),
                'min_score': np.min(scores),
            }
        except Exception as e:
            logging.error(f"Error calculating optimization statistics: {str(e)}")
            return {}