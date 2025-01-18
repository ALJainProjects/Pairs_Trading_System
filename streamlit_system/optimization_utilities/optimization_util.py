"""
Utility functions and model classes for optimization component.
"""

import logging
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List

import numpy as np
import pandas as pd
from datetime import datetime


@dataclass
class ParameterSet:
    """Parameter set for optimization."""
    values: Dict[str, Any]
    bounds: Dict[str, tuple]
    parameter_types: Dict[str, str] = field(default_factory=dict)

    def validate(self) -> bool:
        """Validate parameter values against bounds."""
        try:
            for param, value in self.values.items():
                if param not in self.bounds:
                    raise ValueError(f"No bounds defined for parameter {param}")

                lower, upper = self.bounds[param][:2]
                if not lower <= value <= upper:
                    raise ValueError(
                        f"Parameter {param} value {value} outside bounds [{lower}, {upper}]"
                    )
            return True

        except Exception as e:
            logging.error(f"Parameter validation error: {str(e)}")
            return False


@dataclass
class OptimizationResult:
    """Optimization result container."""
    parameters: Dict[str, Any]
    score: float
    trial_number: int
    timestamp: datetime = field(default_factory=datetime.now)
    additional_metrics: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            'parameters': self.parameters,
            'score': self.score,
            'trial_number': self.trial_number,
            'timestamp': self.timestamp.isoformat(),
            'additional_metrics': self.additional_metrics
        }


class OptimizationConfig:
    """Configuration management for optimization."""

    def __init__(self, config_path: Optional[str] = None):
        """Initialize configuration."""
        self.config_path = Path(config_path) if config_path else None
        self.config = self._load_default_config()

        if self.config_path and self.config_path.exists():
            self._load_config()

    def _load_default_config(self) -> Dict:
        """Load default configuration."""
        return {
            'optimization': {
                'max_trials': 500,
                'default_trials': 100,
                'early_stopping': True,
                'early_stopping_patience': 20,
                'parallel_execution': True,
                'max_workers': 4
            },
            'parameters': {
                'num_grid_points': 10,
                'log_scale_threshold': 100
            },
            'metrics': {
                'primary': 'sharpe_ratio',
                'available': [
                    'sharpe_ratio',
                    'sortino_ratio',
                    'total_return',
                    'max_drawdown'
                ]
            },
            'storage': {
                'save_results': True,
                'output_dir': 'optimization_results',
                'max_saved_runs': 100
            },
            'visualization': {
                'plot_history': True,
                'plot_parameter_importance': True,
                'plot_correlation_matrix': True
            }
        }

    def _load_config(self) -> None:
        """Load configuration from file."""
        try:
            import yaml
            with open(self.config_path, 'r') as f:
                file_config = yaml.safe_load(f)

            # Update config recursively
            self._update_dict_recursive(self.config, file_config)

        except Exception as e:
            logging.error(f"Error loading config: {str(e)}")

    def _update_dict_recursive(self, base_dict: Dict, update_dict: Dict) -> None:
        """Update dictionary recursively."""
        for key, value in update_dict.items():
            if (
                    key in base_dict and
                    isinstance(base_dict[key], dict) and
                    isinstance(value, dict)
            ):
                self._update_dict_recursive(base_dict[key], value)
            else:
                base_dict[key] = value

    def save(self) -> None:
        """Save configuration to file."""
        if not self.config_path:
            return

        try:
            import yaml
            with open(self.config_path, 'w') as f:
                yaml.dump(self.config, f, default_flow_style=False)

        except Exception as e:
            logging.error(f"Error saving config: {str(e)}")

    def get(self, *keys: str, default: Any = None) -> Any:
        """Get configuration value."""
        current = self.config
        try:
            for key in keys:
                current = current[key]
            return current
        except (KeyError, TypeError):
            return default


def setup_logger() -> logging.Logger:
    """Setup logging configuration."""
    logger = logging.getLogger('optimization')
    logger.setLevel(logging.INFO)

    # Create handlers
    c_handler = logging.StreamHandler()
    f_handler = logging.FileHandler('optimization.log')
    c_handler.setLevel(logging.INFO)
    f_handler.setLevel(logging.DEBUG)

    # Create formatters and add to handlers
    c_format = logging.Formatter('%(levelname)s - %(message)s')
    f_format = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    c_handler.setFormatter(c_format)
    f_handler.setFormatter(f_format)

    # Add handlers to the logger
    logger.addHandler(c_handler)
    logger.addHandler(f_handler)

    return logger


def load_strategy(strategy_name: str) -> Any:
    """Load strategy class based on name."""
    try:
        if strategy_name == 'StatisticalStrategy':
            from src.strategy.pairs_strategy_SL import EnhancedStatPairsStrategy
            return EnhancedStatPairsStrategy()
        elif strategy_name == 'MLStrategy':
            from src.strategy.pairs_strategy_ML import MLPairsStrategy
            return MLPairsStrategy()
        elif strategy_name == 'DLStrategy':
            from src.strategy.pairs_strategy_DL import PairsTradingDL
            return PairsTradingDL()
        else:
            raise ValueError(f"Unknown strategy type: {strategy_name}")

    except Exception as e:
        logging.error(f"Error loading strategy: {str(e)}")
        raise


def calculate_metrics(equity_curve: pd.Series) -> Dict[str, float]:
    """Calculate comprehensive performance metrics."""
    try:
        returns = equity_curve.pct_change().dropna()

        if len(returns) < 2:
            return {
                'sharpe_ratio': float('-inf'),
                'sortino_ratio': float('-inf'),
                'total_return': -1.0,
                'max_drawdown': 1.0,
                'volatility': float('inf'),
                'win_rate': 0.0
            }

        # Basic metrics
        total_return = (equity_curve.iloc[-1] / equity_curve.iloc[0]) - 1
        volatility = returns.std() * np.sqrt(252)

        # Sharpe ratio
        risk_free_rate = 0.02  # Configurable
        excess_returns = returns - risk_free_rate / 252
        sharpe_ratio = np.sqrt(252) * (excess_returns.mean() / returns.std()) if returns.std() != 0 else 0.0

        # Sortino ratio
        downside_returns = returns[returns < 0]
        downside_std = np.sqrt(np.mean(downside_returns ** 2))
        sortino_ratio = np.sqrt(252) * (returns.mean() / downside_std) if downside_std != 0 else 0.0

        # Maximum drawdown
        rolling_max = equity_curve.expanding().max()
        drawdowns = (equity_curve - rolling_max) / rolling_max
        max_drawdown = abs(drawdowns.min())

        # Win rate
        win_rate = len(returns[returns > 0]) / len(returns)

        # Calmar ratio
        calmar_ratio = abs(total_return / max_drawdown) if max_drawdown != 0 else 0.0

        return {
            'total_return': total_return,
            'sharpe_ratio': sharpe_ratio,
            'sortino_ratio': sortino_ratio,
            'max_drawdown': max_drawdown,
            'volatility': volatility,
            'win_rate': win_rate,
            'calmar_ratio': calmar_ratio
        }

    except Exception as e:
        logging.error(f"Error calculating metrics: {str(e)}")
        return {}


def validate_optimization_inputs(
        strategy: Any,
        data: pd.DataFrame,
        param_ranges: Dict[str, tuple]
) -> bool:
    """Validate optimization inputs."""
    try:
        # Validate strategy
        if not hasattr(strategy, 'reset') or not callable(getattr(strategy, 'reset')):
            raise ValueError("Strategy must have a reset() method")

        # Validate data
        if not isinstance(data, pd.DataFrame):
            raise ValueError("Data must be a pandas DataFrame")
        if data.empty:
            raise ValueError("Data cannot be empty")
        if not isinstance(data.index, pd.DatetimeIndex):
            raise ValueError("Data must have a datetime index")

        # Validate parameter ranges
        for param, range_vals in param_ranges.items():
            if not isinstance(range_vals, tuple):
                raise ValueError(f"Range for parameter {param} must be a tuple")
            if len(range_vals) not in (2, 3):
                raise ValueError(f"Invalid range format for parameter {param}")
            if len(range_vals) == 3 and range_vals[2] != 'log':
                raise ValueError(f"Invalid range specification for parameter {param}")

        return True

    except Exception as e:
        logging.error(f"Input validation error: {str(e)}")
        return False


class ResultsManager:
    """Manage optimization results storage and retrieval."""

    def __init__(self, base_dir: str = 'optimization_results'):
        """Initialize results manager."""
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def save_results(self, results: Dict[str, Any], run_id: Optional[str] = None) -> str:
        """Save optimization results."""
        try:
            if run_id is None:
                run_id = datetime.now().strftime('%Y%m%d_%H%M%S')

            # Create run directory
            run_dir = self.base_dir / run_id
            run_dir.mkdir(exist_ok=True)

            # Save main results
            with open(run_dir / 'results.json', 'w') as f:
                import json
                json.dump(results, f, indent=4, default=str)

            # Save additional analysis files
            if 'equity_curve' in results:
                results['equity_curve'].to_csv(run_dir / 'equity_curve.csv')

            if 'parameter_importance' in results:
                pd.DataFrame(results['parameter_importance'], index=[0]).to_csv(
                    run_dir / 'parameter_importance.csv'
                )

            return run_id

        except Exception as e:
            logging.error(f"Error saving results: {str(e)}")
            return ""

    def load_results(self, run_id: str) -> Optional[Dict[str, Any]]:
        """Load optimization results."""
        try:
            run_dir = self.base_dir / run_id
            if not run_dir.exists():
                raise ValueError(f"Run {run_id} not found")

            # Load main results
            with open(run_dir / 'results.json', 'r') as f:
                import json
                results = json.load(f)

            # Load additional files if they exist
            if (run_dir / 'equity_curve.csv').exists():
                results['equity_curve'] = pd.read_csv(
                    run_dir / 'equity_curve.csv',
                    index_col=0,
                    parse_dates=True
                )

            if (run_dir / 'parameter_importance.csv').exists():
                results['parameter_importance'] = pd.read_csv(
                    run_dir / 'parameter_importance.csv',
                    index_col=0
                ).to_dict('records')[0]

            return results

        except Exception as e:
            logging.error(f"Error loading results: {str(e)}")
            return None

    def list_runs(self) -> List[str]:
        """List all optimization runs."""
        try:
            return [d.name for d in self.base_dir.iterdir() if d.is_dir()]
        except Exception as e:
            logging.error(f"Error listing runs: {str(e)}")
            return []

    def delete_run(self, run_id: str) -> bool:
        """Delete an optimization run."""
        try:
            run_dir = self.base_dir / run_id
            if run_dir.exists():
                import shutil
                shutil.rmtree(run_dir)
                return True
            return False
        except Exception as e:
            logging.error(f"Error deleting run: {str(e)}")
            return False


def optimize_hyperparameters(strategy: Any, param_ranges: Dict) -> Dict:
    """Optimize strategy hyperparameters using Optuna."""
    try:
        import optuna

        def objective(trial):
            params = {}
            for param, range_vals in param_ranges.items():
                if len(range_vals) == 3 and range_vals[2] == 'log':
                    params[param] = trial.suggest_float(
                        param, range_vals[0], range_vals[1], log=True
                    )
                elif isinstance(range_vals[0], int):
                    params[param] = trial.suggest_int(
                        param, range_vals[0], range_vals[1]
                    )
                else:
                    params[param] = trial.suggest_float(
                        param, range_vals[0], range_vals[1]
                    )

            strategy.reset()
            for param, value in params.items():
                setattr(strategy, param, value)

            # Run strategy evaluation
            try:
                score = strategy.evaluate()
                return score
            except Exception as e:
                logging.error(f"Error evaluating parameters: {str(e)}")
                return float('-inf')

        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=100)

        return {
            'best_params': study.best_params,
            'best_value': study.best_value
        }

    except Exception as e:
        logging.error(f"Hyperparameter optimization error: {str(e)}")
        return {}