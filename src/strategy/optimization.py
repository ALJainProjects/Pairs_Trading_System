"""
Strategy Optimization Module with Real Backtesting

Implements Grid Search or Bayesian Optimization to tune strategy parameters
using actual backtest results.
"""

import pandas as pd
import numpy as np
from typing import Dict, Tuple, Callable, Optional
from src.strategy.base import BaseStrategy
from sklearn.model_selection import ParameterGrid
from src.strategy.backtest import MultiPairBacktester
import optuna
from config.logging_config import logger

class StrategyOptimizer:
    """Optimizes strategy parameters using real backtesting."""

    def __init__(
        self,
        strategy: BaseStrategy,
        returns: pd.DataFrame,
        objective_metric: Callable[[pd.Series, pd.DataFrame], float],
        parameter_grid: Dict,
        initial_capital: float = 100000,
        transaction_cost: float = 0.001,
        max_pairs: Optional[int] = None
    ):
        """
        Initialize the optimizer.

        Args:
            strategy: Strategy to optimize
            returns: Historical returns data
            objective_metric: Function(equity_curve, returns) -> float
            parameter_grid: Parameter names and ranges
            initial_capital: Starting capital for backtests
            transaction_cost: Transaction cost per trade
            max_pairs: Maximum concurrent pairs
        """
        self.strategy = strategy
        self.returns = returns
        self.objective_metric = objective_metric
        self.parameter_grid = parameter_grid
        self.initial_capital = initial_capital
        self.transaction_cost = transaction_cost
        self.max_pairs = max_pairs

        # Store optimization results
        self.optimization_results = []

    def _create_backtester(self) -> MultiPairBacktester:
        """Create a new backtester instance."""
        return MultiPairBacktester(
            strategy=self.strategy,
            returns=self.returns,
            initial_capital=self.initial_capital,
            transaction_cost=self.transaction_cost,
            max_pairs=self.max_pairs
        )

    def _evaluate_parameters(self, params: Dict) -> float:
        """
        Evaluate a parameter set using real backtesting.

        Args:
            params: Dictionary of parameter values

        Returns:
            float: Objective metric value
        """
        # Reset strategy and set parameters
        self.strategy.reset()
        for param, value in params.items():
            setattr(self.strategy, param, value)

        # Run backtest
        backtester = self._create_backtester()
        equity_curve = backtester.run_backtest()

        # Calculate objective metric
        score = self.objective_metric(equity_curve, self.returns)

        # Store results
        self.optimization_results.append({
            'parameters': params,
            'score': score,
            'equity_curve': equity_curve,
            'metrics': backtester.calculate_performance_metrics()
        })

        return score

    def grid_search_optimize(self) -> Tuple[Dict, float]:
        """
        Perform grid search optimization.

        Returns:
            Tuple of (best parameters, best score)
        """
        logger.info("Starting Grid Search optimization")

        grid = list(ParameterGrid(self.parameter_grid))
        best_score = -np.inf
        best_params = None

        for params in grid:
            logger.debug(f"Testing parameters: {params}")
            score = self._evaluate_parameters(params)

            if score > best_score:
                best_score = score
                best_params = params
                logger.info(f"New best score: {score:.4f} with params: {params}")

        return best_params, best_score

    def bayesian_optimize(self, n_trials: int = 100) -> Tuple[Dict, float]:
        """
        Perform Bayesian optimization.

        Args:
            n_trials: Number of optimization trials

        Returns:
            Tuple of (best parameters, best score)
        """
        logger.info("Starting Bayesian Optimization")

        def objective(trial):
            params = {}
            for param, values in self.parameter_grid.items():
                if isinstance(values, list):
                    params[param] = trial.suggest_categorical(param, values)
                elif isinstance(values, tuple) and len(values) == 2:
                    if isinstance(values[0], int):
                        params[param] = trial.suggest_int(
                            param, values[0], values[1]
                        )
                    else:
                        params[param] = trial.suggest_float(
                            param, values[0], values[1]
                        )

            return self._evaluate_parameters(params)

        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=n_trials)

        return study.best_params, study.best_value

    def plot_optimization_results(self) -> None:
        """Plot optimization results using plotly."""
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots

        # Create figure with secondary y-axis
        fig = make_subplots(specs=[[{"secondary_y": True}]])

        # Sort results by score
        sorted_results = sorted(
            self.optimization_results,
            key=lambda x: x['score']
        )

        # Plot scores
        scores = [r['score'] for r in sorted_results]
        fig.add_trace(
            go.Scatter(
                y=scores,
                mode='lines+markers',
                name='Objective Score'
            ),
            secondary_y=False
        )

        # Plot Sharpe ratios
        sharpes = [r['metrics']['Sharpe Ratio'] for r in sorted_results]
        fig.add_trace(
            go.Scatter(
                y=sharpes,
                mode='lines+markers',
                name='Sharpe Ratio'
            ),
            secondary_y=True
        )

        fig.update_layout(
            title="Optimization Results",
            xaxis_title="Trial Number",
            yaxis_title="Objective Score",
            yaxis2_title="Sharpe Ratio"
        )

        fig.show()

    def generate_optimization_report(self,
                                  output_file: Optional[str] = None) -> Dict:
        """
        Generate optimization report.

        Args:
            output_file: Optional Excel file path

        Returns:
            Dictionary with optimization results
        """
        # Convert results to DataFrame
        results_df = pd.DataFrame([
            {
                **r['parameters'],
                'Score': r['score'],
                'Sharpe': r['metrics']['Sharpe Ratio'],
                'Returns': r['metrics']['Total Return'],
                'Drawdown': r['metrics']['Max Drawdown']
            }
            for r in self.optimization_results
        ])

        # Sort by score
        results_df = results_df.sort_values('Score', ascending=False)

        # Save to Excel if requested
        if output_file:
            with pd.ExcelWriter(output_file) as writer:
                results_df.to_excel(
                    writer,
                    sheet_name='Optimization Results',
                    index=False
                )

                # Add best equity curve
                best_result = max(
                    self.optimization_results,
                    key=lambda x: x['score']
                )
                best_result['equity_curve'].to_frame('Value').to_excel(
                    writer,
                    sheet_name='Best Equity Curve'
                )

        return {
            'results': results_df,
            'best_parameters': results_df.iloc[0][self.parameter_grid.keys()].to_dict(),
            'best_score': results_df.iloc[0]['Score']
        }

def main():
    """Example usage."""
    # Define strategy and parameters
    strategy = BaseStrategy()  # Your strategy class
    returns = pd.DataFrame(...)  # Your returns data

    # Define parameter grid
    param_grid = {
        'lookback_period': [10, 20, 30],
        'zscore_threshold': (1.5, 3.0),
        'stop_loss': [0.02, 0.03, 0.04]
    }

    # Define objective function (e.g., Sharpe ratio)
    def objective(equity: pd.Series, returns: pd.DataFrame) -> float:
        ret = equity.pct_change().dropna()
        if len(ret) == 0 or ret.std() == 0:
            return -np.inf
        return ret.mean() / ret.std() * np.sqrt(252)

    # Initialize optimizer
    optimizer = StrategyOptimizer(
        strategy=strategy,
        returns=returns,
        objective_metric=objective,
        parameter_grid=param_grid,
        initial_capital=100000,
        max_pairs=5
    )

    # Run optimization
    best_params, best_score = optimizer.bayesian_optimize(n_trials=100)

    # Generate and save report
    report = optimizer.generate_optimization_report("optimization_results.xlsx")

    # Plot results
    optimizer.plot_optimization_results()

    return report, best_params, best_score

if __name__ == "__main__":
    main()