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
        if len(returns) < 2 or returns.std() == 0: return 0.0
        return (returns.mean() / returns.std()) * np.sqrt(252)

    @staticmethod
    def _calculate_max_drawdown(equity_curve: pd.Series) -> float:
        peaks = equity_curve.expanding().max()
        drawdowns = (equity_curve - peaks) / peaks
        return abs(drawdowns.min())

class OptunaOptimizer:
    """
    Wrapper for Optuna-based Bayesian optimization of strategy parameters.
    """
    def __init__(self, strategy_class: Type[BaseStrategy], data: pd.DataFrame, param_space: Dict, historical_data: pd.DataFrame):
        self.strategy_class = strategy_class
        self.data = data # This is the specific fold data (e.g., train_data)
        self.param_space = param_space
        self.historical_data = historical_data # This is the full history for the backtester

    def _objective(self, trial: optuna.Trial) -> float:
        params = {}
        for name, space in self.param_space.items():
            if space['type'] == 'int':
                params[name] = trial.suggest_int(name, space['low'], space['high'])
            elif space['type'] == 'float':
                params[name] = trial.suggest_float(name, space['low'], space['high'])
        
        try:
            strategy = self.strategy_class(**params)
            # The backtester uses the full historical data up to the end of the test period for lookups
            # but the strategy's fit method will only use the training data fold
            backtester = Backtester(strategy, self.historical_data) 
            strategy.fit(self.data) # Fit strategy on the specific data fold
            results = backtester.run()
            
            sharpe = results['metrics'].get('Sharpe Ratio', -1.0)
            return sharpe if pd.notna(sharpe) and np.isfinite(sharpe) else -1.0
        except Exception as e:
            logger.warning(f"Backtest failed for params {params}: {e}")
            return -1.0

    def run_optimization(self, n_trials: int = 100) -> Dict:
        """Runs the optimization study."""
        study = optuna.create_study(direction='maximize')
        study.optimize(self._objective, n_trials=n_trials, show_progress_bar=True)
        
        logger.info(f"Best trial Sharpe Ratio: {study.best_trial.value:.4f}")
        logger.info(f"Best params: {study.best_params}")
        return study.best_params


class WalkForwardOptimizer(BaseStrategyEvaluator):
    """
    Implements Walk-Forward Optimization (WFO), a robust method for time-series backtesting.
    """
    def __init__(self, data: pd.DataFrame, train_size: int, test_size: int, step_size: int):
        self.data = data
        self.train_size = train_size
        self.test_size = test_size
        self.step_size = step_size
        self.results: List[Dict] = []

    def _generate_windows(self) -> List[Tuple[pd.Index, pd.Index]]:
        """Generates walk-forward windows for training and testing."""
        windows = []
        total_size = len(self.data)
        start = 0
        while start + self.train_size + self.test_size <= total_size:
            train_end = start + self.train_size
            test_end = train_end + self.test_size
            windows.append((
                self.data.index[start:train_end],
                self.data.index[train_end:test_end]
            ))
            start += self.step_size
        return windows

    def run(self, strategy_class: Type[BaseStrategy], param_space: Dict, n_trials_per_fold: int = 50) -> Dict:
        """
        Runs the full walk-forward optimization process.
        """
        windows = self._generate_windows()
        self.results = []
        
        for i, (train_idx, test_idx) in enumerate(tqdm(windows, desc="Walk-Forward Folds")):
            logger.info(f"Running WFO Fold {i+1}/{len(windows)}...")
            train_data = self.data.loc[train_idx]
            test_data_for_backtest = self.data.loc[test_idx]

            optimizer = OptunaOptimizer(strategy_class, train_data, param_space, self.data)
            best_params = optimizer.run_optimization(n_trials=n_trials_per_fold)

            strategy = strategy_class(**best_params)
            backtester = Backtester(strategy, test_data_for_backtest)
            test_results = backtester.run()

            self.results.append({
                'fold': i + 1,
                'train_start': train_idx[0],
                'train_end': train_idx[-1],
                'test_start': test_idx[0],
                'test_end': test_idx[-1],
                'best_params': best_params,
                'test_metrics': test_results['metrics']
            })
            
        return self.aggregate_results()

    def aggregate_results(self) -> Dict:
        """Aggregates and analyzes the results from all walk-forward folds."""
        if not self.results: return {}
        
        all_params = [res['best_params'] for res in self.results]
        all_metrics = [res['test_metrics'] for res in self.results]
        
        param_stability = {}
        if all_params:
            for key in all_params[0].keys():
                values = [p[key] for p in all_params]
                param_stability[key] = {'mean': np.mean(values), 'std': np.std(values)}

        performance_stability = {}
        if all_metrics:
            for key in all_metrics[0].keys():
                values = [m.get(key, 0) for m in all_metrics]
                performance_stability[key] = {'mean': np.mean(values), 'std': np.std(values)}
            
        return {
            'fold_results': self.results,
            'parameter_stability': param_stability,
            'performance_stability': performance_stability
        }

class ParameterSensitivityAnalyzer(BaseStrategyEvaluator):
    """
    Analyzes how sensitive the strategy's performance is to changes in its parameters.
    """
    def __init__(self, strategy_class: Type[BaseStrategy], historical_data: pd.DataFrame, base_params: Dict):
        self.strategy_class = strategy_class
        self.historical_data = historical_data
        self.base_params = base_params
        self.results: Dict[str, pd.DataFrame] = {}

    def run(self, param_to_test: str, test_range: np.ndarray) -> pd.DataFrame:
        performance_data = []
        for value in tqdm(test_range, desc=f"Analyzing {param_to_test}"):
            current_params = self.base_params.copy()
            current_params[param_to_test] = value
            try:
                strategy = self.strategy_class(**current_params)
                backtester = Backtester(strategy, self.historical_data)
                results = backtester.run()
                metrics = results['metrics']
                metrics['param_value'] = value
                performance_data.append(metrics)
            except Exception as e:
                logger.error(f"Backtest failed for {param_to_test}={value}: {e}")
                continue
        results_df = pd.DataFrame(performance_data)
        self.results[param_to_test] = results_df
        return results_df

    def plot_sensitivity(self, param_name: str, metric_name: str = 'Sharpe Ratio'):
        if param_name not in self.results:
            raise ValueError(f"Analysis for '{param_name}' has not been run yet.")
        results_df = self.results[param_name]
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=results_df['param_value'], y=results_df[metric_name], mode='lines+markers'))
        fig.update_layout(title=f"Sensitivity of {metric_name} to {param_name}", xaxis_title=param_name, yaxis_title=metric_name)
        fig.show()

class CrossValidatedOptimizer(BaseStrategyEvaluator):
    """
    Performs optimization using time-series cross-validation.
    """
    def __init__(self, data: pd.DataFrame, n_splits: int = 5):
        self.data = data
        self.n_splits = n_splits
        self.results: List[Dict] = []

    def run(self, strategy_class: Type[BaseStrategy], param_space: Dict, n_trials_per_fold: int = 20) -> Dict:
        tscv = TimeSeriesSplit(n_splits=self.n_splits)
        self.results = []
        
        for i, (train_idx, val_idx) in enumerate(tqdm(tscv.split(self.data), desc="Cross-Validation Folds", total=self.n_splits)):
            train_data = self.data.iloc[train_idx]
            val_data_for_backtest = self.data.iloc[val_idx]
            
            optimizer = OptunaOptimizer(strategy_class, train_data, param_space, self.data)
            best_params = optimizer.run_optimization(n_trials=n_trials_per_fold)
            
            strategy = strategy_class(**best_params)
            backtester = Backtester(strategy, val_data_for_backtest)
            val_results = backtester.run()
            
            self.results.append({
                'fold': i + 1,
                'best_params': best_params,
                'validation_metrics': val_results['metrics']
            })
        
        return self.aggregate_results()

    def aggregate_results(self) -> Dict:
        if not self.results: return {}
        all_metrics = [res['validation_metrics'] for res in self.results]
        performance_stability = {}
        if all_metrics:
            for key in all_metrics[0].keys():
                values = [m.get(key, 0) for m in all_metrics]
                performance_stability[key] = {'mean': np.mean(values), 'std': np.std(values)}
        return {'fold_results': self.results, 'performance_stability': performance_stability}

class MarketImpactModel:
    """
    Models the price impact of trades based on volatility and volume.
    """
    def __init__(self, historical_data: pd.DataFrame, impact_coefficient: float = 0.1):
        self.historical_data = historical_data
        self.impact_coefficient = impact_coefficient
        
        prices_pivot = historical_data.pivot(index='Date', columns='Symbol', values='Adj_Close')
        volume_pivot = historical_data.pivot(index='Date', columns='Symbol', values='Volume')

        self.adv = volume_pivot.rolling(window=21).mean()
        self.volatility = prices_pivot.pct_change().rolling(window=21).std() * np.sqrt(252)

    def calculate_market_impact(self, date: pd.Timestamp, symbol: str, trade_value: float) -> float:
        """
        Calculates the estimated cost of a trade due to market impact.
        """
        try:
            if date not in self.adv.index or date not in self.volatility.index:
                return 0.0
            
            adv = self.adv.loc[date, symbol]
            vol = self.volatility.loc[date, symbol]
            price = self.historical_data[(self.historical_data['Date'] == date) & (self.historical_data['Symbol'] == symbol)]['Adj_Close'].iloc[0]
            
            if pd.isna(adv) or pd.isna(vol) or adv == 0:
                return 0.0
            
            adv_value = adv * price
            price_impact_pct = self.impact_coefficient * vol * (abs(trade_value) / adv_value) if adv_value > 0 else 0
            
            return price_impact_pct * abs(trade_value)
        except (KeyError, IndexError):
            return 0.0

class TransactionCostOptimizer(BaseStrategyEvaluator):
    """
    An optimizer wrapper that incorporates a market impact model into the objective function.
    """
    def __init__(self, strategy_class: Type[BaseStrategy], historical_data: pd.DataFrame, param_space: Dict):
        self.strategy_class = strategy_class
        self.historical_data = historical_data
        self.param_space = param_space
        self.market_impact_model = MarketImpactModel(historical_data)

    def _objective(self, trial: optuna.Trial) -> float:
        params = {}
        for name, space in self.param_space.items():
            if space['type'] == 'int':
                params[name] = trial.suggest_int(name, space['low'], space['high'])
            elif space['type'] == 'float':
                params[name] = trial.suggest_float(name, space['low'], space['high'])
        
        try:
            strategy = self.strategy_class(**params)
            backtester = Backtester(strategy, self.historical_data)
            results = backtester.run()
            
            total_impact_cost = 0
            trades = pd.DataFrame(results['trades'])
            if not trades.empty:
                for _, trade in trades.iterrows():
                    trade_value = trade['price'] * trade['quantity']
                    impact = self.market_impact_model.calculate_market_impact(trade['date'], trade['symbol'], trade_value)
                    total_impact_cost += impact

            sharpe = results['metrics'].get('Sharpe Ratio', 0.0)
            initial_capital = backtester.initial_capital
            cost_penalty = total_impact_cost / initial_capital
            
            return (sharpe if pd.notna(sharpe) and np.isfinite(sharpe) else -1.0) - cost_penalty
        except Exception as e:
            logger.warning(f"Backtest with transaction cost failed for params {params}: {e}")
            return -2.0

    def run(self, n_trials: int = 100) -> Dict:
        """Runs the optimization study."""
        study = optuna.create_study(direction='maximize')
        study.optimize(self._objective, n_trials=n_trials, show_progress_bar=True)
        
        logger.info(f"Best trial (Cost-Adjusted Sharpe): {study.best_trial.value:.4f}")
        logger.info(f"Best params: {study.best_params}")
        return study.best_params