"""
Random baseline strategy implementation for benchmark comparison.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import random
from datetime import timedelta


class RandomBaselineStrategy:
    """
    Random baseline strategy that generates random buy/sell signals for pairs trading.
    Used as a benchmark to compare optimized strategies against.
    """

    def __init__(
        self,
        holding_period: int = 10,
        entry_probability: float = 0.1,
        exit_probability: float = 0.2,
        seed: Optional[int] = None
    ):
        """
        Initialize the random baseline strategy.

        Args:
            holding_period: Average holding period for positions (days)
            entry_probability: Probability of entering a position on any given day
            exit_probability: Probability of exiting a position once it's open
            seed: Random seed for reproducibility
        """
        self.holding_period = holding_period
        self.entry_probability = entry_probability
        self.exit_probability = exit_probability
        
        # Strategy state
        self.pairs = []
        self.positions = {}  # Stores active positions
        self.last_decision_date = {}  # Tracks last decision for each pair
        
        # Set random seed if provided
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
            
        # Stats for compatibility with optimizer
        self.model_confidence = 0.5  # Fixed confidence level
            
    def reset(self):
        """Reset the strategy state."""
        self.positions = {}
        self.last_decision_date = {}
    
    def set_pairs(self, pairs: List[Tuple[str, str]]):
        """Set the pairs to trade."""
        self.pairs = pairs
        self.reset()
        
    def get_default_parameters(self) -> Dict[str, float]:
        """Return default parameters for optimization framework compatibility."""
        return {
            'holding_period': 10,
            'entry_probability': 0.1,
            'exit_probability': 0.2
        }
        
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Generate random trading signals for each pair.
        
        Args:
            data: Price data with DatetimeIndex
            
        Returns:
            DataFrame with signals for each pair
        """
        # Initialize signals DataFrame with zeros
        signals = pd.DataFrame(0, index=data.index, columns=self.pairs)
        
        # For each date, generate random signals
        for date in data.index:
            for pair in self.pairs:
                # Get current position
                current_position = self.positions.get(pair, 0)
                
                # Check if we need to wait before next decision
                if pair in self.last_decision_date:
                    last_date = self.last_decision_date[pair]
                    if (date - last_date) < timedelta(days=1):
                        signals.loc[date, pair] = current_position
                        continue
                
                # Random signal generation
                if current_position == 0:
                    # No position, decide whether to enter
                    if random.random() < self.entry_probability:
                        # Enter randomly long (1) or short (-1)
                        new_position = random.choice([-1, 1])
                        self.positions[pair] = new_position
                        self.last_decision_date[pair] = date
                        signals.loc[date, pair] = new_position
                    else:
                        signals.loc[date, pair] = 0
                else:
                    # Have position, decide whether to exit
                    if random.random() < self.exit_probability:
                        self.positions[pair] = 0
                        self.last_decision_date[pair] = date
                        signals.loc[date, pair] = 0
                    else:
                        signals.loc[date, pair] = current_position
        
        return signals

    def evaluate(self, data: pd.DataFrame) -> float:
        """
        Evaluate strategy performance - used for optimizer compatibility.
        
        Returns:
            Random score between 0 and 1
        """
        return random.random()


class RandomPairTradingSystem:
    """Random version of the MultiPairTradingSystem for baseline comparison."""
    
    def __init__(
        self,
        pairs: List[Tuple[str, str]],
        prices: pd.DataFrame,
        initial_capital: float = 1000000,
        transaction_cost_bps: float = 1.0,
        seed: Optional[int] = None
    ):
        """
        Initialize the random pair trading system.
        
        Args:
            pairs: List of asset pairs to trade
            prices: Price data with DatetimeIndex
            initial_capital: Initial capital for the portfolio
            transaction_cost_bps: Transaction cost in basis points
            seed: Random seed for reproducibility
        """
        self.pairs = pairs
        self.prices = prices
        self.initial_capital = initial_capital
        self.transaction_cost_bps = transaction_cost_bps
        
        # Set seed if provided
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
            
        # Initialize portfolio tracking
        self.portfolio_history = []
        self.trade_history = []
        self.pair_models = {}
        
        # Generate random strategy for each pair
        for pair in pairs:
            self.pair_models[pair] = {
                'active': False,
                'positions': {pair[0]: 0, pair[1]: 0},
                'current_capital': initial_capital / len(pairs)
            }
    
    def run_backtest(self) -> None:
        """Run backtest with random trading decisions."""
        # Get all dates from price data
        dates = self.prices.index
        
        # Initialize portfolio value
        portfolio_value = self.initial_capital
        active_pairs = 0
        
        # Create strategy
        strategy = RandomBaselineStrategy()
        strategy.set_pairs(self.pairs)
        
        # Generate signals for entire period
        signals = strategy.generate_signals(self.prices)
        
        # Process signals daily
        for i, date in enumerate(dates):
            # Skip first day - need previous prices for returns
            if i == 0:
                self.portfolio_history.append({
                    'date': date,
                    'portfolio_value': portfolio_value,
                    'active_pairs': 0
                })
                continue
                
            # Process trades based on signals
            daily_pnl = 0
            active_count = 0
            
            for pair in self.pairs:
                signal = signals.loc[date, pair]
                prev_signal = signals.loc[dates[i-1], pair] if i > 0 else 0
                
                # Check for signal change (trade)
                if signal != prev_signal:
                    # Calculate random P&L from previous position
                    if prev_signal != 0:
                        # Random P&L between -3% and +5%
                        pair_return = (random.random() * 0.08) - 0.03
                        pair_capital = self.pair_models[pair]['current_capital']
                        trade_pnl = pair_capital * pair_return
                        daily_pnl += trade_pnl
                        
                        # Record trade
                        self.trade_history.append({
                            'date': date,
                            'pair': pair,
                            'type': 'exit',
                            'pnl': trade_pnl
                        })
                        
                        # Update pair capital
                        self.pair_models[pair]['current_capital'] += trade_pnl
                        self.pair_models[pair]['active'] = False
                    
                    # Enter new position
                    if signal != 0:
                        # Trade cost for entry
                        pair_capital = self.pair_models[pair]['current_capital']
                        trade_cost = pair_capital * (self.transaction_cost_bps / 10000)
                        daily_pnl -= trade_cost
                        
                        # Record trade
                        self.trade_history.append({
                            'date': date,
                            'pair': pair,
                            'type': 'entry',
                            'direction': 'long' if signal > 0 else 'short',
                            'cost': trade_cost
                        })
                        
                        # Update pair state
                        self.pair_models[pair]['active'] = True
                
                # Update active count
                if self.pair_models[pair]['active']:
                    active_count += 1
            
            # Update portfolio value
            portfolio_value += daily_pnl
            
            # Record portfolio state
            self.portfolio_history.append({
                'date': date,
                'portfolio_value': portfolio_value,
                'active_pairs': active_count,
                'daily_pnl': daily_pnl
            })
    
    def get_portfolio_metrics(self) -> Dict[str, Any]:
        """Calculate portfolio performance metrics."""
        # Convert portfolio history to DataFrame
        if not self.portfolio_history:
            return {'error': 'No portfolio history available'}
            
        portfolio_df = pd.DataFrame(self.portfolio_history)
        portfolio_df.set_index('date', inplace=True)
        
        # Calculate returns and metrics
        returns = portfolio_df['portfolio_value'].pct_change().dropna()
        
        total_return = (portfolio_df['portfolio_value'].iloc[-1] / self.initial_capital - 1) * 100
        
        # Calculate volatility and Sharpe ratio
        annual_vol = returns.std() * np.sqrt(252) * 100
        sharpe_ratio = 0
        if annual_vol > 0:
            sharpe_ratio = (returns.mean() * np.sqrt(252)) / returns.std()
            
        # Calculate drawdown
        peak = portfolio_df['portfolio_value'].cummax()
        drawdown = ((peak - portfolio_df['portfolio_value']) / peak) * 100
        max_dd = drawdown.max() if not drawdown.empty else 0
            
        return {
            'Portfolio Metrics': {
                'Initial Capital': self.initial_capital,
                'Final Portfolio Value': portfolio_df['portfolio_value'].iloc[-1],
                'Total Return (%)': total_return,
                'Annual Volatility (%)': annual_vol,
                'Sharpe Ratio': sharpe_ratio,
                'Max Drawdown (%)': max_dd
            },
            'Trading Activity': {
                'Total Number of Trades': len(self.trade_history),
                'Average Active Pairs': portfolio_df['active_pairs'].mean()
            }
        }
        
    def plot_portfolio_overview(self) -> Any:
        """Placeholder to maintain compatibility with MultiPairTradingSystem."""
        # Will be implemented in visualization code
        pass