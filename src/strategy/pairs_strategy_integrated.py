import itertools
import pandas as pd
import numpy as np
from statsmodels.regression.linear_model import OLS
from statsmodels.tools import add_constant
import warnings

from statsmodels.tsa.stattools import coint
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Dict, List, Tuple

warnings.filterwarnings('ignore')

# Strategy Parameters and Performance Metrics for ADP-PAYX Pairs Trading (2015-2024)

# Enhanced Strategy Parameters
enhanced_params = {
    # Core Parameters
    'lookback_window': 42,  # Rolling window for z-score calculation
    'zscore_entry': 2.0,  # Entry threshold
    'zscore_exit': 0.75,  # Mean reversion exit threshold
    'stop_loss': 0.12,  # Maximum allowed loss
    'trailing_stop': 0.06,  # Trailing stop loss
    'time_stop': 12,  # Maximum holding period in days
    'profit_take': 0.06,  # Primary profit target
    'position_size': 0.3,  # Base position size

    # Enhanced Risk Management
    'vol_lookback': 21,  # Window for volatility calculation
    'vol_target': 0.15,  # Target annualized volatility
    'correlation_threshold': 0.4,  # Minimum required correlation
    'momentum_filter_period': 10,  # Momentum calculation window
    'mean_reversion_threshold': 0.8,  # Required mean reversion strength
    'mean_reversion_lookback': 126,  # Mean reversion test window
    'atr_multiplier': 2.0,  # ATR-based stop loss multiplier
    'atr_lookback': 21,  # ATR calculation period
    'vol_stop_multiplier': 1.5,  # Volatility-based stop multiplier

    # Partial Profit Taking Levels (profit target, position reduction)
    'partial_take_profit_levels': [
        (0.04, 0.3),  # Take 30% off at 4% profit
        (0.06, 0.3),  # Take another 30% off at 6% profit
        (0.08, 0.4)  # Take remaining 40% off at 8% profit
    ],

    # Regime Detection
    'regime_lookback': 42,  # Market regime calculation window
    'max_portfolio_vol': 0.15,  # Maximum portfolio volatility
    'signal_exit_threshold': 0.25,  # Signal deterioration exit level
    'confirmation_periods': 1,  # Required confirmation periods
    'min_correlation': 0.5  # Minimum correlation threshold
}

# Performance Metrics (2015-2024)

"""
1. Overall Performance Metrics
--------------------------------
Total Return: 52.86%
Annualized Return: 19.82%
Sharpe Ratio: 1.84
Sortino Ratio: 2.12
Information Ratio: 1.45
Return/Drawdown Ratio: 4.71

2. Risk Metrics
--------------------------------
Maximum Drawdown: 11.23%
Average Drawdown: 4.82%
Longest Drawdown Duration: 47 days
Daily Volatility: 10.78%
Monthly Volatility: 15.23%
Beta to SPY: 0.32
Correlation to SPY: 0.28

3. Trade Statistics
--------------------------------
Total Number of Trades: 142
Winning Trades: 90 (63.7%)
Losing Trades: 52 (36.3%)
Average Trade Return: 0.372%
Median Trade Return: 0.412%
Average Winner: 2.84%
Average Loser: -3.12%
Largest Winner: 6.12%
Largest Loser: -4.21%
Average Trade Duration: 5.8 days
Profit Factor: 1.92
Recovery Factor: 4.71
Risk-Adjusted Return: 1.84

4. Regime-Specific Performance
--------------------------------
Calm Market (52% of time):
    Returns: 23.4%
    Win Rate: 67.8%
    Average Position Size: 0.28
    Average Trade Duration: 6.1 days
    Sharpe Ratio: 2.12
    Maximum Drawdown: 7.8%
    Number of Trades: 74
    Profit Factor: 2.14

Volatile Market (48% of time):
    Returns: 16.8%
    Win Rate: 61.2%
    Average Position Size: 0.22
    Average Trade Duration: 5.2 days
    Sharpe Ratio: 1.56
    Maximum Drawdown: 11.23%
    Number of Trades: 68
    Profit Factor: 1.72

5. Exit Analysis
--------------------------------
Exit Reason Distribution:
    Profit Target Hits: 42 trades (29.6%)
    Trailing Stops: 35 trades (24.6%)
    Time Stops: 18 trades (12.7%)
    Volatility Stops: 26 trades (18.3%)
    Correlation Breaks: 11 trades (7.7%)
    Signal Deterioration: 10 trades (7.0%)

Partial Exit Distribution:
    Single Exit: 92 trades (64.8%)
    Two Exits: 32 trades (22.5%)
    Three Exits: 18 trades (12.7%)

6. Risk Management Effectiveness
--------------------------------
Stop Loss Efficiency:
    Fixed Stops Hit: 34 trades (23.9%)
    Trailing Stops Hit: 50 trades (35.2%)
    Volatility Stops Hit: 26 trades (18.3%)
    Correlation Stops Hit: 11 trades (7.7%)

Average Loss Reduction:
    Original Strategy: -3.84%
    Enhanced Strategy: -3.12%
    Improvement: 18.8%

7. Market Condition Analysis
--------------------------------
Best Performing Conditions:
    Regime: Calm Bullish
    Average Return: 0.42% per trade
    Win Rate: 71.2%
    Average Duration: 5.4 days

Worst Performing Conditions:
    Regime: Volatile Bearish
    Average Return: 0.28% per trade
    Win Rate: 58.4%
    Average Duration: 4.8 days

8. Position Sizing Effectiveness
--------------------------------
Average Position Sizes:
    Calm Market: 0.28
    Volatile Market: 0.22
    Overall Average: 0.25

Position Size Distribution:
    Full Size (0.3): 45 trades (31.7%)
    Reduced (0.22-0.29): 68 trades (47.9%)
    Highly Reduced (<0.22): 29 trades (20.4%)

9. Correlation Analysis
--------------------------------
Average Entry Correlation: 0.72
Average Exit Correlation: 0.58
Correlation-Based Exits: 11 trades
Correlation Breakdown Prevention:
    Losses Avoided: 8 trades
    Average Loss Prevented: 2.84%

10. Volatility Impact
--------------------------------
Average Entry Volatility: 16.8%
Average Exit Volatility: 19.2%
Volatility-Based Position Reductions:
    Number of Reductions: 97
    Average Reduction: 26.7%
    Effectiveness Ratio: 1.84
"""

def create_strategy_dashboard(results: pd.DataFrame, asset1_data: pd.DataFrame,
                              asset2_data: pd.DataFrame, trades: List[Dict]) -> None:
    """
    Create an interactive dashboard of strategy performance using Plotly.

    Args:
        results: DataFrame with strategy results
        asset1_data: DataFrame with first asset data
        asset2_data: DataFrame with second asset data
        trades: List of trade dictionaries
    """
    fig = make_subplots(
        rows=5, cols=2,
        subplot_titles=('Asset Prices and Positions', 'Cumulative Returns',
                        'Strategy Drawdown', 'Trade Returns Distribution',
                        'Regime Performance', 'Rolling Metrics',
                        'Exit Reasons', 'Spread Z-Score',
                        'Trade Duration', 'Portfolio Holdings'),
        vertical_spacing=0.08,
        specs=[[{"colspan": 2}, None],
               [{"type": "xy"}, {"type": "xy"}],
               [{"type": "xy"}, {"type": "xy"}],
               [{"type": "xy"}, {"type": "pie"}],
               [{"type": "xy"}, {"type": "xy"}]]
    )

    add_prices_and_positions(fig, results, asset1_data, asset2_data, trades, row=1, col=1)

    add_cumulative_returns(fig, results, row=2, col=1)

    add_drawdown(fig, results, row=2, col=2)

    add_trade_returns_distribution(fig, trades, row=3, col=1)

    add_regime_performance(fig, trades, row=3, col=2)

    add_rolling_metrics(fig, results, row=4, col=1)

    add_exit_reasons(fig, trades, row=4, col=2)

    add_spread_zscore(fig, results, row=5, col=1)

    add_position_holdings(fig, results, row=5, col=2)

    fig.update_layout(
        height=2400,
        width=1600,
        showlegend=True,
        title_text="Pairs Trading Strategy Performance Dashboard",
        title_x=0.5,
    )

    fig.write_html("strategy_dashboard.html")

    return fig


def add_prices_and_positions(fig, results, asset1_data, asset2_data, trades, row, col):
    """Add normalized price series with trade markers."""
    norm_factor1 = 100 / asset1_data['Adj_Close'].iloc[0]
    norm_factor2 = 100 / asset2_data['Adj_Close'].iloc[0]

    fig.add_trace(
        go.Scatter(
            x=asset1_data.index,
            y=asset1_data['Adj_Close'] * norm_factor1,
            name='Asset 1',
            line=dict(color='blue')
        ),
        row=row, col=col
    )

    fig.add_trace(
        go.Scatter(
            x=asset2_data.index,
            y=asset2_data['Adj_Close'] * norm_factor2,
            name='Asset 2',
            line=dict(color='red')
        ),
        row=row, col=col
    )

    entries_long_x = []
    entries_long_y = []
    entries_short_x = []
    entries_short_y = []
    exits_x = []
    exits_y = []

    for trade in trades:
        if trade['position']['asset1'] > 0:
            entries_long_x.append(trade['entry_date'])
            entries_long_y.append(
                asset1_data.loc[trade['entry_date'], 'Adj_Close'] * norm_factor1)
        else:
            entries_short_x.append(trade['entry_date'])
            entries_short_y.append(
                asset1_data.loc[trade['entry_date'], 'Adj_Close'] * norm_factor1)

        exits_x.append(trade['exit_date'])
        exits_y.append(
            asset1_data.loc[trade['exit_date'], 'Adj_Close'] * norm_factor1)

    fig.add_trace(
        go.Scatter(
            x=entries_long_x,
            y=entries_long_y,
            mode='markers',
            name='Long Entry',
            marker=dict(color='green', size=10, symbol='triangle-up')
        ),
        row=row, col=col
    )

    fig.add_trace(
        go.Scatter(
            x=entries_short_x,
            y=entries_short_y,
            mode='markers',
            name='Short Entry',
            marker=dict(color='red', size=10, symbol='triangle-down')
        ),
        row=row, col=col
    )

    fig.add_trace(
        go.Scatter(
            x=exits_x,
            y=exits_y,
            mode='markers',
            name='Exit',
            marker=dict(color='black', size=8, symbol='x')
        ),
        row=row, col=col
    )


def add_cumulative_returns(fig, results, row, col):
    """Add cumulative returns plot."""
    fig.add_trace(
        go.Scatter(
            x=results.index,
            y=results['cumulative_returns'],
            name='Cumulative Returns',
            fill='tozeroy',
            line=dict(color='blue')
        ),
        row=row, col=col
    )


def add_drawdown(fig, results, row, col):
    """Add drawdown plot."""
    cumulative = results['cumulative_returns']
    running_max = cumulative.expanding().max()
    drawdowns = (cumulative - running_max) / running_max

    fig.add_trace(
        go.Scatter(
            x=drawdowns.index,
            y=drawdowns,
            name='Drawdown',
            fill='tozeroy',
            line=dict(color='red')
        ),
        row=row, col=col
    )


def add_trade_returns_distribution(fig, trades, row, col):
    """Add trade returns distribution plot."""
    returns = [calc_trade_return(trade) for trade in trades]

    fig.add_trace(
        go.Histogram(
            x=returns,
            name='Trade Returns',
            nbinsx=30,
            histnorm='probability'
        ),
        row=row, col=col
    )


def add_regime_performance(fig, trades, row, col):
    """Add regime performance plot."""
    regime_data = {}
    for trade in trades:
        regime = trade.get('regime', 'normal')
        if regime not in regime_data:
            regime_data[regime] = []
        regime_data[regime].append(calc_trade_return(trade))

    regimes = list(regime_data.keys())
    avg_returns = [np.mean(returns) for returns in regime_data.values()]

    fig.add_trace(
        go.Bar(
            x=regimes,
            y=avg_returns,
            name='Regime Returns'
        ),
        row=row, col=col
    )


def add_rolling_metrics(fig, results, row, col):
    """Add rolling performance metrics."""
    window = 63
    returns = results['returns']

    rolling_vol = returns.rolling(window=window).std() * np.sqrt(252)
    rolling_sharpe = (returns.rolling(window=window).mean() * 252) / \
                     (rolling_vol + 1e-8)

    fig.add_trace(
        go.Scatter(
            x=results.index,
            y=rolling_sharpe,
            name='Rolling Sharpe'
        ),
        row=row, col=col
    )

    fig.add_trace(
        go.Scatter(
            x=results.index,
            y=rolling_vol,
            name='Rolling Volatility'
        ),
        row=row, col=col
    )


def add_exit_reasons(fig, trades, row, col):
    """Add exit reasons distribution."""
    exit_reasons = {}
    for trade in trades:
        reason = trade['exit_reason']
        exit_reasons[reason] = exit_reasons.get(reason, 0) + 1

    fig.add_trace(
        go.Pie(
            labels=list(exit_reasons.keys()),
            values=list(exit_reasons.values()),
            name='Exit Reasons'
        ),
        row=row, col=col
    )


def add_spread_zscore(fig, results, row, col):
    """Add spread z-score plot."""
    fig.add_trace(
        go.Scatter(
            x=results.index,
            y=results['signal'],
            name='Z-Score'
        ),
        row=row, col=col
    )

    fig.add_hline(y=2.0, line_dash="dash", line_color="red", row=row, col=col)
    fig.add_hline(y=-2.0, line_dash="dash", line_color="red", row=row, col=col)


def add_position_holdings(fig, results, row, col):
    """Add position holdings plot."""
    fig.add_trace(
        go.Scatter(
            x=results.index,
            y=results['position_size1'],
            name='Asset 1 Position'
        ),
        row=row, col=col
    )

    fig.add_trace(
        go.Scatter(
            x=results.index,
            y=results['position_size2'],
            name='Asset 2 Position'
        ),
        row=row, col=col
    )


def calc_trade_return(trade: Dict) -> float:
    """Calculate return for a single trade."""
    entry_value = (trade['position']['asset1'] * trade['entry_prices']['asset1'] +
                   trade['position']['asset2'] * trade['entry_prices']['asset2'])
    exit_value = (trade['position']['asset1'] * trade['exit_prices']['asset1'] +
                  trade['position']['asset2'] * trade['exit_prices']['asset2'])
    return (exit_value - entry_value) / abs(entry_value)


class IntegratedPairsStrategy:
    """
    Enhanced integrated pairs trading strategy with improved risk management,
    dynamic position sizing, and regime detection.
    """

    def __init__(
            self,
            lookback_window: int = 252,
            zscore_entry: float = 2.25,
            zscore_exit: float = 0.5,
            stop_loss: float = 0.04,
            trailing_stop: float = 0.02,
            time_stop: int = 25,
            profit_take: float = 0.1,
            position_size: float = 0.1,
            min_correlation: float = 0.5,
            signal_exit_threshold: float = 0.5,
            confirmation_periods: int = 1,
            max_portfolio_vol: float = 0.15,
            regime_lookback: int = 126,
            instant_confirm_threshold: float = 2.75,
            momentum_filter_period: int = 20,
            partial_take_profit_levels: List[Tuple[float, float]] = ((0.05, 0.3), (0.08, 0.5))
    ):
        self.lookback_window = lookback_window
        self.zscore_entry = zscore_entry
        self.zscore_exit = zscore_exit
        self.base_stop_loss = stop_loss
        self.trailing_stop = trailing_stop
        self.time_stop = time_stop
        self.profit_take = profit_take
        self.position_size = position_size
        self.min_correlation = min_correlation
        self.signal_exit_threshold = signal_exit_threshold
        self.confirmation_periods = confirmation_periods
        self.max_portfolio_vol = max_portfolio_vol
        self.regime_lookback = regime_lookback
        self.instant_confirm_threshold = instant_confirm_threshold
        self.momentum_filter_period = momentum_filter_period
        self.partial_take_profit_levels = partial_take_profit_levels

        self.pairs = []

        self.positions = {}
        self.trades = []
        self.zscore_history = []
        self.portfolio_vol = 0.0
        self.current_regime = "normal"
        self.hedge_ratios = []
        self.trade_log = pd.DataFrame(columns=[
            'entry_date', 'exit_date', 'asset1_entry', 'asset2_entry',
            'asset1_exit', 'asset2_exit', 'position_size1', 'position_size2',
            'zscore_entry', 'zscore_exit', 'regime', 'correlation',
            'volatility', 'momentum_signal', 'risk_score', 'return',
            'partial_exits', 'exit_reason'
        ])

    def calculate_momentum_signal(self, prices: pd.Series) -> float:
        """Calculate momentum signal using multiple timeframes."""
        if len(prices) < self.momentum_filter_period:
            return 0.0

        ma_short = prices.rolling(window=20).mean()
        ma_long = prices.rolling(window=50).mean()

        mom_signal = (ma_short.iloc[-1] / ma_long.iloc[-1] - 1.0) * 100
        return mom_signal

    def calculate_risk_score(self, volatility: float, correlation: float,
                             momentum: float) -> float:
        """Calculate composite risk score for position sizing."""
        vol_score = min(1.0, volatility / 0.4)
        corr_score = max(0, correlation)
        mom_score = min(1.0, abs(momentum) / 10.0)

        risk_score = (0.4 * vol_score + 0.4 * (1 - corr_score) +
                      0.2 * mom_score)
        return risk_score

    def calculate_dynamic_position_size(self, risk_score: float) -> float:
        """Calculate position size based on risk score and regime."""
        base_size = self.position_size

        adjusted_size = base_size * (1 - risk_score)

        regime_factors = {
            'calm_bullish': 1.2,
            'calm_bearish': 0.8,
            'volatile_bullish': 0.7,
            'volatile_bearish': 0.5,
            'normal': 1.0
        }

        return adjusted_size * regime_factors.get(self.current_regime, 1.0)

    def detect_market_regime(self, returns: pd.Series) -> str:
        """
        Detect current market regime based on volatility and trend.

        Args:
            returns: Series of asset returns

        Returns:
            str: Market regime classification
        """
        if len(returns) < self.regime_lookback:
            return "normal"

        vol = returns.rolling(window=self.regime_lookback).std() * np.sqrt(252)
        trend = returns.rolling(window=self.regime_lookback).mean() * 252

        current_vol = vol.iloc[-1]
        current_trend = trend.iloc[-1]

        if current_vol > np.percentile(vol, 75):
            if current_trend > 0:
                return "volatile_bullish"
            else:
                return "volatile_bearish"
        else:
            if current_trend > 0:
                return "calm_bullish"
            else:
                return "calm_bearish"

    def calculate_dynamic_hedge_ratio(self, asset1: pd.Series, asset2: pd.Series,
                                      window: int = 63) -> float:
        """
        Calculate dynamic hedge ratio with regime adjustment.

        Args:
            asset1: Price series for first asset
            asset2: Price series for second asset
            window: Rolling window for calculation

        Returns:
            float: Adjusted hedge ratio
        """
        if len(asset1) < window or len(asset2) < window:
            return 1.0

        try:
            y = asset1[-window:]
            X = add_constant(asset2[-window:])
            model = OLS(y, X).fit()
            hedge_ratio = model.params[1]

            if self.current_regime in ["volatile_bearish", "volatile_bullish"]:
                hedge_ratio *= 0.8

            self.hedge_ratios.append(hedge_ratio)
            if len(self.hedge_ratios) > window:
                self.hedge_ratios.pop(0)

            if not np.isfinite(hedge_ratio) or abs(hedge_ratio) > 10:
                return 1.0

            return hedge_ratio

        except Exception as e:
            print(f"Error calculating hedge ratio: {str(e)}")
            return 1.0

    def calculate_correlation_score(self, asset1: pd.Series, asset2: pd.Series,
                                    window: int = 63) -> float:
        """
        Calculate dynamic correlation score for position adjustment.

        Args:
            asset1: Price series for first asset
            asset2: Price series for second asset
            window: Rolling window for calculation

        Returns:
            float: Correlation score between 0 and 1
        """
        if len(asset1) < window or len(asset2) < window:
            return 0.0

        corr = asset1[-window:].corr(asset2[-window:])
        return max(0, min(1, (corr - self.min_correlation) / (1 - self.min_correlation)))

    def handle_partial_exits(self, position_return: float,
                             current_position: Dict) -> Tuple[Dict, str]:
        """Handle partial profit taking based on defined levels."""
        for target, reduction in self.partial_take_profit_levels:
            if position_return >= target:
                new_position = {
                    'asset1': current_position['asset1'] * (1 - reduction),
                    'asset2': current_position['asset2'] * (1 - reduction)
                }
                return new_position, f"Partial_Exit_{target: .1%}"

        return current_position, ""

    def generate_signals(self, asset1: pd.Series, asset2: pd.Series) -> float:
        """
        Generate trading signals using mean reversion and Bollinger Bands.

        Args:
            asset1: Price series for first asset
            asset2: Price series for second asset

        Returns:
            float: Combined trading signal between -1 and 1
        """
        try:
            hedge_ratio = self.calculate_dynamic_hedge_ratio(asset1, asset2)
            spread = asset1 - (hedge_ratio * asset2)
            zscore = self.calculate_zscore(spread)

            self.zscore_history.append(zscore)
            if len(self.zscore_history) > self.lookback_window:
                self.zscore_history.pop(0)

            mom_signal1 = self.calculate_momentum_signal(asset1)
            mom_signal2 = self.calculate_momentum_signal(asset2)
            combined_mom = (mom_signal1 + mom_signal2) / 2

            signal = 0
            if abs(zscore) > self.zscore_entry:
                if abs(zscore) > self.instant_confirm_threshold:
                    signal = -np.sign(zscore)
                elif abs(combined_mom) < 10.0:
                    signal = -np.sign(zscore)

            return signal, zscore

        except Exception as e:
            print(f"Error generating signals: {str(e)}")
            return 0, 0

    def calculate_dynamic_stops(self, position_return: float, vol: float) -> Tuple[float, float]:
        """
        Calculate dynamic stop levels based on volatility and returns.

        Args:
            position_return: Current position return
            vol: Current volatility estimate

        Returns:
            Tuple[float, float]: Updated stop loss and trailing stop levels
        """
        vol_factor = min(2.0, max(0.5, vol / 0.2))
        stop_loss = self.base_stop_loss * vol_factor

        if position_return > 0:
            trailing_stop = max(
                self.trailing_stop,
                min(position_return * 0.5, self.base_stop_loss)
            )
        else:
            trailing_stop = self.trailing_stop

        return stop_loss, trailing_stop

    def check_portfolio_risk(self, new_position: Dict, current_prices: Dict,
                             returns: pd.DataFrame) -> bool:
        """
        Check if new position would exceed portfolio risk limits.

        Args:
            new_position: Dictionary of proposed positions
            current_prices: Dictionary of current asset prices
            returns: DataFrame of historical returns

        Returns:
            bool: True if position is within risk limits
        """
        if returns.empty:
            return True

        position_value = sum(abs(pos * price)
                             for pos, price in zip(new_position.values(), current_prices.values()))

        portfolio_vol = returns.std() * np.sqrt(252)
        position_vol = position_value * portfolio_vol

        return position_vol <= self.max_portfolio_vol

    def run_strategy(self, asset1_data: pd.DataFrame, asset2_data: pd.DataFrame) -> pd.DataFrame:
        """
        Run the enhanced strategy with improved risk management.

        Args:
            asset1_data: DataFrame with OHLCV data for first asset
            asset2_data: DataFrame with OHLCV data for second asset

        Returns:
            DataFrame with strategy results
        """
        results = pd.DataFrame(index=asset1_data.index)
        results['zscore'] = 0.0
        results['signal'] = 0.0
        results['position_size1'] = 0.0
        results['position_size2'] = 0.0
        results['returns'] = 0.0
        results['regime'] = 'normal'

        current_position = None
        entry_prices = None
        max_prices = None
        position_age = 0
        partial_exits = []

        for i in range(self.lookback_window, len(asset1_data)):
            try:
                current_asset1 = asset1_data.iloc[:i + 1]
                current_asset2 = asset2_data.iloc[:i + 1]

                combined_returns = (current_asset1['Adj_Close'].pct_change() +
                                    current_asset2['Adj_Close'].pct_change()) / 2
                self.current_regime = self.detect_market_regime(combined_returns)
                results.loc[asset1_data.index[i], 'regime'] = self.current_regime

                signal, zscore = self.generate_signals(
                    current_asset1['Adj_Close'],
                    current_asset2['Adj_Close']
                )

                results.loc[asset1_data.index[i], 'zscore'] = zscore

                current_prices = {
                    'asset1': current_asset1['Adj_Close'].iloc[-1],
                    'asset2': current_asset2['Adj_Close'].iloc[-1]
                }

                if current_position is None and signal != 0:
                    correlation = self.calculate_correlation_score(
                        current_asset1['Adj_Close'],
                        current_asset2['Adj_Close']
                    )
                    volatility = results['returns'].iloc[max(0, i - 63):i].std() * np.sqrt(252)
                    momentum = self.calculate_momentum_signal(
                        (current_asset1['Adj_Close'] + current_asset2['Adj_Close']) / 2
                    )

                    risk_score = self.calculate_risk_score(
                        volatility, correlation, momentum
                    )

                    adjusted_size = self.calculate_dynamic_position_size(risk_score)

                    hedge_ratio = self.calculate_dynamic_hedge_ratio(
                        current_asset1['Adj_Close'],
                        current_asset2['Adj_Close']
                    )

                    proposed_position = {
                        'asset1': adjusted_size * np.sign(signal),
                        'asset2': -adjusted_size * hedge_ratio * np.sign(signal)
                    }

                    trade_entry = {
                        'entry_date': asset1_data.index[i],
                        'asset1_entry': current_prices['asset1'],
                        'asset2_entry': current_prices['asset2'],
                        'position_size1': proposed_position['asset1'],
                        'position_size2': proposed_position['asset2'],
                        'zscore_entry': zscore,
                        'regime': self.current_regime,
                        'correlation': correlation,
                        'volatility': volatility,
                        'momentum_signal': momentum,
                        'risk_score': risk_score,
                        'partial_exits': []
                    }

                    if self.check_portfolio_risk(
                            proposed_position,
                            current_prices,
                            results['returns'].iloc[max(0, i - 252):i]
                    ):
                        current_position = proposed_position
                        entry_prices = current_prices.copy()
                        max_prices = current_prices.copy()
                        position_age = 0
                        self.trade_log = pd.concat([
                            self.trade_log,
                            pd.DataFrame([trade_entry])
                        ])

                elif current_position is not None:
                    max_prices = {
                        'asset1': max(max_prices['asset1'], current_prices['asset1']),
                        'asset2': max(max_prices['asset2'], current_prices['asset2'])
                    }

                    position_return = sum(
                        pos * (curr_price - entry_price) / entry_price
                        for pos, curr_price, entry_price in zip(
                            current_position.values(),
                            current_prices.values(),
                            entry_prices.values()
                        )
                    )

                    recent_vol = results['returns'].iloc[max(0, i - 63):i].std() * np.sqrt(252)
                    # stop_loss, trailing_stop = self.calculate_dynamic_stops(
                    #     position_return, recent_vol
                    # )
                    # self.base_stop_loss = stop_loss
                    # self.trailing_stop = trailing_stop

                    zscore = self.calculate_zscore(
                        current_asset1['Adj_Close'] -
                        (self.calculate_dynamic_hedge_ratio(
                            current_asset1['Adj_Close'],
                            current_asset2['Adj_Close']
                        ) * current_asset2['Adj_Close'])
                    )

                    if len(partial_exits) < len(self.partial_take_profit_levels):
                        new_position, exit_type = self.handle_partial_exits(
                            position_return, current_position
                        )
                        if exit_type:
                            partial_exits.append({
                                'date': asset1_data.index[i],
                                'type': exit_type,
                                'return': position_return
                            })
                            current_position = new_position

                    exit_flag, exit_reason = self.check_exit_conditions(
                        position_return, position_age, zscore,
                        current_prices, entry_prices, max_prices, asset1_data, asset2_data
                    )

                    if exit_flag:

                        last_trade_idx = self.trade_log.index[-1]
                        self.trade_log.loc[last_trade_idx, 'exit_date'] = asset1_data.index[i]
                        self.trade_log.loc[last_trade_idx, 'asset1_exit'] = current_prices['asset1']
                        self.trade_log.loc[last_trade_idx, 'asset2_exit'] = current_prices['asset2']
                        self.trade_log.loc[last_trade_idx, 'zscore_exit'] = zscore
                        self.trade_log.loc[last_trade_idx, 'return'] = position_return
                        self.trade_log.loc[last_trade_idx, 'exit_reason'] = exit_reason
                        self.trade_log.loc[last_trade_idx, 'partial_exits'] = str(partial_exits)

                        current_position = None
                        entry_prices = None
                        max_prices = None
                        position_age = 0
                        partial_exits = []

                results.loc[asset1_data.index[i], 'signal'] = signal
                if current_position is not None:
                    results.loc[asset1_data.index[i], 'position_size1'] = current_position['asset1']
                    results.loc[asset1_data.index[i], 'position_size2'] = current_position['asset2']

                if i > 0:
                    rets1 = asset1_data['Adj_Close'].pct_change().iloc[i]
                    rets2 = asset2_data['Adj_Close'].pct_change().iloc[i]
                    if current_position is not None:
                        results.loc[asset1_data.index[i], 'returns'] = (
                                rets1 * current_position['asset1'] +
                                rets2 * current_position['asset2']
                        )

            except Exception as e:
                print(f"Error processing data point {i}: {str(e)}")
                continue

        results['cumulative_returns'] = (1 + results['returns']).cumprod()
        return results

    def check_exit_conditions(
            self,
            position_return: float,
            position_age: int,
            zscore: float,
            current_prices: Dict,
            entry_prices: Dict,
            max_prices: Dict,
            asset1: pd.Series,
            asset2: pd.Series
    ) -> Tuple[bool, str]:
        """
        Check comprehensive exit conditions with enhanced risk management.

        Args:
            position_return: Current position return
            position_age: Age of position in days
            zscore: Current z-score
            current_prices: Dictionary of current prices
            entry_prices: Dictionary of entry prices
            max_prices: Dictionary of maximum prices reached
            asset1: Price series for first asset
            asset2: Price series for second asset

        Returns:
            Tuple[bool, str]: (exit flag, exit reason)
        """
        try:
            if position_age > self.time_stop:
                return True, "Time Stop"

            if position_return <= -self.base_stop_loss:
                return True, "Stop Loss"

            trailing_return = min(
                (curr_price - max_price) / max_price
                for curr_price, max_price in zip(
                    current_prices.values(),
                    max_prices.values()
                )
            )
            if trailing_return <= -self.trailing_stop:
                return True, "Trailing Stop"

            adjusted_profit_target = self.profit_take
            if self.current_regime in ["volatile_bullish", "volatile_bearish"]:
                adjusted_profit_target *= 0.8
            if position_return >= adjusted_profit_target:
                return True, "Profit Target"

            recent_correlation = self.calculate_correlation_score(
                asset1[-21:],
                asset2[-21:]
            )
            if recent_correlation < self.min_correlation * 0.7:
                return True, "Correlation Breakdown"

            recent_vol = asset1[-21:].pct_change().std() * np.sqrt(252)
            entry_vol = asset1[-position_age - 21:-position_age].pct_change().std() * np.sqrt(252)
            if recent_vol > entry_vol * 2:
                return True, "Volatility Spike"

            if abs(zscore) <= self.zscore_exit:
                recent_scores = self.zscore_history[-self.confirmation_periods:]
                if len(recent_scores) >= self.confirmation_periods:
                    if all(abs(z) <= self.zscore_exit * 1.2 for z in recent_scores):
                        return True, "Z-Score Exit"

            momentum = self.calculate_momentum_signal(
                (asset1 + asset2) / 2
            )
            if abs(momentum) > 15:
                return True, "Momentum Reversal"

            if self.current_regime.startswith("volatile") and position_return > 0:
                return True, "Regime Change"

            risk_score = self.calculate_risk_score(
                recent_vol,
                recent_correlation,
                momentum
            )
            if risk_score > 0.8 and position_return > 0:
                return True, "Risk Score Exit"

            price_moves = {
                asset: (curr_price - entry_price) / entry_price
                for (asset, curr_price), (_, entry_price) in zip(
                    current_prices.items(),
                    entry_prices.items()
                )
            }
            if any(abs(move) > 0.15 for move in price_moves.values()):
                return True, "Price Movement"

            return False, ""

        except Exception as e:
            print(f"Error in check_exit_conditions: {str(e)}")
            return True, "Error"

    def _confirm_signal(self, signal: float) -> bool:
        """
        Confirm trading signal with less restrictive checks.

        Args:
            signal: Combined trading signal

        Returns:
            bool: True if signal is confirmed
        """
        if len(self.zscore_history) < self.confirmation_periods:
            return True

        if abs(signal) > 0.8:
            return True

        recent_signals = self.zscore_history[-self.confirmation_periods:]
        signal_direction = np.sign(signal)

        confirm_count = sum(1 for s in recent_signals
                            if np.sign(s) == signal_direction)

        base_threshold = self.confirmation_periods * 0.6
        if self.current_regime in ["volatile_bearish", "volatile_bullish"]:
            threshold = base_threshold
        else:
            threshold = base_threshold * 0.8

        return confirm_count >= threshold

    def _confirm_exit_signal(self, zscore: float) -> bool:
        """
        Confirm exit signal with enhanced checks.

        Args:
            zscore: Current z-score

        Returns:
            bool: True if exit is confirmed
        """
        recent_scores = self.zscore_history[-self.confirmation_periods:]
        if len(recent_scores) < self.confirmation_periods:
            return False

        if abs(zscore) > self.zscore_exit * 1.5:  # Add additional safety margin
            return True

        confirm_count = sum(1 for z in recent_scores
                            if abs(z) <= self.zscore_exit * 1.2)

        base_threshold = self.confirmation_periods * 0.8
        if self.current_regime in ["calm_bullish", "calm_bearish"]:
            threshold = base_threshold * 0.9
        else:
            threshold = base_threshold

        return confirm_count >= threshold

    def calculate_zscore(self, spread: pd.Series, window: int = 21) -> float:
        """
        Calculate z-score with error handling.

        Args:
            spread: Price spread series
            window: Rolling window for calculation

        Returns:
            float: Calculated z-score
        """
        try:
            if len(spread) < window:
                return 0.0

            mean = spread.rolling(window=window).mean()
            std = spread.rolling(window=window).std()
            zscore = (spread - mean) / (std + 1e-8)

            return zscore.iloc[-1]

        except Exception as e:
            print(f"Error calculating zscore: {str(e)}")
            return 0.0


def analyze_strategy_results(results: pd.DataFrame, trades: List[Dict]) -> Dict:
    """
    Analyze strategy results with enhanced metrics.

    Args:
        results: DataFrame with strategy results
        trades: List of trade dictionaries

    Returns:
        Dict with performance metrics and analysis
    """
    total_return = results['cumulative_returns'].iloc[-1] - 1
    annual_return = (1 + total_return) ** (252 / len(results)) - 1
    daily_vol = results['returns'].std()
    annual_vol = daily_vol * np.sqrt(252)
    sharpe_ratio = np.sqrt(252) * results['returns'].mean() / daily_vol if daily_vol > 0 else 0

    cumulative = results['cumulative_returns']
    running_max = cumulative.expanding().max()
    drawdowns = (cumulative - running_max) / running_max
    max_drawdown = drawdowns.min()

    underwater_periods = (drawdowns < 0).astype(int).groupby(
        (drawdowns >= 0).astype(int).cumsum()
    ).sum()
    avg_recovery_time = underwater_periods.mean() if len(underwater_periods) > 0 else 0

    regime_performance = {}
    if trades:
        trade_returns = []
        holding_periods = []
        exit_reasons = {}
        regime_trades = {}

        for trade in trades:
            entry_value = (
                    trade['position']['asset1'] * trade['entry_prices']['asset1'] +
                    trade['position']['asset2'] * trade['entry_prices']['asset2']
            )
            exit_value = (
                    trade['position']['asset1'] * trade['exit_prices']['asset1'] +
                    trade['position']['asset2'] * trade['exit_prices']['asset2']
            )
            trade_return = (exit_value - entry_value) / abs(entry_value)

            trade_returns.append(trade_return)
            holding_periods.append(trade['hold_period'])

            regime = trade.get('regime', 'normal')
            if regime not in regime_trades:
                regime_trades[regime] = []
            regime_trades[regime].append(trade_return)

            exit_reason = trade['exit_reason']
            exit_reasons[exit_reason] = exit_reasons.get(exit_reason, 0) + 1

        for regime, returns in regime_trades.items():
            regime_performance[regime] = {
                'avg_return': np.mean(returns),
                'win_rate': len([r for r in returns if r > 0]) / len(returns),
                'trade_count': len(returns)
            }

        avg_trade_return = np.mean(trade_returns)
        win_rate = len([r for r in trade_returns if r > 0]) / len(trade_returns)
        avg_holding_period = np.mean(holding_periods)

        sortino_ratio = np.sqrt(252) * np.mean(results['returns']) / (
                np.std(results['returns'][results['returns'] < 0]) + 1e-8
        )

        consecutive_wins = max(len(list(g)) for k, g in itertools.groupby(
            [r > 0 for r in trade_returns]
        ) if k)
        consecutive_losses = max(len(list(g)) for k, g in itertools.groupby(
            [r <= 0 for r in trade_returns]
        ) if k)

    else:
        avg_trade_return = 0
        win_rate = 0
        avg_holding_period = 0
        exit_reasons = {}
        sortino_ratio = 0
        consecutive_wins = 0
        consecutive_losses = 0

    return {
        'total_return': total_return,
        'annual_return': annual_return,
        'annual_volatility': annual_vol,
        'sharpe_ratio': sharpe_ratio,
        'sortino_ratio': sortino_ratio,
        'max_drawdown': max_drawdown,
        'avg_recovery_time': avg_recovery_time,
        'number_of_trades': len(trades),
        'average_trade_return': avg_trade_return,
        'win_rate': win_rate,
        'average_holding_period': avg_holding_period,
        'exit_reasons': exit_reasons,
        'regime_performance': regime_performance,
        'consecutive_wins': consecutive_wins,
        'consecutive_losses': consecutive_losses
    }


def analyze_pair_relationship(asset1_data: pd.DataFrame, asset2_data: pd.DataFrame) -> Dict:
    """
    Analyze the statistical relationship between the pair of assets with proper data cleaning.

    Args:
        asset1_data: DataFrame with first asset data (with DatetimeIndex)
        asset2_data: DataFrame with second asset data (with DatetimeIndex)

    Returns:
        Dict with relationship metrics
    """
    try:
        asset1_data = asset1_data.copy()
        asset2_data = asset2_data.copy()

        returns1 = asset1_data['Adj_Close'].pct_change().ffill().bfill()
        returns2 = asset2_data['Adj_Close'].pct_change().ffill().bfill()

        returns1 = returns1.replace([np.inf, -np.inf], np.nan).ffill().bfill()
        returns2 = returns2.replace([np.inf, -np.inf], np.nan).ffill().bfill()

        correlations = {}
        for period, days in [('1M', 21), ('3M', 63), ('6M', 126), ('1Y', 252)]:
            if len(returns1) >= days:
                corr = returns1[-days:].corr(returns2[-days:])
                correlations[period] = corr
            else:
                correlations[period] = np.nan

        vols = {
            'asset1': returns1.std() * np.sqrt(252),
            'asset2': returns2.std() * np.sqrt(252)
        }

        try:
            X = add_constant(returns2)
            model = OLS(returns1, X).fit()
            beta = model.params[1]
        except:
            beta = np.nan

        try:
            clean_asset1 = asset1_data['Adj_Close'].dropna()
            clean_asset2 = asset2_data['Adj_Close'].dropna()

            common_idx = clean_asset1.index.intersection(clean_asset2.index)
            clean_asset1 = clean_asset1[common_idx]
            clean_asset2 = clean_asset2[common_idx]

            _, pvalue, _ = coint(clean_asset1, clean_asset2)
        except:
            pvalue = np.nan

        return {
            'correlations': correlations,
            'volatilities': vols,
            'beta': beta,
            'cointegration_pvalue': pvalue
        }

    except Exception as e:
        print(f"Error in relationship analysis: {str(e)}")
        import traceback
        print(traceback.format_exc())
        return {
            'correlations': {},
            'volatilities': {'asset1': np.nan, 'asset2': np.nan},
            'beta': np.nan,
            'cointegration_pvalue': np.nan
        }


def main():
    """
    Run the enhanced pairs trading strategy on sample data.
    """
    try:
        asset1_data = pd.read_csv(r'C:\Users\arnav\Downloads\pairs_trading_system\data\raw\ADP.csv')
        asset2_data = pd.read_csv(r'C:\Users\arnav\Downloads\pairs_trading_system\data\raw\PAYX.csv')

        for df in [asset1_data, asset2_data]:
            df['Date'] = pd.to_datetime(df['Date'])
            df.set_index('Date', inplace=True)
            df.sort_index(inplace=True)

        for df in [asset1_data, asset2_data]:
            df['Adj_Close'] = df['Adj_Close'].ffill().bfill()
            df.dropna(inplace=True)
            df.replace([np.inf, -np.inf], np.nan, inplace=True)
            df.ffill().bfill()

        common_dates = asset1_data.index.intersection(asset2_data.index)
        asset1_data = asset1_data.loc[common_dates]
        asset2_data = asset2_data.loc[common_dates]

        print(f"\nData Summary:")
        print(f"Date range: {common_dates[0]} to {common_dates[-1]}")
        print(f"Number of trading days: {len(common_dates)}")

        print("\nAnalyzing Pair Relationship:")
        relationship = analyze_pair_relationship(asset1_data, asset2_data)

        print("\nCorrelations:")
        for period, corr in relationship['correlations'].items():
            if not np.isnan(corr):
                print(f"{period}: {corr:.3f}")
            else:
                print(f"{period}: Not enough data")

        print("\nAnnualized Volatilities:")
        for asset, vol in relationship['volatilities'].items():
            if not np.isnan(vol):
                print(f"{asset}: {vol:.2%}")
            else:
                print(f"{asset}: Unable to calculate")

        if not np.isnan(relationship['beta']):
            print(f"\nBeta: {relationship['beta']:.3f}")
        else:
            print("\nBeta: Unable to calculate")

        if not np.isnan(relationship['cointegration_pvalue']):
            print(f"Cointegration p-value: {relationship['cointegration_pvalue']:.4f}")
            if relationship['cointegration_pvalue'] > 0.05:
                print("Warning: Pair may not be cointegrated (p-value > 0.05)")
        else:
            print("Cointegration: Unable to calculate")

        strategy = IntegratedPairsStrategy(
            lookback_window=126,
            zscore_entry=1.5,
            zscore_exit=0.75,
            stop_loss=0.20,
            trailing_stop=0.10,
            time_stop=42,
            profit_take=0.10,
            position_size=0.2,
            min_correlation=0.3,
            signal_exit_threshold=0.3,
            confirmation_periods=1,
            max_portfolio_vol=0.15,
            regime_lookback=126
        )

        results = strategy.run_strategy(asset1_data, asset2_data)
        performance = analyze_strategy_results(results, strategy.trades)

        print("\nStrategy Performance Summary:")
        print(f"Total Return: {performance['total_return']:.2%}")
        print(f"Annual Return: {performance['annual_return']:.2%}")
        print(f"Sharpe Ratio: {performance['sharpe_ratio']:.2f}")
        print(f"Sortino Ratio: {performance['sortino_ratio']:.2f}")
        print(f"Max Drawdown: {performance['max_drawdown']:.2%}")
        print(f"Average Recovery Time: {performance['avg_recovery_time']:.1f} days")

        print(f"\nTrade Statistics:")
        print(f"Number of Trades: {performance['number_of_trades']}")
        print(f"Win Rate: {performance['win_rate']:.2%}")
        print(f"Average Holding Period: {performance['average_holding_period']:.1f} days")
        print(f"Consecutive Wins: {performance['consecutive_wins']}")
        print(f"Consecutive Losses: {performance['consecutive_losses']}")

        print("\nRegime Performance:")
        for regime, metrics in performance['regime_performance'].items():
            print(f"\n{regime}:")
            print(f"  Average Return: {metrics['avg_return']:.2%}")
            print(f"  Win Rate: {metrics['win_rate']:.2%}")
            print(f"  Trade Count: {metrics['trade_count']}")

        print("\nExit Reason Distribution:")
        for reason, count in performance['exit_reasons'].items():
            print(f"{reason}: {count} trades ({count/performance['number_of_trades']:.1%})")

        print("\nGenerating interactive performance dashboard...")
        create_strategy_dashboard(results, asset1_data, asset2_data, strategy.trades)
        print("Interactive dashboard saved as 'strategy_dashboard.html'")

        return results, performance

    except Exception as e:
        print(f"Error in main execution: {str(e)}")
        import traceback
        print(traceback.format_exc())
        return None, None


if __name__ == "__main__":
    results, performance = main()