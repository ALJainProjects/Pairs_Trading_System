"""
Enhanced Risk Management for Pairs Trading

Handles:
- Pair-specific position sizing
- Spread-based stop losses
- Portfolio-level risk for multiple pairs
"""
from datetime import datetime

import pandas as pd
import numpy as np
from typing import Dict, Tuple, Optional
import json
from pathlib import Path
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from config.settings import DATA_DIR
from config.logging_config import logger

class PairRiskManager:
    """Risk manager specifically designed for pairs trading."""

    def __init__(
        self,
        max_position_size: float = 0.05,
        max_drawdown: float = 0.20,
        stop_loss_threshold: float = 0.10,
        max_correlation: float = 0.7,
        leverage_limit: float = 2.0
    ):
        """
        Initialize risk manager.

        Args:
            max_position_size: Maximum size per pair
            max_drawdown: Maximum portfolio drawdown
            stop_loss_threshold: Stop loss on spread movement
            max_correlation: Maximum correlation between active pairs
            leverage_limit: Maximum total leverage
        """
        self.max_position_size = max_position_size
        self.max_drawdown = max_drawdown
        self.stop_loss_threshold = stop_loss_threshold
        self.max_correlation = max_correlation
        self.leverage_limit = leverage_limit

    def position_sizing(self,
                       portfolio_value: float,
                       pair_prices: Tuple[float, float],
                       num_active_pairs: int,
                       pair_volatility: Optional[float] = None) -> float:
        """
        Calculate position size for a pair trade.

        Args:
            portfolio_value: Current portfolio value
            pair_prices: Tuple of (price1, price2)
            num_active_pairs: Number of currently active pairs
            pair_volatility: Optional volatility of the pair spread

        Returns:
            float: Position size in units
        """
        adjusted_size = self.max_position_size / max(1, num_active_pairs)

        max_position_value = portfolio_value * adjusted_size

        if pair_volatility is not None:
            vol_adjustment = 1.0 / (1.0 + pair_volatility)
            max_position_value *= vol_adjustment

        price1, price2 = pair_prices
        pair_price = price1 + price2

        if pair_price <= 0:
            return 0

        return max_position_value / pair_price

    def check_pair_correlation(self,
                             new_pair_returns: pd.DataFrame,
                             active_pairs_returns: pd.DataFrame) -> bool:
        """
        Check if adding a new pair would exceed correlation limits.

        Args:
            new_pair_returns: Returns of the new pair
            active_pairs_returns: Returns of currently active pairs

        Returns:
            bool: True if correlation is acceptable
        """
        if active_pairs_returns.empty:
            return True

        correlation = new_pair_returns.corrwith(active_pairs_returns)
        max_corr = abs(correlation).max()

        return max_corr < self.max_correlation

    def check_spread_stop_loss(self,
                             entry_spread: float,
                             current_spread: float,
                             position_type: str) -> bool:
        """
        Check spread-based stop loss.

        Args:
            entry_spread: Spread at entry
            current_spread: Current spread value
            position_type: 'long' or 'short'

        Returns:
            bool: True if stop loss is triggered
        """
        if entry_spread == 0:
            return False

        if position_type == 'long':
            loss = (entry_spread - current_spread) / abs(entry_spread)
        else:
            loss = (current_spread - entry_spread) / abs(entry_spread)

        return loss >= self.stop_loss_threshold

    def calculate_pair_exposure(self,
                              positions: Dict[str, Dict],
                              current_prices: Dict[str, float]) -> float:
        """
        Calculate total exposure from all pair positions.

        Args:
            positions: Dictionary of current positions
            current_prices: Dictionary of current prices

        Returns:
            float: Total exposure
        """
        total_exposure = 0

        for pair, position in positions.items():
            asset1, asset2 = pair
            quantity = position['quantity']

            exposure1 = abs(quantity * current_prices[asset1])
            exposure2 = abs(quantity * current_prices[asset2])
            total_exposure += exposure1 + exposure2

        return total_exposure

    def monitor_portfolio_risk(self,
                             equity_curve: pd.Series,
                             positions: Dict,
                             current_prices: Dict[str, float]) -> Tuple[bool, str]:
        """
        Monitor portfolio-wide risk constraints.

        Args:
            equity_curve: Portfolio equity curve
            positions: Current positions
            current_prices: Current asset prices

        Returns:
            Tuple of (risk_violated, reason)
        """
        if equity_curve.empty:
            return False, ""

        current_equity = equity_curve.iloc[-1]

        drawdown = self.calculate_drawdown(equity_curve)
        if drawdown > self.max_drawdown:
            return True, f"Max drawdown exceeded: {drawdown:.2%}"

        total_exposure = self.calculate_pair_exposure(positions, current_prices)
        current_leverage = total_exposure / current_equity

        if current_leverage > self.leverage_limit:
            return True, f"Leverage limit exceeded: {current_leverage:.2f}x"

        for pair, position in positions.items():
            asset1, asset2 = pair
            quantity = position['quantity']
            position_value = quantity * (
                current_prices[asset1] + current_prices[asset2]
            )
            position_size = position_value / current_equity

            if position_size > self.max_position_size:
                return True, f"Position size exceeded for {pair}: {position_size:.2%}"

        return False, ""

    def calculate_drawdown(self, equity_curve: pd.Series) -> float:
        """Calculate maximum drawdown."""
        peaks = equity_curve.expanding().max()
        drawdowns = (equity_curve - peaks) / peaks
        return abs(float(drawdowns.min()))

    def generate_risk_report(self,
                             equity_curve: pd.Series,
                             positions: Dict,
                             trades: pd.DataFrame) -> Dict:
        """
        Generate comprehensive risk report.

        Args:
            equity_curve: Portfolio equity curve
            positions: Current positions
            trades: Trade history

        Returns:
            Dictionary with risk metrics
        """
        drawdown = self.calculate_drawdown(equity_curve)
        returns = equity_curve.pct_change().dropna()

        latest_equity = equity_curve.iloc[-1]
        exposures = []

        for date in equity_curve.index:
            date_trades = trades[trades['Date'] == date]
            if not date_trades.empty:
                exposure = date_trades['Quantity'].abs() * (date_trades['Price1'] + date_trades['Price2'])
                exposures.append(exposure.sum())
            else:
                exposures.append(0.0)

        exposure_series = pd.Series(exposures, index=equity_curve.index)
        leverage_series = exposure_series / equity_curve

        report = {
            'Current Metrics': {
                'Drawdown': drawdown,
                'Volatility': returns.std() * np.sqrt(252),
                'Current Leverage': leverage_series.iloc[-1],
                'Active Pairs': len(positions)
            },
            'Risk Parameters': {
                'Max Position Size': self.max_position_size,
                'Max Drawdown': self.max_drawdown,
                'Stop Loss': self.stop_loss_threshold,
                'Max Correlation': self.max_correlation,
                'Leverage Limit': self.leverage_limit
            },
            'Time Series': {
                'Leverage': leverage_series,
                'Exposure': exposure_series
            }
        }

        return report

    def check_drawdown(self, equity: pd.Series) -> bool:
        """
        Check if current drawdown exceeds maximum allowed.

        Args:
            equity: Series of portfolio values

        Returns:
            bool: True if drawdown limit is exceeded
        """
        if len(equity) < 2:
            return False

        peak = equity.expanding().max()
        drawdown = (equity - peak) / peak
        current_drawdown = abs(drawdown.iloc[-1])

        return current_drawdown > self.max_drawdown

    def check_portfolio_risk(self,
                             equity: pd.Series,
                             positions: pd.DataFrame) -> bool:
        """
        Check overall portfolio risk constraints.

        Args:
            equity: Series of portfolio values
            positions: DataFrame of current positions

        Returns:
            bool: True if any risk limit is exceeded
        """
        if len(positions) == 0:
            return False

        portfolio_value = equity.iloc[-1]
        for _, pos in positions.iterrows():
            position_value = abs(pos['Quantity'] * pos['Price'])
            if position_value / portfolio_value > self.max_position_size:
                return True

        total_exposure = (positions['Quantity'].abs() * positions['Price']).sum()
        current_leverage = total_exposure / portfolio_value
        if current_leverage > self.leverage_limit:
            return True

        return False

    def calculate_position_size(self,
                                portfolio_value: float,
                                price: float,
                                transaction_cost: float) -> float:
        """
        Calculate safe position size for a trade.

        Args:
            portfolio_value: Current portfolio value
            price: Asset price
            transaction_cost: Transaction cost rate

        Returns:
            float: Recommended position size
        """
        available_capital = portfolio_value * (1 - transaction_cost)

        max_position = available_capital * self.max_position_size

        if price > 0:
            return max_position / price
        return 0.0

    def validate_new_position(self,
                              portfolio_value: float,
                              current_exposure: float,
                              new_position_size: float) -> bool:
        """
        Validate if a new position can be added.

        Args:
            portfolio_value: Current portfolio value
            current_exposure: Current total exposure
            new_position_size: Size of new position

        Returns:
            bool: Whether new position is acceptable
        """
        if new_position_size / portfolio_value > self.max_position_size:
            return False

        total_exposure = current_exposure + new_position_size
        if total_exposure / portfolio_value > self.leverage_limit:
            return False

        return True


def main():
    """Test pairs trading risk management with local data."""
    logger.info("Starting pairs trading risk management test")

    data_dir_edited = DATA_DIR.replace(r'\config', '')
    data_dir = Path(f"{data_dir_edited}/raw")
    OUTPUT_DIR = Path("risk_test")
    OUTPUT_DIR.mkdir(exist_ok=True)

    logger.info(f"Using data directory: {data_dir}")
    logger.info(f"Output directory created at: {OUTPUT_DIR}")

    pairs = [
        ("AAPL", "MSFT"),
        ("GOOGL", "META"),
        ("NVDA", "AMD")
    ]
    logger.info(f"Testing pairs: {pairs}")

    prices = {}
    returns = pd.DataFrame()

    logger.info("Loading price data...")
    for pair in pairs:
        for symbol in pair:
            if symbol not in prices:
                try:
                    df = pd.read_csv(data_dir / f"{symbol}.csv")
                    df['Date'] = pd.to_datetime(df['Date'])
                    df.set_index('Date', inplace=True)
                    prices[symbol] = df['Adj_Close']
                    returns[symbol] = prices[symbol].pct_change()
                    logger.debug(f"Loaded data for {symbol}: {len(df)} rows")
                except Exception as e:
                    logger.error(f"Failed to load data for {symbol}: {str(e)}")
                    raise

    logger.info("Aligning dates across all symbols...")
    common_dates = pd.Index([])
    for series in prices.values():
        if len(common_dates) == 0:
            common_dates = series.index
        else:
            common_dates = common_dates.intersection(series.index)

    logger.info(f"Date range: {common_dates[0]} to {common_dates[-1]}")
    logger.debug(f"Total trading days: {len(common_dates)}")

    for symbol in prices:
        prices[symbol] = prices[symbol][common_dates]
        if symbol in returns.columns:
            returns[symbol] = returns[symbol][common_dates]

    logger.info("Initializing risk manager...")
    risk_manager = PairRiskManager(
        max_position_size=0.05,
        max_drawdown=0.20,
        stop_loss_threshold=0.10,
        max_correlation=0.7,
        leverage_limit=2.0
    )

    positions = {}
    trade_history = []
    equity_curve = pd.Series(index=common_dates, data=100000, dtype=np.float64)
    logger.info("Starting simulation with initial equity: $100,000")

    logger.info("Beginning trading simulation...")
    for date in common_dates[20:]:
        current_prices = {symbol: prices[symbol][date] for symbol in prices.keys()}

        closed_pairs = []
        for pair, position in positions.items():
            entry_spread = position['entry_prices'][0] - position['entry_prices'][1]
            current_spread = current_prices[pair[0]] - current_prices[pair[1]]

            if risk_manager.check_spread_stop_loss(entry_spread, current_spread, position['type']):
                pnl = calculate_position_pnl(
                    entry_prices=position['entry_prices'],
                    exit_prices=(current_prices[pair[0]], current_prices[pair[1]]),
                    quantity=position['quantity'],
                    position_type=position['type']
                )

                equity_curve[date] = equity_curve[date] + pnl

                closed_pairs.append(pair)
                trade_history.append({
                    'Date': date,
                    'Pair': f"{pair[0]}/{pair[1]}",
                    'Action': 'Close',
                    'Reason': 'Stop Loss',
                    'Quantity': position['quantity'],
                    'Price1': current_prices[pair[0]],
                    'Price2': current_prices[pair[1]],
                    'PnL': pnl
                })
                logger.info(f"Stop loss triggered for {pair[0]}/{pair[1]} at {date}, PnL: ${pnl:.2f}")

        for pair in closed_pairs:
            del positions[pair]

        for pair in pairs:
            if pair not in positions:
                if positions:
                    pair_returns = returns[list(pair)]
                    active_pairs_returns = returns[[p[0] for p in positions.keys()]].join(
                        returns[[p[1] for p in positions.keys()]]
                    )
                    if not risk_manager.check_pair_correlation(pair_returns, active_pairs_returns):
                        logger.debug(f"Correlation check failed for {pair[0]}/{pair[1]}")
                        continue

                size = risk_manager.position_sizing(
                    portfolio_value=equity_curve[date],
                    pair_prices=(current_prices[pair[0]], current_prices[pair[1]]),
                    num_active_pairs=len(positions),
                    pair_volatility=returns[list(pair)].std().mean()
                )

                if size > 0:
                    position_type = 'long' if current_prices[pair[0]] > current_prices[pair[1]] else 'short'
                    positions[pair] = {
                        'quantity': size,
                        'entry_date': date,
                        'entry_prices': (current_prices[pair[0]], current_prices[pair[1]]),
                        'type': position_type
                    }

                    trade_history.append({
                        'Date': date,
                        'Pair': f"{pair[0]}/{pair[1]}",
                        'Action': 'Open',
                        'Quantity': size,
                        'Price1': current_prices[pair[0]],
                        'Price2': current_prices[pair[1]]
                    })
                    logger.info(f"Opened {position_type} position for {pair[0]}/{pair[1]} at {date}")

        risk_exceeded, reason = risk_manager.monitor_portfolio_risk(
            equity_curve[:date],
            positions,
            current_prices
        )

        if risk_exceeded:
            logger.warning(f"Risk limits exceeded at {date}: {reason}")
            for pair, position in positions.items():
                pnl = calculate_position_pnl(
                    entry_prices=position['entry_prices'],
                    exit_prices=(current_prices[pair[0]], current_prices[pair[1]]),
                    quantity=position['quantity'],
                    position_type=position['type']
                )

                equity_curve[date] = equity_curve[date] + pnl

                trade_history.append({
                    'Date': date,
                    'Pair': f"{pair[0]}/{pair[1]}",
                    'Action': 'Close',
                    'Reason': reason,
                    'Quantity': position['quantity'],
                    'Price1': current_prices[pair[0]],
                    'Price2': current_prices[pair[1]],
                    'PnL': pnl
                })
            positions = {}

    equity_curve = equity_curve.ffill()

    logger.info("Generating final reports...")
    trades_df = pd.DataFrame(trade_history)
    risk_report = risk_manager.generate_risk_report(
        equity_curve=equity_curve,
        positions=positions,
        trades=trades_df
    )

    logger.info("Saving risk report...")
    with open(OUTPUT_DIR / 'risk_report.json', 'w') as f:
        json_data = make_json_serializable(risk_report)
        json.dump(json_data, f, indent=4)

    logger.info("Saving results...")
    trades_df.to_csv(OUTPUT_DIR / 'trades.csv', index=False)
    equity_curve.to_csv(OUTPUT_DIR / 'equity_curve.csv')

    logger.info("Creating visualizations...")
    fig = make_subplots(
        rows=3, cols=1,
        subplot_titles=['Portfolio Value', 'Leverage', 'Position Exposure'],
        vertical_spacing=0.1
    )

    fig.add_trace(
        go.Scatter(x=equity_curve.index, y=equity_curve, name='Portfolio Value'),
        row=1, col=1
    )

    fig.add_trace(
        go.Scatter(
            x=risk_report['Time Series']['Leverage'].index,
            y=risk_report['Time Series']['Leverage'],
            name='Leverage'
        ),
        row=2, col=1
    )

    fig.add_trace(
        go.Scatter(
            x=risk_report['Time Series']['Exposure'].index,
            y=risk_report['Time Series']['Exposure'],
            name='Exposure'
        ),
        row=3, col=1
    )

    fig.update_layout(
        height=900,
        title_text="Pairs Trading Risk Analysis",
        showlegend=True
    )

    fig.write_html(OUTPUT_DIR / 'risk_analysis.html')

    logger.info("Printing summary statistics...")
    print("\nRisk Analysis Results:")
    print("\nCurrent Metrics:")
    for metric, value in risk_report['Current Metrics'].items():
        print(f"{metric}: {value:.4f}")

    print("\nRisk Parameters:")
    for param, value in risk_report['Risk Parameters'].items():
        print(f"{param}: {value:.4f}")

    logger.info("Creating detailed trading visualizations...")
    create_trading_visualizations(
        equity_curve=equity_curve,
        trades_df=trades_df,
        risk_report=risk_report,
        prices=prices,
        OUTPUT_DIR=OUTPUT_DIR
    )

    logger.info("Simulation completed successfully")
    return {
        'risk_report': risk_report,
        'trades': trades_df,
        'equity_curve': equity_curve,
        'positions': positions,
        'prices': prices
    }


def create_trading_visualizations(equity_curve: pd.Series,
                                  trades_df: pd.DataFrame,
                                  risk_report: dict,
                                  prices: dict,
                                  OUTPUT_DIR: Path) -> None:
    """
    Create comprehensive trading visualizations.

    Args:
        equity_curve: Portfolio equity curve
        trades_df: DataFrame of trading history
        risk_report: Risk metrics dictionary
        prices: Dictionary of price series
        OUTPUT_DIR: Output directory for saving plots
    """
    fig = make_subplots(
        rows=6, cols=1,
        subplot_titles=(
            'Portfolio Performance',
            'Pair Spreads',
            'Position Sizes',
            'Rolling Volatility (20D)',
            'Drawdown',
            'Trade Metrics'
        ),
        vertical_spacing=0.05,
        specs=[[{"secondary_y": True}],
               [{"secondary_y": True}],
               [{"secondary_y": False}],
               [{"secondary_y": False}],
               [{"secondary_y": False}],
               [{"secondary_y": True}]],
        row_heights=[400] * 6
    )

    fig.add_trace(
        go.Scatter(x=equity_curve.index,
                   y=equity_curve,
                   name='Portfolio Value',
                   line=dict(color='blue')),
        row=1, col=1
    )

    if 'PnL' in trades_df.columns:
        cum_pnl = trades_df.set_index('Date')['PnL'].cumsum()
        fig.add_trace(
            go.Scatter(x=cum_pnl.index,
                       y=cum_pnl,
                       name='Cumulative P&L',
                       line=dict(color='green')),
            row=1, col=1, secondary_y=True
        )

    for pair in set(trades_df['Pair'].unique()):
        if '/' in pair:
            symbol1, symbol2 = pair.split('/')
            if symbol1 in prices and symbol2 in prices:
                spread = prices[symbol1] - prices[symbol2]
                fig.add_trace(
                    go.Scatter(x=spread.index,
                               y=spread,
                               name=f'{pair} Spread',
                               line=dict(width=1)),
                    row=2, col=1
                )

    trades_df['Position Size'] = trades_df['Quantity'] * (trades_df['Price1'] + trades_df['Price2'])
    for pair in trades_df['Pair'].unique():
        pair_trades = trades_df[trades_df['Pair'] == pair]
        fig.add_trace(
            go.Scatter(x=pair_trades['Date'],
                       y=pair_trades['Position Size'],
                       name=f'{pair} Position',
                       mode='lines+markers'),
            row=3, col=1
        )

    returns = equity_curve.pct_change()
    rolling_vol = returns.rolling(window=20).std() * np.sqrt(252) * 100
    fig.add_trace(
        go.Scatter(x=rolling_vol.index,
                   y=rolling_vol,
                   name='Rolling Volatility (%)',
                   line=dict(color='red')),
        row=4, col=1
    )

    peaks = equity_curve.expanding().max()
    drawdown = ((equity_curve - peaks) / peaks) * 100
    fig.add_trace(
        go.Scatter(x=drawdown.index,
                   y=drawdown,
                   name='Drawdown (%)',
                   line=dict(color='red'),
                   fill='tonexty'),
        row=5, col=1
    )

    daily_trades = trades_df.groupby('Date').size()
    fig.add_trace(
        go.Scatter(x=daily_trades.index,
                   y=daily_trades,
                   name='Daily Trades',
                   line=dict(color='purple')),
        row=6, col=1
    )

    if 'PnL' in trades_df.columns:
        trades_df['Win'] = trades_df['PnL'] > 0
        rolling_win_rate = trades_df.groupby('Date')['Win'].mean().rolling(20).mean() * 100
        fig.add_trace(
            go.Scatter(x=rolling_win_rate.index,
                       y=rolling_win_rate,
                       name='Rolling Win Rate (%)',
                       line=dict(color='green')),
            row=6, col=1, secondary_y=True
        )

    fig.update_layout(
        title_text="Comprehensive Trading Analysis",
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )

    fig.update_yaxes(title_text="Portfolio Value ($)", row=1, col=1)
    fig.update_yaxes(title_text="P&L ($)", row=1, col=1, secondary_y=True)
    fig.update_yaxes(title_text="Spread Value", row=2, col=1)
    fig.update_yaxes(title_text="Position Size ($)", row=3, col=1)
    fig.update_yaxes(title_text="Volatility (%)", row=4, col=1)
    fig.update_yaxes(title_text="Drawdown (%)", row=5, col=1)
    fig.update_yaxes(title_text="Number of Trades", row=6, col=1)
    fig.update_yaxes(title_text="Win Rate (%)", row=6, col=1, secondary_y=True)

    fig.write_html(OUTPUT_DIR / 'trading_metrics.html')

    corr_matrix = pd.DataFrame()
    for symbol in prices.keys():
        corr_matrix[symbol] = prices[symbol].pct_change()

    correlation = corr_matrix.corr()

    corr_fig = go.Figure(data=go.Heatmap(
        z=correlation,
        x=correlation.columns,
        y=correlation.columns,
        colorscale='RdBu',
        zmin=-1,
        zmax=1
    ))

    corr_fig.update_layout(
        title="Pair Correlations",
        height=600
    )

    corr_fig.write_html(OUTPUT_DIR / 'correlation_heatmap.html')

def make_json_serializable(obj):
    """Convert nested objects to JSON serializable format."""
    if isinstance(obj, dict):
        return {k: make_json_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, pd.Series):
        return {str(k): v for k, v in obj.to_dict().items()}
    elif isinstance(obj, pd.DataFrame):
        return obj.to_dict(orient='records')
    elif isinstance(obj, (pd.Timestamp, datetime)):
        return str(obj)
    elif isinstance(obj, (np.int64, np.int32)):
        return int(obj)
    elif isinstance(obj, (np.float64, np.float32)):
        return float(obj)
    return obj


def calculate_position_pnl(entry_prices: Tuple[float, float],
                           exit_prices: Tuple[float, float],
                           quantity: float,
                           position_type: str) -> float:
    """
    Calculate P&L for a pair trade.

    Args:
        entry_prices: (price1, price2) at entry
        exit_prices: (price1, price2) at exit
        quantity: Position size
        position_type: 'long' or 'short'

    Returns:
        float: P&L amount
    """
    entry_spread = entry_prices[0] - entry_prices[1]
    exit_spread = exit_prices[0] - exit_prices[1]

    if position_type == 'long':
        return quantity * (exit_spread - entry_spread)
    else:
        return quantity * (entry_spread - exit_spread)

if __name__ == "__main__":
    try:
        output = main()
        logger.info("Program completed successfully")
    except Exception as e:
        logger.error(f"Program failed with error: {str(e)}", exc_info=True)
        raise