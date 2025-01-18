from pathlib import Path
import pandas as pd
import plotly.graph_objects as go
from typing import List, Dict
import json
from src.strategy.pairs_strategy_basic import PairsTrader
from src.strategy.backtest import MultiPairBackTester, BacktestVisualizer
from src.strategy.risk import PairRiskManager
from src.strategy.optimization import (
    MultiStrategyOptimizer,
    WalkForwardOptimizer,
    CrossValidatedOptimizer
)
from config.logging_config import logger
from config.settings import DATA_DIR

SYMBOLS = [
    'AAPL', 'MSFT', 'NVDA', 'AMD', 'INTC',
    'QCOM', 'AVGO', 'ASML', 'AMAT', 'MU'
]


def load_market_data(symbols: List[str]) -> pd.DataFrame:
    """Load market data from CSV files."""
    raw_data_dir = Path(DATA_DIR.replace(r'\config', '')) / "raw"

    prices = pd.DataFrame()
    for symbol in symbols:
        try:
            csv_path = raw_data_dir / f"{symbol}.csv"
            if not csv_path.exists():
                logger.warning(f"Data file not found for {symbol}")
                continue

            df = pd.read_csv(csv_path)
            if 'Date' not in df.columns or 'Adj_Close' not in df.columns:
                logger.warning(f"Required columns missing in {symbol}")
                continue

            df['Date'] = pd.to_datetime(df['Date'])
            df.set_index('Date', inplace=True)
            prices[symbol] = df['Adj_Close']
            logger.debug(f"Successfully loaded data for {symbol}")
        except Exception as e:
            logger.error(f"Error reading data for {symbol}: {str(e)}")
            continue

    if prices.empty:
        raise ValueError("No valid price data loaded")

    return prices.ffill().bfill()


def main_backtest() -> Dict:
    """Test the backtesting module with basic pairs strategy."""
    logger.info("Starting backtest with real market data")

    output_dir = Path("backtest_outputs")
    output_dir.mkdir(parents=True, exist_ok=True)

    plots_dir = output_dir / "plots"
    results_dir = output_dir / "results"

    for directory in [plots_dir, results_dir]:
        directory.mkdir(exist_ok=True)

    prices = load_market_data(SYMBOLS)
    logger.info(f"Loaded data for {len(prices.columns)} symbols from {prices.index[0]} to {prices.index[-1]}")

    strategy = PairsTrader(
        correlation_threshold=0.7,
        lookback_period=20,
        entry_threshold=2.0,
        exit_threshold=0.5,
        max_position_size=0.1,
        stop_loss=0.02,
        take_profit=0.04,
        max_drawdown=0.2,
        max_pairs=5
    )

    risk_manager = PairRiskManager(
        max_position_size=0.05,
        max_drawdown=0.20,
        stop_loss_threshold=0.10,
        max_correlation=0.7,
        leverage_limit=2.0,
        var_confidence=0.95,
        min_model_confidence=0.6
    )

    backtester = MultiPairBackTester(
        strategy=strategy,
        returns=prices,
        initial_capital=100000,
        risk_manager=risk_manager
    )

    logger.info("Running backtest...")
    equity_curve = backtester.run_backtest()

    logger.info("Generating backtest report...")
    report = backtester.generate_report(str(results_dir / "backtest_report.json"))

    prices.to_csv(results_dir / "price_data.csv")
    equity_curve.to_csv(results_dir / "equity_curve.csv")

    logger.info("Creating visualizations...")
    visualizer = BacktestVisualizer()
    visualizer.plot_all_analysis(backtester)

    logger.info("Backtest completed successfully")
    return {
        'equity_curve': equity_curve,
        'report': report,
        'data_period': {
            'start': prices.index[0].strftime('%Y-%m-%d'),
            'end': prices.index[-1].strftime('%Y-%m-%d'),
            'symbols': list(prices.columns)
        }
    }


def main_optimization() -> Dict:
    """Test the optimization module with basic pairs strategy."""
    logger.info("Starting optimization with real market data")

    output_dir = Path("optimization_outputs")
    output_dir.mkdir(parents=True, exist_ok=True)

    plots_dir = output_dir / "plots"
    results_dir = output_dir / "results"

    for directory in [plots_dir, results_dir]:
        directory.mkdir(exist_ok=True)

    prices = load_market_data(SYMBOLS)
    logger.info(f"Loaded data for {len(prices.columns)} symbols from {prices.index[0]} to {prices.index[-1]}")

    strategy = PairsTrader(
        correlation_threshold=0.7,
        lookback_period=20,
        entry_threshold=2.0,
        exit_threshold=0.5
    )

    logger.info("Running Bayesian optimization...")
    optimizer = MultiStrategyOptimizer(
        strategy=strategy,
        returns=prices,
        strategy_type='Statistical',
        initial_capital=100000,
        transaction_cost=0.001,
        max_pairs=5
    )

    best_params, best_score = optimizer.bayesian_optimize(n_trials=50)

    optimizer.plot_optimization_results()
    optimizer.generate_optimization_report(
        str(results_dir / "bayesian_optimization_report.xlsx")
    )

    logger.info("Running walk-forward optimization...")
    wf_optimizer = WalkForwardOptimizer(
        data=prices,
        train_size=252,
        test_size=126,
        step_size=63
    )

    wf_results = wf_optimizer.optimize(
        strategy=strategy,
        param_grid={
            'zscore_entry': [1.5, 2.0, 2.5],
            'zscore_exit': [0.0, 0.5],
            'lookback_period': [10, 20, 30],
            'correlation_threshold': [0.6, 0.7, 0.8]
        }
    )

    logger.info("Running cross-validated optimization...")
    cv_optimizer = CrossValidatedOptimizer(
        data=prices,
        n_splits=5,
        validation_size=0.3
    )

    cv_results = cv_optimizer.optimize(
        strategy=strategy,
        param_grid={
            'zscore_entry': [1.5, 2.0, 2.5],
            'zscore_exit': [0.0, 0.5],
            'lookback_period': [10, 20, 30],
            'correlation_threshold': [0.6, 0.7, 0.8]
        }
    )

    optimization_results = {
        'bayesian': {
            'best_parameters': best_params,
            'best_score': best_score
        },
        'walk_forward': wf_results,
        'cross_validation': cv_results,
        'data_period': {
            'start': prices.index[0].strftime('%Y-%m-%d'),
            'end': prices.index[-1].strftime('%Y-%m-%d'),
            'symbols': list(prices.columns)
        }
    }

    with open(results_dir / "optimization_results.json", 'w') as f:
        json.dump(optimization_results, f, indent=4, default=str)

    logger.info("Optimization completed successfully")
    return optimization_results


def main_risk() -> Dict:
    """Test the risk management module with real market data with active position management."""
    logger.info("Starting risk analysis with real market data")

    output_dir = Path("risk_outputs")
    output_dir.mkdir(parents=True, exist_ok=True)

    plots_dir = output_dir / "plots"
    results_dir = output_dir / "results"

    for directory in [plots_dir, results_dir]:
        directory.mkdir(exist_ok=True)

    prices = load_market_data(SYMBOLS)
    logger.info(f"Loaded data for {len(prices.columns)} symbols from {prices.index[0]} to {prices.index[-1]}")

    risk_manager = PairRiskManager(
        max_position_size=0.05,
        max_drawdown=0.20,
        stop_loss_threshold=0.10,
        max_correlation=0.7,
        leverage_limit=2.0,
        var_confidence=0.95,
        min_model_confidence=0.6,
        max_correlation_exposure=0.3
    )

    pairs = [
        (SYMBOLS[i], SYMBOLS[i + 1])
        for i in range(0, len(SYMBOLS) - 1, 2)
    ]

    active_positions = {
        pairs[0]: {
            'quantity': 100,
            'entry_price': prices[pairs[0][0]].iloc[0],
            'entry_date': prices.index[0]
        }
    }

    risk_history = []
    risk_metrics = {}
    trade_history = []
    portfolio_value = 100000

    logger.info("Starting position management and risk monitoring...")
    for current_date in prices.index:
        try:
            current_prices = prices.loc[current_date]
            positions_to_close = []

            for pair in pairs:
                metrics = risk_manager.update_risk_metrics(
                    pair=pair,
                    prices=prices.loc[:current_date],
                    positions=active_positions,
                    model_confidence=0.8
                )

                risk_metrics[f"{pair[0]}/{pair[1]}"] = {
                    'var_95': metrics.var_95,
                    'cvar_95': metrics.cvar_95,
                    'volatility': metrics.volatility,
                    'correlation_risk': metrics.correlation_risk,
                    'model_confidence': metrics.model_confidence,
                    'cointegration_stability': metrics.cointegration_stability
                }

                if pair in active_positions:
                    position = active_positions[pair]
                    current_price = current_prices[pair[0]]

                    entry_value = position['quantity'] * position['entry_price']
                    current_value = position['quantity'] * current_price
                    pnl = current_value - entry_value
                    logger.info(f'Unrealized/Active PnL: {pnl}')

                    if (metrics.var_95 > risk_manager.stop_loss_threshold or
                            metrics.correlation_risk > risk_manager.max_correlation or
                            metrics.model_confidence < risk_manager.min_model_confidence):
                        positions_to_close.append((pair, "Individual position risk limit exceeded"))

            portfolio_value = sum(
                active_positions[pair]['quantity'] * current_prices[pair[0]]
                for pair in active_positions
            )

            exceeded, reason = risk_manager.check_risk_limits(
                equity_curve=pd.Series(portfolio_value),
                positions=active_positions,
                current_prices=current_prices.to_dict()
            )

            if exceeded:
                for pair in list(active_positions.keys()):
                    if pair not in [p[0] for p in positions_to_close]:
                        positions_to_close.append((pair, f"Portfolio risk limits exceeded: {reason}"))

            for pair, close_reason in positions_to_close:
                position = active_positions[pair]
                current_price = current_prices[pair[0]]

                entry_value = position['quantity'] * position['entry_price']
                exit_value = position['quantity'] * current_price
                pnl = exit_value - entry_value

                trade_history.append({
                    'pair': f"{pair[0]}/{pair[1]}",
                    'entry_date': position['entry_date'].strftime('%Y-%m-%d'),
                    'exit_date': current_date.strftime('%Y-%m-%d'),
                    'entry_price': position['entry_price'],
                    'exit_price': current_price,
                    'quantity': position['quantity'],
                    'pnl': pnl,
                    'close_reason': close_reason
                })

                del active_positions[pair]
                logger.warning(f"Closed position for {pair}: {close_reason}")

            daily_status = {
                'date': current_date.strftime('%Y-%m-%d'),
                'risk_limits_exceeded': exceeded,
                'reason': reason if exceeded else None,
                'portfolio_value': portfolio_value,
                'active_positions': len(active_positions),
                'metrics': risk_metrics.copy()
            }
            risk_history.append(daily_status)

        except Exception as e:
            logger.error(f"Error processing date {current_date}: {str(e)}")
            continue

    logger.info("Saving risk analysis results...")

    with open(results_dir / "trade_history.json", "w") as f:
        json.dump(trade_history, f, indent=4)

    with open(results_dir / "risk_history.json", "w") as f:
        json.dump(risk_history, f, indent=4)

    with open(results_dir / "final_risk_metrics.json", "w") as f:
        json.dump(risk_metrics, f, indent=4)

    logger.info("Creating risk visualization dashboard...")
    fig = risk_manager.plot_risk_metrics()
    fig.write_html(str(plots_dir / "risk_dashboard.html"))

    risk_df = pd.DataFrame(risk_history)
    fig_timeseries = go.Figure()

    fig_timeseries.add_trace(
        go.Scatter(
            x=risk_df['date'],
            y=risk_df['portfolio_value'],
            name='Portfolio Value'
        )
    )

    first_pair = list(risk_df['metrics'].iloc[0].keys())[0]
    for metric in ['var_95', 'cvar_95', 'volatility']:
        fig_timeseries.add_trace(
            go.Scatter(
                x=risk_df['date'],
                y=[day['metrics'][first_pair][metric] for day in risk_history],
                name=f'{metric} ({first_pair})'
            )
        )

    for trade in trade_history:
        fig_timeseries.add_trace(
            go.Scatter(
                x=[trade['exit_date']],
                y=[trade['pnl']],
                mode='markers',
                marker=dict(
                    symbol='x',
                    size=10,
                    color='red'
                ),
                name=f"Position Closed: {trade['close_reason']}"
            )
        )

    fig_timeseries.update_layout(
        title='Risk Metrics and Trading Activity Over Time',
        xaxis_title='Date',
        yaxis_title='Value',
        height=800
    )

    fig_timeseries.write_html(str(plots_dir / "risk_metrics_timeseries.html"))

    final_status = {
        'risk_limits_exceeded': risk_history[-1]['risk_limits_exceeded'],
        'reason': risk_history[-1]['reason'],
        'data_period': {
            'start': prices.index[0].strftime('%Y-%m-%d'),
            'end': prices.index[-1].strftime('%Y-%m-%d'),
            'symbols': list(prices.columns)
        },
        'total_risk_events': sum(1 for day in risk_history if day['risk_limits_exceeded']),
        'total_trades_closed': len(trade_history),
        'final_portfolio_value': portfolio_value,
        'active_positions_remaining': len(active_positions)
    }

    with open(results_dir / "risk_status.json", "w") as f:
        json.dump(final_status, f, indent=4)

    logger.info("Risk analysis completed successfully")
    return {
        'risk_metrics': risk_metrics,
        'risk_status': final_status,
        'risk_history': risk_history,
        'trade_history': trade_history
    }


if __name__ == "__main__":
    try:
        logger.info("Starting comprehensive testing of all modules")

        backtest_results = main_backtest()
        print("\nBacktest completed successfully!")

        optimization_results = main_optimization()
        print("\nOptimization completed successfully!")

        risk_results = main_risk()
        print("\nRisk analysis completed successfully!")

    except Exception as e:
        logger.error(f"Error in main execution: {str(e)}")
        import traceback

        logger.error(traceback.format_exc())
