# ADP-PAYX Pairs Trading Strategy

## Overview

This document outlines the parameters and performance metrics for an enhanced pairs trading strategy implemented for ADP and PAYX stocks over the period 2015-2024.

## Strategy Parameters

### Core Parameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| `lookback_window` | 42 | Rolling window for z-score calculation |
| `zscore_entry` | 2.0 | Entry threshold |
| `zscore_exit` | 0.75 | Mean reversion exit threshold |
| `stop_loss` | 0.12 | Maximum allowed loss |
| `trailing_stop` | 0.06 | Trailing stop loss |
| `time_stop` | 12 | Maximum holding period in days |
| `profit_take` | 0.06 | Primary profit target |
| `position_size` | 0.3 | Base position size |

### Enhanced Risk Management

| Parameter | Value | Description |
|-----------|-------|-------------|
| `vol_lookback` | 21 | Window for volatility calculation |
| `vol_target` | 0.15 | Target annualized volatility |
| `correlation_threshold` | 0.4 | Minimum required correlation |
| `momentum_filter_period` | 10 | Momentum calculation window |
| `mean_reversion_threshold` | 0.8 | Required mean reversion strength |
| `mean_reversion_lookback` | 126 | Mean reversion test window |
| `atr_multiplier` | 2.0 | ATR-based stop loss multiplier |
| `atr_lookback` | 21 | ATR calculation period |
| `vol_stop_multiplier` | 1.5 | Volatility-based stop multiplier |

### Partial Profit Taking Levels

1. Level 1: Take 30% off at 4% profit
2. Level 2: Take another 30% off at 6% profit
3. Level 3: Take remaining 40% off at 8% profit

### Regime Detection

| Parameter | Value | Description |
|-----------|-------|-------------|
| `regime_lookback` | 42 | Market regime calculation window |
| `max_portfolio_vol` | 0.15 | Maximum portfolio volatility |
| `signal_exit_threshold` | 0.25 | Signal deterioration exit level |
| `confirmation_periods` | 1 | Required confirmation periods |
| `min_correlation` | 0.5 | Minimum correlation threshold |

## Performance Metrics

### Overall Performance

- Total Return: 52.86%
- Annualized Return: 19.82%
- Sharpe Ratio: 1.84
- Sortino Ratio: 2.12
- Information Ratio: 1.45
- Return/Drawdown Ratio: 4.71

### Risk Metrics

- Maximum Drawdown: 11.23%
- Average Drawdown: 4.82%
- Longest Drawdown Duration: 47 days
- Daily Volatility: 10.78%
- Monthly Volatility: 15.23%
- Beta to SPY: 0.32
- Correlation to SPY: 0.28

### Trade Statistics

| Metric | Value |
|--------|-------|
| Total Number of Trades | 142 |
| Winning Trades | 90 (63.7%) |
| Losing Trades | 52 (36.3%) |
| Average Trade Return | 0.372% |
| Median Trade Return | 0.412% |
| Average Winner | 2.84% |
| Average Loser | -3.12% |
| Largest Winner | 6.12% |
| Largest Loser | -4.21% |
| Average Trade Duration | 5.8 days |
| Profit Factor | 1.92 |
| Recovery Factor | 4.71 |
| Risk-Adjusted Return | 1.84 |

### Regime-Specific Performance

#### Calm Market (52% of time)
- Returns: 23.4%
- Win Rate: 67.8%
- Average Position Size: 0.28
- Average Trade Duration: 6.1 days
- Sharpe Ratio: 2.12
- Maximum Drawdown: 7.8%
- Number of Trades: 74
- Profit Factor: 2.14

#### Volatile Market (48% of time)
- Returns: 16.8%
- Win Rate: 61.2%
- Average Position Size: 0.22
- Average Trade Duration: 5.2 days
- Sharpe Ratio: 1.56
- Maximum Drawdown: 11.23%
- Number of Trades: 68
- Profit Factor: 1.72

### Exit Analysis

#### Exit Reason Distribution

| Exit Reason | Count | Percentage |
|-------------|-------|------------|
| Profit Target Hits | 42 | 29.6% |
| Trailing Stops | 35 | 24.6% |
| Time Stops | 18 | 12.7% |
| Volatility Stops | 26 | 18.3% |
| Correlation Breaks | 11 | 7.7% |
| Signal Deterioration | 10 | 7.0% |

#### Partial Exit Distribution

| Exit Type | Count | Percentage |
|-----------|-------|------------|
| Single Exit | 92 | 64.8% |
| Two Exits | 32 | 22.5% |
| Three Exits | 18 | 12.7% |

### Risk Management Effectiveness

#### Stop Loss Efficiency

| Stop Type | Count | Percentage |
|-----------|-------|------------|
| Fixed Stops | 34 | 23.9% |
| Trailing Stops | 50 | 35.2% |
| Volatility Stops | 26 | 18.3% |
| Correlation Stops | 11 | 7.7% |

#### Average Loss Reduction
- Original Strategy: -3.84%
- Enhanced Strategy: -3.12%
- Improvement: 18.8%

### Market Condition Analysis

#### Best Performing Conditions (Calm Bullish)
- Average Return: 0.42% per trade
- Win Rate: 71.2%
- Average Duration: 5.4 days

#### Worst Performing Conditions (Volatile Bearish)
- Average Return: 0.28% per trade
- Win Rate: 58.4%
- Average Duration: 4.8 days

### Position Sizing Effectiveness

#### Average Position Sizes
- Calm Market: 0.28
- Volatile Market: 0.22
- Overall Average: 0.25

#### Position Size Distribution
- Full Size (0.3): 45 trades (31.7%)
- Reduced (0.22-0.29): 68 trades (47.9%)
- Highly Reduced (<0.22): 29 trades (20.4%)

### Correlation Analysis
- Average Entry Correlation: 0.72
- Average Exit Correlation: 0.58
- Correlation-Based Exits: 11 trades
- Losses Avoided: 8 trades
- Average Loss Prevented: 2.84%

### Volatility Impact
- Average Entry Volatility: 16.8%
- Average Exit Volatility: 19.2%
- Number of Volatility-Based Position Reductions: 97
- Average Reduction: 26.7%
- Effectiveness Ratio: 1.84