# Equity Pair Trading Research Project

## Overview

This project focuses on developing and backtesting an equity pair trading strategy using statistical and machine learning methods. It involves data acquisition, preprocessing, pair selection, strategy development, backtesting, performance evaluation, and interactive analysis through a Streamlit dashboard.

## Project Structure

```
pair_trading/
├── README.md
├── requirements.txt
├── setup.py
├── .gitignore
├── config/
│   ├── __init__.py
│   ├── settings.py
│   └── logging_config.py
├── src/
│   ├── __init__.py
│   ├── data/
│   │   ├── __init__.py
│   │   ├── downloader.py
│   │   ├── database.py
│   │   ├── preprocessor.py
│   │   ├── feature_engineering.py
│   │   └── live_data.py
│   ├── analysis/
│   │   ├── __init__.py
│   │   ├── clustering_analysis.py
│   │   ├── cointegration.py
│   │   ├── correlation_analysis.py
│   │   ├── covariance_estimation.py
│   │   └── denoiser_usage.py
│   ├── strategy/
│   │   ├── __init__.py
│   │   ├── base.py
│   │   ├── pairs_strategy_basic.py
│   │   ├── pairs_strategy_DL.py
│   │   ├── pairs_strategy_ML.py
│   │   ├── pairs_strategy_SL.py
│   │   ├── risk.py
│   │   ├── optimization.py
│   │   └── backtest.py
│   ├── models/
│   │   ├── __init__.py
│   │   ├── statistical.py
│   │   ├── machine_learning.py
│   │   └── deep_learning.py
│   └── utils/
│       ├── __init__.py
│       ├── metrics.py
│       ├── visualization.py
│       ├── validation.py
│       └── parallel_training.py
├── streamlit/
│   ├── app.py
│   └── components/
│       ├── __init__.py
│       ├── data_loader.py
│       ├── pair_analyzer.py
│       ├── strategy_builder.py
│       └── optimization.py
└── execution/
    ├── __init__.py
    └── broker.py
```

## Setup Instructions

1. **Clone the Repository**
   ```bash
   git clone https://github.com/yourusername/pair_trading.git
   cd pair_trading
   ```

2. **Create a Virtual Environment**
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

5. **Launch Streamlit Dashboard**
   ```bash
   streamlit run streamlit/app.py
   ```

6. **Open Jupyter Notebooks**
   ```bash
   jupyter notebook notebooks/
   ```

## Usage

- **Data Loader**: Download and preprocess market data for selected tickers.

- **Pair Analyzer**: Analyze correlations, partial correlations, mean reversion speeds, and use advanced methods like Graph-Based Similarity, DTW, and PCA-based selection to identify highly correlated pairs. Incorporates multiple denoising techniques to stabilize correlation estimates.

- **Strategy Builder**: Configure and build trading strategies with real-time parameter tuning and machine learning integration.

- **Optimization**: Perform strategy parameter optimization using Grid Search or Bayesian Optimization (Optuna) to maximize selected performance metrics.

- **Performance**: Visualize and evaluate the performance of backtested strategies, including comprehensive metrics and interactive plots.

- **Model Performance**: Assess the performance of machine learning models used for spread prediction, including feature importances and training histories.

- **Real-time Monitoring**: Monitor current open positions, view Unrealized PnL, and observe live spread predictions.

## Project Goals

- **Data Acquisition**: Fetch end-of-day and live intraday data for selected indices and securities.

- **Pair Selection**: Identify highly correlated and mean-reverting pairs using advanced clustering algorithms, similarity measures, and denoising techniques.

- **Strategy Development**: Implement mean-reversion trading strategies with configurable parameters and machine learning-based spread prediction.

- **Backtesting**: Simulate strategy performance over historical data with enhanced risk management.

- **Performance Evaluation**: Analyze metrics like Sharpe Ratio, Drawdown, Profit Factor, etc., and evaluate model performance.

- **Interactive Analysis**: Use Streamlit for interactive experimentation with strategy parameters and real-time tuning.

- **Parallel Processing**: Utilize parallel computing to speed up backtesting and model training.

- **Live Predictions**: Integrate live data feeds and display real-time spread movement predictions.

## Future Enhancements

- **Advanced Pair Selection**: Implement additional clustering algorithms or similarity measures for more robust pair selection.

- **Hyperparameter Tuning**: Integrate more sophisticated hyperparameter optimization techniques like Bayesian Optimization.

- **Risk Management**: Incorporate more advanced risk management strategies, including portfolio optimization and dynamic hedging.

- **Deployment**: Deploy the Streamlit dashboard to a cloud platform with user authentication for broader accessibility.

- **Automated Trading**: Extend the backtester to interface with actual trading APIs for live execution.

## Contributing

Contributions are welcome! Please fork the repository and submit a pull request for any enhancements or bug fixes.

## License

This project is licensed under the MIT License.