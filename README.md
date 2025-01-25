# Advanced Pairs Trading System

## Overview

An end-to-end pairs trading platform that enables users to analyze, test, and implement various trading strategies through an intuitive web interface. The system supports multiple strategy types from basic statistical approaches to advanced machine learning models.

## 🌟 Key Features

- **Interactive Web Interface**
  - No coding required for basic usage
  - Real-time strategy visualization
  - Interactive parameter tuning
  - Live performance monitoring

- **Multiple Trading Strategies**
  - Statistical pairs trading
  - Machine learning-based predictions
  - Deep learning with LSTM networks
  - Basic pairs trading for comparison

- **Comprehensive Analysis Tools**
  - Automated pair selection
  - Risk analytics
  - Performance metrics
  - Portfolio optimization

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- SQLite (included in Python)
- Git

### Installation

1. **Clone the Repository**
```bash
git clone https://github.com/yourusername/pair_trading.git
cd pair_trading
```

2. **Create Virtual Environment**
```bash
# On Windows
python -m venv venv
venv\Scripts\activate

# On macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

3. **Install Dependencies**
```bash
pip install -r requirements.txt
```

4. **Set Up the Database**
```bash
python src/data/database.py
```

### 🖥️ Running the System

1. **Start the Streamlit Interface**
```bash
streamlit run streamlit_system/app.py
```

2. **Access the Dashboard**
- Open your browser and go to `http://localhost:8501`
- The system will automatically open your default browser

## 📊 Using the Platform

### 1. Data Loading
- Click on "Data Loading" in the sidebar
- Choose from:
  - Upload your own CSV files
  - Download market data directly
  - Query from database
- Required CSV format:
  ```
  Date,Symbol,Open,High,Low,Close,Adj_Close,Volume
  2024-01-01,AAPL,100,101,99,100.5,100.5,1000000
  ```

### 2. Pair Analysis
- Navigate to "Pair Analysis" section
- Select analysis method:
  - Correlation analysis
  - Cointegration testing
  - Clustering analysis
- Configure parameters and run analysis
- Review and select pairs for trading

### 3. Strategy Building
- Go to "Strategy Builder"
- Choose strategy type:
  - Statistical (best for beginners)
  - Machine Learning
  - Deep Learning
- Configure parameters:
  - Entry/exit thresholds
  - Stop loss/take profit
  - Position sizing
- Run backtests and analyze results

### 4. Optimization
- Use "Optimization" section
- Define parameter ranges
- Select optimization method:
  - Grid Search
  - Bayesian Optimization
- Review results and apply optimized parameters

## 📈 Example Workflow

1. **Load Data**
```python
# Using Python API
from src.data.downloader import DataDownloader

downloader = DataDownloader()
data = downloader.download_historical_data(
    symbols=['AAPL', 'MSFT'],
    years_back=2
)
```

2. **Analyze Pairs**
```python
from src.analysis.correlation_analysis import CorrelationAnalyzer

analyzer = CorrelationAnalyzer(data)
correlated_pairs = analyzer.find_pairs(
    correlation_threshold=0.8
)
```

3. **Build Strategy**
```python
from src.strategy.pairs_strategy_SL import PairsTraderSL

strategy = PairsTraderSL(
    lookback_period=20,
    zscore_threshold=2.0
)
```

4. **Run Backtest**
```python
from src.strategy.backtest import Backtester

backtester = Backtester(
    strategy=strategy,
    data=data,
    initial_capital=100000
)
results = backtester.run()
```

## 🛠️ Configuration

### Database Settings
Edit `config/settings.py`:
```python
DATABASE_URI = "sqlite:///pair_trading.db"
```

### Logging Configuration
Edit `config/logging_config.py`:
```python
LOG_LEVEL = logging.INFO
LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
```

### Strategy Parameters
Default parameters in respective strategy files:
- `src/strategy/pairs_strategy_SL.py`
- `src/strategy/pairs_strategy_ML.py`
- `src/strategy/pairs_strategy_DL.py`

## 🔧 Troubleshooting

### Common Issues

1. **Streamlit Not Starting**
   ```bash
   # Check Python version
   python --version  # Should be 3.8+
   
   # Reinstall streamlit
   pip uninstall streamlit
   pip install streamlit
   ```

2. **Data Loading Errors**
   - Ensure CSV files have correct headers
   - Check date format (YYYY-MM-DD)
   - Verify no missing values in required columns

3. **Database Connection Issues**
   ```bash
   # Reset database
   rm pair_trading.db
   python src/data/database.py
   ```

### Getting Help
- Check the logs in `logs/pair_trading.log`
- Create an issue on GitHub
- Review related documentation

## 🔄 Regular Updates

1. **Update Repository**
```bash
git pull origin main
pip install -r requirements.txt
```

2. **Update Database Schema**
```bash
python src/data/database.py --update
```

## 🎯 Future Enhancements

### Planned Features
- Real-time trading integration
- Additional strategy types
- Enhanced optimization methods
- Mobile app interface
- API access
- Market sentiment analysis
- Alternative data integration
- Custom strategy builder
- Advanced portfolio management
- Multi-asset class support

### Development Roadmap
1. Q2 2024
   - API integration
   - Real-time trading support
   - Enhanced ML models

2. Q3 2024
   - Mobile application
   - Advanced optimization
   - Custom indicators

3. Q4 2024
   - Multi-asset support
   - Portfolio optimization
   - Risk management tools

## 🤝 Contributing

1. Fork the repository
2. Create feature branch
   ```bash
   git checkout -b feature/AmazingFeature
   ```
3. Commit changes
   ```bash
   git commit -m 'Add AmazingFeature'
   ```
4. Push to branch
   ```bash
   git push origin feature/AmazingFeature
   ```
5. Create Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details.

## 🙏 Acknowledgments

- [Streamlit](https://streamlit.io/) for the amazing web framework
- [yfinance](https://github.com/ranaroussi/yfinance) for market data access
- [scikit-learn](https://scikit-learn.org/) for machine learning capabilities
- [TensorFlow](https://www.tensorflow.org/) for deep learning implementation

## 📧 Contact

Your Name - your.email@example.com

Project Link: [https://github.com/yourusername/pair_trading](https://github.com/yourusername/pair_trading)