# Advanced Pairs Trading System

An end-to-end algorithmic trading platform for researching, backtesting, and deploying pairs trading strategies. The system features a sophisticated Streamlit-based web interface, supports multiple strategy paradigms from statistical to deep learning models, and is designed for robust quantitative analysis.

## 🏛️ System Architecture

The system is designed with a modular architecture to separate concerns and improve maintainability:

- **UI Layer (`streamlit_system/`):** An interactive web interface built with Streamlit for managing the entire workflow without writing code.
- **Strategy & Backtesting Layer (`src/strategy/`):** Contains the logic for various trading strategies, a high-fidelity backtesting engine, and risk management modules.
- **Quantitative Analysis Layer (`src/analysis/`):** A suite of tools for statistical analysis, including pair selection through correlation, cointegration, and clustering.
- **Modeling Layer (`src/models/`):** Implements the statistical, machine learning, and deep learning models that drive the trading strategies.
- **Data Layer (`src/data/`):** Handles data acquisition from APIs and files, preprocessing, feature engineering, and database management.
- **Execution Layer (`execution/`):** Provides connectivity to brokers (e.g., Alpaca) for paper and live trading.

---

## 🌟 Key Features

- **Interactive Web Interface:**
  - Intuitive workflow from data loading to optimization.
  - Real-time strategy performance visualization.
  - Interactive parameter tuning and analysis.
- **Diverse Trading Strategies:**
  - Statistical (Cointegration, Z-Score)
  - Machine Learning (Random Forest, Gradient Boosting)
  - Deep Learning (TCN, TimesNet)
  - Dynamic pair selection and rotation.
- **Comprehensive Analysis Tools:**
  - Automated pair selection using cointegration, correlation, and clustering.
  - Advanced covariance estimation and data denoising techniques.
  - In-depth risk and performance analytics.
- **Robust Backtesting & Optimization:**
  - High-fidelity backtester with transaction cost and slippage modeling.
  - Advanced optimization suite (Bayesian, Walk-Forward) via Optuna.
- **Paper & Live Trading Ready:**
  - Includes modules for connecting to brokerage APIs like Alpaca for paper trading.

---

## 🚀 Quick Start

### Prerequisites

- Python 3.9 or higher
- Git

### Installation

1.  **Clone the Repository**
    ```bash
    git clone [https://github.com/AJAllProjects/Pairs_Trading_System.git](https://github.com/AJAllProjects/Pairs_Trading_System.git)
    cd Pairs_Trading_System
    ```

2.  **Create Virtual Environment**
    ```bash
    # It is highly recommended to use a virtual environment
    python3 -m venv venv
    source venv/bin/activate  # On macOS/Linux
    # venv\Scripts\activate  # On Windows
    ```

3.  **Install Dependencies**
    ```bash
    pip install -r requirements.txt
    ```
    
4. **Setup Environment Variables**
    - Create a file named `.env` in the project root.
    - Copy the contents of `.env.example` into it and fill in your Alpaca paper trading API keys.

5.  **Set Up the Database**
    The database is created automatically on first run. To reset it, you can delete the `pair_trading.db` file.

### 🖥️ Running the System

1.  **Start the Streamlit Interface**
    ```bash
    streamlit run streamlit_system/app.py
    ```

2.  **Access the Dashboard**
    - Open your browser and navigate to `http://localhost:8501`.

---

## 📊 Using the Platform

The platform is designed around a sequential workflow, accessible from the sidebar:

1.  **Data Loading:** Upload your own CSV files or download market data directly from sources like Yahoo Finance.
2.  **Pair Analysis:** Use correlation, cointegration, and clustering tools to identify promising pairs for trading.
3.  **Strategy Builder:** Select a strategy type (e.g., Statistical, ML), configure its parameters, and set risk management rules.
4.  **Backtest:** Run the configured strategy on the historical data to evaluate its performance.
5.  **Optimization:** Use advanced methods like Bayesian optimization to find the best parameters for your strategy.

---

## 🛠️ Configuration

- **Database Settings:** Edit `config/settings.py` or set the `DATABASE_URL` environment variable.
- **Logging:** Configure logging levels and formats in `config/logging_config.py`.
- **API Keys:** Store all secret keys in the `.env` file.

---

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1.  Fork the repository.
2.  Create a new feature branch (`git checkout -b feature/AmazingFeature`).
3.  Commit your changes (`git commit -m 'Add some AmazingFeature'`). We follow the PEP 8 style guide.
4.  Push to the branch (`git push origin feature/AmazingFeature`).
5.  Open a Pull Request.

---

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 📧 Contact

Arnav Jain - ajain.careers@gmail.com

Project Link: [https://github.com/AJAllProjects/Pairs_Trading_System](https://github.com/AJAllProjects/Pairs_Trading_System)