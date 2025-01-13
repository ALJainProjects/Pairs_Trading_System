"""
Consolidated Correlation Analysis Module

This module provides functionality for:
1. Standard Pearson correlation analysis
2. Partial correlation analysis
3. Rolling correlation analysis
4. Visualization tools
5. Statistical analysis and hypothesis testing
"""

from __future__ import annotations
import pandas as pd
import numpy as np
from scipy.linalg import pinv
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats
from typing import Dict, List, Tuple, Optional, Union, Any
import os
from pathlib import Path
import warnings
from dataclasses import dataclass
from datetime import datetime
import logging
from functools import lru_cache

from statsmodels.stats.multitest import multipletests

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
@dataclass
class CorrelationConfig:
    """Configuration for correlation analysis."""
    CORRELATION_THRESHOLDS: Dict[str, float] = None
    MIN_OBSERVATIONS: int = 30  # Minimum sample size for statistical significance
    DEFAULT_WINDOW: int = 63
    MAX_ASSETS_DISPLAY: int = 50
    TIMEOUT_SECONDS: int = 300

    def __post_init__(self):
        if self.CORRELATION_THRESHOLDS is None:
            self.CORRELATION_THRESHOLDS = {
                'very_high': 0.9,
                'high': 0.7,
                'moderate': 0.5
            }

class DataValidationError(Exception):
    """Custom exception for data validation errors."""
    pass

class CalculationError(Exception):
    """Custom exception for calculation errors."""
    pass

def validate_pair_name(asset1: str, asset2: str) -> str:
    """Create a unique and safe pair identifier."""
    return f"{asset1}|||{asset2}"  # Using triple pipe as unlikely delimiter

def parse_pair_name(pair_name: str) -> Tuple[str, str]:
    """Parse pair identifier back into asset names."""
    return tuple(pair_name.split('|||'))

class CorrelationAnalyzer:
    """Class for comprehensive correlation analysis."""

    def __init__(self, returns: pd.DataFrame, config: Optional[CorrelationConfig] = None):
        """
        Initialize correlation analyzer.

        Args:
            returns (pd.DataFrame): Asset returns with datetime index
            config (Optional[CorrelationConfig]): Configuration parameters

        Raises:
            DataValidationError: If input data validation fails
        """
        self.config = config or CorrelationConfig()
        self._validate_input(returns)
        self.returns = returns
        self.pearson_corr_: Optional[pd.DataFrame] = None
        self.partial_corr_: Optional[pd.DataFrame] = None
        self.rolling_corr_: Optional[Dict[str, pd.Series]] = None
        self.rolling_window_: Optional[int] = None
        self._last_update = datetime.now()

    def _validate_input(self, returns: pd.DataFrame) -> None:
        """
        Validate input data.

        Raises:
            DataValidationError: If validation fails
        """
        if not isinstance(returns, pd.DataFrame):
            raise DataValidationError("Input must be a pandas DataFrame")

        if not isinstance(returns.index, pd.DatetimeIndex):
            raise DataValidationError("DataFrame must have DatetimeIndex")

        if returns.empty:
            raise DataValidationError("Empty returns DataFrame")

        if returns.isnull().any().any():
            raise DataValidationError("Returns contain missing values")

        if len(returns) < self.config.MIN_OBSERVATIONS:
            raise DataValidationError(
                f"Insufficient observations. Need at least {self.config.MIN_OBSERVATIONS}"
            )

        # Check for constant columns
        zero_var_cols = returns.columns[returns.std() == 0]
        if not zero_var_cols.empty:
            raise DataValidationError(
                f"Constant columns detected: {', '.join(zero_var_cols)}"
            )

    @lru_cache(maxsize=1)
    def calculate_pearson_correlation(self) -> pd.DataFrame:
        """
        Calculate Pearson correlation matrix.

        Returns:
            pd.DataFrame: Correlation matrix

        Raises:
            CalculationError: If calculation fails
        """
        try:
            logger.info("Calculating Pearson correlation matrix.")
            self.pearson_corr_ = self.returns.corr(method="pearson")
            return self.pearson_corr_
        except Exception as e:
            raise CalculationError(f"Failed to calculate Pearson correlation: {str(e)}")

    def calculate_partial_correlation(self) -> pd.DataFrame:
        """
        Calculate partial correlation matrix.

        Returns:
            pd.DataFrame: Partial correlation matrix

        Raises:
            CalculationError: If calculation fails
        """
        try:
            logger.info("Calculating partial correlation matrix.")

            # Calculate covariance and its pseudo-inverse for better numerical stability
            cov = self.returns.cov().values
            prec = pinv(cov)

            # Calculate partial correlations
            d = np.sqrt(np.diag(prec))
            d_outer = np.outer(d, d)

            # Avoid division by zero
            mask = d_outer != 0
            partial_corr = np.zeros_like(prec)
            partial_corr[mask] = -prec[mask] / d_outer[mask]
            np.fill_diagonal(partial_corr, 1.0)

            self.partial_corr_ = pd.DataFrame(
                partial_corr,
                index=self.returns.columns,
                columns=self.returns.columns
            )
            return self.partial_corr_

        except Exception as e:
            raise CalculationError(f"Failed to calculate partial correlation: {str(e)}")

    def calculate_rolling_correlation(self, window: Optional[int] = None) -> Dict[str, pd.Series]:
        """
        Calculate rolling correlations for all pairs.

        Args:
            window (Optional[int]): Rolling window size in days

        Returns:
            Dict[str, pd.Series]: Dictionary of rolling correlations for each pair

        Raises:
            CalculationError: If calculation fails
        """
        try:
            window = window or self.config.DEFAULT_WINDOW
            logger.info(f"Calculating rolling correlations with window={window}")

            if (self.rolling_corr_ is not None and
                self.rolling_window_ == window):
                return self.rolling_corr_

            rolling_corrs = {}
            n_assets = len(self.returns.columns)

            # Pre-calculate all combinations to avoid nested loops
            combinations = [
                (i, j) for i in range(n_assets)
                for j in range(i + 1, n_assets)
            ]

            for i, j in combinations:
                asset1, asset2 = self.returns.columns[i], self.returns.columns[j]
                pair_name = validate_pair_name(asset1, asset2)

                rolling_corr = self.returns[asset1].rolling(
                    window=window,
                    min_periods=self.config.MIN_OBSERVATIONS
                ).corr(self.returns[asset2])

                rolling_corrs[pair_name] = rolling_corr

            self.rolling_corr_ = rolling_corrs
            self.rolling_window_ = window
            return rolling_corrs

        except Exception as e:
            raise CalculationError(f"Failed to calculate rolling correlation: {str(e)}")

    def get_highly_correlated_pairs(self,
                                  correlation_type: str = 'pearson',
                                  threshold: Optional[float] = None,
                                    absolute: bool = False) -> pd.DataFrame:
        """
        Find highly correlated pairs.

        Args:
            correlation_type (str): 'pearson' or 'partial'
            threshold (Optional[float]): Minimum absolute correlation

        Returns:
            pd.DataFrame: Highly correlated pairs with their correlations

        Raises:
            ValueError: If invalid correlation type
            CalculationError: If calculation fails
        """
        threshold = threshold or self.config.CORRELATION_THRESHOLDS['high']

        try:
            if correlation_type == 'pearson':
                if self.pearson_corr_ is None:
                    self.calculate_pearson_correlation()
                corr_matrix = self.pearson_corr_
            elif correlation_type == 'partial':
                if self.partial_corr_ is None:
                    self.calculate_partial_correlation()
                corr_matrix = self.partial_corr_
            else:
                raise ValueError("correlation_type must be 'pearson' or 'partial'")

            # Vectorized operations for better performance
            mask = np.triu(np.abs(corr_matrix) >= threshold, k=1)
            pairs = []

            rows, cols = np.where(mask)
            for i, j in zip(rows, cols):
                pairs.append({
                    'asset1': corr_matrix.columns[i],
                    'asset2': corr_matrix.columns[j],
                    'correlation': corr_matrix.iloc[i, j]
                })

            pairs_df = pd.DataFrame(pairs).sort_values('correlation', ascending=False, key=np.abs if absolute else None)

            return pairs_df

        except Exception as e:
            raise CalculationError(f"Failed to get highly correlated pairs: {str(e)}")

    def analyze_correlation_stability(self,
                                   window: Optional[int] = None) -> pd.DataFrame:
        """
        Analyze stability of correlations over time.

        Args:
            window (Optional[int]): Rolling window size

        Returns:
            pd.DataFrame: Stability metrics for each pair

        Raises:
            CalculationError: If calculation fails
        """
        try:
            if self.rolling_corr_ is None:
                self.calculate_rolling_correlation(window)

            stability_metrics = {}
            for pair, rolling_corr in self.rolling_corr_.items():
                asset1, asset2 = parse_pair_name(pair)

                # Calculate metrics with handling for NaN values
                metrics = {
                    'asset1': asset1,
                    'asset2': asset2,
                    'mean': rolling_corr.mean(),
                    'std': rolling_corr.std(),
                    'min': rolling_corr.min(),
                    'max': rolling_corr.max(),
                    'negative_pct': (rolling_corr < 0).mean() * 100,
                    'missing_pct': rolling_corr.isna().mean() * 100
                }
                stability_metrics[pair] = metrics

            return pd.DataFrame(stability_metrics).T

        except Exception as e:
            raise CalculationError(f"Failed to analyze correlation stability: {str(e)}")

    def determine_correlation_significance(self,
                                   correlation_type: str = 'pearson',
                                   alpha: float = 0.05, method: str = 'bonferroni') -> pd.DataFrame:
        """
        Test significance of correlations.

        Args:
            correlation_type (str): 'pearson' or 'partial'
            alpha (float): Significance level
            adjust_pvalues (bool): Whether to apply multiple testing correction

        Returns:
            pd.DataFrame: P-values for correlation tests

        Raises:
            ValueError: If invalid correlation type
            CalculationError: If calculation fails
        """
        try:
            if correlation_type == 'pearson':
                if self.pearson_corr_ is None:
                    self.calculate_pearson_correlation()
                corr_matrix = self.pearson_corr_
            elif correlation_type == 'partial':
                if self.partial_corr_ is None:
                    self.calculate_partial_correlation()
                corr_matrix = self.partial_corr_
            else:
                raise ValueError("correlation_type must be 'pearson' or 'partial'")

            n = len(self.returns)
            pvalues = pd.DataFrame(
                np.zeros_like(corr_matrix),
                index=corr_matrix.index,
                columns=corr_matrix.columns
            )

            # Vectorized calculation of t-statistics and p-values
            corr_values = corr_matrix.values
            # Avoid division by zero
            denom = np.sqrt(1 - corr_values**2)
            denom[denom == 0] = np.inf

            t_stat = corr_values * np.sqrt((n - 2) / denom)
            pvalues_array = 2 * (1 - stats.t.cdf(np.abs(t_stat), n - 2))

            pvalues.values[:] = pvalues_array
            np.fill_diagonal(pvalues.values, 1.0)

            if method == 'bonferroni':
                reject, pvals_corrected, _, _ = multipletests(pvalues, alpha=alpha, method='bonferroni')
            elif method == 'holm':
                reject, pvals_corrected, _, _ = multipletests(pvalues, alpha=alpha, method='holm')
            elif method == 'fdr_bh':
                reject, pvals_corrected, _, _ = multipletests(pvalues, alpha=alpha, method='fdr_bh')
            else:
                raise ValueError(f"Unsupported multiple testing correction: {method}")

            return pvals_corrected

        except Exception as e:
            raise CalculationError(f"Failed to test correlation significance: {str(e)}")

    def plot_correlation_matrix(self,
                              correlation_type: str = 'pearson',
                              title: Optional[str] = None,
                              colormap: Optional[str] = None,
                              size: Optional[Tuple[int, int]] = None) -> go.Figure:
        """Plot correlation matrix heatmap."""
        try:
            if correlation_type == 'pearson':
                if self.pearson_corr_ is None:
                    self.calculate_pearson_correlation()
                corr_matrix = self.pearson_corr_
                title = title or 'Pearson Correlation Matrix'
            elif correlation_type == 'partial':
                if self.partial_corr_ is None:
                    self.calculate_partial_correlation()
                corr_matrix = self.partial_corr_
                title = title or 'Partial Correlation Matrix'
            else:
                raise ValueError("correlation_type must be 'pearson' or 'partial'")

            # Handle large number of assets
            if len(corr_matrix.columns) > self.config.MAX_ASSETS_DISPLAY:
                warnings.warn(
                    f"Large number of assets ({len(corr_matrix.columns)}). "
                    f"Consider reducing for better visualization."
                )

            size = size or (1000, 1000)

            fig = go.Figure(data=go.Heatmap(
                z=corr_matrix,
                x=corr_matrix.columns,
                y=corr_matrix.columns,
                colorscale=colormap or 'RdBu',
                zmid=0,
                showscale=True
            ))

            fig.update_layout(
                title=title,
                xaxis_title="Assets",
                yaxis_title="Assets",
                width=size[0],
                height=size[1],
                xaxis={'tickangle': 45}
            )

            return fig

        except Exception as e:
            raise CalculationError(f"Failed to plot correlation matrix: {str(e)}")

    def plot_rolling_correlations(self,
                                pairs: Optional[List[Tuple[str, str]]] = None,
                                window: Optional[int] = None,
                                size: Optional[Tuple[int, int]] = None) -> go.Figure:
        """
        Plot rolling correlations for specified pairs.

        Args:
            pairs: List of asset pairs to plot
            window: Rolling window size
            size: Figure size (width, height)

        Returns:
            go.Figure: Plotly figure object
        """
        try:
            if self.rolling_corr_ is None or (window and self.rolling_window_ != window):
                self.calculate_rolling_correlation(window)

            if pairs is None:
                # Get top 5 most correlated pairs
                highly_corr = self.get_highly_correlated_pairs()
                pairs = [(row['asset1'], row['asset2'])
                        for _, row in highly_corr.head().iterrows()]

            size = size or (1200, 600)
            fig = go.Figure()

            for asset1, asset2 in pairs:
                pair_name = validate_pair_name(asset1, asset2)
                if pair_name not in self.rolling_corr_:
                    logger.warning(f"No rolling correlation data for pair: {asset1}-{asset2}")
                    continue

                rolling_corr = self.rolling_corr_[pair_name]

                fig.add_trace(go.Scatter(
                    x=rolling_corr.index,
                    y=rolling_corr,
                    name=f"{asset1} - {asset2}",
                    mode='lines'
                ))

            fig.update_layout(
                title=f"Rolling Correlations (Window: {self.rolling_window_} days)",
                xaxis_title="Date",
                yaxis_title="Correlation",
                showlegend=True,
                width=size[0],
                height=size[1],
                yaxis_range=[-1, 1]
            )

            return fig

        except Exception as e:
            raise CalculationError(f"Failed to plot rolling correlations: {str(e)}")


class DataLoader:
    """Class for loading and preprocessing financial data."""

    def __init__(self, data_dir: Union[str, Path]):
        """
        Initialize data loader.

        Args:
            data_dir: Directory containing data files
        """
        self.data_dir = Path(data_dir)
        if not self.data_dir.exists():
            raise ValueError(f"Data directory does not exist: {data_dir}")

    def load_stock_data(self,
                       required_columns: List[str] = None,
                       min_history: int = 252) -> pd.DataFrame:
        """
        Load and preprocess stock price data.

        Args:
            required_columns: List of required columns in CSV files
            min_history: Minimum number of days required for inclusion

        Returns:
            pd.DataFrame: Processed price data

        Raises:
            ValueError: If no valid data is found
        """
        required_columns = required_columns or ['Date', 'Close']
        logger.info(f"Loading stock data from {self.data_dir}")

        csv_files = list(self.data_dir.glob('*.csv'))
        if not csv_files:
            raise ValueError(f"No CSV files found in {self.data_dir}")

        prices_dict = {}
        errors = []

        for file in csv_files:
            try:
                if file.name == 'combined_prices.csv':
                    continue

                ticker = file.stem
                df = pd.read_csv(file)

                # Validate columns
                missing_cols = set(required_columns) - set(df.columns)
                if missing_cols:
                    errors.append(f"Missing columns in {file.name}: {missing_cols}")
                    continue

                # Process dates
                df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
                df = df.dropna(subset=['Date'])
                df.set_index('Date', inplace=True)

                # Convert prices to numeric, logging any conversion errors
                close_prices = pd.to_numeric(df['Close'], errors='coerce')
                if close_prices.isna().any():
                    errors.append(f"Non-numeric values in Close column of {file.name}")

                # Check sufficient history
                if len(close_prices) < min_history:
                    errors.append(f"Insufficient history for {ticker}: {len(close_prices)} days")
                    continue

                prices_dict[ticker] = close_prices

            except Exception as e:
                errors.append(f"Error processing {file.name}: {str(e)}")

        if not prices_dict:
            raise ValueError("No valid price data loaded. Errors: " + "\n".join(errors))

        # Combine all prices into a DataFrame
        prices_df = pd.DataFrame(prices_dict)

        # Handle missing values
        missing_pct = prices_df.isnull().mean()
        if (missing_pct > 0.1).any():
            warnings.warn("Some assets have >10% missing data")

        # Forward fill then backward fill missing values
        prices_df = prices_df.ffill().bfill()

        # Drop any remaining columns with missing values
        prices_df = prices_df.dropna(axis=1)

        logger.info(f"Successfully loaded data for {len(prices_df.columns)} assets")
        if errors:
            logger.warning("Errors encountered:\n" + "\n".join(errors))

        return prices_df


def run_correlation_analysis(prices_df: pd.DataFrame,
                           output_dir: Union[str, Path],
                           config: Optional[CorrelationConfig] = None) -> None:
    """
    Run comprehensive correlation analysis and save results.

    Args:
        prices_df: DataFrame of price data
        output_dir: Directory to save results
        config: Configuration parameters
    """
    output_dir = Path(output_dir)
    plots_dir = output_dir / "plots"
    results_dir = output_dir / "results"

    # Create directories
    for directory in [output_dir, plots_dir, results_dir]:
        directory.mkdir(parents=True, exist_ok=True)

    # Calculate returns
    returns = prices_df.pct_change().dropna()

    # Initialize analyzer
    analyzer = CorrelationAnalyzer(returns, config)

    try:
        # Calculate correlations
        pearson_corr = analyzer.calculate_pearson_correlation()
        partial_corr = analyzer.calculate_partial_correlation()
        rolling_corr = analyzer.calculate_rolling_correlation()

        # Plot correlation matrices
        pearson_fig = analyzer.plot_correlation_matrix('pearson')
        partial_fig = analyzer.plot_correlation_matrix('partial')

        pearson_fig.write_html(plots_dir / "pearson_correlation.html")
        partial_fig.write_html(plots_dir / "partial_correlation.html")

        # Get highly correlated pairs for different thresholds
        correlation_results = {}
        for threshold_name, threshold in config.CORRELATION_THRESHOLDS.items():
            pearson_pairs = analyzer.get_highly_correlated_pairs('pearson', threshold)
            partial_pairs = analyzer.get_highly_correlated_pairs('partial', threshold)

            correlation_results[f'pearson_{threshold_name}'] = pearson_pairs
            correlation_results[f'partial_{threshold_name}'] = partial_pairs

            pearson_pairs.to_csv(results_dir / f"pearson_pairs_{threshold_name}.csv")
            partial_pairs.to_csv(results_dir / f"partial_pairs_{threshold_name}.csv")

        # Analyze correlation stability
        stability = analyzer.analyze_correlation_stability()
        stability.to_csv(results_dir / "correlation_stability.csv")

        # Plot rolling correlations for different correlation levels
        for threshold_name, pairs_df in correlation_results.items():
            if len(pairs_df) > 0:
                top_pairs = [(row['asset1'], row['asset2'])
                            for _, row in pairs_df.head().iterrows()]
                rolling_fig = analyzer.plot_rolling_correlations(top_pairs)
                rolling_fig.write_html(plots_dir / f"rolling_correlations_{threshold_name}.html")

        # Test correlation significance
        pvalues_pearson = analyzer.determine_correlation_significance('pearson')
        pvalues_partial = analyzer.determine_correlation_significance('partial')

        pvalues_pearson.to_csv(results_dir / "pearson_pvalues.csv")
        pvalues_partial.to_csv(results_dir / "partial_pvalues.csv")

        # Generate summary report
        _write_summary_report(
            analyzer=analyzer,
            output_dir=output_dir,
            correlation_results=correlation_results,
            pvalues_pearson=pvalues_pearson,
            pvalues_partial=pvalues_partial,
            stability=stability
        )

    except Exception as e:
        logger.error(f"Error in correlation analysis: {str(e)}")
        raise


def _write_summary_report(analyzer: CorrelationAnalyzer,
                         output_dir: Path,
                         correlation_results: Dict,
                         pvalues_pearson: pd.DataFrame,
                         pvalues_partial: pd.DataFrame,
                         stability: pd.DataFrame) -> None:
    """Write summary report of correlation analysis."""
    with open(output_dir / "correlation_analysis.txt", "w") as f:
        f.write("Correlation Analysis Summary\n")
        f.write("==========================\n\n")

        # Overall statistics
        f.write("Overall Statistics:\n")
        f.write("-" * 20 + "\n")
        f.write(f"Number of assets analyzed: {len(analyzer.returns.columns)}\n")
        f.write(f"Time period: {analyzer.returns.index[0]} to {analyzer.returns.index[-1]}\n")
        f.write(f"Number of observations: {len(analyzer.returns)}\n\n")

        # Pearson correlation analysis
        f.write("\nPearson Correlation Analysis:\n")
        f.write("-" * 30 + "\n")
        f.write(f"Number of significant pairs (p < 0.05): "
                f"{(pvalues_pearson < 0.05).sum().sum() // 2}\n")

        for threshold_name, threshold in analyzer.config.CORRELATION_THRESHOLDS.items():
            pairs_df = correlation_results[f'pearson_{threshold_name}']
            f.write(f"\n{threshold_name.title()} Correlation Pairs (≥ {threshold}):\n")
            f.write(f"Number of pairs: {len(pairs_df)}\n")
            if len(pairs_df) > 0:
                f.write("Top 5 Most Correlated Pairs:\n")
                for _, row in pairs_df.head().iterrows():
                    f.write(f"  {row['asset1']} - {row['asset2']}: {row['correlation']:.3f}\n")

        # Partial correlation analysis
        f.write("\nPartial Correlation Analysis:\n")
        f.write("-" * 30 + "\n")
        f.write(f"Number of significant pairs (p < 0.05): "
                f"{(pvalues_partial < 0.05).sum().sum() // 2}\n")

        for threshold_name, threshold in analyzer.config.CORRELATION_THRESHOLDS.items():
            pairs_df = correlation_results[f'partial_{threshold_name}']
            f.write(f"\n{threshold_name.title()} Partial Correlation Pairs (≥ {threshold}):\n")
            f.write(f"Number of pairs: {len(pairs_df)}\n")
            if len(pairs_df) > 0:
                f.write("Top 5 Most Correlated Pairs:\n")
                for _, row in pairs_df.head().iterrows():
                    f.write(f"  {row['asset1']} - {row['asset2']}: {row['correlation']:.3f}\n")

        # Correlation stability analysis
        f.write("\nCorrelation Stability Analysis:\n")
        f.write("-" * 30 + "\n")
        f.write("Summary Statistics:\n")
        f.write(f"Average correlation stability: {stability['mean'].mean():.3f}\n")
        f.write(f"Average correlation volatility: {stability['std'].mean():.3f}\n")
        f.write(f"Average negative correlation percentage: {stability['negative_pct'].mean():.1f}%\n")

        # Most stable pairs
        stable_pairs = stability.sort_values('std')
        f.write("\nMost Stable Correlation Pairs:\n")
        for idx in stable_pairs.head().index:
            stats = stable_pairs.loc[idx]
            f.write(f"  {idx}: mean={stats['mean']:.3f}, std={stats['std']:.3f}\n")


if __name__ == "__main__":
    # Example usage
    try:
        data_dir = Path(r'C:\Users\arnav\Downloads\pairs_trading_system\data\raw')
        output_dir = Path("correlation_analysis")

        # Initialize configuration
        config = CorrelationConfig()

        # Load data
        loader = DataLoader(data_dir)
        prices_df = loader.load_stock_data()

        # Run analysis
        run_correlation_analysis(prices_df, output_dir, config)

        logger.info("Correlation analysis completed successfully")
    except Exception as e:
        raise