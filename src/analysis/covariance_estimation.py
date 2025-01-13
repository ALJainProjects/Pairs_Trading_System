"""
Comprehensive Covariance Estimation Module

This module implements multiple covariance estimation methods:
1. Standard Sample Covariance
2. Exponentially Weighted Moving Average (EWMA)
3. Graphical Lasso (Sparse) Covariance
4. OLS-Based Denoised Covariance
5. Robust Covariance (MCD)
6. Kalman Filter Covariance

Each method is implemented as a separate class inheriting from BaseCovariance.
"""

import pandas as pd
import numpy as np
from abc import ABC, abstractmethod
from sklearn.covariance import GraphicalLasso, MinCovDet
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
from typing import Optional, Tuple, Dict, List
from scipy import stats
import warnings
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class BaseCovariance(ABC):
    """Base class for covariance estimation methods.

    Attributes:
        covariance_: Estimated covariance matrix
        correlation_: Estimated correlation matrix
        eigenvalues_: Eigenvalues of the covariance matrix
        column_names_: Names of the assets/columns
    """

    def __init__(self):
        self.covariance_ = None
        self.correlation_ = None
        self.eigenvalues_ = None
        self.column_names_ = None

    @abstractmethod
    def fit(self, returns: pd.DataFrame) -> 'BaseCovariance':
        """
        Fit the covariance estimation model.

        Args:
            returns (pd.DataFrame): Asset returns

        Returns:
            self: The fitted estimator
        """
        pass

    def validate_input(self, returns: pd.DataFrame) -> None:
        """Validate input data."""
        if returns.empty:
            raise ValueError("Empty returns DataFrame")

        if returns.isnull().any().any():
            raise ValueError("Returns contain missing values")

        # Check for insufficient variation
        std_devs = returns.std()
        if (std_devs < 1e-8).any():
            raise ValueError("Some assets show near-zero variation")

        # Check for extreme values
        z_scores = stats.zscore(returns)
        if np.abs(z_scores).max() > 10:
            warnings.warn("Extreme values detected in returns (|z-score| > 10)")

    def analyze(self) -> dict:
        """Analyze properties of estimated covariance matrix."""
        if self.covariance_ is None:
            raise ValueError("Covariance matrix not yet estimated. Call fit() first.")

        analysis = {
            'condition_number': np.linalg.cond(self.covariance_),
            'eigenvalues': {
                'min': float(self.eigenvalues_.min()),
                'max': float(self.eigenvalues_.max()),
                'mean': float(self.eigenvalues_.mean())
            },
            'correlations': {
                'mean': float(np.mean(self.correlation_[np.triu_indices_from(self.correlation_, k=1)])),
                'std': float(np.std(self.correlation_[np.triu_indices_from(self.correlation_, k=1)])),
                'min': float(self.correlation_.min()),
                'max': float(self.correlation_.max())
            },
            'variances': {
                'mean': float(np.mean(np.diag(self.covariance_))),
                'min': float(np.min(np.diag(self.covariance_))),
                'max': float(np.max(np.diag(self.covariance_)))
            }
        }
        return analysis

    def plot(self, title: str = "") -> go.Figure:
        """Plot correlation matrix heatmap."""
        if self.correlation_ is None:
            raise ValueError("Correlation matrix not yet estimated. Call fit() first.")

        fig = go.Figure(data=go.Heatmap(
            z=self.correlation_,
            x=self.column_names_,
            y=self.column_names_,
            colorscale='RdBu',
            zmid=0,
            showscale=True
        ))

        fig.update_layout(
            title=title or self.__class__.__name__,
            xaxis_title="Assets",
            yaxis_title="Assets",
            width=1000,
            height=1000
        )

        return fig

class StandardCovariance(BaseCovariance):
    """Standard sample covariance estimation."""

    def fit(self, returns: pd.DataFrame) -> 'StandardCovariance':
        """Estimate standard sample covariance."""
        self.validate_input(returns)
        self.column_names_ = returns.columns

        self.covariance_ = returns.cov().values
        std = np.sqrt(np.diag(self.covariance_))
        self.correlation_ = self.covariance_ / np.outer(std, std)
        self.eigenvalues_ = np.linalg.eigvals(self.covariance_)

        return self

class EWMACovariance(BaseCovariance):
    """EWMA-based covariance estimation."""

    def __init__(self, span: int = 30, min_periods: Optional[int] = None):
        super().__init__()
        self.span = span
        self.min_periods = min_periods or span
        self.weights_ = None

    def fit(self, returns: pd.DataFrame) -> 'EWMACovariance':
        """Estimate EWMA covariance.

        Args:
            returns: DataFrame of returns with DatetimeIndex

        Returns:
            self: The fitted estimator

        Raises:
            ValueError: If insufficient data points or invalid input
        """
        self.validate_input(returns)

        # Ensure data is properly sorted by date
        if not isinstance(returns.index, pd.DatetimeIndex):
            raise ValueError("Returns index must be a DatetimeIndex")

        returns_sorted = returns.sort_index()

        if len(returns_sorted) < self.min_periods:
            raise ValueError(
                f"Insufficient data points ({len(returns_sorted)}) for EWMA calculation. "
                f"Minimum required: {self.min_periods}"
            )

        self.column_names_ = returns_sorted.columns
        n_assets = len(self.column_names_)

        try:
            # Compute pairwise covariances
            ewm_cov = returns_sorted.ewm(
                span=self.span,
                min_periods=self.min_periods,
                adjust=True  # Use adjusted (normalized) weights
            ).cov()

            # Handle multi-index correctly
            # ewm_cov has a MultiIndex: (time, asset1, asset2)
            latest_date = ewm_cov.index.get_level_values(0).unique()[-1]

            # Extract the final covariance matrix
            latest_cov = ewm_cov.loc[latest_date]

            # Ensure we have a proper n_assets × n_assets matrix
            if isinstance(latest_cov, pd.Series):
                # Convert series to matrix if necessary
                self.covariance_ = latest_cov.unstack().values
            else:
                self.covariance_ = latest_cov.values

            # Additional shape validation
            if self.covariance_.shape != (n_assets, n_assets):
                raise ValueError(
                    f"Invalid covariance matrix shape: {self.covariance_.shape}. "
                    f"Expected: ({n_assets}, {n_assets})"
                )

            # Store weights
            alpha = 2 / (self.span + 1)
            self.weights_ = np.array([(1-alpha)**i for i in range(len(returns))])
            self.weights_ = self.weights_[::-1] / self.weights_.sum()

            # Calculate correlation
            std = np.sqrt(np.diag(self.covariance_))
            self.correlation_ = self.covariance_ / np.outer(std, std)
            self.eigenvalues_ = np.linalg.eigvals(self.covariance_)

            return self

        except Exception as e:
            raise RuntimeError(f"Error computing EWMA covariance: {str(e)}")

    def analyze(self) -> dict:
        """Extended analysis including EWMA-specific metrics."""
        base_analysis = super().analyze()

        ewma_analysis = {
            'ewma_parameters': {
                'span': self.span,
                'min_periods': self.min_periods,
                'effective_sample_size': 1 / (self.weights_**2).sum(),
                'max_weight': float(self.weights_.max()),
                'weight_decay_95': float(
                    np.where(np.cumsum(self.weights_[::-1]) >= 0.95)[0][0]
                )
            }
        }

        base_analysis.update(ewma_analysis)
        return base_analysis

class GraphicalLassoCovariance(BaseCovariance):
    """Graphical Lasso (sparse) covariance estimation."""

    def __init__(self, alpha: float = 0.01, max_iter: int = 100,
                 tol: float = 1e-4):
        super().__init__()
        self.alpha = alpha
        self.max_iter = max_iter
        self.tol = tol
        self.precision_ = None

    def fit(self, returns: pd.DataFrame) -> 'GraphicalLassoCovariance':
        """Estimate sparse covariance using Graphical Lasso."""
        self.validate_input(returns)
        self.column_names_ = returns.columns

        # Standardize returns
        scaler = StandardScaler()
        scaled_returns = scaler.fit_transform(returns)

        # Fit GraphicalLasso
        model = GraphicalLasso(
            alpha=self.alpha,
            max_iter=self.max_iter,
            tol=self.tol
        )
        model.fit(scaled_returns)

        # Transform back to original scale
        scales = returns.std()
        self.covariance_ = model.covariance_ * np.outer(scales, scales)
        self.precision_ = model.precision_ / np.outer(scales, scales)

        # Calculate correlation
        std = np.sqrt(np.diag(self.covariance_))
        self.correlation_ = self.covariance_ / np.outer(std, std)
        self.eigenvalues_ = np.linalg.eigvals(self.covariance_)

        return self

    def analyze(self) -> dict:
        """Extended analysis including sparsity metrics."""
        base_analysis = super().analyze()

        # Add sparsity analysis
        threshold = 1e-5
        precision_sparsity = np.mean(np.abs(self.precision_) < threshold)
        covariance_sparsity = np.mean(np.abs(self.covariance_) < threshold)

        sparsity_analysis = {
            'sparsity': {
                'precision': float(precision_sparsity),
                'covariance': float(covariance_sparsity)
            }
        }

        base_analysis.update(sparsity_analysis)
        return base_analysis

class OLSCovariance(BaseCovariance):
    """Enhanced OLS-based denoised covariance estimation with multi-factor support."""

    def __init__(self, n_factors: Optional[int] = None,
                 min_eigenvalue_ratio: float = 0.01):
        super().__init__()
        self.n_factors = n_factors
        self.min_eigenvalue_ratio = min_eigenvalue_ratio
        self.factors_ = None
        self.factor_loadings_ = None

    def fit(self, returns: pd.DataFrame) -> 'OLSCovariance':
        """Estimate OLS-denoised covariance with optional factor structure."""
        self.validate_input(returns)
        self.column_names_ = returns.columns

        if self.n_factors is not None:
            denoised_returns = self._factor_denoise(returns)
        else:
            denoised_returns = self._pairwise_denoise(returns)

        self.covariance_ = denoised_returns.cov().values
        std = np.sqrt(np.diag(self.covariance_))
        self.correlation_ = self.covariance_ / np.outer(std, std)
        self.eigenvalues_ = np.linalg.eigvals(self.covariance_)

        return self

    def _factor_denoise(self, returns: pd.DataFrame) -> pd.DataFrame:
        """Denoise returns using PCA factors."""
        scaler = StandardScaler()
        scaled_returns = scaler.fit_transform(returns)

        U, S, Vt = np.linalg.svd(scaled_returns, full_matrices=False)

        if self.n_factors is None:
            eigenvalue_ratios = S**2 / (S**2).max()
            self.n_factors = np.sum(eigenvalue_ratios > self.min_eigenvalue_ratio)

        self.factors_ = U[:, :self.n_factors] * S[:self.n_factors]
        self.factor_loadings_ = Vt[:self.n_factors, :]

        denoised_scaled = self.factors_ @ self.factor_loadings_

        denoised_returns = pd.DataFrame(
            scaler.inverse_transform(denoised_scaled),
            index=returns.index,
            columns=returns.columns
        )

        return denoised_returns

    def _pairwise_denoise(self, returns: pd.DataFrame) -> pd.DataFrame:
        """Denoise returns using pairwise OLS regression."""
        denoised = pd.DataFrame(index=returns.index, columns=returns.columns)

        for target in returns.columns:
            X = returns.drop(columns=[target])
            y = returns[target]

            model = LinearRegression()
            model.fit(X, y)

            # Get residuals (noise component)
            systematic = model.predict(X)
            denoised[target] = y - systematic  # Use residuals instead of systematic

        return denoised

    def analyze(self) -> dict:
        """Extended analysis including factor-related metrics."""
        base_analysis = super().analyze()

        if self.factors_ is not None:
            factor_analysis = {
                'factors': {
                    'n_factors': self.n_factors,
                    'explained_variance_ratio': list(
                        np.var(self.factors_, axis=0) /
                        np.var(self.factors_).sum()
                    ),
                    'total_explained_variance': float(
                        np.var(self.factors_).sum() /
                        np.var(self.factors_[:, 0])
                    )
                }
            }
            base_analysis.update(factor_analysis)

        return base_analysis

class RobustCovariance(BaseCovariance):
    """Robust covariance estimation using Minimum Covariance Determinant (MCD)."""

    def __init__(self,
                 contamination: Optional[float] = None,
                 contamination_cap: float = 0.05,
                 support_fraction: Optional[float] = None,
                 random_state: int = 42):
        super().__init__()
        self.contamination = contamination
        self.contamination_cap = contamination_cap
        self.support_fraction = support_fraction
        self.random_state = random_state

        # Additional attributes
        self.outlier_mask_ = None
        self.mahalanobis_dist_ = None
        self.contamination_used_ = None
        self.location_ = None
        self.precision_ = None
        self.support_ = None

    def fit(self, returns: pd.DataFrame) -> 'RobustCovariance':
        """Estimate robust covariance using MCD."""
        self.validate_input(returns)
        self.column_names_ = returns.columns

        # Estimate contamination if not provided
        if self.contamination is None:
            iso = IsolationForest(contamination='auto',
                                random_state=self.random_state)
            iso.fit(returns)
            pred = iso.predict(returns)
            est_cont = np.count_nonzero(pred == -1) / len(pred)
            self.contamination_used_ = min(self.contamination_cap,
                                         est_cont + 0.01)
        else:
            self.contamination_used_ = min(self.contamination,
                                         self.contamination_cap)

        # Set support fraction for MCD
        if self.support_fraction is None:
            self.support_fraction = 1.0 - self.contamination_used_

        try:
            # Fit MCD estimator
            mcd = MinCovDet(support_fraction=self.support_fraction,
                           random_state=self.random_state)
            mcd.fit(returns)

            # Store results
            self.covariance_ = mcd.covariance_
            self.location_ = mcd.location_
            self.precision_ = mcd.precision_
            self.support_ = mcd.support_

            # Calculate Mahalanobis distances and outlier mask
            self.mahalanobis_dist_ = mcd.mahalanobis(returns)
            self.outlier_mask_ = self.mahalanobis_dist_ > mcd.threshold_

            # Calculate correlation matrix
            std = np.sqrt(np.diag(self.covariance_))
            self.correlation_ = self.covariance_ / np.outer(std, std)
            self.eigenvalues_ = np.linalg.eigvals(self.covariance_)

            return self

        except Exception as e:
            raise RuntimeError(f"MCD estimation error: {str(e)}")

    def analyze(self) -> dict:
        """Extended analysis including robustness metrics."""
        base_analysis = super().analyze()

        # Add robustness-specific analysis
        robust_analysis = {
            'robustness_metrics': {
                'contamination_used': float(self.contamination_used_),
                'outliers_detected': int(np.sum(self.outlier_mask_)),
                'outlier_percentage': float(np.mean(self.outlier_mask_)),
                'mahalanobis_distances': {
                    'mean': float(np.mean(self.mahalanobis_dist_)),
                    'median': float(np.median(self.mahalanobis_dist_)),
                    'max': float(np.max(self.mahalanobis_dist_)),
                    'threshold': float(np.percentile(self.mahalanobis_dist_,
                                                   (1 - self.contamination_used_) * 100))
                },
                'support_size': int(len(self.support_)),
                'support_fraction': float(len(self.support_) / len(self.mahalanobis_dist_))
            }
        }

        base_analysis.update(robust_analysis)
        return base_analysis

    def plot_outliers(self, returns: pd.DataFrame, n_assets: int = 5) -> go.Figure:
        """Plot return series with outliers highlighted."""
        if self.outlier_mask_ is None:
            raise ValueError("No outliers detected. Call fit() first.")

        # Select top assets by variance
        top_assets = returns.var().nlargest(n_assets).index

        fig = make_subplots(
            rows=n_assets,
            cols=1,
            subplot_titles=[f"Returns and Outliers - {asset}"
                          for asset in top_assets]
        )

        for i, asset in enumerate(top_assets, 1):
            # Normal returns
            fig.add_trace(
                go.Scatter(
                    x=returns.index[~self.outlier_mask_],
                    y=returns[asset][~self.outlier_mask_],
                    mode='markers',
                    name=f'{asset} Normal',
                    marker=dict(color='blue', size=5)
                ),
                row=i, col=1
            )

            # Outliers
            fig.add_trace(
                go.Scatter(
                    x=returns.index[self.outlier_mask_],
                    y=returns[asset][self.outlier_mask_],
                    mode='markers',
                    name=f'{asset} Outliers',
                    marker=dict(color='red', size=8)
                ),
                row=i, col=1
            )

        fig.update_layout(
            height=300*n_assets,
            showlegend=True,
            title="Return Series with Outliers Highlighted"
        )

        return fig

class KalmanCovariance(BaseCovariance):
    """Kalman Filter based covariance estimation."""

    def __init__(self,
                 process_variance: float = 1e-5,
                 measurement_variance: float = 1e-3,
                 forgetting_factor: float = 0.97,
                 batch_size: int = 50,
                 parameter_set: Optional[str] = None):
        super().__init__()

        # Default parameter sets
        PARAMETER_SETS = {
            'conservative': {
                'process_variance': 1e-6,
                'measurement_variance': 1e-3,
                'forgetting_factor': 0.99
            },
            'moderate': {
                'process_variance': 1e-5,
                'measurement_variance': 1e-3,
                'forgetting_factor': 0.97
            },
            'adaptive': {
                'process_variance': 1e-4,
                'measurement_variance': 1e-3,
                'forgetting_factor': 0.95
            }
        }

        if parameter_set is not None:
            if parameter_set not in PARAMETER_SETS:
                raise ValueError(f"Unknown parameter set: {parameter_set}")
            params = PARAMETER_SETS[parameter_set]
            self.process_variance = params['process_variance']
            self.measurement_variance = params['measurement_variance']
            self.forgetting_factor = params['forgetting_factor']
        else:
            self.process_variance = process_variance
            self.measurement_variance = measurement_variance
            self.forgetting_factor = forgetting_factor

        self.batch_size = batch_size
        self.diagnostics_ = None

    def fit(self, returns: pd.DataFrame) -> 'KalmanCovariance':
        """Estimate time-varying covariance using Kalman Filter."""
        self.validate_input(returns)
        self.column_names_ = returns.columns

        n_assets = returns.shape[1]
        n_obs = returns.shape[0]

        # Initialize diagnostics with only used metrics
        self.diagnostics_ = {
            'condition_numbers': []
        }

        # Initialize state with sample covariance of first batch
        initial_batch = returns.iloc[:min(self.batch_size, n_obs)]
        current_cov = initial_batch.cov().values

        # Process returns in batches
        for start_idx in range(0, n_obs, self.batch_size):
            end_idx = min(start_idx + self.batch_size, n_obs)
            batch_returns = returns.iloc[start_idx:end_idx]

            # Update for each observation in batch
            for _, ret in batch_returns.iterrows():
                # Predict step
                current_cov = current_cov / self.forgetting_factor
                current_cov += self.process_variance * np.eye(n_assets)

                # Update step
                y = ret.values.reshape(-1, 1)
                S = current_cov + self.measurement_variance * np.eye(n_assets)

                try:
                    # Compute Kalman gain using stable solver
                    K = np.linalg.solve(S, current_cov).T

                    # Update estimate
                    innovation = y @ y.T - current_cov
                    current_cov = current_cov + K @ innovation

                    # Ensure symmetry
                    current_cov = (current_cov + current_cov.T) / 2

                    # Force positive definiteness if needed
                    min_eig = np.min(np.real(np.linalg.eigvals(current_cov)))
                    if min_eig < 1e-10:
                        current_cov += (abs(min_eig) + 1e-10) * np.eye(n_assets)

                    # Update diagnostics
                    cond_num = np.linalg.cond(current_cov)
                    self.diagnostics_['condition_numbers'].append(float(cond_num))

                except np.linalg.LinAlgError:
                    # Fallback to robust update
                    current_cov = 0.9 * current_cov + 0.1 * (y @ y.T)

        # Store final estimates
        self.covariance_ = current_cov
        std = np.sqrt(np.diag(self.covariance_))
        self.correlation_ = self.covariance_ / np.outer(std, std)
        self.eigenvalues_ = np.linalg.eigvals(self.covariance_)

        # Final diagnostics
        self.diagnostics_['final_condition_number'] = float(np.linalg.cond(current_cov))
        self.diagnostics_['eigenvalue_range'] = {
            'min': float(np.min(np.real(self.eigenvalues_))),
            'max': float(np.max(np.real(self.eigenvalues_)))
        }

        return self

    def analyze(self) -> dict:
        """Extended analysis including Kalman Filter diagnostics."""
        base_analysis = super().analyze()

        # Add Kalman-specific analysis
        kalman_analysis = {
            'kalman_diagnostics': {
                'final_condition_number': self.diagnostics_['final_condition_number'],
                'condition_number_stats': {
                    'mean': float(np.mean(self.diagnostics_['condition_numbers'])),
                    'std': float(np.std(self.diagnostics_['condition_numbers'])),
                    'max': float(np.max(self.diagnostics_['condition_numbers']))
                },
                'eigenvalue_range': self.diagnostics_['eigenvalue_range']
            }
        }

        base_analysis.update(kalman_analysis)
        return base_analysis

    def plot_diagnostics(self) -> go.Figure:
        """Plot Kalman Filter diagnostics."""
        if self.diagnostics_ is None:
            raise ValueError("No diagnostics available. Call fit() first.")

        fig = make_subplots(rows=1, cols=1)

        # Plot condition numbers
        fig.add_trace(
            go.Scatter(
                y=np.log10(self.diagnostics_['condition_numbers']),
                name='Log10 Condition Number',
                mode='lines'
            )
        )

        fig.update_layout(
            title='Kalman Filter Diagnostics',
            yaxis_title='Log10 Condition Number',
            xaxis_title='Update Step',
            showlegend=True,
            height=600,
            width=800
        )

        return fig

def plot_method_comparison(returns: pd.DataFrame,
                         methods: Dict[str, BaseCovariance],
                         title: str = "Covariance Estimation Comparison",
                         plot_outliers: bool = True) -> List[go.Figure]:
    """Plot comparison of different covariance estimation methods."""
    figures = []

    # Correlation matrix comparison
    n_methods = len(methods)
    fig = make_subplots(
        rows=1,
        cols=n_methods,
        subplot_titles=list(methods.keys())
    )

    for i, (name, estimator) in enumerate(methods.items(), 1):
        fig.add_trace(
            go.Heatmap(
                z=estimator.correlation_,
                x=estimator.column_names_,
                y=estimator.column_names_,
                colorscale='RdBu',
                zmid=0,
                showscale=(i == n_methods)
            ),
            row=1, col=i
        )

    fig.update_layout(
        title=title,
        height=800,
        width=400 * n_methods
    )

    figures.append(fig)

    # Add outlier plots for RobustCovariance
    if plot_outliers:
        robust_estimators = {
            name: est for name, est in methods.items()
            if isinstance(est, RobustCovariance)
        }

        for name, estimator in robust_estimators.items():
            outlier_fig = estimator.plot_outliers(returns)
            outlier_fig.update_layout(title=f"{name} - Outlier Analysis")
            figures.append(outlier_fig)

    return figures

def load_nasdaq100_data(data_dir: str) -> pd.DataFrame:
    """Load price data for NASDAQ 100 stocks from CSV files."""
    logger.info(f"Loading price data from {data_dir}")

    csv_files = [f for f in os.listdir(data_dir)
                if f.endswith('.csv') and f != 'combined_prices.csv']

    prices_dict = {}
    required_columns = ['Date', 'Close']

    for file in csv_files:
        try:
            ticker = file.replace('.csv', '')
            df = pd.read_csv(os.path.join(data_dir, file))

            if not all(col in df.columns for col in required_columns):
                logger.warning(f"Skipping {file}: Missing required columns")
                continue

            df['Date'] = pd.to_datetime(df['Date'])
            df.set_index('Date', inplace=True)

            prices_dict[ticker] = pd.to_numeric(df['Close'], errors='coerce')
            logger.info(f"Loaded {ticker} data: {len(df)} rows")

        except Exception as e:
            logger.error(f"Error loading {file}: {str(e)}")

    if not prices_dict:
        raise ValueError("No valid price data loaded")

    prices_df = pd.DataFrame(prices_dict)
    prices_df = prices_df.ffill().bfill()
    prices_df = prices_df.dropna(axis=1)

    logger.info(f"Loaded {len(prices_df.columns)} stocks with {len(prices_df)} data points each")

    return prices_df
def write_analysis_results(output_dir: str, results: Dict) -> None:
    """Write detailed analysis results to file.

    Args:
        output_dir: Directory to save the results
        results: Dictionary containing analysis results for each method
    """
    with open(os.path.join(output_dir, "analysis_results.txt"), "w") as f:
        f.write("Covariance Estimation Analysis\n")
        f.write("============================\n\n")

        for method, analysis in results.items():
            f.write(f"\n{method} Estimation Results:\n")
            f.write("-" * 40 + "\n")
            write_dict(f, analysis)

def write_dict(f, d: Dict, indent: int = 0) -> None:
    """Helper function to write nested dictionary to file.

    Args:
        f: File handle to write to
        d: Dictionary to write
        indent: Current indentation level
    """
    for key, value in d.items():
        if isinstance(value, dict):
            f.write("    " * indent + f"{key}:\n")
            write_dict(f, value, indent + 1)
        else:
            f.write("    " * indent + f"{key}: {value}\n")

def save_comparison_metrics(output_dir: str, results: Dict) -> None:
    """Save comparison metrics to CSV.

    Args:
        output_dir: Directory to save the metrics
        results: Dictionary containing analysis results for each method
    """
    comparison_metrics = pd.DataFrame({
        'Method': [],
        'Condition_Number': [],
        'Avg_Correlation': [],
        'Max_Eigenvalue': [],
        'Min_Eigenvalue': []
    })

    for method, analysis in results.items():
        comparison_metrics = pd.concat([comparison_metrics, pd.DataFrame({
            'Method': [method],
            'Condition_Number': [analysis['condition_number']],
            'Avg_Correlation': [analysis['correlations']['mean']],
            'Max_Eigenvalue': [analysis['eigenvalues']['max']],
            'Min_Eigenvalue': [analysis['eigenvalues']['min']]
        })])

    comparison_metrics.to_csv(
        os.path.join(output_dir, "method_comparison_metrics.csv"),
        index=False
    )

def main():
    """Main execution function."""
    # Create output directory
    output_dir = "covariance_analysis"
    os.makedirs(output_dir, exist_ok=True)

    # Load data
    logger.info("Loading price data...")
    prices_df = load_nasdaq100_data(r'C:\Users\arnav\Downloads\pairs_trading_system\data\raw')
    returns = prices_df.pct_change().dropna()

    # Initialize estimators
    estimators = {
        'Standard': StandardCovariance(),
        'EWMA': EWMACovariance(span=30),
        'GraphicalLasso': GraphicalLassoCovariance(alpha=0.01),
        'OLS': OLSCovariance(),
        'Robust': RobustCovariance(),
        'Kalman': KalmanCovariance(parameter_set='moderate')
    }

    # Process each estimator
    results = {}
    for name, estimator in estimators.items():
        logger.info(f"Processing {name} estimator...")
        try:
            # Fit estimator
            estimator.fit(returns)

            # Save correlation plot
            estimator.plot(f"{name} Correlation Matrix").write_html(
                os.path.join(output_dir, f"{name.lower()}_correlation.html")
            )

            # Save special plots if available
            if isinstance(estimator, KalmanCovariance):
                estimator.plot_diagnostics().write_html(
                    os.path.join(output_dir, f"{name.lower()}_diagnostics.html")
                )

            # Get analysis
            results[name] = estimator.analyze()

        except Exception as e:
            logger.error(f"Error processing {name}: {str(e)}")
            continue

    # Create and save comparison plots
    figures = plot_method_comparison(returns, estimators)
    for i, fig in enumerate(figures):
        fig.write_html(os.path.join(output_dir, f"comparison_{i}.html"))

    # Save analysis results
    write_analysis_results(output_dir, results)
    save_comparison_metrics(output_dir, results)

    logger.info(f"Analysis complete. Results saved in {output_dir}")

if __name__ == "__main__":
    main()