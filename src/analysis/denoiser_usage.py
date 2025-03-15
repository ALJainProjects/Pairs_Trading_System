"""
Comprehensive Asset Analysis Module

This module provides functionality for:
1. Return Denoising:
   - Wavelet-based denoising (with multiple threshold methods)
   - PCA-based denoising (with flexible scaling)
   - Hybrid denoising approaches
   - Rolling time-variant denoising
2. Pair Selection:
   - PCA-based pair identification
   - Similarity analysis
3. Analysis and Visualization Tools
4. Denoising Quality Assessment

Features:
- Multiple denoising methods with parametric control
- Parallel processing capabilities
- Comprehensive time series metrics
- Flexible data scaling options
"""

import pandas as pd
import numpy as np
import pywt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.linear_model import LinearRegression
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Tuple, Optional, List, Dict, Union
import os
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

LOOKBACK_PERIODS = {
    '1M': 21,
    '3M': 63,
    '6M': 126,
    '12M': 252
}

PARAMETER_SETS = {
    'conservative': {'n_components': 2, 'similarity_threshold': 0.95},
    'moderate': {'n_components': 3, 'similarity_threshold': 0.9},
    'aggressive': {'n_components': 5, 'similarity_threshold': 0.85}
}

class AssetAnalyzer:
    """Main class for comprehensive asset analysis."""

    def __init__(self, n_jobs: int = -1, random_state: int = 42):
        """
        Initialize the analyzer.

        Args:
            n_jobs: Number of parallel jobs (-1 for all cores)
            random_state: Random seed for reproducibility
        """
        self.n_jobs = n_jobs
        self.random_state = random_state

        self.original_returns = None
        self.denoised_returns = {}
        self.denoising_results = {}

        self.selected_pairs = {}
        self.pair_similarities = {}
        self.pca_loadings = {}
        self.explained_variance = {}

    def fit(self, returns: pd.DataFrame) -> 'AssetAnalyzer':
        """Fit the analyzer with return data."""
        self._validate_input(returns)
        self.original_returns = returns
        return self

    def denoise_wavelet(self,
                       wavelet: str = 'db1',
                       level: Optional[int] = None,
                       threshold_mode: str = 'universal',
                       threshold_multiplier: float = 1.0,
                       thresholding_type: str = 'soft') -> pd.DataFrame:
        """Apply enhanced wavelet-based denoising with multiple options.

        Args:
            wavelet: Type of wavelet ('db1', 'sym4', 'coif3', etc)
            level: Decomposition level (None for automatic based on data length)
            threshold_mode: 'universal' or 'adaptive'
            threshold_multiplier: Factor to adjust threshold sensitivity
            thresholding_type: 'soft' or 'hard'
        """
        if self.original_returns is None:
            raise ValueError("Must call fit() before denoising")

        logger.info(f"Applying wavelet denoising: {wavelet}, mode {threshold_mode}")

        denoised = pd.DataFrame(index=self.original_returns.index,
                              columns=self.original_returns.columns)

        # Auto-determine level if not specified
        if level is None:
            max_level = pywt.dwt_max_level(len(self.original_returns), wavelet)
            level = min(max_level, 4)
            logger.info(f"Automatically selected decomposition level {level}")

        # Use fixed level if auto-determination fails
        if level <= 0:
            level = 1
            logger.warning(f"Using default level {level} as auto-determination failed")

        for ticker in self.original_returns.columns:
            series = self.original_returns[ticker].values
            coeff = pywt.wavedec(series, wavelet, level=level)

            # Calculate threshold depending on mode
            if threshold_mode == 'universal':
                # Universal threshold (VisuShrink)
                sigma = np.median(np.abs(coeff[-1])) / 0.6745
                uthresh = sigma * np.sqrt(2 * np.log(len(series))) * threshold_multiplier
            elif threshold_mode == 'adaptive':
                # SURE threshold (Stein's Unbiased Risk Estimator) - simplified version
                sigma = np.median(np.abs(coeff[-1])) / 0.6745
                uthresh = sigma * 0.6745 * threshold_multiplier
            else:
                raise ValueError(f"Unknown threshold mode: {threshold_mode}")

            # Apply thresholding
            denoised_coeff = coeff[:]
            denoised_coeff[1:] = [
                pywt.threshold(c, value=uthresh, mode=thresholding_type)
                for c in denoised_coeff[1:]
            ]

            reconstructed = pywt.waverec(denoised_coeff, wavelet)
            denoised[ticker] = reconstructed[:len(series)]

        self.denoised_returns['wavelet'] = denoised
        self._analyze_denoising('wavelet')
        return denoised

    def denoise_pca(self, n_components: Optional[int] = None,
                  variance_threshold: float = 0.95,
                  return_loadings: bool = False) -> Union[pd.DataFrame, Tuple[pd.DataFrame, np.ndarray]]:
        """Apply PCA-based denoising with adaptive component selection.

        Args:
            n_components: Specific number of components to use (None for automatic selection)
            variance_threshold: Threshold for explained variance when auto-selecting components
            return_loadings: Whether to return the PCA loadings along with denoised returns

        Returns:
            pd.DataFrame: Denoised returns dataframe
            np.ndarray (optional): PCA loadings if return_loadings is True
        """
        if self.original_returns is None:
            raise ValueError("Must call fit() before denoising")

        logger.info("Applying adaptive PCA denoising")

        scaler = StandardScaler()
        scaled_returns = scaler.fit_transform(self.original_returns)

        # First fit PCA without limiting components to determine optimal number
        if n_components is None:
            pca_full = PCA(random_state=self.random_state)
            pca_full.fit(scaled_returns)

            # Determine optimal number of components based on explained variance
            cumulative_variance = np.cumsum(pca_full.explained_variance_ratio_)
            n_components = np.argmax(cumulative_variance >= variance_threshold) + 1
            logger.info(f"Automatically selected {n_components} components explaining {variance_threshold*100:.1f}% of variance")

        # Apply PCA with optimal/specified number of components
        pca = PCA(n_components=n_components, random_state=self.random_state)
        principal_components = pca.fit_transform(scaled_returns)
        reconstructed = pca.inverse_transform(principal_components)

        denoised = pd.DataFrame(
            scaler.inverse_transform(reconstructed),
            index=self.original_returns.index,
            columns=self.original_returns.columns
        )

        self.denoised_returns['pca'] = denoised
        self._analyze_denoising('pca', pca.explained_variance_ratio_)

        if return_loadings:
            return denoised, pca.components_
        return denoised

    def denoise_hybrid(self, weight_pca: float = 0.5,
                      pca_kwargs: Optional[Dict] = None,
                      wavelet_kwargs: Optional[Dict] = None) -> pd.DataFrame:
        """Apply hybrid denoising combining multiple methods.

        Args:
            weight_pca: Weight given to PCA (1-weight_pca will be given to wavelet)
            pca_kwargs: Optional parameters for PCA denoising
            wavelet_kwargs: Optional parameters for wavelet denoising
        """
        if self.original_returns is None:
            raise ValueError("Must call fit() before denoising")

        logger.info(f"Applying hybrid denoising with PCA weight {weight_pca}")

        # Apply individual methods if not already applied
        pca_kwargs = pca_kwargs or {'variance_threshold': 0.9}
        if 'pca' not in self.denoised_returns:
            self.denoise_pca(**pca_kwargs)

        wavelet_kwargs = wavelet_kwargs or {'wavelet': 'db4', 'threshold_mode': 'adaptive'}
        if 'wavelet' not in self.denoised_returns:
            self.denoise_wavelet(**wavelet_kwargs)

        # Combine the results with weighting
        pca_returns = self.denoised_returns['pca']
        wavelet_returns = self.denoised_returns['wavelet']

        hybrid_returns = (weight_pca * pca_returns) + ((1 - weight_pca) * wavelet_returns)

        self.denoised_returns['hybrid'] = hybrid_returns
        self._analyze_denoising('hybrid')
        return hybrid_returns

    def denoise_rolling(self,
                      method: str = 'pca',
                      window_size: int = 126,
                      step_size: Optional[int] = None,
                      **kwargs) -> pd.DataFrame:
        """Apply denoising with rolling windows to account for time-varying market conditions.

        Args:
            method: 'pca' or 'wavelet'
            window_size: Size of rolling window in days
            step_size: Size of step between windows (None for automatic)
            **kwargs: Parameters for the chosen denoising method
        """
        if self.original_returns is None:
            raise ValueError("Must call fit() before denoising")

        if method not in ['pca', 'wavelet']:
            raise ValueError(f"Unsupported method for rolling denoising: {method}")

        logger.info(f"Applying rolling {method} denoising with {window_size}-day window")

        # Set default step size to avoid excessive computation
        if step_size is None:
            step_size = max(1, window_size // 5)

        # Create output dataframe
        result = pd.DataFrame(index=self.original_returns.index,
                            columns=self.original_returns.columns)

        # Use original data for beginning of series where we don't have a full window
        result.iloc[:window_size] = self.original_returns.iloc[:window_size]

        # Calculate rolling denoised returns
        for start_idx in range(0, len(self.original_returns) - window_size, step_size):
            end_idx = start_idx + window_size
            window_data = self.original_returns.iloc[start_idx:end_idx].copy()

            # Apply denoising to window
            temp_analyzer = AssetAnalyzer(random_state=self.random_state)
            temp_analyzer.fit(window_data)

            if method == 'pca':
                denoised_window = temp_analyzer.denoise_pca(**kwargs)
            elif method == 'wavelet':
                denoised_window = temp_analyzer.denoise_wavelet(**kwargs)

            # Store results - we use the last step_size values from each window
            update_start = min(end_idx - step_size, len(self.original_returns) - 1)
            update_end = min(end_idx, len(self.original_returns))
            result_idx = slice(update_start, update_end)
            window_idx = slice(window_size - (update_end - update_start), window_size)

            result.iloc[result_idx] = denoised_window.iloc[window_idx].values

        # Fill any remaining NaN values with original data
        mask = result.isna().any(axis=1)
        if mask.any():
            result.loc[mask] = self.original_returns.loc[mask]

        method_name = f'rolling_{method}'
        self.denoised_returns[method_name] = result
        self._analyze_denoising(method_name)
        return result

    def assess_denoising_quality(self, method: str, holdout_period: int = 21) -> Dict:
        """Evaluate denoising quality using predictive accuracy on holdout period.

        Args:
            method: The denoising method to evaluate
            holdout_period: Number of days to use for evaluation

        Returns:
            dict: Dictionary of quality metrics
        """
        if method not in self.denoised_returns:
            raise ValueError(f"Method {method} has not been applied yet")

        logger.info(f"Assessing quality of {method} denoising")

        # Skip quality assessment if dataset is too small
        if len(self.original_returns) <= holdout_period * 2:
            logger.warning("Dataset too small for holdout evaluation")
            return {"error": "Dataset too small for quality assessment"}

        # Use last holdout_period days for evaluation
        train_data = self.original_returns.iloc[:-holdout_period].copy()
        test_data = self.original_returns.iloc[-holdout_period:].copy()

        # Refit denoising on training data only
        temp_analyzer = AssetAnalyzer(random_state=self.random_state)
        temp_analyzer.fit(train_data)

        # Apply the same denoising method
        if method == 'pca':
            temp_analyzer.denoise_pca()
        elif method == 'wavelet':
            temp_analyzer.denoise_wavelet()
        elif method == 'hybrid':
            temp_analyzer.denoise_hybrid()
        elif method.startswith('rolling_'):
            # Extract base method from rolling_method name
            base_method = method.split('_')[1]
            temp_analyzer.denoise_rolling(method=base_method)
        else:
            raise ValueError(f"Unsupported method for quality assessment: {method}")

        denoised_train = temp_analyzer.denoised_returns[method]

        # Calculate metrics for each asset
        asset_metrics = {}
        for asset in self.original_returns.columns:
            # Correlation preservation
            orig_corr = train_data.corrwith(train_data[asset])
            denoised_corr = denoised_train.corrwith(denoised_train[asset])
            corr_preservation = np.corrcoef(orig_corr, denoised_corr)[0, 1]

            # Calculate signal-to-noise ratio improvement
            noise = train_data[asset] - denoised_train[asset]
            signal_power = np.var(denoised_train[asset])
            noise_power = np.var(noise)
            snr = signal_power / noise_power if noise_power > 0 else float('inf')

            # Predictive accuracy on test data
            try:
                model = LinearRegression()
                X = denoised_train[asset].values.reshape(-1, 1)
                y = train_data[asset].values
                model.fit(X, y)

                X_test = test_data[asset].values.reshape(-1, 1)
                y_test = test_data[asset].values
                prediction_score = model.score(X_test, y_test)
            except Exception as e:
                logger.warning(f"Could not calculate predictive score for {asset}: {str(e)}")
                prediction_score = None

            asset_metrics[asset] = {
                'correlation_preservation': corr_preservation,
                'snr': snr,
                'predictive_score': prediction_score,
            }

        # Calculate overall metrics
        valid_predictive_scores = [m['predictive_score'] for m in asset_metrics.values()
                                 if m['predictive_score'] is not None]

        overall_metrics = {
            'avg_correlation_preservation': np.mean([m['correlation_preservation'] for m in asset_metrics.values()]),
            'avg_snr': np.mean([m['snr'] for m in asset_metrics.values() if not np.isinf(m['snr'])]),
            'avg_predictive_score': np.mean(valid_predictive_scores) if valid_predictive_scores else None,
            'asset_metrics': asset_metrics,
            'method': method,
        }

        return overall_metrics

    def select_pairs(self,
                    n_components: int = 2,
                    similarity_threshold: float = 0.9,
                    parameter_set: Optional[str] = None) -> List[Tuple[str, str]]:
        """Select asset pairs using PCA-based similarity."""
        if self.original_returns is None:
            raise ValueError("Must call fit() before selecting pairs")

        if parameter_set is not None:
            if parameter_set not in PARAMETER_SETS:
                raise ValueError(f"Unknown parameter set: {parameter_set}")
            params = PARAMETER_SETS[parameter_set]
            n_components = params['n_components']
            similarity_threshold = params['similarity_threshold']

        logger.info(f"Selecting pairs using PCA with {n_components} components")

        scaler = StandardScaler()
        scaled = scaler.fit_transform(self.original_returns)

        pca = PCA(n_components=n_components, random_state=self.random_state)
        pca.fit(scaled)
        loadings = pca.components_.T

        norms = np.linalg.norm(loadings, axis=1, keepdims=True)
        norm_loadings = loadings / norms

        cos_sim = cosine_similarity(norm_loadings)

        columns = self.original_returns.columns
        pairs = []
        similarities = []

        for i in range(len(columns)):
            for j in range(i+1, len(columns)):
                if cos_sim[i, j] >= similarity_threshold:
                    pairs.append((columns[i], columns[j]))
                    similarities.append(cos_sim[i, j])

        self.selected_pairs[parameter_set or 'custom'] = pairs
        self.pair_similarities[parameter_set or 'custom'] = similarities
        self.pca_loadings[parameter_set or 'custom'] = loadings
        self.explained_variance[parameter_set or 'custom'] = pca.explained_variance_ratio_

        return pairs

    def analyze_pair_stability(self,
                             parameter_set1: str,
                             parameter_set2: str) -> float:
        """Analyze stability between two sets of pairs."""
        if parameter_set1 not in self.selected_pairs or \
           parameter_set2 not in self.selected_pairs:
            raise ValueError("Both parameter sets must have selected pairs")

        pairs1 = self.selected_pairs[parameter_set1]
        pairs2 = self.selected_pairs[parameter_set2]

        pairs_set1 = set(map(tuple, map(sorted, pairs1)))
        pairs_set2 = set(map(tuple, map(sorted, pairs2)))

        if not pairs_set1 or not pairs_set2:
            return 0.0

        intersection = pairs_set1.intersection(pairs_set2)
        union = pairs_set1.union(pairs_set2)

        return len(intersection) / len(union)

    def _analyze_denoising(self,
                         method: str,
                         explained_variance: Optional[np.ndarray] = None) -> None:
        """Analyze results of a denoising method."""
        if method not in self.denoised_returns:
            raise ValueError(f"No results found for method: {method}")

        denoised = self.denoised_returns[method]

        orig_corr = self.original_returns.corr()
        denoised_corr = denoised.corr()

        reconstruction_error = np.mean(
            (self.original_returns - denoised) ** 2
        )
        corr_diff = orig_corr - denoised_corr

        analysis = {
            'reconstruction_error': float(reconstruction_error),
            'max_correlation_change': float(np.max(np.abs(corr_diff))),
            'avg_correlation_change': float(np.mean(np.abs(corr_diff))),
            'orig_avg_correlation': float(np.mean(np.triu(orig_corr, k=1))),
            'denoised_avg_correlation': float(np.mean(np.triu(denoised_corr, k=1))),
            'variance_reduction': float(
                1 - (denoised.var().mean() / self.original_returns.var().mean())
            )
        }

        if explained_variance is not None:
            analysis['explained_variance_ratio'] = explained_variance.tolist()
            analysis['total_variance_explained'] = float(np.sum(explained_variance))

        self.denoising_results[method] = analysis

    def plot_denoising_comparison(self,
                                methods: Optional[List[str]] = None,
                                n_assets: int = 5) -> go.Figure:
        """Plot comparison of original vs denoised returns."""
        if methods is None:
            methods = list(self.denoised_returns.keys())

        top_assets = self.original_returns.var().nlargest(n_assets).index

        fig = make_subplots(
            rows=n_assets,
            cols=1,
            subplot_titles=[f"Returns Comparison - {asset}"
                          for asset in top_assets]
        )

        colors = {
            'original': 'black',
            'wavelet': 'blue',
            'pca': 'red',
            'hybrid': 'green',
            'rolling_pca': 'purple',
            'rolling_wavelet': 'orange'
        }

        for i, asset in enumerate(top_assets, 1):
            fig.add_trace(
                go.Scatter(
                    x=self.original_returns.index,
                    y=self.original_returns[asset],
                    name=f"{asset} Original",
                    line=dict(color=colors['original'], width=1)
                ),
                row=i, col=1
            )

            for method in methods:
                if method in self.denoised_returns:
                    method_color = colors.get(method, 'gray')
                    fig.add_trace(
                        go.Scatter(
                            x=self.denoised_returns[method].index,
                            y=self.denoised_returns[method][asset],
                            name=f"{asset} {method.capitalize()}",
                            line=dict(color=method_color, width=1)
                        ),
                        row=i, col=1
                    )

        fig.update_layout(
            height=300 * n_assets,
            showlegend=True,
            title="Original vs Denoised Returns Comparison"
        )

        return fig

    def plot_pca_loadings(self, parameter_set: str) -> go.Figure:
        """Plot PCA loadings for pair selection."""
        if parameter_set not in self.pca_loadings:
            raise ValueError(f"No loadings found for parameter set: {parameter_set}")

        loadings = self.pca_loadings[parameter_set]
        if loadings.shape[1] < 2:
            raise ValueError("Need at least 2 components for plotting")

        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=loadings[:, 0],
            y=loadings[:, 1],
            mode='markers+text',
            text=self.original_returns.columns,
            textposition="top center",
            name='Assets'
        ))

        fig.update_layout(
            title=f'PCA Loadings - {parameter_set}',
            xaxis_title='PC1',
            yaxis_title='PC2',
            width=1000,
            height=800
        )

        return fig

    def plot_pair_similarity_matrix(self, parameter_set: str) -> go.Figure:
        """Plot similarity matrix for selected pairs."""
        if parameter_set not in self.selected_pairs:
            raise ValueError(f"No pairs found for parameter set: {parameter_set}")

        pairs = self.selected_pairs[parameter_set]
        similarities = self.pair_similarities[parameter_set]

        assets = list(set([asset for pair in pairs for asset in pair]))
        n_assets = len(assets)
        sim_matrix = np.zeros((n_assets, n_assets))

        asset_to_idx = {asset: i for i, asset in enumerate(assets)}
        for pair, sim in zip(pairs, similarities):
            i, j = asset_to_idx[pair[0]], asset_to_idx[pair[1]]
            sim_matrix[i, j] = sim
            sim_matrix[j, i] = sim

        fig = go.Figure(data=go.Heatmap(
            z=sim_matrix,
            x=assets,
            y=assets,
            colorscale='Viridis',
            zmin=0,
            zmax=1
        ))

        fig.update_layout(
            title=f'Pair Similarities - {parameter_set}',
            width=1000,
            height=800
        )

        return fig

    def plot_denoising_effect_on_correlations(self, method: str) -> go.Figure:
        """Plot the effect of denoising on the correlation structure."""
        if method not in self.denoised_returns:
            raise ValueError(f"No results found for method: {method}")

        orig_corr = self.original_returns.corr()
        denoised_corr = self.denoised_returns[method].corr()

        # Get the difference in correlations
        diff_corr = denoised_corr - orig_corr

        fig = make_subplots(
            rows=1,
            cols=3,
            subplot_titles=[
                "Original Correlations",
                f"Denoised Correlations ({method})",
                "Correlation Difference"
            ],
            horizontal_spacing=0.05
        )

        fig.add_trace(
            go.Heatmap(
                z=orig_corr.values,
                x=orig_corr.columns,
                y=orig_corr.index,
                colorscale='RdBu',
                zmid=0,
                showscale=False
            ),
            row=1, col=1
        )

        fig.add_trace(
            go.Heatmap(
                z=denoised_corr.values,
                x=denoised_corr.columns,
                y=denoised_corr.index,
                colorscale='RdBu',
                zmid=0,
                showscale=False
            ),
            row=1, col=2
        )

        fig.add_trace(
            go.Heatmap(
                z=diff_corr.values,
                x=diff_corr.columns,
                y=diff_corr.index,
                colorscale='RdBu',
                zmid=0,
                showscale=True
            ),
            row=1, col=3
        )

        fig.update_layout(
            height=800,
            width=1500,
            title=f"Effect of {method.capitalize()} Denoising on Correlation Structure"
        )

        fig.update_xaxes(tickangle=45)

        return fig

    @staticmethod
    def _validate_input(returns: pd.DataFrame) -> None:
        """Validate input data."""
        if returns.empty:
            raise ValueError("Empty DataFrame provided")

        if returns.isnull().any().any():
            raise ValueError("Input contains missing values")

        if (returns.std() == 0).any():
            raise ValueError("Input contains constant columns")

    def compare_denoising_methods(self,
                                methods: Optional[List[str]] = None,
                                metrics: Optional[List[str]] = None) -> pd.DataFrame:
        """Compare denoising methods based on various metrics.

        Args:
            methods: List of methods to compare (None for all available)
            metrics: List of metrics to include (None for all available)

        Returns:
            pd.DataFrame: Comparison of denoising methods
        """
        if not self.denoising_results:
            raise ValueError("No denoising methods have been applied yet")

        if methods is None:
            methods = list(self.denoising_results.keys())
        else:
            methods = [m for m in methods if m in self.denoising_results]

        if not methods:
            raise ValueError("No valid methods specified for comparison")

        if metrics is None:
            # Use all metrics from the first method's results
            metrics = list(self.denoising_results[methods[0]].keys())
            # Filter out non-scalar metrics
            metrics = [m for m in metrics if not isinstance(self.denoising_results[methods[0]][m], (list, dict))]

        comparison = {}
        for method in methods:
            comparison[method] = {
                metric: self.denoising_results[method].get(metric, None)
                for metric in metrics
            }

        return pd.DataFrame(comparison).T

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

def main():
    """Main execution function to demonstrate the enhanced AssetAnalyzer capabilities."""
    output_dir = "asset_analysis"
    os.makedirs(output_dir, exist_ok=True)

    try:
        logger.info("Loading price data...")
        prices_df = load_nasdaq100_data(r'data/raw')
        returns = prices_df.pct_change().dropna()

        # Initialize the analyzer
        analyzer = AssetAnalyzer(n_jobs=-1, random_state=42)
        analyzer.fit(returns)

        # Create output directories
        denoising_dir = os.path.join(output_dir, "denoising")
        pairs_dir = os.path.join(output_dir, "pairs")
        plots_dir = os.path.join(output_dir, "plots")
        assessment_dir = os.path.join(output_dir, "assessment")

        for directory in [denoising_dir, pairs_dir, plots_dir, assessment_dir]:
            os.makedirs(directory, exist_ok=True)

        logger.info("Starting enhanced denoising analysis...")

        # Apply different denoising methods
        analyzer.denoise_wavelet(wavelet='db4', threshold_mode='adaptive', threshold_multiplier=1.2)
        analyzer.denoise_pca(variance_threshold=0.9)
        analyzer.denoise_hybrid(weight_pca=0.6)
        analyzer.denoise_rolling(method='pca', window_size=63, variance_threshold=0.9)

        # Compare denoising methods
        comparison_df = analyzer.compare_denoising_methods()
        comparison_df.to_csv(os.path.join(assessment_dir, "denoising_comparison.csv"))

        # Create comparison visualization
        comparison_fig = analyzer.plot_denoising_comparison()
        comparison_fig.write_html(os.path.join(plots_dir, "denoising_comparison.html"))

        # Visualize effect on correlations
        for method in analyzer.denoised_returns.keys():
            corr_effect_fig = analyzer.plot_denoising_effect_on_correlations(method)
            corr_effect_fig.write_html(os.path.join(plots_dir, f"{method}_correlation_effect.html"))

        # Save denoised returns
        for method, returns_df in analyzer.denoised_returns.items():
            returns_df.to_csv(os.path.join(denoising_dir, f"{method}_denoised_returns.csv"))

        logger.info("Assessing denoising quality...")

        # Run quality assessment on each method
        quality_results = {}
        for method in analyzer.denoised_returns.keys():
            quality_results[method] = analyzer.assess_denoising_quality(method)

        # Save quality assessment results
        quality_df = pd.DataFrame({
            method: {
                'correlation_preservation': results['avg_correlation_preservation'],
                'snr': results['avg_snr'],
                'predictive_score': results['avg_predictive_score']
            } for method, results in quality_results.items()
        }).T

        quality_df.to_csv(os.path.join(assessment_dir, "denoising_quality.csv"))

        logger.info("Starting pair selection analysis...")

        # Run pair selection with different parameter sets
        for param_set in PARAMETER_SETS.keys():
            pairs = analyzer.select_pairs(parameter_set=param_set)

            pairs_df = pd.DataFrame(pairs, columns=['Asset1', 'Asset2'])
            pairs_df['Similarity'] = analyzer.pair_similarities[param_set]
            pairs_df.to_csv(os.path.join(pairs_dir, f"{param_set}_pairs.csv"), index=False)

            loadings_fig = analyzer.plot_pca_loadings(param_set)
            loadings_fig.write_html(os.path.join(plots_dir, f"{param_set}_loadings.html"))

            similarity_fig = analyzer.plot_pair_similarity_matrix(param_set)
            similarity_fig.write_html(os.path.join(plots_dir, f"{param_set}_similarities.html"))

        logger.info("Analyzing pair stability...")

        # Analyze stability between different parameter sets
        stability_results = []
        param_sets = list(PARAMETER_SETS.keys())

        for i in range(len(param_sets)):
            for j in range(i + 1, len(param_sets)):
                param1, param2 = param_sets[i], param_sets[j]
                stability = analyzer.analyze_pair_stability(param1, param2)
                stability_results.append({
                    'parameter_set1': param1,
                    'parameter_set2': param2,
                    'stability': stability
                })

        stability_df = pd.DataFrame(stability_results)
        stability_df.to_csv(os.path.join(pairs_dir, "pair_stability.csv"), index=False)

        # Generate comprehensive analysis report
        with open(os.path.join(output_dir, "enhanced_analysis_report.txt"), "w") as f:
            f.write("Enhanced Asset Analysis Report\n")
            f.write("============================\n\n")

            f.write("1. Denoising Analysis\n")
            f.write("-------------------\n\n")

            for method, results in analyzer.denoising_results.items():
                f.write(f"\n{method.capitalize()} Denoising Results:\n")
                f.write("-" * 40 + "\n")
                for key, value in results.items():
                    if isinstance(value, float):
                        f.write(f"{key}: {value:.4f}\n")
                    elif isinstance(value, list):
                        f.write(f"{key}:\n")
                        for i, v in enumerate(value):
                            f.write(f"  Component {i+1}: {v:.4f}\n")
                    else:
                        f.write(f"{key}: {value}\n")

                # Add quality assessment results if available
                if method in quality_results:
                    f.write("\nQuality Assessment:\n")
                    f.write("-" * 25 + "\n")
                    f.write(f"Correlation Preservation: {quality_results[method]['avg_correlation_preservation']:.4f}\n")
                    f.write(f"Signal-to-Noise Ratio: {quality_results[method]['avg_snr']:.4f}\n")
                    if quality_results[method]['avg_predictive_score'] is not None:
                        f.write(f"Predictive Score: {quality_results[method]['avg_predictive_score']:.4f}\n")

            f.write("\n\n2. Pair Selection Analysis\n")
            f.write("------------------------\n\n")

            for param_set in PARAMETER_SETS.keys():
                f.write(f"\nParameter Set: {param_set}\n")
                f.write("-" * 40 + "\n")

                pairs = analyzer.selected_pairs[param_set]
                similarities = analyzer.pair_similarities[param_set]
                explained_var = analyzer.explained_variance[param_set]

                f.write(f"Number of pairs found: {len(pairs)}\n")
                f.write(f"Average similarity: {np.mean(similarities):.4f}\n")
                f.write(f"Variance explained by components:\n")
                for i, var in enumerate(explained_var):
                    f.write(f"  PC{i+1}: {var:.2%}\n")

                f.write("\nTop 10 Most Similar Pairs:\n")
                sorted_pairs = sorted(zip(pairs, similarities), key=lambda x: x[1], reverse=True)
                for pair, sim in sorted_pairs[:10]:
                    f.write(f"  {pair[0]} - {pair[1]}: {sim:.4f}\n")

            f.write("\n\n3. Stability Analysis\n")
            f.write("-------------------\n\n")

            for _, row in stability_df.iterrows():
                f.write(f"{row['parameter_set1']} vs {row['parameter_set2']}: ")
                f.write(f"{row['stability']:.2%} stability\n")

        logger.info(f"Enhanced analysis complete. Results saved in {output_dir}")

    except Exception as e:
        logger.error(f"Error during enhanced analysis: {str(e)}")
        raise

if __name__ == "__main__":
    main()