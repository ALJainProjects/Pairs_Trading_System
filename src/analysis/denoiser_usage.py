"""
Comprehensive Asset Analysis Module

This module provides functionality for:
1. Return Denoising:
   - Wavelet-based denoising (with multiple threshold methods)
   - PCA-based denoising (with flexible scaling)
2. Pair Selection:
   - PCA-based pair identification
   - Similarity analysis
3. Analysis and Visualization Tools

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
from sklearn.preprocessing import (StandardScaler, RobustScaler,
                                 MinMaxScaler, MaxAbsScaler)
from sklearn.metrics.pairwise import cosine_similarity
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Dict, Tuple, Optional, List, Set
from concurrent.futures import ProcessPoolExecutor, as_completed
import warnings
import os
from config.logging_config import logger

# Configuration
LOOKBACK_PERIODS = {
    '1M': 21,    # ~1 month
    '3M': 63,    # ~3 months
    '6M': 126,   # ~6 months
    '12M': 252   # ~12 months
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

        # Denoising attributes
        self.original_returns = None
        self.denoised_returns = {}
        self.denoising_results = {}

        # Pair selection attributes
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
                       level: int = 1,
                       threshold: float = 0.04) -> pd.DataFrame:
        """Apply wavelet-based denoising."""
        if self.original_returns is None:
            raise ValueError("Must call fit() before denoising")

        logger.info(f"Applying wavelet denoising: {wavelet}, level {level}")

        denoised = pd.DataFrame(index=self.original_returns.index,
                              columns=self.original_returns.columns)

        for ticker in self.original_returns.columns:
            series = self.original_returns[ticker]
            coeff = pywt.wavedec(series, wavelet, level=level)

            sigma = np.median(np.abs(coeff[-1])) / 0.6745
            uthresh = sigma * threshold

            denoised_coeff = coeff[:]
            denoised_coeff[1:] = [
                pywt.threshold(c, value=uthresh, mode='soft')
                for c in denoised_coeff[1:]
            ]

            reconstructed = pywt.waverec(denoised_coeff, wavelet)
            denoised[ticker] = reconstructed[:len(series)]

        self.denoised_returns['wavelet'] = denoised
        self._analyze_denoising('wavelet')
        return denoised

    def denoise_pca(self, n_components: int = 2) -> pd.DataFrame:
        """Apply PCA-based denoising."""
        if self.original_returns is None:
            raise ValueError("Must call fit() before denoising")

        logger.info(f"Applying PCA denoising with {n_components} components")

        scaler = StandardScaler()
        scaled_returns = scaler.fit_transform(self.original_returns)

        pca = PCA(n_components=n_components)
        principal_components = pca.fit_transform(scaled_returns)
        reconstructed = pca.inverse_transform(principal_components)

        denoised = pd.DataFrame(
            scaler.inverse_transform(reconstructed),
            index=self.original_returns.index,
            columns=self.original_returns.columns
        )

        self.denoised_returns['pca'] = denoised
        self._analyze_denoising('pca', pca.explained_variance_ratio_)
        return denoised

    def select_pairs(self,
                    n_components: int = 2,
                    similarity_threshold: float = 0.9,
                    parameter_set: Optional[str] = None) -> List[Tuple[str, str]]:
        """Select asset pairs using PCA-based similarity."""
        if self.original_returns is None:
            raise ValueError("Must call fit() before selecting pairs")

        # Use parameter set if provided
        if parameter_set is not None:
            if parameter_set not in PARAMETER_SETS:
                raise ValueError(f"Unknown parameter set: {parameter_set}")
            params = PARAMETER_SETS[parameter_set]
            n_components = params['n_components']
            similarity_threshold = params['similarity_threshold']

        logger.info(f"Selecting pairs using PCA with {n_components} components")

        # Standardize returns
        scaler = StandardScaler()
        scaled = scaler.fit_transform(self.original_returns)

        # Fit PCA
        pca = PCA(n_components=n_components)
        pca.fit(scaled)
        loadings = pca.components_.T

        # Normalize loadings
        norms = np.linalg.norm(loadings, axis=1, keepdims=True)
        norm_loadings = loadings / norms

        # Calculate similarities
        cos_sim = cosine_similarity(norm_loadings)

        # Find pairs
        columns = self.original_returns.columns
        pairs = []
        similarities = []

        for i in range(len(columns)):
            for j in range(i+1, len(columns)):
                if cos_sim[i, j] >= similarity_threshold:
                    pairs.append((columns[i], columns[j]))
                    similarities.append(cos_sim[i, j])

        # Store results
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
            'reconstruction_error': reconstruction_error,
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
            'pca': 'red'
        }

        for i, asset in enumerate(top_assets, 1):
            # Plot original returns
            fig.add_trace(
                go.Scatter(
                    x=self.original_returns.index,
                    y=self.original_returns[asset],
                    name=f"{asset} Original",
                    line=dict(color=colors['original'], width=1)
                ),
                row=i, col=1
            )

            # Plot denoised returns
            for method in methods:
                if method in self.denoised_returns:
                    fig.add_trace(
                        go.Scatter(
                            x=self.denoised_returns[method].index,
                            y=self.denoised_returns[method][asset],
                            name=f"{asset} {method.capitalize()}",
                            line=dict(color=colors[method], width=1)
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

        # Plot first two components
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

        # Create matrix of unique assets
        assets = list(set([asset for pair in pairs for asset in pair]))
        n_assets = len(assets)
        sim_matrix = np.zeros((n_assets, n_assets))

        # Fill similarity matrix
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

    @staticmethod
    def _validate_input(returns: pd.DataFrame) -> None:
        """Validate input data."""
        if returns.empty:
            raise ValueError("Empty DataFrame provided")

        if returns.isnull().any().any():
            raise ValueError("Input contains missing values")

        if (returns.std() == 0).any():
            raise ValueError("Input contains constant columns")

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
    """Main execution function."""
    # Create output directory
    output_dir = "asset_analysis"
    os.makedirs(output_dir, exist_ok=True)

    try:
        # Load data
        logger.info("Loading price data...")
        prices_df = load_nasdaq100_data(r'C:\Users\arnav\Downloads\pairs_trading_system\data\raw')
        returns = prices_df.pct_change().dropna()

        # Initialize analyzer
        analyzer = AssetAnalyzer(n_jobs=-1, random_state=42)
        analyzer.fit(returns)

        # Create output directories
        denoising_dir = os.path.join(output_dir, "denoising")
        pairs_dir = os.path.join(output_dir, "pairs")
        plots_dir = os.path.join(output_dir, "plots")

        for directory in [denoising_dir, pairs_dir, plots_dir]:
            os.makedirs(directory, exist_ok=True)

        # 1. Denoising Analysis
        logger.info("Starting denoising analysis...")

        # Apply different denoising methods
        analyzer.denoise_wavelet(wavelet='db1', level=1, threshold=0.04)
        analyzer.denoise_pca(n_components=5)

        # Generate denoising plots
        comparison_fig = analyzer.plot_denoising_comparison()
        comparison_fig.write_html(
            os.path.join(plots_dir, "denoising_comparison.html")
        )

        # Save denoised returns
        for method, returns_df in analyzer.denoised_returns.items():
            returns_df.to_csv(
                os.path.join(denoising_dir, f"{method}_denoised_returns.csv")
            )

        # 2. Pair Selection Analysis
        logger.info("Starting pair selection analysis...")

        # Apply pair selection with different parameter sets
        for param_set in PARAMETER_SETS.keys():
            pairs = analyzer.select_pairs(parameter_set=param_set)

            # Save pairs
            pairs_df = pd.DataFrame(pairs, columns=['Asset1', 'Asset2'])
            pairs_df['Similarity'] = analyzer.pair_similarities[param_set]
            pairs_df.to_csv(
                os.path.join(pairs_dir, f"{param_set}_pairs.csv"),
                index=False
            )

            # Generate and save plots
            loadings_fig = analyzer.plot_pca_loadings(param_set)
            loadings_fig.write_html(
                os.path.join(plots_dir, f"{param_set}_loadings.html")
            )

            similarity_fig = analyzer.plot_pair_similarity_matrix(param_set)
            similarity_fig.write_html(
                os.path.join(plots_dir, f"{param_set}_similarities.html")
            )

        # 3. Stability Analysis
        logger.info("Analyzing pair stability...")

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
        stability_df.to_csv(
            os.path.join(pairs_dir, "pair_stability.csv"),
            index=False
        )

        # 4. Create Comprehensive Report
        with open(os.path.join(output_dir, "analysis_report.txt"), "w") as f:
            f.write("Comprehensive Asset Analysis Report\n")
            f.write("================================\n\n")

            # Denoising Results
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

            # Pair Selection Results
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
                sorted_pairs = sorted(zip(pairs, similarities),
                                   key=lambda x: x[1], reverse=True)
                for pair, sim in sorted_pairs[:10]:
                    f.write(f"  {pair[0]} - {pair[1]}: {sim:.4f}\n")

            # Stability Analysis
            f.write("\n\n3. Stability Analysis\n")
            f.write("-------------------\n\n")

            for _, row in stability_df.iterrows():
                f.write(f"{row['parameter_set1']} vs {row['parameter_set2']}: ")
                f.write(f"{row['stability']:.2%} stability\n")

        logger.info(f"Analysis complete. Results saved in {output_dir}")

    except Exception as e:
        logger.error(f"Error during analysis: {str(e)}")
        raise

if __name__ == "__main__":
    main()