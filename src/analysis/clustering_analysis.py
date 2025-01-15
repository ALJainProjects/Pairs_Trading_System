"""
Consolidated Clustering and Similarity Analysis Module

This module provides functionality for:
1. Asset clustering using various methods:
   - K-Means (with dimensionality reduction)
   - DBSCAN
   - Agglomerative Clustering (with proper distance metrics)
   - Graph-based clustering
2. Similarity measures:
   - Correlation-based
   - Mutual Information
   - Dynamic Time Warping (with parallelization)
   - Cosine Similarity
3. Visualization and analysis tools

Features:
- Parallel processing for computationally intensive operations
- Proper distance metric handling for each clustering method
- Dimensionality reduction for high-dimensional feature spaces
- Robust error handling and validation
"""
import random
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mutual_info_score, pairwise_distances
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.decomposition import PCA
from statsmodels.tsa.stattools import coint
from dtaidistance import dtw
import networkx as nx
import plotly.graph_objects as go
from typing import List, Dict, Optional, Tuple
from concurrent.futures import ProcessPoolExecutor, as_completed
from config.logging_config import logger
import os

SIMILARITY_METRICS = {
    'correlation': 'correlation',
    'mutual_info': 'mutual_info',
    'dtw': 'dtw',
    'cosine': 'cosine'
}

def calculate_coint_work(
    prices: pd.DataFrame,
    pair: Tuple[int, int]
) -> Tuple[int, int, float]:
    i, j = pair
    _, pvalue, _ = coint(prices.iloc[:, i], prices.iloc[:, j])
    return i, j, pvalue

def calculate_dtw_work(
    i: int,
    j: int,
    series_i: np.ndarray,
    series_j: np.ndarray
) -> Tuple[int, int, float]:
    dist = dtw.distance_fast(series_i, series_j)
    return i, j, dist

class AssetClusteringAnalyzer:
    """Main class for asset clustering analysis."""

    def __init__(self, n_jobs: int = -1, random_state: int = 42):
        """
        Initialize the analyzer.

        Args:
            n_jobs: Number of parallel jobs (-1 for all cores)
            random_state: Random seed for reproducibility
        """
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.similarity_matrices_ = {}
        self.clustering_results_ = {}
        self.feature_importances_ = {}

    def calculate_similarity_matrix(self, returns: pd.DataFrame,
                                  metric: str = 'correlation',
                                  **kwargs) -> pd.DataFrame:
        """Calculate similarity matrix using specified metric."""
        if metric == 'correlation':
            return returns.corr()
        elif metric == 'mutual_info':
            return self._mutual_information(returns, **kwargs)
        elif metric == 'dtw':
            return -self._dtw_distance_matrix(returns)
        elif metric == 'cosine':
            return self._cosine_similarity_matrix(returns)
        else:
            raise ValueError(f"Unknown similarity metric: {metric}")

    # noinspection PyTypeChecker
    def _mutual_information(self, returns: pd.DataFrame,
                          n_bins: int = 10) -> pd.DataFrame:
        """Calculate mutual information between assets."""
        self._validate_input(returns)

        discretizer = KBinsDiscretizer(n_bins=n_bins, encode='ordinal',
                                     strategy='uniform')
        discretized = discretizer.fit_transform(returns.T)
        columns = returns.columns
        n_assets = len(columns)

        mi_matrix = pd.DataFrame(np.zeros((n_assets, n_assets)),
                               index=columns, columns=columns)

        with ProcessPoolExecutor(max_workers=4) as executor:
            futures = []
            for i in range(n_assets):
                for j in range(i + 1, n_assets):
                    futures.append(
                        executor.submit(
                            mutual_info_score,
                            discretized[i],
                            discretized[j]
                        )
                    )

            idx = 0
            for i in range(n_assets):
                for j in range(i + 1, n_assets):
                    mi = futures[idx].result()
                    mi_matrix.iloc[i, j] = mi
                    mi_matrix.iloc[j, i] = mi
                    idx += 1

        return mi_matrix

    def _cosine_similarity_matrix(self, returns: pd.DataFrame) -> pd.DataFrame:
        """Calculate cosine similarity between assets."""
        self._validate_input(returns)
        similarity = 1 - pairwise_distances(returns.T, metric="cosine",
                                          n_jobs=self.n_jobs)
        return pd.DataFrame(similarity, index=returns.columns,
                          columns=returns.columns)

    def _dtw_distance_matrix(self, returns: pd.DataFrame) -> pd.DataFrame:
        """Calculate DTW distance matrix between assets with parallelization."""
        self._validate_input(returns)

        columns = returns.columns
        n_assets = len(columns)
        distance_mat = np.zeros((n_assets, n_assets))

        # def calculate_dtw(i: int, j: int) -> Tuple[int, int, float]:
        #     dist = dtw.distance_fast(returns.iloc[:, i].values,
        #                            returns.iloc[:, j].values)
        #     return i, j, dist

        with ProcessPoolExecutor(max_workers=4) as executor:
            futures = []
            for i in range(n_assets):
                for j in range(i + 1, n_assets):
                    futures.append(
                        executor.submit(calculate_dtw_work, i, j, returns.iloc[:, i].values, returns.iloc[:, j].values)
                    )

            for future in as_completed(futures):
                i, j, dist = future.result()
                distance_mat[i, j] = dist
                distance_mat[j, i] = dist

        # print(pd.DataFrame(distance_mat, index=columns, columns=columns))
        pd.DataFrame(distance_mat, index=columns, columns=columns).to_csv('clustering_analysis/results/dtw_matrix.csv')
        return pd.DataFrame(distance_mat, index=columns, columns=columns)

    def calculate_cointegration_matrix(self, prices: pd.DataFrame,
                                     max_pairs: Optional[int] = None) -> pd.DataFrame:
        """
        Calculate pairwise cointegration p-values with optional sampling.

        Args:
            prices: Price data
            max_pairs: Maximum number of pairs to test (random sampling if exceeded)
        """
        self._validate_input(prices)
        n = len(prices.columns)
        pvalue_matrix = np.zeros((n, n))

        pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
        if max_pairs and len(pairs) > max_pairs:
            pairs = random.sample(pairs, max_pairs)

        # def calculate_coint(pair: Tuple[int, int]) -> Tuple[int, int, float]:
        #     i, j = pair
        #     _, pvalue, _ = coint(prices.iloc[:, i], prices.iloc[:, j])
        #     return i, j, pvalue

        with ProcessPoolExecutor(max_workers=4) as executor:
            futures = [executor.submit(calculate_coint_work, prices, pair) for pair in pairs]

            for future in as_completed(futures):
                i, j, pvalue = future.result()
                pvalue_matrix[i, j] = pvalue
                pvalue_matrix[j, i] = pvalue

        return pd.DataFrame(pvalue_matrix, index=prices.columns,
                          columns=prices.columns)

    def kmeans_clustering(self, prices: pd.DataFrame, n_clusters: int = 5,
                         n_components: Optional[int] = None) -> List[List[str]]:
        """
        Apply K-Means clustering with dimensionality reduction.

        Args:
            prices: Price data
            n_clusters: Number of clusters
            n_components: Number of PCA components (None for automatic selection)
        """
        self._validate_input(prices)

        scaler = StandardScaler()
        scaled_prices = pd.DataFrame(
            scaler.fit_transform(prices),
            columns=prices.columns,
            index=prices.index
        )

        returns = prices.pct_change().dropna()
        volatility = returns.std()
        correlation = returns.corr()

        features = np.column_stack([
            scaled_prices.T,
            volatility.values.reshape(-1, 1),
            correlation.values
        ])

        if n_components is not None:
            pca = PCA(n_components=n_components, random_state=self.random_state)
            features = pca.fit_transform(features)
            self.feature_importances_['kmeans_pca'] = pd.Series(
                pca.explained_variance_ratio_,
                index=[f'PC{i+1}' for i in range(n_components)]
            )

        kmeans = KMeans(n_clusters=n_clusters, random_state=self.random_state,
                       n_init=10)
        labels = kmeans.fit_predict(features)

        clusters = []
        for i in range(n_clusters):
            cluster_assets = prices.columns[labels == i].tolist()
            clusters.append(cluster_assets)

        return clusters

    def dbscan_clustering(self, prices: pd.DataFrame, eps: float = 0.5,
                         min_samples: int = 2, metric: str = 'precomputed',
                         use_cointegration: bool = True) -> List[List[str]]:
        """Apply DBSCAN clustering with flexible distance metrics."""
        self._validate_input(prices)

        if use_cointegration:
            dist_matrix = self.calculate_cointegration_matrix(prices)
            dist_matrix = 1 - dist_matrix
        else:
            returns = prices.pct_change().dropna()
            dist_matrix = 1 - returns.corr()

        dbs = DBSCAN(eps=eps, min_samples=min_samples, metric=metric)
        labels = dbs.fit_predict(dist_matrix)

        clusters = []
        for label in set(labels):
            if label == -1:
                continue
            cluster_assets = prices.columns[labels == label].tolist()
            clusters.append(cluster_assets)

        return clusters

    def agglomerative_clustering(self, prices: pd.DataFrame,
                               n_clusters: int = 5,
                               metric: str = 'euclidean',
                               linkage: str = 'complete') -> List[List[str]]:
        """
        Apply Agglomerative clustering with proper distance metric.

        Note: 'ward' linkage is only allowed with 'euclidean' metric.
        """
        self._validate_input(prices)

        returns = prices.pct_change().dropna()
        if metric == 'precomputed':
            dist_matrix = 1 - returns.corr()
            agglom = AgglomerativeClustering(
                n_clusters=n_clusters,
                metric=metric,
                linkage=linkage
            )
            labels = agglom.fit_predict(dist_matrix)
        else:
            agglom = AgglomerativeClustering(
                n_clusters=n_clusters,
                metric=metric,
                linkage=linkage
            )
            labels = agglom.fit_predict(returns.T)

        clusters = []
        for i in range(n_clusters):
            cluster_assets = prices.columns[labels == i].tolist()
            clusters.append(cluster_assets)

        return clusters

    def graph_based_clustering(self, similarity_matrix: pd.DataFrame,
                             threshold: float = 0.8,
                             min_cluster_size: int = 2) -> List[List[str]]:
        """Identify clusters using graph-based approach with minimum size."""
        G = nx.Graph()
        G.add_nodes_from(similarity_matrix.columns)

        for i in similarity_matrix.columns:
            for j in similarity_matrix.columns:
                if i != j and similarity_matrix.loc[i, j] >= threshold:
                    G.add_edge(i, j)

        clusters = [list(component) for component in nx.connected_components(G)
                   if len(component) >= min_cluster_size]

        return clusters

    def analyze_clusters(self, clusters: List[List[str]],
                        similarity_matrix: pd.DataFrame) -> Dict:
        """Analyze properties of identified clusters."""
        if not clusters:
            return {
                'num_clusters': 0,
                'avg_cluster_size': 0,
                'max_cluster_size': 0,
                'min_cluster_size': 0,
                'total_assets': 0,
                'avg_intra_similarity': 0,
                'silhouette_score': 0
            }

        cluster_sizes = [len(cluster) for cluster in clusters]

        intra_sims = []
        for cluster in clusters:
            if len(cluster) > 1:
                cluster_sim = similarity_matrix.loc[cluster, cluster]
                upper_tri = cluster_sim.values[
                    np.triu_indices_from(cluster_sim.values, k=1)
                ]
                intra_sims.extend(upper_tri)

        analysis = {
            'num_clusters': len(clusters),
            'avg_cluster_size': np.mean(cluster_sizes),
            'max_cluster_size': max(cluster_sizes),
            'min_cluster_size': min(cluster_sizes),
            'total_assets': sum(cluster_sizes),
            'avg_intra_similarity': np.mean(intra_sims) if intra_sims else 0,
            'similarity_std': np.std(intra_sims) if intra_sims else 0
        }

        return analysis

    @staticmethod
    def _validate_input(data: pd.DataFrame) -> None:
        """Validate input data."""
        if data.empty:
            raise ValueError("Empty DataFrame provided")

        if data.isnull().any().any():
            raise ValueError("Input contains missing values")

        if (data.std() == 0).any():
            raise ValueError("Input contains constant columns")

    def plot_cluster_heatmap(self, matrix: pd.DataFrame,
                            clusters: List[List[str]],
                            title: str = "Clustered Heatmap") -> go.Figure:
        """Plot clustered heatmap of similarity/cointegration matrix."""
        ordered = []
        for cluster in clusters:
            ordered.extend(cluster)
        remaining = [c for c in matrix.columns if c not in ordered]
        ordered.extend(remaining)

        ordered_matrix = matrix.loc[ordered, ordered]

        fig = go.Figure(data=go.Heatmap(
            z=ordered_matrix.values,
            x=ordered_matrix.columns,
            y=ordered_matrix.index,
            colorscale='RdBu',
            zmid=0.5,
            showscale=True
        ))

        current_idx = 0
        shapes = []
        for cluster in clusters:
            cluster_size = len(cluster)
            if cluster_size > 1:
                shapes.extend([
                    dict(
                        type="line",
                        x0=-0.5,
                        x1=len(ordered_matrix) - 0.5,
                        y0=current_idx - 0.5,
                        y1=current_idx - 0.5,
                        line=dict(color="black", width=2)
                    ),
                    dict(
                        type="line",
                        x0=-0.5,
                        x1=len(ordered_matrix) - 0.5,
                        y0=current_idx + cluster_size - 0.5,
                        y1=current_idx + cluster_size - 0.5,
                        line=dict(color="black", width=2)
                    ),
                    dict(
                        type="line",
                        x0=current_idx - 0.5,
                        x1=current_idx - 0.5,
                        y0=-0.5,
                        y1=len(ordered_matrix) - 0.5,
                        line=dict(color="black", width=2)
                    ),
                    dict(
                        type="line",
                        x0=current_idx + cluster_size - 0.5,
                        x1=current_idx + cluster_size - 0.5,
                        y0=-0.5,
                        y1=len(ordered_matrix) - 0.5,
                        line=dict(color="black", width=2)
                    )
                ])
            current_idx += cluster_size

        fig.update_layout(
            title=title,
            xaxis_title="Assets",
            yaxis_title="Assets",
            width=1000,
            height=800,
            shapes=shapes
        )

        fig.update_xaxes(tickangle=45)

        return fig

def main():
    """Main execution function."""
    output_dir = "clustering_analysis"
    os.makedirs(output_dir, exist_ok=True)

    analyzer = AssetClusteringAnalyzer(n_jobs=-1, random_state=42)

    try:
        logger.info("Loading price data...")
        prices_df = load_nasdaq100_data(r'C:\Users\arnav\Downloads\pairs_trading_system\data\raw')
        returns = prices_df.pct_change().dropna()

        similarity_matrices = {
            'correlation': analyzer.calculate_similarity_matrix(
                returns, metric='correlation'
            ),
            'mutual_info': analyzer.calculate_similarity_matrix(
                returns, metric='mutual_info', n_bins=10
            ),
            'cosine': analyzer.calculate_similarity_matrix(
                returns, metric='cosine'
            )
        }

        try:
            similarity_matrices['dtw'] = analyzer.calculate_similarity_matrix(
                returns, metric='dtw'
            )
            similarity_matrices['dtw'] *= -1
        except Exception as e:
            logger.warning(f"DTW calculation failed: {str(e)}")

        clustering_results = {
            'kmeans': analyzer.kmeans_clustering(
                prices_df,
                n_clusters=5,
                n_components=10
            ),
            'dbscan': analyzer.dbscan_clustering(
                prices_df,
                eps=0.1,
                min_samples=2,
                use_cointegration=True
            ),
            'agglomerative': analyzer.agglomerative_clustering(
                prices_df,
                n_clusters=5,
                metric='euclidean',
                linkage='complete'
            )
        }

        for metric_name, sim_matrix in similarity_matrices.items():
            clustering_results[f'graph_{metric_name}'] = \
                analyzer.graph_based_clustering(
                    sim_matrix,
                    threshold=0.8,
                    min_cluster_size=2
                )

        plots_dir = os.path.join(output_dir, "plots")
        results_dir = os.path.join(output_dir, "results")
        os.makedirs(plots_dir, exist_ok=True)
        os.makedirs(results_dir, exist_ok=True)

        analysis_results = {}

        for method_name, clusters in clustering_results.items():
            # print(method_name)
            if method_name.startswith('graph_'):
                metric = method_name.split('_', 1)[1]
                sim_matrix = similarity_matrices[metric]
            else:
                sim_matrix = similarity_matrices['correlation']

            analysis = analyzer.analyze_clusters(clusters, sim_matrix)
            analysis_results[method_name] = analysis

            fig = analyzer.plot_cluster_heatmap(
                sim_matrix,
                clusters,
                f"Cluster Analysis - {method_name}"
            )
            fig.write_html(os.path.join(plots_dir, f"{method_name}_clusters.html"))

            with open(os.path.join(results_dir, f"{method_name}_clusters.txt"), "w") as f:
                f.write(f"Clustering Method: {method_name}\n")
                f.write("=" * 50 + "\n\n")
                for i, cluster in enumerate(clusters):
                    f.write(f"Cluster {i + 1} ({len(cluster)} assets):\n")
                    f.write(", ".join(cluster) + "\n\n")

        with open(os.path.join(output_dir, "clustering_summary.txt"), "w") as f:
            f.write("Clustering Analysis Summary\n")
            f.write("=========================\n\n")

            for method_name, analysis in analysis_results.items():
                f.write(f"\nMethod: {method_name}\n")
                f.write("-" * 40 + "\n")
                for key, value in analysis.items():
                    if isinstance(value, float):
                        f.write(f"{key}: {value:.3f}\n")
                    else:
                        f.write(f"{key}: {value}\n")

            if 'kmeans_pca' in analyzer.feature_importances_:
                f.write("\nPCA Component Importance (K-Means):\n")
                f.write("-" * 40 + "\n")
                for idx, imp in analyzer.feature_importances_['kmeans_pca'].items():
                    f.write(f"{idx}: {imp:.3f}\n")

        summary_data = []
        for method_name, analysis in analysis_results.items():
            row = {'method': method_name}
            row.update(analysis)
            summary_data.append(row)

        summary_df = pd.DataFrame(summary_data)
        summary_df.to_csv(os.path.join(results_dir, "clustering_summary.csv"), index=False)

        stability_results = []
        methods = list(clustering_results.keys())

        for i in range(len(methods)):
            for j in range(i + 1, len(methods)):
                method1, method2 = methods[i], methods[j]
                clusters1 = clustering_results[method1]
                clusters2 = clustering_results[method2]

                sets1 = {frozenset(cluster) for cluster in clusters1}
                sets2 = {frozenset(cluster) for cluster in clusters2}

                intersection = len(sets1.intersection(sets2))
                union = len(sets1.union(sets2))
                stability = intersection / union if union > 0 else 0

                stability_results.append({
                    'method1': method1,
                    'method2': method2,
                    'stability': stability
                })

        stability_df = pd.DataFrame(stability_results)
        stability_df.to_csv(os.path.join(results_dir, "cluster_stability.csv"), index=False)

        with open(os.path.join(output_dir, "stability_analysis.txt"), "w") as f:
            f.write("Cluster Stability Analysis\n")
            f.write("========================\n\n")
            for row in stability_results:
                f.write(f"{row['method1']} vs {row['method2']}: {row['stability']:.3f}\n")

        logger.info("Clustering analysis complete. Results saved in: " + output_dir)

    except Exception as e:
        logger.error(f"Error during clustering analysis: {str(e)}")
        raise

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

if __name__ == "__main__":
    main()