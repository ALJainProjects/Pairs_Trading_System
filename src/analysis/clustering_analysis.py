import pandas as pd
import numpy as np
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
import networkx as nx
import plotly.graph_objects as go
from typing import List, Dict, Tuple
from concurrent.futures import ProcessPoolExecutor, as_completed
from config.logging_config import logger

# Note: dtaidistance is a specialized library. Ensure it's in requirements.txt.
try:
    from dtaidistance import dtw
except ImportError:
    logger.error("dtaidistance library not found. Please run 'pip install dtaidistance'.")
    dtw = None

class AssetClusteringAnalyzer:
    """
    Performs asset clustering using various algorithms and similarity metrics,
    with support for dimensionality reduction and cluster evaluation.
    """
    def __init__(self, n_jobs: int = -1, random_state: int = 42):
        self.n_jobs = n_jobs
        self.random_state = random_state

    def _validate_input(self, data: pd.DataFrame):
        if not isinstance(data, pd.DataFrame) or data.empty:
            raise ValueError("Input must be a non-empty pandas DataFrame.")
        if data.isnull().any().any():
            raise ValueError("Input data contains NaN values.")

    def reduce_dimensionality(self, returns: pd.DataFrame, n_components: int = 10) -> pd.DataFrame:
        """
        Reduces the dimensionality of the returns data using PCA.

        Args:
            returns (pd.DataFrame): DataFrame of asset returns.
            n_components (int): The number of principal components to keep.

        Returns:
            pd.DataFrame: A DataFrame of the principal components.
        """
        self._validate_input(returns)
        scaled_returns = StandardScaler().fit_transform(returns.T)
        pca = PCA(n_components=n_components, random_state=self.random_state)
        principal_components = pca.fit_transform(scaled_returns)
        logger.info(f"Explained variance by {n_components} components: {np.sum(pca.explained_variance_ratio_):.2%}")
        return pd.DataFrame(principal_components, index=returns.columns)

    def calculate_dtw_distance_matrix(self, returns: pd.DataFrame) -> pd.DataFrame:
        """Calculates the Dynamic Time Warping distance matrix with parallelization."""
        if dtw is None:
            raise ImportError("dtaidistance library is required for DTW calculations.")
        self._validate_input(returns)
        
        symbols = returns.columns
        n_assets = len(symbols)
        distance_matrix = np.zeros((n_assets, n_assets))
        
        series_list = [returns[symbol].values.astype(np.double) for symbol in symbols]

        with ProcessPoolExecutor(max_workers=self.n_jobs if self.n_jobs != -1 else None) as executor:
            pairs = [(i, j) for i in range(n_assets) for j in range(i + 1, n_assets)]
            futures = {executor.submit(dtw.distance_fast, series_list[i], series_list[j]): (i, j) for i, j in pairs}

            for future in as_completed(futures):
                i, j = futures[future]
                dist = future.result()
                distance_matrix[i, j] = distance_matrix[j, i] = dist
                
        return pd.DataFrame(distance_matrix, index=symbols, columns=symbols)

    def kmeans_clustering(self, returns: pd.DataFrame, n_clusters: int = 5, use_pca: bool = True, n_components: int = 10) -> List[List[str]]:
        """Applies K-Means clustering, optionally on PCA-reduced data."""
        self._validate_input(returns)
        
        data_to_cluster = returns.T
        if use_pca:
            data_to_cluster = self.reduce_dimensionality(returns, n_components=n_components)
            
        kmeans = KMeans(n_clusters=n_clusters, random_state=self.random_state, n_init=10)
        labels = kmeans.fit_predict(data_to_cluster)
        
        clusters = [data_to_cluster.index[labels == i].tolist() for i in range(n_clusters)]
        return clusters

    def dbscan_clustering(self, returns: pd.DataFrame, eps: float = 0.5, min_samples: int = 2) -> List[List[str]]:
        """Applies DBSCAN clustering based on correlation distance."""
        self._validate_input(returns)
        corr_dist = 1 - returns.corr()
        
        dbscan = DBSCAN(eps=eps, min_samples=min_samples, metric='precomputed')
        labels = dbscan.fit_predict(corr_dist)
        
        clusters = [returns.columns[labels == i].tolist() for i in set(labels) if i != -1]
        return clusters

    def agglomerative_clustering(self, returns: pd.DataFrame, n_clusters: int = 5, linkage: str = 'ward') -> List[List[str]]:
        """Applies Agglomerative (Hierarchical) clustering."""
        self._validate_input(returns)
        
        agg = AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage)
        labels = agg.fit_predict(returns.T)
        
        clusters = [returns.columns[labels == i].tolist() for i in range(n_clusters)]
        return clusters

    def graph_based_clustering(self, correlation_matrix: pd.DataFrame, threshold: float = 0.8) -> List[List[str]]:
        """Identifies clusters from a correlation matrix using a graph-based approach."""
        adj_matrix = (correlation_matrix.abs() > threshold).astype(int)
        graph = nx.from_pandas_adjacency(adj_matrix)
        graph.remove_edges_from(nx.selfloop_edges(graph))
        
        clusters = [list(component) for component in nx.connected_components(graph)]
        return clusters

    def calculate_silhouette_score(self, returns: pd.DataFrame, cluster_labels: np.ndarray, metric: str = 'euclidean') -> float:
        """
        Calculates the silhouette score to evaluate the quality of the clustering.
        """
        if len(np.unique(cluster_labels)) < 2:
            logger.warning("Cannot calculate silhouette score with less than 2 clusters.")
            return -1.0
            
        return silhouette_score(returns.T, cluster_labels, metric=metric)
    
    def plot_cluster_heatmap(self, correlation_matrix: pd.DataFrame, clusters: List[List[str]], title: str) -> go.Figure:
        """Generates a heatmap with assets reordered according to their cluster."""
        ordered_assets = [asset for cluster in clusters for asset in cluster]
        unclustered = [asset for asset in correlation_matrix.columns if asset not in ordered_assets]
        final_order = ordered_assets + unclustered
        
        ordered_matrix = correlation_matrix.loc[final_order, final_order]
        
        fig = go.Figure(data=go.Heatmap(
            z=ordered_matrix.values,
            x=ordered_matrix.columns,
            y=ordered_matrix.columns,
            colorscale='RdBu',
            zmid=0
        ))
        
        shapes = []
        current_pos = 0
        for cluster in clusters:
            size = len(cluster)
            current_pos += size
            shapes.append(go.layout.Shape(
                type="rect",
                x0=current_pos - size - 0.5, y0=current_pos - size - 0.5,
                x1=current_pos - 0.5, y1=current_pos - 0.5,
                line=dict(color="black", width=2)
            ))
        
        fig.update_layout(
            title=title,
            height=800,
            width=800,
            shapes=shapes
        )
        return fig