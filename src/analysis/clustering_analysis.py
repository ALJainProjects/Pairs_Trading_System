import pandas as pd
import numpy as np
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering, MeanShift, estimate_bandwidth
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
import networkx as nx
import plotly.graph_objects as go
from typing import List, Dict, Tuple, Optional, Union
from concurrent.futures import ProcessPoolExecutor, as_completed
from config.logging_config import logger

# Note: dtaidistance is a specialized library. Ensure it's in requirements.txt.
try:
    from dtaidistance import dtw
    from dtaidistance import dtw_ndim  # For potential future multi-dimensional DTW
except ImportError:
    logger.error("dtaidistance library not found. Please run 'pip install dtaidistance'.")
    dtw = None
    dtw_ndim = None


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
            raise ValueError("Input data contains NaN values. Please handle them before clustering.")

    def reduce_dimensionality(self, returns: pd.DataFrame, n_components: int = 10) -> pd.DataFrame:
        """
        Reduces the dimensionality of the returns data using PCA.

        Args:
            returns (pd.DataFrame): DataFrame of asset returns. Rows are time, columns are assets.
            n_components (int): The number of principal components to keep.
                                If an integer, the number of components.
                                If a float between 0 and 1, the variance ratio to be explained.

        Returns:
            pd.DataFrame: A DataFrame of the principal components, with assets as index.
                          Returns original data's transpose if only one asset.
        """
        self._validate_input(returns)

        if len(returns.columns) < 2:
            logger.warning("PCA requires at least two assets to be meaningful. Returning original data transposed.")
            return returns.T  # Return original data, transposed, if only one asset

        # PCA expects samples as rows, features as columns. We want to cluster assets,
        # so assets are samples, and time points are features. Hence, returns.T.
        data_for_pca = returns.T

        # Handle n_components for PCA
        if isinstance(n_components, float) and (n_components <= 0 or n_components > 1):
            raise ValueError("n_components (if float) must be between 0 and 1 (exclusive).")

        # Ensure n_components is not greater than the number of features (time points)
        if isinstance(n_components, int) and n_components > data_for_pca.shape[1]:
            logger.warning(
                f"n_components ({n_components}) is greater than the number of time points ({data_for_pca.shape[1]}). Setting n_components to number of time points.")
            n_components = data_for_pca.shape[1]

        scaled_data = StandardScaler().fit_transform(data_for_pca)
        pca = PCA(n_components=n_components, random_state=self.random_state)
        principal_components = pca.fit_transform(scaled_data)

        logger.info(
            f"Explained variance by {pca.n_components_} components: {np.sum(pca.explained_variance_ratio_):.2%}")

        return pd.DataFrame(principal_components, index=returns.columns,
                            columns=[f'PC{i + 1}' for i in range(pca.n_components_)])

    def calculate_dtw_distance_matrix(self, returns: pd.DataFrame) -> pd.DataFrame:
        """
        Calculates the Dynamic Time Warping distance matrix with parallelization.

        Args:
            returns (pd.DataFrame): DataFrame of asset returns. Rows are time, columns are assets.

        Returns:
            pd.DataFrame: A square DataFrame representing the DTW distances between assets.
        """
        if dtw is None:
            raise ImportError("dtaidistance library is required for DTW calculations.")
        self._validate_input(returns)

        symbols = returns.columns
        n_assets = len(symbols)

        if n_assets < 2:
            logger.warning("DTW distance matrix requires at least two assets. Returning an empty matrix.")
            return pd.DataFrame(index=symbols, columns=symbols)

        distance_matrix = np.zeros((n_assets, n_assets))

        # Convert to numpy arrays of double for dtaidistance.dtw.distance_fast
        series_list = [returns[symbol].values.astype(np.double) for symbol in symbols]

        # Using ProcessPoolExecutor for parallel DTW calculations
        with ProcessPoolExecutor(max_workers=self.n_jobs if self.n_jobs != -1 else None) as executor:
            pairs = [(i, j) for i in range(n_assets) for j in range(i + 1, n_assets)]
            futures = {executor.submit(dtw.distance_fast, series_list[i], series_list[j]): (i, j) for i, j in pairs}

            for future in as_completed(futures):
                i, j = futures[future]
                try:
                    dist = future.result()
                    distance_matrix[i, j] = distance_matrix[j, i] = dist
                except Exception as exc:
                    logger.error(f'DTW calculation generated an exception for pair ({symbols[i]}, {symbols[j]}): {exc}')
                    # Set distance to NaN or a very large number to indicate failure
                    distance_matrix[i, j] = distance_matrix[j, i] = np.inf

        return pd.DataFrame(distance_matrix, index=symbols, columns=symbols)

    def kmeans_clustering(self,
                          returns: pd.DataFrame,
                          n_clusters: int = 5,
                          use_pca: bool = True,
                          n_components: int = 10,
                          distance_matrix: Optional[pd.DataFrame] = None) -> Tuple[List[List[str]], np.ndarray]:
        """
        Applies K-Means clustering.

        Args:
            returns (pd.DataFrame): DataFrame of asset returns. Rows are time, columns are assets.
            n_clusters (int): The number of clusters to form.
            use_pca (bool): Whether to apply PCA before clustering. Ignored if `distance_matrix` is provided.
            n_components (int): Number of principal components if `use_pca` is True.
            distance_matrix (Optional[pd.DataFrame]): Precomputed distance matrix. If provided,
                                                      `use_pca` is ignored and clustering is
                                                      performed on this matrix using `metric='precomputed'`.
                                                      Index and columns must be asset symbols.

        Returns:
            Tuple[List[List[str]], np.ndarray]: A tuple containing:
                - List of lists, where each inner list contains asset symbols in a cluster.
                - Numpy array of cluster labels for each asset.
        """
        self._validate_input(returns)

        data_to_cluster = None
        metric_for_kmeans = 'euclidean'

        if distance_matrix is not None:
            # Ensure distance matrix matches assets in returns
            if not all(asset in distance_matrix.index for asset in returns.columns):
                raise ValueError("Distance matrix index/columns must match asset symbols in returns.")
            data_to_cluster = distance_matrix.loc[returns.columns, returns.columns].values  # Ensure order
            metric_for_kmeans = 'precomputed'
            logger.info("K-Means using precomputed distance matrix.")
        else:
            data_to_cluster = returns.T  # Default to Euclidean on transposed returns
            if use_pca:
                data_to_cluster = self.reduce_dimensionality(returns, n_components=n_components)
                logger.info("K-Means using PCA-reduced data.")
            else:
                # Need to scale data if not using PCA and clustering on returns directly
                data_to_cluster = pd.DataFrame(StandardScaler().fit_transform(data_to_cluster),
                                               index=data_to_cluster.index,
                                               columns=data_to_cluster.columns)
                logger.info("K-Means using scaled original returns (transposed).")

        kmeans = KMeans(n_clusters=n_clusters, random_state=self.random_state, n_init=10,
                        init='k-means++' if metric_for_kmeans != 'precomputed' else 'random',
                        # 'k-means++' incompatible with precomputed
                        max_iter=300, algorithm='lloyd'  # explicit default to avoid future warnings
                        )

        # When metric is 'precomputed', fit_predict expects a distance matrix
        if metric_for_kmeans == 'precomputed':
            kmeans.fit(data_to_cluster)
            labels = kmeans.labels_  # labels_ after fitting on precomputed
        else:
            labels = kmeans.fit_predict(data_to_cluster)  # fit_predict for other metrics

        clusters = [data_to_cluster.index[labels == i].tolist() for i in range(n_clusters)]
        return clusters, labels

    def dbscan_clustering(self, returns: pd.DataFrame, eps: float = 0.5, min_samples: int = 2,
                          distance_matrix: Optional[pd.DataFrame] = None) -> Tuple[List[List[str]], np.ndarray]:
        """
        Applies DBSCAN clustering.

        Args:
            returns (pd.DataFrame): DataFrame of asset returns. Rows are time, columns are assets.
            eps (float): The maximum distance between two samples for one to be considered as in the neighborhood of the other.
            min_samples (int): The number of samples (or total weight) in a neighborhood for a point to be considered as a core point.
            distance_matrix (Optional[pd.DataFrame]): Precomputed distance matrix. If provided,
                                                      clustering is performed on this matrix using `metric='precomputed'`.
                                                      Index and columns must be asset symbols. If None,
                                                      1 - correlation_matrix is used as distance.

        Returns:
            Tuple[List[List[str]], np.ndarray]: A tuple containing:
                - List of lists, where each inner list contains asset symbols in a cluster.
                - Numpy array of cluster labels for each asset. Noise points are labeled -1.
        """
        self._validate_input(returns)

        data_for_dbscan = None
        metric_for_dbscan = 'euclidean'  # DBSCAN's default for non-precomputed

        if distance_matrix is not None:
            if not all(asset in distance_matrix.index for asset in returns.columns):
                raise ValueError("Distance matrix index/columns must match asset symbols in returns.")
            data_for_dbscan = distance_matrix.loc[returns.columns, returns.columns].values
            metric_for_dbscan = 'precomputed'
            logger.info("DBSCAN using precomputed distance matrix.")
        else:
            # Default to 1 - correlation as distance if no precomputed matrix
            corr_dist = 1 - returns.corr()
            data_for_dbscan = corr_dist.values  # DBSCAN uses this as the input distance matrix if metric='precomputed'
            metric_for_dbscan = 'precomputed'  # When using corr_dist, it's a precomputed matrix
            logger.info("DBSCAN using 1 - correlation as precomputed distance.")

        dbscan = DBSCAN(eps=eps, min_samples=min_samples, metric=metric_for_dbscan, n_jobs=self.n_jobs)
        labels = dbscan.fit_predict(data_for_dbscan)

        clusters = [returns.columns[labels == i].tolist() for i in set(labels) if i != -1]
        return clusters, labels

    def agglomerative_clustering(self,
                                 returns: pd.DataFrame,
                                 n_clusters: int = 5,
                                 linkage: str = 'ward',
                                 distance_matrix: Optional[pd.DataFrame] = None) -> Tuple[List[List[str]], np.ndarray]:
        """
        Applies Agglomerative (Hierarchical) clustering.

        Args:
            returns (pd.DataFrame): DataFrame of asset returns. Rows are time, columns are assets.
            n_clusters (int): The number of clusters to form.
            linkage (str): Which linkage criterion to use ('ward', 'complete', 'average', 'single').
            distance_matrix (Optional[pd.DataFrame]): Precomputed distance matrix. If provided,
                                                      `linkage` must be 'complete', 'average', or 'single'
                                                      (ward only works on Euclidean distances).
                                                      Index and columns must be asset symbols.

        Returns:
            Tuple[List[List[str]], np.ndarray]: A tuple containing:
                - List of lists, where each inner list contains asset symbols in a cluster.
                - Numpy array of cluster labels for each asset.
        """
        self._validate_input(returns)

        data_for_agg = None
        metric_for_agg = 'euclidean'  # Agglomerative default metric

        if distance_matrix is not None:
            if not all(asset in distance_matrix.index for asset in returns.columns):
                raise ValueError("Distance matrix index/columns must match asset symbols in returns.")
            if linkage == 'ward':
                raise ValueError(
                    "Ward linkage only works with Euclidean distance on raw features. Cannot use with precomputed distance_matrix.")
            data_for_agg = distance_matrix.loc[returns.columns, returns.columns].values
            metric_for_agg = 'precomputed'
            logger.info("Agglomerative Clustering using precomputed distance matrix.")
        else:
            # Agglomerative expects samples as rows, features as columns.
            # We want to cluster assets, so assets are samples, time points are features.
            data_for_agg = returns.T
            # Scale data if not using precomputed distances
            data_for_agg = pd.DataFrame(StandardScaler().fit_transform(data_for_agg),
                                        index=data_for_agg.index,
                                        columns=data_for_agg.columns)
            logger.info("Agglomerative Clustering using scaled original returns (transposed).")

        agg = AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage, metric=metric_for_agg)
        labels = agg.fit_predict(data_for_agg)

        clusters = [data_for_agg.index[labels == i].tolist() for i in range(n_clusters)]
        return clusters, labels

    def mean_shift_clustering(self,
                              returns: pd.DataFrame,
                              bandwidth: Optional[float] = None,
                              n_samples: int = 500) -> Tuple[List[List[str]], np.ndarray]:
        """
        Applies Mean Shift clustering. Does not require specifying number of clusters.

        Args:
            returns (pd.DataFrame): DataFrame of asset returns. Rows are time, columns are assets.
            bandwidth (Optional[float]): Bandwidth parameter. If None, it is estimated using `estimate_bandwidth`.
            n_samples (int): The number of samples to use for bandwidth estimation.

        Returns:
            Tuple[List[List[str]], np.ndarray]: A tuple containing:
                - List of lists, where each inner list contains asset symbols in a cluster.
                - Numpy array of cluster labels for each asset.
        """
        self._validate_input(returns)

        if returns.shape[1] < 2:
            logger.warning("Mean Shift clustering requires at least two assets. Returning empty clusters.")
            return [], np.array([])

        # Mean Shift also expects samples as rows, features as columns (clustering assets)
        data_for_ms = returns.T
        # Scale data for better performance with Euclidean distance
        scaled_data = StandardScaler().fit_transform(data_for_ms)
        scaled_data_df = pd.DataFrame(scaled_data, index=data_for_ms.index, columns=data_for_ms.columns)

        if bandwidth is None:
            logger.info(f"Estimating Mean Shift bandwidth using {n_samples} samples.")
            # Adjust n_samples if data is smaller than n_samples
            actual_n_samples = min(n_samples, scaled_data_df.shape[0])
            bandwidth = estimate_bandwidth(scaled_data_df, quantile=0.2, n_samples=actual_n_samples,
                                           random_state=self.random_state, n_jobs=self.n_jobs)
            logger.info(f"Estimated bandwidth: {bandwidth:.4f}")
            if bandwidth == 0:  # Handle cases where bandwidth might be estimated as 0
                logger.warning(
                    "Estimated bandwidth is 0. Mean Shift might not form meaningful clusters. Consider setting it manually or increasing data variability.")
                # You might choose to raise an error or set a default non-zero small value
                bandwidth = 0.1  # A sensible small default

        ms = MeanShift(bandwidth=bandwidth, n_jobs=self.n_jobs)
        labels = ms.fit_predict(scaled_data_df)

        clusters = [scaled_data_df.index[labels == i].tolist() for i in np.unique(labels)]
        logger.info(f"Mean Shift found {len(clusters)} clusters.")
        return clusters, labels

    def graph_based_clustering(self, correlation_matrix: pd.DataFrame, threshold: float = 0.8) -> List[List[str]]:
        """
        Identifies clusters from a correlation matrix using a graph-based approach (connected components).

        Args:
            correlation_matrix (pd.DataFrame): Square DataFrame of asset correlations.
            threshold (float): Minimum absolute correlation to consider an edge between assets.

        Returns:
            List[List[str]]: A list of lists, where each inner list contains asset symbols in a cluster.
        """
        self._validate_input(correlation_matrix)
        if not (correlation_matrix.index.equals(correlation_matrix.columns) and
                (correlation_matrix.shape[0] == correlation_matrix.shape[1])):
            raise ValueError("Correlation matrix must be square with matching index and columns.")

        adj_matrix = (correlation_matrix.abs() > threshold).astype(int)
        graph = nx.from_pandas_adjacency(adj_matrix)
        graph.remove_edges_from(
            nx.selfloop_edges(graph))  # Remove self-loops as assets are always correlated with themselves

        clusters = [list(component) for component in nx.connected_components(graph)]
        logger.info(f"Graph-based clustering found {len(clusters)} clusters with threshold {threshold}.")
        return clusters

    def calculate_silhouette_score(self,
                                   returns: pd.DataFrame,
                                   cluster_labels: np.ndarray,
                                   metric: Union[str, callable] = 'euclidean',
                                   precomputed_distance_matrix: Optional[pd.DataFrame] = None) -> float:
        """
        Calculates the silhouette score to evaluate the quality of the clustering.

        Args:
            returns (pd.DataFrame): Original DataFrame of asset returns (rows are time, columns are assets).
                                    Used to derive the data matrix for silhouette calculation if `metric` is not 'precomputed'.
            cluster_labels (np.ndarray): Numpy array of cluster labels for each asset.
            metric (Union[str, callable]): The metric to use when calculating the distance between points.
                                           Can be a string (e.g., 'euclidean', 'correlation') or a callable.
                                           If 'precomputed', `precomputed_distance_matrix` must be provided.
            precomputed_distance_matrix (Optional[pd.DataFrame]): If `metric` is 'precomputed', this
                                                                   DataFrame must contain the pairwise
                                                                   distances between assets used for clustering.

        Returns:
            float: The silhouette score (between -1 and 1). -1.0 if calculation is not possible.
        """
        if len(np.unique(cluster_labels)) < 2 or len(np.unique(cluster_labels)) == len(cluster_labels):
            logger.warning(
                "Cannot calculate silhouette score with less than 2 or more than (n_samples - 1) unique clusters.")
            return -1.0

        if len(cluster_labels) != returns.shape[1]:
            raise ValueError(
                f"Number of cluster labels ({len(cluster_labels)}) must match number of assets ({returns.shape[1]}).")

        data_for_silhouette = None
        if metric == 'precomputed':
            if precomputed_distance_matrix is None:
                raise ValueError("For 'precomputed' metric, 'precomputed_distance_matrix' must be provided.")
            if not all(asset in precomputed_distance_matrix.index for asset in returns.columns):
                raise ValueError("Precomputed distance matrix index/columns must match asset symbols in returns.")
            data_for_silhouette = precomputed_distance_matrix.loc[returns.columns, returns.columns].values
            logger.info("Calculating silhouette score with precomputed distance matrix.")
        elif metric == 'correlation':
            # Convert correlation to distance (1 - correlation)
            data_for_silhouette = 1 - returns.corr()
            # If using `correlation` as metric, silhouette expects a (n_samples, n_samples) distance matrix
            # so we pass the 1-corr matrix, and set metric='precomputed' for silhouette_score.
            metric = 'precomputed'
            logger.info("Calculating silhouette score with 1-correlation as precomputed distance.")
        else:
            # For other metrics like 'euclidean', silhouette_score expects feature array
            data_for_silhouette = StandardScaler().fit_transform(returns.T)  # Cluster assets, so transpose
            logger.info(f"Calculating silhouette score with '{metric}' metric on scaled returns (transposed).")

        try:
            score = silhouette_score(data_for_silhouette, cluster_labels, metric=metric, n_jobs=self.n_jobs)
            logger.info(f"Silhouette Score ({metric}): {score:.4f}")
            return score
        except Exception as e:
            logger.error(f"Error calculating silhouette score: {e}")
            return -1.0

    def plot_cluster_heatmap(self, correlation_matrix: pd.DataFrame, clusters: List[List[str]],
                             title: str = "Asset Correlation Heatmap by Cluster",
                             colorscale: str = 'RdBu') -> go.Figure:
        """
        Generates a heatmap of the correlation matrix with assets reordered according to their cluster.

        Args:
            correlation_matrix (pd.DataFrame): Square DataFrame of asset correlations.
            clusters (List[List[str]]): List of lists, where each inner list contains asset symbols in a cluster.
            title (str): Title of the heatmap.
            colorscale (str): Plotly colorscale (e.g., 'RdBu', 'Viridis', 'Plasma').

        Returns:
            go.Figure: A Plotly Figure object.
        """
        self._validate_input(correlation_matrix)
        if not (correlation_matrix.index.equals(correlation_matrix.columns) and
                (correlation_matrix.shape[0] == correlation_matrix.shape[1])):
            raise ValueError("Correlation matrix must be square with matching index and columns.")

        ordered_assets = []
        for cluster in clusters:
            # Ensure assets in cluster exist in correlation matrix
            valid_assets_in_cluster = [asset for asset in cluster if asset in correlation_matrix.columns]
            ordered_assets.extend(valid_assets_in_cluster)

        # Add any assets that were not assigned to a cluster (e.g., DBSCAN noise, or not in provided clusters)
        unclustered_assets = [asset for asset in correlation_matrix.columns if asset not in ordered_assets]
        final_order = ordered_assets + unclustered_assets

        # Ensure final_order has unique elements and matches correlation_matrix.columns
        final_order_unique = list(dict.fromkeys(final_order))
        if len(final_order_unique) != len(correlation_matrix.columns):
            logger.warning(
                "Discrepancy in assets between clusters/unclustered and correlation matrix columns. Plotting may be incomplete.")
            # Adjust final_order to only include assets present in the correlation_matrix
            final_order_unique = [asset for asset in final_order_unique if asset in correlation_matrix.columns]

        ordered_matrix = correlation_matrix.loc[final_order_unique, final_order_unique]

        fig = go.Figure(data=go.Heatmap(
            z=ordered_matrix.values,
            x=ordered_matrix.columns,
            y=ordered_matrix.columns,
            colorscale=colorscale,
            zmid=0  # Center the colorscale around 0 for correlation
        ))

        shapes = []
        current_idx = 0
        for cluster in clusters:
            cluster_members_in_order = [asset for asset in final_order_unique if asset in cluster]
            if not cluster_members_in_order: continue  # Skip if cluster has no members in final_order

            start_pos = current_idx
            end_pos = current_idx + len(cluster_members_in_order)

            shapes.append(go.layout.Shape(
                type="rect",
                # The -0.5 is because heatmap cells are centered on tick marks
                x0=start_pos - 0.5, y0=start_pos - 0.5,
                x1=end_pos - 0.5, y1=end_pos - 0.5,
                line=dict(color="black", width=2, dash="solid")
            ))
            current_idx = end_pos  # Move to the end of the current cluster for the next rectangle

        fig.update_layout(
            title=title,
            height=800,
            width=800,
            xaxis_showgrid=False,
            yaxis_showgrid=False,
            xaxis_zeroline=False,
            yaxis_zeroline=False,
            shapes=shapes,
            hovermode='closest'
        )
        return fig


# --- Example Usage ---
def main():
    # 1. Generate Sample Data (more realistic for clustering)
    np.random.seed(42)
    dates = pd.date_range(start='2023-01-01', periods=100)

    # Create distinct clusters
    num_assets = 50
    cluster_size = 10

    # Cluster 1: High correlation, medium volatility
    returns_c1 = np.random.normal(0.0005, 0.01, (len(dates), cluster_size))
    common_factor_c1 = np.random.normal(0, 0.005, len(dates)).reshape(-1, 1)
    assets_c1 = pd.DataFrame(returns_c1 + common_factor_c1, index=dates,
                             columns=[f'Asset_C1_{i}' for i in range(cluster_size)])

    # Cluster 2: Medium correlation, low volatility
    returns_c2 = np.random.normal(0.0001, 0.005, (len(dates), cluster_size))
    common_factor_c2 = np.random.normal(0, 0.002, len(dates)).reshape(-1, 1)
    assets_c2 = pd.DataFrame(returns_c2 + common_factor_c2, index=dates,
                             columns=[f'Asset_C2_{i}' for i in range(cluster_size)])

    # Cluster 3: Low correlation, high volatility
    returns_c3 = np.random.normal(0.001, 0.02, (len(dates), cluster_size))
    assets_c3 = pd.DataFrame(returns_c3, index=dates,
                             columns=[f'Asset_C3_{i}' for i in range(cluster_size)])

    # Cluster 4: Some noise assets
    returns_noise = np.random.normal(0, 0.015, (len(dates), num_assets - 3 * cluster_size))
    assets_noise = pd.DataFrame(returns_noise, index=dates,
                                columns=[f'Asset_Noise_{i}' for i in range(num_assets - 3 * cluster_size)])

    all_returns = pd.concat([assets_c1, assets_c2, assets_c3, assets_noise], axis=1)

    # Introduce some NaN values to test validation (should raise error)
    # all_returns.iloc[10, 5] = np.nan

    # 2. Initialize Analyzer
    analyzer = AssetClusteringAnalyzer(n_jobs=-1, random_state=42)

    # 3. Test PCA for dimensionality reduction
    print("\n--- Testing PCA ---")
    pca_returns = analyzer.reduce_dimensionality(all_returns, n_components=5)
    print(f"Original shape: {all_returns.shape}, PCA shape: {pca_returns.shape}")
    # Test PCA with variance threshold
    pca_returns_var = analyzer.reduce_dimensionality(all_returns, n_components=0.90)
    print(f"Original shape: {all_returns.shape}, PCA (0.90 var) shape: {pca_returns_var.shape}")
    # Test PCA with single asset (should warn and return transposed original)
    single_asset_returns = all_returns[['Asset_C1_0']]
    pca_single = analyzer.reduce_dimensionality(single_asset_returns, n_components=2)
    print(f"Single asset PCA shape: {pca_single.shape}")

    # 4. Test DTW Distance Matrix
    print("\n--- Testing DTW Distance Matrix ---")
    try:
        dtw_matrix = analyzer.calculate_dtw_distance_matrix(all_returns.iloc[:, :10])  # Use a subset for speed
        print(f"DTW Distance Matrix shape: {dtw_matrix.shape}")
        # print(dtw_matrix.head())
    except ImportError as e:
        print(e)
        dtw_matrix = None  # Set to None if dtw not available

    # 5. Test K-Means Clustering
    print("\n--- Testing K-Means Clustering ---")
    n_clusters_kmeans = 4  # Expecting 3 clear clusters + noise

    # K-Means with PCA
    kmeans_clusters_pca, kmeans_labels_pca = analyzer.kmeans_clustering(
        all_returns, n_clusters=n_clusters_kmeans, use_pca=True, n_components=5
    )
    print(f"K-Means (PCA) Clusters: {len(kmeans_clusters_pca)} clusters found.")
    for i, cluster in enumerate(kmeans_clusters_pca):
        print(f"  Cluster {i + 1} ({len(cluster)} assets): {cluster[:5]}...")

    # K-Means with DTW distance (if DTW matrix computed)
    if dtw_matrix is not None:
        kmeans_clusters_dtw, kmeans_labels_dtw = analyzer.kmeans_clustering(
            all_returns.iloc[:, :10], n_clusters=2, use_pca=False, distance_matrix=dtw_matrix
        )
        print(f"K-Means (DTW) Clusters: {len(kmeans_clusters_dtw)} clusters found.")
        for i, cluster in enumerate(kmeans_clusters_dtw):
            print(f"  Cluster {i + 1} ({len(cluster)} assets): {cluster[:5]}...")

    # 6. Test DBSCAN Clustering
    print("\n--- Testing DBSCAN Clustering ---")
    dbscan_clusters, dbscan_labels = analyzer.dbscan_clustering(
        all_returns, eps=0.5, min_samples=5  # Adjust eps/min_samples based on data characteristics
    )
    print(f"DBSCAN Clusters: {len(dbscan_clusters)} clusters found.")
    for i, cluster in enumerate(dbscan_clusters):
        print(f"  Cluster {i + 1} ({len(cluster)} assets): {cluster[:5]}...")
    # Test DBSCAN with DTW distance (if DTW matrix computed)
    if dtw_matrix is not None:
        dbscan_clusters_dtw, dbscan_labels_dtw = analyzer.dbscan_clustering(
            all_returns.iloc[:, :10], eps=5.0, min_samples=2, distance_matrix=dtw_matrix
        )
        print(f"DBSCAN (DTW) Clusters: {len(dbscan_clusters_dtw)} clusters found.")
        for i, cluster in enumerate(dbscan_clusters_dtw):
            print(f"  Cluster {i + 1} ({len(cluster)} assets): {cluster[:5]}...")

    # 7. Test Agglomerative Clustering
    print("\n--- Testing Agglomerative Clustering ---")
    agg_clusters, agg_labels = analyzer.agglomerative_clustering(
        all_returns, n_clusters=n_clusters_kmeans, linkage='ward'
    )
    print(f"Agglomerative Clusters: {len(agg_clusters)} clusters found.")
    for i, cluster in enumerate(agg_clusters):
        print(f"  Cluster {i + 1} ({len(cluster)} assets): {cluster[:5]}...")

    # Test Agglomerative with DTW distance (if DTW matrix computed) - must use non-ward linkage
    if dtw_matrix is not None:
        agg_clusters_dtw, agg_labels_dtw = analyzer.agglomerative_clustering(
            all_returns.iloc[:, :10], n_clusters=2, linkage='average', distance_matrix=dtw_matrix
        )
        print(f"Agglomerative (DTW, Average) Clusters: {len(agg_clusters_dtw)} clusters found.")
        for i, cluster in enumerate(agg_clusters_dtw):
            print(f"  Cluster {i + 1} ({len(cluster)} assets): {cluster[:5]}...")

    # 8. Test Mean Shift Clustering
    print("\n--- Testing Mean Shift Clustering ---")
    ms_clusters, ms_labels = analyzer.mean_shift_clustering(all_returns)
    print(f"Mean Shift Clusters: {len(ms_clusters)} clusters found.")
    for i, cluster in enumerate(ms_clusters):
        print(f"  Cluster {i + 1} ({len(cluster)} assets): {cluster[:5]}...")

    # 9. Test Graph-Based Clustering
    print("\n--- Testing Graph-Based Clustering ---")
    correlation_matrix = all_returns.corr()
    graph_clusters = analyzer.graph_based_clustering(correlation_matrix, threshold=0.7)
    print(f"Graph-Based Clusters: {len(graph_clusters)} clusters found.")
    for i, cluster in enumerate(graph_clusters):
        print(f"  Cluster {i + 1} ({len(cluster)} assets): {cluster[:5]}...")

    # 10. Test Silhouette Score
    print("\n--- Testing Silhouette Scores ---")
    if len(np.unique(kmeans_labels_pca)) > 1:  # Check if enough clusters for score
        silhouette_euclidean = analyzer.calculate_silhouette_score(all_returns, kmeans_labels_pca, metric='euclidean')
        print(f"K-Means (PCA) Silhouette Score (Euclidean): {silhouette_euclidean:.4f}")

    if len(np.unique(dbscan_labels)) > 1:  # Check if enough clusters for score
        # For DBSCAN, often default to correlation for evaluation if that's how it was clustered
        silhouette_corr = analyzer.calculate_silhouette_score(all_returns, dbscan_labels, metric='correlation')
        print(f"DBSCAN Silhouette Score (Correlation): {silhouette_corr:.4f}")

    if dtw_matrix is not None and len(np.unique(kmeans_labels_dtw)) > 1:
        silhouette_dtw = analyzer.calculate_silhouette_score(all_returns.iloc[:, :10], kmeans_labels_dtw,
                                                             metric='precomputed',
                                                             precomputed_distance_matrix=dtw_matrix)
        print(f"K-Means (DTW) Silhouette Score (Precomputed DTW): {silhouette_dtw:.4f}")

    # 11. Test Heatmap Plotting
    print("\n--- Generating Heatmap Plot ---")
    if len(kmeans_clusters_pca) > 0:  # Ensure clusters were found
        fig = analyzer.plot_cluster_heatmap(correlation_matrix, kmeans_clusters_pca,
                                            title="K-Means (PCA) Clusters Correlation Heatmap")
        # fig.show() # Uncomment to display the plot in a browser

    if len(dbscan_clusters) > 0:
        fig_dbscan = analyzer.plot_cluster_heatmap(correlation_matrix, dbscan_clusters,
                                                   title="DBSCAN Clusters Correlation Heatmap", colorscale='Plasma')
        # fig_dbscan.show() # Uncomment to display the plot

    print("\n--- Script finished ---")


if __name__ == "__main__":
    main()