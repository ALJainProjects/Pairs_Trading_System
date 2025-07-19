import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib_venn import venn3, venn2  # Import venn2 for 2-set diagrams
from typing import List, Dict, Tuple, Set, Optional, Any

# Import analysis modules
from src.analysis.correlation_analysis import CorrelationAnalyzer
from src.analysis.cointegration import find_cointegrated_pairs, rolling_cointegration_test, find_cointegrating_vectors
from src.analysis.clustering_analysis import AssetClusteringAnalyzer

# Import Covariance Estimation modules
from src.analysis.covariance_estimation import (
    StandardCovariance,
    EWMACovariance,
    LedoitWolfShrinkage,
    GraphicalLassoCovariance,
    ResidualCovarianceFromRegression,
    RobustCovariance,
    BaseCovariance  # For type hinting
)


# --- Helper Function for Session State Initialization ---
def _init_pair_analysis_session_state():
    """Initializes necessary session state variables for pair analysis."""
    if 'latest_analysis' not in st.session_state:
        st.session_state.latest_analysis = {
            'correlated_pairs': pd.DataFrame(),
            'cointegrated_eg_pairs': pd.DataFrame(),  # Engle-Granger results
            'cointegrated_johansen_pairs': pd.DataFrame(),  # Johansen results
            'clustering_results': {},  # Store algorithm, clusters, labels, metrics
            'dtw_matrix': None,  # Store DTW matrix if calculated
            'correlation_matrix': None,  # Store Pearson correlation matrix (calculated once for consistency)
            'estimated_covariance_matrix': None,  # New: Store the estimated covariance matrix
            'estimated_correlation_matrix': None  # New: Store the estimated correlation matrix from chosen estimator
        }
    if 'selected_pairs' not in st.session_state:
        st.session_state.selected_pairs = []

    # Ensure any temporary plots are cleared
    if 'analysis_fig_heatmap' not in st.session_state:
        st.session_state.analysis_fig_heatmap = None
    if 'analysis_fig_spread' not in st.session_state:  # For cointegration spread plots
        st.session_state.analysis_fig_spread = None


# --- Main Streamlit Renderer Function ---
def render_pair_analyzer_page():
    """Renders the UI for analyzing and selecting trading pairs."""
    st.title("🔬 Pair Analyzer")

    _init_pair_analysis_session_state()

    if st.session_state.pivot_prices.empty:
        st.warning("Please load data in the 'Data Loader' page first to perform pair analysis.")
        return

    prices = st.session_state.pivot_prices
    # Ensure returns are clean before passing to analyzers
    returns = prices.pct_change().dropna()

    if returns.empty:
        st.warning("Returns data is empty after processing. Cannot perform analysis. Check 'Data Loader' for issues.")
        return

    # Calculate Pearson correlation matrix once, as it's a common input for many analyses
    corr_analyzer = CorrelationAnalyzer(returns)
    st.session_state.latest_analysis['correlation_matrix'] = corr_analyzer.calculate_correlation('pearson')

    cluster_analyzer = AssetClusteringAnalyzer()

    # --- Tabs ---
    tab1, tab2, tab3, tab4 = st.tabs(
        ["📊 Correlation Analysis", "📈 Cointegration Analysis", "📦 Cluster Analysis", "🔍 Covariance Estimation"])

    # --- Tab 1: Correlation Analysis ---
    with tab1:
        st.subheader("Correlation-Based Pair Selection")

        corr_method = st.selectbox(
            "Select Correlation Method",
            ["pearson", "spearman", "partial"],
            key="corr_tab_method",
            help="Choose the statistical method for calculating correlation between asset returns."
        )

        corr_threshold = st.slider(
            "Correlation Threshold (absolute value)",
            0.5, 1.0, 0.8, 0.05,
            key="corr_tab_threshold",
            help="Pairs with absolute correlation above this value will be identified."
        )

        st.markdown("---")
        st.subheader("Correlation Significance & Multiple Testing")
        if corr_method != 'pearson':
            st.info(
                f"💡 P-value and multiple testing correction are typically applied to Pearson correlations. Results might not be meaningful for '{corr_method}'.")

        alpha_sig = st.slider("Significance Level (alpha)", 0.01, 0.10, 0.05, 0.01, key="alpha_sig")
        mtest_method = st.selectbox(
            "Multiple Testing Correction Method",
            ['fdr_bh', 'bonferroni', 'holm', 'simes-hotelling', 'hsidk'],
            key="mtest_method",
            help="Method to correct p-values for multiple comparisons (e.g., Bonferroni for strict control, FDR for false discovery rate)."
        )

        if st.button("Find Correlated Pairs & Assess Significance", key="run_corr_analysis"):
            with st.spinner(f"Calculating {corr_method} correlations and significance..."):
                try:
                    # Calculate correlation matrix
                    current_corr_matrix = corr_analyzer.calculate_correlation(corr_method)
                    st.session_state.latest_analysis[
                        'correlation_matrix'] = current_corr_matrix  # Update with chosen method's matrix

                    # Find highly correlated pairs
                    correlated_pairs_df = corr_analyzer.get_highly_correlated_pairs(corr_method, corr_threshold)
                    st.session_state.latest_analysis['correlated_pairs'] = correlated_pairs_df

                    # Calculate p-values (only for Pearson)
                    raw_p_values_df = pd.DataFrame()
                    corrected_p_values_df = pd.DataFrame()
                    if corr_method == 'pearson':
                        raw_p_values_df = corr_analyzer.calculate_correlation_significance(current_corr_matrix)
                        corrected_p_values_df = corr_analyzer.correct_pvalues_for_multiple_testing(raw_p_values_df,
                                                                                                   alpha=alpha_sig,
                                                                                                   method=mtest_method)
                        st.session_state.latest_analysis['raw_p_values'] = raw_p_values_df
                        st.session_state.latest_analysis['corrected_p_values'] = corrected_p_values_df
                        st.success("Correlation analysis complete.")
                    else:
                        st.info(
                            "P-value calculation in this tool is optimized for Pearson correlation. Skipping p-value calculation for other methods.")
                        st.success("Correlation analysis complete (without p-value calculation).")

                except ValueError as e:
                    st.error(f"Error in correlation analysis: {e}. Check data and parameters.")
                except Exception as e:
                    st.error(f"An unexpected error occurred during correlation analysis: {e}.")

        # Display Results
        if not st.session_state.latest_analysis['correlated_pairs'].empty:
            st.markdown("#### Highly Correlated Pairs Found:")
            st.dataframe(st.session_state.latest_analysis['correlated_pairs'])

            st.markdown("#### Correlation Heatmap:")
            fig_corr_heatmap = corr_analyzer.plot_correlation_matrix(corr_method,
                                                                     title=f"{corr_method.capitalize()} Correlation Heatmap")
            st.plotly_chart(fig_corr_heatmap, use_container_width=True)

            if corr_method == 'pearson' and 'corrected_p_values' in st.session_state.latest_analysis and not \
            st.session_state.latest_analysis['corrected_p_values'].empty:
                st.markdown(f"#### Corrected P-values ({mtest_method.upper()}) for Pearson Correlation:")
                st.dataframe(st.session_state.latest_analysis['corrected_p_values'])

                # Filter significant pairs based on corrected p-values
                sig_pairs = st.session_state.latest_analysis['corrected_p_values'].where(
                    np.triu(np.ones(st.session_state.latest_analysis['corrected_p_values'].shape), k=1).astype(bool))
                sig_pairs = sig_pairs[sig_pairs < alpha_sig].stack().reset_index()
                sig_pairs.columns = ['Asset1', 'Asset2', 'Corrected_P_Value']
                if not sig_pairs.empty:
                    st.info(f"💡 Found {len(sig_pairs)} statistically significant pairs (Corrected P < {alpha_sig}).")
                    st.dataframe(sig_pairs.sort_values('Corrected_P_Value'))

    # --- Tab 2: Cointegration Analysis ---
    with tab2:
        st.subheader("Engle-Granger Cointegration Test (for pairs)")

        col_coint1, col_coint2, col_coint3 = st.columns(3)
        eg_p_value_threshold = col_coint1.slider("P-Value Threshold (Engle-Granger)", 0.01, 0.1, 0.05, 0.01,
                                                 key="eg_p_thresh")
        eg_min_half_life = col_coint2.slider("Min Half-Life (days)", 5, 252, 10, key="eg_min_hl")
        eg_max_half_life = col_coint3.slider("Max Half-Life (days)", 60, 500, 120, key="eg_max_hl")

        col_coint4, col_coint5 = st.columns(2)
        eg_min_obs = col_coint4.slider("Min Observations for Test", 20, 252, 60, key="eg_min_obs")
        eg_adf_reg_type = col_coint5.selectbox("ADF Regression Type", ['c', 'ct', 'nc'], key="eg_adf_reg_type",
                                               help="Trend assumption for ADF test (constant, constant+trend, no trend/constant).")

        if st.button("Find Engle-Granger Cointegrated Pairs", key="run_eg_coint"):
            with st.spinner("Running Engle-Granger cointegration tests..."):
                try:
                    eg_coint_results = find_cointegrated_pairs(
                        prices,  # Pass raw prices for cointegration tests
                        p_threshold=eg_p_value_threshold,
                        min_half_life=eg_min_half_life,
                        max_half_life=eg_max_half_life,
                        integration_test_min_obs=eg_min_obs,
                        adf_regression_type=eg_adf_reg_type
                    )
                    st.session_state.latest_analysis['cointegrated_eg_pairs'] = pd.DataFrame(eg_coint_results)
                    st.success("Engle-Granger cointegration analysis complete.")
                except ValueError as e:
                    st.error(f"Error in Engle-Granger cointegration analysis: {e}. Check data and parameters.")
                except Exception as e:
                    st.error(f"An unexpected error occurred during Engle-Granger cointegration: {e}.")

        # Display EG results
        if not st.session_state.latest_analysis['cointegrated_eg_pairs'].empty:
            st.markdown("#### Engle-Granger Cointegrated Pairs Found:")
            st.dataframe(st.session_state.latest_analysis['cointegrated_eg_pairs'])

            # Optional: Plot spread and z-score for a selected EG pair
            eg_pairs_for_plot = [f"{p.asset1}-{p.asset2}" for idx, p in
                                 st.session_state.latest_analysis['cointegrated_eg_pairs'].iterrows()]
            if eg_pairs_for_plot:
                selected_pair_eg_plot = st.selectbox("Select an EG pair to visualize its Spread & Z-Score:",
                                                     eg_pairs_for_plot, key="select_eg_pair_plot")
                if selected_pair_eg_plot:
                    asset1_plot, asset2_plot = selected_pair_eg_plot.split('-')
                    # Retrieve the hedge ratio for the selected pair
                    pair_row = st.session_state.latest_analysis['cointegrated_eg_pairs'][
                        (st.session_state.latest_analysis['cointegrated_eg_pairs']['asset1'] == asset1_plot) &
                        (st.session_state.latest_analysis['cointegrated_eg_pairs']['asset2'] == asset2_plot)
                        ].iloc[0]

                    hedge_ratio_plot = pair_row['hedge_ratio']

                    # Calculate spread and z-score for the entire price history for plotting
                    from src.models.statistical import StatisticalModel  # Import StatisticalModel
                    stat_model_for_plot = StatisticalModel()

                    spread_series_plot = stat_model_for_plot.calculate_spread(
                        prices[asset1_plot], prices[asset2_plot], hedge_ratio_plot
                    )
                    z_score_series_plot = stat_model_for_plot.calculate_zscore(
                        spread_series_plot, window=st.session_state.get('lookback_window', 60)
                        # Use default strategy lookback or 60
                    )

                    from src.utils.visualizer import PlotlyVisualizer
                    visualizer = PlotlyVisualizer()
                    fig_spread_z = visualizer.plot_spread_analysis(
                        spread_series_plot, z_score_series_plot,
                        title=f"Spread & Z-Score for {asset1_plot}-{asset2_plot}",
                        entry_zscore=st.session_state.get('zscore_entry', 2.0),
                        exit_zscore=st.session_state.get('zscore_exit', 0.5)
                    )
                    st.plotly_chart(fig_spread_z, use_container_width=True)

        st.markdown("---")
        st.subheader("Johansen Cointegration Test (Multivariate)")
        st.info(
            "💡 Johansen test finds cointegrating relationships among 2 or more assets. It returns cointegrating vectors, not direct pairs.")

        col_johansen1, col_johansen2 = st.columns(2)
        johansen_det_order = col_johansen1.selectbox("Deterministic Trend Order", [-1, 0, 1], index=1,
                                                     key="johansen_det_order",
                                                     help="-1: no trend, 0: constant, 1: linear trend.")
        johansen_k_ar_diff = col_johansen2.slider("VAR Lagged Differences (k_ar_diff)", 1,
                                                  min(5, returns.shape[0] - prices.shape[1] - 2), 1,
                                                  key="johansen_k_ar_diff")

        johansen_normalize_to = st.selectbox(
            "Normalize Cointegrating Vector to (optional)",
            ['None'] + prices.columns.tolist(),
            key="johansen_normalize_to",
            help="Select an asset to normalize its coefficient in the cointegrating vector to 1. Leave 'None' for no normalization."
        )

        if st.button("Run Johansen Test", key="run_johansen_coint"):
            with st.spinner("Running Johansen cointegration test..."):
                try:
                    johansen_vectors = find_cointegrating_vectors(
                        prices,  # Johansen test also uses raw prices
                        det_order=johansen_det_order,
                        k_ar_diff=johansen_k_ar_diff,
                        normalize_to_asset=johansen_normalize_to if johansen_normalize_to != 'None' else None
                    )
                    st.session_state.latest_analysis['cointegrated_johansen_pairs'] = johansen_vectors
                    st.success("Johansen test complete.")
                except ValueError as e:
                    st.error(f"Error in Johansen test: {e}. Check data (min 2 assets) and parameters.")
                except Exception as e:
                    st.error(f"An unexpected error occurred during Johansen test: {e}.")

        if not st.session_state.latest_analysis['cointegrated_johansen_pairs'].empty:
            st.markdown("#### Johansen Cointegrating Vectors Found:")
            st.dataframe(st.session_state.latest_analysis['cointegrated_johansen_pairs'])
            st.info("💡 These vectors represent the linear combinations of assets that are stationary.")

        st.markdown("---")
        st.subheader("Rolling Cointegration Test (Stability)")
        st.info("💡 Assesses the stability of cointegration and hedge ratio over time for a selected pair.")

        all_symbols_for_rolling = prices.columns.tolist()
        if len(all_symbols_for_rolling) < 2:
            st.warning("Not enough symbols for rolling cointegration. Need at least 2.")
        else:
            col_roll1, col_roll2 = st.columns(2)
            roll_asset1 = col_roll1.selectbox("Asset 1 (Dependent)", all_symbols_for_rolling, key="roll_asset1_select")
            # Ensure asset2 is different from asset1
            other_symbols = [s for s in all_symbols_for_rolling if s != roll_asset1]
            if not other_symbols:
                st.warning("Only one asset selected. Cannot form a pair for rolling cointegration.")
                roll_asset2 = all_symbols_for_rolling[0]  # Fallback to avoid error
            else:
                # Default to the first different symbol
                default_idx_roll2 = 0
                if roll_asset1 == all_symbols_for_rolling[0] and len(all_symbols_for_rolling) > 1:
                    default_idx_roll2 = 1
                elif roll_asset1 == all_symbols_for_rolling[1] and len(all_symbols_for_rolling) > 2:
                    default_idx_roll2 = 2

                try:  # Ensure index doesn't go out of bounds
                    roll_asset2 = col_roll2.selectbox("Asset 2 (Independent)", other_symbols, index=other_symbols.index(
                        all_symbols_for_rolling[default_idx_roll2]) if all_symbols_for_rolling[
                                                                           default_idx_roll2] in other_symbols else 0,
                                                      key="roll_asset2_select")
                except IndexError:
                    roll_asset2 = col_roll2.selectbox("Asset 2 (Independent)", other_symbols, index=0,
                                                      key="roll_asset2_select")

            roll_window = st.slider("Rolling Window Size (days)", 30, 252, 60, key="roll_window")
            roll_min_obs = st.slider("Min Observations for Rolling Cointegration", 20, roll_window, 30,
                                     key="roll_min_obs")
            roll_adf_reg_type = st.selectbox("ADF Regression Type for Rolling Test", ['c', 'ct', 'nc'],
                                             key="roll_adf_reg_type_select")

            if st.button("Run Rolling Cointegration Test", key="run_rolling_coint"):
                with st.spinner(f"Running rolling cointegration test for {roll_asset1}-{roll_asset2}..."):
                    try:
                        rolling_results_df = rolling_cointegration_test(
                            prices[roll_asset1], prices[roll_asset2],  # Pass raw prices
                            window=roll_window,
                            min_observations_for_coint=roll_min_obs,
                            adf_regression_type=roll_adf_reg_type
                        )
                        st.session_state.latest_analysis['rolling_coint_results'] = rolling_results_df
                        st.success("Rolling cointegration analysis complete.")
                    except ValueError as e:
                        st.error(f"Error in rolling cointegration: {e}. Check data, selected assets, and window size.")
                    except Exception as e:
                        st.error(f"An unexpected error occurred during rolling cointegration: {e}.")

            if 'rolling_coint_results' in st.session_state.latest_analysis and not st.session_state.latest_analysis[
                'rolling_coint_results'].empty:
                st.markdown(f"#### Rolling Cointegration Results for {roll_asset1}-{roll_asset2}:")
                st.dataframe(st.session_state.latest_analysis['rolling_coint_results'])

                # Plot rolling p-value and hedge ratio
                fig_roll = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.1,
                                         subplot_titles=('Rolling P-Value', 'Rolling Hedge Ratio'))

                import plotly.graph_objects as go  # ensure go is imported if not already
                fig_roll.add_trace(go.Scatter(x=st.session_state.latest_analysis['rolling_coint_results'].index,
                                              y=st.session_state.latest_analysis['rolling_coint_results'][
                                                  'p_value_coint'],
                                              mode='lines', name='P-Value'), row=1, col=1)
                # Use the EG p-value threshold for reference
                fig_roll.add_hline(y=eg_p_value_threshold, line_dash="dash", line_color="red",
                                   annotation_text=f"EG P-value Threshold ({eg_p_value_threshold})",
                                   annotation_position="top right", row=1, col=1)

                fig_roll.add_trace(go.Scatter(x=st.session_state.latest_analysis['rolling_coint_results'].index,
                                              y=st.session_state.latest_analysis['rolling_coint_results'][
                                                  'hedge_ratio_ols'],
                                              mode='lines', name='Hedge Ratio'), row=2, col=1)
                fig_roll.update_layout(title_text=f"Rolling Cointegration for {roll_asset1}-{roll_asset2}", height=700)
                st.plotly_chart(fig_roll, use_container_width=True)

    # --- Tab 3: Cluster Analysis ---
    with tab3:
        st.subheader("Asset Clustering for Pairs Identification")

        cluster_algo_options = {
            "K-Means": "KMeans",
            "DBSCAN": "DBSCAN",
            "Agglomerative": "Agglomerative",
            "Mean Shift": "Mean Shift",
            "Graph-Based (Correlation)": "Graph-Based"
        }
        cluster_algo = st.selectbox(
            "Clustering Algorithm",
            list(cluster_algo_options.keys()),
            key="cluster_algo_select"
        )
        selected_algo_internal_name = cluster_algo_options[cluster_algo]

        # --- Common Clustering Parameters ---
        st.markdown("##### Pre-clustering Steps & Distance Metric")
        use_pca = st.checkbox(
            "Apply PCA for Dimensionality Reduction (before K-Means/Mean Shift)",
            value=True,
            key="cluster_use_pca",
            help="Reduces noise and speeds up clustering for high-dimensional data. Applied to asset returns (transposed)."
        )
        if use_pca:
            pca_n_components_value = st.slider("PCA Components / Variance Explained Value", 1,
                                               min(10, returns.shape[1] - 1), 5, key="pca_n_components_value",
                                               help="If 'PCA is Variance Ratio' is checked, this is the ratio (0.0-1.0). Otherwise, it's the number of components.")
            pca_n_components_is_ratio = st.checkbox("PCA N Components is Variance Ratio (0.0-1.0)", value=False,
                                                    key="pca_is_ratio")
            if pca_n_components_is_ratio:
                pca_n_components_value = st.slider("PCA Variance Explained Ratio", 0.05, 0.99, 0.9, 0.01,
                                                   key="pca_var_ratio_value")

            pca_n_components_final = pca_n_components_value if pca_n_components_is_ratio else int(
                pca_n_components_value)

        distance_metric_options = {
            "Euclidean (scaled returns)": "euclidean",
            "DTW (Dynamic Time Warping)": "dtw",
            "1 - Correlation (Pearson)": "1_minus_correlation"
        }
        distance_metric_for_cluster_display = st.selectbox(
            "Distance Metric / Input Data for Clustering",
            list(distance_metric_options.keys()),
            key="cluster_dist_metric_select",
            help="Choose how similarity between assets is measured. DTW can be slow for many assets."
        )
        distance_metric_internal_name = distance_metric_options[distance_metric_for_cluster_display]

        # Force metric for Graph-Based
        if selected_algo_internal_name == "Graph-Based":
            st.info("💡 Graph-based clustering inherently uses a correlation threshold for edge creation.")
            distance_metric_internal_name = "1_minus_correlation"  # Force for Graph-Based
            use_pca = False  # PCA not directly used for graph-based

        # --- Algorithm-Specific Parameters ---
        st.markdown("##### Algorithm-Specific Parameters")
        cluster_params: Dict[str, Any] = {}  # Initialize dict for params

        if selected_algo_internal_name == "KMeans":
            cluster_params['n_clusters'] = st.slider("Number of Clusters", 2, min(20, returns.shape[1]), 5,
                                                     key="kmeans_n_clusters")
        elif selected_algo_internal_name == "DBSCAN":
            cluster_params['eps'] = st.slider("EPS (Neighborhood Radius)", 0.1, 5.0, 0.5, 0.1, key="dbscan_eps")
            cluster_params['min_samples'] = st.slider("Min Samples (Core Point)", 2, min(20, returns.shape[1]), 5,
                                                      key="dbscan_min_samples")
        elif selected_algo_internal_name == "Agglomerative":
            cluster_params['n_clusters'] = st.slider("Number of Clusters", 2, min(20, returns.shape[1]), 5,
                                                     key="agg_n_clusters")
            cluster_params['linkage'] = st.selectbox("Linkage Method", ['ward', 'complete', 'average', 'single'],
                                                     key="agg_linkage")
            if distance_metric_internal_name != "euclidean" and cluster_params['linkage'] == 'ward':
                st.warning(
                    "Ward linkage only works with Euclidean distance on raw features. Select 'Euclidean' or change linkage.")
        elif selected_algo_internal_name == "Mean Shift":
            ms_bandwidth_option = st.slider("Bandwidth (Set 0 to Estimate Automatically)", 0.0, 5.0, 0.0, 0.1,
                                            key="ms_bandwidth_val")
            cluster_params['bandwidth'] = None if ms_bandwidth_option == 0.0 else ms_bandwidth_option
            if cluster_params['bandwidth'] is None:
                cluster_params['n_samples'] = st.slider("N Samples for Bandwidth Estimation", 50, 1000, 500,
                                                        key="ms_n_samples")
        elif selected_algo_internal_name == "Graph-Based":
            cluster_params['threshold'] = st.slider("Correlation Threshold (for Edges)", 0.5, 1.0, 0.7, 0.05,
                                                    key="graph_threshold")

        if st.button("Run Clustering Analysis", key="run_clustering_analysis"):
            with st.spinner("Performing clustering..."):
                try:
                    dtw_matrix_obj = None
                    precomputed_dist_for_algo: Optional[pd.DataFrame] = None

                    if distance_metric_internal_name == "dtw":
                        st.info("Calculating DTW distance matrix... This may take time for many assets.")
                        dtw_matrix_obj = cluster_analyzer.calculate_dtw_distance_matrix(returns)
                        st.session_state.latest_analysis['dtw_matrix'] = dtw_matrix_obj
                        if dtw_matrix_obj.empty:
                            st.error("DTW matrix calculation failed or resulted in empty matrix. Check logs.")
                            raise ValueError("DTW matrix empty.")
                        precomputed_dist_for_algo = dtw_matrix_obj
                    elif distance_metric_internal_name == "1_minus_correlation":
                        # Use the pre-calculated Pearson correlation matrix
                        if st.session_state.latest_analysis['correlation_matrix'] is None or \
                                st.session_state.latest_analysis['correlation_matrix'].empty:
                            st.error(
                                "Pearson Correlation Matrix not available for '1 - Correlation' distance. Run Correlation Analysis first.")
                            raise ValueError("Correlation matrix empty for clustering.")
                        precomputed_dist_for_algo = 1 - st.session_state.latest_analysis['correlation_matrix']

                    # --- Run chosen clustering algorithm ---
                    clusters = []
                    labels = np.array([])

                    if selected_algo_internal_name == "KMeans":
                        clusters, labels = cluster_analyzer.kmeans_clustering(
                            returns,
                            n_clusters=cluster_params['n_clusters'],
                            use_pca=use_pca,
                            n_components=pca_n_components_final,
                            distance_matrix=precomputed_dist_for_algo
                        )
                    elif selected_algo_internal_name == "DBSCAN":
                        clusters, labels = cluster_analyzer.dbscan_clustering(
                            returns,
                            eps=cluster_params['eps'],
                            min_samples=cluster_params['min_samples'],
                            distance_matrix=precomputed_dist_for_algo
                        )
                    elif selected_algo_internal_name == "Agglomerative":
                        clusters, labels = cluster_analyzer.agglomerative_clustering(
                            returns,
                            n_clusters=cluster_params['n_clusters'],
                            linkage=cluster_params['linkage'],
                            distance_matrix=precomputed_dist_for_algo
                        )
                    elif selected_algo_internal_name == "Mean Shift":
                        clusters, labels = cluster_analyzer.mean_shift_clustering(
                            returns,
                            bandwidth=cluster_params.get('bandwidth'),
                            n_samples=cluster_params.get('n_samples', 500)
                        )
                    elif selected_algo_internal_name == "Graph-Based":
                        if st.session_state.latest_analysis['correlation_matrix'] is None or \
                                st.session_state.latest_analysis['correlation_matrix'].empty:
                            st.error(
                                "Correlation Matrix not available for Graph-Based clustering. Run Correlation Analysis first.")
                            raise ValueError("Correlation matrix empty for Graph-Based.")
                        clusters = cluster_analyzer.graph_based_clustering(
                            st.session_state.latest_analysis['correlation_matrix'],
                            threshold=cluster_params['threshold']
                        )
                        # Assign labels for silhouette calculation if desired
                        labels = np.full(len(returns.columns), -1)  # Default to -1 (noise/unassigned)
                        asset_to_label = {}
                        for idx, c in enumerate(clusters):
                            for asset in c:
                                asset_to_label[asset] = idx
                        for i, asset in enumerate(returns.columns):
                            if asset in asset_to_label:
                                labels[i] = asset_to_label[asset]

                    st.session_state.latest_analysis['clustering_results'] = {
                        'algorithm': selected_algo_internal_name,
                        'params': cluster_params,
                        'clusters': clusters,
                        'labels': labels,
                        'distance_metric_used': distance_metric_internal_name
                    }

                    st.success("Clustering analysis complete.")

                    # Calculate Silhouette Score
                    # Silhouette needs >1 cluster and < N_samples. Handles noise (-1) by ignoring.
                    if len(np.unique(labels)) > 1 and len(np.unique(labels[labels != -1])) < len(labels[labels != -1]):
                        try:
                            silhouette_metric = distance_metric_internal_name
                            # Silhouette score metric needs to map internal name to sklearn's metric.
                            # 'dtw' is not a direct sklearn metric, needs precomputed.
                            if silhouette_metric == 'dtw':
                                silhouette_metric = 'precomputed'
                            elif silhouette_metric == '1_minus_correlation':
                                silhouette_metric = 'precomputed'
                            elif silhouette_metric == 'euclidean':
                                silhouette_metric = 'euclidean'  # Sklearn's default

                            silhouette = cluster_analyzer.calculate_silhouette_score(
                                returns, labels,
                                metric=silhouette_metric,  # Use internal sklearn name
                                precomputed_distance_matrix=precomputed_dist_for_algo if silhouette_metric == 'precomputed' else None
                            )
                            st.write(f"**Silhouette Score:** {silhouette:.4f}")
                        except Exception as e:
                            st.warning(
                                f"Could not calculate Silhouette Score: {e}. Check cluster count or data variability.")

                    st.markdown("#### Identified Clusters:")
                    for i, cluster in enumerate(clusters):
                        st.write(f"**Cluster {i + 1}** ({len(cluster)} assets): `{', '.join(cluster)}`")

                    # Plot Clustered Correlation Heatmap
                    if st.session_state.latest_analysis['correlation_matrix'] is not None and not \
                    st.session_state.latest_analysis['correlation_matrix'].empty:
                        st.markdown("#### Clustered Correlation Heatmap:")
                        # Pass returns.corr() for heatmap as it expects original Pearson correlation
                        fig_cluster_heatmap = cluster_analyzer.plot_cluster_heatmap(
                            st.session_state.latest_analysis['correlation_matrix'],
                            clusters,
                            title=f"Clustered Correlation Matrix ({cluster_algo})"
                        )
                        st.plotly_chart(fig_cluster_heatmap, use_container_width=True)
                    else:
                        st.warning("Correlation matrix not available or empty for heatmap plotting.")

                except ValueError as e:
                    st.error(
                        f"Error in clustering analysis: {e}. Check data, selected assets, or algorithm parameters.")
                except Exception as e:
                    st.error(f"An unexpected error occurred during clustering analysis: {e}.")

    # --- Tab 4: Covariance Estimation ---
    with tab4:
        st.subheader("Covariance Matrix Estimation")
        st.info(
            "💡 Different methods estimate asset covariance matrices, impacting portfolio optimization and risk. These matrices are derived from asset returns.")

        covariance_estimator_options = {
            "Standard (Sample)": StandardCovariance,
            "EWMA": EWMACovariance,
            "Ledoit-Wolf Shrinkage": LedoitWolfShrinkage,
            "Graphical Lasso": GraphicalLassoCovariance,
            "Residual from Regression": ResidualCovarianceFromRegression,
            "Robust (MinCovDet)": RobustCovariance
        }
        selected_estimator_name = st.selectbox(
            "Select Covariance Estimator",
            list(covariance_estimator_options.keys()),
            key="cov_estimator_select"
        )
        selected_estimator_class = covariance_estimator_options[selected_estimator_name]

        # --- Estimator-Specific Parameters ---
        estimator_params: Dict[str, Any] = {}
        st.markdown("##### Estimator Parameters")

        if selected_estimator_name == "EWMA":
            col_ewma_1, col_ewma_2 = st.columns(2)
            ewma_span = col_ewma_1.number_input("Span", min_value=2, value=60, key="ewma_span")
            ewma_alpha = col_ewma_2.number_input("Alpha (0.0-1.0, overrides Span if >0)", min_value=0.0, max_value=1.0,
                                                 value=0.0, step=0.01, format="%.2f", key="ewma_alpha")
            if ewma_alpha > 0:
                estimator_params['alpha'] = ewma_alpha
            else:
                estimator_params['span'] = ewma_span
        elif selected_estimator_name == "Graphical Lasso":
            estimator_params['alpha'] = st.slider("Alpha (Regularization)", 0.001, 0.5, 0.01, 0.001, format="%.3f",
                                                  key="gl_alpha")
            estimator_params['max_iter'] = st.slider("Max Iterations", 50, 500, 100, key="gl_max_iter")
        elif selected_estimator_name == "Robust (MinCovDet)":
            estimator_params['contamination'] = st.slider("Contamination (Proportion of Outliers)", 0.0, 0.49, 0.05,
                                                          0.01, format="%.2f", key="mcd_contamination")

        # No specific params needed for Standard, Ledoit-Wolf, Residual from Regression

        if st.button("Estimate Covariance Matrix", key="run_covariance_estimation"):
            with st.spinner(f"Estimating covariance using {selected_estimator_name} method..."):
                try:
                    estimator: BaseCovariance = selected_estimator_class(**estimator_params)
                    estimator.fit(returns)  # Fit on asset returns

                    cov_matrix = estimator.get_covariance()
                    corr_matrix = estimator.get_correlation()

                    st.session_state.latest_analysis['estimated_covariance_matrix'] = cov_matrix
                    st.session_state.latest_analysis['estimated_correlation_matrix'] = corr_matrix

                    st.success(f"Covariance estimation complete using {selected_estimator_name}.")

                    st.markdown("#### Estimated Covariance Matrix (Head):")
                    st.dataframe(cov_matrix.head())
                    st.markdown("#### Estimated Correlation Matrix (Head):")
                    st.dataframe(corr_matrix.head())

                    st.markdown("#### Visualizations:")
                    fig_cov = estimator.plot_matrix(matrix_type='covariance',
                                                    title=f'{selected_estimator_name} Covariance Matrix')
                    st.plotly_chart(fig_cov, use_container_width=True)

                    fig_corr_est = estimator.plot_matrix(matrix_type='correlation',
                                                         title=f'{selected_estimator_name} Correlation Matrix')
                    st.plotly_chart(fig_corr_est, use_container_width=True)

                except ValueError as e:
                    st.error(f"Error in covariance estimation: {e}. Check data or estimator parameters.")
                except Exception as e:
                    st.error(f"An unexpected error occurred during covariance estimation: {e}.")

    st.markdown("---")
    st.subheader("Final Pair Selection for Backtesting")

    # --- Consolidate Pairs from all analysis types ---
    all_pairs_sets = {}  # Dict to hold sets for Venn Diagram

    # Correlation Pairs
    corr_pairs_set = set()
    if not st.session_state.latest_analysis['correlated_pairs'].empty:
        df = st.session_state.latest_analysis['correlated_pairs']
        corr_pairs_set = {tuple(sorted((p.Asset1, p.Asset2))) for idx, p in df.iterrows()}
    all_pairs_sets['Correlated Pairs'] = corr_pairs_set

    # Engle-Granger Cointegrated Pairs
    coint_eg_pairs_set = set()
    if not st.session_state.latest_analysis['cointegrated_eg_pairs'].empty:
        df = st.session_state.latest_analysis['cointegrated_eg_pairs']
        coint_eg_pairs_set = {tuple(sorted((p.asset1, p.asset2))) for idx, p in df.iterrows()}
    all_pairs_sets['EG Cointegrated'] = coint_eg_pairs_set

    # Cluster-based pairs: Form pairs from assets within the same cluster that are also highly correlated.
    cluster_pairs_set = set()
    if 'clustering_results' in st.session_state.latest_analysis and st.session_state.latest_analysis[
        'clustering_results']:
        clusters_list = st.session_state.latest_analysis['clustering_results'].get('clusters', [])
        # Use the Pearson correlation matrix for intra-cluster pair filtering
        current_pearson_corr = st.session_state.latest_analysis['correlation_matrix']

        if not current_pearson_corr.empty:
            for cluster_members in clusters_list:
                if len(cluster_members) > 1:
                    # Filter clusters to only include members present in the correlation matrix
                    valid_cluster_members = [m for m in cluster_members if m in current_pearson_corr.columns]
                    if len(valid_cluster_members) < 2: continue  # Need at least 2 for pairs

                    cluster_corr_subset = current_pearson_corr.loc[valid_cluster_members, valid_cluster_members]

                    # Use a moderate threshold for intra-cluster correlation for pair formation
                    intra_cluster_threshold = 0.7  # This could be configurable in the UI if needed
                    # Use a temporary CorrelationAnalyzer just for this subset of returns
                    temp_corr_analyzer = CorrelationAnalyzer(
                        returns[valid_cluster_members])  # Pass subset of original returns

                    intra_cluster_highly_correlated_pairs = temp_corr_analyzer.get_highly_correlated_pairs(
                        'pearson', intra_cluster_threshold
                    )

                    for _, row in intra_cluster_highly_correlated_pairs.iterrows():
                        # Ensure both assets are indeed from the current cluster (already filtered by subset of corr_matrix)
                        cluster_pairs_set.add(tuple(sorted((row['Asset1'], row['Asset2']))))
        all_pairs_sets['Clustered Pairs'] = cluster_pairs_set

    # --- Venn Diagram Visualization ---
    st.markdown("#### Overlap of Pair Identification Methods")
    sets_for_venn = []
    labels_for_venn = []

    # Dynamically add up to 3 sets based on what's available and has members
    if all_pairs_sets['Correlated Pairs']:
        sets_for_venn.append(all_pairs_sets['Correlated Pairs'])
        labels_for_venn.append('Correlated')
    if all_pairs_sets['EG Cointegrated']:
        sets_for_venn.append(all_pairs_sets['EG Cointegrated'])
        labels_for_venn.append('EG Cointegrated')
    if all_pairs_sets['Clustered Pairs']:  # Add clustered pairs as a 3rd set if available
        sets_for_venn.append(all_pairs_sets['Clustered Pairs'])
        labels_for_venn.append('Clustered')

    if len(sets_for_venn) >= 2:
        fig_venn, ax_venn = plt.subplots(figsize=(8, 8))
        if len(sets_for_venn) == 2:
            venn2(sets_for_venn, set_labels=labels_for_venn, ax=ax_venn)
        elif len(sets_for_venn) == 3:
            venn3(sets_for_venn, set_labels=labels_for_venn, ax=ax_venn)
        ax_venn.set_title("Overlap of Pair Selection Methods")
        st.pyplot(fig_venn)
    else:
        st.warning(
            "Need at least two types of pair analysis results (Correlation, EG Cointegration, or Clustering) with found pairs to show Venn diagram.")

    # --- Final Multi-select for User Selection ---
    # Combine all unique pairs found across all methods for the multi-select dropdown
    all_unique_pairs_found = set()
    for s in all_pairs_sets.values():
        all_unique_pairs_found.update(s)

    # Sort for consistent display
    all_unique_pairs_sorted = sorted(list(all_unique_pairs_found))

    # Set default selection to the intersection of Correlated and EG Cointegrated if both exist
    default_selected_pairs_for_multiselect = st.session_state.selected_pairs
    # If no pairs are currently selected AND both Correlation & EG Cointegration found pairs,
    # default to their intersection.
    if not default_selected_pairs_for_multiselect and 'Correlated Pairs' in all_pairs_sets and 'EG Cointegrated' in all_pairs_sets and \
            all_pairs_sets['Correlated Pairs'] and all_pairs_sets['EG Cointegrated']:
        default_selected_pairs_for_multiselect = list(
            all_pairs_sets['Correlated Pairs'].intersection(all_pairs_sets['EG Cointegrated']))
        st.info(
            f"💡 Defaulting selection to {len(default_selected_pairs_for_multiselect)} pairs found by both Correlation and Engle-Granger Cointegration.")
    elif not default_selected_pairs_for_multiselect and len(all_unique_pairs_sorted) > 0:
        # If no default from intersection, just default to the first few found pairs, or none.
        # For simplicity, if intersection is empty or no selection, keep it empty.
        pass  # No change to default_selected_pairs_for_multiselect

    # Use a custom format function for display in the multiselect
    st.session_state.selected_pairs = st.multiselect(
        "**Select pairs for backtesting and live trading:**",
        options=all_unique_pairs_sorted,
        default=default_selected_pairs_for_multiselect,
        format_func=lambda x: f"{x[0]}-{x[1]}",
        help="Choose the specific asset pairs you want to use for subsequent strategy backtesting or live trading."
    )

    if st.session_state.selected_pairs:
        st.success(f"You have selected {len(st.session_state.selected_pairs)} pairs.")
        st.write("Selected Pairs:")
        for pair in st.session_state.selected_pairs:
            st.write(f"- {pair[0]}-{pair[1]}")
    else:
        st.info("No pairs currently selected. Please choose from the list above.")