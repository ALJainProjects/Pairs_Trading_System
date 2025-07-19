import streamlit as st
import pandas as pd
from src.analysis.correlation_analysis import CorrelationAnalyzer
from src.analysis.cointegration import find_cointegrated_pairs
from src.analysis.clustering_analysis import AssetClusteringAnalyzer
from matplotlib_venn import venn3
import matplotlib.pyplot as plt


def render_pair_analyzer_page():
    """Renders the UI for analyzing and selecting trading pairs."""
    st.title("🔬 Pair Analyzer")

    if st.session_state.pivot_prices.empty:
        st.warning("Please load data in the 'Data Loader' step first.")
        return

    prices = st.session_state.pivot_prices
    returns = prices.pct_change().dropna()

    corr_analyzer = CorrelationAnalyzer(returns)
    cluster_analyzer = AssetClusteringAnalyzer()

    tab1, tab2, tab3 = st.tabs(["Correlation Analysis", "Cointegration Analysis", "Cluster Analysis"])

    with tab1:
        st.subheader("Correlation-Based Pair Selection")
        corr_method = st.selectbox("Correlation Method", ["pearson", "spearman"])
        corr_threshold = st.slider("Correlation Threshold", 0.5, 1.0, 0.8, 0.05)

        if st.button("Find Correlated Pairs"):
            with st.spinner("Calculating correlations..."):
                correlated_pairs = corr_analyzer.get_highly_correlated_pairs(corr_method, corr_threshold)
                st.session_state.latest_analysis['correlated_pairs'] = correlated_pairs
                st.dataframe(correlated_pairs)
                fig = corr_analyzer.plot_correlation_matrix(corr_method)
                st.plotly_chart(fig, use_container_width=True)

    with tab2:
        st.subheader("Cointegration-Based Pair Selection")
        p_value_threshold = st.slider("P-Value Threshold", 0.01, 0.1, 0.05, 0.01)

        if st.button("Find Cointegrated Pairs"):
            with st.spinner("Running cointegration tests..."):
                coint_results = find_cointegrated_pairs(prices, p_threshold=p_value_threshold)
                st.session_state.latest_analysis['cointegrated_pairs'] = pd.DataFrame(coint_results)
                st.dataframe(st.session_state.latest_analysis['cointegrated_pairs'])

    with tab3:
        st.subheader("Cluster-Based Pair Selection")
        n_clusters = st.slider("Number of Clusters (for K-Means)", 2, 20, 5)

        if st.button("Run Clustering"):
            with st.spinner("Performing clustering analysis..."):
                clusters = cluster_analyzer.kmeans_clustering(returns, n_clusters=n_clusters)
                st.session_state.latest_analysis['clusters'] = clusters
                st.write(f"Found {len(clusters)} clusters:")
                st.json(clusters)
                fig = cluster_analyzer.plot_cluster_heatmap(returns.corr(), clusters, "Clustered Correlation Matrix")
                st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    st.subheader("Final Pair Selection")

    corr_pairs_set = set()
    if 'correlated_pairs' in st.session_state.latest_analysis:
        df = st.session_state.latest_analysis['correlated_pairs']
        corr_pairs_set = {tuple(sorted(p)) for p in df[['Asset1', 'Asset2']].values}

    coint_pairs_set = set()
    if 'cointegrated_pairs' in st.session_state.latest_analysis:
        df = st.session_state.latest_analysis['cointegrated_pairs']
        coint_pairs_set = {tuple(sorted((p['asset1'], p['asset2']))) for idx, p in df.iterrows()}

    if corr_pairs_set and coint_pairs_set:
        st.write("Overlap between Correlation and Cointegration methods:")
        fig, ax = plt.subplots()
        venn3([corr_pairs_set, coint_pairs_set, set()], set_labels=('Correlated', 'Cointegrated', ''))
        st.pyplot(fig)

    all_found_pairs = sorted(list(corr_pairs_set.union(coint_pairs_set)))
    intersection_pairs = list(corr_pairs_set.intersection(coint_pairs_set))

    st.session_state.selected_pairs = st.multiselect(
        "Select pairs for backtesting:",
        options=all_found_pairs,
        default=st.session_state.selected_pairs or intersection_pairs,
        format_func=lambda x: f"{x[0]}-{x[1]}"
    )