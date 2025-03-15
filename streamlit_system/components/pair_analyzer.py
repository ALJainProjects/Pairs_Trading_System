import time

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Dict, List, Tuple
from statsmodels.tsa.stattools import coint

from src.analysis.correlation_analysis import CorrelationAnalyzer
from src.analysis.cointegration import find_cointegrated_pairs, calculate_half_life
from src.analysis.clustering_analysis import AssetClusteringAnalyzer
from src.analysis.denoiser_usage import AssetAnalyzer
from src.models.statistical import StatisticalModel
from src.utils.visualization import PlotlyVisualizer

from src.analysis.covariance_estimation import (
    GraphicalLassoCovariance, KalmanCovariance, OLSCovariance
)



class EnhancedPairAnalyzer:
    """Enhanced pair analysis component with comprehensive analytics."""

    def __init__(self):
        self.correlation_analyzer = None
        self.cointegration_analyzer = None
        self.clustering_analyzer = AssetClusteringAnalyzer()
        self.visualizer = PlotlyVisualizer()

    def render(self):
        """Render the enhanced pair analysis interface."""
        st.header("Pair Analysis & Selection")

        self.status_container = st.empty()
        self.progress_bar = st.empty()

        if 'historical_data' not in st.session_state:
            st.warning("Please load data first in the Data Loading section.")
            return

        try:
            with self.status_container.container():
                st.info("Initializing analysis components...")
            if self.correlation_analyzer is None:
                returns = self._calculate_returns(st.session_state['historical_data'])
                self.correlation_analyzer = CorrelationAnalyzer(returns=returns)
                self.status_container.empty()
        except Exception as e:
            self.status_container.error(f"Initialization failed: {str(e)}")
            return

        tab1, tab2, tab3, tab4 = st.tabs([
            "Correlation Analysis",
            "Cointegration Analysis",
            "Clustering Analysis",
            "Selected Pairs"
        ])

        with st.expander("About Denoising Methods", expanded=False):
            st.markdown("""
            **Denoising Methods:**
            - **None**: Uses raw returns without denoising
            - **Wavelet**: Uses wavelet transform to remove high-frequency noise while preserving significant trends
            - **PCA**: Uses Principal Component Analysis to reconstruct data using only principal components, removing minor variations

            Denoising can help identify more stable relationships between assets by reducing market noise.
            """)

        with tab1:
            self._render_correlation_analysis()

        with tab2:
            self._render_cointegration_analysis()

        with tab3:
            self._render_clustering_analysis()

        with tab4:
            self._render_selected_pairs()

    def _render_correlation_analysis(self):
        """Render correlation analysis section with full history visualization."""
        progress_bar = st.progress(0)
        status_text = st.empty()
        st.subheader("Correlation Analysis")

        col1, col2 = st.columns(2)
        with col1:
            min_correlation, max_correlation = st.slider(
                "Correlation Range",
                min_value=0.0,
                max_value=1.0,
                value=(0.7, 0.95),
                step=0.05,
                help="Select minimum and maximum correlation thresholds"
            )
            lookback = st.number_input(
                "Lookback Period (days)",
                min_value=30,
                max_value=1008,
                value=504
            )

        with col2:
            correlation_method = st.selectbox(
                "Correlation Method",
                ["pearson", "partial"]
            )
            rolling_window = st.number_input(
                "Rolling Window (days)",
                min_value=20,
                max_value=756,
                value=63
            )
            denoising_method = st.selectbox(
                "Denoising Method",
                ["None", "Wavelet", "PCA"],
                help="Select a method to reduce noise in return data before analysis"
            )
            st.session_state['denoising_method_correlation'] = denoising_method

        if st.button("Run Correlation Analysis"):
            try:
                status_text.text("Calculating returns...")
                progress_bar.progress(10)

                method = None
                if denoising_method == "Wavelet":
                    method = "wavelet"
                    status_text.text("Calculating and denoising returns with Wavelet method...")
                elif denoising_method == "PCA":
                    method = "pca"
                    status_text.text("Calculating and denoising returns with PCA method...")
                else:
                    status_text.text("Calculating returns...")

                full_returns = self._calculate_returns(st.session_state['historical_data'], method=method)
                lookback_returns = full_returns.tail(lookback).copy()

                status_text.text("Initializing correlation analyzer...")
                progress_bar.progress(20)

                self.correlation_analyzer = CorrelationAnalyzer(returns=lookback_returns)

                status_text.text(f"Calculating {correlation_method} correlation matrix...")
                progress_bar.progress(40)

                if correlation_method == 'pearson':
                    corr_matrix = self.correlation_analyzer.calculate_pearson_correlation()
                else:
                    corr_matrix = self.correlation_analyzer.calculate_partial_correlation()

                status_text.text("Identifying correlated pairs...")
                progress_bar.progress(60)

                pairs = self.correlation_analyzer.get_highly_correlated_pairs(
                    correlation_type=correlation_method,
                    threshold=min_correlation,
                    absolute=True
                )
                pairs = pairs[
                    (pairs['correlation'].abs() >= min_correlation) &
                    (pairs['correlation'].abs() <= max_correlation)
                    ]
                pairs['abs_correlation'] = pairs['correlation'].abs()
                pairs = pairs.sort_values('abs_correlation', ascending=False).drop('abs_correlation', axis=1)

                status_text.text("Computing rolling correlations...")
                progress_bar.progress(80)

                st.session_state['correlation_results'] = {
                    'matrix': corr_matrix,
                    'pairs': pairs,
                    'method': correlation_method
                }

                status_text.text("Generating visualizations...")
                progress_bar.progress(95)

                fig_matrix = self.correlation_analyzer.plot_correlation_matrix(
                    correlation_type=correlation_method,
                    title=f"{correlation_method.capitalize()} Correlation Matrix"
                )
                st.plotly_chart(fig_matrix)

                st.subheader("Top 5 Most Correlated Pairs")
                if len(pairs) > 0:
                    top_5_pairs = pairs.head(5)
                    st.dataframe(top_5_pairs.round(4))

                    full_returns_copy = full_returns.copy()

                    fig = make_subplots(
                        rows=5,
                        cols=1,
                        subplot_titles=[
                            f"{row['asset1']} - {row['asset2']} "
                            f"(Full Period: {full_returns[row['asset1']].corr(full_returns[row['asset2']]):.3f}, "
                            f"Recent: {full_returns[row['asset1']].tail(63).corr(full_returns_copy[row['asset2']].tail(63)):.3f})"
                            for _, row in top_5_pairs.iterrows()
                        ],
                        vertical_spacing=0.05,
                        row_heights=[300] * 5
                    )

                    for idx, (_, pair) in enumerate(top_5_pairs.iterrows(), 1):
                        ticker1, ticker2 = pair['asset1'], pair['asset2']

                        historical_corr = full_returns[ticker1].rolling(window=rolling_window).corr(
                            full_returns[ticker2])

                        fig.add_trace(
                            go.Scatter(
                                x=full_returns.index,
                                y=historical_corr,
                                name=f'{ticker1}-{ticker2}',
                                mode='lines',
                                line=dict(width=2),
                                hovertemplate='Date: %{x}<br>Correlation: %{y:.3f}<extra></extra>'
                            ),
                            row=idx,
                            col=1
                        )

                        fig.update_yaxes(
                            range=[-1, 1],
                            title_text="Correlation",
                            row=idx,
                            col=1,
                            gridcolor='lightgray'
                        )

                        fig.update_xaxes(
                            gridcolor='lightgray',
                            row=idx,
                            col=1,
                            range=[full_returns.index.min(), full_returns.index.max()]
                        )

                    fig.update_layout(
                        height=1500,
                        title="Historical Correlation Analysis - Top 5 Pairs",
                        showlegend=True,
                        legend=dict(
                            orientation="h",
                            yanchor="bottom",
                            y=1.02,
                            xanchor="right",
                            x=1
                        ),
                        plot_bgcolor='white'
                    )

                    st.plotly_chart(fig)

                    st.subheader("Correlation Stability Metrics")
                    stability_metrics = []
                    for _, pair in top_5_pairs.iterrows():
                        ticker1, ticker2 = pair['asset1'], pair['asset2']
                        historical_corr = full_returns[ticker1].rolling(window=5).corr(
                            full_returns[ticker2])

                        metrics = {
                            'Pair': f"{ticker1}-{ticker2}",
                            'Full Period Mean': historical_corr.mean(),
                            'Recent Mean (63d)': historical_corr.tail(63).mean(),
                            'All-time Min': historical_corr.min(),
                            'All-time Max': historical_corr.max(),
                            'Std Dev': historical_corr.std()
                        }
                        stability_metrics.append(metrics)

                    st.dataframe(pd.DataFrame(stability_metrics).round(4))

                else:
                    st.warning("No pairs found within the correlation range.")

                progress_bar.progress(100)
                status_text.success("Correlation analysis complete!")

            except Exception as e:
                status_text.error(f"Error in correlation analysis: {str(e)}")
                st.error(f"Full error: {str(e)}")
            finally:
                time.sleep(1)
                progress_bar.empty()

    def _render_cointegration_analysis(self):
        """Render cointegration analysis section with progress tracking."""
        st.subheader("Cointegration Analysis")

        col1, col2 = st.columns(2)
        with col1:
            significance_level = st.slider(
                "Significance Level",
                min_value=0.01,
                max_value=0.30,
                value=0.05,
                step=0.01
            )
            max_pairs = st.number_input(
                "Maximum Pairs to Test",
                min_value=10,
                max_value=10000,
                value=5050
            )

        with col2:
            min_half_life = st.number_input(
                "Minimum Half-Life (days)",
                min_value=1,
                max_value=150,
                value=5
            )
            max_half_life = st.number_input(
                "Maximum Half-Life (days)",
                min_value=31,
                max_value=504,
                value=126
            )
            lookback_window = st.number_input(
                "Lookback_window (days)",
                min_value=31,
                max_value=1008,
                value=504
            )
            denoising_method = st.selectbox(
                "Denoising Method",
                ["None", "Wavelet", "PCA"],
                help="Select a method to reduce noise in return data before analysis"
            )
            st.session_state['denoising_method_cointegration'] = denoising_method

        if st.button("Run Cointegration Analysis"):
            progress_bar = st.progress(0)
            status_text = st.empty()

            try:
                status_text.text("Preparing price data...")
                progress_bar.progress(10)

                method = None
                if denoising_method == "Wavelet":
                    method = "wavelet"
                    status_text.text("Preparing and denoising price data with Wavelet method...")
                elif denoising_method == "PCA":
                    method = "pca"
                    status_text.text("Preparing and denoising price data with PCA method...")
                else:
                    status_text.text("Preparing price data...")

                prices = self._get_price_data(st.session_state['historical_data'])
                prices = prices.ffill().bfill()

                # Add denoising for returns if needed for derived features
                if method:
                    returns = self._calculate_returns(st.session_state['historical_data'], method=method)
                    st.session_state['returns_denoised'] = returns

                status_text.text("Initializing cointegration analysis...")
                progress_bar.progress(20)

                cointegrated_pairs = find_cointegrated_pairs(
                    prices=prices.ffill().bfill(),
                    significance_level=significance_level,
                    lookback_period=lookback_window,
                    max_pairs=max_pairs,
                    min_half_life=min_half_life,
                    max_half_life=max_half_life,
                )

                status_text.text("Storing results...")
                progress_bar.progress(90)
                st.session_state['cointegration_results'] = cointegrated_pairs

                status_text.text("Generating visualizations...")
                progress_bar.progress(95)

                self._display_cointegration_results(cointegrated_pairs, prices)

                progress_bar.progress(100)
                status_text.success("Cointegration analysis complete!")

            except Exception as e:
                status_text.error(f"Error in cointegration analysis: {str(e)}")
            finally:
                time.sleep(1)
                progress_bar.empty()

    def _render_clustering_analysis(self):
        """Render clustering analysis section with progress tracking."""
        st.subheader("Clustering Analysis")

        col1, col2 = st.columns(2)
        with col1:
            clustering_method = st.selectbox(
                "Clustering Method",
                ["kmeans", "dbscan", "hierarchical", "graph"]
            )
            n_clusters = st.number_input(
                "Number of Clusters",
                min_value=2,
                max_value=20,
                value=5
            ) if clustering_method in ["kmeans", "hierarchical"] else None

        with col2:
            similarity_threshold = st.slider(
                "Similarity Threshold",
                min_value=0.0,
                max_value=1.0,
                value=0.8,
                step=0.05
            )
            min_cluster_size = st.number_input(
                "Minimum Cluster Size",
                min_value=2,
                max_value=10,
                value=2
            )
            denoising_method = st.selectbox(
                "Denoising Method",
                ["None", "Wavelet", "PCA"],
                help="Select a method to reduce noise in return data before clustering"
            )
            st.session_state['denoising_method_clustering'] = denoising_method

        if st.button("Run Clustering Analysis"):
            progress_bar = st.progress(0)
            status_text = st.empty()

            try:
                status_text.text("Preparing return data...")
                progress_bar.progress(10)
                # Update inside the "Run Clustering Analysis" button handler
                method = None
                if denoising_method == "Wavelet":
                    method = "wavelet"
                    status_text.text("Preparing and denoising returns with Wavelet method...")
                elif denoising_method == "PCA":
                    method = "pca"
                    status_text.text("Preparing and denoising returns with PCA method...")
                else:
                    status_text.text("Preparing return data...")

                returns = self._calculate_returns(st.session_state['historical_data'], method=method)
                returns = returns.ffill().bfill()
                returns = returns.dropna()

                status_text.text(f"Initializing {clustering_method} clustering...")
                progress_bar.progress(30)

                status_text.text("Running clustering algorithm...")
                progress_bar.progress(50)

                if clustering_method == "kmeans":
                    clusters = self.clustering_analyzer.kmeans_clustering(
                        returns,
                        n_clusters=n_clusters
                    )
                elif clustering_method == "dbscan":
                    clusters = self.clustering_analyzer.dbscan_clustering(
                        returns,
                        eps=1 - similarity_threshold,
                        min_samples=min_cluster_size
                    )
                elif clustering_method == "hierarchical":
                    clusters = self.clustering_analyzer.agglomerative_clustering(
                        returns,
                        n_clusters=n_clusters
                    )
                else:
                    clusters = self.clustering_analyzer.graph_based_clustering(
                        returns.corr(),
                        threshold=similarity_threshold,
                        min_cluster_size=min_cluster_size
                    )

                status_text.text("Storing clustering results...")
                progress_bar.progress(80)
                st.session_state['clustering_results'] = clusters

                status_text.text("Generating visualizations...")
                progress_bar.progress(90)
                self._display_clustering_results(clusters, returns)

                progress_bar.progress(100)
                status_text.success("Clustering analysis complete!")

            except Exception as e:
                status_text.error(f"Error in clustering analysis: {str(e)}")
            finally:
                time.sleep(1)
                progress_bar.empty()

    def _render_selected_pairs(self):
        """Render selected pairs section with progress tracking."""
        st.subheader("Selected Trading Pairs")

        progress_bar = st.progress(0)
        status_text = st.empty()

        try:
            status_text.text("Gathering pairs from analyses...")
            progress_bar.progress(20)

            correlation_pairs = set(
                self._get_pairs_from_correlation()
                if 'correlation_results' in st.session_state else set()
            )
            cointegration_pairs = set(
                self._get_pairs_from_cointegration()
                if 'cointegration_results' in st.session_state else set()
            )
            clustering_pairs = set(
                self._get_pairs_from_clustering()
                if 'clustering_results' in st.session_state else set()
            )

            status_text.text("Finding pair intersections...")
            progress_bar.progress(40)

            use_clustering = st.checkbox(
                "Include clustering pairs in intersection",
                value=False,
                help="When checked, pairs must appear in correlation, cointegration, AND clustering analyses. When unchecked, pairs only need to appear in correlation AND cointegration analyses."
            )

            intersection = correlation_pairs.intersection(cointegration_pairs)
            if clustering_pairs and use_clustering:
                intersection = intersection.intersection(clustering_pairs)

            st.write(f"Correlation pairs: {correlation_pairs}")
            st.write(f"Cointegration pairs: {cointegration_pairs}")
            if clustering_pairs:
                st.write(f"Clustering pairs: {clustering_pairs}")
            st.write(f"Intersection pairs: {intersection}")

            status_text.text("Generating pair overlap visualization...")
            progress_bar.progress(60)
            self._plot_pair_overlap(
                correlation_pairs,
                cointegration_pairs,
                clustering_pairs
            )

            status_text.text("Preparing pair selection interface...")
            progress_bar.progress(80)
            st.subheader("Pair Selection")
            selected_pairs = st.multiselect(
                "Select pairs for trading",
                list(correlation_pairs.union(cointegration_pairs, clustering_pairs)),
                default=list(intersection)
            )

            if st.button("Confirm Selected Pairs"):
                if selected_pairs:
                    status_text.text("Saving selected pairs...")
                    progress_bar.progress(90)
                    st.session_state['selected_pairs'] = pd.DataFrame({
                        'Pair': selected_pairs
                    })
                    progress_bar.progress(100)
                    status_text.success(f"Selected {len(selected_pairs)} pairs for trading!")
                else:
                    status_text.warning("Please select at least one pair.")

        except Exception as e:
            status_text.error(f"Error in pair selection: {str(e)}")
        finally:
            time.sleep(1)
            progress_bar.empty()

    def _calculate_returns(self, data: pd.DataFrame, method=None) -> pd.DataFrame:
        """Calculate returns from price data."""
        prices = data.pivot(index='Date', columns='Symbol', values='Adj_Close')

        prices = prices.sort_index()

        prices = prices.ffill().bfill()

        raw_returns = prices.pct_change()

        # print("Original Data Shape:", data.shape)
        # print("Pivoted Prices Shape:", prices.shape)
        # print("Raw Returns Shape Before Dropna:", raw_returns.shape)

        raw_returns = raw_returns.dropna()

        # print("Raw Returns Shape After Dropna:", raw_returns.shape)

        if method == "wavelet":
            asset_analyzer = AssetAnalyzer()
            asset_analyzer.fit(raw_returns)
            return asset_analyzer.denoise_wavelet(wavelet='db1', level=1, threshold=0.04)
        elif method == "pca":
            asset_analyzer = AssetAnalyzer()
            asset_analyzer.fit(raw_returns)
            return asset_analyzer.denoise_pca(n_components=5)
        else:
            return raw_returns

    def _get_price_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Get price data in proper format."""
        return data.pivot(
            index='Date',
            columns='Symbol',
            values='Adj_Close'
        )

    def _display_correlation_results(self, pairs, returns, rolling_corrs, correlation_method):
        """Display correlation analysis results with the correct correlation type."""
        try:
            fig = self.correlation_analyzer.plot_correlation_matrix(
                correlation_type=correlation_method,
                title=f"{correlation_method.capitalize()} Correlation Matrix"
            )
            st.plotly_chart(fig)

            st.subheader("Top Correlated Pairs")
            if len(pairs) > 0:
                display_pairs = pairs.copy()
                if 'correlation' in display_pairs.columns:
                    display_pairs['correlation'] = display_pairs['correlation'].round(4)
                st.dataframe(display_pairs)
            else:
                st.warning("No pairs found above the correlation threshold.")

            try:
                stability = self.correlation_analyzer.analyze_correlation_stability()
                st.subheader("Correlation Stability Analysis")
                st.dataframe(stability.round(4))
            except Exception as e:
                st.warning(f"Could not calculate correlation stability: {str(e)}")

            if len(pairs) > 0:
                selected_pair = st.selectbox(
                    "Select pair for detailed analysis",
                    [(row['asset1'], row['asset2']) for _, row in pairs.iterrows()]
                )

                if selected_pair:
                    with st.spinner("Generating pair analysis..."):
                        self._display_pair_details(selected_pair, returns, rolling_corrs)

                    try:
                        stability_fig = self._analyze_pair_stability(selected_pair, returns)
                        st.plotly_chart(stability_fig)
                    except Exception as e:
                        st.warning(f"Could not generate stability analysis: {str(e)}")

        except Exception as e:
            st.error(f"Error displaying correlation results: {str(e)}")

    def _display_pair_details(self,
                              pair: Tuple[str, str],
                              returns: pd.DataFrame,
                              rolling_corrs: Dict[str, pd.Series]):
        """Display detailed analysis for a selected pair."""
        try:
            ticker1, ticker2 = pair
            pair_name = self.correlation_analyzer._validate_pair_name(ticker1, ticker2)

            fig = make_subplots(
                rows=2, cols=1,
                subplot_titles=["Returns", "Rolling Correlation"],
                vertical_spacing=0.15
            )

            for ticker, name in [(ticker1, "Asset 1"), (ticker2, "Asset 2")]:
                if ticker in returns.columns:
                    fig.add_trace(
                        go.Scatter(
                            x=returns.index,
                            y=returns[ticker],
                            name=f"{ticker} Returns",
                            mode='lines'
                        ),
                        row=1, col=1
                    )
                else:
                    st.warning(f"No return data found for {ticker}")

            if rolling_corrs and pair_name in rolling_corrs:
                roll_corr = rolling_corrs[pair_name]
                fig.add_trace(
                    go.Scatter(
                        x=roll_corr.index,
                        y=roll_corr,
                        name="Rolling Correlation",
                        mode='lines'
                    ),
                    row=2, col=1
                )
            else:
                st.info("No rolling correlation data available for this pair")

            fig.update_layout(
                height=800,
                title=f"Pair Analysis: {ticker1} - {ticker2}",
                showlegend=True,
                yaxis2_range=[-1, 1]
            )

            st.plotly_chart(fig)

            if ticker1 in returns.columns and ticker2 in returns.columns:
                stats = {
                    "Full Period Correlation": returns[ticker1].corr(returns[ticker2]),
                    "Recent Correlation (63d)": returns[ticker1].tail(63).corr(returns[ticker2].tail(63)),
                    "Correlation Stability": returns[ticker1].rolling(63).corr(returns[ticker2]).std()
                }
                st.write("### Correlation Statistics")
                st.write(pd.Series(stats).round(4))

        except Exception as e:
            st.error(f"Error displaying pair details: {str(e)}")

    def _display_cointegration_results(self, cointegrated_pairs: List[Dict], prices: pd.DataFrame):
        """Display cointegration analysis results with separate plots for each pair."""
        if not cointegrated_pairs:
            st.warning("No cointegrated pairs found.")
            return

        prices = prices.ffill().bfill()

        pairs_df = pd.DataFrame(cointegrated_pairs)
        top_5_pairs = pairs_df.nsmallest(5, 'p_value')
        st.dataframe(top_5_pairs)

        for _, pair in top_5_pairs.iterrows():
            ticker1, ticker2 = pair['stock1'], pair['stock2']
            pair_name = f"{ticker1}-{ticker2}"

            full_spread = StatisticalModel().calculate_spread(
                prices[ticker1],
                prices[ticker2]
            )

            rolling_mean = full_spread.rolling(window=21).mean()
            rolling_std = full_spread.rolling(window=21).std()
            upper_band = rolling_mean + 2 * rolling_std
            lower_band = rolling_mean - 2 * rolling_std
            z_score = (full_spread - rolling_mean) / rolling_std

            spread_fig = go.Figure()

            spread_fig.add_trace(
                go.Scatter(
                    x=full_spread.index,
                    y=full_spread,
                    name='Spread',
                    mode='lines',
                    line=dict(color='blue')
                )
            )

            spread_fig.add_trace(
                go.Scatter(
                    x=rolling_mean.index,
                    y=rolling_mean,
                    name='Mean',
                    mode='lines',
                    line=dict(color='gray', dash='dash')
                )
            )

            spread_fig.add_trace(
                go.Scatter(
                    x=upper_band.index,
                    y=upper_band,
                    name='+2σ',
                    mode='lines',
                    line=dict(color='red', dash='dash')
                )
            )

            spread_fig.add_trace(
                go.Scatter(
                    x=lower_band.index,
                    y=lower_band,
                    name='-2σ',
                    mode='lines',
                    line=dict(color='red', dash='dash')
                )
            )

            latest_spread = full_spread.iloc[-1]
            spread_fig.add_annotation(
                x=full_spread.index[-1],
                y=latest_spread,
                text=f'Latest Spread: {latest_spread:.2f}',
                showarrow=True,
                arrowhead=1
            )

            spread_fig.update_layout(
                title=f"Historical Spread: {pair_name}",
                xaxis_title="Date",
                yaxis_title="Spread Value",
                height=500,
                showlegend=True,
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1
                )
            )

            st.plotly_chart(spread_fig)

            zscore_fig = go.Figure()

            zscore_fig.add_trace(
                go.Scatter(
                    x=z_score.index,
                    y=z_score,
                    name='Z-Score',
                    mode='lines',
                    line=dict(color='purple')
                )
            )

            zscore_fig.add_hline(y=2, line_dash="dash", line_color="red")
            zscore_fig.add_hline(y=-2, line_dash="dash", line_color="red")
            zscore_fig.add_hline(y=0, line_dash="dash", line_color="gray")

            latest_z = z_score.iloc[-1]
            zscore_fig.add_annotation(
                x=z_score.index[-1],
                y=latest_z,
                text=f'Latest Z-Score: {latest_z:.2f}',
                showarrow=True,
                arrowhead=1
            )

            zscore_fig.update_layout(
                title=f"Z-Score Analysis: {pair_name}",
                xaxis_title="Date",
                yaxis_title="Z-Score",
                height=500,
                showlegend=True,
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1
                )
            )

            st.plotly_chart(zscore_fig)

        stats_data = []
        for _, pair in top_5_pairs.iterrows():
            ticker1, ticker2 = pair['stock1'], pair['stock2']
            spread = StatisticalModel().calculate_spread(
                prices[ticker1],
                prices[ticker2]
            )
            z_score = (spread - spread.rolling(21).mean()) / spread.rolling(21).std()

            stats = {
                'Pair': f"{ticker1}-{ticker2}",
                'Half-Life (days)': pair['half_life'],
                'P-Value': pair['p_value'],
                'Spread Std Dev': spread.std(),
                'Spread Mean': spread.mean(),
                'Latest Z-Score': z_score.iloc[-1],
                'Max Z-Score': z_score.max(),
                'Min Z-Score': z_score.min(),
                'Mean Reversion Time (days)': np.median(
                    z_score[abs(z_score) >= 2].index.to_series().diff().dt.days
                )
            }
            stats_data.append(stats)

        stats_df = pd.DataFrame(stats_data)
        st.write("### Cointegration Statistics")
        st.dataframe(stats_df.round(4))

    def _display_clustering_results(self,
                                    clusters: List[List[str]],
                                    returns: pd.DataFrame):
        """Display clustering analysis results."""
        full_corr = returns.corr()

        st.subheader("Clustering Overview")
        st.write(f"Total number of clusters: {len(clusters)}")
        st.write(f"Average cluster size: {np.mean([len(c) for c in clusters]):.2f}")

        sorted_clusters = sorted(clusters, key=len, reverse=True)

        for i, cluster in enumerate(sorted_clusters, 1):
            with st.expander(f"Cluster {i} ({len(cluster)} assets)", expanded=True):
                cluster_corr = full_corr.loc[cluster, cluster]

                fig = go.Figure(data=go.Heatmap(
                    z=cluster_corr.values,
                    x=cluster_corr.columns,
                    y=cluster_corr.index,
                    colorscale='RdBu',
                    zmid=0,
                    showscale=True
                ))

                fig.update_layout(
                    title=f"Correlation Heatmap - Cluster {i}",
                    width=800,
                    height=max(400, len(cluster) * 30),
                    xaxis_title="Assets",
                    yaxis_title="Assets"
                )

                fig.update_xaxes(tickangle=45)

                avg_corr = cluster_corr.values[np.triu_indices_from(cluster_corr.values, k=1)].mean()
                min_corr = cluster_corr.values[np.triu_indices_from(cluster_corr.values, k=1)].min()
                max_corr = cluster_corr.values[np.triu_indices_from(cluster_corr.values, k=1)].max()

                col1, col2, col3 = st.columns(3)
                col1.metric("Average Correlation", f"{avg_corr:.3f}")
                col2.metric("Min Correlation", f"{min_corr:.3f}")
                col3.metric("Max Correlation", f"{max_corr:.3f}")

                st.plotly_chart(fig)

                st.write("**Assets in this cluster:**")
                st.write(", ".join(sorted(cluster)))

                st.markdown("---")

        st.subheader("Full Clustering Heatmap")
        full_fig = self.clustering_analyzer.plot_cluster_heatmap(
            returns.corr(),
            sorted_clusters,
            "Complete Clustered Asset Correlation"
        )
        st.plotly_chart(full_fig)

    def _display_spread_analysis(self, pair: Tuple[str, str], prices: pd.DataFrame):
        """Display spread analysis for a cointegrated pair."""
        ticker1, ticker2 = pair

        spread = StatisticalModel().calculate_spread(
            prices[ticker1],
            prices[ticker2]
        )

        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=spread.index,
                y=spread,
                name='Spread',
                mode='lines'
            )
        )

        mean = spread.mean()
        std = spread.std()
        fig.add_hline(y=mean, line_dash="dash", line_color="gray")
        fig.add_hline(y=mean + 2 * std, line_dash="dash", line_color="red")
        fig.add_hline(y=mean - 2 * std, line_dash="dash", line_color="red")

        fig.update_layout(
            title=f"Spread Analysis: {ticker1} - {ticker2}",
            height=600
        )
        st.plotly_chart(fig)

    def _plot_pair_overlap(self,
                           correlation_pairs: set,
                           cointegration_pairs: set,
                           clustering_pairs: set):
        """Plot Venn diagram of pair overlap in a Streamlit app."""
        from matplotlib_venn import venn3
        import matplotlib.pyplot as plt

        if not (isinstance(correlation_pairs, set) and
                isinstance(cointegration_pairs, set) and
                isinstance(clustering_pairs, set)):
            st.error("All input arguments must be sets.")
            return

        fig, ax = plt.subplots(figsize=(10, 10))
        venn3(
            subsets=(correlation_pairs, cointegration_pairs, clustering_pairs),
            set_labels=('Correlation', 'Cointegration', 'Clustering')
        )

        plt.title("Venn Diagram of Pair Overlap")

        st.pyplot(fig)

    def _get_pairs_from_correlation(self) -> set:
        """Extract pairs from correlation results."""
        if 'correlation_results' not in st.session_state:
            return set()

        pairs = st.session_state['correlation_results']['pairs']
        return set(
            tuple(sorted([row['asset1'], row['asset2']]))
            for _, row in pairs.iterrows()
        )

    def _get_pairs_from_cointegration(self) -> set:
        """Extract pairs from cointegration results."""
        if 'cointegration_results' not in st.session_state:
            return set()

        pairs = st.session_state['cointegration_results']
        return set(
            tuple(sorted([p['stock1'], p['stock2']]))
            for p in pairs
        )

    def _get_pairs_from_clustering(self) -> set:
        """Extract pairs from clustering results."""
        if 'clustering_results' not in st.session_state:
            return set()

        clusters = st.session_state['clustering_results']
        pairs = set()
        for cluster in clusters:
            for i, asset1 in enumerate(cluster):
                for asset2 in cluster[i + 1:]:
                    pairs.add(tuple(sorted([asset1, asset2])))
        return pairs

    def save_selected_pairs(self, pairs: List[Tuple[str, str]]):
        """Save selected pairs with detailed metrics."""
        if not pairs:
            return

        pair_metrics = []
        returns = self._calculate_returns(st.session_state['historical_data'])
        prices = self._get_price_data(st.session_state['historical_data'])
        prices = prices.ffill().bfill()

        def _single_coint_test(asset1: pd.Series,
                               asset2: pd.Series) -> float:
            """Perform single cointegration test."""
            try:
                _, p_val, _ = coint(asset1, asset2)
                return p_val
            except:
                return 1.0

        for pair in pairs:
            ticker1, ticker2 = pair
            metrics = {
                'pair': pair,
                'correlation': returns[ticker1].corr(returns[ticker2]),
                'cointegration_pval':_single_coint_test(
                    prices[ticker1],
                    prices[ticker2]
                ),
                'half_life': calculate_half_life(
                    prices[ticker1] - prices[ticker2]
                )
            }
            pair_metrics.append(metrics)

        metrics_df = pd.DataFrame(pair_metrics)
        st.session_state['pair_metrics'] = metrics_df

        st.session_state['selected_pairs'] = pd.DataFrame({
            'Pair': pairs,
            'Asset1': [p[0] for p in pairs],
            'Asset2': [p[1] for p in pairs]
        })

    def _analyze_pair_stability(self, pair: Tuple[str, str], returns: pd.DataFrame):
        """Analyze the stability of a trading pair over time."""
        ticker1, ticker2 = pair

        window_sizes = [21, 63, 126]
        rolling_metrics = {}

        for window in window_sizes:
            roll_corr = returns[ticker1].rolling(window=window).corr(returns[ticker2])

            roll_beta = (
                    returns[[ticker1, ticker2]]
                    .rolling(window=window)
                    .cov()
                    .unstack()[ticker2][ticker1] /
                    returns[ticker2]
                    .rolling(window=window)
                    .var()
            )

            rolling_metrics[f'{window}d'] = {
                'correlation': roll_corr,
                'beta': roll_beta
            }

        fig = make_subplots(
            rows=2,
            cols=1,
            subplot_titles=['Rolling Correlation', 'Rolling Beta'],
            vertical_spacing=0.15
        )

        colors = ['blue', 'green', 'red']
        for (period, metrics), color in zip(rolling_metrics.items(), colors):
            fig.add_trace(
                go.Scatter(
                    x=returns.index,
                    y=metrics['correlation'],
                    name=f'{period} Correlation',
                    line=dict(color=color)
                ),
                row=1,
                col=1
            )

            fig.add_trace(
                go.Scatter(
                    x=returns.index,
                    y=metrics['beta'],
                    name=f'{period} Beta',
                    line=dict(color=color)
                ),
                row=2,
                col=1
            )

        fig.update_layout(
            height=800,
            title=f"Pair Stability Analysis: {ticker1} - {ticker2}",
            showlegend=True
        )

        return fig

    def _analyze_pair_profitability(self,
                                    pair: Tuple[str, str],
                                    prices: pd.DataFrame,
                                    returns: pd.DataFrame):
        """Analyze historical profitability of a pair."""
        ticker1, ticker2 = pair

        spread = StatisticalModel().calculate_spread(
            prices[ticker1],
            prices[ticker2]
        )

        z_score = (spread - spread.rolling(window=21).mean()) / \
                  spread.rolling(window=21).std()

        long_entry = z_score < -2.0
        short_entry = z_score > 2.0
        exit_signal = abs(z_score) < 0.5

        strategy_returns = pd.Series(0, index=returns.index)
        position = 0

        for i in range(1, len(returns)):
            if position == 0:
                if long_entry.iloc[i]:
                    position = 1
                elif short_entry.iloc[i]:
                    position = -1
            else:
                if exit_signal.iloc[i]:
                    position = 0

            strategy_returns.iloc[i] = position * (
                    returns[ticker1].iloc[i] - returns[ticker2].iloc[i]
            )

        metrics = {
            'Total Return': (1 + strategy_returns).prod() - 1,
            'Annual Return': (1 + strategy_returns).prod() ** (252 / len(returns)) - 1,
            'Volatility': strategy_returns.std() * np.sqrt(252),
            'Sharpe Ratio': strategy_returns.mean() / strategy_returns.std() * np.sqrt(252),
            'Max Drawdown': (
                    (1 + strategy_returns).cumprod() /
                    (1 + strategy_returns).cumprod().cummax() - 1
            ).min()
        }

        return strategy_returns, metrics

    def _analyze_pair_risks(self, pair: Tuple[str, str], returns: pd.DataFrame):
        """Analyze risks associated with a pair."""
        ticker1, ticker2 = pair

        pair_returns = returns[ticker1] - returns[ticker2]

        var_95 = np.percentile(pair_returns, 5)
        var_99 = np.percentile(pair_returns, 1)

        es_95 = pair_returns[pair_returns <= var_95].mean()
        es_99 = pair_returns[pair_returns <= var_99].mean()

        market_returns = returns.mean(axis=1)
        tail_mask = market_returns < np.percentile(market_returns, 5)
        tail_corr = pair_returns[tail_mask].corr(market_returns[tail_mask])

        risk_metrics = {
            'VaR (95%)': var_95,
            'VaR (99%)': var_99,
            'Expected Shortfall (95%)': es_95,
            'Expected Shortfall (99%)': es_99,
            'Tail Correlation': tail_corr
        }

        return risk_metrics