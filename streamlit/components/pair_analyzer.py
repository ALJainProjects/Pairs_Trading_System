import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime
from typing import Dict, List, Tuple

from src.analysis.correlation_analysis import CorrelationAnalyzer
from src.models.statistical import StatisticalModel
from src.analysis.clustering_analysis import AssetClusteringAnalyzer
from src.utils.visualization import PlotlyVisualizer


class EnhancedPairAnalyzer:
    """Enhanced pair analysis component with comprehensive analytics."""

    def __init__(self):
        self.correlation_analyzer = CorrelationAnalyzer()
        self.cointegration_analyzer = StatisticalModel()
        self.clustering_analyzer = AssetClusteringAnalyzer()
        self.visualizer = PlotlyVisualizer()

    def render(self):
        """Render the enhanced pair analysis interface."""
        st.header("Pair Analysis & Selection")

        if 'historical_data' not in st.session_state:
            st.warning("Please load data first in the Data Loading section.")
            return

        # Create tabs for different analysis methods
        tab1, tab2, tab3, tab4 = st.tabs([
            "Correlation Analysis",
            "Cointegration Analysis",
            "Clustering Analysis",
            "Selected Pairs"
        ])

        with tab1:
            self._render_correlation_analysis()

        with tab2:
            self._render_cointegration_analysis()

        with tab3:
            self._render_clustering_analysis()

        with tab4:
            self._render_selected_pairs()

    def _render_correlation_analysis(self):
        """Render correlation analysis section."""
        st.subheader("Correlation Analysis")

        # Parameters
        col1, col2 = st.columns(2)
        with col1:
            correlation_threshold = st.slider(
                "Correlation Threshold",
                min_value=0.0,
                max_value=1.0,
                value=0.7,
                step=0.05
            )
            lookback = st.number_input(
                "Lookback Period (days)",
                min_value=30,
                max_value=252,
                value=126
            )

        with col2:
            correlation_method = st.selectbox(
                "Correlation Method",
                ["pearson", "spearman", "partial"]
            )
            rolling_window = st.number_input(
                "Rolling Window (days)",
                min_value=20,
                max_value=126,
                value=63
            )

        if st.button("Run Correlation Analysis"):
            try:
                with st.spinner("Running correlation analysis..."):
                    # Get returns data
                    returns = self._calculate_returns(st.session_state['historical_data'])

                    # Run analysis
                    corr_matrix = self.correlation_analyzer.calculate_correlation_matrix(
                        returns,
                        method=correlation_method,
                        min_periods=lookback
                    )

                    # Get highly correlated pairs
                    pairs = self.correlation_analyzer.get_highly_correlated_pairs(
                        correlation_type=correlation_method,
                        threshold=correlation_threshold,
                        absolute=True
                    )

                    # Store results
                    st.session_state['correlation_results'] = {
                        'matrix': corr_matrix,
                        'pairs': pairs
                    }

                    # Display results
                    self._display_correlation_results(corr_matrix, pairs, returns)

            except Exception as e:
                st.error(f"Error in correlation analysis: {str(e)}")

    def _render_cointegration_analysis(self):
        """Render cointegration analysis section."""
        st.subheader("Cointegration Analysis")

        # Parameters
        col1, col2 = st.columns(2)
        with col1:
            significance_level = st.slider(
                "Significance Level",
                min_value=0.01,
                max_value=0.10,
                value=0.05,
                step=0.01
            )
            max_pairs = st.number_input(
                "Maximum Pairs to Test",
                min_value=10,
                max_value=1000,
                value=100
            )

        with col2:
            min_half_life = st.number_input(
                "Minimum Half-Life (days)",
                min_value=1,
                max_value=30,
                value=5
            )
            max_half_life = st.number_input(
                "Maximum Half-Life (days)",
                min_value=31,
                max_value=252,
                value=126
            )

        if st.button("Run Cointegration Analysis"):
            try:
                with st.spinner("Running cointegration analysis..."):
                    # Get price data
                    prices = self._get_price_data(st.session_state['historical_data'])

                    # Run analysis
                    cointegrated_pairs = self.cointegration_analyzer.find_cointegrated_pairs(
                        prices=prices,
                        significance_level=significance_level,
                        max_pairs=max_pairs,
                        min_half_life=min_half_life,
                        max_half_life=max_half_life
                    )

                    # Store results
                    st.session_state['cointegration_results'] = cointegrated_pairs

                    # Display results
                    self._display_cointegration_results(cointegrated_pairs, prices)

            except Exception as e:
                st.error(f"Error in cointegration analysis: {str(e)}")

    def _render_clustering_analysis(self):
        """Render clustering analysis section."""
        st.subheader("Clustering Analysis")

        # Parameters
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

        if st.button("Run Clustering Analysis"):
            try:
                with st.spinner("Running clustering analysis..."):
                    # Get returns data
                    returns = self._calculate_returns(st.session_state['historical_data'])

                    # Run clustering
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
                    else:  # graph
                        clusters = self.clustering_analyzer.graph_based_clustering(
                            returns.corr(),
                            threshold=similarity_threshold,
                            min_cluster_size=min_cluster_size
                        )

                    # Store results
                    st.session_state['clustering_results'] = clusters

                    # Display results
                    self._display_clustering_results(clusters, returns)

            except Exception as e:
                st.error(f"Error in clustering analysis: {str(e)}")

    def _render_selected_pairs(self):
        """Render selected pairs section."""
        st.subheader("Selected Trading Pairs")

        # Get pairs from different methods
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

        # Find pairs that appear in multiple methods
        intersection = correlation_pairs.intersection(cointegration_pairs)
        if clustering_pairs:
            intersection = intersection.intersection(clustering_pairs)

        # Display Venn diagram of overlap
        self._plot_pair_overlap(
            correlation_pairs,
            cointegration_pairs,
            clustering_pairs
        )

        # Allow manual selection/deselection
        st.subheader("Pair Selection")
        selected_pairs = st.multiselect(
            "Select pairs for trading",
            list(correlation_pairs.union(cointegration_pairs, clustering_pairs)),
            default=list(intersection)
        )

        if st.button("Confirm Selected Pairs"):
            if selected_pairs:
                # Store in session state
                st.session_state['selected_pairs'] = pd.DataFrame({
                    'Pair': selected_pairs
                })
                st.success(f"Selected {len(selected_pairs)} pairs for trading!")
            else:
                st.warning("Please select at least one pair.")

    def _calculate_returns(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate returns from price data."""
        prices = data.pivot(
            index='date',
            columns='ticker',
            values='adj_close'
        )
        returns = prices.pct_change().dropna()
        return returns

    def _get_price_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Get price data in proper format."""
        return data.pivot(
            index='date',
            columns='ticker',
            values='adj_close'
        )

    def _display_correlation_results(self,
                                     corr_matrix: pd.DataFrame,
                                     pairs: pd.DataFrame,
                                     returns: pd.DataFrame):
        """Display correlation analysis results."""
        # Plot correlation heatmap
        fig = self.visualizer.plot_correlation_matrix(
            corr_matrix,
            title="Asset Correlation Matrix"
        )
        st.plotly_chart(fig)

        # Show top pairs
        st.subheader("Top Correlated Pairs")
        st.dataframe(pairs)

        # Allow detailed pair analysis
        if len(pairs) > 0:
            pair = st.selectbox(
                "Select pair for detailed analysis",
                pairs['Pair'].tolist()
            )
            if pair:
                self._display_pair_details(pair, returns)

    def _display_cointegration_results(self,
                                       cointegrated_pairs: List[Dict],
                                       prices: pd.DataFrame):
        """Display cointegration analysis results."""
        if not cointegrated_pairs:
            st.warning("No cointegrated pairs found.")
            return

        # Convert to DataFrame for display
        pairs_df = pd.DataFrame(cointegrated_pairs)
        st.dataframe(pairs_df)

        # Allow detailed spread analysis
        pair = st.selectbox(
            "Select pair for spread analysis",
            [(p['stock1'], p['stock2']) for p in cointegrated_pairs]
        )
        if pair:
            self._display_spread_analysis(pair, prices)

    def _display_clustering_results(self,
                                    clusters: List[List[str]],
                                    returns: pd.DataFrame):
        """Display clustering analysis results."""
        # Display clusters
        for i, cluster in enumerate(clusters, 1):
            st.write(f"Cluster {i}: {', '.join(cluster)}")

        # Plot cluster visualization
        fig = self.clustering_analyzer.plot_cluster_heatmap(
            returns.corr(),
            clusters,
            "Clustered Asset Correlation"
        )
        st.plotly_chart(fig)

    def _display_pair_details(self, pair: Tuple[str, str], returns: pd.DataFrame):
        """Display detailed analysis for a selected pair."""
        ticker1, ticker2 = pair

        # Create subplots
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=["Returns", "Rolling Correlation"]
        )

        # Plot returns
        fig.add_trace(
            go.Scatter(
                x=returns.index,
                y=returns[ticker1],
                name=f"{ticker1} Returns",
                mode='lines'
            ),
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(
                x=returns.index,
                y=returns[ticker2],
                name=f"{ticker2} Returns",
                mode='lines'
            ),
            row=1, col=1
        )

        # Plot rolling correlation
        roll_corr = returns[ticker1].rolling(window=63).corr(returns[ticker2])
        fig.add_trace(
            go.Scatter(
                x=returns.index,
                y=roll_corr,
                name="Rolling Correlation",
                mode='lines'
            ),
            row=2, col=1
        )

        fig.update_layout(height=800)
        st.plotly_chart(fig)

    def _display_spread_analysis(self, pair: Tuple[str, str], prices: pd.DataFrame):
        """Display spread analysis for a cointegrated pair."""
        ticker1, ticker2 = pair

        # Calculate spread
        spread = self.cointegration_analyzer.calculate_spread(
            prices[ticker1],
            prices[ticker2]
        )

        # Plot spread
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=spread.index,
                y=spread,
                name='Spread',
                mode='lines'
            )
        )

        # Add mean and standard deviation bands
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
        import streamlit as st

        # Ensure the input sets are valid
        if not (isinstance(correlation_pairs, set) and
                isinstance(cointegration_pairs, set) and
                isinstance(clustering_pairs, set)):
            st.error("All input arguments must be sets.")
            return

        # Create the Venn diagram
        fig, ax = plt.subplots(figsize=(10, 10))
        venn3(
            subsets=[correlation_pairs, cointegration_pairs, clustering_pairs],
            set_labels=('Correlation', 'Cointegration', 'Clustering')
        )

        # Add a title to the plot
        plt.title("Venn Diagram of Pair Overlap")

        # Display the plot in Streamlit
        st.pyplot(fig)

    def _get_pairs_from_correlation(self) -> set:
        """Extract pairs from correlation results."""
        if 'correlation_results' not in st.session_state:
            return set()

        pairs = st.session_state['correlation_results']['pairs']
        return set(tuple(sorted(p)) for p in pairs['Pair'])

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
            # Create pairs from assets in same cluster
            for i, asset1 in enumerate(cluster):
                for asset2 in cluster[i + 1:]:
                    pairs.add(tuple(sorted([asset1, asset2])))
        return pairs

    def save_selected_pairs(self, pairs: List[Tuple[str, str]]):
        """Save selected pairs with detailed metrics."""
        if not pairs:
            return

        # Calculate comprehensive metrics for each pair
        pair_metrics = []
        returns = self._calculate_returns(st.session_state['historical_data'])
        prices = self._get_price_data(st.session_state['historical_data'])

        for pair in pairs:
            ticker1, ticker2 = pair
            metrics = {
                'pair': pair,
                'correlation': returns[ticker1].corr(returns[ticker2]),
                'cointegration_pval': self.cointegration_analyzer._single_coint_test(
                    prices[ticker1],
                    prices[ticker2]
                ),
                'half_life': self.cointegration_analyzer.calculate_half_life(
                    prices[ticker1] - prices[ticker2]
                )
            }
            pair_metrics.append(metrics)

        # Create DataFrame and store
        metrics_df = pd.DataFrame(pair_metrics)
        st.session_state['pair_metrics'] = metrics_df

        # Save pairs in standardized format
        st.session_state['selected_pairs'] = pd.DataFrame({
            'Pair': pairs,
            'Asset1': [p[0] for p in pairs],
            'Asset2': [p[1] for p in pairs]
        })

    def _analyze_pair_stability(self, pair: Tuple[str, str], returns: pd.DataFrame):
        """Analyze the stability of a trading pair over time."""
        ticker1, ticker2 = pair

        # Calculate rolling metrics
        window_sizes = [21, 63, 126]  # 1m, 3m, 6m
        rolling_metrics = {}

        for window in window_sizes:
            # Rolling correlation
            roll_corr = returns[ticker1].rolling(window=window).corr(returns[ticker2])

            # Rolling beta
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

        # Plot rolling metrics
        fig = make_subplots(
            rows=2,
            cols=1,
            subplot_titles=['Rolling Correlation', 'Rolling Beta'],
            vertical_spacing=0.15
        )

        colors = ['blue', 'green', 'red']
        for (period, metrics), color in zip(rolling_metrics.items(), colors):
            # Correlation plot
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

            # Beta plot
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

        # Calculate spread and z-score
        spread = self.cointegration_analyzer.calculate_spread(
            prices[ticker1],
            prices[ticker2]
        )

        z_score = (spread - spread.rolling(window=21).mean()) / \
                  spread.rolling(window=21).std()

        # Simulate basic signals
        long_entry = z_score < -2.0
        short_entry = z_score > 2.0
        exit_signal = abs(z_score) < 0.5

        # Calculate strategy returns
        strategy_returns = pd.Series(0, index=returns.index)
        position = 0

        for i in range(1, len(returns)):
            if position == 0:  # No position
                if long_entry.iloc[i]:
                    position = 1
                elif short_entry.iloc[i]:
                    position = -1
            else:  # In position
                if exit_signal.iloc[i]:
                    position = 0

            strategy_returns.iloc[i] = position * (
                    returns[ticker1].iloc[i] - returns[ticker2].iloc[i]
            )

        # Calculate metrics
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

        # Calculate various risk metrics
        pair_returns = returns[ticker1] - returns[ticker2]

        # Value at Risk
        var_95 = np.percentile(pair_returns, 5)
        var_99 = np.percentile(pair_returns, 1)

        # Expected Shortfall
        es_95 = pair_returns[pair_returns <= var_95].mean()
        es_99 = pair_returns[pair_returns <= var_99].mean()

        # Tail correlation with market
        market_returns = returns.mean(axis=1)  # simple proxy
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