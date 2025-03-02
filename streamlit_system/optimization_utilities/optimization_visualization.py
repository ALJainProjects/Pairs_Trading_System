"""
Visualization components for optimization results.
"""

from typing import Dict, Any, Optional, List, Tuple
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import logging


class OptimizationVisualizer:
    """Generate visualizations for optimization results."""

    def __init__(self):
        """Initialize visualizer."""
        self.color_scheme = {
            'primary': '#1f77b4',
            'secondary': '#ff7f0e',
            'highlight': '#2ca02c',
            'alert': '#d62728'
        }

    def plot_optimization_history(
            self,
            results: Dict[str, Any],
            show_running_best: bool = True
    ) -> go.Figure:
        """Plot optimization history."""
        try:
            # Extract history data
            history = pd.DataFrame(results['results_history'] if 'results_history' in results
                                  else results.get('optimization_history', []))

            fig = make_subplots(
                rows=2, cols=1,
                subplot_titles=(
                    'Optimization Progress',
                    'Parameter Evolution'
                ),
                vertical_spacing=0.15
            )

            # Score evolution
            fig.add_trace(
                go.Scatter(
                    x=history.index if hasattr(history, 'index') and not isinstance(history.index, pd.RangeIndex)
                      else list(range(len(history))),
                    y=history['score'],
                    mode='markers',
                    name='Trial Scores',
                    marker=dict(color=self.color_scheme['primary'])
                ),
                row=1, col=1
            )

            if show_running_best:
                running_best = history['score'].cummax()
                fig.add_trace(
                    go.Scatter(
                        x=history.index if hasattr(history, 'index') and not isinstance(history.index, pd.RangeIndex)
                          else list(range(len(history))),
                        y=running_best,
                        mode='lines',
                        name='Best Score',
                        line=dict(color=self.color_scheme['highlight'])
                    ),
                    row=1, col=1
                )

            # Parameter evolution
            if 'parameters' in history.columns:
                param_data = pd.DataFrame(history['parameters'].tolist())
                for param in param_data.columns:
                    fig.add_trace(
                        go.Scatter(
                            x=history.index if hasattr(history, 'index') and not isinstance(history.index, pd.RangeIndex)
                              else list(range(len(history))),
                            y=param_data[param],
                            mode='markers',
                            name=param,
                            opacity=0.6
                        ),
                        row=2, col=1
                    )
            elif 'params' in history.columns:
                param_data = pd.DataFrame(history['params'].tolist())
                for param in param_data.columns:
                    fig.add_trace(
                        go.Scatter(
                            x=history.index if hasattr(history, 'index') and not isinstance(history.index, pd.RangeIndex)
                              else list(range(len(history))),
                            y=param_data[param],
                            mode='markers',
                            name=param,
                            opacity=0.6
                        ),
                        row=2, col=1
                    )

            fig.update_layout(
                height=800,
                title_text='Optimization History',
                showlegend=True,
                template='plotly_white'
            )

            fig.update_xaxes(title_text="Trial", row=2, col=1)
            fig.update_yaxes(title_text="Score", row=1, col=1)
            fig.update_yaxes(title_text="Parameter Value", row=2, col=1)

            return fig

        except Exception as e:
            logging.error(f"Error plotting optimization history: {str(e)}")
            return go.Figure()

    def plot_parameter_analysis(self, results: Dict[str, Any]) -> go.Figure:
        """Plot parameter analysis."""
        try:
            # Extract history data - handle both formats
            history = pd.DataFrame(results['results_history'] if 'results_history' in results
                                  else results.get('optimization_history', []))

            # Extract parameter data with compatibility for different formats
            if 'parameters' in history.columns:
                param_data = pd.DataFrame(history['parameters'].tolist())
            elif 'params' in history.columns:
                param_data = pd.DataFrame(history['params'].tolist())
            else:
                # Fall back to best parameters if no history
                param_data = pd.DataFrame([results.get('best_parameters', {})])

            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=(
                    'Parameter Importance',
                    'Parameter Correlations',
                    'Parameter Distributions',
                    'Score vs Parameters'
                ),
                vertical_spacing=0.15
            )

            # Parameter importance
            importance = results.get('parameter_importance', {})
            if importance:
                fig.add_trace(
                    go.Bar(
                        x=list(importance.keys()),
                        y=list(importance.values()),
                        name='Importance',
                        marker=dict(color=self.color_scheme['primary'])
                    ),
                    row=1, col=1
                )

            # Parameter correlations
            corr_matrix = param_data.corr()
            fig.add_trace(
                go.Heatmap(
                    z=corr_matrix.values,
                    x=corr_matrix.columns,
                    y=corr_matrix.columns,
                    colorscale='RdBu',
                    name='Correlations'
                ),
                row=1, col=2
            )

            # Parameter distributions
            for param in param_data.columns:
                fig.add_trace(
                    go.Histogram(
                        x=param_data[param],
                        name=param,
                        opacity=0.7
                    ),
                    row=2, col=1
                )

            # Score vs parameters
            for param in param_data.columns:
                fig.add_trace(
                    go.Scatter(
                        x=param_data[param],
                        y=history['score'],
                        mode='markers',
                        name=f'Score vs {param}',
                        opacity=0.6
                    ),
                    row=2, col=2
                )

            fig.update_layout(
                height=1000,
                title_text='Parameter Analysis',
                showlegend=True,
                template='plotly_white'
            )

            return fig

        except Exception as e:
            logging.error(f"Error plotting parameter analysis: {str(e)}")
            return go.Figure()

    def plot_performance_analysis(self, results: Dict[str, Any]) -> go.Figure:
        """Plot performance analysis."""
        try:
            # Check for equity curve - handle both formats
            if 'equity_curve' in results:
                equity_curve = results['equity_curve']
            elif 'metrics' in results and 'equity_curve' in results['metrics']:
                equity_curve = results['metrics']['equity_curve']
            elif 'best_result' in results and 'equity_curve' in results['best_result']:
                equity_curve = results['best_result']['equity_curve']
            else:
                logging.warning("No equity curve found in results")
                return go.Figure()

            # Handle different types of equity curve data
            if isinstance(equity_curve, dict):
                # Convert dictionary to series
                equity_curve = pd.Series(equity_curve)
            elif isinstance(equity_curve, pd.DataFrame):
                # Take first column if dataframe
                equity_curve = equity_curve.iloc[:, 0]

            # Calculate returns
            returns = equity_curve.pct_change().dropna()

            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=(
                    'Equity Curve',
                    'Returns Distribution',
                    'Drawdown Analysis',
                    'Rolling Metrics'
                ),
                vertical_spacing=0.15
            )

            # Equity curve
            fig.add_trace(
                go.Scatter(
                    x=equity_curve.index,
                    y=equity_curve,
                    name='Portfolio Value',
                    line=dict(color=self.color_scheme['primary'])
                ),
                row=1, col=1
            )

            # Returns distribution
            fig.add_trace(
                go.Histogram(
                    x=returns,
                    name='Returns Distribution',
                    nbinsx=50,
                    histnorm='probability',
                    marker=dict(color=self.color_scheme['secondary'])
                ),
                row=1, col=2
            )

            # Add normal distribution overlay
            x_range = np.linspace(returns.min(), returns.max(), 100)
            fig.add_trace(
                go.Scatter(
                    x=x_range,
                    y=np.exp(-(x_range - returns.mean()) ** 2 / (2 * returns.std() ** 2)) / (
                                returns.std() * np.sqrt(2 * np.pi)),
                    name='Normal Distribution',
                    line=dict(color=self.color_scheme['highlight'])
                ),
                row=1, col=2
            )

            # Drawdown analysis
            rolling_max = equity_curve.expanding().max()
            drawdowns = (equity_curve - rolling_max) / rolling_max

            fig.add_trace(
                go.Scatter(
                    x=drawdowns.index,
                    y=drawdowns * 100,  # Convert to percentage
                    name='Drawdown',
                    fill='tozeroy',
                    line=dict(color=self.color_scheme['alert'])
                ),
                row=2, col=1
            )

            # Rolling metrics
            window = min(252, len(returns) // 4)  # Use quarterly window if data less than 1 year

            # Rolling Sharpe ratio
            rolling_sharpe = (
                                     returns.rolling(window=window).mean() /
                                     returns.rolling(window=window).std()
                             ) * np.sqrt(252)

            fig.add_trace(
                go.Scatter(
                    x=rolling_sharpe.index,
                    y=rolling_sharpe,
                    name='Rolling Sharpe',
                    line=dict(color=self.color_scheme['primary'])
                ),
                row=2, col=2
            )

            # Rolling volatility
            rolling_vol = returns.rolling(window=window).std() * np.sqrt(252)

            fig.add_trace(
                go.Scatter(
                    x=rolling_vol.index,
                    y=rolling_vol,
                    name='Rolling Volatility',
                    line=dict(color=self.color_scheme['secondary'])
                ),
                row=2, col=2
            )

            # Update layout
            fig.update_layout(
                height=1000,
                title_text='Performance Analysis',
                showlegend=True,
                template='plotly_white'
            )

            # Update axes
            fig.update_xaxes(title_text="Date", row=1, col=1)
            fig.update_xaxes(title_text="Return", row=1, col=2)
            fig.update_xaxes(title_text="Date", row=2, col=1)
            fig.update_xaxes(title_text="Date", row=2, col=2)

            fig.update_yaxes(title_text="Portfolio Value", row=1, col=1)
            fig.update_yaxes(title_text="Probability", row=1, col=2)
            fig.update_yaxes(title_text="Drawdown %", row=2, col=1)
            fig.update_yaxes(title_text="Value", row=2, col=2)

            return fig

        except Exception as e:
            logging.error(f"Error plotting performance analysis: {str(e)}")
            return go.Figure()

    def plot_correlation_network(
            self,
            results: Dict[str, Any],
            threshold: float = 0.5
    ) -> go.Figure:
        """Plot parameter correlation network."""
        try:
            # Extract history data - handle both formats
            history = pd.DataFrame(results['results_history'] if 'results_history' in results
                                  else results.get('optimization_history', []))

            # Extract parameter data
            if 'parameters' in history.columns:
                param_data = pd.DataFrame(history['parameters'].tolist())
            elif 'params' in history.columns:
                param_data = pd.DataFrame(history['params'].tolist())
            else:
                # Fall back to best parameters if no history
                param_data = pd.DataFrame([results.get('best_parameters', {})])

            # Calculate correlations
            corr_matrix = param_data.corr().abs()

            # Create network layout
            import networkx as nx
            G = nx.Graph()

            # Add nodes
            for param in param_data.columns:
                G.add_node(param)

            # Add edges for correlations above threshold
            for i in range(len(corr_matrix)):
                for j in range(i + 1, len(corr_matrix)):
                    if corr_matrix.iloc[i, j] >= threshold:
                        G.add_edge(
                            corr_matrix.index[i],
                            corr_matrix.columns[j],
                            weight=corr_matrix.iloc[i, j]
                        )

            # Generate layout
            pos = nx.spring_layout(G)

            # Create figure
            edge_x = []
            edge_y = []
            edge_weights = []

            for edge in G.edges(data=True):
                x0, y0 = pos[edge[0]]
                x1, y1 = pos[edge[1]]
                edge_x.extend([x0, x1, None])
                edge_y.extend([y0, y1, None])
                edge_weights.append(edge[2]['weight'])

            node_x = []
            node_y = []
            node_text = []

            for node in G.nodes():
                x, y = pos[node]
                node_x.append(x)
                node_y.append(y)
                node_text.append(node)

            # Create plot
            fig = go.Figure()

            # Add edges
            fig.add_trace(
                go.Scatter(
                    x=edge_x,
                    y=edge_y,
                    mode='lines',
                    line=dict(
                        width=1,
                        color='#888'
                    ),
                    hoverinfo='none'
                )
            )

            # Add nodes
            fig.add_trace(
                go.Scatter(
                    x=node_x,
                    y=node_y,
                    mode='markers+text',
                    marker=dict(
                        size=30,
                        color=self.color_scheme['primary']
                    ),
                    text=node_text,
                    textposition='middle center',
                    hoverinfo='text'
                )
            )

            fig.update_layout(
                title='Parameter Correlation Network',
                showlegend=False,
                hovermode='closest',
                margin=dict(b=20, l=5, r=5, t=40),
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                template='plotly_white'
            )

            return fig

        except Exception as e:
            logging.error(f"Error plotting correlation network: {str(e)}")
            return go.Figure()

    def plot_multi_pair_dashboard(self, system) -> go.Figure:
        """
        Create a dashboard for MultiPairTradingSystem results.

        Args:
            system: The optimized MultiPairTradingSystem instance

        Returns:
            Plotly figure with dashboard
        """
        try:
            if not hasattr(system, 'portfolio_history') or not system.portfolio_history:
                logging.warning("No portfolio history found in system")
                return go.Figure()

            # Convert portfolio history to DataFrame
            portfolio_df = pd.DataFrame(system.portfolio_history)
            portfolio_df.set_index('date', inplace=True)

            # Create subplots for dashboard
            fig = make_subplots(
                rows=3, cols=2,
                subplot_titles=(
                    'Portfolio Value',
                    'Active Pairs',
                    'Drawdown',
                    'Pair Performance',
                    'Daily Returns',
                    'Capital Allocation'
                ),
                vertical_spacing=0.1
            )

            # 1. Portfolio Value
            fig.add_trace(
                go.Scatter(
                    x=portfolio_df.index,
                    y=portfolio_df['portfolio_value'],
                    name='Portfolio Value',
                    line=dict(color=self.color_scheme['primary'])
                ),
                row=1, col=1
            )

            # 2. Active Pairs
            fig.add_trace(
                go.Scatter(
                    x=portfolio_df.index,
                    y=portfolio_df['active_pairs'],
                    name='Active Pairs',
                    line=dict(color=self.color_scheme['secondary'])
                ),
                row=1, col=2
            )

            # 3. Drawdown
            peak = portfolio_df['portfolio_value'].cummax()
            drawdown = (portfolio_df['portfolio_value'] - peak) / peak * 100

            fig.add_trace(
                go.Scatter(
                    x=drawdown.index,
                    y=drawdown,
                    name='Drawdown',
                    fill='tozeroy',
                    line=dict(color=self.color_scheme['alert'])
                ),
                row=2, col=1
            )

            # 4. Pair Performance
            pair_performance = []
            for pair, model in system.pair_models.items():
                if hasattr(model, 'portfolio_history') and model.portfolio_history:
                    initial = model.initial_capital
                    final = model.portfolio_history[-1]['portfolio_value']
                    pair_performance.append({
                        'pair': f"{pair[0]}-{pair[1]}",
                        'return': (final / initial - 1) * 100
                    })

            if pair_performance:
                pair_df = pd.DataFrame(pair_performance)
                fig.add_trace(
                    go.Bar(
                        x=pair_df['pair'],
                        y=pair_df['return'],
                        name='Pair Returns (%)',
                        marker_color=self.color_scheme['highlight']
                    ),
                    row=2, col=2
                )

            # 5. Daily Returns
            returns = portfolio_df['portfolio_value'].pct_change() * 100
            fig.add_trace(
                go.Scatter(
                    x=returns.index,
                    y=returns,
                    name='Daily Returns (%)',
                    line=dict(color='green')
                ),
                row=3, col=1
            )

            # 6. Capital Allocation
            capital_data = []
            for pair, model in system.pair_models.items():
                if hasattr(model, 'current_capital'):
                    capital_data.append({
                        'pair': f"{pair[0]}-{pair[1]}",
                        'capital': model.current_capital
                    })

            if capital_data:
                capital_df = pd.DataFrame(capital_data)
                fig.add_trace(
                    go.Pie(
                        labels=capital_df['pair'],
                        values=capital_df['capital'],
                        name='Capital Allocation'
                    ),
                    row=3, col=2
                )

            # Update layout
            fig.update_layout(
                height=1200,
                showlegend=True,
                title_text='Multi-Pair Trading System Dashboard',
                template='plotly_white'
            )

            return fig

        except Exception as e:
            logging.error(f"Error creating multi-pair dashboard: {str(e)}")
            return go.Figure()

    def create_optimization_report(
            self,
            results: Dict[str, Any],
            output_path: Optional[str] = None
    ) -> Optional[str]:
        """Create comprehensive HTML report of optimization results."""
        try:
            from jinja2 import Template

            # Load template
            template_str = """
            <html>
                <head>
                    <title>Optimization Results Report</title>
                    <style>
                        body { font-family: Arial, sans-serif; margin: 40px; }
                        .section { margin-bottom: 30px; }
                        .metric { display: inline-block; margin: 10px; padding: 15px; border: 1px solid #ddd; }
                    </style>
                </head>
                <body>
                    <h1>Optimization Results Report</h1>

                    <div class="section">
                        <h2>Best Parameters</h2>
                        <pre>{{ best_parameters | tojson(indent=2) }}</pre>
                    </div>

                    <div class="section">
                        <h2>Performance Metrics</h2>
                        {% for metric, value in metrics.items() %}
                        <div class="metric">
                            <h3>{{ metric }}</h3>
                            <p>{{ "%.4f"|format(value) }}</p>
                        </div>
                        {% endfor %}
                    </div>

                    <div class="section">
                        <h2>Optimization Progress</h2>
                        {{ optimization_plot }}
                    </div>

                    <div class="section">
                        <h2>Parameter Analysis</h2>
                        {{ parameter_plot }}
                    </div>

                    <div class="section">
                        <h2>Performance Analysis</h2>
                        {{ performance_plot }}
                    </div>
                </body>
            </html>
            """

            # Generate plots
            opt_plot = self.plot_optimization_history(results)
            param_plot = self.plot_parameter_analysis(results)
            perf_plot = self.plot_performance_analysis(results)

            # Prepare template data
            template_data = {
                'best_parameters': results.get('best_parameters', {}),
                'metrics': results.get('metrics', {}),
                'optimization_plot': opt_plot.to_html(full_html=False),
                'parameter_plot': param_plot.to_html(full_html=False),
                'performance_plot': perf_plot.to_html(full_html=False)
            }

            # Render template
            template = Template(template_str)
            html_report = template.render(**template_data)

            # Save if path provided
            if output_path:
                with open(output_path, 'w') as f:
                    f.write(html_report)
                return output_path

            return html_report

        except Exception as e:
            logging.error(f"Error creating optimization report: {str(e)}")
            return None

def plot_strategy_comparison(
        optimized_equity: pd.Series,
        baseline_equity: pd.Series,
        title: str = "Strategy Comparison"
) -> go.Figure:
    """
    Plot comparison between optimized strategy and baseline strategy.

    Args:
        optimized_equity: Equity curve of optimized strategy
        baseline_equity: Equity curve of baseline strategy
        title: Plot title

    Returns:
        Plotly figure with comparison visualization
    """
    # Ensure the same timeframe
    common_dates = optimized_equity.index.intersection(baseline_equity.index)

    if common_dates.empty:
        # No overlapping dates, return empty figure
        fig = go.Figure()
        fig.update_layout(title="No overlapping dates between strategies")
        return fig

    # Normalize to start from the same point for fair comparison
    opt_normalized = optimized_equity[common_dates] / optimized_equity[common_dates].iloc[0]
    baseline_normalized = baseline_equity[common_dates] / baseline_equity[common_dates].iloc[0]

    # Calculate cumulative returns
    opt_cum_return = opt_normalized - 1
    baseline_cum_return = baseline_normalized - 1

    # Calculate outperformance
    outperformance = opt_normalized - baseline_normalized

    # Calculate rolling metrics (90-day window)
    window = min(90, len(common_dates) // 2)
    if window >= 5:  # Ensure enough data for rolling window
        opt_rolling_vol = opt_normalized.pct_change().rolling(window=window).std() * np.sqrt(252)
        baseline_rolling_vol = baseline_normalized.pct_change().rolling(window=window).std() * np.sqrt(252)

        # Rolling Sharpe (using 0 as risk-free rate for simplicity)
        opt_rolling_sharpe = (opt_normalized.pct_change().rolling(window=window).mean() * 252) / \
                             (opt_normalized.pct_change().rolling(window=window).std() * np.sqrt(252))
        baseline_rolling_sharpe = (baseline_normalized.pct_change().rolling(window=window).mean() * 252) / \
                                  (baseline_normalized.pct_change().rolling(window=window).std() * np.sqrt(252))
    else:
        opt_rolling_vol = pd.Series(index=common_dates)
        baseline_rolling_vol = pd.Series(index=common_dates)
        opt_rolling_sharpe = pd.Series(index=common_dates)
        baseline_rolling_sharpe = pd.Series(index=common_dates)

    # Create figure with subplots
    fig = make_subplots(
        rows=3, cols=2,
        subplot_titles=(
            "Normalized Equity Curves",
            "Cumulative Returns",
            "Outperformance",
            "Drawdowns",
            "Rolling Volatility (90-day)",
            "Rolling Sharpe Ratio (90-day)"
        ),
        vertical_spacing=0.1,
        shared_xaxes=True
    )

    # 1. Normalized equity curves
    fig.add_trace(
        go.Scatter(
            x=common_dates,
            y=opt_normalized.values,
            mode='lines',
            name='Optimized Strategy',
            line=dict(color='blue')
        ),
        row=1, col=1
    )

    fig.add_trace(
        go.Scatter(
            x=common_dates,
            y=baseline_normalized.values,
            mode='lines',
            name='Baseline Strategy',
            line=dict(color='red', dash='dash')
        ),
        row=1, col=1
    )

    # 2. Cumulative returns
    fig.add_trace(
        go.Scatter(
            x=common_dates,
            y=opt_cum_return.values * 100,  # Convert to percentage
            mode='lines',
            name='Optimized Returns',
            line=dict(color='blue')
        ),
        row=1, col=2
    )

    fig.add_trace(
        go.Scatter(
            x=common_dates,
            y=baseline_cum_return.values * 100,  # Convert to percentage
            mode='lines',
            name='Baseline Returns',
            line=dict(color='red', dash='dash')
        ),
        row=1, col=2
    )

    # 3. Outperformance
    fig.add_trace(
        go.Scatter(
            x=common_dates,
            y=outperformance.values,
            mode='lines',
            name='Outperformance',
            line=dict(color='green'),
            fill='tozeroy'
        ),
        row=2, col=1
    )

    # Add zero line for reference
    fig.add_trace(
        go.Scatter(
            x=[common_dates[0], common_dates[-1]],
            y=[0, 0],
            mode='lines',
            name='Zero',
            line=dict(color='black', dash='dash', width=1),
            showlegend=False
        ),
        row=2, col=1
    )

    # 4. Drawdowns
    # Calculate drawdowns
    opt_peak = opt_normalized.cummax()
    opt_drawdown = (opt_normalized - opt_peak) / opt_peak * 100

    baseline_peak = baseline_normalized.cummax()
    baseline_drawdown = (baseline_normalized - baseline_peak) / baseline_peak * 100

    fig.add_trace(
        go.Scatter(
            x=common_dates,
            y=opt_drawdown.values,
            mode='lines',
            name='Optimized Drawdown',
            line=dict(color='blue')
        ),
        row=2, col=2
    )

    fig.add_trace(
        go.Scatter(
            x=common_dates,
            y=baseline_drawdown.values,
            mode='lines',
            name='Baseline Drawdown',
            line=dict(color='red', dash='dash')
        ),
        row=2, col=2
    )

    # 5. Rolling volatility
    if not opt_rolling_vol.empty and not baseline_rolling_vol.empty:
        fig.add_trace(
            go.Scatter(
                x=opt_rolling_vol.index,
                y=opt_rolling_vol.values * 100,  # Convert to percentage
                mode='lines',
                name='Optimized Volatility',
                line=dict(color='blue')
            ),
            row=3, col=1
        )

        fig.add_trace(
            go.Scatter(
                x=baseline_rolling_vol.index,
                y=baseline_rolling_vol.values * 100,  # Convert to percentage
                mode='lines',
                name='Baseline Volatility',
                line=dict(color='red', dash='dash')
            ),
            row=3, col=1
        )

    # 6. Rolling Sharpe ratio
    if not opt_rolling_sharpe.empty and not baseline_rolling_sharpe.empty:
        fig.add_trace(
            go.Scatter(
                x=opt_rolling_sharpe.index,
                y=opt_rolling_sharpe.values,
                mode='lines',
                name='Optimized Sharpe',
                line=dict(color='blue')
            ),
            row=3, col=2
        )

        fig.add_trace(
            go.Scatter(
                x=baseline_rolling_sharpe.index,
                y=baseline_rolling_sharpe.values,
                mode='lines',
                name='Baseline Sharpe',
                line=dict(color='red', dash='dash')
            ),
            row=3, col=2
        )

    # Update layout
    fig.update_layout(
        height=900,
        title=title,
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )

    # Update axes titles
    fig.update_yaxes(title_text="Normalized Value", row=1, col=1)
    fig.update_yaxes(title_text="Return (%)", row=1, col=2)
    fig.update_yaxes(title_text="Difference", row=2, col=1)
    fig.update_yaxes(title_text="Drawdown (%)", row=2, col=2)
    fig.update_yaxes(title_text="Volatility (%)", row=3, col=1)
    fig.update_yaxes(title_text="Sharpe Ratio", row=3, col=2)

    return fig

def create_performance_summary_table(
        optimized_metrics: Dict[str, Any],
        baseline_metrics: Dict[str, Any]
) -> pd.DataFrame:
    """
    Create a summary table comparing optimized strategy against baseline.

    Args:
        optimized_metrics: Performance metrics for optimized strategy
        baseline_metrics: Performance metrics for baseline strategy

    Returns:
        DataFrame with comparison metrics
    """
    # Define metrics to compare
    metric_mapping = {
        'Total Return (%)': ['total_return', 'Total Return (%)'],
        'Sharpe Ratio': ['sharpe_ratio', 'Sharpe Ratio'],
        'Max Drawdown (%)': ['max_drawdown', 'Max Drawdown (%)'],
        'Annual Volatility (%)': ['annual_vol', 'Annual Volatility (%)'],
        'Win Rate (%)': ['win_rate', 'Win Rate (%)'],
        'Profit Factor': ['profit_factor', 'Profit Factor'],
        'Recovery Factor': ['recovery_factor', 'Recovery Factor']
    }

    # Get values for each metric
    data = []

    for display_name, keys in metric_mapping.items():
        opt_key, base_key = keys

        # Try to get optimized metric value
        if isinstance(optimized_metrics, dict):
            opt_value = optimized_metrics.get(opt_key, None)
            if opt_value is None and 'Portfolio Metrics' in optimized_metrics:
                opt_value = optimized_metrics['Portfolio Metrics'].get(base_key, None)
        else:
            opt_value = None

        # Try to get baseline metric value
        if isinstance(baseline_metrics, dict):
            base_value = baseline_metrics.get(opt_key, None)
            if base_value is None and 'Portfolio Metrics' in baseline_metrics:
                base_value = baseline_metrics['Portfolio Metrics'].get(base_key, None)
        else:
            base_value = None

        # Calculate difference if both values exist
        if opt_value is not None and base_value is not None:
            difference = opt_value - base_value

            # For drawdown and volatility, lower is better, so reverse the sign
            if 'Drawdown' in display_name or 'Volatility' in display_name:
                difference = -difference

            # Calculate percentage improvement
            if base_value != 0:
                pct_improvement = (difference / abs(base_value)) * 100
            else:
                pct_improvement = float('inf') if difference > 0 else float('-inf') if difference < 0 else 0

            data.append({
                'Metric': display_name,
                'Optimized Strategy': opt_value,
                'Baseline Strategy': base_value,
                'Difference': difference,
                'Improvement (%)': pct_improvement
            })

    # Filter out rows where we don't have both values
    data = [row for row in data if row['Optimized Strategy'] is not None and row['Baseline Strategy'] is not None]

    if not data:
        # Return empty DataFrame with correct columns
        return pd.DataFrame(
            columns=['Metric', 'Optimized Strategy', 'Baseline Strategy', 'Difference', 'Improvement (%)'])

    return pd.DataFrame(data)

def plot_trade_comparison(
        optimized_trades: List[Dict],
        baseline_trades: List[Dict]
) -> go.Figure:
    """
    Create visualization comparing trade characteristics between strategies.

    Args:
        optimized_trades: List of trade dictionaries from optimized strategy
        baseline_trades: List of trade dictionaries from baseline strategy

    Returns:
        Plotly figure comparing trade characteristics
    """
    # Create DataFrames from trade lists
    if optimized_trades:
        opt_df = pd.DataFrame(optimized_trades)
    else:
        opt_df = pd.DataFrame(columns=['date', 'pnl', 'type', 'duration'])

    if baseline_trades:
        base_df = pd.DataFrame(baseline_trades)
    else:
        base_df = pd.DataFrame(columns=['date', 'pnl', 'type', 'duration'])

    # Convert dates if needed
    for df in [opt_df, base_df]:
        if 'date' in df.columns and not pd.api.types.is_datetime64_dtype(df['date']):
            df['date'] = pd.to_datetime(df['date'])

    # Calculate trade statistics
    def calculate_stats(df):
        if df.empty:
            return {
                'count': 0,
                'win_count': 0,
                'loss_count': 0,
                'win_rate': 0,
                'avg_win': 0,
                'avg_loss': 0,
                'avg_pnl': 0,
                'profit_factor': 0
            }

        # Add default pnl column if it doesn't exist
        if 'pnl' not in df.columns:
            df['pnl'] = 0

        win_trades = df[df['pnl'] > 0]
        loss_trades = df[df['pnl'] < 0]

        win_count = len(win_trades)
        loss_count = len(loss_trades)
        total_count = len(df)

        return {
            'count': total_count,
            'win_count': win_count,
            'loss_count': loss_count,
            'win_rate': win_count / total_count if total_count > 0 else 0,
            'avg_win': win_trades['pnl'].mean() if not win_trades.empty else 0,
            'avg_loss': loss_trades['pnl'].mean() if not loss_trades.empty else 0,
            'avg_pnl': df['pnl'].mean() if not df.empty else 0,
            'profit_factor': abs(win_trades['pnl'].sum() / loss_trades['pnl'].sum())
            if not loss_trades.empty and loss_trades['pnl'].sum() != 0 else 0
        }

    opt_stats = calculate_stats(opt_df)
    base_stats = calculate_stats(base_df)

    # Create figure
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            "Trade Count Comparison",
            "Win Rate Comparison",
            "Average P&L Comparison",
            "Trade Distribution"
        ),
        specs=[
            [{"type": "bar"}, {"type": "bar"}],
            [{"type": "bar"}, {"type": "histogram"}]
        ]
    )

    # 1. Trade Count
    counts = {
        'Strategy': ['Optimized', 'Baseline'],
        'Trade Count': [opt_stats['count'], base_stats['count']],
        'Winning Trades': [opt_stats['win_count'], base_stats['win_count']],
        'Losing Trades': [opt_stats['loss_count'], base_stats['loss_count']]
    }

    fig.add_trace(
        go.Bar(
            x=counts['Strategy'],
            y=counts['Trade Count'],
            name='Total Trades',
            marker_color='blue'
        ),
        row=1, col=1
    )

    fig.add_trace(
        go.Bar(
            x=counts['Strategy'],
            y=counts['Winning Trades'],
            name='Winning Trades',
            marker_color='green'
        ),
        row=1, col=1
    )

    fig.add_trace(
        go.Bar(
            x=counts['Strategy'],
            y=counts['Losing Trades'],
            name='Losing Trades',
            marker_color='red'
        ),
        row=1, col=1
    )

    # 2. Win Rate
    win_rates = {
        'Strategy': ['Optimized', 'Baseline'],
        'Win Rate': [opt_stats['win_rate'] * 100, base_stats['win_rate'] * 100]
    }

    fig.add_trace(
        go.Bar(
            x=win_rates['Strategy'],
            y=win_rates['Win Rate'],
            name='Win Rate (%)',
            marker_color=['blue', 'red']
        ),
        row=1, col=2
    )

    # 3. Average P&L
    pnl_comparison = {
        'Strategy': ['Optimized', 'Baseline'],
        'Avg P&L': [opt_stats['avg_pnl'], base_stats['avg_pnl']],
        'Avg Win': [opt_stats['avg_win'], base_stats['avg_win']],
        'Avg Loss': [opt_stats['avg_loss'], base_stats['avg_loss']]
    }

    fig.add_trace(
        go.Bar(
            x=pnl_comparison['Strategy'],
            y=pnl_comparison['Avg P&L'],
            name='Average P&L',
            marker_color='purple'
        ),
        row=2, col=1
    )

    fig.add_trace(
        go.Bar(
            x=pnl_comparison['Strategy'],
            y=pnl_comparison['Avg Win'],
            name='Average Win',
            marker_color='green'
        ),
        row=2, col=1
    )

    fig.add_trace(
        go.Bar(
            x=pnl_comparison['Strategy'],
            y=pnl_comparison['Avg Loss'],
            name='Average Loss',
            marker_color='red'
        ),
        row=2, col=1
    )

    # 4. P&L Distribution
    if not opt_df.empty and 'pnl' in opt_df.columns:
        fig.add_trace(
            go.Histogram(
                x=opt_df['pnl'],
                name='Optimized P&L',
                marker_color='blue',
                opacity=0.6,
                nbinsx=20
            ),
            row=2, col=2
        )

    if not base_df.empty and 'pnl' in base_df.columns:
        fig.add_trace(
            go.Histogram(
                x=base_df['pnl'],
                name='Baseline P&L',
                marker_color='red',
                opacity=0.6,
                nbinsx=20
            ),
            row=2, col=2
        )

    # Update layout
    fig.update_layout(
        height=800,
        title="Trade Comparison: Optimized vs Baseline",
        barmode='group',
        showlegend=True
    )

    return fig