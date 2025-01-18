"""
Visualization components for optimization results.
"""

from typing import Dict, Any, Optional
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
            history = pd.DataFrame(results['results_history'])

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
                    x=history.index,
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
                        x=history.index,
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
                            x=history.index,
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
            history = pd.DataFrame(results['results_history'])
            param_data = pd.DataFrame(history['parameters'].tolist())

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
            if 'equity_curve' not in results:
                return go.Figure()

            equity_curve = results['equity_curve']
            returns = equity_curve.pct_change()

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
            history = pd.DataFrame(results['results_history'])
            param_data = pd.DataFrame(history['parameters'].tolist())

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
                'best_parameters': results['best_parameters'],
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