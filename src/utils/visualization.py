"""
Enhanced Visualization Module (Plotly-based)

Features:
1. Figure return options
2. File saving capabilities
3. Complex plot combinations
4. Customization options
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from typing import Dict, List, Union, Optional, Tuple
from pathlib import Path
from config.logging_config import logger

class PlotlyVisualizer:
    """Class for creating and managing Plotly visualizations."""

    def __init__(self, theme: str = 'plotly_white'):
        """
        Initialize visualizer.

        Args:
            theme: Plotly template theme
        """
        self.theme = theme

    def plot_equity_curve(self,
                         equity_curve: pd.Series,
                         title: str = 'Equity Curve',
                         include_drawdown: bool = True,
                         show_fig: bool = True,
                         output_file: Optional[str] = None) -> go.Figure:
        """
        Plot equity curve with optional drawdown.

        Args:
            equity_curve: Portfolio value time series
            title: Chart title
            include_drawdown: Whether to include drawdown subplot
            show_fig: Whether to display the figure
            output_file: Optional file path to save figure

        Returns:
            Plotly Figure object
        """
        if equity_curve.empty:
            logger.warning("Empty equity curve")
            return go.Figure()

        # Create figure with secondary y-axis if including drawdown
        fig = make_subplots(
            rows=2 if include_drawdown else 1,
            cols=1,
            subplot_titles=['Portfolio Value', 'Drawdown'] if include_drawdown
                          else [title],
            vertical_spacing=0.12
        )

        # Plot equity curve
        fig.add_trace(
            go.Scatter(
                x=equity_curve.index,
                y=equity_curve.values,
                mode='lines',
                name='Portfolio Value',
                line=dict(color='blue')
            ),
            row=1, col=1
        )

        # Add drawdown subplot
        if include_drawdown:
            drawdown = (equity_curve - equity_curve.cummax()) / equity_curve.cummax()
            fig.add_trace(
                go.Scatter(
                    x=drawdown.index,
                    y=drawdown.values,
                    mode='lines',
                    name='Drawdown',
                    line=dict(color='red')
                ),
                row=2, col=1
            )

        # Update layout
        fig.update_layout(
            title=title,
            template=self.theme,
            height=800 if include_drawdown else 500,
            showlegend=True
        )

        # Update axes titles
        fig.update_yaxes(title_text="Portfolio Value", row=1, col=1)
        if include_drawdown:
            fig.update_yaxes(title_text="Drawdown", row=2, col=1)

        self._handle_figure_output(fig, show_fig, output_file)
        return fig

    def plot_performance_metrics(self,
                               metrics: Dict[str, float],
                               title: str = 'Performance Metrics',
                               show_fig: bool = True,
                               output_file: Optional[str] = None) -> go.Figure:
        """Plot performance metrics as horizontal bar chart."""
        if not metrics:
            logger.warning("No metrics to plot")
            return go.Figure()

        # Convert to DataFrame and sort
        df_metrics = pd.DataFrame(
            list(metrics.items()),
            columns=['Metric', 'Value']
        ).sort_values('Value', ascending=True)

        # Create horizontal bar chart
        fig = go.Figure(
            go.Bar(
                x=df_metrics['Value'],
                y=df_metrics['Metric'],
                orientation='h',
                text=df_metrics['Value'].round(4),
                textposition='auto'
            )
        )

        fig.update_layout(
            title=title,
            template=self.theme,
            height=max(400, len(metrics) * 30),  # Dynamic height
            yaxis_title='Metrics',
            xaxis_title='Values',
            margin=dict(l=200)  # More space for metric names
        )

        self._handle_figure_output(fig, show_fig, output_file)
        return fig

    def plot_trade_analysis(self,
                          trades: pd.DataFrame,
                          title: str = 'Trade Analysis',
                          show_fig: bool = True,
                          output_file: Optional[str] = None) -> go.Figure:
        """
        Plot comprehensive trade analysis.

        Args:
            trades: DataFrame with trade history
            title: Chart title
            show_fig: Whether to display figure
            output_file: Optional file path to save figure
        """
        if trades.empty:
            logger.warning("No trades to analyze")
            return go.Figure()

        # Create subplots
        fig = make_subplots(
            rows=2,
            cols=2,
            subplot_titles=[
                'Trade PnL Distribution',
                'Cumulative PnL',
                'Trade Durations',
                'Win/Loss Ratio by Pair'
            ]
        )

        # 1. PnL Distribution
        pnl_hist = go.Histogram(
            x=trades['PnL'],
            name='PnL Distribution',
            nbinsx=30
        )
        fig.add_trace(pnl_hist, row=1, col=1)

        # 2. Cumulative PnL
        cum_pnl = trades['PnL'].cumsum()
        fig.add_trace(
            go.Scatter(
                x=trades.index,
                y=cum_pnl,
                name='Cumulative PnL'
            ),
            row=1, col=2
        )

        # 3. Trade Durations
        if 'Duration' in trades.columns:
            duration_hist = go.Histogram(
                x=trades['Duration'],
                name='Trade Durations'
            )
            fig.add_trace(duration_hist, row=2, col=1)

        # 4. Win/Loss by Pair
        if 'Pair' in trades.columns:
            win_rates = trades.groupby('Pair')['PnL'].apply(
                lambda x: (x > 0).mean()
            ).sort_values()

            fig.add_trace(
                go.Bar(
                    x=win_rates.index,
                    y=win_rates.values,
                    name='Win Rate'
                ),
                row=2, col=2
            )

        # Update layout
        fig.update_layout(
            title=title,
            template=self.theme,
            height=800,
            showlegend=True
        )

        self._handle_figure_output(fig, show_fig, output_file)
        return fig

    def plot_pair_correlations(self,
                             returns: pd.DataFrame,
                             title: str = 'Pair Correlations',
                             show_fig: bool = True,
                             output_file: Optional[str] = None) -> go.Figure:
        """Plot correlation matrix for pairs."""
        if returns.empty:
            logger.warning("Empty returns data")
            return go.Figure()

        # Calculate correlation matrix
        corr_matrix = returns.corr()

        # Create heatmap
        fig = go.Figure(
            data=go.Heatmap(
                z=corr_matrix.values,
                x=corr_matrix.columns,
                y=corr_matrix.columns,
                colorscale='RdBu',
                zmid=0,
                text=np.round(corr_matrix.values, 2),
                texttemplate='%{text}',
                textfont={"size": 10},
                hoverongaps=False
            )
        )

        fig.update_layout(
            title=title,
            template=self.theme,
            height=800,
            width=800
        )

        self._handle_figure_output(fig, show_fig, output_file)
        return fig

    def plot_spread_analysis(self,
                           spread: pd.Series,
                           z_scores: pd.Series,
                           trades: Optional[pd.DataFrame] = None,
                           title: str = 'Spread Analysis',
                           show_fig: bool = True,
                           output_file: Optional[str] = None) -> go.Figure:
        """Plot spread and z-score analysis with optional trade markers."""
        if spread.empty:
            logger.warning("Empty spread data")
            return go.Figure()

        # Create subplots
        fig = make_subplots(
            rows=2,
            cols=1,
            subplot_titles=['Spread', 'Z-Score'],
            vertical_spacing=0.12
        )

        # Plot spread
        fig.add_trace(
            go.Scatter(
                x=spread.index,
                y=spread.values,
                mode='lines',
                name='Spread'
            ),
            row=1, col=1
        )

        # Add trade markers if provided
        if trades is not None:
            # Entry points
            entries = trades[trades['Action'] == 'ENTRY']
            fig.add_trace(
                go.Scatter(
                    x=entries.index,
                    y=spread[entries.index],
                    mode='markers',
                    marker=dict(size=10, color='green'),
                    name='Entry Points'
                ),
                row=1, col=1
            )

            # Exit points
            exits = trades[trades['Action'] == 'EXIT']
            fig.add_trace(
                go.Scatter(
                    x=exits.index,
                    y=spread[exits.index],
                    mode='markers',
                    marker=dict(size=10, color='red'),
                    name='Exit Points'
                ),
                row=1, col=1
            )

        # Plot z-scores
        fig.add_trace(
            go.Scatter(
                x=z_scores.index,
                y=z_scores.values,
                mode='lines',
                name='Z-Score'
            ),
            row=2, col=1
        )

        # Add threshold lines
        for threshold in [-2, -1, 0, 1, 2]:
            fig.add_hline(
                y=threshold,
                line_dash="dash",
                line_color="gray",
                row=2, col=1
            )

        # Update layout
        fig.update_layout(
            title=title,
            template=self.theme,
            height=800,
            showlegend=True
        )

        self._handle_figure_output(fig, show_fig, output_file)
        return fig

    def plot_optimization_results(self,
                                results: pd.DataFrame,
                                param_cols: List[str],
                                metric_col: str,
                                title: str = 'Optimization Results',
                                show_fig: bool = True,
                                output_file: Optional[str] = None) -> go.Figure:
        """Plot optimization results with parameter combinations."""
        if results.empty:
            logger.warning("Empty optimization results")
            return go.Figure()

        if len(param_cols) > 2:
            logger.warning("Can only visualize up to 2 parameters")
            param_cols = param_cols[:2]

        if len(param_cols) == 1:
            # Single parameter visualization
            fig = go.Figure(
                go.Scatter(
                    x=results[param_cols[0]],
                    y=results[metric_col],
                    mode='markers',
                    text=results[metric_col].round(4),
                    name=metric_col
                )
            )

            fig.update_layout(
                title=title,
                template=self.theme,
                xaxis_title=param_cols[0],
                yaxis_title=metric_col
            )

        else:
            # Two parameter visualization
            fig = go.Figure(
                go.Scatter(
                    x=results[param_cols[0]],
                    y=results[param_cols[1]],
                    mode='markers',
                    marker=dict(
                        size=10,
                        color=results[metric_col],
                        colorscale='Viridis',
                        showscale=True,
                        colorbar=dict(title=metric_col)
                    ),
                    text=results[metric_col].round(4)
                )
            )

            fig.update_layout(
                title=title,
                template=self.theme,
                xaxis_title=param_cols[0],
                yaxis_title=param_cols[1]
            )

        self._handle_figure_output(fig, show_fig, output_file)
        return fig

    @staticmethod
    def _handle_figure_output(fig: go.Figure,
                            show_fig: bool = True,
                            output_file: Optional[str] = None) -> None:
        """Handle figure display and saving."""
        if output_file:
            # Create directory if needed
            output_path = Path(output_file)
            output_path.parent.mkdir(parents=True, exist_ok=True)

            # Save based on extension
            if output_file.endswith('.html'):
                fig.write_html(output_file)
            else:
                fig.write_image(output_file)

        if show_fig:
            fig.show()

def main():
    """Example usage of visualization functions."""
    # Sample data
    dates = pd.date_range('2023-01-01', '2023-12-31', freq='D')
    equity_curve = pd.Series(
        np.random.normal(0, 1, len(dates)).cumsum() + 100000,
        index=dates
    )

    # Initialize visualizer
    viz = PlotlyVisualizer()

    # Create and save plots
    fig1 = viz.plot_equity_curve(
        equity_curve,
        title='Sample Equity Curve',
        output_file='equity_curve.html'
    )

    # Performance metrics
    metrics = {
        'Sharpe Ratio': 1.5,
        'Max Drawdown': -0.15,
        'Annual Return': 0.25
    }

    fig2 = viz.plot_performance_metrics(
        metrics,
        output_file='performance.html'
    )

    return fig1, fig2

if __name__ == "__main__":
    main()