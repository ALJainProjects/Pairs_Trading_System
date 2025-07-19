import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Dict, Optional


class PlotlyVisualizer:
    """Class for creating and managing Plotly visualizations for quant finance."""

    def __init__(self, theme: str = 'plotly_white'):
        """
        Initializes the PlotlyVisualizer.

        Args:
            theme (str): Plotly theme (e.g., 'plotly_white', 'plotly_dark', 'ggplot2', 'seaborn').
        """
        self.theme = theme

    def plot_equity_curve(self, equity_curve: pd.Series, title: str = 'Equity Curve') -> go.Figure:
        """
        Plots equity curve and drawdown in a single figure.

        Args:
            equity_curve (pd.Series): Time series of portfolio equity values.
            title (str): Title of the plot.

        Returns:
            go.Figure: A Plotly figure object.
        """
        if equity_curve.empty:
            return go.Figure().update_layout(title_text=f"{title} (No Data)", template=self.theme)

        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.1,
                            subplot_titles=('Portfolio Value', 'Drawdown'))

        # Ensure equity curve starts from a non-zero value for normalization purposes if needed for ratio comparisons
        if equity_curve.iloc[0] == 0:
            equity_curve = equity_curve.replace(0, np.nan).ffill().bfill()  # Replace 0s with NaNs then fill
            if equity_curve.iloc[0] == 0:  # If still 0 after fill, adjust starting point to 1
                equity_curve = equity_curve + 1  # Add 1 to all values to make them positive for ratios. THIS IS A VISUAL ADJUSTMENT
                # User should ensure proper equity_curve generation from Backtester.

        # Plot Equity Curve
        fig.add_trace(go.Scatter(x=equity_curve.index, y=equity_curve, mode='lines',
                                 name='Equity', line=dict(color='blue'),
                                 hovertemplate='<b>Date:</b> %{x}<br><b>Value:</b> $%{y:,.2f}<extra></extra>'), row=1,
                      col=1)

        # Plot Drawdown
        running_max = equity_curve.expanding().max()
        drawdown = (equity_curve - running_max) / running_max
        fig.add_trace(go.Scatter(x=drawdown.index, y=drawdown, mode='lines', fill='tozeroy',
                                 name='Drawdown', line=dict(color='red'),
                                 hovertemplate='<b>Date:</b> %{x}<br><b>Drawdown:</b> %{y:.2%}<extra></extra>'), row=2,
                      col=1)

        fig.update_layout(title_text=title, template=self.theme, height=700)
        fig.update_yaxes(title_text="Portfolio Value ($)", row=1, col=1)
        fig.update_yaxes(title_text="Drawdown (%)", tickformat=".0%", row=2, col=1)
        fig.update_xaxes(title_text="Date", row=2, col=1)  # Add x-axis title to bottom subplot

        return fig

    def plot_underwater(self, equity_curve: pd.Series, title: str = 'Underwater Plot') -> go.Figure:
        """
        Plots the underwater curve, showing time spent in drawdown.
        This is essentially the drawdown plot from `plot_equity_curve` as a standalone.

        Args:
            equity_curve (pd.Series): Time series of portfolio equity values.
            title (str): Title of the plot.

        Returns:
            go.Figure: A Plotly figure object.
        """
        if equity_curve.empty or equity_curve.iloc[0] == 0:
            return go.Figure().update_layout(title_text=f"{title} (No Data)", template=self.theme)

        running_max = equity_curve.expanding().max()
        drawdown = (equity_curve - running_max) / running_max

        fig = go.Figure(data=go.Scatter(
            x=drawdown.index,
            y=drawdown,
            fill='tozeroy',
            mode='lines',
            line=dict(color='red'),
            name='Drawdown',
            hovertemplate='<b>Date:</b> %{x}<br><b>Drawdown:</b> %{y:.2%}<extra></extra>'
        ))
        fig.update_layout(
            title_text=title,
            template=self.theme,
            yaxis_title="Drawdown (%)",
            yaxis_tickformat=".0%",
            xaxis_title="Date",
            height=500
        )
        return fig

    def plot_performance_metrics(self, metrics: Dict[str, float], title: str = 'Performance Metrics') -> go.Figure:
        """
        Plots performance metrics as a horizontal bar chart.
        Metrics are formatted based on their typical representation (percentage, ratio, currency).

        Args:
            metrics (Dict[str, float]): A dictionary of performance metrics.
            title (str): Title of the plot.

        Returns:
            go.Figure: A Plotly figure object.
        """
        if not metrics:
            return go.Figure().update_layout(title_text=f"{title} (No Metrics)", template=self.theme)

        df_metrics = pd.DataFrame(list(metrics.items()), columns=['Metric', 'Value'])

        # Sort metrics for better visualization (e.g., by value for horizontal bars)
        # Or sort alphabetically if preferred: df_metrics = df_metrics.sort_values(by='Metric', ascending=True)
        # For this plot, sorting by value makes sense for horizontal bars
        df_metrics['AbsValue'] = df_metrics['Value'].apply(abs)  # Use absolute for sorting
        df_metrics = df_metrics.sort_values(by='AbsValue', ascending=True)  # Ascending for horizontal bar charts

        # Custom text formatting based on metric name heuristics
        def format_metric_text(metric_name: str, value: float) -> str:
            if any(s in metric_name for s in ['Return', 'Drawdown', 'Rate', 'Pct']):
                return f'{value:.2%}'  # Percentage
            elif any(s in metric_name for s in ['Ratio', 'Factor']):
                return f'{value:.2f}'  # Ratio/Factor
            elif any(s in metric_name for s in ['$', 'Profit', 'Loss', 'Cost', 'Capital']):
                return f'${value:,.2f}'  # Currency
            elif any(s in metric_name for s in ['Trades']):
                return f'{int(value)}'  # Integer
            return f'{value:.2f}'  # Default to 2 decimal places

        df_metrics['FormattedValue'] = df_metrics.apply(lambda row: format_metric_text(row['Metric'], row['Value']),
                                                        axis=1)

        # Assign colors based on value (e.g., green for positive, red for negative for some metrics)
        colors = ['red' if v < 0 else 'green' for v in df_metrics['Value']]
        # Exceptions: Drawdown should be red, but its value is usually positive (max_dd is abs).
        # We can refine this:
        actual_colors = []
        for _, row in df_metrics.iterrows():
            if 'Drawdown' in row['Metric'] and row['Value'] > 0:  # Max Drawdown is positive, but still represents loss
                actual_colors.append('red')
            elif row['Value'] < 0:
                actual_colors.append('red')
            else:
                actual_colors.append('green')

        fig = go.Figure(go.Bar(
            x=df_metrics['Value'],
            y=df_metrics['Metric'],
            orientation='h',
            text=df_metrics['FormattedValue'],
            textposition='auto',
            marker_color=actual_colors  # Apply dynamic colors
        ))

        fig.update_layout(title_text=title, template=self.theme,
                          xaxis_title="Value", yaxis_title="Metric",
                          height=max(500, len(metrics) * 30),  # Adjust height dynamically
                          margin=dict(l=150, r=20, t=50, b=20)  # Adjust margins for labels
                          )
        return fig

    def plot_spread_analysis(self, spread: pd.Series, z_score: pd.Series,
                             title: str = 'Spread Analysis',
                             entry_zscore: Optional[float] = None,
                             exit_zscore: Optional[float] = None) -> go.Figure:
        """
        Plots the spread and its z-score, with optional entry/exit thresholds.

        Args:
            spread (pd.Series): Time series of the calculated spread values.
            z_score (pd.Series): Time series of the spread's z-score.
            title (str): Title of the plot.
            entry_zscore (Optional[float]): Z-score threshold for entering trades (e.g., 2.0).
            exit_zscore (Optional[float]): Z-score threshold for exiting trades (e.g., 0.5).

        Returns:
            go.Figure: A Plotly figure object.
        """
        if spread.empty or z_score.empty:
            return go.Figure().update_layout(title_text=f"{title} (No Data)", template=self.theme)

        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.1,
                            subplot_titles=('Pair Spread', 'Spread Z-Score'))

        # Plot Spread
        fig.add_trace(go.Scatter(x=spread.index, y=spread, mode='lines', name='Spread',
                                 hovertemplate='<b>Date:</b> %{x}<br><b>Spread:</b> %{y:,.4f}<extra></extra>'), row=1,
                      col=1)

        # Plot Z-Score
        fig.add_trace(go.Scatter(x=z_score.index, y=z_score, mode='lines', name='Z-Score',
                                 hovertemplate='<b>Date:</b> %{x}<br><b>Z-Score:</b> %{y:.2f}<extra></extra>'), row=2,
                      col=1)

        # Add Z-score thresholds
        if entry_zscore is not None:
            fig.add_hline(y=entry_zscore, line_dash="dash", line_color="red", annotation_text=f"Entry: {entry_zscore}",
                          annotation_position="top right", row=2, col=1)
            fig.add_hline(y=-entry_zscore, line_dash="dash", line_color="red",
                          annotation_text=f"Entry: {-entry_zscore}",
                          annotation_position="bottom right", row=2, col=1)

        if exit_zscore is not None:
            fig.add_hline(y=exit_zscore, line_dash="dot", line_color="green", annotation_text=f"Exit: {exit_zscore}",
                          annotation_position="top left", row=2, col=1)
            fig.add_hline(y=-exit_zscore, line_dash="dot", line_color="green", annotation_text=f"Exit: {-exit_zscore}",
                          annotation_position="bottom left", row=2, col=1)

        fig.add_hline(y=0, line_dash="dot", line_color="black", row=2, col=1, annotation_text="Mean",
                      annotation_position="top center")

        fig.update_layout(title_text=title, template=self.theme, height=700)
        fig.update_yaxes(title_text="Spread Value", row=1, col=1)
        fig.update_yaxes(title_text="Z-Score", row=2, col=1)
        fig.update_xaxes(title_text="Date", row=2, col=1)

        return fig