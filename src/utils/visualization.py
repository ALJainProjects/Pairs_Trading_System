import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Dict

class PlotlyVisualizer:
    """Class for creating and managing Plotly visualizations for quant finance."""

    def __init__(self, theme: str = 'plotly_white'):
        self.theme = theme

    def plot_equity_curve(self, equity_curve: pd.Series, title: str = 'Equity Curve') -> go.Figure:
        """Plots equity curve and drawdown in a single figure."""
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.1,
                            subplot_titles=('Portfolio Value', 'Drawdown'))

        fig.add_trace(go.Scatter(x=equity_curve.index, y=equity_curve, mode='lines',
                                 name='Equity', line=dict(color='blue')), row=1, col=1)

        running_max = equity_curve.expanding().max()
        drawdown = (equity_curve - running_max) / running_max
        fig.add_trace(go.Scatter(x=drawdown.index, y=drawdown, mode='lines', fill='tozeroy',
                                 name='Drawdown', line=dict(color='red')), row=2, col=1)

        fig.update_layout(title_text=title, template=self.theme, height=700)
        fig.update_yaxes(title_text="Portfolio Value ($)", row=1, col=1)
        fig.update_yaxes(title_text="Drawdown (%)", tickformat=".0%", row=2, col=1)
        return fig

    def plot_underwater(self, equity_curve: pd.Series, title: str = 'Underwater Plot') -> go.Figure:
        """
        Plots the underwater curve, showing time spent in drawdown.
        """
        running_max = equity_curve.expanding().max()
        drawdown = (equity_curve - running_max) / running_max
        
        fig = go.Figure(data=go.Scatter(
            x=drawdown.index,
            y=drawdown,
            fill='tozeroy',
            mode='lines',
            line=dict(color='red')
        ))
        fig.update_layout(
            title_text=title,
            template=self.theme,
            yaxis_title="Drawdown (%)",
            yaxis_tickformat=".0%",
            height=500
        )
        return fig
    
    def plot_performance_metrics(self, metrics: Dict[str, float], title: str = 'Performance Metrics') -> go.Figure:
        """Plots performance metrics as a horizontal bar chart."""
        df_metrics = pd.DataFrame(list(metrics.items()), columns=['Metric', 'Value'])
        fig = go.Figure(go.Bar(
            x=df_metrics['Value'],
            y=df_metrics['Metric'],
            orientation='h',
            text=df_metrics['Value'].apply(lambda x: f'{x:.2f}'),
            textposition='auto'
        ))
        fig.update_layout(title_text=title, template=self.theme)
        return fig

    def plot_spread_analysis(self, spread: pd.Series, z_score: pd.Series, title: str = 'Spread Analysis') -> go.Figure:
        """Plots the spread and its z-score."""
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.1,
                            subplot_titles=('Pair Spread', 'Spread Z-Score'))

        fig.add_trace(go.Scatter(x=spread.index, y=spread, mode='lines', name='Spread'), row=1, col=1)
        
        fig.add_trace(go.Scatter(x=z_score.index, y=z_score, mode='lines', name='Z-Score'), row=2, col=1)
        fig.add_hline(y=2, line_dash="dash", line_color="red", row=2, col=1)
        fig.add_hline(y=-2, line_dash="dash", line_color="red", row=2, col=1)
        fig.add_hline(y=0, line_dash="dot", line_color="black", row=2, col=1)
        
        fig.update_layout(title_text=title, template=self.theme, height=700)
        return fig