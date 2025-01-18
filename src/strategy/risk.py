from typing import Dict, Tuple, Optional
import pandas as pd
import numpy as np
from dataclasses import dataclass, field
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from config.logging_config import logger


class MarketImpactModel:
    """Model to estimate market impact costs."""
    def __init__(self):
        self.impact_coefficient = 0.1
        self.min_spread = 0.0001

    def calculate_market_impact(self, trade_size: float, price: float) -> float:
        """Calculate market impact cost for a trade."""
        return trade_size * price * self.impact_coefficient

@dataclass
class RiskMetrics:
    """Enhanced risk metrics structure"""
    var_95: float = 0.0
    cvar_95: float = 0.0
    volatility: float = 0.0
    correlation_risk: float = 0.0
    model_confidence: float = 0.0
    cointegration_stability: float = 0.0
    pair_metrics: Dict = field(default_factory=dict)

class PairRiskManager:
    """Enhanced risk manager with integrated ML/DL confidence scores and advanced metrics"""

    def __init__(
            self,
            max_position_size: float = 0.05,
            max_drawdown: float = 0.20,
            stop_loss_threshold: float = 0.10,
            max_correlation: float = 0.7,
            leverage_limit: float = 2.0,
            var_confidence: float = 0.95,
            min_model_confidence: float = 0.6,
            max_correlation_exposure: float = 0.3
    ):
        self.cost_model = MarketImpactModel()
        self.max_position_size = max_position_size
        self.max_drawdown = max_drawdown
        self.stop_loss_threshold = stop_loss_threshold
        self.max_correlation = max_correlation
        self.leverage_limit = leverage_limit
        self.var_confidence = var_confidence
        self.min_model_confidence = min_model_confidence
        self.max_correlation_exposure = max_correlation_exposure

        self.risk_metrics = {}
        self.correlation_matrix = pd.DataFrame()
        self.position_history = []

    def calculate_market_impact_cost(self, trade_size: float, price: float) -> float:
        """Calculate market impact cost for a trade."""
        return self.cost_model.calculate_market_impact(trade_size, price)

    def _calculate_pair_correlation(self, pair1: Tuple[str, str],
                                    pair2: Tuple[str, str]) -> float:
        """Calculate correlation between two pairs"""
        try:
            spread1 = self._get_pair_spread(pair1)
            spread2 = self._get_pair_spread(pair2)

            if spread1 is not None and spread2 is not None:
                correlation = spread1.corr(spread2)
                return correlation if not np.isnan(correlation) else 0.0
            return 0.0

        except Exception as e:
            logger.warning(f"Error calculating pair correlation: {str(e)}")
            return 0.0

    def _get_pair_spread(self, pair: Tuple[str, str]) -> Optional[pd.Series]:
        """Calculate spread series for a pair"""
        asset1, asset2 = pair
        if asset1 in self.correlation_matrix.index and asset2 in self.correlation_matrix.index:
            spread = self.correlation_matrix[asset1] - self.correlation_matrix[asset2]
            return spread
        return None

    def calculate_drawdown(self, equity_curve: pd.Series) -> float:
        """Calculate current drawdown"""
        if len(equity_curve) < 1:
            return 0.0
        peak = equity_curve.expanding().max()
        drawdown = (equity_curve - peak) / peak
        return abs(float(drawdown.iloc[-1]))

    def calculate_var_cvar(self, returns: pd.Series) -> Tuple[float, float]:
        """Calculate Value at Risk and Conditional Value at Risk"""
        if len(returns) < 2:
            return 0.0, 0.0

        sorted_returns = np.sort(returns.values)
        var_idx = int(len(returns) * (1 - self.var_confidence))
        var = float(sorted_returns[var_idx])
        cvar = float(sorted_returns[:var_idx].mean())

        return abs(var), abs(cvar)

    def update_risk_metrics(self,
                            pair: Tuple[str, str],
                            prices: pd.DataFrame,
                            positions: Dict,
                            model_confidence: Optional[float] = None) -> RiskMetrics:
        """Update comprehensive risk metrics for a pair"""
        returns = self._calculate_pair_returns(pair, prices)
        var, cvar = self.calculate_var_cvar(returns)

        self.correlation_matrix = prices.pct_change().dropna()

        metrics = RiskMetrics(
            var_95=var,
            cvar_95=cvar,
            volatility=returns.std() if len(returns) > 1 else 0.0,
            correlation_risk=self._calculate_correlation_risk(pair, positions),
            model_confidence=model_confidence or 1.0,
            cointegration_stability=self._calculate_cointegration_stability(pair, prices)
        )

        self.risk_metrics[pair] = metrics
        return metrics
        
    def calculate_position_size(self,
                              portfolio_value: float,
                              pair: Tuple[str, str],
                              prices: pd.DataFrame,
                              model_confidence: Optional[float] = None,
                              correlation_matrix: Optional[pd.DataFrame] = None) -> float:
        """Calculate position size with enhanced risk adjustments"""
        base_size = self.max_position_size * portfolio_value

        if model_confidence is not None:
            if model_confidence < self.min_model_confidence:
                return 0.0
            base_size *= model_confidence

        pair_returns = self._calculate_pair_returns(pair, prices)
        vol_adjustment = self._calculate_volatility_adjustment(pair_returns)
        base_size *= vol_adjustment

        if correlation_matrix is not None:
            corr_adjustment = self._calculate_correlation_adjustment(pair, correlation_matrix)
            base_size *= corr_adjustment

        var, _ = self.calculate_var_cvar(pair_returns)
        var_adjustment = 1.0 / (1.0 + var)
        base_size *= var_adjustment
        
        return min(base_size, self.max_position_size * portfolio_value)
        
    def _calculate_pair_returns(self, pair: Tuple[str, str], prices: pd.DataFrame) -> pd.Series:
        """Calculate returns for a pair"""
        asset1, asset2 = pair
        spread = prices[asset1] - prices[asset2]
        return spread.pct_change().dropna()
        
    def _calculate_volatility_adjustment(self, returns: pd.Series) -> float:
        """Calculate volatility-based position adjustment"""
        if len(returns) < 2:
            return 1.0
        vol = returns.std()
        return 1.0 / (1.0 + vol)
        
    def _calculate_correlation_adjustment(self, 
                                       pair: Tuple[str, str],
                                       correlation_matrix: pd.DataFrame) -> float:
        """Calculate correlation-based position adjustment"""
        asset1, asset2 = pair
        if asset1 not in correlation_matrix.index or asset2 not in correlation_matrix.index:
            return 1.0
            
        pair_correlations = correlation_matrix.loc[asset1] + correlation_matrix.loc[asset2]
        avg_correlation = pair_correlations.abs().mean()
        
        if avg_correlation > self.max_correlation:
            return 0.0
            
        return 1.0 - (avg_correlation / self.max_correlation)
        
    def _calculate_correlation_risk(self, pair: Tuple[str, str], positions: Dict) -> float:
        """Calculate correlation-based risk exposure"""
        if not positions or not self.correlation_matrix.empty:
            return 0.0
            
        active_pairs = list(positions.keys())
        if not active_pairs:
            return 0.0
            
        total_correlation = 0.0
        count = 0
        
        for active_pair in active_pairs:
            if active_pair == pair:
                continue
                
            corr = self._calculate_pair_correlation(pair, active_pair)
            if not np.isnan(corr):
                total_correlation += abs(corr)
                count += 1
                
        return total_correlation / count if count > 0 else 0.0
        
    def _calculate_cointegration_stability(self, 
                                         pair: Tuple[str, str],
                                         prices: pd.DataFrame,
                                         window: int = 63) -> float:
        """Calculate stability of cointegration relationship"""
        asset1, asset2 = pair
        if asset1 not in prices.columns or asset2 not in prices.columns:
            return 0.0
            
        from statsmodels.tsa.stattools import coint
        stability_scores = []
        
        for i in range(window, len(prices)):
            _, pvalue, _ = coint(
                prices[asset1].iloc[i-window:i],
                prices[asset2].iloc[i-window:i]
            )
            stability_scores.append(1 - pvalue)
            
        return np.mean(stability_scores) if stability_scores else 0.0
        
    def check_risk_limits(self,
                         equity_curve: pd.Series,
                         positions: Dict,
                         current_prices: Dict[str, float]) -> Tuple[bool, str]:
        """Check comprehensive risk limits"""
        if self.calculate_drawdown(equity_curve) > self.max_drawdown:
            return True, "Maximum drawdown exceeded"

        portfolio_returns = equity_curve.pct_change().dropna()
        var, cvar = self.calculate_var_cvar(portfolio_returns)
        if cvar > self.max_drawdown:
            return True, "CVaR limit exceeded"

        total_correlation_exposure = sum(
            self.risk_metrics[pair].correlation_risk
            for pair in positions
            if pair in self.risk_metrics
        )
        if total_correlation_exposure > self.max_correlation_exposure:
            return True, "Correlation exposure limit exceeded"

        low_confidence_positions = [
            pair for pair in positions
            if pair in self.risk_metrics
            and self.risk_metrics[pair].model_confidence < self.min_model_confidence
        ]
        if low_confidence_positions:
            return True, "Model confidence below threshold"
            
        return False, ""
        
    def plot_risk_metrics(self) -> go.Figure:
        """Create visualization of risk metrics"""
        metrics_df = pd.DataFrame({
            pair: {
                'VaR': metrics.var_95,
                'CVaR': metrics.cvar_95,
                'Volatility': metrics.volatility,
                'Correlation Risk': metrics.correlation_risk,
                'Model Confidence': metrics.model_confidence,
                'Cointegration Stability': metrics.cointegration_stability
            }
            for pair, metrics in self.risk_metrics.items()
        }).T
        
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                'Risk Metrics Distribution',
                'Model Confidence vs. Correlation Risk',
                'VaR/CVaR Analysis',
                'Cointegration Stability'
            )
        )

        for col in metrics_df.columns:
            fig.add_trace(
                go.Box(y=metrics_df[col], name=col),
                row=1, col=1
            )

        fig.add_trace(
            go.Scatter(
                x=metrics_df['Model Confidence'],
                y=metrics_df['Correlation Risk'],
                mode='markers',
                name='Confidence vs Risk'
            ),
            row=1, col=2
        )

        fig.add_trace(
            go.Scatter(
                x=metrics_df.index,
                y=metrics_df['VaR'],
                name='VaR'
            ),
            row=2, col=1
        )
        fig.add_trace(
            go.Scatter(
                x=metrics_df.index,
                y=metrics_df['CVaR'],
                name='CVaR'
            ),
            row=2, col=1
        )

        fig.add_trace(
            go.Bar(
                x=metrics_df.index,
                y=metrics_df['Cointegration Stability'],
                name='Cointegration Stability'
            ),
            row=2, col=2
        )
        
        fig.update_layout(height=800, title_text="Risk Analysis Dashboard")
        return fig