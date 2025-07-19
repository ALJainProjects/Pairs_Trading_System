import pandas as pd
import numpy as np
from abc import ABC, abstractmethod
from sklearn.covariance import GraphicalLasso, MinCovDet, LedoitWolf
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import plotly.graph_objects as go
from typing import List, Optional
from config.logging_config import logger

class BaseCovariance(ABC):
    """Abstract base class for covariance estimation methods."""
    def __init__(self):
        self.covariance_: Optional[np.ndarray] = None
        self.correlation_: Optional[np.ndarray] = None
        self.column_names_: Optional[List[str]] = None

    def _validate_and_prepare(self, returns: pd.DataFrame) -> pd.DataFrame:
        """Validates input and returns a cleaned copy."""
        if not isinstance(returns, pd.DataFrame) or returns.empty:
            raise ValueError("Input must be a non-empty pandas DataFrame.")
        
        # Handle potential NaNs from pct_change() at the first row
        returns = returns.iloc[1:].copy()
        
        if returns.isnull().any().any():
            logger.warning("Input contains NaNs. Applying forward fill.")
            returns.ffill(inplace=True)
            returns.bfill(inplace=True)
        
        return returns

    @abstractmethod
    def fit(self, returns: pd.DataFrame) -> 'BaseCovariance':
        """Fit the covariance estimation model."""
        pass
    
    def _finalize_estimation(self, covariance: np.ndarray):
        """
        Ensures the final matrix is numerically stable (positive semi-definite)
        and computes correlation matrix.
        """
        min_eig = np.min(np.linalg.eigh(covariance)[0])
        if min_eig < 0:
            covariance -= (min_eig - 1e-12) * np.eye(covariance.shape[0])

        self.covariance_ = covariance
        std = np.sqrt(np.diag(self.covariance_))
        outer_std = np.outer(std, std)
        # Avoid division by zero for constant series
        self.correlation_ = np.divide(self.covariance_, outer_std, out=np.zeros_like(self.covariance_), where=outer_std!=0)
        np.fill_diagonal(self.correlation_, 1)


class StandardCovariance(BaseCovariance):
    """
    Standard sample covariance estimator.
    
    Use Case: Best for datasets where the number of observations (T) is much larger
    than the number of assets (N). Can be unstable otherwise.
    """
    def fit(self, returns: pd.DataFrame) -> 'StandardCovariance':
        returns = self._validate_and_prepare(returns)
        self.column_names_ = returns.columns.tolist()
        covariance = returns.cov().values
        self._finalize_estimation(covariance)
        return self

class EWMACovariance(BaseCovariance):
    """
    Exponentially Weighted Moving Average (EWMA) covariance estimator.
    
    Use Case: Gives more weight to recent observations, making it adaptive to
    changing market conditions. Excellent for capturing time-varying volatility and correlations.
    """
    def __init__(self, span: int = 60):
        super().__init__()
        self.span = span

    def fit(self, returns: pd.DataFrame) -> 'EWMACovariance':
        returns = self._validate_and_prepare(returns)
        self.column_names_ = returns.columns.tolist()
        covariance = returns.ewm(span=self.span).cov().values[-len(self.column_names_):]
        self._finalize_estimation(covariance)
        return self

class LedoitWolfShrinkage(BaseCovariance):
    """
    Ledoit-Wolf shrinkage estimator. Shrinks the sample covariance matrix towards a
    structured estimator, reducing estimation error.
    
    Use Case: Excellent general-purpose estimator, especially when N is close to T.
    Provides a good balance between the instability of the sample matrix and the bias of simpler models.
    """
    def fit(self, returns: pd.DataFrame) -> 'LedoitWolfShrinkage':
        returns = self._validate_and_prepare(returns)
        self.column_names_ = returns.columns.tolist()
        lw = LedoitWolf()
        lw.fit(returns)
        self._finalize_estimation(lw.covariance_)
        return self

class GraphicalLassoCovariance(BaseCovariance):
    """
    Sparse covariance estimation using the Graphical Lasso (L1 penalty).
    
    Use Case: Ideal for high-dimensional problems (many assets) where it's assumed
    that many pairwise correlations are zero. Helps uncover sparse underlying dependency structures.
    """
    def __init__(self, alpha: float = 0.01):
        super().__init__()
        self.alpha = alpha

    def fit(self, returns: pd.DataFrame) -> 'GraphicalLassoCovariance':
        returns = self._validate_and_prepare(returns)
        self.column_names_ = returns.columns.tolist()
        
        scaled_returns = StandardScaler().fit_transform(returns)
        model = GraphicalLasso(alpha=self.alpha, assume_centered=True)
        model.fit(scaled_returns)
        
        scales = np.diag(returns.std())
        covariance = scales @ model.covariance_ @ scales
        self._finalize_estimation(covariance)
        return self

class OLSCovariance(BaseCovariance):
    """
    OLS-based denoised covariance. Assumes a factor model structure and removes
    the idiosyncratic noise from each asset's returns.
    
    Use Case: Effective when returns are driven by a few common underlying factors (e.g., market, sectors).
    Helps to create a more stable covariance matrix by focusing on systematic risk.
    """
    def fit(self, returns: pd.DataFrame) -> 'OLSCovariance':
        returns = self._validate_and_prepare(returns)
        self.column_names_ = returns.columns.tolist()
        
        residuals = pd.DataFrame(index=returns.index)
        for col in returns.columns:
            y = returns[col]
            X = returns.drop(columns=col)
            model = LinearRegression().fit(X, y)
            residuals[col] = y - model.predict(X)
        
        covariance = residuals.cov().values
        self._finalize_estimation(covariance)
        return self

class RobustCovariance(BaseCovariance):
    """
    Robust covariance estimation using Minimum Covariance Determinant (MCD).
    
    Use Case: Excellent for datasets with significant outliers (e.g., during market crashes).
    It fits the covariance to a subset of "normal" data points, ignoring the outliers.
    """
    def __init__(self, contamination: float = 0.1, random_state: int = 42):
        super().__init__()
        self.contamination = contamination
        self.random_state = random_state

    def fit(self, returns: pd.DataFrame) -> 'RobustCovariance':
        returns = self._validate_and_prepare(returns)
        self.column_names_ = returns.columns.tolist()

        mcd = MinCovDet(support_fraction=1-self.contamination, random_state=self.random_state)
        mcd.fit(returns)

        self._finalize_estimation(mcd.covariance_)
        return self