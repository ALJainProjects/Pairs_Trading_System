import pandas as pd
import numpy as np
from scipy.linalg import pinv
from scipy import stats
import plotly.graph_objects as go
import seaborn as sns
import matplotlib.pyplot as plt
from typing import Tuple, Optional
from functools import lru_cache
from statsmodels.stats.multitest import multipletests
from config.logging_config import logger


class CorrelationAnalyzer:
    """
    Performs comprehensive correlation analysis on asset returns, including Pearson,
    Spearman, and partial correlations, along with stability analysis and advanced visualizations.
    """

    def __init__(self, returns: pd.DataFrame):
        self._validate_input(returns)
        self.returns = returns
        self.pearson_corr_: Optional[pd.DataFrame] = None
        self.spearman_corr_: Optional[pd.DataFrame] = None
        self.partial_corr_: Optional[pd.DataFrame] = None

    def _validate_input(self, returns: pd.DataFrame):
        if not isinstance(returns, pd.DataFrame) or not isinstance(returns.index, pd.DatetimeIndex):
            raise ValueError("Input must be a pandas DataFrame with a DatetimeIndex.")
        if returns.isnull().any().any():
            raise ValueError("Input returns contain NaN values.")

    @lru_cache(maxsize=2)
    def calculate_correlation(self, method: str = 'pearson') -> pd.DataFrame:
        """
        Calculates the correlation matrix using the specified method.
        """
        logger.info(f"Calculating {method} correlation matrix.")
        if method == 'pearson':
            if self.pearson_corr_ is None:
                self.pearson_corr_ = self.returns.corr(method='pearson')
            return self.pearson_corr_

        elif method == 'spearman':
            if self.spearman_corr_ is None:
                self.spearman_corr_ = self.returns.corr(method='spearman')
            return self.spearman_corr_

        elif method == 'partial':
            if self.partial_corr_ is None:
                self.partial_corr_ = self._calculate_partial_correlation()
            return self.partial_corr_
        else:
            raise ValueError("Method must be one of 'pearson', 'spearman', or 'partial'.")

    def _calculate_partial_correlation(self) -> pd.DataFrame:
        cov = self.returns.cov().values
        precision_matrix = pinv(cov)
        diag = np.sqrt(np.diag(precision_matrix))
        outer_diag = np.outer(diag, diag)
        partial_corr = -precision_matrix / outer_diag
        np.fill_diagonal(partial_corr, 1)
        return pd.DataFrame(partial_corr, index=self.returns.columns, columns=self.returns.columns)

    def get_highly_correlated_pairs(self, method: str = 'pearson', threshold: float = 0.7) -> pd.DataFrame:
        """
        Finds pairs of assets with correlation above a given threshold.
        """
        corr_matrix = self.calculate_correlation(method)
        upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        high_corr = upper_tri[abs(upper_tri) > threshold].stack().reset_index()
        high_corr.columns = ['Asset1', 'Asset2', 'Correlation']
        high_corr['Abs_Correlation'] = high_corr['Correlation'].abs()
        return high_corr.sort_values('Abs_Correlation', ascending=False).drop(columns='Abs_Correlation')

    def calculate_correlation_significance(self, corr_matrix: pd.DataFrame) -> pd.DataFrame:
        """
        Calculates the raw p-values for a Pearson correlation matrix.
        """
        n = len(self.returns)
        t_stat_squared = (n - 2) * (corr_matrix ** 2) / (1 - corr_matrix ** 2)
        p_values = stats.f.sf(t_stat_squared, 1, n - 2)

        p_value_df = pd.DataFrame(p_values, index=corr_matrix.index, columns=corr_matrix.columns)
        np.fill_diagonal(p_value_df.values, np.nan)
        return p_value_df

    def correct_pvalues_for_multiple_testing(self, p_values: pd.DataFrame, alpha: float = 0.05,
                                             method: str = 'fdr_bh') -> pd.DataFrame:
        """
        Applies a multiple hypothesis testing correction to a matrix of p-values.

        Args:
            p_values (pd.DataFrame): The matrix of raw p-values.
            alpha (float): The significance level.
            method (str): The correction method to use ('bonferroni', 'holm', 'fdr_bh').

        Returns:
            pd.DataFrame: A matrix of the corrected p-values.
        """
        # Extract the upper triangle of the p-values, excluding the diagonal
        p_values_flat = p_values.where(np.triu(np.ones(p_values.shape), k=1).astype(bool)).stack().values

        if len(p_values_flat) == 0:
            return p_values  # Return original if no tests to correct

        # Apply the correction
        reject, pvals_corrected, _, _ = multipletests(p_values_flat, alpha=alpha, method=method)

        # Reconstruct the corrected p-value matrix
        corrected_df = p_values.copy()
        p_value_map = dict(
            zip(p_values.where(np.triu(np.ones(p_values.shape), k=1).astype(bool)).stack().index, pvals_corrected))

        for (row, col), val in p_value_map.items():
            corrected_df.loc[row, col] = val
            corrected_df.loc[col, row] = val

        return corrected_df

    def plot_correlation_matrix(self, method: str = 'pearson', title: Optional[str] = None) -> go.Figure:
        """
        Generates an interactive heatmap of the correlation matrix.
        """
        corr_matrix = self.calculate_correlation(method)
        fig = go.Figure(data=go.Heatmap(
            z=corr_matrix.values,
            x=corr_matrix.columns,
            y=corr_matrix.columns,
            colorscale='RdBu',
            zmid=0
        ))
        fig.update_layout(
            title=title or f"{method.capitalize()} Correlation Matrix",
            height=800,
            width=800
        )
        return fig

    def plot_clustered_correlation_matrix(self, method: str = 'pearson', title: Optional[str] = None):
        """
        Generates a clustered heatmap to reveal underlying correlation structures.
        """
        logger.info(f"Generating clustered heatmap for {method} correlation.")
        corr_matrix = self.calculate_correlation(method)

        cg = sns.clustermap(
            corr_matrix,
            cmap='RdBu_r',
            vmin=-1,
            vmax=1,
            annot=False,
            figsize=(12, 12)
        )
        plt.setp(cg.ax_heatmap.xaxis.get_majorticklabels(), rotation=90)
        cg.fig.suptitle(title or f'Clustered {method.capitalize()} Correlation Matrix')
