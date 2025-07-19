import pandas as pd
import numpy as np
from scipy.linalg import pinv
from scipy import stats
import plotly.graph_objects as go
import seaborn as sns
import matplotlib.pyplot as plt
from typing import Tuple, Optional, Dict, Union
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
        # Initialize cached attributes
        self.pearson_corr_: Optional[pd.DataFrame] = None
        self.spearman_corr_: Optional[pd.DataFrame] = None
        self.partial_corr_: Optional[pd.DataFrame] = None

    def _validate_input(self, returns: pd.DataFrame):
        """
        Validates the input DataFrame for correlation analysis.
        Ensures it's a DataFrame with DatetimeIndex, no NaNs, and at least 2 columns.
        """
        if not isinstance(returns, pd.DataFrame):
            raise ValueError("Input must be a pandas DataFrame.")
        if not isinstance(returns.index, pd.DatetimeIndex):
            raise ValueError("Input DataFrame must have a DatetimeIndex.")
        if returns.empty:
            raise ValueError("Input DataFrame is empty.")
        if returns.isnull().any().any():
            raise ValueError(
                "Input returns contain NaN values. Please handle them before analysis (e.g., with dropna(), fillna()).")
        if returns.shape[1] < 2:
            raise ValueError("Input DataFrame must contain at least two assets (columns) for correlation analysis.")
        if returns.shape[0] < 2:
            raise ValueError(
                "Input DataFrame must contain at least two time observations (rows) for correlation analysis.")

    @lru_cache(maxsize=2)  # Cache Pearson and Spearman results
    def calculate_correlation(self, method: str = 'pearson') -> pd.DataFrame:
        """
        Calculates the correlation matrix using the specified method.

        Args:
            method (str): The correlation method to use ('pearson', 'spearman', or 'partial').

        Returns:
            pd.DataFrame: The calculated correlation matrix.
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

    @lru_cache(maxsize=1)  # Cache partial correlation result as it's more expensive
    def _calculate_partial_correlation(self) -> pd.DataFrame:
        """
        Helper method to calculate the partial correlation matrix using the precision matrix.
        """
        if self.returns.shape[0] <= self.returns.shape[1]:
            logger.warning(
                "Number of observations is less than or equal to number of assets. Covariance matrix may be singular, partial correlation might be unstable or undefined.")
            # For extremely short series, cov might be singular, pinv might produce bad results.
            # You might want to raise an error here or return an empty/NaN matrix.
            # For now, let pinv attempt to calculate.

        cov = self.returns.cov().values

        try:
            # Calculate the pseudo-inverse (precision matrix)
            precision_matrix = pinv(cov)
        except np.linalg.LinAlgError as e:
            logger.error(f"Linear algebra error during pseudo-inverse calculation for partial correlation: {e}")
            return pd.DataFrame(np.nan, index=self.returns.columns, columns=self.returns.columns)

        # Calculate partial correlation from precision matrix
        # Formula: P_ij = - (Precision_ij) / sqrt(Precision_ii * Precision_jj)
        diag = np.sqrt(np.diag(precision_matrix))
        outer_diag = np.outer(diag, diag)

        # Handle potential division by zero if a diagonal element is zero (very rare for real data)
        # Using np.divide with where for robustness
        partial_corr = np.divide(-precision_matrix, outer_diag,
                                 out=np.full_like(precision_matrix, np.nan),
                                 where=outer_diag != 0)

        np.fill_diagonal(partial_corr, 1)  # Diagonal elements should be 1

        return pd.DataFrame(partial_corr, index=self.returns.columns, columns=self.returns.columns)

    def get_highly_correlated_pairs(self, method: str = 'pearson', threshold: float = 0.7) -> pd.DataFrame:
        """
        Finds pairs of assets with correlation (absolute value) above a given threshold.

        Args:
            method (str): The correlation method to use ('pearson', 'spearman', or 'partial').
            threshold (float): The absolute correlation threshold (between 0 and 1).

        Returns:
            pd.DataFrame: A DataFrame with columns 'Asset1', 'Asset2', 'Correlation',
                          sorted by absolute correlation in descending order.
        """
        if not (0 <= threshold <= 1):
            raise ValueError("Threshold must be between 0 and 1.")

        corr_matrix = self.calculate_correlation(method)

        # Use upper triangle to avoid duplicate pairs and self-correlation
        upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

        # Filter by absolute threshold
        high_corr = upper_tri[abs(upper_tri) > threshold].stack().reset_index()
        high_corr.columns = ['Asset1', 'Asset2', 'Correlation']
        high_corr['Abs_Correlation'] = high_corr['Correlation'].abs()

        return high_corr.sort_values('Abs_Correlation', ascending=False).drop(columns='Abs_Correlation')

    def calculate_correlation_significance(self, corr_matrix: pd.DataFrame) -> pd.DataFrame:
        """
        Calculates the raw p-values for a Pearson correlation matrix.
        Note: This method is specifically designed for Pearson correlation.
        P-value calculation for other correlation methods (Spearman, Partial) is different
        and typically more complex or requires different assumptions/approaches.

        Args:
            corr_matrix (pd.DataFrame): The Pearson correlation matrix.

        Returns:
            pd.DataFrame: A matrix of the raw p-values.
        """
        n = len(self.returns)
        if n < 3:  # Need at least 3 observations for t-statistic (n-2 degrees of freedom)
            logger.warning(
                "Not enough observations (n<3) to calculate meaningful p-values for correlation. Returning NaN matrix.")
            return pd.DataFrame(np.nan, index=corr_matrix.index, columns=corr_matrix.columns)

        # Ensure we're working with a copy to avoid modifying the input correlation matrix
        # And ensure it's a Pearson matrix if not already
        if self.pearson_corr_ is None or not corr_matrix.equals(self.pearson_corr_):
            # If input corr_matrix is not the cached Pearson, assume it's Pearson and proceed
            logger.warning(
                "`calculate_correlation_significance` is designed for Pearson correlation. Ensure input `corr_matrix` is Pearson.")

        # Calculate t-statistic squared for correlation significance
        # t = r * sqrt((n-2) / (1-r^2))
        # t^2 = r^2 * (n-2) / (1-r^2)
        # We use F-distribution (F = t^2 for 1 degree of freedom in numerator)
        r_squared = corr_matrix ** 2
        # Handle cases where r_squared is 1 (perfect correlation), to avoid division by zero
        r_squared = r_squared.replace(1.0, 0.99999999)  # Very close to 1, but not exactly

        t_stat_squared = (n - 2) * r_squared / (1 - r_squared)

        # Get p-values from F-distribution (two-sided test)
        p_values = stats.f.sf(t_stat_squared, 1, n - 2)

        p_value_df = pd.DataFrame(p_values, index=corr_matrix.index, columns=corr_matrix.columns)
        np.fill_diagonal(p_value_df.values, np.nan)  # Correlation with self is always 1, not tested
        return p_value_df

    def correct_pvalues_for_multiple_testing(self, p_values: pd.DataFrame, alpha: float = 0.05,
                                             method: str = 'fdr_bh') -> pd.DataFrame:
        """
        Applies a multiple hypothesis testing correction to a matrix of p-values.

        Args:
            p_values (pd.DataFrame): The matrix of raw p-values (e.g., from `calculate_correlation_significance`).
            alpha (float): The significance level.
            method (str): The correction method to use. Common options:
                          'bonferroni': Bonferroni correction (very strict).
                          'holm': Holm-Bonferroni method (less strict than Bonferroni).
                          'fdr_bh': Benjamini-Hochberg FDR correction (controls False Discovery Rate).
                          Refer to `statsmodels.stats.multitest.multipletests` for full list.

        Returns:
            pd.DataFrame: A matrix of the corrected p-values.
        """
        # Extract the upper triangle of the p-values, excluding the diagonal and NaNs
        # Only test unique pairs, so exclude diagonal and lower triangle
        mask = np.triu(np.ones(p_values.shape), k=1).astype(bool)
        p_values_flat = p_values.where(mask).stack().values

        if len(p_values_flat) == 0:
            logger.warning("No p-values found to correct for multiple testing. Returning original DataFrame.")
            return p_values.copy()

        # Apply the correction
        try:
            reject, pvals_corrected, alphacSidak, alphacBonf = multipletests(p_values_flat, alpha=alpha, method=method)
            logger.info(f"Multiple testing correction '{method}' applied. Rejected {np.sum(reject)} hypotheses.")
        except ValueError as e:
            logger.error(
                f"Error applying multiple testing correction with method '{method}': {e}. Returning original p-values.")
            return p_values.copy()

        # Reconstruct the corrected p-value matrix
        corrected_df = p_values.copy()

        # Create a mapping from original (row, col) indices to corrected p-values
        # The stack().index is a MultiIndex of (row, col) tuples
        original_indices = p_values.where(mask).stack().index
        p_value_map = dict(zip(original_indices, pvals_corrected))

        for (row, col), val in p_value_map.items():
            corrected_df.loc[row, col] = val
            corrected_df.loc[col, row] = val  # Fill symmetric element

        # Ensure diagonal remains NaN or 1 if desired (depending on original p_values)
        np.fill_diagonal(corrected_df.values, np.nan)

        return corrected_df

    def calculate_rolling_correlation(self, window: int, method: str = 'pearson') -> Dict[pd.Timestamp, pd.DataFrame]:
        """
        Calculates rolling correlation matrices over time.

        Args:
            window (int): The size of the rolling window (number of observations).
            method (str): The correlation method to use ('pearson' or 'spearman'). Partial correlation
                          is not typically calculated in a rolling manner due to computational cost
                          and stability issues with small windows.

        Returns:
            Dict[pd.Timestamp, pd.DataFrame]: A dictionary where keys are the end dates
                                              of the rolling window and values are the
                                              correlation matrices for that window.
        """
        if window < 2:
            raise ValueError("Rolling window size must be at least 2.")
        if len(self.returns) < window:
            logger.warning(
                f"Data length ({len(self.returns)}) is less than window size ({window}). Cannot perform rolling correlation.")
            return {}
        if method not in ['pearson', 'spearman']:
            raise ValueError(
                "Rolling correlation method must be 'pearson' or 'spearman'. Partial correlation is not supported for rolling.")

        logger.info(f"Calculating rolling {method} correlation with window {window}.")

        rolling_corr_matrices = {}
        for i in range(len(self.returns) - window + 1):
            window_data = self.returns.iloc[i: i + window]
            end_date = window_data.index[-1]

            # Ensure enough observations in window for correlation
            if window_data.shape[0] < window:  # Should not happen with current loop, but for safety
                logger.debug(
                    f"Skipping window ending {end_date}: not enough observations ({window_data.shape[0]} < {window}).")
                continue

            # Ensure non-constant series in window if method is pearson/spearman
            if window_data.nunique().min() <= 1:
                logger.warning(
                    f"Window ending {end_date} contains constant series. Correlation might be ill-defined or NaN. Skipping.")
                rolling_corr_matrices[end_date] = pd.DataFrame(np.nan, index=self.returns.columns,
                                                               columns=self.returns.columns)
                continue

            try:
                corr_matrix = window_data.corr(method=method)
                rolling_corr_matrices[end_date] = corr_matrix
            except Exception as e:
                logger.error(f"Error calculating {method} correlation for window ending {end_date}: {e}")
                rolling_corr_matrices[end_date] = pd.DataFrame(np.nan, index=self.returns.columns,
                                                               columns=self.returns.columns)

        return rolling_corr_matrices

    def plot_correlation_matrix(self, method: str = 'pearson', title: Optional[str] = None) -> go.Figure:
        """
        Generates an interactive heatmap of the correlation matrix.

        Args:
            method (str): The correlation method to use ('pearson', 'spearman', or 'partial').
            title (Optional[str]): Custom title for the plot. If None, a default title is generated.

        Returns:
            go.Figure: A Plotly Figure object.
        """
        corr_matrix = self.calculate_correlation(method)
        fig = go.Figure(data=go.Heatmap(
            z=corr_matrix.values,
            x=corr_matrix.columns,
            y=corr_matrix.columns,
            colorscale='RdBu',
            zmid=0  # Center the colorscale around 0
        ))
        fig.update_layout(
            title=title or f"{method.capitalize()} Correlation Matrix",
            height=800,
            width=800,
            xaxis_nticks=len(corr_matrix.columns),  # Show all labels
            yaxis_nticks=len(corr_matrix.columns)
        )
        return fig

    def plot_clustered_correlation_matrix(self, method: str = 'pearson', title: Optional[str] = None):
        """
        Generates a clustered heatmap using Seaborn to reveal underlying correlation structures.
        This uses matplotlib internally and will display the plot directly.

        Args:
            method (str): The correlation method to use ('pearson', 'spearman', or 'partial').
            title (Optional[str]): Custom title for the plot. If None, a default title is generated.
        """
        logger.info(f"Generating clustered heatmap for {method} correlation.")
        corr_matrix = self.calculate_correlation(method)

        # Drop any rows/columns that are all NaN in the correlation matrix (can happen with partial)
        corr_matrix_clean = corr_matrix.dropna(how='all').dropna(axis=1, how='all')
        if corr_matrix_clean.empty:
            logger.warning("Correlation matrix is empty after dropping NaNs. Cannot generate clustered heatmap.")
            return  # Don't raise error, just return

        # clustermap internally performs hierarchical clustering
        cg = sns.clustermap(
            corr_matrix_clean,  # Use clean matrix
            cmap='RdBu_r',  # Reversed RdBu for more intuitive (red = positive, blue = negative)
            vmin=-1,  # Ensure color scale fixed from -1 to 1
            vmax=1,
            annot=False,  # Set to True for annotations, but can be cluttered for many assets
            figsize=(12, 12)
        )
        # Adjust tick labels for readability
        plt.setp(cg.ax_heatmap.xaxis.get_majorticklabels(), rotation=90)
        plt.setp(cg.ax_heatmap.yaxis.get_majorticklabels(), rotation=0)  # Often preferred for y-axis

        cg.fig.suptitle(title or f'Clustered {method.capitalize()} Correlation Matrix',
                        y=1.02)  # y adjusts title height
        plt.tight_layout()  # Adjust layout to prevent labels overlapping
        plt.show()  # Display the plot


# --- Example Usage ---
def main():
    # 1. Generate Sample Data
    np.random.seed(42)
    dates = pd.date_range(start='2020-01-01', periods=252)  # 1 year of daily data

    # Create 3 "clusters" of assets
    # Cluster 1: Highly correlated positive
    c1_factor = np.random.normal(0, 0.01, len(dates)).cumsum()
    assets_c1 = pd.DataFrame({
        'Asset_A': c1_factor + np.random.normal(0, 0.005, len(dates)),
        'Asset_B': c1_factor * 1.1 + np.random.normal(0, 0.006, len(dates)),
        'Asset_C': c1_factor * 0.9 + np.random.normal(0, 0.004, len(dates))
    }, index=dates)

    # Cluster 2: Moderately correlated, different trend
    c2_factor = np.random.normal(0.001, 0.008, len(dates)).cumsum()
    assets_c2 = pd.DataFrame({
        'Asset_D': c2_factor + np.random.normal(0, 0.005, len(dates)),
        'Asset_E': c2_factor * 0.8 + np.random.normal(0, 0.007, len(dates)),
        'Asset_F': c2_factor * 1.2 + np.random.normal(0, 0.006, len(dates))
    }, index=dates)

    # Cluster 3: Lowly correlated / idiosyncratic
    assets_c3 = pd.DataFrame({
        'Asset_G': np.random.normal(0, 0.015, len(dates)).cumsum(),
        'Asset_H': np.random.normal(0, 0.012, len(dates)).cumsum(),
    }, index=dates)

    # Combine into a single price DataFrame, then convert to returns
    prices = pd.concat([assets_c1, assets_c2, assets_c3], axis=1)
    returns = prices.pct_change().dropna()

    # Introduce some perfect correlation to see edge case in partial/significance
    returns['Asset_A_Copy'] = returns['Asset_A']
    # Introduce NaNs to test validation
    # returns.iloc[10, 0] = np.nan

    print("--- Original Returns Data ---")
    print(returns.head())
    print(f"Data shape: {returns.shape}")

    # 2. Initialize Analyzer
    try:
        analyzer = CorrelationAnalyzer(returns)
    except ValueError as e:
        print(f"Initialization Error: {e}")
        return

    # 3. Calculate and Get Correlation Matrices
    print("\n--- Calculating Correlation Matrices ---")
    pearson_corr = analyzer.calculate_correlation('pearson')
    print("\nPearson Correlation (first 5x5):\n", pearson_corr.iloc[:5, :5])

    spearman_corr = analyzer.calculate_correlation('spearman')
    print("\nSpearman Correlation (first 5x5):\n", spearman_corr.iloc[:5, :5])

    partial_corr = analyzer.calculate_correlation('partial')
    print("\nPartial Correlation (first 5x5):\n", partial_corr.iloc[:5, :5])

    # 4. Get Highly Correlated Pairs
    print("\n--- Highly Correlated Pairs (Pearson > 0.8) ---")
    high_corr_pairs = analyzer.get_highly_correlated_pairs(method='pearson', threshold=0.8)
    print(high_corr_pairs.head(10))

    print("\n--- Highly Correlated Pairs (Partial > 0.5) ---")
    high_partial_corr_pairs = analyzer.get_highly_correlated_pairs(method='partial', threshold=0.5)
    print(high_partial_corr_pairs.head(10))

    # 5. Calculate and Correct P-values
    print("\n--- Correlation Significance ---")
    raw_p_values = analyzer.calculate_correlation_significance(pearson_corr)
    print("\nRaw P-values (first 5x5):\n", raw_p_values.iloc[:5, :5])

    # Corrected P-values (FDR-BH)
    corrected_p_values_fdr = analyzer.correct_pvalues_for_multiple_testing(raw_p_values, method='fdr_bh')
    print("\nCorrected P-values (FDR-BH, first 5x5):\n", corrected_p_values_fdr.iloc[:5, :5])

    # Corrected P-values (Bonferroni)
    corrected_p_values_bonf = analyzer.correct_pvalues_for_multiple_testing(raw_p_values, method='bonferroni')
    print("\nCorrected P-values (Bonferroni, first 5x5):\n", corrected_p_values_bonf.iloc[:5, :5])

    # How many significant correlations at alpha=0.05 after correction?
    print(f"\nSignificant Pearson correlations (alpha=0.05, FDR-BH): {(corrected_p_values_fdr < 0.05).sum().sum() / 2}")
    print(
        f"Significant Pearson correlations (alpha=0.05, Bonferroni): {(corrected_p_values_bonf < 0.05).sum().sum() / 2}")

    # 6. Calculate Rolling Correlation
    print("\n--- Rolling Correlation ---")
    rolling_pearson = analyzer.calculate_rolling_correlation(window=60, method='pearson')
    print(f"Number of rolling correlation matrices: {len(rolling_pearson)}")
    if rolling_pearson:
        print("First rolling correlation matrix (last 5x5):\n", list(rolling_pearson.values())[0].iloc[:5, :5])
        print("Last rolling correlation matrix (last 5x5):\n", list(rolling_pearson.values())[-1].iloc[:5, :5])

    # 7. Plotting
    print("\n--- Generating Plots ---")
    fig_pearson = analyzer.plot_correlation_matrix('pearson', title='Pearson Correlation Matrix')
    # fig_pearson.show() # Uncomment to display interactive Plotly figure

    fig_partial = analyzer.plot_correlation_matrix('partial', title='Partial Correlation Matrix')
    # fig_partial.show() # Uncomment to display interactive Plotly figure

    # Matplotlib/Seaborn clustered heatmap
    analyzer.plot_clustered_correlation_matrix('pearson', title='Clustered Pearson Correlation Heatmap')
    # plt.show() # Make sure to call this if not in an interactive environment

    print("\n--- Script finished ---")


if __name__ == "__main__":
    main()