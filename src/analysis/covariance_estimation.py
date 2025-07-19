import pandas as pd
import numpy as np
from abc import ABC, abstractmethod
from sklearn.covariance import GraphicalLasso, MinCovDet, LedoitWolf
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import plotly.graph_objects as go
from typing import List, Optional, Union, Dict
from config.logging_config import logger  # Assuming this is configured


class BaseCovariance(ABC):
    """Abstract base class for covariance estimation methods."""

    def __init__(self):
        self._covariance: Optional[np.ndarray] = None
        self._correlation: Optional[np.ndarray] = None
        self._column_names: Optional[List[str]] = None

    def _validate_and_prepare(self, returns: pd.DataFrame) -> pd.DataFrame:
        """
        Validates input and returns a cleaned copy of the returns DataFrame.
        Handles NaNs by dropping rows with any NaNs.
        """
        if not isinstance(returns, pd.DataFrame) or returns.empty:
            raise ValueError("Input must be a non-empty pandas DataFrame.")

        returns_clean = returns.dropna(how='any').copy()

        if returns_clean.empty:
            raise ValueError("Input DataFrame is empty after dropping NaN values. Cannot estimate covariance.")

        if returns_clean.shape[1] < 2:
            raise ValueError(
                f"Input DataFrame must contain at least two assets (columns) for covariance estimation. Found {returns_clean.shape[1]}.")

        if returns_clean.shape[0] < 2:
            raise ValueError(
                f"Input DataFrame must contain at least two time observations (rows) for covariance estimation. Found {returns_clean.shape[0]}.")

        # Warning for N > T, which can lead to singular covariance matrices
        if returns_clean.shape[0] < returns_clean.shape[1]:
            logger.warning(
                f"Number of observations (T={returns_clean.shape[0]}) is less than number of assets (N={returns_clean.shape[1]}). "
                "The sample covariance matrix will be singular. Consider using shrinkage or robust estimators.")

        self._column_names = returns_clean.columns.tolist()
        return returns_clean

    @abstractmethod
    def fit(self, returns: pd.DataFrame) -> 'BaseCovariance':
        """Fit the covariance estimation model."""
        pass

    def _finalize_estimation(self, covariance: np.ndarray):
        """
        Ensures the final matrix is numerically stable (positive semi-definite)
        and computes correlation matrix.
        """
        # Ensure matrix is symmetric
        covariance = (covariance + covariance.T) / 2

        # Ensure positive semi-definite by adjusting eigenvalues
        try:
            eigvals, eigvecs = np.linalg.eigh(covariance)
            min_eig = np.min(eigvals)
            if min_eig < 1e-12:  # Check if eigenvalues are negative or very close to zero
                logger.warning(
                    f"Covariance matrix is not positive semi-definite (min_eig={min_eig:.2e}). Adjusting eigenvalues.")
                # Add a small epsilon to negative/zero eigenvalues
                eigvals[eigvals < 1e-12] = 1e-12
                covariance = eigvecs @ np.diag(eigvals) @ eigvecs.T
                # Ensure symmetry after reconstruction
                covariance = (covariance + covariance.T) / 2
        except np.linalg.LinAlgError as e:
            logger.error(f"Linear algebra error during eigenvalue decomposition for PSD adjustment: {e}. "
                         "Covariance matrix might be ill-conditioned.")
            # Fallback for severe cases, adding a diagonal to ensure PSD
            covariance += 1e-6 * np.eye(covariance.shape[0])

        self._covariance = covariance

        # Compute correlation matrix
        std = np.sqrt(np.diag(self._covariance))
        outer_std = np.outer(std, std)

        # Avoid division by zero for constant series (where std might be 0)
        # Using np.divide with 'out' and 'where' ensures division by zero results in 0 (or NaN if preferred)
        self._correlation = np.divide(self._covariance, outer_std,
                                      out=np.zeros_like(self._covariance),
                                      where=outer_std != 0)
        np.fill_diagonal(self._correlation, 1)  # Ensure diagonal is 1 for correlation matrix

    def get_covariance(self) -> Optional[pd.DataFrame]:
        """Returns the estimated covariance matrix as a DataFrame."""
        if self._covariance is None:
            raise RuntimeError("Estimator has not been fitted. Call .fit() first.")
        return pd.DataFrame(self._covariance, index=self._column_names, columns=self._column_names)

    def get_correlation(self) -> Optional[pd.DataFrame]:
        """Returns the estimated correlation matrix as a DataFrame."""
        if self._correlation is None:
            raise RuntimeError("Estimator has not been fitted. Call .fit() first.")
        return pd.DataFrame(self._correlation, index=self._column_names, columns=self._column_names)

    def plot_matrix(self, matrix_type: str = 'covariance', title: Optional[str] = None) -> go.Figure:
        """
        Generates an interactive heatmap of the estimated covariance or correlation matrix.

        Args:
            matrix_type (str): 'covariance' or 'correlation'.
            title (Optional[str]): Custom title for the plot. If None, a default title is generated.

        Returns:
            go.Figure: A Plotly Figure object.
        """
        if matrix_type == 'covariance':
            matrix = self.get_covariance()
            default_title = "Estimated Covariance Matrix"
            colorscale = 'Viridis'  # Good for positive values
            zmid = None
        elif matrix_type == 'correlation':
            matrix = self.get_correlation()
            default_title = "Estimated Correlation Matrix"
            colorscale = 'RdBu'  # Good for symmetric values around zero
            zmid = 0  # Center colorscale at zero for correlation
        else:
            raise ValueError("matrix_type must be 'covariance' or 'correlation'.")

        if matrix is None:
            raise RuntimeError(f"No {matrix_type} matrix available. Call .fit() first.")

        fig = go.Figure(data=go.Heatmap(
            z=matrix.values,
            x=matrix.columns,
            y=matrix.columns,
            colorscale=colorscale,
            zmid=zmid
        ))
        fig.update_layout(
            title=title or default_title,
            height=800,
            width=800,
            xaxis_nticks=len(matrix.columns),
            yaxis_nticks=len(matrix.columns)
        )
        return fig


class StandardCovariance(BaseCovariance):
    """
    Standard sample covariance estimator.

    Use Case: Best for datasets where the number of observations (T) is much larger
    than the number of assets (N). Can be unstable otherwise.
    """

    def fit(self, returns: pd.DataFrame) -> 'StandardCovariance':
        returns = self._validate_and_prepare(returns)
        covariance = returns.cov().values
        self._finalize_estimation(covariance)
        logger.info("StandardCovariance estimated.")
        return self


class EWMACovariance(BaseCovariance):
    """
    Exponentially Weighted Moving Average (EWMA) covariance estimator.

    Use Case: Gives more weight to recent observations, making it adaptive to
    changing market conditions. Excellent for capturing time-varying volatility and correlations.

    Args:
        span (int): The span for the EWMA calculation (similar to number of periods for SMA).
                    Alternatively, `alpha` can be provided.
        alpha (float, optional): The decay factor for EWMA (0 < alpha <= 1).
                                 If provided, span is calculated as 2/alpha - 1.
                                 If both are provided, span takes precedence.
    """

    def __init__(self, span: Optional[int] = 60, alpha: Optional[float] = None):
        super().__init__()
        if span is None and alpha is None:
            raise ValueError("Either 'span' or 'alpha' must be provided for EWMACovariance.")
        if span is not None:
            self.span = span
        elif alpha is not None:
            if not (0 < alpha <= 1):
                raise ValueError("Alpha must be between 0 (exclusive) and 1 (inclusive).")
            self.span = (2 / alpha) - 1  # Convert alpha to span
        else:
            self.span = 60  # Default if somehow both are None and passed initial checks

    def fit(self, returns: pd.DataFrame) -> 'EWMACovariance':
        returns = self._validate_and_prepare(returns)

        # EWMA covariance needs to be based on the latest observation in the series
        # returns.ewm(span=self.span).cov() computes a rolling EWMA covariance.
        # We need the last one.
        ewma_cov_series = returns.ewm(span=self.span, adjust=False).cov()

        # The result of ewm().cov() is a multi-indexed DataFrame.
        # We need to extract the last full covariance matrix.
        # The last unique level 0 index will be the date of the last matrix.
        last_date = ewma_cov_series.index.get_level_values(0).unique()[-1]
        covariance = ewma_cov_series.loc[last_date].values

        self._finalize_estimation(covariance)
        logger.info(f"EWMACovariance estimated with span={self.span:.2f}.")
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
        lw = LedoitWolf(assume_centered=False)  # assume_centered=False is default, works for non-zero mean returns
        lw.fit(returns)
        self._finalize_estimation(lw.covariance_)
        logger.info("LedoitWolfShrinkage estimated.")
        return self


class GraphicalLassoCovariance(BaseCovariance):
    """
    Sparse covariance estimation using the Graphical Lasso (L1 penalty).
    Estimates a sparse inverse covariance matrix (precision matrix).

    Use Case: Ideal for high-dimensional problems (many assets, N > T or N ~ T) where it's assumed
    that many pairwise conditional dependencies (partial correlations) are zero. Helps uncover sparse
    underlying dependency structures.

    Args:
        alpha (float): The regularization parameter. Higher alpha means more sparsity (more zeros
                       in the inverse covariance matrix, corresponding to conditional independence).
        max_iter (int): Maximum number of iterations for the optimization algorithm.
    """

    def __init__(self, alpha: float = 0.01, max_iter: int = 100):
        super().__init__()
        if not (alpha >= 0):
            raise ValueError("Alpha must be non-negative.")
        if not (max_iter > 0):
            raise ValueError("Max_iter must be positive.")
        self.alpha = alpha
        self.max_iter = max_iter

    def fit(self, returns: pd.DataFrame) -> 'GraphicalLassoCovariance':
        returns = self._validate_and_prepare(returns)

        # GraphicalLasso assumes data is centered and scaled for optimal alpha interpretation
        # We transform it and then scale back the resulting covariance
        scaler = StandardScaler()
        scaled_returns = scaler.fit_transform(returns)

        model = GraphicalLasso(alpha=self.alpha, assume_centered=True, max_iter=self.max_iter,
                               tol=1e-4)  # tol for convergence

        try:
            model.fit(scaled_returns)
        except Exception as e:
            logger.error(f"GraphicalLasso failed to fit: {e}. Returning NaN covariance.")
            self._finalize_estimation(np.full((returns.shape[1], returns.shape[1]), np.nan))
            return self

        # The covariance_ attribute of GraphicalLasso is the estimated covariance matrix
        # of the *scaled* data. We need to unscale it to get the covariance of original returns.
        # Unscaling: Sigma_orig = D @ Sigma_scaled @ D
        # where D is a diagonal matrix of standard deviations of original data.
        scales = np.diag(returns.std().values)
        covariance = scales @ model.covariance_ @ scales

        self._finalize_estimation(covariance)
        logger.info(f"GraphicalLassoCovariance estimated with alpha={self.alpha}.")
        return self


class ResidualCovarianceFromRegression(BaseCovariance):
    """
    Estimates covariance matrix based on the residuals from a linear regression.
    This approach implicitly assumes a factor model where each asset's returns are
    explained by the returns of other assets, and the residual covariance captures
    the idiosyncratic risk.

    Note: For a true common factor model, explicit factors should be used.
          This method regresses each asset on all other assets.

    Use Case: Can be used as a simple denoising method, isolating the idiosyncratic
              component of risk by regressing out commonalities with other assets.
              Less common than explicit factor models but offers a different perspective.
    """

    def fit(self, returns: pd.DataFrame) -> 'ResidualCovarianceFromRegression':
        returns = self._validate_and_prepare(returns)

        residuals = pd.DataFrame(index=returns.index)

        if returns.shape[1] == 1:
            logger.warning("Only one asset provided. Residual covariance is just its variance.")
            covariance = np.array([[returns.var().iloc[0]]])
            self._finalize_estimation(covariance)
            return self

        for col_name in returns.columns:
            y = returns[col_name]
            X_cols = [c for c in returns.columns if c != col_name]
            X = returns[X_cols]

            # Ensure X has enough features and samples for regression
            if X.empty or X.shape[0] < X.shape[1] + 2:  # N_samples < N_features + 2 (for intercept and 1 feature)
                logger.warning(
                    f"Not enough data points or too many features for regression for asset {col_name}. Skipping regression.")
                residuals[col_name] = y  # If regression fails, consider residual as original returns
                continue

            try:
                model = LinearRegression(fit_intercept=True)  # Fit with intercept
                model.fit(X, y)
                residuals[col_name] = y - model.predict(X)
            except Exception as e:
                logger.error(
                    f"Error performing OLS regression for {col_name} to get residuals: {e}. Using original returns as proxy for residuals.")
                residuals[col_name] = y  # Fallback if regression fails

        covariance = residuals.cov().values
        self._finalize_estimation(covariance)
        logger.info("ResidualCovarianceFromRegression estimated.")
        return self


class RobustCovariance(BaseCovariance):
    """
    Robust covariance estimation using Minimum Covariance Determinant (MCD).

    Use Case: Excellent for datasets with significant outliers (e.g., during market crashes).
    It fits the covariance to a subset of "normal" data points, ignoring the outliers.

    Args:
        contamination (float): The expected proportion of outliers in the data (between 0 and 0.5).
                               Used to set `support_fraction` for MinCovDet.
        random_state (int, optional): Seed for the random number generator for reproducibility.
    """

    def __init__(self, contamination: float = 0.1, random_state: int = 42):
        super().__init__()
        if not (0 <= contamination < 0.5):
            raise ValueError("Contamination must be between 0 (inclusive) and 0.5 (exclusive).")
        self.contamination = contamination
        self.random_state = random_state

    def fit(self, returns: pd.DataFrame) -> 'RobustCovariance':
        returns = self._validate_and_prepare(returns)

        # MinCovDet needs at least N_features + 1 observations.
        # Its default support_fraction is None, meaning (n_samples + n_features + 1) / 2
        # If explicitly setting support_fraction, it should be > (n_features / n_samples)
        # And min_samples = n_features + 1

        n_samples, n_features = returns.shape
        # Default support_fraction logic from sklearn documentation
        min_support_fraction = (n_features + 1) / n_samples

        # Adjust support_fraction based on contamination, ensuring it's valid
        support_fraction = 1 - self.contamination
        if support_fraction < min_support_fraction:
            logger.warning(
                f"Calculated support_fraction ({support_fraction:.2f}) from contamination is too low for current data (min required {min_support_fraction:.2f}). Adjusting to minimum required.")
            support_fraction = min_support_fraction

        mcd = MinCovDet(support_fraction=support_fraction, random_state=self.random_state, assume_centered=False)

        try:
            mcd.fit(returns)
        except Exception as e:
            logger.error(f"MinCovDet (RobustCovariance) failed to fit: {e}. Returning NaN covariance.")
            self._finalize_estimation(np.full((n_features, n_features), np.nan))
            return self

        self._finalize_estimation(mcd.covariance_)
        logger.info(f"RobustCovariance estimated with contamination={self.contamination}.")
        return self


# --- Example Usage ---
def main():
    # 1. Generate Sample Data
    np.random.seed(42)
    dates = pd.date_range(start='2020-01-01', periods=100)
    num_assets = 5

    # Create synthetic returns data
    # Normal assets
    returns_data = np.random.normal(0.001, 0.02, (len(dates), num_assets))
    # Introduce some correlation
    returns_data[:, 0] = returns_data[:, 1] * 0.5 + returns_data[:, 0] * 0.5  # Asset 0 and 1 are correlated
    returns_data[:, 2] = returns_data[:, 3] * 0.7 + returns_data[:, 2] * 0.3  # Asset 2 and 3 are correlated

    returns_df = pd.DataFrame(returns_data, index=dates, columns=[f'Asset_{i}' for i in range(num_assets)])

    # Introduce some outliers for RobustCovariance testing
    returns_df.iloc[20:25, [0, 1]] += 0.1  # Outliers

    # Introduce some NaNs at the beginning and middle for _validate_and_prepare testing
    returns_df.iloc[0, 0] = np.nan
    returns_df.iloc[50, 2] = np.nan
    returns_df.iloc[51, 3] = np.nan

    print("--- Original Returns Data ---")
    print(returns_df.head())
    print(f"Original Data shape: {returns_df.shape}")

    estimators = {
        "Standard": StandardCovariance(),
        "EWMA (span=30)": EWMACovariance(span=30),
        "EWMA (alpha=0.1)": EWMACovariance(alpha=0.1),
        "Ledoit-Wolf": LedoitWolfShrinkage(),
        "Graphical Lasso (alpha=0.05)": GraphicalLassoCovariance(alpha=0.05),
        "Residual from Regression": ResidualCovarianceFromRegression(),
        "Robust (MCD, cont=0.05)": RobustCovariance(contamination=0.05)
    }

    results = {}

    for name, estimator in estimators.items():
        print(f"\n--- Estimating with {name} ---")
        try:
            estimator.fit(returns_df.copy())  # Pass a copy to ensure each estimator gets fresh data
            results[name] = {
                "covariance": estimator.get_covariance(),
                "correlation": estimator.get_correlation()
            }
            print(f"{name} Covariance (first 3x3):\n", results[name]["covariance"].iloc[:3, :3])
            print(f"{name} Correlation (first 3x3):\n", results[name]["correlation"].iloc[:3, :3])

            # Plotting example
            # fig_cov = estimator.plot_matrix(matrix_type='covariance', title=f'{name} Covariance Matrix')
            # fig_cov.show()

            # fig_corr = estimator.plot_matrix(matrix_type='correlation', title=f'{name} Correlation Matrix')
            # fig_corr.show()

        except ValueError as e:
            logger.error(f"Failed to estimate with {name} due to input error: {e}")
        except RuntimeError as e:
            logger.error(f"Runtime error with {name}: {e}")
        except Exception as e:
            logger.error(f"An unexpected error occurred with {name}: {e}")

    # Test edge cases for validation
    print("\n--- Testing validation edge cases ---")

    # Test empty DataFrame
    try:
        StandardCovariance().fit(pd.DataFrame())
    except ValueError as e:
        print(f"Caught expected error for empty DataFrame: {e}")

    # Test DataFrame with NaNs that leads to empty after dropna
    returns_all_nan = pd.DataFrame(np.nan, index=dates, columns=['A', 'B'])
    try:
        StandardCovariance().fit(returns_all_nan)
    except ValueError as e:
        print(f"Caught expected error for all-NaN DataFrame: {e}")

    # Test single asset
    returns_single_asset = returns_df[['Asset_0']]
    try:
        StandardCovariance().fit(returns_single_asset)
    except ValueError as e:
        print(f"Caught expected error for single asset DataFrame: {e}")

    # Test too few observations (N > T)
    returns_few_obs = returns_df.iloc[:2]  # Only 2 observations, but 5 assets
    try:
        StandardCovariance().fit(returns_few_obs)  # Should log warning, but proceed
        print("StandardCovariance with few observations (N>T) passed (check log for warning).")
        # Try a specific estimator that might fail or be unstable
        # GraphicalLasso can be sensitive to N>T for certain alphas
        GraphicalLassoCovariance(alpha=0.01).fit(returns_few_obs)
        print("GraphicalLasso with few observations (N>T) passed (check log for warning).")
    except ValueError as e:
        print(f"Caught unexpected error for few observations DataFrame: {e}")
    except RuntimeError as e:
        print(f"Caught RuntimeError for few observations DataFrame: {e}")


if __name__ == "__main__":
    main()