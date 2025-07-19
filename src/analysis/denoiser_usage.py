import pandas as pd
import numpy as np
import pywt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from typing import Optional, List, Union
from config.logging_config import logger  # Assuming configured

try:
    from filterpy.kalman import KalmanFilter
except ImportError:
    logger.error("filterpy is not installed. Run 'pip install filterpy'.")
    KalmanFilter = None


class ReturnDenoiser:
    """
    A class for denoising asset return series using various techniques to
    improve the signal-to-noise ratio for downstream modeling.
    """

    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.original_returns: Optional[pd.DataFrame] = None
        self._fitted = False  # Internal flag to check if fit has been called

    def _validate_input_for_fit(self, returns: pd.DataFrame):
        """
        Validates the input DataFrame during the fit process.
        This method is for initial data checks and preparation.
        """
        if not isinstance(returns, pd.DataFrame):
            raise ValueError("Input must be a pandas DataFrame.")
        if returns.empty:
            raise ValueError("Input DataFrame is empty.")
        if returns.shape[0] < 2:
            raise ValueError("Input DataFrame must contain at least two time observations (rows).")
        if returns.shape[1] < 1:
            raise ValueError("Input DataFrame must contain at least one asset (column).")

        # Consistent NaN handling for the entire DataFrame during fit
        # We drop rows with any NaN values to ensure common index for PCA
        # For Kalman/Wavelet, individual series are handled, but PCA needs complete cases.
        if returns.isnull().any().any():
            initial_rows = returns.shape[0]
            returns_clean = returns.dropna(how='any')
            if returns_clean.empty:
                raise ValueError("Input DataFrame becomes empty after dropping NaN values. Cannot fit denoiser.")
            logger.warning(
                f"Input returns contained NaNs. Dropped {initial_rows - returns_clean.shape[0]} rows with NaNs during fit.")
            return returns_clean

        return returns.copy()  # Return a copy to avoid modifying original input DF

    def _check_is_fitted(self):
        """Raises an error if the denoiser has not been fitted."""
        if not self._fitted or self.original_returns is None:
            raise RuntimeError("Denoiser has not been fitted. Call .fit(returns) first.")

    def fit(self, returns: pd.DataFrame) -> 'ReturnDenoiser':
        """
        Fits the denoiser with the original return data.
        Performs initial validation and NaN handling (drops rows with any NaNs).
        """
        self.original_returns = self._validate_input_for_fit(returns)
        self._fitted = True
        logger.info(
            f"ReturnDenoiser fitted with {self.original_returns.shape[0]} observations and {self.original_returns.shape[1]} assets after NaN handling.")
        return self

    def denoise_wavelet(self, wavelet: str = 'db4', level: int = 1) -> pd.DataFrame:
        """
        Applies wavelet transform to denoise each return series individually.
        Handles NaNs by dropping them per series before transformation, and then
        realigns the denoised series to the original DataFrame's index.

        Args:
            wavelet (str): The name of the wavelet to use (e.g., 'db4', 'sym4').
            level (int): The decomposition level. Lower levels remove higher-frequency noise.

        Returns:
            pd.DataFrame: A DataFrame of the denoised return series, with the same
                          index and columns as the original fitted data. NaNs will be
                          reintroduced where original NaNs were present.
        """
        self._check_is_fitted()

        denoised_data = {}
        original_indices = self.original_returns.index
        original_columns = self.original_returns.columns

        for col in original_columns:
            series = self.original_returns[col]
            series_clean = series.dropna()

            if series_clean.empty:
                logger.warning(f"Column '{col}' is all NaNs after dropping. Skipping wavelet denoising.")
                denoised_data[col] = pd.Series(np.nan, index=original_indices)
                continue

            # pywt.wavedec requires input length to be power of 2 for exact reconstruction
            # or pad=True to handle arbitrary lengths, but generally handles it.
            # However, for correct reconstruction, length should match.
            # Using discrete wavelet transform for approximation and detail coefficients
            coeffs = pywt.wavedec(series_clean, wavelet, level=level)

            # Universal threshold (VisuShrink)
            # sigma: median absolute deviation for robust noise estimation
            sigma = np.median(np.abs(coeffs[-1])) / 0.6745
            threshold = sigma * np.sqrt(2 * np.log(len(series_clean)))

            # Apply soft thresholding
            new_coeffs = [pywt.threshold(c, threshold, mode='soft') for c in coeffs]

            # Reconstruct the series
            reconstructed = pywt.waverec(new_coeffs, wavelet)

            # Align reconstructed data back to the original index
            # This handles cases where original series had NaNs, re-inserting them
            temp_series = pd.Series(reconstructed, index=series_clean.index)
            denoised_data[col] = temp_series.reindex(original_indices)

        return pd.DataFrame(denoised_data, index=original_indices, columns=original_columns)

    def denoise_pca(self, n_components: Union[int, float] = 0.95) -> pd.DataFrame:
        """
        Denoises data by reconstructing it using a specified number of principal components.
        Assumes `original_returns` has no NaNs (handled by `_validate_input_for_fit`).

        Args:
            n_components (Union[int, float]): Number of principal components to keep.
                                              If int, the number of components.
                                              If float (0.0 to 1.0), the fraction of variance to explain.

        Returns:
            pd.DataFrame: A DataFrame of the denoised return series, with the same
                          index and columns as the original fitted data.
        """
        self._check_is_fitted()

        # Input to PCA should be scaled
        scaler = StandardScaler()
        scaled_returns = scaler.fit_transform(self.original_returns)

        # PCA for dimensionality reduction
        pca = PCA(n_components=n_components, random_state=self.random_state)
        principal_components = pca.fit_transform(scaled_returns)

        logger.info(
            f"PCA denoising: Explained variance by selected components: {np.sum(pca.explained_variance_ratio_):.2%}")

        # Reconstruct data from selected principal components
        reconstructed = pca.inverse_transform(principal_components)

        # Inverse transform back to original scale
        denoised = scaler.inverse_transform(reconstructed)
        return pd.DataFrame(denoised, index=self.original_returns.index, columns=self.original_returns.columns)

    def denoise_kalman(self, process_variance: float = 1e-5, measurement_variance: float = 1e-2) -> pd.DataFrame:
        """
        Denoises each return series individually using a Kalman Filter.
        Handles NaNs by filling them temporarily during filtering and then
        reintroducing original NaNs in the output.

        Args:
            process_variance (float): The variance of the underlying process noise (Q).
                                      A smaller value implies a smoother underlying process.
            measurement_variance (float): The variance of the measurement noise (R).
                                          A smaller value implies more trust in observations.

        Returns:
            pd.DataFrame: A DataFrame of the denoised return series, with the same
                          index and columns as the original fitted data. NaNs will be
                          reintroduced where original NaNs were present.
        """
        self._check_is_fitted()
        if KalmanFilter is None:
            raise ImportError("filterpy must be installed to use Kalman denoising.")

        if process_variance <= 0 or measurement_variance <= 0:
            raise ValueError("Process and measurement variances must be positive.")

        denoised_data = {}
        original_indices = self.original_returns.index
        original_columns = self.original_returns.columns

        for col in original_columns:
            series = self.original_returns[col]
            series_clean = series.dropna()  # Use only non-NaN values for filter operation

            if series_clean.empty:
                logger.warning(f"Column '{col}' is all NaNs after dropping. Skipping Kalman denoising.")
                denoised_data[col] = pd.Series(np.nan, index=original_indices)
                continue

            # Initialize Kalman Filter
            kf = KalmanFilter(dim_x=1, dim_z=1)

            # Initial state estimate (x0) and its covariance (P0)
            kf.x = np.array([series_clean.iloc[0]])  # Initialize with first non-NaN measurement
            kf.P = measurement_variance  # Initial uncertainty, can be tuned

            # Measurement noise covariance
            kf.R = measurement_variance
            # Process noise covariance
            kf.Q = process_variance

            # State transition matrix (identity for random walk model)
            kf.F = np.array([[1.]])
            # Measurement function (maps state to measurement)
            kf.H = np.array([[1.]])

            # Filter the measurements
            measurements = series_clean.values.astype(float)

            # Perform filtering
            filtered_state_means, _ = kf.filter(measurements)

            # Align filtered data back to the original index
            temp_series = pd.Series(filtered_state_means.flatten(), index=series_clean.index)
            denoised_data[col] = temp_series.reindex(original_indices)

        return pd.DataFrame(denoised_data, index=original_indices, columns=original_columns)


# --- Example Usage ---
def main():
    # 1. Generate Sample Data
    np.random.seed(42)
    dates = pd.date_range(start='2020-01-01', periods=200)
    num_assets = 5

    # Create synthetic returns data with underlying signal + noise
    # Base signal
    signal = np.sin(np.linspace(0, 4 * np.pi, len(dates))) * 0.01 + np.random.normal(0, 0.001, len(dates)).cumsum()

    # Returns = signal + noise
    returns_data = np.outer(signal, np.ones(num_assets)) + np.random.normal(0, 0.01, (len(dates), num_assets))

    returns_df = pd.DataFrame(returns_data, index=dates, columns=[f'Asset_{i}' for i in range(num_assets)])

    # Introduce some NaNs to test robustness
    returns_df.iloc[10:15, 0] = np.nan
    returns_df.iloc[50, 2] = np.nan
    returns_df.iloc[100:105, 3] = np.nan
    returns_df.iloc[150:, 4] = np.nan  # Trailing NaNs
    returns_df.iloc[5:8, :] = np.nan  # Rows with all NaNs

    print("--- Original Returns Data ---")
    print(returns_df.head(10))
    print(f"Original Data shape: {returns_df.shape}")
    print(f"Original NaNs:\n{returns_df.isnull().sum()}")

    # 2. Initialize and Fit Denoiser
    denoiser = ReturnDenoiser(random_state=42)

    try:
        denoiser.fit(returns_df.copy())  # Pass a copy to ensure original is not modified
    except ValueError as e:
        print(f"Denoiser Fit Error: {e}")
        return
    except Exception as e:
        print(f"An unexpected error during fit: {e}")
        return

    # 3. Apply Denoising Techniques
    print("\n--- Applying Denoising ---")

    denoised_wavelet = None
    denoised_pca = None
    denoised_kalman = None

    # Wavelet Denoising
    try:
        denoised_wavelet = denoiser.denoise_wavelet(wavelet='db4', level=2)
        print("\nWavelet Denoised (head):\n", denoised_wavelet.head(10))
        print("Wavelet Denoised NaNs:\n", denoised_wavelet.isnull().sum())
    except Exception as e:
        logger.error(f"Wavelet Denoising Failed: {e}")

    # PCA Denoising
    try:
        denoised_pca = denoiser.denoise_pca(n_components=0.90)  # Explain 90% variance
        print("\nPCA Denoised (head):\n", denoised_pca.head(10))
        print("PCA Denoised NaNs:\n",
              denoised_pca.isnull().sum())  # Should have no NaNs if original_returns was cleaned
    except Exception as e:
        logger.error(f"PCA Denoising Failed: {e}")

    # Kalman Filter Denoising
    try:
        denoised_kalman = denoiser.denoise_kalman(process_variance=1e-6, measurement_variance=1e-1)
        print("\nKalman Denoised (head):\n", denoised_kalman.head(10))
        print("Kalman Denoised NaNs:\n", denoised_kalman.isnull().sum())
    except ImportError as e:
        print(f"Skipping Kalman Denoising: {e}")
    except Exception as e:
        logger.error(f"Kalman Denoising Failed: {e}")

    # Compare original vs denoised for a single asset visually (optional, requires plotting library)
    import matplotlib.pyplot as plt
    if 'Asset_0' in returns_df.columns:
        plt.figure(figsize=(12, 6))
        plt.plot(returns_df['Asset_0'], label='Original Asset_0', alpha=0.7)
        if 'denoised_wavelet' in locals():
            plt.plot(denoised_wavelet['Asset_0'], label='Wavelet Denoised Asset_0', linestyle='--')
        if 'denoised_pca' in locals():
            plt.plot(denoised_pca['Asset_0'], label='PCA Denoised Asset_0', linestyle='-.')
        if 'denoised_kalman' in locals():
            plt.plot(denoised_kalman['Asset_0'], label='Kalman Denoised Asset_0', linestyle=':')
        plt.title('Original vs. Denoised Returns for Asset_0')
        plt.legend()
        plt.grid(True)
        plt.show()

    print("\n--- Testing edge cases/errors ---")

    # Test unfitted denoiser
    unfitted_denoiser = ReturnDenoiser()
    try:
        unfitted_denoiser.denoise_pca()
    except RuntimeError as e:
        print(f"Caught expected error for unfitted denoiser: {e}")

    # Test empty DataFrame after NaN drop (simulated)
    empty_after_nan_df = pd.DataFrame(np.nan, index=pd.date_range('2020-01-01', periods=5), columns=['A', 'B'])
    try:
        denoiser_empty = ReturnDenoiser()
        denoiser_empty.fit(empty_after_nan_df)
    except ValueError as e:
        print(f"Caught expected error for DataFrame empty after NaN drop: {e}")

    # Test DataFrame with too few observations
    too_few_obs_df = pd.DataFrame(np.random.rand(1, 3), index=[pd.Timestamp('2020-01-01')], columns=['A', 'B', 'C'])
    try:
        denoiser_few_obs = ReturnDenoiser()
        denoiser_few_obs.fit(too_few_obs_df)
    except ValueError as e:
        print(f"Caught expected error for too few observations: {e}")


if __name__ == "__main__":
    main()