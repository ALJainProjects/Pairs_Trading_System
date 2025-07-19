import pandas as pd
import numpy as np
import pywt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from typing import Optional, List, Union
from config.logging_config import logger

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

    def fit(self, returns: pd.DataFrame) -> 'ReturnDenoiser':
        """Fit the denoiser with the original return data."""
        if not isinstance(returns, pd.DataFrame):
            raise ValueError("Input must be a pandas DataFrame.")
        self.original_returns = returns.copy()
        return self

    def denoise_wavelet(self, wavelet: str = 'db4', level: int = 1) -> pd.DataFrame:
        """
        Applies wavelet transform to denoise the return series.

        Guidance: 'db4' or 'sym4' are often good starting points. The level determines
        the scale of noise being removed; lower levels remove higher-frequency noise.
        """
        denoised_data = {}
        for col in self.original_returns.columns:
            series = self.original_returns[col].dropna()
            coeffs = pywt.wavedec(series, wavelet, level=level)
            sigma = np.median(np.abs(coeffs[-1])) / 0.6745
            threshold = sigma * np.sqrt(2 * np.log(len(series)))

            new_coeffs = [pywt.threshold(c, threshold, mode='soft') for c in coeffs]
            reconstructed = pywt.waverec(new_coeffs, wavelet)
            denoised_data[col] = reconstructed[:len(self.original_returns)]

        return pd.DataFrame(denoised_data, index=self.original_returns.index)

    def denoise_pca(self, n_components: Union[int, float] = 0.95) -> pd.DataFrame:
        """
        Denoises data by reconstructing it using a specified number of principal components.

        Guidance: n_components can be an int (number of components) or a float
        (fraction of variance to explain, e.g., 0.95). Using a float is often more robust.
        """
        scaler = StandardScaler()
        scaled_returns = scaler.fit_transform(self.original_returns)

        pca = PCA(n_components=n_components, random_state=self.random_state)
        principal_components = pca.fit_transform(scaled_returns)
        reconstructed = pca.inverse_transform(principal_components)

        denoised = scaler.inverse_transform(reconstructed)
        return pd.DataFrame(denoised, index=self.original_returns.index, columns=self.original_returns.columns)

    def denoise_kalman(self, process_variance: float = 1e-5, measurement_variance: float = 1e-2) -> pd.DataFrame:
        """
        Denoises each return series individually using a Kalman Filter.

        Guidance: The ratio of process_variance to measurement_variance controls smoothness.
        A smaller ratio results in a smoother (more denoised) series.
        """
        if KalmanFilter is None:
            raise ImportError("filterpy must be installed to use Kalman denoising.")

        denoised_data = {}
        for col in self.original_returns.columns:
            kf = KalmanFilter(dim_x=1, dim_z=1)
            kf.x = np.array([0.])
            kf.P = 1.
            kf.R = measurement_variance
            kf.Q = process_variance
            kf.H = np.array([[1.]])
            kf.F = np.array([[1.]])

            measurements = self.original_returns[col].fillna(0).values
            filtered_state_means, _ = kf.filter(measurements)

            denoised_data[col] = filtered_state_means.flatten()

        return pd.DataFrame(denoised_data, index=self.original_returns.index)