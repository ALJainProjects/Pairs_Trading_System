import pandas as pd
import numpy as np
from typing import Dict, Tuple, List, Optional, Union
from pathlib import Path
import joblib  # For general Python object saving if needed

from .base import BaseStrategy
from src.models.deep_learning import TimesNetModel
from src.data.feature_engineering import FeatureEngineer
from config.logging_config import logger

try:
    from arch import arch_model
    # Also import ARCHModelResult for type hinting and specific saving/loading
    from arch.univariate.base import ARCHModelResult
except ImportError:
    logger.error("arch library not installed. Please run 'pip install arch'.")
    arch_model = None
    ARCHModelResult = type(None)  # Define a dummy type for type hinting if arch is not available


class AdvancedMLStrategy(BaseStrategy):
    """
    An advanced pairs trading strategy using a TimesNet model to predict spread
    movements and a GARCH model for adaptive risk management.
    """

    def __init__(self,
                 name: str = "AdvancedMLStrategy_TimesNet",
                 lookback_window: int = 60,
                 sequence_length: int = 30,
                 max_pairs: int = 5,
                 timesnet_epochs: int = 50,
                 timesnet_batch_size: int = 32,
                 timesnet_n_filters: int = 64,
                 timesnet_kernel_size: int = 3,
                 timesnet_stack_size: int = 2,
                 garch_p: int = 1,
                 garch_q: int = 1,
                 model_dir: str = "strategy_models"):  # Directory to save strategy-specific models

        super().__init__(name, lookback_window=lookback_window, sequence_length=sequence_length, max_pairs=max_pairs)

        self.feature_engineer = FeatureEngineer()
        # Stores TimesNetModel instances and GARCH fit results
        self.pair_models: Dict[Tuple[str, str], Dict[str, Union[TimesNetModel, ARCHModelResult]]] = {}
        # This will be populated by a pair selection method in a real scenario
        # For this example, it remains hardcoded for demonstration consistency.
        self.tradeable_pairs: List[Tuple[str, str]] = []

        # Hyperparameters for TimesNet
        self.timesnet_epochs = timesnet_epochs
        self.timesnet_batch_size = timesnet_batch_size
        self.timesnet_n_filters = timesnet_n_filters
        self.timesnet_kernel_size = timesnet_kernel_size
        self.timesnet_stack_size = timesnet_stack_size

        # Hyperparameters for GARCH
        self.garch_p = garch_p
        self.garch_q = garch_q

        # Define model saving paths for each pair
        self._strategy_model_path = Path(model_dir) / self.name
        self._strategy_model_path.mkdir(parents=True, exist_ok=True)
        logger.info(f"{self.name} initialized. Models will be saved/loaded from {self._strategy_model_path}")

    def _train_garch_model(self, spread_series: pd.Series, pair_name: Tuple[str, str]) -> Optional[ARCHModelResult]:
        """
        Trains a GARCH(p,q) model on spread returns for volatility forecasting.
        """
        if arch_model is None:
            logger.warning("ARCH library not available. Skipping GARCH model training.")
            return None

        # GARCH models typically work on returns, not absolute spread values
        spread_returns = spread_series.pct_change().dropna() * 100  # Convert to percentage returns

        # Ensure sufficient data for GARCH fitting
        min_garch_data = 100  # A reasonable minimum for GARCH(1,1)
        if len(spread_returns) < min_garch_data:
            logger.warning(
                f"Insufficient data points ({len(spread_returns)}) for GARCH model training for {pair_name}. Need at least {min_garch_data}. Skipping GARCH.")
            return None

        try:
            # Instantiate and fit GARCH model
            garch = arch_model(spread_returns, vol='Garch', p=self.garch_p, q=self.garch_q, rescale=False)
            garch_fit = garch.fit(update_freq=10, disp='off')
            logger.info(f"GARCH({self.garch_p},{self.garch_q}) model trained for {pair_name}.")
            return garch_fit
        except Exception as e:
            logger.error(f"Error training GARCH model for {pair_name}: {e}")
            return None

    def _train_timesnet_model(self,
                              pair_data: pd.DataFrame,
                              pair_name: Tuple[str, str]) -> Optional[TimesNetModel]:
        """
        Trains a TimesNet model to predict future spread movements.
        """
        # TimesNet expects a 'close' equivalent for its input features
        # Here, the 'spread' itself will be the feature for TimesNet.
        features_df = pd.DataFrame({'Adj_Close': pair_data['spread']}).dropna()
        # Target is the next spread value
        target_series = features_df['Adj_Close'].shift(-1)

        # Align features and target, dropping rows with NaN in either
        features_df, target_series = features_df.align(target_series, join='inner', axis=0)

        # Create sequences for TimesNet
        # TimesNetModel.create_sequences expects pd.DataFrame for features and pd.Series for target
        # and returns np.ndarray
        X, y = TimesNetModel.create_sequences(features_df, target_series, self.params['sequence_length'])

        min_timesnet_data = 50  # A reasonable minimum for deep learning training
        if len(X) < min_timesnet_data:
            logger.warning(
                f"Insufficient sequenced data points ({len(X)}) for TimesNet model training for {pair_name}. Need at least {min_timesnet_data}. Skipping TimesNet.")
            return None

        # Split data for training and validation
        split_idx = int(len(X) * 0.8)
        X_train, X_val = X[:split_idx], X[split_idx:]
        y_train, y_val = y[:split_idx], y[split_idx:]

        timesnet = TimesNetModel(f"TimesNet_{pair_name[0]}_{pair_name[1]}",
                                 model_dir=str(self._strategy_model_path))  # Pass model_dir

        # Build model with specific TimesNet hyperparameters
        timesnet.build_model(
            input_shape=(X_train.shape[1], X_train.shape[2]),
            n_filters=self.timesnet_n_filters,
            kernel_size=self.timesnet_kernel_size,
            stack_size=self.timesnet_stack_size
        )

        # Train TimesNet model
        timesnet.train(X_train, y_train, X_val, y_val,
                       epochs=self.timesnet_epochs, batch_size=self.timesnet_batch_size)

        logger.info(f"TimesNet model trained for {pair_name}.")
        return timesnet

    def fit(self, historical_data: pd.DataFrame):
        """
        Finds pairs (in a real scenario) and trains a dedicated TimesNet and GARCH model for each pair.
        The `historical_data` is assumed to contain 'Adj_Close' for each asset indexed by date.
        """
        logger.info(f"Starting fitting process for {self.name}...")

        # In a production scenario, you would implement a pair selection algorithm here
        # (e.g., cointegration tests, distance methods, machine learning clustering).
        # For this example, we keep them hardcoded for consistency.
        # Ensure these symbols are actually present in historical_data.
        self.tradeable_pairs = [('AAPL', 'MSFT'), ('GOOGL', 'GOOG')]

        available_assets = historical_data.columns.tolist()
        self.tradeable_pairs = [
            pair for pair in self.tradeable_pairs
            if pair[0] in available_assets and pair[1] in available_assets
        ]
        if not self.tradeable_pairs:
            logger.error("No tradeable pairs found in historical data. Aborting fit.")
            return

        for pair in self.tradeable_pairs:
            asset1, asset2 = pair
            logger.info(f"--- Processing pair: {pair} ---")

            # Extract data for the pair, assuming 'Adj_Close' is available for each asset
            # Ensure proper handling of multi-index if historical_data comes in long format
            # For simplicity, assume historical_data is already wide format with asset names as columns
            if (asset1 not in historical_data.columns) or (asset2 not in historical_data.columns):
                logger.warning(f"Assets {asset1} or {asset2} not found in historical_data. Skipping pair {pair}.")
                continue

            # Use 'Adj_Close' if available, otherwise assume the column directly contains prices
            price_col1 = historical_data[asset1] if asset1 in historical_data.columns else historical_data.iloc[:,
                                                                                           historical_data.columns.get_loc(
                                                                                               asset1)]
            price_col2 = historical_data[asset2] if asset2 in historical_data.columns else historical_data.iloc[:,
                                                                                           historical_data.columns.get_loc(
                                                                                               asset2)]

            # Align prices to ensure common index and no NaNs before spread calculation
            temp_df = pd.DataFrame({asset1: price_col1, asset2: price_col2}).dropna()

            if temp_df.empty:
                logger.warning(f"No common non-NaN data for pair {pair}. Skipping model training.")
                continue

            # Calculate spread
            # Note: A static hedge ratio (e.g., from OLS) could be used here.
            # For simplicity, using simple difference as initial spread.
            # For highly sophisticated, you could run StatModel.calculate_rolling_hedge_ratio here
            # and then calculate spread = asset1 - hedge_ratio * asset2
            # For now, stick to direct difference for TimesNet prediction as per original
            pair_data = pd.DataFrame(index=temp_df.index)
            pair_data['spread'] = temp_df[asset1] - temp_df[asset2]

            # Train GARCH model
            garch_model_fit = self._train_garch_model(pair_data['spread'], pair)

            # Train TimesNet model
            timesnet_model = self._train_timesnet_model(pair_data, pair)

            if timesnet_model is None:  # If TimesNet training was skipped
                logger.warning(f"TimesNet model not trained for {pair}. This pair will not generate signals.")
                continue

            self.pair_models[pair] = {'timesnet': timesnet_model, 'garch': garch_model_fit}
            logger.info(f"Finished processing pair: {pair}. Models stored.")

        self.is_fitted = True
        logger.info(f"Fitting process for {self.name} complete. {len(self.pair_models)} pairs have models.")
        # After fitting, save the entire strategy state including all sub-models
        self.save_strategy()

    def generate_signals(self, current_data_window: pd.DataFrame, portfolio_context: Dict) -> Dict[
        Tuple[str, str], float]:
        """
        Generates trading signals for each fitted pair based on predicted spread movement
        and adaptive volatility thresholds.

        Args:
            current_data_window (pd.DataFrame): A DataFrame containing the most recent
                                                'lookback_window' data points for all assets.
                                                Expected columns are asset symbols.
            portfolio_context (Dict): Current portfolio state (e.g., cash, open positions).
                                      Not directly used for signal generation logic in this snippet
                                      but available for more complex rules.

        Returns:
            Dict[Tuple[str, str], float]: A dictionary where keys are pairs (e.g., ('AAPL', 'MSFT'))
                                          and values are signals (-1.0 for short spread, 1.0 for long spread,
                                          0.0 for no trade).
        """
        signals = {}
        if not self.is_fitted:
            logger.warning("Strategy not fitted. No signals can be generated.")
            return signals

        if not self.pair_models:
            logger.warning("No models are available for generating signals. Skipping signal generation.")
            return signals

        for pair, models in self.pair_models.items():
            asset1, asset2 = pair
            timesnet = models.get('timesnet')
            garch_fit = models.get('garch')

            if timesnet is None or not timesnet.is_fitted:
                logger.warning(f"TimesNet model not available or not fitted for {pair}. Skipping signal for this pair.")
                signals[pair] = 0.0
                continue

            # 1. Volatility Forecast from GARCH
            vol_forecast = 1.0  # Default if GARCH fails or not available
            if garch_fit:
                # Need historical returns leading up to current_data_window to get latest GARCH state
                # The GARCH model was trained on historical spread returns.
                # For a true forecast, you'd feed the *current* spread return to the fitted GARCH model.
                # However, `garch_fit.forecast` from the `arch` library uses the *last* available data
                # it was fitted with to make forecasts. If `current_data_window` is simply the
                # *latest* data, `spread_returns` needs to reflect this.

                # Extract the spread from the current window
                current_spread_series = current_data_window[asset1] - current_data_window[asset2]

                # Calculate returns on the current spread window for GARCH
                # Need at least 2 points to calculate a return.
                if len(current_spread_series) > 1:
                    window_spread_returns = current_spread_series.pct_change().dropna() * 100
                    if not window_spread_returns.empty:
                        try:
                            # Use the last actual return in the window to make the next forecast
                            # ARCH's forecast method automatically uses the internal state from fit.
                            # For out-of-sample forecasting, you might need to re-estimate or use rolling forecasts.
                            # For simplicity here, we assume the fitted garch_fit's internal state is sufficient
                            # to provide a relevant forecast based on its last observation during fitting.
                            # To be more rigorous, you might need to update the GARCH model or
                            # pass recent data for a conditional forecast.
                            # For now, just use the built-in forecast on the fitted model.
                            forecast_result = garch_fit.forecast(horizon=1,
                                                                 method='analytic')  # 'analytic' is faster for simple models
                            vol_forecast = forecast_result.variance.iloc[-1, 0]
                            if vol_forecast <= 0:  # Ensure volatility is positive
                                vol_forecast = 1e-6
                        except Exception as e:
                            logger.warning(
                                f"Error forecasting GARCH volatility for {pair}: {e}. Using default vol_forecast={vol_forecast}.")
                    else:
                        logger.warning(
                            f"Empty spread returns in current window for GARCH forecast for {pair}. Using default vol_forecast={vol_forecast}.")
                else:
                    logger.warning(
                        f"Insufficient spread data in current window for GARCH forecast for {pair}. Using default vol_forecast={vol_forecast}.")

            # 2. TimesNet Prediction of next spread
            # Create features for TimesNet from the current spread window
            # Adj_Close is a placeholder column name TimesNetModel.create_sequences expects
            features_for_timesnet = pd.DataFrame(
                {'Adj_Close': current_data_window[asset1] - current_data_window[asset2]})
            features_for_timesnet.dropna(inplace=True)

            if len(features_for_timesnet) < self.params['sequence_length']:
                logger.warning(f"Insufficient data in current_data_window ({len(features_for_timesnet)} points) "
                               f"for TimesNet sequence of length {self.params['sequence_length']} for {pair}. Skipping signal.")
                signals[pair] = 0.0
                continue

            # Extract the latest sequence for prediction
            # TimesNetModel.predict expects a 3D numpy array: (num_samples, sequence_length, num_features)
            # Here, num_samples is 1, num_features is 1 (spread).
            sequence_for_prediction = features_for_timesnet['Adj_Close'].tail(
                self.params['sequence_length']).values.reshape(1, self.params['sequence_length'], 1)

            try:
                predicted_spread = timesnet.predict(sequence_for_prediction)[0][0]
            except Exception as e:
                logger.error(f"Error during TimesNet prediction for {pair}: {e}. Skipping signal.")
                signals[pair] = 0.0
                continue

            current_spread = features_for_timesnet['Adj_Close'].iloc[-1]
            predicted_change = predicted_spread - current_spread

            # 3. Signal Generation using adaptive threshold
            # The entry threshold scales with the predicted volatility of the spread
            entry_threshold = 0.5 * np.sqrt(vol_forecast)  # Example: 0.5 standard deviations

            logger.debug(f"Pair {pair}: Current Spread={current_spread:.4f}, Predicted Spread={predicted_spread:.4f}, "
                         f"Predicted Change={predicted_change:.4f}, Vol Forecast={vol_forecast:.4f}, Threshold={entry_threshold:.4f}")

            if predicted_change > entry_threshold:
                signals[pair] = 1.0  # Long the spread (buy asset1, sell asset2)
                logger.info(
                    f"Signal for {pair}: LONG SPREAD (Predicted change {predicted_change:.4f} > Threshold {entry_threshold:.4f})")
            elif predicted_change < -entry_threshold:
                signals[pair] = -1.0  # Short the spread (sell asset1, buy asset2)
                logger.info(
                    f"Signal for {pair}: SHORT SPREAD (Predicted change {predicted_change:.4f} < -Threshold {-entry_threshold:.4f})")
            else:
                signals[pair] = 0.0  # No trade
                logger.info(f"Signal for {pair}: NO TRADE (Predicted change {predicted_change:.4f} within threshold)")

        return signals

    def save_strategy(self):
        """
        Saves the entire strategy state, including fitted TimesNet and GARCH models for each pair.
        """
        logger.info(f"Saving {self.name} strategy to {self._strategy_model_path}...")

        # Save TimesNet models using their internal save mechanism
        for pair, models in self.pair_models.items():
            pair_str = f"{pair[0]}_{pair[1]}"
            if models['timesnet'] and models['timesnet'].is_fitted:
                # TimesNetModel's save method already uses its own model_dir (which is _strategy_model_path here)
                models[
                    'timesnet'].save_model()  # This will save to self._strategy_model_path / "TimesNet_{asset1}_{asset2}" / "best_model.keras" etc.
                logger.debug(f"Saved TimesNet model for {pair_str}.")

            # Save GARCH model parameters (or the model itself using arch's save)
            if models['garch']:
                garch_path = self._strategy_model_path / f"garch_model_{pair_str}.pkl"
                try:
                    # arch_model.fit() result (ARCHModelResult) has its own save/load
                    models['garch'].save(garch_path)
                    logger.debug(f"Saved GARCH model for {pair_str} to {garch_path}.")
                except Exception as e:
                    logger.warning(f"Could not save GARCH model for {pair_str}: {e}. Saving parameters via joblib.")
                    # Fallback to saving parameters if full model save fails
                    joblib.dump(models['garch'].params, self._strategy_model_path / f"garch_params_{pair_str}.joblib")
                    joblib.dump(models['garch'].volatility_params,
                                self._strategy_model_path / f"garch_vol_params_{pair_str}.joblib")

        # Save strategy metadata (e.g., tradeable_pairs, hyperparameters)
        strategy_metadata = {
            'name': self.name,
            'lookback_window': self.params['lookback_window'],
            'sequence_length': self.params['sequence_length'],
            'max_pairs': self.params['max_pairs'],
            'timesnet_epochs': self.timesnet_epochs,
            'timesnet_batch_size': self.timesnet_batch_size,
            'timesnet_n_filters': self.timesnet_n_filters,
            'timesnet_kernel_size': self.timesnet_kernel_size,
            'timesnet_stack_size': self.timesnet_stack_size,
            'garch_p': self.garch_p,
            'garch_q': self.garch_q,
            'tradeable_pairs': self.tradeable_pairs,  # Save the list of pairs that were actually fitted
            'is_fitted': self.is_fitted
        }
        joblib.dump(strategy_metadata, self._strategy_model_path / "strategy_metadata.joblib")
        logger.info(f"{self.name} strategy state saved successfully.")

    def load_strategy(self):
        """
        Loads the entire strategy state, including fitted TimesNet and GARCH models for each pair.
        """
        logger.info(f"Loading {self.name} strategy from {self._strategy_model_path}...")

        metadata_path = self._strategy_model_path / "strategy_metadata.joblib"
        if not metadata_path.exists():
            logger.warning(f"Strategy metadata not found at {metadata_path}. Cannot load strategy state.")
            return

        try:
            strategy_metadata = joblib.load(metadata_path)
            self.name = strategy_metadata.get('name', self.name)
            # Update parameters from loaded metadata
            self.params['lookback_window'] = strategy_metadata.get('lookback_window', self.params['lookback_window'])
            self.params['sequence_length'] = strategy_metadata.get('sequence_length', self.params['sequence_length'])
            self.params['max_pairs'] = strategy_metadata.get('max_pairs', self.params['max_pairs'])
            self.timesnet_epochs = strategy_metadata.get('timesnet_epochs', self.timesnet_epochs)
            self.timesnet_batch_size = strategy_metadata.get('timesnet_batch_size', self.timesnet_batch_size)
            self.timesnet_n_filters = strategy_metadata.get('timesnet_n_filters', self.timesnet_n_filters)
            self.timesnet_kernel_size = strategy_metadata.get('timesnet_kernel_size', self.timesnet_kernel_size)
            self.timesnet_stack_size = strategy_metadata.get('timesnet_stack_size', self.timesnet_stack_size)
            self.garch_p = strategy_metadata.get('garch_p', self.garch_p)
            self.garch_q = strategy_metadata.get('garch_q', self.garch_q)
            self.tradeable_pairs = strategy_metadata.get('tradeable_pairs', [])
            self.is_fitted = strategy_metadata.get('is_fitted', False)

        except Exception as e:
            logger.error(
                f"Error loading strategy metadata for {self.name}: {e}. Strategy may be in an inconsistent state.")
            return

        # Load TimesNet and GARCH models for each pair
        self.pair_models = {}
        for pair in self.tradeable_pairs:
            asset1, asset2 = pair
            pair_str = f"{asset1}_{asset2}"

            loaded_timesnet = None
            loaded_garch_fit = None

            # Load TimesNet model
            timesnet_model_instance = TimesNetModel(f"TimesNet_{pair_str}",
                                                    model_dir=str(self._strategy_model_path))
            try:
                timesnet_model_instance.load_model()
                if timesnet_model_instance.model is not None:
                    loaded_timesnet = timesnet_model_instance
                    logger.debug(f"Loaded TimesNet model for {pair_str}.")
            except Exception as e:
                logger.warning(
                    f"Could not load TimesNet model for {pair_str}: {e}. It might not have been saved or path is incorrect.")

            # Load GARCH model
            garch_path = self._strategy_model_path / f"garch_model_{pair_str}.pkl"
            if garch_path.exists():
                try:
                    # arch_model.fit() result (ARCHModelResult) has its own save/load
                    from arch.univariate.base import ARCHModelResult  # Import again just in case
                    loaded_garch_fit = ARCHModelResult.load(garch_path)
                    logger.debug(f"Loaded GARCH model for {pair_str} from {garch_path}.")
                except Exception as e:
                    logger.warning(
                        f"Could not load GARCH model for {pair_str} from {garch_path}: {e}. Attempting to load parameters.")
                    # Fallback to loading parameters and reconstructing, though forecasting won't work without full model
                    try:
                        garch_params = joblib.load(self._strategy_model_path / f"garch_params_{pair_str}.joblib")
                        garch_vol_params = joblib.load(
                            self._strategy_model_path / f"garch_vol_params_{pair_str}.joblib")
                        logger.warning(
                            f"Loaded GARCH parameters for {pair_str}. Full model not loaded, forecasting might be limited.")
                        # You can potentially reconstruct a dummy GARCH model with these params,
                        # but it won't have the internal state for `.forecast()` correctly.
                        # For robustness, you'd likely re-fit GARCH if full load fails.
                    except Exception as e_params:
                        logger.warning(f"Could not load GARCH parameters for {pair_str}: {e_params}.")

            if loaded_timesnet or loaded_garch_fit:
                self.pair_models[pair] = {'timesnet': loaded_timesnet, 'garch': loaded_garch_fit}
            else:
                logger.warning(
                    f"No models (TimesNet or GARCH) successfully loaded for {pair}. This pair will be skipped during signal generation.")

        logger.info(f"{self.name} strategy state loaded. {len(self.pair_models)} pairs with models loaded.")