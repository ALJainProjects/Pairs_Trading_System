import pandas as pd
import numpy as np
from typing import Tuple, List

from .base_model import BaseModel
from config.logging_config import logger

try:
    import tensorflow as tf
    from tensorflow.keras.models import Model
    from tensorflow.keras.layers import (
        Input, Dense, Dropout, Layer, Conv1D, SpatialDropout1D, add,
        Activation, LayerNormalization, Conv2D, MaxPooling2D, Concatenate, GlobalAveragePooling1D
    )
    from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.models import load_model  # Added for explicit model loading
except ImportError:
    raise ImportError("TensorFlow is not installed. Please run 'pip install tensorflow'.")


class TCNResidualBlock(Layer):
    """
    Implements a TCN (Temporal Convolutional Network) residual block.

    This block uses dilated causal convolutions and a residual connection,
    which is a key component of Temporal Convolutional Networks.
    """

    def __init__(self, n_filters: int, kernel_size: int, dilation_rate: int, **kwargs):
        """
        Initializes the TCNResidualBlock.

        Args:
            n_filters (int): Number of filters for the convolutional layers.
            kernel_size (int): Kernel size for the convolutional layers.
            dilation_rate (int): Dilation rate for the convolutional layers.
            **kwargs: Additional keyword arguments for the Layer base class.
        """
        super(TCNResidualBlock, self).__init__(**kwargs)
        self.n_filters = n_filters
        self.kernel_size = kernel_size
        self.dilation_rate = dilation_rate
        # Layers are created in the build method for better flexibility with input shapes
        self.conv1 = None
        self.dropout1 = None
        self.conv2 = None
        self.dropout2 = None
        self.downsample = None
        self.add_layer = add  # Renamed to avoid conflict with built-in 'add'
        self.activation_layer = Activation('relu')  # Renamed to avoid conflict with built-in 'Activation'

    def build(self, input_shape: Tuple[tf.TensorShape, tf.TensorShape]):
        """
        Creates the layers' weights based on the input shape.

        Args:
            input_shape (Tuple[tf.TensorShape, tf.TensorShape]): A tuple of
                                                                 input shapes, where
                                                                 input_shape[0] is the shape
                                                                 of the main input 'x' and
                                                                 input_shape[1] is the shape
                                                                 of the residual input.
        """
        self.conv1 = Conv1D(filters=self.n_filters, kernel_size=self.kernel_size,
                            dilation_rate=self.dilation_rate, padding='causal', activation='relu')
        self.dropout1 = SpatialDropout1D(0.2)
        self.conv2 = Conv1D(filters=self.n_filters, kernel_size=self.kernel_size,
                            dilation_rate=self.dilation_rate, padding='causal', activation='relu')
        self.dropout2 = SpatialDropout1D(0.2)

        # Only create downsample if the number of filters needs to change for the residual connection
        if input_shape[0][-1] != self.n_filters:
            self.downsample = Conv1D(filters=self.n_filters, kernel_size=1, padding='same')
        super(TCNResidualBlock, self).build(input_shape)

    def call(self, inputs: Tuple[tf.Tensor, tf.Tensor]) -> tf.Tensor:
        """
        Forward pass for the TCN residual block.

        Args:
            inputs (Tuple[tf.Tensor, tf.Tensor]): A tuple containing the
                                                  current input tensor (x) and
                                                  the residual connection tensor (residual).

        Returns:
            tf.Tensor: The output tensor after applying the residual block.
        """
        x, residual = inputs

        # Apply convolutional layers and dropouts
        _x = self.conv1(x)
        _x = self.dropout1(_x)
        _x = self.conv2(_x)
        _x = self.dropout2(_x)

        # Handle residual connection: downsample if feature dimensions don't match
        if self.downsample is not None:
            residual = self.downsample(residual)

        # Add residual and apply activation
        return self.activation_layer(self.add_layer([_x, residual]))


class TCNModel(BaseModel):
    """
    A Temporal Convolutional Network (TCN) model for time series forecasting.
    Inherits from BaseModel to provide standard training and prediction interfaces.
    """

    def build_model(self, input_shape: Tuple, n_filters: int = 64, kernel_size: int = 3, stack_size: int = 2):
        """
        Builds the TCN model architecture.

        Args:
            input_shape (Tuple): The shape of the input time series data (sequence_length, num_features).
            n_filters (int): Number of filters for the TCN residual blocks.
            kernel_size (int): Kernel size for the TCN residual blocks.
            stack_size (int): Number of residual block stacks. Each stack contains
                              blocks with different dilation rates.
        """
        inputs = Input(shape=input_shape)
        x = inputs
        # Store initial input for the residual connections if desired, current design feeds 'x' back
        # initial_input_for_stack = inputs # If you wanted true residual from the start of the stack

        for i in range(stack_size):
            # Apply residual blocks with increasing dilation rates
            for d in [1, 2, 4, 8]:  # Example dilation rates
                # The TCNResidualBlock expects [x, residual]. Here, x is also the residual.
                x = TCNResidualBlock(n_filters, kernel_size, d, name=f'tcn_block_{i}_{d}')([x, x])

        # Global pooling and dense layers for output
        # Take the last timestep's output as representation (typical for sequence-to-one prediction)
        x = Dense(32, activation='relu')(x[:, -1, :])
        outputs = Dense(1)(x)  # Single output for regression

        model = Model(inputs=inputs, outputs=outputs)
        model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')
        self.model = model
        logger.info(f"{self.model_name} model built with input shape {input_shape}")
        self.model.summary(print_fn=lambda x: logger.info(x))

    def train(self, X_train: np.ndarray, y_train: np.ndarray, X_val: np.ndarray, y_val: np.ndarray, epochs: int,
              batch_size: int):
        """
        Trains the TCN model.

        Args:
            X_train (np.ndarray): Training features.
            y_train (np.ndarray): Training targets.
            X_val (np.ndarray): Validation features.
            y_val (np.ndarray): Validation targets.
            epochs (int): Number of training epochs.
            batch_size (int): Batch size for training.
        """
        if self.model is None:
            raise RuntimeError("Model must be built before training.")
        logger.info(f"Training {self.model_name} for {epochs} epochs...")

        # Ensure model_path exists
        self.model_path.mkdir(parents=True, exist_ok=True)

        callbacks = [
            EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True),
            ModelCheckpoint(filepath=str(self.model_path / "best_model.keras"), save_best_only=True, monitor='val_loss')
        ]
        history = self.model.fit(X_train, y_train, validation_data=(X_val, y_val),
                                 epochs=epochs, batch_size=batch_size,
                                 callbacks=callbacks, verbose=1)
        self.is_fitted = True
        logger.info("Training complete.")
        # Optionally save the final model state in addition to the best checkpoint
        self.model.save(str(self.model_path / "final_model.keras"))
        logger.info(f"Final model state saved to {self.model_path / 'final_model.keras'}")

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Makes predictions using the trained TCN model.

        Args:
            X (np.ndarray): Input features for prediction.

        Returns:
            np.ndarray: Predicted values.
        """
        if not self.is_fitted:
            raise RuntimeError("Model must be trained before prediction.")
        logger.info("Making predictions with TCNModel...")
        return self.model.predict(X)

    @staticmethod
    def create_sequences(features: pd.DataFrame, target: pd.Series, sequence_length: int) -> Tuple[
        np.ndarray, np.ndarray]:
        """
        Creates sequences from time series features and target for TCN input.

        Args:
            features (pd.DataFrame): DataFrame of input features.
            target (pd.Series): Series of target values.
            sequence_length (int): The length of each input sequence.

        Returns:
            Tuple[np.ndarray, np.ndarray]: A tuple containing numpy arrays of
                                           sequences (X) and corresponding targets (y).
        """
        X, y = [], []
        for i in range(len(features) - sequence_length):
            X.append(features.iloc[i:(i + sequence_length)].values)
            y.append(target.iloc[i + sequence_length])
        logger.info(f"Created {len(X)} sequences of length {sequence_length}.")
        return np.array(X), np.array(y)


class TimesNetModel(BaseModel):
    """
    An implementation of the TimesNet architecture for time series forecasting.
    This model utilizes a frequency-domain analysis to extract periodicities
    and applies an Inception-like block on reshaped periodic components.
    """

    def build_model(self, input_shape: Tuple, top_k_periods: int = 5, num_kernels: int = 32):
        """
        Builds the TimesNet model architecture.

        Args:
            input_shape (Tuple): The shape of the input time series data (sequence_length, num_features).
            top_k_periods (int): The number of top dominant frequencies (periods) to consider.
            num_kernels (int): Number of filters for the convolutional layers within the Inception block.
        """

        def inception_block(input_layer: tf.Tensor, n_filters: int) -> tf.Tensor:
            """
            An Inception-like block for 2D convolutions, applied to the reshaped periodic components.
            """
            conv1 = Conv2D(filters=n_filters, kernel_size=(1, 1), padding='same', activation='relu')(input_layer)
            conv3 = Conv2D(filters=n_filters, kernel_size=(3, 3), padding='same', activation='relu')(input_layer)
            conv5 = Conv2D(filters=n_filters, kernel_size=(5, 5), padding='same', activation='relu')(input_layer)
            # MaxPooling and 1x1 conv for dimensionality reduction
            pool = MaxPooling2D(pool_size=(3, 3), strides=1, padding='same')(input_layer)
            pool = Conv2D(filters=n_filters, kernel_size=(1, 1), padding='same', activation='relu')(pool)
            return Concatenate(axis=-1)([conv1, conv3, conv5, pool])

        def times_block(inputs: tf.Tensor) -> tf.Tensor:
            """
            The core TimesBlock that performs frequency analysis and applies Inception blocks.
            """
            # Perform Real FFT
            fft = tf.signal.rfft(inputs)
            amp = tf.math.abs(fft)  # Get amplitude spectrum
            freqs = tf.signal.rfftfreq(tf.shape(inputs)[1])  # Get corresponding frequencies

            # Ensure top_k_periods is valid (cannot exceed available frequencies)
            tf.Assert(tf.less_equal(top_k_periods, tf.shape(amp)[1]),
                      ["top_k_periods must be less than or equal to the length of the FFT result."])

            # Get top-k dominant frequencies and calculate corresponding periods
            top_k_indices = tf.math.top_k(amp, k=top_k_periods).indices
            top_k_freqs = tf.gather(freqs, top_k_indices, batch_dims=1)

            # Calculate periods, ensuring a minimum period of 1 to avoid issues
            periods = tf.cast(tf.round(tf.cast(tf.shape(inputs)[1], tf.float32) / (top_k_freqs + 1e-6)),
                              dtype=tf.int32)  # Add small epsilon to avoid div by zero
            periods = tf.maximum(periods, 1)  # Ensure period is at least 1

            outputs = []
            for i in range(top_k_periods):
                # Extract the period for the current top-k frequency
                period = periods[:, i, 0]  # shape (batch_size,)

                # Calculate padding to make input length a multiple of the period
                current_sequence_length = tf.shape(inputs)[1]
                padding = (period - (current_sequence_length % period)) % period
                padded_inputs = tf.pad(inputs, [[0, 0], [0, padding], [0, 0]])

                # Reshape to (batch_size, num_segments, period, num_features)
                # Then expand dims for Conv2D: (batch_size, num_segments, period, num_features, 1)
                reshaped = tf.reshape(padded_inputs, (tf.shape(inputs)[0], -1, period, inputs.shape[-1]))
                reshaped = tf.expand_dims(reshaped, axis=-1)

                # Apply Inception block
                conv_out = inception_block(reshaped, num_kernels)

                # Flatten back and crop to original sequence length
                flattened = tf.reshape(conv_out, (tf.shape(inputs)[0], -1, tf.shape(conv_out)[-1]))
                outputs.append(flattened[:, :current_sequence_length, :])

            # Sum the outputs from all top-k periods
            return add(outputs)

        input_layer = Input(shape=input_shape)
        # Apply Layer Normalization before TimesBlock for stability
        x = LayerNormalization()(input_layer)
        x = times_block(x)

        # Global average pooling across the sequence dimension
        x = GlobalAveragePooling1D()(x)
        x = Dense(64, activation='relu')(x)
        x = Dropout(0.2)(x)
        output_layer = Dense(1)(x)

        model = Model(inputs=input_layer, outputs=output_layer)
        model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')
        self.model = model
        logger.info(f"{self.model_name} (TimesNet) model built with input shape {input_shape}")
        self.model.summary(print_fn=lambda x: logger.info(x))

    def train(self, X_train: np.ndarray, y_train: np.ndarray, X_val: np.ndarray, y_val: np.ndarray, epochs: int,
              batch_size: int):
        """
        Trains the TimesNet model.

        Args:
            X_train (np.ndarray): Training features.
            y_train (np.ndarray): Training targets.
            X_val (np.ndarray): Validation features.
            y_val (np.ndarray): Validation targets.
            epochs (int): Number of training epochs.
            batch_size (int): Batch size for training.
        """
        if self.model is None:
            raise RuntimeError("Model must be built before training.")
        logger.info(f"Training {self.model_name} (TimesNet) for {epochs} epochs...")

        # Ensure model_path exists
        self.model_path.mkdir(parents=True, exist_ok=True)

        callbacks = [
            EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True),
            # Note: For custom layers, custom_objects might be needed on load
            ModelCheckpoint(filepath=str(self.model_path / "best_model_timesnet.keras"), save_best_only=True,
                            monitor='val_loss')
        ]
        history = self.model.fit(X_train, y_train, validation_data=(X_val, y_val),
                                 epochs=epochs, batch_size=batch_size,
                                 callbacks=callbacks, verbose=1)
        self.is_fitted = True
        logger.info("Training complete.")
        self.model.save(str(self.model_path / "final_model_timesnet.keras"))
        logger.info(f"Final TimesNet model state saved to {self.model_path / 'final_model_timesnet.keras'}")

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Makes predictions using the trained TimesNet model.

        Args:
            X (np.ndarray): Input features for prediction.

        Returns:
            np.ndarray: Predicted values.
        """
        if not self.is_fitted:
            raise RuntimeError("Model must be trained before prediction.")
        logger.info("Making predictions with TimesNetModel...")
        return self.model.predict(X)

    @staticmethod
    def create_sequences(features: pd.DataFrame, target: pd.Series, sequence_length: int) -> Tuple[
        np.ndarray, np.ndarray]:
        """
        Creates sequences from time series features and target for TimesNet input.

        Args:
            features (pd.DataFrame): DataFrame of input features.
            target (pd.Series): Series of target values.
            sequence_length (int): The length of each input sequence.

        Returns:
            Tuple[np.ndarray, np.ndarray]: A tuple containing numpy arrays of
                                           sequences (X) and corresponding targets (y).
        """
        X, y = [], []
        for i in range(len(features) - sequence_length):
            feature_slice = features.iloc[i:(i + sequence_length)]
            # Ensure feature_slice is always 2D (sequence_length, num_features)
            if feature_slice.ndim == 1:
                feature_slice = feature_slice.to_frame()
            X.append(feature_slice.values)
            y.append(target.iloc[i + sequence_length])
        logger.info(f"Created {len(X)} sequences of length {sequence_length}.")
        return np.array(X), np.array(y)