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
        Activation, LayerNormalization, Conv2D, MaxPooling2D, Concatenate
    )
    from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
    from tensorflow.keras.optimizers import Adam
except ImportError:
    raise ImportError("TensorFlow is not installed. Please run 'pip install tensorflow'.")

class TCNResidualBlock(Layer):
    def __init__(self, n_filters, kernel_size, dilation_rate, **kwargs):
        super(TCNResidualBlock, self).__init__(**kwargs)
        self.n_filters = n_filters
        self.kernel_size = kernel_size
        self.dilation_rate = dilation_rate
        self.conv1 = Conv1D(filters=n_filters, kernel_size=kernel_size,
                            dilation_rate=dilation_rate, padding='causal', activation='relu')
        self.dropout1 = SpatialDropout1D(0.2)
        self.conv2 = Conv1D(filters=n_filters, kernel_size=kernel_size,
                            dilation_rate=dilation_rate, padding='causal', activation='relu')
        self.dropout2 = SpatialDropout1D(0.2)
        self.downsample = Conv1D(filters=n_filters, kernel_size=1, padding='same')
        self.add = add
        self.activation = Activation('relu')

    def call(self, inputs):
        x, residual = inputs
        x = self.conv1(x)
        x = self.dropout1(x)
        x = self.conv2(x)
        x = self.dropout2(x)
        if residual.shape[-1] != x.shape[-1]:
            residual = self.downsample(residual)
        return self.activation(self.add([x, residual]))

class TCNModel(BaseModel):
    def build_model(self, input_shape: Tuple, n_filters=64, kernel_size=3, stack_size=2):
        inputs = Input(shape=input_shape)
        x = inputs
        for i in range(stack_size):
            for d in [1, 2, 4, 8]:
                x = TCNResidualBlock(n_filters, kernel_size, d)([x, x])
        x = Dense(32, activation='relu')(x[:, -1, :])
        outputs = Dense(1)(x)
        model = Model(inputs=inputs, outputs=outputs)
        model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')
        self.model = model

    def train(self, X_train: np.ndarray, y_train: np.ndarray, X_val: np.ndarray, y_val: np.ndarray, epochs: int, batch_size: int):
        if self.model is None:
            raise RuntimeError("Model must be built before training.")
        logger.info(f"Training {self.model_name} for {epochs} epochs...")
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True),
            ModelCheckpoint(filepath=self.model_path / "best_model.keras", save_best_only=True, monitor='val_loss')
        ]
        self.model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=epochs, batch_size=batch_size, callbacks=callbacks, verbose=1)
        self.is_fitted = True
        logger.info("Training complete.")

    def predict(self, X: np.ndarray) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError("Model must be trained before prediction.")
        return self.model.predict(X)

    @staticmethod
    def create_sequences(features: pd.DataFrame, target: pd.Series, sequence_length: int) -> Tuple[np.ndarray, np.ndarray]:
        X, y = [], []
        for i in range(len(features) - sequence_length):
            X.append(features.iloc[i:(i + sequence_length)].values)
            y.append(target.iloc[i + sequence_length])
        return np.array(X), np.array(y)


class TimesNetModel(BaseModel):
    """
    An implementation of the TimesNet architecture for time series forecasting.
    """
    def build_model(self, input_shape: Tuple, top_k_periods: int = 5, num_kernels: int = 32):
        
        def inception_block(input_layer, n_filters):
            conv1 = Conv2D(filters=n_filters, kernel_size=(1, 1), padding='same', activation='relu')(input_layer)
            conv3 = Conv2D(filters=n_filters, kernel_size=(3, 3), padding='same', activation='relu')(input_layer)
            conv5 = Conv2D(filters=n_filters, kernel_size=(5, 5), padding='same', activation='relu')(input_layer)
            pool = MaxPooling2D(pool_size=(3, 3), strides=1, padding='same')(input_layer)
            pool = Conv2D(filters=n_filters, kernel_size=(1, 1), padding='same', activation='relu')(pool)
            return Concatenate(axis=-1)([conv1, conv3, conv5, pool])

        def times_block(inputs):
            fft = tf.signal.rfft(inputs)
            amp = tf.math.abs(fft)
            freqs = tf.signal.rfftfreq(inputs.shape[1])
            
            top_k_indices = tf.math.top_k(amp, k=top_k_periods).indices
            top_k_freqs = tf.gather(freqs, top_k_indices, batch_dims=1)
            periods = tf.cast(tf.round(inputs.shape[1] / top_k_freqs), dtype=tf.int32)

            outputs = []
            for i in range(top_k_periods):
                period = periods[:, i, 0]
                padding = (period - (inputs.shape[1] % period)) % period
                padded_inputs = tf.pad(inputs, [[0, 0], [0, padding], [0, 0]])
                reshaped = tf.reshape(padded_inputs, (tf.shape(inputs)[0], -1, period, inputs.shape[-1]))
                reshaped = tf.expand_dims(reshaped, axis=-1)
                conv_out = inception_block(reshaped, num_kernels)
                flattened = tf.reshape(conv_out, (tf.shape(inputs)[0], -1, conv_out.shape[-1]))
                outputs.append(flattened[:, :inputs.shape[1], :])
            
            return add(outputs)

        input_layer = Input(shape=input_shape)
        x = LayerNormalization()(input_layer)
        x = times_block(x)
        x = tf.keras.layers.GlobalAveragePooling1D()(x)
        x = Dense(64, activation='relu')(x)
        x = Dropout(0.2)(x)
        output_layer = Dense(1)(x)

        model = Model(inputs=input_layer, outputs=output_layer)
        model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')
        self.model = model

    def train(self, X_train: np.ndarray, y_train: np.ndarray, X_val: np.ndarray, y_val: np.ndarray, epochs: int, batch_size: int):
        if self.model is None:
            raise RuntimeError("Model must be built before training.")
        logger.info(f"Training {self.model_name} (TimesNet) for {epochs} epochs...")
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True),
            ModelCheckpoint(filepath=self.model_path / "best_model.keras", save_best_only=True, monitor='val_loss')
        ]
        self.model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=epochs, batch_size=batch_size, callbacks=callbacks, verbose=1)
        self.is_fitted = True
        logger.info("Training complete.")

    def predict(self, X: np.ndarray) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError("Model must be trained before prediction.")
        return self.model.predict(X)

    @staticmethod
    def create_sequences(features: pd.DataFrame, target: pd.Series, sequence_length: int) -> Tuple[np.ndarray, np.ndarray]:
        X, y = [], []
        for i in range(len(features) - sequence_length):
            feature_slice = features.iloc[i:(i + sequence_length)]
            if feature_slice.ndim == 1:
                feature_slice = feature_slice.to_frame()
            X.append(feature_slice.values)
            y.append(target.iloc[i + sequence_length])
        return np.array(X), np.array(y)