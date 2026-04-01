"""Table 2: amplitude demodulation with a DONN-style model."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import tensorflow as tf

from src.HopfLayer import HopfLayer, set_seed


def generate_demod_dataset(
    num_samples: int,
    dt: float,
    duration: float,
    carrier_hz: float,
    num_components: int,
    msg_fmin: float,
    msg_fmax: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate the synthetic demodulation task from the paper description."""
    rng = np.random.default_rng(seed)
    t = np.arange(0.0, duration, dt, dtype=np.float32)

    x = np.zeros((num_samples, t.shape[0], 1), dtype=np.float32)
    y = np.zeros((num_samples, t.shape[0], 1), dtype=np.float32)

    for n in range(num_samples):
        freqs = rng.uniform(msg_fmin, msg_fmax, size=num_components).astype(np.float32)
        message = np.sum(np.sin(2.0 * np.pi * freqs[:, None] * t[None, :]), axis=0)
        carrier = np.sin(2.0 * np.pi * carrier_hz * t)
        modulated = (1.0 + message) * carrier
        x[n, :, 0] = modulated
        y[n, :, 0] = message
    return x, y, t


def split_train_test(
    x: np.ndarray,
    y: np.ndarray,
    test_ratio: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    idx = np.arange(x.shape[0])
    rng = np.random.default_rng(seed)
    rng.shuffle(idx)
    n_test = int(round(x.shape[0] * test_ratio))
    test_idx = idx[:n_test]
    train_idx = idx[n_test:]
    return x[train_idx], y[train_idx], x[test_idx], y[test_idx]


class DONNDemodulator(tf.keras.Model):
    """Linear frontend -> Hopf oscillators -> temporal readout."""

    def __init__(
        self,
        num_steps: int,
        units: int = 40,
        min_omega_hz: float = 0.1,
        max_omega_hz: float = 12.0,
        dt: float = 0.01,
        carrier_hz: float = 8.0,
        hopf_input_scale: float = 10.0,
        use_linear_frontend: bool = True,
        readout_channels: int = 64,
        temporal_kernel: int = 0,
        use_input_skip: bool = True,
    ) -> None:
        super().__init__()
        self.units = units
        self.use_linear_frontend = use_linear_frontend
        self.use_input_skip = use_input_skip

        self.in_r = tf.keras.layers.Dense(units, activation="relu")
        self.in_i = tf.keras.layers.Dense(units, activation="relu")
        self.hopf = HopfLayer(
            units=units,
            num_steps=num_steps,
            min_omega_hz=min_omega_hz,
            max_omega_hz=max_omega_hz,
            dt=dt,
            input_scale=hopf_input_scale,
        )
        self.td_in_r = tf.keras.layers.TimeDistributed(self.in_r)
        self.td_in_i = tf.keras.layers.TimeDistributed(self.in_i)

        if temporal_kernel <= 0:
            temporal_kernel = max(5, int(round(2.0 / (carrier_hz * dt))))
        if temporal_kernel % 2 == 0:
            temporal_kernel += 1

        self.norm = tf.keras.layers.LayerNormalization(axis=-1)
        self.temporal_conv1 = tf.keras.layers.Conv1D(
            readout_channels,
            kernel_size=temporal_kernel,
            padding="same",
            activation="tanh",
        )
        self.temporal_conv2 = tf.keras.layers.Conv1D(
            readout_channels,
            kernel_size=temporal_kernel,
            padding="same",
            activation="tanh",
        )
        self.out_conv = tf.keras.layers.Conv1D(1, kernel_size=1, padding="same", activation="linear")

    def call(self, x: tf.Tensor) -> tf.Tensor:
        if self.use_linear_frontend:
            x_r = self.td_in_r(x)
            x_i = self.td_in_i(x)
        else:
            x_r = tf.tile(x, [1, 1, self.units])
            x_i = tf.zeros_like(x_r)

        z_r, z_i = self.hopf(x_r, x_i)
        features = [z_r, z_i]
        if self.use_input_skip:
            features.extend([x, tf.abs(x)])
        h = self.norm(tf.concat(features, axis=2))
        h = self.temporal_conv1(h)
        h = self.temporal_conv2(h)
        return self.out_conv(h)


@dataclass
class Metrics:
    test_mse: float
    val_mse: float


def train_one_run(
    x: np.ndarray,
    y: np.ndarray,
    dt: float,
    carrier_hz: float,
    seed: int,
    epochs: int,
    batch_size: int,
    test_ratio: float,
    learning_rate: float,
    units: int,
    hopf_input_scale: float,
    use_linear_frontend: bool,
    readout_channels: int,
    temporal_kernel: int,
    use_input_skip: bool,
) -> tuple[Metrics, np.ndarray, np.ndarray, np.ndarray]:
    set_seed(seed)
    x_train, y_train, x_test, y_test = split_train_test(x=x, y=y, test_ratio=test_ratio, seed=seed)

    model = DONNDemodulator(
        num_steps=x.shape[1],
        units=units,
        dt=dt,
        carrier_hz=carrier_hz,
        hopf_input_scale=hopf_input_scale,
        use_linear_frontend=use_linear_frontend,
        readout_channels=readout_channels,
        temporal_kernel=temporal_kernel,
        use_input_skip=use_input_skip,
    )
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="mse",
    )
    history = model.fit(
        x_train,
        y_train,
        validation_split=0.2,
        epochs=epochs,
        batch_size=batch_size,
        verbose=0,
    )

    pred = model.predict(x_test, batch_size=batch_size, verbose=0)
    test_mse = float(np.mean((pred - y_test) ** 2))
    val_mse = float(history.history["val_loss"][-1])
    return Metrics(test_mse=test_mse, val_mse=val_mse), x_test, y_test, pred
