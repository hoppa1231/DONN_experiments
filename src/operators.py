"""Table 3: integration and differentiation with a DONN-style temporal regressor."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import tensorflow as tf

from src.HopfLayer import HopfLayer, set_seed

TASKS = ("integration", "differentiation")


@dataclass
class OperatorMetrics:
    task: str
    test_mse: float
    val_mse: float
    baseline_mse: float
    test_corr: float


class DONNOperatorRegressor(tf.keras.Model):
    """Linear frontend -> Hopf oscillators -> temporal Conv1D readout."""

    def __init__(
        self,
        num_steps: int,
        units: int = 20,
        min_omega_hz: float = 0.1,
        max_omega_hz: float = 20.0,
        dt: float = 0.001,
        hopf_input_scale: float = 5.0,
        use_linear_frontend: bool = True,
        use_input_skip: bool = True,
        readout_channels: int = 48,
        temporal_kernel: int = 33,
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

        if temporal_kernel < 3:
            temporal_kernel = 3
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
            features.append(x)

        h = self.norm(tf.concat(features, axis=2))
        h = self.temporal_conv1(h)
        h = self.temporal_conv2(h)
        return self.out_conv(h)


def generate_operator_dataset(
    task: str,
    num_samples: int,
    dt: float,
    duration: float,
    num_components: int,
    fmin_hz: float,
    fmax_hz: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate article-style operator data directly from the formulas."""
    rng = np.random.default_rng(seed)
    t = np.arange(0.0, duration, dt, dtype=np.float32)
    x = np.zeros((num_samples, t.shape[0], 1), dtype=np.float32)
    y = np.zeros((num_samples, t.shape[0], 1), dtype=np.float32)

    for n in range(num_samples):
        amps = rng.normal(0.0, 1.0, size=num_components).astype(np.float32)
        phases = rng.normal(0.0, np.pi, size=num_components).astype(np.float32)
        freq_hz = rng.uniform(fmin_hz, fmax_hz, size=num_components).astype(np.float32)
        omega = 2.0 * np.pi * freq_hz

        input_sig = np.sum(
            amps[:, None] * np.sin(omega[:, None] * t[None, :] + phases[:, None]),
            axis=0,
        )
        if task == "integration":
            output_sig = np.sum(
                -(amps[:, None] / omega[:, None]) * np.cos(omega[:, None] * t[None, :] + phases[:, None]),
                axis=0,
            )
        elif task == "differentiation":
            output_sig = np.sum(
                (amps[:, None] * omega[:, None]) * np.cos(omega[:, None] * t[None, :] + phases[:, None]),
                axis=0,
            )
        else:
            raise ValueError(f"Unsupported task: {task}")

        x[n, :, 0] = input_sig
        y[n, :, 0] = output_sig

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


def _numeric_prediction(task: str, x: np.ndarray, y: np.ndarray, dt: float) -> np.ndarray:
    sig = x[:, :, 0]
    if task == "differentiation":
        pred = np.gradient(sig, dt, axis=1)
    else:
        area = np.cumsum((sig[:, 1:] + sig[:, :-1]) * 0.5 * dt, axis=1)
        pred = np.concatenate([np.zeros((sig.shape[0], 1), dtype=sig.dtype), area], axis=1)
        offset = y[:, :, 0].mean(axis=1, keepdims=True) - pred.mean(axis=1, keepdims=True)
        pred = pred + offset
    return pred[:, :, None].astype(np.float32)


def numeric_baseline(task: str, x: np.ndarray, y: np.ndarray, dt: float) -> tuple[np.ndarray, float]:
    pred = _numeric_prediction(task=task, x=x, y=y, dt=dt)
    mse = float(np.mean((pred - y) ** 2))
    return pred, mse


def _safe_scale(x: np.ndarray) -> float:
    return max(float(np.std(x)), 1e-6)


def train_one_task(
    task: str,
    num_samples: int,
    dt: float,
    duration: float,
    num_components: int,
    fmin_hz: float,
    fmax_hz: float,
    seed: int,
    epochs: int,
    batch_size: int,
    test_ratio: float,
    learning_rate: float,
    units: int,
    hopf_input_scale: float,
    use_linear_frontend: bool,
    use_input_skip: bool,
    readout_channels: int,
    temporal_kernel: int,
) -> tuple[OperatorMetrics, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Train one task and return metrics plus test-set traces for visualization."""
    set_seed(seed)
    x, y, t = generate_operator_dataset(
        task=task,
        num_samples=num_samples,
        dt=dt,
        duration=duration,
        num_components=num_components,
        fmin_hz=fmin_hz,
        fmax_hz=fmax_hz,
        seed=seed,
    )
    x_train, y_train, x_test, y_test = split_train_test(x=x, y=y, test_ratio=test_ratio, seed=seed)

    x_scale = _safe_scale(x_train)
    y_scale = _safe_scale(y_train)
    baseline_pred, baseline_mse = numeric_baseline(task=task, x=x_test, y=y_test, dt=dt)

    model = DONNOperatorRegressor(
        num_steps=x.shape[1],
        units=units,
        dt=dt,
        hopf_input_scale=hopf_input_scale,
        use_linear_frontend=use_linear_frontend,
        use_input_skip=use_input_skip,
        readout_channels=readout_channels,
        temporal_kernel=temporal_kernel,
    )
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="mse",
    )
    history = model.fit(
        x_train / x_scale,
        y_train / y_scale,
        validation_split=0.2,
        epochs=epochs,
        batch_size=batch_size,
        verbose=0,
    )

    pred = model.predict(x_test / x_scale, batch_size=batch_size, verbose=0) * y_scale
    test_mse = float(np.mean((pred - y_test) ** 2))
    val_mse = float(history.history["val_loss"][-1] * (y_scale**2))
    test_corr = float(np.corrcoef(pred.reshape(-1), y_test.reshape(-1))[0, 1])

    metrics = OperatorMetrics(
        task=task,
        test_mse=test_mse,
        val_mse=val_mse,
        baseline_mse=baseline_mse,
        test_corr=test_corr,
    )
    return metrics, t, x_test, y_test, pred.astype(np.float32), baseline_pred
