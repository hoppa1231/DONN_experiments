"""Paper-style DONN building blocks with complex static layers."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import tensorflow as tf

from src.HopfLayer import HopfLayer, set_seed


def _activation(name: str | None):
    if name is None or name == "linear":
        return None
    if name == "relu":
        return tf.nn.relu
    if name == "tanh":
        return tf.nn.tanh
    if name == "sigmoid":
        return tf.nn.sigmoid
    raise ValueError(f"Unsupported activation: {name}")


class ComplexDense(tf.keras.layers.Layer):
    """Complex dense layer followed by split real/imag activation from Eq. (20)."""

    def __init__(self, units: int, activation: str | None = None, **kwargs) -> None:
        super().__init__(**kwargs)
        self.units = units
        self.activation_name = activation
        self.activation = _activation(activation)

    def build(self, input_shape) -> None:
        in_dim = int(input_shape[0][-1])
        init = tf.keras.initializers.GlorotUniform()
        self.w_r = self.add_weight(name="w_r", shape=(in_dim, self.units), initializer=init)
        self.w_i = self.add_weight(name="w_i", shape=(in_dim, self.units), initializer=init)
        self.b_r = self.add_weight(name="b_r", shape=(self.units,), initializer="zeros")
        self.b_i = self.add_weight(name="b_i", shape=(self.units,), initializer="zeros")
        super().build(input_shape)

    def call(self, inputs: tuple[tf.Tensor, tf.Tensor]) -> tuple[tf.Tensor, tf.Tensor]:
        z_r, z_i = inputs
        out_r = tf.tensordot(z_r, self.w_r, axes=1) - tf.tensordot(z_i, self.w_i, axes=1) + self.b_r
        out_i = tf.tensordot(z_r, self.w_i, axes=1) + tf.tensordot(z_i, self.w_r, axes=1) + self.b_i
        if self.activation is not None:
            out_r = self.activation(out_r)
            out_i = self.activation(out_i)
        return out_r, out_i


class PaperSequenceDONN(tf.keras.Model):
    """ReLU -> Hopf -> ReLU -> Hopf -> tanh -> output sequence architecture."""

    def __init__(
        self,
        num_steps: int,
        units: int,
        output_dim: int,
        min_omega_hz: float,
        max_omega_hz: float,
        dt: float,
        hopf_input_scale: float,
        mu: float = 1.0,
        beta: float = -100.0,
        trainable_omegas: bool = False,
    ) -> None:
        super().__init__()
        self.static1 = ComplexDense(units, activation="relu")
        self.hopf1 = HopfLayer(
            units=units,
            num_steps=num_steps,
            min_omega_hz=min_omega_hz,
            max_omega_hz=max_omega_hz,
            dt=dt,
            mu=mu,
            beta=beta,
            input_scale=hopf_input_scale,
            trainable_omegas=trainable_omegas,
        )
        self.static2 = ComplexDense(units, activation="relu")
        self.hopf2 = HopfLayer(
            units=units,
            num_steps=num_steps,
            min_omega_hz=min_omega_hz,
            max_omega_hz=max_omega_hz,
            dt=dt,
            mu=mu,
            beta=beta,
            input_scale=hopf_input_scale,
            trainable_omegas=trainable_omegas,
        )
        self.static3 = ComplexDense(units, activation="tanh")
        self.out = ComplexDense(output_dim, activation="linear")

    def call(self, x: tf.Tensor) -> tf.Tensor:
        z_r = x
        z_i = tf.zeros_like(z_r)
        z_r, z_i = self.static1((z_r, z_i))
        z_r, z_i = self.hopf1(z_r, z_i)
        z_r, z_i = self.static2((z_r, z_i))
        z_r, z_i = self.hopf2(z_r, z_i)
        z_r, z_i = self.static3((z_r, z_i))
        out_r, _ = self.out((z_r, z_i))
        return out_r


class PaperClassificationDONN(tf.keras.Model):
    """Linear -> Hopf -> tanh -> output(2) sequence architecture from Table 1."""

    def __init__(
        self,
        num_steps: int,
        units: int,
        num_classes: int,
        min_omega_hz: float,
        max_omega_hz: float,
        dt: float,
        hopf_input_scale: float,
        mu: float = 1.0,
        beta: float = -100.0,
    ) -> None:
        super().__init__()
        self.linear = ComplexDense(units, activation="linear")
        self.hopf = HopfLayer(
            units=units,
            num_steps=num_steps,
            min_omega_hz=min_omega_hz,
            max_omega_hz=max_omega_hz,
            dt=dt,
            mu=mu,
            beta=beta,
            input_scale=hopf_input_scale,
            trainable_omegas=False,
        )
        self.hidden = ComplexDense(units, activation="tanh")
        self.out = ComplexDense(num_classes, activation="linear")

    def call(self, x: tf.Tensor) -> tf.Tensor:
        z_r = x
        z_i = tf.zeros_like(z_r)
        z_r, z_i = self.linear((z_r, z_i))
        z_r, z_i = self.hopf(z_r, z_i)
        z_r, z_i = self.hidden((z_r, z_i))
        out_r, _ = self.out((z_r, z_i))
        return out_r


@dataclass
class PaperSequenceMetrics:
    test_mse: float
    val_mse: float
    test_corr: float


def _safe_scale(x: np.ndarray) -> float:
    return max(float(np.std(x)), 1e-6)


def train_paper_sequence_regressor(
    x: np.ndarray,
    y: np.ndarray,
    seed: int,
    epochs: int,
    batch_size: int,
    test_ratio: float,
    learning_rate: float,
    clipnorm: float | None,
    units: int,
    min_omega_hz: float,
    max_omega_hz: float,
    dt: float,
    hopf_input_scale: float,
    mu: float = 1.0,
    beta: float = -100.0,
    trainable_omegas: bool = False,
    scale_data: bool = False,
) -> tuple[PaperSequenceMetrics, np.ndarray, np.ndarray, np.ndarray, dict[str, float]]:
    set_seed(seed)
    idx = np.arange(x.shape[0])
    rng = np.random.default_rng(seed)
    rng.shuffle(idx)
    n_test = int(round(x.shape[0] * test_ratio))
    test_idx = idx[:n_test]
    train_idx = idx[n_test:]
    x_train, y_train = x[train_idx], y[train_idx]
    x_test, y_test = x[test_idx], y[test_idx]

    x_scale = _safe_scale(x_train) if scale_data else 1.0
    y_scale = _safe_scale(y_train) if scale_data else 1.0
    model = PaperSequenceDONN(
        num_steps=x.shape[1],
        units=units,
        output_dim=y.shape[2],
        min_omega_hz=min_omega_hz,
        max_omega_hz=max_omega_hz,
        dt=dt,
        hopf_input_scale=hopf_input_scale,
        mu=mu,
        beta=beta,
        trainable_omegas=trainable_omegas,
    )
    optimizer = tf.keras.optimizers.Adam(learning_rate, clipnorm=clipnorm) if clipnorm else tf.keras.optimizers.Adam(learning_rate)
    model.compile(optimizer=optimizer, loss="mse")
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
    metrics = PaperSequenceMetrics(test_mse=test_mse, val_mse=val_mse, test_corr=test_corr)
    scale_info = {"x_scale": float(x_scale), "y_scale": float(y_scale)}
    return metrics, x_test, y_test, pred.astype(np.float32), scale_info
