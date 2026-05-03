"""Table 5 helpers: OCNN-style action-recognition smoke control."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import tensorflow as tf

from src.HopfLayer import set_seed


@dataclass
class ActionMetrics:
    val_acc: float
    val_loss: float
    train_acc: float
    train_loss: float


def _make_ramp_targets(labels: np.ndarray, num_steps: int, num_classes: int) -> np.ndarray:
    ramp = np.linspace(0.0, 1.0, num_steps, dtype=np.float32)
    y = np.zeros((labels.shape[0], num_steps, num_classes), dtype=np.float32)
    y[np.arange(labels.shape[0]), :, labels.astype(np.int64)] = ramp[None, :]
    return y


def sequence_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    true_cls = np.argmax(np.sum(y_true, axis=1), axis=1)
    pred_cls = np.argmax(np.sum(y_pred, axis=1), axis=1)
    return float(np.mean(true_cls == pred_cls))


def generate_synthetic_action_dataset(
    num_samples: int,
    num_frames: int,
    frame_size: int,
    square_size: int,
    num_classes: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate a tiny video task for checking the Table-5 OCNN code path.

    This is not UCF11.  Class 0 moves a colored square horizontally; class 1
    moves it vertically.  Additional classes, when requested, use simple
    diagonal variants so the model and metric paths can be tested.
    """
    rng = np.random.default_rng(seed)
    x = np.zeros((num_samples, num_frames, frame_size, frame_size, 3), dtype=np.float32)
    labels = rng.integers(0, num_classes, size=num_samples, dtype=np.int64)
    max_pos = frame_size - square_size

    for sample_idx, label in enumerate(labels):
        start = int(rng.integers(0, max_pos + 1))
        fixed = int(rng.integers(0, max_pos + 1))
        color = np.zeros(3, dtype=np.float32)
        color[label % 3] = 1.0
        for t in range(num_frames):
            pos = int(round(t * max_pos / max(1, num_frames - 1)))
            if label % 4 == 0:
                row, col = fixed, pos
            elif label % 4 == 1:
                row, col = pos, fixed
            elif label % 4 == 2:
                row, col = pos, pos
            else:
                row, col = pos, max_pos - pos
            row = (row + start) % (max_pos + 1)
            col = (col + fixed) % (max_pos + 1)
            x[sample_idx, t, row : row + square_size, col : col + square_size] = color

    y = _make_ramp_targets(labels, num_steps=num_frames, num_classes=num_classes)
    return x, y, labels


@tf.function
def _convosc_rollout(
    x_r: tf.Tensor,
    x_i: tf.Tensor,
    omegas: tf.Tensor,
    num_steps: int,
    dt: float,
    mu: float,
    beta: float,
    input_scale: float,
    radius_epsilon: float,
) -> tuple[tf.Tensor, tf.Tensor]:
    """Article-style Hopf forcing on a convolutional feature map."""
    state_shape = tf.shape(x_r[:, 0])
    r_t = tf.ones(state_shape, dtype=tf.float32)
    theta_t = tf.zeros(state_shape, dtype=tf.float32)
    r_arr = tf.TensorArray(dtype=tf.float32, size=num_steps)
    theta_arr = tf.TensorArray(dtype=tf.float32, size=num_steps)

    for t in tf.range(num_steps):
        input_real = input_scale * x_r[:, t]
        input_imag = input_scale * x_i[:, t]
        input_amp = tf.sqrt(tf.square(input_real) + tf.square(input_imag))
        input_phase = tf.atan2(input_imag, input_real)
        psi_t = theta_t - input_phase
        safe_r = tf.maximum(r_t, radius_epsilon)
        r_dot = mu * r_t + beta * tf.pow(r_t, 3) + input_amp * tf.math.cos(psi_t)
        theta_dot = omegas - (input_amp / safe_r) * tf.math.sin(psi_t)
        r_t = tf.maximum(r_t + r_dot * dt, radius_epsilon)
        theta_t = theta_t + theta_dot * dt
        r_arr = r_arr.write(t, r_t)
        theta_arr = theta_arr.write(t, theta_t)

    r = tf.transpose(r_arr.stack(), [1, 0, 2, 3, 4])
    theta = tf.transpose(theta_arr.stack(), [1, 0, 2, 3, 4])
    return r, theta


class ConvOscLayer(tf.keras.layers.Layer):
    """Convolutional frontend followed by Hopf oscillator dynamics."""

    def __init__(
        self,
        filters: int,
        num_steps: int,
        kernel_size: int = 3,
        min_omega_hz: float = 1.0,
        max_omega_hz: float = 15.0,
        dt: float = 0.02,
        mu: float = 1.0,
        beta: float = -0.01,
        input_scale: float = 0.5,
        radius_epsilon: float = 1e-6,
        trainable_omegas: bool = True,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.filters = filters
        self.num_steps = num_steps
        self.dt = dt
        self.mu = mu
        self.beta = beta
        self.input_scale = input_scale
        self.radius_epsilon = radius_epsilon
        self.conv_r = tf.keras.layers.Conv2D(filters, kernel_size, padding="same", activation="relu")
        self.conv_i = tf.keras.layers.Conv2D(filters, kernel_size, padding="same", activation="relu")
        hz = np.linspace(min_omega_hz, max_omega_hz, filters, dtype=np.float32)
        omega_init = (hz * (2.0 * np.pi)).reshape(1, 1, 1, filters)
        self.omegas = self.add_weight(
            name="omegas",
            shape=(1, 1, 1, filters),
            dtype=tf.float32,
            initializer=tf.constant_initializer(omega_init),
            trainable=trainable_omegas,
        )

    def call(self, x: tf.Tensor) -> tf.Tensor:
        x_r = tf.keras.layers.TimeDistributed(self.conv_r)(x)
        x_i = tf.keras.layers.TimeDistributed(self.conv_i)(x)
        r, theta = _convosc_rollout(
            x_r=x_r,
            x_i=x_i,
            omegas=tf.convert_to_tensor(self.omegas),
            num_steps=self.num_steps,
            dt=self.dt,
            mu=self.mu,
            beta=self.beta,
            input_scale=self.input_scale,
            radius_epsilon=self.radius_epsilon,
        )
        return tf.concat([r * tf.math.cos(theta), r * tf.math.sin(theta)], axis=-1)


class ActionOCNNSmokeModel(tf.keras.Model):
    """2 x OCNN(3x3, filters) -> flatten -> output(num_classes)."""

    def __init__(
        self,
        num_steps: int,
        num_classes: int,
        filters: int = 8,
        dt: float = 0.02,
        input_scale: float = 0.5,
    ) -> None:
        super().__init__()
        self.osc1 = ConvOscLayer(filters, num_steps=num_steps, dt=dt, input_scale=input_scale)
        self.post1 = tf.keras.layers.Activation("relu")
        self.pool1 = tf.keras.layers.TimeDistributed(tf.keras.layers.AveragePooling2D(pool_size=2))
        self.osc2 = ConvOscLayer(filters, num_steps=num_steps, dt=dt, input_scale=input_scale)
        self.post2 = tf.keras.layers.Activation("relu")
        self.pool2 = tf.keras.layers.TimeDistributed(tf.keras.layers.AveragePooling2D(pool_size=2))
        self.flatten = tf.keras.layers.TimeDistributed(tf.keras.layers.Flatten())
        self.out = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(num_classes, activation="linear"))

    def call(self, x: tf.Tensor) -> tf.Tensor:
        h = self.post1(self.osc1(x))
        h = self.pool1(h)
        h = self.post2(self.osc2(h))
        h = self.pool2(h)
        return self.out(self.flatten(h))


def train_synthetic_smoke_run(
    seed: int,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    num_samples: int,
    num_frames: int,
    frame_size: int,
    square_size: int,
    num_classes: int,
    filters: int,
    val_ratio: float,
) -> tuple[ActionMetrics, np.ndarray, np.ndarray, np.ndarray, dict[str, list[float]], int]:
    set_seed(seed)
    x, y, _labels = generate_synthetic_action_dataset(
        num_samples=num_samples,
        num_frames=num_frames,
        frame_size=frame_size,
        square_size=square_size,
        num_classes=num_classes,
        seed=seed,
    )
    idx = np.arange(num_samples)
    rng = np.random.default_rng(seed)
    rng.shuffle(idx)
    n_val = int(round(num_samples * val_ratio))
    val_idx = idx[:n_val]
    train_idx = idx[n_val:]

    model = ActionOCNNSmokeModel(
        num_steps=num_frames,
        num_classes=num_classes,
        filters=filters,
        dt=0.02,
    )
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate), loss="mse")
    history = model.fit(
        x[train_idx],
        y[train_idx],
        validation_data=(x[val_idx], y[val_idx]),
        epochs=epochs,
        batch_size=batch_size,
        verbose=0,
    )
    pred_train = model.predict(x[train_idx], batch_size=batch_size, verbose=0)
    pred_val = model.predict(x[val_idx], batch_size=batch_size, verbose=0)
    train_loss = float(np.mean((pred_train - y[train_idx]) ** 2))
    val_loss = float(np.mean((pred_val - y[val_idx]) ** 2))
    metrics = ActionMetrics(
        val_acc=sequence_accuracy(y[val_idx], pred_val),
        val_loss=val_loss,
        train_acc=sequence_accuracy(y[train_idx], pred_train),
        train_loss=train_loss,
    )
    return metrics, x[val_idx], y[val_idx], pred_val, {k: [float(v) for v in vals] for k, vals in history.history.items()}, int(model.count_params())


def find_local_ucf_candidates(root: Path) -> list[str]:
    patterns = ("ucf", "action", "video")
    candidates: list[str] = []
    if not root.exists():
        return candidates
    for path in root.rglob("*"):
        if path.is_dir() and any(token in path.name.lower() for token in patterns):
            candidates.append(str(path))
            if len(candidates) >= 20:
                break
    return candidates
