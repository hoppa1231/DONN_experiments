"""Reproduce DONN signal generation/classification experiment (Table 1).

This script trains a small DONN-style model on:
  artifacts/signal_generation/X.npy
  artifacts/signal_generation/Y.npy

Reference from paper (Table 1):
  Initial frequency range of oscillators: [0.1-20 Hz]
  Architecture: Linear(20) -> Hopf(20) -> tanh(20) -> output(2)
  Input type to oscillators: I(t)
  Frequency of oscillators: Not trained
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import numpy as np
import tensorflow as tf


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    tf.random.set_seed(seed)


@tf.function
def _real_part(r: tf.Tensor, phi: tf.Tensor) -> tf.Tensor:
    return r * tf.math.cos(phi)


@tf.function
def _imag_part(r: tf.Tensor, phi: tf.Tensor) -> tf.Tensor:
    return r * tf.math.sin(phi)


@tf.function
def _hopf_rollout(
    x_r: tf.Tensor,
    x_i: tf.Tensor,
    omegas: tf.Tensor,
    num_steps: int,
    dt: float,
    mu: float,
    beta: float,
    input_scale: float,
) -> Tuple[tf.Tensor, tf.Tensor]:
    """Euler integration for a batch of Hopf oscillators."""
    batch_size = tf.shape(x_r)[0]
    dim = tf.shape(x_r)[2]

    r_t = tf.ones((batch_size, dim), dtype=tf.float32)
    phi_t = tf.zeros((batch_size, dim), dtype=tf.float32)

    r_arr = tf.TensorArray(dtype=tf.float32, size=num_steps)
    phi_arr = tf.TensorArray(dtype=tf.float32, size=num_steps)

    for t in tf.range(num_steps):
        input_r = input_scale * x_r[:, t, :] * tf.math.cos(phi_t)
        input_phi = input_scale * x_i[:, t, :] * tf.math.sin(phi_t)
        r_t = r_t + ((mu - beta * tf.square(r_t)) * r_t + input_r) * dt
        phi_t = phi_t + (omegas - input_phi) * dt
        r_arr = r_arr.write(t, r_t)
        phi_arr = phi_arr.write(t, phi_t)

    r = tf.transpose(r_arr.stack(), [1, 0, 2])
    phi = tf.transpose(phi_arr.stack(), [1, 0, 2])
    return r, phi


class HopfLayer(tf.keras.layers.Layer):
    def __init__(
        self,
        units: int,
        num_steps: int,
        min_omega_hz: float = 0.1,
        max_omega_hz: float = 20.0,
        dt: float = 0.001,
        mu: float = 1.0,
        beta: float = 0.01,
        input_scale: float = 0.1,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.units = units
        self.num_steps = num_steps
        self.dt = dt
        self.mu = mu
        self.beta = beta
        self.input_scale = input_scale

        hz = tf.linspace(min_omega_hz, max_omega_hz, units)
        self.omegas = tf.cast(tf.expand_dims(hz * (2.0 * np.pi), 0), tf.float32)

    def call(self, x_r: tf.Tensor, x_i: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        r, phi = _hopf_rollout(
            x_r=x_r,
            x_i=x_i,
            omegas=self.omegas,
            num_steps=self.num_steps,
            dt=self.dt,
            mu=self.mu,
            beta=self.beta,
            input_scale=self.input_scale,
        )
        return _real_part(r, phi), _imag_part(r, phi)


class DONNClassifier(tf.keras.Model):
    """Linear(20) -> Hopf(20) -> tanh(20) -> output(2), time-distributed."""

    def __init__(
        self,
        num_steps: int,
        units: int = 20,
        out_dim: int = 2,
        use_linear_frontend: bool = True,
    ) -> None:
        super().__init__()
        self.units = units
        self.use_linear_frontend = use_linear_frontend
        self.in_r = tf.keras.layers.Dense(units, activation="relu")
        self.in_i = tf.keras.layers.Dense(units, activation="relu")
        self.hopf = HopfLayer(units=units, num_steps=num_steps)
        self.tanh = tf.keras.layers.Dense(units, activation="tanh")
        self.out = tf.keras.layers.Dense(out_dim, activation="linear")
        self.td_in_r = tf.keras.layers.TimeDistributed(self.in_r)
        self.td_in_i = tf.keras.layers.TimeDistributed(self.in_i)
        self.td_tanh = tf.keras.layers.TimeDistributed(self.tanh)
        self.td_out = tf.keras.layers.TimeDistributed(self.out)

    def call(self, x: tf.Tensor) -> tf.Tensor:
        if self.use_linear_frontend:
            x_r = self.td_in_r(x)
            x_i = self.td_in_i(x)
        else:
            x_r = tf.tile(x, [1, 1, self.units])
            x_i = tf.zeros_like(x_r)
        z_r, z_i = self.hopf(x_r, x_i)
        z = tf.concat([z_r, z_i], axis=2)
        h = self.td_tanh(z)
        y = self.td_out(h)
        return y


@dataclass
class Metrics:
    test_mse: float
    test_acc: float
    val_mse: float


def train_one_run(
    x: np.ndarray,
    y: np.ndarray,
    seed: int,
    epochs: int,
    batch_size: int,
    test_ratio: float,
    use_linear_frontend: bool,
) -> Metrics:
    set_seed(seed)

    n = x.shape[0]
    idx = np.arange(n)
    np.random.shuffle(idx)

    n_test = int(round(n * test_ratio))
    test_idx = idx[:n_test]
    train_idx = idx[n_test:]

    x_train, y_train = x[train_idx], y[train_idx]
    x_test, y_test = x[test_idx], y[test_idx]

    model = DONNClassifier(
        num_steps=x.shape[1],
        units=20,
        out_dim=y.shape[2],
        use_linear_frontend=use_linear_frontend,
    )
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-2),
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

    pred_cls = np.argmax(np.sum(pred, axis=1), axis=1)
    true_cls = np.argmax(np.sum(y_test, axis=1), axis=1)
    test_acc = float(np.mean(pred_cls == true_cls))
    pred_hist = np.bincount(pred_cls, minlength=y.shape[2])
    true_hist = np.bincount(true_cls, minlength=y.shape[2])
    print(f"  class_hist true={true_hist.tolist()} pred={pred_hist.tolist()}")

    val_mse = float(history.history["val_loss"][-1])
    return Metrics(test_mse=test_mse, test_acc=test_acc, val_mse=val_mse)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--x-path", type=Path, default=Path("artifacts/signal_generation/X.npy"))
    parser.add_argument("--y-path", type=Path, default=Path("artifacts/signal_generation/Y.npy"))
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--test-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--use-linear-frontend",
        action="store_true",
        help="Use learnable linear frontend before Hopf layer (paper table says Linear(20)).",
    )
    args = parser.parse_args()

    x = np.load(args.x_path).astype(np.float32)
    y = np.load(args.y_path).astype(np.float32)

    print(f"Loaded X={x.shape}, Y={y.shape}")
    all_metrics = []
    for run in range(args.runs):
        seed = args.seed + run
        m = train_one_run(
            x=x,
            y=y,
            seed=seed,
            epochs=args.epochs,
            batch_size=args.batch_size,
            test_ratio=args.test_ratio,
            use_linear_frontend=args.use_linear_frontend,
        )
        all_metrics.append(m)
        print(
            f"Run {run + 1}/{args.runs} seed={seed}: "
            f"test_acc={m.test_acc:.4f}, test_mse={m.test_mse:.6f}, val_mse={m.val_mse:.6f}"
        )

    acc = np.array([m.test_acc for m in all_metrics])
    mse = np.array([m.test_mse for m in all_metrics])
    val_mse = np.array([m.val_mse for m in all_metrics])
    print("---- Summary ----")
    print(f"Test accuracy mean±std: {acc.mean():.4f} ± {acc.std():.4f}")
    print(f"Test MSE mean±std: {mse.mean():.6f} ± {mse.std():.6f}")
    print(f"Val  MSE mean±std: {val_mse.mean():.6f} ± {val_mse.std():.6f}")


if __name__ == "__main__":
    main()
