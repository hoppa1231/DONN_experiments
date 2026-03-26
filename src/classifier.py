"""Alternative reproduction for task 1 using explicit classification objective.

Key difference from `donn_signal_classification.py`:
  - trains with sparse categorical cross-entropy on class labels
  - class labels are derived from ramp targets in Y
  - predicts class logits directly (instead of MSE over ramp sequences)
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import tensorflow as tf

from HopfLayer import HopfLayer, set_seed


def labels_from_y(y: np.ndarray) -> np.ndarray:
    """Convert ramp targets [N, T, 2] to class labels [N]."""
    return np.argmax(np.sum(y, axis=1), axis=1).astype(np.int64)


class DONNClassifierCE(tf.keras.Model):
    """Linear -> Hopf -> tanh projection -> temporal pooling -> class logits."""

    def __init__(
        self,
        num_steps: int,
        units: int = 20,
        proj_dim: int = 32,
        num_classes: int = 2,
        use_linear_frontend: bool = True,
        dropout: float = 0.0,
        hopf_input_scale: float = 0.1,
    ) -> None:
        super().__init__()
        self.units = units
        self.use_linear_frontend = use_linear_frontend

        self.in_r = tf.keras.layers.Dense(units, activation="relu")
        self.in_i = tf.keras.layers.Dense(units, activation="relu")
        self.hopf = HopfLayer(units=units, num_steps=num_steps, input_scale=hopf_input_scale)

        self.proj = tf.keras.layers.Dense(proj_dim, activation="tanh")
        self.pool = tf.keras.layers.GlobalAveragePooling1D()
        self.dropout = tf.keras.layers.Dropout(dropout)
        self.head = tf.keras.layers.Dense(num_classes, activation="linear")

        self.td_in_r = tf.keras.layers.TimeDistributed(self.in_r)
        self.td_in_i = tf.keras.layers.TimeDistributed(self.in_i)
        self.td_proj = tf.keras.layers.TimeDistributed(self.proj)

    def call(self, x: tf.Tensor, training: bool = False) -> tf.Tensor:
        if self.use_linear_frontend:
            x_r = self.td_in_r(x)
            x_i = self.td_in_i(x)
        else:
            x_r = tf.tile(x, [1, 1, self.units])
            x_i = tf.zeros_like(x_r)

        z_r, z_i = self.hopf(x_r, x_i)
        z = tf.concat([z_r, z_i], axis=2)
        h = self.td_proj(z)
        pooled = self.pool(h)
        pooled = self.dropout(pooled, training=training)
        logits = self.head(pooled)
        return logits


@dataclass
class Metrics:
    test_acc: float
    val_acc: float
    test_loss: float


def train_one_run(
    x: np.ndarray,
    y_cls: np.ndarray,
    seed: int,
    epochs: int,
    batch_size: int,
    test_ratio: float,
    learning_rate: float,
    use_linear_frontend: bool,
    units: int,
    proj_dim: int,
    dropout: float,
    hopf_input_scale: float,
) -> Metrics:
    set_seed(seed)

    n = x.shape[0]
    idx = np.arange(n)
    np.random.shuffle(idx)

    n_test = int(round(n * test_ratio))
    test_idx = idx[:n_test]
    train_idx = idx[n_test:]

    x_train, y_train = x[train_idx], y_cls[train_idx]
    x_test, y_test = x[test_idx], y_cls[test_idx]

    model = DONNClassifierCE(
        num_steps=x.shape[1],
        units=units,
        proj_dim=proj_dim,
        num_classes=2,
        use_linear_frontend=use_linear_frontend,
        dropout=dropout,
        hopf_input_scale=hopf_input_scale,
    )
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy(name="acc")],
    )

    history = model.fit(
        x_train,
        y_train,
        validation_split=0.2,
        epochs=epochs,
        batch_size=batch_size,
        verbose=0,
    )

    test_loss, test_acc = model.evaluate(x_test, y_test, batch_size=batch_size, verbose=0)
    logits = model.predict(x_test, batch_size=batch_size, verbose=0)
    pred_cls = np.argmax(logits, axis=1)
    true_hist = np.bincount(y_test, minlength=2)
    pred_hist = np.bincount(pred_cls, minlength=2)
    print(f"  class_hist true={true_hist.tolist()} pred={pred_hist.tolist()}")

    val_acc = float(history.history["val_acc"][-1])
    return Metrics(test_acc=float(test_acc), val_acc=val_acc, test_loss=float(test_loss))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--x-path", type=Path, default=Path("artifacts/signal_generation/X.npy"))
    parser.add_argument("--y-path", type=Path, default=Path("artifacts/signal_generation/Y.npy"))
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--test-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--units", type=int, default=20)
    parser.add_argument("--proj-dim", type=int, default=32)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument(
        "--hopf-input-scale",
        type=float,
        default=5.0,
        help="Input coupling scale for Hopf layer (higher values improve class separability in this setup).",
    )
    parser.add_argument(
        "--use-linear-frontend",
        action="store_true",
        help="Use learnable linear frontend before Hopf layer.",
    )
    args = parser.parse_args()

    x = np.load(args.x_path).astype(np.float32)
    y = np.load(args.y_path).astype(np.float32)
    y_cls = labels_from_y(y)

    print(f"Loaded X={x.shape}, Y={y.shape}, y_cls={y_cls.shape}")
    print(f"Label histogram: {np.bincount(y_cls, minlength=2).tolist()}")

    all_metrics: list[Metrics] = []
    for run in range(args.runs):
        seed = args.seed + run
        m = train_one_run(
            x=x,
            y_cls=y_cls,
            seed=seed,
            epochs=args.epochs,
            batch_size=args.batch_size,
            test_ratio=args.test_ratio,
            learning_rate=args.learning_rate,
            use_linear_frontend=args.use_linear_frontend,
            units=args.units,
            proj_dim=args.proj_dim,
            dropout=args.dropout,
            hopf_input_scale=args.hopf_input_scale,
        )
        all_metrics.append(m)
        print(
            f"Run {run + 1}/{args.runs} seed={seed}: "
            f"test_acc={m.test_acc:.4f}, test_loss={m.test_loss:.6f}, val_acc={m.val_acc:.4f}"
        )

    acc = np.array([m.test_acc for m in all_metrics])
    loss = np.array([m.test_loss for m in all_metrics])
    val_acc = np.array([m.val_acc for m in all_metrics])
    print("---- Summary ----")
    print(f"Test accuracy mean+/-std: {acc.mean():.4f} +/- {acc.std():.4f}")
    print(f"Test loss mean+/-std: {loss.mean():.6f} +/- {loss.std():.6f}")
    print(f"Val  acc  mean+/-std: {val_acc.mean():.4f} +/- {val_acc.std():.4f}")


if __name__ == "__main__":
    main()
