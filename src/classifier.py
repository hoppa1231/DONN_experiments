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

from src.HopfLayer import HopfLayer, set_seed


def labels_from_y(y: np.ndarray) -> np.ndarray:
    """Convert ramp targets [N, T, 2] to class labels [N]."""
    return np.argmax(np.sum(y, axis=1), axis=1).astype(np.int64)


def make_ramp_targets(labels: np.ndarray, num_steps: int, num_classes: int = 2) -> np.ndarray:
    """Create paper-style ramp labels for sequence classification."""
    ramp = np.linspace(0.0, 1.0, num_steps, dtype=np.float32)
    y = np.zeros((labels.shape[0], num_steps, num_classes), dtype=np.float32)
    y[np.arange(labels.shape[0]), :, labels.astype(np.int64)] = ramp[None, :]
    return y


def ramp_accuracy(pred: np.ndarray, y_true: np.ndarray) -> float:
    """Pick the class whose output neuron has the highest mean across time."""
    pred_cls = np.argmax(np.mean(pred, axis=1), axis=1)
    true_cls = labels_from_y(y_true)
    return float(np.mean(pred_cls == true_cls))


def ramp_classification_report(pred: np.ndarray, y_true: np.ndarray) -> dict[str, float | list[int]]:
    """Evaluate ramp outputs with several plausible class readouts."""
    true_cls = labels_from_y(y_true)
    pred_mean = np.argmax(np.mean(pred, axis=1), axis=1)
    pred_sum = np.argmax(np.sum(pred, axis=1), axis=1)
    pred_final = np.argmax(pred[:, -1, :], axis=1)

    class_templates = []
    for class_id in range(y_true.shape[2]):
        labels = np.full(y_true.shape[0], class_id, dtype=np.int64)
        class_templates.append(make_ramp_targets(labels=labels, num_steps=y_true.shape[1]))
    template_mse = np.stack([np.mean((pred - template) ** 2, axis=(1, 2)) for template in class_templates], axis=1)
    pred_template = np.argmin(template_mse, axis=1)

    return {
        "acc_mean": float(np.mean(pred_mean == true_cls)),
        "acc_sum": float(np.mean(pred_sum == true_cls)),
        "acc_final": float(np.mean(pred_final == true_cls)),
        "acc_template_mse": float(np.mean(pred_template == true_cls)),
        "class_hist_true": np.bincount(true_cls, minlength=y_true.shape[2]).tolist(),
        "class_hist_mean": np.bincount(pred_mean, minlength=y_true.shape[2]).tolist(),
        "class_hist_final": np.bincount(pred_final, minlength=y_true.shape[2]).tolist(),
        "class_hist_template_mse": np.bincount(pred_template, minlength=y_true.shape[2]).tolist(),
    }


def generate_classification_dataset(
    samples_per_class: int = 500,
    num_steps: int = 1000,
    dt: float = 0.001,
    num_components: int = 5,
    seed: int = 42,
    source: str = "article",
) -> tuple[np.ndarray, np.ndarray]:
    """Generate Table-1-style data from either the article text or the notebook.

    source="article":
      - sine waves
      - amplitudes U[-3, 3]
      - phases U[0, 2*pi]
      - low band [0, 10] Hz, high band [10, 20] Hz
      - no extra noise term mentioned in the text

    source="supplement-notebook":
      - cosine waves
      - amplitudes sampled from np.arange(-3, 3, 0.1)
      - no phase offsets
      - low band np.arange(0.1, 10, 1), high band np.arange(10.1, 20, 1)
      - additive white noise N(0, 0.15)
    """
    if source not in {"article", "supplement-notebook"}:
        raise ValueError("source must be 'article' or 'supplement-notebook'")

    rng = np.random.default_rng(seed)
    t = np.arange(0.0, num_steps * dt, dt, dtype=np.float32)
    total_samples = samples_per_class * 2
    x = np.zeros((total_samples, t.shape[0], 1), dtype=np.float32)
    labels = np.zeros(total_samples, dtype=np.int64)

    for class_id in range(2):
        if source == "article":
            fmin, fmax = ((0.0, 10.0) if class_id == 0 else (10.0, 20.0))
        else:
            freq_grid = (
                np.arange(0.1, 10.0, 1.0, dtype=np.float32)
                if class_id == 0
                else np.arange(10.1, 20.0, 1.0, dtype=np.float32)
            )
            amp_grid = np.arange(-3.0, 3.0, 0.1, dtype=np.float32)

        for sample_offset in range(samples_per_class):
            sample_idx = class_id * samples_per_class + sample_offset
            labels[sample_idx] = class_id
            signal = np.zeros_like(t)

            if source == "article":
                amps = rng.uniform(-3.0, 3.0, size=num_components).astype(np.float32)
                freqs = rng.uniform(fmin, fmax, size=num_components).astype(np.float32)
                phases = rng.uniform(0.0, 2.0 * np.pi, size=num_components).astype(np.float32)
                for amp, freq, phase in zip(amps, freqs, phases, strict=True):
                    signal += amp * np.sin(2.0 * np.pi * freq * t + phase)
            else:
                amps = rng.choice(amp_grid, size=num_components, replace=True).astype(np.float32)
                freqs = rng.choice(freq_grid, size=num_components, replace=True).astype(np.float32)
                for amp, freq in zip(amps, freqs, strict=True):
                    signal += amp * np.cos(2.0 * np.pi * freq * t)
                signal += rng.normal(0.0, 0.15, size=t.shape[0]).astype(np.float32)

            x[sample_idx, :, 0] = signal

    y = make_ramp_targets(labels=labels, num_steps=t.shape[0])
    return x, y


class DONNClassifierCE(tf.keras.Model):
    """Linear -> Hopf -> tanh projection -> temporal pooling -> class logits."""

    def __init__(
        self,
        num_steps: int,
        units: int = 20,
        proj_dim: int = 20,
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


class DONNClassifierRamp(tf.keras.Model):
    """Closer Table-1 control: linear frontend -> Hopf -> tanh -> ramp outputs."""

    def __init__(
        self,
        num_steps: int,
        units: int = 20,
        proj_dim: int = 20,
        num_classes: int = 2,
        min_omega_hz: float = 0.1,
        max_omega_hz: float = 20.0,
        dt: float = 0.001,
        hopf_input_scale: float = 0.1,
    ) -> None:
        super().__init__()
        self.units = units
        self.in_r = tf.keras.layers.Dense(units, activation="linear")
        self.hopf = HopfLayer(
            units=units,
            num_steps=num_steps,
            min_omega_hz=min_omega_hz,
            max_omega_hz=max_omega_hz,
            dt=dt,
            input_scale=hopf_input_scale,
            trainable_omegas=False,
        )
        self.proj = tf.keras.layers.Dense(proj_dim, activation="tanh")
        self.head = tf.keras.layers.Dense(num_classes, activation="linear")

        self.td_in_r = tf.keras.layers.TimeDistributed(self.in_r)
        self.td_proj = tf.keras.layers.TimeDistributed(self.proj)
        self.td_head = tf.keras.layers.TimeDistributed(self.head)

    def call(self, x: tf.Tensor) -> tf.Tensor:
        x_r = self.td_in_r(x)
        x_i = tf.zeros_like(x_r)
        z_r, z_i = self.hopf(x_r, x_i)
        amp = tf.sqrt(tf.square(z_r) + tf.square(z_i))
        h = self.td_proj(amp)
        return self.td_head(h)


@dataclass
class Metrics:
    test_acc: float
    val_acc: float
    test_loss: float


def _split_train_test(
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


def _split_train_val(
    x: np.ndarray,
    y: np.ndarray,
    val_ratio: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    idx = np.arange(x.shape[0])
    rng = np.random.default_rng(seed)
    rng.shuffle(idx)
    n_val = int(round(x.shape[0] * val_ratio))
    val_idx = idx[:n_val]
    train_idx = idx[n_val:]
    return x[train_idx], y[train_idx], x[val_idx], y[val_idx]


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


def train_one_ramp_run(
    x: np.ndarray,
    y: np.ndarray,
    seed: int,
    epochs: int,
    batch_size: int,
    test_ratio: float,
    learning_rate: float,
    units: int,
    proj_dim: int,
    hopf_input_scale: float,
    min_omega_hz: float,
    max_omega_hz: float,
    dt: float,
) -> tuple[Metrics, np.ndarray, np.ndarray, np.ndarray, DONNClassifierRamp]:
    """Train the article-style ramp classifier and return predictions."""
    set_seed(seed)
    x_train, y_train, x_test, y_test = _split_train_test(x=x, y=y, test_ratio=test_ratio, seed=seed)
    x_fit, y_fit, x_val, y_val = _split_train_val(x=x_train, y=y_train, val_ratio=0.2, seed=seed)

    model = DONNClassifierRamp(
        num_steps=x.shape[1],
        units=units,
        proj_dim=proj_dim,
        hopf_input_scale=hopf_input_scale,
        min_omega_hz=min_omega_hz,
        max_omega_hz=max_omega_hz,
        dt=dt,
    )
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="mse",
    )
    model.fit(
        x_fit,
        y_fit,
        validation_data=(x_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        verbose=0,
    )

    pred_test = model.predict(x_test, batch_size=batch_size, verbose=0)
    pred_val = model.predict(x_val, batch_size=batch_size, verbose=0)
    test_loss = float(np.mean((pred_test - y_test) ** 2))
    test_acc = ramp_accuracy(pred_test, y_test)
    val_acc = ramp_accuracy(pred_val, y_val)
    return (
        Metrics(test_acc=test_acc, val_acc=val_acc, test_loss=test_loss),
        x_test,
        y_test,
        pred_test,
        model,
    )


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
    parser.add_argument("--proj-dim", type=int, default=20)
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
