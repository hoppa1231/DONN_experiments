"""Strict paper-architecture control for Table 1 classification."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.HopfLayer import set_seed
from src.classifier import generate_classification_dataset, labels_from_y, ramp_classification_report
from src.paper_donn import PaperClassificationDONN


def split_train_test(x: np.ndarray, y: np.ndarray, test_ratio: float, seed: int):
    idx = np.arange(x.shape[0])
    rng = np.random.default_rng(seed)
    rng.shuffle(idx)
    n_test = int(round(x.shape[0] * test_ratio))
    return x[idx[n_test:]], y[idx[n_test:]], x[idx[:n_test]], y[idx[:n_test]]


def plot_report(out_path: Path, t: np.ndarray, x_test: np.ndarray, y_test: np.ndarray, pred: np.ndarray, result):
    fig, axes = plt.subplots(2, 2, figsize=(14, 8), sharex=True)
    axes = axes.ravel()
    labels = labels_from_y(y_test)
    for idx, class_id in enumerate([0, 1]):
        choices = np.where(labels == class_id)[0]
        sample_idx = int(choices[0]) if len(choices) else 0
        axes[2 * idx].plot(t, x_test[sample_idx, :, 0], color="tab:purple", lw=1.2)
        axes[2 * idx].set_title(f"input | class {class_id}")
        axes[2 * idx].grid(alpha=0.25)
        axes[2 * idx + 1].plot(t, y_test[sample_idx, :, 0], "k--", lw=1.0, label="target 0")
        axes[2 * idx + 1].plot(t, y_test[sample_idx, :, 1], "gray", ls="--", lw=1.0, label="target 1")
        axes[2 * idx + 1].plot(t, pred[sample_idx, :, 0], color="tab:blue", lw=1.2, label="pred 0")
        axes[2 * idx + 1].plot(t, pred[sample_idx, :, 1], color="tab:red", lw=1.2, label="pred 1")
        axes[2 * idx + 1].set_title(f"ramp target vs prediction | class {class_id}")
        axes[2 * idx + 1].grid(alpha=0.25)
        axes[2 * idx + 1].legend(fontsize=8)
    fig.suptitle(
        "Table 1 strict paper architecture | "
        f"mean_acc={result['accuracy_report']['acc_mean']:.4f}, "
        f"final_acc={result['accuracy_report']['acc_final']:.4f}, loss={result['test_loss']:.6f}",
        fontsize=13,
    )
    for ax in axes:
        ax.set_xlabel("time, s")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=170, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-source", choices=["article", "supplement-notebook"], default="article")
    parser.add_argument("--samples-per-class", type=int, default=500)
    parser.add_argument("--num-steps", type=int, default=1000)
    parser.add_argument("--dt", type=float, default=0.001)
    parser.add_argument("--num-components", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--test-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--clipnorm", type=float, default=None)
    parser.add_argument("--units", type=int, default=20)
    parser.add_argument("--min-omega-hz", type=float, default=0.1)
    parser.add_argument("--max-omega-hz", type=float, default=20.0)
    parser.add_argument("--hopf-input-scale", type=float, default=0.1)
    parser.add_argument("--mu", type=float, default=1.0)
    parser.add_argument("--beta", type=float, default=-100.0)
    parser.add_argument(
        "--out-path",
        type=Path,
        default=Path("artifacts/plots/paper_exact/classification_paper_sequence_summary.png"),
    )
    parser.add_argument(
        "--metrics-path",
        type=Path,
        default=Path("artifacts/plots/paper_exact/classification_paper_sequence_metrics.json"),
    )
    args = parser.parse_args()

    set_seed(args.seed)
    x, y = generate_classification_dataset(
        samples_per_class=args.samples_per_class,
        num_steps=args.num_steps,
        dt=args.dt,
        num_components=args.num_components,
        seed=args.seed,
        source=args.dataset_source,
    )
    x_train, y_train, x_test, y_test = split_train_test(x, y, args.test_ratio, args.seed)
    model = PaperClassificationDONN(
        num_steps=x.shape[1],
        units=args.units,
        num_classes=y.shape[2],
        min_omega_hz=args.min_omega_hz,
        max_omega_hz=args.max_omega_hz,
        dt=args.dt,
        hopf_input_scale=args.hopf_input_scale,
        mu=args.mu,
        beta=args.beta,
    )
    optimizer = (
        tf.keras.optimizers.Adam(args.learning_rate, clipnorm=args.clipnorm)
        if args.clipnorm
        else tf.keras.optimizers.Adam(args.learning_rate)
    )
    model.compile(optimizer=optimizer, loss="mse")
    history = model.fit(
        x_train,
        y_train,
        validation_split=0.2,
        epochs=args.epochs,
        batch_size=args.batch_size,
        verbose=0,
    )
    pred = model.predict(x_test, batch_size=args.batch_size, verbose=0)
    report = ramp_classification_report(pred, y_test)
    result = {
        "variant": "strict_paper_table1_architecture",
        "is_paper_architecture": True,
        "dataset_source": args.dataset_source,
        "article_architecture": "Linear(20), Hopf(20), tanh(20), output(2)",
        "paper_reported_accuracy": 0.99,
        "test_loss": float(np.mean((pred - y_test) ** 2)),
        "val_loss": float(history.history["val_loss"][-1]),
        "accuracy_report": report,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "seed": args.seed,
        "learning_rate": args.learning_rate,
        "clipnorm": args.clipnorm,
        "units": args.units,
        "min_omega_hz": args.min_omega_hz,
        "max_omega_hz": args.max_omega_hz,
        "hopf_input_scale": args.hopf_input_scale,
        "mu": args.mu,
        "beta": args.beta,
        "samples_per_class": args.samples_per_class,
        "num_steps": args.num_steps,
        "dt": args.dt,
        "num_components": args.num_components,
        "assumptions": [
            "Static layers are implemented as complex dense layers with split real/imag activation per Eq. (20).",
            "The real part of the final complex output layer is compared to ramp targets.",
            "The paper does not specify exact train/test split, batch size, optimizer settings, or seed.",
        ],
    }
    t = np.arange(x.shape[1], dtype=np.float32) * args.dt
    plot_report(args.out_path, t, x_test, y_test, pred, result)
    args.metrics_path.parent.mkdir(parents=True, exist_ok=True)
    args.metrics_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    print(f"Saved figure: {args.out_path}")
    print(f"Saved metrics: {args.metrics_path}")


if __name__ == "__main__":
    main()
