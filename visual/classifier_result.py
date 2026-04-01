"""Visualize task-1 classification using the CE-based DONN classifier.

This script trains `DONNClassifierCE`, then produces paper-like plots:
  A) example input signals from both classes
  B) target ramp outputs vs predicted class-probability ramps
  C) hidden Hopf oscillator amplitude profile by intrinsic frequency
"""

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

from src.classifier import DONNClassifierCE, labels_from_y, set_seed


def split_train_test(
    x: np.ndarray, y: np.ndarray, y_cls: np.ndarray, test_ratio: float, seed: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    idx = np.arange(x.shape[0])
    rng = np.random.default_rng(seed)
    rng.shuffle(idx)
    n_test = int(round(x.shape[0] * test_ratio))
    test_idx = idx[:n_test]
    train_idx = idx[n_test:]
    return (
        x[train_idx],
        y[train_idx],
        y_cls[train_idx],
        x[test_idx],
        y[test_idx],
        y_cls[test_idx],
    )


def select_two_classes(y_cls: np.ndarray) -> tuple[int, int]:
    idx0 = np.where(y_cls == 0)[0]
    idx1 = np.where(y_cls == 1)[0]
    if len(idx0) == 0 or len(idx1) == 0:
        raise RuntimeError("Both classes must be present in test split.")
    return int(idx0[0]), int(idx1[0])


def get_oscillator_amplitudes(model: DONNClassifierCE, x_batch: np.ndarray) -> np.ndarray:
    x_tf = tf.convert_to_tensor(x_batch, dtype=tf.float32)
    if model.use_linear_frontend:
        x_r = model.td_in_r(x_tf)
        x_i = model.td_in_i(x_tf)
    else:
        x_r = tf.tile(x_tf, [1, 1, model.units])
        x_i = tf.zeros_like(x_r)
    z_r, z_i = model.hopf(x_r, x_i)
    amp = tf.sqrt(tf.square(z_r) + tf.square(z_i))
    return amp.numpy()


def logits_to_ramps(logits: np.ndarray, y_template: np.ndarray) -> np.ndarray:
    probs = tf.nn.softmax(tf.convert_to_tensor(logits), axis=1).numpy()
    ramp = np.maximum(y_template[:, :, 0], y_template[:, :, 1])
    pred = np.zeros_like(y_template)
    pred[:, :, 0] = probs[:, 0][:, None] * ramp
    pred[:, :, 1] = probs[:, 1][:, None] * ramp
    return pred


def plot_report(
    out_path: Path,
    x_test: np.ndarray,
    y_test: np.ndarray,
    pred_seq: np.ndarray,
    amp: np.ndarray,
    hz: np.ndarray,
    i0: int,
    i1: int,
    metrics: dict[str, float | int | list[int] | bool],
) -> None:
    t = np.arange(x_test.shape[1], dtype=np.float32) * 0.001

    fig = plt.figure(figsize=(14, 10))
    gs = fig.add_gridspec(3, 2, height_ratios=[1.0, 1.1, 1.2], hspace=0.35, wspace=0.25)

    ax_a0 = fig.add_subplot(gs[0, 0])
    ax_a1 = fig.add_subplot(gs[0, 1])
    ax_b0 = fig.add_subplot(gs[1, 0])
    ax_b1 = fig.add_subplot(gs[1, 1])
    ax_c = fig.add_subplot(gs[2, :])

    ax_a0.plot(t, x_test[i0, :, 0], color="tab:blue", lw=1.2)
    ax_a0.set_title("A1) Input signal (class 0, low band)")
    ax_a0.set_xlabel("Time, s")
    ax_a0.set_ylabel("Amplitude")
    ax_a0.grid(alpha=0.25)

    ax_a1.plot(t, x_test[i1, :, 0], color="tab:orange", lw=1.2)
    ax_a1.set_title("A2) Input signal (class 1, high band)")
    ax_a1.set_xlabel("Time, s")
    ax_a1.set_ylabel("Amplitude")
    ax_a1.grid(alpha=0.25)

    ax_b0.plot(t, y_test[i0, :, 0], "k--", lw=1.2, label="target ch0")
    ax_b0.plot(t, y_test[i0, :, 1], "gray", ls="--", lw=1.2, label="target ch1")
    ax_b0.plot(t, pred_seq[i0, :, 0], color="tab:blue", lw=1.2, label="pred ch0")
    ax_b0.plot(t, pred_seq[i0, :, 1], color="tab:red", lw=1.2, label="pred ch1")
    ax_b0.set_title("B1) Target ramps vs CE probability ramps (class 0 sample)")
    ax_b0.set_xlabel("Time, s")
    ax_b0.set_ylabel("Output")
    ax_b0.grid(alpha=0.25)
    ax_b0.legend(loc="upper left", fontsize=8, ncol=2)

    ax_b1.plot(t, y_test[i1, :, 0], "k--", lw=1.2, label="target ch0")
    ax_b1.plot(t, y_test[i1, :, 1], "gray", ls="--", lw=1.2, label="target ch1")
    ax_b1.plot(t, pred_seq[i1, :, 0], color="tab:blue", lw=1.2, label="pred ch0")
    ax_b1.plot(t, pred_seq[i1, :, 1], color="tab:red", lw=1.2, label="pred ch1")
    ax_b1.set_title("B2) Target ramps vs CE probability ramps (class 1 sample)")
    ax_b1.set_xlabel("Time, s")
    ax_b1.set_ylabel("Output")
    ax_b1.grid(alpha=0.25)
    ax_b1.legend(loc="upper left", fontsize=8, ncol=2)

    amp_mean0 = amp[0].mean(axis=0)
    amp_mean1 = amp[1].mean(axis=0)
    ax_c.axvspan(0.1, 10.0, color="tab:blue", alpha=0.08, label="Low band 0.1-10 Hz")
    ax_c.axvspan(10.0, 20.0, color="tab:orange", alpha=0.08, label="High band 10-20 Hz")
    ax_c.plot(hz, amp_mean0, marker="o", lw=1.4, color="tab:blue", label="Class 0 sample")
    ax_c.plot(hz, amp_mean1, marker="o", lw=1.4, color="tab:orange", label="Class 1 sample")
    ax_c.set_title("C) Hidden Hopf oscillator amplitude profile by intrinsic frequency")
    ax_c.set_xlabel("Oscillator intrinsic frequency, Hz")
    ax_c.set_ylabel("Mean amplitude over time")
    ax_c.grid(alpha=0.25)
    ax_c.legend(loc="upper right", fontsize=9)

    fig.suptitle(
        "Task 1 Visual Report (CE DONN) | "
        f"test_acc={metrics['test_acc']:.4f}, test_loss={metrics['test_loss']:.6f}, "
        f"plot_mse={metrics['plot_mse']:.6f}",
        fontsize=13,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=170, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--x-path", type=Path, default=Path("artifacts/signal_generation/X.npy"))
    parser.add_argument("--y-path", type=Path, default=Path("artifacts/signal_generation/Y.npy"))
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--test-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--units", type=int, default=20)
    parser.add_argument("--proj-dim", type=int, default=32)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--hopf-input-scale", type=float, default=5.0)
    parser.add_argument("--use-linear-frontend", action="store_true")
    parser.add_argument(
        "--out-path",
        type=Path,
        default=Path("artifacts/plots/first_work_visual_comparison_ce.png"),
    )
    parser.add_argument(
        "--metrics-path",
        type=Path,
        default=Path("artifacts/plots/first_work_visual_metrics_ce.json"),
    )
    args = parser.parse_args()

    set_seed(args.seed)
    x = np.load(args.x_path).astype(np.float32)
    y = np.load(args.y_path).astype(np.float32)
    y_cls = labels_from_y(y)

    x_train, y_train, y_cls_train, x_test, y_test, y_cls_test = split_train_test(
        x=x,
        y=y,
        y_cls=y_cls,
        test_ratio=args.test_ratio,
        seed=args.seed,
    )

    model = DONNClassifierCE(
        num_steps=x.shape[1],
        units=args.units,
        proj_dim=args.proj_dim,
        use_linear_frontend=args.use_linear_frontend,
        dropout=args.dropout,
        hopf_input_scale=args.hopf_input_scale,
    )
    model.compile(
        optimizer=tf.keras.optimizers.Adam(args.learning_rate),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy(name="acc")],
    )
    model.fit(
        x_train,
        y_cls_train,
        validation_split=0.2,
        epochs=args.epochs,
        batch_size=args.batch_size,
        verbose=0,
    )

    test_loss, test_acc = model.evaluate(x_test, y_cls_test, batch_size=args.batch_size, verbose=0)
    logits = model.predict(x_test, batch_size=args.batch_size, verbose=0)
    pred_cls = np.argmax(logits, axis=1)
    pred_seq = logits_to_ramps(logits, y_test)
    plot_mse = float(np.mean((pred_seq - y_test) ** 2))

    pred_hist = np.bincount(pred_cls, minlength=2).tolist()
    true_hist = np.bincount(y_cls_test, minlength=2).tolist()

    i0, i1 = select_two_classes(y_cls_test)
    x_pair = np.stack([x_test[i0], x_test[i1]], axis=0)
    amp_pair = get_oscillator_amplitudes(model, x_pair)
    hz = (model.hopf.omegas.numpy().squeeze() / (2.0 * np.pi)).astype(np.float32)

    metrics = {
        "test_acc": float(test_acc),
        "test_loss": float(test_loss),
        "plot_mse": plot_mse,
        "class_hist_true": true_hist,
        "class_hist_pred": pred_hist,
        "epochs": args.epochs,
        "seed": args.seed,
        "use_linear_frontend": bool(args.use_linear_frontend),
        "hopf_input_scale": args.hopf_input_scale,
    }

    plot_report(
        out_path=args.out_path,
        x_test=x_test,
        y_test=y_test,
        pred_seq=pred_seq,
        amp=amp_pair,
        hz=hz,
        i0=i0,
        i1=i1,
        metrics=metrics,
    )

    args.metrics_path.parent.mkdir(parents=True, exist_ok=True)
    args.metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))
    print(f"Saved figure: {args.out_path}")
    print(f"Saved metrics: {args.metrics_path}")


if __name__ == "__main__":
    main()
