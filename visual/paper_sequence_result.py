"""Strict paper-architecture controls for Table 2 and Table 3 sequence tasks."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.demodulation import generate_demod_dataset
from src.operators import generate_operator_dataset, numeric_baseline
from src.paper_donn import train_paper_sequence_regressor


def build_dataset(args: argparse.Namespace) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, object]]:
    if args.task == "demodulation":
        x, y, t = generate_demod_dataset(
            num_samples=args.num_samples,
            dt=args.dt,
            duration=args.duration,
            carrier_hz=args.carrier_hz,
            num_components=args.num_components,
            msg_fmin=args.msg_fmin,
            msg_fmax=args.msg_fmax,
            seed=args.seed,
        )
        article = {
            "table": "Table 2",
            "reported_validation_mse": 0.02,
            "architecture": "ReLU(40), Hopf(40), ReLU(40), Hopf(40), tanh(40), output(1)",
            "initial_frequency_range_hz": [0.1, 12.0],
            "input_type": "I(t)",
            "oscillator_frequencies": "not trained",
            "carrier_hz": args.carrier_hz,
            "message_range_hz": [args.msg_fmin, args.msg_fmax],
        }
    else:
        x, y, t = generate_operator_dataset(
            task=args.task,
            num_samples=args.num_samples,
            dt=args.dt,
            duration=args.duration,
            num_components=args.num_components,
            fmin_hz=args.fmin_hz,
            fmax_hz=args.fmax_hz,
            seed=args.seed,
        )
        article = {
            "table": "Table 3",
            "reported_validation_mse": 0.08 if args.task == "integration" else 0.1,
            "architecture": "ReLU(20), Hopf(20), ReLU(20), Hopf(20), tanh(20), output(1)",
            "initial_frequency_range_hz": [1.0, 10.0],
            "input_type": "I(t)",
            "oscillator_frequencies": "not trained",
            "input_range_hz": [args.fmin_hz, args.fmax_hz],
        }
    return x, y, t, article


def plot_report(
    out_path: Path,
    t: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    pred: np.ndarray,
    result: dict[str, object],
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(14, 8), sharex=True)
    axes = axes.ravel()
    for idx, sample_idx in enumerate([0, min(1, x_test.shape[0] - 1)]):
        axes[2 * idx].plot(t, x_test[sample_idx, :, 0], color="tab:purple", lw=1.2)
        axes[2 * idx].set_title(f"input | sample {sample_idx}")
        axes[2 * idx].grid(alpha=0.25)
        axes[2 * idx + 1].plot(t, y_test[sample_idx, :, 0], color="tab:orange", lw=1.3, label="target")
        axes[2 * idx + 1].plot(t, pred[sample_idx, :, 0], color="tab:blue", lw=1.3, label="prediction")
        axes[2 * idx + 1].set_title(f"target vs prediction | sample {sample_idx}")
        axes[2 * idx + 1].grid(alpha=0.25)
        axes[2 * idx + 1].legend()
    fig.suptitle(
        f"{result['task']} strict paper architecture | "
        f"test_mse={result['test_mse']:.6f}, val_mse={result['val_mse']:.6f}, "
        f"corr={result['test_corr']:.4f}",
        fontsize=13,
    )
    for ax in axes:
        ax.set_xlabel("time, s")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=170, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=["demodulation", "integration", "differentiation"], required=True)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--test-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--clipnorm", type=float, default=None)
    parser.add_argument("--num-samples", type=int, default=200)
    parser.add_argument("--dt", type=float, default=0.01)
    parser.add_argument("--duration", type=float, default=1.0)
    parser.add_argument("--num-components", type=int, default=5)
    parser.add_argument("--carrier-hz", type=float, default=8.0)
    parser.add_argument("--msg-fmin", type=float, default=1.0)
    parser.add_argument("--msg-fmax", type=float, default=5.0)
    parser.add_argument("--fmin-hz", type=float, default=1.0)
    parser.add_argument("--fmax-hz", type=float, default=5.0)
    parser.add_argument("--units", type=int, default=None)
    parser.add_argument("--min-omega-hz", type=float, default=None)
    parser.add_argument("--max-omega-hz", type=float, default=None)
    parser.add_argument("--hopf-input-scale", type=float, default=0.1)
    parser.add_argument("--mu", type=float, default=1.0)
    parser.add_argument("--beta", type=float, default=-100.0)
    parser.add_argument("--scale-data", action="store_true")
    parser.add_argument("--out-path", type=Path, default=None)
    parser.add_argument("--metrics-path", type=Path, default=None)
    args = parser.parse_args()

    if args.units is None:
        args.units = 40 if args.task == "demodulation" else 20
    if args.min_omega_hz is None:
        args.min_omega_hz = 0.1 if args.task == "demodulation" else 1.0
    if args.max_omega_hz is None:
        args.max_omega_hz = 12.0 if args.task == "demodulation" else 10.0
    if args.out_path is None:
        args.out_path = Path(f"artifacts/plots/paper_exact/{args.task}_paper_sequence_summary.png")
    if args.metrics_path is None:
        args.metrics_path = Path(f"artifacts/plots/paper_exact/{args.task}_paper_sequence_metrics.json")

    x, y, t, article = build_dataset(args)
    metrics, x_test, y_test, pred, scale_info = train_paper_sequence_regressor(
        x=x,
        y=y,
        seed=args.seed,
        epochs=args.epochs,
        batch_size=args.batch_size,
        test_ratio=args.test_ratio,
        learning_rate=args.learning_rate,
        clipnorm=args.clipnorm,
        units=args.units,
        min_omega_hz=args.min_omega_hz,
        max_omega_hz=args.max_omega_hz,
        dt=args.dt,
        hopf_input_scale=args.hopf_input_scale,
        mu=args.mu,
        beta=args.beta,
        trainable_omegas=False,
        scale_data=args.scale_data,
    )

    result = {
        "variant": "strict_paper_sequence_architecture",
        "is_paper_architecture": True,
        "task": args.task,
        "article": article,
        "test_mse": metrics.test_mse,
        "val_mse": metrics.val_mse,
        "test_corr": metrics.test_corr,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "seed": args.seed,
        "learning_rate": args.learning_rate,
        "clipnorm": args.clipnorm,
        "num_samples": args.num_samples,
        "dt": args.dt,
        "duration": args.duration,
        "num_components": args.num_components,
        "units": args.units,
        "min_omega_hz": args.min_omega_hz,
        "max_omega_hz": args.max_omega_hz,
        "hopf_input_scale": args.hopf_input_scale,
        "mu": args.mu,
        "beta": args.beta,
        "scale_data": bool(args.scale_data),
        **scale_info,
        "assumptions": [
            "Static layers are implemented as complex dense layers with split real/imag activation per Eq. (20).",
            "The real part of the final complex output layer is used as the real-valued target sequence.",
            "The paper does not specify dataset size, train/test split, batch size, or exact initialization seed.",
        ],
    }
    if args.task in {"integration", "differentiation"}:
        _, baseline_mse = numeric_baseline(task=args.task, x=x_test, y=y_test, dt=args.dt)
        result["numeric_baseline_mse"] = baseline_mse

    plot_report(args.out_path, t, x_test, y_test, pred, result)
    args.metrics_path.parent.mkdir(parents=True, exist_ok=True)
    args.metrics_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    print(f"Saved figure: {args.out_path}")
    print(f"Saved metrics: {args.metrics_path}")


if __name__ == "__main__":
    main()
