"""Visual report for Table 2: amplitude demodulation."""

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

from src.demodulation import generate_demod_dataset, train_one_run


def plot_report(
    out_path: Path,
    t: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    pred: np.ndarray,
    metrics: dict[str, float | int | bool],
) -> None:
    fig = plt.figure(figsize=(14, 9))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 1.05], hspace=0.3, wspace=0.25)

    ax_a0 = fig.add_subplot(gs[0, 0])
    ax_a1 = fig.add_subplot(gs[0, 1])
    ax_b0 = fig.add_subplot(gs[1, 0])
    ax_b1 = fig.add_subplot(gs[1, 1])

    ax_a0.plot(t, y_test[0, :, 0], color="tab:orange", lw=1.3)
    ax_a0.set_title("A1) Message signal m(t)")
    ax_a0.set_xlabel("Time, s")
    ax_a0.set_ylabel("Amplitude")
    ax_a0.grid(alpha=0.25)

    ax_a1.plot(t, x_test[0, :, 0], color="tab:purple", lw=1.3)
    ax_a1.set_title("A2) Modulated input M(t)")
    ax_a1.set_xlabel("Time, s")
    ax_a1.set_ylabel("Amplitude")
    ax_a1.grid(alpha=0.25)

    ax_b0.plot(t, y_test[0, :, 0], color="tab:orange", lw=1.4, label="target m(t)")
    ax_b0.plot(t, pred[0, :, 0], color="tab:blue", lw=1.4, label="predicted m(t)")
    ax_b0.set_title("B1) Demodulation result | sample 0")
    ax_b0.set_xlabel("Time, s")
    ax_b0.set_ylabel("Amplitude")
    ax_b0.grid(alpha=0.25)
    ax_b0.legend(loc="upper right")

    ax_b1.plot(t, y_test[1, :, 0], color="tab:orange", lw=1.4, label="target m(t)")
    ax_b1.plot(t, pred[1, :, 0], color="tab:blue", lw=1.4, label="predicted m(t)")
    ax_b1.set_title("B2) Demodulation result | sample 1")
    ax_b1.set_xlabel("Time, s")
    ax_b1.set_ylabel("Amplitude")
    ax_b1.grid(alpha=0.25)
    ax_b1.legend(loc="upper right")

    fig.suptitle(
        "Task 2 Visual Report (Amplitude Demodulation) | "
        f"test_mse={metrics['test_mse']:.6f}, val_mse={metrics['val_mse']:.6f}",
        fontsize=13,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=170, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--test-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--learning-rate", type=float, default=1e-2)
    parser.add_argument("--units", type=int, default=40)
    parser.add_argument("--hopf-input-scale", type=float, default=10.0)
    parser.add_argument("--use-linear-frontend", action="store_true")
    parser.add_argument("--use-input-skip", action="store_true")
    parser.add_argument("--readout-channels", type=int, default=64)
    parser.add_argument(
        "--temporal-kernel",
        type=int,
        default=0,
        help="Conv1D readout kernel. 0 means auto from carrier period.",
    )
    parser.add_argument("--num-samples", type=int, default=400)
    parser.add_argument("--dt", type=float, default=0.01)
    parser.add_argument("--duration", type=float, default=1.0)
    parser.add_argument("--carrier-hz", type=float, default=8.0)
    parser.add_argument("--num-components", type=int, default=5)
    parser.add_argument("--msg-fmin", type=float, default=1.0)
    parser.add_argument("--msg-fmax", type=float, default=5.0)
    parser.add_argument(
        "--out-path",
        type=Path,
        default=Path("artifacts/plots/second_work_visual_comparison_fixed.png"),
    )
    parser.add_argument(
        "--metrics-path",
        type=Path,
        default=Path("artifacts/plots/second_work_visual_metrics_fixed.json"),
    )
    args = parser.parse_args()

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

    metrics, x_test, y_test, pred = train_one_run(
        x=x,
        y=y,
        dt=args.dt,
        carrier_hz=args.carrier_hz,
        seed=args.seed,
        epochs=args.epochs,
        batch_size=args.batch_size,
        test_ratio=args.test_ratio,
        learning_rate=args.learning_rate,
        units=args.units,
        hopf_input_scale=args.hopf_input_scale,
        use_linear_frontend=args.use_linear_frontend,
        readout_channels=args.readout_channels,
        temporal_kernel=args.temporal_kernel,
        use_input_skip=args.use_input_skip,
    )

    result = {
        "test_mse": metrics.test_mse,
        "val_mse": metrics.val_mse,
        "epochs": args.epochs,
        "seed": args.seed,
        "use_linear_frontend": bool(args.use_linear_frontend),
        "use_input_skip": bool(args.use_input_skip),
        "hopf_input_scale": args.hopf_input_scale,
        "learning_rate": args.learning_rate,
        "units": args.units,
        "readout_channels": args.readout_channels,
        "temporal_kernel": args.temporal_kernel,
        "num_samples": args.num_samples,
        "dt": args.dt,
        "duration": args.duration,
    }

    plot_report(
        out_path=args.out_path,
        t=t,
        x_test=x_test,
        y_test=y_test,
        pred=pred,
        metrics=result,
    )

    args.metrics_path.parent.mkdir(parents=True, exist_ok=True)
    args.metrics_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    print(f"Saved figure: {args.out_path}")
    print(f"Saved metrics: {args.metrics_path}")


if __name__ == "__main__":
    main()
