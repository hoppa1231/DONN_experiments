"""Visual report for Table 3: integration and differentiation."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.operators import TASKS, train_one_task


def _task_labels(task: str) -> tuple[str, str, str]:
    if task == "integration":
        return "A", "B", "tab:blue"
    return "C", "D", "tab:orange"


def plot_task_row(
    axes: list[plt.Axes],
    task: str,
    t,
    x_test,
    y_test,
    pred,
    baseline_pred,
    metrics: dict[str, float | str],
) -> None:
    input_prefix, result_prefix, target_color = _task_labels(task)
    pred_color = "tab:blue" if task == "differentiation" else "tab:orange"
    task_ru = "интегрирование" if task == "integration" else "дифференцирование"

    axes[0].plot(t, x_test[0, :, 0], color="tab:purple", lw=1.2)
    axes[0].set_title(f"{input_prefix}1) Вход {task_ru} I(t)")
    axes[0].set_xlabel("Время, с")
    axes[0].set_ylabel("Амплитуда")
    axes[0].grid(alpha=0.25)
    axes[0].text(
        0.02,
        0.98,
        (
            f"DONN test MSE={metrics['test_mse']:.4f}\n"
            f"val MSE={metrics['val_mse']:.4f}\n"
            f"базовый MSE={metrics['baseline_mse']:.6f}\n"
            f"corr={metrics['test_corr']:.4f}"
        ),
        transform=axes[0].transAxes,
        va="top",
        ha="left",
        fontsize=9,
        bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.85},
    )

    for sample_idx, ax in enumerate(axes[1:3]):
        ax.plot(t, y_test[sample_idx, :, 0], color=target_color, lw=1.3, label="цель O(t)")
        ax.plot(t, pred[sample_idx, :, 0], color=pred_color, lw=1.3, label="предсказание DONN")
        ax.plot(
            t,
            baseline_pred[sample_idx, :, 0],
            color="0.35",
            lw=1.1,
            ls="--",
            label="численный базовый метод",
        )
        ax.set_title(f"{result_prefix}{sample_idx + 1}) Результат {task_ru} | пример {sample_idx}")
        ax.set_xlabel("Время, с")
        ax.set_ylabel("Амплитуда")
        ax.grid(alpha=0.25)
        ax.legend(loc="upper right", fontsize=8)


def plot_report(
    out_path: Path,
    t,
    results: dict[str, dict[str, object]],
) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(17, 10), constrained_layout=True)

    for row_idx, task in enumerate(TASKS):
        task_result = results[task]
        plot_task_row(
            axes=list(axes[row_idx]),
            task=task,
            t=t,
            x_test=task_result["x_test"],
            y_test=task_result["y_test"],
            pred=task_result["pred"],
            baseline_pred=task_result["baseline_pred"],
            metrics=task_result["metrics"],
        )

    fig.suptitle(
        "Задача 3: визуальный отчёт (математические операторы) | "
        f"интегрирование test_mse={results['integration']['metrics']['test_mse']:.4f}, "
        f"дифференцирование test_mse={results['differentiation']['metrics']['test_mse']:.4f}",
        fontsize=13,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=170, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--test-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--units", type=int, default=20)
    parser.add_argument("--hopf-input-scale", type=float, default=5.0)
    parser.add_argument("--readout-channels", type=int, default=48)
    parser.add_argument("--temporal-kernel", type=int, default=33)
    parser.add_argument("--use-linear-frontend", action="store_true")
    parser.add_argument(
        "--no-input-skip",
        action="store_true",
        help="Disable raw-input skip features in the temporal readout.",
    )
    parser.add_argument("--num-samples", type=int, default=200)
    parser.add_argument("--dt", type=float, default=0.001)
    parser.add_argument("--duration", type=float, default=1.0)
    parser.add_argument("--num-components", type=int, default=5)
    parser.add_argument("--fmin-hz", type=float, default=1.0)
    parser.add_argument("--fmax-hz", type=float, default=5.0)
    parser.add_argument(
        "--out-path",
        type=Path,
        default=Path("artifacts/plots/third_work_visual_summary.png"),
    )
    parser.add_argument(
        "--metrics-path",
        type=Path,
        default=Path("artifacts/plots/third_work_visual_metrics.json"),
    )
    args = parser.parse_args()

    results: dict[str, dict[str, object]] = {}
    shared_t = None

    for task in TASKS:
        metrics, t, x_test, y_test, pred, baseline_pred = train_one_task(
            task=task,
            num_samples=args.num_samples,
            dt=args.dt,
            duration=args.duration,
            num_components=args.num_components,
            fmin_hz=args.fmin_hz,
            fmax_hz=args.fmax_hz,
            seed=args.seed,
            epochs=args.epochs,
            batch_size=args.batch_size,
            test_ratio=args.test_ratio,
            learning_rate=args.learning_rate,
            units=args.units,
            hopf_input_scale=args.hopf_input_scale,
            use_linear_frontend=args.use_linear_frontend,
            use_input_skip=not args.no_input_skip,
            readout_channels=args.readout_channels,
            temporal_kernel=args.temporal_kernel,
        )
        shared_t = t
        results[task] = {
            "metrics": {
                "task": metrics.task,
                "test_mse": metrics.test_mse,
                "val_mse": metrics.val_mse,
                "baseline_mse": metrics.baseline_mse,
                "test_corr": metrics.test_corr,
            },
            "x_test": x_test,
            "y_test": y_test,
            "pred": pred,
            "baseline_pred": baseline_pred,
        }

    plot_report(out_path=args.out_path, t=shared_t, results=results)

    payload = {
        task: results[task]["metrics"] for task in TASKS
    }
    payload["_meta"] = {
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "seed": args.seed,
        "learning_rate": args.learning_rate,
        "units": args.units,
        "hopf_input_scale": args.hopf_input_scale,
        "readout_channels": args.readout_channels,
        "temporal_kernel": args.temporal_kernel,
        "use_linear_frontend": bool(args.use_linear_frontend),
        "use_input_skip": not args.no_input_skip,
        "num_samples": args.num_samples,
        "dt": args.dt,
        "duration": args.duration,
        "num_components": args.num_components,
        "fmin_hz": args.fmin_hz,
        "fmax_hz": args.fmax_hz,
    }

    args.metrics_path.parent.mkdir(parents=True, exist_ok=True)
    args.metrics_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))
    print(f"Saved figure: {args.out_path}")
    print(f"Saved metrics: {args.metrics_path}")


if __name__ == "__main__":
    main()
