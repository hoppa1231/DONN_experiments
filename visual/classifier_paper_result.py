"""Paper-style control for Table 1 classification with ramp targets."""

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

from src.classifier import (
    DONNClassifierRamp,
    generate_classification_dataset,
    labels_from_y,
    ramp_classification_report,
    train_one_ramp_run,
)


def select_two_classes(y_cls: np.ndarray) -> tuple[int, int]:
    idx0 = np.where(y_cls == 0)[0]
    idx1 = np.where(y_cls == 1)[0]
    if len(idx0) == 0 or len(idx1) == 0:
        raise RuntimeError("Both classes must be present in test split.")
    return int(idx0[0]), int(idx1[0])


def get_oscillator_amplitudes(model: DONNClassifierRamp, x_batch: np.ndarray) -> np.ndarray:
    x_tf = tf.convert_to_tensor(x_batch, dtype=tf.float32)
    x_r = model.td_in_r(x_tf)
    x_i = tf.zeros_like(x_r)
    z_r, z_i = model.hopf(x_r, x_i)
    amp = tf.sqrt(tf.square(z_r) + tf.square(z_i))
    return amp.numpy()


def plot_report(
    out_path: Path,
    x_test: np.ndarray,
    y_test: np.ndarray,
    pred_seq: np.ndarray,
    amp: np.ndarray,
    hz: np.ndarray,
    i0: int,
    i1: int,
    metrics: dict[str, float | int | str | list[int] | bool],
) -> None:
    t = np.arange(x_test.shape[1], dtype=np.float32) * metrics["dt"]

    fig = plt.figure(figsize=(14, 10))
    gs = fig.add_gridspec(3, 2, height_ratios=[1.0, 1.1, 1.2], hspace=0.35, wspace=0.25)

    ax_a0 = fig.add_subplot(gs[0, 0])
    ax_a1 = fig.add_subplot(gs[0, 1])
    ax_b0 = fig.add_subplot(gs[1, 0])
    ax_b1 = fig.add_subplot(gs[1, 1])
    ax_c = fig.add_subplot(gs[2, :])

    ax_a0.plot(t, x_test[i0, :, 0], color="tab:blue", lw=1.2)
    ax_a0.set_title("A1) Входной сигнал (класс 0, низкий диапазон)")
    ax_a0.set_xlabel("Время, с")
    ax_a0.set_ylabel("Амплитуда")
    ax_a0.grid(alpha=0.25)

    ax_a1.plot(t, x_test[i1, :, 0], color="tab:orange", lw=1.2)
    ax_a1.set_title("A2) Входной сигнал (класс 1, высокий диапазон)")
    ax_a1.set_xlabel("Время, с")
    ax_a1.set_ylabel("Амплитуда")
    ax_a1.grid(alpha=0.25)

    ax_b0.plot(t, y_test[i0, :, 0], "k--", lw=1.2, label="цель ch0")
    ax_b0.plot(t, y_test[i0, :, 1], "gray", ls="--", lw=1.2, label="цель ch1")
    ax_b0.plot(t, pred_seq[i0, :, 0], color="tab:blue", lw=1.2, label="пред ch0")
    ax_b0.plot(t, pred_seq[i0, :, 1], color="tab:red", lw=1.2, label="пред ch1")
    ax_b0.set_title("B1) Ramp-цели и paper-style предсказание | класс 0")
    ax_b0.set_xlabel("Время, с")
    ax_b0.set_ylabel("Выход")
    ax_b0.grid(alpha=0.25)
    ax_b0.legend(loc="upper left", fontsize=8, ncol=2)

    ax_b1.plot(t, y_test[i1, :, 0], "k--", lw=1.2, label="цель ch0")
    ax_b1.plot(t, y_test[i1, :, 1], "gray", ls="--", lw=1.2, label="цель ch1")
    ax_b1.plot(t, pred_seq[i1, :, 0], color="tab:blue", lw=1.2, label="пред ch0")
    ax_b1.plot(t, pred_seq[i1, :, 1], color="tab:red", lw=1.2, label="пред ch1")
    ax_b1.set_title("B2) Ramp-цели и paper-style предсказание | класс 1")
    ax_b1.set_xlabel("Время, с")
    ax_b1.set_ylabel("Выход")
    ax_b1.grid(alpha=0.25)
    ax_b1.legend(loc="upper left", fontsize=8, ncol=2)

    amp_mean0 = amp[0].mean(axis=0)
    amp_mean1 = amp[1].mean(axis=0)
    ax_c.axvspan(0.1, 10.0, color="tab:blue", alpha=0.08, label="Низкий диапазон 0.1-10 Гц")
    ax_c.axvspan(10.0, 20.0, color="tab:orange", alpha=0.08, label="Высокий диапазон 10-20 Гц")
    ax_c.plot(hz, amp_mean0, marker="o", lw=1.4, color="tab:blue", label="Пример класса 0")
    ax_c.plot(hz, amp_mean1, marker="o", lw=1.4, color="tab:orange", label="Пример класса 1")
    ax_c.set_title("C) Профиль амплитуд скрытых осцилляторов Хопфа")
    ax_c.set_xlabel("Собственная частота осциллятора, Гц")
    ax_c.set_ylabel("Средняя амплитуда по времени")
    ax_c.grid(alpha=0.25)
    ax_c.legend(loc="upper right", fontsize=9)

    fig.suptitle(
        "Задача 1: paper-style контроль | "
        f"source={metrics['dataset_source']}, test_acc={metrics['test_acc']:.4f}, "
        f"val_acc={metrics['val_acc']:.4f}, test_loss={metrics['test_loss']:.6f}",
        fontsize=13,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=170, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset-source",
        choices=["article", "supplement-notebook", "saved-arrays"],
        default="article",
        help="Use the article text generator, the supplementary notebook generator, or the saved X/Y arrays.",
    )
    parser.add_argument("--x-path", type=Path, default=Path("artifacts/signal_generation/X.npy"))
    parser.add_argument("--y-path", type=Path, default=Path("artifacts/signal_generation/Y.npy"))
    parser.add_argument("--samples-per-class", type=int, default=500)
    parser.add_argument("--num-steps", type=int, default=1000)
    parser.add_argument("--dt", type=float, default=0.001)
    parser.add_argument("--num-components", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--test-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--units", type=int, default=20)
    parser.add_argument("--proj-dim", type=int, default=20)
    parser.add_argument("--hopf-input-scale", type=float, default=0.1)
    parser.add_argument("--min-omega-hz", type=float, default=0.1)
    parser.add_argument("--max-omega-hz", type=float, default=20.0)
    parser.add_argument(
        "--out-path",
        type=Path,
        default=Path("artifacts/plots/table1/first_work_paper_style_summary.png"),
    )
    parser.add_argument(
        "--metrics-path",
        type=Path,
        default=Path("artifacts/plots/table1/first_work_paper_style_metrics.json"),
    )
    args = parser.parse_args()

    if args.dataset_source == "saved-arrays":
        x = np.load(args.x_path).astype(np.float32)
        y = np.load(args.y_path).astype(np.float32)
    else:
        x, y = generate_classification_dataset(
            samples_per_class=args.samples_per_class,
            num_steps=args.num_steps,
            dt=args.dt,
            num_components=args.num_components,
            seed=args.seed,
            source=args.dataset_source,
        )

    metrics, x_test, y_test, pred_seq, model = train_one_ramp_run(
        x=x,
        y=y,
        seed=args.seed,
        epochs=args.epochs,
        batch_size=args.batch_size,
        test_ratio=args.test_ratio,
        learning_rate=args.learning_rate,
        units=args.units,
        proj_dim=args.proj_dim,
        hopf_input_scale=args.hopf_input_scale,
        min_omega_hz=args.min_omega_hz,
        max_omega_hz=args.max_omega_hz,
        dt=args.dt,
    )

    y_cls_test = labels_from_y(y_test)
    accuracy_report = ramp_classification_report(pred_seq, y_test)

    i0, i1 = select_two_classes(y_cls_test)
    x_pair = np.stack([x_test[i0], x_test[i1]], axis=0)
    amp_pair = get_oscillator_amplitudes(model, x_pair)
    hz = (model.hopf.omegas.numpy().squeeze() / (2.0 * np.pi)).astype(np.float32)

    result = {
        "variant": "paper_style_ramp_control",
        "dataset_source": args.dataset_source,
        "test_acc": metrics.test_acc,
        "val_acc": metrics.val_acc,
        "test_loss": metrics.test_loss,
        "accuracy_report": accuracy_report,
        "class_hist_true": accuracy_report["class_hist_true"],
        "class_hist_pred": accuracy_report["class_hist_mean"],
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "seed": args.seed,
        "learning_rate": args.learning_rate,
        "units": args.units,
        "proj_dim": args.proj_dim,
        "hopf_input_scale": args.hopf_input_scale,
        "min_omega_hz": args.min_omega_hz,
        "max_omega_hz": args.max_omega_hz,
        "samples_per_class": args.samples_per_class,
        "num_steps": args.num_steps,
        "dt": args.dt,
        "num_components": args.num_components,
        "assumptions": [
            "paper-style ramp targets with MSE objective",
            "Linear(20) is modeled as a real-valued TimeDistributed Dense before Hopf",
            "Hidden readout uses oscillator amplitudes before tanh(20) -> output(2)",
            "Oscillator frequencies are fixed, matching Table 1",
        ],
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
        metrics=result,
    )

    args.metrics_path.parent.mkdir(parents=True, exist_ok=True)
    args.metrics_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    print(f"Saved figure: {args.out_path}")
    print(f"Saved metrics: {args.metrics_path}")


if __name__ == "__main__":
    main()
