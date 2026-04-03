"""Visual report for Table 4: IMDB sentiment analysis."""

from __future__ import annotations

import argparse
import json
import sys
import textwrap
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.sentiment import (
    aggregate_sequence_logits,
    decode_review,
    get_imdb_decoder,
    train_bilstm_baseline,
    train_one_run,
)


def select_negative_positive(labels: np.ndarray) -> tuple[int, int]:
    neg_idx = np.where(labels == 0)[0]
    pos_idx = np.where(labels == 1)[0]
    if len(neg_idx) == 0 or len(pos_idx) == 0:
        raise RuntimeError("Both sentiment classes must be present in the test split.")
    return int(neg_idx[0]), int(pos_idx[0])


def _format_review(title: str, review: str) -> str:
    wrapped = textwrap.fill(review, width=58)
    return f"{title}\n\n{wrapped}"


def plot_text_panel(ax: plt.Axes, title: str, review: str, predicted_label: str, true_label: str) -> None:
    ax.axis("off")
    ax.text(
        0.02,
        0.98,
        _format_review(title, review),
        ha="left",
        va="top",
        fontsize=10,
        family="monospace",
    )
    ax.text(
        0.02,
        0.08,
        f"истина={true_label} | предсказание={predicted_label}",
        ha="left",
        va="bottom",
        fontsize=10,
        bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.9},
    )


def plot_trace_panel(
    ax: plt.Axes,
    title: str,
    y_true: np.ndarray,
    pred: np.ndarray,
    true_label: int,
) -> None:
    t = np.arange(y_true.shape[0], dtype=np.float32)
    target_curve = y_true[:, true_label]
    pred_true = pred[:, true_label]
    pred_other = pred[:, 1 - true_label]

    true_name_ru = "негатив" if true_label == 0 else "позитив"
    other_name_ru = "позитив" if true_label == 0 else "негатив"
    ax.plot(t, target_curve, color="tab:orange", lw=1.3, label=f"целевая рампа {true_name_ru}")
    ax.plot(t, pred_true, color="tab:blue", lw=1.3, label=f"предсказанная оценка {true_name_ru}")
    ax.plot(t, pred_other, color="0.35", lw=1.1, ls="--", label=f"предсказанная оценка {other_name_ru}")
    ax.set_title(title)
    ax.set_xlabel("Позиция слова")
    ax.set_ylabel("Выход последовательности")
    ax.grid(alpha=0.25)
    ax.legend(loc="upper left", fontsize=8)


def plot_report(
    out_path: Path,
    x_test: np.ndarray,
    y_test: np.ndarray,
    pred: np.ndarray,
    labels: np.ndarray,
    metrics: dict[str, float | int | bool],
) -> None:
    decoder = get_imdb_decoder()
    neg_idx, pos_idx = select_negative_positive(labels)

    agg = aggregate_sequence_logits(pred)
    pred_labels = np.argmax(agg, axis=1)

    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 1.2], hspace=0.28, wspace=0.22)

    ax_a0 = fig.add_subplot(gs[0, 0])
    ax_a1 = fig.add_subplot(gs[0, 1])
    ax_b0 = fig.add_subplot(gs[1, 0])
    ax_b1 = fig.add_subplot(gs[1, 1])

    plot_text_panel(
        ax_a0,
        "A1) Фрагмент отрицательного отзыва",
        decode_review(x_test[neg_idx], decoder, max_words=36),
        predicted_label="негатив" if pred_labels[neg_idx] == 0 else "позитив",
        true_label="негатив",
    )
    plot_text_panel(
        ax_a1,
        "A2) Фрагмент положительного отзыва",
        decode_review(x_test[pos_idx], decoder, max_words=36),
        predicted_label="негатив" if pred_labels[pos_idx] == 0 else "позитив",
        true_label="позитив",
    )
    plot_trace_panel(ax_b0, "B1) Выходы последовательности | отрицательный пример", y_test[neg_idx], pred[neg_idx], true_label=0)
    plot_trace_panel(ax_b1, "B2) Выходы последовательности | положительный пример", y_test[pos_idx], pred[pos_idx], true_label=1)

    fig.suptitle(
        "Задача 4: визуальный отчёт (анализ тональности IMDB) | "
        f"DONN test_acc={metrics['test_acc']:.4f}, val_acc={metrics['val_acc']:.4f}, "
        f"test_loss={metrics['test_loss']:.6f}"
        + (
            f", baseline_acc={metrics['bilstm_test_acc']:.4f}"
            if "bilstm_test_acc" in metrics
            else ""
        ),
        fontsize=13,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=170, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--vocab-size", type=int, default=35000)
    parser.add_argument("--max-len", type=int, default=500)
    parser.add_argument("--train-samples", type=int, default=600)
    parser.add_argument("--test-samples", type=int, default=300)
    parser.add_argument("--val-ratio", type=float, default=0.3)
    parser.add_argument("--embed-dim", type=int, default=100)
    parser.add_argument("--units", type=int, default=32)
    parser.add_argument("--proj-dim", type=int, default=20)
    parser.add_argument("--hopf-input-scale", type=float, default=0.2)
    parser.add_argument(
        "--skip-bilstm-baseline",
        action="store_true",
        help="Skip the Bidirectional LSTM reference run.",
    )
    parser.add_argument("--bilstm-units", type=int, default=32)
    parser.add_argument(
        "--out-path",
        type=Path,
        default=Path("artifacts/plots/fourth_work_visual_summary.png"),
    )
    parser.add_argument(
        "--metrics-path",
        type=Path,
        default=Path("artifacts/plots/fourth_work_visual_metrics.json"),
    )
    args = parser.parse_args()

    metrics, x_test, y_test, pred, labels = train_one_run(
        vocab_size=args.vocab_size,
        max_len=args.max_len,
        train_samples=args.train_samples,
        test_samples=args.test_samples,
        seed=args.seed,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        embed_dim=args.embed_dim,
        units=args.units,
        proj_dim=args.proj_dim,
        hopf_input_scale=args.hopf_input_scale,
        val_ratio=args.val_ratio,
    )

    result = {
        "test_acc": metrics.test_acc,
        "val_acc": metrics.val_acc,
        "test_loss": metrics.test_loss,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "seed": args.seed,
        "learning_rate": args.learning_rate,
        "vocab_size": args.vocab_size,
        "max_len": args.max_len,
        "train_samples": args.train_samples,
        "test_samples": args.test_samples,
        "val_ratio": args.val_ratio,
        "embed_dim": args.embed_dim,
        "units": args.units,
        "proj_dim": args.proj_dim,
        "hopf_input_scale": args.hopf_input_scale,
        "paper_table_acc": 0.852,
        "paper_bilstm_acc": 0.875,
    }

    if not args.skip_bilstm_baseline:
        bilstm = train_bilstm_baseline(
            vocab_size=args.vocab_size,
            max_len=args.max_len,
            train_samples=args.train_samples,
            test_samples=args.test_samples,
            seed=args.seed,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            embed_dim=args.embed_dim,
            lstm_units=args.bilstm_units,
            val_ratio=args.val_ratio,
        )
        result["bilstm_test_acc"] = bilstm.test_acc
        result["bilstm_val_acc"] = bilstm.val_acc
        result["bilstm_test_loss"] = bilstm.test_loss
        result["bilstm_units"] = args.bilstm_units

    plot_report(
        out_path=args.out_path,
        x_test=x_test,
        y_test=y_test,
        pred=pred,
        labels=labels,
        metrics=result,
    )

    args.metrics_path.parent.mkdir(parents=True, exist_ok=True)
    args.metrics_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    print(f"Saved figure: {args.out_path}")
    print(f"Saved metrics: {args.metrics_path}")


if __name__ == "__main__":
    main()
