"""Control run for the Table 4 paper-style DONN sentiment architecture."""

from __future__ import annotations

import argparse
import json
import sys
import textwrap
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.sentiment import decode_review, get_imdb_decoder, train_paper_exact_run


ARTICLE_ARCH = (
    "Embedding(100) -> Hopf(100) -> ReLU(100) -> Hopf(100) -> ReLU(100) "
    "-> tanh(20) -> output(2)"
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


def plot_prob_panel(ax: plt.Axes, title: str, logits: np.ndarray, true_label: int) -> None:
    probs = tf.nn.softmax(tf.convert_to_tensor(logits[None, :]), axis=1).numpy()[0]
    names = ["негатив", "позитив"]
    colors = ["tab:red" if i == 0 else "tab:green" for i in range(2)]
    bars = ax.bar(names, probs, color=colors, alpha=0.8)
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("Вероятность класса")
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.25)
    for i, bar in enumerate(bars):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.03,
            f"{probs[i]:.3f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    ax.text(
        0.03,
        0.92,
        f"ожидаемый класс: {names[true_label]}",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=10,
        bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.9},
    )


def plot_report(
    out_path: Path,
    x_test: np.ndarray,
    labels: np.ndarray,
    pred_logits: np.ndarray,
    metrics: dict[str, float | int | str | list[float]],
) -> None:
    decoder = get_imdb_decoder()
    neg_idx, pos_idx = select_negative_positive(labels)
    pred_labels = np.argmax(pred_logits, axis=1)

    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 1.0], hspace=0.28, wspace=0.22)

    ax_a0 = fig.add_subplot(gs[0, 0])
    ax_a1 = fig.add_subplot(gs[0, 1])
    ax_b0 = fig.add_subplot(gs[1, 0])
    ax_b1 = fig.add_subplot(gs[1, 1])

    plot_text_panel(
        ax_a0,
        "A1) Отрицательный отзыв",
        decode_review(x_test[neg_idx], decoder, max_words=36),
        predicted_label="негатив" if pred_labels[neg_idx] == 0 else "позитив",
        true_label="негатив",
    )
    plot_text_panel(
        ax_a1,
        "A2) Положительный отзыв",
        decode_review(x_test[pos_idx], decoder, max_words=36),
        predicted_label="негатив" if pred_labels[pos_idx] == 0 else "позитив",
        true_label="позитив",
    )
    plot_prob_panel(ax_b0, "B1) Вероятности классов | отрицательный пример", pred_logits[neg_idx], true_label=0)
    plot_prob_panel(ax_b1, "B2) Вероятности классов | положительный пример", pred_logits[pos_idx], true_label=1)

    fig.suptitle(
        "Задача 4: контрольный paper-style DONN | "
        f"test_acc={metrics['test_acc']:.4f}, val_acc={metrics['val_acc']:.4f}, "
        f"test_loss={metrics['test_loss']:.6f}, params={metrics['total_params']}, "
        f"paper_params={metrics['paper_reported_total_params']}",
        fontsize=13,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=170, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--vocab-size", type=int, default=35000)
    parser.add_argument("--max-len", type=int, default=500)
    parser.add_argument("--train-samples", type=int, default=25000)
    parser.add_argument("--test-samples", type=int, default=25000)
    parser.add_argument("--val-ratio", type=float, default=0.3)
    parser.add_argument("--embed-dim", type=int, default=100)
    parser.add_argument("--units", type=int, default=100)
    parser.add_argument("--proj-dim", type=int, default=20)
    parser.add_argument("--hopf-input-scale", type=float, default=0.2)
    parser.add_argument(
        "--out-path",
        type=Path,
        default=Path("artifacts/plots/fourth_work_paper_exact_summary.png"),
    )
    parser.add_argument(
        "--metrics-path",
        type=Path,
        default=Path("artifacts/plots/fourth_work_paper_exact_metrics.json"),
    )
    args = parser.parse_args()

    metrics, x_test, labels, pred_logits, history, param_info = train_paper_exact_run(
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
        "variant": "paper_exact_control",
        "assumption": "many_to_one_mse_on_one_hot_labels",
        "article_architecture": ARTICLE_ARCH,
        "paper_reported_acc": 0.852,
        "paper_reported_total_params": 26798,
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
        "history": history,
        **param_info,
    }
    result["param_gap_total_vs_paper"] = int(result["total_params"] - result["paper_reported_total_params"])
    result["param_gap_non_embedding_vs_paper"] = int(
        result["non_embedding_params"] - result["paper_reported_total_params"]
    )

    plot_report(
        out_path=args.out_path,
        x_test=x_test,
        labels=labels,
        pred_logits=pred_logits,
        metrics=result,
    )

    args.metrics_path.parent.mkdir(parents=True, exist_ok=True)
    args.metrics_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    print(f"Saved figure: {args.out_path}")
    print(f"Saved metrics: {args.metrics_path}")


if __name__ == "__main__":
    main()
