"""Visual report for Table 5: OCNN action-recognition availability/smoke control."""

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

from src.action_recognition import find_local_ucf_candidates, train_synthetic_smoke_run


ARTICLE_ARCH = (
    "2 x OCNN (3x3,40), flatten, output (2); initial frequency range [1-15 Hz], "
    "input type I(t), trainable oscillator frequencies"
)


def plot_report(out_path: Path, frames: np.ndarray, y_true: np.ndarray, y_pred: np.ndarray, metrics: dict[str, object]) -> None:
    sample_idx = 0
    true_curve = y_true[sample_idx]
    pred_curve = y_pred[sample_idx]
    frame_ids = np.linspace(0, frames.shape[1] - 1, num=min(4, frames.shape[1]), dtype=int)

    fig = plt.figure(figsize=(14, 7))
    gs = fig.add_gridspec(2, len(frame_ids), height_ratios=[1.0, 1.1], hspace=0.35, wspace=0.15)
    for col, frame_id in enumerate(frame_ids):
        ax = fig.add_subplot(gs[0, col])
        ax.imshow(np.clip(frames[sample_idx, frame_id], 0.0, 1.0))
        ax.set_title(f"frame {int(frame_id)}")
        ax.axis("off")

    ax_curve = fig.add_subplot(gs[1, :])
    t = np.arange(true_curve.shape[0])
    for cls in range(true_curve.shape[1]):
        ax_curve.plot(t, true_curve[:, cls], ls="--", lw=1.2, label=f"target {cls}")
        ax_curve.plot(t, pred_curve[:, cls], lw=1.2, label=f"pred {cls}")
    ax_curve.set_xlabel("frame")
    ax_curve.set_ylabel("ramp output")
    ax_curve.grid(alpha=0.25)
    ax_curve.legend(loc="upper left", ncol=2, fontsize=8)

    fig.suptitle(
        "Table 5 OCNN smoke control | "
        f"val_acc={float(metrics['val_acc']):.4f}, val_loss={float(metrics['val_loss']):.6f}",
        fontsize=13,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=170, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--external-root", type=Path, default=Path("/home/user/Projects/test-ai-capabilities/external"))
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-samples", type=int, default=48)
    parser.add_argument("--num-frames", type=int, default=12)
    parser.add_argument("--frame-size", type=int, default=24)
    parser.add_argument("--square-size", type=int, default=5)
    parser.add_argument("--num-classes", type=int, default=2)
    parser.add_argument("--filters", type=int, default=4)
    parser.add_argument("--val-ratio", type=float, default=0.25)
    parser.add_argument(
        "--out-path",
        type=Path,
        default=Path("artifacts/plots/table5/fifth_work_ocnn_smoke_summary.png"),
    )
    parser.add_argument(
        "--metrics-path",
        type=Path,
        default=Path("artifacts/plots/table5/fifth_work_ocnn_smoke_metrics.json"),
    )
    args = parser.parse_args()

    ucf_candidates = find_local_ucf_candidates(args.external_root)
    metrics, x_val, y_val, pred_val, history, total_params = train_synthetic_smoke_run(
        seed=args.seed,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        num_samples=args.num_samples,
        num_frames=args.num_frames,
        frame_size=args.frame_size,
        square_size=args.square_size,
        num_classes=args.num_classes,
        filters=args.filters,
        val_ratio=args.val_ratio,
    )

    result = {
        "variant": "table5_ocnn_synthetic_smoke",
        "is_ucf11_reproduction": False,
        "reason_not_ucf11": "No local UCF11/UCF50 video dataset was found in artifacts/ or the configured external root.",
        "local_ucf_candidates": ucf_candidates,
        "article_dataset": "UCF11 YouTube Action dataset; article appendix points to Kaggle UCF50 mirror",
        "article_train_val": {"train": 1290, "validation": 305},
        "article_num_frames": 50,
        "article_frame_size": [48, 48, 3],
        "article_architecture": ARTICLE_ARCH,
        "paper_reported_ocnn_val_acc": 0.9864,
        "paper_text_reports_figure4_acc": 0.9975,
        "paper_reported_val_mse": 0.0564,
        "smoke_dataset": {
            "num_samples": args.num_samples,
            "num_frames": args.num_frames,
            "frame_size": args.frame_size,
            "square_size": args.square_size,
            "num_classes": args.num_classes,
        },
        "filters": args.filters,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "seed": args.seed,
        "total_params": total_params,
        "history": history,
        "val_acc": metrics.val_acc,
        "val_loss": metrics.val_loss,
        "train_acc": metrics.train_acc,
        "train_loss": metrics.train_loss,
    }

    plot_report(args.out_path, x_val, y_val, pred_val, result)
    args.metrics_path.parent.mkdir(parents=True, exist_ok=True)
    args.metrics_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    print(f"Saved figure: {args.out_path}")
    print(f"Saved metrics: {args.metrics_path}")


if __name__ == "__main__":
    main()
