"""Visual report for Case study 1: temporal binding in moving-bar videos."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.temporal_binding import (
    CLASS_NAMES,
    audit_deterministic_classifier,
    audit_dataset,
    generate_moving_bar_dataset,
    load_case_study_arrays,
    run_binding_control,
)


def plot_report(
    out_path: Path,
    x: np.ndarray,
    labels: np.ndarray,
    audit: dict[str, object],
    classifier_audit: dict[str, object],
    results: list[dict[str, object]],
) -> None:
    fig = plt.figure(figsize=(15, 9))
    gs = fig.add_gridspec(2, 3, height_ratios=[1.0, 1.15], hspace=0.32, wspace=0.28)

    example_axes = [fig.add_subplot(gs[0, i]) for i in range(3)]
    bar_ax = fig.add_subplot(gs[1, :2])
    audit_ax = fig.add_subplot(gs[1, 2])

    sample_idx = 0
    frames = [0, x.shape[0] // 2, x.shape[0] - 1]
    for ax, frame in zip(example_axes, frames, strict=True):
        img = np.transpose(x[frame, sample_idx], (1, 2, 0))
        ax.imshow(np.clip(img, 0.0, 1.0))
        ax.set_title(f"sample {sample_idx}, t={frame}, {CLASS_NAMES[int(labels[sample_idx])]}")
        ax.set_xticks([])
        ax.set_yticks([])

    names = [str(r["class_name"]) for r in results]
    group = np.array([float(r["group_synchrony_mean"]) for r in results])
    residuary = np.array([float(r["residuary_synchrony_mean"]) for r in results])
    group_err = np.array([float(r["group_synchrony_std"]) for r in results])
    residuary_err = np.array([float(r["residuary_synchrony_std"]) for r in results])

    pos = np.arange(len(names))
    width = 0.36
    bar_ax.bar(pos - width / 2, group, width, yerr=group_err, label="group", color="tab:blue", alpha=0.85)
    bar_ax.bar(
        pos + width / 2,
        residuary,
        width,
        yerr=residuary_err,
        label="residuary",
        color="tab:orange",
        alpha=0.85,
    )
    bar_ax.set_ylim(0.0, 1.05)
    bar_ax.set_xticks(pos)
    bar_ax.set_xticklabels(names, rotation=20, ha="right")
    bar_ax.set_ylabel("Synchrony S")
    bar_ax.set_title("Temporal-binding control: group vs residuary synchrony")
    bar_ax.grid(axis="y", alpha=0.25)
    bar_ax.legend()

    audit_ax.axis("off")
    audit_text = "\n".join(
        [
            "Dataset audit",
            f"X: {audit['x_shape']}",
            f"Y: {audit['y_shape']}",
            f"class counts: {audit['class_counts']}",
            f"blank frames: {float(audit['blank_frame_fraction']):.3f}",
            f"samples with blanks: {audit['samples_with_blank_frames']}",
            f"max blank/sample: {audit['max_blank_frames_in_sample']}",
            f"direct classifier acc: {float(classifier_audit['accuracy']):.3f}",
            "",
            "Note: control uses a deterministic",
            "oscillatory probe, not a trained OCNN.",
        ]
    )
    audit_ax.text(0.0, 1.0, audit_text, ha="left", va="top", family="monospace", fontsize=10)

    fig.suptitle("Case study 1: temporal binding analysis check", fontsize=14)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=170, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source",
        choices=["saved", "generated-fixed", "generated-buggy"],
        default="saved",
        help="Use saved arrays, a corrected generator, or the original buggy generator.",
    )
    parser.add_argument("--x-path", type=Path, default=Path("artifacts/case_study/X.npy"))
    parser.add_argument("--y-path", type=Path, default=Path("artifacts/case_study/Y.npy"))
    parser.add_argument("--generated-samples", type=int, default=100)
    parser.add_argument("--seq-len", type=int, default=500)
    parser.add_argument("--canvas-dim", type=int, default=32)
    parser.add_argument("--threshold-quantile", type=float, default=0.8)
    parser.add_argument("--grid", type=int, default=8)
    parser.add_argument("--trials", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--out-path",
        type=Path,
        default=Path("artifacts/plots/case_study/case_study_temporal_binding_summary.png"),
    )
    parser.add_argument(
        "--metrics-path",
        type=Path,
        default=Path("artifacts/plots/case_study/case_study_temporal_binding_metrics.json"),
    )
    args = parser.parse_args()

    if args.source == "saved":
        x, y, labels = load_case_study_arrays(args.x_path, args.y_path)
    else:
        x, y, labels = generate_moving_bar_dataset(
            num_samples=args.generated_samples,
            seq_len=args.seq_len,
            canvas_dim=args.canvas_dim,
            fix_wrap=args.source == "generated-fixed",
            seed=args.seed,
        )
    audit = audit_dataset(x=x, y=y, labels=labels)
    classifier_audit = audit_deterministic_classifier(x=x, labels=labels)
    results = run_binding_control(
        x=x,
        labels=labels,
        threshold_quantile=args.threshold_quantile,
        grid=args.grid,
        trials=args.trials,
        seed=args.seed,
    )

    result = {
        "variant": "temporal_binding_dataset_and_algorithm_control",
        "is_full_trained_ocnn_reproduction": False,
        "source": args.source,
        "paper_claim": "group oscillators have higher synchrony than residuary oscillators",
        "article_table6_counts": {
            "A_hat": 545,
            "X_hat": 435,
            "B_hat": 38,
            "Y_hat": 48,
        },
        "threshold_quantile": args.threshold_quantile,
        "grid": args.grid,
        "trials": args.trials,
        "seed": args.seed,
        "generated_samples": args.generated_samples if args.source != "saved" else None,
        "seq_len": args.seq_len if args.source != "saved" else None,
        "canvas_dim": args.canvas_dim if args.source != "saved" else None,
        "dataset_audit": asdict(audit),
        "deterministic_classifier_audit": asdict(classifier_audit),
        "binding_results": [asdict(r) for r in results],
    }
    result["all_group_means_above_residuary"] = all(
        r.group_synchrony_mean > r.residuary_synchrony_mean for r in results
    )
    result["mean_group_minus_residuary"] = float(
        np.mean([r.group_synchrony_mean - r.residuary_synchrony_mean for r in results])
    )

    plot_report(
        out_path=args.out_path,
        x=x,
        labels=labels,
        audit=result["dataset_audit"],
        classifier_audit=result["deterministic_classifier_audit"],
        results=result["binding_results"],
    )

    args.metrics_path.parent.mkdir(parents=True, exist_ok=True)
    args.metrics_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    print(f"Saved figure: {args.out_path}")
    print(f"Saved metrics: {args.metrics_path}")


if __name__ == "__main__":
    main()
