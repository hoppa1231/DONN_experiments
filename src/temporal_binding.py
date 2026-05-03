"""Case study 1 helpers: temporal binding analysis for moving-bar videos.

This module intentionally keeps the first pass lightweight.  It audits the
published/supplied moving-bar dataset and runs the set-selection/synchrony
calculation described in the paper on a deterministic oscillatory probe.
It is not a full trained ConvOsc reproduction.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


CLASS_NAMES = [
    "channel0_vertical",
    "channel1_vertical",
    "channel0_horizontal",
    "channel1_horizontal",
]


@dataclass
class DatasetAudit:
    x_shape: tuple[int, ...]
    y_shape: tuple[int, ...]
    class_counts: list[int]
    blank_frame_fraction: float
    samples_with_blank_frames: int
    max_blank_frames_in_sample: int
    blank_frame_fraction_by_class: list[float]


@dataclass
class BindingResult:
    class_name: str
    group_size: int
    residuary_size: int
    group_synchrony_mean: float
    group_synchrony_std: float
    residuary_synchrony_mean: float
    residuary_synchrony_std: float


@dataclass
class ClassificationAudit:
    accuracy: float
    confusion_matrix: list[list[int]]
    predicted_class_counts: list[int]


def moving_bar_sample(
    rng: np.random.Generator,
    seq_len: int = 500,
    canvas_dim: int = 32,
    fix_wrap: bool = True,
) -> tuple[np.ndarray, int]:
    """Generate one moving-bar video following the article notebook setup.

    When ``fix_wrap`` is false, this intentionally reproduces the wraparound bug
    in the supplied notebook.  When true, bars crossing the image boundary are
    drawn in two slices instead of disappearing for that frame.
    """
    bar_width_max = 10
    bar_width_min = 2
    canvas = np.zeros((seq_len, 3, canvas_dim, canvas_dim), dtype=np.float32)
    color_pick = int(rng.integers(0, 2))
    bar_x_pos = int(rng.integers(0, canvas_dim))
    bar_width = int(rng.integers(bar_width_min, bar_width_max))
    disp = int(rng.integers(1, 8))
    orientation = int(rng.integers(0, 2))

    def draw_slice(t: int, start: int, stop: int) -> None:
        if start == stop:
            return
        if orientation:
            canvas[t, color_pick, start:stop, :] = 1.0
        else:
            canvas[t, color_pick, :, start:stop] = 1.0

    for t in range(seq_len):
        bar_on = bar_x_pos % canvas_dim
        bar_off = (bar_x_pos + bar_width) % canvas_dim
        if bar_on > bar_off:
            if fix_wrap:
                draw_slice(t, bar_on, canvas_dim)
                draw_slice(t, 0, bar_off)
            else:
                # This mirrors the notebook bug: both indices become equal,
                # leaving the slice empty and the frame blank.
                swap = bar_on
                bar_off = bar_on
                bar_on = swap
                draw_slice(t, bar_on, bar_off)
        else:
            draw_slice(t, bar_on, bar_off)
        bar_x_pos += disp

    return canvas, 2 * orientation + color_pick


def generate_moving_bar_dataset(
    num_samples: int,
    seq_len: int = 500,
    canvas_dim: int = 32,
    fix_wrap: bool = True,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate a moving-bar dataset in the same layout as the saved arrays."""
    rng = np.random.default_rng(seed)
    x_samples = []
    y = np.zeros((num_samples, seq_len, 4), dtype=np.float32)
    ramp = np.linspace(0.0, 2.0, seq_len, dtype=np.float32)
    labels = np.zeros(num_samples, dtype=np.int64)
    for sample_idx in range(num_samples):
        video, class_id = moving_bar_sample(
            rng=rng,
            seq_len=seq_len,
            canvas_dim=canvas_dim,
            fix_wrap=fix_wrap,
        )
        x_samples.append(video)
        y[sample_idx, :, class_id] = ramp
        labels[sample_idx] = class_id

    x = np.transpose(np.asarray(x_samples, dtype=np.float32), (1, 0, 2, 3, 4))
    y = np.transpose(y, (1, 0, 2))
    return x, y, labels


def load_case_study_arrays(x_path: Path, y_path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load the case-study arrays and derive one label per sample.

    The notebook stores arrays as X=[T, N, C, H, W] and Y=[T, N, 4].
    """
    x = np.load(x_path).astype(np.float32)
    y = np.load(y_path).astype(np.float32)
    if x.ndim != 5:
        raise ValueError(f"Expected X with 5 dimensions [T, N, C, H, W], got {x.shape}")
    if y.ndim != 3:
        raise ValueError(f"Expected Y with 3 dimensions [T, N, classes], got {y.shape}")
    if x.shape[:2] != y.shape[:2]:
        raise ValueError(f"X/Y time and sample dimensions do not match: {x.shape} vs {y.shape}")
    labels = np.argmax(y.sum(axis=0), axis=1).astype(np.int64)
    return x, y, labels


def audit_dataset(x: np.ndarray, y: np.ndarray, labels: np.ndarray) -> DatasetAudit:
    active_pixels = x.sum(axis=(2, 3, 4))
    blank = active_pixels == 0
    by_class: list[float] = []
    for class_id in range(y.shape[2]):
        mask = labels == class_id
        by_class.append(float(blank[:, mask].mean()) if np.any(mask) else float("nan"))

    return DatasetAudit(
        x_shape=tuple(int(v) for v in x.shape),
        y_shape=tuple(int(v) for v in y.shape),
        class_counts=np.bincount(labels, minlength=y.shape[2]).astype(int).tolist(),
        blank_frame_fraction=float(blank.mean()),
        samples_with_blank_frames=int(np.sum(blank.any(axis=0))),
        max_blank_frames_in_sample=int(blank.sum(axis=0).max()),
        blank_frame_fraction_by_class=by_class,
    )


def infer_moving_bar_labels(x: np.ndarray) -> np.ndarray:
    """Infer class labels directly from color channel and bar orientation.

    This is a deterministic sanity-check classifier for the synthetic moving-bar
    task.  It reads the same two factors used by the data generator:
    class = 2 * orientation + color, with orientation 0=vertical and
    orientation 1=horizontal.
    """
    color_scores = x[:, :, :2].sum(axis=(0, 3, 4))
    pred_color = np.argmax(color_scores, axis=1).astype(np.int64)

    intensity = x[:, :, :2].sum(axis=2)
    col_profile = intensity.sum(axis=2)
    row_profile = intensity.sum(axis=3)
    col_var = np.var(col_profile, axis=2).mean(axis=0)
    row_var = np.var(row_profile, axis=2).mean(axis=0)
    pred_orientation = (row_var > col_var).astype(np.int64)
    return 2 * pred_orientation + pred_color


def audit_deterministic_classifier(x: np.ndarray, labels: np.ndarray, num_classes: int = 4) -> ClassificationAudit:
    pred = infer_moving_bar_labels(x)
    confusion = np.zeros((num_classes, num_classes), dtype=np.int64)
    for true_label, pred_label in zip(labels, pred, strict=True):
        confusion[int(true_label), int(pred_label)] += 1
    return ClassificationAudit(
        accuracy=float(np.mean(pred == labels)),
        confusion_matrix=confusion.astype(int).tolist(),
        predicted_class_counts=np.bincount(pred, minlength=num_classes).astype(int).tolist(),
    )


def _block_reduce_mean(video: np.ndarray, grid: int) -> np.ndarray:
    """Average frames into a grid while preserving time and channel dimensions."""
    time_steps, channels, height, width = video.shape
    if height % grid != 0 or width % grid != 0:
        raise ValueError(f"Grid {grid} must evenly divide video size {(height, width)}")
    cell_h = height // grid
    cell_w = width // grid
    blocks = video.reshape(time_steps, channels, grid, cell_h, grid, cell_w)
    return blocks.mean(axis=(3, 5))


def make_feature_amplitudes(video: np.ndarray, grid: int = 8) -> np.ndarray:
    """Create color/orientation-selective probe amplitudes for one video.

    The output is [T, 4 * grid * grid].  Feature groups are:
    channel0, channel1, vertical-bar evidence, horizontal-bar evidence.
    """
    pooled = _block_reduce_mean(video, grid=grid)
    color0 = pooled[:, 0]
    color1 = pooled[:, 1]
    intensity = pooled[:, :2].sum(axis=1)

    col_profile = video[:, :2].sum(axis=(1, 2))
    row_profile = video[:, :2].sum(axis=(1, 3))
    vertical_strength = np.var(col_profile, axis=1)
    horizontal_strength = np.var(row_profile, axis=1)
    denom = vertical_strength + horizontal_strength + 1e-6
    vertical_gate = (vertical_strength / denom)[:, None, None]
    horizontal_gate = (horizontal_strength / denom)[:, None, None]

    vertical = intensity * vertical_gate
    horizontal = intensity * horizontal_gate
    features = np.stack([color0, color1, vertical, horizontal], axis=1)
    return features.reshape(video.shape[0], -1).astype(np.float32)


def make_oscillatory_probe(features: np.ndarray, seed: int = 42) -> np.ndarray:
    """Turn feature amplitudes into complex traces for synchrony checks.

    Active features are phase-entrained by the same input-derived carrier, while
    weak features retain more independent phase drift.  This is a deterministic
    probe for the binding-analysis algorithm, not a trained ConvOsc hidden layer.
    """
    rng = np.random.default_rng(seed)
    time_steps, units = features.shape
    t = np.linspace(0.0, 1.0, time_steps, dtype=np.float32)
    freqs = rng.uniform(1.0, 10.0, size=units).astype(np.float32)
    base_phase = rng.uniform(-np.pi, np.pi, size=units).astype(np.float32)

    norm = features / (np.max(features, axis=0, keepdims=True) + 1e-6)
    carrier = 2.0 * np.pi * 4.0 * t[:, None]
    free_phase = carrier * freqs[None, :] + base_phase[None, :]
    entrained_phase = carrier + 0.15 * base_phase[None, :]
    phase = norm * entrained_phase + (1.0 - norm) * free_phase
    amplitude = 0.05 + features
    return amplitude * np.exp(1j * phase)


def synchrony(z: np.ndarray, indices: np.ndarray) -> float:
    """Compute the paper's order-parameter synchrony for a set of oscillators."""
    if len(indices) == 0:
        return float("nan")
    selected = z[:, indices]
    normalized = selected / (np.abs(selected) + 1e-8)
    return float(np.mean(np.abs(np.mean(normalized, axis=1))))


def _active_indices(z: np.ndarray, threshold_quantile: float) -> set[int]:
    amplitudes = np.mean(np.abs(z), axis=0)
    threshold = np.quantile(amplitudes, threshold_quantile)
    return set(np.flatnonzero(amplitudes >= threshold).astype(int).tolist())


def _class_sets(
    x: np.ndarray,
    labels: np.ndarray,
    threshold_quantile: float,
    grid: int,
    seed: int,
) -> list[set[int]]:
    sets: list[set[int]] = []
    # Use the first sample of each class, matching the paper's set construction
    # over AX, AY, BX, BY before repeating the synchrony measurement.
    for class_id in range(4):
        sample_idx = int(np.flatnonzero(labels == class_id)[0])
        features = make_feature_amplitudes(x[:, sample_idx], grid=grid)
        z = make_oscillatory_probe(features, seed=seed + class_id)
        sets.append(_active_indices(z, threshold_quantile=threshold_quantile))
    return sets


def run_binding_control(
    x: np.ndarray,
    labels: np.ndarray,
    threshold_quantile: float = 0.8,
    grid: int = 8,
    trials: int = 20,
    seed: int = 42,
) -> list[BindingResult]:
    """Run a local temporal-binding control using the paper's set algebra."""
    # Notebook labels are class = 2 * orientation + color, so the saved order is
    # AX, BX, AY, BY for color A/B and orientation X/Y.
    ax, bx, ay, by = _class_sets(
        x=x,
        labels=labels,
        threshold_quantile=threshold_quantile,
        grid=grid,
        seed=seed,
    )

    a_hat = (ax | ay) - (bx | by)
    b_hat = (bx | by) - (ax | ay)
    x_hat = (ax | bx) - (ay | by)
    y_hat = (ay | by) - (ax | bx)

    groups = [a_hat | x_hat, b_hat | x_hat, a_hat | y_hat, b_hat | y_hat]
    universe = set(range(4 * grid * grid))

    results: list[BindingResult] = []
    for class_id, group in enumerate(groups):
        group_idx = np.array(sorted(group), dtype=np.int64)
        residuary_idx = np.array(sorted(universe - group), dtype=np.int64)
        sample_pool = np.flatnonzero(labels == class_id)
        group_scores: list[float] = []
        residuary_scores: list[float] = []
        for trial in range(trials):
            sample_idx = int(sample_pool[trial % len(sample_pool)])
            features = make_feature_amplitudes(x[:, sample_idx], grid=grid)
            z = make_oscillatory_probe(features, seed=seed + 1000 * class_id + trial)
            group_scores.append(synchrony(z, group_idx))
            residuary_scores.append(synchrony(z, residuary_idx))

        results.append(
            BindingResult(
                class_name=CLASS_NAMES[class_id],
                group_size=int(len(group_idx)),
                residuary_size=int(len(residuary_idx)),
                group_synchrony_mean=float(np.nanmean(group_scores)),
                group_synchrony_std=float(np.nanstd(group_scores)),
                residuary_synchrony_mean=float(np.nanmean(residuary_scores)),
                residuary_synchrony_std=float(np.nanstd(residuary_scores)),
            )
        )

    return results
