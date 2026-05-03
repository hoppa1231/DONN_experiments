"""Compare local oscillator runs against simple non-oscillator baselines."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.HopfLayer import set_seed
from src.classifier import generate_classification_dataset, labels_from_y
from src.demodulation import generate_demod_dataset, split_train_test
from src.operators import generate_operator_dataset, split_train_test as split_operator_train_test


def _load_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _class_split(
    x: np.ndarray,
    y: np.ndarray,
    test_ratio: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    idx = np.arange(x.shape[0])
    rng = np.random.default_rng(seed)
    rng.shuffle(idx)
    n_test = int(round(x.shape[0] * test_ratio))
    test_idx = idx[:n_test]
    return x[test_idx], y[test_idx]


def spectral_table1_baseline(
    source: str,
    seed: int,
    samples_per_class: int,
    num_steps: int,
    dt: float,
    num_components: int,
    test_ratio: float,
) -> dict[str, Any]:
    if source == "saved-arrays":
        x = np.load("artifacts/signal_generation/X.npy").astype(np.float32)
        y = np.load("artifacts/signal_generation/Y.npy").astype(np.float32)
    else:
        x, y = generate_classification_dataset(
            samples_per_class=samples_per_class,
            num_steps=num_steps,
            dt=dt,
            num_components=num_components,
            seed=seed,
            source=source,
        )

    x_test, y_test = _class_split(x=x, y=y, test_ratio=test_ratio, seed=seed)
    labels = labels_from_y(y_test)
    freq = np.fft.rfftfreq(x_test.shape[1], d=dt)
    spectrum = np.abs(np.fft.rfft(x_test[:, :, 0], axis=1)) ** 2
    low = spectrum[:, (freq >= 0.1) & (freq < 10.0)].sum(axis=1)
    high = spectrum[:, (freq >= 10.0) & (freq <= 20.0)].sum(axis=1)
    pred = (high > low).astype(np.int64)
    return {
        "baseline": "fft_band_energy",
        "dataset_source": source,
        "test_acc": float(np.mean(pred == labels)),
        "class_hist_true": np.bincount(labels, minlength=2).tolist(),
        "class_hist_pred": np.bincount(pred, minlength=2).tolist(),
    }


class TemporalConvDemodBaseline(tf.keras.Model):
    """Temporal Conv1D demodulator without Hopf dynamics."""

    def __init__(self, carrier_hz: float, dt: float, channels: int, temporal_kernel: int) -> None:
        super().__init__()
        if temporal_kernel <= 0:
            temporal_kernel = max(5, int(round(2.0 / (carrier_hz * dt))))
        if temporal_kernel % 2 == 0:
            temporal_kernel += 1
        self.norm = tf.keras.layers.LayerNormalization(axis=-1)
        self.conv1 = tf.keras.layers.Conv1D(channels, temporal_kernel, padding="same", activation="tanh")
        self.conv2 = tf.keras.layers.Conv1D(channels, temporal_kernel, padding="same", activation="tanh")
        self.out = tf.keras.layers.Conv1D(1, 1, padding="same", activation="linear")

    def call(self, x: tf.Tensor) -> tf.Tensor:
        h = self.norm(tf.concat([x, tf.abs(x)], axis=2))
        h = self.conv1(h)
        h = self.conv2(h)
        return self.out(h)


class MatchedNoHopfTemporalRegressor(tf.keras.Model):
    """Dense frontend plus temporal readout, matching the DONN readout without Hopf."""

    def __init__(
        self,
        units: int,
        channels: int,
        temporal_kernel: int,
        use_input_skip: bool,
        include_abs_skip: bool,
    ) -> None:
        super().__init__()
        if temporal_kernel < 3:
            temporal_kernel = 3
        if temporal_kernel % 2 == 0:
            temporal_kernel += 1
        self.use_input_skip = use_input_skip
        self.include_abs_skip = include_abs_skip
        self.in_r = tf.keras.layers.Dense(units, activation="relu")
        self.in_i = tf.keras.layers.Dense(units, activation="relu")
        self.td_in_r = tf.keras.layers.TimeDistributed(self.in_r)
        self.td_in_i = tf.keras.layers.TimeDistributed(self.in_i)
        self.norm = tf.keras.layers.LayerNormalization(axis=-1)
        self.conv1 = tf.keras.layers.Conv1D(channels, temporal_kernel, padding="same", activation="tanh")
        self.conv2 = tf.keras.layers.Conv1D(channels, temporal_kernel, padding="same", activation="tanh")
        self.out = tf.keras.layers.Conv1D(1, 1, padding="same", activation="linear")

    def call(self, x: tf.Tensor) -> tf.Tensor:
        features = [self.td_in_r(x), self.td_in_i(x)]
        if self.use_input_skip:
            features.append(x)
        if self.include_abs_skip:
            features.append(tf.abs(x))
        h = self.norm(tf.concat(features, axis=2))
        h = self.conv1(h)
        h = self.conv2(h)
        return self.out(h)


def table2_temporal_conv_baseline(
    seed: int,
    num_samples: int,
    dt: float,
    duration: float,
    carrier_hz: float,
    num_components: int,
    msg_fmin: float,
    msg_fmax: float,
    test_ratio: float,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    channels: int,
    temporal_kernel: int,
) -> dict[str, Any]:
    set_seed(seed)
    x, y, _ = generate_demod_dataset(
        num_samples=num_samples,
        dt=dt,
        duration=duration,
        carrier_hz=carrier_hz,
        num_components=num_components,
        msg_fmin=msg_fmin,
        msg_fmax=msg_fmax,
        seed=seed,
    )
    x_train, y_train, x_test, y_test = split_train_test(x=x, y=y, test_ratio=test_ratio, seed=seed)
    model = TemporalConvDemodBaseline(
        carrier_hz=carrier_hz,
        dt=dt,
        channels=channels,
        temporal_kernel=temporal_kernel,
    )
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate), loss="mse")
    history = model.fit(
        x_train,
        y_train,
        validation_split=0.2,
        epochs=epochs,
        batch_size=batch_size,
        verbose=0,
    )
    pred = model.predict(x_test, batch_size=batch_size, verbose=0)
    return {
        "baseline": "temporal_conv_no_hopf",
        "test_mse": float(np.mean((pred - y_test) ** 2)),
        "val_mse": float(history.history["val_loss"][-1]),
        "epochs": epochs,
        "num_samples": num_samples,
        "channels": channels,
    }


def table2_matched_no_hopf_baseline(
    seed: int,
    num_samples: int,
    dt: float,
    duration: float,
    carrier_hz: float,
    num_components: int,
    msg_fmin: float,
    msg_fmax: float,
    test_ratio: float,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    units: int,
    channels: int,
    temporal_kernel: int,
) -> dict[str, Any]:
    set_seed(seed)
    x, y, _ = generate_demod_dataset(
        num_samples=num_samples,
        dt=dt,
        duration=duration,
        carrier_hz=carrier_hz,
        num_components=num_components,
        msg_fmin=msg_fmin,
        msg_fmax=msg_fmax,
        seed=seed,
    )
    x_train, y_train, x_test, y_test = split_train_test(x=x, y=y, test_ratio=test_ratio, seed=seed)
    if temporal_kernel <= 0:
        temporal_kernel = max(5, int(round(2.0 / (carrier_hz * dt))))
    model = MatchedNoHopfTemporalRegressor(
        units=units,
        channels=channels,
        temporal_kernel=temporal_kernel,
        use_input_skip=True,
        include_abs_skip=True,
    )
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate), loss="mse")
    history = model.fit(
        x_train,
        y_train,
        validation_split=0.2,
        epochs=epochs,
        batch_size=batch_size,
        verbose=0,
    )
    pred = model.predict(x_test, batch_size=batch_size, verbose=0)
    return {
        "baseline": "matched_dense_temporal_no_hopf",
        "test_mse": float(np.mean((pred - y_test) ** 2)),
        "val_mse": float(history.history["val_loss"][-1]),
        "epochs": epochs,
        "num_samples": num_samples,
        "units": units,
        "channels": channels,
        "temporal_kernel": temporal_kernel,
    }


def _safe_scale(x: np.ndarray) -> float:
    return max(float(np.std(x)), 1e-6)


def table3_matched_no_hopf_baseline(
    task: str,
    seed: int,
    num_samples: int,
    dt: float,
    duration: float,
    num_components: int,
    fmin_hz: float,
    fmax_hz: float,
    test_ratio: float,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    units: int,
    channels: int,
    temporal_kernel: int,
) -> dict[str, Any]:
    set_seed(seed)
    x, y, _ = generate_operator_dataset(
        task=task,
        num_samples=num_samples,
        dt=dt,
        duration=duration,
        num_components=num_components,
        fmin_hz=fmin_hz,
        fmax_hz=fmax_hz,
        seed=seed,
    )
    x_train, y_train, x_test, y_test = split_operator_train_test(x=x, y=y, test_ratio=test_ratio, seed=seed)
    x_scale = _safe_scale(x_train)
    y_scale = _safe_scale(y_train)
    model = MatchedNoHopfTemporalRegressor(
        units=units,
        channels=channels,
        temporal_kernel=temporal_kernel,
        use_input_skip=True,
        include_abs_skip=False,
    )
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate), loss="mse")
    history = model.fit(
        x_train / x_scale,
        y_train / y_scale,
        validation_split=0.2,
        epochs=epochs,
        batch_size=batch_size,
        verbose=0,
    )
    pred = model.predict(x_test / x_scale, batch_size=batch_size, verbose=0) * y_scale
    target_std = _safe_scale(y_test)
    test_mse = float(np.mean((pred - y_test) ** 2))
    return {
        "baseline": "matched_dense_temporal_no_hopf",
        "task": task,
        "test_mse": test_mse,
        "val_mse": float(history.history["val_loss"][-1] * (y_scale**2)),
        "test_corr": float(np.corrcoef(pred.reshape(-1), y_test.reshape(-1))[0, 1]),
        "normalized_mse": float(test_mse / (target_std**2)),
        "relative_rmse": float(np.sqrt(test_mse) / target_std),
        "epochs": epochs,
        "num_samples": num_samples,
        "units": units,
        "channels": channels,
        "temporal_kernel": temporal_kernel,
    }


def _moving_average(x: np.ndarray, kernel_size: int) -> np.ndarray:
    kernel = np.ones(kernel_size, dtype=np.float32) / float(kernel_size)
    pad = kernel_size // 2
    padded = np.pad(x, ((0, 0), (pad, pad)), mode="edge")
    rows = [np.convolve(row, kernel, mode="valid")[: x.shape[1]] for row in padded]
    return np.stack(rows, axis=0).astype(np.float32)


def table2_coherent_demod_baseline(
    seed: int,
    num_samples: int,
    dt: float,
    duration: float,
    carrier_hz: float,
    num_components: int,
    msg_fmin: float,
    msg_fmax: float,
    test_ratio: float,
) -> dict[str, Any]:
    x, y, t = generate_demod_dataset(
        num_samples=num_samples,
        dt=dt,
        duration=duration,
        carrier_hz=carrier_hz,
        num_components=num_components,
        msg_fmin=msg_fmin,
        msg_fmax=msg_fmax,
        seed=seed,
    )
    _, _, x_test, y_test = split_train_test(x=x, y=y, test_ratio=test_ratio, seed=seed)
    carrier = np.sin(2.0 * np.pi * carrier_hz * t)[None, :]
    mixed = 2.0 * x_test[:, :, 0] * carrier

    candidates: list[dict[str, Any]] = []
    for kernel_size in [5, 9, 13, 17, 25, 33, 41, 51]:
        pred = _moving_average(mixed, kernel_size=kernel_size) - 1.0
        candidates.append(
            {
                "kernel_size": kernel_size,
                "test_mse": float(np.mean((pred[:, :, None] - y_test) ** 2)),
                "test_corr": float(np.corrcoef(pred.reshape(-1), y_test[:, :, 0].reshape(-1))[0, 1]),
            }
        )
    best = min(candidates, key=lambda item: item["test_mse"])
    return {
        "baseline": "coherent_demod_known_carrier",
        "best": best,
        "candidates": candidates,
        "note": "Uses the fixed carrier frequency stated by the task, then low-pass filters by moving average.",
    }


def collect_existing_metrics() -> dict[str, Any]:
    table3 = _load_json(Path("artifacts/plots/table3/third_work_visual_metrics.json")) or {}
    return {
        "table1_donn_ce_saved": _load_json(Path("artifacts/plots/table1/first_work_visual_metrics_ce.json")),
        "table1_donn_ramp_supplement": _load_json(
            Path("artifacts/plots/table1/first_work_paper_style_supplement_metrics.json")
        ),
        "table1_donn_ramp_article": _load_json(
            Path("artifacts/plots/table1/first_work_paper_style_article_metrics.json")
        ),
        "table2_donn": _load_json(Path("artifacts/plots/table2/second_work_visual_metrics_fixed.json")),
        "table3_donn_and_numeric": table3,
        "table4_donn_1k1e": _load_json(
            Path("artifacts/plots/table4/fourth_work_paper_exact_metrics_1k1e_post_hopf.json")
        ),
        "table4_donn_2k2e": _load_json(
            Path("artifacts/plots/table4/fourth_work_paper_exact_metrics_2k2e_post_hopf.json")
        ),
        "table4_bilstm_1k1e": _load_json(Path("artifacts/plots/table4/fourth_work_paper_baseline_metrics_1k1e.json")),
        "table5_ocnn_smoke": _load_json(Path("artifacts/plots/table5/fifth_work_ocnn_smoke_metrics.json")),
    }


def summarize(metrics: dict[str, Any]) -> list[dict[str, Any]]:
    existing = metrics["existing"]
    summary: list[dict[str, Any]] = []

    t1_fft_saved = metrics["new"]["table1_fft_saved"]
    t1_fft_supp = metrics["new"]["table1_fft_supplement"]
    t1_fft_article = metrics["new"]["table1_fft_article"]
    summary.append(
        {
            "task": "Table 1 classification",
            "oscillator_result": {
                "CE saved": existing["table1_donn_ce_saved"]["test_acc"],
                "ramp supplement": existing["table1_donn_ramp_supplement"]["test_acc"],
                "ramp article": existing["table1_donn_ramp_article"]["test_acc"],
            },
            "non_oscillator_result": {
                "FFT saved": t1_fft_saved["test_acc"],
                "FFT supplement": t1_fft_supp["test_acc"],
                "FFT article": t1_fft_article["test_acc"],
            },
            "local_winner": "non_oscillator_baseline",
        }
    )

    t2_donn = existing["table2_donn"]
    t2_conv = metrics["new"]["table2_temporal_conv_no_hopf"]
    t2_matched = metrics["new"]["table2_matched_no_hopf"]
    t2_coherent = metrics["new"]["table2_coherent_known_carrier"]
    summary.append(
        {
            "task": "Table 2 demodulation",
            "oscillator_result": {"DONN test_mse": t2_donn["test_mse"], "DONN val_mse": t2_donn["val_mse"]},
            "non_oscillator_result": {
                "TemporalConv test_mse": t2_conv["test_mse"],
                "TemporalConv val_mse": t2_conv["val_mse"],
                "Matched no-Hopf test_mse": t2_matched["test_mse"],
                "Matched no-Hopf val_mse": t2_matched["val_mse"],
                "Coherent demod best test_mse": t2_coherent["best"]["test_mse"],
                "Coherent demod best corr": t2_coherent["best"]["test_corr"],
            },
            "local_winner": (
                "oscillator"
                if t2_donn["test_mse"]
                < min(t2_conv["test_mse"], t2_matched["test_mse"], t2_coherent["best"]["test_mse"])
                else "non_oscillator_baseline"
            ),
        }
    )

    table3 = existing["table3_donn_and_numeric"]
    t3_matched_integration = metrics["new"]["table3_matched_no_hopf_integration"]
    t3_matched_differentiation = metrics["new"]["table3_matched_no_hopf_differentiation"]
    summary.append(
        {
            "task": "Table 3 operators",
            "oscillator_result": {
                "integration test_mse": table3["integration"]["test_mse"],
                "differentiation test_mse": table3["differentiation"]["test_mse"],
            },
            "non_oscillator_result": {
                "integration numeric_mse": table3["integration"]["baseline_mse"],
                "differentiation numeric_mse": table3["differentiation"]["baseline_mse"],
                "integration matched no-Hopf test_mse": t3_matched_integration["test_mse"],
                "differentiation matched no-Hopf test_mse": t3_matched_differentiation["test_mse"],
            },
            "local_winner": "non_oscillator_task_baseline",
        }
    )

    summary.append(
        {
            "task": "Table 4 sentiment",
            "oscillator_result": {
                "DONN 1k1e test_acc": existing["table4_donn_1k1e"]["test_acc"],
                "DONN 2k2e test_acc": existing["table4_donn_2k2e"]["test_acc"],
            },
            "non_oscillator_result": {"BiLSTM 1k1e test_acc": existing["table4_bilstm_1k1e"]["test_acc"]},
            "local_winner": "non_oscillator_baseline",
        }
    )
    return summary


def plot_summary(out_path: Path, summary: list[dict[str, Any]]) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    axes = axes.ravel()

    axes[0].bar(
        ["DONN CE", "DONN ramp", "FFT"],
        [
            summary[0]["oscillator_result"]["CE saved"],
            summary[0]["oscillator_result"]["ramp supplement"],
            summary[0]["non_oscillator_result"]["FFT saved"],
        ],
    )
    axes[0].set_ylim(0, 1.05)
    axes[0].set_title("Table 1 accuracy")
    axes[0].grid(axis="y", alpha=0.25)

    axes[1].bar(
        ["DONN", "raw Conv", "matched", "coherent"],
        [
            summary[1]["oscillator_result"]["DONN test_mse"],
            summary[1]["non_oscillator_result"]["TemporalConv test_mse"],
            summary[1]["non_oscillator_result"]["Matched no-Hopf test_mse"],
            summary[1]["non_oscillator_result"]["Coherent demod best test_mse"],
        ],
    )
    axes[1].set_title("Table 2 test MSE")
    axes[1].grid(axis="y", alpha=0.25)

    axes[2].bar(
        ["DONN int", "matched int", "numeric int", "DONN diff", "matched diff", "numeric diff"],
        [
            summary[2]["oscillator_result"]["integration test_mse"],
            summary[2]["non_oscillator_result"]["integration matched no-Hopf test_mse"],
            summary[2]["non_oscillator_result"]["integration numeric_mse"],
            summary[2]["oscillator_result"]["differentiation test_mse"],
            summary[2]["non_oscillator_result"]["differentiation matched no-Hopf test_mse"],
            summary[2]["non_oscillator_result"]["differentiation numeric_mse"],
        ],
    )
    axes[2].set_yscale("log")
    axes[2].set_title("Table 3 MSE, log scale")
    axes[2].grid(axis="y", alpha=0.25)

    axes[3].bar(
        ["DONN 1k1e", "DONN 2k2e", "BiLSTM 1k1e"],
        [
            summary[3]["oscillator_result"]["DONN 1k1e test_acc"],
            summary[3]["oscillator_result"]["DONN 2k2e test_acc"],
            summary[3]["non_oscillator_result"]["BiLSTM 1k1e test_acc"],
        ],
    )
    axes[3].set_ylim(0, 1.0)
    axes[3].set_title("Table 4 accuracy")
    axes[3].grid(axis="y", alpha=0.25)

    fig.suptitle("Oscillator layers vs simple local baselines", fontsize=14)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=170, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--table2-epochs", type=int, default=60)
    parser.add_argument("--table2-samples", type=int, default=400)
    parser.add_argument("--table2-batch-size", type=int, default=32)
    parser.add_argument("--table2-channels", type=int, default=64)
    parser.add_argument("--table3-epochs", type=int, default=30)
    parser.add_argument("--table3-samples", type=int, default=200)
    parser.add_argument("--table3-batch-size", type=int, default=16)
    parser.add_argument("--table3-channels", type=int, default=48)
    parser.add_argument("--metrics-path", type=Path, default=Path("artifacts/plots/ablation/oscillator_vs_baselines_metrics.json"))
    parser.add_argument("--out-path", type=Path, default=Path("artifacts/plots/ablation/oscillator_vs_baselines_summary.png"))
    args = parser.parse_args()

    new_metrics = {
        "table1_fft_saved": spectral_table1_baseline(
            source="saved-arrays",
            seed=args.seed,
            samples_per_class=500,
            num_steps=1000,
            dt=0.001,
            num_components=5,
            test_ratio=0.2,
        ),
        "table1_fft_supplement": spectral_table1_baseline(
            source="supplement-notebook",
            seed=args.seed,
            samples_per_class=500,
            num_steps=1000,
            dt=0.001,
            num_components=5,
            test_ratio=0.2,
        ),
        "table1_fft_article": spectral_table1_baseline(
            source="article",
            seed=args.seed,
            samples_per_class=500,
            num_steps=1000,
            dt=0.001,
            num_components=5,
            test_ratio=0.2,
        ),
        "table2_temporal_conv_no_hopf": table2_temporal_conv_baseline(
            seed=args.seed,
            num_samples=args.table2_samples,
            dt=0.01,
            duration=1.0,
            carrier_hz=8.0,
            num_components=5,
            msg_fmin=1.0,
            msg_fmax=5.0,
            test_ratio=0.2,
            epochs=args.table2_epochs,
            batch_size=args.table2_batch_size,
            learning_rate=0.01,
            channels=args.table2_channels,
            temporal_kernel=0,
        ),
        "table2_matched_no_hopf": table2_matched_no_hopf_baseline(
            seed=args.seed,
            num_samples=args.table2_samples,
            dt=0.01,
            duration=1.0,
            carrier_hz=8.0,
            num_components=5,
            msg_fmin=1.0,
            msg_fmax=5.0,
            test_ratio=0.2,
            epochs=args.table2_epochs,
            batch_size=args.table2_batch_size,
            learning_rate=0.01,
            units=40,
            channels=args.table2_channels,
            temporal_kernel=0,
        ),
        "table2_coherent_known_carrier": table2_coherent_demod_baseline(
            seed=args.seed,
            num_samples=args.table2_samples,
            dt=0.01,
            duration=1.0,
            carrier_hz=8.0,
            num_components=5,
            msg_fmin=1.0,
            msg_fmax=5.0,
            test_ratio=0.2,
        ),
        "table3_matched_no_hopf_integration": table3_matched_no_hopf_baseline(
            task="integration",
            seed=args.seed,
            num_samples=args.table3_samples,
            dt=0.001,
            duration=1.0,
            num_components=5,
            fmin_hz=1.0,
            fmax_hz=5.0,
            test_ratio=0.2,
            epochs=args.table3_epochs,
            batch_size=args.table3_batch_size,
            learning_rate=0.001,
            units=20,
            channels=args.table3_channels,
            temporal_kernel=33,
        ),
        "table3_matched_no_hopf_differentiation": table3_matched_no_hopf_baseline(
            task="differentiation",
            seed=args.seed,
            num_samples=args.table3_samples,
            dt=0.001,
            duration=1.0,
            num_components=5,
            fmin_hz=1.0,
            fmax_hz=5.0,
            test_ratio=0.2,
            epochs=args.table3_epochs,
            batch_size=args.table3_batch_size,
            learning_rate=0.001,
            units=20,
            channels=args.table3_channels,
            temporal_kernel=33,
        ),
    }
    metrics = {"existing": collect_existing_metrics(), "new": new_metrics}
    metrics["summary"] = summarize(metrics)
    metrics["overall_local_conclusion"] = (
        "In the current local reproductions there is no broad empirical advantage for Hopf/oscillator layers. "
        "Simple non-oscillator baselines win Table 1, Table 3, and Table 4. Table 2 is the one local case where "
        "the current Hopf model beats the checked raw Conv1D, matched no-Hopf temporal readout, and known-carrier "
        "coherent demodulation baselines."
    )

    plot_summary(args.out_path, metrics["summary"])
    args.metrics_path.parent.mkdir(parents=True, exist_ok=True)
    args.metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))
    print(f"Saved figure: {args.out_path}")
    print(f"Saved metrics: {args.metrics_path}")


if __name__ == "__main__":
    main()
