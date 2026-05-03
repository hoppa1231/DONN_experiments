"""Temporary Table-4 correction sweep.

Runs multiple plausible fixes around the paper-style sentiment setup and writes
a comparable metrics report for each variant.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.HopfLayer import HopfLayer, set_seed


@dataclass(frozen=True)
class Variant:
    name: str
    padding: str  # pre | post
    readout: str  # last | last_valid | mean_valid
    zero_pad_inputs: bool
    loss: str  # mse | ce


def _select_subset(x: np.ndarray, y: np.ndarray, limit: int | None, seed: int) -> tuple[np.ndarray, np.ndarray]:
    if limit is None or limit >= x.shape[0]:
        return x, y
    idx = np.arange(x.shape[0])
    rng = np.random.default_rng(seed)
    rng.shuffle(idx)
    idx = idx[:limit]
    return x[idx], y[idx]


def split_train_val(
    x: np.ndarray,
    y: np.ndarray,
    val_ratio: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    idx = np.arange(x.shape[0])
    rng = np.random.default_rng(seed)
    rng.shuffle(idx)
    n_val = int(round(x.shape[0] * val_ratio))
    val_idx = idx[:n_val]
    train_idx = idx[n_val:]
    return x[train_idx], y[train_idx], x[val_idx], y[val_idx]


def load_imdb_with_padding(
    vocab_size: int,
    max_len: int,
    padding: str,
    train_samples: int | None,
    test_samples: int | None,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    (x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=vocab_size)
    x_train, y_train = _select_subset(np.array(x_train, dtype=object), np.array(y_train), train_samples, seed)
    x_test, y_test = _select_subset(np.array(x_test, dtype=object), np.array(y_test), test_samples, seed + 1)

    x_train = pad_sequences(x_train, maxlen=max_len, padding=padding, truncating="pre")
    x_test = pad_sequences(x_test, maxlen=max_len, padding=padding, truncating="pre")
    return x_train.astype(np.int32), y_train.astype(np.int64), x_test.astype(np.int32), y_test.astype(np.int64)


class PaperDONNVariant(tf.keras.Model):
    def __init__(
        self,
        vocab_size: int,
        num_steps: int,
        embed_dim: int,
        units: int,
        proj_dim: int,
        readout: str,
        zero_pad_inputs: bool,
        hopf_input_scale: float,
    ) -> None:
        super().__init__()
        self.readout = readout
        self.zero_pad_inputs = zero_pad_inputs

        self.embed = tf.keras.layers.Embedding(vocab_size, embed_dim, mask_zero=False)
        self.hopf1 = HopfLayer(
            units=units,
            num_steps=num_steps,
            min_omega_hz=1.0,
            max_omega_hz=15.0,
            dt=0.001,
            input_scale=hopf_input_scale,
            trainable_omegas=True,
        )
        self.post1 = tf.keras.layers.Dense(units, activation="relu")
        self.hopf2 = HopfLayer(
            units=units,
            num_steps=num_steps,
            min_omega_hz=1.0,
            max_omega_hz=15.0,
            dt=0.001,
            input_scale=hopf_input_scale,
            trainable_omegas=True,
        )
        self.post2 = tf.keras.layers.Dense(units, activation="relu")
        self.proj = tf.keras.layers.Dense(proj_dim, activation="tanh")
        self.head = tf.keras.layers.Dense(2, activation="linear")
        self.td_post1 = tf.keras.layers.TimeDistributed(self.post1)
        self.td_post2 = tf.keras.layers.TimeDistributed(self.post2)

    def _readout(self, x_tokens: tf.Tensor, h2: tf.Tensor) -> tf.Tensor:
        if self.readout == "last":
            return h2[:, -1, :]

        lengths = tf.reduce_sum(tf.cast(x_tokens != 0, tf.int32), axis=1)
        if self.readout == "last_valid":
            last_idx = tf.maximum(lengths - 1, 0)
            batch_idx = tf.range(tf.shape(h2)[0], dtype=tf.int32)
            gather_idx = tf.stack([batch_idx, last_idx], axis=1)
            return tf.gather_nd(h2, gather_idx)

        if self.readout == "mean_valid":
            mask = tf.cast(x_tokens != 0, h2.dtype)[:, :, None]
            denom = tf.maximum(tf.reduce_sum(mask, axis=1), 1.0)
            return tf.reduce_sum(h2 * mask, axis=1) / denom

        raise ValueError(f"Unsupported readout mode: {self.readout}")

    def call(self, x_tokens: tf.Tensor) -> tf.Tensor:
        h0 = self.embed(x_tokens)
        if self.zero_pad_inputs:
            pad_mask = tf.cast(x_tokens != 0, h0.dtype)[:, :, None]
            h0 = h0 * pad_mask

        z1_r, z1_i = self.hopf1(h0, tf.zeros_like(h0))
        h1 = self.td_post1(tf.concat([z1_r, z1_i], axis=2))
        z2_r, z2_i = self.hopf2(h1, tf.zeros_like(h1))
        h2 = self.td_post2(tf.concat([z2_r, z2_i], axis=2))

        readout = self._readout(x_tokens, h2)
        h3 = self.proj(readout)
        return self.head(h3)


def one_hot(labels: np.ndarray) -> np.ndarray:
    return tf.keras.utils.to_categorical(labels, num_classes=2).astype(np.float32)


def evaluate_accuracy(logits: np.ndarray, y_true_labels: np.ndarray) -> float:
    pred = np.argmax(logits, axis=1)
    return float(np.mean(pred == y_true_labels))


def run_single_variant(
    variant: Variant,
    train_samples: int,
    test_samples: int,
    epochs: int,
    batch_size: int,
    seed: int,
    learning_rate: float,
    hopf_input_scale: float,
) -> dict[str, float | int | str]:
    set_seed(seed)
    x_train_full, y_train_labels, x_test, y_test_labels = load_imdb_with_padding(
        vocab_size=35000,
        max_len=500,
        padding=variant.padding,
        train_samples=train_samples,
        test_samples=test_samples,
        seed=seed,
    )

    if variant.loss == "mse":
        y_train_full = one_hot(y_train_labels)
    elif variant.loss == "ce":
        y_train_full = y_train_labels
    else:
        raise ValueError(f"Unsupported loss type: {variant.loss}")

    x_train, y_train, x_val, y_val = split_train_val(x_train_full, y_train_full, val_ratio=0.3, seed=seed)
    if variant.loss == "mse":
        y_val_labels = np.argmax(y_val, axis=1)
    else:
        y_val_labels = y_val

    model = PaperDONNVariant(
        vocab_size=35000,
        num_steps=500,
        embed_dim=100,
        units=100,
        proj_dim=20,
        readout=variant.readout,
        zero_pad_inputs=variant.zero_pad_inputs,
        hopf_input_scale=hopf_input_scale,
    )
    if variant.loss == "mse":
        loss_fn = "mse"
    else:
        loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss=loss_fn,
    )

    started = perf_counter()
    hist = model.fit(
        x_train,
        y_train,
        validation_data=(x_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        verbose=0,
    )
    elapsed = perf_counter() - started

    pred_test = model.predict(x_test, batch_size=batch_size, verbose=0)
    pred_val = model.predict(x_val, batch_size=batch_size, verbose=0)
    test_acc = evaluate_accuracy(pred_test, y_test_labels)
    val_acc = evaluate_accuracy(pred_val, y_val_labels)

    if variant.loss == "mse":
        test_loss = float(np.mean((pred_test - one_hot(y_test_labels)) ** 2))
    else:
        ce = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        test_loss = float(ce(y_test_labels, pred_test).numpy())

    out = {
        "name": variant.name,
        "padding": variant.padding,
        "readout": variant.readout,
        "zero_pad_inputs": variant.zero_pad_inputs,
        "loss_type": variant.loss,
        "train_samples": train_samples,
        "test_samples": test_samples,
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "hopf_input_scale": hopf_input_scale,
        "test_acc": test_acc,
        "val_acc": val_acc,
        "test_loss": test_loss,
        "train_loss_last": float(hist.history["loss"][-1]),
        "val_loss_last": float(hist.history["val_loss"][-1]),
        "seconds": float(elapsed),
    }
    return out


def default_variants() -> list[Variant]:
    return [
        Variant("baseline_pre_last_mse", "pre", "last", False, "mse"),
        Variant("pre_last_zeroPad_mse", "pre", "last", True, "mse"),
        Variant("pre_lastValid_mse", "pre", "last_valid", False, "mse"),
        Variant("pre_lastValid_zeroPad_mse", "pre", "last_valid", True, "mse"),
        Variant("pre_meanValid_zeroPad_mse", "pre", "mean_valid", True, "mse"),
        Variant("pre_lastValid_zeroPad_ce", "pre", "last_valid", True, "ce"),
        Variant("pre_meanValid_zeroPad_ce", "pre", "mean_valid", True, "ce"),
        Variant("post_lastValid_zeroPad_mse", "post", "last_valid", True, "mse"),
        Variant("post_meanValid_zeroPad_ce", "post", "mean_valid", True, "ce"),
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-samples", type=int, default=1024)
    parser.add_argument("--test-samples", type=int, default=1024)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--hopf-input-scale", type=float, default=0.2)
    parser.add_argument(
        "--variant-names",
        type=str,
        default="",
        help="Comma-separated subset of variant names to run. Empty means all.",
    )
    parser.add_argument(
        "--out-path",
        type=Path,
        default=Path("artifacts/plots/table4/fourth_work_temp_variants_results.json"),
    )
    args = parser.parse_args()

    variants = default_variants()
    if args.variant_names.strip():
        requested = {name.strip() for name in args.variant_names.split(",") if name.strip()}
        variants = [variant for variant in variants if variant.name in requested]
        missing = sorted(requested - {variant.name for variant in variants})
        if missing:
            print(f"[warn] Unknown variant names skipped: {', '.join(missing)}")

    if not variants:
        raise ValueError("No variants selected to run.")

    results: list[dict[str, float | int | str]] = []
    for variant in variants:
        print(f"[run] {variant.name}")
        result = run_single_variant(
            variant=variant,
            train_samples=args.train_samples,
            test_samples=args.test_samples,
            epochs=args.epochs,
            batch_size=args.batch_size,
            seed=args.seed,
            learning_rate=args.learning_rate,
            hopf_input_scale=args.hopf_input_scale,
        )
        results.append(result)
        print(
            f"  test_acc={result['test_acc']:.4f} val_acc={result['val_acc']:.4f} "
            f"test_loss={result['test_loss']:.6f} time={result['seconds']:.1f}s"
        )

    results_sorted = sorted(results, key=lambda x: float(x["test_acc"]), reverse=True)
    payload = {
        "purpose": "temporary_table4_correction_sweep",
        "paper_reported_acc": 0.852,
        "run_config": {
            "train_samples": args.train_samples,
            "test_samples": args.test_samples,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "seed": args.seed,
            "learning_rate": args.learning_rate,
            "hopf_input_scale": args.hopf_input_scale,
        },
        "results": results_sorted,
    }

    args.out_path.parent.mkdir(parents=True, exist_ok=True)
    args.out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Saved: {args.out_path}")


if __name__ == "__main__":
    main()
