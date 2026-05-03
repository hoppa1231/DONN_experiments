"""Temporary Table-4 correction sweep v2.

This script explores stronger variants intended to improve sentiment accuracy.
It is temporary and does not modify production runners.
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
    max_len: int = 500
    padding: str = "pre"
    truncating: str = "pre"
    readout: str = "last"
    zero_pad_inputs: bool = False
    loss: str = "mse"
    use_input_projections: bool = False
    trainable_omegas: bool = True
    learning_rate: float = 1e-3
    hopf_input_scale: float = 0.2
    units: int = 100
    embed_dim: int = 100
    proj_dim: int = 20


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


def load_imdb_variant(
    vocab_size: int,
    max_len: int,
    padding: str,
    truncating: str,
    train_samples: int | None,
    test_samples: int | None,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    (x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=vocab_size)
    x_train, y_train = _select_subset(np.array(x_train, dtype=object), np.array(y_train), train_samples, seed)
    x_test, y_test = _select_subset(np.array(x_test, dtype=object), np.array(y_test), test_samples, seed + 1)

    x_train = pad_sequences(x_train, maxlen=max_len, padding=padding, truncating=truncating)
    x_test = pad_sequences(x_test, maxlen=max_len, padding=padding, truncating=truncating)
    return x_train.astype(np.int32), y_train.astype(np.int64), x_test.astype(np.int32), y_test.astype(np.int64)


class EnhancedDONNVariant(tf.keras.Model):
    def __init__(self, variant: Variant) -> None:
        super().__init__()
        self.variant = variant

        self.embed = tf.keras.layers.Embedding(35000, variant.embed_dim, mask_zero=False)
        self.embed_proj = None
        self.td_embed_proj = None
        if (not variant.use_input_projections) and variant.embed_dim != variant.units:
            self.embed_proj = tf.keras.layers.Dense(variant.units, activation="linear")
            self.td_embed_proj = tf.keras.layers.TimeDistributed(self.embed_proj)

        self.td_in1_r = None
        self.td_in1_i = None
        self.td_in2_r = None
        self.td_in2_i = None
        if variant.use_input_projections:
            self.td_in1_r = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(variant.units, activation="relu"))
            self.td_in1_i = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(variant.units, activation="relu"))
            self.td_in2_r = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(variant.units, activation="relu"))
            self.td_in2_i = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(variant.units, activation="relu"))

        self.hopf1 = HopfLayer(
            units=variant.units,
            num_steps=variant.max_len,
            min_omega_hz=1.0,
            max_omega_hz=15.0,
            dt=0.001,
            input_scale=variant.hopf_input_scale,
            trainable_omegas=variant.trainable_omegas,
        )
        self.hopf2 = HopfLayer(
            units=variant.units,
            num_steps=variant.max_len,
            min_omega_hz=1.0,
            max_omega_hz=15.0,
            dt=0.001,
            input_scale=variant.hopf_input_scale,
            trainable_omegas=variant.trainable_omegas,
        )

        self.post1 = tf.keras.layers.Dense(variant.units, activation="relu")
        self.post2 = tf.keras.layers.Dense(variant.units, activation="relu")
        self.td_post1 = tf.keras.layers.TimeDistributed(self.post1)
        self.td_post2 = tf.keras.layers.TimeDistributed(self.post2)

        self.proj = tf.keras.layers.Dense(variant.proj_dim, activation="tanh")
        self.head = tf.keras.layers.Dense(2, activation="linear")

    def _valid_mask(self, x_tokens: tf.Tensor, dtype: tf.dtypes.DType) -> tf.Tensor:
        return tf.cast(x_tokens != 0, dtype)[:, :, None]

    def _readout(self, x_tokens: tf.Tensor, h2: tf.Tensor) -> tf.Tensor:
        mode = self.variant.readout
        if mode == "last":
            return h2[:, -1, :]

        lengths = tf.reduce_sum(tf.cast(x_tokens != 0, tf.int32), axis=1)
        if mode == "last_valid":
            last_idx = tf.maximum(lengths - 1, 0)
            batch_idx = tf.range(tf.shape(h2)[0], dtype=tf.int32)
            idx = tf.stack([batch_idx, last_idx], axis=1)
            return tf.gather_nd(h2, idx)

        mask = self._valid_mask(x_tokens, h2.dtype)
        denom = tf.maximum(tf.reduce_sum(mask, axis=1), 1.0)
        mean_pool = tf.reduce_sum(h2 * mask, axis=1) / denom
        if mode == "mean_valid":
            return mean_pool
        if mode == "meanmax_valid":
            neg_inf = tf.constant(-1e9, dtype=h2.dtype)
            masked = tf.where(mask > 0, h2, neg_inf)
            max_pool = tf.reduce_max(masked, axis=1)
            return tf.concat([mean_pool, max_pool], axis=1)
        raise ValueError(f"Unsupported readout mode: {mode}")

    def call(self, x_tokens: tf.Tensor) -> tf.Tensor:
        h0 = self.embed(x_tokens)
        if self.variant.zero_pad_inputs:
            h0 = h0 * self._valid_mask(x_tokens, h0.dtype)

        if self.td_embed_proj is not None:
            h0 = self.td_embed_proj(h0)

        if self.variant.use_input_projections:
            x1_r = self.td_in1_r(h0)
            x1_i = self.td_in1_i(h0)
        else:
            x1_r = h0
            x1_i = tf.zeros_like(h0)

        z1_r, z1_i = self.hopf1(x1_r, x1_i)
        h1 = self.td_post1(tf.concat([z1_r, z1_i], axis=2))

        if self.variant.use_input_projections:
            x2_r = self.td_in2_r(h1)
            x2_i = self.td_in2_i(h1)
        else:
            x2_r = h1
            x2_i = tf.zeros_like(h1)

        z2_r, z2_i = self.hopf2(x2_r, x2_i)
        h2 = self.td_post2(tf.concat([z2_r, z2_i], axis=2))
        readout = self._readout(x_tokens, h2)
        return self.head(self.proj(readout))


def one_hot(labels: np.ndarray) -> np.ndarray:
    return tf.keras.utils.to_categorical(labels, num_classes=2).astype(np.float32)


def accuracy_from_logits(logits: np.ndarray, y_true_labels: np.ndarray) -> float:
    pred = np.argmax(logits, axis=1)
    return float(np.mean(pred == y_true_labels))


def run_variant(
    variant: Variant,
    train_samples: int,
    test_samples: int,
    epochs: int,
    batch_size: int,
    seed: int,
) -> dict[str, float | int | bool | str]:
    set_seed(seed)
    x_train_full, y_train_labels, x_test, y_test_labels = load_imdb_variant(
        vocab_size=35000,
        max_len=variant.max_len,
        padding=variant.padding,
        truncating=variant.truncating,
        train_samples=train_samples,
        test_samples=test_samples,
        seed=seed,
    )

    if variant.loss == "mse":
        y_train_full = one_hot(y_train_labels)
    elif variant.loss == "ce":
        y_train_full = y_train_labels
    else:
        raise ValueError(f"Unsupported loss: {variant.loss}")

    x_train, y_train, x_val, y_val = split_train_val(x_train_full, y_train_full, val_ratio=0.3, seed=seed)
    y_val_labels = np.argmax(y_val, axis=1) if variant.loss == "mse" else y_val

    model = EnhancedDONNVariant(variant)
    if variant.loss == "mse":
        loss_fn = "mse"
    else:
        loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=variant.learning_rate),
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
    test_acc = accuracy_from_logits(pred_test, y_test_labels)
    val_acc = accuracy_from_logits(pred_val, y_val_labels)

    if variant.loss == "mse":
        test_loss = float(np.mean((pred_test - one_hot(y_test_labels)) ** 2))
    else:
        ce = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        test_loss = float(ce(y_test_labels, pred_test).numpy())

    return {
        "name": variant.name,
        "max_len": variant.max_len,
        "padding": variant.padding,
        "truncating": variant.truncating,
        "readout": variant.readout,
        "zero_pad_inputs": variant.zero_pad_inputs,
        "loss_type": variant.loss,
        "use_input_projections": variant.use_input_projections,
        "trainable_omegas": variant.trainable_omegas,
        "learning_rate": variant.learning_rate,
        "hopf_input_scale": variant.hopf_input_scale,
        "units": variant.units,
        "embed_dim": variant.embed_dim,
        "proj_dim": variant.proj_dim,
        "train_samples": train_samples,
        "test_samples": test_samples,
        "epochs": epochs,
        "batch_size": batch_size,
        "test_acc": test_acc,
        "val_acc": val_acc,
        "test_loss": test_loss,
        "train_loss_last": float(hist.history["loss"][-1]),
        "val_loss_last": float(hist.history["val_loss"][-1]),
        "seconds": float(elapsed),
    }


def all_variants_v2() -> list[Variant]:
    return [
        Variant(
            name="v2_baseline_pre_last_mse",
            max_len=500,
            padding="pre",
            truncating="pre",
            readout="last",
            zero_pad_inputs=False,
            loss="mse",
            use_input_projections=False,
        ),
        Variant(
            name="v2_post_lastValid_ce_500",
            max_len=500,
            padding="post",
            truncating="post",
            readout="last_valid",
            zero_pad_inputs=True,
            loss="ce",
            use_input_projections=False,
        ),
        Variant(
            name="v2_post_meanValid_ce_300",
            max_len=300,
            padding="post",
            truncating="post",
            readout="mean_valid",
            zero_pad_inputs=True,
            loss="ce",
            use_input_projections=False,
        ),
        Variant(
            name="v2_post_meanmax_ce_256",
            max_len=256,
            padding="post",
            truncating="post",
            readout="meanmax_valid",
            zero_pad_inputs=True,
            loss="ce",
            use_input_projections=False,
        ),
        Variant(
            name="v2_post_meanmax_ce_256_proj",
            max_len=256,
            padding="post",
            truncating="post",
            readout="meanmax_valid",
            zero_pad_inputs=True,
            loss="ce",
            use_input_projections=True,
        ),
        Variant(
            name="v2_post_meanmax_ce_300_proj_lr5e4",
            max_len=300,
            padding="post",
            truncating="post",
            readout="meanmax_valid",
            zero_pad_inputs=True,
            loss="ce",
            use_input_projections=True,
            learning_rate=5e-4,
        ),
        Variant(
            name="v2_post_meanmax_ce_300_proj_fixedOmega",
            max_len=300,
            padding="post",
            truncating="post",
            readout="meanmax_valid",
            zero_pad_inputs=True,
            loss="ce",
            use_input_projections=True,
            trainable_omegas=False,
        ),
        Variant(
            name="v2_pre_meanmax_ce_300_proj",
            max_len=300,
            padding="pre",
            truncating="pre",
            readout="meanmax_valid",
            zero_pad_inputs=True,
            loss="ce",
            use_input_projections=True,
        ),
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-samples", type=int, default=2048)
    parser.add_argument("--test-samples", type=int, default=2048)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--variant-names",
        type=str,
        default="",
        help="Comma-separated subset of variant names. Empty means all.",
    )
    parser.add_argument(
        "--out-path",
        type=Path,
        default=Path("artifacts/plots/table4/fourth_work_temp_variants_v2_results.json"),
    )
    args = parser.parse_args()

    variants = all_variants_v2()
    if args.variant_names.strip():
        requested = {name.strip() for name in args.variant_names.split(",") if name.strip()}
        variants = [variant for variant in variants if variant.name in requested]
        missing = sorted(requested - {variant.name for variant in variants})
        if missing:
            print(f"[warn] Unknown variant names skipped: {', '.join(missing)}")
    if not variants:
        raise ValueError("No variants selected to run.")

    results: list[dict[str, float | int | bool | str]] = []
    for variant in variants:
        print(f"[run] {variant.name}")
        result = run_variant(
            variant=variant,
            train_samples=args.train_samples,
            test_samples=args.test_samples,
            epochs=args.epochs,
            batch_size=args.batch_size,
            seed=args.seed,
        )
        results.append(result)
        print(
            f"  test_acc={result['test_acc']:.4f} val_acc={result['val_acc']:.4f} "
            f"test_loss={result['test_loss']:.6f} time={result['seconds']:.1f}s"
        )

    sorted_results = sorted(results, key=lambda r: float(r["test_acc"]), reverse=True)
    payload = {
        "purpose": "temporary_table4_correction_sweep_v2",
        "paper_reported_acc": 0.852,
        "run_config": {
            "train_samples": args.train_samples,
            "test_samples": args.test_samples,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "seed": args.seed,
        },
        "results": sorted_results,
    }
    args.out_path.parent.mkdir(parents=True, exist_ok=True)
    args.out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Saved: {args.out_path}")


if __name__ == "__main__":
    main()
