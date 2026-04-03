"""Table 4: IMDB sentiment analysis with a DONN-style sequence classifier."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences

from src.HopfLayer import HopfLayer, set_seed


def make_ramp_targets(labels: np.ndarray, num_steps: int, num_classes: int = 2) -> np.ndarray:
    """Create paper-style sequence targets with a rising ramp in the true class channel."""
    ramp = np.linspace(0.0, 1.0, num_steps, dtype=np.float32)
    y = np.zeros((labels.shape[0], num_steps, num_classes), dtype=np.float32)
    y[np.arange(labels.shape[0]), :, labels.astype(np.int64)] = ramp[None, :]
    return y


def labels_from_targets(y: np.ndarray) -> np.ndarray:
    return np.argmax(np.sum(y, axis=1), axis=1).astype(np.int64)


def make_one_hot_targets(labels: np.ndarray, num_classes: int = 2) -> np.ndarray:
    return tf.keras.utils.to_categorical(labels, num_classes=num_classes).astype(np.float32)


def aggregate_sequence_logits(logits: np.ndarray) -> np.ndarray:
    return np.sum(logits, axis=1)


def sequence_accuracy(logits: np.ndarray, y_true: np.ndarray) -> float:
    pred = np.argmax(aggregate_sequence_logits(logits), axis=1)
    true = labels_from_targets(y_true)
    return float(np.mean(pred == true))


def classification_accuracy(logits: np.ndarray, y_true: np.ndarray) -> float:
    pred = np.argmax(logits, axis=1)
    true = np.argmax(y_true, axis=1)
    return float(np.mean(pred == true))


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


def _select_subset(x: np.ndarray, y: np.ndarray, limit: int | None, seed: int) -> tuple[np.ndarray, np.ndarray]:
    if limit is None or limit >= x.shape[0]:
        return x, y
    idx = np.arange(x.shape[0])
    rng = np.random.default_rng(seed)
    rng.shuffle(idx)
    idx = idx[:limit]
    return x[idx], y[idx]


def load_imdb_dataset(
    vocab_size: int,
    max_len: int,
    train_samples: int | None,
    test_samples: int | None,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load IMDB, keep the top vocab, pad reviews, and optionally subsample."""
    (x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=vocab_size)
    x_train, y_train = _select_subset(np.array(x_train, dtype=object), np.array(y_train), train_samples, seed)
    x_test, y_test = _select_subset(np.array(x_test, dtype=object), np.array(y_test), test_samples, seed + 1)

    x_train = pad_sequences(x_train, maxlen=max_len, padding="pre", truncating="pre")
    x_test = pad_sequences(x_test, maxlen=max_len, padding="pre", truncating="pre")
    return x_train.astype(np.int32), y_train.astype(np.int64), x_test.astype(np.int32), y_test.astype(np.int64)


def get_imdb_decoder() -> dict[int, str] | None:
    """Build an index-to-word mapping. Falls back to None if unavailable."""
    try:
        word_index = imdb.get_word_index()
    except Exception:
        return None

    decoder = {index + 3: word for word, index in word_index.items()}
    decoder[0] = "<pad>"
    decoder[1] = "<start>"
    decoder[2] = "<unk>"
    decoder[3] = "<unused>"
    return decoder


def decode_review(tokens: np.ndarray, decoder: dict[int, str] | None, max_words: int = 80) -> str:
    if decoder is None:
        words = [str(int(token)) for token in tokens if int(token) != 0][:max_words]
        return " ".join(words)

    words = []
    for token in tokens:
        token = int(token)
        if token == 0:
            continue
        words.append(decoder.get(token, "<unk>"))
        if len(words) >= max_words:
            break
    return " ".join(words)


class DONNSentimentClassifier(tf.keras.Model):
    """Embedding -> Hopf -> ReLU -> Hopf -> ReLU -> tanh -> sequence logits."""

    def __init__(
        self,
        vocab_size: int,
        num_steps: int,
        embed_dim: int = 100,
        units: int = 64,
        proj_dim: int = 20,
        num_classes: int = 2,
        min_omega_hz: float = 1.0,
        max_omega_hz: float = 15.0,
        dt: float = 0.001,
        hopf_input_scale: float = 0.2,
    ) -> None:
        super().__init__()

        self.embed = tf.keras.layers.Embedding(vocab_size, embed_dim, mask_zero=False)

        self.in1_r = tf.keras.layers.Dense(units, activation="relu")
        self.in1_i = tf.keras.layers.Dense(units, activation="relu")
        self.hopf1 = HopfLayer(
            units=units,
            num_steps=num_steps,
            min_omega_hz=min_omega_hz,
            max_omega_hz=max_omega_hz,
            dt=dt,
            input_scale=hopf_input_scale,
            trainable_omegas=True,
        )
        self.post1 = tf.keras.layers.Dense(units, activation="relu")

        self.in2_r = tf.keras.layers.Dense(units, activation="relu")
        self.in2_i = tf.keras.layers.Dense(units, activation="relu")
        self.hopf2 = HopfLayer(
            units=units,
            num_steps=num_steps,
            min_omega_hz=min_omega_hz,
            max_omega_hz=max_omega_hz,
            dt=dt,
            input_scale=hopf_input_scale,
            trainable_omegas=True,
        )
        self.post2 = tf.keras.layers.Dense(units, activation="relu")
        self.proj = tf.keras.layers.Dense(proj_dim, activation="tanh")
        self.head = tf.keras.layers.Dense(num_classes, activation="linear")

        self.td_in1_r = tf.keras.layers.TimeDistributed(self.in1_r)
        self.td_in1_i = tf.keras.layers.TimeDistributed(self.in1_i)
        self.td_post1 = tf.keras.layers.TimeDistributed(self.post1)
        self.td_in2_r = tf.keras.layers.TimeDistributed(self.in2_r)
        self.td_in2_i = tf.keras.layers.TimeDistributed(self.in2_i)
        self.td_post2 = tf.keras.layers.TimeDistributed(self.post2)
        self.td_proj = tf.keras.layers.TimeDistributed(self.proj)
        self.td_head = tf.keras.layers.TimeDistributed(self.head)

    def call(self, x: tf.Tensor) -> tf.Tensor:
        emb = self.embed(x)

        x1_r = self.td_in1_r(emb)
        x1_i = self.td_in1_i(emb)
        z1_r, z1_i = self.hopf1(x1_r, x1_i)
        h1 = self.td_post1(tf.concat([z1_r, z1_i], axis=2))

        x2_r = self.td_in2_r(h1)
        x2_i = self.td_in2_i(h1)
        z2_r, z2_i = self.hopf2(x2_r, x2_i)
        h2 = self.td_post2(tf.concat([z2_r, z2_i], axis=2))

        h3 = self.td_proj(h2)
        return self.td_head(h3)


class PaperDONNSentimentClassifier(tf.keras.Model):
    """Table-4-style many-to-one DONN with MSE on 2D sentiment targets."""

    def __init__(
        self,
        vocab_size: int,
        num_steps: int,
        embed_dim: int = 100,
        units: int = 100,
        proj_dim: int = 20,
        num_classes: int = 2,
        min_omega_hz: float = 1.0,
        max_omega_hz: float = 15.0,
        dt: float = 0.001,
        hopf_input_scale: float = 0.2,
    ) -> None:
        super().__init__()

        self.embed = tf.keras.layers.Embedding(vocab_size, embed_dim, mask_zero=False)
        self.embed_proj = None
        self.td_embed_proj = None
        if embed_dim != units:
            self.embed_proj = tf.keras.layers.Dense(units, activation="linear")
            self.td_embed_proj = tf.keras.layers.TimeDistributed(self.embed_proj)

        self.hopf1 = HopfLayer(
            units=units,
            num_steps=num_steps,
            min_omega_hz=min_omega_hz,
            max_omega_hz=max_omega_hz,
            dt=dt,
            input_scale=hopf_input_scale,
            trainable_omegas=True,
        )
        self.post1 = tf.keras.layers.Dense(units, activation="relu")

        self.hopf2 = HopfLayer(
            units=units,
            num_steps=num_steps,
            min_omega_hz=min_omega_hz,
            max_omega_hz=max_omega_hz,
            dt=dt,
            input_scale=hopf_input_scale,
            trainable_omegas=True,
        )
        self.post2 = tf.keras.layers.Dense(units, activation="relu")
        self.proj = tf.keras.layers.Dense(proj_dim, activation="tanh")
        self.head = tf.keras.layers.Dense(num_classes, activation="linear")

        self.td_post1 = tf.keras.layers.TimeDistributed(self.post1)
        self.td_post2 = tf.keras.layers.TimeDistributed(self.post2)

    def call(self, x: tf.Tensor) -> tf.Tensor:
        h0 = self.embed(x)
        if self.td_embed_proj is not None:
            h0 = self.td_embed_proj(h0)

        z1_r, z1_i = self.hopf1(h0, tf.zeros_like(h0))
        h1 = self.td_post1(tf.concat([z1_r, z1_i], axis=2))

        z2_r, z2_i = self.hopf2(h1, tf.zeros_like(h1))
        h2 = self.td_post2(tf.concat([z2_r, z2_i], axis=2))

        last = h2[:, -1, :]
        h3 = self.proj(last)
        return self.head(h3)


@dataclass
class SentimentMetrics:
    test_acc: float
    val_acc: float
    test_loss: float


def train_one_run(
    vocab_size: int,
    max_len: int,
    train_samples: int | None,
    test_samples: int | None,
    seed: int,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    embed_dim: int,
    units: int,
    proj_dim: int,
    hopf_input_scale: float,
    val_ratio: float,
) -> tuple[SentimentMetrics, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    set_seed(seed)
    x_train_full, y_train_labels, x_test, y_test_labels = load_imdb_dataset(
        vocab_size=vocab_size,
        max_len=max_len,
        train_samples=train_samples,
        test_samples=test_samples,
        seed=seed,
    )

    y_train_full = make_ramp_targets(y_train_labels, num_steps=max_len)
    y_test = make_ramp_targets(y_test_labels, num_steps=max_len)
    x_train, y_train, x_val, y_val = split_train_val(x=x_train_full, y=y_train_full, val_ratio=val_ratio, seed=seed)

    model = DONNSentimentClassifier(
        vocab_size=vocab_size,
        num_steps=max_len,
        embed_dim=embed_dim,
        units=units,
        proj_dim=proj_dim,
        hopf_input_scale=hopf_input_scale,
    )
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="mse",
    )
    model.fit(
        x_train,
        y_train,
        validation_data=(x_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        verbose=0,
    )

    pred_test = model.predict(x_test, batch_size=batch_size, verbose=0)
    pred_val = model.predict(x_val, batch_size=batch_size, verbose=0)
    test_loss = float(np.mean((pred_test - y_test) ** 2))
    val_acc = sequence_accuracy(pred_val, y_val)
    test_acc = sequence_accuracy(pred_test, y_test)
    return SentimentMetrics(test_acc=test_acc, val_acc=val_acc, test_loss=test_loss), x_test, y_test, pred_test, y_test_labels


def train_bilstm_baseline(
    vocab_size: int,
    max_len: int,
    train_samples: int | None,
    test_samples: int | None,
    seed: int,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    embed_dim: int,
    lstm_units: int,
    val_ratio: float,
) -> SentimentMetrics:
    """Simple reference baseline on the same IMDB subset and split."""
    set_seed(seed)
    x_train_full, y_train_labels, x_test, y_test_labels = load_imdb_dataset(
        vocab_size=vocab_size,
        max_len=max_len,
        train_samples=train_samples,
        test_samples=test_samples,
        seed=seed,
    )
    x_train, y_train, x_val, y_val = split_train_val(
        x=x_train_full,
        y=y_train_labels,
        val_ratio=val_ratio,
        seed=seed,
    )

    model = tf.keras.Sequential(
        [
            tf.keras.layers.Embedding(vocab_size, embed_dim, mask_zero=False),
            tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(lstm_units)),
            tf.keras.layers.Dense(2, activation="linear"),
        ]
    )
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy(name="acc")],
    )
    model.fit(
        x_train,
        y_train,
        validation_data=(x_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        verbose=0,
    )
    val_loss, val_acc = model.evaluate(x_val, y_val, batch_size=batch_size, verbose=0)
    test_loss, test_acc = model.evaluate(x_test, y_test_labels, batch_size=batch_size, verbose=0)
    return SentimentMetrics(test_acc=float(test_acc), val_acc=float(val_acc), test_loss=float(test_loss))


def train_paper_exact_run(
    vocab_size: int,
    max_len: int,
    train_samples: int | None,
    test_samples: int | None,
    seed: int,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    embed_dim: int,
    units: int,
    proj_dim: int,
    hopf_input_scale: float,
    val_ratio: float,
) -> tuple[SentimentMetrics, np.ndarray, np.ndarray, np.ndarray, dict[str, list[float]], dict[str, int]]:
    """Train the many-to-one Table-4-style DONN with MSE on one-hot labels."""
    set_seed(seed)
    x_train_full, y_train_labels, x_test, y_test_labels = load_imdb_dataset(
        vocab_size=vocab_size,
        max_len=max_len,
        train_samples=train_samples,
        test_samples=test_samples,
        seed=seed,
    )

    y_train_full = make_one_hot_targets(y_train_labels)
    y_test = make_one_hot_targets(y_test_labels)
    x_train, y_train, x_val, y_val = split_train_val(
        x=x_train_full,
        y=y_train_full,
        val_ratio=val_ratio,
        seed=seed,
    )

    model = PaperDONNSentimentClassifier(
        vocab_size=vocab_size,
        num_steps=max_len,
        embed_dim=embed_dim,
        units=units,
        proj_dim=proj_dim,
        hopf_input_scale=hopf_input_scale,
    )
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="mse",
    )
    history = model.fit(
        x_train,
        y_train,
        validation_data=(x_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        verbose=0,
    )

    pred_test = model.predict(x_test, batch_size=batch_size, verbose=0)
    pred_val = model.predict(x_val, batch_size=batch_size, verbose=0)
    test_loss = float(np.mean((pred_test - y_test) ** 2))
    val_acc = classification_accuracy(pred_val, y_val)
    test_acc = classification_accuracy(pred_test, y_test)

    total_params = int(model.count_params())
    embed_params = int(vocab_size * embed_dim)
    param_info = {
        "total_params": total_params,
        "embedding_params": embed_params,
        "non_embedding_params": total_params - embed_params,
    }
    return (
        SentimentMetrics(test_acc=test_acc, val_acc=val_acc, test_loss=test_loss),
        x_test,
        y_test_labels,
        pred_test,
        {key: [float(v) for v in values] for key, values in history.history.items()},
        param_info,
    )
