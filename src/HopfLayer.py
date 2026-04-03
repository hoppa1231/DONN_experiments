
from __future__ import annotations

from typing import Tuple

import numpy as np
import tensorflow as tf


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    tf.random.set_seed(seed)

    
@tf.function
def _real_part(r: tf.Tensor, phi: tf.Tensor) -> tf.Tensor:
    return r * tf.math.cos(phi)


@tf.function
def _imag_part(r: tf.Tensor, phi: tf.Tensor) -> tf.Tensor:
    return r * tf.math.sin(phi)


@tf.function
def _hopf_rollout(
    x_r: tf.Tensor,
    x_i: tf.Tensor,
    omegas: tf.Tensor,
    num_steps: int,
    dt: float,
    mu: float,
    beta: float,
    input_scale: float,
) -> Tuple[tf.Tensor, tf.Tensor]:
    """Euler integration for a batch of Hopf oscillators."""
    batch_size = tf.shape(x_r)[0]
    dim = tf.shape(x_r)[2]

    r_t = tf.ones((batch_size, dim), dtype=tf.float32)
    phi_t = tf.zeros((batch_size, dim), dtype=tf.float32)

    r_arr = tf.TensorArray(dtype=tf.float32, size=num_steps)
    phi_arr = tf.TensorArray(dtype=tf.float32, size=num_steps)

    for t in tf.range(num_steps):
        input_r = input_scale * x_r[:, t, :] * tf.math.cos(phi_t)
        input_phi = input_scale * x_i[:, t, :] * tf.math.sin(phi_t)
        r_t = r_t + ((mu - beta * tf.square(r_t)) * r_t + input_r) * dt
        phi_t = phi_t + (omegas - input_phi) * dt
        r_arr = r_arr.write(t, r_t)
        phi_arr = phi_arr.write(t, phi_t)

    r = tf.transpose(r_arr.stack(), [1, 0, 2])
    phi = tf.transpose(phi_arr.stack(), [1, 0, 2])
    return r, phi


class HopfLayer(tf.keras.layers.Layer):
    def __init__(
        self,
        units: int,
        num_steps: int,
        min_omega_hz: float = 0.1,
        max_omega_hz: float = 20.0,
        dt: float = 0.001,
        mu: float = 1.0,
        beta: float = 0.01,
        input_scale: float = 0.1,
        trainable_omegas: bool = False,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.units = units
        self.num_steps = num_steps
        self.dt = dt
        self.mu = mu
        self.beta = beta
        self.input_scale = input_scale
        self.trainable_omegas = trainable_omegas

        hz = np.linspace(min_omega_hz, max_omega_hz, units, dtype=np.float32)
        omega_init = np.expand_dims(hz * (2.0 * np.pi), 0)
        self.omegas = self.add_weight(
            name="omegas",
            shape=(1, units),
            dtype=tf.float32,
            initializer=tf.constant_initializer(omega_init),
            trainable=trainable_omegas,
        )

    def call(self, x_r: tf.Tensor, x_i: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        r, phi = _hopf_rollout(
            x_r=x_r,
            x_i=x_i,
            omegas=self.omegas,
            num_steps=self.num_steps,
            dt=self.dt,
            mu=self.mu,
            beta=self.beta,
            input_scale=self.input_scale,
        )
        return _real_part(r, phi), _imag_part(r, phi)
