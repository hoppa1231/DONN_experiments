
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
    beta2: float,
    epsilon: float,
    input_scale: float,
    radius_epsilon: float,
) -> Tuple[tf.Tensor, tf.Tensor]:
    """Euler integration for article-style Hopf oscillators.

    This implements the paper's supercritical/critical regime in polar form:

        r_dot = mu * r + beta * r^3
                + epsilon * beta2 * r^5 / (1 - epsilon * r^2)
                + A(t) * cos(psi)
        psi_dot = Omega - A(t) / r * sin(psi)

    beta2 defaults to 0 for the supercritical/critical experiments in the
    article.  The input pair is interpreted as a complex forcing signal
    I(t)=A(t) exp(i * input_phase), represented by x_r + i*x_i.  The state
    stores theta, so theta_dot = omega - A(t)/r * sin(theta - input_phase).
    """
    batch_size = tf.shape(x_r)[0]
    dim = tf.shape(x_r)[2]

    r_t = tf.ones((batch_size, dim), dtype=tf.float32)
    phi_t = tf.zeros((batch_size, dim), dtype=tf.float32)

    r_arr = tf.TensorArray(dtype=tf.float32, size=num_steps)
    phi_arr = tf.TensorArray(dtype=tf.float32, size=num_steps)

    for t in tf.range(num_steps):
        input_real = input_scale * x_r[:, t, :]
        input_imag = input_scale * x_i[:, t, :]
        safe_r = tf.maximum(r_t, radius_epsilon)
        cos_phi = tf.math.cos(phi_t)
        sin_phi = tf.math.sin(phi_t)
        radial_forcing = input_real * cos_phi + input_imag * sin_phi
        phase_forcing = input_real * sin_phi - input_imag * cos_phi

        if beta2 == 0.0:
            quintic_term = tf.zeros_like(r_t)
        else:
            denominator = 1.0 - epsilon * tf.square(r_t)
            safe_denominator = tf.where(
                tf.abs(denominator) < radius_epsilon,
                tf.where(denominator >= 0.0, radius_epsilon, -radius_epsilon),
                denominator,
            )
            quintic_term = epsilon * beta2 * tf.pow(r_t, 5) / safe_denominator
        r_dot = mu * r_t + beta * tf.pow(r_t, 3) + quintic_term + radial_forcing
        phi_dot = omegas - phase_forcing / safe_r
        r_t = tf.maximum(r_t + r_dot * dt, radius_epsilon)
        phi_t = phi_t + phi_dot * dt
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
        beta: float = -100.0,
        beta2: float = 0.0,
        epsilon: float = 1.0,
        input_scale: float = 0.1,
        radius_epsilon: float = 1e-6,
        trainable_omegas: bool = False,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.units = units
        self.num_steps = num_steps
        self.dt = dt
        self.mu = mu
        self.beta = beta
        self.beta2 = beta2
        self.epsilon = epsilon
        self.input_scale = input_scale
        self.radius_epsilon = radius_epsilon
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
            omegas=tf.convert_to_tensor(self.omegas),
            num_steps=self.num_steps,
            dt=self.dt,
            mu=self.mu,
            beta=self.beta,
            beta2=self.beta2,
            epsilon=self.epsilon,
            input_scale=self.input_scale,
            radius_epsilon=self.radius_epsilon,
        )
        return _real_part(r, phi), _imag_part(r, phi)
