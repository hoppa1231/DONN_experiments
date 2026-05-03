"""Case study 2 helpers: STDP-like kernel from coupled Hopf oscillators."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class STDPConfig:
    frequency_hz: float = 5.0
    mu: float = 1.0
    eta: float = 0.2
    pulse_amplitude: float = 20.0
    pulse_sigma: float = 0.006
    dt: float = 5e-4
    total_time: float = 0.9
    pulse_time: float = 0.35
    tau_min: float = -10.0
    tau_max: float = 10.0
    tau_units_per_period: float = 20.0
    num_delays: int = 161
    hebbian_product: str = "paper-conjugate"


@dataclass
class STDPSweep:
    tau: list[float]
    delay_seconds: list[float]
    real_weight: list[float]
    imag_weight: list[float]
    real_peak_to_peak: float
    imag_peak_to_peak: float
    imag_positive_delay_mean: float
    imag_negative_delay_mean: float
    config: dict[str, float | int | str]


def _pulse(t: float, center: float, amplitude: float, sigma: float) -> float:
    return float(amplitude * np.exp(-0.5 * ((t - center) / sigma) ** 2))


def simulate_coupled_hopf(delay: float, config: STDPConfig) -> complex:
    """Simulate equations (6)-(8) from the article for one pulse delay.

    The article text uses conjugate coupling:
    ``zdot2 = ... + conjugate(W) * z1`` and
    ``Wdot = -W + eta * z1 * conjugate(z2)``.  The literal ``z1*z2`` path is
    kept as a diagnostic because an earlier screenshot was easy to misread.
    """
    product = config.hebbian_product
    if product == "conjugate-control":
        product = "paper-conjugate"
    if product == "literal":
        product = "literal-control"
    if product not in {"paper-conjugate", "literal-control"}:
        raise ValueError("hebbian_product must be 'paper-conjugate' or 'literal-control'")

    omega = 2.0 * np.pi * config.frequency_hz
    steps = int(round(config.total_time / config.dt))
    z1 = 0.0 + 0.0j
    z2 = 0.0 + 0.0j
    weight = 0.0 + 0.0j

    for step in range(steps):
        t = step * config.dt
        p1 = _pulse(t, config.pulse_time, config.pulse_amplitude, config.pulse_sigma)
        p2 = _pulse(t, config.pulse_time + delay, config.pulse_amplitude, config.pulse_sigma)

        dz1 = (config.mu + 1j * omega) * z1 - (abs(z1) ** 2) * z1 + weight * z2 + p1
        coupling_21 = np.conj(weight) * z1 if product == "paper-conjugate" else weight * z1
        dz2 = (config.mu + 1j * omega) * z2 - (abs(z2) ** 2) * z2 + coupling_21 + p2
        z1 += dz1 * config.dt
        z2 += dz2 * config.dt

        if product == "paper-conjugate":
            hebbian_term = z1 * np.conj(z2)
        else:
            hebbian_term = z1 * z2
        weight += (-weight + config.eta * hebbian_term) * config.dt

    return weight


def run_stdp_sweep(config: STDPConfig) -> STDPSweep:
    period = 1.0 / config.frequency_hz
    tau = np.linspace(config.tau_min, config.tau_max, config.num_delays)
    delay_seconds = tau * (period / config.tau_units_per_period)
    weights = np.array([simulate_coupled_hopf(float(delay), config) for delay in delay_seconds])
    negative_mask = tau < 0
    positive_mask = tau > 0

    return STDPSweep(
        tau=[float(v) for v in tau],
        delay_seconds=[float(v) for v in delay_seconds],
        real_weight=[float(v) for v in weights.real],
        imag_weight=[float(v) for v in weights.imag],
        real_peak_to_peak=float(np.ptp(weights.real)),
        imag_peak_to_peak=float(np.ptp(weights.imag)),
        imag_positive_delay_mean=float(np.mean(weights.imag[positive_mask])),
        imag_negative_delay_mean=float(np.mean(weights.imag[negative_mask])),
        config={
            "frequency_hz": config.frequency_hz,
            "period": period,
            "mu": config.mu,
            "eta": config.eta,
            "pulse_amplitude": config.pulse_amplitude,
            "pulse_sigma": config.pulse_sigma,
            "dt": config.dt,
            "total_time": config.total_time,
            "pulse_time": config.pulse_time,
            "tau_min": config.tau_min,
            "tau_max": config.tau_max,
            "tau_units_per_period": config.tau_units_per_period,
            "num_delays": config.num_delays,
            "hebbian_product": config.hebbian_product,
        },
    )
