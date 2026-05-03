"""Visual report for Case study 2: STDP-like kernel in coupled Hopf oscillators."""

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

from src.stdp_kernel import STDPConfig, run_stdp_sweep


def plot_report(out_path: Path, result: dict[str, object]) -> None:
    tau = np.array(result["tau"], dtype=np.float64)
    real_weight = np.array(result["real_weight"], dtype=np.float64)
    imag_weight = np.array(result["imag_weight"], dtype=np.float64)
    config = result["config"]

    fig, axes = plt.subplots(2, 1, figsize=(7, 8), constrained_layout=True)
    axes[0].plot(tau, real_weight, color="black", lw=2.0)
    axes[0].axvline(0.0, color="0.2", lw=1.0, ls="--")
    axes[0].set_xlim(float(config["tau_min"]), float(config["tau_max"]))
    axes[0].set_ylabel("Re(W)")
    axes[0].grid(alpha=0.25)

    axes[1].plot(tau, imag_weight, color="black", lw=2.0)
    axes[1].axvline(0.0, color="0.2", lw=1.0, ls="--")
    axes[1].set_xlim(float(config["tau_min"]), float(config["tau_max"]))
    axes[1].set_xlabel("tau")
    axes[1].set_ylabel("Im(W)")
    axes[1].grid(alpha=0.25)

    fig.suptitle(
        "Case study 2 equation-level control | "
        f"f={float(config['frequency_hz']):.1f} Hz, product={config['hebbian_product']}, "
        f"Im peak-to-peak={float(result['imag_peak_to_peak']):.4g}",
        fontsize=12,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=170, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--frequency-hz", type=float, default=5.0)
    parser.add_argument("--mu", type=float, default=1.0)
    parser.add_argument("--eta", type=float, default=0.2)
    parser.add_argument("--pulse-amplitude", type=float, default=20.0)
    parser.add_argument("--pulse-sigma", type=float, default=0.006)
    parser.add_argument("--dt", type=float, default=5e-4)
    parser.add_argument("--total-time", type=float, default=0.9)
    parser.add_argument("--pulse-time", type=float, default=0.35)
    parser.add_argument("--tau-min", type=float, default=-10.0)
    parser.add_argument("--tau-max", type=float, default=10.0)
    parser.add_argument(
        "--tau-units-per-period",
        type=float,
        default=20.0,
        help="Map article-style tau units to seconds; 20 units equals one oscillator period by default.",
    )
    parser.add_argument("--num-delays", type=int, default=161)
    parser.add_argument(
        "--hebbian-product",
        choices=["paper-conjugate", "literal-control", "literal", "conjugate-control"],
        default="paper-conjugate",
        help="Use the paper conjugate rule or a literal z1*z2 diagnostic.",
    )
    parser.add_argument(
        "--out-path",
        type=Path,
        default=Path("artifacts/plots/case_study/case_study_stdp_kernel_summary.png"),
    )
    parser.add_argument(
        "--metrics-path",
        type=Path,
        default=Path("artifacts/plots/case_study/case_study_stdp_kernel_metrics.json"),
    )
    args = parser.parse_args()

    sweep = run_stdp_sweep(
        STDPConfig(
            frequency_hz=args.frequency_hz,
            mu=args.mu,
            eta=args.eta,
            pulse_amplitude=args.pulse_amplitude,
            pulse_sigma=args.pulse_sigma,
            dt=args.dt,
            total_time=args.total_time,
            pulse_time=args.pulse_time,
            tau_min=args.tau_min,
            tau_max=args.tau_max,
            tau_units_per_period=args.tau_units_per_period,
            num_delays=args.num_delays,
            hebbian_product=args.hebbian_product,
        )
    )

    used_w_update = (
        "Wdot = -W + eta z1 z2"
        if args.hebbian_product in {"literal", "literal-control"}
        else "Wdot = -W + eta z1 conj(z2)"
    )
    used_z2_coupling = (
        "conj(W) z1"
        if args.hebbian_product in {"paper-conjugate", "conjugate-control"}
        else "W z1"
    )
    result = {
        "variant": "stdp_kernel_equation_level_control",
        "is_exact_figure7_reproduction": False,
        "reason_not_exact": "The article gives equations but does not specify pulse shape or numerical parameters.",
        "paper_equations": {
            "z1": "zdot1 = (mu + i omega1) z1 - |z1|^2 z1 + W z2 + p(t)",
            "z2": "zdot2 = (mu + i omega2) z2 - |z2|^2 z2 + conjugate(W) z1 + p(t + tau)",
            "W": "Wdot = -W + eta z1 conjugate(z2)",
        },
        "used_W_update": used_w_update,
        "used_z2_coupling": used_z2_coupling,
        **asdict(sweep),
    }

    plot_report(args.out_path, result)
    args.metrics_path.parent.mkdir(parents=True, exist_ok=True)
    args.metrics_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    print(f"Saved figure: {args.out_path}")
    print(f"Saved metrics: {args.metrics_path}")


if __name__ == "__main__":
    main()
