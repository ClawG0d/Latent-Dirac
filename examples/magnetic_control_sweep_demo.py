"""Magnetic control sweep demo for charge-sign separation."""

from __future__ import annotations

from collections.abc import Sequence
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from examples.charge_sign_splitter_demo import make_initial_pair
from latent_dirac.fields.uniform import UniformField
from latent_dirac.solvers.relativistic_boris import RelativisticBorisSolver
from latent_dirac.state.particle_cloud import ParticleCloud

DEFAULT_FIELD_VALUES_T = tuple(float(value) for value in np.linspace(0.0, 0.6, 7))


def _validate_inputs(
    field_values_t: Sequence[float],
    particle_count: int,
    aperture_radius_m: float,
    dt_s: float,
    steps: int,
) -> tuple[float, ...]:
    field_values = tuple(float(value) for value in field_values_t)
    if not field_values:
        raise ValueError("field_values_t must not be empty")
    if particle_count <= 0:
        raise ValueError("particle_count must be positive")
    if aperture_radius_m <= 0.0:
        raise ValueError("aperture_radius_m must be positive")
    if dt_s <= 0.0:
        raise ValueError("dt_s must be positive")
    if steps <= 0:
        raise ValueError("steps must be positive")
    return field_values


def _accepted_x_mask(cloud: ParticleCloud, aperture_radius_m: float) -> np.ndarray:
    return np.abs(cloud.position_m[:, 0]) <= aperture_radius_m


def _field_result(
    by_tesla: float,
    positron_cloud: ParticleCloud,
    electron_cloud: ParticleCloud,
    aperture_radius_m: float,
    solver: RelativisticBorisSolver,
) -> dict[str, float]:
    field = UniformField(B_vector_t=np.array([0.0, by_tesla, 0.0]))
    positron_out = solver.propagate(positron_cloud, field)
    electron_out = solver.propagate(electron_cloud, field)

    positron_mean_x = float(np.mean(positron_out.position_m[:, 0]))
    electron_mean_x = float(np.mean(electron_out.position_m[:, 0]))
    positron_accepted = _accepted_x_mask(positron_out, aperture_radius_m)
    electron_accepted = _accepted_x_mask(electron_out, aperture_radius_m)

    accepted_particles = int(np.count_nonzero(positron_accepted) + np.count_nonzero(electron_accepted))
    total_particles = positron_out.position_m.shape[0] + electron_out.position_m.shape[0]
    accepted_fraction = accepted_particles / total_particles
    loss_fraction = 1.0 - accepted_fraction

    return {
        "field_by_tesla": float(by_tesla),
        "positron_mean_x_m": positron_mean_x,
        "electron_mean_x_m": electron_mean_x,
        "mean_separation_m": float(abs(electron_mean_x - positron_mean_x)),
        "accepted_fraction": float(accepted_fraction),
        "loss_fraction": float(loss_fraction),
        "accepted_particles": float(accepted_particles),
        "lost_particles": float(total_particles - accepted_particles),
    }


def run_sweep(
    field_values_t: Sequence[float] = DEFAULT_FIELD_VALUES_T,
    particle_count: int = 96,
    aperture_radius_m: float = 0.035,
    dt_s: float = 2.0e-12,
    steps: int = 80,
    seed: int = 2031,
) -> list[dict[str, float]]:
    """Scan transverse magnetic field strength and return acceptance diagnostics."""

    field_values = _validate_inputs(field_values_t, particle_count, aperture_radius_m, dt_s, steps)
    positron_cloud, electron_cloud = make_initial_pair(particle_count=particle_count, seed=seed)
    solver = RelativisticBorisSolver(dt_s=dt_s, steps=steps)

    return [
        _field_result(by_tesla, positron_cloud, electron_cloud, aperture_radius_m, solver)
        for by_tesla in field_values
    ]


def format_report(
    results: list[dict[str, float]],
    *,
    aperture_radius_m: float,
    particle_count: int,
    dt_s: float,
    steps: int,
) -> str:
    """Format magnetic control sweep diagnostics as a compact text report."""

    if not results:
        raise ValueError("results must not be empty")

    lines = [
        "Magnetic control sweep demo",
        "",
        "Shared setup:",
        f"- macro-particles per species: {particle_count}",
        "- source state: matched positron/electron clouds",
        "- transport model: relativistic Boris solver",
        f"- solver step: dt={dt_s:.3g} s, steps={steps}",
        "",
        "Magnetic field status:",
        "- field model: uniform transverse field",
        "- B vector [T]: [0, By, 0]",
        f"- sweep range: {results[0]['field_by_tesla']:.3g} T to {results[-1]['field_by_tesla']:.3g} T",
        "",
        "Aperture status:",
        f"- transverse x acceptance: abs(x) <= {aperture_radius_m:.3g} m",
        "- accepted and lost fractions are diagnostics for this fixed window",
        "",
        "Sweep table:",
        "By [T] | positron mean x [m] | electron mean x [m] | separation [m] | accepted fraction | loss fraction",
    ]
    for row in results:
        lines.append(
            " | ".join(
                [
                    f"{row['field_by_tesla']:.3f}",
                    f"{row['positron_mean_x_m']:.6g}",
                    f"{row['electron_mean_x_m']:.6g}",
                    f"{row['mean_separation_m']:.6g}",
                    f"{row['accepted_fraction']:.3f}",
                    f"{row['loss_fraction']:.3f}",
                ]
            )
        )
    lines.extend(
        [
            "",
            "Scope note:",
            "- this is a magnetic transport and aperture diagnostic only",
        ]
    )
    return "\n".join(lines)


def run_report(
    field_values_t: Sequence[float] = DEFAULT_FIELD_VALUES_T,
    particle_count: int = 96,
    aperture_radius_m: float = 0.035,
    dt_s: float = 2.0e-12,
    steps: int = 80,
) -> str:
    results = run_sweep(
        field_values_t=field_values_t,
        particle_count=particle_count,
        aperture_radius_m=aperture_radius_m,
        dt_s=dt_s,
        steps=steps,
    )
    return format_report(
        results,
        aperture_radius_m=aperture_radius_m,
        particle_count=particle_count,
        dt_s=dt_s,
        steps=steps,
    )


if __name__ == "__main__":
    print(run_report())
