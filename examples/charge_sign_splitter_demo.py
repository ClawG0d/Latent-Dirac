"""Charge-sign splitter demo.

This demo uses the same initial phase-space distribution for positrons and
electrons, then transports both clouds through the same uniform transverse
magnetic field. Because the particles have equal mass and opposite charge,
the Lorentz-force curvature separates the two clouds in opposite transverse
directions.
"""

from __future__ import annotations

import numpy as np

from latent_dirac.core.species import electron, positron
from latent_dirac.core.units import momentum_gev_c_to_si
from latent_dirac.fields.uniform import UniformField
from latent_dirac.solvers.relativistic_boris import RelativisticBorisSolver
from latent_dirac.state.particle_cloud import ParticleCloud


def _make_cloud(species, position_m: np.ndarray, momentum_kg_m_s: np.ndarray) -> ParticleCloud:
    count = position_m.shape[0]
    return ParticleCloud(
        species=species,
        position_m=position_m,
        momentum_kg_m_s=momentum_kg_m_s,
        time_s=np.zeros(count),
        weight=np.ones(count),
        alive=np.ones(count, dtype=bool),
        particle_id=np.arange(count),
        parent_id=np.full(count, -1),
        metadata={
            "source": "charge_sign_splitter_demo",
            "model_type": "diagnostic",
            "physics_note": "Equal-mass opposite-charge transport in a shared magnetic field.",
        },
    )


def make_initial_pair(particle_count: int = 96, seed: int = 2030) -> tuple[ParticleCloud, ParticleCloud]:
    """Create matched positron/electron clouds with identical initial states."""

    if particle_count <= 0:
        raise ValueError("particle_count must be positive")

    rng = np.random.default_rng(seed)
    position_m = np.column_stack(
        [
            rng.normal(0.0, 1.2e-3, size=particle_count),
            rng.normal(0.0, 8.0e-4, size=particle_count),
            rng.normal(0.0, 2.0e-4, size=particle_count),
        ]
    )
    pz = momentum_gev_c_to_si(0.004)
    momentum_kg_m_s = np.column_stack(
        [
            rng.normal(0.0, 0.025 * pz, size=particle_count),
            rng.normal(0.0, 0.010 * pz, size=particle_count),
            rng.normal(pz, 0.015 * pz, size=particle_count),
        ]
    )

    return (
        _make_cloud(positron, position_m.copy(), momentum_kg_m_s.copy()),
        _make_cloud(electron, position_m.copy(), momentum_kg_m_s.copy()),
    )


def run_demo(
    particle_count: int = 96,
    b_tesla: float = 0.45,
    dt_s: float = 2.0e-12,
    steps: int = 80,
) -> dict[str, float]:
    """Run the charge-sign splitter and return summary diagnostics."""

    positron_cloud, electron_cloud = make_initial_pair(particle_count=particle_count)
    field = UniformField(B_vector_t=np.array([0.0, b_tesla, 0.0]))
    solver = RelativisticBorisSolver(dt_s=dt_s, steps=steps)

    positron_out = solver.propagate(positron_cloud, field)
    electron_out = solver.propagate(electron_cloud, field)
    positron_mean_x = float(np.mean(positron_out.position_m[:, 0]))
    electron_mean_x = float(np.mean(electron_out.position_m[:, 0]))
    positron_mean_z = float(np.mean(positron_out.position_m[:, 2]))
    electron_mean_z = float(np.mean(electron_out.position_m[:, 2]))

    return {
        "particle_count": float(particle_count),
        "field_b_tesla": float(b_tesla),
        "positron_mean_x_m": positron_mean_x,
        "electron_mean_x_m": electron_mean_x,
        "positron_mean_z_m": positron_mean_z,
        "electron_mean_z_m": electron_mean_z,
        "mean_separation_m": float(abs(electron_mean_x - positron_mean_x)),
    }


def run_report(particle_count: int = 96) -> str:
    summary = run_demo(particle_count=particle_count)
    return "\n".join(
        [
            "Charge-sign splitter demo",
            "",
            "Shared setup:",
            f"- macro-particles per species: {int(summary['particle_count'])}",
            f"- transverse magnetic field By: {summary['field_b_tesla']:.3g} T",
            "- transport model: relativistic Boris solver",
            "",
            "Lorentz-force separation:",
            f"- positron mean x: {summary['positron_mean_x_m']:.6g} m",
            f"- electron mean x: {summary['electron_mean_x_m']:.6g} m",
            f"- mean transverse separation: {summary['mean_separation_m']:.6g} m",
            "",
            "Scope note:",
            "- this is a charge-sign transport and acceptance diagnostic only",
        ]
    )


if __name__ == "__main__":
    print(run_report())
