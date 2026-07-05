"""Magnetic mirror bottle demo, driven by a table-based field map.

An analytic magnetic-mirror field (B_z = B0 * (1 + (z/L)^2) on axis, with
the divergence-free transverse component B_r = -B0 * r * z / L^2) is
sampled onto a regular grid, written as a COMSOL-style CSV, and loaded back
through the real `load_comsol_grid_csv` import pipeline. Particles inside
the trapping cone bounce between the throats.

Fidelity: table-based field map generated from an analytic expression —
synthetic, not a real FEM export, and labeled as such.
"""

from __future__ import annotations

import sys
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from latent_dirac.core.species import positron
from latent_dirac.fields.field_map import FieldMapField, load_comsol_grid_csv
from latent_dirac.solvers.kernels import (
    boris_step,
    dimensionless_to_momentum,
    momentum_to_dimensionless,
)

B0_TESLA = 1.0
MIRROR_HALF_LENGTH_M = 0.05
TRANSVERSE_EXTENT_M = 0.008
KINETIC_ENERGY_EV = 2000.0
FIELD_MODEL_LABEL = "table-based field map (synthetic analytic mirror)"


def mirror_field(points: np.ndarray) -> np.ndarray:
    x, y, z = points[:, 0], points[:, 1], points[:, 2]
    scale = B0_TESLA / MIRROR_HALF_LENGTH_M**2
    return np.column_stack(
        [
            -scale * x * z,
            -scale * y * z,
            B0_TESLA * (1.0 + (z / MIRROR_HALF_LENGTH_M) ** 2),
        ]
    )


def write_mirror_field_csv(path: str | Path) -> Path:
    """Sample the analytic mirror field onto a grid and write a COMSOL-style CSV."""

    path = Path(path)
    x = np.linspace(-TRANSVERSE_EXTENT_M, TRANSVERSE_EXTENT_M, 15)
    z = np.linspace(-MIRROR_HALF_LENGTH_M, MIRROR_HALF_LENGTH_M, 61)
    grid = np.stack(np.meshgrid(x, x, z, indexing="ij"), axis=-1).reshape(-1, 3)
    b_values = mirror_field(grid)

    lines = [
        "% Latent Dirac synthetic magnetic mirror (analytic sample, SI units)",
        "% x, y, z, Bx, By, Bz",
    ]
    for point, b in zip(grid, b_values, strict=True):
        lines.append(f"{point[0]:.9g},{point[1]:.9g},{point[2]:.9g},{b[0]:.9g},{b[1]:.9g},{b[2]:.9g}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def load_mirror_field(directory: str | Path) -> FieldMapField:
    csv_path = write_mirror_field_csv(Path(directory) / "mirror_field.csv")
    return load_comsol_grid_csv(csv_path)


def sample_particles(count: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Positrons at the bottle center with isotropic pitch angles."""

    kinetic_j = KINETIC_ENERGY_EV * 1.602176634e-19
    momentum = np.sqrt(2.0 * positron.mass_kg * kinetic_j)  # non-relativistic at keV

    cos_theta = rng.uniform(0.0, 1.0, size=count)
    sin_theta = np.sqrt(1.0 - cos_theta**2)
    phi = rng.uniform(0.0, 2.0 * np.pi, size=count)

    momenta = momentum * np.column_stack([sin_theta * np.cos(phi), sin_theta * np.sin(phi), cos_theta])
    positions = np.zeros((count, 3))
    positions[:, 0] = rng.normal(0.0, 2.0e-4, size=count)
    positions[:, 1] = rng.normal(0.0, 2.0e-4, size=count)
    return positions, momenta


def run_trajectories(
    count: int = 24,
    steps: int = 6000,
    dt_s: float = 2.0e-12,
    record_every: int = 30,
    seed: int = 2026,
) -> dict:
    """Propagate positrons in the mirror and record sampled positions."""

    rng = np.random.default_rng(seed)
    with TemporaryDirectory() as tmp:
        field = load_mirror_field(tmp)

    positions, momenta = sample_particles(count, rng)
    u = momentum_to_dimensionless(momenta, positron.mass_kg)
    time_s = np.zeros(count)
    alive = np.ones(count, dtype=bool)

    history = [positions.copy()]
    vz_sign_changes = np.zeros(count, dtype=int)
    previous_vz_sign = np.sign(u[:, 2])

    for step in range(steps):
        e_field = field.E(positions, time_s)
        b_field = field.B(positions, time_s)
        positions, u, time_s = boris_step(
            positions,
            u,
            time_s,
            alive,
            dt_s=dt_s,
            charge_c=positron.charge_c,
            mass_kg=positron.mass_kg,
            e_field=e_field,
            b_field=b_field,
        )
        current_sign = np.sign(u[:, 2])
        vz_sign_changes += (current_sign != previous_vz_sign) & (current_sign != 0)
        previous_vz_sign = current_sign
        if (step + 1) % record_every == 0:
            history.append(positions.copy())

    trajectory = np.stack(history)
    final_momenta = dimensionless_to_momentum(u, positron.mass_kg)
    trapped = (np.abs(trajectory[:, :, 2]).max(axis=0) < MIRROR_HALF_LENGTH_M) & (vz_sign_changes >= 1)
    return {
        "trajectory": trajectory,
        "trapped": trapped,
        "final_momenta": final_momenta,
        "vz_sign_changes": vz_sign_changes,
    }


def run_demo(count: int = 24, steps: int = 6000) -> dict:
    outcome = run_trajectories(count=count, steps=steps)
    trajectory = outcome["trajectory"]
    trapped = outcome["trapped"]
    trapped_z = np.abs(trajectory[:, trapped, 2]) if np.any(trapped) else np.zeros((1, 1))
    return {
        "trapped_fraction": float(np.mean(trapped)),
        "max_abs_z_m": float(trapped_z.max()),
        "mirror_half_length_m": MIRROR_HALF_LENGTH_M,
        "mirror_ratio": 2.0,
        "field_model": FIELD_MODEL_LABEL,
    }


def run_report() -> str:
    summary = run_demo()
    return "\n".join(
        [
            "Magnetic mirror bottle demo",
            "",
            "Magnetic field status:",
            f"- field model: {summary['field_model']}",
            f"- on-axis B: {B0_TESLA:g} T at center, mirror ratio {summary['mirror_ratio']:g}",
            f"- bottle half-length: {summary['mirror_half_length_m']:g} m",
            "",
            "Trapping:",
            f"- trapped fraction: {summary['trapped_fraction']:.3f}",
            f"- max |z| reached: {summary['max_abs_z_m']:.4g} m",
            "",
            "Scope note:",
            "- single-particle transport in a static synthetic field map;",
            "  no collisions, no space charge, no trap operations",
        ]
    )


if __name__ == "__main__":
    print(run_report())
