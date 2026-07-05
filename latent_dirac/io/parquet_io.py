"""Placeholder Parquet I/O hooks.

The core package currently avoids mandatory Parquet dependencies. These
functions define the intended integration points for a future optional extra.
"""

from __future__ import annotations

from pathlib import Path

from latent_dirac.state.particle_state import ParticleState


def write_particle_cloud_parquet(cloud: ParticleState, path: str | Path) -> None:
    raise NotImplementedError("Parquet I/O is a placeholder for a future optional adapter.")


def read_particle_cloud_parquet(path: str | Path) -> ParticleState:
    raise NotImplementedError("Parquet I/O is a placeholder for a future optional adapter.")
