"""Placeholder HDF5 I/O hooks.

The core package currently avoids mandatory HDF5 dependencies. These functions
define the intended integration points for a future optional HDF5 extra.
"""

from __future__ import annotations

from pathlib import Path

from latent_dirac.state.particle_cloud import ParticleCloud


def write_particle_cloud_hdf5(cloud: ParticleCloud, path: str | Path) -> None:
    raise NotImplementedError("HDF5 I/O is a placeholder for a future optional adapter.")


def read_particle_cloud_hdf5(path: str | Path) -> ParticleCloud:
    raise NotImplementedError("HDF5 I/O is a placeholder for a future optional adapter.")
