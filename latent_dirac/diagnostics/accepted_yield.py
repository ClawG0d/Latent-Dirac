"""Accepted-yield diagnostics."""

from __future__ import annotations

from latent_dirac.state.particle_cloud import ParticleCloud


def accepted_yield(accepted_weighted_count: float, primary_count: float) -> float:
    if primary_count <= 0.0:
        raise ValueError("primary_count must be positive")
    return float(accepted_weighted_count / primary_count)


def accepted_yield_from_cloud(cloud: ParticleCloud, primary_count: float) -> float:
    return accepted_yield(cloud.weighted_count(), primary_count)
