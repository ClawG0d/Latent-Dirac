"""Simple spectrum summaries."""

from __future__ import annotations

import numpy as np

from latent_dirac.core.units import joule_to_ev
from latent_dirac.state.particle_state import ParticleState


def energy_spectrum_summary(cloud: ParticleState) -> dict[str, float]:
    live = cloud.alive
    if not np.any(live):
        return {
            "count": 0.0,
            "mean_energy_MeV": 0.0,
            "min_energy_MeV": 0.0,
            "max_energy_MeV": 0.0,
        }

    energies_mev = joule_to_ev(cloud.kinetic_energy_joule()[live]) / 1.0e6
    weights = cloud.weight[live]
    return {
        "count": float(np.sum(weights)),
        "mean_energy_MeV": float(np.average(energies_mev, weights=weights)),
        "min_energy_MeV": float(np.min(energies_mev)),
        "max_energy_MeV": float(np.max(energies_mev)),
    }
