"""Table-based positron source fed by an engine yield table.

The table is the offline exchange artifact of the Geant4 engine track
(M3 route): `engine/positrongen` records the phase space of positrons
exiting a tungsten converter into a CSV whose contract is defined in
docs/superpowers/specs/2026-07-07-positron-yield-table-design.md. This
source replays those records as macro-particles; there is no runtime
coupling to the engine.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from pydantic import field_validator

from latent_dirac.core.species import positron
from latent_dirac.core.units import momentum_gev_c_to_si
from latent_dirac.sources.antiproton_table import _parse_table
from latent_dirac.sources.base import SourceTerm, get_rng, particle_arrays, validate_positive
from latent_dirac.state.particle_state import ParticleState


class PositronYieldTableSource(SourceTerm):
    """Replay an engine-produced positron phase-space table.

    Weight model: the table holds every positron produced by
    ``n_primaries`` simulated electrons (header key). Each row therefore
    represents ``primary_electron_count / n_primaries`` physical
    positrons. Optional ``macro_particles`` bootstrap-resamples rows
    with replacement and renormalizes weights so the represented total
    is unchanged. Fidelity tier: table_based.
    """

    table_path: str
    primary_electron_count: float
    macro_particles: int | None = None

    @field_validator("primary_electron_count")
    @classmethod
    def _positive(cls, value, info):
        return validate_positive(info.field_name, value)

    @field_validator("macro_particles")
    @classmethod
    def _positive_when_set(cls, value, info):
        if value is None:
            return value
        return validate_positive(info.field_name, value)

    def sample(self, rng: np.random.Generator | None = None) -> ParticleState:
        rng = get_rng(rng)
        path = Path(self.table_path)
        header, records = _parse_table(path)

        if "n_primaries" not in header:
            raise ValueError(f"yield table {path} is missing the required '# n_primaries = <int>' header")
        raw_primaries = header["n_primaries"]
        try:
            primaries_value = float(raw_primaries)
        except ValueError as exc:
            raise ValueError(
                f"yield table {path} header n_primaries is not a number: {raw_primaries!r}"
            ) from exc
        if not primaries_value.is_integer() or primaries_value <= 0:
            raise ValueError(
                f"yield table {path} header n_primaries must be a positive integer, got {raw_primaries!r}"
            )
        n_primaries = int(primaries_value)

        if header.get("complete") != "true":
            raise ValueError(
                f"yield table {path} is missing the trailing '# complete = true' marker; "
                "the generating run may have been interrupted (weights would be biased)"
            )

        n_rows = records.shape[0]
        total_yield = self.primary_electron_count * n_rows / n_primaries

        if self.macro_particles is None:
            selected = records
        else:
            indices = rng.integers(0, n_rows, size=int(self.macro_particles))
            selected = records[indices]
        count = selected.shape[0]

        provenance = {
            "geant4_version": header.get("geant4_version", "unknown"),
            "physics_list": header.get("physics_list", "unknown"),
            "datasets": header.get("datasets", "unknown"),
            "patches": header.get("patches", "none"),
            "n_primaries": n_primaries,
        }

        return ParticleState(
            species=positron,
            position_m=selected[:, 0:3],
            momentum_kg_m_s=momentum_gev_c_to_si(selected[:, 3:6]),
            time_s=np.zeros(count),
            weight=np.full(count, total_yield / count),
            metadata={
                "source": "PositronYieldTableSource",
                "model_type": "table_based",
                "physics_note": (
                    "Phase-space replay of an offline engine yield table; no runtime engine coupling."
                ),
                "provenance": provenance,
                "table": {"path": str(path), "rows": int(n_rows)},
            },
            **particle_arrays(count),
        )
