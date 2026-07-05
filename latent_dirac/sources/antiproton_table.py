"""Table-based antiproton source fed by an engine yield table.

The table is the offline exchange artifact of the Geant4 engine track
(M3 route): `engine/yieldgen` records the phase space of antiprotons
exiting a production target into a CSV whose contract is defined in
docs/superpowers/specs/2026-07-05-engine-yieldgen-demo-design.md. This
source replays those records as macro-particles; there is no runtime
coupling to the engine.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from pydantic import field_validator

from latent_dirac.core.species import antiproton
from latent_dirac.core.units import momentum_gev_c_to_si
from latent_dirac.sources.base import SourceTerm, get_rng, particle_arrays, validate_positive
from latent_dirac.state.particle_state import ParticleState

_COLUMNS = 6  # x_m, y_m, z_m, px_gev_c, py_gev_c, pz_gev_c


def _parse_table(path: Path) -> tuple[dict[str, str], np.ndarray]:
    """Parse header key/value pairs and the (N, 6) data block.

    Comment lines (``# key = value``) may appear anywhere; the generator
    writes the provenance header before the data and the mandatory
    ``# complete = true`` marker after the last row, so a truncated file
    fails validation instead of silently biasing weights.
    """

    header: dict[str, str] = {}
    rows: list[list[float]] = []
    for line_no, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw.strip()
        if not line:
            continue
        if line.startswith("#"):
            body = line.lstrip("#").strip()
            if "=" in body:
                key, _, value = body.partition("=")
                header[key.strip()] = value.strip()
            continue
        parts = line.split(",")
        if len(parts) != _COLUMNS:
            raise ValueError(f"yield table row {line_no} has {len(parts)} columns, expected {_COLUMNS}")
        try:
            rows.append([float(part) for part in parts])
        except ValueError as exc:
            raise ValueError(f"yield table row {line_no} is not numeric: {line!r}") from exc

    if not rows:
        raise ValueError(f"yield table {path} contains no data rows")
    return header, np.asarray(rows, dtype=float)


class AntiprotonYieldTableSource(SourceTerm):
    """Replay an engine-produced antiproton phase-space table.

    Weight model: the table holds every antiproton produced by
    ``n_primaries`` simulated protons (header key). Each row therefore
    represents ``primary_proton_count / n_primaries`` physical
    antiprotons. Optional ``macro_particles`` bootstrap-resamples rows
    with replacement and renormalizes weights so the represented total
    is unchanged.
    """

    table_path: str
    primary_proton_count: float
    macro_particles: int | None = None

    @field_validator("primary_proton_count")
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
                f"yield table {path} header n_primaries must be a positive integer, "
                f"got {raw_primaries!r}"
            )
        n_primaries = int(primaries_value)

        if header.get("complete") != "true":
            raise ValueError(
                f"yield table {path} is missing the trailing '# complete = true' marker; "
                "the generating run may have been interrupted (weights would be biased)"
            )

        n_rows = records.shape[0]
        total_yield = self.primary_proton_count * n_rows / n_primaries

        if self.macro_particles is None:
            selected = records
        else:
            indices = rng.integers(0, n_rows, size=int(self.macro_particles))
            selected = records[indices]
        count = selected.shape[0]

        position_m = selected[:, 0:3]
        momentum_kg_m_s = momentum_gev_c_to_si(selected[:, 3:6])

        provenance = {
            "geant4_version": header.get("geant4_version", "unknown"),
            "physics_list": header.get("physics_list", "unknown"),
            "datasets": header.get("datasets", "unknown"),
            "patches": header.get("patches", "none"),
            "n_primaries": n_primaries,
        }

        return ParticleState(
            species=antiproton,
            position_m=position_m,
            momentum_kg_m_s=momentum_kg_m_s,
            time_s=np.zeros(count),
            weight=np.full(count, total_yield / count),
            metadata={
                "source": "AntiprotonYieldTableSource",
                "model_type": "table_based",
                "physics_note": (
                    "Phase-space replay of an offline engine yield table; "
                    "no runtime engine coupling."
                ),
                "provenance": provenance,
                "table": {"path": str(path), "rows": int(n_rows)},
            },
            **particle_arrays(count),
        )
