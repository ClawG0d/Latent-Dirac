"""Geant4 Matter adapter: the Matter component of the solver layer.

A `ParticleState` cloud passes through a slab of real material;
vanilla Geant4 (`engine/transformer`, FTFP_BERT) computes energy loss,
multiple scattering, nuclear interaction, and antiproton annihilation.
Surviving primaries come back with engine phase space; absorbed ones
come back dead and the `Stage` wrapper stamps the loss ledger.

Exchange is subprocess + files following the phase-space contract in
docs/superpowers/specs/2026-07-05-geant4-matter-adapter-design.md.
Gradients stop at this boundary (transformer form); results carry the
engine provenance four-tuple in the state metadata, which
`scene_report` prints. This module never imports engine code — it is
importable and constructible with no Geant4 build present.
"""

from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path, PureWindowsPath

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, field_validator

from latent_dirac.core.units import momentum_gev_c_to_si, momentum_si_to_gev_c
from latent_dirac.pipeline.stage import Stage
from latent_dirac.state.particle_state import ParticleState

#: latent-dirac species name -> Geant4 particle name
GEANT4_SPECIES_NAMES = {
    "electron": "e-",
    "positron": "e+",
    "proton": "proton",
    "antiproton": "anti_proton",
}

_COLUMNS = "id,x_m,y_m,z_m,px_gev_c,py_gev_c,pz_gev_c"


def windows_to_wsl_path(path: Path) -> str:
    """C:\\dir\\file -> /mnt/c/dir/file (for a WSL-hosted engine build)."""

    pure = PureWindowsPath(path)
    drive = pure.drive.rstrip(":").lower()
    return f"/mnt/{drive}/" + "/".join(pure.parts[1:])


def _write_phase_space(path: Path, species_name: str, ids, positions, momenta_gev_c) -> None:
    lines = [
        "# latent-dirac phase space v1",
        f"# species = {species_name}",
        f"# n_rows = {len(ids)}",
        f"# columns = {_COLUMNS}",
    ]
    for row_id, pos, mom in zip(ids, positions, momenta_gev_c, strict=True):
        lines.append(
            f"{int(row_id)},{pos[0]:.9g},{pos[1]:.9g},{pos[2]:.9g},"
            f"{mom[0]:.9g},{mom[1]:.9g},{mom[2]:.9g}"
        )
    lines.append("# complete = true")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _parse_phase_space(path: Path) -> tuple[dict[str, str], np.ndarray]:
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
        if len(parts) != 7:
            raise ValueError(f"phase-space row {line_no} has {len(parts)} columns, expected 7")
        try:
            rows.append([float(part) for part in parts])
        except ValueError as exc:
            raise ValueError(f"phase-space row {line_no} is not numeric: {line!r}") from exc

    if header.get("complete") != "true":
        raise ValueError(
            f"phase-space file {path} is missing the trailing '# complete = true' marker; "
            "the engine run may have been interrupted"
        )
    data = np.asarray(rows, dtype=float) if rows else np.empty((0, 7))
    return header, data


class Geant4MatterAdapter(BaseModel):
    """Track a cloud through a material slab via the engine transformer.

    `command` is the transformer invocation prefix — the native binary,
    or a bridge such as WSL. The slab front sits at `entry_z_m` in
    pipeline coordinates; survivors return at the engine's downstream
    scoring plane, ~1 mm past the slab exit (the exact z is field-free
    drift, not load-bearing).
    """

    model_config = ConfigDict(extra="forbid")

    command: tuple[str, ...] = Field(min_length=1)
    material: str
    thickness_mm: float = Field(gt=0)
    entry_z_m: float = 0.0
    workdir: str | None = None
    path_style: str = "native"  # "native" | "wsl"
    # Must match the engine/transformer build: the slab and scoring plane
    # span |x|,|y| <= transverse_half_width_m, inside a vacuum world of
    # half-length world_half_length_m. Particles outside the aperture would
    # miss the slab and be silently booked as material losses; a vertex
    # outside the world aborts the whole engine run. We fail fast instead.
    transverse_half_width_m: float = 0.20
    world_half_length_m: float = 0.60

    @field_validator("path_style")
    @classmethod
    def _known_style(cls, value):
        if value not in ("native", "wsl"):
            raise ValueError("path_style must be 'native' or 'wsl'")
        return value

    def _render(self, path: Path) -> str:
        return windows_to_wsl_path(path) if self.path_style == "wsl" else str(path)

    def stage(self, name: str) -> Stage:
        return Stage(name, self.apply)

    def apply(self, state: ParticleState) -> ParticleState:
        species_name = GEANT4_SPECIES_NAMES.get(state.species.name)
        if species_name is None:
            raise ValueError(
                f"species {state.species.name!r} is not supported by the Geant4 matter "
                f"adapter (supported: {sorted(GEANT4_SPECIES_NAMES)})"
            )

        result = state.copy()
        sent = np.flatnonzero(state.alive)
        if sent.size == 0:
            return result

        positions = state.position_m[sent].copy()
        positions[:, 2] -= self.entry_z_m  # contract frame: slab front at z = 0
        momenta = momentum_si_to_gev_c(state.momentum_kg_m_s[sent])

        # Fail fast on particles the engine geometry cannot represent:
        # outside the transverse aperture they would miss the slab and be
        # miscounted as material losses; outside the world they abort the run.
        transverse = np.max(np.abs(positions[:, :2]), axis=1)
        if np.any(transverse > self.transverse_half_width_m):
            raise ValueError(
                f"{int(np.sum(transverse > self.transverse_half_width_m))} particle(s) exceed the "
                f"engine transverse aperture (|x|,|y| <= {self.transverse_half_width_m} m); they "
                "would miss the slab and be miscounted as losses"
            )
        if np.any(np.abs(positions[:, 2]) > self.world_half_length_m):
            raise ValueError(
                "particle vertex lies outside the engine world "
                f"(|z - entry_z_m| <= {self.world_half_length_m} m); the engine run would abort"
            )

        with tempfile.TemporaryDirectory(dir=self.workdir) as exchange:
            in_path = Path(exchange) / "in.csv"
            out_path = Path(exchange) / "out.csv"
            _write_phase_space(in_path, species_name, sent, positions, momenta)

            argv = [
                *self.command,
                self._render(in_path),
                self._render(out_path),
                self.material,
                f"{self.thickness_mm:g}",
            ]
            # bytes + replace: a failing WSL bridge emits UTF-16LE that
            # would otherwise kill the reader thread and null out the streams.
            run = subprocess.run(argv, capture_output=True)
            if run.returncode != 0:
                stderr = (run.stderr or b"").decode("utf-8", "replace").strip()
                stdout = (run.stdout or b"").decode("utf-8", "replace").strip()
                raise RuntimeError(
                    f"engine transformer failed (exit {run.returncode}): "
                    f"{stderr or stdout or '<no output>'}"
                )
            header, rows = _parse_phase_space(out_path)

        survivor_ids = rows[:, 0].astype(int) if rows.size else np.empty(0, dtype=int)
        unknown = np.setdiff1d(survivor_ids, sent)
        if unknown.size:
            raise ValueError(f"engine returned unknown particle id(s): {unknown.tolist()}")
        if np.unique(survivor_ids).size != survivor_ids.size:
            raise ValueError("engine returned duplicate particle ids")

        absorbed = np.setdiff1d(sent, survivor_ids)
        result.alive[absorbed] = False
        if survivor_ids.size:
            new_positions = rows[:, 1:4].copy()
            new_positions[:, 2] += self.entry_z_m
            result.position_m[survivor_ids] = new_positions
            result.momentum_kg_m_s[survivor_ids] = momentum_gev_c_to_si(rows[:, 4:7])

        engine_provenance = {
            "geant4_version": header.get("geant4_version", "unknown"),
            "physics_list": header.get("physics_list", "unknown"),
            "datasets": header.get("datasets", "unknown"),
            "patches": header.get("patches", "none"),
            "n_primaries": int(header.get("n_primaries", sent.size)),
        }
        result.metadata = dict(result.metadata)
        # A transform, not a source: never clobber the source's model_type or
        # provenance (e.g. an engine yield-table source upstream). The matter
        # engine's own provenance lives under the `matter` key; top-level keys
        # are only filled when the cloud has none yet.
        result.metadata.setdefault("model_type", "engine_transformer")
        result.metadata.setdefault("provenance", engine_provenance)
        result.metadata.update(
            {
                "matter": {
                    "material": self.material,
                    "thickness_mm": self.thickness_mm,
                    "entry_z_m": self.entry_z_m,
                    "provenance": engine_provenance,
                    "engine": "geant4-transformer",
                },
            }
        )
        return result
