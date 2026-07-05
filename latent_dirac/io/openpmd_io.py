"""openPMD particle output: the first real Analysis sink.

Writes `ParticleState` snapshots as openPMD particle species through the
optional `openpmd-api` dependency (`pip install "latent-dirac[openpmd]"`).
All stored values are SI, so every `unitSI` is 1.0 — the State boundary
rule keeps the arrays SI already. Writing only; reading openPMD back and
mesh (field) records are later extensions (see the 2026-07-05 openPMD
output spec).
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from importlib import metadata as importlib_metadata
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from latent_dirac.state.particle_state import ParticleState

if TYPE_CHECKING:  # pragma: no cover - typing only
    from latent_dirac.scene.build import SceneRunResult

FINAL_LABEL = "final"

# metadata["provenance"] keys lifted to flat, greppable species attributes;
# the engine four-tuple must stay visible on engine-backed output.
_PROVENANCE_ATTRIBUTES = {
    "geant4_version": "latentDirac_geant4Version",
    "physics_list": "latentDirac_physicsList",
    "datasets": "latentDirac_datasets",
    "patches": "latentDirac_patches",
}


def _require_api():
    try:
        import openpmd_api as api
    except ImportError as exc:  # pragma: no cover - exercised without extra
        raise ImportError(
            "openPMD output requires the optional [openpmd] extra: "
            'pip install "latent-dirac[openpmd]"'
        ) from exc
    return api


def _package_version() -> str:
    try:
        return importlib_metadata.version("latent-dirac")
    except importlib_metadata.PackageNotFoundError:  # pragma: no cover
        return "unknown"


def _store_vector(api, particles, name: str, values: np.ndarray, buffers: list) -> None:
    for column, axis in enumerate("xyz"):
        component = particles[name][axis]
        data = np.ascontiguousarray(values[:, column], dtype=np.float64)
        component.reset_dataset(api.Dataset(data.dtype, data.shape))
        component.store_chunk(data)
        component.unit_SI = 1.0
        buffers.append(data)


def _store_scalar(api, particles, name: str, values: np.ndarray, buffers: list) -> None:
    component = particles[name][api.Record_Component.SCALAR]
    data = np.ascontiguousarray(values)
    component.reset_dataset(api.Dataset(data.dtype, data.shape))
    component.store_chunk(data)
    component.unit_SI = 1.0
    buffers.append(data)


def _store_constant(api, particles, name: str, value: float, count: int) -> None:
    component = particles[name][api.Record_Component.SCALAR]
    component.reset_dataset(api.Dataset(np.dtype("float64"), (count,)))
    component.make_constant(float(value))
    component.unit_SI = 1.0


def _write_state(api, iteration, state: ParticleState, buffers: list) -> None:
    species = state.species
    count = state.position_m.shape[0]
    if count == 0:
        # openpmd-api treats zero-extent datasets as constant components and
        # rejects store_chunk on them with an opaque RuntimeError
        raise ValueError("cannot write a zero-particle ParticleState")
    particles = iteration.particles[species.name]
    dim = api.Unit_Dimension

    _store_vector(api, particles, "position", state.position_m, buffers)
    particles["position"].unit_dimension = {dim.L: 1}
    for axis in "xyz":
        component = particles["positionOffset"][axis]
        component.reset_dataset(api.Dataset(np.dtype("float64"), (count,)))
        component.make_constant(0.0)
        component.unit_SI = 1.0
    particles["positionOffset"].unit_dimension = {dim.L: 1}

    _store_vector(api, particles, "momentum", state.momentum_kg_m_s, buffers)
    particles["momentum"].unit_dimension = {dim.M: 1, dim.L: 1, dim.T: -1}

    _store_scalar(api, particles, "weighting", state.weight.astype(np.float64), buffers)
    _store_scalar(api, particles, "time", state.time_s.astype(np.float64), buffers)
    particles["time"].unit_dimension = {dim.T: 1}
    _store_scalar(api, particles, "id", state.particle_id.astype(np.int64), buffers)
    _store_scalar(api, particles, "parentId", state.parent_id.astype(np.int64), buffers)
    _store_scalar(api, particles, "alive", state.alive.astype(np.uint8), buffers)
    _store_scalar(
        api, particles, "lostAtElement", state.lost_at_element.astype(np.int32), buffers
    )

    _store_constant(api, particles, "charge", species.charge_c, count)
    particles["charge"].unit_dimension = {dim.T: 1, dim.I: 1}
    _store_constant(api, particles, "mass", species.mass_kg, count)
    particles["mass"].unit_dimension = {dim.M: 1}

    particles.set_attribute("latentDirac_pdgId", int(species.pdg_id))
    particles.set_attribute("latentDirac_isAntimatter", bool(species.is_antimatter))
    particles.set_attribute("latentDirac_symbol", species.symbol)
    particles.set_attribute(
        "latentDirac_metadata", json.dumps(state.metadata, default=str)
    )
    provenance = state.metadata.get("provenance")
    if isinstance(provenance, Mapping):
        for key, attribute in _PROVENANCE_ATTRIBUTES.items():
            if key in provenance:
                particles.set_attribute(attribute, str(provenance[key]))


def write_particle_states(
    path: str | Path,
    states: Mapping[str, ParticleState],
    *,
    author: str | None = None,
) -> None:
    """Write labeled ParticleState snapshots as one openPMD iteration each.

    Iterations are numbered 0..N-1 in mapping insertion order; each label
    is stored as the iteration attribute ``latentDirac_label``. The file
    backend follows the path suffix (``.h5`` and ``.json`` are tested;
    ``.bp`` works but ADIOS2 discourages group-based encoding). An
    existing file at ``path`` is overwritten.
    """
    if not states:
        raise ValueError("states must contain at least one labeled ParticleState")
    api = _require_api()

    series = api.Series(str(Path(path)), api.Access.create)
    try:
        series.set_software("latent-dirac", _package_version())
        if author is not None:
            series.author = author
        for index, (label, state) in enumerate(states.items()):
            iteration = series.iterations[index]
            iteration.set_attribute("latentDirac_label", str(label))
            buffers: list = []
            _write_state(api, iteration, state, buffers)
            series.flush()
    finally:
        series.close()


def write_scene_result(
    path: str | Path,
    result: SceneRunResult,
    *,
    author: str | None = None,
) -> None:
    """Write every monitor snapshot, then the final pipeline state.

    Monitors keep their scene labels and insertion order; the accepted
    final cloud is appended under the label ``"final"``.
    """
    if FINAL_LABEL in result.monitors:
        raise ValueError(
            f"monitor label {FINAL_LABEL!r} collides with the final-state label"
        )
    states: dict[str, ParticleState] = dict(result.monitors)
    states[FINAL_LABEL] = result.pipeline_result.final_cloud
    write_particle_states(path, states, author=author)
