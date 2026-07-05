"""ROOT file I/O via uproot: the second Analysis sink, and the first that reads.

Writes `ParticleState` snapshots as flat ROOT TTrees (one per label) with
SI-unit branch names, plus a JSON metadata sidecar (`TObjString` at
``{label}__metadata``) carrying the full species definition, the state
metadata (including any engine provenance four-tuple), and the format
version. Round-trips back through `read_particle_state`. Requires the
optional `[root]` extra (`pip install "latent-dirac[root]"`); uproot is
pure Python — no ROOT installation involved.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from latent_dirac.core.species import ParticleSpecies
from latent_dirac.state.particle_state import ParticleState

if TYPE_CHECKING:  # pragma: no cover - typing only
    from latent_dirac.scene.build import SceneRunResult

FINAL_LABEL = "final"
_METADATA_SUFFIX = "__metadata"
_FORMAT = {"schema": 1, "units": "SI; branch names carry the unit suffix"}


def _require_uproot():
    try:
        import uproot
    except ImportError as exc:  # pragma: no cover - exercised without extra
        raise ImportError(
            'ROOT I/O requires the optional [root] extra: pip install "latent-dirac[root]"'
        ) from exc
    return uproot


def _tree_payload(state: ParticleState) -> dict[str, np.ndarray]:
    return {
        "x_m": np.ascontiguousarray(state.position_m[:, 0], dtype=np.float64),
        "y_m": np.ascontiguousarray(state.position_m[:, 1], dtype=np.float64),
        "z_m": np.ascontiguousarray(state.position_m[:, 2], dtype=np.float64),
        "px_kg_m_s": np.ascontiguousarray(state.momentum_kg_m_s[:, 0], dtype=np.float64),
        "py_kg_m_s": np.ascontiguousarray(state.momentum_kg_m_s[:, 1], dtype=np.float64),
        "pz_kg_m_s": np.ascontiguousarray(state.momentum_kg_m_s[:, 2], dtype=np.float64),
        "time_s": state.time_s.astype(np.float64),
        "weight": state.weight.astype(np.float64),
        "particle_id": state.particle_id.astype(np.int64),
        "parent_id": state.parent_id.astype(np.int64),
        "alive": state.alive.astype(np.uint8),
        "lost_at_element": state.lost_at_element.astype(np.int32),
    }


def _sidecar_json(state: ParticleState) -> str:
    return json.dumps(
        {
            "species": state.species.model_dump(),
            "metadata": state.metadata,
            "format": _FORMAT,
        },
        default=str,
    )


def write_particle_states(
    path: str | Path,
    states: Mapping[str, ParticleState],
) -> None:
    """Write labeled ParticleState snapshots as one ROOT TTree each.

    Tree names are the mapping keys; each tree gets a JSON sidecar
    `TObjString` at ``{label}__metadata``. An existing file at ``path``
    is overwritten. Zero-particle states, empty mappings, and labels
    that would break ROOT paths (``/``, ``;``, empty, ``__metadata``
    suffix) raise ``ValueError`` (uniform sink contract with
    `openpmd_io`). Metadata values survive only as JSON-native types;
    anything else is stringified on write.
    """
    if not states:
        raise ValueError("states must contain at least one labeled ParticleState")
    for label, state in states.items():
        label = str(label)
        if not label:
            raise ValueError("tree labels must be non-empty")
        if "/" in label or ";" in label:
            # "/" creates ROOT directories, ";" is the key-cycle separator:
            # both write "successfully" and then break readers
            raise ValueError(f"label {label!r} must not contain '/' or ';'")
        if label.endswith(_METADATA_SUFFIX):
            raise ValueError(
                f"label {label!r} collides with the {_METADATA_SUFFIX!r} sidecar keys"
            )
        if state.position_m.shape[0] == 0:
            raise ValueError(
                f"cannot write a zero-particle ParticleState (label {label!r})"
            )
    uproot = _require_uproot()

    # serialize everything before touching the file: uproot.recreate
    # truncates immediately, so a late failure would destroy prior content
    payloads = [
        (str(label), _tree_payload(state), _sidecar_json(state))
        for label, state in states.items()
    ]

    with uproot.recreate(str(Path(path))) as f:
        for label, payload, sidecar in payloads:
            # explicit TTrees: dict assignment writes RNTuple on newer
            # uproot, which older ROOT releases cannot read
            f.mktree(label, {name: array.dtype for name, array in payload.items()})
            f[label].extend(payload)
            f[label + _METADATA_SUFFIX] = sidecar


def read_particle_state(path: str | Path, label: str) -> ParticleState:
    """Round-trip a labeled snapshot back into a ParticleState.

    The species is rebuilt from the sidecar (custom species survive);
    a missing label — or a file written by other tools without the
    ``{label}__metadata`` sidecar — raises uproot's ``KeyInFileError``
    (a ``KeyError`` naming the missing key).
    """
    uproot = _require_uproot()

    with uproot.open(str(Path(path))) as f:
        arrays = f[str(label)].arrays(library="np")
        sidecar = json.loads(str(f[str(label) + _METADATA_SUFFIX]))

    return ParticleState(
        species=ParticleSpecies(**sidecar["species"]),
        position_m=np.stack([arrays["x_m"], arrays["y_m"], arrays["z_m"]], axis=1),
        momentum_kg_m_s=np.stack(
            [arrays["px_kg_m_s"], arrays["py_kg_m_s"], arrays["pz_kg_m_s"]], axis=1
        ),
        time_s=arrays["time_s"].astype(np.float64),
        weight=arrays["weight"].astype(np.float64),
        alive=arrays["alive"].astype(bool),
        particle_id=arrays["particle_id"].astype(np.int64),
        parent_id=arrays["parent_id"].astype(np.int64),
        lost_at_element=arrays["lost_at_element"].astype(np.int32),
        metadata=sidecar["metadata"],
    )


def write_scene_result(path: str | Path, result: SceneRunResult) -> None:
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
    write_particle_states(path, states)
