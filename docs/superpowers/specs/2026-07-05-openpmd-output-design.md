# openPMD output design (closed-loop v1, item 1)

Date: 2026-07-05
Status: adopted (the deferred Spec 2e, scheduled first in closed-loop v1
by the solver-zoo composition spec)

## Decision

Latent Dirac writes particle clouds to the openPMD standard through the
`openpmd-api` reference library, behind a new optional extra
`[openpmd]`. This is the first real member of the Analysis (sink)
component in the solver zoo: `ParticleState` snapshots become openPMD
particle species with correct SI unit metadata, so the wider ecosystem
(openPMD viewers, postprocessing stacks, PIC tools) can read our output
without custom parsers.

Scope of v1: writing only. Reading openPMD back into `ParticleState`,
ADIOS2 streaming, and mesh records (fields) are explicitly out of this
spec. The CLI stays untouched; a `--openpmd` flag can come with the
uproot work when the Analysis sink grows a second format.

## Module and API

New module `latent_dirac/io/openpmd_io.py` (naming follows the existing
`hdf5_io.py` / `parquet_io.py` placeholders, which stay untouched):

- `write_particle_states(path, states: Mapping[str, ParticleState],
  *, author: str | None = None) -> None` — one openPMD iteration per
  mapping entry, in insertion order (iteration indices 0..N-1). The
  mapping key (e.g. a monitor label) is stored on the iteration as the
  attribute `latentDirac_label`. Empty mappings raise `ValueError`.
- `write_scene_result(path, result: SceneRunResult, *, author=None)` —
  convenience wrapper: every monitor snapshot in `result.monitors`
  (insertion order) followed by the final pipeline state under the
  label `"final"`.

The file backend is chosen by `openpmd-api` from the path suffix
(`.h5`, `.json`, `.bp`); the Series is group-based (one file per
Series). Tests parametrize over `.json` (always available) and `.h5`
(when the installed wheel carries HDF5 support per
`openpmd_api.variants`).

## openPMD mapping

Per iteration, one particle species named `state.species.name`:

| Record | Source | Unit dimension |
| --- | --- | --- |
| `position` (x/y/z) | `position_m` columns | L |
| `positionOffset` (x/y/z) | constant 0.0 (required by the standard) | L |
| `momentum` (x/y/z) | `momentum_kg_m_s` columns | M·L·T⁻¹ |
| `weighting` | `weight` | dimensionless |
| `time` | `time_s` (per-particle, custom record) | T |
| `id` / `parentId` | `particle_id` / `parent_id` | dimensionless |
| `alive` | `alive` as uint8 | dimensionless |
| `lostAtElement` | `lost_at_element` (int32, −1 = alive) | dimensionless |
| `charge` | constant `species.charge_c` | T·I |
| `mass` | constant `species.mass_kg` | M |

All stored values are SI, so every `unitSI` is 1.0 — the State
boundary rule does the unit work for us. Species-level attributes:
`latentDirac_pdgId`, `latentDirac_isAntimatter`, `latentDirac_symbol`.

Provenance: `state.metadata` is serialized to the species attribute
`latentDirac_metadata` as a JSON string (`default=str` guards numpy
scalars). When `metadata["provenance"]` exists (the engine four-tuple
of the yield-table source), its entries are additionally lifted to
flat, greppable attributes `latentDirac_geant4Version`,
`latentDirac_physicsList`, `latentDirac_datasets`,
`latentDirac_patches` — engine-backed output must keep its four-tuple
visible, per the solver-zoo interface contract.

Series-level metadata: software name/version (`latent-dirac`, package
version) and the optional author.

## Packaging

`pyproject.toml` gains `openpmd = ["openpmd-api>=0.15"]` under
`[project.optional-dependencies]`. The core stays lightweight; imports
happen inside the writer functions with a clear error message when the
extra is missing.

## Tests (TDD)

`tests/test_openpmd_output.py`, `pytest.importorskip("openpmd_api")`
at module level (same pattern as the xopt/plotly optional tests):

1. Round-trip: write a small constructed `ParticleState`, read the file
   back with raw `openpmd_api`, assert positions/momenta/weights/ids
   byte-accurate, `alive`/`lostAtElement` preserved, charge/mass
   constants equal to species values, momentum `unitDimension`
   correct and `unitSI == 1.0`.
2. Scene-result convenience: run the hello-beamline scene; the Series
   holds `len(monitors) + 1` iterations, labels in order, final label
   `"final"`.
3. Provenance lift: a state with a `provenance` metadata dict exposes
   the four flat attributes; a state without it exposes none of them
   but always carries `latentDirac_metadata`.
4. Empty mapping raises `ValueError`; missing dependency raises
   `ImportError` with the extra name in the message (tested via
   monkeypatched import, optional).

## Honesty notes

The writer records exactly what the pipeline produced — no resampling,
no unit massaging beyond SI passthrough. Fidelity tier stays whatever
the source declared; openPMD output is transport, not physics.
