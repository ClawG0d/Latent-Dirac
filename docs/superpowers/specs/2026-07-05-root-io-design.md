# ROOT I/O via uproot design (closed-loop v1, item 2)

Date: 2026-07-05
Status: adopted

## Decision

Latent Dirac reads and writes ROOT files through `uproot` (pure-Python,
no ROOT installation), behind a new optional extra `[root]`. This is
the second Analysis-sink member of the solver zoo and the first one
that also *reads*: a `ParticleState` written here round-trips back,
so ROOT-ecosystem analyses (the reference chain's fourth tool) can
both consume our output and produce input we replay.

Scope of v1: flat particle TTrees plus a JSON metadata sidecar object
per label. RNTuple, histograms, and streaming stay out; the
`latent_dirac/adapters/root/` placeholder is untouched (this is file
I/O, not an adapter integration — the placeholder gate still flips
with the Xsuite adapter per the solver-zoo spec).

## Module and API

New module `latent_dirac/io/root_io.py`:

- `write_particle_states(path, states: Mapping[str, ParticleState])` —
  one TTree per mapping entry, named by label. Empty mappings and
  zero-particle states raise `ValueError` (same contract as
  `openpmd_io`; a uniform sink contract beats per-backend quirks).
  Existing files are overwritten (`uproot.recreate`).
- `read_particle_state(path, label) -> ParticleState` — full
  round-trip reconstruction, including species (rebuilt from the
  metadata sidecar, not from the built-in registry, so custom species
  survive) and `metadata`.
- `write_scene_result(path, result)` — monitors in insertion order,
  then the final cloud under `"final"` (collision raises, as in
  `openpmd_io`).

## TTree layout

Branch names carry SI units explicitly — ROOT convention often means
GeV/c, so ambiguity is the failure mode to design away:

| Branch | Source | dtype |
| --- | --- | --- |
| `x_m`, `y_m`, `z_m` | `position_m` columns | float64 |
| `px_kg_m_s`, `py_kg_m_s`, `pz_kg_m_s` | `momentum_kg_m_s` columns | float64 |
| `time_s` | `time_s` | float64 |
| `weight` | `weight` | float64 |
| `particle_id`, `parent_id` | ids | int64 |
| `alive` | `alive` | bool → uint8 |
| `lost_at_element` | ledger channel | int32 |

Sidecar: a `TObjString` at key `{label}__metadata` holding JSON with
three blocks: `species` (all `ParticleSpecies` fields), `metadata`
(the state's dict, `default=str`), and `format` (schema version 1,
unit note). The engine provenance four-tuple therefore travels inside
`metadata` unchanged; readers on the ROOT side get it by parsing one
JSON string.

## Packaging and CI

`pyproject.toml` gains `root = ["uproot>=5"]`. CI installs it in its
own non-fatal step (same pattern as `[openpmd]`); the tests
`importorskip("uproot")`.

## Tests (TDD)

`tests/test_root_output.py`:

1. Write→read round-trip: arrays byte-accurate (positions, momenta,
   time, weight, ids, alive, ledger), species fields equal, metadata
   dict (with a provenance four-tuple) preserved.
2. Scene-result convenience: hello-beamline; tree names = monitor
   labels + `"final"`; reading `"final"` matches the pipeline's
   final cloud.
3. Raw-uproot readability: branches visible with expected names via
   plain `uproot.open` (the point is ROOT-ecosystem consumption, not
   only our own reader).
4. Zero-particle state and empty mapping raise `ValueError`;
   `"final"` monitor-label collision raises.
5. Unknown-species round-trip: a custom `ParticleSpecies` (not in the
   built-in registry) survives write→read.

## Honesty notes

Pass-through transport of what the pipeline produced; SI throughout;
no unit conversion, no resampling. Reading constructs a fresh
`ParticleState` and revalidates shapes via its `__post_init__`.
