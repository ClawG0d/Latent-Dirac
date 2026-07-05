# Engine yield-table generator and engine-backed target demo

Date: 2026-07-05
Status: shipped in this change (first engine-track deliverable: M1'
build recipe + M3-lite yield table + engine-backed Chain 2b demo)

## Goal

Replace the drawn-annotation target of the Chain 2 demo with an honest,
engine-backed antiproton source: vanilla Geant4 (FTFP_BERT) simulates
proton-on-iridium production offline and exports a phase-space yield
table; Latent Dirac consumes the table through a new `table_based`
source. No runtime coupling: the exchange artifact is a versioned CSV.
`latent_dirac/adapters/` stays placeholder-only (this is the M3 offline
route, not the M2 adapter).

## Architecture

```
engine/yieldgen (first-party C++, WSL/Linux build)
  p (26 GeV/c) -> Ir target, FTFP_BERT, MT
  -> records every antiproton EXITING the target surface
  -> pbar_yield CSV (phase space + provenance header)
        |
        v
latent_dirac.sources.antiproton_table.AntiprotonYieldTableSource
  (fidelity tier: table-based; provenance carried in metadata)
        |
        v
existing pipeline: transport -> acceptance -> ledger -> report/3D
```

## CSV contract (v1)

Header lines start with `#`; keys are `# key = value`:

```
# latent-dirac antiproton yield table v1
# generator = engine/yieldgen
# geant4_version = <tag>            (provenance four-tuple, part 1)
# physics_list = FTFP_BERT          (part 2)
# datasets = <G4EMLOW-x.y,...>      (part 3; from env at run time)
# patches = none                    (part 4; vendored tree is frozen)
# primary = proton
# primary_momentum_gev_c = 26.0
# n_primaries = <int>               (REQUIRED for weight normalization)
# target = iridium cylinder r_mm=1.5 half_length_mm=27.5
# columns = x_m,y_m,z_m,px_gev_c,py_gev_c,pz_gev_c
<data rows>
# complete = true
```

The trailing `# complete = true` marker is written after the run
finishes and is REQUIRED by the parser: an interrupted or truncated run
leaves a table whose header claims more primaries than were simulated,
which would silently bias every weight — incomplete tables are rejected
instead.

Weight model: each row is one simulated antiproton from `n_primaries`
protons. With user-declared `primary_proton_count` physical protons,
total represented yield = primary_proton_count * n_rows / n_primaries.
Rows map 1:1 to macro-particles by default; optional `macro_particles`
resamples rows with replacement (bootstrap) and renormalizes weights so
the weighted total is unchanged.

## C++ application (engine/yieldgen)

- Geometry: vacuum world; NIST `G4_Ir` cylinder (r = 1.5 mm,
  half-length 27.5 mm) centred at origin, axis on z (AD-like target,
  labeled parameterized geometry — not facility engineering).
- Physics: `G4PhysListFactory` -> FTFP_BERT (reference list).
- Primaries: `G4ParticleGun`, proton, +z pencil beam, 26 GeV/c,
  starting upstream of the target.
- Scoring: `G4UserSteppingAction`; when an `anti_proton` track's step
  crosses from the target volume into the world, record post-step
  position and momentum. Mutex-guarded writer (rare events; MT-safe).
- Run: `yieldgen <n_events> <out.csv> [momentum_gev_c]`, batch only,
  `G4RunManagerFactory` default (tasking/MT).
- Provenance header written from `G4Version.hh` + physics list name;
  the datasets field comes from the `YIELDGEN_DATASETS` env var, which
  the runner precomputes from `geant4-config --datasets` (see
  `engine/README.md`; the classic `G4*DATA` env vars are a fallback for
  custom-data setups — Geant4 11.x locates datasets from the install
  prefix, so those vars are normally unset).

## Python source (latent_dirac/sources/antiproton_table.py)

`AntiprotonYieldTableSource(SourceTerm)`:
- fields: `table_path: str`, `primary_proton_count: float > 0`,
  `macro_particles: int | None = None` (None -> one macro per row)
- parses the header (requires `n_primaries`), rejects missing/malformed
  rows, converts GeV/c -> SI momentum at the boundary
- `sample(rng)` -> antiproton `ParticleState`; resampling (if
  requested) uses the passed rng for reproducibility
- `metadata`: `model_type="table_based"`, provenance four-tuple echoed
  from the header, table path and row count
- scene schema: new source `type: antiproton_yield_table`; wired in
  `scene/build.py` `_SOURCE_CLASSES`. Sources sample on NumPy at the
  State boundary, so the JAX backend and the differentiable mirror are
  untouched (no element-loop change).

## Demo

`examples/scenes/target_production_engine.yaml`: yield-table source +
collection solenoid + aperture + momentum window (3.0-4.2 GeV/c,
AD-like) + monitor. A relative `table_path` in a scene file resolves
against the scene file's directory (loader rule), so the demo runs from
any cwd. Committed table asset:
`examples/data/pbar_yield_ftfp_bert_26gevc_ir.csv` (curated run,
provenance in header). README Chain 2 section gains the engine-backed
variant; the vendored-engine section's "no build ships" wording updates
to reflect the in-repo `engine/` build recipe while adapters stay
placeholders.

## Testing

- TDD: `tests/test_antiproton_table_source.py` written first
  (header parsing, weight normalization, GeV/c->SI conversion, bootstrap
  resampling determinism, malformed-input errors, scene-schema wiring)
- existing suite must stay green; positioning tests unaffected
  (adapters untouched)
- C++ side validated by a smoke run (100 events) plus physical sanity
  checks on the full run (yield per proton within literature order of
  magnitude for 26 GeV/c p-on-Ir; forward-peaked momentum spectrum)

## Honesty notes

- The demo is labeled: table-based source, and the scene report prints
  a "Source provenance (engine four-tuple)" section (implemented in
  `latent_dirac.diagnostics.scene_report`, fed from the source's cloud
  metadata); target geometry is a parameterized stand-in, not facility
  engineering.
- Performance numbers, if any, come from the WSL2/CUDA environment and
  must be labeled as such; none are required for this demo.
