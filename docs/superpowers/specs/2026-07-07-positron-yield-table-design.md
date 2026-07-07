# Positron yield table (M3-a: engine/positrongen + table-based source)

Date: 2026-07-07. Status: accepted (owner decision 2026-07-06:
10 MeV electrons on tungsten, production-only table; the moderation
table is a separate later deliverable).

## Physics

A 10 MeV electron beam on a tungsten converter: bremsstrahlung
followed by pair production yields positrons — the standard
accelerator-driven positron source topology, delegated entirely to
the vendored vanilla Geant4 engine (FTFP_BERT reference list; EM
physics dominates at these energies). The converter is a parameterized
stand-in — a tungsten disc r = 5 mm, thickness 2 mm (~0.57 radiation
lengths; W X0 ≈ 3.5 mm) — not facility target engineering. Every
positron exit event from the converter surface is recorded (same
exit-event convention as the antiproton table: re-entry/re-exit counts
once per exit; acceptable at the table-based tier).

## Engine application

`engine/positrongen` — a sibling of `engine/yieldgen`, same guards and
contract discipline:

- Usage: `positrongen <n_events> <out.csv> [kinetic_mev=10]`.
- Records PDG −11 exits from the target volume; CSV columns
  `x_m,y_m,z_m,px_gev_c,py_gev_c,pz_gev_c`.
- Header: `# latent-dirac positron yield table v1`, generator,
  Geant4 version, physics list, datasets (YIELDGEN_DATASETS
  convention), patches, `primary = e-`,
  `primary_kinetic_mev`, `n_primaries`, target description; trailing
  `# complete = true` marker (the parser refuses tables without it).
- INT_MAX guard on n_events (BeamOn takes G4int), MT-safe mutex-held
  ofstream — both inherited from yieldgen.

## Python source

`latent_dirac/sources/positron_table.py`:
`PositronYieldTableSource(table_path, primary_electron_count,
macro_particles=None)` — mirrors the antiproton table source (shared
`_parse_table`; completion-marker and n_primaries validation; each
row represents `primary_electron_count / n_primaries` physical
positrons; optional bootstrap resampling preserves the represented
total; provenance four-tuple into metadata). Species: positron.
Scene registration: `positron_yield_table` in the source union and
`_SOURCE_CLASSES`. Fidelity tier: table_based.

## Committed artifact

`examples/data/positron_yield_ftfp_bert_10mev_w.csv`, produced on the
project's WSL engine build; statistics sized after a timed smoke run
(target: O(10^4+) recorded positrons). The run command and environment
are reproducible from `engine/README.md` plus this spec.

## Validation

- Python: table fixtures (happy path, missing completion marker,
  missing/invalid n_primaries, wrong column count) mirroring the
  antiproton source tests; weight-model total preserved under
  bootstrap; species/charge assertions (positron, +e).
- Engine-side sanity on the committed table: yield per primary in the
  literature ballpark for ~0.5 X0 W at 10 MeV (1e-3–1e-2 e+/e-);
  forward-hemisphere dominance; energy spectrum bounded by the
  primary energy.
- Status sync: README Solvers table Source row, roadmap M3 entry,
  CHANGELOG — same commit.
