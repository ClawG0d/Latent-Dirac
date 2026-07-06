# Geant4 Matter adapter design (M2)

Date: 2026-07-05
Status: shipped in this change (adapter real; `matter_slab` scene
element deferred to M2b)

## Goal

Make the Geant4 adapter real: the Matter component of the solver layer,
at `latent_dirac/adapters/geant4/` (location confirmed by the owner;
the conceptual "Solvers" layer maps engine transformers to `adapters/`,
mirroring the Xsuite Lattice precedent). A `ParticleState` cloud passes
through a configurable slab of real material; vanilla Geant4 (FTFP_BERT)
computes energy loss, multiple scattering, nuclear interaction, and —
for antiprotons — annihilation; the surviving primaries come back with
updated phase space and the absorbed ones die into the loss ledger.

Exchange is subprocess + files (the official Python bindings left the
Geant4 tree in 11.1); gradients stop at this boundary (transformer form,
per the solver composition spec).

## Phase-space exchange contract v1

Shared CSV grammar with the yield-table contract (`#` header lines,
`key = value`, mandatory trailing `# complete = true`):

```
# latent-dirac phase space v1
# species = anti_proton              (Geant4 particle name)
# n_rows = <int>
# columns = id,x_m,y_m,z_m,px_gev_c,py_gev_c,pz_gev_c
<rows>
# complete = true
```

The engine output additionally carries the provenance four-tuple
(`geant4_version`, `physics_list`, `datasets`, `patches`), the transform
parameters (`material`, `thickness_mm`), and `n_primaries` (= input
rows). `id` ties every output row to its input row: macro-particle
identity survives the engine boundary.

Species map (latent-dirac → Geant4): electron→e-, positron→e+,
proton→proton, antiproton→anti_proton. Unsupported species are rejected
before any subprocess runs.

## engine/transformer (C++)

`transformer <input.csv> <output.csv> <nist_material> <thickness_mm>`

- Geometry: vacuum world of half-length 0.6 m (`kWorldHalfLength`);
  slab of the NIST material, transverse half-width 20 cm
  (`kHalfWidth`), occupying z ∈ [0, thickness]; a thin vacuum scoring
  plane 1 mm downstream (front face at z ≈ thickness + 0.95 mm — the
  exact z is just field-free drift and is not load-bearing). A
  `thickness_mm` that would push the plane outside the world exits 2.
- Physics: FTFP_BERT reference list (annihilation, energy loss, MSC,
  nuclear interaction — the reason this adapter exists).
- Primaries: one event per input row, fired from the row's position and
  momentum in contract coordinates (slab front at z = 0).
- Survivor semantics: the primary track (parent id 0, input PDG) is
  recorded once when it enters the downstream scoring plane. Absorbed,
  annihilated, stopped, backscattered, or transversely-escaped primaries
  produce no row and come back dead. Secondaries are outside contract v1
  (fidelity note).
- Values are written at 9 significant digits (matching the Python
  writer's `%.9g`); an unsupported species name exits 2 before the run
  builds (a serial `G4RunManager` builds the primary generator eagerly,
  so the guard runs pre-Initialize); MT with the yieldgen mutex-writer
  pattern, provenance header from `G4Version` + `YIELDGEN_DATASETS`,
  `# complete = true` on success, `n_rows` guarded against G4int
  overflow.

## Python adapter (latent_dirac/adapters/geant4/adapter.py)

`Geant4MatterAdapter` (pydantic, importable and constructible without
any engine present; the subprocess is the only engine contact):

- `command: tuple[str, ...]` — transformer invocation prefix (native
  binary, or a WSL bridge); `path_style="wsl"` renders exchange-file
  paths through the `windows_to_wsl_path()` helper
  (`C:\…` → `/mnt/c/…`), so the bridge command sources the engine
  environment and execs the transformer
- `material: str` (NIST name), `thickness_mm: float > 0`,
  `entry_z_m: float = 0.0` — slab front in pipeline coordinates; the
  adapter shifts z by −entry_z_m on write and back on read
- `transverse_half_width_m` / `world_half_length_m` — must match the
  engine build; particles outside the aperture (would be miscounted as
  losses) or the world (would abort the run) are rejected up front
- `apply(state) -> ParticleState` — writes the alive particles as input
  rows (ids = original indices), runs the subprocess, parses the output,
  and returns a new state: survivors get engine positions/momenta,
  non-survivors among the sent ids get `alive=False` (the Stage wrapper
  stamps `lost_at_element`), already-dead particles pass through
  untouched, weights unchanged
- `stage(name) -> Stage` — pipeline integration; the ledger comes free
- metadata: `model_type="engine_transformer"`, the provenance
  four-tuple (printed by `scene_report`), and the matter parameters

Weight semantics: survival is sampled per macro-particle (binary), so
weights pass through unchanged; with strongly non-uniform weights the
survival estimate gains variance — documented fidelity caveat.

## Out of scope for M2

- A `matter_slab` scene element (M2b: NumPy-pipeline-only element like
  `annihilation_plate`; needs schema + build wiring; the JAX backend and
  the differentiable mirror are untouched by design — transformers are
  engine boundaries)
- Secondary-particle transport across the boundary; GDML translation of
  full scenes (the slab is parameterized geometry, not facility CAD)

## Testing

- TDD, all engine-free: a stub transformer script (invoked through the
  real subprocess path) implements the contract; round-trip mapping,
  id/complete-marker validation, species mapping, provenance metadata,
  Stage/ledger integration, weight preservation.
- Gate: `test_adapter_status_matches_roadmap` extends to assert the
  geant4 adapter is real (module imports without the engine, placeholder
  gone), keeping `ALLOWED_ADAPTERS` fixed.
- Physical validation (WSL, recorded here, not in CI): 3.6 GeV/c
  antiprotons through 1 mm G4_Al must mostly survive with ~sub-MeV/mm
  energy loss (λ_I(Al) ≈ 39 cm); 100 MeV-KE antiprotons into 50 mm G4_Al
  must mostly annihilate (range ≈ 37 mm < thickness). Results below.

## Physical validation results

WSL build (vanilla Geant4 v11.4.2, FTFP_BERT, 12 datasets), 200
antiprotons per case, pencil beam on G4_Al:

- **Thin foil** (3.6 GeV/c, 1 mm): 200/200 crossed the scoring plane;
  mean momentum loss 0.5 MeV/c per mm (textbook MIP dE/dx in Al is
  ~0.43 MeV/mm) and multiple-scattering transverse kick RMS 2.7 MeV/c
  (Highland estimate ~2 MeV/c) — both at the expected scale.
- **Stopping block** (100 MeV kinetic, 50 mm): 0/200 survived — the
  ~37 mm range is inside the block; every antiproton stops and
  annihilates, exactly the ledger endpoint the Matter component owes.
- **End-to-end**: the Windows-side `Geant4MatterAdapter`
  (`path_style="wsl"`, WSL bridge command) drove the engine through the
  real subprocess path inside a `PipelineRunner`; survivors returned in
  pipeline coordinates with the provenance four-tuple in metadata and
  the stage ledger stamped for absorbed particles.
