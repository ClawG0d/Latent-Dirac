# matter_slab scene element design (M2b)

Date: 2026-07-06
Status: adopted (Mac-side task T1 per TASK-SPLIT.md)

## Decision

Wire the already-real Geant4 Matter adapter into the declarative scene
layer: a new `matter_slab` element type lets a scene YAML place a slab
of NIST material in the beamline. The adapter body
(`latent_dirac/adapters/geant4/adapter.py`, `Geant4MatterAdapter`) is
**not touched** — this is schema + build + viz wiring only, following
the Xsuite-adapter-to-scene precedent conceptually (though Lattice has
no scene element yet either; this is the first engine element in a
scene).

## The machine-vs-scene boundary (the load-bearing design point)

A scene is declarative and machine-independent; the transformer binary
path is machine-specific. So the split is:

- **In the scene element** (physics, portable): `material` (NIST name),
  `thickness_mm`, `entry_z_m`, and the engine geometry envelope
  `transverse_half_width_m` / `world_half_length_m` (these must match
  the compiled transformer build — documented as such, defaulting to
  the adapter's 0.20 / 0.60).
- **Injected at run time** (machine-specific, never in committed YAML):
  the transformer command and path style. Resolved from the environment
  variable `LATENT_DIRAC_G4_TRANSFORMER` (a command string, shlex-split)
  plus optional `LATENT_DIRAC_G4_PATH_STYLE` (`native` | `wsl`,
  default `native`).

Constructability without the engine: a scene with `matter_slab` must
build, validate, and render with no binary present. Only *running* the
stage requires the binary. Therefore:

- `scene_from_mapping` / schema validation: always succeeds.
- `scene_3d` rendering: always works (adds a box; see below).
- `run_scene`: the `matter_slab` stage resolves the command from the
  env var when the stage is built; if unset, the stage's action raises
  a clear `RuntimeError` ("set LATENT_DIRAC_G4_TRANSFORMER ...") when
  invoked — not at construction.

## Integration points

1. `scene/schema.py`: `MatterSlabElement(type="matter_slab", material,
   thickness_mm>0, entry_z_m=0.0, transverse_half_width_m=0.20,
   world_half_length_m=0.60)`, added to `ElementSpec`, `extra="forbid"`.
2. `scene/build.py`: `_matter_slab_action(element)` resolves the env-var
   command and builds a `Geant4MatterAdapter`, returning its `.apply`;
   if the env var is unset, returns an action that raises the clear
   error. Wired in the stage loop like the other loss elements.
3. `viz/scene_3d.py`: **required** (roadmap 2b: every element type has a
   3D representation, or it renders silently invisible with no error).
   `_element_segments` gains a `matter_slab` branch — a filled-ish box
   at `entry_z_m` spanning the slab thickness and the transverse
   envelope; `FIDELITY_LABELS` gains a `matter_slab` entry
   (`"fidelity: engine transformer (vanilla Geant4, FTFP_BERT)"`).
4. JAX side: **zero code**. `matter_slab` is not in `_SWEEPABLE_PARAMS`,
   so `_base_params` rejects it ("not supported by the JAX backend
   yet"). Do NOT touch the mirror-pair element loops. A test asserts the
   rejection.
5. Docs: `docs/scene_schema.md`, README/roadmap/CHANGELOG status sync in
   the same commit (M2b done).

## Provenance in the scene report

The adapter already writes the engine four-tuple into
`state.metadata["matter"]` provenance. `scene_report` should surface it
for a `matter_slab` stage (engine version / physics list / datasets /
patches), matching how the yield-table source's provenance is printed —
the safety-scope provenance rule applies to engine-derived results.

## Tests (TDD, engine-free)

Reuse the stub-transformer pattern from
`tests/test_geant4_matter_adapter.py` (a Python script invoked via
`command=(sys.executable, stub)`), driven here through a full scene:

1. A `matter_slab` scene runs end-to-end against the stub (env var set
   to the stub): survivors updated, absorbed particles ledgered at the
   slab's stage index.
2. Env var unset → running the scene raises the clear "set
   LATENT_DIRAC_G4_TRANSFORMER" error; but `scene_from_mapping` and
   `render_scene_3d` succeed without it.
3. Schema: `thickness_mm<=0` rejected; unknown field rejected
   (`extra="forbid"`).
4. JAX backend rejects a `matter_slab` scene.
5. `scene_3d` produces a trace for the slab (not silently invisible).
6. Provenance from the stub header appears in the scene report.

## Honesty notes

Fidelity tier: engine transformer (vanilla Geant4 v11.4.2, FTFP_BERT) —
the element is a real engine boundary, not a parameterized stand-in
(that was Chain 2's drawn annotation). Naming rules from the engine
positioning spec apply. No performance wording.

## Coordination

Mac-owned files for this task (TASK-SPLIT.md §五): `scene/schema.py`,
`scene/build.py`, `viz/scene_3d.py`, `docs/`. The adapter body and
anything under `adapters/geant4/` or `engine/` stay Windows-owned — if
this task turns out to need an adapter change, flag the owner first.
