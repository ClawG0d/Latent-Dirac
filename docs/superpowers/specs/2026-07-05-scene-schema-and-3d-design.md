# Scene Schema and Minimal 3D Design (Spec 2b)

## Objective

Add a declarative, serializable scene description that maps one-to-one onto
the existing pipeline, plus a function-level 3D rendering API driven by that
description. This is the visual-flagship half of Phase 2: "the declarative
scene is visible from day one."

Scope decision (user, 2026-07-05): schema + build + 3D function API only.
The `latent-dirac` CLI and the hello-beamline example are deferred to a
follow-up spec. Two new zero-physics elements are included: `drift`
(zero-field transport) and `monitor` (phase-space snapshot). Material
interaction elements (degraders, moderators) stay out until the Phase 4
Geant4 adapter.

## Schema

YAML is the primary hand-written format, JSON is equally supported; both
parse into the same pydantic `Scene` model (pyyaml becomes a core
dependency by user decision).

```yaml
schema_version: 1
name: capture-line
seed: 2026
source:
  type: positron_pair        # positron_pair | beta_plus | antiproton_surrogate
  label: pair-source
  params: { primary_count: 10000, yield_eplus_per_primary: 0.02, ... }
solver:
  type: relativistic_boris
  dt_s: 2.0e-12
  steps: 100
elements:
  - { type: solenoid, label: capture-solenoid, b_tesla: 0.8, radius_m: 0.05, length_m: 0.5 }
  - { type: drift, label: gap-1, steps: 20 }
  - { type: aperture, label: collimator, radius_m: 0.04, z_m: 0.06 }
  - { type: momentum_window, label: momentum-cut, p_min_gev_c: 0.001, p_max_gev_c: 0.020 }
  - { type: monitor, label: end-station }
```

Rules:

- `schema_version` is required and must equal 1.
- Every element and the source carry a required `label`; labels must be
  unique across the scene. Labels become pipeline stage names, so the loss
  ledger is anchored to scene labels.
- Validation is fail-fast: unknown element types, unknown keys
  (`extra="forbid"`), missing labels, and duplicate labels are rejected at
  load time. Source `params` are validated by the source classes themselves
  (they are pydantic models).
- Element vocabulary: `uniform_field`, `solenoid`, `dipole`, `quadrupole`
  (transport through the corresponding field model), `drift` (zero-field
  transport), `aperture`, `momentum_window` (acceptance; momenta given in
  GeV/c and converted at build time), `monitor` (records a cloud snapshot,
  no physics).
- Field elements accept an optional `steps` override; otherwise the global
  solver `steps` applies.
- Batch convention (documented, not implemented): every numeric parameter
  must remain liftable into a batch-dimension array in Phase 3; the schema
  must not grow structures that prevent vmap over configurations.

## Modules

- `latent_dirac/scene/schema.py` — pydantic models (`Scene`, `SourceSpec`,
  `SolverSpec`, element discriminated union on `type`).
- `latent_dirac/scene/loader.py` — `load_scene(path)` (suffix-detected
  YAML/JSON), `scene_from_mapping(dict)`.
- `latent_dirac/scene/build.py` — `build_source(scene)`,
  `build_stages(scene, ...)`, and `run_scene(scene, rng=None,
  record_trajectories=False)` returning the pipeline result, monitor
  snapshots by label, and optional per-stage position histories for
  rendering.
- `latent_dirac/viz/scene_3d.py` — `render_scene_3d(scene, run_result,
  max_particles=...)` returning a Plotly figure. Optional viz layer; the
  scene core must not import plotly.

## 3D rendering

- Every geometric element type has a 3D representation, rendered as
  wireframes (a deliberate choice: wireframes stay readable with
  trajectories behind them): solenoid as a cylinder wireframe,
  dipole/quadrupole as hard-edge boxes over their z-extent, aperture as
  concentric circles at its z position, monitor as a plane wireframe at the
  mean z of its surviving snapshot particles. `uniform_field` (global) and
  `drift` (no geometry) render no solid; they appear in trajectories only.
- Element hover text includes the label, type, and fidelity tier label
  (honesty visualization).
- Trajectories come from `record_trajectories=True` runs; lost particles
  are visually distinct from accepted ones in the final-state scatter.

## Validation

- Schema: valid YAML and JSON load identically; unknown type, unknown key,
  missing label, duplicate label, wrong `schema_version` all raise.
- Build/run: stage names equal scene labels; transport advances particles;
  drift preserves kinetic energy; monitor snapshot matches the cloud at its
  pipeline position; GeV/c momentum window converts correctly; fixed seed
  gives deterministic results.
- Rendering (plotly importorskip): figure contains element geometry and
  trajectory traces; fidelity labels present in hover metadata.
- Full suite and ruff stay green; core modules stay viz-free.

## Non-Goals

- CLI entry point and hello-beamline example (deferred follow-up)
- batch execution (Phase 3), JAX backend (Phase 3)
- new physics of any kind; State/Model/Control refactor (2c) — the scene
  layer wraps today's `ParticleCloud`/`PipelineRunner` and will be
  re-targeted onto the 2c containers when they land
