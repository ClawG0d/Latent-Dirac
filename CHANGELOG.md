# Changelog

Latent Dirac uses 0.x semantics: minor releases may break APIs without
deprecation shims. Notable changes are recorded here starting from 0.2.0.

## Unreleased (0.2.0)

- Added `FieldMapField`: table-based fields on a regular grid with
  trilinear interpolation, a COMSOL regular-grid CSV importer, and 3D
  field-magnitude volume rendering.
- Added the declarative scene schema (YAML/JSON, `schema_version: 1`) with
  fail-fast validation, label-anchored loss accounting, the new `drift`
  and `monitor` elements, and `run_scene` with optional trajectory
  recording. pyyaml is now a core dependency.
- Added scene-driven 3D rendering (`latent_dirac.viz.scene_3d`) with
  per-element fidelity labels in hover text.
- Added the composable field model library: `CompositeField`,
  hard-edge `DipoleField`, and hard-edge `QuadrupoleField`.
- Repositioned the project as an open interactive simulation platform for
  antimatter factories, with an enforced documentation honesty discipline.
- Added the 3D charge-sign splitter README animation rendered from a
  recorded Boris-solver `Trajectory`, plus an interactive Plotly HTML export
  (`tools/generate_hero_3d_webp.py`).
- Added CI (pytest matrix + ruff), a PyPI trusted-publishing workflow,
  CONTRIBUTING.md, and this changelog.

## 0.1.0

- Initial architecture skeleton: SI units and species, `ParticleCloud`
  state, positron pair / beta-plus / surrogate antiproton sources, uniform
  and idealized solenoid fields, relativistic Boris solver, aperture and
  momentum-window acceptance, pipeline loss accounting, accepted-yield
  diagnostics, optional Matplotlib/Plotly backends, placeholder adapters
  for Geant4, Xsuite, and ROOT.
