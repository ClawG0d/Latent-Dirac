# Composite-field scene element (field superposition)

Date: 2026-07-07
Status: implementation spec. Owner decision (2026-07-07, via dialog):
build a composite-field scene element so multiple field models act in the
same region simultaneously — the immediate motivation is a rotating wall
superimposed on a Penning trap, which the sequential-stage pipeline cannot
express today.

## What it is

The scene pipeline runs each field element as its own transport stage, so
fields never overlap in time/space. `composite_field` bundles two or more
field models into ONE stage whose field is their exact superposition
(`CompositeField`, which already exists and sums component E and B). This
unlocks rotating-wall-in-a-trap, solenoid + steering dipole, trap + bias
field, etc.

## Scope (slice 1: NumPy reference backend)

- NumPy pipeline: full support. `CompositeField` sums the components; the
  Boris solver transports through the summed field.
- JAX / differentiable backends: **slice 2.** For now the existing guard
  (`element.type not in _SWEEPABLE_PARAMS`) rejects `composite_field` with
  the standard "not supported by the JAX backend yet; use the NumPy
  pipeline" error. Composition is deterministic and JAX-expressible (a sum
  of the component field fns), so this is a follow-up, not a permanent
  limit like the stochastic `buffer_gas_cooling`.

## Schema — `CompositeFieldElement`

- `type: "composite_field"`
- `fields: list[<field sub-spec>]`, `min_length = 2` — each a field element
  spec (`uniform_field`, `solenoid`, `dipole`, `quadrupole`,
  `penning_trap`, `rotating_wall`), a discriminated union on `type`.
- `steps: int | None` (per-element solver-step override; the composite has
  ONE step count for the summed field).

Sub-fields reuse the existing field element schemas, so each sub-field
keeps its own field-model parameters and its optional time gate
(`t_on_s`/`t_off_s`) — e.g. a rotating wall that switches on partway
through the composite stage. A sub-field's own `steps` / `space_charge`
are ignored (the composite owns stepping; space charge is a
composite-level concern deferred to later); this is documented, not
silently surprising. Nesting a `composite_field` inside `fields` is
rejected (fields must be leaf field models).

## Build

- `build._base_field_for`: `composite_field` →
  `CompositeField(fields=[_field_for(sub) for sub in element.fields])`.
  Using `_field_for` (not `_base_field_for`) on each sub means per-sub time
  gating (the `TimeGatedField` wrap) is honored.
- Add `composite_field` to `FIELD_ELEMENT_TYPES` so it transports as a
  field stage.

## Viz / report

- `viz.field_lines.element_field_line_bundles`: a `composite_field` branch
  that renders each component's field lines (recurse into each sub-spec
  paired with the corresponding built sub-field from `CompositeField.fields`).
- `viz.scene_3d.FIDELITY_LABELS["composite_field"]` (honesty coverage test).
- `diagnostics.scene_report`: a field-status block listing the components
  ("composite field: penning_trap + rotating_wall (m=2)"), each an exact
  superposition; fidelity follows the component models.

## Validation / tests (TDD)

- The summed field equals the elementwise sum of its components at sample
  points and times (e.g. penning_trap + rotating_wall).
- A scene with a composite transports correctly (compare a
  rotating-wall-in-trap composite run vs the same fields applied and
  summed by hand at a few steps, or vs a NumPy CompositeField reference).
- Schema: `fields` requires ≥2 entries; a nested composite is rejected;
  each sub-field validates by its own schema.
- Per-sub time gate works (a gated sub-field contributes 0 outside its
  window inside the composite).
- The JAX backend raises the clear "not supported yet" error for a
  composite scene (slice-2 boundary pinned).
- A committed demo: `rotating_wall` + `penning_trap` composite
  (`examples/scenes/rotating_wall_in_trap.yaml`).

## Honesty

The composition itself is exact superposition; each component keeps its
own fidelity tier and caveats (e.g. the rotating wall's single-particle /
no-plasma-compression note still applies). The demo is labelled
accordingly. No new physics claims.

## Non-goals

- No JAX/differentiable support yet (slice 2).
- No composite-level space charge; no nested composites; no per-sub step
  counts. m ≥ 3 sub-fields are allowed (the list is unbounded) but each
  must be a leaf field model.
