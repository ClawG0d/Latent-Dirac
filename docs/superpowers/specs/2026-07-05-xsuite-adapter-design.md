# Xsuite adapter design (closed-loop v1, item 3)

Date: 2026-07-05
Status: draft (implementation pending; the first adapter to become real)

## Decision

The Xsuite adapter turns `latent_dirac/adapters/xsuite/` from a
placeholder into the zoo's Lattice component: bidirectional
`ParticleState` ↔ `xtrack.Particles` conversion plus a stage-level
tracking wrapper that runs an `xtrack.Line` and returns a
`ParticleState` with xtrack losses stamped into our ledger. This is
the first engine boundary the exchange-currency contract crosses in
process (Geant4 exchanges offline tables; Xsuite runs in the same
Python process), and it flips the placeholder gate test.

Scope of v1: conversion + line tracking on the NumPy pipeline. No JAX
path (xtrack has its own compute contexts; gradients stop here — the
Lattice component is a stepper for physics purposes but an engine
boundary for autodiff purposes). No lattice *construction* helpers:
the user supplies the `xtrack.Line`; scene-schema lattice elements are
a later extension.

## Frame convention (the physics contract)

Our `ParticleState` is Cartesian SI; xtrack is accelerator coordinates
around a reference particle. The adapter demands an explicit
`ReferenceFrame`:

- beam axis: `+z` (matching every existing scene); `s ≡ z`
- `p0c_ev`: reference momentum × c in eV — explicit, never silently
  inferred (a wrong implicit reference is the classic silent bug);
  a `reference_from_state` helper computes the weighted-mean |p| and
  returns a frame, but using it is the caller's visible choice
- mapping: `x = x_m`, `y = y_m`, `px = p_x/p0`, `py = p_y/p0`,
  `delta = (|p| − p0)/p0`, `zeta = z − β₀ c t` (per-particle `time_s`
  enters here; on the way back, `t = (z − zeta)/(β₀ c)`)
- species: `q0` = charge in units of e, `mass0` in eV from `mass_kg`

Round-trip closure (state → Particles → state) must hold to float64
precision on all coordinates, weights, and ids for alive particles.

## Loss/ledger contract

- alive → xtrack `state = 1`; dead-on-entry → xtrack `state = 0`
  particles carried along (particles are never deleted — shapes stay
  static on both sides; xtrack keeps inactive particles in place)
- after tracking, particles with xtrack `state <= 0` become
  `alive = False`; the wrapper stamps `lost_at_element` through the
  normal `Stage` mechanism (one scene stage = one xtrack line), so the
  ledger stays label-addressable; xtrack's finer `at_element` index is
  preserved in `metadata["xtrack_at_element"]` for diagnostics
- resurrection is forbidden as everywhere: the wrapper AND-masks

## API sketch

`latent_dirac/adapters/xsuite/adapter.py`:

- `ReferenceFrame(p0c_ev: float)` — pydantic model (Model layer)
- `reference_from_state(state) -> ReferenceFrame`
- `to_xtrack_particles(state, frame) -> xtrack.Particles`
- `from_xtrack_particles(particles, species, frame,
  template_state=None) -> ParticleState`
- `track_state(state, line, frame, *, num_turns=1) -> ParticleState`

The placeholder module is deleted in the same change;
`test_only_placeholder_adapters_are_present` is replaced by an
adapter-status test asserting xsuite is real (imports without
NotImplementedError) while geant4/root remain placeholders.

## Packaging and CI

`pyproject.toml` gains `xsuite = ["xtrack>=0.65"]` (pin decided at
implementation time against the installed version). CI: same
non-fatal extra-install step; tests `importorskip("xtrack")`.

## Tests (TDD)

1. Round-trip: state → Particles → state closure (float64) for
   positions, momenta, time, weight, ids; dead-on-entry particles
   survive the round trip as dead.
2. Species mapping: positron and antiproton q0/mass0 correct (sign
   conventions: antiproton q0 = −1).
3. Drift equivalence: an `xtrack.Line` of one drift vs our `drift`
   scene element — final transverse positions agree to documented
   tolerance for a paraxial cloud (this is the calibration handshake,
   not a benchmark).
4. Loss stamping: a line with an aperture restriction kills particles;
   the returned state has them dead, ledger stamped via the stage
   mechanism, survivors untouched.
5. Explicit-reference discipline: `to_xtrack_particles` with a frame
   whose `p0c_ev` differs from the cloud mean still round-trips
   (delta absorbs the offset).
6. Gate flip: the new adapter-status test; geant4/root placeholders
   still raise NotImplementedError.

## Honesty notes

Fidelity tier of tracking results: whatever xtrack provides —
externally computed, cited as "tracked by Xsuite (xtrack vX.Y)";
provenance recorded in `metadata["xtrack_version"]`. The drift
equivalence test is a correctness handshake with a documented
tolerance, not a performance claim.
