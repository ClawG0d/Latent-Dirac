# Antimatter Factory Chain Demos Design (D0-D4)

## Objective

Add the company narrative's four physics stages to the README as one
coherent chain: beta-plus decay emission, antiproton production at a
target (surrogate), electrostatic deceleration with dynamic trap capture,
and annihilation as a ledgered endpoint with two-photon visualization.
Decisions confirmed by the user on 2026-07-05.

## D0 — minimal scope amendment

The safety-scope exclusion `annihilation physics` becomes `annihilation
energetics (energy release or deposition calculations; annihilation is
modeled only as a loss endpoint with kinematic two-photon emission for
visualization)`, synchronized verbatim across `docs/safety_scope.md`,
`AGENTS.md`, and the README (the honesty tests parse the bullets and
require the README to contain them). Everything energetic stays excluded;
what opens up is PET-textbook kinematics only.

## D1 — decay emission demo (no new physics)

`examples/scenes/decay_emission.yaml` uses the existing
`BetaPlusPositronSource` (Na-22-like parameters) with a guide solenoid,
collimator, and monitor. New generator coloring mode `"energy"`: trails
colored by initial kinetic energy (the initial state is reproducible via
`build_source(scene).sample(default_rng(scene.seed))`). Title labels the
Beta(3,3) spectrum approximation and the absence of nuclear detail.

## D2 — target-production surrogate demo

`examples/scenes/target_production.yaml` uses the existing
`AntiprotonSurrogateSource` with strong-field transport and a momentum
window. `tools/mpl3d.py` gains pure-visual annotation helpers
`draw_block` (target block) and `draw_beam_arrow` (incoming proton beam
sketch) — drawn, not simulated, and the title says so:
`surrogate accepted-source model | target physics NOT simulated
(Geant4 adapter: roadmap)`.

## D3 — TimeGatedField and the deceleration-capture demo

Physics correction folded into the narrative: magnetic fields do no work;
deceleration is electrostatic here (RF deceleration stays a roadmap
field-library extension).

New capability, both backends:

- `latent_dirac/fields/time_gated.py`: `TimeGatedField(inner, t_on_s,
  t_off_s)` multiplies the inner field by a per-particle time gate
  (`t_on <= t < t_off`); `t_off > t_on` validated.
- `UniformFieldElement` gains optional `t_on_s`/`t_off_s` (None = always
  on); the builder wraps with `TimeGatedField` when set.
- JAX field functions change signature to `(jnp, positions, time_s,
  params)` (mechanical migration of all field fns; the differentiable
  mirror follows); the gate is a `jnp.where`. Parity tests extend to a
  gated scene.

`examples/scenes/decel_capture.yaml` (keV positrons): drift in, a
retarding uniform-E section (analytic dKE = qE dz), a Penning trap, and a
gated exit-barrier E field that switches on after the bunch arrives — the
real accumulator capture sequence. Tests: analytic energy loss, free
passage before the gate closes, axial bouncing and bounded |z| after.

## D4 — annihilation plate and two-photon visualization

New acceptance-type scene element `annihilation_plate(z_m, radius_m)`:
positrons crossing the plane inside the radius are killed (standard
ledger stamp) and recorded as annihilation events — position plus one
isotropically oriented back-to-back photon direction pair (at-rest
approximation; 511 keV appears as a label only, no energetics, no
in-flight annihilation). Events land in `SceneRunResult.annihilations`
keyed by element label (the monitor-collector pattern). Fidelity label:
`parameterized (at-rest two-photon kinematics; no energetics)`. The JAX
backend rejects the element via the existing unsupported-type error for
now. `examples/scenes/annihilation_endpoint.yaml` renders trails ending
at the plate with golden back-to-back photon rays.

## Validation

- honesty tests green throughout (D0 first)
- D1: energy-coloring unit test; D3: dKE = qE dz, gate-open free passage,
  post-gate bounded axial bouncing, NumPy/JAX gated parity; D4: event
  count equals plate kills, photon direction dot product = -1, ledger
  agreement
- all four scenes runnable via `latent-dirac run`; 44-frame assets
  visually inspected; full suite + ruff green; review per sub-step

## Non-Goals

- real target/shower physics (Geant4 adapter), RF deceleration,
  in-flight annihilation kinematics, any energy deposition
