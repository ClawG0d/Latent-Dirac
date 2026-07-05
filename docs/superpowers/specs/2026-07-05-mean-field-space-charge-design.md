# Mean-field space charge design (closed-loop v1, item 4)

Date: 2026-07-05
Status: adopted

## Decision

The Collective component gets its first rung: a **parameterized
uniform-sphere mean-field self-field**, recomputed from the alive cloud
once per transport step on the NumPy pipeline. This is rung 2 of the
declared solver ladder (between single-particle tracking and
self-consistent PIC), scoped to the trap regime.

Fidelity tier: **parameterized**. Validity envelope (explicit, printed
by the scene report):

- electrostatic mean field only; the self-magnetic field is neglected —
  valid for non-relativistic clouds (β ≪ 1, the trap regime)
- the cloud is approximated by ONE uniform-density sphere fitted to the
  alive particles (charge-weighted centroid, R = √(5/3) · r_rms);
  hollow, multi-bunch, or strongly aspherical clouds are outside the
  envelope
- no self-consistent equilibrium iteration: the field is frozen within
  each step (leapfrog-consistent refresh, dt must resolve ω_p)
- macroparticle weights carry physical charge: Q = Σ w·q

## Physics

Uniform sphere of total charge Q, radius R, center r₀; d = x − r₀:

- inside (|d| ≤ R): E = Q d / (4πε₀ R³) — linear
- outside: E = Q d̂ / (4πε₀ |d|²)
- B = 0

Same-species clouds always self-repel (Q and q share sign, so qE points
outward regardless of the species' charge sign — positrons and
antiprotons behave identically here; a sign error would flip this, so
it is pinned by test).

## Architecture

- `latent_dirac/fields/space_charge.py`:
  `UniformSphereSelfField(center_m, radius_m, total_charge_c)` — an
  ordinary pure `Field` (E(x,t), B=0), plus
  `fit_uniform_sphere(state) -> UniformSphereSelfField | None`
  (alive-weighted fit; `None` for clouds with < 2 alive particles or
  zero radius — no field rather than a singular one).
- Scene schema: optional `space_charge: "uniform_sphere"` on
  `uniform_field` and `penning_trap` elements (v1 scope; other elements
  reject it by omission). Default off — space charge is an explicit
  modeling choice, never a hidden default.
- `scene/build._transport_action` gains the per-step branch: when
  space charge is on, each step re-fits the sphere and transports one
  step through `CompositeField([element_field, self_field])`. The Boris
  kernel stays pure — the self-field enters as a plain field argument.
- JAX backend and the differentiable mirror **reject** the parameter
  with a clear error (same pattern as `annihilation_plate`): the
  state-dependent field breaks the static-program assumption; a
  jit-compatible version is a later, separate design.
- Scene report: when enabled, the field status block prints the model
  name, tier, and the validity envelope line.

## Tests (TDD)

1. Field formula: interior linearity (E(R/2) = ½·E(R)), exterior 1/r²
   (E(2R) = ¼·E(R)), continuity at R, direction outward for positive Q.
2. Sign discipline: a cold positron sphere AND a cold antiproton sphere
   both expand (r_rms strictly grows) under pure self-field transport;
   leading-order magnitude check: a surface particle's early
   displacement matches Δr ≈ ½ (qE(R)/γm) t² within a few percent for
   small t.
3. Momentum conservation: total weighted momentum drift from the pure
   self-field stays at float64 noise (the fitted field exerts zero net
   force on the fitting cloud by symmetry of the linear interior +
   radial exterior only when the cloud is the fit source — assert the
   centroid stays put to tight tolerance).
4. Trap qualitative: a centered cold sphere in a `penning_trap` with
   ω_p² ≪ ω_z² stays confined with space charge on, and its free
   expansion (no trap) is suppressed by the trap — ordering assertions,
   not mode frequencies (Dubin cold-fluid mode validation is a roadmap
   item, noted in physics_scope).
5. Ledger/masking: dead particles neither feel nor source the field
   (fit is alive-only); resurrection guard untouched.
6. JAX rejection: building a batched program from a scene with
   `space_charge` raises with a clear message (both jax_scene and the
   differentiable mirror).
7. Schema: unknown model string fails fast; elements without the param
   behave exactly as before (regression: existing scenes unchanged).

## Honesty notes

The scene report line and the docs name the tier and the envelope; the
roadmap keeps rung 3 (self-consistent PIC via WarpX) and the Dubin-mode
validation as explicit next steps. No performance claims.
