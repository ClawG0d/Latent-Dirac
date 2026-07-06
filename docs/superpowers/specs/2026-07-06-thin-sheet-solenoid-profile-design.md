# Thin-sheet solenoid profile (smooth fringe fields for demos)

Date: 2026-07-06. Status: accepted.

## Motivation

Every solenoid in the demos is the hard-edge model: uniform Bz inside a
cylinder, exactly zero outside. Trajectories kink non-physically at the
faces, a particle entering parallel to the axis off-center feels no
force at all (v × B = 0 with B purely axial), and the fringe physics
that makes real capture optics look and behave right — funneling,
Larmor rotation from flux linkage (Busch's theorem), adiabatic spiral
compression — is absent. The smooth finite-length solenoid was already
recorded as a design direction (ONBOARDING §七). Hard-edge B is also a
discontinuous integrand for the differentiable objective; a smooth
profile improves autodiff quality as a side effect.

## Model

Ideal thin cylindrical current sheet of radius `R = radius_m`, length
`L = length_m`, centered at `z_c = center_z_m`. On-axis profile (exact
for the thin sheet, elementary functions only):

    zeta_pm(z) = (z - z_c) ± L/2
    f(u)  = u / sqrt(u^2 + R^2)
    b(z)  = (B0/2) * (f(zeta_+) - f(zeta_-))
    b'(z) = (B0/2) * R^2 * ((zeta_+^2+R^2)^-3/2 - (zeta_-^2+R^2)^-3/2)

Off-axis, the first radial order of the axisymmetric expansion:

    B_z(r, z) = b(z)
    B_r(r, z) = -(r/2) * b'(z)
    E = 0

This pair is **exactly divergence-free at every point** — it is
curl(A) for the vector potential `A_phi = r b(z) / 2` — so flux and
canonical-angular-momentum physics are exact, and there is no field
blow-up anywhere. It is curl-free only to first order in r (the
truncation error is O(r * b'')), so accuracy degrades off-axis:
intended for `r ≲ radius_m`, stated in the fidelity note.

`b_tesla` is the **sheet strength** `B0 = mu0 n I` (the infinite-length
interior field). This keeps the integrated on-axis strength exactly

    ∫ B_z(0, z) dz = B0 * L

— identical to the hard-edge element with the same parameters, so the
two profiles are equivalent in integrated optics strength. The field at
the center of a short solenoid is `B0 * L / sqrt(L^2 + 4 R^2) < B0`
(≈ 0.96 B0 for the scene-tour geometry); documented in the class
docstring.

Fidelity tier: parameterized (analytic thin-sheet model, first-order
radial fringe).

## Rejected alternatives

- **Exact thin-sheet solution (complete elliptic integrals)**: exact at
  all r, but `ellipk`/`ellipe` are not available in `jax.numpy` /
  `jax.scipy.special` — it would break the one-kernel-two-backends
  symmetry. The paraxial form is elementary and jit/vmap/grad-clean.
- **Second radial order** (`B_z = b - (r²/4) b''`, `B_r = -(r/2) b' +
  (r³/16) b'''`): also exactly divergence-free, more accurate in-bore,
  but B_z grows quadratically with r off the bore — escaped (lost)
  particles far from the axis would see unbounded nonsense field unless
  clamped. First order keeps B_z bounded everywhere and the model
  robust for demo clouds with escapees; the field-map route exists for
  higher fidelity.
- **New element type**: rejected in review discussion — a `profile`
  enum on the existing `solenoid` element keeps existing scenes valid
  with zero changes and reads better in YAML.

## Schema and wiring

- `SolenoidElement` gains `profile: Literal["hard_edge", "thin_sheet"]
  = "hard_edge"`. Default preserves every existing scene bit-for-bit.
- `fields/solenoid.py`: new `ThinSheetSolenoidField(BaseModel, Field)`
  next to `SolenoidField`, dual-shape query convention, validators
  shared in spirit (positive radius/length).
- `scene/build.py::_base_field_for`: solenoid branch dispatches on
  `element.profile`.
- `backends/jax_scene.py`: new `_thin_sheet_solenoid_field` (same
  algebra under `jnp`), plus a `_field_fn_for(element)` helper that
  dispatches `solenoid` on `element.profile` and falls back to
  `_FIELD_FNS[element.type]` otherwise. `_make_simulator` uses the
  helper. `_SWEEPABLE_PARAMS["solenoid"]` is unchanged — all four
  numeric params parameterize both profiles, so sweeps and overrides
  work identically.
- `backends/differentiable.py`: the mirrored element loop switches its
  `_FIELD_FNS[element.type]` lookup to the shared `_field_fn_for`
  (one-line mirror change; profile is static at trace time).
- `viz/scene_3d.py`: fidelity hover for solenoid becomes
  profile-aware via a `_fidelity_label(element)` helper (geometry
  unchanged; the wireframe cylinder still marks the sheet).
- Docs: `docs/scene_schema.md` documents `profile`; README
  "Implemented" gains a bullet; CHANGELOG entry; ONBOARDING §七 flips
  the smooth-solenoid direction to done.

Demo scenes are NOT switched in this spec — the visual rollout (which
scenes adopt `thin_sheet`, WebP regeneration, annihilation animation)
is the companion demo-visuals spec, so assets regenerate once.

## Validation (TDD)

1. Long-solenoid limit: `b(z_c) -> B0` for `L = 100 R` (rtol 5e-4).
2. Integrated strength: dense trapezoid of `b(z)` over ±40 L equals
   `B0 * L` (rtol 1e-4; the dipole z^-3 tails beyond ±40 L contribute
   ~7e-6 relative).
3. Exactly divergence-free: central finite differences (h = 1e-6 m) at
   random points in and out of the bore, including the former hard
   edges, abs tol 1e-6 T/m — three orders above the finite-difference
   truncation floor and ~7 orders below the O(b') ~ 8 T/m signal any
   sign or factor error in the fringe would produce.
4. Fringe funnels: below the entrance face at x > 0, B_x < 0 (field
   lines converge into the bore); symmetric above the exit.
5. Continuity: no jump in any component along a radial line crossing
   r = R and along an axial line crossing the faces (the model is
   smooth everywhere by construction — this pins the implementation).
6. Busch/canonical angular momentum: Boris-propagate a positron
   entering off-axis parallel to the axis from the weak-field region
   into the bore; `P_phi = (x p_y - y p_x) + q r^2 b(z) / 2` is
   conserved along the whole trajectory to 1% of the q r0^2 B0 / 2
   scale (the tolerance floor is the Boris half-step staggering skew
   ~0.5% when evaluating P_phi from a single synchronized state — the
   true conservation drift sits far below; an implementation sign or
   factor error would deviate at ~50% of scale). The instantaneous
   azimuthal rate inside matches the Larmor rate
   `-q b(z) / (2 gamma m)` pointwise at r > 0.5 r0 (rtol 5e-2), and
   transverse momentum is genuinely acquired through the fringe.
7. Schema: default is `hard_edge` (existing scene mapping unchanged),
   `thin_sheet` accepted, unknown profile rejected.
8. End-to-end NumPy: the same scene run under the two profiles
   produces measurably different transport (rtol-only comparison —
   SI momenta are ~1e-21, so allclose's default atol would mask any
   difference). The axis-parallel zero-kick artifact of `hard_edge`
   is asserted at solver level in the companion test.
9. JAX parity: the thin-sheet scene matches the NumPy pipeline
   element-wise (x64, rtol 1e-9), and a `b_tesla` batch override runs.
10. Differentiable objective: value_and_grad w.r.t.
    `solenoid.b_tesla` is finite and matches central finite
    differences (rtol 1e-3) on a thin-sheet scene.
11. Viz: hover label for a thin-sheet solenoid carries the
    profile-specific fidelity note (skipped without plotly).
