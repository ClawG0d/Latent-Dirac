# Residual-gas storage lifetime design

Date: 2026-07-06
Status: adopted

## Decision

A new scene element `residual_gas_loss` models the finite storage
lifetime of trapped antiparticles: while held in a region at finite
vacuum, they annihilate on residual-gas molecules at a rate set by the
gas density and the (velocity-dependent) annihilation cross-section.
This is the first antimatter-native loss physics beyond geometry — the
"how long does it live" question that has no positron-machine analogue
(a stored proton just scatters off residual gas; a stored antiproton is
destroyed by it).

The element is a **loss endpoint**, fully inside the safety scope
(annihilation modeled only as a ledgered loss — no energetics, no
deposition). It reuses the existing per-particle kill ledger, so a
storage-stage annihilation becomes an addressable ledger event exactly
like an aperture loss.

## Model (fidelity tier: parameterized)

Exponential survival over a hold time:

    p_survive = exp(-hold_time_s / mean_lifetime_s)

Each alive particle draws an independent uniform and survives if it
falls under `p_survive`; the rest are killed. `mean_lifetime_s` (τ) is
a **direct input**, not derived — the honest parameterized tier. The
physically-derived form τ = 1 / (n σ(v) v), with number density n from
the vacuum pressure and σ(v) the annihilation cross-section, is the
future upgrade; it needs a curated, provenance-tracked σ(v) dataset
(the same buffer-gas cross-section research the roadmap defers) and
would graduate the element toward `table_based`. The docstring records
this path.

Stochastic, not weight-decay: the choice is deliberate. Stochastic kill
(a) makes each annihilation a real, addressable per-particle ledger
event (pillar 3), (b) reuses the existing `lost_at_element` channel so
the "by killing element" and "weighted stage" loss views stay
consistent, and (c) keeps the array shape static (killed particles stay
in place with `alive=False`, exactly like `aperture`), preserving the
vmap invariant. The cost — non-differentiability of the hard kill —
matches how the hard `aperture` / `momentum_window` already behave.

Differentiable form (shipped alongside): the soft objective models the
element's EXPECTED survival, a smooth factor `exp(-hold_time_s /
mean_lifetime_s)` applied uniformly, differentiable in both parameters.
This is the general principle for stochastic losses — hard = a random
draw, soft = its expectation — and it lets a design loop jointly
optimize capture efficiency against storage survival (the
antimatter-native objective). The batched simulator still rejects the
element (stochastic kill has no static-program form); the divergence
between the mirror pair is intentional and documented at both sites.

## Behaviour

- RNG: seeded per stage, `default_rng(scene.seed + 6229 + stage_index)`
  (distinct offset from the annihilation-plate stream), so runs are
  reproducible and independent across stages.
- Survivors: `time_s += hold_time_s` (they aged through the full hold).
  Killed particles keep their entry time (death time within the hold is
  not modeled — a stated fidelity limitation).
- Dead-on-entry particles are untouched: the survival mask only ANDs the
  alive mask (`apply_alive_mask`), never resurrects; `Stage.run` stamps
  only the newly dead.
- Species-agnostic: applies to any stored antiparticle (antiproton on
  gas nuclei, positron on gas electrons); τ is the caller's input.
- `hold_time_s = 0` → all survive; `hold_time_s ≫ τ` → nearly all die.

## Schema and integration

- `ResidualGasLossElement(type="residual_gas_loss", mean_lifetime_s>0,
  hold_time_s>=0)` in `scene/schema.py`, added to the `ElementSpec`
  discriminated union.
- `scene/build.py`: a `_residual_gas_loss_action(element, rng)` closure,
  wired like `annihilation_plate` (own seeded RNG per stage).
- NumPy pipeline only. The JAX backend rejects it automatically (not in
  `_SWEEPABLE_PARAMS`, so `_base_params` raises the existing
  "not supported by the JAX backend yet" error) — no new code, but a
  test pins it.
- No scene-report field line (it is not a field); the stage accounting
  and loss ledger already surface the attenuation.

## Tests (TDD)

1. Survival fraction ≈ exp(-hold/τ) over large N (binomial tolerance).
2. Determinism: same seed → identical survivor set; different seed →
   different set, same fraction within tolerance.
3. Ledger: killed particles carry this stage's index in
   `lost_at_element`; survivors keep -1.
4. Time: survivors' `time_s` advanced by `hold_time_s`; killed
   particles' time unchanged.
5. Limits: `hold_time_s=0` keeps all alive; `hold_time_s=20τ` kills
   nearly all.
6. Validation: `mean_lifetime_s<=0` and `hold_time_s<0` fail fast.
7. Dead-on-entry stay dead and are not re-stamped.
8. JAX backend raises on a scene containing the element.
9. Works for both antiproton and positron clouds.

## Honesty notes

Parameterized tier stated in the element docstring and CHANGELOG; τ is
an input, so no physics is claimed beyond "exponential survival at the
given lifetime". The cross-section-derived upgrade and its data
provenance requirement are recorded here and in the roadmap.
