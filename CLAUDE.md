@AGENTS.md

# Claude session notes

AGENTS.md (imported above) is the single source of truth for physics
rules, engineering rules, scope, and the current phase. This file adds
only Claude-Code-specific operational notes.

## Commands

- tests: `.venv/bin/python -m pytest -q`
- lint: `.venv/bin/python -m ruff check .` (run `ruff format` on new files
  only; several pre-existing files are not format-clean by choice)
- demo assets: `.venv/bin/python tools/generate_hero_3d_webp.py` and
  `.venv/bin/python tools/generate_scene_demo_webps.py` (44 frames;
  extract a late frame and inspect visually before committing assets)
- CLI: `.venv/bin/latent-dirac run|render <scene.yaml>`
- JAX tests enable x64 inside the test files; this machine is CPU-only —
  never produce performance numbers here (honesty discipline requires
  hardware-labeled numbers from Linux CUDA)

## Workflow (every feature)

1. Design spec to `docs/superpowers/specs/YYYY-MM-DD-<topic>-design.md`
2. TDD: write the failing test first and confirm it fails for the right
   reason
3. Full `pytest -q` + `ruff check .` green
4. `superpowers:code-reviewer` review before committing — it has caught
   real physics bugs (e.g. a silent antiproton sign error) in 7/7 runs;
   do not skip
5. Conventional commit (`feat:` / `chore:` / `refactor!:` / `docs:`) with
   the Co-Authored-By trailer, then push

## Traps (learned the hard way)

- The safety-scope exclusion bullets must stay **verbatim-identical**
  across `docs/safety_scope.md`, `AGENTS.md`, and the README —
  `tests/test_project_positioning.py` parses the bullets and asserts the
  README contains each one. Change one file, change all three.
- Before editing README wording, read `tests/test_project_positioning.py`
  (comparative performance phrases without a benchmark reference fail CI).
- `latent_dirac/backends/differentiable.py` mirrors the element loop in
  `jax_scene._make_simulator` — new element types must be added to both.
- JAX field functions have the signature `(jnp, positions, time_s,
  params)`; the Boris kernel stays a pure, `xp`-generic function.
- State containers are pytree dataclasses, never pydantic (pydantic owns
  the Model/scene layer only); kernels work in dimensionless u = p/(m c),
  SI exists only at State boundaries (float32 + SI momenta underflow).
