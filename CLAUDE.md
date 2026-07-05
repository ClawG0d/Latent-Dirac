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

## Vendored Geant4 tree

- `geant4-v11.4.2/` is read-only vendored upstream code (vanilla
  v11.4.2). Never edit, lint, format, or run tests over it.
- Exclude it from Grep/Glob searches by default — it holds 17k+ files;
  a plain `du`/`find` across it has hit the 2-minute command timeout.
- ruff (`extend-exclude`), pytest (`testpaths`), and packaging
  (`MANIFEST.in prune`) are already configured around it; keep all three
  exclusions in place.
- Commits that touch the vendored tree use the `vendor:` prefix and must
  keep it byte-identical to the upstream release (`.gitattributes
  -text` prevents EOL rewriting).

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
6. After pushing, confirm the run's final CI status
   (`gh run list --repo ClawG0d/Latent-Dirac --limit 1`) — the local venv
   is Python 3.14, but the matrix tests 3.10–3.14 (a `tomllib` import
   once broke only the 3.10 jobs and went unnoticed for five pushes)

## Traps (learned the hard way)

- The safety-scope exclusion bullets are pinned in **four** places:
  `EXPECTED_EXCLUSIONS` in `tests/test_project_positioning.py` (exact
  tuple equality) plus verbatim-identical copies in
  `docs/safety_scope.md`, `AGENTS.md`, and the README (substring
  asserts). Change one, change all four — and scope changes require a
  positioning spec first.
- Before editing README wording, read `tests/test_project_positioning.py`
  (comparative performance phrases without a benchmark reference fail CI).
- `latent_dirac/backends/differentiable.py` mirrors the element loop in
  `jax_scene._make_simulator` — new element types must be added to both.
- JAX field functions have the signature `(jnp, positions, time_s,
  params)`; the Boris kernel stays a pure, `xp`-generic function.
- State containers are pytree dataclasses, never pydantic (pydantic owns
  the Model/scene layer only); kernels work in dimensionless u = p/(m c),
  SI exists only at State boundaries (float32 + SI momenta underflow).
