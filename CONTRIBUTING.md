# Contributing to Latent Dirac

Thanks for your interest in Latent Dirac. This project is an open interactive
simulation platform for antimatter factories; see
[docs/roadmap.md](docs/roadmap.md) for where it is heading and
[AGENTS.md](AGENTS.md) for the working rules that also apply to human
contributors.

## Development setup

```bash
python -m venv .venv
.venv/bin/python -m pip install -e ".[dev,viz]"
.venv/bin/python -m pytest -q
.venv/bin/python -m ruff check .
```

Both the test suite and `ruff check` must pass before a pull request is
reviewed. CI runs them on Linux and macOS across Python 3.10–3.14.

## Fidelity declaration duty

Every new physics model (source, field, solver, beamline element,
interaction) must declare its fidelity tier in its docstring and
documentation: **placeholder**, **parameterized**, **surrogate**,
**table-based**, or **externally calibrated**. Pull requests that add
physics without a declared tier will be asked to add one before review.

Every physical assumption must be explicit, and every new physics feature
must include tests (analytic cases where possible).

## Honesty discipline

- No comparative performance wording ("fastest", "N× faster") anywhere in
  the documentation without an open, reproducible benchmark reference.
- Performance numbers always carry their settings: integrator, timestep,
  particle count, batch size, approximation tier, and hardware.
- These rules are enforced mechanically by
  `tests/test_project_positioning.py`.

## Safety scope

Contributions must stay inside [docs/safety_scope.md](docs/safety_scope.md).
Pull requests that implement excluded topics (weaponization scenarios,
energetic-release applications, real facility control systems, and the rest
of the exclusion list) will be closed.

## Versioning

Latent Dirac is pre-1.0. Under 0.x semantics, minor releases may break APIs
without deprecation shims; breaking changes are recorded in
[CHANGELOG.md](CHANGELOG.md).

## Releases (maintainers)

Releases publish to PyPI via trusted publishing: bump `version` in
`pyproject.toml`, update `CHANGELOG.md`, tag, and publish a GitHub release.
The `publish.yml` workflow handles the upload; the one-time PyPI pending
publisher setup is documented at the top of that workflow file.
