# CLI and Hello-Beamline Design (Spec 3c)

## Objective

Close the scope deliberately deferred from Spec 2b: a `latent-dirac`
console script and the 60-second hello-beamline. One command from a YAML
scene to a text report; one command to an interactive 3D HTML.

## Design

- `[project.scripts] latent-dirac = "latent_dirac.cli:main"`.
- `latent-dirac run <scene.(yaml|json)>` — loads the scene, runs the
  pipeline, prints the scene report (stage accounting, loss ledger, field
  status, scope note). Core dependencies only.
- `latent-dirac render <scene> -o <out.html> [--max-particles N]` — runs
  with trajectory recording and writes the interactive Plotly rendering;
  requires the `viz` extra and says so clearly when missing.
- Errors (missing file, bad suffix, schema validation, missing extra)
  print a one-line message to stderr and exit non-zero; no tracebacks for
  expected failures.
- `scene_report`/`field_status_lines` move from `examples/scene_report.py`
  into `latent_dirac.diagnostics.scene_report` (the CLI needs them and
  `examples/` is not packaged); example scripts import from the package.
- `examples/scenes/hello_beamline.yaml`: a compact scene for the README's
  "60-second hello beamline" — install, one `run`, one `render`.

## Validation

- `run` prints the report for the hello scene (exit 0).
- `render` writes a non-empty HTML (plotly importorskip); missing-viz path
  raises the clear message.
- Missing file and invalid schema exit non-zero with a one-line stderr
  message.
- Console entry point resolves (`latent-dirac --help` via the installed
  script or `python -m latent_dirac.cli`).
- Existing example scripts keep passing with the relocated report module.

## Non-Goals

- sweep/optimize subcommands (the batched backend and evaluator are
  library APIs for now)
- interactive viewer application (still Phase 3 later work)
