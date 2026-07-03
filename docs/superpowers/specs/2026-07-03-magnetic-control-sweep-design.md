# Magnetic Control Sweep Demo Design

## Objective

Add a README-facing demo that shows magnetic control of matched positron and
electron clouds. The demo scans a uniform transverse magnetic field and reports
how charge-sign separation, aperture acceptance, and losses change with field
strength.

## Scope

The demo uses only the existing lightweight physics core:

- matched positron and electron `ParticleCloud` inputs from the charge-sign
  splitter example
- `UniformField` with `B_vector_t = [0, By, 0]`
- `RelativisticBorisSolver`
- a fixed transverse aperture acceptance check
- simple loss and accepted-fraction diagnostics

The field scan range is `By = 0.0 T` through `0.6 T`. Each field point reuses
the same initial matched clouds so changes are attributable to the magnetic
field, not sampling noise.

## Non-Goals

This demo does not model particle collisions, annihilation physics, material
interactions, target engineering, release physics, facility controls, or
operational optimization recipes. It is a charge-sign transport and acceptance
diagnostic only.

## User-Facing Behavior

Add `examples/magnetic_control_sweep_demo.py`. Running it prints a compact text
report with a table containing:

- `By [T]`
- positron mean `x`
- electron mean `x`
- transverse separation
- accepted fraction
- loss fraction

The report should explicitly state the aperture radius, solver model, field
model, and that each field point uses matched initial clouds.

## README Demo

Add a new README demo section before the optional report-figures section. The
primary story is that stronger transverse magnetic fields produce larger
opposite charge-sign bending. Aperture loss and accepted fraction are presented
as diagnostics, not as a prescriptive optimization workflow.

The README references a new animated WebP asset:

```text
assets/demos/magnetic_control_sweep.webp
```

The existing optional report-figures demo is renumbered after the new sweep
demo.

## Animated WebP

Extend `tools/generate_demo_webp.py` to generate
`magnetic_control_sweep.webp`.

Each frame represents one field strength in the sweep. The animation shows:

- positron and electron tracks in the same plot
- fixed aperture boundaries
- the current `B vector [T]`
- current separation
- accepted and lost counts or fractions
- a bottom stage bar showing source, field sweep, aperture, losses, and report

The WebP stays lightweight and uses the existing Pillow-based generator. No
runtime visualization package is imported by core simulation modules.

## Architecture

The sweep logic should be testable without generating images. Prefer a small
data object or plain dictionaries for per-field results. Keep the example
module independent from README animation code, so the text demo remains usable
without Pillow.

Suggested functions:

- `run_sweep(...) -> list[dict[str, float]]`
- `format_report(results, ...) -> str`
- `run_report(...) -> str`

The WebP generator may call the same `run_sweep` helpers where useful, but it
can also generate frame-level clouds directly to draw trajectories.

## Error Handling

Validate that particle count, aperture radius, field list, time step, and step
count are positive where applicable. Empty field scans should raise
`ValueError` with a clear message.

## Testing

Add tests for:

- sweep output contains the expected number of field points
- separation increases from the zero-field point to the strongest field point
- accepted fraction and loss fraction stay in `[0, 1]`
- `run_report` includes magnetic field status, aperture status, and the sweep
  table
- README references `assets/demos/magnetic_control_sweep.webp`
- WebP generator includes and creates the fourth animated asset

Keep existing project-positioning and optional-viz import tests passing.

## Verification

Before committing implementation:

- run the full pytest suite
- run `compileall` over `latent_dirac`, `examples`, and `tools`
- run `git diff --check`
- verify all README WebP assets are animated and have nonzero size
- visually inspect at least one generated sweep WebP frame

