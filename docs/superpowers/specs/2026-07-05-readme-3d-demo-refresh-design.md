# README 3D Demo Refresh Design

## Objective

Replace every 2D README demo with a 3D animation rendered from real
simulation output, reorganize the README around the narrative pillars, and
make declarative scenes the way demos are defined — not just one demo among
many. Approved by the user on 2026-07-05 (all cases, full 2D retirement,
unified matplotlib→WebP pipeline).

## Demo lineup (hero + 7)

| # | Demo | Capability shown | Defined by |
|---|------|------------------|-----------|
| hero | Charge-sign splitter 3D (existing) | core physics | `tools/generate_hero_3d_webp.py` (unchanged; two matched species are not scene-expressible yet) |
| 1 | YAML scene → 3D tour | 2b platform narrative | `examples/scenes/scene_tour.yaml`, shown verbatim in the README next to its rendering |
| 2 | Positron spiral capture | source + solenoid | `examples/scenes/positron_capture.yaml` |
| 3 | Dipole + quadrupole beamline | 2a field library | `examples/scenes/dipole_quad_line.yaml` |
| 4 | Wien velocity filter (single-species e+) | crossed E×B fields | `examples/scenes/wien_filter.yaml` |
| 5 | Magnetic mirror bottle | 2d FieldMap + trap teaser | analytic mirror field sampled to a grid, written as a COMSOL-style CSV, loaded through `load_comsol_grid_csv` (honestly labeled synthetic) |
| 6 | Antiproton loss ledger | 2c ledger | `examples/scenes/antiproton_ledger.yaml`; trajectories colored by `lost_at_element` |
| 7 | Magnetic control sweep 3D | parameter sweeps → GPU teaser | reuses `run_sweep`/`make_initial_pair` (two species; not scene-expressible), By ramp per frame |

## Rendering pipeline

One shared matplotlib 3D framework (`tools/mpl3d.py`), extending the proven
hero pipeline: 920×500 frames, rotating camera, progressive trajectory
reveal, Pillow WebP assembly. New capabilities:

- element geometry drawn from scene descriptions: translucent cylinder
  (solenoid), box (dipole/quadrupole), disc with hole (aperture), plane
  (monitor)
- coloring modes: per-species, accepted/lost, and ledger mode (one color
  per killing stage, from `lost_at_element`)
- every title carries the field model and a fidelity/scope note (honesty
  discipline)

`tools/generate_scene_demo_webps.py` holds one config per demo and
regenerates all assets. Each demo also writes an interactive Plotly HTML
next to its WebP (linked from the README; GitHub cannot embed them).

## Retirement

- `tools/generate_demo_webp.py` (Pillow 2D canvas) and the four 2D WebP
  assets are deleted.
- `examples/antiproton_transport_demo.py` is replaced by the ledger demo;
  `examples/positron_capture_demo.py` is rewritten to load its scene YAML.
  `charge_sign_splitter_demo.py` and `magnetic_control_sweep_demo.py` stay
  (hero and sweep reuse them).
- Affected tests are migrated: asset-generator tests target the new tool,
  README-reference tests target the new asset list, field-status tests
  target the rewritten example reports. The honesty-discipline tests in
  `test_project_positioning.py` keep passing throughout.

## README restructure

- badges: add Python versions next to CI and license
- hero animation moves directly under the positioning statement
- each demo: 3D animation + one command + `<details>`-folded text output
  (the sweep's full acceptance table folds into `<details>`)
- demo 1 shows the scene YAML verbatim beside its rendering

## Validation

- generator produces animated WebPs for all demos (small-frame smoke test)
- README references every asset; stale asset references fail the test
- physics assertions: mirror demo particles reflect inside the bottle
  (z-velocity reverses, |z| bounded); Wien demo passes matched-velocity
  particles and cuts mismatched ones; ledger demo final state carries at
  least two distinct killing-stage indices
- full suite green, ruff clean, demos runnable, honesty tests intact

## Non-Goals

- packaged CLI (still deferred), interactive web viewer (Phase 3)
- new physics approximations; kaleido or any new rendering dependency
