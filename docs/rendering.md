# Rendering

Rendering in Latent Dirac is optional and separated from physics.

Core simulation modules do not import visualization packages. Source terms,
fields, solvers, beamline acceptance, pipeline loss accounting, and diagnostics
remain usable with only the core dependencies.

Install the optional rendering dependencies with:

```bash
pip install "latent-dirac[viz]"
```

## Current Backends

`MatplotlibBackend` provides static report figures:
- energy spectrum
- phase-space scatter plot
- losses by pipeline stage
- basic report figure export

`PlotlyBackend` provides interactive figures:
- 3D trajectory plot
- interactive phase-space scatter plot
- interactive losses by stage

`latent_dirac.viz.scene_3d.render_scene_3d` renders a declarative scene
(see [scene_schema.md](scene_schema.md)) with element wireframes, recorded
trajectories, accepted/lost final states, and per-element fidelity labels
in the hover text.

`latent_dirac.viz.scene_3d.render_scene_animation` renders the same scene
as a play/pause + scrub animation of the recorded cloud traversing the
beamline (lost particles freeze at their loss point — the loss ledger in
motion); it needs `run_scene(..., record_trajectories=True)`. From the
CLI, add `--animate`:

```bash
latent-dirac render scene.yaml -o scene_anim.html --animate
```

`latent_dirac.viz.field_3d.render_field_magnitude_3d` renders |B| of a
table-based field map as a translucent volume, labeled with its fidelity
tier.

## README Animations

Every README demo is a 3D WebP animation rendered from real simulation
output through the shared matplotlib pipeline in `tools/mpl3d.py`. Most
demos are defined by the declarative scenes under `examples/scenes/`; the
scene-driven ones also export interactive Plotly HTML files next to the
WebPs.

```bash
.venv/bin/python tools/generate_hero_3d_webp.py
.venv/bin/python tools/generate_scene_demo_webps.py
```

The tools import Pillow and Matplotlib only when they run. Core physics
modules do not depend on the animation generators.

## Future Backend

PyVista is a future optional backend for scientific 3D beamline geometry and
field-line visualization. It is not included yet.
