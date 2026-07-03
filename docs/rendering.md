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

## README Animations

The animated README demos are generated as WebP assets under `assets/demos/`.
They are built from deterministic source and transport runs, then rendered by
the repository tool:

```bash
.venv/bin/python tools/generate_demo_webp.py
```

The tool imports Pillow only when it runs. Core physics modules do not depend
on the animation generator.

## Future Backend

PyVista is a future optional backend for scientific 3D beamline geometry and
field-line visualization. It is not included yet.
