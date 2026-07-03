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

## Future Backend

PyVista is a future optional backend for scientific 3D beamline geometry and
field-line visualization. It is not included yet.
