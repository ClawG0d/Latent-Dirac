<p align="center">
  <img src="assets/logoD.png" alt="Latent Dirac logo" width="220">
</p>

# Latent Dirac

Latent Dirac is an open modular simulation platform for positron and
antiproton source-to-acceptance modeling.

It provides a lightweight Python core for fast scenario modeling, parameter
sweeps, transport studies, acceptance accounting, and future calibration
against external scientific tools such as Geant4 and Xsuite.

## Focus

- positron source term models
- antiproton surrogate source term models
- relativistic charged-particle transport in electromagnetic fields
- beamline acceptance
- loss accounting
- accepted-yield diagnostics
- optional visualization backends separated from the physics core

Latent Dirac does not try to replace high-fidelity particle-matter simulation
tools. The current package is a source-to-acceptance modeling skeleton with
explicit assumptions and placeholder adapters for future calibration workflows.

## Current Status

This repository contains the first architecture skeleton and minimal working
simulation demos.

Implemented:

- SI-unit constants, unit conversions, and particle species
- `ParticleCloud` as the universal intermediate state
- parameterized positron pair source model
- simplified beta-plus positron source model
- surrogate antiproton source model
- uniform and idealized solenoid fields
- relativistic Boris transport
- aperture and momentum-window acceptance
- staged pipeline loss accounting
- accepted-yield and spectrum diagnostics
- optional Matplotlib and Plotly visualization backends
- placeholder adapters for Geant4, Xsuite, and ROOT

Not implemented yet:

- full electromagnetic or hadronic shower physics
- detailed target engineering
- real facility control systems
- high-yield operational recipes
- material activation or shielding design
- PyVista scientific 3D backend

## Installation

Create a virtual environment and install the package in editable mode:

```bash
python -m venv .venv
.venv/bin/python -m pip install -e ".[dev]"
```

Install optional visualization dependencies:

```bash
.venv/bin/python -m pip install -e ".[dev,viz]"
```

The simulation core does not import Matplotlib or Plotly. Visualization
packages are only loaded by `latent_dirac.viz` backend methods.

## Demos

The demos exercise the first end-to-end workflow:

```text
source model -> field transport -> beamline acceptance -> loss accounting -> report
```

### Demo 1: Positron Capture

This demo samples a parameterized positron pair source, transports the cloud
through an idealized solenoid field, applies an aperture and momentum window,
then reports accepted yield.

```bash
.venv/bin/python examples/positron_capture_demo.py
```

Example output:

```text
Latent Dirac simulation report

Stage accounting:
- solenoid transport: input=200, output=200, transmission=1, losses=0
- aperture: input=200, output=200, transmission=1, losses=0
- momentum window: input=200, output=200, transmission=1, losses=0

Accepted cloud:
- weighted count: 200
- mean kinetic energy: 3.01583 MeV
- accepted yield: 0.02
```

### Demo 2: Antiproton Transport

This demo samples a surrogate antiproton source, transports it through a
uniform magnetic field, applies a momentum acceptance window, and summarizes
the accepted weighted yield.

```bash
.venv/bin/python examples/antiproton_transport_demo.py
```

Example output:

```text
Latent Dirac simulation report

Stage accounting:
- uniform-field transport: input=1, output=1, transmission=1, losses=0
- momentum window: input=1, output=1, transmission=1, losses=0

Accepted cloud:
- weighted count: 1
- mean kinetic energy: 2209.39 MeV
- accepted yield: 2e-05
```

### Demo 3: Optional Report Figures

Install the visualization extra and save static report figures from any
`PipelineResult`, such as the `result` object built in the API sketch below:

```bash
.venv/bin/python -m pip install -e ".[dev,viz]"
```

```python
from latent_dirac.viz.matplotlib_backend import MatplotlibBackend

backend = MatplotlibBackend()
backend.save_all_basic_report_figures(result, "reports/positron_capture")
```

Interactive Plotly figures are available through `PlotlyBackend`:

```python
from latent_dirac.viz.plotly_backend import PlotlyBackend

fig = PlotlyBackend().plot_losses_interactive(result)
fig.show()
```

## Minimal API Sketch

```python
import numpy as np

from latent_dirac.beamline.aperture import Aperture
from latent_dirac.beamline.momentum_window import MomentumWindow
from latent_dirac.core.units import momentum_gev_c_to_si
from latent_dirac.diagnostics.reports import text_report
from latent_dirac.fields.solenoid import SolenoidField
from latent_dirac.pipeline.runner import PipelineRunner
from latent_dirac.pipeline.stage import Stage
from latent_dirac.solvers.relativistic_boris import RelativisticBorisSolver
from latent_dirac.sources.positron_pair import PositronPairSource

source = PositronPairSource(
    primary_count=10_000,
    yield_eplus_per_primary=0.02,
    mean_energy_MeV=3.0,
    energy_spread_MeV=0.4,
    angular_rms_rad=0.03,
    source_sigma_m=1.0e-3,
    bunch_length_s=1.0e-12,
    macro_particles=512,
)

field = SolenoidField(b_tesla=0.8, radius_m=0.05, length_m=0.5)
solver = RelativisticBorisSolver(dt_s=2.0e-12, steps=100)
cloud = source.sample(np.random.default_rng(2026))

result = PipelineRunner(
    stages=[
        Stage("solenoid transport", lambda c: solver.propagate(c, field)),
        Stage("aperture", Aperture(radius_m=0.04, z_m=0.06).apply),
        Stage(
            "momentum window",
            MomentumWindow(momentum_gev_c_to_si(0.001), momentum_gev_c_to_si(0.020)).apply,
        ),
    ]
).run(cloud)

print(text_report(result.stage_results, result.final_cloud, primary_count=10_000))
```

## Visualization

Static report figures:

```python
from latent_dirac.viz.matplotlib_backend import MatplotlibBackend

backend = MatplotlibBackend()
fig = backend.plot_energy_spectrum(result)
backend.save_all_basic_report_figures(result, "reports/basic")
```

Interactive figures:

```python
from latent_dirac.viz.plotly_backend import PlotlyBackend

backend = PlotlyBackend()
fig = backend.plot_losses_interactive(result)
```

See [docs/rendering.md](docs/rendering.md) for the rendering strategy.

## Tests

```bash
.venv/bin/python -m pytest -q
```

The test suite covers species assumptions, unit conversions, particle-cloud
state handling, source models, relativistic motion in uniform fields, Larmor
radius validation, pipeline losses, accepted yield, project positioning, and
optional visualization behavior.

## Documentation

- [Architecture](docs/architecture.md)
- [Physics scope](docs/physics_scope.md)
- [Source models](docs/source_models.md)
- [Solver backends](docs/solver_backends.md)
- [Rendering](docs/rendering.md)
- [Validation plan](docs/validation_plan.md)
- [Safety scope](docs/safety_scope.md)
- [License strategy](docs/license_strategy.md)
- [Roadmap](docs/roadmap.md)

## Safety Scope

Latent Dirac is scoped to open, lightweight source-to-acceptance simulation
architecture and diagnostics. It excludes weaponization scenarios,
energetic-release applications, real facility controls, detailed accelerator
target engineering, high-yield operational recipes, full shower physics,
annihilation physics, material activation, and radiation shielding design.

## License

Apache-2.0. See [LICENSE](LICENSE).
