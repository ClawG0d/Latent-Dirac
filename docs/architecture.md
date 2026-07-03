# Architecture

Latent Dirac is organized around a universal `ParticleCloud` state. Source
terms create clouds, solvers transport clouds through electromagnetic fields,
beamline elements update the `alive` acceptance mask, and diagnostics summarize
accepted yield and spectra.

The first skeleton is intentionally lightweight:
- `core`: constants, unit conversion, species, and random helpers
- `state`: particle cloud and trajectory containers
- `sources`: parameterized, simplified, and surrogate source models
- `fields`: uniform and idealized solenoid field models
- `solvers`: relativistic Boris transport
- `beamline`: aperture and momentum-window acceptance
- `pipeline`: staged execution and loss accounting
- `diagnostics`: accepted-yield and text-report utilities
- `adapters`: placeholders for future optional Geant4, Xsuite, and ROOT integrations

External scientific ecosystems are not integrated in this phase.
