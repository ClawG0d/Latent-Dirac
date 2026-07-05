# Safety Scope

Latent Dirac is scoped to open simulation architecture and diagnostics for
antimatter facility design studies. Particle-matter interaction physics
(showers, stopping, energy deposition) is delegated to the vendored
vanilla Geant4 engine track and is in scope only as diagnostics; the red
lines below sit at the application layer. No engine build or runtime
coupling ships yet; adapters remain placeholders.

Out of scope:
- weaponization scenarios
- energetic-release applications (antimatter as an energy source or destructive payload in any form)
- real facility control systems
- detailed accelerator target engineering (thermal, mechanical, and materials design of production targets)
- high-yield operational recipes
- in-house shower physics (electromagnetic and hadronic showers are delegated to the vendored vanilla Geant4 engine; the Python core does not implement them)
- annihilation energetics as a figure of merit (energy deposition is in scope only as an engine-computed diagnostic; the Python core models annihilation only as a loss endpoint with kinematic two-photon emission for visualization)
- material activation
- radiation shielding design

Source models must state whether they are placeholder, parameterized,
surrogate, table-based, or externally calibrated. Engine-derived results
must carry the Geant4 version, physics list, dataset versions, and patch
list (empty while the vendored tree is frozen).

The digital-twin direction is limited to offline forward simulation, replay
of measured data, and historical parameter calibration. Latent Dirac provides
no real-time control loops and no interfaces that write back to a facility.
