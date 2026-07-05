# Safety Scope

Latent Dirac is scoped to open, lightweight source-to-acceptance simulation
architecture and diagnostics.

Out of scope:
- weaponization scenarios
- energetic-release applications
- real facility control systems
- detailed accelerator target engineering
- high-yield operational recipes
- full shower physics
- annihilation physics
- material activation
- radiation shielding design

Source models must state whether they are placeholder, parameterized,
surrogate, table-based, or externally calibrated.

The digital-twin direction is limited to offline forward simulation, replay
of measured data, and historical parameter calibration. Latent Dirac provides
no real-time control loops and no interfaces that write back to a facility.
