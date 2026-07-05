# License Strategy

First-party code in Latent Dirac is licensed under Apache-2.0 (see
`LICENSE`). The vendored vanilla Geant4 v11.4.2 tree at `geant4-v11.4.2/`
keeps its own Geant4 Software License (text inside the vendored tree);
any redistribution must reproduce that license and the Geant4
Collaboration attribution recorded in `NOTICE`.

The pip package should remain lightweight and permissively licensed; the
vendored engine tree is not part of the Python distribution. Optional
external integrations must stay behind adapters so their dependencies and
licenses do not become mandatory for the core simulation path.

Any future adapter should document its upstream dependency, license, install
extra, and data exchange boundary.
