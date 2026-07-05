# Source Models

`PositronPairSource` is a parameterized source term. It uses user-provided
primary count, yield, kinetic-energy distribution, angular spread, source size,
and bunch length. It is explicitly not full shower physics.

`BetaPlusPositronSource` is a simplified beta-plus source. The first version
uses a Beta(3, 3) approximation scaled to endpoint kinetic energy, isotropic
directions, and a uniform spherical source volume. It omits nuclear detail,
material effects, and annihilation physics.

`AntiprotonSurrogateSource` is a surrogate accepted-source model. It samples
from user-provided primary count, accepted yield, momentum distribution,
angular spread, source size, and bunch length. It is not a detailed target
model.

`AntiprotonYieldTableSource` is a table-based source. It replays the
phase space of antiprotons recorded by the offline engine yield-table
generator (`engine/yieldgen`: vanilla Geant4, FTFP_BERT, proton on an
iridium stand-in target) and carries the engine provenance four-tuple
(Geant4 version, physics list, datasets, patch list) in its metadata.
Weights normalize the table's `n_primaries` to a user-declared physical
proton count. There is no runtime engine coupling; the CSV contract is
defined in the yieldgen demo spec.
