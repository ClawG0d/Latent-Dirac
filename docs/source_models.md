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
