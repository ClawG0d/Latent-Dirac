# Solver Backends

The first solver backend is `RelativisticBorisSolver`.

It updates relativistic momentum and position for multiple particles using:
- momentum in kg m/s
- gamma from momentum and mass
- velocity from relativistic momentum
- Lorentz force through electric and magnetic fields

The solver is validated against uniform magnetic-field behavior, including
kinetic-energy preservation and Larmor-radius motion.

Future backends can be added behind the same solver interface.
