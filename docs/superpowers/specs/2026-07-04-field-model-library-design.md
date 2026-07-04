# Field Model Library Design

## Objective

Extend Latent Dirac's classical electromagnetic field layer from isolated
uniform and solenoid fields into a small, composable field model library for
relativistic charged-particle transport.

The first implementation phase adds:

- `CompositeField`
- `DipoleField`
- `QuadrupoleField`

These models keep the solver contract unchanged: every field object answers
`E(x, t)` in V/m and `B(x, t)` in tesla.

## Scope

This phase stays inside classical relativistic electromagnetic transport. It
supports source-to-acceptance studies where fields steer, focus, defocus, or
combine beamline effects before acceptance diagnostics are applied.

The implementation will live under `latent_dirac/fields/` and follow the
existing `Field` interface in `latent_dirac/fields/base.py`.

## Field Models

### CompositeField

`CompositeField` combines multiple field objects and returns the sum of their
electric and magnetic components:

```text
E_total(x, t) = sum(E_i(x, t))
B_total(x, t) = sum(B_i(x, t))
```

It enables beamline-like configurations without changing
`RelativisticBorisSolver`.

### DipoleField

`DipoleField` represents an idealized hard-edge magnetic dipole. The first
version uses a uniform magnetic field inside a configurable longitudinal
extent, and zero field outside that extent.

Required parameters:

- `B_vector_t`
- `length_m`
- `center_z_m`, default `0.0`

The field is useful for bending and charge-sign separation demos.

### QuadrupoleField

`QuadrupoleField` represents an idealized hard-edge magnetic quadrupole. The
first version uses the linear transverse field:

```text
B_x = gradient_t_m * y
B_y = gradient_t_m * x
B_z = 0
```

inside a configurable longitudinal extent, and zero field outside that extent.
Changing the sign of `gradient_t_m` swaps the focusing and defocusing planes.

Required parameters:

- `gradient_t_m`
- `length_m`
- `center_z_m`, default `0.0`

This is a beam optics model, not a magnet engineering model.

## Public API

The intended imports are:

```python
from latent_dirac.fields.composite import CompositeField
from latent_dirac.fields.dipole import DipoleField
from latent_dirac.fields.quadrupole import QuadrupoleField
```

Optional convenience exports may be added to `latent_dirac/fields/__init__.py`
if they match the existing package style.

Example usage:

```python
field = CompositeField(
    fields=[
        DipoleField(B_vector_t=[0.0, 0.35, 0.0], length_m=0.2),
        QuadrupoleField(gradient_t_m=8.0, length_m=0.1, center_z_m=0.2),
    ]
)

cloud_out = RelativisticBorisSolver(dt_s=2.0e-12, steps=100).propagate(cloud, field)
```

## Validation

Add focused tests for:

- `CompositeField` sums electric and magnetic field components from multiple
  fields.
- `DipoleField` returns its configured field inside its hard-edge length and
  zero outside.
- `QuadrupoleField` returns the expected linear field at known positions.
- `QuadrupoleField` reverses focusing/defocusing sign when the gradient sign
  changes.
- Existing uniform-field, solenoid, and solver tests continue to pass.

Future validation can add trajectory-level optics checks, such as weak
quadrupole focusing trends and dipole bend direction tests.

## Documentation

Update field-related docs to explain that the current field layer is:

- classical
- SI-unit based
- callable by position and time
- composable
- intentionally separate from quantum wavefunction or full field-solver work

The roadmap should list RF fields and field-map loading as later field-library
extensions.

## Non-Goals

This phase does not implement:

- Maxwell equation solvers
- Dirac, Klein-Gordon, or wavefunction evolution
- spin tracking or BMT precession
- RF cavities
- field-map interpolation
- detailed magnet design
- material interaction physics
- facility control systems

Those may be separate future specs.

## Acceptance Criteria

The phase is complete when:

- the three new field models exist under `latent_dirac/fields/`
- their behavior is covered by pytest tests
- docs mention the expanded field-library direction
- the full test suite passes
- core modules still do not import visualization packages

