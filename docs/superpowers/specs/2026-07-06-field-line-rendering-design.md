# Field-line rendering (make the model fields visible)

Date: 2026-07-06. Status: accepted.

## Motivation

Every demo draws trajectories and element wireframes, but the fields
themselves — the actual physics content of each scene — are invisible.
The analytic field models are exact functions; their field lines can be
rendered honestly (a streamline of the model field is a faithful
diagnostic of the model, exactly like the mirror demo's flux-tube
outline). This spec adds a small field-line library and per-element
line bundles to both renderers.

## Design

New module `latent_dirac/viz/field_lines.py` (pure NumPy, no plotting):

- `field_line(field, seed, length_m, step_m, direction, kind, t_s)`:
  midpoint (RK2) integration of dX/ds = F/|F| for F = B or E of any
  `Field`; stops early where |F| falls below a floor (hard edges, zero
  fields). Returns an (N, 3) polyline.
- `element_field_line_bundles(element, field, extent)`: per-element
  seed strategies returning `(kind, polyline)` tuples, `extent` being
  the beam bounding half-widths so seeding scales with each scene:
  - `solenoid` (both profiles): a seed ring inside the bore upstream,
    integrated downstream — thin-sheet lines funnel in through the
    fringe and splay out past the exit; hard-edge lines exist only
    inside the envelope (the integrator stops at the edge, which is
    itself an honest picture of the hard-edge model).
  - `uniform_field`: straight B lines through a small transverse grid;
    straight E lines (Wien's crossed pair renders as two orthogonal
    line families).
  - `penning_trap`: straight axial B lines plus quadrupole E lines
    (they satisfy r^2 |z - zc| = const, bending from the midplane
    toward the endcaps) — B confines radially, E confines axially,
    now visible.
  - `dipole`: straight lines along the dipole vector inside the gap.
  - `quadrupole`: transverse hyperbolic B lines at the element center
    (the field has no z-component; the integrator stays in-plane).
- Rendering: `mpl3d.draw_field_polylines` (WebP demos; B lines steel
  blue, E lines amber, thin and translucent) and `_field_line_traces`
  in `render_scene_3d` and the animated viewer (HTML; hover text
  "field lines of the model field" plus the element fidelity label).
- Dedupe: `field_elements_for_lines(scene)` drops physics-identical
  repeated field elements (label/steps excluded from the fingerprint)
  so a scene that re-declares one physical trap per pipeline stage
  draws its lines once — stacked translucent copies would render
  near-opaque.

Honesty: lines are computed from the exact model field being
simulated — no artistic smoothing; the hard-edge/ideal/global
idealizations show up as-is (e.g. hard-edge lines end abruptly at the
envelope, ideal-trap lines extend to the frame edge).

## Validation

- Uniform field: the line is straight along B to 1e-12.
- Thin-sheet solenoid: a line seeded off-axis in the fringe converges
  radially inside the bore (r decreases), and the polyline tangent is
  parallel to the local B everywhere (cross product below tolerance).
- Penning trap E line: r^2 |z - zc| is conserved along the line
  (analytic invariant of the quadrupole field-line ODE).
- Hard edge: a line seeded outside the envelope has length 1 (stops
  immediately); a line inside stops at the boundary.
- Renderer smoke: scene_3d gains field-line traces for a
  field-carrying scene (skipped without plotly); WebP demos regenerate
  and late frames are visually inspected.
