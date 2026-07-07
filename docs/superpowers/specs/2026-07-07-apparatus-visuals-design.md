# Apparatus visuals: every non-particle, non-field element made real

Date: 2026-07-07. Status: accepted.

## Motivation

Fields got their streamlines and the clouds their ledger colors; the
apparatus is still placeholder wireframes — a solenoid is two circles
and four lines, a quadrupole is the same gray box as a dipole, an
aperture is a flat ring. This pass gives every element a glyph that
reads as the physical object it stands for, without inventing geometry
the model doesn't have: glyphs visualize the model's own parameters
(radius, length, gap, thickness), and idealized elements stay
schematic.

## Element glyphs (WebP via `tools/mpl3d.py`; HTML mirrors the key ones)

- **Solenoid (both profiles)** — a wound coil: a copper helix at
  `radius_m` spanning the length (turn count scaled by length), plus a
  very faint bore surface for volume. The winding is a glyph for "this
  is a coil"; the field model stays what the profile says.
- **Dipole** — two pole faces: slabs perpendicular to the transverse
  component of the B vector at ±gap/2, north warm / south cool,
  connected by faint yoke edges. Gap flux exits the north pole face,
  so north sits on the −B side of the gap. Gap and face size scale
  with the beam frame (display scale, like the plate crop).
- **Quadrupole** — the classic four hyperbolic pole tips at 45°/135°/
  225°/315°, alternating polarity colors conditioned on the sign of
  `gradient_t_m` (with B = g·(y, x, 0), flux enters the 45° tip for
  g > 0, making it a south pole), extruded across `length_m` (tip
  profiles at both ends plus spine lines).
- **Frame-interval cropping (WebP path)** — the renderer passes the
  framed per-axis intervals (`{"x": (lo, hi), "y": (lo, hi),
  "z": (lo, hi)}`) instead of one isotropic scale. mplot3d does not
  clip to the axes box, and a bent beam makes the frame asymmetric
  about the element axis, so display-scaled accents size and crop to
  the actual frame via per-axis reach (the farthest each glyph point
  may sit from the element axis and still lie inside the frame
  rectangle — sound for tilted B, not just axis-aligned): pole gap
  and faces clamp inside the frame, tip apexes fit the nearest
  transverse edge, screens center on the frame midpoint, foils crop
  to the frame intersected with the slab's real transverse extent
  (`transverse_half_width_m`, an engine aperture — never drawn past
  it), the washer rim caps at the frame edge (skipped entirely when
  the model opening already swallows the framed region), and
  extrusions (coil, poles, tips) crop their z span to the framed
  interval. Model geometry (bore radius, aperture radius, slab
  half-width, element z positions) is never rescaled — only the
  accents that have no model geometry are.
- **Aperture** — a washer with thickness: front/back annular faces and
  a rim, steel gray; reads as a collimator plate instead of a floating
  ring.
- **Monitor** — a framed screen: faint glass pane plus a solid frame
  border and corner ticks.
- **Annihilation plate** — a target-style disc: bold outer ring, mid
  ring, faint center fill (display-radius crop kept).
- **Matter slab (new in the WebP path)** — a metallic foil: thin
  filled plane at `entry_z_m` with a double-edged border; the µm
  thickness renders as a plane (drawing its true thickness would be
  invisible), sized to the display scale.
- **Penning trap / uniform field** — intentionally no apparatus: the
  ideal models have none; their field lines carry the structure.

## Furniture

`render_frames` (shared by all WebP demos, including the charge-sign
splitter hero's own renderer): near-white panes, lighter grid, darker
tick labels, bolder title line one — one consistent style everywhere.

## Honesty

Glyphs draw only model parameters plus explicitly display-scaled
accents (pole gap, face size, foil extent — same convention as the
plate's display-radius crop). No element gains physics it doesn't
have; fidelity labels and hover text are unchanged.

## Validation

Drawing helpers get artist-count smoke tests (each glyph adds artists
to a 3D axes without error); every demo regenerates (engine env for
the gated ones) and late frames are visually inspected per the
demo-assets rule; full suite + ruff; HTML renderer keeps its
fidelity-hover tests green.
