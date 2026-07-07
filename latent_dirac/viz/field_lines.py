"""Field-line polylines: faithful streamlines of the model fields.

Visual diagnostics only — lines are integrated from the exact `Field`
being simulated, so every idealization renders as-is: hard-edge lines
stop abruptly at the envelope, ideal-trap lines run to the frame edge,
thin-sheet fringes funnel smoothly. No plotting here; renderers consume
the polylines. Design record:
docs/superpowers/specs/2026-07-06-field-line-rendering-design.md.
"""

from __future__ import annotations

import numpy as np

_FIELD_FLOOR = 1.0e-12


def field_line(
    field,
    seed,
    length_m: float,
    step_m: float,
    direction: float = 1.0,
    kind: str = "B",
    t_s: float = 0.0,
) -> np.ndarray:
    """Integrate dX/ds = F/|F| (midpoint rule) from `seed`; returns (N, 3).

    Stops early where |F| falls under a floor (hard edges, zero fields);
    a seed in zero field returns just the seed point.
    """

    if kind not in ("B", "E"):
        raise ValueError(f"kind must be 'B' or 'E', got {kind!r}")
    evaluate = field.B if kind == "B" else field.E

    def unit(point: np.ndarray) -> np.ndarray | None:
        vector = np.asarray(evaluate(point, t_s), dtype=float)
        norm = float(np.linalg.norm(vector))
        if norm < _FIELD_FLOOR:
            return None
        return vector / norm

    position = np.asarray(seed, dtype=float)
    points = [position]
    for _ in range(max(int(length_m / step_m), 1)):
        first = unit(position)
        if first is None:
            break
        midpoint = unit(position + direction * 0.5 * step_m * first)
        if midpoint is None:
            break
        position = position + direction * step_m * midpoint
        points.append(position)
    return np.asarray(points)


def _two_sided(field, seed, length_m, step_m, kind, t_s) -> np.ndarray:
    forward = field_line(field, seed, 0.5 * length_m, step_m, 1.0, kind, t_s)
    backward = field_line(field, seed, 0.5 * length_m, step_m, -1.0, kind, t_s)
    if backward.shape[0] > 1:
        return np.vstack([backward[::-1], forward[1:]])
    return forward


def _ring_seeds(radius_m: float, z_m: float, count: int) -> list[tuple[float, float, float]]:
    angles = np.linspace(0.0, 2.0 * np.pi, count, endpoint=False)
    return [(radius_m * np.cos(a), radius_m * np.sin(a), z_m) for a in angles]


def _perpendicular_pair(direction: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    helper = np.array([1.0, 0.0, 0.0])
    if abs(float(direction @ helper)) > 0.9:
        helper = np.array([0.0, 1.0, 0.0])
    first = np.cross(direction, helper)
    first /= np.linalg.norm(first)
    return first, np.cross(direction, first)


def field_elements_for_lines(scene):
    """Field elements with physics-identical repeats dropped (global dedupe).

    A scene that re-declares one physical field per pipeline stage (e.g.
    trap -> cool -> trap -> ...) must draw that field's lines once, not
    once per stage — stacked translucent copies render near-opaque.
    """

    from latent_dirac.scene.schema import FIELD_ELEMENT_TYPES

    seen: set[str] = set()
    for element in scene.elements:
        if element.type not in FIELD_ELEMENT_TYPES:
            continue
        fingerprint = repr(sorted(element.model_dump(exclude={"label", "steps"}).items()))
        if fingerprint in seen:
            continue
        seen.add(fingerprint)
        yield element


def element_field_line_bundles(element, field, extent: dict) -> list[tuple[str, np.ndarray]]:
    """Per-element seed strategies; returns (kind, polyline) tuples.

    `extent` carries the beam bounding half-widths
    (`transverse_m`, `axial_m`) so seeding scales with each scene. Gated
    elements are sampled at their switch-on time — the field is drawn
    as-when-active.
    """

    transverse = float(extent["transverse_m"])
    axial = float(extent["axial_m"])
    t_s = float(getattr(element, "t_on_s", None) or 0.0)
    bundles: list[tuple[str, np.ndarray]] = []

    if element.type == "solenoid":
        radius, length, center = element.radius_m, element.length_m, element.center_z_m
        # seed at the smaller of the bore and the framed beam scale, so the
        # funneling stays inside the rendered frame
        seed_radius = min(0.55 * radius, 0.7 * transverse)
        if element.profile == "thin_sheet":
            z_start = center - 0.5 * length - 0.8 * radius
            span = length + 2.4 * radius
        else:
            z_start = center - 0.45 * length
            span = 0.95 * length
        for seed in _ring_seeds(seed_radius, z_start, 8):
            bundles.append(("B", field_line(field, seed, span, span / 400.0, 1.0, "B", t_s)))

    elif element.type == "uniform_field":

        def _extent_along(vector: np.ndarray) -> float:
            scale = np.array([transverse, transverse, axial])
            return float(np.linalg.norm(vector * scale))

        for kind, vector in (("B", element.B_vector_t), ("E", element.E_vector_v_m)):
            magnitude = float(np.linalg.norm(vector))
            if magnitude == 0.0:
                continue
            direction = np.asarray(vector, dtype=float) / magnitude
            first, second = _perpendicular_pair(direction)
            span = 2.2 * _extent_along(direction)
            anchor = np.array([0.0, 0.0, axial])
            # offsets scale with the frame extent along each perpendicular,
            # so the family spreads across the frame instead of bunching
            for u in (-0.8, -0.4, 0.0, 0.4, 0.8):
                seed = anchor + u * _extent_along(first) * first
                bundles.append((kind, _two_sided(field, seed, span, span / 200.0, kind, t_s)))
            for u in (-0.6, 0.6):
                seed = anchor + u * _extent_along(second) * second
                bundles.append((kind, _two_sided(field, seed, span, span / 200.0, kind, t_s)))

    elif element.type == "penning_trap":
        center = element.center_z_m
        span = 2.6 * axial
        for u in (-0.7, -0.35, 0.35, 0.7):
            seed = (u * transverse, 0.0, center - 1.3 * axial)
            bundles.append(("B", field_line(field, seed, span, span / 150.0, 1.0, "B", t_s)))
        for u in (-0.5, 0.5):
            seed = (0.0, u * transverse, center - 1.3 * axial)
            bundles.append(("B", field_line(field, seed, span, span / 150.0, 1.0, "B", t_s)))
        # quadrupole E lines: bend from the endcap regions toward the
        # midplane and outward (r^2 |z - zc| stays constant)
        e_span = 2.0 * axial
        for sign in (1.0, -1.0):
            for seed in _ring_seeds(0.3 * transverse, center + sign * 0.8 * axial, 4):
                bundles.append(("E", _two_sided(field, seed, e_span, e_span / 300.0, "E", t_s)))

    elif element.type == "dipole":
        length, center = element.length_m, element.center_z_m
        span = 2.6 * transverse
        for x_offset in (-0.35, 0.35):
            for z_offset in (-0.25, 0.25):
                seed = (x_offset * transverse, 0.0, center + z_offset * length)
                bundles.append(("B", _two_sided(field, seed, span, span / 120.0, "B", 0.0)))

    elif element.type == "quadrupole":
        center = element.center_z_m
        span = 1.4 * transverse
        for seed in _ring_seeds(0.6 * transverse, center, 8):
            bundles.append(("B", _two_sided(field, seed, span, span / 200.0, "B", 0.0)))

    elif element.type == "rotating_wall":
        # snapshot the transverse E pattern at the switch-on time: m=1 is a
        # uniform rotating field (parallel lines), m=2 a rotating quadrupole
        # (hyperbolic lines). It is time-dependent — this is one instant.
        span = 1.8 * transverse
        for seed in _ring_seeds(0.6 * transverse, axial, 8):
            bundles.append(("E", _two_sided(field, seed, span, span / 200.0, "E", t_s)))

    elif element.type == "composite_field":
        # draw each component's own lines (in its own field), so a composite
        # shows the superposed families (trap B lines + rotating-wall E lines)
        for sub_spec, sub_field in zip(element.fields, field.fields, strict=True):
            bundles.extend(element_field_line_bundles(sub_spec, sub_field, extent))

    return [(kind, line) for kind, line in bundles if line.shape[0] >= 2]
