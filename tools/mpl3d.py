"""Shared matplotlib 3D rendering for README demo animations.

Beamline convention: plot X = beam z, plot Y = beam x, plot Z = beam y,
matching the hero animation. All drawing helpers take beam coordinates.
Every animation is rendered from real simulation output; titles must carry
the field model and a fidelity/scope note (honesty discipline).
"""

from __future__ import annotations

import numpy as np

CANVAS_INCHES = (9.2, 5.0)
CANVAS_DPI = 100

POSITRON = (20 / 255, 118 / 255, 198 / 255)
ELECTRON = (225 / 255, 118 / 255, 35 / 255)
ACCEPTED = (20 / 255, 143 / 255, 87 / 255)
LOST = (210 / 255, 76 / 255, 69 / 255)
ELEMENT = (0.45, 0.5, 0.58)
# ledger colors alias modulo 4 for scenes with more than four killing stages
LEDGER_PALETTE = (
    (210 / 255, 76 / 255, 69 / 255),
    (239 / 255, 159 / 255, 39 / 255),
    (129 / 255, 80 / 255, 197 / 255),
    (212 / 255, 83 / 255, 126 / 255),
)


def load_matplotlib():
    try:
        import matplotlib
    except ModuleNotFoundError as exc:
        raise ImportError(
            "Matplotlib is required to render README demos. Install it with "
            '`pip install "latent-dirac[viz]"`.'
        ) from exc
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator

    return plt, MaxNLocator


def load_pillow():
    try:
        from PIL import Image
    except ModuleNotFoundError as exc:
        raise ImportError(
            'Pillow is required to assemble README demos. Install it with `pip install "latent-dirac[viz]"`.'
        ) from exc
    return Image


def save_webp(frames, path, duration_ms: int = 85) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(
        path,
        format="WEBP",
        save_all=True,
        append_images=frames[1:],
        duration=duration_ms,
        loop=0,
        quality=76,
        method=6,
    )


def render_frames(
    draw,
    frame_count: int,
    title: str,
    limits: list[tuple[float, float]],
    elev: float = 18.0,
    azim_start: float = -62.0,
    azim_sweep: float = 80.0,
    box_aspect: tuple[float, float, float] = (2.0, 1.2, 0.8),
    zoom: float = 1.16,
    axis_labels: tuple[str, str, str] = ("z [m]", "x [m]", "y [m]"),
):
    """Render `frame_count` frames; `draw(axes, index, count)` adds content.

    `limits` are beam-coordinate (x, y, z) axis limits; the camera sweeps
    `azim_sweep` degrees across the animation.
    """

    plt, MaxNLocator = load_matplotlib()
    image_module = load_pillow()

    frames = []
    for index in range(frame_count):
        azimuth = azim_start + azim_sweep * index / max(frame_count - 1, 1)

        fig = plt.figure(figsize=CANVAS_INCHES, dpi=CANVAS_DPI)
        axes = fig.add_subplot(projection="3d")

        draw(axes, index, frame_count)

        axes.set_xlim(*limits[2])
        axes.set_ylim(*limits[0])
        axes.set_zlim(*limits[1])
        axes.set_xlabel(axis_labels[0], labelpad=10, color=(0.25, 0.28, 0.32))
        axes.set_ylabel(axis_labels[1], labelpad=10, color=(0.25, 0.28, 0.32))
        axes.set_zlabel(axis_labels[2], labelpad=6, color=(0.25, 0.28, 0.32))
        for axis in (axes.xaxis, axes.yaxis, axes.zaxis):
            axis.set_major_locator(MaxNLocator(4))
        style_axes(axes)
        axes.tick_params(labelsize=8, pad=2, labelcolor=(0.30, 0.33, 0.38))
        axes.view_init(elev=elev, azim=azimuth)
        axes.set_box_aspect(box_aspect, zoom=zoom)
        axes.set_position((0.02, 0.05, 0.96, 0.87))
        legend = axes.get_legend_handles_labels()
        if legend[0]:
            axes.legend(loc="upper left", frameon=False, bbox_to_anchor=(0.04, 0.96), fontsize=8)
        title_lines = title.split("\n", 1)
        fig.text(0.5, 0.965, title_lines[0], ha="center", fontsize=10.5,
                 fontweight="semibold", color=(0.15, 0.17, 0.20))
        if len(title_lines) > 1:
            fig.text(0.5, 0.935, title_lines[1], ha="center", fontsize=8.2,
                     color=(0.40, 0.43, 0.48))

        fig.canvas.draw()
        buffer = np.asarray(fig.canvas.buffer_rgba())
        frames.append(image_module.fromarray(buffer[..., :3].copy()))
        plt.close(fig)

    return frames


def style_axes(axes) -> None:
    """Shared furniture: near-white panes, light grid (best effort)."""

    try:
        for axis in (axes.xaxis, axes.yaxis, axes.zaxis):
            axis.set_pane_color((0.985, 0.985, 0.992, 0.7))
            axis._axinfo["grid"].update(
                {"color": (0.78, 0.81, 0.85, 0.6), "linewidth": 0.5}
            )
    except (AttributeError, KeyError):  # private mpl API: degrade silently
        pass


def axis_limits(
    *position_arrays: np.ndarray,
    pad_fraction: float = 0.08,
    percentiles: tuple[float, float] = (1.0, 99.0),
):
    """Axis limits from percentiles, so runaway lost particles don't zoom out the frame."""

    stacked = np.concatenate([array.reshape(-1, 3) for array in position_arrays])
    limits = []
    for axis in range(3):
        low = float(np.nanpercentile(stacked[:, axis], percentiles[0]))
        high = float(np.nanpercentile(stacked[:, axis], percentiles[1]))
        pad = pad_fraction * max(high - low, 1e-6)
        limits.append((low - pad, high + pad))
    return limits


def draw_trajectories(axes, positions: np.ndarray, reveal: int, colors, linewidth=0.7, alpha=0.55):
    """Draw per-particle trails up to `reveal` snapshots.

    `colors` is either one RGB tuple or a per-particle list.
    """

    revealed = positions[:reveal]
    per_particle = not _is_single_color(colors)
    for particle in range(revealed.shape[1]):
        color = colors[particle] if per_particle else colors
        axes.plot(
            revealed[:, particle, 2],
            revealed[:, particle, 0],
            revealed[:, particle, 1],
            color=color,
            linewidth=linewidth,
            alpha=alpha,
        )


def _is_single_color(colors) -> bool:
    return isinstance(colors, tuple) and len(colors) in (3, 4) and isinstance(colors[0], (int, float))


def draw_points(axes, positions: np.ndarray, colors, size=9, label=None):
    per_particle = not _is_single_color(colors)
    axes.scatter(
        positions[:, 2],
        positions[:, 0],
        positions[:, 1],
        c=colors if per_particle else [colors],
        s=size,
        depthshade=False,
        label=label,
    )


# apparatus glyph palette (see the 2026-07-07 apparatus-visuals spec)
COPPER = (0.72, 0.44, 0.20)
POLE_N = (0.78, 0.30, 0.26)
POLE_S = (0.27, 0.42, 0.70)
STEEL = (0.55, 0.58, 0.62)
GLASS = (0.55, 0.70, 0.80)


def draw_coil(axes, radius: float, z0: float, z1: float):
    """A wound-coil glyph: copper helix over a faint bore surface.

    The winding count scales with the aspect so short and long coils
    both read as coils; the glyph visualizes radius/length only.
    """

    length = z1 - z0
    turns = int(np.clip(length / (0.25 * radius), 10, 44))
    theta = np.linspace(0.0, 2.0 * np.pi * turns, turns * 24)
    z = np.linspace(z0, z1, theta.size)
    axes.plot(
        z,
        radius * np.cos(theta),
        radius * np.sin(theta),
        color=COPPER,
        linewidth=0.95,
        alpha=0.45,
    )
    draw_cylinder(axes, radius, z0, z1, alpha=0.05)


def draw_dipole_poles(axes, b_direction, z0: float, z1: float, gap_half: float, face_half):
    """Two pole faces perpendicular to the dipole B vector, plus yoke edges.

    Gap and face size are display-scaled accents (the model has no pole
    geometry); north is warm, south is cool. `face_half` is either a
    half-width or a `(lo, hi)` interval along the width direction — the
    interval form lets a bent-beam frame crop the faces asymmetrically
    (mplot3d does not clip to the axes box).
    """

    b_unit = np.asarray(b_direction, dtype=float)
    norm = float(np.linalg.norm(b_unit))
    if norm == 0.0:
        return
    b_unit = b_unit / norm
    width_dir = np.cross(b_unit, [0.0, 0.0, 1.0])
    width_norm = float(np.linalg.norm(width_dir))
    if width_norm < 1e-9:  # B along z: no transverse pole picture
        return
    width_dir /= width_norm

    if np.isscalar(face_half):
        face_lo, face_hi = -float(face_half), float(face_half)
    else:
        face_lo, face_hi = (float(value) for value in face_half)

    z_line = np.linspace(z0, z1, 2)
    s_line = np.linspace(face_lo, face_hi, 2)
    z_grid, s_grid = np.meshgrid(z_line, s_line)
    corners = []
    # gap flux exits the north pole face and enters the south face, so
    # north sits on the -B side of the gap
    for sign, color in ((-1.0, POLE_N), (1.0, POLE_S)):
        center = sign * gap_half * b_unit
        x = center[0] + s_grid * width_dir[0]
        y = center[1] + s_grid * width_dir[1]
        z = z_grid + center[2]
        axes.plot_surface(z, x, y, color=color, alpha=0.38, linewidth=0, shade=False)
        corners.append(
            [
                center[:2] + s * width_dir[:2]
                for s in (face_lo, face_hi)
            ]
        )
    for (top, bottom) in zip(corners[0], corners[1], strict=True):
        for z_edge in (z0, z1):
            axes.plot(
                [z_edge, z_edge], [top[0], bottom[0]], [top[1], bottom[1]],
                color=ELEMENT, linewidth=0.7, alpha=0.3,
            )


def draw_quadrupole_tips(axes, z0: float, z1: float, r0: float, gradient_sign: float = 1.0):
    """The classic four hyperbolic pole tips, alternating polarity.

    Tip profiles follow 2xy = ±r0² (apex distance r0, a display-scaled
    accent — the quadrupole model carries gradient and length only),
    extruded across the element with spine lines. Polarity follows the
    gradient sign: with B = g*(y, x, 0), flux enters the 45° tip for
    g > 0, making it a south pole.
    """

    t = np.linspace(-0.55, 0.55, 25)
    base_x = (r0 / np.sqrt(2.0)) * np.exp(t)
    base_y = (r0 / np.sqrt(2.0)) * np.exp(-t)
    first, second = (POLE_S, POLE_N) if gradient_sign >= 0.0 else (POLE_N, POLE_S)
    rotations = [
        (base_x, base_y, first),
        (-base_y, base_x, second),
        (-base_x, -base_y, first),
        (base_y, -base_x, second),
    ]
    for x_curve, y_curve, color in rotations:
        for z_edge in (z0, z1):
            axes.plot(
                np.full_like(x_curve, z_edge), x_curve, y_curve,
                color=color, linewidth=1.2, alpha=0.6,
            )
        for index in (0, x_curve.size // 2, -1):
            axes.plot(
                [z0, z1], [x_curve[index]] * 2, [y_curve[index]] * 2,
                color=color, linewidth=0.7, alpha=0.4,
            )


def draw_washer(axes, inner_radius: float, outer_radius: float, z: float):
    """An aperture as a collimator washer: annular faces plus a rim."""

    thickness = 0.35 * (outer_radius - inner_radius)
    for face_z in (z - 0.5 * thickness, z + 0.5 * thickness):
        draw_disc(axes, inner_radius, outer_radius, face_z, color=STEEL, alpha=0.35)
    theta = np.linspace(0.0, 2.0 * np.pi, 36)
    z_line = np.array([z - 0.5 * thickness, z + 0.5 * thickness])
    theta_grid, z_grid = np.meshgrid(theta, z_line)
    for rim in (inner_radius, outer_radius):
        axes.plot_surface(
            z_grid, rim * np.cos(theta_grid), rim * np.sin(theta_grid),
            color=STEEL, alpha=0.45, linewidth=0, shade=False,
        )


def _rect_plane(axes, half_x: float, half_y: float, z: float, color, alpha: float,
                center=(0.0, 0.0)):
    x_grid, y_grid = np.meshgrid(
        [center[0] - half_x, center[0] + half_x], [center[1] - half_y, center[1] + half_y]
    )
    axes.plot_surface(np.full_like(x_grid, z), x_grid, y_grid,
                      color=color, alpha=alpha, linewidth=0, shade=False)


def draw_screen(axes, half_x: float, half_y: float, z: float, center=(0.0, 0.0)):
    """A monitor as a framed screen: faint glass, solid border, corner ticks."""

    cx, cy = center
    _rect_plane(axes, half_x, half_y, z, GLASS, 0.06, center=center)
    border = np.array(
        [[-half_x, -half_y], [half_x, -half_y], [half_x, half_y], [-half_x, half_y], [-half_x, -half_y]]
    ) + np.array([cx, cy])
    axes.plot(
        np.full(border.shape[0], z), border[:, 0], border[:, 1],
        color=(0.35, 0.5, 0.58), linewidth=1.4, alpha=0.75,
    )
    tick_x, tick_y = 0.18 * half_x, 0.18 * half_y
    for sx in (-1.0, 1.0):
        for sy in (-1.0, 1.0):
            axes.plot([z, z], [cx + sx * half_x, cx + sx * (half_x - tick_x)],
                      [cy + sy * half_y, cy + sy * half_y],
                      color=(0.35, 0.5, 0.58), linewidth=2.0, alpha=0.85)
            axes.plot([z, z], [cx + sx * half_x, cx + sx * half_x],
                      [cy + sy * half_y, cy + sy * (half_y - tick_y)],
                      color=(0.35, 0.5, 0.58), linewidth=2.0, alpha=0.85)


def draw_foil(axes, half_x: float, half_y: float, z: float, center=(0.0, 0.0)):
    """A matter slab as a metallic foil: faint sheet with a doubled edge.

    Micrometer thicknesses render as a plane — drawing the true
    thickness would be invisible at beamline scale.
    """

    _rect_plane(axes, half_x, half_y, z, (0.62, 0.66, 0.72), 0.18, center=center)
    for scale, lw in ((1.0, 1.3), (0.92, 0.6)):
        wx, wy = scale * half_x, scale * half_y
        border = np.array(
            [[-wx, -wy], [wx, -wy], [wx, wy], [-wx, wy], [-wx, -wy]]
        ) + np.array(center)
        axes.plot(
            np.full(border.shape[0], z), border[:, 0], border[:, 1],
            color=(0.45, 0.49, 0.55), linewidth=lw, alpha=0.8,
        )


def draw_cylinder(axes, radius: float, z0: float, z1: float, color=ELEMENT, alpha=0.08):
    theta = np.linspace(0.0, 2.0 * np.pi, 36)
    z = np.linspace(z0, z1, 2)
    theta_grid, z_grid = np.meshgrid(theta, z)
    axes.plot_surface(
        z_grid,
        radius * np.cos(theta_grid),
        radius * np.sin(theta_grid),
        color=color,
        alpha=alpha,
        linewidth=0,
        shade=False,
    )
    for z_edge in (z0, z1):
        axes.plot(
            np.full_like(theta, z_edge),
            radius * np.cos(theta),
            radius * np.sin(theta),
            color=color,
            linewidth=0.8,
            alpha=0.6,
        )


def draw_disc(axes, inner_radius: float, outer_radius: float, z: float, color=ELEMENT, alpha=0.25):
    theta = np.linspace(0.0, 2.0 * np.pi, 36)
    r = np.array([inner_radius, outer_radius])
    theta_grid, r_grid = np.meshgrid(theta, r)
    axes.plot_surface(
        np.full_like(theta_grid, z),
        r_grid * np.cos(theta_grid),
        r_grid * np.sin(theta_grid),
        color=color,
        alpha=alpha,
        linewidth=0,
        shade=False,
    )


def draw_block(axes, z0: float, z1: float, half_width: float, color=(0.35, 0.37, 0.4), alpha=0.55):
    """Solid-looking block annotation (e.g. a target) - drawn, not simulated."""

    span = np.array([-half_width, half_width])
    face_y, face_z = np.meshgrid(span, span)
    for z_face in (z0, z1):
        axes.plot_surface(
            np.full_like(face_y, z_face),
            face_y,
            face_z,
            color=color,
            alpha=alpha,
            linewidth=0,
            shade=False,
        )
    z_span = np.array([z0, z1])
    for fixed in span:
        wall_z, wall_w = np.meshgrid(z_span, span)
        axes.plot_surface(
            wall_z,
            np.full_like(wall_z, fixed),
            wall_w,
            color=color,
            alpha=alpha * 0.8,
            linewidth=0,
            shade=False,
        )
        axes.plot_surface(
            wall_z,
            wall_w,
            np.full_like(wall_z, fixed),
            color=color,
            alpha=alpha * 0.8,
            linewidth=0,
            shade=False,
        )


def draw_beam_arrow(axes, z_start: float, z_end: float, color=(0.85, 0.3, 0.25), linewidth=3.0):
    """Incoming-beam sketch along the axis - drawn, not simulated."""

    axes.plot([z_start, z_end], [0.0, 0.0], [0.0, 0.0], color=color, linewidth=linewidth, alpha=0.9)
    head = 0.12 * abs(z_end - z_start)
    for dy, dz in ((0.4, 0.0), (-0.4, 0.0), (0.0, 0.4), (0.0, -0.4)):
        axes.plot(
            [z_end - head, z_end],
            [dy * head, 0.0],
            [dz * head, 0.0],
            color=color,
            linewidth=linewidth * 0.7,
            alpha=0.9,
        )


FIELD_LINE_COLORS = {"B": (0.35, 0.5, 0.75), "E": (0.85, 0.6, 0.2)}


def draw_field_polylines(axes, bundles, alpha=0.45, linewidth=0.9):
    """Field-line polylines from `viz.field_lines` (B steel blue, E amber)."""

    for kind, line in bundles:
        axes.plot(
            line[:, 2],
            line[:, 0],
            line[:, 1],
            color=FIELD_LINE_COLORS[kind],
            linewidth=linewidth,
            alpha=alpha,
        )


def draw_photon_burst(
    axes,
    positions: np.ndarray,
    photon_directions: np.ndarray,
    progress: np.ndarray,
    max_length: float,
    color=(0.85, 0.65, 0.13),
    flash_color=(1.0, 0.92, 0.55),
):
    """Back-to-back photon pairs growing with per-event progress in [0, 1].

    Kinematic visualization only: 511 keV is a label, no energetics (see
    the safety scope). `positions` is (K, 3), `photon_directions` is
    (K, 2, 3), `progress` is (K,) — 0 draws nothing for that event, 1 is
    a fully grown ray pair; events still in progress get a fading star
    flash at the vertex.
    """

    progress = np.clip(np.asarray(progress, dtype=float), 0.0, 1.0)
    active = progress > 0.0
    if not np.any(active):
        return
    for direction_index in (0, 1):
        ends = positions + (progress[:, np.newaxis] * max_length) * photon_directions[:, direction_index, :]
        for start, end, fraction in zip(positions[active], ends[active], progress[active], strict=True):
            axes.plot(
                [start[2], end[2]],
                [start[0], end[0]],
                [start[1], end[1]],
                color=color,
                linewidth=0.7,
                alpha=0.25 + 0.4 * (1.0 - fraction),
            )
    fresh = active & (progress < 1.0)
    if np.any(fresh):
        flare = positions[fresh]
        sizes = 8.0 + 30.0 * (1.0 - progress[fresh])
        axes.scatter(
            flare[:, 2],
            flare[:, 0],
            flare[:, 1],
            c=[flash_color],
            s=sizes,
            alpha=0.7,
            depthshade=False,
            marker="*",
        )


def draw_scene_elements(axes, scene, run_result, plane_half_width=0.01, display_scale=None):
    """Draw an apparatus glyph for every element with a spatial footprint.

    `display_scale` is a per-axis dict ``{"x": (lo, hi), "y": (lo, hi)}``
    of the framed intervals (a plain float means an isotropic ±float
    frame); it sizes and crops the display-scaled accents — plate crop,
    pole gap and faces, tip apex, screen and foil extents: mplot3d does
    not clip to the axes box, several models carry no transverse
    geometry of their own, and a bent beam makes the frame asymmetric
    about the element axis. Glyph conventions are recorded in the
    2026-07-07 apparatus-visuals spec.
    """

    frame_z = None
    if display_scale is None:
        width = 3.0 * plane_half_width
        frame_x = frame_y = (-width, width)
    elif isinstance(display_scale, dict):
        frame_x = tuple(float(value) for value in display_scale["x"])
        frame_y = tuple(float(value) for value in display_scale["y"])
        if "z" in display_scale:
            frame_z = tuple(float(value) for value in display_scale["z"])
    else:
        width = float(display_scale)
        frame_x = frame_y = (-width, width)
    mid_x, mid_y = 0.5 * (frame_x[0] + frame_x[1]), 0.5 * (frame_y[0] + frame_y[1])
    half_x, half_y = 0.5 * (frame_x[1] - frame_x[0]), 0.5 * (frame_y[1] - frame_y[0])
    # distance from the element axis (x=y=0) to the nearest frame edge:
    # axis-centered glyphs sized past this poke out of a bent-beam frame
    edge_x = max(min(-frame_x[0], frame_x[1]), 1e-9)
    edge_y = max(min(-frame_y[0], frame_y[1]), 1e-9)

    def _reach_from(origin_xy, direction_xy) -> float:
        # farthest t >= 0 with origin + t*direction still inside the frame
        # rectangle (per-axis containment, not a projection bound)
        t_max = np.inf
        for origin, component, (lo, hi) in (
            (origin_xy[0], direction_xy[0], frame_x),
            (origin_xy[1], direction_xy[1], frame_y),
        ):
            if component > 1e-12:
                t_max = min(t_max, (hi - origin) / component)
            elif component < -1e-12:
                t_max = min(t_max, (lo - origin) / component)
        return float(max(t_max, 0.0))

    def _z_span(element) -> tuple[float, float] | None:
        # crop the extrusion to the framed z interval: an element that
        # sticks past the frame end otherwise draws outside the box
        z0 = element.center_z_m - 0.5 * element.length_m
        z1 = element.center_z_m + 0.5 * element.length_m
        if frame_z is not None:
            z0, z1 = max(z0, frame_z[0]), min(z1, frame_z[1])
        return (z0, z1) if z1 > z0 else None

    for element in scene.elements:
        if element.type == "solenoid":
            span = _z_span(element)
            if span is None:
                continue
            draw_coil(axes, element.radius_m, span[0], span[1])
        elif element.type == "dipole":
            span = _z_span(element)
            if span is None:
                continue
            z0, z1 = span
            # the pole picture is transverse: build it from the transverse
            # component of B so the faces stay in the framed z span
            b = np.asarray(element.B_vector_t, dtype=float)
            b_xy = np.array([b[0], b[1], 0.0])
            b_norm = float(np.linalg.norm(b_xy))
            if b_norm > 1e-12:
                b_unit = b_xy / b_norm
                width_dir = np.cross(b_unit, [0.0, 0.0, 1.0])
                gap_half = 0.92 * min(
                    _reach_from((0.0, 0.0), b_unit[:2]),
                    _reach_from((0.0, 0.0), -b_unit[:2]),
                )
                pole_centers = (gap_half * b_unit[:2], -gap_half * b_unit[:2])
                face_hi = 0.9 * min(_reach_from(c, width_dir[:2]) for c in pole_centers)
                face_lo = -0.9 * min(_reach_from(c, -width_dir[:2]) for c in pole_centers)
                if gap_half > 0.0 and face_hi > face_lo:
                    draw_dipole_poles(
                        axes, b_unit, z0, z1,
                        gap_half=gap_half, face_half=(face_lo, face_hi),
                    )
        elif element.type == "quadrupole":
            span = _z_span(element)
            if span is None:
                continue
            # the tip arcs reach ~1.23*r0 from the axis: keep them inside
            # the nearest frame edge
            draw_quadrupole_tips(
                axes, span[0], span[1], r0=0.75 * min(edge_x, edge_y),
                gradient_sign=1.0 if element.gradient_t_m >= 0.0 else -1.0,
            )
        elif element.type == "aperture":
            outer_radius = 1.6 * element.radius_m
            if display_scale is not None:
                # the hole is model geometry; the rim is a display accent.
                # An opening wider than the framed region leaves nothing of
                # the washer in view — drawing it would flood the figure.
                cap = min(edge_x, edge_y)
                if element.radius_m >= cap:
                    continue
                outer_radius = min(outer_radius, cap)
            draw_washer(axes, element.radius_m, outer_radius, element.z_m)
        elif element.type == "matter_slab":
            # the slab has real transverse geometry (transverse_half_width_m,
            # an engine aperture): crop to the frame, never past the model
            hw = element.transverse_half_width_m
            x_lo, x_hi = max(frame_x[0], -hw), min(frame_x[1], hw)
            y_lo, y_hi = max(frame_y[0], -hw), min(frame_y[1], hw)
            if x_hi > x_lo and y_hi > y_lo:
                draw_foil(
                    axes,
                    0.4 * (x_hi - x_lo), 0.4 * (y_hi - y_lo), element.entry_z_m,
                    center=(0.5 * (x_lo + x_hi), 0.5 * (y_lo + y_hi)),
                )
        elif element.type == "annihilation_plate":
            radius = (
                min(element.radius_m, 1.05 * min(edge_x, edge_y))
                if display_scale is not None
                else element.radius_m
            )
            theta = np.linspace(0.0, 2.0 * np.pi, 60)
            for ring, lw in ((radius, 1.6), (0.55 * radius, 1.0)):
                axes.plot(
                    np.full_like(theta, element.z_m),
                    ring * np.cos(theta),
                    ring * np.sin(theta),
                    color=(0.55, 0.45, 0.35),
                    linewidth=lw,
                    alpha=0.75,
                )
            draw_disc(axes, 0.0, 0.55 * radius, element.z_m, color=(0.55, 0.45, 0.35), alpha=0.08)
        elif element.type == "monitor":
            snapshot = run_result.monitors.get(element.label)
            if snapshot is None:
                continue
            alive = snapshot.alive
            positions = snapshot.position_m[alive] if np.any(alive) else snapshot.position_m
            draw_screen(
                axes, 0.85 * half_x, 0.85 * half_y, float(np.mean(positions[:, 2])),
                center=(mid_x, mid_y),
            )
