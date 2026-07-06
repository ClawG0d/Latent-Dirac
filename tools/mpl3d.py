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
        axes.set_xlabel(axis_labels[0], labelpad=10)
        axes.set_ylabel(axis_labels[1], labelpad=10)
        axes.set_zlabel(axis_labels[2], labelpad=6)
        for axis in (axes.xaxis, axes.yaxis, axes.zaxis):
            axis.set_major_locator(MaxNLocator(4))
        axes.tick_params(labelsize=8, pad=2)
        axes.view_init(elev=elev, azim=azimuth)
        axes.set_box_aspect(box_aspect, zoom=zoom)
        axes.set_position((0.02, 0.05, 0.96, 0.87))
        legend = axes.get_legend_handles_labels()
        if legend[0]:
            axes.legend(loc="upper left", frameon=False, bbox_to_anchor=(0.04, 0.96), fontsize=8)
        fig.suptitle(title, fontsize=10, y=0.97)

        fig.canvas.draw()
        buffer = np.asarray(fig.canvas.buffer_rgba())
        frames.append(image_module.fromarray(buffer[..., :3].copy()))
        plt.close(fig)

    return frames


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


def draw_box(axes, half_width: float, z0: float, z1: float, color=ELEMENT, alpha=0.5):
    w = half_width
    corners = [(-w, -w), (w, -w), (w, w), (-w, w), (-w, -w)]
    for z_edge in (z0, z1):
        axes.plot(
            [z_edge] * 5,
            [c[0] for c in corners],
            [c[1] for c in corners],
            color=color,
            linewidth=0.8,
            alpha=alpha,
        )
    for cx, cy in corners[:4]:
        axes.plot([z0, z1], [cx, cx], [cy, cy], color=color, linewidth=0.8, alpha=alpha)


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


def draw_plane(axes, half_width: float, z: float, color=ELEMENT, alpha=0.15):
    span = np.array([-half_width, half_width])
    x_grid, y_grid = np.meshgrid(span, span)
    axes.plot_surface(
        np.full_like(x_grid, z),
        x_grid,
        y_grid,
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


def draw_scene_elements(axes, scene, run_result, plane_half_width=0.01, plate_display_radius=None):
    """Draw geometry for every scene element that has a spatial footprint.

    `plate_display_radius` visually crops an annihilation plate to the
    framed region: mplot3d does not clip surfaces to the axes box, so a
    plate much wider than the beam would bleed past the frame edge.
    """

    for element in scene.elements:
        if element.type == "solenoid":
            z0 = element.center_z_m - 0.5 * element.length_m
            z1 = element.center_z_m + 0.5 * element.length_m
            draw_cylinder(axes, element.radius_m, z0, z1)
        elif element.type in ("dipole", "quadrupole"):
            z0 = element.center_z_m - 0.5 * element.length_m
            z1 = element.center_z_m + 0.5 * element.length_m
            draw_box(axes, plane_half_width, z0, z1)
        elif element.type == "aperture":
            draw_disc(axes, element.radius_m, 1.6 * element.radius_m, element.z_m)
        elif element.type == "annihilation_plate":
            radius = element.radius_m
            if plate_display_radius is not None:
                radius = min(radius, plate_display_radius)
            theta = np.linspace(0.0, 2.0 * np.pi, 60)
            for ring in (radius, 0.6 * radius):
                axes.plot(
                    np.full_like(theta, element.z_m),
                    ring * np.cos(theta),
                    ring * np.sin(theta),
                    color=(0.55, 0.45, 0.35),
                    linewidth=1.2,
                    alpha=0.7,
                )
        elif element.type == "monitor":
            snapshot = run_result.monitors.get(element.label)
            if snapshot is None:
                continue
            alive = snapshot.alive
            positions = snapshot.position_m[alive] if np.any(alive) else snapshot.position_m
            draw_plane(axes, plane_half_width, float(np.mean(positions[:, 2])))
