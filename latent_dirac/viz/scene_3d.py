"""Scene-driven 3D rendering on the Plotly backend.

Every geometric element type gets a wireframe representation, and each
element trace carries its fidelity tier in the hover text so the rendered
scene stays honest about its approximation level.
"""

from __future__ import annotations

import numpy as np

from latent_dirac.scene.build import SceneRunResult
from latent_dirac.scene.schema import Scene
from latent_dirac.viz.base import import_optional

FIDELITY_LABELS = {
    "uniform_field": "fidelity: parameterized (uniform field)",
    "solenoid": "fidelity: parameterized (hard-edge optics model)",
    "dipole": "fidelity: parameterized (hard-edge optics model)",
    "quadrupole": "fidelity: parameterized (hard-edge optics model)",
    "penning_trap": "fidelity: parameterized (ideal quadrupole well, no electrode geometry)",
    "rotating_wall": (
        "fidelity: parameterized (rotating multipole E field; single-particle "
        "only — no self-consistent plasma compression)"
    ),
    "composite_field": "fidelity: exact superposition of component fields (tiers follow components)",
    "drift": "fidelity: exact zero-field transport",
    "aperture": "fidelity: diagnostic acceptance cut",
    "momentum_window": "fidelity: diagnostic acceptance cut",
    "annihilation_plate": "fidelity: parameterized (at-rest two-photon kinematics; no energetics)",
    "residual_gas_loss": "fidelity: parameterized (exponential storage survival; no cross-section)",
    "buffer_gas_cooling": (
        "fidelity: parameterized (constant-rate) or table-based "
        "(curated cross sections; see report provenance)"
    ),
    "matter_slab": "fidelity: engine transformer (vanilla Geant4 v11.4.2, FTFP_BERT)",
    "xsuite_lattice": "fidelity: externally tracked (Xsuite / xtrack)",
    "monitor": "fidelity: diagnostic snapshot",
}

_BOX_HALF_WIDTH_M = 0.05
_CIRCLE_POINTS = 41

# Animated-cloud coloring (Plotly hex), matching the WebP demo conventions.
_ACCEPTED_COLOR = "#2ca02c"
_LOST_COLOR = "#d62728"
_LEDGER_PALETTE = (
    "#1f77b4", "#ff7f0e", "#9467bd", "#8c564b",
    "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
)
_CLOUD_COLOR_MODES = ("none", "fate", "ledger", "energy")


def _cloud_marker(scene: Scene, run_result: SceneRunResult, count: int, color: str) -> dict:
    """Per-particle marker dict for the animated cloud (color is static across
    frames — it encodes a final/initial property, not a per-step one)."""
    if color not in _CLOUD_COLOR_MODES:
        raise ValueError(f"color must be one of {_CLOUD_COLOR_MODES}, got {color!r}")
    base = {"size": 3}
    if color == "none":
        return base

    final = run_result.pipeline_result.final_cloud
    if color == "energy":
        from latent_dirac.scene.build import build_source

        initial = build_source(scene).sample(np.random.default_rng(scene.seed))
        energies = initial.kinetic_energy_joule()[:count]
        return {**base, "color": energies, "colorscale": "Plasma", "showscale": True,
                "colorbar": {"title": "KE [J]"}}

    alive = final.alive[:count]
    if color == "fate":
        colors = [_ACCEPTED_COLOR if a else _LOST_COLOR for a in alive]
    else:  # ledger: accepted green, else colored by killing-element index
        lost_at = final.lost_at_element[:count]
        colors = [
            _ACCEPTED_COLOR if a else _LEDGER_PALETTE[int(idx) % len(_LEDGER_PALETTE)]
            for a, idx in zip(alive, lost_at, strict=True)
        ]
    return {**base, "color": colors}


def _fidelity_label(element) -> str:
    if element.type == "solenoid" and element.profile == "thin_sheet":
        return "fidelity: parameterized (thin-sheet profile, first-order fringe)"
    return FIDELITY_LABELS[element.type]


def render_scene_3d(scene: Scene, run_result: SceneRunResult, max_particles: int = 64):
    """Render scene elements, trajectories, and final states as a 3D figure."""

    if max_particles <= 0:
        raise ValueError("max_particles must be positive")

    go = import_optional("plotly.graph_objects", "plotly")
    figure = go.Figure()

    for element in scene.elements:
        segments = _element_segments(element, run_result)
        if not segments:
            continue
        name = f"{element.label} [{element.type}]"
        figure.add_trace(
            _wire_trace(
                go,
                name=name,
                hovertext=f"{name}<br>{_fidelity_label(element)}",
                segments=segments,
            )
        )

    for trace in _field_line_traces(go, scene, run_result):
        figure.add_trace(trace)

    combined = _combined_trajectories(scene, run_result)
    if combined is not None:
        figure.add_trace(_trajectory_trace(go, combined, max_particles))

    _add_final_state_traces(go, figure, run_result)
    _add_annihilation_traces(go, figure, run_result)

    figure.update_layout(
        title=scene.name,
        scene={
            "xaxis_title": "x [m]",
            "yaxis_title": "y [m]",
            "zaxis_title": "z [m]",
        },
    )
    return figure


def render_scene_animation(
    scene: Scene,
    run_result: SceneRunResult,
    max_particles: int = 64,
    trail: bool = True,
    color: str = "fate",
):
    """Animate the recorded cloud traversing the scene (play/pause + scrub).

    Requires `run_scene(..., record_trajectories=True)`. Element wireframes
    and the optional faint full paths are static; a single marker cloud is
    animated over the recorded steps. Lost particles are frozen in the
    recording, so they visibly stop at their loss point. Self-contained
    Plotly figure — no server.

    `color` sets the cloud coloring (a static per-particle property, so it
    holds across frames): "fate" (accepted vs lost, default), "ledger"
    (accepted vs the killing element), "energy" (initial kinetic energy on
    a Plasma ramp), or "none" (uniform).
    """
    if max_particles <= 0:
        raise ValueError("max_particles must be positive")

    combined = _combined_trajectories(scene, run_result)
    if combined is None:
        raise ValueError(
            "the animation viewer needs recorded trajectories; call "
            "run_scene(scene, record_trajectories=True) first"
        )

    go = import_optional("plotly.graph_objects", "plotly")
    count = min(combined.shape[1], max_particles)
    steps = combined.shape[0]
    marker = _cloud_marker(scene, run_result, count, color)

    static_traces = []
    for element in scene.elements:
        segments = _element_segments(element, run_result)
        if not segments:
            continue
        name = f"{element.label} [{element.type}]"
        static_traces.append(
            _wire_trace(
                go,
                name=name,
                hovertext=f"{name}<br>{_fidelity_label(element)}",
                segments=segments,
            )
        )
    static_traces.extend(_field_line_traces(go, scene, run_result))
    if trail:
        static_traces.append(_trajectory_trace(go, combined, max_particles))

    def _cloud(step: int):
        frame = combined[step, :count, :]
        return go.Scatter3d(
            x=frame[:, 0],
            y=frame[:, 1],
            z=frame[:, 2],
            mode="markers",
            name="cloud",
            marker=marker,
        )

    figure = go.Figure(data=[*static_traces, _cloud(0)])
    figure.frames = [
        go.Frame(data=[_cloud(step)], traces=[len(static_traces)], name=str(step))
        for step in range(steps)
    ]

    play = {
        "label": "Play",
        "method": "animate",
        "args": [None, {"frame": {"duration": 60, "redraw": True}, "fromcurrent": True}],
    }
    pause = {
        "label": "Pause",
        "method": "animate",
        "args": [[None], {"frame": {"duration": 0, "redraw": False}, "mode": "immediate"}],
    }
    slider = {
        "steps": [
            {
                "label": str(step),
                "method": "animate",
                "args": [[str(step)], {"frame": {"duration": 0, "redraw": True}, "mode": "immediate"}],
            }
            for step in range(steps)
        ],
        "x": 0.1,
        "len": 0.9,
        "currentvalue": {"prefix": "step "},
    }
    figure.update_layout(
        title=scene.name,
        scene={"xaxis_title": "x [m]", "yaxis_title": "y [m]", "zaxis_title": "z [m]"},
        updatemenus=[{"type": "buttons", "buttons": [play, pause]}],
        sliders=[slider],
    )
    return figure


def _element_segments(element, run_result: SceneRunResult) -> list[np.ndarray]:
    if element.type == "solenoid":
        return _cylinder_segments(element.radius_m, element.center_z_m, element.length_m)
    if element.type in {"dipole", "quadrupole"}:
        return _box_segments(element.center_z_m, element.length_m, _BOX_HALF_WIDTH_M)
    if element.type == "aperture":
        return [
            _circle(element.radius_m, element.z_m),
            _circle(1.5 * element.radius_m, element.z_m),
        ]
    if element.type == "momentum_window":
        return []
    if element.type == "annihilation_plate":
        return [
            _circle(element.radius_m, element.z_m),
            _circle(0.5 * element.radius_m, element.z_m),
        ]
    if element.type == "matter_slab":
        thickness_m = element.thickness_mm / 1000.0
        center_z_m = element.entry_z_m + thickness_m / 2.0
        return _box_segments(center_z_m, thickness_m, _BOX_HALF_WIDTH_M)
    if element.type == "xsuite_lattice":
        if element.length_m is not None:
            return _box_segments(element.center_z_m, element.length_m, _BOX_HALF_WIDTH_M)
        return _square_segments(element.center_z_m, _BOX_HALF_WIDTH_M)
    if element.type == "monitor":
        snapshot = run_result.monitors.get(element.label)
        if snapshot is None:
            return []
        positions = snapshot.position_m[snapshot.alive] if np.any(snapshot.alive) else snapshot.position_m
        z_m = float(np.mean(positions[:, 2]))
        return _square_segments(z_m, _BOX_HALF_WIDTH_M)
    return []


def _circle(radius_m: float, z_m: float) -> np.ndarray:
    theta = np.linspace(0.0, 2.0 * np.pi, _CIRCLE_POINTS)
    return np.column_stack([radius_m * np.cos(theta), radius_m * np.sin(theta), np.full_like(theta, z_m)])


def _cylinder_segments(radius_m: float, center_z_m: float, length_m: float) -> list[np.ndarray]:
    z_low = center_z_m - 0.5 * length_m
    z_high = center_z_m + 0.5 * length_m
    segments = [_circle(radius_m, z_low), _circle(radius_m, z_high)]
    for theta in np.linspace(0.0, 2.0 * np.pi, 4, endpoint=False):
        x = radius_m * np.cos(theta)
        y = radius_m * np.sin(theta)
        segments.append(np.array([[x, y, z_low], [x, y, z_high]]))
    return segments


def _square(half_width_m: float, z_m: float) -> np.ndarray:
    return np.array(
        [
            [-half_width_m, -half_width_m, z_m],
            [half_width_m, -half_width_m, z_m],
            [half_width_m, half_width_m, z_m],
            [-half_width_m, half_width_m, z_m],
            [-half_width_m, -half_width_m, z_m],
        ]
    )


def _square_segments(z_m: float, half_width_m: float) -> list[np.ndarray]:
    square = _square(half_width_m, z_m)
    return [square, square[[0, 2]], square[[1, 3]]]


def _box_segments(center_z_m: float, length_m: float, half_width_m: float) -> list[np.ndarray]:
    z_low = center_z_m - 0.5 * length_m
    z_high = center_z_m + 0.5 * length_m
    segments = [_square(half_width_m, z_low), _square(half_width_m, z_high)]
    for x_sign in (-1.0, 1.0):
        for y_sign in (-1.0, 1.0):
            segments.append(
                np.array(
                    [
                        [x_sign * half_width_m, y_sign * half_width_m, z_low],
                        [x_sign * half_width_m, y_sign * half_width_m, z_high],
                    ]
                )
            )
    return segments


def _wire_trace(go, name: str, hovertext: str, segments: list[np.ndarray]):
    xs: list[float | None] = []
    ys: list[float | None] = []
    zs: list[float | None] = []
    for segment in segments:
        xs.extend(segment[:, 0].tolist())
        ys.extend(segment[:, 1].tolist())
        zs.extend(segment[:, 2].tolist())
        xs.append(None)
        ys.append(None)
        zs.append(None)
    return go.Scatter3d(
        x=xs,
        y=ys,
        z=zs,
        mode="lines",
        name=name,
        hovertext=hovertext,
        hoverinfo="text",
        line={"width": 3},
    )


def _combined_trajectories(scene: Scene, run_result: SceneRunResult) -> np.ndarray | None:
    parts: list[np.ndarray] = []
    for element in scene.elements:
        history = run_result.trajectories.get(element.label)
        if history is None:
            continue
        parts.append(history if not parts else history[1:])
    if not parts:
        return None
    return np.concatenate(parts, axis=0)


def _trajectory_trace(go, combined: np.ndarray, max_particles: int):
    particle_count = min(combined.shape[1], max_particles)
    xs: list[float | None] = []
    ys: list[float | None] = []
    zs: list[float | None] = []
    for particle in range(particle_count):
        path = combined[:, particle, :]
        xs.extend(path[:, 0].tolist())
        ys.extend(path[:, 1].tolist())
        zs.extend(path[:, 2].tolist())
        xs.append(None)
        ys.append(None)
        zs.append(None)
    return go.Scatter3d(
        x=xs,
        y=ys,
        z=zs,
        mode="lines",
        name="trajectories",
        opacity=0.6,
        line={"width": 2},
    )


_FIELD_LINE_TRACE_COLORS = {"B": "#597fbf", "E": "#d9982e"}


def _field_line_traces(go, scene: Scene, run_result: SceneRunResult) -> list:
    """Field-line traces of each model field (visual diagnostics).

    Extent comes from the recorded trajectories; without them the beam
    footprint is unknown and no lines are drawn.
    """

    from latent_dirac.scene.build import _field_for
    from latent_dirac.viz.field_lines import element_field_line_bundles, field_elements_for_lines

    combined = _combined_trajectories(scene, run_result)
    if combined is None:
        return []
    extent = {
        "transverse_m": max(float(np.nanmax(np.abs(combined[..., :2]))), 1e-6),
        "axial_m": max(
            0.5 * (float(np.nanmax(combined[..., 2])) - float(np.nanmin(combined[..., 2]))), 1e-6
        ),
    }
    traces = []
    for element in field_elements_for_lines(scene):
        bundles = element_field_line_bundles(element, _field_for(element), extent)
        for kind, color in _FIELD_LINE_TRACE_COLORS.items():
            xs: list[float | None] = []
            ys: list[float | None] = []
            zs: list[float | None] = []
            for bundle_kind, line in bundles:
                if bundle_kind != kind:
                    continue
                xs.extend([*line[:, 0].tolist(), None])
                ys.extend([*line[:, 1].tolist(), None])
                zs.extend([*line[:, 2].tolist(), None])
            if not xs:
                continue
            traces.append(
                go.Scatter3d(
                    x=xs,
                    y=ys,
                    z=zs,
                    mode="lines",
                    name=f"{element.label} {kind} field lines",
                    hovertext=f"field lines of the model field<br>{_fidelity_label(element)}",
                    hoverinfo="text",
                    line={"width": 1.5, "color": color},
                    opacity=0.45,
                )
            )
    return traces


def _add_annihilation_traces(go, figure, run_result: SceneRunResult) -> None:
    """Back-to-back photon rays per annihilation event (kinematics only).

    511 keV appears in the hover text as a label; no energetics — see the
    safety scope.
    """

    for label, events in run_result.annihilations.items():
        positions = events.get("positions")
        if positions is None or positions.shape[0] == 0:
            continue
        directions = events["photon_directions"]
        extent = float(np.max(np.ptp(positions, axis=0))) if positions.shape[0] > 1 else 0.0
        ray_length = max(extent, 0.01)
        xs: list[float | None] = []
        ys: list[float | None] = []
        zs: list[float | None] = []
        for start, pair in zip(positions, directions, strict=True):
            for direction in pair:
                end = start + ray_length * direction
                xs.extend([start[0], end[0], None])
                ys.extend([start[1], end[1], None])
                zs.extend([start[2], end[2], None])
        figure.add_trace(
            go.Scatter3d(
                x=xs,
                y=ys,
                z=zs,
                mode="lines",
                name=f"{label} photons",
                hovertext="at-rest two-photon kinematics; 511 keV label only; no energetics",
                hoverinfo="text",
                line={"width": 2, "color": "goldenrod"},
                opacity=0.55,
            )
        )


def _add_final_state_traces(go, figure, run_result: SceneRunResult) -> None:
    final = run_result.pipeline_result.final_cloud
    for name, mask in (("accepted", final.alive), ("lost", ~final.alive)):
        positions = final.position_m[mask]
        figure.add_trace(
            go.Scatter3d(
                x=positions[:, 0],
                y=positions[:, 1],
                z=positions[:, 2],
                mode="markers",
                name=name,
                marker={"size": 3},
            )
        )
