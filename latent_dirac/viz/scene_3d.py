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
    "drift": "fidelity: exact zero-field transport",
    "aperture": "fidelity: diagnostic acceptance cut",
    "momentum_window": "fidelity: diagnostic acceptance cut",
    "annihilation_plate": "fidelity: parameterized (at-rest two-photon kinematics; no energetics)",
    "residual_gas_loss": "fidelity: parameterized (exponential storage survival; no cross-section)",
    "matter_slab": "fidelity: engine transformer (vanilla Geant4 v11.4.2, FTFP_BERT)",
    "xsuite_lattice": "fidelity: externally tracked (Xsuite / xtrack)",
    "monitor": "fidelity: diagnostic snapshot",
}

_BOX_HALF_WIDTH_M = 0.05
_CIRCLE_POINTS = 41


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
                hovertext=f"{name}<br>{FIDELITY_LABELS[element.type]}",
                segments=segments,
            )
        )

    combined = _combined_trajectories(scene, run_result)
    if combined is not None:
        figure.add_trace(_trajectory_trace(go, combined, max_particles))

    _add_final_state_traces(go, figure, run_result)

    figure.update_layout(
        title=scene.name,
        scene={
            "xaxis_title": "x [m]",
            "yaxis_title": "y [m]",
            "zaxis_title": "z [m]",
        },
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
