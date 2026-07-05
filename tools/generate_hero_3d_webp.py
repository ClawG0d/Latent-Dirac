"""Generate the animated 3D charge-sign splitter hero asset for the README.

The animation is rendered from the recorded ``Trajectory`` of a real
relativistic Boris-solver run: matched positron/electron clouds in the same
uniform transverse magnetic field. Matplotlib renders the 3D frames and
Pillow assembles the WebP. When Plotly is installed, an interactive HTML
version is also written through ``PlotlyBackend.plot_trajectory_3d``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from examples.charge_sign_splitter_demo import make_initial_pair
from latent_dirac.fields.uniform import UniformField
from latent_dirac.solvers.relativistic_boris import RelativisticBorisSolver
from latent_dirac.state.particle_state import ParticleState
from latent_dirac.state.trajectory import Trajectory

HERO_WEBP_FILE = "charge_sign_splitter_3d.webp"
HERO_HTML_FILE = "charge_sign_splitter_3d.html"

POSITRON_COLOR = (20 / 255, 118 / 255, 198 / 255)
ELECTRON_COLOR = (225 / 255, 118 / 255, 35 / 255)

FIELD_BY_TESLA = 0.45
DT_S = 2.0e-12
STEPS = 80


def _load_pillow():
    try:
        from PIL import Image
    except ModuleNotFoundError as exc:
        raise ImportError(
            'Pillow is required to generate the hero WebP. Install it with '
            '`pip install "latent-dirac[viz]"`.'
        ) from exc
    return Image


def _load_matplotlib():
    try:
        import matplotlib
    except ModuleNotFoundError as exc:
        raise ImportError(
            'Matplotlib is required to generate the hero WebP. Install it with '
            '`pip install "latent-dirac[viz]"`.'
        ) from exc
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator

    return plt, MaxNLocator


def record_trajectory(
    cloud: ParticleState,
    field,
    dt_s: float,
    steps: int,
) -> Trajectory:
    """Propagate one step at a time and record the full particle history."""

    solver = RelativisticBorisSolver(dt_s=dt_s, steps=1)
    clouds = [cloud]
    current = cloud
    for _ in range(steps):
        current = solver.propagate(current, field)
        clouds.append(current)

    return Trajectory(
        time_s=np.array([float(np.mean(snapshot.time_s)) for snapshot in clouds]),
        position_m=np.stack([snapshot.position_m for snapshot in clouds]),
        momentum_kg_m_s=np.stack([snapshot.momentum_kg_m_s for snapshot in clouds]),
    )


def make_hero_trajectories(particle_count: int, seed: int = 2030) -> tuple[Trajectory, Trajectory]:
    positron_cloud, electron_cloud = make_initial_pair(particle_count=particle_count, seed=seed)
    field = UniformField(B_vector_t=np.array([0.0, FIELD_BY_TESLA, 0.0]))
    return (
        record_trajectory(positron_cloud, field, DT_S, STEPS),
        record_trajectory(electron_cloud, field, DT_S, STEPS),
    )


def _axis_limits(trajectories: tuple[Trajectory, ...]) -> list[tuple[float, float]]:
    stacked = np.concatenate([trajectory.position_m.reshape(-1, 3) for trajectory in trajectories])
    limits = []
    for axis in range(3):
        low = float(stacked[:, axis].min())
        high = float(stacked[:, axis].max())
        pad = 0.08 * max(high - low, 1e-6)
        limits.append((low - pad, high + pad))
    return limits


def _render_frames(
    positron_trajectory: Trajectory,
    electron_trajectory: Trajectory,
    frame_count: int,
):
    plt, MaxNLocator = _load_matplotlib()
    image_module = _load_pillow()

    limits = _axis_limits((positron_trajectory, electron_trajectory))
    total_steps = positron_trajectory.position_m.shape[0]

    frames = []
    for index in range(frame_count):
        reveal = 2 + int(round((total_steps - 2) * index / max(frame_count - 1, 1)))
        azimuth = -62.0 + 80.0 * index / max(frame_count - 1, 1)

        fig = plt.figure(figsize=(9.2, 5.0), dpi=100)
        axes = fig.add_subplot(projection="3d")
        for trajectory, color, label in (
            (positron_trajectory, POSITRON_COLOR, "positron e+"),
            (electron_trajectory, ELECTRON_COLOR, "electron e-"),
        ):
            positions = trajectory.position_m[:reveal]
            for particle in range(positions.shape[1]):
                axes.plot(
                    positions[:, particle, 2],
                    positions[:, particle, 0],
                    positions[:, particle, 1],
                    color=color,
                    linewidth=0.7,
                    alpha=0.55,
                )
            axes.scatter(
                positions[-1, :, 2],
                positions[-1, :, 0],
                positions[-1, :, 1],
                color=color,
                s=9,
                depthshade=False,
                label=label,
            )

        axes.set_xlim(*limits[2])
        axes.set_ylim(*limits[0])
        axes.set_zlim(*limits[1])
        axes.set_xlabel("z [m]", labelpad=10)
        axes.set_ylabel("x [m]", labelpad=10)
        axes.set_zlabel("y [m]", labelpad=6)
        for axis in (axes.xaxis, axes.yaxis, axes.zaxis):
            axis.set_major_locator(MaxNLocator(4))
        axes.tick_params(labelsize=8, pad=2)
        axes.view_init(elev=18.0, azim=azimuth)
        axes.set_box_aspect((2.0, 1.2, 0.8), zoom=1.16)
        axes.set_position((0.02, 0.05, 0.96, 0.87))
        axes.legend(loc="upper left", frameon=False, bbox_to_anchor=(0.06, 0.94))
        fig.suptitle(
            "Charge-sign splitter, 3D transport\n"
            f"uniform transverse By = {FIELD_BY_TESLA:.2f} T | relativistic Boris solver | "
            "transport diagnostic only",
            fontsize=10,
            y=0.97,
        )

        fig.canvas.draw()
        buffer = np.asarray(fig.canvas.buffer_rgba())
        frames.append(image_module.fromarray(buffer[..., :3].copy()))
        plt.close(fig)

    return frames


def _save_webp(frames, path: Path, duration_ms: int) -> None:
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


def _save_interactive_html(
    positron_trajectory: Trajectory,
    electron_trajectory: Trajectory,
    path: Path,
) -> Path | None:
    try:
        from latent_dirac.viz.plotly_backend import PlotlyBackend
        combined = Trajectory(
            time_s=positron_trajectory.time_s,
            position_m=np.concatenate(
                [positron_trajectory.position_m, electron_trajectory.position_m], axis=1
            ),
            momentum_kg_m_s=np.concatenate(
                [positron_trajectory.momentum_kg_m_s, electron_trajectory.momentum_kg_m_s], axis=1
            ),
        )
        figure = PlotlyBackend().plot_trajectory_3d(combined, max_particles=64)
    except ImportError:
        return None

    figure.write_html(path, include_plotlyjs="cdn")
    return path


def generate_hero_3d_webp(
    output_dir: str | Path = "assets/demos",
    frame_count: int = 44,
    particle_count: int = 48,
    duration_ms: int = 85,
    write_html: bool = True,
) -> dict[str, Path]:
    """Generate the 3D hero animation and return output paths by file name."""

    if frame_count < 2:
        raise ValueError("frame_count must be at least 2")
    if particle_count < 1:
        raise ValueError("particle_count must be positive")

    positron_trajectory, electron_trajectory = make_hero_trajectories(particle_count)

    output_path = Path(output_dir)
    outputs = {HERO_WEBP_FILE: output_path / HERO_WEBP_FILE}
    frames = _render_frames(positron_trajectory, electron_trajectory, frame_count)
    _save_webp(frames, outputs[HERO_WEBP_FILE], duration_ms)

    if write_html:
        html_path = _save_interactive_html(
            positron_trajectory, electron_trajectory, output_path / HERO_HTML_FILE
        )
        if html_path is not None:
            outputs[HERO_HTML_FILE] = html_path

    return outputs


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="assets/demos")
    parser.add_argument("--frames", type=int, default=44)
    parser.add_argument("--particles", type=int, default=48)
    parser.add_argument("--duration-ms", type=int, default=85)
    parser.add_argument("--no-html", action="store_true")
    args = parser.parse_args()

    generated = generate_hero_3d_webp(
        args.output_dir,
        args.frames,
        args.particles,
        args.duration_ms,
        write_html=not args.no_html,
    )
    for path in generated.values():
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
