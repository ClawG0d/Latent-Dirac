"""Generate the 3D README demo animations from real simulation runs.

Scene-driven demos are defined by the YAML files under `examples/scenes/`
and rendered through the shared matplotlib pipeline in `tools/mpl3d.py`.
The magnetic mirror and the magnetic control sweep are rendered from
direct kernel runs (two matched species and table-based field maps are not
scene-expressible yet). Scene-driven demos also export an interactive
Plotly HTML via `render_scene_3d` when Plotly is available.
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
from examples.magnetic_mirror_demo import FIELD_MODEL_LABEL, MIRROR_HALF_LENGTH_M, run_trajectories
from latent_dirac.fields.uniform import UniformField
from latent_dirac.scene.build import build_source, run_scene
from latent_dirac.scene.loader import load_scene
from latent_dirac.solvers.relativistic_boris import RelativisticBorisSolver
from latent_dirac.viz.scene_3d import _combined_trajectories
from tools import mpl3d

SCENES_DIR = PROJECT_ROOT / "examples" / "scenes"

SCENE_DEMOS = {
    "decay_emission_3d.webp": {
        "scene": "decay_emission.yaml",
        "title": "Beta-plus decay emission - isotropic e+ from a Na-22-like pellet\n"
        "parameterized Beta(3,3) spectrum approximation | no nuclear detail | "
        "trail color = initial kinetic energy",
        "coloring": "energy",
    },
    "scene_tour_3d.webp": {
        "scene": "scene_tour.yaml",
        "title": "YAML scene tour - beta+ source, guide solenoid, collimator\n"
        "hard-edge optics models | relativistic Boris solver | transport diagnostic only",
        "coloring": "fate",
    },
    "positron_capture_3d.webp": {
        "scene": "positron_capture.yaml",
        "title": "Positron capture - pair source spiraling in a solenoid\n"
        "hard-edge solenoid | relativistic Boris solver | transport and acceptance diagnostic only",
        "coloring": "fate",
    },
    "dipole_quad_line_3d.webp": {
        "scene": "dipole_quad_line.yaml",
        "title": "Dipole bend + quadrupole focusing\n"
        "hard-edge optics models | relativistic Boris solver | transport diagnostic only",
        "coloring": "fate",
    },
    "wien_filter_3d.webp": {
        "scene": "wien_filter.yaml",
        "title": "Wien velocity filter - crossed E and B fields select v = E/B\n"
        "uniform fields | relativistic Boris solver | velocity-selection diagnostic only",
        "coloring": "fate",
    },
    "target_production_3d.webp": {
        "scene": "target_production.yaml",
        "title": "Antiproton production - SURROGATE accepted-source model\n"
        "target physics NOT simulated (drawn only; Geant4 adapter: roadmap) | "
        "uniform capture field | relativistic Boris solver",
        "coloring": "fate",
        "annotate": "target",
    },
    "target_production_engine_3d.webp": {
        "scene": "target_production_engine.yaml",
        "title": "Antiproton production - ENGINE-BACKED table-based source\n"
        "vanilla Geant4 v11.4.2 FTFP_BERT yield table (2M protons on Ir) | "
        "relativistic Boris solver | acceptance diagnostic only",
        "coloring": "fate",
        "annotate": "target_engine",
        # the yield table holds the full exit phase space; wide-angle
        # spirals would blow up the auto limits, so clamp the view to the
        # collection line and draw a readable subset of the 2547 trails
        "limits": [(-0.12, 0.12), (-0.12, 0.12), (-0.08, 0.62)],
        "max_trails": 480,
        "render": {"box_aspect": (2.4, 1.1, 1.1), "azim_start": -70, "azim_sweep": 50},
    },
    "decel_capture_3d.webp": {
        "scene": "decel_capture.yaml",
        "title": "Electrostatic deceleration and dynamic trap capture\n"
        "magnets do no work - the E field decelerates | trap gated on after entry | "
        "relativistic Boris solver",
        "coloring": "fate",
        "render": {"box_aspect": (3.0, 1.0, 1.0), "azim_start": -78, "azim_sweep": 36, "elev": 14},
    },
    "annihilation_endpoint_3d.webp": {
        "scene": "annihilation_endpoint.yaml",
        "title": "Annihilation endpoint - every positron's fate is ledgered\n"
        "at-rest two-photon kinematics (511 keV label only; NO energetics) | "
        "relativistic Boris solver",
        "coloring": "ledger",
        "annotate": "annihilation",
    },
    "antiproton_ledger_3d.webp": {
        "scene": "antiproton_ledger.yaml",
        "title": "Antiproton loss ledger - trajectory color = killing element\n"
        "uniform field | relativistic Boris solver | acceptance ledger diagnostic only",
        "coloring": "ledger",
    },
}

DIRECT_DEMOS = ("magnetic_mirror_3d.webp", "magnetic_control_sweep_3d.webp", "batched_sweep_3d.webp")
DEMO_WEBP_FILES = tuple(SCENE_DEMOS) + DIRECT_DEMOS


def _particle_colors_energy(scene):
    """Trail colors from initial kinetic energy (plasma ramp, slow to fast)."""

    initial = build_source(scene).sample(np.random.default_rng(scene.seed))
    energies = initial.kinetic_energy_joule()
    low, high = float(energies.min()), float(energies.max())
    normalized = (energies - low) / max(high - low, 1e-300)
    plt, _ = mpl3d.load_matplotlib()
    colormap = plt.get_cmap("plasma")
    return [colormap(float(value))[:3] for value in normalized]


def _particle_colors(final_state, coloring: str):
    if coloring == "ledger":
        colors = []
        for alive, lost_at in zip(final_state.alive, final_state.lost_at_element, strict=True):
            if alive:
                colors.append(mpl3d.ACCEPTED)
            else:
                colors.append(mpl3d.LEDGER_PALETTE[int(lost_at) % len(mpl3d.LEDGER_PALETTE)])
        return colors
    return [mpl3d.ACCEPTED if alive else mpl3d.LOST for alive in final_state.alive]


def _scene_demo_frames(scene_name: str, title: str, coloring: str, frame_count: int, config=None):
    scene = load_scene(SCENES_DIR / scene_name)
    run_result = run_scene(scene, record_trajectories=True)
    combined = _combined_trajectories(scene, run_result)
    final_state = run_result.pipeline_result.final_cloud
    if coloring == "energy":
        colors = _particle_colors_energy(scene)
    else:
        colors = _particle_colors(final_state, coloring)

    max_trails = (config or {}).get("max_trails")
    if max_trails is not None and combined.shape[1] > max_trails:
        picked = np.random.default_rng(scene.seed).choice(combined.shape[1], max_trails, replace=False)
        combined = combined[:, picked]
        colors = [colors[int(index)] for index in picked]

    limits = (config or {}).get("limits") or mpl3d.axis_limits(combined)
    total = combined.shape[0]

    annotate = config.get("annotate") if config else None
    beam_extent = float(np.nanmax(np.abs(combined[..., :2])))
    z_span = float(combined[..., 2].max()) - float(combined[..., 2].min())
    photon_rays = None
    if annotate == "annihilation" and run_result.annihilations:
        events = next(iter(run_result.annihilations.values()))
        ray_length = 0.25 * z_span
        starts = events["positions"]
        photon_rays = [
            (starts, starts + ray_length * events["photon_directions"][:, 0, :]),
            (starts, starts + ray_length * events["photon_directions"][:, 1, :]),
        ]

    def draw(axes, index, count):
        reveal = 2 + int(round((total - 2) * index / max(count - 1, 1)))
        if annotate == "target":
            mpl3d.draw_block(axes, -0.12 * z_span, 0.0, 1.1 * beam_extent)
            mpl3d.draw_beam_arrow(axes, -0.45 * z_span, -0.13 * z_span)
        if annotate == "target_engine":
            # the real yieldgen target (r=1.5 mm, half-length 27.5 mm) is
            # invisible at beamline scale; draw a modest, clearly labeled
            # stand-in at its true z extent instead of a scaled block
            mpl3d.draw_block(axes, -0.0275, 0.0275, 0.014)
            mpl3d.draw_beam_arrow(axes, -0.07, -0.032)
        mpl3d.draw_scene_elements(axes, scene, run_result)
        mpl3d.draw_trajectories(axes, combined, reveal, colors)
        mpl3d.draw_points(axes, combined[reveal - 1], colors)
        if photon_rays is not None and index > count * 0.55:
            gold = (0.85, 0.65, 0.13)
            for starts_arr, ends_arr in photon_rays:
                for start, end in zip(starts_arr, ends_arr, strict=True):
                    axes.plot(
                        [start[2], end[2]],
                        [start[0], end[0]],
                        [start[1], end[1]],
                        color=gold,
                        linewidth=0.6,
                        alpha=0.6,
                    )

    render_kwargs = (config or {}).get("render", {})
    return mpl3d.render_frames(draw, frame_count, title, limits, **render_kwargs)


def _write_scene_html(scene_name: str, output_path: Path) -> Path | None:
    scene = load_scene(SCENES_DIR / scene_name)
    run_result = run_scene(scene, record_trajectories=True)
    try:
        from latent_dirac.viz.scene_3d import render_scene_3d

        figure = render_scene_3d(scene, run_result)
    except ImportError:
        return None
    figure.write_html(output_path, include_plotlyjs="cdn")
    return output_path


def _mirror_frames(frame_count: int):
    outcome = run_trajectories(count=24, steps=6000, record_every=30)
    trajectory = outcome["trajectory"]
    trapped = outcome["trapped"]
    colors = [mpl3d.ACCEPTED if flag else mpl3d.LOST for flag in trapped]

    # flux-tube outline: r(z) = r_edge * sqrt(B0 / Bz(z)) narrows at the throats
    r_edge = 0.0015
    z_line = np.linspace(-MIRROR_HALF_LENGTH_M, MIRROR_HALF_LENGTH_M, 61)
    r_line = r_edge / np.sqrt(1.0 + (z_line / MIRROR_HALF_LENGTH_M) ** 2)

    bottle = np.array([r_line, z_line])
    inside = np.abs(trajectory[:, :, 2]) <= MIRROR_HALF_LENGTH_M
    visible = trajectory
    total = visible.shape[0]
    limits = [
        (-0.002, 0.002),
        (-0.002, 0.002),
        (-MIRROR_HALF_LENGTH_M * 1.1, MIRROR_HALF_LENGTH_M * 1.1),
    ]

    title = (
        "Magnetic mirror bottle - trapped positrons bounce between throats\n"
        f"{FIELD_MODEL_LABEL} | relativistic Boris solver | single-particle transport only"
    )

    def draw(axes, index, count):
        reveal = 2 + int(round((total - 2) * index / max(count - 1, 1)))
        theta = np.linspace(0.0, 2.0 * np.pi, 24)
        for angle in theta[::4]:
            axes.plot(
                bottle[1],
                bottle[0] * np.cos(angle),
                bottle[0] * np.sin(angle),
                color=mpl3d.ELEMENT,
                linewidth=0.5,
                alpha=0.35,
            )
        for z_throat in (-MIRROR_HALF_LENGTH_M, MIRROR_HALF_LENGTH_M):
            mpl3d.draw_disc(axes, 0.001, 0.0018, z_throat, alpha=0.2)
        mask = inside[:reveal]
        clipped = np.where(mask[..., np.newaxis], visible[:reveal], np.nan)
        mpl3d.draw_trajectories(axes, clipped, reveal, colors, linewidth=0.6, alpha=0.6)

    return mpl3d.render_frames(
        draw,
        frame_count,
        title,
        limits,
        box_aspect=(2.4, 1.0, 1.0),
        azim_start=-70,
        azim_sweep=60,
    )


def _sweep_frames(frame_count: int, particle_count: int = 64):
    field_values = np.linspace(0.0, 0.6, frame_count)
    aperture_x = 0.035
    dt_s, steps, snapshots = 2.0e-12, 80, 17

    all_runs = []
    for by_tesla in field_values:
        positron_cloud, electron_cloud = make_initial_pair(particle_count=particle_count, seed=2031)
        field = UniformField(B_vector_t=np.array([0.0, by_tesla, 0.0]))
        stepper = RelativisticBorisSolver(dt_s=dt_s, steps=steps // (snapshots - 1))
        histories = []
        for cloud in (positron_cloud, electron_cloud):
            current = cloud
            track = [current.position_m.copy()]
            for _ in range(snapshots - 1):
                current = stepper.propagate(current, field)
                track.append(current.position_m.copy())
            histories.append(np.stack(track))
        all_runs.append((by_tesla, histories))

    limits = mpl3d.axis_limits(*[h for _, hs in all_runs for h in hs])

    def draw(axes, index, count):
        by_tesla, histories = all_runs[index]
        for history, color in zip(histories, (mpl3d.POSITRON, mpl3d.ELECTRON), strict=True):
            accepted = np.abs(history[-1, :, 0]) <= aperture_x
            colors = [color if flag else mpl3d.LOST for flag in accepted]
            mpl3d.draw_trajectories(axes, history, len(history), colors, alpha=0.5)
            mpl3d.draw_points(axes, history[-1], colors)
        axes.text2D(
            0.72,
            0.9,
            f"By = {by_tesla:.2f} T",
            transform=axes.transAxes,
            fontsize=11,
        )

    title = (
        "Magnetic control sweep - charge-sign separation vs By\n"
        "uniform transverse field | relativistic Boris solver | aperture diagnostic only"
    )
    return mpl3d.render_frames(draw, frame_count, title, limits, azim_sweep=50)


def _batched_sweep_frames(frame_count: int, config_count: int = 24):
    """One jit-compiled vmap launch runs every configuration; JAX required."""

    try:
        import jax
    except ModuleNotFoundError as exc:
        raise ImportError('the batched sweep demo requires jax: pip install "latent-dirac[jax]"') from exc
    jax.config.update("jax_enable_x64", True)
    from latent_dirac.backends.jax_scene import BatchedSceneProgram

    scene = load_scene(SCENES_DIR / "batched_sweep.yaml")
    program = BatchedSceneProgram(scene, override_keys=("sweep-field.B_vector_t",), record_stride=4)
    by_values = np.linspace(0.0, 0.6, config_count)
    b_vectors = np.stack([np.zeros_like(by_values), by_values, np.zeros_like(by_values)], axis=1)
    result = program.run({"sweep-field.B_vector_t": b_vectors})
    trajectories = result.trajectories  # (B, S, N, 3)

    plt, _ = mpl3d.load_matplotlib()
    colormap = plt.get_cmap("viridis")
    colors = [colormap(index / max(config_count - 1, 1))[:3] for index in range(config_count)]
    limits = mpl3d.axis_limits(trajectories)
    total = trajectories.shape[1]

    title = (
        f"Batched sweep - one launch, {config_count} configurations (By 0 to 0.6 T)\n"
        "uniform transverse field | relativistic Boris solver | vmap over configurations | "
        "transport diagnostic only"
    )

    def draw(axes, index, count):
        reveal = 2 + int(round((total - 2) * index / max(count - 1, 1)))
        for config in range(config_count):
            mpl3d.draw_trajectories(
                axes, trajectories[config], reveal, colors[config], linewidth=0.5, alpha=0.45
            )
            mpl3d.draw_points(axes, trajectories[config, reveal - 1], colors[config], size=5)

    return mpl3d.render_frames(draw, frame_count, title, limits, azim_sweep=60)


def generate_scene_demo_webps(
    output_dir: str | Path = "assets/demos",
    frame_count: int = 44,
    duration_ms: int = 85,
    write_html: bool = True,
) -> dict[str, Path]:
    """Generate all 3D demo animations; returns output paths by file name."""

    if frame_count < 2:
        raise ValueError("frame_count must be at least 2")

    output_path = Path(output_dir)
    outputs: dict[str, Path] = {}

    for file_name, config in SCENE_DEMOS.items():
        frames = _scene_demo_frames(
            config["scene"], config["title"], config["coloring"], frame_count, config=config
        )
        target = output_path / file_name
        mpl3d.save_webp(frames, target, duration_ms)
        outputs[file_name] = target
        if write_html:
            html_name = file_name.replace(".webp", ".html")
            written = _write_scene_html(config["scene"], output_path / html_name)
            if written is not None:
                outputs[html_name] = written

    mirror_target = output_path / "magnetic_mirror_3d.webp"
    mpl3d.save_webp(_mirror_frames(frame_count), mirror_target, duration_ms)
    outputs["magnetic_mirror_3d.webp"] = mirror_target

    sweep_target = output_path / "magnetic_control_sweep_3d.webp"
    mpl3d.save_webp(_sweep_frames(frame_count), sweep_target, duration_ms)
    outputs["magnetic_control_sweep_3d.webp"] = sweep_target

    batched_target = output_path / "batched_sweep_3d.webp"
    mpl3d.save_webp(_batched_sweep_frames(frame_count), batched_target, duration_ms)
    outputs["batched_sweep_3d.webp"] = batched_target

    return outputs


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="assets/demos")
    parser.add_argument("--frames", type=int, default=44)
    parser.add_argument("--duration-ms", type=int, default=85)
    parser.add_argument("--no-html", action="store_true")
    args = parser.parse_args()

    generated = generate_scene_demo_webps(
        args.output_dir, args.frames, args.duration_ms, write_html=not args.no_html
    )
    for path in generated.values():
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
