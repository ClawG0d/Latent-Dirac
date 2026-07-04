"""Generate animated WebP demo assets for the README."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from latent_dirac.beamline.aperture import Aperture
from latent_dirac.beamline.momentum_window import MomentumWindow
from latent_dirac.core.units import momentum_gev_c_to_si
from latent_dirac.fields.solenoid import SolenoidField
from latent_dirac.fields.uniform import UniformField
from latent_dirac.solvers.relativistic_boris import RelativisticBorisSolver
from latent_dirac.sources.antiproton_surrogate import AntiprotonSurrogateSource
from latent_dirac.sources.positron_pair import PositronPairSource
from latent_dirac.state.particle_cloud import ParticleCloud
from examples.charge_sign_splitter_demo import make_initial_pair
from examples.magnetic_control_sweep_demo import DEFAULT_FIELD_VALUES_T, run_sweep

DEMO_WEBP_FILES = (
    "charge_sign_splitter.webp",
    "positron_capture.webp",
    "antiproton_transport.webp",
    "magnetic_control_sweep.webp",
)

CANVAS_SIZE = (920, 500)
PLOT_BOX = (82, 112, 654, 292)
BACKGROUND = (252, 253, 255)
INK = (18, 22, 28)
MUTED = (105, 116, 130)
GRID = (223, 229, 237)
POSITRON = (20, 118, 198)
ELECTRON = (225, 118, 35)
ANTIPROTON = (129, 80, 197)
ACCEPTED = (20, 143, 87)
LOST = (210, 76, 69)
FIELD = (236, 245, 255)


def _load_pillow():
    try:
        from PIL import Image, ImageDraw, ImageFont
    except ModuleNotFoundError as exc:
        raise ImportError(
            'Pillow is required to generate README WebP demos. Install it with '
            '`pip install "latent-dirac[viz]"`.'
        ) from exc
    return Image, ImageDraw, ImageFont


def _font(size: int, bold: bool = False):
    _, _, image_font = _load_pillow()
    candidates = (
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/Library/Fonts/Arial.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    )
    for path in candidates:
        if ("Bold" in path) != bold and "DejaVuSans-Bold" not in path:
            continue
        try:
            return image_font.truetype(path, size=size)
        except OSError:
            continue
    return image_font.load_default()


def _make_base_frame(title: str, subtitle: str, particle_color: tuple[int, int, int]):
    image_module, image_draw, _ = _load_pillow()
    image = image_module.new("RGB", CANVAS_SIZE, BACKGROUND)
    draw = image_draw.Draw(image)

    draw.rounded_rectangle((26, 24, 894, 474), radius=18, fill=(255, 255, 255), outline=(221, 228, 237), width=2)
    draw.text((52, 42), title, fill=INK, font=_font(24, bold=True))
    draw.text((52, 76), subtitle, fill=MUTED, font=_font(14))

    x0, y0, width, height = PLOT_BOX
    draw.rounded_rectangle((x0, y0, x0 + width, y0 + height), radius=10, fill=(248, 250, 253), outline=(220, 227, 236))
    for i in range(1, 5):
        x = x0 + width * i / 5
        draw.line((x, y0 + 12, x, y0 + height - 12), fill=GRID, width=1)
    for i in range(1, 4):
        y = y0 + height * i / 4
        draw.line((x0 + 12, y, x0 + width - 12, y), fill=GRID, width=1)

    draw.rectangle((x0 + 22, y0 + 30, x0 + width - 22, y0 + height - 30), outline=(232, 238, 245), width=1)
    draw.text((x0 + 18, y0 + height + 12), "beamline z [m]", fill=MUTED, font=_font(12))
    draw.text((x0 + 12, y0 - 22), "transverse x [m]", fill=MUTED, font=_font(12))

    legend_x = 770
    draw.rounded_rectangle((744, 128, 872, 250), radius=10, fill=(248, 250, 253), outline=(222, 229, 238))
    draw.ellipse((legend_x, 154, legend_x + 11, 165), fill=particle_color)
    draw.text((legend_x + 18, 151), "in transport", fill=INK, font=_font(12))
    draw.ellipse((legend_x, 184, legend_x + 11, 195), fill=ACCEPTED)
    draw.text((legend_x + 18, 181), "accepted", fill=INK, font=_font(12))
    draw.ellipse((legend_x, 214, legend_x + 11, 225), fill=LOST)
    draw.text((legend_x + 18, 211), "outside cut", fill=INK, font=_font(12))
    return image, draw


def _pipeline_snapshots(cloud: ParticleCloud, field, dt_s: float, frame_count: int) -> list[ParticleCloud]:
    solver = RelativisticBorisSolver(dt_s=dt_s, steps=1)
    snapshots = []
    current = cloud
    for _ in range(frame_count):
        snapshots.append(current)
        current = solver.propagate(current, field)
    return snapshots


def _sampled_transport_snapshots(
    cloud: ParticleCloud,
    field,
    dt_s: float,
    steps: int,
    snapshot_count: int,
) -> list[ParticleCloud]:
    if snapshot_count < 2:
        raise ValueError("snapshot_count must be at least 2")
    solver = RelativisticBorisSolver(dt_s=dt_s, steps=max(1, steps // (snapshot_count - 1)))
    snapshots = [cloud]
    current = cloud
    for _ in range(snapshot_count - 1):
        current = solver.propagate(current, field)
        snapshots.append(current)
    return snapshots


def _map_positions(
    positions: np.ndarray,
    z_limit: float,
    transverse_limit: float,
) -> np.ndarray:
    x0, y0, width, height = PLOT_BOX
    z_values = np.clip(positions[:, 2], 0.0, z_limit)
    x_values = np.clip(positions[:, 0], -transverse_limit, transverse_limit)
    px = x0 + 22 + (width - 44) * (z_values / z_limit)
    py = y0 + height / 2 - (height * 0.42) * (x_values / transverse_limit)
    return np.column_stack([px, py])


def _draw_stage_bar(draw, frame_index: int, frame_count: int, stages: tuple[str, ...]):
    left, top, width = 52, 438, 816
    segment = width / len(stages)
    progress = frame_index / max(frame_count - 1, 1)
    active_index = min(int(progress * len(stages)), len(stages) - 1)

    for index, name in enumerate(stages):
        x0 = left + segment * index
        x1 = left + segment * (index + 1) - 8
        fill = (228, 239, 252) if index <= active_index else (241, 244, 248)
        outline = (128, 176, 228) if index == active_index else (222, 229, 238)
        draw.rounded_rectangle((x0, top, x1, top + 26), radius=8, fill=fill, outline=outline)
        draw.text((x0 + 10, top + 7), name, fill=INK if index <= active_index else MUTED, font=_font(11))


def _draw_particles(
    draw,
    snapshots: list[ParticleCloud],
    frame_index: int,
    accepted_mask: np.ndarray,
    particle_color: tuple[int, int, int],
    z_limit: float,
    transverse_limit: float,
):
    start = max(0, frame_index - 8)
    trail_positions = [_map_positions(snapshot.position_m, z_limit, transverse_limit) for snapshot in snapshots[start : frame_index + 1]]
    current_positions = trail_positions[-1]

    for particle_index in range(current_positions.shape[0]):
        trail = [tuple(frame_positions[particle_index]) for frame_positions in trail_positions]
        if len(trail) > 1:
            draw.line(trail, fill=(198, 207, 218), width=1)

    for particle_index, (x, y) in enumerate(current_positions):
        color = particle_color
        if frame_index > len(snapshots) * 0.68:
            color = ACCEPTED if accepted_mask[particle_index] else LOST
        radius = 3 if accepted_mask[particle_index] else 2
        draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=color)


def _draw_acceptance_guides(draw, aperture_radius: float, momentum_label: str, z_limit: float, transverse_limit: float):
    x0, y0, width, height = PLOT_BOX
    aperture_y = y0 + height / 2 - (height * 0.42) * (aperture_radius / transverse_limit)
    aperture_y2 = y0 + height / 2 + (height * 0.42) * (aperture_radius / transverse_limit)
    draw.line((x0 + 24, aperture_y, x0 + width - 24, aperture_y), fill=(129, 190, 151), width=2)
    draw.line((x0 + 24, aperture_y2, x0 + width - 24, aperture_y2), fill=(129, 190, 151), width=2)
    draw.text((744, 366), f"aperture: +/- {aperture_radius:.3f} m", fill=MUTED, font=_font(12))
    draw.text((744, 388), momentum_label, fill=MUTED, font=_font(12))
    draw.text((x0 + width - 56, y0 + height - 24), f"z max {z_limit:.2f} m", fill=MUTED, font=_font(11))


def _draw_transverse_aperture(draw, aperture_radius: float, z_limit: float, transverse_limit: float):
    x0, y0, width, height = PLOT_BOX
    aperture_y = y0 + height / 2 - (height * 0.42) * (aperture_radius / transverse_limit)
    aperture_y2 = y0 + height / 2 + (height * 0.42) * (aperture_radius / transverse_limit)
    draw.line((x0 + 24, aperture_y, x0 + width - 24, aperture_y), fill=(129, 190, 151), width=2)
    draw.line((x0 + 24, aperture_y2, x0 + width - 24, aperture_y2), fill=(129, 190, 151), width=2)
    draw.text((x0 + width - 56, y0 + height - 24), f"z max {z_limit:.2f} m", fill=MUTED, font=_font(11))


def _draw_field_status(draw, lines: tuple[str, ...]):
    left, top, right, bottom = 744, 266, 888, 350
    draw.rounded_rectangle((left, top, right, bottom), radius=10, fill=FIELD, outline=(199, 218, 240))
    draw.text((left + 10, top + 9), "Magnetic field status", fill=INK, font=_font(11, bold=True))
    for index, line in enumerate(lines):
        draw.text((left + 10, top + 31 + index * 16), line, fill=MUTED, font=_font(10))


def _draw_splitter_legend(draw):
    legend_x = 770
    draw.rounded_rectangle((744, 128, 872, 250), radius=10, fill=(248, 250, 253), outline=(222, 229, 238))
    draw.ellipse((legend_x, 154, legend_x + 11, 165), fill=POSITRON)
    draw.text((legend_x + 18, 151), "positron e+", fill=INK, font=_font(12))
    draw.ellipse((legend_x, 184, legend_x + 11, 195), fill=ELECTRON)
    draw.text((legend_x + 18, 181), "electron e-", fill=INK, font=_font(12))
    draw.line((legend_x, 219, legend_x + 12, 219), fill=(87, 143, 204), width=3)
    draw.text((legend_x + 18, 211), "same By field", fill=INK, font=_font(12))


def _draw_cloud_tracks(
    draw,
    snapshots: list[ParticleCloud],
    frame_index: int,
    color: tuple[int, int, int],
    z_limit: float,
    transverse_limit: float,
):
    start = max(0, frame_index - 8)
    trail_positions = [_map_positions(snapshot.position_m, z_limit, transverse_limit) for snapshot in snapshots[start : frame_index + 1]]
    current_positions = trail_positions[-1]

    pale = tuple(int(0.55 * channel + 0.45 * 255) for channel in color)
    for particle_index in range(current_positions.shape[0]):
        trail = [tuple(frame_positions[particle_index]) for frame_positions in trail_positions]
        if len(trail) > 1:
            draw.line(trail, fill=pale, width=1)

    for x, y in current_positions:
        draw.ellipse((x - 3, y - 3, x + 3, y + 3), fill=color)


def _draw_sweep_cloud_tracks(
    draw,
    snapshots: list[ParticleCloud],
    accepted_mask: np.ndarray,
    color: tuple[int, int, int],
    z_limit: float,
    transverse_limit: float,
):
    mapped = [_map_positions(snapshot.position_m, z_limit, transverse_limit) for snapshot in snapshots]
    pale = tuple(int(0.62 * channel + 0.38 * 255) for channel in color)
    for particle_index in range(mapped[-1].shape[0]):
        trail = [tuple(frame_positions[particle_index]) for frame_positions in mapped]
        draw.line(trail, fill=pale, width=1)

    for particle_index, (x, y) in enumerate(mapped[-1]):
        fill = color if accepted_mask[particle_index] else LOST
        radius = 3 if accepted_mask[particle_index] else 2
        draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=fill)


def _make_charge_sign_splitter_frames(frame_count: int, particle_count: int):
    positron_cloud, electron_cloud = make_initial_pair(particle_count=particle_count, seed=2030)
    field = UniformField(B_vector_t=np.array([0.0, 0.45, 0.0]))
    positron_snapshots = _pipeline_snapshots(positron_cloud, field, dt_s=2.0e-12, frame_count=frame_count)
    electron_snapshots = _pipeline_snapshots(electron_cloud, field, dt_s=2.0e-12, frame_count=frame_count)

    frames = []
    for index in range(frame_count):
        image, draw = _make_base_frame(
            "Charge-sign splitter demo",
            "Matched e+ / e- clouds in the same transverse magnetic field",
            POSITRON,
        )
        _draw_splitter_legend(draw)
        x0, y0, width, height = PLOT_BOX
        draw.line((x0 + 24, y0 + height / 2, x0 + width - 24, y0 + height / 2), fill=(190, 199, 211), width=2)
        _draw_field_status(
            draw,
            (
                "model: transverse",
                "B vector [T]: [0, 0.45, 0]",
                "status: active",
            ),
        )
        draw.text((744, 382), "diagnostic: track separation", fill=INK, font=_font(13, bold=True))
        _draw_cloud_tracks(draw, positron_snapshots, index, POSITRON, 0.045, 0.040)
        _draw_cloud_tracks(draw, electron_snapshots, index, ELECTRON, 0.045, 0.040)
        _draw_stage_bar(draw, index, frame_count, ("matched source", "shared field", "opposite bend", "separation", "diagnostic"))
        frames.append(image)
    return frames


def _make_magnetic_control_sweep_frames(frame_count: int, particle_count: int):
    field_values = np.linspace(DEFAULT_FIELD_VALUES_T[0], DEFAULT_FIELD_VALUES_T[-1], frame_count)
    aperture_radius = 0.035
    dt_s = 2.0e-12
    steps = 80
    z_limit = 0.060
    transverse_limit = 0.045
    metrics = run_sweep(
        field_values_t=field_values,
        particle_count=particle_count,
        aperture_radius_m=aperture_radius,
        dt_s=dt_s,
        steps=steps,
        seed=2031,
    )

    frames = []
    for index, (by_tesla, row) in enumerate(zip(field_values, metrics, strict=True)):
        positron_cloud, electron_cloud = make_initial_pair(particle_count=particle_count, seed=2031)
        field = UniformField(B_vector_t=np.array([0.0, by_tesla, 0.0]))
        positron_snapshots = _sampled_transport_snapshots(positron_cloud, field, dt_s, steps, snapshot_count=18)
        electron_snapshots = _sampled_transport_snapshots(electron_cloud, field, dt_s, steps, snapshot_count=18)
        positron_accepted = np.abs(positron_snapshots[-1].position_m[:, 0]) <= aperture_radius
        electron_accepted = np.abs(electron_snapshots[-1].position_m[:, 0]) <= aperture_radius

        image, draw = _make_base_frame(
            "Magnetic control sweep demo",
            "Matched e+ / e- clouds under increasing transverse By",
            POSITRON,
        )
        _draw_splitter_legend(draw)
        _draw_transverse_aperture(draw, aperture_radius, z_limit, transverse_limit)
        _draw_field_status(
            draw,
            (
                "model: transverse sweep",
                f"B vector [T]: [0, {by_tesla:.2f}, 0]",
                "status: active",
            ),
        )
        draw.text((744, 362), f"separation: {row['mean_separation_m']:.4f} m", fill=INK, font=_font(12, bold=True))
        draw.text((744, 384), f"accepted: {row['accepted_fraction']:.3f}", fill=ACCEPTED, font=_font(12, bold=True))
        draw.text((744, 406), f"loss: {row['loss_fraction']:.3f}", fill=LOST, font=_font(12, bold=True))
        _draw_sweep_cloud_tracks(draw, positron_snapshots, positron_accepted, POSITRON, z_limit, transverse_limit)
        _draw_sweep_cloud_tracks(draw, electron_snapshots, electron_accepted, ELECTRON, z_limit, transverse_limit)
        _draw_stage_bar(draw, index, frame_count, ("matched source", "field sweep", "aperture", "losses", "report"))
        frames.append(image)
    return frames


def _make_positron_frames(frame_count: int, particle_count: int):
    source = PositronPairSource(
        primary_count=10_000,
        yield_eplus_per_primary=0.02,
        mean_energy_MeV=3.0,
        energy_spread_MeV=0.7,
        angular_rms_rad=0.08,
        source_sigma_m=0.0035,
        bunch_length_s=1.0e-12,
        macro_particles=particle_count,
    )
    cloud = source.sample(np.random.default_rng(2026))
    field = SolenoidField(b_tesla=0.8, radius_m=0.05, length_m=0.5)
    snapshots = _pipeline_snapshots(cloud, field, dt_s=3.0e-12, frame_count=frame_count)
    final_cloud = snapshots[-1]
    aperture_radius = 0.010
    accepted = MomentumWindow(momentum_gev_c_to_si(0.001), momentum_gev_c_to_si(0.016)).apply(
        Aperture(radius_m=aperture_radius, z_m=0.06).apply(final_cloud)
    ).alive

    frames = []
    for index in range(frame_count):
        image, draw = _make_base_frame(
            "Positron capture demo",
            "Parameterized e+ source -> solenoid transport -> acceptance cuts",
            POSITRON,
        )
        _draw_acceptance_guides(draw, aperture_radius, "p: 0.001-0.016 GeV/c", 0.045, 0.020)
        _draw_field_status(
            draw,
            (
                "model: solenoid",
                "B vector [T]: [0, 0, 0.8]",
                "status: in envelope",
            ),
        )
        _draw_particles(draw, snapshots, index, accepted, POSITRON, 0.045, 0.020)
        _draw_stage_bar(draw, index, frame_count, ("source", "transport", "aperture", "momentum", "yield"))
        accepted_count = int(np.count_nonzero(accepted))
        draw.text((744, 410), f"accepted: {accepted_count}/{particle_count}", fill=INK, font=_font(13, bold=True))
        frames.append(image)
    return frames


def _make_antiproton_frames(frame_count: int, particle_count: int):
    source = AntiprotonSurrogateSource(
        primary_proton_count=50_000,
        yield_pbar_per_primary_in_acceptance=2.0e-5,
        central_momentum_GeV_c=3.0,
        momentum_spread_fraction=0.08,
        angular_rms_rad=0.025,
        source_sigma_m=0.002,
        bunch_length_s=2.0e-12,
        macro_particles=particle_count,
    )
    cloud = source.sample(np.random.default_rng(2027))
    field = UniformField(B_vector_t=np.array([0.0, 0.0, 0.15]))
    snapshots = _pipeline_snapshots(cloud, field, dt_s=4.5e-11, frame_count=frame_count)
    final_cloud = snapshots[-1]
    aperture_radius = 0.025
    accepted = MomentumWindow(momentum_gev_c_to_si(2.75), momentum_gev_c_to_si(3.25)).apply(
        Aperture(radius_m=aperture_radius, z_m=0.5).apply(final_cloud)
    ).alive

    frames = []
    for index in range(frame_count):
        image, draw = _make_base_frame(
            "Antiproton transport demo",
            "Surrogate pbar source -> uniform B transport -> momentum window",
            ANTIPROTON,
        )
        _draw_acceptance_guides(draw, aperture_radius, "p: 2.75-3.25 GeV/c", 0.62, 0.045)
        _draw_field_status(
            draw,
            (
                "model: uniform",
                "B vector [T]: [0, 0, 0.15]",
                "status: active",
            ),
        )
        _draw_particles(draw, snapshots, index, accepted, ANTIPROTON, 0.62, 0.045)
        _draw_stage_bar(draw, index, frame_count, ("source", "transport", "momentum", "losses", "yield"))
        accepted_count = int(np.count_nonzero(accepted))
        draw.text((744, 410), f"accepted: {accepted_count}/{particle_count}", fill=INK, font=_font(13, bold=True))
        frames.append(image)
    return frames


def _save_webp(frames, path: Path, duration_ms: int):
    path.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(
        path,
        format="WEBP",
        save_all=True,
        append_images=frames[1:],
        duration=duration_ms,
        loop=0,
        quality=82,
        method=6,
    )


def generate_demo_webps(
    output_dir: str | Path = "assets/demos",
    frame_count: int = 44,
    particle_count: int = 90,
    duration_ms: int = 85,
) -> dict[str, Path]:
    """Generate README demo animations and return output paths by file name."""

    if frame_count < 2:
        raise ValueError("frame_count must be at least 2")
    if particle_count < 1:
        raise ValueError("particle_count must be positive")

    output_path = Path(output_dir)
    outputs = {
        DEMO_WEBP_FILES[0]: output_path / DEMO_WEBP_FILES[0],
        DEMO_WEBP_FILES[1]: output_path / DEMO_WEBP_FILES[1],
        DEMO_WEBP_FILES[2]: output_path / DEMO_WEBP_FILES[2],
        DEMO_WEBP_FILES[3]: output_path / DEMO_WEBP_FILES[3],
    }
    _save_webp(_make_charge_sign_splitter_frames(frame_count, particle_count), outputs[DEMO_WEBP_FILES[0]], duration_ms)
    _save_webp(_make_positron_frames(frame_count, particle_count), outputs[DEMO_WEBP_FILES[1]], duration_ms)
    _save_webp(_make_antiproton_frames(frame_count, particle_count), outputs[DEMO_WEBP_FILES[2]], duration_ms)
    _save_webp(_make_magnetic_control_sweep_frames(frame_count, particle_count), outputs[DEMO_WEBP_FILES[3]], duration_ms)
    return outputs


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="assets/demos")
    parser.add_argument("--frames", type=int, default=44)
    parser.add_argument("--particles", type=int, default=90)
    parser.add_argument("--duration-ms", type=int, default=85)
    args = parser.parse_args()

    generated = generate_demo_webps(args.output_dir, args.frames, args.particles, args.duration_ms)
    for path in generated.values():
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
