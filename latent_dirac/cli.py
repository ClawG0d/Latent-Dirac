"""The latent-dirac command line: run and render declarative scenes.

`latent-dirac run scene.yaml` prints the scene report; `latent-dirac
render scene.yaml -o out.html` writes the interactive 3D rendering
(requires the `viz` extra). Expected failures print one line to stderr
and exit non-zero.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from pydantic import ValidationError

from latent_dirac.diagnostics.scene_report import scene_report
from latent_dirac.scene.build import run_scene
from latent_dirac.scene.loader import load_scene


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="latent-dirac",
        description="Run and render declarative Latent Dirac scenes.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run", help="run a scene and print its report")
    run_parser.add_argument("scene", help="path to a scene .yaml/.yml/.json file")

    render_parser = subparsers.add_parser("render", help="run a scene and write an interactive 3D HTML")
    render_parser.add_argument("scene", help="path to a scene .yaml/.yml/.json file")
    render_parser.add_argument("-o", "--output", required=True, help="output HTML path for the 3D rendering")
    render_parser.add_argument(
        "--max-particles",
        type=int,
        default=64,
        help="maximum particle trajectories to draw (default: 64)",
    )
    render_parser.add_argument(
        "--animate",
        action="store_true",
        help="write a play/pause animation of the cloud traversing the scene",
    )
    return parser


def _load(scene_path: str):
    path = Path(scene_path)
    if not path.exists():
        raise FileNotFoundError(f"scene file {path} does not exist")
    return load_scene(path)


def _command_run(args: argparse.Namespace) -> int:
    scene = _load(args.scene)
    result = run_scene(scene)
    print(scene_report(scene, result, "transport and acceptance diagnostic only"))
    return 0


def _command_render(args: argparse.Namespace) -> int:
    scene = _load(args.scene)
    result = run_scene(scene, record_trajectories=True)
    try:
        from latent_dirac.viz.scene_3d import render_scene_3d, render_scene_animation

        if args.animate:
            figure = render_scene_animation(scene, result, max_particles=args.max_particles)
        else:
            figure = render_scene_3d(scene, result, max_particles=args.max_particles)
    except ImportError as exc:
        raise ImportError('rendering requires the viz extra: pip install "latent-dirac[viz]"') from exc
    output = Path(args.output)
    figure.write_html(output, include_plotlyjs="cdn")
    kind = "animation" if args.animate else "3D scene"
    print(f"interactive {kind} written to {output}")
    return 0


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        if args.command == "run":
            return _command_run(args)
        if args.command == "render":
            return _command_render(args)
    except (FileNotFoundError, ValueError, ValidationError, ImportError) as exc:
        print(f"latent-dirac: {exc}", file=sys.stderr)
        return 1
    return 2  # pragma: no cover - argparse enforces the command choices


if __name__ == "__main__":
    raise SystemExit(main())
