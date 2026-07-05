"""Tests for the latent-dirac command-line interface."""

from pathlib import Path

import pytest

from latent_dirac.cli import main

HELLO_SCENE = Path("examples/scenes/hello_beamline.yaml")


def test_run_prints_scene_report(capsys):
    exit_code = main(["run", str(HELLO_SCENE)])

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "Latent Dirac scene report" in captured.out
    assert "Loss ledger (weighted, by killing element):" in captured.out
    assert "Magnetic field status:" in captured.out


def test_render_writes_interactive_html(tmp_path):
    pytest.importorskip("plotly")
    output = tmp_path / "hello.html"

    exit_code = main(["render", str(HELLO_SCENE), "-o", str(output)])

    assert exit_code == 0
    assert output.stat().st_size > 0
    assert "plotly" in output.read_text(encoding="utf-8")[:5000].lower()


def test_missing_scene_file_exits_nonzero(capsys):
    exit_code = main(["run", "does/not/exist.yaml"])

    captured = capsys.readouterr()
    assert exit_code != 0
    assert "exist.yaml" in captured.err
    assert "Traceback" not in captured.err


def test_invalid_scene_exits_nonzero(tmp_path, capsys):
    bad = tmp_path / "bad.yaml"
    bad.write_text("schema_version: 99\nname: broken\n", encoding="utf-8")

    exit_code = main(["run", str(bad)])

    captured = capsys.readouterr()
    assert exit_code != 0
    assert "Traceback" not in captured.err


def test_unsupported_suffix_exits_nonzero(tmp_path, capsys):
    bad = tmp_path / "scene.toml"
    bad.write_text("x = 1", encoding="utf-8")

    exit_code = main(["run", str(bad)])

    captured = capsys.readouterr()
    assert exit_code != 0
    assert "suffix" in captured.err


def test_console_entry_point_is_declared():
    import tomllib

    pyproject = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))
    assert pyproject["project"]["scripts"]["latent-dirac"] == "latent_dirac.cli:main"


def test_examples_import_report_from_package():
    from latent_dirac.diagnostics.scene_report import field_status_lines, scene_report

    assert callable(scene_report)
    assert callable(field_status_lines)
