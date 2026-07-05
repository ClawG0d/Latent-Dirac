from pathlib import Path

import pytest


def test_scene_demo_generator_creates_animated_webp_files(tmp_path):
    image_module = pytest.importorskip("PIL.Image")
    pytest.importorskip("matplotlib")
    pytest.importorskip("jax")  # the batched sweep demo runs on the JAX backend
    from tools.generate_scene_demo_webps import DEMO_WEBP_FILES, generate_scene_demo_webps

    assert "antiproton_ledger_3d.webp" in DEMO_WEBP_FILES
    assert "magnetic_mirror_3d.webp" in DEMO_WEBP_FILES

    generated = generate_scene_demo_webps(tmp_path, frame_count=2, write_html=False)

    for name in DEMO_WEBP_FILES:
        path = generated[name]
        assert path == tmp_path / name
        assert path.stat().st_size > 0
        with image_module.open(path) as image:
            assert image.format == "WEBP"
            assert getattr(image, "is_animated", False)
            assert image.n_frames == 2


def test_hero_3d_generator_creates_animated_webp_from_trajectory(tmp_path):
    image_module = pytest.importorskip("PIL.Image")
    pytest.importorskip("matplotlib")
    from tools.generate_hero_3d_webp import HERO_WEBP_FILE, generate_hero_3d_webp

    generated = generate_hero_3d_webp(tmp_path, frame_count=3, particle_count=6, write_html=False)

    path = generated[HERO_WEBP_FILE]
    assert path == tmp_path / HERO_WEBP_FILE
    assert path.stat().st_size > 0
    with image_module.open(path) as image:
        assert image.format == "WEBP"
        assert getattr(image, "is_animated", False)
        assert image.n_frames == 3


def test_readme_references_all_3d_demo_assets():
    readme = Path("README.md").read_text(encoding="utf-8")

    from tools.generate_hero_3d_webp import HERO_WEBP_FILE
    from tools.generate_scene_demo_webps import DEMO_WEBP_FILES

    for name in (HERO_WEBP_FILE, *DEMO_WEBP_FILES):
        assert f"assets/demos/{name}" in readme, f"README must reference {name}"


def test_retired_2d_assets_are_gone():
    demos = Path("assets/demos")
    for retired in (
        "charge_sign_splitter.webp",
        "positron_capture.webp",
        "antiproton_transport.webp",
        "magnetic_control_sweep.webp",
    ):
        assert not (demos / retired).exists(), f"{retired} should be retired"
    assert not Path("tools/generate_demo_webp.py").exists()
