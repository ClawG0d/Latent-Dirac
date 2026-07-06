from pathlib import Path

import pytest


def test_scene_demo_generator_creates_animated_webp_files(tmp_path, monkeypatch):
    image_module = pytest.importorskip("PIL.Image")
    pytest.importorskip("matplotlib")
    pytest.importorskip("jax")  # the batched sweep demo runs on the JAX backend
    from tools.generate_scene_demo_webps import (
        DEMO_WEBP_FILES,
        SCENE_DEMOS,
        generate_scene_demo_webps,
    )

    assert "antiproton_ledger_3d.webp" in DEMO_WEBP_FILES
    assert "magnetic_mirror_3d.webp" in DEMO_WEBP_FILES

    # without the engine env var, engine-gated demos are skipped (their
    # committed assets stay in place) and everything else still renders;
    # xsuite-gated demos render exactly when xtrack is importable
    monkeypatch.delenv("LATENT_DIRAC_G4_TRANSFORMER", raising=False)
    import importlib.util

    xsuite_available = importlib.util.find_spec("xtrack") is not None
    generated = generate_scene_demo_webps(tmp_path, frame_count=2, write_html=False)

    for name in DEMO_WEBP_FILES:
        config = SCENE_DEMOS.get(name, {})
        if config.get("requires_engine"):
            assert name not in generated
            continue
        if config.get("requires_xsuite") and not xsuite_available:
            assert name not in generated
            continue
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
