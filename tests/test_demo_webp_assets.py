from pathlib import Path

import pytest


def test_demo_webp_generator_creates_animated_webp_files(tmp_path):
    image_module = pytest.importorskip("PIL.Image")
    from tools.generate_demo_webp import DEMO_WEBP_FILES, generate_demo_webps

    generated = generate_demo_webps(tmp_path, frame_count=4, particle_count=16)

    assert set(generated) == set(DEMO_WEBP_FILES)
    for name, path in generated.items():
        assert path == tmp_path / name
        assert path.suffix == ".webp"
        assert path.stat().st_size > 0
        with image_module.open(path) as image:
            assert image.format == "WEBP"
            assert getattr(image, "is_animated", False)
            assert image.n_frames == 4


def test_readme_references_demo_webp_assets():
    readme = Path("README.md").read_text(encoding="utf-8")

    assert "assets/demos/charge_sign_splitter.webp" in readme
    assert "assets/demos/positron_capture.webp" in readme
    assert "assets/demos/antiproton_transport.webp" in readme
