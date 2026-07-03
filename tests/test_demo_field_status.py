from pathlib import Path


def test_positron_capture_report_shows_magnetic_field_status():
    from examples.positron_capture_demo import run_demo

    report = run_demo()

    assert "Magnetic field status:" in report
    assert "- field model: idealized solenoid" in report
    assert "- B vector [T]: [0, 0, 0.8] inside solenoid envelope" in report
    assert "- status: active inside radius 0.05 m and length 0.5 m" in report


def test_antiproton_transport_report_shows_magnetic_field_status():
    from examples.antiproton_transport_demo import run_demo

    report = run_demo()

    assert "Magnetic field status:" in report
    assert "- field model: uniform magnetic field" in report
    assert "- B vector [T]: [0, 0, 0.15]" in report
    assert "- status: active over all sampled positions" in report


def test_readme_and_webp_generator_show_demo_field_status():
    readme = Path("README.md").read_text(encoding="utf-8")
    generator = Path("tools/generate_demo_webp.py").read_text(encoding="utf-8")

    assert readme.count("Magnetic field status:") >= 3
    assert "Each animation includes a magnetic field status panel." in readme
    assert "Magnetic field status" in generator
    assert "B vector [T]" in generator
