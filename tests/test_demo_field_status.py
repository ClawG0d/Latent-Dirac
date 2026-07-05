from pathlib import Path


def test_positron_capture_report_shows_magnetic_field_status():
    from examples.positron_capture_demo import run_report

    report = run_report()

    assert "Magnetic field status:" in report
    assert "- field model: idealized solenoid (hard-edge)" in report
    assert "- status: active inside radius 0.02 m and length 0.15 m" in report
    assert "Loss ledger (weighted, by killing element):" in report


def test_antiproton_ledger_report_shows_magnetic_field_status():
    from examples.antiproton_ledger_demo import run_report

    report = run_report()

    assert "Magnetic field status:" in report
    assert "- field model: uniform field" in report
    assert "- B vector [T]: [0, 0, 1.5]" in report
    assert "Loss ledger (weighted, by killing element):" in report


def test_wien_report_declares_crossed_fields():
    from examples.wien_filter_demo import run_report

    report = run_report()

    assert "Magnetic field status:" in report
    assert "- E vector [V/m]:" in report
    assert "matched velocity E/B" in report


def test_readme_and_generator_show_demo_field_status():
    readme = Path("README.md").read_text(encoding="utf-8")
    generator = Path("tools/generate_scene_demo_webps.py").read_text(encoding="utf-8")

    assert readme.count("Magnetic field status:") >= 3
    assert "diagnostic only" in generator
    assert "Boris solver" in generator
