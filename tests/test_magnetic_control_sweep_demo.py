import subprocess
import sys

import pytest


def test_magnetic_control_sweep_separation_and_losses_are_reported():
    from examples.magnetic_control_sweep_demo import run_report, run_sweep

    results = run_sweep(field_values_t=[0.0, 0.3, 0.6], particle_count=36, aperture_radius_m=0.05)

    assert len(results) == 3
    assert results[0]["field_by_tesla"] == 0.0
    assert results[-1]["field_by_tesla"] == 0.6
    assert results[-1]["mean_separation_m"] > results[0]["mean_separation_m"]
    for row in results:
        assert 0.0 <= row["accepted_fraction"] <= 1.0
        assert 0.0 <= row["loss_fraction"] <= 1.0
        assert row["accepted_fraction"] + row["loss_fraction"] == pytest.approx(1.0)

    report = run_report(field_values_t=[0.0, 0.3, 0.6], particle_count=36, aperture_radius_m=0.05)

    assert "Magnetic control sweep demo" in report
    assert "Magnetic field status:" in report
    assert "Aperture status:" in report
    assert "By [T]" in report
    assert "accepted fraction" in report
    assert "loss fraction" in report


def test_magnetic_control_sweep_default_aperture_shows_high_field_loss():
    from examples.magnetic_control_sweep_demo import run_sweep

    results = run_sweep()

    assert results[0]["loss_fraction"] == 0.0
    assert results[-1]["loss_fraction"] > 0.0


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"field_values_t": []}, "field_values_t must not be empty"),
        ({"particle_count": 0}, "particle_count must be positive"),
        ({"aperture_radius_m": 0.0}, "aperture_radius_m must be positive"),
        ({"dt_s": 0.0}, "dt_s must be positive"),
        ({"steps": 0}, "steps must be positive"),
    ],
)
def test_magnetic_control_sweep_validates_inputs(kwargs, message):
    from examples.magnetic_control_sweep_demo import run_sweep

    with pytest.raises(ValueError, match=message):
        run_sweep(**kwargs)


def test_magnetic_control_sweep_script_runs_directly():
    completed = subprocess.run(
        [sys.executable, "examples/magnetic_control_sweep_demo.py"],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "Magnetic control sweep demo" in completed.stdout
    assert "Sweep table:" in completed.stdout
