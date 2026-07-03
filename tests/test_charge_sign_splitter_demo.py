from pathlib import Path

import numpy as np


def test_charge_sign_splitter_separates_opposite_charge_tracks():
    from examples.charge_sign_splitter_demo import run_demo

    result = run_demo(particle_count=48)

    assert result["field_b_tesla"] > 0.0
    assert result["positron_mean_x_m"] < 0.0
    assert result["electron_mean_x_m"] > 0.0
    assert result["mean_separation_m"] > 0.0
    assert np.sign(result["positron_mean_x_m"]) == -np.sign(result["electron_mean_x_m"])


def test_charge_sign_splitter_report_names_scope_without_annihilation():
    from examples.charge_sign_splitter_demo import run_report

    report = run_report(particle_count=24)

    assert "Charge-sign splitter demo" in report
    assert "Lorentz-force" in report
    assert "annihilation" not in report.lower()
    assert "energy release" not in report.lower()


def test_charge_sign_splitter_example_source_stays_in_scope():
    source_text = Path("examples/charge_sign_splitter_demo.py").read_text(encoding="utf-8")

    assert "annihilation" not in source_text.lower()
    assert "energy release" not in source_text.lower()
