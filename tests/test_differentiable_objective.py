"""Tests for the differentiable capture objective (soft relaxation)."""

from pathlib import Path

import numpy as np
import pytest

jax = pytest.importorskip("jax")
jax.config.update("jax_enable_x64", True)

from latent_dirac.backends.differentiable import make_differentiable_objective  # noqa: E402
from latent_dirac.backends.jax_scene import run_scene_batched  # noqa: E402
from latent_dirac.scene.loader import load_scene  # noqa: E402

HELLO_SCENE = Path("examples/scenes/hello_beamline.yaml")

VARIABLES = ["hello-solenoid.b_tesla", "hello-aperture.radius_m"]
BASE_INPUTS = {"hello-solenoid.b_tesla": 0.8, "hello-aperture.radius_m": 0.008}


def test_gradient_matches_finite_differences():
    scene = load_scene(HELLO_SCENE)
    objective = make_differentiable_objective(scene, variables=VARIABLES, sharpness=50.0)

    value, grads = objective.value_and_grad(BASE_INPUTS)
    assert 0.0 < value < 1.0

    for name in VARIABLES:
        step = max(abs(BASE_INPUTS[name]), 1.0) * 1e-6
        plus = dict(BASE_INPUTS)
        minus = dict(BASE_INPUTS)
        plus[name] += step
        minus[name] -= step
        numeric = (objective.value(plus) - objective.value(minus)) / (2.0 * step)
        assert grads[name] == pytest.approx(numeric, rel=1e-4, abs=1e-8)


def test_soft_objective_converges_to_hard_accepted_fraction():
    scene = load_scene(HELLO_SCENE)
    hard = float(run_scene_batched(scene).accepted_fraction[0])

    errors = []
    for sharpness in (50.0, 500.0, 5000.0):
        objective = make_differentiable_objective(scene, variables=VARIABLES, sharpness=sharpness)
        errors.append(abs(objective.value(BASE_INPUTS) - hard))

    assert errors[-1] < 5e-3
    assert errors[-1] <= errors[0]


def test_gradient_sign_is_physical():
    scene = load_scene(HELLO_SCENE)
    objective = make_differentiable_objective(scene, variables=VARIABLES, sharpness=50.0)

    _, grads = objective.value_and_grad(BASE_INPUTS)

    # a wider aperture always accepts more of the beam
    assert grads["hello-aperture.radius_m"] > 0.0
    # the field gradient flows through the transport, not just the cuts
    assert abs(grads["hello-solenoid.b_tesla"]) > 0.0


def test_gradient_ascent_improves_soft_objective():
    scene = load_scene(HELLO_SCENE)
    objective = make_differentiable_objective(scene, variables=VARIABLES, sharpness=50.0)

    inputs = dict(BASE_INPUTS)
    start = objective.value(inputs)
    learning_rates = {"hello-solenoid.b_tesla": 0.05, "hello-aperture.radius_m": 1e-4}
    for _ in range(5):
        _, grads = objective.value_and_grad(inputs)
        for name in inputs:
            inputs[name] += learning_rates[name] * np.sign(grads[name])

    assert objective.value(inputs) > start


def make_window_scene():
    from latent_dirac.scene.loader import scene_from_mapping

    return scene_from_mapping(
        {
            "schema_version": 1,
            "name": "window-line",
            "seed": 2026,
            "source": {
                "type": "positron_pair",
                "label": "pair-source",
                "params": {
                    "primary_count": 10000,
                    "yield_eplus_per_primary": 0.02,
                    "mean_energy_MeV": 3.0,
                    "energy_spread_MeV": 0.5,
                    "angular_rms_rad": 0.05,
                    "source_sigma_m": 0.002,
                    "bunch_length_s": 1.0e-12,
                    "macro_particles": 48,
                },
            },
            "solver": {"type": "relativistic_boris", "dt_s": 3.0e-12, "steps": 40},
            "elements": [
                {
                    "type": "solenoid",
                    "label": "coil",
                    "b_tesla": 0.8,
                    "radius_m": 0.02,
                    "length_m": 0.1,
                    "center_z_m": 0.05,
                },
                {
                    "type": "momentum_window",
                    "label": "window",
                    "p_min_gev_c": 0.003,
                    "p_max_gev_c": 0.0045,
                },
            ],
        }
    )


def test_momentum_window_branch_gradient_matches_finite_differences():
    scene = make_window_scene()
    variables = ["window.p_min_gev_c", "window.p_max_gev_c"]
    objective = make_differentiable_objective(scene, variables=variables, sharpness=50.0)
    inputs = {"window.p_min_gev_c": 0.003, "window.p_max_gev_c": 0.0045}

    value, grads = objective.value_and_grad(inputs)
    assert 0.0 < value < 1.0
    # narrowing from below cuts yield
    assert grads["window.p_min_gev_c"] < 0.0

    for name in variables:
        step = abs(inputs[name]) * 1e-5
        plus, minus = dict(inputs), dict(inputs)
        plus[name] += step
        minus[name] -= step
        numeric = (objective.value(plus) - objective.value(minus)) / (2.0 * step)
        assert grads[name] == pytest.approx(numeric, rel=1e-4, abs=1e-6)

    hard = float(run_scene_batched(scene).accepted_fraction[0])
    sharp = make_differentiable_objective(scene, variables=variables, sharpness=5000.0)
    assert sharp.value(inputs) == pytest.approx(hard, abs=5e-3)


def test_inverted_regimes_collapse_toward_zero_survival():
    scene = load_scene(HELLO_SCENE)
    objective = make_differentiable_objective(scene, variables=VARIABLES, sharpness=50.0)

    negative_radius = dict(BASE_INPUTS)
    negative_radius["hello-aperture.radius_m"] = -0.004
    assert objective.value(negative_radius) < 1e-6

    window_scene = make_window_scene()
    window_objective = make_differentiable_objective(
        window_scene, variables=["window.p_min_gev_c", "window.p_max_gev_c"], sharpness=50.0
    )
    inverted = {"window.p_min_gev_c": 0.006, "window.p_max_gev_c": 0.002}
    assert window_objective.value(inverted) < 1e-3


def test_penning_trap_parameters_are_differentiable():
    from latent_dirac.scene.loader import scene_from_mapping

    scene = scene_from_mapping(
        {
            "schema_version": 1,
            "name": "trap-grad",
            "seed": 2026,
            "source": {
                "type": "positron_pair",
                "label": "pair-source",
                "params": {
                    "primary_count": 1000,
                    "yield_eplus_per_primary": 0.02,
                    "mean_energy_MeV": 0.001,
                    "energy_spread_MeV": 0.0002,
                    "angular_rms_rad": 0.3,
                    "source_sigma_m": 0.0005,
                    "bunch_length_s": 1.0e-12,
                    "macro_particles": 24,
                },
            },
            "solver": {"type": "relativistic_boris", "dt_s": 5.0e-12, "steps": 150},
            "elements": [
                {"type": "penning_trap", "label": "trap", "v0_volt": 10.0, "d_m": 0.005, "b_tesla": 1.0},
                {"type": "aperture", "label": "iris", "radius_m": 0.0006, "z_m": 0.0},
            ],
        }
    )
    objective = make_differentiable_objective(scene, variables=["trap.v0_volt"], sharpness=50.0)
    inputs = {"trap.v0_volt": 10.0}

    value, grads = objective.value_and_grad(inputs)
    step = 1e-4
    numeric = (
        objective.value({"trap.v0_volt": 10.0 + step}) - objective.value({"trap.v0_volt": 10.0 - step})
    ) / (2.0 * step)

    assert 0.0 < value < 1.0
    assert grads["trap.v0_volt"] == pytest.approx(numeric, rel=1e-3, abs=1e-8)


def test_variable_validation_is_shared_with_evaluator():
    scene = load_scene(HELLO_SCENE)
    with pytest.raises(ValueError, match="warp_factor"):
        make_differentiable_objective(scene, variables=["hello-solenoid.warp_factor"])
    with pytest.raises(ValueError, match="missing"):
        make_differentiable_objective(scene, variables=VARIABLES).value({"hello-solenoid.b_tesla": 0.8})


def _capture_then_store_scene(mean_lifetime_s, hold_time_s):
    from latent_dirac.scene.loader import scene_from_mapping

    return scene_from_mapping(
        {
            "schema_version": 1,
            "name": "capture-then-store",
            "seed": 4,
            "solver": {"dt_s": 4.0e-12, "steps": 40},
            "source": {
                "type": "positron_pair",
                "label": "pairs",
                "params": {
                    "primary_count": 10000,
                    "yield_eplus_per_primary": 0.02,
                    "mean_energy_MeV": 2.0,
                    "energy_spread_MeV": 0.3,
                    "angular_rms_rad": 0.15,
                    "source_sigma_m": 0.001,
                    "bunch_length_s": 1.0e-10,
                    "macro_particles": 200,
                },
            },
            "elements": [
                {
                    "type": "solenoid",
                    "label": "capture",
                    "b_tesla": 0.8,
                    "radius_m": 0.03,
                    "length_m": 0.2,
                    "center_z_m": 0.1,
                },
                {"type": "aperture", "label": "collimator", "radius_m": 0.008, "z_m": 0.1},
                {
                    "type": "residual_gas_loss",
                    "label": "storage",
                    "mean_lifetime_s": mean_lifetime_s,
                    "hold_time_s": hold_time_s,
                },
            ],
        }
    )


def test_storage_applies_expected_survival_factor():
    tau, hold = 100.0, 50.0
    variables = ["capture.b_tesla", "collimator.radius_m"]
    inputs = {"capture.b_tesla": 0.8, "collimator.radius_m": 0.008}

    with_storage = make_differentiable_objective(
        _capture_then_store_scene(tau, hold), variables=variables, sharpness=50.0
    )
    without_storage = make_differentiable_objective(
        _capture_then_store_scene(tau, 0.0), variables=variables, sharpness=50.0
    )

    ratio = with_storage.value(inputs) / without_storage.value(inputs)
    assert ratio == pytest.approx(np.exp(-hold / tau), rel=1e-6)


def test_storage_survival_gradient_is_analytic():
    tau, hold = 80.0, 40.0
    variables = ["storage.hold_time_s"]
    scene = _capture_then_store_scene(tau, hold)
    objective = make_differentiable_objective(scene, variables=variables, sharpness=50.0)

    value, grads = objective.value_and_grad({"storage.hold_time_s": hold})
    # d/dhold [F * exp(-hold/tau)] = -(1/tau) * value  ->  longer hold, worse
    assert grads["storage.hold_time_s"] == pytest.approx(-value / tau, rel=1e-6)
    assert grads["storage.hold_time_s"] < 0.0


def test_storage_leaves_field_gradient_flowing():
    # storage is a uniform factor, so it must not zero the capture-field gradient
    scene = _capture_then_store_scene(100.0, 50.0)
    variables = ["capture.b_tesla", "collimator.radius_m"]
    objective = make_differentiable_objective(scene, variables=variables, sharpness=50.0)

    _, grads = objective.value_and_grad({"capture.b_tesla": 0.8, "collimator.radius_m": 0.008})
    assert grads["collimator.radius_m"] > 0.0
    assert abs(grads["capture.b_tesla"]) > 0.0


def test_storage_lifetime_variable_at_zero_gives_no_nan():
    # if mean_lifetime_s is an optimization variable, the optimizer can
    # transit through 0; the floored tau must keep value and grad finite
    import math

    scene = _capture_then_store_scene(80.0, 40.0)
    objective = make_differentiable_objective(
        scene, variables=["storage.mean_lifetime_s"], sharpness=50.0
    )
    value, grads = objective.value_and_grad({"storage.mean_lifetime_s": 0.0})
    assert math.isfinite(value)
    assert math.isfinite(grads["storage.mean_lifetime_s"])
    assert value == pytest.approx(0.0, abs=1e-9)  # tau -> 0 kills everything
