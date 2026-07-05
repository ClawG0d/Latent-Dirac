"""Tests for the Xopt-compatible scene evaluator and program reuse."""

from pathlib import Path

import numpy as np
import pytest

jax = pytest.importorskip("jax")
jax.config.update("jax_enable_x64", True)

from latent_dirac.backends.evaluator import make_scene_evaluator  # noqa: E402
from latent_dirac.backends.jax_scene import (  # noqa: E402
    BatchedSceneProgram,
    run_scene_batched,
)
from latent_dirac.scene.build import run_scene  # noqa: E402
from latent_dirac.scene.loader import load_scene, scene_from_mapping  # noqa: E402

SCENES_DIR = Path("examples/scenes")

SOURCE_PARAMS = {
    "primary_count": 10000,
    "yield_eplus_per_primary": 0.02,
    "mean_energy_MeV": 3.0,
    "energy_spread_MeV": 0.4,
    "angular_rms_rad": 0.03,
    "source_sigma_m": 0.001,
    "bunch_length_s": 1.0e-12,
    "macro_particles": 48,
}


def sweep_scene_mapping() -> dict:
    return {
        "schema_version": 1,
        "name": "eval-line",
        "seed": 2031,
        "source": {"type": "positron_pair", "label": "pair-source", "params": dict(SOURCE_PARAMS)},
        "solver": {"type": "relativistic_boris", "dt_s": 2.0e-12, "steps": 60},
        "elements": [
            {
                "type": "uniform_field",
                "label": "sweep-field",
                "B_vector_t": [0.0, 0.3, 0.0],
                "steps": 60,
            },
            {"type": "aperture", "label": "window", "radius_m": 0.02, "z_m": 0.03},
            {"type": "momentum_window", "label": "cut", "p_min_gev_c": 0.003, "p_max_gev_c": 0.0045},
        ],
    }


def test_program_reuse_matches_fresh_runs():
    scene = scene_from_mapping(sweep_scene_mapping())
    program = BatchedSceneProgram(scene, override_keys=("sweep-field.B_vector_t",))

    first_values = np.array([[0.0, 0.1, 0.0], [0.0, 0.5, 0.0]])
    second_values = np.array([[0.0, 0.2, 0.0], [0.0, 0.6, 0.0]])

    for values in (first_values, second_values):
        reused = program.run({"sweep-field.B_vector_t": values})
        fresh = run_scene_batched(scene, overrides={"sweep-field.B_vector_t": values})
        np.testing.assert_array_equal(reused.alive, fresh.alive)
        np.testing.assert_allclose(reused.position_m, fresh.position_m, rtol=1e-12)


def test_evaluator_matches_numpy_pipeline():
    scene = load_scene(SCENES_DIR / "positron_capture.yaml")
    evaluate = make_scene_evaluator(scene, variables=["capture-solenoid.b_tesla"])

    outputs = evaluate({"capture-solenoid.b_tesla": 0.8})

    reference = run_scene(scene).pipeline_result.final_cloud
    expected_fraction = reference.weighted_count() / float(np.sum(reference.weight))
    assert outputs["accepted_fraction"] == pytest.approx(expected_fraction, rel=1e-12)
    assert outputs["accepted_weighted"] == pytest.approx(reference.weighted_count(), rel=1e-12)


def test_vector_component_variable_reconstructs_full_override():
    scene = scene_from_mapping(sweep_scene_mapping())
    evaluate = make_scene_evaluator(scene, variables=["sweep-field.B_vector_t[1]"])

    outputs = evaluate({"sweep-field.B_vector_t[1]": 0.45})

    reference = run_scene_batched(scene, overrides={"sweep-field.B_vector_t": np.array([[0.0, 0.45, 0.0]])})
    assert outputs["accepted_fraction"] == pytest.approx(float(reference.accepted_fraction[0]), rel=1e-12)


def test_two_components_of_same_vector_compose_into_one_override():
    scene = scene_from_mapping(sweep_scene_mapping())
    evaluate = make_scene_evaluator(
        scene, variables=["sweep-field.B_vector_t[0]", "sweep-field.B_vector_t[2]"]
    )

    outputs = evaluate({"sweep-field.B_vector_t[0]": 0.1, "sweep-field.B_vector_t[2]": 0.05})

    # untouched component [1] keeps its base value 0.3
    reference = run_scene_batched(scene, overrides={"sweep-field.B_vector_t": np.array([[0.1, 0.3, 0.05]])})
    assert outputs["accepted_fraction"] == pytest.approx(float(reference.accepted_fraction[0]), rel=1e-12)

    batched = evaluate.batch(
        {
            "sweep-field.B_vector_t[0]": np.array([0.1, 0.2]),
            "sweep-field.B_vector_t[2]": np.array([0.05, 0.0]),
        }
    )
    reference_batch = run_scene_batched(
        scene,
        overrides={"sweep-field.B_vector_t": np.array([[0.1, 0.3, 0.05], [0.2, 0.3, 0.0]])},
    )
    np.testing.assert_allclose(batched["accepted_fraction"], reference_batch.accepted_fraction, rtol=1e-12)


def test_batch_evaluation_matches_per_point_calls():
    scene = scene_from_mapping(sweep_scene_mapping())
    evaluate = make_scene_evaluator(scene, variables=["sweep-field.B_vector_t[1]"])
    values = np.array([0.0, 0.2, 0.4, 0.6])

    batched = evaluate.batch({"sweep-field.B_vector_t[1]": values})

    assert batched["accepted_fraction"].shape == (4,)
    for index, value in enumerate(values):
        single = evaluate({"sweep-field.B_vector_t[1]": float(value)})
        assert batched["accepted_fraction"][index] == pytest.approx(single["accepted_fraction"], rel=1e-12)


def test_unknown_variable_raises_at_construction():
    scene = scene_from_mapping(sweep_scene_mapping())
    with pytest.raises(ValueError, match="warp_factor"):
        make_scene_evaluator(scene, variables=["sweep-field.warp_factor"])
    with pytest.raises(ValueError, match="index"):
        make_scene_evaluator(scene, variables=["sweep-field.B_vector_t[7]"])
    with pytest.raises(ValueError, match="scalar"):
        make_scene_evaluator(scene, variables=["sweep-field.B_vector_t"])


def test_wrong_input_keys_raise_at_call():
    scene = scene_from_mapping(sweep_scene_mapping())
    evaluate = make_scene_evaluator(scene, variables=["sweep-field.B_vector_t[1]"])

    with pytest.raises(ValueError, match="missing"):
        evaluate({})
    with pytest.raises(ValueError, match="unexpected"):
        evaluate({"sweep-field.B_vector_t[1]": 0.1, "extra": 1.0})


def test_end_to_end_with_xopt_if_installed():
    xopt = pytest.importorskip("xopt")

    scene = scene_from_mapping(sweep_scene_mapping())
    evaluate = make_scene_evaluator(scene, variables=["sweep-field.B_vector_t[1]"])

    problem = xopt.Xopt(
        vocs=xopt.VOCS(
            variables={"sweep-field.B_vector_t[1]": [0.0, 0.6]},
            objectives={"accepted_fraction": "MAXIMIZE"},
        ),
        evaluator=xopt.Evaluator(function=evaluate),
        generator=xopt.generators.random.RandomGenerator(
            vocs=xopt.VOCS(
                variables={"sweep-field.B_vector_t[1]": [0.0, 0.6]},
                objectives={"accepted_fraction": "MAXIMIZE"},
            )
        ),
    )
    problem.random_evaluate(3)
    assert len(problem.data) == 3
