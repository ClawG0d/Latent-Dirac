"""Tests for trajectory recording inside the batched JAX program."""

import numpy as np
import pytest

jax = pytest.importorskip("jax")
jax.config.update("jax_enable_x64", True)

from latent_dirac.backends.jax_scene import BatchedSceneProgram, run_scene_batched  # noqa: E402
from latent_dirac.scene.build import run_scene  # noqa: E402
from latent_dirac.scene.loader import scene_from_mapping  # noqa: E402

SOURCE_PARAMS = {
    "primary_count": 10000,
    "yield_eplus_per_primary": 0.02,
    "mean_energy_MeV": 3.0,
    "energy_spread_MeV": 0.4,
    "angular_rms_rad": 0.03,
    "source_sigma_m": 0.001,
    "bunch_length_s": 1.0e-12,
    "macro_particles": 16,
}


def make_scene() -> dict:
    return {
        "schema_version": 1,
        "name": "trajectory-line",
        "seed": 2031,
        "source": {"type": "positron_pair", "label": "pair-source", "params": dict(SOURCE_PARAMS)},
        "solver": {"type": "relativistic_boris", "dt_s": 2.0e-12, "steps": 12},
        "elements": [
            {"type": "uniform_field", "label": "field-a", "B_vector_t": [0.0, 0.3, 0.0], "steps": 12},
            {"type": "aperture", "label": "cut", "radius_m": 0.02, "z_m": 0.01},
            {"type": "drift", "label": "gap", "steps": 6},
        ],
    }


def test_recorded_trajectories_match_numpy_pipeline_at_stride():
    scene = scene_from_mapping(make_scene())
    stride = 3

    batched = run_scene_batched(scene, record_stride=stride)
    reference = run_scene(scene, record_trajectories=True)

    # NumPy per-stage histories: field-a (13, N, 3), gap (7, N, 3); the
    # first row of a later stage repeats the previous stage's last row
    combined = np.concatenate([reference.trajectories["field-a"], reference.trajectories["gap"][1:]], axis=0)
    expected = combined[::stride]

    assert batched.trajectories is not None
    assert batched.trajectories.shape == (1, expected.shape[0], 16, 3)
    np.testing.assert_allclose(batched.trajectories[0], expected, rtol=1e-9, atol=1e-15)


def test_trajectories_are_none_when_recording_is_off():
    scene = scene_from_mapping(make_scene())
    assert run_scene_batched(scene).trajectories is None


def test_batched_recording_shapes_over_configurations():
    scene = scene_from_mapping(make_scene())
    program = BatchedSceneProgram(scene, override_keys=("field-a.B_vector_t",), record_stride=2)
    values = np.array([[0.0, 0.1, 0.0], [0.0, 0.3, 0.0], [0.0, 0.5, 0.0]])

    result = program.run({"field-a.B_vector_t": values})

    total_steps = 12 + 6
    expected_snapshots = total_steps // 2 + 1
    assert result.trajectories.shape == (3, expected_snapshots, 16, 3)
    # configuration 1 matches the unbatched scene run at the same stride
    single = run_scene_batched(scene, record_stride=2)
    np.testing.assert_allclose(result.trajectories[1], single.trajectories[0], rtol=1e-12)


def test_invalid_record_stride_is_rejected():
    scene = scene_from_mapping(make_scene())
    with pytest.raises(ValueError, match="record_stride"):
        BatchedSceneProgram(scene, record_stride=0)
