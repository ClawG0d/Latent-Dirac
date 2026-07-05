import numpy as np
import pytest

from latent_dirac.core.species import positron
from latent_dirac.state.particle_state import ParticleState


def make_state(count: int = 4) -> ParticleState:
    return ParticleState(
        species=positron,
        position_m=np.zeros((count, 3)),
        momentum_kg_m_s=np.full((count, 3), 1.0e-22),
        time_s=np.zeros(count),
        weight=np.full(count, 2.0),
        alive=np.ones(count, dtype=bool),
        particle_id=np.arange(count),
        parent_id=np.full(count, -1),
    )


def test_construction_coerces_dtypes_and_defaults_ledger_to_minus_one():
    state = make_state()

    assert state.position_m.dtype == np.float64
    assert state.alive.dtype == np.bool_
    assert state.lost_at_element.dtype == np.int32
    np.testing.assert_array_equal(state.lost_at_element, [-1, -1, -1, -1])


def test_shape_validation_is_fail_fast():
    with pytest.raises(ValueError):
        ParticleState(
            species=positron,
            position_m=np.zeros((4, 2)),
            momentum_kg_m_s=np.zeros((4, 3)),
            time_s=np.zeros(4),
            weight=np.ones(4),
            alive=np.ones(4, dtype=bool),
            particle_id=np.arange(4),
            parent_id=np.full(4, -1),
        )
    with pytest.raises(ValueError):
        ParticleState(
            species=positron,
            position_m=np.zeros((4, 3)),
            momentum_kg_m_s=np.zeros((4, 3)),
            time_s=np.zeros(3),
            weight=np.ones(4),
            alive=np.ones(4, dtype=bool),
            particle_id=np.arange(4),
            parent_id=np.full(4, -1),
        )


def test_negative_weights_are_rejected():
    with pytest.raises(ValueError):
        ParticleState(
            species=positron,
            position_m=np.zeros((2, 3)),
            momentum_kg_m_s=np.zeros((2, 3)),
            time_s=np.zeros(2),
            weight=np.array([1.0, -0.5]),
            alive=np.ones(2, dtype=bool),
            particle_id=np.arange(2),
            parent_id=np.full(2, -1),
        )


def test_weighted_count_counts_only_alive():
    state = make_state()
    state.apply_alive_mask(np.array([True, False, True, True]))

    assert state.weighted_count() == pytest.approx(6.0)


def test_apply_alive_mask_is_an_and_operation():
    state = make_state()
    state.apply_alive_mask(np.array([True, False, True, True]))
    state.apply_alive_mask(np.array([True, True, False, True]))

    np.testing.assert_array_equal(state.alive, [True, False, False, True])


def test_copy_is_deeply_independent():
    state = make_state()
    clone = state.copy()
    clone.position_m[0, 0] = 9.9
    clone.apply_alive_mask(np.array([False, True, True, True]))
    clone.lost_at_element[0] = 3
    clone.metadata["note"] = "copied"

    assert state.position_m[0, 0] == 0.0
    assert bool(state.alive[0]) is True
    assert int(state.lost_at_element[0]) == -1
    assert "note" not in state.metadata


def test_tree_flatten_unflatten_roundtrip():
    state = make_state()
    state.metadata["origin"] = "test"
    leaves, aux = state.tree_flatten()

    assert all(isinstance(leaf, np.ndarray) for leaf in leaves)

    rebuilt = ParticleState.tree_unflatten(aux, leaves)
    np.testing.assert_array_equal(rebuilt.position_m, state.position_m)
    np.testing.assert_array_equal(rebuilt.lost_at_element, state.lost_at_element)
    assert rebuilt.species == state.species
    assert rebuilt.metadata == state.metadata


def test_kinematic_helpers_match_units_module():
    from latent_dirac.core.units import gamma_from_momentum

    state = make_state()
    np.testing.assert_allclose(state.gamma(), gamma_from_momentum(state.momentum_kg_m_s, positron.mass_kg))
    assert state.mean_kinetic_energy_joule() > 0.0
