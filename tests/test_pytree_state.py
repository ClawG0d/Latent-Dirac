"""ParticleState pytree registration: jit/vmap-safe, metadata-explicit.

Design record: docs/superpowers/specs/2026-07-06-gpu-float32-validation-design.md.
"""

from __future__ import annotations

import numpy as np
import pytest

jax = pytest.importorskip("jax")
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402

from latent_dirac.backends.pytree_state import register_particle_state_pytree  # noqa: E402
from latent_dirac.core.species import positron  # noqa: E402
from latent_dirac.state.particle_state import ParticleState  # noqa: E402

register_particle_state_pytree()


def make_state(count: int = 4) -> ParticleState:
    rng = np.random.default_rng(3)
    return ParticleState(
        species=positron,
        position_m=rng.normal(size=(count, 3)),
        momentum_kg_m_s=rng.normal(size=(count, 3)) * 1e-21,
        time_s=np.zeros(count),
        weight=np.ones(count),
        alive=np.ones(count, dtype=bool),
        particle_id=np.arange(count),
        parent_id=np.full(count, -1),
        metadata={"model_type": "placeholder"},
    )


def test_registration_is_idempotent():
    register_particle_state_pytree()
    register_particle_state_pytree()


def test_state_round_trips_through_jit():
    state = make_state()

    @jax.jit
    def shift(s: ParticleState) -> ParticleState:
        return jax.tree_util.tree_map(lambda leaf: leaf, s)

    out = shift(state)
    assert isinstance(out, ParticleState)
    np.testing.assert_allclose(np.asarray(out.position_m), state.position_m, rtol=1e-15)
    np.testing.assert_array_equal(np.asarray(out.alive), state.alive)
    assert out.species is positron


def test_tracers_do_not_hit_post_init_validation():
    state = make_state()

    @jax.jit
    def kick(s: ParticleState, delta):
        # tree_map builds new ParticleStates from tracer leaves: this would
        # explode inside __post_init__'s shape/dtype coercion if unflatten
        # ran it
        leaves, treedef = jax.tree_util.tree_flatten(s)
        moved = [leaves[0] + delta, *leaves[1:]]
        return jax.tree_util.tree_unflatten(treedef, moved)

    out = kick(state, 0.5)
    np.testing.assert_allclose(np.asarray(out.position_m), state.position_m + 0.5, rtol=1e-15)


def test_metadata_does_not_cross_the_jit_boundary():
    state = make_state()

    @jax.jit
    def identity(s: ParticleState) -> ParticleState:
        return jax.tree_util.tree_map(lambda leaf: leaf, s)

    out = identity(state)
    # metadata is provenance, not simulation state: it is deliberately
    # dropped at the boundary (documented), never smuggled into jit keys
    assert out.metadata == {}
    assert state.metadata == {"model_type": "placeholder"}  # input untouched


def test_jit_cache_is_stable_across_instances():
    calls = {"count": 0}

    def body(s: ParticleState) -> ParticleState:
        calls["count"] += 1
        return jax.tree_util.tree_map(lambda leaf: leaf * 2.0, s)

    jitted = jax.jit(body)
    jitted(make_state())
    jitted(make_state())  # fresh metadata dict must NOT force a retrace
    assert calls["count"] == 1


def test_vmap_over_a_batched_leaf():
    state = make_state()
    deltas = jnp.asarray([0.0, 1.0, 2.0])

    @jax.jit
    def shift_many(s: ParticleState, delta):
        def one(d):
            leaves, treedef = jax.tree_util.tree_flatten(s)
            return jax.tree_util.tree_unflatten(treedef, [leaves[0] + d, *leaves[1:]]).position_m

        return jax.vmap(one)(delta)

    out = shift_many(state, deltas)
    assert out.shape == (3, 4, 3)
    np.testing.assert_allclose(np.asarray(out[2]), state.position_m + 2.0, rtol=1e-15)
