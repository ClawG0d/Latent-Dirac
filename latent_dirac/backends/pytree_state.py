"""Opt-in JAX pytree registration for `ParticleState`.

`ParticleState.tree_flatten/tree_unflatten` follow the JAX conventions
but cannot be registered as-is (the class docstring records why):
`tree_unflatten` re-runs `__post_init__`, whose dtype coercion and
shape validation are illegal on tracer leaves, and the mutable
`metadata` dict is not hashable for jit cache keys.

This module registers dedicated flatten/unflatten functions instead:

- unflatten builds the instance via ``object.__new__`` and sets fields
  directly — tracer and placeholder leaves pass through untouched;
- **metadata does not cross the boundary**: the static aux is the
  species alone (a frozen, hashable pydantic model), and unflattened
  states carry ``metadata={}``. Metadata is provenance and diagnostics,
  not simulation state — the engineering rules already keep it off hot
  paths; callers who need it re-attach it outside the jit boundary.

Registration is opt-in (call :func:`register_particle_state_pytree`)
and idempotent; importing this module has no side effects.

Design record:
docs/superpowers/specs/2026-07-06-gpu-float32-validation-design.md.
"""

from __future__ import annotations

from latent_dirac.state.particle_state import _ARRAY_FIELDS, ParticleState

_registered = False


def _flatten(state: ParticleState):
    leaves = tuple(getattr(state, name) for name in _ARRAY_FIELDS)
    return leaves, state.species


def _unflatten(species, leaves) -> ParticleState:
    state = object.__new__(ParticleState)  # bypass __init__/__post_init__
    state.species = species
    for name, leaf in zip(_ARRAY_FIELDS, leaves, strict=True):
        object.__setattr__(state, name, leaf)
    state.metadata = {}
    return state


def register_particle_state_pytree() -> None:
    """Register `ParticleState` as a JAX pytree node (idempotent)."""

    global _registered
    if _registered:
        return
    from jax.tree_util import register_pytree_node

    register_pytree_node(ParticleState, _flatten, _unflatten)
    _registered = True
