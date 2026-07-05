"""Xsuite adapter: the Lattice component of the solver zoo.

Bidirectional `ParticleState` ↔ `xtrack.Particles` conversion plus a
stage-level tracking wrapper. Frame convention (see the 2026-07-05
Xsuite adapter spec): beam axis +z with `s ≡ z`, explicit reference
momentum `p0c_ev` (never silently inferred), `zeta = z − β₀ c t`.
Back-conversion assumes forward-going particles (`p_z > 0`), the
paraxial regime every current scene uses.

xtrack runs in-process on the NumPy pipeline only: gradients stop at
this boundary (the Lattice component is an engine boundary for autodiff
purposes), and tracked results carry `xtrack_version` provenance in the
state metadata. Requires the optional `[xsuite]` extra; imports of
xtrack stay inside functions so this module is importable without it.
"""

from __future__ import annotations

import numpy as np
from pydantic import BaseModel, Field

from latent_dirac.core.constants import c, e
from latent_dirac.pipeline.stage import Stage
from latent_dirac.state.particle_state import ParticleState


def _require_xtrack():
    try:
        import xtrack
    except ImportError as exc:  # pragma: no cover - exercised without extra
        raise ImportError(
            'the Xsuite adapter requires the optional [xsuite] extra: '
            'pip install "latent-dirac[xsuite]"'
        ) from exc
    return xtrack


class ReferenceFrame(BaseModel):
    """Explicit accelerator reference: momentum × c in eV, beam along +z."""

    p0c_ev: float = Field(gt=0)

    def p0_si(self) -> float:
        return self.p0c_ev * e / c

    def beta0(self, mass_kg: float) -> float:
        mass_ev = mass_kg * c**2 / e
        return self.p0c_ev / float(np.hypot(self.p0c_ev, mass_ev))


def reference_from_state(state: ParticleState) -> ReferenceFrame:
    """Weighted-mean |p| of the alive particles — a helper, never a hidden default."""
    mask = state.alive if np.any(state.alive) else np.ones_like(state.alive)
    p_norm = np.linalg.norm(state.momentum_kg_m_s[mask], axis=1)
    p0_si = float(np.average(p_norm, weights=state.weight[mask]))
    return ReferenceFrame(p0c_ev=p0_si * c / e)


def to_xtrack_particles(state: ParticleState, frame: ReferenceFrame):
    """Convert a forward-going ParticleState into xtrack.Particles."""
    xtrack = _require_xtrack()
    species = state.species
    p0_si = frame.p0_si()
    beta0 = frame.beta0(species.mass_kg)

    momentum = state.momentum_kg_m_s
    if np.any(state.alive & (momentum[:, 2] <= 0.0)):
        raise ValueError(
            "the Xsuite adapter requires forward-going particles "
            "(p_z > 0 for every alive particle); only |p| survives the "
            "conversion, so a backward p_z would silently flip sign"
        )
    p_norm = np.linalg.norm(momentum, axis=1)
    z = state.position_m[:, 2]

    particles = xtrack.Particles(
        p0c=frame.p0c_ev,
        mass0=species.mass_kg * c**2 / e,
        q0=int(round(species.charge_c / e)),
        x=state.position_m[:, 0].copy(),
        y=state.position_m[:, 1].copy(),
        px=momentum[:, 0] / p0_si,
        py=momentum[:, 1] / p0_si,
        delta=p_norm / p0_si - 1.0,
        zeta=z - beta0 * c * state.time_s,
        s=z.copy(),
        weight=state.weight.copy(),
        particle_id=state.particle_id.astype(np.int64),
        parent_particle_id=state.parent_id.astype(np.int64),
    )
    particles.state = np.where(state.alive, 1, 0).astype(particles.state.dtype)
    return particles


def from_xtrack_particles(
    particles,
    species,
    frame: ReferenceFrame,
    template_state: ParticleState | None = None,
    line_length_m: float | None = None,
) -> ParticleState:
    """Convert xtrack.Particles back, restoring the template's order.

    xtrack reorders lost particles to the tail during tracking, so rows
    are realigned by ``particle_id``. Dead particles never resurrect:
    the result's alive mask is ANDed with the template's.

    xtrack resets per-particle ``s`` to 0 when a turn completes, so any
    tracking that finishes a turn needs ``line_length_m`` (and the
    template for the initial z offsets) to reconstruct ``z`` and the
    per-particle time; ``xsuite_tracking_stage`` passes it
    automatically. ``metadata["xtrack_at_element"]`` is only meaningful
    for lost rows — xtrack wraps it back to 0 for survivors.
    """
    xtrack = _require_xtrack()
    p0_si = frame.p0_si()
    beta0 = frame.beta0(species.mass_kg)

    ids = np.asarray(particles.particle_id)
    if template_state is not None:
        desired = template_state.particle_id
    else:
        desired = np.sort(ids)
    order = {int(pid): row for row, pid in enumerate(ids)}
    index = np.array([order[int(pid)] for pid in desired], dtype=int)

    def col(name):
        return np.asarray(getattr(particles, name), dtype=np.float64)[index]

    px_si = col("px") * p0_si
    py_si = col("py") * p0_si
    p_norm = (1.0 + col("delta")) * p0_si
    pz_sq = p_norm**2 - px_si**2 - py_si**2
    pz_si = np.sqrt(np.maximum(pz_sq, 0.0))  # forward-going convention

    at_turn = np.asarray(particles.at_turn)[index].astype(np.int64)
    if np.any(at_turn > 0):
        if line_length_m is None:
            raise ValueError(
                "particles completed at least one turn; pass line_length_m "
                "(and a template_state for initial z offsets) to reconstruct z"
            )
        # survivors: s was reset to 0 at turn end and the initial z offset
        # was wiped with it; rows lost mid-turn keep their s (with the
        # offset already inside for turn 0)
        z_in = (
            template_state.position_m[:, 2]
            if template_state is not None
            else np.zeros(len(desired))
        )
        z = at_turn * float(line_length_m) + col("s") + np.where(at_turn > 0, z_in, 0.0)
    else:
        z = col("s")
    time_s = (z - col("zeta")) / (beta0 * c)
    alive = np.asarray(particles.state)[index] > 0
    if template_state is not None:
        alive = alive & template_state.alive

    return ParticleState(
        species=species,
        position_m=np.stack([col("x"), col("y"), z], axis=1),
        momentum_kg_m_s=np.stack([px_si, py_si, pz_si], axis=1),
        time_s=time_s,
        weight=col("weight"),
        alive=alive,
        particle_id=desired.copy(),
        parent_id=np.asarray(particles.parent_particle_id)[index].astype(int),
        lost_at_element=(
            template_state.lost_at_element.copy()
            if template_state is not None and template_state.lost_at_element is not None
            else np.full(len(desired), -1, dtype=np.int32)
        ),
        metadata={
            **(dict(template_state.metadata) if template_state is not None else {}),
            "xtrack_version": xtrack.__version__,
            "xtrack_at_element": np.asarray(particles.at_element)[index].copy(),
        },
    )


def xsuite_tracking_stage(
    label: str,
    line,
    frame: ReferenceFrame,
    *,
    num_turns: int = 1,
) -> Stage:
    """Wrap an xtrack.Line as a pipeline Stage.

    The Stage mechanism owns the ledger: particles xtrack loses are
    stamped with this stage's index by `Stage.run`, keeping losses
    label-addressable across the engine boundary.
    """

    def action(state: ParticleState) -> ParticleState:
        particles = to_xtrack_particles(state, frame)
        line.track(particles, num_turns=num_turns)
        return from_xtrack_particles(
            particles,
            state.species,
            frame,
            template_state=state,
            line_length_m=float(line.get_length()),
        )

    return Stage(name=label, action=action)
