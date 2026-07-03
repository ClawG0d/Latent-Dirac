"""Particle species definitions with explicit mass and charge assumptions."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict

from latent_dirac.core.constants import e, m_e, m_p


class ParticleSpecies(BaseModel):
    """A particle identity used by source, transport, and diagnostic modules."""

    model_config = ConfigDict(frozen=True)

    name: str
    symbol: str
    mass_kg: float
    charge_c: float
    pdg_id: int
    is_antimatter: bool


electron = ParticleSpecies(
    name="electron",
    symbol="e-",
    mass_kg=m_e,
    charge_c=-e,
    pdg_id=11,
    is_antimatter=False,
)

positron = ParticleSpecies(
    name="positron",
    symbol="e+",
    mass_kg=m_e,
    charge_c=e,
    pdg_id=-11,
    is_antimatter=True,
)

proton = ParticleSpecies(
    name="proton",
    symbol="p+",
    mass_kg=m_p,
    charge_c=e,
    pdg_id=2212,
    is_antimatter=False,
)

antiproton = ParticleSpecies(
    name="antiproton",
    symbol="pbar",
    mass_kg=m_p,
    charge_c=-e,
    pdg_id=-2212,
    is_antimatter=True,
)

BUILTIN_SPECIES = {
    species.name: species
    for species in (electron, positron, proton, antiproton)
}
