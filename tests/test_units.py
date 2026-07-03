import numpy as np

from latent_dirac.core.units import (
    ev_to_joule,
    gev_to_joule,
    joule_to_ev,
    momentum_gev_c_to_si,
    momentum_si_to_gev_c,
)


def test_ev_joule_conversion_round_trip():
    energy_ev = np.array([1.0, 10.0, 1.0e6])
    assert np.allclose(joule_to_ev(ev_to_joule(energy_ev)), energy_ev)


def test_gev_c_to_si_momentum_round_trip():
    momentum_gev_c = np.array([0.01, 1.0, 7.5])
    momentum_si = momentum_gev_c_to_si(momentum_gev_c)
    assert np.allclose(momentum_si_to_gev_c(momentum_si), momentum_gev_c)


def test_gev_energy_conversion_uses_ev_scale():
    assert np.isclose(gev_to_joule(1.0), ev_to_joule(1.0e9))
