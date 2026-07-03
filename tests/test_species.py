import numpy as np

from latent_dirac.core.species import antiproton, electron, positron, proton


def test_electron_and_positron_have_same_mass_and_opposite_charge():
    assert electron.name == "electron"
    assert positron.name == "positron"
    assert np.isclose(electron.mass_kg, positron.mass_kg)
    assert np.isclose(electron.charge_c, -positron.charge_c)
    assert not electron.is_antimatter
    assert positron.is_antimatter


def test_proton_and_antiproton_have_same_mass_and_opposite_charge():
    assert proton.name == "proton"
    assert antiproton.name == "antiproton"
    assert np.isclose(proton.mass_kg, antiproton.mass_kg)
    assert np.isclose(proton.charge_c, -antiproton.charge_c)
    assert not proton.is_antimatter
    assert antiproton.is_antimatter
