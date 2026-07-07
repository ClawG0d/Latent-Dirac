"""Positron yield-table source: contract, weights, committed artifact.

Design record: docs/superpowers/specs/2026-07-07-positron-yield-table-design.md.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from latent_dirac.sources.positron_table import PositronYieldTableSource

COMMITTED_TABLE = Path("examples/data/positron_yield_ftfp_bert_10mev_w.csv")

HEADER = """\
# latent-dirac positron yield table v1
# generator = engine/positrongen
# geant4_version = test-11.4.2
# physics_list = FTFP_BERT
# datasets = test-data
# patches = none
# primary = e-
# primary_kinetic_mev = 10
# n_primaries = 1000
# columns = x_m,y_m,z_m,px_gev_c,py_gev_c,pz_gev_c
"""

ROWS = "0.001,0.0,0.001,0.001,0.0,0.002\n-0.001,0.0005,0.001,0.0,0.0005,0.0015\n"


def write_table(tmp_path, body=HEADER + ROWS + "# complete = true\n") -> Path:
    table = tmp_path / "table.csv"
    table.write_text(body, encoding="utf-8")
    return table


def test_replay_weights_and_species(tmp_path):
    source = PositronYieldTableSource(table_path=str(write_table(tmp_path)), primary_electron_count=5.0e5)
    state = source.sample(np.random.default_rng(1))

    assert state.species.name == "positron"
    assert state.species.charge_c > 0.0
    # 2 rows / 1000 primaries * 5e5 electrons = 1000 physical positrons
    assert state.weighted_count() == pytest.approx(1000.0)
    assert state.metadata["model_type"] == "table_based"
    assert state.metadata["provenance"]["physics_list"] == "FTFP_BERT"


def test_bootstrap_preserves_represented_total(tmp_path):
    source = PositronYieldTableSource(
        table_path=str(write_table(tmp_path)), primary_electron_count=5.0e5, macro_particles=64
    )
    state = source.sample(np.random.default_rng(2))
    assert state.alive.size == 64
    assert state.weighted_count() == pytest.approx(1000.0)


def test_missing_completion_marker_rejected(tmp_path):
    table = write_table(tmp_path, HEADER + ROWS)  # no trailing marker
    source = PositronYieldTableSource(table_path=str(table), primary_electron_count=1.0)
    with pytest.raises(ValueError, match="complete"):
        source.sample(np.random.default_rng(0))


def test_bad_n_primaries_rejected(tmp_path):
    body = HEADER.replace("# n_primaries = 1000", "# n_primaries = many") + ROWS + "# complete = true\n"
    source = PositronYieldTableSource(table_path=str(write_table(tmp_path, body)), primary_electron_count=1.0)
    with pytest.raises(ValueError, match="n_primaries"):
        source.sample(np.random.default_rng(0))


def test_committed_table_replays_with_sane_physics():
    source = PositronYieldTableSource(
        table_path=str(COMMITTED_TABLE), primary_electron_count=1.0e10, macro_particles=256
    )
    state = source.sample(np.random.default_rng(3))

    # yield per primary in the literature ballpark for ~0.57 X0 W at 10 MeV
    rows = state.metadata["table"]["rows"]
    n_primaries = state.metadata["provenance"]["n_primaries"]
    assert 1e-3 < rows / n_primaries < 1e-2

    # kinematics bounded by the primary energy; forward hemisphere dominates
    ke_mev = state.kinetic_energy_joule() / 1.602176634e-13
    assert float(ke_mev.max()) < 10.0
    forward = float(np.mean(state.momentum_kg_m_s[:, 2] > 0.0))
    assert forward > 0.6
