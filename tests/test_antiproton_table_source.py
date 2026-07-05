"""Tests for the engine yield-table antiproton source (table-based tier).

The CSV contract is defined in
docs/superpowers/specs/2026-07-05-engine-yieldgen-demo-design.md: header
lines start with '#', `# n_primaries = <int>` is mandatory, data rows are
x_m,y_m,z_m,px_gev_c,py_gev_c,pz_gev_c.
"""

from __future__ import annotations

import numpy as np
import pytest

from latent_dirac.core.units import momentum_gev_c_to_si
from latent_dirac.scene.loader import scene_from_mapping
from latent_dirac.sources.antiproton_table import AntiprotonYieldTableSource

HEADER = """# latent-dirac antiproton yield table v1
# generator = engine/yieldgen
# geant4_version = geant4-11-04-patch-02
# physics_list = FTFP_BERT
# n_primaries = 1000
# columns = x_m,y_m,z_m,px_gev_c,py_gev_c,pz_gev_c
"""

ROWS = [
    (0.0000, 0.0005, 0.0275, 0.10, -0.05, 3.60),
    (0.0010, -0.0002, 0.0275, -0.20, 0.10, 3.10),
    (-0.0003, 0.0001, 0.0275, 0.05, 0.02, 4.20),
    (0.0002, 0.0000, 0.0275, 0.00, -0.01, 2.80),
]


def write_table(path, header=HEADER, rows=ROWS, complete=True):
    lines = [header.rstrip("\n")]
    lines += [",".join(f"{v:.6g}" for v in row) for row in rows]
    if complete:
        lines.append("# complete = true")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def test_one_macro_particle_per_row_with_normalized_weights(tmp_path):
    table = write_table(tmp_path / "yield.csv")
    source = AntiprotonYieldTableSource(
        table_path=str(table),
        primary_proton_count=5.0e13,
    )
    cloud = source.sample(np.random.default_rng(7))

    assert cloud.species.name == "antiproton"
    assert cloud.position_m.shape == (len(ROWS), 3)
    # each row represents primary_proton_count / n_primaries physical pbars
    expected_weight = 5.0e13 / 1000
    np.testing.assert_allclose(cloud.weight, expected_weight)
    # total represented yield = primary_count * n_rows / n_primaries
    assert cloud.weighted_count() == pytest.approx(5.0e13 * len(ROWS) / 1000)


def test_momentum_and_position_come_from_the_table(tmp_path):
    table = write_table(tmp_path / "yield.csv")
    source = AntiprotonYieldTableSource(table_path=str(table), primary_proton_count=1.0e12)
    cloud = source.sample(np.random.default_rng(7))

    np.testing.assert_allclose(cloud.position_m[0], [0.0, 0.0005, 0.0275])
    expected_p = momentum_gev_c_to_si(np.array([0.10, -0.05, 3.60]))
    np.testing.assert_allclose(cloud.momentum_kg_m_s[0], expected_p)


def test_bootstrap_resampling_is_reproducible_and_weight_preserving(tmp_path):
    table = write_table(tmp_path / "yield.csv")
    source = AntiprotonYieldTableSource(
        table_path=str(table),
        primary_proton_count=1.0e12,
        macro_particles=64,
    )
    cloud_a = source.sample(np.random.default_rng(11))
    cloud_b = source.sample(np.random.default_rng(11))

    assert cloud_a.position_m.shape == (64, 3)
    np.testing.assert_allclose(cloud_a.position_m, cloud_b.position_m)
    # renormalized: weighted total unchanged by resampling
    assert cloud_a.weighted_count() == pytest.approx(1.0e12 * len(ROWS) / 1000)


def test_provenance_is_carried_in_metadata(tmp_path):
    table = write_table(tmp_path / "yield.csv")
    source = AntiprotonYieldTableSource(table_path=str(table), primary_proton_count=1.0)
    cloud = source.sample(np.random.default_rng(0))

    assert cloud.metadata["model_type"] == "table_based"
    assert cloud.metadata["provenance"]["geant4_version"] == "geant4-11-04-patch-02"
    assert cloud.metadata["provenance"]["physics_list"] == "FTFP_BERT"
    assert cloud.metadata["provenance"]["n_primaries"] == 1000


def test_missing_n_primaries_header_is_rejected(tmp_path):
    header = HEADER.replace("# n_primaries = 1000\n", "")
    table = write_table(tmp_path / "yield.csv", header=header)

    with pytest.raises(ValueError, match="n_primaries"):
        AntiprotonYieldTableSource(table_path=str(table), primary_proton_count=1.0).sample(
            np.random.default_rng(0)
        )


def test_malformed_row_is_rejected(tmp_path):
    table = tmp_path / "yield.csv"
    table.write_text(HEADER + "0.0,0.0,0.0,not_a_number,0.0,3.0\n", encoding="utf-8")

    with pytest.raises(ValueError, match="row"):
        AntiprotonYieldTableSource(table_path=str(table), primary_proton_count=1.0).sample(
            np.random.default_rng(0)
        )


def test_empty_table_is_rejected(tmp_path):
    table = tmp_path / "yield.csv"
    table.write_text(HEADER, encoding="utf-8")

    with pytest.raises(ValueError, match="no data rows"):
        AntiprotonYieldTableSource(table_path=str(table), primary_proton_count=1.0).sample(
            np.random.default_rng(0)
        )


def test_missing_completion_marker_is_rejected(tmp_path):
    table = write_table(tmp_path / "yield.csv", complete=False)

    with pytest.raises(ValueError, match="complete"):
        AntiprotonYieldTableSource(table_path=str(table), primary_proton_count=1.0).sample(
            np.random.default_rng(0)
        )


def test_integral_float_n_primaries_is_accepted_and_junk_is_contextualized(tmp_path):
    float_header = HEADER.replace("# n_primaries = 1000", "# n_primaries = 1000.0")
    table = write_table(tmp_path / "yield.csv", header=float_header)
    cloud = AntiprotonYieldTableSource(table_path=str(table), primary_proton_count=1000.0).sample(
        np.random.default_rng(0)
    )
    np.testing.assert_allclose(cloud.weight, 1.0)

    junk_header = HEADER.replace("# n_primaries = 1000", "# n_primaries = lots")
    table = write_table(tmp_path / "junk.csv", header=junk_header)
    with pytest.raises(ValueError, match="n_primaries"):
        AntiprotonYieldTableSource(table_path=str(table), primary_proton_count=1.0).sample(
            np.random.default_rng(0)
        )


def test_relative_table_path_resolves_against_the_scene_file(tmp_path, monkeypatch):
    from latent_dirac.scene.build import build_source
    from latent_dirac.scene.loader import load_scene

    (tmp_path / "data").mkdir()
    (tmp_path / "scenes").mkdir()
    (tmp_path / "elsewhere").mkdir()
    write_table(tmp_path / "data" / "yield.csv")
    scene_path = tmp_path / "scenes" / "engine.yaml"
    scene_path.write_text(
        "\n".join(
            [
                "schema_version: 1",
                "name: cwd-independent",
                "seed: 1",
                "source:",
                "  type: antiproton_yield_table",
                "  label: pbar-table",
                "  params: { table_path: ../data/yield.csv, primary_proton_count: 1.0e+12 }",
                "solver: { type: relativistic_boris, dt_s: 1.0e-11, steps: 5 }",
                "elements:",
                "  - { type: monitor, label: end }",
            ]
        ),
        encoding="utf-8",
    )

    monkeypatch.chdir(tmp_path / "elsewhere")
    scene = load_scene(scene_path)
    cloud = build_source(scene).sample(np.random.default_rng(1))
    assert cloud.position_m.shape == (len(ROWS), 3)


def test_scene_report_prints_the_provenance_four_tuple(tmp_path):
    from latent_dirac.diagnostics.scene_report import scene_report
    from latent_dirac.scene.build import run_scene

    table = write_table(tmp_path / "yield.csv")
    scene = scene_from_mapping(
        {
            "schema_version": 1,
            "name": "provenance-check",
            "seed": 1,
            "source": {
                "type": "antiproton_yield_table",
                "label": "pbar-table",
                "params": {"table_path": str(table), "primary_proton_count": 1.0e12},
            },
            "solver": {"type": "relativistic_boris", "dt_s": 1.0e-11, "steps": 5},
            "elements": [{"type": "monitor", "label": "end"}],
        }
    )
    report = scene_report(scene, run_scene(scene), "table replay diagnostic only")

    assert "Source provenance (engine four-tuple):" in report
    assert "geant4-11-04-patch-02" in report
    assert "FTFP_BERT" in report
    assert "table primaries: 1000" in report


def test_scene_schema_accepts_the_new_source_type(tmp_path):
    table = write_table(tmp_path / "yield.csv")
    scene = scene_from_mapping(
        {
            "schema_version": 1,
            "name": "engine-target",
            "seed": 3,
            "source": {
                "type": "antiproton_yield_table",
                "label": "pbar-table",
                "params": {
                    "table_path": str(table),
                    "primary_proton_count": 1.0e13,
                },
            },
            "solver": {"type": "relativistic_boris", "dt_s": 1.0e-11, "steps": 10},
            "elements": [{"type": "monitor", "label": "end"}],
        }
    )
    from latent_dirac.scene.build import build_source

    source = build_source(scene)
    cloud = source.sample(np.random.default_rng(3))
    assert cloud.species.name == "antiproton"
    assert cloud.position_m.shape == (len(ROWS), 3)
