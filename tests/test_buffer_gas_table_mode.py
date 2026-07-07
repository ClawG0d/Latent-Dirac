"""Tests for table-based buffer_gas_cooling (T5, slice 3).

The element gains a second mode: a curated cross-section table
(`cross_section_path` + `gas_pressure_pa`) drives the null-collision
operator, versus the existing constant-rate parameterized mode. A scene
`model_validator` enforces exactly one mode.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from pydantic import ValidationError

from latent_dirac.diagnostics.scene_report import scene_report
from latent_dirac.scene.build import build_source, run_scene
from latent_dirac.scene.loader import load_scene, scene_from_mapping

TOY = Path(__file__).resolve().parents[1] / "examples/data/cross_sections/n2_positron_toy.csv"
DEMO = Path(__file__).resolve().parents[1] / "examples/scenes/buffer_gas_table_cooling.yaml"
DEMO_CF4 = Path(__file__).resolve().parents[1] / "examples/scenes/buffer_gas_cf4_cooling.yaml"


def table_scene(
    cross_section_path=str(TOY),
    gas_pressure_pa=0.1,
    hold_time_s=1e-4,
    gas_temperature_k=300.0,
    mean_energy_MeV=4e-6,  # ~4 eV: below the toy Ps/electronic thresholds
    macro_particles=400,
    seed=11,
    extra_element=None,
):
    element = {
        "type": "buffer_gas_cooling",
        "label": "cooler",
        "hold_time_s": hold_time_s,
        "gas_temperature_k": gas_temperature_k,
    }
    if cross_section_path is not None:
        element["cross_section_path"] = cross_section_path
    if gas_pressure_pa is not None:
        element["gas_pressure_pa"] = gas_pressure_pa
    if extra_element:
        element.update(extra_element)
    return scene_from_mapping(
        {
            "schema_version": 1,
            "name": "buffer-gas-table",
            "seed": seed,
            "solver": {"dt_s": 1e-9, "steps": 1},
            "source": {
                "type": "positron_pair",
                "label": "pairs",
                "params": {
                    "primary_count": 10000,
                    "yield_eplus_per_primary": 0.02,
                    "mean_energy_MeV": mean_energy_MeV,
                    "energy_spread_MeV": mean_energy_MeV * 0.2,
                    "angular_rms_rad": 0.2,
                    "source_sigma_m": 1e-3,
                    "bunch_length_s": 1e-10,
                    "macro_particles": macro_particles,
                },
            },
            "elements": [element, {"type": "monitor", "label": "end"}],
        }
    )


def test_table_mode_requires_gas_pressure():
    with pytest.raises(ValidationError, match="gas_pressure_pa"):
        table_scene(gas_pressure_pa=None)


def test_table_mode_rejects_constant_rate_fields():
    with pytest.raises(ValidationError, match="constant-rate|cross_section_path"):
        table_scene(extra_element={"collision_rate_hz": 2e8})


def test_constant_mode_requires_its_fields():
    # no path -> constant-rate mode, which needs the three rate fields
    with pytest.raises(ValidationError, match="collision_rate_hz|constant-rate"):
        table_scene(cross_section_path=None, gas_pressure_pa=None)


def test_gas_pressure_only_in_table_mode():
    with pytest.raises(ValidationError, match="gas_pressure_pa|table"):
        scene_from_mapping(
            {
                "schema_version": 1,
                "name": "x",
                "seed": 1,
                "solver": {"dt_s": 1e-9, "steps": 1},
                "source": {"type": "positron_pair", "label": "s", "params": {"macro_particles": 4}},
                "elements": [
                    {
                        "type": "buffer_gas_cooling",
                        "label": "c",
                        "hold_time_s": 1e-6,
                        "collision_rate_hz": 2e8,
                        "energy_loss_ev": 8.5,
                        "ps_fraction": 0.0,
                        "gas_pressure_pa": 0.1,  # invalid without a table
                    }
                ],
            }
        )


def test_table_mode_cools_the_cloud():
    scene = table_scene()
    initial = build_source(scene).sample(np.random.default_rng(scene.seed))
    result = run_scene(scene)
    final = result.pipeline_result.final_cloud
    assert final.mean_kinetic_energy_joule() < 0.5 * initial.kinetic_energy_joule().mean()
    assert final.alive.all()  # below the Ps threshold: no losses


def test_report_carries_cross_section_provenance():
    scene = table_scene()
    result = run_scene(scene)
    text = scene_report(scene, result, scope_note="test")
    assert "cross-section provenance" in text.lower()
    assert "N2" in text
    assert "parameterized" in text  # synthetic toy stays parameterized


_TABLE_BASED_CSV = """\
# gas: CF4
# fidelity_tier: table-based
# source: fixture standing in for a cited dataset
# doi: 10.0000/test.fixture
# method: test fixture
# energy_unit: eV
# sigma_unit: m^2
# channels: elastic,vibrational
# thresholds_ev: elastic=0.0,vibrational=0.3
energy_eV,elastic,vibrational
0.1,5.0e-20,0.0
5.0,4.5e-20,1.0e-20
20.0,3.0e-20,6.0e-21
"""


def test_report_tier_follows_the_table_not_a_hardcode(tmp_path):
    # honesty: the reported tier must come FROM the table, so a table
    # declaring table-based reports table-based (guards against hardcoding
    # "parameterized"). The fixture is a throwaway test CSV, not real data.
    csv = tmp_path / "cited.csv"
    csv.write_text(_TABLE_BASED_CSV)
    scene = table_scene(cross_section_path=str(csv))
    text = scene_report(scene, run_scene(scene), scope_note="test")
    assert "tier table-based" in text
    assert "tier parameterized" not in text


def test_gas_number_density_is_pressure_over_kbt():
    # pin the core slice-3 conversion n = P / (k_B T) (guards a units swap)
    from latent_dirac.core.constants import k_B

    scene = table_scene(gas_pressure_pa=0.2, gas_temperature_k=250.0)
    result = run_scene(scene)
    prov = result.pipeline_result.final_cloud.metadata["buffer_gas"]["cooler"]
    assert prov["n_gas_m3"] == pytest.approx(0.2 / (k_B * 250.0))


def test_table_mode_rejects_zero_temperature():
    # the only zero-guard on the density denominator (Field allows ge=0)
    with pytest.raises(ValidationError, match="gas_temperature_k"):
        table_scene(gas_temperature_k=0.0)


def test_table_mode_losses_are_ledgered_at_the_cooler_stage():
    # inject ABOVE the toy Ps threshold (8.8 eV): some positrons form
    # positronium and are killed; verify the ledger through the scene wiring
    scene = table_scene(
        mean_energy_MeV=2e-5,  # ~20 eV, above Ps/electronic thresholds
        hold_time_s=8e-6,
        macro_particles=60,
        seed=3,
    )
    final = run_scene(scene).pipeline_result.final_cloud
    killed = ~final.alive
    assert killed.any() and final.alive.any()  # a genuine mix
    assert np.all(final.lost_at_element[killed] == 0)  # cooler is stage 0
    assert np.all(final.lost_at_element[final.alive] == -1)


def test_table_mode_is_deterministic():
    a = run_scene(table_scene(seed=5))
    b = run_scene(table_scene(seed=5))
    np.testing.assert_array_equal(
        a.pipeline_result.final_cloud.alive, b.pipeline_result.final_cloud.alive
    )
    np.testing.assert_allclose(
        a.pipeline_result.final_cloud.momentum_kg_m_s,
        b.pipeline_result.final_cloud.momentum_kg_m_s,
    )


def test_relative_cross_section_path_resolves_against_scene_dir(tmp_path):
    (tmp_path / "xs.csv").write_text(TOY.read_text())
    scene_file = tmp_path / "scene.yaml"
    scene_file.write_text(
        "schema_version: 1\n"
        "name: rel\n"
        "seed: 1\n"
        "solver: {dt_s: 1.0e-9, steps: 1}\n"
        "source: {type: positron_pair, label: s, params: {primary_count: 100,"
        " yield_eplus_per_primary: 0.02, mean_energy_MeV: 4.0e-6, energy_spread_MeV: 8.0e-7,"
        " angular_rms_rad: 0.3, source_sigma_m: 0.001, bunch_length_s: 1.0e-10,"
        " macro_particles: 8}}\n"
        "elements:\n"
        "  - {type: buffer_gas_cooling, label: c, hold_time_s: 1.0e-5,"
        " cross_section_path: xs.csv, gas_pressure_pa: 0.1}\n"
    )
    scene = load_scene(scene_file)
    # the relative path is now absolute and points at the copied table
    assert scene.elements[0].cross_section_path == str((tmp_path / "xs.csv").resolve())
    run_scene(scene)  # runs without a cwd-relative file error


def test_committed_demo_scene_runs():
    scene = load_scene(DEMO)
    result = run_scene(scene)
    assert result.pipeline_result.final_cloud.alive.any()


def test_committed_cf4_demo_scene_cools_without_losses():
    # CF4 is the fast Surko coolant; injected below the toy thresholds the
    # cloud cools via elastic+vibrational with no annihilation losses
    scene = load_scene(DEMO_CF4)
    initial = build_source(scene).sample(np.random.default_rng(scene.seed))
    final = run_scene(scene).pipeline_result.final_cloud
    assert final.alive.all()  # injected below electronic/Ps thresholds
    cooled = final.kinetic_energy_joule()[final.alive].mean()
    assert cooled < 0.25 * initial.kinetic_energy_joule().mean()
    # the reported tier stays parameterized (the CF4 toy table is synthetic)
    assert "parameterized" in scene_report(scene, run_scene(scene), scope_note="test")


def test_every_element_type_has_a_fidelity_label():
    # honesty gate: no element renders without a declared fidelity tier
    from typing import get_args

    from latent_dirac.scene.schema import ElementSpec
    from latent_dirac.viz.scene_3d import FIDELITY_LABELS

    union = get_args(ElementSpec)[0]  # Annotated[Union[...], Field(...)] -> the Union
    types = {get_args(member.model_fields["type"].annotation)[0] for member in get_args(union)}
    assert types <= set(FIDELITY_LABELS), f"missing labels: {types - set(FIDELITY_LABELS)}"
