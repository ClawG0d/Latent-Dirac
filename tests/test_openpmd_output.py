"""Tests for openPMD particle output (Spec: 2026-07-05 openPMD output).

Requires the optional [openpmd] extra; skipped when openpmd-api is not
installed (same pattern as the xopt/plotly optional tests).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

api = pytest.importorskip("openpmd_api")

from latent_dirac.core.species import positron  # noqa: E402
from latent_dirac.io.openpmd_io import (  # noqa: E402
    write_particle_states,
    write_scene_result,
)
from latent_dirac.scene.build import run_scene  # noqa: E402
from latent_dirac.scene.loader import load_scene  # noqa: E402
from latent_dirac.sources.base import particle_arrays  # noqa: E402
from latent_dirac.state.particle_state import ParticleState  # noqa: E402

HELLO_SCENE = Path("examples/scenes/hello_beamline.yaml")

BACKENDS = [".json"] + ([".h5"] if api.variants.get("hdf5", False) else [])


def make_state(count: int = 5, metadata: dict | None = None) -> ParticleState:
    rng = np.random.default_rng(11)
    state = ParticleState(
        species=positron,
        position_m=rng.normal(size=(count, 3)),
        momentum_kg_m_s=rng.normal(size=(count, 3)) * 1e-21,
        time_s=np.linspace(0.0, 1e-9, count),
        weight=np.full(count, 2.5),
        metadata=metadata or {},
        **particle_arrays(count),
    )
    if count:
        state.alive[-1] = False
        state.lost_at_element[-1] = 3
    return state


@pytest.mark.parametrize("suffix", BACKENDS)
def test_round_trip_single_state(tmp_path, suffix):
    state = make_state()
    path = tmp_path / f"cloud{suffix}"

    write_particle_states(path, {"snapshot": state})

    series = api.Series(str(path), api.Access.read_only)
    iteration = series.iterations[0]
    assert iteration.get_attribute("latentDirac_label") == "snapshot"
    particles = iteration.particles[positron.name]

    # load_chunk buffers fill only at flush; stack afterwards
    pos_parts = [particles["position"][ax].load_chunk() for ax in "xyz"]
    mom_parts = [particles["momentum"][ax].load_chunk() for ax in "xyz"]
    weight = particles["weighting"][api.Record_Component.SCALAR].load_chunk()
    ids = particles["id"][api.Record_Component.SCALAR].load_chunk()
    parents = particles["parentId"][api.Record_Component.SCALAR].load_chunk()
    times = particles["time"][api.Record_Component.SCALAR].load_chunk()
    alive = particles["alive"][api.Record_Component.SCALAR].load_chunk()
    lost = particles["lostAtElement"][api.Record_Component.SCALAR].load_chunk()
    series.flush()
    pos = np.stack(pos_parts, axis=1)
    mom = np.stack(mom_parts, axis=1)

    np.testing.assert_array_equal(pos, state.position_m)
    np.testing.assert_array_equal(mom, state.momentum_kg_m_s)
    np.testing.assert_array_equal(weight, state.weight)
    np.testing.assert_array_equal(ids, state.particle_id)
    np.testing.assert_array_equal(parents, state.parent_id)
    np.testing.assert_array_equal(times, state.time_s)
    np.testing.assert_array_equal(alive.astype(bool), state.alive)
    np.testing.assert_array_equal(lost, state.lost_at_element)

    # openPMD dimension order: (L, M, T, I, theta, N, J)
    assert tuple(particles["momentum"].unit_dimension) == (1.0, 1.0, -1.0, 0.0, 0.0, 0.0, 0.0)
    assert tuple(particles["charge"].unit_dimension) == (0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0)
    assert particles["momentum"]["x"].unit_SI == 1.0

    charge = particles["charge"][api.Record_Component.SCALAR]
    mass = particles["mass"][api.Record_Component.SCALAR]
    assert charge.get_attribute("value") == pytest.approx(positron.charge_c)
    assert mass.get_attribute("value") == pytest.approx(positron.mass_kg)
    series.close()


def test_scene_result_writes_monitors_then_final(tmp_path):
    scene = load_scene(HELLO_SCENE)
    result = run_scene(scene)
    path = tmp_path / "hello.json"

    write_scene_result(path, result)

    series = api.Series(str(path), api.Access.read_only)
    labels = [
        series.iterations[i].get_attribute("latentDirac_label")
        for i in sorted(series.iterations)
    ]
    assert labels == list(result.monitors) + ["final"]
    assert len(labels) == len(result.monitors) + 1
    series.close()


def test_provenance_four_tuple_is_lifted(tmp_path):
    provenance = {
        "geant4_version": "geant4-11-04-patch-02",
        "physics_list": "FTFP_BERT",
        "datasets": "G4NDL4.7.1",
        "patches": "none",
    }
    state = make_state(metadata={"provenance": provenance, "note": "x"})
    path = tmp_path / "prov.json"

    write_particle_states(path, {"table": state})

    series = api.Series(str(path), api.Access.read_only)
    particles = series.iterations[0].particles[positron.name]
    assert particles.get_attribute("latentDirac_geant4Version") == provenance["geant4_version"]
    assert particles.get_attribute("latentDirac_physicsList") == "FTFP_BERT"
    assert particles.get_attribute("latentDirac_datasets") == "G4NDL4.7.1"
    assert particles.get_attribute("latentDirac_patches") == "none"
    stored = json.loads(particles.get_attribute("latentDirac_metadata"))
    assert stored["note"] == "x"
    series.close()


def test_state_without_provenance_has_no_lifted_attributes(tmp_path):
    state = make_state()
    path = tmp_path / "plain.json"

    write_particle_states(path, {"s": state})

    series = api.Series(str(path), api.Access.read_only)
    particles = series.iterations[0].particles[positron.name]
    assert "latentDirac_geant4Version" not in particles.attributes
    assert "latentDirac_metadata" in particles.attributes
    series.close()


def test_empty_mapping_raises(tmp_path):
    with pytest.raises(ValueError):
        write_particle_states(tmp_path / "empty.json", {})


def test_zero_particle_state_raises_clear_error(tmp_path):
    state = make_state(count=0)
    with pytest.raises(ValueError, match="zero-particle"):
        write_particle_states(tmp_path / "zero.json", {"empty": state})


def test_monitor_label_final_collision_raises(tmp_path):
    from latent_dirac.pipeline.runner import PipelineResult

    state = make_state()
    result_type = type(
        "FakeResult",
        (),
        {
            "monitors": {"final": state},
            "pipeline_result": PipelineResult(final_cloud=state, stage_results=[]),
        },
    )
    with pytest.raises(ValueError, match="final"):
        write_scene_result(tmp_path / "clash.json", result_type())
