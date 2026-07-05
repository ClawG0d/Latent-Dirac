"""Tests for ROOT I/O via uproot (Spec: 2026-07-05 ROOT I/O).

Requires the optional [root] extra; skipped when uproot is missing.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

uproot = pytest.importorskip("uproot")

from latent_dirac.core.species import ParticleSpecies, positron  # noqa: E402
from latent_dirac.io.root_io import (  # noqa: E402
    read_particle_state,
    write_particle_states,
    write_scene_result,
)
from latent_dirac.scene.build import run_scene  # noqa: E402
from latent_dirac.scene.loader import load_scene  # noqa: E402
from latent_dirac.sources.base import particle_arrays  # noqa: E402
from latent_dirac.state.particle_state import ParticleState  # noqa: E402

HELLO_SCENE = Path("examples/scenes/hello_beamline.yaml")

PROVENANCE = {
    "geant4_version": "geant4-11-04-patch-02",
    "physics_list": "FTFP_BERT",
    "datasets": "G4NDL4.7.1",
    "patches": "none",
}


def make_state(count: int = 6, species=positron, metadata: dict | None = None) -> ParticleState:
    rng = np.random.default_rng(23)
    state = ParticleState(
        species=species,
        position_m=rng.normal(size=(count, 3)),
        momentum_kg_m_s=rng.normal(size=(count, 3)) * 1e-21,
        time_s=np.linspace(0.0, 2e-9, count),
        weight=np.full(count, 1.5),
        metadata=metadata or {},
        **particle_arrays(count),
    )
    if count:
        state.alive[0] = False
        state.lost_at_element[0] = 2
    return state


def test_write_read_round_trip(tmp_path):
    state = make_state(metadata={"provenance": dict(PROVENANCE), "note": "x"})
    path = tmp_path / "cloud.root"

    write_particle_states(path, {"snapshot": state})
    loaded = read_particle_state(path, "snapshot")

    np.testing.assert_array_equal(loaded.position_m, state.position_m)
    np.testing.assert_array_equal(loaded.momentum_kg_m_s, state.momentum_kg_m_s)
    np.testing.assert_array_equal(loaded.time_s, state.time_s)
    np.testing.assert_array_equal(loaded.weight, state.weight)
    np.testing.assert_array_equal(loaded.particle_id, state.particle_id)
    np.testing.assert_array_equal(loaded.parent_id, state.parent_id)
    np.testing.assert_array_equal(loaded.alive, state.alive)
    np.testing.assert_array_equal(loaded.lost_at_element, state.lost_at_element)
    assert loaded.species == state.species
    assert loaded.metadata["provenance"] == PROVENANCE
    assert loaded.metadata["note"] == "x"


def test_branches_readable_with_plain_uproot(tmp_path):
    state = make_state()
    path = tmp_path / "plain.root"

    write_particle_states(path, {"snap": state})

    with uproot.open(path) as f:
        # a TTree, not an RNTuple: universal ROOT-ecosystem readability is
        # the point, and uproot's dict-assignment default has changed before
        assert f.classnames()["snap;1"] == "TTree"
        tree = f["snap"]
        arrays = tree.arrays(library="np")
    expected = {
        "x_m", "y_m", "z_m",
        "px_kg_m_s", "py_kg_m_s", "pz_kg_m_s",
        "time_s", "weight", "particle_id", "parent_id",
        "alive", "lost_at_element",
    }
    assert expected.issubset(arrays.keys())
    np.testing.assert_array_equal(arrays["x_m"], state.position_m[:, 0])
    np.testing.assert_array_equal(arrays["pz_kg_m_s"], state.momentum_kg_m_s[:, 2])


def test_scene_result_monitors_then_final(tmp_path):
    scene = load_scene(HELLO_SCENE)
    result = run_scene(scene)
    path = tmp_path / "hello.root"

    write_scene_result(path, result)

    with uproot.open(path) as f:
        tree_names = {key.split(";")[0] for key in f.keys(cycle=True)}
    for label in result.monitors:
        assert label in tree_names
    assert "final" in tree_names

    final = read_particle_state(path, "final")
    np.testing.assert_array_equal(
        final.position_m, result.pipeline_result.final_cloud.position_m
    )
    np.testing.assert_array_equal(
        final.lost_at_element, result.pipeline_result.final_cloud.lost_at_element
    )


def test_custom_species_survives_round_trip(tmp_path):
    muon_like = ParticleSpecies(
        name="antimuon",
        symbol="mu+",
        mass_kg=1.883531627e-28,
        charge_c=1.602176634e-19,
        pdg_id=-13,
        is_antimatter=True,
    )
    state = make_state(species=muon_like)
    path = tmp_path / "muon.root"

    write_particle_states(path, {"s": state})
    loaded = read_particle_state(path, "s")

    assert loaded.species == muon_like


def test_zero_particle_state_raises(tmp_path):
    state = make_state(count=0)
    with pytest.raises(ValueError, match="zero-particle"):
        write_particle_states(tmp_path / "zero.root", {"z": state})


def test_empty_mapping_raises(tmp_path):
    with pytest.raises(ValueError):
        write_particle_states(tmp_path / "empty.root", {})


def test_missing_label_raises(tmp_path):
    state = make_state()
    path = tmp_path / "one.root"
    write_particle_states(path, {"a": state})

    with pytest.raises(KeyError):
        read_particle_state(path, "does-not-exist")


@pytest.mark.parametrize("label", ["a/b", "a;1", "", "a__metadata"])
def test_unsafe_labels_are_rejected(tmp_path, label):
    # "/" creates ROOT directories, ";" is the cycle separator (write
    # succeeds but our own reader then fails), "__metadata" collides with
    # the sidecar key of another label
    state = make_state()
    with pytest.raises(ValueError):
        write_particle_states(tmp_path / "bad.root", {label: state})


def test_failed_write_preserves_existing_file(tmp_path):
    good = make_state()
    path = tmp_path / "keep.root"
    write_particle_states(path, {"good": good})

    bad = make_state(metadata={np.int64(3): "non-string key"})
    with pytest.raises(TypeError):
        write_particle_states(path, {"good": good, "bad": bad})

    # serialization happens before the file is opened: the original
    # content must survive the failed attempt
    reloaded = read_particle_state(path, "good")
    np.testing.assert_array_equal(reloaded.position_m, good.position_m)


def test_monitor_label_final_collision_raises(tmp_path):
    from latent_dirac.pipeline.runner import PipelineResult

    state = make_state()
    fake = type(
        "FakeResult",
        (),
        {
            "monitors": {"final": state},
            "pipeline_result": PipelineResult(final_cloud=state, stage_results=[]),
        },
    )
    with pytest.raises(ValueError, match="final"):
        write_scene_result(tmp_path / "clash.root", fake())
