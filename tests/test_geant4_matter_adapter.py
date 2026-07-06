"""Tests for the real Geant4 Matter adapter (engine-free via a stub).

The stub transformer is a Python script invoked through the adapter's
real subprocess path; it implements the phase-space exchange contract
from docs/superpowers/specs/2026-07-05-geant4-matter-adapter-design.md.
"""

from __future__ import annotations

import sys
import textwrap

import numpy as np
import pytest

from latent_dirac.adapters.geant4.adapter import Geant4MatterAdapter
from latent_dirac.core.species import antiproton, positron
from latent_dirac.core.units import momentum_gev_c_to_si
from latent_dirac.sources.base import particle_arrays
from latent_dirac.state.particle_state import ParticleState

STUB = textwrap.dedent(
    """
    import sys

    in_path, out_path, material, thickness_mm = sys.argv[1:5]
    header = {}
    rows = []
    for line in open(in_path, encoding="utf-8"):
        line = line.strip()
        if line.startswith("#"):
            body = line.lstrip("#").strip()
            if "=" in body:
                key, _, value = body.partition("=")
                header[key.strip()] = value.strip()
        elif line:
            rows.append(line.split(","))

    with open(out_path, "w", encoding="utf-8") as out:
        out.write("# latent-dirac phase space v1\\n")
        out.write(f"# species = {header['species']}\\n")
        out.write("# generator = stub-transformer\\n")
        out.write("# geant4_version = stub-11.4.2\\n")
        out.write("# physics_list = FTFP_BERT\\n")
        out.write("# datasets = stub-data\\n")
        out.write("# patches = none\\n")
        out.write(f"# material = {material}\\n")
        out.write(f"# thickness_mm = {thickness_mm}\\n")
        out.write(f"# n_primaries = {len(rows)}\\n")
        out.write("# columns = id,x_m,y_m,z_m,px_gev_c,py_gev_c,pz_gev_c\\n")
        for row in rows:
            pid = int(row[0])
            if pid % 2 == 1:
                continue  # odd ids are absorbed
            x, y, z = float(row[1]), float(row[2]), float(row[3])
            px, py, pz = float(row[4]), float(row[5]), float(row[6])
            new_z = float(thickness_mm) / 1000.0 + 0.001
            out.write(
                f"{pid},{x},{y},{new_z},{px * 0.9},{py * 0.9},{pz * 0.9}\\n"
            )
        MARKER
    """
)


def write_stub(tmp_path, marker='out.write("# complete = true\\n")'):
    stub = tmp_path / "stub_transformer.py"
    stub.write_text(STUB.replace("MARKER", marker), encoding="utf-8")
    return stub


def make_cloud(count=6, species=antiproton, momentum_gev_c=3.6):
    momentum = momentum_gev_c_to_si(momentum_gev_c)
    return ParticleState(
        species=species,
        position_m=np.column_stack(
            [np.linspace(-0.002, 0.002, count), np.zeros(count), np.full(count, 0.05)]
        ),
        momentum_kg_m_s=np.column_stack([np.zeros(count), np.zeros(count), np.full(count, momentum)]),
        time_s=np.zeros(count),
        weight=np.full(count, 7.0),
        **particle_arrays(count),
    )


def make_adapter(stub, **overrides):
    params = {
        "command": (sys.executable, str(stub)),
        "material": "G4_Al",
        "thickness_mm": 1.0,
        "entry_z_m": 0.05,
    }
    params.update(overrides)
    return Geant4MatterAdapter(**params)


def test_survivors_update_and_absorbed_die(tmp_path):
    adapter = make_adapter(write_stub(tmp_path))
    cloud = make_cloud(count=6)
    out = adapter.apply(cloud)

    # stub keeps even ids only
    np.testing.assert_array_equal(out.alive, [True, False, True, False, True, False])
    # survivor momentum scaled by 0.9 (stub's energy-loss stand-in)
    np.testing.assert_allclose(
        out.momentum_kg_m_s[out.alive][:, 2], cloud.momentum_kg_m_s[out.alive][:, 2] * 0.9
    )
    # z comes back in pipeline coordinates: entry_z + thickness + 1mm plane
    np.testing.assert_allclose(out.position_m[out.alive][:, 2], 0.05 + 0.001 + 0.001)
    # weights and dead-particle phase space untouched
    np.testing.assert_allclose(out.weight, 7.0)
    np.testing.assert_allclose(out.position_m[~out.alive], cloud.position_m[~out.alive])


def test_already_dead_particles_are_not_sent(tmp_path):
    stub = write_stub(tmp_path)
    adapter = make_adapter(stub)
    cloud = make_cloud(count=4)
    cloud.alive[0] = False
    cloud.lost_at_element[0] = 3

    out = adapter.apply(cloud)

    assert not out.alive[0] and out.lost_at_element[0] == 3
    # id 0 was never sent, so the stub's even-id survival applies to 1..3
    np.testing.assert_array_equal(out.alive, [False, False, True, False])


def test_provenance_and_matter_metadata(tmp_path):
    adapter = make_adapter(write_stub(tmp_path))
    out = adapter.apply(make_cloud())

    # a bare cloud has no source provenance, so the transform fills the top level
    assert out.metadata["model_type"] == "engine_transformer"
    provenance = out.metadata["provenance"]
    assert provenance["geant4_version"] == "stub-11.4.2"
    assert provenance["physics_list"] == "FTFP_BERT"
    matter = out.metadata["matter"]
    assert matter["material"] == "G4_Al"
    assert matter["thickness_mm"] == 1.0
    # the matter engine four-tuple is always namespaced under `matter`
    assert matter["provenance"]["geant4_version"] == "stub-11.4.2"


def test_upstream_source_provenance_is_preserved(tmp_path):
    adapter = make_adapter(write_stub(tmp_path))
    cloud = make_cloud()
    cloud.metadata.update(
        {
            "model_type": "table_based",
            "provenance": {"geant4_version": "source-11.4.2", "n_primaries": 2_000_000},
        }
    )
    out = adapter.apply(cloud)

    # a transform must not clobber the source's identity/provenance
    assert out.metadata["model_type"] == "table_based"
    assert out.metadata["provenance"]["geant4_version"] == "source-11.4.2"
    assert out.metadata["provenance"]["n_primaries"] == 2_000_000
    # but the matter engine's own provenance is still recorded under `matter`
    assert out.metadata["matter"]["provenance"]["geant4_version"] == "stub-11.4.2"


def test_entry_z_shift_round_trips(tmp_path):
    # echo each row's own z back (no z change) so the write-side -entry_z_m and
    # the read-side +entry_z_m must cancel exactly. The stub parses z but the
    # default fabricates new_z from thickness; swap in an echo of the input z.
    echo = write_stub(tmp_path)
    echo.write_text(
        echo.read_text(encoding="utf-8").replace(
            "new_z = float(thickness_mm) / 1000.0 + 0.001", "new_z = z"
        ),
        encoding="utf-8",
    )

    adapter = make_adapter(echo, entry_z_m=0.05)
    cloud = make_cloud(count=4)
    out = adapter.apply(cloud)

    survivors = out.alive
    np.testing.assert_allclose(out.position_m[survivors, 2], cloud.position_m[survivors, 2])


def test_particles_outside_transverse_aperture_are_rejected(tmp_path):
    adapter = make_adapter(write_stub(tmp_path), transverse_half_width_m=0.001)
    cloud = make_cloud(count=6)  # x spans -0.002..0.002 m, beyond 1 mm aperture
    with pytest.raises(ValueError, match="aperture"):
        adapter.apply(cloud)


def test_vertex_outside_world_is_rejected(tmp_path):
    adapter = make_adapter(write_stub(tmp_path), entry_z_m=2.0, world_half_length_m=0.6)
    # cloud sits at pipeline z = 0.05, so contract-frame z = 0.05 - 2.0 = -1.95 m
    with pytest.raises(ValueError, match="world"):
        adapter.apply(make_cloud(count=4))


def test_missing_completion_marker_is_rejected(tmp_path):
    adapter = make_adapter(write_stub(tmp_path, marker="pass"))
    with pytest.raises(ValueError, match="complete"):
        adapter.apply(make_cloud())


def test_unknown_survivor_id_is_rejected(tmp_path):
    marker = 'out.write("999,0,0,0,0,0,1\\n"); out.write("# complete = true\\n")'
    adapter = make_adapter(write_stub(tmp_path, marker=marker))
    with pytest.raises(ValueError, match="id"):
        adapter.apply(make_cloud())


def test_species_header_uses_geant4_names(tmp_path):
    stub = write_stub(tmp_path)
    adapter = make_adapter(stub)
    seen = tmp_path / "seen.txt"
    marker = (
        f'open(r"{seen}", "w").write(header["species"]); '
        'out.write("# complete = true\\n")'
    )
    adapter = make_adapter(write_stub(tmp_path, marker=marker))
    adapter.apply(make_cloud(species=positron))
    assert seen.read_text() == "e+"


def test_unsupported_species_is_rejected_before_subprocess(tmp_path):
    from latent_dirac.core.species import ParticleSpecies

    muon = ParticleSpecies(
        name="muon", symbol="mu-", mass_kg=1.88e-28, charge_c=-1.6e-19, pdg_id=13, is_antimatter=False
    )
    adapter = make_adapter(tmp_path / "never_invoked.py")
    with pytest.raises(ValueError, match="species"):
        adapter.apply(make_cloud(species=muon))


def test_stage_integration_stamps_the_ledger(tmp_path):
    from latent_dirac.pipeline.runner import PipelineRunner

    adapter = make_adapter(write_stub(tmp_path))
    result = PipelineRunner(stages=[adapter.stage("aluminium-degrader")]).run(make_cloud(count=6))

    final = result.final_cloud
    assert result.stage_results[0].stage_name == "aluminium-degrader"
    np.testing.assert_array_equal(final.lost_at_element[~final.alive], 0)
    assert final.weighted_count() == pytest.approx(3 * 7.0)


def test_failed_subprocess_surfaces_stderr(tmp_path):
    bad = tmp_path / "bad.py"
    bad.write_text("import sys; sys.stderr.write('boom'); sys.exit(3)", encoding="utf-8")
    adapter = make_adapter(bad)
    with pytest.raises(RuntimeError, match="transformer"):
        adapter.apply(make_cloud())
