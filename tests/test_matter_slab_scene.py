"""Tests for the matter_slab scene element (M2b), engine-free via a stub.

The stub transformer implements the phase-space exchange contract and is
invoked through the real subprocess path; the transformer binary is
injected via LATENT_DIRAC_G4_TRANSFORMER, never stored in the scene.
"""

from __future__ import annotations

import sys
import textwrap

import numpy as np
import pytest

from latent_dirac.scene.build import run_scene
from latent_dirac.scene.loader import scene_from_mapping

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
                continue  # odd ids absorbed
            x, y, z = float(row[1]), float(row[2]), float(row[3])
            px, py, pz = float(row[4]), float(row[5]), float(row[6])
            new_z = float(thickness_mm) / 1000.0 + 0.001
            out.write(f"{pid},{x},{y},{new_z},{px * 0.9},{py * 0.9},{pz * 0.9}\\n")
        out.write("# complete = true\\n")
    """
)


def write_stub(tmp_path):
    stub = tmp_path / "stub_transformer.py"
    stub.write_text(STUB, encoding="utf-8")
    return stub


def slab_scene(thickness_mm=5.0, count=6):
    return scene_from_mapping(
        {
            "schema_version": 1,
            "name": "matter-slab",
            "seed": 5,
            "solver": {"dt_s": 1e-9, "steps": 1},
            "source": {
                "type": "cold_uniform_sphere",
                "label": "cloud",
                "params": {
                    "species_name": "antiproton",
                    "macro_particles": count,
                    "radius_m": 1e-3,
                    "weight": 7.0,
                },
            },
            "elements": [
                {
                    "type": "matter_slab",
                    "label": "degrader",
                    "material": "G4_Al",
                    "thickness_mm": thickness_mm,
                    "entry_z_m": 0.0,
                },
                {"type": "monitor", "label": "end"},
            ],
        }
    )


def set_transformer(monkeypatch, stub):
    monkeypatch.setenv("LATENT_DIRAC_G4_TRANSFORMER", f"{sys.executable} {stub}")


def test_scene_runs_through_the_stub(tmp_path, monkeypatch):
    set_transformer(monkeypatch, write_stub(tmp_path))
    result = run_scene(slab_scene(count=6))
    cloud = result.pipeline_result.final_cloud

    even = cloud.particle_id % 2 == 0
    assert cloud.alive[even].all()  # even ids survive
    assert not cloud.alive[~even].any()  # odd ids absorbed
    # absorbed particles are ledgered at the slab's stage index (0)
    assert np.all(cloud.lost_at_element[~even] == 0)
    # survivors carry the engine's 0.9 momentum scaling (all-zero here) and
    # the downstream scoring-plane z
    assert cloud.position_m[even][:, 2].min() > 0.0


def test_provenance_reaches_the_report(tmp_path, monkeypatch):
    set_transformer(monkeypatch, write_stub(tmp_path))
    from latent_dirac.diagnostics.scene_report import scene_report

    scene = slab_scene()
    result = run_scene(scene)
    report = scene_report(scene, result, "matter-slab transport diagnostic")
    # the matter block reads metadata["matter"] independently of the top-level
    # provenance, so it survives even when an upstream source set its own
    assert "Matter engine provenance" in report
    assert "FTFP_BERT" in report
    assert "stub-11.4.2" in report
    assert "G4_Al" in report


def test_missing_transformer_env_fails_at_run_not_construction(monkeypatch):
    monkeypatch.delenv("LATENT_DIRAC_G4_TRANSFORMER", raising=False)
    # construction must succeed with no binary present...
    scene = slab_scene()
    assert scene is not None
    # ...only running the slab stage requires the binary
    with pytest.raises(RuntimeError, match="LATENT_DIRAC_G4_TRANSFORMER"):
        run_scene(scene)


def test_scene_3d_does_not_silently_hide_the_slab():
    # the roadmap-2b trap: an element with no _element_segments branch and no
    # FIDELITY_LABELS entry renders invisibly with no error
    from latent_dirac.viz.scene_3d import FIDELITY_LABELS, _element_segments

    scene = slab_scene()
    slab = scene.elements[0]
    assert slab.type == "matter_slab"
    segments = _element_segments(slab, None)
    assert segments  # a box is drawn
    assert "matter_slab" in FIDELITY_LABELS


def test_schema_rejects_bad_params():
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        slab_scene(thickness_mm=0.0)  # thickness_mm must be > 0
    with pytest.raises(ValidationError):
        scene_from_mapping(
            {
                "schema_version": 1,
                "name": "bad",
                "seed": 1,
                "solver": {"dt_s": 1e-9, "steps": 1},
                "source": {
                    "type": "cold_uniform_sphere",
                    "label": "c",
                    "params": {
                        "species_name": "antiproton",
                        "macro_particles": 4,
                        "radius_m": 1e-3,
                        "weight": 1.0,
                    },
                },
                "elements": [
                    {
                        "type": "matter_slab",
                        "label": "slab",
                        "material": "G4_Al",
                        "thickness_mm": 1.0,
                        "bogus": 1,  # extra field forbidden
                    }
                ],
            }
        )


def test_jax_backend_rejects_matter_slab():
    pytest.importorskip("jax")
    from latent_dirac.backends.jax_scene import run_scene_batched

    with pytest.raises(ValueError, match="JAX backend"):
        run_scene_batched(slab_scene(), overrides={})
