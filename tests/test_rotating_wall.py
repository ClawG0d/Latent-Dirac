"""Tests for the rotating-wall trap element (Phase 4 trap physics).

A rotating multipole transverse E field (parameterized tier): a uniform
rotating field for the dipole (m=1), and a quadrupole pattern linear in
transverse position for m=2. Single-particle field only — collective
plasma compression is out of scope (see the 2026-07-07 rotating-wall spec).
"""

from __future__ import annotations

import numpy as np
import pytest

from latent_dirac.fields.rotating_wall import RotatingWallField


def test_dipole_is_a_uniform_field_rotating_at_omega():
    f = 1.0e6
    fld = RotatingWallField(multipole=1, amplitude_v_m=500.0, radius_m=0.02, frequency_hz=f)
    positions = np.array([[0.0, 0.0, 0.0], [0.01, -0.03, 0.05]])  # uniform -> same everywhere

    e0 = fld.E(positions, np.zeros(2))  # theta = 0 -> +x
    np.testing.assert_allclose(e0, np.tile([500.0, 0.0, 0.0], (2, 1)), atol=1e-9)

    e_quarter = fld.E(positions, np.full(2, 1.0 / (4.0 * f)))  # theta = pi/2 -> +y
    np.testing.assert_allclose(e_quarter, np.tile([0.0, 500.0, 0.0], (2, 1)), atol=1e-6)

    e_period = fld.E(positions, np.full(2, 1.0 / f))  # full period -> back to +x
    np.testing.assert_allclose(e_period, e0, atol=1e-6)


def test_quadrupole_is_linear_in_radius():
    amp, radius = 500.0, 0.02
    fld = RotatingWallField(multipole=2, amplitude_v_m=amp, radius_m=radius, frequency_hz=1.0e6)
    # theta = 0 -> E = -g*[x, -y, 0] = [-g x, +g y, 0] (a true quadrupole)
    pos = np.array([[radius, 0.0, 0.0], [0.0, radius, 0.0], [0.0, 0.0, 0.0]])
    e = fld.E(pos, np.zeros(3))
    np.testing.assert_allclose(e[0], [-amp, 0.0, 0.0], atol=1e-9)  # |E| = amp at r = radius
    np.testing.assert_allclose(e[1], [0.0, amp, 0.0], atol=1e-9)
    np.testing.assert_allclose(e[2], [0.0, 0.0, 0.0], atol=1e-12)  # zero on axis
    # magnitude scales linearly with radius
    half = fld.E(np.array([[0.5 * radius, 0.0, 0.0]]), np.zeros(1))
    np.testing.assert_allclose(half[0], [-0.5 * amp, 0.0, 0.0], atol=1e-9)

    # a real quadrupole is divergence- and curl-free (not a radial field):
    # check div E = dEx/dx + dEy/dy ~= 0 by central differences at theta != 0
    h, t = 1e-6, 0.3e-6
    ex_px = fld.E(np.array([[radius + h, 0.5 * radius, 0.0]]), np.array([t]))[0, 0]
    ex_mx = fld.E(np.array([[radius - h, 0.5 * radius, 0.0]]), np.array([t]))[0, 0]
    ey_py = fld.E(np.array([[radius, 0.5 * radius + h, 0.0]]), np.array([t]))[0, 1]
    ey_my = fld.E(np.array([[radius, 0.5 * radius - h, 0.0]]), np.array([t]))[0, 1]
    div = (ex_px - ex_mx) / (2 * h) + (ey_py - ey_my) / (2 * h)
    assert abs(div) < 1e-6 * amp / radius


def test_b_is_zero_and_shapes_follow_the_contract():
    fld = RotatingWallField(multipole=2, amplitude_v_m=300.0, radius_m=0.02, frequency_hz=5.0e5)
    single = fld.E(np.array([0.01, 0.0, 0.0]), 0.0)  # (3,) in -> (3,) out
    assert single.shape == (3,)
    np.testing.assert_allclose(fld.B(np.array([0.01, 0.0, 0.0]), 0.0), [0.0, 0.0, 0.0])
    batch_b = fld.B(np.zeros((4, 3)), np.zeros(4))
    assert batch_b.shape == (4, 3) and not np.any(batch_b)


def test_schema_validates_parameters():
    from pydantic import ValidationError

    from latent_dirac.scene.loader import scene_from_mapping

    def scene(**overrides):
        element = {
            "type": "rotating_wall",
            "label": "rw",
            "multipole": 2,
            "amplitude_v_m": 500.0,
            "radius_m": 0.02,
            "frequency_hz": 1.0e6,
        }
        element.update(overrides)
        return scene_from_mapping(
            {
                "schema_version": 1,
                "name": "rw",
                "seed": 1,
                "solver": {"dt_s": 1e-11, "steps": 10},
                "source": {
                    "type": "positron_pair",
                    "label": "s",
                    "params": {
                        "primary_count": 100,
                        "yield_eplus_per_primary": 0.02,
                        "mean_energy_MeV": 1e-5,
                        "energy_spread_MeV": 1e-6,
                        "angular_rms_rad": 0.1,
                        "source_sigma_m": 1e-3,
                        "bunch_length_s": 1e-10,
                        "macro_particles": 8,
                    },
                },
                "elements": [element],
            }
        )

    scene()  # valid
    for bad in ({"amplitude_v_m": 0.0}, {"frequency_hz": 0.0}, {"radius_m": 0.0}, {"multipole": 3}):
        with pytest.raises(ValidationError):
            scene(**bad)


def test_runs_in_a_scene_and_deflects():
    from latent_dirac.scene.build import build_source, run_scene
    from latent_dirac.scene.loader import scene_from_mapping

    scene = scene_from_mapping(
        {
            "schema_version": 1,
            "name": "rw-run",
            "seed": 7,
            "solver": {"dt_s": 1e-11, "steps": 200},
            "source": {
                "type": "positron_pair",
                "label": "s",
                "params": {
                    "primary_count": 100,
                    "yield_eplus_per_primary": 0.02,
                    "mean_energy_MeV": 1e-5,
                    "energy_spread_MeV": 1e-6,
                    "angular_rms_rad": 0.05,
                    "source_sigma_m": 2e-3,
                    "bunch_length_s": 1e-10,
                    "macro_particles": 16,
                },
            },
            "elements": [
                {
                    "type": "rotating_wall",
                    "label": "rw",
                    "multipole": 2,
                    "amplitude_v_m": 5.0e4,
                    "radius_m": 0.02,
                    "frequency_hz": 1.0e7,
                },
            ],
        }
    )
    initial = build_source(scene).sample(np.random.default_rng(scene.seed))
    final = run_scene(scene).pipeline_result.final_cloud
    assert final.alive.all()
    # the transverse field imparted transverse momentum (momenta are ~1e-24
    # kg·m/s, far below np.allclose's default atol, so compare the change)
    change = np.max(np.abs(final.momentum_kg_m_s[:, :2] - initial.momentum_kg_m_s[:, :2]))
    assert change > 1e-26


@pytest.mark.parametrize("multipole", [1, 2])
def test_field_lines_are_rendered(multipole):
    # the 3D viz draws the rotating E pattern (snapshot); guards the
    # "0 bundles, no field visualization" gap
    from latent_dirac.scene.schema import RotatingWallElement
    from latent_dirac.viz.field_lines import element_field_line_bundles

    element = RotatingWallElement(
        type="rotating_wall",
        label="rw",
        multipole=multipole,
        amplitude_v_m=5.0e4,
        radius_m=0.02,
        frequency_hz=1.0e6,
    )
    field = RotatingWallField(multipole=multipole, amplitude_v_m=5.0e4, radius_m=0.02, frequency_hz=1.0e6)
    bundles = element_field_line_bundles(element, field, {"transverse_m": 0.01, "axial_m": 0.02})
    assert len(bundles) > 0 and all(kind == "E" for kind, _ in bundles)


def test_committed_demo_scene_runs():
    from pathlib import Path

    from latent_dirac.scene.build import run_scene
    from latent_dirac.scene.loader import load_scene

    demo = Path(__file__).resolve().parents[1] / "examples/scenes/rotating_wall_drive.yaml"
    result = run_scene(load_scene(demo))
    assert result.pipeline_result.final_cloud.alive.all()


@pytest.mark.parametrize("multipole", [1, 2])
def test_numpy_jax_parity(multipole):
    pytest.importorskip("jax")
    from latent_dirac.backends.jax_scene import run_scene_batched
    from latent_dirac.scene.build import run_scene
    from latent_dirac.scene.loader import scene_from_mapping

    scene = scene_from_mapping(
        {
            "schema_version": 1,
            "name": "rw-parity",
            "seed": 3,
            "solver": {"dt_s": 1e-11, "steps": 100},
            "source": {
                "type": "positron_pair",
                "label": "s",
                "params": {
                    "primary_count": 100,
                    "yield_eplus_per_primary": 0.02,
                    "mean_energy_MeV": 1e-5,
                    "energy_spread_MeV": 1e-6,
                    "angular_rms_rad": 0.05,
                    "source_sigma_m": 2e-3,
                    "bunch_length_s": 1e-10,
                    "macro_particles": 16,
                },
            },
            "elements": [
                {"type": "uniform_field", "label": "b", "B_vector_t": [0.0, 0.0, 1.0]},
                {
                    "type": "rotating_wall",
                    "label": "rw",
                    "multipole": multipole,
                    "amplitude_v_m": 1.0e4,
                    "radius_m": 0.02,
                    "frequency_hz": 1.0e7,
                },
            ],
        }
    )
    ref = run_scene(scene).pipeline_result.final_cloud
    batched = run_scene_batched(scene, overrides={})
    got = np.asarray(batched.position_m)[0]
    np.testing.assert_allclose(got, ref.position_m, rtol=1e-5, atol=1e-9)
