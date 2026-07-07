"""Tests for the composite_field scene element (field superposition).

Bundles two or more field models into one transport stage whose field is
their exact superposition (CompositeField). Motivating case: a rotating
wall superimposed on a Penning trap. NumPy backend (slice 1); the JAX
backend rejects it for now (slice 2). See the 2026-07-07 composite-field spec.
"""

from __future__ import annotations

import numpy as np
import pytest

from latent_dirac.scene.build import _base_field_for, build_source, run_scene
from latent_dirac.scene.loader import scene_from_mapping
from latent_dirac.scene.schema import Scene


def _composite_scene(fields=None, steps=50, extra_solver=None):
    fields = fields or [
        {"type": "penning_trap", "label": "trap", "v0_volt": -20.0, "d_m": 0.01, "b_tesla": 1.0},
        {
            "type": "rotating_wall",
            "label": "wall",
            "multipole": 2,
            "amplitude_v_m": 3.0e4,
            "radius_m": 0.02,
            "frequency_hz": 5.0e6,
        },
    ]
    return scene_from_mapping(
        {
            "schema_version": 1,
            "name": "composite",
            "seed": 5,
            "solver": {"dt_s": 1e-11, "steps": 4},
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
                {"type": "composite_field", "label": "combo", "fields": fields, "steps": steps},
                {"type": "monitor", "label": "m"},
            ],
        }
    )


def test_field_is_the_sum_of_components():
    scene = _composite_scene()
    combo = scene.elements[0]
    field = _base_field_for(combo)  # a CompositeField

    trap = _base_field_for(combo.fields[0])
    wall = _base_field_for(combo.fields[1])
    x = np.array([[0.003, -0.004, 0.001], [0.0, 0.006, -0.002]])
    for t in (0.0, 3.0e-8):
        ts = np.full(x.shape[0], t)
        np.testing.assert_allclose(field.E(x, ts), trap.E(x, ts) + wall.E(x, ts), rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(field.B(x, ts), trap.B(x, ts) + wall.B(x, ts), rtol=1e-12, atol=1e-12)


def test_composite_scene_runs():
    scene = _composite_scene()
    initial = build_source(scene).sample(np.random.default_rng(scene.seed))
    final = run_scene(scene).pipeline_result.final_cloud
    assert final.alive.all()
    # both the trap well and the rotating wall act -> transverse momentum changed
    change = np.max(np.abs(final.momentum_kg_m_s[:, :2] - initial.momentum_kg_m_s[:, :2]))
    assert change > 1e-26


def test_requires_at_least_two_fields():
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        _composite_scene(
            fields=[{"type": "penning_trap", "label": "t", "v0_volt": -20.0, "d_m": 0.01, "b_tesla": 1.0}]
        )


def test_nested_composite_is_rejected():
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        _composite_scene(
            fields=[
                {"type": "uniform_field", "label": "u", "B_vector_t": [0, 0, 1.0]},
                {
                    "type": "composite_field",
                    "label": "nested",
                    "fields": [
                        {"type": "uniform_field", "label": "a", "B_vector_t": [0, 0, 1.0]},
                        {"type": "uniform_field", "label": "b", "B_vector_t": [0, 0, 1.0]},
                    ],
                },
            ]
        )


def test_per_subfield_time_gate_applies_inside_the_composite():
    # a gated uniform E sub-field contributes 0 before its window, full after
    scene = _composite_scene(
        fields=[
            {"type": "uniform_field", "label": "steady", "B_vector_t": [0.0, 0.0, 1.0]},
            {
                "type": "uniform_field",
                "label": "pulsed",
                "E_vector_v_m": [1000.0, 0.0, 0.0],
                "t_on_s": 1.0e-8,
                "t_off_s": 2.0e-8,
            },
        ]
    )
    field = _base_field_for(scene.elements[0])
    x = np.array([[0.0, 0.0, 0.0]])
    before = field.E(x, np.array([0.0]))  # pulsed off -> only steady (E=0)
    during = field.E(x, np.array([1.5e-8]))  # pulsed on -> E = 1000 x
    np.testing.assert_allclose(before[0], [0.0, 0.0, 0.0], atol=1e-12)
    np.testing.assert_allclose(during[0], [1000.0, 0.0, 0.0], atol=1e-9)


def test_jax_backend_rejects_composite():
    pytest.importorskip("jax")
    from latent_dirac.backends.jax_scene import run_scene_batched

    with pytest.raises(ValueError, match="JAX backend"):
        run_scene_batched(_composite_scene(), overrides={})


def test_committed_demo_scene_runs():
    from pathlib import Path

    from latent_dirac.scene.loader import load_scene

    demo = Path(__file__).resolve().parents[1] / "examples/scenes/rotating_wall_in_trap.yaml"
    result = run_scene(load_scene(demo))
    assert result.pipeline_result.final_cloud.alive.any()


def test_isinstance_scene():
    assert isinstance(_composite_scene(), Scene)
