# Scene Schema

A scene is a declarative, serializable description of a beamline pipeline:
one source, one solver, and an ordered list of elements. YAML is the primary
hand-written format; JSON is equally supported. Both parse into the same
validated model via `latent_dirac.scene.loader.load_scene`.

```yaml
schema_version: 1
name: capture-line
seed: 2026
source:
  type: positron_pair        # positron_pair | beta_plus | antiproton_surrogate
  label: pair-source
  params: { primary_count: 10000, yield_eplus_per_primary: 0.02, mean_energy_MeV: 3.0,
            energy_spread_MeV: 0.4, angular_rms_rad: 0.03, source_sigma_m: 0.001,
            bunch_length_s: 1.0e-12, macro_particles: 512 }
solver:
  type: relativistic_boris
  dt_s: 2.0e-12
  steps: 100
elements:
  - { type: solenoid, label: capture-solenoid, b_tesla: 0.8, radius_m: 0.05, length_m: 0.5 }
  - { type: drift, label: gap-1, steps: 20 }
  - { type: aperture, label: collimator, radius_m: 0.04, z_m: 0.06 }
  - { type: momentum_window, label: momentum-cut, p_min_gev_c: 0.001, p_max_gev_c: 0.020 }
  - { type: monitor, label: end-station }
```

## Rules

- `schema_version` is required and must equal 1. Pre-1.0, compatibility is
  only guaranteed within a minor release series.
- Every element and the source carry a required `label`, unique across the
  scene. Labels become pipeline stage names, so loss accounting is anchored
  to the scene description.
- Validation is fail-fast: unknown element types, unknown keys, missing or
  duplicate labels, and version mismatches are rejected at load time.
- Element vocabulary: `uniform_field`, `solenoid`, `dipole`, `quadrupole`
  (transport through the corresponding field model, optional per-element
  `steps` override), `drift` (zero-field transport), `aperture`,
  `momentum_window` (momenta in GeV/c), `monitor` (cloud snapshot, no
  physics).
- Batch convention: every numeric parameter must remain liftable into a
  batch-dimension array. This keeps the schema compatible with the Phase 3
  batched (vmap) execution plan.

## Running and rendering

```python
from latent_dirac.scene.loader import load_scene
from latent_dirac.scene.build import run_scene

scene = load_scene("capture-line.yaml")
result = run_scene(scene, record_trajectories=True)
print(result.pipeline_result.stage_results)
print(result.monitors["end-station"])
```

With the `viz` extra installed, render the scene and its recorded
trajectories in 3D. Element hover text carries the fidelity tier of each
model, so the rendered scene states its approximation level explicitly:

```python
from latent_dirac.viz.scene_3d import render_scene_3d

figure = render_scene_3d(scene, result)
figure.write_html("capture-line.html")
```

The scene core does not import any visualization package; rendering lives
behind the optional `viz` extra.
