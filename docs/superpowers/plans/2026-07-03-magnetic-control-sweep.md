# Magnetic Control Sweep Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a README-facing magnetic field sweep demo for matched positron/electron charge-sign separation with aperture loss diagnostics.

**Architecture:** Add a standalone example module that computes sweep diagnostics without Pillow or visualization dependencies. Extend the existing Pillow README animation generator to include a fourth animated WebP using the same physics primitives. Update README and tests so the text demo, rendered asset list, and generated animation stay in sync.

**Tech Stack:** Python, numpy, pytest, existing Latent Dirac core modules, optional Pillow only inside `tools/generate_demo_webp.py`.

## Global Constraints

- Do not integrate Genesis World.
- Do not add UI or 3D world visualization.
- Core simulation modules must not import visualization packages.
- Use SI units internally.
- Keep optional external integrations behind adapters.
- Do not model particle collisions, annihilation physics, material interactions, target engineering, release physics, facility controls, operational optimization recipes, full shower physics, material activation, or shielding design.
- Keep the demo focused on charge-sign transport, aperture acceptance, loss accounting, and accepted-yield diagnostics.

---

### Task 1: Magnetic Sweep Text Demo

**Files:**
- Create: `examples/magnetic_control_sweep_demo.py`
- Create: `tests/test_magnetic_control_sweep_demo.py`

**Interfaces:**
- Consumes: `examples.charge_sign_splitter_demo.make_initial_pair(particle_count: int, seed: int) -> tuple[ParticleCloud, ParticleCloud]`
- Produces: `run_sweep(field_values_t: Sequence[float] = DEFAULT_FIELD_VALUES_T, particle_count: int = 96, aperture_radius_m: float = 0.035, dt_s: float = 2.0e-12, steps: int = 80, seed: int = 2031) -> list[dict[str, float]]`
- Produces: `format_report(results: list[dict[str, float]], *, aperture_radius_m: float, particle_count: int, dt_s: float, steps: int) -> str`
- Produces: `run_report(field_values_t: Sequence[float] = DEFAULT_FIELD_VALUES_T, particle_count: int = 96, aperture_radius_m: float = 0.035, dt_s: float = 2.0e-12, steps: int = 80) -> str`

- [ ] **Step 1: Write the failing test**

```python
def test_magnetic_control_sweep_separation_and_losses_are_reported():
    from examples.magnetic_control_sweep_demo import run_report, run_sweep

    results = run_sweep(field_values_t=[0.0, 0.3, 0.6], particle_count=36, aperture_radius_m=0.05)

    assert len(results) == 3
    assert results[0]["field_by_tesla"] == 0.0
    assert results[-1]["field_by_tesla"] == 0.6
    assert results[-1]["mean_separation_m"] > results[0]["mean_separation_m"]
    for row in results:
        assert 0.0 <= row["accepted_fraction"] <= 1.0
        assert 0.0 <= row["loss_fraction"] <= 1.0
        assert row["accepted_fraction"] + row["loss_fraction"] == pytest.approx(1.0)

    report = run_report(field_values_t=[0.0, 0.3, 0.6], particle_count=36, aperture_radius_m=0.05)

    assert "Magnetic control sweep demo" in report
    assert "Magnetic field status:" in report
    assert "Aperture status:" in report
    assert "By [T]" in report
    assert "accepted fraction" in report
    assert "loss fraction" in report
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_magnetic_control_sweep_demo.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'examples.magnetic_control_sweep_demo'`.

- [ ] **Step 3: Write minimal implementation**

Implement `examples/magnetic_control_sweep_demo.py` with these public functions and behavior:

```python
DEFAULT_FIELD_VALUES_T = tuple(float(value) for value in np.linspace(0.0, 0.6, 7))
```

`run_sweep` validates that the field list is nonempty, particle count is
positive, aperture radius is positive, `dt_s` is positive, and `steps` is
positive. It creates matched initial positron/electron clouds once per field
value using the same seed, propagates them through `UniformField(B_vector_t=[0,
By, 0])`, computes the mean x position for each species, computes absolute
mean separation, marks particles accepted when `abs(x) <= aperture_radius_m`,
and returns one dictionary per field value with keys:

```python
{
    "field_by_tesla": float,
    "positron_mean_x_m": float,
    "electron_mean_x_m": float,
    "mean_separation_m": float,
    "accepted_fraction": float,
    "loss_fraction": float,
    "accepted_particles": float,
    "lost_particles": float,
}
```

Use these signatures:

```python
def run_sweep(
    field_values_t: Sequence[float] = DEFAULT_FIELD_VALUES_T,
    particle_count: int = 96,
    aperture_radius_m: float = 0.035,
    dt_s: float = 2.0e-12,
    steps: int = 80,
    seed: int = 2031,
) -> list[dict[str, float]]:


def format_report(
    results: list[dict[str, float]],
    *,
    aperture_radius_m: float,
    particle_count: int,
    dt_s: float,
    steps: int,
) -> str:


def run_report(
    field_values_t: Sequence[float] = DEFAULT_FIELD_VALUES_T,
    particle_count: int = 96,
    aperture_radius_m: float = 0.035,
    dt_s: float = 2.0e-12,
    steps: int = 80,
) -> str:
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_magnetic_control_sweep_demo.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add examples/magnetic_control_sweep_demo.py tests/test_magnetic_control_sweep_demo.py
git commit -m "feat: add magnetic control sweep demo"
```

### Task 2: README Demo Section

**Files:**
- Modify: `README.md`
- Modify: `tests/test_demo_webp_assets.py`

**Interfaces:**
- Consumes: `examples.magnetic_control_sweep_demo.run_report(field_values_t: Sequence[float] = DEFAULT_FIELD_VALUES_T, particle_count: int = 96, aperture_radius_m: float = 0.035, dt_s: float = 2.0e-12, steps: int = 80) -> str`
- Produces: README reference to `assets/demos/magnetic_control_sweep.webp`

- [ ] **Step 1: Write the failing test**

```python
def test_readme_references_demo_webp_assets():
    readme = Path("README.md").read_text(encoding="utf-8")

    assert "assets/demos/charge_sign_splitter.webp" in readme
    assert "assets/demos/positron_capture.webp" in readme
    assert "assets/demos/antiproton_transport.webp" in readme
    assert "assets/demos/magnetic_control_sweep.webp" in readme
    assert "Magnetic Control Sweep" in readme
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_demo_webp_assets.py::test_readme_references_demo_webp_assets -q`
Expected: FAIL because README does not reference `assets/demos/magnetic_control_sweep.webp`.

- [ ] **Step 3: Update README**

Add a new demo section before optional report figures:

```markdown
### Demo 4: Magnetic Control Sweep

This demo scans a uniform transverse magnetic field over matched positron and
electron clouds. It shows the charge-sign separation trend and reports fixed
aperture acceptance and loss diagnostics.

![Animated magnetic control sweep demo](assets/demos/magnetic_control_sweep.webp)
```

Renumber the optional report figures section to Demo 5.

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_demo_webp_assets.py::test_readme_references_demo_webp_assets -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add README.md tests/test_demo_webp_assets.py
git commit -m "docs: add magnetic control sweep demo"
```

### Task 3: Animated WebP Asset

**Files:**
- Modify: `tools/generate_demo_webp.py`
- Create: `assets/demos/magnetic_control_sweep.webp`
- Modify: `tests/test_demo_webp_assets.py`

**Interfaces:**
- Consumes: `examples.magnetic_control_sweep_demo.DEFAULT_FIELD_VALUES_T`
- Consumes: `examples.magnetic_control_sweep_demo.run_sweep(field_values_t: Sequence[float] = DEFAULT_FIELD_VALUES_T, particle_count: int = 96, aperture_radius_m: float = 0.035, dt_s: float = 2.0e-12, steps: int = 80, seed: int = 2031) -> list[dict[str, float]]`
- Produces: `DEMO_WEBP_FILES` containing four file names

- [ ] **Step 1: Write the failing test**

```python
def test_demo_webp_generator_creates_animated_webp_files(tmp_path):
    image_module = pytest.importorskip("PIL.Image")
    from tools.generate_demo_webp import DEMO_WEBP_FILES, generate_demo_webps

    assert "magnetic_control_sweep.webp" in DEMO_WEBP_FILES

    generated = generate_demo_webps(tmp_path, frame_count=4, particle_count=16)

    assert set(generated) == set(DEMO_WEBP_FILES)
    for name, path in generated.items():
        assert path == tmp_path / name
        assert path.suffix == ".webp"
        assert path.stat().st_size > 0
        with image_module.open(path) as image:
            assert image.format == "WEBP"
            assert getattr(image, "is_animated", False)
            assert image.n_frames == 4
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_demo_webp_assets.py::test_demo_webp_generator_creates_animated_webp_files -q`
Expected: FAIL because `DEMO_WEBP_FILES` does not contain `magnetic_control_sweep.webp`.

- [ ] **Step 3: Extend generator**

Update `DEMO_WEBP_FILES`, add `_make_magnetic_control_sweep_frames`, add an output entry, and save the generated frames. The new frames draw matched positron/electron tracks, aperture guide lines, a magnetic field status panel, separation, accepted fraction, and loss fraction.

- [ ] **Step 4: Generate assets**

Run: `.venv/bin/python tools/generate_demo_webp.py`
Expected output includes `assets/demos/magnetic_control_sweep.webp`.

- [ ] **Step 5: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_demo_webp_assets.py -q`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add tools/generate_demo_webp.py tests/test_demo_webp_assets.py assets/demos/magnetic_control_sweep.webp assets/demos/charge_sign_splitter.webp assets/demos/positron_capture.webp assets/demos/antiproton_transport.webp
git commit -m "docs: add magnetic control sweep animation"
```

### Task 4: Full Verification and Push

**Files:**
- Verify all changed files from Tasks 1-3.

**Interfaces:**
- Consumes: all implemented demo, README, test, and WebP assets.
- Produces: pushed `master` branch with the plan, implementation, docs, tests, and assets.

- [ ] **Step 1: Run full tests**

Run: `.venv/bin/python -m pytest -q`
Expected: all tests pass.

- [ ] **Step 2: Run compile check**

Run: `.venv/bin/python -m compileall -q latent_dirac examples tools`
Expected: exit code 0.

- [ ] **Step 3: Run whitespace check**

Run: `git diff --check`
Expected: exit code 0.

- [ ] **Step 4: Verify WebP assets**

Run a Pillow check that every file in `assets/demos/*.webp` is animated, has nonzero size, and has at least two frames.

- [ ] **Step 5: Visual inspect**

Open `assets/demos/magnetic_control_sweep.webp` and confirm the field status, separation, accepted fraction, loss fraction, and aperture boundaries are visible without text overflow.

- [ ] **Step 6: Push**

```bash
git push origin master
```
