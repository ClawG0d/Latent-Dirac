"""Honest benchmark suite: measure, label fully, write docs/benchmarks.md.

Honesty discipline (AGENTS.md): numbers carry integrator, timestep,
particle count, batch size, fidelity tier, and the full hardware/stack
labels; no comparative wording — the table speaks. Official numbers
come only from the project's WSL2 GPU box.

    python tools/run_benchmarks.py --output docs/benchmarks.md

GPU cases are skipped (and marked absent) when no JAX GPU backend
exists; the committed document must come from a full run.
"""

from __future__ import annotations

import argparse
import platform
import statistics
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from latent_dirac.core.constants import ELECTRON_MASS_KG, ELEMENTARY_CHARGE_C  # noqa: E402
from latent_dirac.solvers.kernels import boris_step  # noqa: E402

REPEATS = 5
KERNEL_STEPS = 1000
KERNEL_DT_S = 2.0e-12
KERNEL_B_T = 0.5
FIDELITY_NOTE = "uniform-field transport, parameterized tier; relativistic Boris integrator"


def _median_time(run, sync=lambda result: None) -> float:
    run()  # warmup: compile + caches excluded from timing
    samples = []
    for _ in range(REPEATS):
        start = time.perf_counter()
        result = run()
        sync(result)
        samples.append(time.perf_counter() - start)
    return statistics.median(samples)


def _initial_arrays(count: int, rng: np.random.Generator):
    position = rng.normal(0.0, 1e-3, size=(count, 3))
    u = rng.normal(0.0, 2.0, size=(count, 3))  # dimensionless p/(mc), MeV-scale
    time_s = np.zeros(count)
    alive = np.ones(count, dtype=bool)
    e_field = np.zeros((count, 3))
    b_field = np.tile([0.0, 0.0, KERNEL_B_T], (count, 1))
    return position, u, time_s, alive, e_field, b_field


def kernel_case_numpy(count: int) -> float:
    rng = np.random.default_rng(1)
    position, u, time_s, alive, e_field, b_field = _initial_arrays(count, rng)

    def run():
        pos, uu, tt = position, u, time_s
        for _ in range(KERNEL_STEPS):
            pos, uu, tt = boris_step(
                pos, uu, tt, alive, dt_s=KERNEL_DT_S, charge_c=ELEMENTARY_CHARGE_C,
                mass_kg=ELECTRON_MASS_KG, e_field=e_field, b_field=b_field, xp=np,
            )
        return pos

    return _median_time(run)


def kernel_case_jax(count: int, dtype_name: str, device) -> float:
    import jax
    import jax.numpy as jnp

    rng = np.random.default_rng(1)
    arrays = _initial_arrays(count, rng)
    dtype = jnp.float32 if dtype_name == "float32" else jnp.float64
    with jax.default_device(device):
        position, u, time_s = (jnp.asarray(a, dtype=dtype) for a in arrays[:3])
        alive = jnp.asarray(arrays[3])
        e_field, b_field = (jnp.asarray(a, dtype=dtype) for a in arrays[4:])

        @jax.jit
        def program(pos, uu, tt):
            def step(carry, _):
                p, w, t = carry
                return boris_step(
                    p, w, t, alive, dt_s=KERNEL_DT_S, charge_c=ELEMENTARY_CHARGE_C,
                    mass_kg=ELECTRON_MASS_KG, e_field=e_field, b_field=b_field, xp=jnp,
                ), None

            (pos, uu, tt), _ = jax.lax.scan(step, (pos, uu, tt), None, length=KERNEL_STEPS)
            return pos

        return _median_time(lambda: program(position, u, time_s), sync=lambda r: r.block_until_ready())


def batch_case(batch_size: int) -> float:
    """Amortized per-configuration wall time for a hello-beamline sweep."""

    from latent_dirac.backends.jax_scene import BatchedSceneProgram
    from latent_dirac.scene.loader import load_scene

    scene = load_scene(PROJECT_ROOT / "examples/scenes/hello_beamline.yaml")
    program = BatchedSceneProgram(scene, override_keys=("hello-solenoid.b_tesla",))
    values = {"hello-solenoid.b_tesla": np.linspace(0.4, 1.2, batch_size)}
    return _median_time(lambda: program.run(values)) / batch_size


def environment_block() -> list[str]:
    lines = [f"- python: {platform.python_version()} on {platform.platform()}"]
    try:
        import jax
        import jaxlib

        lines.append(f"- jax {jax.__version__}, jaxlib {jaxlib.__version__}")
        for device in jax.devices():
            lines.append(f"- jax device: {device.device_kind} ({device.platform})")
    except ImportError:
        lines.append("- jax: not installed")
    try:
        smi = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=10,
        )
        if smi.returncode == 0 and smi.stdout.strip():
            lines.append(f"- nvidia-smi: {smi.stdout.strip()} (WSL2 passthrough)")
    except (OSError, subprocess.TimeoutExpired):
        pass
    lines.append(f"- numpy {np.__version__}")
    return lines


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="docs/benchmarks.md")
    parser.add_argument("--quick", action="store_true", help="tiny sizes (smoke only, do not commit)")
    args = parser.parse_args()

    counts = [1_000, 10_000] if args.quick else [10_000, 100_000, 1_000_000]
    numpy_counts = [1_000] if args.quick else [10_000, 100_000]
    batches = [1, 4] if args.quick else [1, 8, 64, 256]

    try:
        import jax

        gpu_devices = [d for d in jax.devices() if d.platform == "gpu"]
        cpu_device = jax.devices("cpu")[0]
    except (ImportError, RuntimeError):
        jax, gpu_devices, cpu_device = None, [], None

    kernel_rows = []
    for count in counts:
        row = {"count": count}
        if gpu_devices:
            row["gpu_fp32"] = kernel_case_jax(count, "float32", gpu_devices[0])
        if cpu_device is not None:
            row["cpu_fp32"] = kernel_case_jax(count, "float32", cpu_device)
        if count in numpy_counts:
            row["numpy_fp64"] = kernel_case_numpy(count)
        kernel_rows.append(row)

    batch_rows = []
    if gpu_devices:
        for batch in batches:
            batch_rows.append({"batch": batch, "per_config_s": batch_case(batch)})

    def fmt(row: dict, key: str) -> str:
        value = row.get(key)
        if value is None:
            return "—"
        rate = row["count"] * KERNEL_STEPS / value
        return f"{value * 1e3:.1f} ms ({rate:.2e} particle·steps/s)"

    lines = [
        "# Benchmarks",
        "",
        "Reproducible measurements of the transport kernel and the batched",
        "scene program. Every number carries its full label per the honesty",
        "discipline; regenerate with:",
        "",
        "```bash",
        "python tools/run_benchmarks.py --output docs/benchmarks.md",
        "```",
        "",
        "## Environment",
        "",
        *environment_block(),
        "",
        "## Boris kernel, uniform B field",
        "",
        f"{KERNEL_STEPS} steps of `boris_step` under `jax.lax.scan`/`jit`",
        f"(dt = {KERNEL_DT_S:g} s, B = {KERNEL_B_T} T, positron-mass particles;",
        f"{FIDELITY_NOTE}). Median of {REPEATS} runs after a warmup that",
        "excludes compilation; GPU timings synchronize with",
        "`block_until_ready`. The NumPy column is the float64 reference",
        "pipeline (per-step Python loop, the same pure kernel).",
        "",
        "| particles | fp32 GPU (jit+scan) | fp32 CPU (jit+scan) | fp64 NumPy loop |",
        "| --------- | ------------------- | ------------------- | --------------- |",
    ]
    for row in kernel_rows:
        lines.append(
            f"| {row['count']:,} | {fmt(row, 'gpu_fp32')} | {fmt(row, 'cpu_fp32')} | {fmt(row, 'numpy_fp64')} |"
        )

    if batch_rows:
        lines += [
            "",
            "## Batched scene sweep (hello_beamline)",
            "",
            "`BatchedSceneProgram` over `hello-solenoid.b_tesla`",
            "(thin-sheet solenoid + aperture, 100 steps, dt = 3e-12 s,",
            "64 macro-particles per configuration, parameterized tier),",
            "fp32 on the GPU device; amortized wall time per configuration,",
            f"median of {REPEATS} runs after warmup.",
            "",
            "| batch size | per-configuration time |",
            "| ---------- | ---------------------- |",
        ]
        for row in batch_rows:
            lines.append(f"| {row['batch']} | {row['per_config_s'] * 1e3:.2f} ms |")

    lines += [
        "",
        "Notes: correctness of the fp32 GPU lane against the float64 CPU",
        "reference is enforced separately by the tiered validation suite",
        "(`tests/test_gpu_float32_validation.py`); the jax pin and its",
        "rationale are documented in `docs/solver_backends.md`.",
        "",
    ]

    output = Path(args.output)
    output.write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
