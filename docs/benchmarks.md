# Benchmarks

Reproducible measurements of the transport kernel and the batched
scene program. Every number carries its full label per the honesty
discipline; regenerate with:

```bash
python tools/run_benchmarks.py --output docs/benchmarks.md
```

## Environment

- python: 3.12.3 on Linux-6.18.33.2-microsoft-standard-WSL2-x86_64-with-glibc2.39
- jax 0.9.1, jaxlib 0.9.1
- jax device: NVIDIA GeForce RTX 5070 Ti (gpu)
- nvidia-smi: NVIDIA GeForce RTX 5070 Ti, 591.86 (WSL2 passthrough)
- numpy 2.5.1

## Boris kernel, uniform B field

1000 steps of `boris_step` under `jax.lax.scan`/`jit`
(dt = 2e-12 s, B = 0.5 T, positron-mass particles;
uniform-field transport, parameterized tier; relativistic Boris integrator). Median of 5 runs after a warmup that
excludes compilation; GPU timings synchronize with
`block_until_ready`. The NumPy column is the float64 reference
pipeline (per-step Python loop, the same pure kernel).

| particles | fp32 GPU (jit+scan) | fp32 CPU (jit+scan) | fp64 NumPy loop |
| --------- | ------------------- | ------------------- | --------------- |
| 10,000 | 16.2 ms (6.16e+08 particle·steps/s) | 23.9 ms (4.18e+08 particle·steps/s) | 858.2 ms (1.17e+07 particle·steps/s) |
| 100,000 | 22.3 ms (4.49e+09 particle·steps/s) | 240.5 ms (4.16e+08 particle·steps/s) | 11280.4 ms (8.86e+06 particle·steps/s) |
| 1,000,000 | 130.1 ms (7.69e+09 particle·steps/s) | 3598.8 ms (2.78e+08 particle·steps/s) | — |

## Batched scene sweep (hello_beamline)

`BatchedSceneProgram` over `hello-solenoid.b_tesla`
(thin-sheet solenoid + aperture, 100 steps, dt = 3e-12 s,
64 macro-particles per configuration, parameterized tier),
fp32 on the GPU device; amortized wall time per configuration,
median of 5 runs after warmup.

| batch size | per-configuration time |
| ---------- | ---------------------- |
| 1 | 6.40 ms |
| 8 | 0.69 ms |
| 64 | 0.10 ms |
| 256 | 0.04 ms |

Notes: correctness of the fp32 GPU lane against the float64 CPU
reference is enforced separately by the tiered validation suite
(`tests/test_gpu_float32_validation.py`); the jax pin and its
rationale are documented in `docs/solver_backends.md`.
