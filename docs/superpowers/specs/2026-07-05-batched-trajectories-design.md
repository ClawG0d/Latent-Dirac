# Batched Trajectory Recording and Flagship Sweep Demo (Spec 3e)

## Objective

The Phase 3 flagship demo: one JAX launch, a whole family of beamlines,
rendered as 3D trajectories colored by configuration. Requires trajectory
recording inside the batched program.

## Design

- `BatchedSceneProgram(..., record_stride: int | None = None)`: a
  compile-time option (it changes the traced program). Transport scans
  emit per-step positions; the host concatenates transports and applies
  the stride. `BatchedSceneResult` gains
  `trajectories: np.ndarray | None` shaped `(B, S, N, 3)` where `S` is
  the strided snapshot count (initial position included).
- Memory is `B x T x N x 3` doubles at emission before striding — fine
  for demo scales, documented honestly; true streaming for extreme scales
  (the 1000 x 1e4 x 1e3 case) remains a later design, per the roadmap.
- `run_scene_batched(..., record_stride=...)` passes through.
- Demo asset `batched_sweep_3d.webp`: a positron line under a By sweep,
  24 configurations executed in a single `program.run`, trails colored by
  configuration on a color ramp. The title carries the honest framing
  ("one launch, 24 configurations") with the physics settings; no speed
  claims.

## Validation

- Parity: for B=1 and stride s, the recorded snapshots equal the NumPy
  `run_scene(record_trajectories=True)` per-stage histories sampled at
  the same steps (x64).
- Shapes: (B, S, N, 3) with S = floor(total_steps / s) + 1; acceptance
  stages contribute no snapshots.
- `record_stride` validation (>= 1); `trajectories is None` when off.
- The demo generator produces the animated asset and the README
  references it (extends the existing asset tests).

## Non-Goals

- streaming/chunked recording for extreme scales
- monitor snapshots in the batched backend (still pending)
- performance numbers
