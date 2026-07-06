# Interactive 3D viewer design (Plotly-first, slice 1)

Date: 2026-07-06
Status: adopted

## Decision

Add an **animated** interactive 3D scene viewer on the existing Plotly
backend, per the roadmap's "interactive 3D viewer (plotly first, then
web)". Slice 1 turns the static `render_scene_3d` output into a
play/pause + scrub animation of the recorded cloud traversing the
beamline: you watch the particle cloud move through the elements, and
lost particles freeze in place at the step they were killed — the loss
ledger made visible in motion.

This is a first-party viz increment in `latent_dirac/viz/scene_3d.py`
(Mac-owned lane). It reuses the element wireframes and the combined
trajectory array already built for the static view; no new dependency
beyond the existing optional `viz` (Plotly) extra. It is self-contained
HTML — no server, no web framework. A web/three.js viewer and
parameter-sweep controls are later slices.

## API

New function alongside the unchanged `render_scene_3d`:

- `render_scene_animation(scene, run_result, max_particles=64,
  trail=True) -> plotly Figure`

Requires `run_result` to carry recorded trajectories
(`run_scene(..., record_trajectories=True)`); otherwise raises a clear
`ValueError` naming that requirement (the static renderer degrades
gracefully without trajectories, but an animation with nothing to
animate is a user error, not a silent empty figure).

## Structure (standard Plotly animation pattern)

The combined trajectory is `(S, N, 3)` — S recorded steps, N particles.
Let `count = min(N, max_particles)`.

Base `figure.data`:
1. element wireframes (static across all frames; reuse `_element_segments`
   / `_wire_trace` / `_fidelity_label`).
2. faint full trajectory lines for orientation (reuse `_trajectory_trace`
   at low opacity) when `trail=True`.
3. the moving cloud: a `Scatter3d` marker trace at step 0
   (`combined[0, :count]`). This is the only animated trace.

`figure.frames`: one frame per step `s`, each supplying just the moving
cloud trace (`combined[s, :count]`), targeted by trace index so the
static traces persist.

Layout controls:
- `updatemenus`: Play and Pause buttons (Plotly `animate` method) with a
  per-frame duration.
- `sliders`: one step slider labeled by step index so the user can
  scrub; slider steps reference the frame names.
- axis titles and title reused from the static renderer.

Frame count equals `S`; `record_stride` on the batched program (or fewer
solver steps) is the knob for keeping S — and the HTML size — modest.
Dead particles are frozen in the recorded positions, so the moving cloud
naturally shows them stopping at their loss point (no per-step alive
mask needed).

## Tests (TDD; require plotly via importorskip)

1. The figure has `frames`, and `len(frames) == S` (the combined
   trajectory step count) for a small recorded run.
2. The moving cloud trace in each frame has `count` points
   (`min(N, max_particles)`).
3. Layout carries animation controls: `updatemenus` with a Play button
   and a `sliders` entry.
4. `trail=True` adds the faint full-path trace; `trail=False` omits it.
5. Element wireframes are present as static traces (a scene with a
   solenoid yields its wire trace, unchanged from the static renderer).
6. No recorded trajectories → `ValueError` naming `record_trajectories`.
7. `max_particles<=0` → `ValueError` (matching the static renderer).
8. The figure writes to self-contained interactive HTML
   (`write_html`) containing the plotly runtime and the frames.

Structural tests only; visual quality (smoothness, framing) is checked
by opening the HTML, like the WebP demo assets.

## Honesty / scope

No physics change; the animation replays recorded positions exactly.
Fidelity labels ride on the element hover text as in the static view.
No performance wording. Later slices (web viewer, sweep-parameter
sliders that recompute) are roadmap continuations, noted not built.
