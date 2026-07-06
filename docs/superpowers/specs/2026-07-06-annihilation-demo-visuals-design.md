# Annihilation demo visuals and thin-sheet demo rollout

Date: 2026-07-06. Status: accepted. Companion to the same-day
thin-sheet solenoid profile spec (the field capability); this spec is
the visual rollout.

## Motivation

The two-photon kinematics data recorded by `annihilation_plate` has
never been rendered: `render_scene_3d` draws neither the plate nor the
photon pairs, and the WebP generator shows only static gold rays that
all pop in at 55% of the animation regardless of when each positron
actually reaches the plate. Meanwhile every demo solenoid still runs
the hard-edge profile the thin-sheet model was built to replace.

## Scope guards (unchanged physics, unchanged safety scope)

- Visualization only: at-rest isotropic back-to-back unit vectors;
  511 keV stays a label; no energetics anywhere.
- The plate gains a species guard: it models e+ annihilation on
  matter, so a non-positron cloud is a scene-construction error
  (antiproton annihilation is engine physics — multi-pion final
  states must not be faked as two photons).
- No in-flight annihilation (cross-section is negligible; recorded
  project decision).

## Changes

1. `build.py::_annihilation_action` records two new event channels:
   `time_s` (the killed particle's clock at the plate) and
   `particle_id` — enough to animate each event at the moment its
   trail reaches the plate. Raises for non-positron clouds.
2. `viz/scene_3d.py`: the plate renders as two concentric circles at
   its z plane, and a gold photon-ray trace (two rays per event) is
   added when the run result carries annihilation events; hover text
   carries the kinematics-only fidelity note.
3. `tools/mpl3d.py`: `draw_photon_burst` — per-event progress in
   [0, 1] grows each ray pair from its vertex with a fading star
   flash at the vertex; `draw_scene_elements` draws the plate disc.
4. `tools/generate_scene_demo_webps.py`: the static ray block becomes
   time-resolved — each event's start frame is derived from the
   snapshot where that particle's recorded trail first crosses the
   plate plane (mapped through the reveal schedule), rays grow over
   the following frames.
5. Demo rollout: `positron_capture` (the README hero) gains a
   collector drift + annihilation plate after the end-station
   monitor — the accepted core now ends in a photon burst; the
   solenoid demos (`positron_capture`, `scene_tour`,
   `decay_emission`, `target_production_engine`, `hello_beamline`)
   switch to `profile: thin_sheet`. Demo titles and README captions
   and report blocks are regenerated to match (fidelity notes change
   from hard-edge to thin-sheet where switched).

## Validation

- Event channels: `time_s`/`particle_id` shapes match the event
  count; recorded ids are actually the killed particles; times equal
  the cloud clock at the kill.
- Species guard: an antiproton cloud through a plate raises with a
  message pointing at the engine route.
- Plotly: the plate contributes geometry segments; a scene with
  annihilation events renders a photon trace whose hover text
  carries the kinematics-only note; a scene without events renders
  no such trace.
- WebP path: `draw_photon_burst` handles zero-progress (draws
  nothing), partial and saturated progress.
- The switched demo scenes still load and run; the hero scene's
  accepted core annihilates at the collector (surviving weight 0,
  collector-plate ledger entry positive).
- Full suite green; WebP assets regenerated and a late frame
  visually inspected before committing.
