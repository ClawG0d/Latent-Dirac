# Geant4 engine positioning design

Date: 2026-07-05
Status: adopted (M0 shipped in this change; M1'+ are roadmap milestones)

## Decision

Latent Dirac's positioning is upgraded from "lightweight core with future
adapter calibration" to **an antimatter-factory simulation platform with a
built-in Geant4 engine**. The engine form, after evaluating alternatives
against the Geant4 v11.4.2 source tree, is:

**vanilla Geant4, vendored in-repo, plus a companion acceleration library,
plus a controlled patch protocol.**

- The complete unmodified Geant4 v11.4.2 source tree is vendored at
  `geant4-v11.4.2/` (commit `vendor:`-prefixed, byte-identical to the
  upstream release, enforced by `.gitattributes -text`).
- Algorithm work happens first in a companion acceleration library hooked
  in through Geant4's fast-simulation interface — never by editing the
  vendored tree.
- Modifications to Geant4 itself, when truly unavoidable, go through the
  patch protocol below. Until that protocol's infrastructure exists, the
  vendored tree is frozen read-only.
- The JAX stack is unaffected: it remains the throughput and
  differentiability substrate. Geant4 supplies fidelity (particle-matter
  physics) and the calibration anchor; JAX supplies the batched,
  differentiable design-iteration loop. Neither replaces the other.

## Three tracks

1. **Engine baseline (vendored tree + build recipes).** `recipes/` (M1')
   adds a minimal-physics build configuration: no visualization, UI, or
   analysis category libraries; datasets restricted to what the selected
   physics list needs. Output: reproducible engine binaries for the
   adapter.
2. **Companion acceleration library (`engine/`, M4).** First-party C++
   living beside the vendored tree, attached via
   `G4VFastSimulationModel` (`IsApplicable` / `ModelTrigger` / `DoIt`),
   `G4Region` envelopes, and `G4FastSimulationPhysics`. This is where
   performance and algorithm R&D happens (the AdePT/Celeritas pattern).
   Performance numbers must cite an open benchmark against the vanilla
   engine on labeled hardware.
3. **Calibration loop (M2/M3).** The adapter becomes real: scene → GDML
   geometry exchange plus subprocess/macro driving (the official Python
   bindings left the Geant4 tree in 11.1, so no in-process binding is
   assumed). Yield tables generated offline feed `table_based` sources;
   surrogate source parameters fitted to engine output graduate to
   `externally calibrated`.

## Vendored-tree rules

Recorded normatively in AGENTS.md; summarized here with rationale:

- **Read-only vanilla baseline.** Never edit files inside
  `geant4-v11.4.2/`. In-tree edits silently create a hard fork — the
  failure mode GeantV documented (multi-year rewrite, ~1.5–2x measured
  gain, project wound down) and the reason the collaboration's own GPU
  efforts (AdePT, Celeritas) chose companion libraries over forks.
- **Tooling exclusions are load-bearing.** ruff `extend-exclude`, pytest
  `testpaths`, and `MANIFEST.in prune` keep lint, tests, and the Python
  distribution clear of the 17k-file tree. The pip package stays
  lightweight; the engine ships separately.
- **Patch protocol (reserved, not yet active).** When a change cannot be
  expressed through official hooks: one diff file per change under
  `patches/`, applied at build time; at most 5 patches live at any time,
  each ≤500 lines; each carries regression evidence against the vanilla
  baseline in `validation/`; anything upstreamable is submitted to the
  collaboration and deleted here once accepted. Exceeding the budget is
  an architecture signal — move the work to the companion library or
  upstream it.

## Naming, attribution, licensing

- The tree is redistributed under the Geant4 Software License (text
  inside the vendored tree); the root `NOTICE` carries the required
  collaboration attribution and must stay intact.
- Public wording: "vanilla Geant4 v11.4.2" (plus the published patch
  list once patches exist). The Geant4 name must never be used for
  endorsement or promotion — that requires written permission from the
  collaboration. Modified builds must be described distinguishably.
- Published patches are deemed licensed back to collaboration members
  under the license's terms, and Geant4 modifications cannot enter
  patent applications — consider this before adding any patch.

## Validation discipline

Every engine-derived result carries a four-tuple: **Geant4 version,
physics list, dataset versions, patch list** (empty while frozen).
Results from a patched engine must not be called "Geant4 results".
For antiproton production studies the reference physics list is
FTFP_BERT (FTF strings above ~3 GeV, Bertini cascade below;
`G4FTFAnnihilation` covers antiproton annihilation); physics-list choice
is part of the result provenance, not a hidden default.

## Safety scope rewrite

With shower physics and energy deposition entering scope as
engine-computed diagnostics, the red lines move from the capability
layer to the application layer. The canonical exclusion list lives in
`docs/safety_scope.md` and is pinned verbatim by
`tests/test_project_positioning.py` (now also asserting the AGENTS.md
copy). Kept excluded: weaponization scenarios, energetic-release
applications, real facility control systems, detailed accelerator target
engineering, high-yield operational recipes, material activation,
radiation shielding design. Reworded: in-house shower physics (delegated
to the engine) and annihilation energetics as a figure of merit
(deposition is a diagnostic only). The digital twin stays offline-only.

## Alternatives considered and rejected

- **Hard fork of Geant4 internals** — loses the validation pedigree that
  is the entire reason to use Geant4; GeantV precedent; annual-release
  rebase treadmill.
- **Extracting only the needed subsets** — the dependency graph converges
  (FTF → fragmentation → nuclei → precompound/de-excitation → particles/
  materials/global; a target shower additionally needs the kernel,
  geometry, and EM); genuinely droppable categories are ~11% of the
  tree; "subset of Geant4" weakens the validation claim; selective
  category linking at build time achieves the same footprint without a
  fork-by-subtraction.
- **Separate engine repository with build-time tarball fetch** — the
  recommended-by-review option; declined by the owner in favor of
  in-repo vendoring ("built-in" positioning, single tree). The
  containment measures above (byte-identical vendoring, read-only rules,
  tooling exclusions, patch budget) port that architecture's safety
  properties into the monorepo form.

## Source-research basis (Geant4 v11.4.2, read 2026-07-05)

- Fast-simulation hook chain verified end-to-end in source (model
  trigger → manager → process → track diversion and return) — the
  companion library needs no core edits.
- GDML read/write in `source/persistency/gdml` — the geometry exchange
  path for the adapter.
- Category libraries are the official modularity unit; 11.4 split
  `G4processes` into hadronic + core, easing minimal builds.
- License terms, naming restriction, and attribution sentence taken from
  the vendored `LICENSE`; Python bindings externalized since 11.1.

## Milestones

- **M0 (this change):** vendored baseline committed; tooling and license
  compliance; AGENTS/CLAUDE rules; safety-scope rewrite with pinned
  canonical list; roadmap update.
- **M1':** `recipes/` minimal-physics engine build (CI outside the
  Python matrix).
- **M2:** adapter made real (GDML + subprocess);
  `test_only_placeholder_adapters_are_present` is deliberately flipped
  in the same change.
- **M3:** yield-table pipeline and `table_based` antiproton source; the
  Chain 2 demo's drawn-annotation target becomes an honest engine-backed
  source.
- **M4:** companion acceleration library R&D (EM first), benchmarked
  against vanilla.

## Addendum (2026-07-05): verified vendoring fidelity, one known deviation

A full byte-level comparison against the upstream v11.4.2 tag (GitHub
mirror archive) confirmed all 17,643 regular files identical, with no
missing or extra files. One known deviation: upstream `CHANGELOG` is a
symlink to `ReleaseNotes`; the vendored tree stores it as a regular
file containing the target path (zip-archive semantics — the tree was
imported from a zip archive, which cannot represent symlinks). Kept
deliberately: a Windows-native checkout of this repository is
maintained, where git symlinks degrade silently without developer-mode
configuration. The deviation is content-preserving; `NOTICE` carries
the same clarification. Do not "fix" it back to a symlink.
