"""Honesty-discipline checks for outward-facing project documentation.

These tests enforce the positioning rules adopted in
docs/superpowers/specs/2026-07-05-platform-positioning-and-roadmap-design.md:

- the README must separate design intent from current status
- source-model fidelity tiers must be declared in the README
- every safety-scope exclusion must survive verbatim in AGENTS.md; the
  README links to docs/safety_scope.md instead of mirroring the list
  (owner decision, 2026-07-05 — see the engine positioning spec
  addendum)
- comparative performance wording requires a benchmark reference in the
  same document
"""

from __future__ import annotations

import re
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
README_PATH = PROJECT_ROOT / "README.md"
AGENTS_PATH = PROJECT_ROOT / "AGENTS.md"
SAFETY_SCOPE_PATH = PROJECT_ROOT / "docs" / "safety_scope.md"
ALLOWED_ADAPTERS = {"geant4", "root", "xsuite"}

# Canonical safety-scope exclusions for the Geant4-engine era, adopted in
# docs/superpowers/specs/2026-07-05-geant4-engine-positioning-design.md.
# Red lines sit at the application layer: shower physics and energy
# deposition are delegated to the vendored vanilla Geant4 engine as
# diagnostics, while weaponization, energetic-release applications, and
# facility write-back stay excluded. These strings must appear verbatim in
# docs/safety_scope.md and AGENTS.md (the README links to the canonical
# file instead of mirroring it).
EXPECTED_EXCLUSIONS = (
    "weaponization scenarios",
    "energetic-release applications (antimatter as an energy source or destructive payload in any form)",
    "real facility control systems",
    "detailed accelerator target engineering (thermal, mechanical, and materials design of production "
    "targets)",
    "high-yield operational recipes",
    "in-house shower physics (electromagnetic and hadronic showers are delegated to the vendored vanilla "
    "Geant4 engine; the Python core does not implement them)",
    "annihilation energetics as a figure of merit (energy deposition is in scope only as an "
    "engine-computed diagnostic; the Python core models annihilation only as a loss endpoint with "
    "kinematic two-photon emission for visualization)",
    "material activation",
    "radiation shielding design",
)

FIDELITY_TIERS = (
    "placeholder",
    "parameterized",
    "surrogate",
    "table-based",
    "externally calibrated",
)

# Comparative performance wording is only allowed next to a reproducible
# benchmark reference. Scope: outward-facing docs (README + top-level docs/),
# not internal specs/plans that record external research notes.
PERFORMANCE_CLAIM_PATTERN = re.compile(
    r"(?i)(\d+(?:\.\d+)?\s*[x×]\s*faster|faster\s+than|fastest|blazing(?:ly)?\s+fast)"
)


def outward_facing_docs():
    yield README_PATH
    yield from sorted((PROJECT_ROOT / "docs").glob("*.md"))


def safety_scope_exclusions() -> list[str]:
    lines = SAFETY_SCOPE_PATH.read_text(encoding="utf-8").splitlines()
    return [line.removeprefix("- ").strip() for line in lines if line.startswith("- ")]


def test_readme_separates_design_intent_from_current_status():
    readme = README_PATH.read_text(encoding="utf-8")

    assert "## Current Status" in readme
    assert "Implemented:" in readme
    assert "Not implemented yet:" in readme


def test_readme_declares_fidelity_tiers():
    readme = README_PATH.read_text(encoding="utf-8").lower()

    for tier in FIDELITY_TIERS:
        assert tier in readme, f"README must declare the fidelity tier: {tier}"


def test_safety_scope_is_the_canonical_exclusion_list():
    assert tuple(safety_scope_exclusions()) == EXPECTED_EXCLUSIONS, (
        "docs/safety_scope.md exclusions drifted from the canonical list; "
        "safety-scope changes require a positioning spec and a coordinated "
        "update of docs/safety_scope.md, AGENTS.md, and this test"
    )


def test_readme_links_to_the_canonical_safety_scope():
    # The README no longer mirrors the exclusion list (owner decision,
    # 2026-07-05); it must at least keep the canonical file one click away.
    readme = README_PATH.read_text(encoding="utf-8")

    assert "docs/safety_scope.md" in readme


def test_safety_scope_exclusions_survive_in_agents_md():
    agents = AGENTS_PATH.read_text(encoding="utf-8").lower()
    missing = [item for item in safety_scope_exclusions() if item.lower() not in agents]

    assert missing == [], f"AGENTS.md safety scope is missing exclusions: {missing}"


def test_no_unsubstantiated_performance_claims():
    offenders = []
    for path in outward_facing_docs():
        text = path.read_text(encoding="utf-8")
        match = PERFORMANCE_CLAIM_PATTERN.search(text)
        if match and "benchmark" not in text.lower():
            offenders.append(
                f"{path.relative_to(PROJECT_ROOT)} claims {match.group(0)!r} without referencing a benchmark"
            )

    assert offenders == []


def test_only_placeholder_adapters_are_present():
    adapter_dirs = {
        path.name
        for path in (PROJECT_ROOT / "latent_dirac" / "adapters").iterdir()
        if path.is_dir() and path.name != "__pycache__"
    }

    assert adapter_dirs == ALLOWED_ADAPTERS
