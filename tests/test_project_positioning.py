"""Honesty-discipline checks for outward-facing project documentation.

These tests enforce the positioning rules adopted in
docs/superpowers/specs/2026-07-05-platform-positioning-and-roadmap-design.md:

- the README must separate design intent from current status
- source-model fidelity tiers must be declared in the README
- every safety-scope exclusion must survive verbatim in the README
- comparative performance wording requires a benchmark reference in the
  same document
"""

from __future__ import annotations

import re
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
README_PATH = PROJECT_ROOT / "README.md"
SAFETY_SCOPE_PATH = PROJECT_ROOT / "docs" / "safety_scope.md"
ALLOWED_ADAPTERS = {"geant4", "root", "xsuite"}

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


def test_safety_scope_exclusions_survive_in_readme():
    exclusions = safety_scope_exclusions()
    assert len(exclusions) >= 9, "safety scope exclusions must not be trimmed"

    readme = README_PATH.read_text(encoding="utf-8").lower()
    missing = [item for item in exclusions if item.lower() not in readme]

    assert missing == [], f"README safety scope is missing exclusions: {missing}"


def test_no_unsubstantiated_performance_claims():
    offenders = []
    for path in outward_facing_docs():
        text = path.read_text(encoding="utf-8")
        match = PERFORMANCE_CLAIM_PATTERN.search(text)
        if match and "benchmark" not in text.lower():
            offenders.append(
                f"{path.relative_to(PROJECT_ROOT)} claims {match.group(0)!r} "
                "without referencing a benchmark"
            )

    assert offenders == []


def test_only_placeholder_adapters_are_present():
    adapter_dirs = {
        path.name
        for path in (PROJECT_ROOT / "latent_dirac" / "adapters").iterdir()
        if path.is_dir() and path.name != "__pycache__"
    }

    assert adapter_dirs == ALLOWED_ADAPTERS
