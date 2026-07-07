"""Buffer-gas collision physics (Surko-type positron cooling).

Cross-section tables (curated per publication with provenance — positron
data is not in a single open database) plus the null-collision Monte
Carlo operator. See
docs/superpowers/specs/2026-07-06-buffer-gas-collisions-design.md and
docs/superpowers/specs/2026-07-06-buffer-gas-table-based-landing-design.md.
"""

from __future__ import annotations

from latent_dirac.collisions.cross_sections import (
    CrossSectionTable,
    load_cross_sections,
    sigma,
)

__all__ = ["CrossSectionTable", "load_cross_sections", "sigma"]
