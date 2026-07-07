# Positron cross-section data sources for buffer-gas cooling (N₂, CF₄)

Reference catalogue for curating a real, DOI-cited cross-section table to
promote `buffer_gas_cooling` from the synthetic placeholder
(`parameterized`) to `table-based` (see
`docs/superpowers/specs/2026-07-06-buffer-gas-collisions-design.md` and
the table loader in `latent_dirac/collisions/cross_sections.py`).

**This file catalogues SOURCES and their data availability only — it
contains no cross-section values.** Per the honesty discipline, no
number is transcribed until it comes from the cited source itself, and
each curated table carries its own provenance header (gas, tier, source,
DOI, method, energy range). Compiled 2026-07-07 from an adversarially
verified literature sweep (24/25 claims confirmed); DOIs are the anchors,
availability notes must be re-checked against the actual PDFs before
curation.

## Bottom line for a cooling model

The dominant COOLING channels differ by gas, and direct annihilation is
**not** a cooling channel:

- **CF₄** cools chiefly via **vibrational** excitation (the low-threshold
  ν₃ mode ≈ 0.16 eV is why CF₄ is the fast Surko coolant).
- **N₂** cools chiefly via **rotational** (quadrupole J→J+2) excitation —
  *not* vibrational.
- **Direct annihilation (Z_eff ≈ 10)** is negligible for cooling in both
  gases (Swann & Green 2023); it matters only as a loss endpoint, which
  the engine/annihilation path already handles. A cooling table does not
  need a Z_eff channel.

Consequence for our synthetic toys: `n2_positron_toy.csv` uses a
"vibrational" cooling channel as a stand-in, but the *real* N₂ coolant is
rotational — a real N₂ table should add a rotational channel (our loader
accepts arbitrary channel names, so this is a header/data change, not a
code change). `cf4_positron_toy.csv`'s vibrational-dominated shape is
qualitatively right for CF₄.

## Candidate datasets

### Best single citable source covering BOTH gases

- **Brunger, Buckman & Ratnavelu (2017)** — *Positron Scattering from
  Molecules: An Experimental Cross Section Compilation for Positron
  Transport Studies and Benchmarking Theory*, J. Phys. Chem. Ref. Data
  **46**, 023102. DOI `10.1063/1.4982827`. Experimental **compilation**;
  covers N₂ and CF₄ across elastic, vibrational, electronic excitation,
  Ps formation, ionization, grand total. **Availability:** JPCRD printed
  numeric tables of recommended absolute values + figures (transcribable,
  no plot digitization) — *not* a downloadable file. Paywalled (AIP).
  **This is the recommended anchor.** (Its 2019 companion, JPCRD 48,
  023102, DOI `10.1063/1.5089638`, is atoms-only — not useful here.)

### N₂

| Source | DOI | Type | Channels | Energy | Availability |
|---|---|---|---|---|---|
| Marler & Surko 2005, PRA 72, 062713 | `10.1103/PhysRevA.72.062713` | exp. | Ps formation, direct+total ionization, near-threshold electronic exc. (a¹Π, a′¹Σ) | thr–90 eV | in-paper tables/figures; paywalled |
| Zecca et al. 2011, New J. Phys. 13, 115001 | `10.1088/1367-2630/13/11/115001` | exp. | total CS | ~0.1–50 eV | **open access**; tables/figures |
| Poveda et al. 2019, PRA 100, 062706 | `10.1103/PhysRevA.100.062706` | theory (close-coupling) | elastic, rotational, vibrational (below Ps) + a cooling model | low-energy | figures/tables; paywalled |
| Rawlins et al. 2023, PRL 130, 263001 | `10.1103/PhysRevLett.130.263001` (arXiv:2303.02083, **OA**) | theory (many-body) | elastic + annihilation rate | low-energy | figures; arXiv PDF |
| **N₂ rotational (the coolant)** — Mukherjee & Mukherjee 2015, PRA 91, 062706; Barp et al. 2018, J. Phys. B 51, 205201 | `10.1103/PhysRevA.91.062706`; `10.1088/1361-6455/aae1d3` | theory | rotational J→J+2 (dominant N₂ cooling channel) | low-energy | to be checked |

### CF₄

| Source | DOI | Type | Channels | Energy | Availability |
|---|---|---|---|---|---|
| **Marler & Surko 2005, PRA 72, 062702** | `10.1103/PhysRevA.72.062702` | exp. | **ν₃ vibrational excitation** (the landmark CF₄ coolant dataset) | 0.1–2 eV | in-paper table/figure; paywalled |
| Sueoka, Mori & Hamada 1994, J. Phys. B 27, 1453 | `10.1088/0953-4075/27/7/019` | exp. | total CS | 0.7–400 eV | in-paper tables/figures; paywalled |
| Mori et al. 2024, J. Phys. B 57, 225201 | `10.1088/1361-6455/ad7f35` | theory (CCC+IAM) | total, elastic, momentum-transfer, Ps, ionization, electron-loss, bound-state exc., stopping power | thr–5000 eV | figures/PDF; an OA mirror exists (PMC) |
| Yadav & Antony 2026, RSC Adv. 16, 19300 | `10.1039/d6ra01378c` | theory (SCOP/CSP-ic) | elastic, inelastic, total, ionization, Ps | 1–5000 eV | **open access**; data in figures + PDF SI (needs parsing) |
| **CF₄ vibrational (the coolant)** — Franz, Baccarelli, Caprasecca & Gianturco 2009, PRA 80, 012709 | `10.1103/PhysRevA.80.012709` | theory | vibrational (e⁻ + e⁺) | low-energy | to be checked |

### Cooling-physics synthesis (not a dataset, but the map)

- **Swann & Green 2023** — *Positron cooling via inelastic collisions in
  CF₄ and N₂ gases*, PRL 130, 033001 (arXiv:2105.06904, **OA**). DOI
  `10.1103/PhysRevLett.130.033001`. Establishes the dominant cooling
  channels above and points to the specific channel-input papers (Franz
  2009; Mukherjee 2015; Barp 2018). Cross sections shown as figures only.
- **Gribakin, Young & Surko 2010** — Rev. Mod. Phys. 82, 2557, DOI
  `10.1103/RevModPhys.82.2557`. The Z_eff/annihilation review (below Ps
  threshold). Useful for the annihilation *endpoint*, not for cooling
  channels.

## Databases

- **LXCat has NO positron data** — it hosts electron- and ion-neutral
  cross sections only (it does have electron N₂/CF₄ sets). No open
  positron analogue of LXCat, and no NIST positron cross-section
  database, was found. Positron data must be sourced paper-by-paper.

## Recommendation (for owner decision)

1. **Anchor:** obtain the 2017 JPCRD compilation (DOI 10.1063/1.4982827)
   PDF and transcribe its recommended N₂ and CF₄ tables — broadest
   channel + gas coverage, recommended (evaluated) values, one citation.
2. **Add the coolant channels it may under-cover:** CF₄ ν₃ vibrational
   from Marler & Surko 2005 (062702); N₂ rotational from Mukherjee 2015 /
   Barp 2018.
3. **Fully-open quick start** (if avoiding paywalls first): Zecca 2011
   (N₂ total, OA) + Yadav & Antony 2026 (CF₄ multi-channel, OA SI). Lower
   fidelity/coverage, but citable and immediately obtainable.
4. Label each curated table's `fidelity_tier: table-based` with its own
   DOI/method; mix experiment and theory only with a per-source note.

## Caveats

- **No machine-readable positron CS file** was confirmed for either gas;
  every route needs transcription from printed tables (JPCRD, best case)
  or digitization from figures (worst case). Budget curation effort
  accordingly.
- No single source covers both gases and all channels.
- CF₄ data mostly starts ≳0.7–1 eV; the 0.1–0.7 eV tail is thin.
- Availability notes marked "paywalled/403" were not read in full text;
  re-verify table-vs-figure status against each PDF before curating.
- Open question: does the 2017 JPCRD volume tabulate CF₄ vibrational and
  N₂ rotational specifically, or only total/elastic/Ps/ionization? Check
  its per-channel content before relying on it for the coolant channels.
