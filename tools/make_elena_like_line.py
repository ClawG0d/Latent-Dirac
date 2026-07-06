"""Build the ELENA-like ring lattice artifact (xtrack Line JSON).

An ELENA-inspired, license-clean parameterized ring constructed
programmatically from public machine-scale parameters (circumference
~30.4 m, six-fold symmetry, 60-degree bends, ~0.93 m bending radius,
13.7 MeV/c antiprotons). This is NOT the real ELENA optics — quadrupole
strengths are chosen here for a stable FODO-like cell — and every
caption says so (fidelity tier: parameterized). The committed JSON at
`examples/data/elena_like_ring.json` is fully reproducible from this
script, which is its provenance.

Run where xsuite is installed (e.g. the WSL box):

    python tools/make_elena_like_line.py [output.json]
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

# public machine-scale parameters (ELENA design values, rounded)
CIRCUMFERENCE_M = 30.4056
N_CELLS = 6
BEND_ANGLE_RAD = 2.0 * math.pi / N_CELLS
BEND_RADIUS_M = 0.927
P0C_EV = 13.7e6  # extraction momentum x c
QUAD_LENGTH_M = 0.25
# (edge angle, k1) candidates scanned deterministically for a stable cell.
# A ring this tight over-focuses one plane on bend geometry alone: pure
# sector bends (edge 0) over-focus horizontally, full rectangular bends
# (edge theta/2) over-focus vertically — the stable window sits between,
# exactly the balance the real machine family strikes with pole-face
# rotation.
EDGE_ANGLE_CANDIDATES = [round(0.10 + 0.02 * i, 2) for i in range(20)]
QUAD_K1_CANDIDATES = [round(0.2 + 0.1 * i, 2) for i in range(25)]


def build_line(quad_k1_m2: float, edge_angle_rad: float):
    import xtrack as xt

    bend_length = BEND_RADIUS_M * BEND_ANGLE_RAD
    cell_length = CIRCUMFERENCE_M / N_CELLS
    drift_budget = cell_length - bend_length - 2.0 * QUAD_LENGTH_M
    outer_drift = 0.55
    inner_drift = 0.5 * (drift_budget - 2.0 * outer_drift)

    elements = {}
    names = []
    for cell in range(N_CELLS):
        tag = f"c{cell}"
        parts = [
            (f"do_{tag}_a", xt.Drift(length=outer_drift)),
            (f"qf_{tag}", xt.Quadrupole(length=QUAD_LENGTH_M, k1=quad_k1_m2)),
            (f"di_{tag}_a", xt.Drift(length=inner_drift)),
            (
                f"mb_{tag}",
                xt.Bend(
                    length=bend_length,
                    angle=BEND_ANGLE_RAD,
                    edge_entry_angle=edge_angle_rad,
                    edge_exit_angle=edge_angle_rad,
                ),
            ),
            (f"di_{tag}_b", xt.Drift(length=inner_drift)),
            (f"qd_{tag}", xt.Quadrupole(length=QUAD_LENGTH_M, k1=-quad_k1_m2)),
            (f"do_{tag}_b", xt.Drift(length=outer_drift)),
        ]
        for name, element in parts:
            elements[name] = element
            names.append(name)

    line = xt.Line(elements=elements, element_names=names)
    line.particle_ref = xt.Particles(mass0=xt.PROTON_MASS_EV, q0=-1, p0c=P0C_EV)
    return line


def _fractional_tune_ok(tune: float) -> bool:
    fractional = tune % 1.0
    return 0.1 < fractional < 0.42 or 0.58 < fractional < 0.9


def main() -> int:
    target = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("examples/data/elena_like_ring.json")

    found = None
    for edge_angle_rad in EDGE_ANGLE_CANDIDATES:
        for quad_k1_m2 in QUAD_K1_CANDIDATES:
            line = build_line(quad_k1_m2, edge_angle_rad)
            length = line.get_length()
            assert abs(length - CIRCUMFERENCE_M) < 1e-6, length
            line.build_tracker()
            try:
                twiss = line.twiss(method="4d")
            except Exception:
                continue  # unstable working point
            qx, qy = float(twiss.qx), float(twiss.qy)
            if _fractional_tune_ok(qx) and _fractional_tune_ok(qy):
                found = (edge_angle_rad, quad_k1_m2, qx, qy)
                break
        if found:
            break
    if not found:
        raise SystemExit("no stable working point found in the (edge, k1) scan")
    edge_angle_rad, quad_k1_m2, qx, qy = found

    line.to_json(str(target))
    print(f"wrote {target}")
    print(f"circumference: {length:.4f} m, p0c: {P0C_EV / 1e6:.1f} MeV/c (antiproton)")
    print(
        f"edge angle = {edge_angle_rad} rad, k1 = {quad_k1_m2} 1/m^2, "
        f"tunes: qx = {qx:.4f}, qy = {qy:.4f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
