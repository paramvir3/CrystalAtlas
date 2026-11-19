"""Measurement M2: the starting pair separations of every replayed attempt.

Registered in R1_LOG.md, "Measurement M2" (2026-11-15).  Runs ``m1_contacts.main``
unchanged, with the same arguments, after two arithmetic-neutral wrappers:

* ``describe_row`` also returns a ``start`` block for every solved attempt (an attempt of
  a recorded padded solve): the cell and atoms of the plan state at step 0
  (``PaddedCoverPlan.expand`` of the recorded initial state), the exact image distances up
  to ``--radius`` from ``m1_contacts.pair_geometry``, the closest contact class, the
  shortest lattice vector, pair counts at most 0.01, 0.1 and 0.5 angstrom, and the
  starting volume per atom;
* the result is written with this script's schema and sha256 next to the M1 provenance.

The targets file must trace no attempt (checked before the run); every solved attempt then
gets its end state and its start.  Research diagnostic: it reads private attributes of the
generator and changes no sampled value.
"""

from __future__ import annotations

import hashlib
import json
import math
import sys
from pathlib import Path

import m1_contacts as m1
import numpy as np
import torch

SCHEMA = "crystalatlas.research-v7-m2-starts.v1"
COUNT_THRESHOLDS = (0.01, 0.1, 0.5)
M1_SHA256 = "1f8332a87fc8c707c71ad35a745862665bc2b4c7cd19ebd71465950e0d3ca432"


def start_summary(record, row, model, *, radius: float, max_reach: int):  # type: ignore[no-untyped-def]
    """Geometry of row ``row`` of the recorded solve at step 0."""

    plan = record["plan"]
    cover = record["covers"][row]
    orbit_of, numbers, orbits = m1.cell_atoms(model, cover)
    count = plan.atom_counts[row]
    with torch.no_grad():
        expanded = plan.expand(record["z"][0], record["eta"][0])
        lattice = expanded.lattice[row].to(dtype=torch.float64)
        fractional = expanded.fractional_coordinates[row, :count].to(dtype=torch.float64)
    volume = abs(float(torch.linalg.det(lattice)))
    entry: dict[str, object] = {
        "time": float(record["times"][0]),
        "atoms": int(count),
        "volume_per_atom": m1.significant(volume / int(count)),
    }
    geometry = m1.pair_geometry(lattice, fractional, radius, max_reach)
    if geometry is None:
        entry["resolved"] = False
        return entry
    distance, rho, reduced = geometry
    entry["resolved"] = True
    classes = m1.class_minima(distance, rho, orbit_of, len(orbits))
    best = min(classes.items(), key=lambda item: item[1][0], default=None)
    closest = None
    if best is not None and math.isfinite(best[1][0]):
        (low, high), (value, smooth, smooth_max, i, j) = best
        closest = {
            "class": [low, high],
            "mates": low == high,
            "atoms": [i, j],
            "atomic_numbers": [numbers[i], numbers[j]],
            "wyckoff": [orbits[low][0], orbits[high][0]],
            "r": m1.significant(value),
            "rho": m1.significant(smooth),
            "rho_max": m1.significant(smooth_max),
            "delta": [m1.significant(float(v)) for v in reduced[i, j]],
        }
    first, second = np.triu_indices(int(count), 1)
    between = distance[first, second]
    entry["closest"] = closest
    entry["lattice_min"] = m1.significant(float(distance[0, 0]))
    entry["pairs_at_most"] = {
        str(threshold): int(np.count_nonzero(between <= threshold)) for threshold in COUNT_THRESHOLDS
    }
    return entry


def check_inputs(argv: list[str]) -> None:
    """Refuse a changed M1 script or a targets file that traces any attempt, before any work."""

    found = hashlib.sha256(Path(m1.__file__).read_bytes()).hexdigest()
    if found != M1_SHA256:
        raise m1.InvariantViolation(f"m1_contacts.py has sha256 {found}, not the registered {M1_SHA256}")
    if "--targets" not in argv:
        raise m1.InvariantViolation("--targets is required")
    wanted = json.loads(Path(argv[argv.index("--targets") + 1]).read_text())
    if wanted.get("replay") != {} or wanted.get("oracle") != {}:
        raise m1.InvariantViolation("M2 runs with no traced attempt: the targets file must be empty")


def install() -> None:
    describe = m1.describe_row

    def describe_with_start(record, row, model, *, steps, radius, max_reach, **rest):  # type: ignore[no-untyped-def]
        entry = describe(record, row, model, steps=steps, radius=radius, max_reach=max_reach, **rest)
        entry["start"] = start_summary(record, row, model, radius=radius, max_reach=max_reach)
        return entry

    m1.describe_row = describe_with_start

    write = m1.runner._write_new
    script = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()

    def write_with_provenance(path, result):  # type: ignore[no-untyped-def]
        traced = sum(1 for case in result["records"] for kind in ("replay", "oracle")
                     for attempt in case[kind] if attempt["traced"])
        if traced:
            raise m1.InvariantViolation("M2 runs with no traced attempt")
        write(path, {**result, "schema": SCHEMA, "m1_schema": m1.SCHEMA,
                     "m1_script_sha256": result["script_sha256"], "m2_script_sha256": script})

    m1.runner._write_new = write_with_provenance


if __name__ == "__main__":
    check_inputs(sys.argv)
    install()
    m1.main()
