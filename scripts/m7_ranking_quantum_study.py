"""Review M7: primitive-ranking rejections against the metric quantum on the MP-20 pilot.

Each frozen pilot record is standardized once (pinned spglib pass at the given
tau, exactly as in production), and the resulting idealized primitive cell is
ranked by ``canonicalize_primitive_presentation`` under the legacy absolute
quantum and under each relative quantum ``c`` (``c * max_i G_ii``).  Outcomes
are classified by the ranking's own rejection messages:

* ``bin-wall``: a guarded value meets a ranking-bin boundary (falls as e/q);
* ``unresolved-tie``: distinct resolved representatives share one key (grows
  as q coarsens);
* ``resolution``, ``resource``: the other declared rejections.

This is the trade-off study the review requests before fixing q.  Records whose
standardization fails at this tau are counted separately and never refilled.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np

from crystalatlas.benchmark_data import read_benchmark_cif_csv, select_training_pilot
from crystalatlas.errors import CrystalAtlasError
from crystalatlas.full_catalogue import pinned_hall_numbers
from crystalatlas.ingestion import parse_cif_structure
from crystalatlas.primitive_canonicalization import canonicalize_primitive_presentation
from crystalatlas.standardization import FloatingStructure, SpglibPolicy, SpglibStandardizer

TRAIN_SHA256 = "133bab58dba02d316f9f91174839f17802dce648be4e8659d1a42b32b97fa01d"
QUANTA = (None, 1e-11, 1e-10, 1e-9, 1e-8, 1e-7)
CLASSES = (
    ("intersects a ranking-bin boundary", "bin-wall"),
    ("ties between resolved distinct representatives", "unresolved-tie"),
    ("numerical resolution", "resolution"),
    ("float resolution", "resolution"),
    ("resource limit", "resource"),
)


def classify(message: str) -> str:
    return next((name for key, name in CLASSES if key in message), "other")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-csv", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--symprec", type=float, required=True)
    parser.add_argument("--count", type=int, default=512)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output must be new")
    rows = read_benchmark_cif_csv(
        args.train_csv, split="train", expected_sha256=TRAIN_SHA256, expected_count=27136
    )
    pilot = select_training_pilot(rows, count=512, seed=20260913)[: args.count]
    standardizer = SpglibStandardizer(
        pinned_hall_numbers(),
        SpglibPolicy(symprec=args.symprec, idealization_tolerance_angstrom=args.symprec),
    )
    outcomes: dict[str, Counter[str]] = {str(q): Counter() for q in QUANTA}
    standardization_rejected = 0
    records = []
    for row in pilot:
        raw = parse_cif_structure(row.cif, source_id=row.material_id)
        structure = FloatingStructure(raw.lattice, raw.fractional_positions, raw.atomic_numbers)
        try:
            initial = standardizer._canonicalize_pinned(structure)
        except CrystalAtlasError as exc:
            standardization_rejected += 1
            records.append({"material_id": row.material_id, "standardization": str(exc)[:120]})
            continue
        encoded, _ = standardizer._encode_species(initial.primitive_species)
        lattice = np.asarray(initial.primitive_lattice)
        record: dict[str, object] = {
            "material_id": row.material_id,
            "max_metric_diagonal": float(np.max(np.diag(lattice @ lattice.T))),
        }
        for quantum in QUANTA:
            try:
                canonicalize_primitive_presentation(
                    lattice,
                    initial.primitive_fractional_positions,
                    encoded,
                    decimals=12,
                    relative_metric_quantum=quantum,
                )
                outcome = "accepted"
            except CrystalAtlasError as exc:
                outcome = classify(str(exc))
            outcomes[str(quantum)][outcome] += 1
            record[str(quantum)] = outcome
        records.append(record)
    report = {
        "schema": "crystalatlas.m7-ranking-quantum-study.v1",
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "symprec_angstrom": args.symprec,
        "records": len(pilot),
        "standardization_rejected": standardization_rejected,
        "outcomes_by_quantum": {q: dict(c) for q, c in outcomes.items()},
        "attempts": records,
    }
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=1, allow_nan=False)
    summary = {k: report[k] for k in ("standardization_rejected", "outcomes_by_quantum")}
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
