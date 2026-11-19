"""Non-training preflight of a fixed prefix of the hash-selected MP-20 pilot.

No replacement of rejected inputs; no model sampling, fitting, or evaluation.
Unexpected exceptions abort, rather than becoming convenient dataset rejections.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import time
from collections import Counter
from dataclasses import asdict
from pathlib import Path

import numpy as np
import spglib

import crystalatlas
from crystalatlas.benchmark_data import read_benchmark_cif_csv, select_training_pilot
from crystalatlas.certification import CrystallographyCertification
from crystalatlas.config import DomainConfig
from crystalatlas.dataset_builder import (
    CatalogueDatasetTargetBuilder,
    canonical_training_target_fingerprint,
)
from crystalatlas.dataset_identity import (
    AmbiguousIdentityError,
    IdentityPolicy,
    build_canonical_identity,
)
from crystalatlas.errors import DomainError, RecordPreprocessingError
from crystalatlas.full_catalogue import build_full_catalogue, pinned_hall_numbers
from crystalatlas.ingestion import CatalogueTargetAssigner, parse_cif_structure
from crystalatlas.standardization import SpglibPolicy, SpglibStandardizer

TRAIN_SHA256 = "133bab58dba02d316f9f91174839f17802dce648be4e8659d1a42b32b97fa01d"
CERTIFICATION = "6ec73dccec43b9e53fa76e7a8bba44bd2cfec25d9733df624a55da57637d50d7"


def sources():
    root = Path(crystalatlas.__file__).resolve().parent
    return {
        p.relative_to(root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(root.rglob("*.py"))
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-csv", type=Path, required=True)
    parser.add_argument("--certification", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--count", type=int, default=32)
    args = parser.parse_args()
    if args.output.exists() or not 1 <= args.count <= 512:
        parser.error("output must be new and preflight count between1 and512")
    before = sources()
    rows = read_benchmark_cif_csv(
        args.train_csv, split="train", expected_sha256=TRAIN_SHA256, expected_count=27136
    )
    pilot = select_training_pilot(rows, count=512, seed=20260913)
    certification = CrystallographyCertification.read_json(args.certification)
    if certification.receipt_fingerprint != CERTIFICATION:
        raise ValueError("certification receipt disagrees with prior verified external pin")
    start = time.perf_counter()
    print("Building and binding all-230 catalogue", flush=True)
    catalogue = build_full_catalogue()
    certification.require_for(catalogue)
    builder = CatalogueDatasetTargetBuilder(
        SpglibStandardizer(pinned_hall_numbers(), SpglibPolicy()),
        CatalogueTargetAssigner(catalogue),
        DomainConfig(lattice_chart="ball-v1"),  # as run (pre loewner-v2 default)
    )
    setup_seconds = time.perf_counter() - start
    attempts = []
    fatal = None
    for index, row in enumerate(pilot[: args.count]):
        attempt = {
            "index": index,
            "material_id": row.material_id,
            "row_index": row.row_index,
            "cif_sha256": row.cif_sha256,
            "upstream_spacegroup": dict(row.metadata).get("spacegroup.number"),
            "status": "started",
            "stage": "parse",
        }
        attempts.append(attempt)
        started = time.perf_counter()
        try:
            try:
                raw = parse_cif_structure(row.cif, source_id=row.material_id)
            except DomainError as exc:
                attempt.update(status="rejected", exception=type(exc).__name__, reason=str(exc))
                continue
            attempt.update(raw_atoms=len(raw.atomic_numbers), stage="crystallography")
            try:
                record = builder(raw)
            except RecordPreprocessingError as exc:
                attempt.update(status="rejected", exception=type(exc).__name__, reason=str(exc))
                continue
            receipt = record.canonicalized.receipt
            attempt.update(
                stage="identity",
                roundtrip_angstrom=receipt.roundtrip_max_cartesian_error,
                roundtrip_tolerance_angstrom=receipt.symmetry_tolerance,
                hall_number=receipt.hall_number,
                all_receipt_gates=receipt.passed,
                target_fingerprint=canonical_training_target_fingerprint(record.target),
                canonical_atoms=record.target.layout.expanded_atom_count,
                target_setting=record.target.layout.setting.key,
            )
            try:
                build_canonical_identity(record, catalogue=catalogue, policy=IdentityPolicy())
            except AmbiguousIdentityError as exc:
                attempt.update(status="rejected", exception=type(exc).__name__, reason=str(exc))
                continue
            attempt.update(status="passed", stage="complete")
        except Exception as exc:
            fatal = {"exception": type(exc).__name__, "reason": str(exc)}
            attempt.update(status="fatal", **fatal)
            break
        finally:
            attempt["seconds"] = time.perf_counter() - started
            print(json.dumps(attempt, allow_nan=False, sort_keys=True), flush=True)
    after = sources()
    result = {
        "schema": "crystalatlas.mp20-preprocessing-preflight.v1",
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "package_sources": before,
        "source_tree_unchanged": before == after,
        "python": platform.python_version(),
        "numpy": np.__version__,
        "spglib": spglib.__version__,
        "domain": asdict(builder.domain),
        "standardizer_policy": asdict(builder.standardizer.policy),
        "builder_fingerprint": builder.builder_fingerprint,
        "catalogue_fingerprint": catalogue.fingerprint,
        "certification_fingerprint": certification.receipt_fingerprint,
        "train_csv_sha256": TRAIN_SHA256,
        "published_train_count": 27136,
        "pilot_size": 512,
        "selection_seed": 20260913,
        "scheduled_preflight_count": args.count,
        "scheduled_ids": [r.material_id for r in pilot[: args.count]],
        "attempts": attempts,
        "fatal": fatal,
        "status_counts": dict(Counter(a["status"] for a in attempts)),
        "not_run_count": args.count - len(attempts),
        "setup_seconds": setup_seconds,
        "total_seconds": time.perf_counter() - start,
        "scope": "input preprocessing diagnostics only; no fitting, no training store, no refills",
    }
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")
    if fatal or before != after:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
