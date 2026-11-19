"""Theory obligation S2: symmetry-tolerance sensitivity of MP-20 preprocessing.

Runs the production preprocessing path of the frozen 512-record pilot
(``artifacts/mp20_pilot_2026-09-14/audit_mp20_preprocessing.py``: same pinned
CSV, selection, certification receipt, builder, and identity step) once with
the legacy v6 policy and once per symmetry tolerance ``tau`` on a grid, where
``SpglibPolicy(symprec=tau, idealization_tolerance_angstrom=tau)`` bounds both
classification and raw-to-ideal displacement by the same length (review B1).

No input is replaced or refilled; unexpected exceptions abort.  Per grid point
the report gives per-crystal-family acceptance (families from the upstream
label), agreement of the accepted space group with the upstream label and the
direction of each disagreement, rejection reasons, and the recorded raw-to-
ideal displacements.  Selection rule (declared before the run, review B1):
choose the smallest tau at which no family's acceptance rate is below half of
the best family's, unless a stated physical reason justifies an exception.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import statistics
import time
from collections import Counter
from pathlib import Path

import numpy as np
import spglib

from crystalatlas.benchmark_data import read_benchmark_cif_csv, select_training_pilot
from crystalatlas.certification import CrystallographyCertification
from crystalatlas.config import DomainConfig
from crystalatlas.dataset_builder import CatalogueDatasetTargetBuilder
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
LEGACY_V6_BUILDER = "a6e24ded30b9d4a69f5ca07213e23359c49d3908290914bfb2abd9240855bf37"
DEFAULT_GRID = (1e-5, 1e-4, 1e-3, 1e-2, 1e-1)
FAMILIES = (
    ("triclinic", 1, 2),
    ("monoclinic", 3, 15),
    ("orthorhombic", 16, 74),
    ("tetragonal", 75, 142),
    ("trigonal", 143, 167),
    ("hexagonal", 168, 194),
    ("cubic", 195, 230),
)


def family(number: int) -> str:
    for name, lower, upper in FAMILIES:
        if lower <= number <= upper:
            return name
    raise ValueError(f"space-group number {number} outside 1..230")


_POINT_GROUP_ORDER: dict[int, int] = {}


def point_group_order(number: int) -> int:
    """Number of distinct rotations of a space-group type (spglib database).

    Database operations of a centred setting repeat each rotation once per
    centring vector, so distinct rotations are counted.
    """

    if number not in _POINT_GROUP_ORDER:
        hall = next(h for h in range(1, 531) if spglib.get_spacegroup_type(h).number == number)
        rotations = spglib.get_symmetry_from_database(hall)["rotations"]
        _POINT_GROUP_ORDER[number] = len({tuple(np.asarray(r).ravel()) for r in rotations})
    return _POINT_GROUP_ORDER[number]


def run_arm(name, policy, pilot, catalogue, domain, *, merge_orbits):  # type: ignore[no-untyped-def]
    builder = CatalogueDatasetTargetBuilder(
        SpglibStandardizer(pinned_hall_numbers(), policy),
        CatalogueTargetAssigner(catalogue, merge_quantization_equivalent_orbits=merge_orbits),
        domain,
    )
    attempts = []
    started_arm = time.perf_counter()
    for index, row in enumerate(pilot):
        upstream = dict(row.metadata).get("spacegroup.number")
        attempt: dict[str, object] = {
            "index": index,
            "material_id": row.material_id,
            "upstream_spacegroup": int(upstream) if upstream else None,
            "status": "started",
        }
        attempts.append(attempt)
        try:
            raw = parse_cif_structure(row.cif, source_id=row.material_id)
        except DomainError as exc:
            attempt.update(status="rejected", stage="parse", reason=str(exc))
            continue
        try:
            record = builder(raw)
        except RecordPreprocessingError as exc:
            attempt.update(
                status="rejected",
                stage="crystallography",
                exception=type(exc).__name__,
                reason=str(exc),
            )
            continue
        receipt = record.canonicalized.receipt
        attempt.update(
            accepted_spacegroup=int(spglib.get_spacegroup_type(receipt.hall_number).number),
            hall_number=receipt.hall_number,
            idealization_lattice_displacement_angstrom=(
                receipt.idealization_lattice_displacement_angstrom
            ),
            idealization_site_displacement_angstrom=(
                receipt.idealization_site_displacement_angstrom
            ),
        )
        try:
            build_canonical_identity(record, catalogue=catalogue, policy=IdentityPolicy())
        except AmbiguousIdentityError as exc:
            attempt.update(status="rejected", stage="identity", reason=str(exc))
            continue
        attempt.update(status="passed")
    print(f"{name}: {time.perf_counter() - started_arm:.1f} s", flush=True)
    return builder.builder_fingerprint, attempts


def summarize(attempts):  # type: ignore[no-untyped-def]
    by_family = {}
    for name, _, _ in FAMILIES:
        members = [a for a in attempts if a["upstream_spacegroup"] and
                   family(a["upstream_spacegroup"]) == name]
        accepted = sum(a["status"] == "passed" for a in members)
        chart = sum(
            a["status"] == "rejected" and "bounded chart ball" in str(a.get("reason", ""))
            for a in members
        )
        by_family[name] = {
            "n": len(members),
            "accepted": accepted,
            "rate": round(accepted / len(members), 4) if members else None,
            "chart_domain_rejected": chart,
            "rate_excluding_chart_domain": (
                round((accepted + chart) / len(members), 4) if members else None
            ),
        }
    passed = [a for a in attempts if a["status"] == "passed"]
    agree = [a for a in passed if a["accepted_spacegroup"] == a["upstream_spacegroup"]]
    disagreements = [
        {
            "material_id": a["material_id"],
            "upstream": a["upstream_spacegroup"],
            "accepted": a["accepted_spacegroup"],
            "accepted_point_group_order": point_group_order(a["accepted_spacegroup"]),
            "upstream_point_group_order": point_group_order(a["upstream_spacegroup"]),
        }
        for a in passed
        if a["accepted_spacegroup"] != a["upstream_spacegroup"]
    ]
    lattice = [a["idealization_lattice_displacement_angstrom"] for a in passed
               if a.get("idealization_lattice_displacement_angstrom") is not None]
    site = [a["idealization_site_displacement_angstrom"] for a in passed
            if a.get("idealization_site_displacement_angstrom") is not None]
    rates = [v["rate"] for v in by_family.values() if v["rate"] is not None]
    pre_chart = [
        v["rate_excluding_chart_domain"]
        for v in by_family.values()
        if v["rate_excluding_chart_domain"] is not None
    ]
    return {
        "accepted": len(passed),
        "attempted": len(attempts),
        "by_family": by_family,
        "worst_to_best_family_rate": (
            round(min(rates) / max(rates), 4) if rates and max(rates) else None
        ),
        "worst_to_best_family_rate_excluding_chart_domain": (
            round(min(pre_chart) / max(pre_chart), 4) if pre_chart and max(pre_chart) else None
        ),
        "label_agreement_among_accepted": (
            round(len(agree) / len(passed), 4) if passed else None
        ),
        "label_disagreements": disagreements,
        "rejection_reasons": dict(
            Counter(
                str(a.get("reason", "")).split(":")[0][:90]
                for a in attempts
                if a["status"] == "rejected"
            ).most_common()
        ),
        "lattice_displacement_angstrom": {
            "median": statistics.median(lattice), "max": max(lattice)
        } if lattice else None,
        "site_displacement_angstrom": {
            "median": statistics.median(site), "max": max(site)
        } if site else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-csv", type=Path, required=True)
    parser.add_argument("--certification", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--count", type=int, default=512)
    parser.add_argument("--grid", type=float, nargs="+", default=list(DEFAULT_GRID))
    parser.add_argument("--reuse-legacy", type=Path, default=None)
    parser.add_argument("--relative-metric-quantum", type=float, default=1e-9)
    parser.add_argument("--max-primitive-atoms", type=int, default=20)
    args = parser.parse_args()
    if args.output.exists() or not 1 <= args.count <= 512:
        parser.error("output must be new and count between 1 and 512")
    rows = read_benchmark_cif_csv(
        args.train_csv, split="train", expected_sha256=TRAIN_SHA256, expected_count=27136
    )
    pilot = select_training_pilot(rows, count=512, seed=20260913)[: args.count]
    certification = CrystallographyCertification.read_json(args.certification)
    if certification.receipt_fingerprint != CERTIFICATION:
        raise ValueError("certification receipt disagrees with the verified external pin")
    catalogue = build_full_catalogue()
    certification.require_for(catalogue)
    # The study ran on the ball-v1 chart; loewner-v2 became the default afterwards.
    legacy_domain = DomainConfig(lattice_chart="ball-v1")
    # The deployed v7 candidate: dimensioned tau, scale-relative ranking quantum,
    # quantization-equivalent orbit merging, and the primitive atom bound (B1/B2).
    domain = DomainConfig(
        max_primitive_atoms=args.max_primitive_atoms,
        max_raw_atoms=4 * args.max_primitive_atoms,
        lattice_chart="ball-v1",
    )
    arms = [("legacy-v6", SpglibPolicy(), legacy_domain, False)]
    arms += [
        (
            f"tau={tau:g}",
            SpglibPolicy(
                symprec=tau,
                idealization_tolerance_angstrom=tau,
                relative_metric_quantum=args.relative_metric_quantum,
            ),
            domain,
            True,
        )
        for tau in args.grid
    ]
    report: dict[str, object] = {
        "schema": "crystalatlas.s2-symmetry-tolerance-study.v1",
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "spglib": spglib.__version__,
        "count": args.count,
        "arms": {},
    }
    for name, policy, arm_domain, merge in arms:
        if name == "legacy-v6" and args.reuse_legacy is not None:
            saved = json.loads(args.reuse_legacy.read_text())["arms"]["legacy-v6"]
            if saved["builder_fingerprint"] != LEGACY_V6_BUILDER:
                raise SystemExit("reused legacy arm does not carry the v6 builder fingerprint")
            saved["summary"] = summarize(saved["attempts"])
            report["arms"][name] = saved  # type: ignore[index]
            args.output.with_suffix(".partial.json").write_text(
                json.dumps(report, indent=1, allow_nan=False, default=str) + "\n"
            )
            continue
        fingerprint, attempts = run_arm(
            name, policy, pilot, catalogue, arm_domain, merge_orbits=merge
        )
        if name == "legacy-v6" and fingerprint != LEGACY_V6_BUILDER:
            raise SystemExit("legacy arm no longer reproduces the v6 builder fingerprint")
        report["arms"][name] = {  # type: ignore[index]
            "domain": arm_domain.identity_payload(),
            "merge_quantization_equivalent_orbits": merge,
            "policy": policy.fingerprint_payload(),
            "builder_fingerprint": fingerprint,
            "summary": summarize(attempts),
            "attempts": attempts,
        }
        # Persist after every arm so a later abort cannot discard finished arms.
        partial = args.output.with_suffix(".partial.json")
        partial.write_text(json.dumps(report, indent=1, allow_nan=False, default=str) + "\n")
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=1, allow_nan=False, default=str)
        stream.write("\n")
    args.output.with_suffix(".partial.json").unlink()
    for name in report["arms"]:  # type: ignore[union-attr]
        summary = report["arms"][name]["summary"]  # type: ignore[index]
        print(name, json.dumps({k: summary.get(k) for k in (
            "accepted", "worst_to_best_family_rate",
            "worst_to_best_family_rate_excluding_chart_domain",
            "label_agreement_among_accepted")}))


if __name__ == "__main__":
    main()
