"""Review B2: chart support, atom bound, and ranking resolution on the MP-20 pilot.

Runs the production canonicalizer of the frozen 512-record pilot under one
standardization policy, then records for every canonicalized record:

* setting, crystal family, centring index I_G, nu, |c|, conventional and
  primitive atom counts (primitive = nu |c| / I_G);
* the invariant-chart coefficients ``b`` of the conventional metric under the
  production reference shape at 20 A^3/atom, and ``||b||`` against B = 2;
* the final target-lift outcome under a supplied domain.

Coverage of a data-fitted reference shape is evaluated offline by 2-fold
cross-fitting within each setting: the centre is the mean coefficient vector of
one fold, decoded as an invariant metric shape, and coverage is measured on the
other fold, so no record is scored against a centre fitted on itself.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from crystalatlas.benchmark_data import read_benchmark_cif_csv, select_training_pilot
from crystalatlas.certification import CrystallographyCertification
from crystalatlas.config import DomainConfig
from crystalatlas.errors import CrystalAtlasError, RecordPreprocessingError
from crystalatlas.full_catalogue import build_full_catalogue, pinned_hall_numbers
from crystalatlas.ingestion import (
    CatalogueTargetAssigner,
    ComposedCanonicalizer,
    parse_cif_structure,
)
from crystalatlas.lattice import InvariantSPDChart, scale_metric_to_volume
from crystalatlas.standardization import SpglibPolicy, SpglibStandardizer
from crystalatlas.training_targets import lift_canonical_catalogue_target

TRAIN_SHA256 = "133bab58dba02d316f9f91174839f17802dce648be4e8659d1a42b32b97fa01d"
CERTIFICATION = "6ec73dccec43b9e53fa76e7a8bba44bd2cfec25d9733df624a55da57637d50d7"
FAMILIES = (
    ("triclinic", 1, 2), ("monoclinic", 3, 15), ("orthorhombic", 16, 74),
    ("tetragonal", 75, 142), ("trigonal", 143, 167), ("hexagonal", 168, 194),
    ("cubic", 195, 230),
)


def family(number: int) -> str:
    return next(name for name, low, high in FAMILIES if low <= number <= high)


def chart_for(setting, atoms, shape=None):  # type: ignore[no-untyped-def]
    reference = (
        np.asarray(setting.reference_metric, dtype=np.float64) if shape is None else shape
    )
    rotations = [np.asarray(r, dtype=np.float64) for r in setting.point_group_rotations]
    return InvariantSPDChart(
        rotations, reference_metric=scale_metric_to_volume(reference, 20.0 * atoms), bound=2.0
    )


def cross_fit(records: list[dict], settings) -> dict:  # type: ignore[no-untyped-def]
    """2-fold cross-fitted coverage of a per-setting centred reference shape."""

    by_setting: dict[str, list[dict]] = defaultdict(list)
    for record in records:
        by_setting[record["setting"]].append(record)
    scored = []
    for key, members in by_setting.items():
        setting = settings[key]
        members = sorted(members, key=lambda r: r["material_id"])
        folds = (members[0::2], members[1::2])
        for fit, test in ((folds[0], folds[1]), (folds[1], folds[0])):
            if not fit:
                # A singleton setting has no other record to fit on.
                scored.extend((r, None) for r in test)
                continue
            centre = np.mean([r["coefficients"] for r in fit], axis=0)
            shape = chart_for(setting, 1).decode(centre)
            for record in test:
                chart = chart_for(setting, record["conventional_atoms"], shape)
                b = chart.encode(np.asarray(record["metric"]))
                scored.append((record, float(np.linalg.norm(b))))
    result: dict[str, dict] = {}
    for name, _, _ in FAMILIES:
        rows = [(r, n) for r, n in scored if r["family"] == name]
        fitted = [n for _, n in rows if n is not None]
        result[name] = {
            "n": len(rows),
            "singleton_settings_unfitted": sum(n is None for _, n in rows),
            "inside_production_ball": sum(r["coefficient_norm"] < 2.0 for r, _ in rows),
            "inside_cross_fitted_ball": sum(n < 2.0 for n in fitted),
            "cross_fitted_norm_p99": float(np.quantile(fitted, 0.99)) if fitted else None,
            "cross_fitted_norm_max": max(fitted) if fitted else None,
        }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-csv", type=Path, required=True)
    parser.add_argument("--certification", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--symprec", type=float, required=True)
    parser.add_argument("--relative-metric-quantum", type=float, default=None)
    parser.add_argument("--max-primitive-atoms", type=int, default=None)
    parser.add_argument("--count", type=int, default=512)
    parser.add_argument(
        "--lattice-chart", choices=("ball-v1", "loewner-v2"), default="ball-v1"
    )
    parser.add_argument(
        "--selection-seed",
        type=int,
        default=20260913,
        help="hash-rank seed of the training-split sample (default: the frozen pilot)",
    )
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output must be new")
    rows = read_benchmark_cif_csv(
        args.train_csv, split="train", expected_sha256=TRAIN_SHA256, expected_count=27136
    )
    pilot = select_training_pilot(rows, count=args.count, seed=args.selection_seed)
    certification = CrystallographyCertification.read_json(args.certification)
    if certification.receipt_fingerprint != CERTIFICATION:
        raise ValueError("certification receipt disagrees with the verified external pin")
    catalogue = build_full_catalogue()
    certification.require_for(catalogue)
    policy = SpglibPolicy(
        symprec=args.symprec,
        idealization_tolerance_angstrom=args.symprec,
        relative_metric_quantum=args.relative_metric_quantum,
    )
    domain = (
        DomainConfig(lattice_chart=args.lattice_chart)
        if args.max_primitive_atoms is None
        else DomainConfig(
            max_primitive_atoms=args.max_primitive_atoms,
            max_raw_atoms=4 * args.max_primitive_atoms,
            lattice_chart=args.lattice_chart,
        )
    )
    assigner = CatalogueTargetAssigner(catalogue, merge_quantization_equivalent_orbits=True)
    canonicalizer = ComposedCanonicalizer(
        SpglibStandardizer(pinned_hall_numbers(), policy), assigner
    )
    settings = {s.key: s for s in catalogue.settings.values()}
    attempts = []
    for index, row in enumerate(pilot):
        attempt: dict[str, object] = {"index": index, "material_id": row.material_id}
        attempts.append(attempt)
        try:
            raw = parse_cif_structure(row.cif, source_id=row.material_id)
            canonicalized = canonicalizer.canonicalize(raw)
        except (RecordPreprocessingError, CrystalAtlasError) as exc:
            attempt.update(status="canonicalization-rejected", reason=str(exc)[:160])
            continue
        scaffold = canonicalized.scaffold
        setting = catalogue.get(scaffold.space_group)
        metric = np.asarray(canonicalized.conventional_structure.metric)
        formula = sum(count for _, count in scaffold.reduced_composition)
        conventional = len(canonicalized.conventional_structure.atomic_numbers)
        coefficients = chart_for(setting, conventional).encode(metric)
        attempt.update(
            setting=setting.key,
            family=family(setting.international_number),
            centering_index=setting.centering_index,
            nu=scaffold.nu,
            formula_atoms=formula,
            conventional_atoms=conventional,
            primitive_atoms=conventional // setting.centering_index,
            metric=metric.tolist(),
            coefficients=coefficients.tolist(),
            coefficient_norm=float(np.linalg.norm(coefficients)),
        )
        try:
            lift_canonical_catalogue_target(
                scaffold, metric, catalogue=catalogue, domain=domain
            )
            attempt.update(status="lifted")
        except CrystalAtlasError as exc:
            attempt.update(status="lift-rejected", reason=str(exc)[:160])
    canonical = [a for a in attempts if "setting" in a]
    report = {
        "schema": "crystalatlas.b2-chart-support-study.v1",
        "selection_seed": args.selection_seed,
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "policy": policy.fingerprint_payload(),
        "merge_quantization_equivalent_orbits": True,
        "domain": domain.identity_payload(),
        "status_counts": {
            s: sum(a["status"] == s for a in attempts)
            for s in ("canonicalization-rejected", "lift-rejected", "lifted")
        },
        "cross_fitted_coverage": cross_fit(canonical, settings),  # type: ignore[arg-type]
        "attempts": attempts,
    }
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=1, allow_nan=False, default=str)
    summary = {k: report[k] for k in ("status_counts", "cross_fitted_coverage")}
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
