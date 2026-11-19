"""Summarize a pinned MP-20 preprocessing report offline, without rerunning it.

Counts describe the recorded input-preprocessing policy, not physical validity,
model performance, source rights, or authorization to train. A report's source
freeze assertion is checked for consistency but not independently reproduced.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import stat
from collections import Counter
from pathlib import Path
from typing import Any

REPORT_SCHEMA = "crystalatlas.mp20-preprocessing-preflight.v1"
SUMMARY_SCHEMA = "crystalatlas.mp20-preprocessing-summary.v1"
TRAIN_SHA256 = "133bab58dba02d316f9f91174839f17802dce648be4e8659d1a42b32b97fa01d"
PILOT_SIZE = 512
SELECTION_SEED = 20260913
PUBLISHED_TRAIN_COUNT = 27136
MAX_REPORT_BYTES = 8 * 1024 * 1024
_DIGEST = re.compile(r"[0-9a-f]{64}\Z")
_STATUSES = ("passed", "rejected", "fatal")
_STAGES = ("parse", "crystallography", "identity")


def _integer(value: Any, name: str, lower: int, upper: int) -> int:
    if type(value) is not int or not lower <= value <= upper:
        raise ValueError(f"{name} must be an integer in [{lower}, {upper}]")
    return value


def _text(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be nonempty text")
    return value


def _digest(value: Any, name: str) -> str:
    if not isinstance(value, str) or _DIGEST.fullmatch(value) is None:
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return value


def _nonnegative(value: Any, name: str) -> float:
    if type(value) not in (int, float):
        raise ValueError(f"{name} must be a finite nonnegative number")
    try:
        result = float(value)
    except OverflowError as exc:
        raise ValueError(f"{name} must be finite") from exc
    if not math.isfinite(result) or result < 0:
        raise ValueError(f"{name} must be a finite nonnegative number")
    return result


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"report JSON repeats a key: {key}")
        result[key] = value
    return result


def _finite_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed):
        raise ValueError("report JSON contains a nonfinite number")
    return parsed


def _reject_constant(value: str) -> Any:
    raise ValueError(f"report JSON contains nonfinite constant: {value}")


def _read_report(path: Path, expected_sha256: str) -> tuple[dict[str, Any], str]:
    _digest(expected_sha256, "expected_report_sha256")
    # Reject final symlinks and FIFOs without blocking. This is not a sandbox
    # against an adversary replacing ancestor directories during the read.
    descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    with os.fdopen(descriptor, "rb") as stream:
        if not stat.S_ISREG(os.fstat(stream.fileno()).st_mode):
            raise ValueError("report must be a regular file")
        raw = stream.read(MAX_REPORT_BYTES + 1)
    if len(raw) > MAX_REPORT_BYTES:
        raise ValueError("report exceeds the byte limit")
    observed = hashlib.sha256(raw).hexdigest()
    if observed != expected_sha256:
        raise ValueError("report SHA-256 disagrees with the external pin")
    try:
        report = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=_unique_object,
            parse_float=_finite_float,
            parse_constant=_reject_constant,
        )
    except (UnicodeDecodeError, RecursionError) as exc:
        raise ValueError("report is not bounded-depth UTF-8 JSON") from exc
    if not isinstance(report, dict):
        raise ValueError("report must be a JSON object")
    return report, observed


def _upstream_group(value: Any) -> str:
    # The original importer retains this upstream CSV field as text. Missing
    # metadata is kept explicitly unknown; it is never inferred from a setting.
    if value is None or value == "":
        return "unknown"
    if not isinstance(value, str) or not re.fullmatch(r"[1-9][0-9]{0,2}", value):
        raise ValueError("upstream_spacegroup must be text in 1..230 or missing")
    _integer(int(value), "upstream_spacegroup", 1, 230)
    return value


def validate_and_summarize(report: dict[str, Any]) -> dict[str, Any]:
    """Check the v1 report contract and derive counts without selecting records."""
    if report.get("schema") != REPORT_SCHEMA:
        raise ValueError("unsupported preprocessing report schema")
    if report.get("source_tree_unchanged") is not True:
        raise ValueError("report does not assert an unchanged source tree")
    for field, expected in (
        ("pilot_size", PILOT_SIZE),
        ("selection_seed", SELECTION_SEED),
        ("published_train_count", PUBLISHED_TRAIN_COUNT),
    ):
        _integer(report.get(field), field, expected, expected)
    if report.get("train_csv_sha256") != TRAIN_SHA256:
        raise ValueError("report does not bind the frozen MP-20 train container")
    provenance = {}
    for field in (
        "script_sha256",
        "builder_fingerprint",
        "catalogue_fingerprint",
        "certification_fingerprint",
    ):
        provenance[field] = _digest(report.get(field), field)
    package_sources = report.get("package_sources")
    if not isinstance(package_sources, dict) or not package_sources:
        raise ValueError("package_sources must be a nonempty source-digest mapping")
    for name, digest in package_sources.items():
        _text(name, "package source name")
        _digest(digest, "package source digest")
    scheduled = _integer(report.get("scheduled_preflight_count"), "scheduled count", 1, 512)
    ids = report.get("scheduled_ids")
    if not isinstance(ids, list) or len(ids) != scheduled:
        raise ValueError("scheduled_ids length disagrees with the scheduled count")
    for material_id in ids:
        _text(material_id, "scheduled material_id")
    if len(set(ids)) != scheduled:
        raise ValueError("scheduled_ids must be unique")
    attempts = report.get("attempts")
    if not isinstance(attempts, list) or not 1 <= len(attempts) <= scheduled:
        raise ValueError("attempts must be a nonempty scheduled prefix")

    actual: Counter[str] = Counter()
    reasons: Counter[tuple[str, str, str, str]] = Counter()
    groups: dict[str, Counter[str]] = {}
    settings: Counter[str] = Counter()
    row_indices: set[int] = set()
    for index, attempt in enumerate(attempts):
        if not isinstance(attempt, dict):
            raise ValueError("each attempt must be an object")
        _integer(attempt.get("index"), "attempt index", index, index)
        if attempt.get("material_id") != ids[index]:
            raise ValueError("attempt IDs must be the exact ordered scheduled prefix")
        row_index = _integer(
            attempt.get("row_index"), "source row_index", 0, PUBLISHED_TRAIN_COUNT - 1
        )
        if row_index in row_indices:
            raise ValueError("attempts repeat a source row_index")
        row_indices.add(row_index)
        _digest(attempt.get("cif_sha256"), "attempt cif_sha256")
        _nonnegative(attempt.get("seconds"), "attempt seconds")
        status = attempt.get("status")
        if not isinstance(status, str) or status not in _STATUSES:
            raise ValueError("attempt has an unknown or unfinished status")
        actual[status] += 1
        group = _upstream_group(attempt.get("upstream_spacegroup"))
        groups.setdefault(group, Counter())[status] += 1
        if status == "passed":
            if (
                attempt.get("stage") != "complete"
                or attempt.get("all_receipt_gates") is not True
            ):
                raise ValueError(
                    "passed attempt must have complete stage and all receipt gates"
                )
            if "exception" in attempt or "reason" in attempt:
                raise ValueError("passed attempt must not carry a rejection or fatal reason")
            _integer(attempt.get("hall_number"), "hall_number", 1, 530)
            _integer(attempt.get("raw_atoms"), "raw_atoms", 1, 20)
            _integer(attempt.get("canonical_atoms"), "canonical_atoms", 1, 100000)
            _digest(attempt.get("target_fingerprint"), "target_fingerprint")
            settings[_text(attempt.get("target_setting"), "target_setting")] += 1
            error = _nonnegative(attempt.get("roundtrip_angstrom"), "roundtrip_angstrom")
            tolerance = _nonnegative(
                attempt.get("roundtrip_tolerance_angstrom"), "roundtrip_tolerance_angstrom"
            )
            if tolerance == 0 or error > tolerance:
                raise ValueError("passed attempt violates its recorded round-trip tolerance")
        else:
            stage = attempt.get("stage")
            if not isinstance(stage, str) or stage not in _STAGES:
                raise ValueError("rejected/fatal attempt must identify an incomplete stage")
            exception = _text(attempt.get("exception"), "attempt exception")
            reason = _text(attempt.get("reason"), "attempt reason")
            reasons[(status, stage, exception, reason)] += 1
            if status == "fatal" and index != len(attempts) - 1:
                raise ValueError("a fatal attempt must be the final attempt")

    declared = report.get("status_counts")
    if not isinstance(declared, dict):
        raise ValueError("status_counts must be an object")
    for status, count in declared.items():
        if status not in _STATUSES:
            raise ValueError("status_counts contains an unknown status")
        _integer(count, "status count", 1, scheduled)
    if declared != dict(actual):
        raise ValueError("status_counts disagrees with the attempts")
    not_run = scheduled - len(attempts)
    _integer(report.get("not_run_count"), "not_run_count", not_run, not_run)
    if "fatal" not in report:
        raise ValueError("report is missing fatal state")
    if actual["fatal"]:
        expected_fatal = {key: attempts[-1][key] for key in ("exception", "reason")}
        if report["fatal"] != expected_fatal:
            raise ValueError("report fatal details disagree with its final fatal attempt")
    elif report["fatal"] is not None or not_run:
        raise ValueError("a nonfatal report must account for every scheduled attempt")
    setup_seconds = _nonnegative(report.get("setup_seconds"), "setup_seconds")
    total_seconds = _nonnegative(report.get("total_seconds"), "total_seconds")
    if total_seconds < setup_seconds:
        raise ValueError("total_seconds is less than setup_seconds")

    counts = {
        "scheduled": scheduled,
        "attempted": len(attempts),
        **{status: actual[status] for status in _STATUSES},
        "not_run": not_run,
    }
    return {
        "schema": SUMMARY_SCHEMA,
        "scope": (
            "Recorded input-preprocessing diagnostics only; not physical validity or quality"
        ),
        "training_authorized_by_this_summary": False,
        "source_tree_unchanged_reported": True,
        "source_freeze_independently_reproduced": False,
        "run_state": "aborted_fatal" if actual["fatal"] else "completed",
        "all_scheduled_attempted": not_run == 0,
        "counts": counts,
        "denominator_note": (
            "No rejected, fatal, or not-run records are replaced. Upstream space-group counts "
            "cover attempted records only; metadata for unattempted records is unavailable."
        ),
        "pilot_size": PILOT_SIZE,
        "selection_seed": SELECTION_SEED,
        "published_train_count": PUBLISHED_TRAIN_COUNT,
        "train_csv_sha256": TRAIN_SHA256,
        "report_provenance": provenance,
        "reason_counts": [
            {
                "status": key[0],
                "stage": key[1],
                "exception": key[2],
                "reason": key[3],
                "count": count,
            }
            for key, count in sorted(reasons.items())
        ],
        "upstream_spacegroup_outcomes": {
            group: {
                "attempted": sum(outcomes.values()),
                **{status: outcomes[status] for status in _STATUSES},
            }
            for group, outcomes in sorted(
                groups.items(), key=lambda pair: 231 if pair[0] == "unknown" else int(pair[0])
            )
        },
        "passed_target_setting_counts": dict(sorted(settings.items())),
        "target_setting_vs_upstream_spacegroup_compared": False,
        "fatal": report["fatal"],
        "setup_seconds": setup_seconds,
        "total_seconds": total_seconds,
    }


def summarize_mp20_preflight(
    report_path: str | Path, output: str | Path, *, expected_report_sha256: str
) -> dict[str, Any]:
    """Read one pinned report and exclusively create one deterministic summary."""
    report_path, output = Path(report_path), Path(output)
    report, digest = _read_report(report_path, expected_report_sha256)
    summary = validate_and_summarize(report)
    summary["input_report"] = {"name": report_path.name, "sha256": digest}
    summary["summarizer_script_sha256"] = hashlib.sha256(
        Path(__file__).read_bytes()
    ).hexdigest()
    payload = (json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
    # Exclusive creation also refuses a dangling destination symlink. Do not
    # create directories or overwrite any original report or prior summary.
    with output.open("xb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--expected-report-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        summary = summarize_mp20_preflight(
            args.report, args.output, expected_report_sha256=args.expected_report_sha256
        )
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    print(json.dumps({"run_state": summary["run_state"], "counts": summary["counts"]}))


if __name__ == "__main__":
    main()
