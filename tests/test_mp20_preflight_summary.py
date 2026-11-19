from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "summarize_mp20_preflight.py"
_SPEC = importlib.util.spec_from_file_location("mp20_preflight_summary_script", _SCRIPT)
assert _SPEC is not None and _SPEC.loader is not None
summary_tool = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(summary_tool)


def _report(*, fatal: bool = False) -> dict[str, Any]:
    digest = "a" * 64
    attempts = [
        {
            "index": 0,
            "material_id": "mp-1",
            "row_index": 10,
            "cif_sha256": digest,
            "seconds": 0.1,
            "upstream_spacegroup": "1",
            "status": "passed",
            "stage": "complete",
            "all_receipt_gates": True,
            "raw_atoms": 3,
            "canonical_atoms": 3,
            "hall_number": 1,
            "target_fingerprint": digest,
            "target_setting": "P1",
            "roundtrip_angstrom": 1e-15,
            "roundtrip_tolerance_angstrom": 2e-8,
        },
        {
            "index": 1,
            "material_id": "mp-2",
            "row_index": 11,
            "cif_sha256": digest,
            "seconds": 0.2,
            "upstream_spacegroup": "225",
            "status": "fatal" if fatal else "rejected",
            "stage": "crystallography",
            "exception": "InvariantViolation" if fatal else "AmbiguousPreprocessingError",
            "reason": "synthetic fixture reason",
        },
    ]
    return {
        "schema": summary_tool.REPORT_SCHEMA,
        "pilot_size": 512,
        "selection_seed": 20260913,
        "published_train_count": 27136,
        "train_csv_sha256": summary_tool.TRAIN_SHA256,
        "script_sha256": digest,
        "builder_fingerprint": digest,
        "catalogue_fingerprint": digest,
        "certification_fingerprint": digest,
        "package_sources": {"fixture.py": digest},
        "source_tree_unchanged": True,
        "scheduled_ids": ["mp-1", "mp-2", "mp-3"] if fatal else ["mp-1", "mp-2"],
        "scheduled_preflight_count": 3 if fatal else 2,
        "attempts": attempts,
        "not_run_count": 1 if fatal else 0,
        "status_counts": {"passed": 1, "fatal" if fatal else "rejected": 1},
        "fatal": {key: attempts[-1][key] for key in ("exception", "reason")} if fatal else None,
        "setup_seconds": 1.0,
        "total_seconds": 1.3,
    }


def _write_report(tmp_path: Path, report: Any) -> tuple[Path, str]:
    path = tmp_path / "report.json"
    raw = json.dumps(report, allow_nan=False).encode()
    path.write_bytes(raw)
    return path, hashlib.sha256(raw).hexdigest()


@pytest.mark.parametrize("fatal", [False, True])
def test_summary_keeps_all_denominators_and_original_bytes(tmp_path: Path, fatal: bool) -> None:
    report = _report(fatal=fatal)
    path, pin = _write_report(tmp_path, report)
    before = path.read_bytes(), path.stat().st_mtime_ns
    result = summary_tool.summarize_mp20_preflight(
        path, tmp_path / "summary.json", expected_report_sha256=pin
    )
    assert result["counts"] == {
        "scheduled": 3 if fatal else 2,
        "attempted": 2,
        "passed": 1,
        "rejected": 0 if fatal else 1,
        "fatal": 1 if fatal else 0,
        "not_run": 1 if fatal else 0,
    }
    assert result["run_state"] == ("aborted_fatal" if fatal else "completed")
    assert result["all_scheduled_attempted"] is (not fatal)
    assert result["training_authorized_by_this_summary"] is False
    assert result["source_freeze_independently_reproduced"] is False
    assert result["target_setting_vs_upstream_spacegroup_compared"] is False
    assert result["input_report"]["sha256"] == pin
    assert (
        result["summarizer_script_sha256"] == hashlib.sha256(_SCRIPT.read_bytes()).hexdigest()
    )
    assert result["passed_target_setting_counts"] == {"P1": 1}
    assert result["upstream_spacegroup_outcomes"]["225"] == {
        "attempted": 1,
        "passed": 0,
        "rejected": 0 if fatal else 1,
        "fatal": int(fatal),
    }
    assert sum(item["count"] for item in result["reason_counts"]) == 1
    assert json.loads((tmp_path / "summary.json").read_bytes()) == result
    assert (path.read_bytes(), path.stat().st_mtime_ns) == before


def test_fatal_at_last_scheduled_record_is_not_reported_completed(tmp_path: Path) -> None:
    report = _report(fatal=True)
    report.update(scheduled_ids=["mp-1", "mp-2"], scheduled_preflight_count=2, not_run_count=0)
    path, pin = _write_report(tmp_path, report)
    result = summary_tool.summarize_mp20_preflight(
        path, tmp_path / "summary.json", expected_report_sha256=pin
    )
    assert result["all_scheduled_attempted"] is True
    assert result["run_state"] == "aborted_fatal"
    assert result["counts"]["fatal"] == 1


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("schema", "unrecognized.v1"),
        ("source_tree_unchanged", False),
        ("source_tree_unchanged", 1),
        ("pilot_size", 511),
        ("selection_seed", 42),
        ("published_train_count", 27135),
        ("train_csv_sha256", "b" * 64),
        ("scheduled_ids", ["mp-1", "mp-1"]),
        ("scheduled_ids", ["mp-1"]),
        ("scheduled_ids", ["mp-1", 2]),
        ("scheduled_preflight_count", True),
        ("status_counts", {"passed": 2}),
        ("status_counts", {"passed": True, "rejected": 1}),
        ("status_counts", {"passed": 1, "rejected": 1, "started": 0}),
        ("not_run_count", 1),
        ("fatal", {"exception": "False", "reason": "False"}),
        ("package_sources", {}),
        ("package_sources", {"fixture.py": "not a digest"}),
        ("builder_fingerprint", "bad"),
        ("total_seconds", 0.5),
        ("setup_seconds", -1),
        ("attempts", []),
    ],
)
def test_malformed_report_fails_before_output(tmp_path: Path, field: str, value: Any) -> None:
    report = _report()
    report[field] = value
    path, pin = _write_report(tmp_path, report)
    output = tmp_path / "summary.json"
    with pytest.raises(ValueError):
        summary_tool.summarize_mp20_preflight(path, output, expected_report_sha256=pin)
    assert not output.exists()


@pytest.mark.parametrize(
    ("index", "field", "value"),
    [
        (0, "index", 1),
        (0, "index", False),
        (1, "material_id", "mp-3"),
        (1, "row_index", 10),
        (1, "row_index", -1),
        (0, "cif_sha256", "bad"),
        (0, "status", "started"),
        (0, "status", []),
        (0, "stage", "identity"),
        (0, "all_receipt_gates", False),
        (0, "all_receipt_gates", 1),
        (0, "roundtrip_angstrom", 3e-8),
        (0, "roundtrip_tolerance_angstrom", 0),
        (0, "reason", "not genuinely passed"),
        (0, "raw_atoms", 21),
        (0, "target_setting", ""),
        (0, "target_fingerprint", "bad"),
        (0, "hall_number", 531),
        (1, "stage", "complete"),
        (1, "exception", ""),
        (1, "reason", ""),
        (0, "upstream_spacegroup", "231"),
        (0, "upstream_spacegroup", 1),
        (0, "seconds", True),
    ],
)
def test_malformed_attempt_fails_before_output(
    tmp_path: Path, index: int, field: str, value: Any
) -> None:
    report = _report()
    report["attempts"][index][field] = value
    path, pin = _write_report(tmp_path, report)
    with pytest.raises(ValueError):
        summary_tool.summarize_mp20_preflight(
            path, tmp_path / "summary.json", expected_report_sha256=pin
        )
    assert not (tmp_path / "summary.json").exists()


@pytest.mark.parametrize("mutation", ["middle_fatal", "missing_fatal", "mismatch", "partial"])
def test_inconsistent_fatal_state_rejected(tmp_path: Path, mutation: str) -> None:
    report = _report(fatal=True)
    if mutation == "middle_fatal":
        final = copy.deepcopy(report["attempts"][0])
        final.update(index=2, material_id="mp-3", row_index=12)
        report["attempts"].append(final)
        report.update(not_run_count=0, status_counts={"passed": 2, "fatal": 1})
    elif mutation == "missing_fatal":
        del report["fatal"]
    elif mutation == "mismatch":
        report["fatal"]["reason"] = "different reason"
    else:
        report["attempts"][-1]["status"] = "rejected"
        report.update(fatal=None, status_counts={"passed": 1, "rejected": 1})
    path, pin = _write_report(tmp_path, report)
    with pytest.raises(ValueError):
        summary_tool.summarize_mp20_preflight(
            path, tmp_path / "summary.json", expected_report_sha256=pin
        )


@pytest.mark.parametrize("unknown", [None, ""])
def test_missing_upstream_group_remains_unknown(unknown: Any) -> None:
    report = _report()
    report["attempts"][0]["upstream_spacegroup"] = unknown
    result = summary_tool.validate_and_summarize(report)
    assert result["upstream_spacegroup_outcomes"]["unknown"]["passed"] == 1


def test_reason_counts_aggregate_exact_status_stage_exception_and_reason() -> None:
    report = _report()
    last = copy.deepcopy(report["attempts"][-1])
    last.update(index=2, material_id="mp-3", row_index=12)
    report["attempts"].append(last)
    report["scheduled_ids"].append("mp-3")
    report.update(scheduled_preflight_count=3, status_counts={"passed": 1, "rejected": 2})
    result = summary_tool.validate_and_summarize(report)
    assert len(result["reason_counts"]) == 1
    assert result["reason_counts"][0]["count"] == 2
    assert result["upstream_spacegroup_outcomes"]["225"]["attempted"] == 2


@pytest.mark.parametrize(
    "raw",
    [
        b'{"schema": 1, "schema": 2}',
        b'{"nested": {"a": 1, "a": 2}}',
        b'{"value": NaN}',
        b'{"value": Infinity}',
        b'{"value": -Infinity}',
        b'{"value": 1e400}',
        b"[]",
        b"\xff",
    ],
)
def test_ambiguous_nonfinite_or_nonobject_json_rejected(tmp_path: Path, raw: bytes) -> None:
    path = tmp_path / "report.json"
    path.write_bytes(raw)
    with pytest.raises(ValueError):
        summary_tool.summarize_mp20_preflight(
            path,
            tmp_path / "summary.json",
            expected_report_sha256=hashlib.sha256(raw).hexdigest(),
        )
    assert not (tmp_path / "summary.json").exists()


@pytest.mark.parametrize("pin", ["b" * 64, "A" * 64, "bad", ""])
def test_external_pin_is_required_and_verified(tmp_path: Path, pin: str) -> None:
    path, _ = _write_report(tmp_path, _report())
    with pytest.raises(ValueError):
        summary_tool.summarize_mp20_preflight(
            path, tmp_path / "summary.json", expected_report_sha256=pin
        )


def test_tampered_pinned_bytes_are_not_reinterpreted(tmp_path: Path) -> None:
    path, pin = _write_report(tmp_path, _report())
    path.write_bytes(path.read_bytes() + b" ")
    with pytest.raises(ValueError, match="external pin"):
        summary_tool.summarize_mp20_preflight(
            path, tmp_path / "summary.json", expected_report_sha256=pin
        )


def test_existing_output_is_never_overwritten(tmp_path: Path) -> None:
    path, pin = _write_report(tmp_path, _report())
    output = tmp_path / "summary.json"
    output.write_bytes(b"previous evidence\n")
    with pytest.raises(FileExistsError):
        summary_tool.summarize_mp20_preflight(path, output, expected_report_sha256=pin)
    assert output.read_bytes() == b"previous evidence\n"


def test_dangling_output_symlink_is_not_followed(tmp_path: Path) -> None:
    path, pin = _write_report(tmp_path, _report())
    output, missing = tmp_path / "summary.json", tmp_path / "missing.json"
    output.symlink_to(missing)
    with pytest.raises(FileExistsError):
        summary_tool.summarize_mp20_preflight(path, output, expected_report_sha256=pin)
    assert not missing.exists()


def test_input_symlink_and_fifo_are_rejected(tmp_path: Path) -> None:
    path, pin = _write_report(tmp_path, _report())
    symlink = tmp_path / "symlink.json"
    symlink.symlink_to(path)
    with pytest.raises(OSError):
        summary_tool.summarize_mp20_preflight(
            symlink, tmp_path / "summary.json", expected_report_sha256=pin
        )
    fifo = tmp_path / "fifo.json"
    os.mkfifo(fifo)
    with pytest.raises(ValueError, match="regular file"):
        summary_tool.summarize_mp20_preflight(
            fifo, tmp_path / "summary.json", expected_report_sha256=pin
        )


def test_read_size_bound(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path, pin = _write_report(tmp_path, _report())
    monkeypatch.setattr(summary_tool, "MAX_REPORT_BYTES", 10)
    with pytest.raises(ValueError, match="byte limit"):
        summary_tool.summarize_mp20_preflight(
            path, tmp_path / "summary.json", expected_report_sha256=pin
        )


def test_cli_uses_only_stdlib_and_does_not_require_package(tmp_path: Path) -> None:
    path, pin = _write_report(tmp_path, _report())
    result = subprocess.run(
        [
            sys.executable,
            "-I",
            str(_SCRIPT),
            "--report",
            str(path),
            "--expected-report-sha256",
            pin,
            "--output",
            str(tmp_path / "summary.json"),
        ],
        check=True,
        capture_output=True,
        text=True,
        cwd=tmp_path,
    )
    assert json.loads(result.stdout)["run_state"] == "completed"
