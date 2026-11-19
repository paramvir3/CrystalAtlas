"""Offline, byte-preserving extraction of a frozen train-only MP-20 pilot.

This is an acquisition/preprocessing handoff, not a source-rights approval or a
training launcher. Existing destinations are never reused. A receipt is written
last; an interrupted directory without that receipt is incomplete and must not
be consumed. Partial output is retained for inspection rather than deleted.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any

from crystalatlas.benchmark_data import (
    BenchmarkCifRow,
    read_benchmark_cif_csv,
    select_training_pilot,
)
from crystalatlas.dataset_provenance import (
    canonical_record_index_bytes,
    canonical_record_payload_index_bytes,
)
from crystalatlas.errors import DomainError, InvariantViolation

REVISION = "f857f598d6f6cca5dc1ea0582d228f12dcc2c2ea"
BASE_URI = f"https://raw.githubusercontent.com/txie-93/cdvae/{REVISION}/data/mp_20"
FROZEN_FILES = {
    "README.md": (
        1622,
        "498076c5663bc478e41a3405e8c2da530973f06a",
        "b3ba7e4732552ee2a72424be45c271d65d238b5c0adfbdf33ae098d6213ff9b2",
    ),
    "train.csv": (
        34975028,
        "b1c9d498565ac2fd8d075a51ff2ef5e51ba6c21f",
        "133bab58dba02d316f9f91174839f17802dce648be4e8659d1a42b32b97fa01d",
    ),
    "val.csv": (
        11617279,
        "7efe4d7ebeb4f460ceaad35e5326c42111d9c9a1",
        "7b5f4085464b3eac9fd2123acee97ec372214004ce7e983b535b33395abc3e29",
    ),
    "test.csv": (
        11599133,
        "5dc643041df53f5ce4970a79930ad262b4f07acd",
        "dd605c7543cf95a13e150aed630d0915e74edc8dea17e85301ee6fbfeb7f500a",
    ),
}
SPLIT_COUNTS = {"train": 27136, "val": 9047, "test": 9046}
_DIGEST = re.compile(r"[0-9a-f]{64}\Z")
_ACQUISITION_LIMIT_BYTES = 1024 * 1024


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False
    ).encode("utf-8")


def _strict_json(raw: bytes) -> Any:
    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise DomainError("acquisition JSON repeats a key")
            result[key] = value
        return result

    def reject_nonfinite(value: str) -> object:
        raise DomainError(f"acquisition JSON has a nonfinite value: {value}")

    try:
        return json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=reject_duplicates,
            parse_constant=reject_nonfinite,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise DomainError("acquisition receipt must be strict UTF-8 JSON") from exc


def _plain_path(path: str | Path, *, must_exist: bool) -> Path:
    candidate = Path(path).absolute()
    if candidate.is_symlink() or candidate.resolve(strict=must_exist) != candidate:
        raise DomainError("pilot paths must not traverse symlinks or '..'")
    return candidate


def _read_regular(path: Path, *, maximum_bytes: int) -> bytes:
    if path.is_symlink() or not path.is_file():
        raise DomainError(f"expected a nonsymlink regular file: {path.name}")
    with path.open("rb") as stream:
        raw = stream.read(maximum_bytes + 1)
    if len(raw) > maximum_bytes:
        raise DomainError(f"input exceeds its frozen byte limit: {path.name}")
    return raw


def _verify_acquisition(source: Path, expected_sha256: str) -> tuple[dict[str, Any], ...]:
    if not isinstance(expected_sha256, str) or not _DIGEST.fullmatch(expected_sha256):
        raise DomainError("expected acquisition SHA-256 must be a lowercase digest")
    raw = _read_regular(source / "acquisition.json", maximum_bytes=_ACQUISITION_LIMIT_BYTES)
    if hashlib.sha256(raw).hexdigest() != expected_sha256:
        raise DomainError("acquisition receipt disagrees with its external byte pin")
    receipt = _strict_json(raw)
    fields = {
        "schema",
        "revision",
        "retrieved_utc",
        "artifacts",
        "training_authorized_by_this_receipt",
        "notice",
    }
    if not isinstance(receipt, dict) or set(receipt) != fields:
        raise DomainError("acquisition receipt has missing or unexpected fields")
    if (
        receipt["schema"] != "crystalatlas.mp20-byte-acquisition.v1"
        or receipt["revision"] != REVISION
        or receipt["training_authorized_by_this_receipt"] is not False
        or not isinstance(receipt["retrieved_utc"], str)
        or not isinstance(receipt["notice"], str)
        or not isinstance(receipt["artifacts"], list)
    ):
        raise DomainError("acquisition receipt has an unsupported or misleading contract")
    artifacts = receipt["artifacts"]
    if len(artifacts) != len(FROZEN_FILES):
        raise DomainError("acquisition artifact count differs from the frozen source")
    observed: dict[str, dict[str, Any]] = {}
    for artifact in artifacts:
        if not isinstance(artifact, dict) or set(artifact) != {
            "path",
            "uri",
            "byte_count",
            "git_blob_sha1",
            "sha256",
        }:
            raise DomainError("acquisition artifact fields are malformed")
        name = artifact["path"]
        if not isinstance(name, str) or name not in FROZEN_FILES or name in observed:
            raise DomainError("acquisition repeats or misidentifies a frozen artifact")
        size, blob, digest = FROZEN_FILES[name]
        expected = {
            "path": name,
            "uri": f"{BASE_URI}/{name}",
            "byte_count": size,
            "git_blob_sha1": blob,
            "sha256": digest,
        }
        if type(artifact["byte_count"]) is not int or artifact != expected:
            raise DomainError("acquisition artifact identity differs from the frozen source")
        payload = _read_regular(source / name, maximum_bytes=size)
        actual_blob = hashlib.sha1(f"blob {len(payload)}\0".encode() + payload).hexdigest()
        if (
            len(payload) != size
            or hashlib.sha256(payload).hexdigest() != digest
            or actual_blob != blob
        ):
            raise DomainError(f"frozen source artifact bytes changed: {name}")
        observed[name] = expected
    return tuple(observed[name] for name in sorted(observed))


def _row_metadata(row: BenchmarkCifRow) -> dict[str, object]:
    return {
        "schema": "crystalatlas.benchmark-cif-row.v1",
        "split": row.split,
        "row_index": row.row_index,
        "material_id": row.material_id,
        "cif_sha256": row.cif_sha256,
        "metadata": dict(row.metadata),
    }


def _write_exclusive(path: Path, payload: bytes) -> None:
    with path.open("xb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())


def prepare_mp20_pilot(
    source_root: str | Path,
    destination: str | Path,
    *,
    expected_acquisition_sha256: str,
    count: int = 512,
    seed: int = 20260913,
) -> dict[str, Any]:
    """Verify all splits, then extract exactly the preselected training rows.

    The output carries no source-scope assertions or machine-use permissions.
    Those must be supplied separately with an archived, reviewed declaration.
    Published validation and test rows are indexed, never selected or copied as
    pilot CIFs. No CIF parser, physical validity test or property filter runs here.
    """
    if isinstance(count, bool) or not isinstance(count, int) or not 1 <= count <= 512:
        raise DomainError("this bounded pilot requires between 1 and 512 selected records")
    source = _plain_path(source_root, must_exist=True)
    if not source.is_dir():
        raise DomainError("source root must be a directory")
    target = _plain_path(destination, must_exist=False)
    if target.exists():
        raise DomainError("pilot destination already exists and will not be overwritten")
    artifacts = _verify_acquisition(source, expected_acquisition_sha256)
    split_rows = {
        split: read_benchmark_cif_csv(
            source / f"{split}.csv",
            split=split,
            expected_sha256=FROZEN_FILES[f"{split}.csv"][2],
            expected_count=expected_count,
        )
        for split, expected_count in SPLIT_COUNTS.items()
    }
    all_rows = tuple(row for rows in split_rows.values() for row in rows)
    if len({row.material_id for row in all_rows}) != len(all_rows):
        raise DomainError("source material identifiers overlap across published splits")
    selected = select_training_pilot(split_rows["train"], count=count, seed=seed)
    selected_ids = frozenset(row.material_id for row in selected)
    if selected_ids.intersection(
        row.material_id for split in ("val", "test") for row in split_rows[split]
    ):
        raise InvariantViolation("a holdout identifier entered the training selection")

    selection = {
        "schema": "crystalatlas.mp20-pilot-selection-spec.v1",
        "source_revision": REVISION,
        "source_acquisition_sha256": expected_acquisition_sha256,
        "source_partition": "train",
        "count": count,
        "seed": seed,
        "rank_protocol": "sha256-json-[crystalatlas.train-pilot-selection.v1,seed,ID]",
        "rank_tie_breaker": "material_id",
        "output_order": "ascending-hash-rank-then-material-id",
        "selection_depends_on": ["seed", "material_id"],
        "selection_excludes": [
            "energy",
            "species",
            "space_group",
            "geometry",
            "target_success",
            "model_output",
        ],
        "refill_after_preprocessing_rejection": False,
        "published_holdouts_available_to_training": False,
    }
    # Validate and parse everything before creating any output. mkdir(exist_ok=False)
    # reserves a new directory; a receipt is its completion marker, written last.
    target.parent.mkdir(parents=True, exist_ok=True)
    if _plain_path(target.parent, must_exist=True) != target.parent:
        raise DomainError("pilot parent changed during preparation")
    target.mkdir(exist_ok=False)
    (target / "cifs").mkdir()
    (target / "metadata").mkdir()
    output_files: dict[str, dict[str, Any]] = {}

    def publish(name: str, payload: bytes, role: str) -> None:
        digest = hashlib.sha256(payload).hexdigest()
        if name in output_files:
            # Different source rows may have identical CIF bytes. Preserve all
            # row bindings but store that one payload once, never deduplicate IDs.
            if output_files[name]["sha256"] != digest:
                raise InvariantViolation("content-addressed payload name collided")
            return
        _write_exclusive(target / name, payload)
        output_files[name] = {
            "path": name,
            "sha256": digest,
            "byte_count": len(payload),
            "role": role,
        }

    publish("selection.json", _canonical_bytes(selection), "selection_spec")
    all_index = []
    for row in all_rows:
        metadata = _row_metadata(row)
        if hashlib.sha256(_canonical_bytes(metadata)).hexdigest() != row.metadata_sha256:
            raise InvariantViolation("row metadata serialization disagrees with its API digest")
        all_index.append(
            _canonical_bytes(
                {
                    "schema": "crystalatlas.mp20-source-row-binding.v1",
                    "source_revision": REVISION,
                    "source_container_path": f"{row.split}.csv",
                    "source_container_sha256": FROZEN_FILES[f"{row.split}.csv"][2],
                    "source_metadata_sha256": row.metadata_sha256,
                    "selected_for_pilot": row.material_id in selected_ids,
                    "row": metadata,
                }
            )
            + b"\n"
        )
    publish("source-rows.jsonl", b"".join(all_index), "all_source_rows")
    bindings = []
    for rank, row in enumerate(selected):
        cif_path = f"cifs/{row.cif_sha256}.cif"
        metadata_path = f"metadata/{row.metadata_sha256}.json"
        publish(cif_path, row.cif.encode("utf-8"), "selected_cif")
        publish(metadata_path, _canonical_bytes(_row_metadata(row)), "selected_metadata")
        bindings.append(
            {
                "selection_rank": rank,
                "source_partition": row.split,
                "source_row_index": row.row_index,
                "source_record_id": row.material_id,
                "source_container_path": "train.csv",
                "source_container_sha256": FROZEN_FILES["train.csv"][2],
                "source_uri": f"{BASE_URI}/train.csv",
                "cif_path": cif_path,
                "payload_sha256": row.cif_sha256,
                "metadata_path": metadata_path,
                "source_metadata_sha256": row.metadata_sha256,
            }
        )
    publish(
        "selected-bindings.json",
        _canonical_bytes(
            {
                "schema": "crystalatlas.mp20-selected-row-bindings.v1",
                "bindings": bindings,
            }
        ),
        "selected_row_bindings",
    )
    publish(
        "record-index.json",
        canonical_record_index_bytes(tuple(row.material_id for row in selected)),
        "record_index",
    )
    publish(
        "record-payload-index.json",
        canonical_record_payload_index_bytes(
            tuple((row.material_id, row.cif_sha256) for row in selected)
        ),
        "payload_index",
    )
    receipt = {
        "schema": "crystalatlas.mp20-pilot-extraction.v1",
        "source_revision": REVISION,
        "source_acquisition_sha256": expected_acquisition_sha256,
        "source_artifacts": list(artifacts),
        "parsed_source_counts": {split: len(rows) for split, rows in split_rows.items()},
        "indexed_source_rows": len(all_rows),
        "selected_count": len(selected),
        "selected_partition": "train",
        "seed": seed,
        "source_rights_status": "unresolved",
        "training_authorized_by_this_receipt": False,
        "notice": (
            "Historical MP-20 inspection/pilot extraction only. No source-scope or licence "
            "approval is created. A separate reviewed DatasetSourceManifest, archived "
            "evidence and external trust pins are required before production training. "
            "Original CIF payloads and property metadata are preserved; no structure or "
            "energy filtering and no holdout training occurred. This receipt is not signed."
        ),
        "files": [output_files[name] for name in sorted(output_files)],
    }
    receipt["receipt_fingerprint"] = hashlib.sha256(_canonical_bytes(receipt)).hexdigest()
    # Re-read every output before making the completion claim.
    for artifact in receipt["files"]:
        payload = _read_regular(target / artifact["path"], maximum_bytes=artifact["byte_count"])
        if (
            len(payload) != artifact["byte_count"]
            or hashlib.sha256(payload).hexdigest() != artifact["sha256"]
        ):
            raise InvariantViolation("extraction output changed before receipt publication")
    _write_exclusive(target / "extraction-receipt.json", _canonical_bytes(receipt) + b"\n")
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source_root", type=Path)
    parser.add_argument("destination", type=Path)
    parser.add_argument("--expected-acquisition-sha256", required=True)
    parser.add_argument("--count", type=int, default=512)
    parser.add_argument("--seed", type=int, default=20260913)
    arguments = parser.parse_args()
    receipt = prepare_mp20_pilot(
        arguments.source_root,
        arguments.destination,
        expected_acquisition_sha256=arguments.expected_acquisition_sha256,
        count=arguments.count,
        seed=arguments.seed,
    )
    print(
        json.dumps(
            {
                "receipt_fingerprint": receipt["receipt_fingerprint"],
                "parsed_source_counts": receipt["parsed_source_counts"],
                "selected_count": receipt["selected_count"],
                "source_rights_status": receipt["source_rights_status"],
                "training_authorized_by_this_receipt": False,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
