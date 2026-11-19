"""Deterministic, tamper-evident shards for compiled crystal datasets.

The storage boundary is intentionally narrower than a general serializer.  It
accepts only :class:`~crystalatlas.dataset_builder.CompiledDataset` and writes
the numerical tensors needed by the current supervised objectives as typed
``.npy`` files.  Ragged blocks use explicit int64 offsets; pickle and NumPy
object arrays are never used.  Exact scaffold data, identities and source
aliases live in canonical JSON alongside those tensors.

Publication is transactional at directory granularity: every file is written
and fsynced in a private sibling directory, the complete tree is audited, and
only then is that directory renamed to its final name.  Existing destinations
are never overwritten.  Readers authenticate the self-fingerprinted storage
manifest, the embedded build manifest, every file size/hash and every shard
schema before exposing read-only memory maps.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import tempfile
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from fractions import Fraction
from hashlib import sha256
from pathlib import Path, PurePosixPath
from types import MappingProxyType
from typing import TypeAlias, cast

import numpy as np
import numpy.typing as npt

from .dataset_builder import (
    CompiledDataset,
    CompiledDatasetRecord,
    DatasetBuildManifest,
    SourceRecordReference,
)
from .dataset_provenance import DatasetSourceManifest, SourceRecordScopeEvidence
from .dataset_splits import (
    DatasetSplitManifest,
    FrozenSplitKeys,
    SplitMode,
    SplitRecord,
    audit_dataset_split,
)
from .errors import DomainError, InvariantViolation
from .ingestion import CanonicalizationReceipt
from .training_targets import (
    CanonicalCatalogueTrainingTarget,
    ExactFixtureTrainingTarget,
)

FloatArray: TypeAlias = npt.NDArray[np.float64]
IntArray: TypeAlias = npt.NDArray[np.int64]
BoolArray: TypeAlias = npt.NDArray[np.bool_]

_STORAGE_SCHEMA = "crystalatlas.dataset-storage.v3"
_SHARD_METADATA_SCHEMA = "crystalatlas.dataset-storage-shard-metadata.v2"
_RECORD_INDEX_SCHEMA = "crystalatlas.dataset-storage-record-index.v1"
_MANIFEST_NAME = "storage-manifest.json"
_BUILD_MANIFEST_NAME = "build-manifest.json"
_SOURCE_MANIFEST_NAME = "source-manifest.json"
_RECORD_INDEX_NAME = "record-index.json"
_NOTICE_NAME = "NOTICE.txt"
_SPLIT_DIRECTORY = "splits"
_DIGEST_PATTERN = re.compile(r"[0-9a-f]{64}\Z")
_SHARD_NAME_PATTERN = re.compile(r"shard-[0-9]{5}\Z")
_DESTINATION_NAME_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,254}\Z")
_INT64_MIN = -(1 << 63)
_INT64_MAX = (1 << 63) - 1

_ARRAY_DTYPES: Mapping[str, np.dtype[np.generic]] = MappingProxyType(
    {
        "composition_atomic_numbers": np.dtype("u1"),
        "composition_coefficients": np.dtype("<i8"),
        "composition_offsets": np.dtype("<i8"),
        "group_legal_mask": np.dtype("?"),
        "group_legal_offsets": np.dtype("<i8"),
        "group_target_index": np.dtype("<i8"),
        "intrinsic_coordinates": np.dtype("<f8"),
        "intrinsic_offsets": np.dtype("<i8"),
        "lattice_latent": np.dtype("<f8"),
        "lattice_offsets": np.dtype("<i8"),
        "legal_action_offsets": np.dtype("<i8"),
        "legal_action_values": np.dtype("<i8"),
        "metric": np.dtype("<f8"),
        "nu": np.dtype("<i8"),
        "nu_legal_mask": np.dtype("?"),
        "nu_legal_offsets": np.dtype("<i8"),
        "nu_target_index": np.dtype("<i8"),
        "period_basis": np.dtype("<f8"),
        "period_dimensions": np.dtype("<i8"),
        "period_offsets": np.dtype("<i8"),
        "row_action_offsets": np.dtype("<i8"),
        "row_action_values": np.dtype("<i8"),
        "row_free_dimensions": np.dtype("<i8"),
        "row_legal_action_offsets": np.dtype("<i8"),
        "row_multiplicities": np.dtype("<i8"),
        "row_offsets": np.dtype("<i8"),
        "row_residual_after": np.dtype("<i8"),
        "row_residual_before": np.dtype("<i8"),
        "row_selected_action_indices": np.dtype("<i8"),
    }
)


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    )


def _canonical_json_bytes(value: object) -> bytes:
    return (_canonical_json(value) + "\n").encode("ascii")


def _fingerprint(value: object) -> str:
    return sha256(_canonical_json(value).encode("ascii")).hexdigest()


def _require_digest(value: object, *, name: str) -> str:
    if not isinstance(value, str) or _DIGEST_PATTERN.fullmatch(value) is None:
        raise DomainError(f"{name} must be a lowercase SHA-256 digest")
    return value


def _require_int(value: object, *, name: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise DomainError(f"{name} must be an integer of at least {minimum}")
    return value


def _require_text(value: object, *, name: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise DomainError(f"{name} must be nonempty stripped text")
    if any(ord(character) < 32 for character in value):
        raise DomainError(f"{name} cannot contain control characters")
    return value


def _require_fields(
    payload: object, expected: frozenset[str], *, name: str
) -> Mapping[str, object]:
    if not isinstance(payload, Mapping) or set(payload) != expected:
        raise DomainError(f"{name} has missing or unexpected fields")
    if any(not isinstance(key, str) for key in payload):
        raise DomainError(f"{name} keys must be strings")
    return cast(Mapping[str, object], payload)


def _sha256_file(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _write_bytes(path: Path, payload: bytes) -> None:
    with path.open("xb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())


def _write_npy(path: Path, array: npt.NDArray[np.generic]) -> None:
    if array.dtype.hasobject:
        raise InvariantViolation("object arrays are forbidden in dataset shards")
    with path.open("xb") as stream:
        np.save(stream, array, allow_pickle=False)
        stream.flush()
        os.fsync(stream.fileno())


def _int64(values: Sequence[int], *, name: str) -> IntArray:
    normalized: list[int] = []
    for value in values:
        if isinstance(value, bool) or not isinstance(value, int):
            raise InvariantViolation(f"{name} contains a non-integer")
        if not _INT64_MIN <= value <= _INT64_MAX:
            raise InvariantViolation(f"{name} exceeds the exact int64 storage domain")
        normalized.append(value)
    return np.asarray(normalized, dtype="<i8")


def _float64(values: npt.ArrayLike, *, name: str) -> FloatArray:
    array = np.asarray(values, dtype="<f8")
    if not np.all(np.isfinite(array)):
        raise InvariantViolation(f"{name} contains a nonfinite value")
    return cast(FloatArray, array)


def _fraction_payload(value: Fraction) -> list[int]:
    return [value.numerator, value.denominator]


def _fraction_vector_payload(values: Sequence[Fraction]) -> list[list[int]]:
    return [_fraction_payload(value) for value in values]


def _notice_bytes_from_manifest(manifest: DatasetBuildManifest) -> bytes:
    obligations = manifest.policy_decision.obligations
    attributions = tuple(
        sorted(
            {
                attribution
                for record in manifest.records
                for evidence in record.alias_scope_evidence
                for attribution in evidence.attribution
            }
        )
    )
    lines = [
        "CrystalAtlas compiled dataset notice",
        f"Build manifest fingerprint: {manifest.fingerprint}",
        f"Policy decision fingerprint: {manifest.policy_decision.fingerprint}",
        "",
        "Source licence and attribution obligations:",
    ]
    lines.extend(f"- {obligation}" for obligation in obligations)
    if not obligations:
        lines.append("- None recorded by the authorized policy decision.")
    lines.extend(("", "Record-scope attributions:"))
    lines.extend(f"- {attribution}" for attribution in attributions)
    if not attributions:
        lines.append("- None recorded in record-scope evidence.")
    return ("\n".join(lines) + "\n").encode("utf-8")


def _notice_bytes(dataset: CompiledDataset) -> bytes:
    return _notice_bytes_from_manifest(dataset.manifest)


@dataclass(frozen=True, slots=True, order=True)
class StoredFile:
    """One regular file authenticated by byte count and SHA-256."""

    relative_path: str
    byte_count: int
    sha256: str

    def __post_init__(self) -> None:
        path = _require_text(self.relative_path, name="relative_path")
        candidate = PurePosixPath(path)
        if (
            candidate.is_absolute()
            or candidate.as_posix() != path
            or "\\" in path
            or not candidate.parts
            or any(part in {"", ".", ".."} for part in candidate.parts)
        ):
            raise DomainError("stored file paths must be normalized relative POSIX paths")
        _require_int(self.byte_count, name="byte_count")
        _require_digest(self.sha256, name="file sha256")

    @classmethod
    def from_path(cls, path: Path, *, relative_path: str) -> StoredFile:
        return cls(relative_path, path.stat().st_size, _sha256_file(path))

    def to_dict(self) -> dict[str, object]:
        return {
            "relative_path": self.relative_path,
            "byte_count": self.byte_count,
            "sha256": self.sha256,
        }

    @classmethod
    def from_dict(cls, payload: object) -> StoredFile:
        value = _require_fields(
            payload,
            frozenset({"relative_path", "byte_count", "sha256"}),
            name="stored file",
        )
        return cls(
            relative_path=value["relative_path"],  # type: ignore[arg-type]
            byte_count=value["byte_count"],  # type: ignore[arg-type]
            sha256=value["sha256"],  # type: ignore[arg-type]
        )


@dataclass(frozen=True, slots=True, order=True)
class StoredSplit:
    """One authenticated split receipt retained with the training tensors."""

    mode: SplitMode
    manifest_fingerprint: str
    file: StoredFile

    def __post_init__(self) -> None:
        if not isinstance(self.mode, SplitMode):
            raise TypeError("stored split mode must be SplitMode")
        _require_digest(self.manifest_fingerprint, name="split manifest fingerprint")
        if not isinstance(self.file, StoredFile):
            raise TypeError("stored split file must be StoredFile")
        expected = f"{_SPLIT_DIRECTORY}/{self.mode.value}.json"
        if self.file.relative_path != expected:
            raise DomainError("stored split receipt has a noncanonical path")

    def to_dict(self) -> dict[str, object]:
        return {
            "mode": self.mode.value,
            "manifest_fingerprint": self.manifest_fingerprint,
            "file": self.file.to_dict(),
        }

    @classmethod
    def from_dict(cls, payload: object) -> StoredSplit:
        value = _require_fields(
            payload,
            frozenset({"mode", "manifest_fingerprint", "file"}),
            name="stored split",
        )
        try:
            mode = SplitMode(value["mode"])
        except (TypeError, ValueError) as exc:
            raise DomainError("stored split has an unknown mode") from exc
        return cls(
            mode=mode,
            manifest_fingerprint=value["manifest_fingerprint"],  # type: ignore[arg-type]
            file=StoredFile.from_dict(value["file"]),
        )


@dataclass(frozen=True, slots=True)
class StoredRecordIndexEntry:
    """Authenticated location and scientific bindings for one stored record."""

    record_id: str
    global_index: int
    shard_index: int
    local_index: int
    representative_payload_sha256: str
    target_fingerprint: str
    canonicalization_receipt_fingerprint: str

    def __post_init__(self) -> None:
        _require_digest(self.record_id, name="indexed record identifier")
        _require_int(self.global_index, name="global record index")
        _require_int(self.shard_index, name="record shard index")
        _require_int(self.local_index, name="record local index")
        for name in (
            "representative_payload_sha256",
            "target_fingerprint",
            "canonicalization_receipt_fingerprint",
        ):
            _require_digest(getattr(self, name), name=name)

    def to_dict(self) -> dict[str, object]:
        return {
            "record_id": self.record_id,
            "global_index": self.global_index,
            "shard_index": self.shard_index,
            "local_index": self.local_index,
            "representative_payload_sha256": self.representative_payload_sha256,
            "target_fingerprint": self.target_fingerprint,
            "canonicalization_receipt_fingerprint": (
                self.canonicalization_receipt_fingerprint
            ),
        }

    @classmethod
    def from_dict(cls, payload: object) -> StoredRecordIndexEntry:
        value = _require_fields(
            payload,
            frozenset(
                {
                    "record_id",
                    "global_index",
                    "shard_index",
                    "local_index",
                    "representative_payload_sha256",
                    "target_fingerprint",
                    "canonicalization_receipt_fingerprint",
                }
            ),
            name="stored record-index entry",
        )
        return cls(
            record_id=value["record_id"],  # type: ignore[arg-type]
            global_index=value["global_index"],  # type: ignore[arg-type]
            shard_index=value["shard_index"],  # type: ignore[arg-type]
            local_index=value["local_index"],  # type: ignore[arg-type]
            representative_payload_sha256=value[
                "representative_payload_sha256"
            ],  # type: ignore[arg-type]
            target_fingerprint=value["target_fingerprint"],  # type: ignore[arg-type]
            canonicalization_receipt_fingerprint=value[
                "canonicalization_receipt_fingerprint"
            ],  # type: ignore[arg-type]
        )


@dataclass(frozen=True, slots=True)
class StoredShard:
    """A contiguous interval of canonically ordered dataset records."""

    index: int
    start: int
    stop: int
    files: tuple[StoredFile, ...]

    def __post_init__(self) -> None:
        _require_int(self.index, name="shard index")
        _require_int(self.start, name="shard start")
        _require_int(self.stop, name="shard stop")
        if self.stop <= self.start:
            raise DomainError("a stored shard interval must be nonempty")
        files = tuple(self.files)
        object.__setattr__(self, "files", files)
        expected_directory = f"shard-{self.index:05d}"
        if tuple(sorted(files)) != files or len(set(files)) != len(files):
            raise DomainError("stored shard files must be sorted and unique")
        expected_names = {"metadata.json", *[f"{name}.npy" for name in _ARRAY_DTYPES]}
        actual_names = {Path(item.relative_path).name for item in files}
        if actual_names != expected_names or any(
            Path(item.relative_path).parts[0] != expected_directory for item in files
        ):
            raise DomainError("stored shard file catalogue is incomplete or inconsistent")

    @property
    def record_count(self) -> int:
        return self.stop - self.start

    def to_dict(self) -> dict[str, object]:
        return {
            "index": self.index,
            "start": self.start,
            "stop": self.stop,
            "files": [item.to_dict() for item in self.files],
        }

    @classmethod
    def from_dict(cls, payload: object) -> StoredShard:
        value = _require_fields(
            payload,
            frozenset({"index", "start", "stop", "files"}),
            name="stored shard",
        )
        files = value["files"]
        if not isinstance(files, list):
            raise DomainError("stored shard files must be a JSON array")
        return cls(
            index=value["index"],  # type: ignore[arg-type]
            start=value["start"],  # type: ignore[arg-type]
            stop=value["stop"],  # type: ignore[arg-type]
            files=tuple(StoredFile.from_dict(item) for item in files),
        )


@dataclass(frozen=True, slots=True)
class DatasetStorageManifest:
    """Self-authenticating root binding shards to one build manifest."""

    build_manifest_fingerprint: str
    build_manifest_byte_count: int
    build_manifest_sha256: str
    source_manifest_fingerprint: str
    source_manifest_byte_count: int
    source_manifest_sha256: str
    record_index: StoredFile
    notice: StoredFile
    record_count: int
    records_per_shard: int
    shards: tuple[StoredShard, ...]
    splits: tuple[StoredSplit, ...]
    storage_fingerprint: str | None = None

    def __post_init__(self) -> None:
        _require_digest(self.build_manifest_fingerprint, name="build manifest fingerprint")
        _require_int(self.build_manifest_byte_count, name="build manifest byte count")
        _require_digest(self.build_manifest_sha256, name="build manifest file hash")
        _require_digest(self.source_manifest_fingerprint, name="source manifest fingerprint")
        _require_int(self.source_manifest_byte_count, name="source manifest byte count")
        _require_digest(self.source_manifest_sha256, name="source manifest file hash")
        if not isinstance(self.record_index, StoredFile):
            raise TypeError("record_index must be StoredFile")
        if self.record_index.relative_path != _RECORD_INDEX_NAME:
            raise DomainError("record index has a noncanonical path")
        if not isinstance(self.notice, StoredFile):
            raise TypeError("notice must be StoredFile")
        if self.notice.relative_path != _NOTICE_NAME:
            raise DomainError("dataset notice has a noncanonical path")
        _require_int(self.record_count, name="record count")
        _require_int(self.records_per_shard, name="records_per_shard", minimum=1)
        shards = tuple(self.shards)
        object.__setattr__(self, "shards", shards)
        cursor = 0
        for expected_index, shard in enumerate(shards):
            if expected_index >= 100_000:
                raise DomainError("dataset storage exceeds the five-digit shard namespace")
            if shard.index != expected_index or shard.start != cursor:
                raise DomainError("stored shards do not form one canonical contiguous sequence")
            if shard.record_count > self.records_per_shard:
                raise DomainError("stored shard exceeds records_per_shard")
            if (
                expected_index < len(shards) - 1
                and shard.record_count != self.records_per_shard
            ):
                raise DomainError("every non-final shard must have records_per_shard records")
            cursor = shard.stop
        if cursor != self.record_count:
            raise DomainError("stored shard intervals do not cover the declared record count")
        if bool(shards) != bool(self.record_count):
            raise DomainError("zero records require zero shards and vice versa")
        splits = tuple(self.splits)
        object.__setattr__(self, "splits", splits)
        if tuple(sorted(splits, key=lambda item: item.mode.value)) != splits:
            raise DomainError("stored splits must be in canonical mode order")
        if len({item.mode for item in splits}) != len(splits):
            raise DomainError("stored split modes must be unique")
        expected = _fingerprint(self._unsigned_payload())
        if self.storage_fingerprint is None:
            object.__setattr__(self, "storage_fingerprint", expected)
        elif self.storage_fingerprint != expected:
            raise DomainError("dataset storage manifest is internally inconsistent")

    @property
    def fingerprint(self) -> str:
        assert self.storage_fingerprint is not None
        return self.storage_fingerprint

    def _unsigned_payload(self) -> dict[str, object]:
        return {
            "schema": _STORAGE_SCHEMA,
            "build_manifest": {
                "relative_path": _BUILD_MANIFEST_NAME,
                "byte_count": self.build_manifest_byte_count,
                "sha256": self.build_manifest_sha256,
                "fingerprint": self.build_manifest_fingerprint,
            },
            "source_manifest": {
                "relative_path": _SOURCE_MANIFEST_NAME,
                "byte_count": self.source_manifest_byte_count,
                "sha256": self.source_manifest_sha256,
                "fingerprint": self.source_manifest_fingerprint,
            },
            "record_index": self.record_index.to_dict(),
            "notice": self.notice.to_dict(),
            "record_count": self.record_count,
            "records_per_shard": self.records_per_shard,
            "shards": [shard.to_dict() for shard in self.shards],
            "splits": [split.to_dict() for split in self.splits],
        }

    def to_dict(self) -> dict[str, object]:
        return {**self._unsigned_payload(), "storage_fingerprint": self.fingerprint}

    @classmethod
    def from_dict(cls, payload: object) -> DatasetStorageManifest:
        value = _require_fields(
            payload,
            frozenset(
                {
                    "schema",
                    "build_manifest",
                    "source_manifest",
                    "record_index",
                    "notice",
                    "record_count",
                    "records_per_shard",
                    "shards",
                    "splits",
                    "storage_fingerprint",
                }
            ),
            name="dataset storage manifest",
        )
        if value["schema"] != _STORAGE_SCHEMA:
            raise DomainError("unsupported dataset storage schema")
        build = _require_fields(
            value["build_manifest"],
            frozenset({"relative_path", "byte_count", "sha256", "fingerprint"}),
            name="stored build manifest",
        )
        if build["relative_path"] != _BUILD_MANIFEST_NAME:
            raise DomainError("build manifest has a noncanonical path")
        source = _require_fields(
            value["source_manifest"],
            frozenset({"relative_path", "byte_count", "sha256", "fingerprint"}),
            name="stored source manifest",
        )
        if source["relative_path"] != _SOURCE_MANIFEST_NAME:
            raise DomainError("source manifest has a noncanonical path")
        shards = value["shards"]
        splits = value["splits"]
        if not isinstance(shards, list) or not isinstance(splits, list):
            raise DomainError("dataset storage shards/splits must be JSON arrays")
        return cls(
            build_manifest_fingerprint=build["fingerprint"],  # type: ignore[arg-type]
            build_manifest_byte_count=build["byte_count"],  # type: ignore[arg-type]
            build_manifest_sha256=build["sha256"],  # type: ignore[arg-type]
            source_manifest_fingerprint=source["fingerprint"],  # type: ignore[arg-type]
            source_manifest_byte_count=source["byte_count"],  # type: ignore[arg-type]
            source_manifest_sha256=source["sha256"],  # type: ignore[arg-type]
            record_index=StoredFile.from_dict(value["record_index"]),
            notice=StoredFile.from_dict(value["notice"]),
            record_count=value["record_count"],  # type: ignore[arg-type]
            records_per_shard=value["records_per_shard"],  # type: ignore[arg-type]
            shards=tuple(StoredShard.from_dict(item) for item in shards),
            splits=tuple(StoredSplit.from_dict(item) for item in splits),
            storage_fingerprint=value["storage_fingerprint"],  # type: ignore[arg-type]
        )


def _identity_metadata(record: CompiledDatasetRecord) -> dict[str, object]:
    identity = record.identity
    return {
        "exact_key": identity.exact_key,
        "structure_key": identity.structure_key,
        "polymorph_key": identity.polymorph_key,
        "scaffold_key": identity.scaffold_key,
        "prototype_key": identity.prototype_key,
        "composition_key": identity.composition_key,
        "element_set_key": identity.element_set_key,
        "policy_fingerprint": identity.policy_fingerprint,
        "catalogue_fingerprint": identity.catalogue_fingerprint,
        "metric_bins": list(identity.metric_bins),
        "shape_bins": list(identity.shape_bins),
        "exact_payload": identity.exact_payload,
        "structure_payload": identity.structure_payload,
        "polymorph_payload": identity.polymorph_payload,
    }


def _record_metadata(record: CompiledDatasetRecord) -> dict[str, object]:
    target = record.canonical_record.target
    if isinstance(target, ExactFixtureTrainingTarget):
        target_type = "exact-fixture"
    elif isinstance(target, CanonicalCatalogueTrainingTarget):
        target_type = "canonical-catalogue"
    else:  # pragma: no cover - CompiledDataset's closed target alias protects this
        raise TypeError("unsupported canonical training target")
    scaffold = target.canonical_scaffold
    receipt = record.canonical_record.canonicalized.receipt
    summary = record.manifest_summary()
    return {
        "record_id": record.record_id,
        "representative_payload_sha256": record.representative_payload_sha256,
        "target_fingerprint": summary.target_fingerprint,
        "canonicalization_receipt_fingerprint": (
            summary.canonicalization_receipt_fingerprint
        ),
        "target_type": target_type,
        "space_group": target.cover.space_group,
        "row_labels": [item.label for item in target.cover_decisions.rows],
        "orbits": [
            {
                "wyckoff_label": assignment.wyckoff_label,
                "atomic_number": int(assignment.species),
                "parameter": [_fraction_payload(value) for value in assignment.parameter],
            }
            for assignment in scaffold.orbits
        ],
        "aliases": [alias.to_dict() for alias in record.aliases],
        "alias_scope_evidence": [
            evidence.to_dict() for evidence in record.alias_scope_evidence
        ],
        "identity": _identity_metadata(record),
        "canonicalization": _canonicalization_payload(receipt),
    }


def _canonicalization_payload(receipt: CanonicalizationReceipt) -> dict[str, object]:
    """Stored receipt; optional idealization metadata is written only when present."""

    payload: dict[str, object] = {
        "source_fingerprint": receipt.source_fingerprint,
        "standardization_fingerprint": receipt.standardization_fingerprint,
        "catalogue_fingerprint": receipt.catalogue_fingerprint,
        "hall_number": receipt.hall_number,
        "hall_symbol": receipt.hall_symbol,
        "source_to_primitive": [
            _fraction_vector_payload(row) for row in receipt.source_to_primitive
        ],
        "origin_shift": _fraction_vector_payload(receipt.origin_shift),
        "free_origin_shift": _fraction_vector_payload(receipt.free_origin_shift),
        "translation_index": receipt.translation_index,
        "symmetry_tolerance": receipt.symmetry_tolerance.hex(),
        "roundtrip_max_cartesian_error": (
            receipt.roundtrip_max_cartesian_error.hex()
        ),
        "symmetry_stability_verified": receipt.symmetry_stability_verified,
        "catalogue_closure_verified": receipt.catalogue_closure_verified,
        "wyckoff_multiplicity_verified": receipt.wyckoff_multiplicity_verified,
        "symmetry_verified": receipt.symmetry_verified,
        "canonical_origin_verified": receipt.canonical_origin_verified,
        "scaffold_roundtrip_verified": receipt.scaffold_roundtrip_verified,
    }
    if receipt.idealization_lattice_displacement_angstrom is not None:
        assert receipt.idealization_site_displacement_angstrom is not None
        payload["idealization_lattice_displacement_angstrom"] = (
            receipt.idealization_lattice_displacement_angstrom.hex()
        )
        payload["idealization_site_displacement_angstrom"] = (
            receipt.idealization_site_displacement_angstrom.hex()
        )
    return payload


def _append_vector(values: list[float], offsets: list[int], vector: npt.ArrayLike) -> None:
    array = _float64(vector, name="continuous target vector").reshape(-1)
    values.extend(float(item) for item in array)
    offsets.append(len(values))


def _build_shard_arrays(
    records: Sequence[CompiledDatasetRecord],
) -> tuple[dict[str, npt.NDArray[np.generic]], list[dict[str, object]]]:
    metrics: list[FloatArray] = []
    lattice_values: list[float] = []
    lattice_offsets = [0]
    intrinsic_values: list[float] = []
    intrinsic_offsets = [0]
    period_values: list[float] = []
    period_offsets = [0]
    period_dimensions: list[int] = []
    composition_z: list[int] = []
    composition_n: list[int] = []
    composition_offsets = [0]
    nu_values: list[int] = []
    nu_masks: list[bool] = []
    nu_mask_offsets = [0]
    nu_targets: list[int] = []
    group_masks: list[bool] = []
    group_mask_offsets = [0]
    group_targets: list[int] = []
    row_offsets = [0]
    row_multiplicities: list[int] = []
    row_free_dimensions: list[int] = []
    row_selected_indices: list[int] = []
    row_action_values: list[int] = []
    row_action_offsets = [0]
    residual_before: list[int] = []
    residual_after: list[int] = []
    legal_values: list[int] = []
    legal_action_offsets = [0]
    row_legal_action_offsets = [0]
    metadata: list[dict[str, object]] = []

    for record in records:
        target = record.canonical_record.target
        metrics.append(_float64(target.metric, name="metric"))
        _append_vector(lattice_values, lattice_offsets, target.lattice_latent)
        _append_vector(intrinsic_values, intrinsic_offsets, target.intrinsic_coordinates)
        period = _float64(target.period_basis, name="period_basis")
        dimension = int(target.intrinsic_coordinates.shape[0])
        if period.shape != (dimension, dimension):
            raise InvariantViolation("period basis and intrinsic dimension disagree")
        period_values.extend(float(item) for item in period.reshape(-1))
        period_offsets.append(len(period_values))
        period_dimensions.append(dimension)

        for atomic_number, coefficient in target.composition.counts:
            composition_z.append(int(atomic_number))
            composition_n.append(coefficient)
        composition_offsets.append(len(composition_z))
        nu_values.append(target.cover.nu)

        decisions = target.cover_decisions
        nu_masks.extend(decisions.nu_legal_mask)
        nu_mask_offsets.append(len(nu_masks))
        nu_targets.append(decisions.nu_target_index)
        group_masks.extend(decisions.group_legal_mask)
        group_mask_offsets.append(len(group_masks))
        group_targets.append(decisions.group_target_index)

        for decision in decisions.row_decisions:
            row_multiplicities.append(decision.row.multiplicity)
            row_free_dimensions.append(decision.row.free_dimension)
            row_selected_indices.append(decision.selected_action_index)
            row_action_values.extend(decision.selected_action.counts)
            row_action_offsets.append(len(row_action_values))
            residual_before.extend(decision.residual_before)
            residual_after.extend(decision.residual_after)
            for action in decision.legal_actions:
                legal_values.extend(action.counts)
                legal_action_offsets.append(len(legal_values))
            row_legal_action_offsets.append(len(legal_action_offsets) - 1)
        row_offsets.append(len(row_multiplicities))
        metadata.append(_record_metadata(record))

    arrays: dict[str, npt.NDArray[np.generic]] = {
        "composition_atomic_numbers": np.asarray(composition_z, dtype="u1"),
        "composition_coefficients": _int64(composition_n, name="composition coefficients"),
        "composition_offsets": _int64(composition_offsets, name="composition offsets"),
        "group_legal_mask": np.asarray(group_masks, dtype="?"),
        "group_legal_offsets": _int64(group_mask_offsets, name="group-mask offsets"),
        "group_target_index": _int64(group_targets, name="group targets"),
        "intrinsic_coordinates": _float64(intrinsic_values, name="intrinsic coordinates"),
        "intrinsic_offsets": _int64(intrinsic_offsets, name="intrinsic offsets"),
        "lattice_latent": _float64(lattice_values, name="lattice latent"),
        "lattice_offsets": _int64(lattice_offsets, name="lattice offsets"),
        "legal_action_offsets": _int64(legal_action_offsets, name="legal-action offsets"),
        "legal_action_values": _int64(legal_values, name="legal-action values"),
        "metric": (
            np.stack(metrics).astype("<f8", copy=False)
            if metrics
            else np.empty((0, 3, 3), dtype="<f8")
        ),
        "nu": _int64(nu_values, name="formula-unit multipliers"),
        "nu_legal_mask": np.asarray(nu_masks, dtype="?"),
        "nu_legal_offsets": _int64(nu_mask_offsets, name="nu-mask offsets"),
        "nu_target_index": _int64(nu_targets, name="nu targets"),
        "period_basis": _float64(period_values, name="period basis"),
        "period_dimensions": _int64(period_dimensions, name="period dimensions"),
        "period_offsets": _int64(period_offsets, name="period offsets"),
        "row_action_offsets": _int64(row_action_offsets, name="row-action offsets"),
        "row_action_values": _int64(row_action_values, name="row-action values"),
        "row_free_dimensions": _int64(row_free_dimensions, name="row free dimensions"),
        "row_legal_action_offsets": _int64(
            row_legal_action_offsets, name="row legal-action offsets"
        ),
        "row_multiplicities": _int64(row_multiplicities, name="row multiplicities"),
        "row_offsets": _int64(row_offsets, name="row offsets"),
        "row_residual_after": _int64(residual_after, name="row residual-after values"),
        "row_residual_before": _int64(residual_before, name="row residual-before values"),
        "row_selected_action_indices": _int64(
            row_selected_indices, name="selected row-action indices"
        ),
    }
    if set(arrays) != set(_ARRAY_DTYPES):
        raise InvariantViolation("internal shard array catalogue is inconsistent")
    return arrays, metadata


def _validate_dataset(dataset: CompiledDataset) -> None:
    if not isinstance(dataset, CompiledDataset):
        raise TypeError("dataset must be a CompiledDataset")
    if dataset.manifest.output_count != len(dataset.records):
        raise InvariantViolation("build manifest record count disagrees with compiled records")
    summaries = tuple(record.manifest_summary() for record in dataset.records)
    if dataset.manifest.records != summaries:
        raise InvariantViolation("build manifest summaries disagree with compiled records")
    if dataset.manifest.policy_decision != dataset.policy_decision:
        raise InvariantViolation("build manifest is bound to a different policy decision")
    if dataset.manifest.source_manifest_fingerprint != dataset.source_manifest.fingerprint:
        raise InvariantViolation("build manifest is bound to different source receipts")
    if dataset.rejections != dataset.manifest.rejections:
        raise InvariantViolation("build manifest rejections disagree with compiled dataset")


def _record_index_entries(
    dataset: CompiledDataset, *, records_per_shard: int
) -> tuple[StoredRecordIndexEntry, ...]:
    entries: list[StoredRecordIndexEntry] = []
    for global_index, record in enumerate(dataset.records):
        summary = dataset.manifest.records[global_index]
        shard_index, local_index = divmod(global_index, records_per_shard)
        entries.append(
            StoredRecordIndexEntry(
                record_id=record.record_id,
                global_index=global_index,
                shard_index=shard_index,
                local_index=local_index,
                representative_payload_sha256=summary.representative_payload_sha256,
                target_fingerprint=summary.target_fingerprint,
                canonicalization_receipt_fingerprint=(
                    summary.canonicalization_receipt_fingerprint
                ),
            )
        )
    return tuple(entries)


def _record_index_bytes(
    dataset: CompiledDataset, *, records_per_shard: int
) -> bytes:
    unsigned: dict[str, object] = {
        "schema": _RECORD_INDEX_SCHEMA,
        "build_manifest_fingerprint": dataset.manifest.fingerprint,
        "record_count": len(dataset.records),
        "records": [
            entry.to_dict()
            for entry in _record_index_entries(
                dataset, records_per_shard=records_per_shard
            )
        ],
    }
    return _canonical_json_bytes(
        {**unsigned, "index_fingerprint": _fingerprint(unsigned)}
    )


def write_compiled_dataset(
    dataset: CompiledDataset,
    destination: str | Path,
    *,
    records_per_shard: int = 1024,
) -> DatasetStorageManifest:
    """Atomically publish deterministic, authenticated training shards.

    ``destination`` must not exist.  If construction or validation fails, the
    private temporary tree is removed and no dataset is published.
    """

    _validate_dataset(dataset)
    _require_int(records_per_shard, name="records_per_shard", minimum=1)
    requested = Path(destination)
    if _DESTINATION_NAME_PATTERN.fullmatch(requested.name) is None:
        raise DomainError("destination must have a safe, portable dataset-directory name")
    requested.parent.mkdir(parents=True, exist_ok=True)
    parent = requested.parent.resolve(strict=True)
    final = parent / requested.name
    if final.exists() or final.is_symlink():
        raise DomainError("dataset storage destination already exists")

    temporary = Path(tempfile.mkdtemp(prefix=f".{requested.name}.tmp-", dir=parent))
    committed = False
    try:
        build_payload = _canonical_json_bytes(dataset.manifest.to_dict())
        build_path = temporary / _BUILD_MANIFEST_NAME
        _write_bytes(build_path, build_payload)
        source_payload = _canonical_json_bytes(dataset.source_manifest.to_dict())
        source_path = temporary / _SOURCE_MANIFEST_NAME
        _write_bytes(source_path, source_payload)
        record_index_path = temporary / _RECORD_INDEX_NAME
        _write_bytes(
            record_index_path,
            _record_index_bytes(dataset, records_per_shard=records_per_shard),
        )
        notice_path = temporary / _NOTICE_NAME
        _write_bytes(notice_path, _notice_bytes(dataset))

        shards: list[StoredShard] = []
        for index, start in enumerate(range(0, len(dataset.records), records_per_shard)):
            stop = min(start + records_per_shard, len(dataset.records))
            shard_name = f"shard-{index:05d}"
            shard_path = temporary / shard_name
            shard_path.mkdir()
            arrays, record_metadata = _build_shard_arrays(dataset.records[start:stop])
            for name in sorted(arrays):
                _write_npy(shard_path / f"{name}.npy", arrays[name])
            metadata_payload = {
                "schema": _SHARD_METADATA_SCHEMA,
                "index": index,
                "start": start,
                "stop": stop,
                "records": record_metadata,
            }
            _write_bytes(shard_path / "metadata.json", _canonical_json_bytes(metadata_payload))
            _fsync_directory(shard_path)
            files = tuple(
                StoredFile.from_path(path, relative_path=f"{shard_name}/{path.name}")
                for path in sorted(shard_path.iterdir(), key=lambda item: item.name)
            )
            shards.append(StoredShard(index, start, stop, files))

        stored_splits: list[StoredSplit] = []
        if dataset.splits:
            split_root = temporary / _SPLIT_DIRECTORY
            split_root.mkdir()
            for mode, split in dataset.splits:
                relative_path = f"{_SPLIT_DIRECTORY}/{mode.value}.json"
                split_path = temporary.joinpath(*PurePosixPath(relative_path).parts)
                _write_bytes(split_path, _canonical_json_bytes(split.manifest.to_dict()))
                stored_splits.append(
                    StoredSplit(
                        mode=mode,
                        manifest_fingerprint=split.manifest.manifest_fingerprint,
                        file=StoredFile.from_path(
                            split_path,
                            relative_path=relative_path,
                        ),
                    )
                )
            _fsync_directory(split_root)

        manifest = DatasetStorageManifest(
            build_manifest_fingerprint=dataset.manifest.fingerprint,
            build_manifest_byte_count=len(build_payload),
            build_manifest_sha256=sha256(build_payload).hexdigest(),
            source_manifest_fingerprint=dataset.source_manifest.fingerprint,
            source_manifest_byte_count=len(source_payload),
            source_manifest_sha256=sha256(source_payload).hexdigest(),
            record_index=StoredFile.from_path(
                record_index_path,
                relative_path=_RECORD_INDEX_NAME,
            ),
            notice=StoredFile.from_path(notice_path, relative_path=_NOTICE_NAME),
            record_count=len(dataset.records),
            records_per_shard=records_per_shard,
            shards=tuple(shards),
            splits=tuple(stored_splits),
        )
        _write_bytes(
            temporary / _MANIFEST_NAME,
            _canonical_json_bytes(manifest.to_dict()),
        )
        _fsync_directory(temporary)

        # Authenticate the complete private tree with the same reader used by consumers.
        DatasetStore.open(
            temporary,
            expected_build_manifest_fingerprint=dataset.manifest.fingerprint,
            expected_source_manifest_fingerprint=dataset.source_manifest.fingerprint,
            expected_storage_fingerprint=manifest.fingerprint,
        ).close()
        if final.exists() or final.is_symlink():
            raise DomainError("dataset storage destination appeared during construction")
        temporary.rename(final)
        committed = True
        _fsync_directory(parent)
        return manifest
    finally:
        if not committed and temporary.exists():
            shutil.rmtree(temporary)


@dataclass(frozen=True, slots=True)
class StoredIdentity:
    exact_key: str
    structure_key: str
    polymorph_key: str
    scaffold_key: str
    prototype_key: str
    composition_key: str
    element_set_key: str
    policy_fingerprint: str
    catalogue_fingerprint: str
    metric_bins: tuple[int, ...]
    shape_bins: tuple[int, ...]
    exact_payload: str
    structure_payload: str
    polymorph_payload: str


@dataclass(frozen=True, slots=True)
class StoredOrbit:
    wyckoff_label: str
    atomic_number: int
    parameter: tuple[Fraction, ...]


@dataclass(frozen=True, slots=True)
class StoredRowDecision:
    row_label: str
    multiplicity: int
    free_dimension: int
    selected_action_index: int
    selected_action: IntArray
    legal_actions: IntArray
    residual_before: IntArray
    residual_after: IntArray


@dataclass(frozen=True, slots=True)
class StoredTrainingRecord:
    """One read-only, mostly zero-copy training view into a mapped shard."""

    record_id: str
    target_type: str
    space_group: str
    aliases: tuple[SourceRecordReference, ...]
    alias_scope_evidence: tuple[SourceRecordScopeEvidence, ...]
    identity: StoredIdentity
    representative_payload_sha256: str
    target_fingerprint: str
    canonicalization_receipt_fingerprint: str
    canonicalization: CanonicalizationReceipt
    orbits: tuple[StoredOrbit, ...]
    atomic_numbers: npt.NDArray[np.uint8]
    composition_coefficients: IntArray
    nu: int
    metric: FloatArray
    lattice_latent: FloatArray
    intrinsic_coordinates: FloatArray
    period_basis: FloatArray
    nu_legal_mask: BoolArray
    nu_target_index: int
    group_legal_mask: BoolArray
    group_target_index: int
    row_decisions: tuple[StoredRowDecision, ...]


def _reject_duplicate_json_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
    value: dict[str, object] = {}
    for key, item in pairs:
        if key in value:
            raise ValueError(f"duplicate JSON key {key!r}")
        value[key] = item
    return value


def _reject_json_constant(value: str) -> object:
    raise ValueError(f"nonfinite JSON number {value!r}")


def _read_json(path: Path, *, name: str) -> object:
    try:
        return json.loads(
            path.read_text(encoding="ascii"),
            object_pairs_hook=_reject_duplicate_json_keys,
            parse_constant=_reject_json_constant,
        )
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as exc:
        raise DomainError(f"cannot read {name}") from exc


def _verify_regular_file(root: Path, item: StoredFile) -> None:
    path = root.joinpath(*PurePosixPath(item.relative_path).parts)
    if path.is_symlink() or not path.is_file():
        raise DomainError(
            "stored artifact is missing, linked, or non-regular: " + item.relative_path
        )
    if path.stat().st_size != item.byte_count:
        raise DomainError(f"stored artifact byte count changed: {item.relative_path}")
    if _sha256_file(path) != item.sha256:
        raise DomainError(f"stored artifact hash changed: {item.relative_path}")


def _parse_record_index(
    payload: object,
    *,
    storage_manifest: DatasetStorageManifest,
    build_manifest: DatasetBuildManifest,
) -> tuple[StoredRecordIndexEntry, ...]:
    value = _require_fields(
        payload,
        frozenset(
            {
                "schema",
                "build_manifest_fingerprint",
                "record_count",
                "records",
                "index_fingerprint",
            }
        ),
        name="stored record index",
    )
    if value["schema"] != _RECORD_INDEX_SCHEMA:
        raise DomainError("unsupported stored record-index schema")
    records = value["records"]
    if not isinstance(records, list):
        raise DomainError("stored record-index records must be a JSON array")
    entries = tuple(StoredRecordIndexEntry.from_dict(item) for item in records)
    supplied_fingerprint = _require_digest(
        value["index_fingerprint"], name="record-index fingerprint"
    )
    unsigned = {
        "schema": value["schema"],
        "build_manifest_fingerprint": value["build_manifest_fingerprint"],
        "record_count": value["record_count"],
        "records": [entry.to_dict() for entry in entries],
    }
    if supplied_fingerprint != _fingerprint(unsigned):
        raise DomainError("stored record index is internally inconsistent")
    if (
        value["build_manifest_fingerprint"] != build_manifest.fingerprint
        or value["record_count"] != storage_manifest.record_count
        or len(entries) != storage_manifest.record_count
    ):
        raise DomainError("stored record index belongs to a different dataset")
    for global_index, (entry, summary) in enumerate(
        zip(entries, build_manifest.records, strict=True)
    ):
        shard_index, local_index = divmod(
            global_index, storage_manifest.records_per_shard
        )
        if (
            entry.global_index != global_index
            or entry.shard_index != shard_index
            or entry.local_index != local_index
            or entry.record_id != summary.record_id
            or entry.representative_payload_sha256
            != summary.representative_payload_sha256
            or entry.target_fingerprint != summary.target_fingerprint
            or entry.canonicalization_receipt_fingerprint
            != summary.canonicalization_receipt_fingerprint
        ):
            raise DomainError("stored record index disagrees with the build manifest")
    return entries


def _validate_offsets(
    offsets: npt.NDArray[np.generic],
    *,
    expected_length: int,
    terminal: int,
    name: str,
) -> None:
    if offsets.ndim != 1 or len(offsets) != expected_length:
        raise DomainError(f"{name} has the wrong shape")
    if len(offsets) == 0 or int(offsets[0]) != 0 or int(offsets[-1]) != terminal:
        raise DomainError(f"{name} has invalid endpoints")
    integer_offsets = cast(IntArray, offsets)
    if np.any(integer_offsets[1:] < integer_offsets[:-1]):
        raise DomainError(f"{name} is not monotone")


@dataclass(slots=True)
class _MappedShard:
    descriptor: StoredShard
    arrays: Mapping[str, npt.NDArray[np.generic]]
    metadata: tuple[Mapping[str, object], ...]

    @classmethod
    def open(cls, root: Path, descriptor: StoredShard) -> _MappedShard:
        shard_path = root / f"shard-{descriptor.index:05d}"
        arrays: dict[str, npt.NDArray[np.generic]] = {}
        for name, dtype in _ARRAY_DTYPES.items():
            try:
                value = np.load(shard_path / f"{name}.npy", mmap_mode="r", allow_pickle=False)
            except (OSError, ValueError) as exc:
                raise DomainError(f"cannot map shard array {name}") from exc
            if value.dtype != dtype or value.dtype.hasobject:
                raise DomainError(f"shard array {name} has an unsupported dtype")
            arrays[name] = value

        payload = _require_fields(
            _read_json(shard_path / "metadata.json", name="shard metadata"),
            frozenset({"schema", "index", "start", "stop", "records"}),
            name="shard metadata",
        )
        if (
            payload["schema"] != _SHARD_METADATA_SCHEMA
            or payload["index"] != descriptor.index
            or payload["start"] != descriptor.start
            or payload["stop"] != descriptor.stop
            or not isinstance(payload["records"], list)
        ):
            raise DomainError("shard metadata disagrees with its authenticated descriptor")
        raw_metadata = cast(list[object], payload["records"])
        if len(raw_metadata) != descriptor.record_count or any(
            not isinstance(item, Mapping) for item in raw_metadata
        ):
            raise DomainError("shard metadata record count is malformed")
        metadata = tuple(cast(Mapping[str, object], item) for item in raw_metadata)
        cls._validate_array_shapes(arrays, descriptor.record_count, metadata)
        return cls(descriptor, MappingProxyType(arrays), metadata)

    @staticmethod
    def _validate_array_shapes(
        arrays: Mapping[str, npt.NDArray[np.generic]],
        count: int,
        metadata: tuple[Mapping[str, object], ...],
    ) -> None:
        one_per_record = ("nu", "nu_target_index", "group_target_index", "period_dimensions")
        if arrays["metric"].shape != (count, 3, 3):
            raise DomainError("metric shard has the wrong shape")
        if any(arrays[name].shape != (count,) for name in one_per_record):
            raise DomainError("fixed per-record shard array has the wrong shape")
        for name in _ARRAY_DTYPES:
            if name != "metric" and arrays[name].ndim != 1:
                raise DomainError(f"shard array {name} must be one-dimensional")
        pairs = (
            ("composition_offsets", "composition_atomic_numbers"),
            ("intrinsic_offsets", "intrinsic_coordinates"),
            ("lattice_offsets", "lattice_latent"),
            ("nu_legal_offsets", "nu_legal_mask"),
            ("group_legal_offsets", "group_legal_mask"),
            ("period_offsets", "period_basis"),
            ("row_offsets", "row_multiplicities"),
        )
        for offset_name, value_name in pairs:
            _validate_offsets(
                arrays[offset_name],
                expected_length=count + 1,
                terminal=len(arrays[value_name]),
                name=offset_name,
            )
        if len(arrays["composition_coefficients"]) != len(
            arrays["composition_atomic_numbers"]
        ):
            raise DomainError("composition coefficient arrays do not align")
        row_count = len(arrays["row_multiplicities"])
        for name in (
            "row_free_dimensions",
            "row_selected_action_indices",
            "row_legal_action_offsets",
        ):
            expected = row_count + 1 if name == "row_legal_action_offsets" else row_count
            if len(arrays[name]) != expected:
                raise DomainError(f"row array {name} does not align")
        _validate_offsets(
            arrays["row_action_offsets"],
            expected_length=row_count + 1,
            terminal=len(arrays["row_action_values"]),
            name="row_action_offsets",
        )
        if len(arrays["row_residual_before"]) != len(arrays["row_action_values"]) or len(
            arrays["row_residual_after"]
        ) != len(arrays["row_action_values"]):
            raise DomainError("row residual arrays do not align with selected actions")
        legal_count = len(arrays["legal_action_offsets"]) - 1
        _validate_offsets(
            arrays["legal_action_offsets"],
            expected_length=legal_count + 1,
            terminal=len(arrays["legal_action_values"]),
            name="legal_action_offsets",
        )
        _validate_offsets(
            arrays["row_legal_action_offsets"],
            expected_length=row_count + 1,
            terminal=legal_count,
            name="row_legal_action_offsets",
        )
        atomic_numbers = cast(
            npt.NDArray[np.uint8], arrays["composition_atomic_numbers"]
        )
        if np.any(atomic_numbers == 0):
            raise DomainError("stored composition contains atomic number zero")
        if np.any(atomic_numbers > 118):
            raise DomainError("stored composition contains an invalid atomic number")
        composition_coefficients = cast(IntArray, arrays["composition_coefficients"])
        nu = cast(IntArray, arrays["nu"])
        period_dimensions = cast(IntArray, arrays["period_dimensions"])
        if np.any(composition_coefficients <= 0):
            raise DomainError("stored composition coefficients must be positive")
        if np.any(nu <= 0) or np.any(period_dimensions < 0):
            raise DomainError("stored target dimensions are outside their domains")
        for name in (
            "metric",
            "lattice_latent",
            "intrinsic_coordinates",
            "period_basis",
        ):
            if not np.all(np.isfinite(arrays[name])):
                raise DomainError(f"stored continuous tensor {name} contains nonfinite values")
        metrics = cast(FloatArray, arrays["metric"])
        if not np.array_equal(metrics, np.swapaxes(metrics, 1, 2)):
            raise DomainError("stored metrics must be exactly symmetric")
        try:
            eigenvalues = np.linalg.eigvalsh(metrics)
        except np.linalg.LinAlgError as exc:
            raise DomainError("stored metric eigendecomposition failed") from exc
        if np.any(eigenvalues <= 0.0):
            raise DomainError("stored metrics must be positive definite")
        for name in (
            "row_multiplicities",
            "row_free_dimensions",
            "row_selected_action_indices",
            "row_action_values",
            "legal_action_values",
            "row_residual_before",
            "row_residual_after",
        ):
            exact_values = cast(IntArray, arrays[name])
            if np.any(exact_values < 0):
                raise DomainError(f"stored exact tensor {name} contains negative values")
        row_multiplicities = cast(IntArray, arrays["row_multiplicities"])
        row_free_dimensions = cast(IntArray, arrays["row_free_dimensions"])
        if np.any(row_multiplicities <= 0):
            raise DomainError("stored Wyckoff multiplicities must be positive")
        if np.any(row_free_dimensions > 3):
            raise DomainError("stored Wyckoff free dimensions exceed three")
        for index, item in enumerate(metadata):
            rows = item.get("row_labels")
            if not isinstance(rows, list) or not all(isinstance(label, str) for label in rows):
                raise DomainError("stored row labels are malformed")
            row_start = int(arrays["row_offsets"][index])
            row_stop = int(arrays["row_offsets"][index + 1])
            if len(rows) != row_stop - row_start:
                raise DomainError("stored row labels do not align with row tensors")
            dimension = int(arrays["period_dimensions"][index])
            p0 = int(arrays["period_offsets"][index])
            p1 = int(arrays["period_offsets"][index + 1])
            if p1 - p0 != dimension * dimension:
                raise DomainError("stored period basis is not square")
            i0 = int(arrays["intrinsic_offsets"][index])
            i1 = int(arrays["intrinsic_offsets"][index + 1])
            if i1 - i0 != dimension:
                raise DomainError("stored period and intrinsic dimensions disagree")
            c0 = int(arrays["composition_offsets"][index])
            c1 = int(arrays["composition_offsets"][index + 1])
            species = cast(
                npt.NDArray[np.uint8], arrays["composition_atomic_numbers"][c0:c1]
            )
            coefficients = cast(IntArray, arrays["composition_coefficients"][c0:c1])
            if len(species) == 0 or np.any(species[1:] <= species[:-1]):
                raise DomainError("stored composition support must be sorted and unique")
            nu_mask_start = int(arrays["nu_legal_offsets"][index])
            nu_mask_stop = int(arrays["nu_legal_offsets"][index + 1])
            nu_target = int(arrays["nu_target_index"][index])
            nu_mask = cast(BoolArray, arrays["nu_legal_mask"][nu_mask_start:nu_mask_stop])
            if (
                len(nu_mask) == 0
                or not 0 <= nu_target < len(nu_mask)
                or not bool(nu_mask[nu_target])
                or nu_target != int(arrays["nu"][index]) - 1
            ):
                raise DomainError("stored formula-unit selector target is not exactly legal")
            group_start = int(arrays["group_legal_offsets"][index])
            group_stop = int(arrays["group_legal_offsets"][index + 1])
            group_target = int(arrays["group_target_index"][index])
            group_mask = cast(
                BoolArray, arrays["group_legal_mask"][group_start:group_stop]
            )
            if (
                len(group_mask) == 0
                or not 0 <= group_target < len(group_mask)
                or not bool(group_mask[group_target])
            ):
                raise DomainError("stored space-group selector target is not exactly legal")
            residual = coefficients * int(arrays["nu"][index])
            for row_index in range(row_start, row_stop):
                action_start = int(arrays["row_action_offsets"][row_index])
                action_stop = int(arrays["row_action_offsets"][row_index + 1])
                if action_stop - action_start != len(coefficients):
                    raise DomainError("stored row-action width disagrees with composition")
                before = cast(
                    IntArray,
                    arrays["row_residual_before"][action_start:action_stop],
                )
                after = cast(
                    IntArray,
                    arrays["row_residual_after"][action_start:action_stop],
                )
                selected = cast(
                    IntArray, arrays["row_action_values"][action_start:action_stop]
                )
                if not np.array_equal(before, residual):
                    raise DomainError("stored teacher-forced residual trace is discontinuous")
                expected_after = (
                    before
                    - int(arrays["row_multiplicities"][row_index]) * selected
                )
                if np.any(expected_after < 0) or not np.array_equal(after, expected_after):
                    raise DomainError("stored row residual does not satisfy rho' = rho - mu*a")
                legal_start = int(arrays["row_legal_action_offsets"][row_index])
                legal_stop = int(arrays["row_legal_action_offsets"][row_index + 1])
                selected_index = int(arrays["row_selected_action_indices"][row_index])
                if (
                    legal_stop <= legal_start
                    or not 0 <= selected_index < legal_stop - legal_start
                ):
                    raise DomainError("stored selected row action lies outside its legal fibre")
                chosen_legal_index = legal_start + selected_index
                legal_value_start = int(arrays["legal_action_offsets"][chosen_legal_index])
                legal_value_stop = int(arrays["legal_action_offsets"][chosen_legal_index + 1])
                if legal_value_stop - legal_value_start != len(selected) or not np.array_equal(
                    arrays["legal_action_values"][legal_value_start:legal_value_stop],
                    selected,
                ):
                    raise DomainError("stored row target disagrees with its exact legal action")
                for legal_index in range(legal_start, legal_stop):
                    a0 = int(arrays["legal_action_offsets"][legal_index])
                    a1 = int(arrays["legal_action_offsets"][legal_index + 1])
                    if a1 - a0 != len(selected):
                        raise DomainError("stored legal row actions have inconsistent widths")
                residual = after
            if np.any(residual):
                raise DomainError("stored teacher-forced cover trace is not terminal")

    def close(self) -> None:
        # NumPy memmaps close when references are released.  Clearing this map
        # deterministically drops the store's ownership without invalidating
        # views already returned to callers.
        self.arrays = MappingProxyType({})


def _parse_fraction(payload: object) -> Fraction:
    if (
        not isinstance(payload, list)
        or len(payload) != 2
        or any(isinstance(value, bool) or not isinstance(value, int) for value in payload)
        or payload[1] == 0
    ):
        raise DomainError("stored exact fraction is malformed")
    return Fraction(payload[0], payload[1])


def _parse_fraction_vector(payload: object, *, name: str) -> tuple[Fraction, ...]:
    if not isinstance(payload, list):
        raise DomainError(f"stored {name} must be a JSON array")
    return tuple(_parse_fraction(item) for item in payload)


def _parse_hex_float(
    payload: object,
    *,
    name: str,
    positive: bool = False,
    nonnegative: bool = False,
) -> float:
    if not isinstance(payload, str):
        raise DomainError(f"stored {name} must be an exact hexadecimal float")
    try:
        value = float.fromhex(payload)
    except ValueError as exc:
        raise DomainError(f"stored {name} is not a hexadecimal float") from exc
    if not np.isfinite(value) or (positive and value <= 0.0) or (
        nonnegative and value < 0.0
    ):
        raise DomainError(f"stored {name} is outside its numerical domain")
    if value.hex() != payload:
        raise DomainError(f"stored {name} is not in canonical hexadecimal form")
    return value


def _parse_canonicalization(payload: object) -> CanonicalizationReceipt:
    expected = frozenset(
        {
            "source_fingerprint",
            "standardization_fingerprint",
            "catalogue_fingerprint",
            "hall_number",
            "hall_symbol",
            "source_to_primitive",
            "origin_shift",
            "free_origin_shift",
            "translation_index",
            "symmetry_tolerance",
            "roundtrip_max_cartesian_error",
            "symmetry_stability_verified",
            "catalogue_closure_verified",
            "wyckoff_multiplicity_verified",
            "symmetry_verified",
            "canonical_origin_verified",
            "scaffold_roundtrip_verified",
        }
    )
    optional = frozenset(
        {
            "idealization_lattice_displacement_angstrom",
            "idealization_site_displacement_angstrom",
        }
    )
    present = optional & set(payload) if isinstance(payload, Mapping) else frozenset()
    value = _require_fields(
        payload, expected | present, name="stored canonicalization receipt"
    )
    metadata: dict[str, float | None] = {name: None for name in optional}
    for name in present:
        metadata[name] = _parse_hex_float(
            value[name], name=name.replace("_", " "), nonnegative=True
        )
    matrix = value["source_to_primitive"]
    if not isinstance(matrix, list) or len(matrix) != 3:
        raise DomainError("stored source-to-primitive matrix must be exact 3x3")
    parsed_matrix = tuple(
        _parse_fraction_vector(row, name="source-to-primitive row") for row in matrix
    )
    if any(len(row) != 3 for row in parsed_matrix):
        raise DomainError("stored source-to-primitive matrix must be exact 3x3")
    gates = (
        "symmetry_stability_verified",
        "catalogue_closure_verified",
        "wyckoff_multiplicity_verified",
        "symmetry_verified",
        "canonical_origin_verified",
        "scaffold_roundtrip_verified",
    )
    if any(not isinstance(value[name], bool) for name in gates):
        raise DomainError("stored canonicalization gates must be booleans")
    receipt = CanonicalizationReceipt(
        source_fingerprint=value["source_fingerprint"],  # type: ignore[arg-type]
        standardization_fingerprint=value[
            "standardization_fingerprint"
        ],  # type: ignore[arg-type]
        catalogue_fingerprint=value["catalogue_fingerprint"],  # type: ignore[arg-type]
        hall_number=value["hall_number"],  # type: ignore[arg-type]
        hall_symbol=value["hall_symbol"],  # type: ignore[arg-type]
        source_to_primitive=parsed_matrix,
        origin_shift=_parse_fraction_vector(value["origin_shift"], name="origin shift"),
        free_origin_shift=_parse_fraction_vector(
            value["free_origin_shift"], name="free-origin shift"
        ),
        translation_index=value["translation_index"],  # type: ignore[arg-type]
        symmetry_tolerance=_parse_hex_float(
            value["symmetry_tolerance"],
            name="symmetry tolerance",
            positive=True,
        ),
        roundtrip_max_cartesian_error=_parse_hex_float(
            value["roundtrip_max_cartesian_error"],
            name="round-trip Cartesian error",
            nonnegative=True,
        ),
        symmetry_stability_verified=cast(bool, value["symmetry_stability_verified"]),
        catalogue_closure_verified=cast(bool, value["catalogue_closure_verified"]),
        wyckoff_multiplicity_verified=cast(
            bool, value["wyckoff_multiplicity_verified"]
        ),
        symmetry_verified=cast(bool, value["symmetry_verified"]),
        canonical_origin_verified=cast(bool, value["canonical_origin_verified"]),
        scaffold_roundtrip_verified=cast(
            bool, value["scaffold_roundtrip_verified"]
        ),
        idealization_lattice_displacement_angstrom=metadata[
            "idealization_lattice_displacement_angstrom"
        ],
        idealization_site_displacement_angstrom=metadata[
            "idealization_site_displacement_angstrom"
        ],
    )
    if not receipt.passed:
        raise DomainError("stored canonicalization receipt did not pass every exact gate")
    return receipt


def _parse_identity(payload: object) -> StoredIdentity:
    expected = frozenset(
        {
            "exact_key",
            "structure_key",
            "polymorph_key",
            "scaffold_key",
            "prototype_key",
            "composition_key",
            "element_set_key",
            "policy_fingerprint",
            "catalogue_fingerprint",
            "metric_bins",
            "shape_bins",
            "exact_payload",
            "structure_payload",
            "polymorph_payload",
        }
    )
    value = _require_fields(payload, expected, name="stored identity")
    digest_names = tuple(name for name in expected if name.endswith("key")) + (
        "policy_fingerprint",
        "catalogue_fingerprint",
    )
    digests = {name: _require_digest(value[name], name=name) for name in digest_names}
    bins: dict[str, tuple[int, ...]] = {}
    for name in ("metric_bins", "shape_bins"):
        raw = value[name]
        if (
            not isinstance(raw, list)
            or len(raw) != 6
            or any(isinstance(item, bool) or not isinstance(item, int) for item in raw)
        ):
            raise DomainError(f"stored identity {name} is malformed")
        bins[name] = tuple(raw)
    texts = {
        name: _require_text(value[name], name=name)
        for name in ("exact_payload", "structure_payload", "polymorph_payload")
    }
    return StoredIdentity(
        exact_key=digests["exact_key"],
        structure_key=digests["structure_key"],
        polymorph_key=digests["polymorph_key"],
        scaffold_key=digests["scaffold_key"],
        prototype_key=digests["prototype_key"],
        composition_key=digests["composition_key"],
        element_set_key=digests["element_set_key"],
        policy_fingerprint=digests["policy_fingerprint"],
        catalogue_fingerprint=digests["catalogue_fingerprint"],
        metric_bins=bins["metric_bins"],
        shape_bins=bins["shape_bins"],
        exact_payload=texts["exact_payload"],
        structure_payload=texts["structure_payload"],
        polymorph_payload=texts["polymorph_payload"],
    )


class DatasetStore:
    """Authenticated lazy reader for a published sharded dataset."""

    def __init__(
        self,
        root: Path,
        manifest: DatasetStorageManifest,
        build_manifest: DatasetBuildManifest,
        source_manifest: DatasetSourceManifest,
        record_index: tuple[StoredRecordIndexEntry, ...],
        split_manifests: Mapping[SplitMode, DatasetSplitManifest],
        notice_text: str,
        metadata: tuple[tuple[Mapping[str, object], ...], ...],
    ) -> None:
        self.root = root
        self.manifest = manifest
        self.build_manifest = build_manifest
        self.source_manifest = source_manifest
        self.record_index = record_index
        self.split_manifests = MappingProxyType(dict(split_manifests))
        self.notice_text = notice_text
        self._metadata = metadata
        self._mapped: dict[int, _MappedShard] = {}
        self._locations: dict[str, tuple[int, int]] = {}
        for entry in record_index:
            if entry.record_id in self._locations:
                raise DomainError("stored record identifiers are not unique")
            self._locations[entry.record_id] = (entry.shard_index, entry.local_index)

    @classmethod
    def open(
        cls,
        root: str | Path,
        *,
        expected_build_manifest_fingerprint: str | None = None,
        expected_source_manifest_fingerprint: str | None = None,
        expected_storage_fingerprint: str | None = None,
    ) -> DatasetStore:
        requested = Path(root)
        try:
            resolved = requested.resolve(strict=True)
        except OSError as exc:
            raise DomainError("dataset storage root does not exist") from exc
        if requested.is_symlink() or not resolved.is_dir():
            raise DomainError("dataset storage root must be a regular directory, not a link")
        manifest_path = resolved / _MANIFEST_NAME
        if manifest_path.is_symlink() or not manifest_path.is_file():
            raise DomainError("dataset storage manifest is missing or linked")
        manifest = DatasetStorageManifest.from_dict(
            _read_json(manifest_path, name="dataset storage manifest")
        )
        if expected_storage_fingerprint is not None:
            expected_storage = _require_digest(
                expected_storage_fingerprint,
                name="expected storage fingerprint",
            )
            if manifest.fingerprint != expected_storage:
                raise DomainError(
                    "dataset storage disagrees with the externally pinned manifest"
                )
        if expected_build_manifest_fingerprint is not None:
            expected = _require_digest(
                expected_build_manifest_fingerprint,
                name="expected build manifest fingerprint",
            )
            if manifest.build_manifest_fingerprint != expected:
                raise DomainError("dataset storage belongs to a different build manifest")

        build_path = resolved / _BUILD_MANIFEST_NAME
        if build_path.is_symlink() or not build_path.is_file():
            raise DomainError("embedded build manifest is missing or linked")
        if build_path.stat().st_size != manifest.build_manifest_byte_count:
            raise DomainError("embedded build manifest byte count changed")
        if _sha256_file(build_path) != manifest.build_manifest_sha256:
            raise DomainError("embedded build manifest hash changed")
        build_payload = _read_json(build_path, name="embedded build manifest")
        if not isinstance(build_payload, Mapping):
            raise DomainError("embedded build manifest JSON root must be an object")
        build_manifest = DatasetBuildManifest.from_dict(
            cast(Mapping[str, object], build_payload)
        )
        if build_manifest.fingerprint != manifest.build_manifest_fingerprint:
            raise DomainError("storage manifest and embedded build manifest disagree")
        if build_manifest.output_count != manifest.record_count:
            raise DomainError("storage and build manifests disagree on record count")

        source_path = resolved / _SOURCE_MANIFEST_NAME
        if source_path.is_symlink() or not source_path.is_file():
            raise DomainError("embedded source manifest is missing or linked")
        if source_path.stat().st_size != manifest.source_manifest_byte_count:
            raise DomainError("embedded source manifest byte count changed")
        if _sha256_file(source_path) != manifest.source_manifest_sha256:
            raise DomainError("embedded source manifest hash changed")
        source_manifest = DatasetSourceManifest.from_dict(
            _read_json(source_path, name="embedded source manifest")
        )
        if (
            source_manifest.fingerprint != manifest.source_manifest_fingerprint
            or source_manifest.fingerprint != build_manifest.source_manifest_fingerprint
        ):
            raise DomainError("stored source receipts disagree with the build manifest")
        if expected_source_manifest_fingerprint is not None:
            expected_source = _require_digest(
                expected_source_manifest_fingerprint,
                name="expected source manifest fingerprint",
            )
            if source_manifest.fingerprint != expected_source:
                raise DomainError("dataset storage belongs to different source receipts")

        _verify_regular_file(resolved, manifest.record_index)
        record_index = _parse_record_index(
            _read_json(resolved / _RECORD_INDEX_NAME, name="stored record index"),
            storage_manifest=manifest,
            build_manifest=build_manifest,
        )
        _verify_regular_file(resolved, manifest.notice)
        try:
            notice_payload = (resolved / _NOTICE_NAME).read_bytes()
        except OSError as exc:  # pragma: no cover - regular-file audit normally catches this
            raise DomainError("cannot read stored dataset notice") from exc
        if notice_payload != _notice_bytes_from_manifest(build_manifest):
            raise DomainError("stored dataset notice disagrees with source obligations")
        notice_text = notice_payload.decode("utf-8")

        expected_paths = {
            _MANIFEST_NAME,
            _BUILD_MANIFEST_NAME,
            _SOURCE_MANIFEST_NAME,
            manifest.record_index.relative_path,
            manifest.notice.relative_path,
        }
        expected_directories = {f"shard-{shard.index:05d}" for shard in manifest.shards}
        metadata_groups: list[tuple[Mapping[str, object], ...]] = []
        split_records: list[SplitRecord] = []
        summaries = iter(build_manifest.records)
        for shard in manifest.shards:
            for item in shard.files:
                _verify_regular_file(resolved, item)
                expected_paths.add(item.relative_path)
            mapped = _MappedShard.open(resolved, shard)
            entries = mapped.metadata
            try:
                for local_index, entry in enumerate(entries):
                    summary = next(summaries)
                    cls._audit_metadata_summary(entry, summary)
                    identity = _parse_identity(entry["identity"])
                    c0 = int(mapped.arrays["composition_offsets"][local_index])
                    c1 = int(mapped.arrays["composition_offsets"][local_index + 1])
                    support = tuple(
                        int(value)
                        for value in mapped.arrays["composition_atomic_numbers"][c0:c1]
                    )
                    split_records.append(
                        SplitRecord(
                            summary.record_id,
                            FrozenSplitKeys(
                                composition_key=identity.composition_key,
                                polymorph_key=identity.polymorph_key,
                                prototype_key=identity.prototype_key,
                                scaffold_key=identity.scaffold_key,
                                element_set=support,
                            ),
                        )
                    )
            finally:
                mapped.close()
            metadata_groups.append(entries)
        try:
            next(summaries)
        except StopIteration:
            pass
        else:  # pragma: no cover - count equality normally catches this first
            raise DomainError("stored records do not exhaust the build manifest")

        stored_split_bindings = tuple(
            (item.mode.value, item.manifest_fingerprint) for item in manifest.splits
        )
        if stored_split_bindings != build_manifest.split_manifest_fingerprints:
            raise DomainError("stored split receipts disagree with the build manifest")
        parsed_splits: dict[SplitMode, DatasetSplitManifest] = {}
        if manifest.splits:
            expected_directories.add(_SPLIT_DIRECTORY)
        for descriptor in manifest.splits:
            _verify_regular_file(resolved, descriptor.file)
            expected_paths.add(descriptor.file.relative_path)
            split_payload = _read_json(
                resolved.joinpath(*PurePosixPath(descriptor.file.relative_path).parts),
                name=f"{descriptor.mode.value} split receipt",
            )
            if not isinstance(split_payload, Mapping):
                raise DomainError("stored split receipt JSON root must be an object")
            split_manifest = DatasetSplitManifest.from_dict(
                cast(Mapping[str, object], split_payload)
            )
            if (
                split_manifest.mode is not descriptor.mode
                or split_manifest.manifest_fingerprint
                != descriptor.manifest_fingerprint
            ):
                raise DomainError("stored split descriptor and receipt disagree")
            audit_dataset_split(split_records, split_manifest)
            parsed_splits[descriptor.mode] = split_manifest

        actual_paths: set[str] = set()
        actual_directories: set[str] = set()
        for path in resolved.rglob("*"):
            relative = path.relative_to(resolved).as_posix()
            if path.is_symlink():
                raise DomainError(f"dataset storage contains a symbolic link: {relative}")
            if path.is_file():
                actual_paths.add(relative)
            elif path.is_dir():
                actual_directories.add(relative)
            else:
                raise DomainError(f"dataset storage contains a non-regular entry: {relative}")
        if actual_paths != expected_paths or actual_directories != expected_directories:
            raise DomainError("dataset storage contains missing or unauthenticated entries")
        return cls(
            resolved,
            manifest,
            build_manifest,
            source_manifest,
            record_index,
            parsed_splits,
            notice_text,
            tuple(metadata_groups),
        )

    @staticmethod
    def _audit_metadata_summary(entry: Mapping[str, object], summary: object) -> None:
        from .dataset_builder import DatasetRecordSummary

        if not isinstance(summary, DatasetRecordSummary):
            raise TypeError("build manifest record summary has an unexpected type")
        expected = frozenset(
            {
                "record_id",
                "representative_payload_sha256",
                "target_fingerprint",
                "canonicalization_receipt_fingerprint",
                "target_type",
                "space_group",
                "row_labels",
                "orbits",
                "aliases",
                "alias_scope_evidence",
                "identity",
                "canonicalization",
            }
        )
        value = _require_fields(entry, expected, name="stored record metadata")
        aliases = value["aliases"]
        if not isinstance(aliases, list) or any(
            not isinstance(item, Mapping) for item in aliases
        ):
            raise DomainError("stored aliases are malformed")
        parsed_aliases = tuple(
            SourceRecordReference.from_dict(cast(Mapping[str, object], item))
            for item in aliases
        )
        evidence = value["alias_scope_evidence"]
        if not isinstance(evidence, list):
            raise DomainError("stored alias scope evidence is malformed")
        parsed_evidence = tuple(
            SourceRecordScopeEvidence.from_dict(item) for item in evidence
        )
        identity = _parse_identity(value["identity"])
        if (
            value["record_id"] != summary.record_id
            or parsed_aliases != summary.aliases
            or parsed_evidence != summary.alias_scope_evidence
            or value["representative_payload_sha256"]
            != summary.representative_payload_sha256
            or value["target_fingerprint"] != summary.target_fingerprint
            or value["canonicalization_receipt_fingerprint"]
            != summary.canonicalization_receipt_fingerprint
            or identity.exact_key != summary.exact_key
            or identity.composition_key != summary.composition_key
            or identity.element_set_key != summary.element_set_key
            or identity.polymorph_key != summary.polymorph_key
            or identity.prototype_key != summary.prototype_key
            or identity.scaffold_key != summary.scaffold_key
            or identity.structure_key != summary.structure_key
        ):
            raise DomainError("stored record metadata disagrees with the build manifest")
        if value["target_type"] not in {"exact-fixture", "canonical-catalogue"}:
            raise DomainError("stored target type is unsupported")
        _require_text(value["space_group"], name="stored space group")
        for name in (
            "representative_payload_sha256",
            "target_fingerprint",
            "canonicalization_receipt_fingerprint",
        ):
            _require_digest(value[name], name=f"stored {name}")
        receipt = _parse_canonicalization(value["canonicalization"])
        if receipt.catalogue_fingerprint != identity.catalogue_fingerprint:
            raise DomainError("stored identity and receipt use different catalogues")
        row_labels = value["row_labels"]
        if (
            not isinstance(row_labels, list)
            or not row_labels
            or any(not isinstance(label, str) or not label for label in row_labels)
            or len(set(row_labels)) != len(row_labels)
        ):
            raise DomainError("stored row labels are malformed")
        raw_orbits = value["orbits"]
        if not isinstance(raw_orbits, list) or not raw_orbits:
            raise DomainError("stored orbits are malformed")
        for raw_orbit in raw_orbits:
            orbit = _require_fields(
                raw_orbit,
                frozenset({"wyckoff_label", "atomic_number", "parameter"}),
                name="stored orbit",
            )
            _require_text(orbit["wyckoff_label"], name="stored Wyckoff label")
            atomic_number = _require_int(
                orbit["atomic_number"], name="stored orbit atomic number", minimum=1
            )
            if atomic_number > 118:
                raise DomainError("stored orbit has an invalid atomic number")
            parameter = _parse_fraction_vector(
                orbit["parameter"], name="stored orbit parameter"
            )
            if len(parameter) > 3:
                raise DomainError("stored orbit parameter dimension exceeds three")

    def __len__(self) -> int:
        return self.manifest.record_count

    @property
    def record_ids(self) -> tuple[str, ...]:
        return tuple(self._locations)

    def split_manifest_for(self, mode: SplitMode) -> DatasetSplitManifest:
        """Return the authenticated leakage audit receipt for ``mode``."""

        if not isinstance(mode, SplitMode):
            raise TypeError("mode must be SplitMode")
        try:
            return self.split_manifests[mode]
        except KeyError as exc:
            raise KeyError(f"dataset has no {mode.value!r} split receipt") from exc

    def _mapped_shard(self, index: int) -> _MappedShard:
        mapped = self._mapped.get(index)
        if mapped is None:
            mapped = _MappedShard.open(self.root, self.manifest.shards[index])
            self._mapped[index] = mapped
        return mapped

    def __getitem__(self, key: int | str) -> StoredTrainingRecord:
        if isinstance(key, str):
            try:
                shard_index, local_index = self._locations[key]
            except KeyError as exc:
                raise KeyError(f"unknown dataset record {key!r}") from exc
        elif isinstance(key, int) and not isinstance(key, bool):
            index = key + len(self) if key < 0 else key
            if not 0 <= index < len(self):
                raise IndexError("dataset record index out of range")
            shard_index = index // self.manifest.records_per_shard
            local_index = index - self.manifest.shards[shard_index].start
        else:
            raise TypeError("dataset keys must be integer positions or record identifiers")
        return self._record(self._mapped_shard(shard_index), local_index)

    @staticmethod
    def _record(shard: _MappedShard, index: int) -> StoredTrainingRecord:
        arrays = shard.arrays
        metadata = shard.metadata[index]

        def record_slice(offset_name: str, value_name: str) -> npt.NDArray[np.generic]:
            start = int(arrays[offset_name][index])
            stop = int(arrays[offset_name][index + 1])
            return arrays[value_name][start:stop]

        aliases_raw = cast(list[object], metadata["aliases"])
        aliases = tuple(
            SourceRecordReference.from_dict(cast(Mapping[str, object], item))
            for item in aliases_raw
        )
        evidence_raw = cast(list[object], metadata["alias_scope_evidence"])
        alias_scope_evidence = tuple(
            SourceRecordScopeEvidence.from_dict(item) for item in evidence_raw
        )
        raw_orbits = metadata["orbits"]
        if not isinstance(raw_orbits, list):
            raise DomainError("stored orbits are malformed")
        orbits: list[StoredOrbit] = []
        for raw in raw_orbits:
            value = _require_fields(
                raw,
                frozenset({"wyckoff_label", "atomic_number", "parameter"}),
                name="stored orbit",
            )
            parameter = value["parameter"]
            if not isinstance(parameter, list):
                raise DomainError("stored orbit parameter is malformed")
            orbits.append(
                StoredOrbit(
                    _require_text(value["wyckoff_label"], name="Wyckoff label"),
                    _require_int(value["atomic_number"], name="atomic number", minimum=1),
                    tuple(_parse_fraction(item) for item in parameter),
                )
            )

        row_start = int(arrays["row_offsets"][index])
        row_stop = int(arrays["row_offsets"][index + 1])
        row_labels = cast(list[str], metadata["row_labels"])
        rows: list[StoredRowDecision] = []
        for local_row, row_index in enumerate(range(row_start, row_stop)):
            action_start = int(arrays["row_action_offsets"][row_index])
            action_stop = int(arrays["row_action_offsets"][row_index + 1])
            width = action_stop - action_start
            legal_start = int(arrays["row_legal_action_offsets"][row_index])
            legal_stop = int(arrays["row_legal_action_offsets"][row_index + 1])
            legal_value_start = int(arrays["legal_action_offsets"][legal_start])
            legal_value_stop = int(arrays["legal_action_offsets"][legal_stop])
            legal_flat = cast(
                IntArray,
                arrays["legal_action_values"][legal_value_start:legal_value_stop],
            )
            if width == 0 or len(legal_flat) != (legal_stop - legal_start) * width:
                raise DomainError("stored legal row actions have inconsistent widths")
            rows.append(
                StoredRowDecision(
                    row_label=row_labels[local_row],
                    multiplicity=int(arrays["row_multiplicities"][row_index]),
                    free_dimension=int(arrays["row_free_dimensions"][row_index]),
                    selected_action_index=int(
                        arrays["row_selected_action_indices"][row_index]
                    ),
                    selected_action=cast(
                        IntArray, arrays["row_action_values"][action_start:action_stop]
                    ),
                    legal_actions=legal_flat.reshape(legal_stop - legal_start, width),
                    residual_before=cast(
                        IntArray, arrays["row_residual_before"][action_start:action_stop]
                    ),
                    residual_after=cast(
                        IntArray, arrays["row_residual_after"][action_start:action_stop]
                    ),
                )
            )

        dimension = int(arrays["period_dimensions"][index])
        p0 = int(arrays["period_offsets"][index])
        p1 = int(arrays["period_offsets"][index + 1])
        return StoredTrainingRecord(
            record_id=cast(str, metadata["record_id"]),
            target_type=cast(str, metadata["target_type"]),
            space_group=cast(str, metadata["space_group"]),
            aliases=aliases,
            alias_scope_evidence=alias_scope_evidence,
            identity=_parse_identity(metadata["identity"]),
            representative_payload_sha256=cast(
                str, metadata["representative_payload_sha256"]
            ),
            target_fingerprint=cast(str, metadata["target_fingerprint"]),
            canonicalization_receipt_fingerprint=cast(
                str, metadata["canonicalization_receipt_fingerprint"]
            ),
            canonicalization=_parse_canonicalization(metadata["canonicalization"]),
            orbits=tuple(orbits),
            atomic_numbers=cast(
                npt.NDArray[np.uint8],
                record_slice("composition_offsets", "composition_atomic_numbers"),
            ),
            composition_coefficients=cast(
                IntArray,
                record_slice("composition_offsets", "composition_coefficients"),
            ),
            nu=int(arrays["nu"][index]),
            metric=cast(FloatArray, arrays["metric"][index]),
            lattice_latent=cast(
                FloatArray, record_slice("lattice_offsets", "lattice_latent")
            ),
            intrinsic_coordinates=cast(
                FloatArray, record_slice("intrinsic_offsets", "intrinsic_coordinates")
            ),
            period_basis=cast(
                FloatArray,
                arrays["period_basis"][p0:p1].reshape(dimension, dimension),
            ),
            nu_legal_mask=cast(
                BoolArray, record_slice("nu_legal_offsets", "nu_legal_mask")
            ),
            nu_target_index=int(arrays["nu_target_index"][index]),
            group_legal_mask=cast(
                BoolArray, record_slice("group_legal_offsets", "group_legal_mask")
            ),
            group_target_index=int(arrays["group_target_index"][index]),
            row_decisions=tuple(rows),
        )

    def __iter__(self) -> Iterator[StoredTrainingRecord]:
        for index in range(len(self)):
            yield self[index]

    def close(self) -> None:
        for mapped in self._mapped.values():
            mapped.close()
        self._mapped.clear()

    def __enter__(self) -> DatasetStore:
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()


__all__ = [
    "DatasetStorageManifest",
    "DatasetStore",
    "StoredFile",
    "StoredIdentity",
    "StoredOrbit",
    "StoredRowDecision",
    "StoredShard",
    "StoredTrainingRecord",
    "write_compiled_dataset",
]
