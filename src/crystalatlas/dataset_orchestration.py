"""Authenticated local-CIF orchestration from source receipts to tensor shards.

The core compiler intentionally accepts in-memory :class:`DatasetCandidate`
objects.  This module supplies the narrower filesystem boundary needed by a
reproducible local build: every candidate payload must itself be a pinned
``DATA`` artifact, and the archived record, payload, and scope indexes must be
canonical and agree in exactly the same order.

This loader does not download, unpack, infer rights for, or silently select
records from a database archive.  Compressed exports need a separately pinned
and audited extraction/indexing stage before they can use this entry point.
External source-manifest and crystallography-receipt fingerprints remain the
caller's trust anchors; self-fingerprints are not signatures.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path, PurePosixPath
from typing import Any, cast

from .dataset_builder import (
    DEFAULT_SPLIT_FRACTIONS,
    DatasetCandidate,
    DatasetCompiler,
    SourceRecordReference,
)
from .dataset_provenance import (
    ArtifactChecksum,
    ArtifactRole,
    DatasetSourceManifest,
    SourceRecordScopeEvidence,
    canonical_record_index_bytes,
    canonical_record_payload_index_bytes,
    canonical_record_scope_index_bytes,
    ordered_record_ids_sha256,
    ordered_record_payloads_sha256,
    ordered_record_scopes_sha256,
)
from .dataset_splits import SplitFractions, SplitMode
from .dataset_storage import DatasetStore, write_compiled_dataset
from .errors import DomainError, InvariantViolation
from .historical_mp20 import require_historical_mp20_archive

_INPUT_SCHEMA = "crystalatlas.local-cif-input-manifest.v1"
_BUILD_SCHEMA = "crystalatlas.local-dataset-build-receipt.v1"
_LOADER_PROTOCOL = "direct-pinned-data-artifact-v1"
_DEFAULT_MAX_CIF_BYTES = 16 * 1024 * 1024


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    )


def _fingerprint(value: object) -> str:
    return sha256(_canonical_json(value).encode("ascii")).hexdigest()


def _require_digest(value: object, *, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise DomainError(f"{name} must be a lowercase SHA-256 digest")
    return value


def _require_count(value: object, *, name: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise DomainError(f"{name} must be an integer of at least {minimum}")
    return value


def _strict_json_loads(payload: bytes, *, name: str) -> object:
    def reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise DomainError(f"{name} repeats JSON key {key!r}")
            result[key] = value
        return result

    def reject_nonfinite(token: str) -> object:
        raise DomainError(f"{name} contains non-standard JSON number {token!r}")

    try:
        text = payload.decode("utf-8", errors="strict")
        value = json.loads(
            text,
            object_pairs_hook=reject_duplicate_keys,
            parse_constant=reject_nonfinite,
        )
    except DomainError:
        raise
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise DomainError(f"{name} is not strict UTF-8 JSON") from exc
    return value


def _require_object(
    value: object, *, fields: frozenset[str], name: str
) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or set(value) != fields:
        raise DomainError(f"{name} has missing or unexpected fields")
    if any(not isinstance(key, str) for key in value):
        raise DomainError(f"{name} contains a non-text key")
    return cast(Mapping[str, object], value)


def _artifact_for_role(
    artifacts: Sequence[ArtifactChecksum], role: ArtifactRole
) -> ArtifactChecksum:
    matches = tuple(artifact for artifact in artifacts if artifact.role is role)
    if len(matches) != 1:
        raise InvariantViolation(f"snapshot must contain exactly one {role.value} artifact")
    return matches[0]


def _artifact_path(root: Path, artifact: ArtifactChecksum) -> Path:
    return root.joinpath(*PurePosixPath(artifact.relative_path).parts)


def _read_record_index(path: Path) -> tuple[str, ...]:
    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise DomainError(f"cannot read archived record index {path}") from exc
    payload = _require_object(
        _strict_json_loads(raw, name="record index"),
        fields=frozenset({"schema", "record_ids"}),
        name="record index",
    )
    if payload["schema"] != "crystalatlas.dataset-record-index.v1" or not isinstance(
        payload["record_ids"], list
    ):
        raise DomainError("record index has an unsupported schema or malformed rows")
    record_ids = tuple(payload["record_ids"])
    if any(not isinstance(value, str) for value in record_ids):
        raise DomainError("record index identifiers must be strings")
    normalized = cast(tuple[str, ...], record_ids)
    if canonical_record_index_bytes(normalized) != raw:
        raise DomainError("record index is not the canonical archived representation")
    return normalized


def _read_payload_index(path: Path) -> tuple[tuple[str, str], ...]:
    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise DomainError(f"cannot read archived payload index {path}") from exc
    payload = _require_object(
        _strict_json_loads(raw, name="payload index"),
        fields=frozenset({"schema", "records"}),
        name="payload index",
    )
    records = payload["records"]
    if payload["schema"] != "crystalatlas.dataset-record-payload-index.v1" or not isinstance(
        records, list
    ):
        raise DomainError("payload index has an unsupported schema or malformed rows")
    normalized: list[tuple[str, str]] = []
    for row in records:
        if (
            not isinstance(row, list)
            or len(row) != 2
            or not isinstance(row[0], str)
            or not isinstance(row[1], str)
        ):
            raise DomainError("payload index rows must be [record_id, sha256]")
        normalized.append((row[0], row[1]))
    result = tuple(normalized)
    if canonical_record_payload_index_bytes(result) != raw:
        raise DomainError("payload index is not the canonical archived representation")
    return result


def _read_scope_index(
    path: Path,
) -> tuple[tuple[str, SourceRecordScopeEvidence], ...]:
    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise DomainError(f"cannot read archived scope index {path}") from exc
    payload = _require_object(
        _strict_json_loads(raw, name="scope index"),
        fields=frozenset({"schema", "records"}),
        name="scope index",
    )
    records = payload["records"]
    if payload["schema"] != "crystalatlas.dataset-record-scope-index.v1" or not isinstance(
        records, list
    ):
        raise DomainError("scope index has an unsupported schema or malformed rows")
    normalized: list[tuple[str, SourceRecordScopeEvidence]] = []
    for row in records:
        value = _require_object(
            row,
            fields=frozenset({"record_id", "scope"}),
            name="scope index row",
        )
        if not isinstance(value["record_id"], str):
            raise DomainError("scope index record identifiers must be strings")
        normalized.append(
            (
                value["record_id"],
                SourceRecordScopeEvidence.from_dict(value["scope"]),
            )
        )
    result = tuple(normalized)
    if canonical_record_scope_index_bytes(result) != raw:
        raise DomainError("scope index is not the canonical archived representation")
    return result


@dataclass(frozen=True, slots=True, order=True)
class LocalCifCandidateBinding:
    """One source row bound to the exact local ``DATA`` artifact used as CIF."""

    source_id: str
    snapshot_id: str
    source_record_id: str
    data_artifact_relative_path: str
    payload_sha256: str
    scope_evidence_fingerprint: str

    def __post_init__(self) -> None:
        for name in ("source_id", "snapshot_id", "source_record_id"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value or value != value.strip():
                raise DomainError(f"{name} must be nonempty stripped text")
        path = self.data_artifact_relative_path
        if not isinstance(path, str):
            raise DomainError("data artifact path must be normalized relative POSIX text")
        candidate = PurePosixPath(path)
        if (
            not path
            or candidate.is_absolute()
            or candidate.as_posix() != path
            or any(part in {"", ".", ".."} for part in candidate.parts)
        ):
            raise DomainError("data artifact path must be normalized relative POSIX text")
        _require_digest(self.payload_sha256, name="candidate payload fingerprint")
        _require_digest(
            self.scope_evidence_fingerprint,
            name="candidate scope-evidence fingerprint",
        )

    def to_dict(self) -> dict[str, str]:
        return {
            "source_id": self.source_id,
            "snapshot_id": self.snapshot_id,
            "source_record_id": self.source_record_id,
            "data_artifact_relative_path": self.data_artifact_relative_path,
            "payload_sha256": self.payload_sha256,
            "scope_evidence_fingerprint": self.scope_evidence_fingerprint,
        }

    @classmethod
    def from_dict(cls, payload: object) -> LocalCifCandidateBinding:
        value = _require_object(
            payload,
            fields=frozenset(
                {
                    "source_id",
                    "snapshot_id",
                    "source_record_id",
                    "data_artifact_relative_path",
                    "payload_sha256",
                    "scope_evidence_fingerprint",
                }
            ),
            name="local CIF candidate binding",
        )
        return cls(
            source_id=value["source_id"],  # type: ignore[arg-type]
            snapshot_id=value["snapshot_id"],  # type: ignore[arg-type]
            source_record_id=value["source_record_id"],  # type: ignore[arg-type]
            data_artifact_relative_path=value["data_artifact_relative_path"],  # type: ignore[arg-type]
            payload_sha256=value["payload_sha256"],  # type: ignore[arg-type]
            scope_evidence_fingerprint=value["scope_evidence_fingerprint"],  # type: ignore[arg-type]
        )


@dataclass(frozen=True, slots=True)
class LocalCifInputManifest:
    """Path-independent receipt for one deterministic local candidate load."""

    source_manifest_fingerprint: str
    max_cif_bytes: int
    bindings: tuple[LocalCifCandidateBinding, ...]
    manifest_fingerprint: str | None = None

    def __post_init__(self) -> None:
        _require_digest(
            self.source_manifest_fingerprint,
            name="source manifest fingerprint",
        )
        _require_count(self.max_cif_bytes, name="max_cif_bytes", minimum=1)
        bindings = tuple(self.bindings)
        if not bindings or any(
            not isinstance(item, LocalCifCandidateBinding) for item in bindings
        ):
            raise DomainError("local CIF input manifest needs candidate bindings")
        references = tuple(
            (item.source_id, item.snapshot_id, item.source_record_id) for item in bindings
        )
        if len(set(references)) != len(references):
            raise DomainError("local CIF input manifest repeats a source-row reference")
        object.__setattr__(self, "bindings", bindings)
        expected = _fingerprint(self._unsigned_payload())
        if self.manifest_fingerprint is None:
            object.__setattr__(self, "manifest_fingerprint", expected)
        elif (
            _require_digest(
                self.manifest_fingerprint,
                name="local CIF input manifest fingerprint",
            )
            != expected
        ):
            raise DomainError("local CIF input manifest is internally inconsistent")

    @property
    def fingerprint(self) -> str:
        assert self.manifest_fingerprint is not None
        return self.manifest_fingerprint

    def _unsigned_payload(self) -> dict[str, object]:
        return {
            "schema": _INPUT_SCHEMA,
            "loader_protocol": _LOADER_PROTOCOL,
            "source_manifest_fingerprint": self.source_manifest_fingerprint,
            "max_cif_bytes": self.max_cif_bytes,
            "bindings": [item.to_dict() for item in self.bindings],
        }

    def to_dict(self) -> dict[str, object]:
        value = self._unsigned_payload()
        value["manifest_fingerprint"] = self.fingerprint
        return value

    def to_json(self) -> str:
        return _canonical_json(self.to_dict()) + "\n"

    @classmethod
    def from_dict(cls, payload: object) -> LocalCifInputManifest:
        value = _require_object(
            payload,
            fields=frozenset(
                {
                    "schema",
                    "loader_protocol",
                    "source_manifest_fingerprint",
                    "max_cif_bytes",
                    "bindings",
                    "manifest_fingerprint",
                }
            ),
            name="local CIF input manifest",
        )
        if value["schema"] != _INPUT_SCHEMA or value["loader_protocol"] != _LOADER_PROTOCOL:
            raise DomainError("unsupported local CIF input manifest schema or protocol")
        bindings = value["bindings"]
        if not isinstance(bindings, list):
            raise DomainError("local CIF input manifest bindings must be an array")
        return cls(
            source_manifest_fingerprint=value["source_manifest_fingerprint"],  # type: ignore[arg-type]
            max_cif_bytes=value["max_cif_bytes"],  # type: ignore[arg-type]
            bindings=tuple(LocalCifCandidateBinding.from_dict(item) for item in bindings),
            manifest_fingerprint=value["manifest_fingerprint"],  # type: ignore[arg-type]
        )

    @classmethod
    def from_json(cls, payload: str) -> LocalCifInputManifest:
        if not isinstance(payload, str):
            raise TypeError("local CIF input manifest JSON must be text")
        try:
            encoded = payload.encode("utf-8", errors="strict")
        except UnicodeEncodeError as exc:
            raise DomainError("local CIF input manifest is not valid UTF-8 text") from exc
        return cls.from_dict(_strict_json_loads(encoded, name="local CIF input manifest"))


@dataclass(frozen=True, slots=True)
class LoadedLocalCifCandidates:
    """In-memory candidates plus their path-independent authenticated receipt."""

    candidates: tuple[DatasetCandidate, ...]
    manifest: LocalCifInputManifest

    def __post_init__(self) -> None:
        candidates = tuple(self.candidates)
        if not candidates or any(not isinstance(item, DatasetCandidate) for item in candidates):
            raise DomainError("loaded local CIF candidates cannot be empty")
        object.__setattr__(self, "candidates", candidates)
        if not isinstance(self.manifest, LocalCifInputManifest):
            raise TypeError("manifest must be LocalCifInputManifest")
        if len(candidates) != len(self.manifest.bindings):
            raise InvariantViolation("candidate payloads and local bindings disagree")
        for candidate, binding in zip(candidates, self.manifest.bindings, strict=True):
            reference = candidate.reference
            if (
                reference.source_id,
                reference.snapshot_id,
                reference.source_record_id,
                candidate.payload_sha256,
                candidate.scope_evidence.fingerprint,
            ) != (
                binding.source_id,
                binding.snapshot_id,
                binding.source_record_id,
                binding.payload_sha256,
                binding.scope_evidence_fingerprint,
            ):
                raise InvariantViolation("candidate payload disagrees with its local binding")


@dataclass(frozen=True, slots=True)
class LocalDatasetBuildReceipt:
    """CLI-friendly result binding local inputs, compilation, and final storage."""

    local_input_manifest: LocalCifInputManifest
    source_manifest_fingerprint: str
    policy_decision_fingerprint: str
    crystallography_receipt_fingerprint: str
    target_builder_fingerprint: str
    build_manifest_fingerprint: str
    storage_manifest_fingerprint: str
    input_count: int
    accepted_pre_dedup_count: int
    output_count: int
    rejection_count: int
    records_per_shard: int
    split_manifest_fingerprints: tuple[tuple[str, str], ...]
    receipt_fingerprint: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.local_input_manifest, LocalCifInputManifest):
            raise TypeError("local_input_manifest must be LocalCifInputManifest")
        for name in (
            "source_manifest_fingerprint",
            "policy_decision_fingerprint",
            "crystallography_receipt_fingerprint",
            "target_builder_fingerprint",
            "build_manifest_fingerprint",
            "storage_manifest_fingerprint",
        ):
            _require_digest(getattr(self, name), name=name)
        if (
            self.local_input_manifest.source_manifest_fingerprint
            != self.source_manifest_fingerprint
        ):
            raise DomainError("local inputs and source manifest fingerprint disagree")
        for name in (
            "input_count",
            "accepted_pre_dedup_count",
            "output_count",
            "rejection_count",
        ):
            _require_count(getattr(self, name), name=name)
        _require_count(self.records_per_shard, name="records_per_shard", minimum=1)
        if self.accepted_pre_dedup_count + self.rejection_count != self.input_count:
            raise DomainError("build receipt input accounting is inconsistent")
        if self.output_count > self.accepted_pre_dedup_count:
            raise DomainError("build receipt output exceeds accepted input count")
        split_fingerprints = tuple(self.split_manifest_fingerprints)
        if tuple(sorted(split_fingerprints)) != split_fingerprints:
            raise DomainError("split manifest fingerprints must be canonically sorted")
        if len({name for name, _ in split_fingerprints}) != len(split_fingerprints):
            raise DomainError("split manifest fingerprint modes must be unique")
        for mode, digest in split_fingerprints:
            try:
                SplitMode(mode)
            except ValueError as exc:
                raise DomainError("build receipt contains an unknown split mode") from exc
            _require_digest(digest, name=f"{mode} split manifest fingerprint")
        object.__setattr__(self, "split_manifest_fingerprints", split_fingerprints)
        expected = _fingerprint(self._unsigned_payload())
        if self.receipt_fingerprint is None:
            object.__setattr__(self, "receipt_fingerprint", expected)
        elif (
            _require_digest(
                self.receipt_fingerprint,
                name="local dataset build receipt fingerprint",
            )
            != expected
        ):
            raise DomainError("local dataset build receipt is internally inconsistent")

    @property
    def fingerprint(self) -> str:
        assert self.receipt_fingerprint is not None
        return self.receipt_fingerprint

    @property
    def local_input_manifest_fingerprint(self) -> str:
        return self.local_input_manifest.fingerprint

    def _unsigned_payload(self) -> dict[str, object]:
        return {
            "schema": _BUILD_SCHEMA,
            "local_input_manifest": self.local_input_manifest.to_dict(),
            "source_manifest_fingerprint": self.source_manifest_fingerprint,
            "policy_decision_fingerprint": self.policy_decision_fingerprint,
            "crystallography_receipt_fingerprint": (self.crystallography_receipt_fingerprint),
            "target_builder_fingerprint": self.target_builder_fingerprint,
            "build_manifest_fingerprint": self.build_manifest_fingerprint,
            "storage_manifest_fingerprint": self.storage_manifest_fingerprint,
            "input_count": self.input_count,
            "accepted_pre_dedup_count": self.accepted_pre_dedup_count,
            "output_count": self.output_count,
            "rejection_count": self.rejection_count,
            "records_per_shard": self.records_per_shard,
            "split_manifest_fingerprints": dict(self.split_manifest_fingerprints),
        }

    def to_dict(self) -> dict[str, object]:
        value = self._unsigned_payload()
        value["receipt_fingerprint"] = self.fingerprint
        return value

    def to_json(self) -> str:
        return _canonical_json(self.to_dict()) + "\n"

    @classmethod
    def from_dict(cls, payload: object) -> LocalDatasetBuildReceipt:
        value = _require_object(
            payload,
            fields=frozenset(
                {
                    "schema",
                    "local_input_manifest",
                    "source_manifest_fingerprint",
                    "policy_decision_fingerprint",
                    "crystallography_receipt_fingerprint",
                    "target_builder_fingerprint",
                    "build_manifest_fingerprint",
                    "storage_manifest_fingerprint",
                    "input_count",
                    "accepted_pre_dedup_count",
                    "output_count",
                    "rejection_count",
                    "records_per_shard",
                    "split_manifest_fingerprints",
                    "receipt_fingerprint",
                }
            ),
            name="local dataset build receipt",
        )
        if value["schema"] != _BUILD_SCHEMA:
            raise DomainError("unsupported local dataset build receipt schema")
        split_fingerprints = value["split_manifest_fingerprints"]
        if not isinstance(split_fingerprints, Mapping) or any(
            not isinstance(key, str) or not isinstance(digest, str)
            for key, digest in split_fingerprints.items()
        ):
            raise DomainError("build receipt split fingerprints must map text to text")
        return cls(
            local_input_manifest=LocalCifInputManifest.from_dict(value["local_input_manifest"]),
            source_manifest_fingerprint=value["source_manifest_fingerprint"],  # type: ignore[arg-type]
            policy_decision_fingerprint=value["policy_decision_fingerprint"],  # type: ignore[arg-type]
            crystallography_receipt_fingerprint=value["crystallography_receipt_fingerprint"],  # type: ignore[arg-type]
            target_builder_fingerprint=value["target_builder_fingerprint"],  # type: ignore[arg-type]
            build_manifest_fingerprint=value["build_manifest_fingerprint"],  # type: ignore[arg-type]
            storage_manifest_fingerprint=value["storage_manifest_fingerprint"],  # type: ignore[arg-type]
            input_count=value["input_count"],  # type: ignore[arg-type]
            accepted_pre_dedup_count=value["accepted_pre_dedup_count"],  # type: ignore[arg-type]
            output_count=value["output_count"],  # type: ignore[arg-type]
            rejection_count=value["rejection_count"],  # type: ignore[arg-type]
            records_per_shard=value["records_per_shard"],  # type: ignore[arg-type]
            split_manifest_fingerprints=tuple(sorted(split_fingerprints.items())),
            receipt_fingerprint=value["receipt_fingerprint"],  # type: ignore[arg-type]
        )

    @classmethod
    def from_json(cls, payload: str) -> LocalDatasetBuildReceipt:
        if not isinstance(payload, str):
            raise TypeError("local dataset build receipt JSON must be text")
        try:
            encoded = payload.encode("utf-8", errors="strict")
        except UnicodeEncodeError as exc:
            raise DomainError("local dataset build receipt is not valid UTF-8 text") from exc
        return cls.from_dict(_strict_json_loads(encoded, name="local dataset build receipt"))


def load_local_cif_candidates(
    source_manifest: DatasetSourceManifest,
    *,
    expected_source_manifest_fingerprint: str,
    artifact_roots: Mapping[tuple[str, str], str | Path],
    max_cif_bytes: int = _DEFAULT_MAX_CIF_BYTES,
) -> LoadedLocalCifCandidates:
    """Load exact UTF-8 CIF rows whose bytes are individually pinned artifacts.

    ``expected_source_manifest_fingerprint`` must come from outside the
    acquisition tree (or from a verified reviewer signature).  Snapshot roots
    are operational locations and therefore excluded from scientific
    fingerprints; all files they contribute are content-addressed instead.
    """

    if not isinstance(source_manifest, DatasetSourceManifest):
        raise TypeError("source_manifest must be DatasetSourceManifest")
    expected = _require_digest(
        expected_source_manifest_fingerprint,
        name="expected source manifest fingerprint",
    )
    if source_manifest.fingerprint != expected:
        raise DomainError("source manifest disagrees with the external trust anchor")
    _require_count(max_cif_bytes, name="max_cif_bytes", minimum=1)
    expected_roots = {
        (receipt.source.source_id, receipt.snapshot.snapshot_id)
        for receipt in source_manifest.receipts
    }
    if set(artifact_roots) != expected_roots:
        missing = sorted(expected_roots.difference(artifact_roots))
        extra = sorted(set(artifact_roots).difference(expected_roots))
        raise DomainError(
            "snapshot artifact roots disagree with source receipts; "
            f"missing={missing}, extra={extra}"
        )

    candidates: list[DatasetCandidate] = []
    bindings: list[LocalCifCandidateBinding] = []
    for receipt in source_manifest.receipts:
        source = receipt.source
        snapshot = receipt.snapshot
        key = (source.source_id, snapshot.snapshot_id)
        root = Path(artifact_roots[key])
        snapshot.require_files(root)
        try:
            resolved_root = root.resolve(strict=True)
        except OSError as exc:  # pragma: no cover - require_files normally catches this
            raise DomainError(f"cannot resolve snapshot artifact root {root}") from exc

        record_artifact = _artifact_for_role(snapshot.artifacts, ArtifactRole.RECORD_INDEX)
        payload_artifact = _artifact_for_role(snapshot.artifacts, ArtifactRole.PAYLOAD_INDEX)
        scope_artifact = _artifact_for_role(snapshot.artifacts, ArtifactRole.SCOPE_INDEX)
        record_ids = _read_record_index(_artifact_path(resolved_root, record_artifact))
        payload_rows = _read_payload_index(_artifact_path(resolved_root, payload_artifact))
        scope_rows = _read_scope_index(_artifact_path(resolved_root, scope_artifact))
        if record_ids != tuple(
            record_id for record_id, _ in payload_rows
        ) or record_ids != tuple(record_id for record_id, _ in scope_rows):
            raise DomainError(
                f"snapshot {key!r} record, payload, and scope indexes disagree in order"
            )
        if len(record_ids) != snapshot.record_count:
            raise DomainError(f"snapshot {key!r} index length disagrees with record_count")
        # Recheck the semantic index digests after reading, rather than relying
        # only on the earlier whole-file verification.  This closes the
        # verification/read boundary and binds the exact rows returned by this
        # loader to the snapshot receipt even under an adversarial file swap.
        if ordered_record_ids_sha256(record_ids) != snapshot.ordered_record_ids_sha256:
            raise DomainError(f"snapshot {key!r} record index digest disagrees")
        if (
            ordered_record_payloads_sha256(payload_rows)
            != snapshot.ordered_record_payloads_sha256
        ):
            raise DomainError(f"snapshot {key!r} payload index digest disagrees")
        if ordered_record_scopes_sha256(scope_rows) != snapshot.ordered_record_scopes_sha256:
            raise DomainError(f"snapshot {key!r} scope index digest disagrees")

        require_historical_mp20_archive(
            receipt, root,
            tuple((record_id, digest, scope) for (record_id, digest), (_, scope)
                  in zip(payload_rows, scope_rows, strict=True)),
        )

        data_by_digest: dict[str, list[ArtifactChecksum]] = {}
        for artifact in snapshot.artifacts:
            if artifact.role is ArtifactRole.DATA:
                data_by_digest.setdefault(artifact.sha256, []).append(artifact)
        for (record_id, payload_digest), (_, scope) in zip(
            payload_rows, scope_rows, strict=True
        ):
            scope.require_for(source)
            matches = tuple(data_by_digest.get(payload_digest, ()))
            if len(matches) != 1:
                raise DomainError(
                    f"snapshot {key!r} record {record_id!r} must map by digest to "
                    "exactly one pinned DATA artifact"
                )
            artifact = matches[0]
            if artifact.byte_count > max_cif_bytes:
                raise DomainError(
                    f"snapshot {key!r} record {record_id!r} exceeds max_cif_bytes"
                )
            path = _artifact_path(resolved_root, artifact)
            try:
                raw_bytes = path.read_bytes()
                cif_text = raw_bytes.decode("utf-8", errors="strict")
            except (OSError, UnicodeDecodeError) as exc:
                raise DomainError(
                    f"snapshot {key!r} record {record_id!r} is not readable UTF-8 CIF"
                ) from exc
            if not cif_text.strip():
                raise DomainError(f"snapshot {key!r} record {record_id!r} is empty")
            if sha256(raw_bytes).hexdigest() != payload_digest:
                raise InvariantViolation("verified DATA artifact changed while being read")
            reference = SourceRecordReference(source.source_id, snapshot.snapshot_id, record_id)
            candidate = DatasetCandidate(reference, cif_text, scope)
            if candidate.payload_sha256 != payload_digest:
                raise InvariantViolation("UTF-8 CIF payload digest changed during decoding")
            candidates.append(candidate)
            bindings.append(
                LocalCifCandidateBinding(
                    source_id=source.source_id,
                    snapshot_id=snapshot.snapshot_id,
                    source_record_id=record_id,
                    data_artifact_relative_path=artifact.relative_path,
                    payload_sha256=payload_digest,
                    scope_evidence_fingerprint=scope.fingerprint,
                )
            )

    manifest = LocalCifInputManifest(
        source_manifest_fingerprint=source_manifest.fingerprint,
        max_cif_bytes=max_cif_bytes,
        bindings=tuple(bindings),
    )
    return LoadedLocalCifCandidates(tuple(candidates), manifest)


def compile_local_cif_dataset(
    compiler: DatasetCompiler,
    *,
    expected_source_manifest_fingerprint: str,
    expected_crystallography_receipt_fingerprint: str,
    artifact_roots: Mapping[tuple[str, str], str | Path],
    destination: str | Path,
    split_modes: Sequence[SplitMode] = tuple(SplitMode),
    split_fractions: SplitFractions = DEFAULT_SPLIT_FRACTIONS,
    split_seed: int = 0,
    require_nonempty_splits: bool = True,
    records_per_shard: int = 1024,
    max_cif_bytes: int = _DEFAULT_MAX_CIF_BYTES,
) -> LocalDatasetBuildReceipt:
    """Load, compile, publish, and authenticate one local direct-CIF corpus.

    The destination must not exist.  The returned JSON-ready receipt is
    path-independent; callers should persist its fingerprint in a trusted
    system separate from the source and stored dataset trees.
    """

    if not isinstance(compiler, DatasetCompiler):
        raise TypeError("compiler must be DatasetCompiler")
    expected_crystallography = _require_digest(
        expected_crystallography_receipt_fingerprint,
        name="expected crystallography receipt fingerprint",
    )
    if compiler.crystallography_certification.receipt_fingerprint != expected_crystallography:
        raise DomainError("crystallography certification disagrees with the external pin")
    loaded = load_local_cif_candidates(
        compiler.source_manifest,
        expected_source_manifest_fingerprint=expected_source_manifest_fingerprint,
        artifact_roots=artifact_roots,
        max_cif_bytes=max_cif_bytes,
    )
    dataset = compiler.compile(
        loaded.candidates,
        artifact_roots=artifact_roots,
        split_modes=split_modes,
        split_fractions=split_fractions,
        split_seed=split_seed,
        require_nonempty_splits=require_nonempty_splits,
    )
    storage = write_compiled_dataset(
        dataset,
        destination,
        records_per_shard=records_per_shard,
    )
    with DatasetStore.open(
        destination,
        expected_build_manifest_fingerprint=dataset.manifest.fingerprint,
        expected_source_manifest_fingerprint=dataset.source_manifest.fingerprint,
        expected_storage_fingerprint=storage.fingerprint,
    ) as store:
        if len(store) != dataset.manifest.output_count:
            raise InvariantViolation(
                "published dataset count changed after final authentication"
            )
    return LocalDatasetBuildReceipt(
        local_input_manifest=loaded.manifest,
        source_manifest_fingerprint=dataset.source_manifest.fingerprint,
        policy_decision_fingerprint=dataset.policy_decision.fingerprint,
        crystallography_receipt_fingerprint=(
            dataset.manifest.crystallography_receipt_fingerprint
        ),
        target_builder_fingerprint=dataset.manifest.target_builder_fingerprint,
        build_manifest_fingerprint=dataset.manifest.fingerprint,
        storage_manifest_fingerprint=storage.fingerprint,
        input_count=dataset.manifest.input_count,
        accepted_pre_dedup_count=dataset.manifest.accepted_pre_dedup_count,
        output_count=dataset.manifest.output_count,
        rejection_count=dataset.manifest.rejection_count,
        records_per_shard=records_per_shard,
        split_manifest_fingerprints=dataset.manifest.split_manifest_fingerprints,
    )


__all__ = [
    "LoadedLocalCifCandidates",
    "LocalCifCandidateBinding",
    "LocalCifInputManifest",
    "LocalDatasetBuildReceipt",
    "compile_local_cif_dataset",
    "load_local_cif_candidates",
]
