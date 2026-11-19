"""Archive binding for the narrowly pinned historical MP20 scope.

No downloading, permission inference, canonicalization or filtering occurs here.
Both ingestion entry points call this validator before any target is built.
"""

from __future__ import annotations

import os
import stat
from hashlib import sha256
from pathlib import Path

from .dataset_provenance import (
    ArtifactChecksum,
    ArtifactRole,
    DatasetSourceReceipt,
    RecordScopePolicy,
    SourceRecordScopeEvidence,
    _strict_json_loads,
    historical_cdvae_mp20_source,
)
from .errors import DomainError


def _read_bound(root: Path, artifact: ArtifactChecksum, limit: int) -> bytes:
    if artifact.byte_count > limit:
        raise DomainError("historical MP20 artifact exceeds its role's byte limit")
    path = root / artifact.relative_path
    prefix = root
    for part in Path(artifact.relative_path).parts:
        prefix /= part
        if prefix.is_symlink():
            raise DomainError("historical MP20 artifact uses a symbolic link")
    try:
        descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
        with os.fdopen(descriptor, "rb") as stream:
            if not stat.S_ISREG(os.fstat(stream.fileno()).st_mode):
                raise DomainError("historical MP20 artifact is not a regular file")
            raw = stream.read(limit + 1)
    except OSError as exc:
        raise DomainError("cannot read historical MP20 artifact") from exc
    if len(raw) != artifact.byte_count or sha256(raw).hexdigest() != artifact.sha256:
        raise DomainError("historical MP20 artifact changed during its verified read")
    return raw


def require_historical_mp20_archive(
    receipt: DatasetSourceReceipt,
    root: str | Path,
    rows: tuple[tuple[str, str, SourceRecordScopeEvidence], ...],
) -> None:
    """Require metadata/CIF/ordered-binding evidence, including for direct callers.

    Index hashes and all other snapshot files are checked by the caller's generic
    snapshot contract. Here the actual bytes consumed are rehashed, not trusted
    merely because an earlier integrity pass saw a matching path. This is not a
    sandbox against an adversary replacing ancestor directories during a read.
    """
    if receipt.source.record_scope_policy is not RecordScopePolicy.CDVAE_MP20_HISTORICAL:
        return
    if not rows:
        raise DomainError("historical MP20 archive has no records")
    base = Path(root)
    if base.is_symlink():
        raise DomainError("historical MP20 root must not be a symbolic link")
    base = base.resolve(strict=True)
    for _, _, scope in rows:
        scope.require_for(receipt.source)
    claims = dict(rows[0][2].claims)
    template = historical_cdvae_mp20_source(
        reviewed_on=receipt.source.license.reviewed_on,
        training_use=receipt.source.license.training_use,
    )
    if receipt.source != template:
        raise DomainError(
            "historical MP20 requires its reviewed declaration and dated evidence"
        )
    by_role_digest: dict[tuple[ArtifactRole, str], list[ArtifactChecksum]] = {}
    for artifact in receipt.snapshot.artifacts:
        by_role_digest.setdefault((artifact.role, artifact.sha256), []).append(artifact)

    def artifact_for(role: ArtifactRole, digest: str) -> ArtifactChecksum:
        matches = by_role_digest.get((role, digest), [])
        if len(matches) != 1:
            raise DomainError("historical MP20 requires exactly one artifact of the cited role")
        return matches[0]

    def read_json(artifact: ArtifactChecksum) -> object:
        try:
            return _strict_json_loads(
                _read_bound(base, artifact, 2 * 1024 * 1024).decode("utf-8"),
                "historical MP20 metadata/bindings",
            )
        except UnicodeError as exc:
            raise DomainError("historical MP20 metadata is not UTF-8") from exc

    _read_bound(
        base,
        artifact_for(ArtifactRole.TERMS, template.license.terms_evidence_sha256),
        2 * 1024 * 1024,
    )
    _read_bound(
        base,
        artifact_for(ArtifactRole.AUXILIARY, claims["review_artifact_sha256"]),
        2 * 1024 * 1024,
    )
    _read_bound(
        base,
        artifact_for(ArtifactRole.AUXILIARY, claims["source_container_sha256"]),
        64 * 1024 * 1024,
    )
    bindings = read_json(
        artifact_for(ArtifactRole.AUXILIARY, claims["selected_bindings_sha256"])
    )
    if (
        not isinstance(bindings, dict)
        or set(bindings) != {"schema", "bindings"}
        or (
            bindings["schema"] != "crystalatlas.mp20-selected-row-bindings.v1"
            or not isinstance(bindings["bindings"], list)
            or len(bindings["bindings"]) != len(rows)
        )
    ):
        raise DomainError("historical MP20 selected bindings are malformed or incomplete")
    expected_uri = (
        "https://raw.githubusercontent.com/txie-93/cdvae/"
        f"{claims['source_revision']}/data/mp_20/train.csv"
    )
    for rank, ((record_id, payload_sha, scope), binding) in enumerate(
        zip(rows, bindings["bindings"], strict=True)
    ):
        cif_artifact = artifact_for(ArtifactRole.DATA, payload_sha)
        metadata_artifact = artifact_for(ArtifactRole.AUXILIARY, scope.source_metadata_sha256)
        metadata = read_json(metadata_artifact)
        if not isinstance(metadata, dict) or set(metadata) != {
            "schema",
            "split",
            "row_index",
            "material_id",
            "cif_sha256",
            "metadata",
        }:
            raise DomainError("historical MP20 row metadata has an invalid schema")
        fields = metadata["metadata"]
        if (
            metadata["schema"] != "crystalatlas.benchmark-cif-row.v1"
            or metadata["split"] != "train"
            or metadata["material_id"] != record_id
            or metadata["cif_sha256"] != payload_sha
            or type(metadata["row_index"]) is not int
            or metadata["row_index"] < 0
            or not isinstance(fields, dict)
            or fields.get("material_id") != record_id
            or any(not isinstance(k, str) or not isinstance(v, str) for k, v in fields.items())
        ):
            raise DomainError("historical MP20 metadata does not describe the candidate row")
        expected_binding = {
            "cif_path": cif_artifact.relative_path,
            "metadata_path": metadata_artifact.relative_path,
            "payload_sha256": payload_sha,
            "selection_rank": rank,
            "source_container_path": "train.csv",
            "source_container_sha256": claims["source_container_sha256"],
            "source_metadata_sha256": scope.source_metadata_sha256,
            "source_partition": "train",
            "source_record_id": record_id,
            "source_row_index": metadata["row_index"],
            "source_uri": expected_uri,
        }
        if (
            not isinstance(binding, dict)
            or binding != expected_binding
            or type(binding.get("selection_rank")) is not int
            or type(binding.get("source_row_index")) is not int
            or scope.evidence_uri != expected_uri
        ):
            raise DomainError(
                "historical MP20 ordered row binding disagrees with archived bytes"
            )
        _read_bound(base, cif_artifact, 16 * 1024 * 1024)
