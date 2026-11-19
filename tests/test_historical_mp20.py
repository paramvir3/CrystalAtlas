"""Synthetic archives test enforcement; these fixtures grant no real data rights."""

from __future__ import annotations

import json
from dataclasses import replace
from hashlib import sha256
from pathlib import Path

import pytest
from test_dataset_builder import _compiler, _FixtureTargetBuilder
from test_dataset_orchestration import _source_manifest

import crystalatlas.dataset_provenance as provenance
from crystalatlas.dataset_builder import DatasetCandidate, SourceRecordReference
from crystalatlas.dataset_orchestration import load_local_cif_candidates
from crystalatlas.dataset_provenance import (
    ArtifactChecksum,
    ArtifactRole,
    DatasetSourceManifest,
    DatasetSourceReceipt,
    Permission,
    RecordScopePolicy,
    SourceRecordScopeEvidence,
    canonical_record_scope_index_bytes,
    historical_cdvae_mp20_source,
    ordered_record_scopes_sha256,
)
from crystalatlas.errors import DomainError
from crystalatlas.historical_mp20 import require_historical_mp20_archive


def _bytes(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def _archive(tmp_path, monkeypatch, *, mutate_metadata=None, mutate_binding=None):
    manifest, roots = _source_manifest(tmp_path)
    old = manifest.receipts[0]
    root = next(iter(roots.values()))
    source = historical_cdvae_mp20_source(
        reviewed_on="2026-09-30", training_use=Permission.ALLOWED
    )
    claims = dict(provenance._SCOPE_REQUIREMENTS[RecordScopePolicy.CDVAE_MP20_HISTORICAL])
    project = Path(__file__).resolve().parents[1]
    evidence = project / "artifacts/mp20_pilot_2026-09-14"
    additions = []

    def artifact(name, payload, role=ArtifactRole.AUXILIARY):
        (root / name).write_bytes(payload)
        result = ArtifactChecksum(role, name, sha256(payload).hexdigest(), len(payload))
        additions.append(result)
        return result

    artifact("review.md", (evidence / "SOURCE_REVIEW.md").read_bytes())
    artifact(
        "terms.json",
        (evidence / "mp-provider-clarification-45437.json").read_bytes(),
        ArtifactRole.TERMS,
    )
    container = artifact("train.csv", b"synthetic test container, not MP20 data\n")
    claims["source_container_sha256"] = container.sha256
    data = next(a for a in old.snapshot.artifacts if a.role is ArtifactRole.DATA)
    record_id = "synthetic-row-0001"
    metadata = {
        "schema": "crystalatlas.benchmark-cif-row.v1",
        "split": "train",
        "row_index": 0,
        "material_id": record_id,
        "cif_sha256": data.sha256,
        "metadata": {"material_id": record_id},
    }
    if mutate_metadata:
        mutate_metadata(metadata)
    meta = artifact("metadata.json", _bytes(metadata))
    uri = (
        "https://raw.githubusercontent.com/txie-93/cdvae/"
        f"{claims['source_revision']}/data/mp_20/train.csv"
    )
    binding = {
        "cif_path": data.relative_path,
        "metadata_path": meta.relative_path,
        "payload_sha256": data.sha256,
        "selection_rank": 0,
        "source_container_path": "train.csv",
        "source_container_sha256": container.sha256,
        "source_metadata_sha256": meta.sha256,
        "source_partition": "train",
        "source_record_id": record_id,
        "source_row_index": 0,
        "source_uri": uri,
    }
    if mutate_binding:
        mutate_binding(binding)
    bound = artifact(
        "bindings.json",
        _bytes(
            {
                "schema": "crystalatlas.mp20-selected-row-bindings.v1",
                "bindings": [binding],
            }
        ),
    )
    claims["selected_bindings_sha256"] = bound.sha256
    # Explicitly scoped synthetic trust anchors; never modify production pins.
    monkeypatch.setitem(
        provenance._SCOPE_REQUIREMENTS, RecordScopePolicy.CDVAE_MP20_HISTORICAL, claims
    )
    scope = SourceRecordScopeEvidence(
        RecordScopePolicy.CDVAE_MP20_HISTORICAL,
        tuple(claims.items()),
        meta.sha256,
        uri,
        ("Synthetic archive test only",),
    )
    scope_rows = ((record_id, scope),)
    scoped = artifact(
        "historical-scopes.json",
        canonical_record_scope_index_bytes(scope_rows),
        ArtifactRole.SCOPE_INDEX,
    )
    retained = tuple(
        a
        for a in old.snapshot.artifacts
        if a.role not in {ArtifactRole.TERMS, ArtifactRole.SCOPE_INDEX}
    )
    snapshot = replace(
        old.snapshot,
        source_id=source.source_id,
        artifacts=retained + tuple(additions),
        ordered_record_scopes_sha256=ordered_record_scopes_sha256(scope_rows),
    )
    assert scoped.sha256 == snapshot.ordered_record_scopes_sha256
    receipt = DatasetSourceReceipt(source, snapshot)
    manifest = DatasetSourceManifest((receipt,))
    roots = {(source.source_id, snapshot.snapshot_id): root}
    candidate = DatasetCandidate(
        SourceRecordReference(source.source_id, snapshot.snapshot_id, record_id),
        (root / data.relative_path).read_text(),
        scope,
    )
    return manifest, roots, candidate


def _forbidden_builder(raw):
    raise AssertionError("scientific preprocessing must not run for invalid provenance")


def test_factory_is_unresolved_by_default_and_does_not_grant_other_uses():
    source = historical_cdvae_mp20_source(reviewed_on="2026-09-30")
    assert source.license.training_use is Permission.UNRESOLVED
    allowed = historical_cdvae_mp20_source(
        reviewed_on="2026-09-30", training_use=Permission.ALLOWED
    )
    assert allowed.license.training_use is Permission.ALLOWED
    for field in ("raw_redistribution", "derived_redistribution", "commercial_use"):
        assert getattr(allowed.license, field) is Permission.UNRESOLVED


def test_valid_archive_loads_and_direct_compiler_checks_it(tmp_path, monkeypatch):
    manifest, roots, candidate = _archive(tmp_path, monkeypatch)
    loaded = load_local_cif_candidates(
        manifest,
        expected_source_manifest_fingerprint=manifest.fingerprint,
        artifact_roots=roots,
    )
    assert loaded.candidates == (candidate,)
    compiler = _compiler(manifest, _FixtureTargetBuilder(_forbidden_builder))
    compiler._verify_snapshot_contract((candidate,), roots)


@pytest.mark.parametrize(
    "field,value",
    [
        ("material_id", "wrong"),
        ("split", "test"),
        ("row_index", True),
        ("row_index", -1),
        ("cif_sha256", "a" * 64),
        ("schema", "wrong"),
        ("metadata", {"material_id": "wrong"}),
        ("metadata", {"material_id": 3}),
    ],
)
def test_self_consistent_hashes_do_not_hide_false_row_metadata(
    tmp_path,
    monkeypatch,
    field,
    value,
):
    manifest, roots, candidate = _archive(
        tmp_path, monkeypatch, mutate_metadata=lambda row: row.update({field: value})
    )
    with pytest.raises(DomainError, match="metadata"):
        load_local_cif_candidates(
            manifest,
            expected_source_manifest_fingerprint=manifest.fingerprint,
            artifact_roots=roots,
        )
    with pytest.raises(DomainError, match="metadata"):
        _compiler(manifest, _FixtureTargetBuilder(_forbidden_builder)).compile(
            (candidate,), artifact_roots=roots, split_modes=()
        )


@pytest.mark.parametrize(
    "field,value",
    [
        ("selection_rank", 1),
        ("selection_rank", False),
        ("source_row_index", False),
        ("source_record_id", "other"),
        ("source_partition", "test"),
        ("source_container_sha256", "f" * 64),
        ("metadata_path", "wrong.json"),
        ("source_uri", "https://example.invalid/train.csv"),
        ("extra", "claim"),
    ],
)
def test_false_ordered_binding_is_rejected_in_both_entry_points(
    tmp_path,
    monkeypatch,
    field,
    value,
):
    manifest, roots, candidate = _archive(
        tmp_path, monkeypatch, mutate_binding=lambda row: row.update({field: value})
    )
    with pytest.raises(DomainError, match="binding"):
        load_local_cif_candidates(
            manifest,
            expected_source_manifest_fingerprint=manifest.fingerprint,
            artifact_roots=roots,
        )
    with pytest.raises(DomainError, match="binding"):
        _compiler(manifest, _FixtureTargetBuilder(_forbidden_builder)).compile(
            (candidate,), artifact_roots=roots, split_modes=()
        )


@pytest.mark.parametrize(
    "filename", ["metadata.json", "review.md", "bindings.json", "train.csv"]
)
def test_missing_archived_evidence_cannot_be_replaced_by_a_hash(
    tmp_path,
    monkeypatch,
    filename,
):
    manifest, roots, candidate = _archive(tmp_path, monkeypatch)
    receipt = manifest.receipts[0]
    stripped = replace(
        receipt.snapshot,
        artifacts=tuple(a for a in receipt.snapshot.artifacts if a.relative_path != filename),
    )
    manifest = DatasetSourceManifest((DatasetSourceReceipt(receipt.source, stripped),))
    with pytest.raises(DomainError, match="cited role"):
        load_local_cif_candidates(
            manifest,
            expected_source_manifest_fingerprint=manifest.fingerprint,
            artifact_roots=roots,
        )
    with pytest.raises(DomainError, match="cited role"):
        _compiler(manifest, _FixtureTargetBuilder(_forbidden_builder)).compile(
            (candidate,), artifact_roots=roots, split_modes=()
        )


def test_read_boundary_rehashes_metadata_and_rejects_symlinks(tmp_path, monkeypatch):
    manifest, roots, candidate = _archive(tmp_path, monkeypatch)
    root = next(iter(roots.values()))
    receipt = manifest.receipts[0]
    rows = (
        (
            candidate.reference.source_record_id,
            candidate.payload_sha256,
            candidate.scope_evidence,
        ),
    )
    receipt.snapshot.require_files(root)
    (root / "metadata.json").write_text("changed after preliminary integrity check")
    with pytest.raises(DomainError, match="changed"):
        require_historical_mp20_archive(receipt, root, rows)
    (root / "symlink-root").symlink_to(root, target_is_directory=True)
    with pytest.raises(DomainError, match="symbolic"):
        require_historical_mp20_archive(receipt, root / "symlink-root", rows)


def test_extra_modern_scope_claim_is_not_accepted(tmp_path, monkeypatch):
    manifest, _, candidate = _archive(tmp_path, monkeypatch)
    scope = replace(
        candidate.scope_evidence,
        claims=candidate.scope_evidence.claims + (("mpcontribs", "false"),),
    )
    with pytest.raises(DomainError, match="exactly"):
        scope.require_for(manifest.receipts[0].source)


@pytest.mark.parametrize("permission", [True, "allowed", None])
def test_training_permission_requires_the_explicit_enum(permission):
    with pytest.raises(DomainError):
        historical_cdvae_mp20_source(reviewed_on="2026-09-30", training_use=permission)


def test_new_policy_serializes_without_changing_existing_receipt_schemas(tmp_path, monkeypatch):
    manifest, _, _ = _archive(tmp_path, monkeypatch)
    restored = DatasetSourceManifest.from_json(manifest.to_json())
    assert restored == manifest
    assert restored.fingerprint == manifest.fingerprint


@pytest.mark.parametrize(
    "field,value",
    [
        ("reviewed_lineage", "no-third-party-content"),
        ("record_license_metadata", "provided"),
        ("provider_signed", "true"),
        ("upstream_database_version", "current"),
    ],
)
def test_historical_unknowns_cannot_be_relabelled_as_verified(
    tmp_path,
    monkeypatch,
    field,
    value,
):
    manifest, _, candidate = _archive(tmp_path, monkeypatch)
    claims = dict(candidate.scope_evidence.claims)
    claims[field] = value
    scope = replace(candidate.scope_evidence, claims=tuple(claims.items()))
    with pytest.raises(DomainError, match="legal scope"):
        scope.require_for(manifest.receipts[0].source)


@pytest.mark.parametrize(
    "filename", ["metadata.json", "review.md", "bindings.json", "train.csv"]
)
def test_wrong_semantic_role_does_not_satisfy_a_matching_digest(
    tmp_path, monkeypatch, filename
):
    manifest, roots, candidate = _archive(tmp_path, monkeypatch)
    receipt = manifest.receipts[0]
    snapshot = replace(
        receipt.snapshot,
        artifacts=tuple(
            replace(a, role=ArtifactRole.DATA) if a.relative_path == filename else a
            for a in receipt.snapshot.artifacts
        ),
    )
    receipt = DatasetSourceReceipt(receipt.source, snapshot)
    with pytest.raises(DomainError, match="cited role"):
        require_historical_mp20_archive(
            receipt,
            next(iter(roots.values())),
            (
                (
                    candidate.reference.source_record_id,
                    candidate.payload_sha256,
                    candidate.scope_evidence,
                ),
            ),
        )


def test_permission_cannot_be_expanded_by_a_custom_historical_declaration(
    tmp_path, monkeypatch
):
    manifest, roots, candidate = _archive(tmp_path, monkeypatch)
    receipt = manifest.receipts[0]
    source = replace(
        receipt.source,
        license=replace(receipt.source.license, commercial_use=Permission.ALLOWED),
    )
    receipt = DatasetSourceReceipt(source, receipt.snapshot)
    with pytest.raises(DomainError, match="reviewed declaration"):
        require_historical_mp20_archive(
            receipt,
            next(iter(roots.values())),
            (
                (
                    candidate.reference.source_record_id,
                    candidate.payload_sha256,
                    candidate.scope_evidence,
                ),
            ),
        )
