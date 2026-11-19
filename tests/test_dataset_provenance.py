from __future__ import annotations

import copy
from dataclasses import replace
from pathlib import Path

import pytest

from crystalatlas.dataset_provenance import (
    SELECTED_SOURCE_IDS,
    ArtifactChecksum,
    ArtifactRole,
    DatasetSnapshot,
    DatasetSourceManifest,
    DatasetSourceReceipt,
    DatasetUseRequest,
    Permission,
    RecordScopePolicy,
    SourceRecordScopeEvidence,
    alexandria_source,
    canonical_record_index_bytes,
    canonical_record_payload_index_bytes,
    canonical_record_scope_index_bytes,
    crystallography_open_database_source,
    evaluate_dataset_use,
    materials_project_core_source,
    nomad_open_source,
    oqmd_generated_prototypes_source,
    ordered_record_ids_sha256,
    ordered_record_payloads_sha256,
    ordered_record_scopes_sha256,
)
from crystalatlas.errors import DomainError

_TERMS_BYTES = b"archived official terms evidence"


def _cod_scope(record_id: str) -> SourceRecordScopeEvidence:
    return SourceRecordScopeEvidence(
        policy=RecordScopePolicy.COD_MAIN_CC0,
        claims=(("collection", "cod-main"), ("record_license", "CC0-1.0")),
        source_metadata_sha256="c" * 64,
        evidence_uri=f"https://example.org/cod/metadata/{record_id}",
        attribution=("Crystallography Open Database",),
    )


def _snapshot(tmp_path: Path, source_id: str) -> tuple[DatasetSnapshot, Path]:
    root = tmp_path / source_id
    root.mkdir()
    artifact = root / "records.cif.gz"
    artifact.write_bytes(b"immutable-test-snapshot")
    terms = root / "terms.html"
    terms.write_bytes(_TERMS_BYTES)
    ids = ("row-1", "row-2")
    payloads = (("row-1", "a" * 64), ("row-2", "b" * 64))
    scopes = tuple((record_id, _cod_scope(record_id)) for record_id in ids)
    selection = root / "selection.json"
    selection.write_bytes(b'{"query":"fixture-v1"}')
    record_index = root / "record-index.json"
    record_index.write_bytes(canonical_record_index_bytes(ids))
    payload_index = root / "record-payload-index.json"
    payload_index.write_bytes(canonical_record_payload_index_bytes(payloads))
    scope_index = root / "record-scope-index.json"
    scope_index.write_bytes(canonical_record_scope_index_bytes(scopes))
    return (
        DatasetSnapshot(
            source_id=source_id,
            snapshot_id="release-2026-08-01",
            retrieved_utc="2026-08-25T12:00:00Z",
            retrieval_uri="https://example.org/releases/2026-08-01",
            selection_spec_sha256=ArtifactChecksum.from_file(
                selection, role=ArtifactRole.SELECTION_SPEC
            ).sha256,
            ordered_record_ids_sha256=ordered_record_ids_sha256(ids),
            ordered_record_payloads_sha256=ordered_record_payloads_sha256(payloads),
            ordered_record_scopes_sha256=ordered_record_scopes_sha256(scopes),
            record_count=len(ids),
            artifacts=(
                ArtifactChecksum.from_file(artifact, role=ArtifactRole.DATA),
                ArtifactChecksum.from_file(
                    payload_index, role=ArtifactRole.PAYLOAD_INDEX
                ),
                ArtifactChecksum.from_file(record_index, role=ArtifactRole.RECORD_INDEX),
                ArtifactChecksum.from_file(scope_index, role=ArtifactRole.SCOPE_INDEX),
                ArtifactChecksum.from_file(selection, role=ArtifactRole.SELECTION_SPEC),
                ArtifactChecksum.from_file(terms, role=ArtifactRole.TERMS),
            ),
        ),
        root,
    )


def test_selected_source_factories_are_narrow_and_explicit() -> None:
    factories = (
        crystallography_open_database_source,
        nomad_open_source,
        alexandria_source,
        oqmd_generated_prototypes_source,
        materials_project_core_source,
    )
    declarations = tuple(
        factory(terms_evidence_sha256="a" * 64, reviewed_on="2026-08-25")
        for factory in factories
    )
    assert tuple(sorted(source.source_id for source in declarations)) == SELECTED_SOURCE_IDS
    assert all(source.license.training_use is Permission.ALLOWED for source in declarations)
    assert (
        "restricted"
        in nomad_open_source(
            terms_evidence_sha256="a" * 64, reviewed_on="2026-08-25"
        ).data_scope
    )
    assert (
        "ICSD"
        in oqmd_generated_prototypes_source(
            terms_evidence_sha256="a" * 64, reviewed_on="2026-08-25"
        ).data_scope
    )
    assert (
        "MPContribs"
        in materials_project_core_source(
            terms_evidence_sha256="a" * 64, reviewed_on="2026-08-25"
        ).data_scope
    )


def test_snapshot_manifest_binds_terms_index_and_artifacts(tmp_path: Path) -> None:
    terms_path = tmp_path / "terms-evidence"
    terms_path.write_bytes(_TERMS_BYTES)
    source = crystallography_open_database_source(
        terms_evidence_sha256=ArtifactChecksum.from_file(
            terms_path, role=ArtifactRole.TERMS
        ).sha256,
        reviewed_on="2026-08-25",
    )
    snapshot, root = _snapshot(tmp_path, source.source_id)
    receipt = DatasetSourceReceipt(source, snapshot)
    manifest = DatasetSourceManifest((receipt,))
    restored = DatasetSourceManifest.from_json(manifest.to_json())

    assert restored == manifest
    assert restored.fingerprint == manifest.fingerprint
    snapshot.require_files(root)
    assert evaluate_dataset_use(manifest, DatasetUseRequest()).authorized

    (root / "records.cif.gz").write_bytes(b"tampered")
    with pytest.raises(DomainError, match="size mismatch|hash mismatch"):
        snapshot.require_files(root)


def test_any_unresolved_requested_permission_denies_the_manifest(tmp_path: Path) -> None:
    terms_path = tmp_path / "terms-evidence"
    terms_path.write_bytes(_TERMS_BYTES)
    source = crystallography_open_database_source(
        terms_evidence_sha256=ArtifactChecksum.from_file(
            terms_path, role=ArtifactRole.TERMS
        ).sha256,
        reviewed_on="2026-08-25",
    )
    source = replace(
        source,
        license=replace(source.license, training_use=Permission.UNRESOLVED),
    )
    snapshot, _ = _snapshot(tmp_path, source.source_id)
    decision = evaluate_dataset_use(
        DatasetSourceManifest((DatasetSourceReceipt(source, snapshot),)),
        DatasetUseRequest(train_model=True),
    )
    assert not decision.authorized
    with pytest.raises(DomainError, match="dataset use denied"):
        decision.require_authorized()


def test_ordered_record_index_is_order_sensitive_and_duplicate_free() -> None:
    assert ordered_record_ids_sha256(("a", "b")) != ordered_record_ids_sha256(("b", "a"))
    with pytest.raises(DomainError, match="duplicate"):
        ordered_record_ids_sha256(("a", "a"))
    with pytest.raises(DomainError, match="cannot be empty"):
        ordered_record_ids_sha256(())


def test_ordered_payload_index_binds_identifiers_order_and_content() -> None:
    first = (("a", "1" * 64), ("b", "2" * 64))
    assert ordered_record_payloads_sha256(first) != ordered_record_payloads_sha256(
        tuple(reversed(first))
    )
    assert ordered_record_payloads_sha256(first) != ordered_record_payloads_sha256(
        (("a", "1" * 64), ("b", "3" * 64))
    )
    with pytest.raises(DomainError, match="repeat identifiers"):
        ordered_record_payloads_sha256((("a", "1" * 64), ("a", "2" * 64)))


def test_snapshot_requires_archived_selection_and_index_evidence(tmp_path: Path) -> None:
    snapshot, _ = _snapshot(tmp_path, "cod")
    with pytest.raises(DomainError, match="ordered-record-payload-index"):
        replace(
            snapshot,
            artifacts=tuple(
                item
                for item in snapshot.artifacts
                if item.sha256 != snapshot.ordered_record_payloads_sha256
            ),
        )


def test_snapshot_artifact_roles_are_exact_and_require_data(tmp_path: Path) -> None:
    snapshot, _ = _snapshot(tmp_path, "cod")
    without_data = tuple(
        replace(item, role=ArtifactRole.AUXILIARY)
        if item.role is ArtifactRole.DATA
        else item
        for item in snapshot.artifacts
    )
    with pytest.raises(DomainError, match="at least one data artifact"):
        replace(snapshot, artifacts=without_data)

    wrong_selection = tuple(
        replace(item, sha256="f" * 64)
        if item.role is ArtifactRole.SELECTION_SPEC
        else item
        for item in snapshot.artifacts
    )
    with pytest.raises(DomainError, match="selection-spec artifact digest disagrees"):
        replace(snapshot, artifacts=wrong_selection)

    duplicate_scope_role = tuple(
        replace(item, role=ArtifactRole.SCOPE_INDEX)
        if item.role is ArtifactRole.PAYLOAD_INDEX
        else item
        for item in snapshot.artifacts
    )
    with pytest.raises(DomainError, match="ordered-record-payload-index artifact"):
        replace(snapshot, artifacts=duplicate_scope_role)


def test_terms_digest_must_come_from_the_terms_role(tmp_path: Path) -> None:
    terms_path = tmp_path / "terms-evidence"
    terms_path.write_bytes(_TERMS_BYTES)
    source = crystallography_open_database_source(
        terms_evidence_sha256=ArtifactChecksum.from_file(
            terms_path, role=ArtifactRole.TERMS
        ).sha256,
        reviewed_on="2026-08-25",
    )
    snapshot, _ = _snapshot(tmp_path, source.source_id)
    wrong_terms = tuple(
        replace(item, sha256="e" * 64)
        if item.role is ArtifactRole.TERMS
        else item
        for item in snapshot.artifacts
    )
    with pytest.raises(DomainError, match="terms artifact digest disagrees"):
        DatasetSourceReceipt(source, replace(snapshot, artifacts=wrong_terms))


def test_manifest_json_rejects_duplicate_keys_at_every_depth(tmp_path: Path) -> None:
    terms_path = tmp_path / "terms-evidence"
    terms_path.write_bytes(_TERMS_BYTES)
    source = crystallography_open_database_source(
        terms_evidence_sha256=ArtifactChecksum.from_file(
            terms_path, role=ArtifactRole.TERMS
        ).sha256,
        reviewed_on="2026-08-25",
    )
    snapshot, _ = _snapshot(tmp_path, source.source_id)
    manifest = DatasetSourceManifest((DatasetSourceReceipt(source, snapshot),))

    root_duplicate = manifest.to_json().replace(
        '"schema":', '"schema":"shadow","schema":', 1
    )
    with pytest.raises(DomainError, match="duplicate JSON key 'schema'"):
        DatasetSourceManifest.from_json(root_duplicate)

    nested_duplicate = manifest.to_json().replace(
        '"source_id":"cod"', '"source_id":"shadow","source_id":"cod"', 1
    )
    with pytest.raises(DomainError, match="duplicate JSON key 'source_id'"):
        DatasetSourceManifest.from_json(nested_duplicate)


def test_read_json_checks_an_external_manifest_and_receipt_pin(tmp_path: Path) -> None:
    terms_path = tmp_path / "terms-evidence"
    terms_path.write_bytes(_TERMS_BYTES)
    source = crystallography_open_database_source(
        terms_evidence_sha256=ArtifactChecksum.from_file(
            terms_path, role=ArtifactRole.TERMS
        ).sha256,
        reviewed_on="2026-08-25",
    )
    snapshot, _ = _snapshot(tmp_path, source.source_id)
    receipt = DatasetSourceReceipt(source, snapshot)
    manifest = DatasetSourceManifest((receipt,))
    manifest_path = tmp_path / "manifest.json"
    receipt_path = tmp_path / "receipt.json"
    manifest.write_json(manifest_path)
    receipt_path.write_text(receipt.to_json(), encoding="utf-8")

    assert not tuple(tmp_path.glob(".manifest.json.*.tmp"))

    assert (
        DatasetSourceManifest.read_json(
            manifest_path, expected_fingerprint=manifest.fingerprint
        )
        == manifest
    )
    assert (
        DatasetSourceReceipt.read_json(
            receipt_path, expected_fingerprint=receipt.fingerprint
        )
        == receipt
    )
    with pytest.raises(DomainError, match="externally pinned fingerprint"):
        DatasetSourceManifest.read_json(manifest_path, expected_fingerprint="f" * 64)
    with pytest.raises(DomainError, match="externally pinned fingerprint"):
        DatasetSourceReceipt.read_json(receipt_path, expected_fingerprint="f" * 64)


@pytest.mark.parametrize(
    ("path", "replacement"),
    (
        (("receipts", 0, "source", "license", "training_use"), {"bad": "shape"}),
        (("receipts", 0, "snapshot", "artifacts", 0, "role"), {"bad": "shape"}),
        (("receipts", 0, "snapshot", "artifacts", 0, "byte_count"), "large"),
        (("receipts", 0, "source", "citations", 0), 7),
    ),
)
def test_malformed_nested_manifest_fields_fail_as_domain_errors(
    tmp_path: Path, path: tuple[str | int, ...], replacement: object
) -> None:
    terms_path = tmp_path / "terms-evidence"
    terms_path.write_bytes(_TERMS_BYTES)
    source = crystallography_open_database_source(
        terms_evidence_sha256=ArtifactChecksum.from_file(
            terms_path, role=ArtifactRole.TERMS
        ).sha256,
        reviewed_on="2026-08-25",
    )
    snapshot, _ = _snapshot(tmp_path, source.source_id)
    payload = copy.deepcopy(
        DatasetSourceManifest((DatasetSourceReceipt(source, snapshot),)).to_dict()
    )
    cursor: object = payload
    for key in path[:-1]:
        assert isinstance(cursor, dict | list)
        cursor = cursor[key]  # type: ignore[index]
    assert isinstance(cursor, dict | list)
    cursor[path[-1]] = replacement  # type: ignore[index]

    with pytest.raises(DomainError):
        DatasetSourceManifest.from_dict(payload)


def test_restricted_record_scope_fails_closed() -> None:
    restricted = SourceRecordScopeEvidence(
        policy=RecordScopePolicy.NOMAD_OPEN_CC_BY,
        claims=(
            ("access", "restricted"),
            ("record_license", "CC-BY-4.0"),
            ("reusable_structure", "true"),
        ),
        source_metadata_sha256="d" * 64,
        evidence_uri="https://example.org/nomad/restricted-row",
        attribution=("Example submitter",),
    )
    source = nomad_open_source(terms_evidence_sha256="e" * 64, reviewed_on="2026-08-25")
    with pytest.raises(DomainError, match="outside the declared legal scope"):
        restricted.require_for(source)


def test_materials_project_scope_explicitly_excludes_gnome_and_mpcontribs() -> None:
    source = materials_project_core_source(
        terms_evidence_sha256="e" * 64,
        reviewed_on="2026-08-25",
    )
    claims = (
        ("access_controlled", "false"),
        ("collection", "core"),
        ("gnome", "false"),
        ("mpcontribs", "false"),
        ("record_license", "CC-BY-4.0"),
    )
    evidence = SourceRecordScopeEvidence(
        policy=RecordScopePolicy.MP_CORE_CC_BY,
        claims=claims,
        source_metadata_sha256="f" * 64,
        evidence_uri="https://example.org/materials-project/core-row",
        attribution=("Materials Project",),
    )
    evidence.require_for(source)
    with pytest.raises(DomainError, match="gnome='true'"):
        replace(evidence, claims=tuple(
            (key, "true" if key == "gnome" else value) for key, value in claims
        )).require_for(source)
