from __future__ import annotations

import json
from dataclasses import replace
from fractions import Fraction
from hashlib import sha256
from pathlib import Path

import numpy as np
import pytest

import crystalatlas.dataset_storage as storage_module
from crystalatlas.catalogue import MINIMAL_CATALOGUE
from crystalatlas.certification import certify_crystallography
from crystalatlas.config import DomainConfig
from crystalatlas.dataset_builder import (
    DatasetCandidate,
    DatasetCompiler,
    SourceRecordReference,
)
from crystalatlas.dataset_identity import DeduplicationLevel, IdentityPolicy
from crystalatlas.dataset_provenance import (
    ArtifactChecksum,
    ArtifactRole,
    DatasetSnapshot,
    DatasetSourceManifest,
    DatasetSourceReceipt,
    DatasetUseRequest,
    RecordScopePolicy,
    SourceRecordScopeEvidence,
    canonical_record_index_bytes,
    canonical_record_payload_index_bytes,
    canonical_record_scope_index_bytes,
    crystallography_open_database_source,
    ordered_record_ids_sha256,
    ordered_record_payloads_sha256,
    ordered_record_scopes_sha256,
)
from crystalatlas.dataset_splits import SplitFractions, SplitMode
from crystalatlas.dataset_storage import DatasetStore, StoredFile
from crystalatlas.decoder import CrystalScaffold, ExactCrystalDecoder, OrbitAssignment
from crystalatlas.errors import DomainError
from crystalatlas.ingestion import (
    CanonicalizationReceipt,
    CanonicalizedStructure,
    CanonicalTrainingRecord,
    RawPeriodicStructure,
)
from crystalatlas.training_targets import ExactFixtureTrainingTarget, lift_exact_fixture_target


def _canonical_record(
    *, shift: Fraction, source_id: str, domain: DomainConfig | None = None
) -> CanonicalTrainingRecord[ExactFixtureTrainingTarget]:
    scaffold = CrystalScaffold.from_mapping(
        nu=1,
        space_group="P1",
        reduced_composition={6: 1, 7: 1, 8: 1},
        orbits=(
            OrbitAssignment("1a", 6, (Fraction(0), Fraction(0), Fraction(0))),
            OrbitAssignment("1a", 7, (Fraction(1, 7) + shift, Fraction(2, 9), Fraction(3, 11))),
            OrbitAssignment("1a", 8, (Fraction(2, 5), Fraction(3, 8), Fraction(4, 13))),
        ),
    )
    target = lift_exact_fixture_target(
        scaffold,
        np.diag([16.0, 25.0, 36.0]),
        catalogue=MINIMAL_CATALOGUE,
        domain=domain,
    )
    decoded = ExactCrystalDecoder(MINIMAL_CATALOGUE).decode(
        target.canonical_scaffold, target.metric
    )
    raw = RawPeriodicStructure.from_arrays(
        np.diag([4.0, 5.0, 6.0]),
        decoded.fractional_positions_float,
        tuple(int(value) for value in decoded.species),
        source_id=source_id,
    )
    receipt = CanonicalizationReceipt(
        source_fingerprint=raw.fingerprint,
        standardization_fingerprint="1" * 64,
        catalogue_fingerprint=MINIMAL_CATALOGUE.fingerprint,
        hall_number=1,
        hall_symbol="P 1",
        source_to_primitive=(
            (Fraction(1), Fraction(0), Fraction(0)),
            (Fraction(0), Fraction(1), Fraction(0)),
            (Fraction(0), Fraction(0), Fraction(1)),
        ),
        origin_shift=(Fraction(0), Fraction(0), Fraction(0)),
        free_origin_shift=(Fraction(0), Fraction(0), Fraction(0)),
        translation_index=1,
        symmetry_tolerance=1.0e-8,
        roundtrip_max_cartesian_error=0.0,
        symmetry_stability_verified=True,
        catalogue_closure_verified=True,
        wyckoff_multiplicity_verified=True,
        symmetry_verified=True,
        canonical_origin_verified=True,
        scaffold_roundtrip_verified=True,
    )
    return CanonicalTrainingRecord(
        raw,
        CanonicalizedStructure(raw, raw, target.canonical_scaffold, receipt),
        target,
    )


def _compiled_dataset(  # type: ignore[no-untyped-def]
    tmp_path: Path, domain: DomainConfig | None = None
):
    templates = (
        _canonical_record(shift=Fraction(0), source_id="template-a", domain=domain),
        _canonical_record(shift=Fraction(1, 17), source_id="template-b", domain=domain),
    )
    by_fingerprint = {item.raw_structure.fingerprint: item for item in templates}

    class TargetBuilder:
        certified_preprocessing_only = True
        builder_fingerprint = "9" * 64

        def __call__(
            self,
            raw: RawPeriodicStructure,
        ) -> CanonicalTrainingRecord[ExactFixtureTrainingTarget]:
            template = by_fingerprint[raw.fingerprint]
            receipt = replace(
                template.canonicalized.receipt,
                source_fingerprint=raw.fingerprint,
            )
            return replace(
                template,
                raw_structure=raw,
                canonicalized=replace(template.canonicalized, receipt=receipt),
            )

    source_root = tmp_path / "source"
    source_root.mkdir()
    artifact = source_root / "selected-cifs.tar.gz"
    artifact.write_bytes(b"pinned selected fixture")
    terms = source_root / "terms.html"
    terms.write_bytes(b"archived COD CC0 terms")
    terms_artifact = ArtifactChecksum.from_file(terms, role=ArtifactRole.TERMS)
    source = crystallography_open_database_source(
        terms_evidence_sha256=terms_artifact.sha256,
        reviewed_on="2026-08-25",
    )
    record_ids = ("phase-a", "phase-b")
    candidates_payloads = tuple(
        (record_id, template.raw_structure.fingerprint)
        for record_id, template in zip(record_ids, templates, strict=True)
    )
    scopes = tuple(
        (
            record_id,
            SourceRecordScopeEvidence(
                policy=RecordScopePolicy.COD_MAIN_CC0,
                claims=(("collection", "cod-main"), ("record_license", "CC0-1.0")),
                source_metadata_sha256=sha256(record_id.encode("utf-8")).hexdigest(),
                evidence_uri=f"https://example.org/cod/metadata/{record_id}",
                attribution=("Crystallography Open Database",),
            ),
        )
        for record_id in record_ids
    )
    selection = source_root / "selection.json"
    selection.write_bytes(b'{"query":"fixture-release-1"}')
    record_index = source_root / "record-index.json"
    record_index.write_bytes(canonical_record_index_bytes(record_ids))
    payload_index = source_root / "record-payload-index.json"
    payload_index.write_bytes(canonical_record_payload_index_bytes(candidates_payloads))
    scope_index = source_root / "record-scope-index.json"
    scope_index.write_bytes(canonical_record_scope_index_bytes(scopes))
    selection_artifact = ArtifactChecksum.from_file(
        selection,
        role=ArtifactRole.SELECTION_SPEC,
    )
    snapshot = DatasetSnapshot(
        source_id=source.source_id,
        snapshot_id="fixture-release-1",
        retrieved_utc="2026-08-25T12:00:00Z",
        retrieval_uri="https://example.org/cod/fixture-release-1",
        selection_spec_sha256=selection_artifact.sha256,
        ordered_record_ids_sha256=ordered_record_ids_sha256(record_ids),
        ordered_record_payloads_sha256=ordered_record_payloads_sha256(
            candidates_payloads
        ),
        ordered_record_scopes_sha256=ordered_record_scopes_sha256(scopes),
        record_count=2,
        artifacts=(
            ArtifactChecksum.from_file(artifact, role=ArtifactRole.DATA),
            ArtifactChecksum.from_file(payload_index, role=ArtifactRole.PAYLOAD_INDEX),
            ArtifactChecksum.from_file(record_index, role=ArtifactRole.RECORD_INDEX),
            ArtifactChecksum.from_file(scope_index, role=ArtifactRole.SCOPE_INDEX),
            selection_artifact,
            terms_artifact,
        ),
    )
    manifest = DatasetSourceManifest((DatasetSourceReceipt(source, snapshot),))
    candidates = tuple(
        DatasetCandidate(
            SourceRecordReference("cod", "fixture-release-1", record_id),
            template.raw_structure,
            scope,
        )
        for (record_id, template), (_, scope) in zip(
            zip(record_ids, templates, strict=True),
            scopes,
            strict=True,
        )
    )
    compiler = DatasetCompiler(
        catalogue=MINIMAL_CATALOGUE,
        crystallography_certification=certify_crystallography(
            MINIMAL_CATALOGUE,
            require_all_230=False,
            crosscheck_external_operations=False,
        ),
        source_manifest=manifest,
        use_request=DatasetUseRequest(train_model=True),
        target_builder=TargetBuilder(),
        identity_policy=IdentityPolicy(require_all_230=False),
        deduplication_level=DeduplicationLevel.POLYMORPH,
        require_all_230=False,
        require_external_operation_database=False,
    )
    return compiler.compile(
        candidates,
        artifact_roots={(source.source_id, snapshot.snapshot_id): source_root},
        split_modes=(SplitMode.COMPOSITION,),
        split_fractions=SplitFractions.from_values("1", "0", "0"),
    )


def _tree_bytes(root: Path) -> dict[str, bytes]:
    return {
        path.relative_to(root).as_posix(): path.read_bytes()
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def test_shards_are_deterministic_authenticated_and_lazy(tmp_path: Path) -> None:
    dataset = _compiled_dataset(tmp_path)
    first_root = tmp_path / "stored-a"
    second_root = tmp_path / "stored-b"
    first = storage_module.write_compiled_dataset(
        dataset, first_root, records_per_shard=1
    )
    second = storage_module.write_compiled_dataset(
        dataset, second_root, records_per_shard=1
    )

    assert first == second
    assert first.fingerprint == second.fingerprint
    assert _tree_bytes(first_root) == _tree_bytes(second_root)
    assert [shard.start for shard in first.shards] == [0, 1]
    assert [shard.stop for shard in first.shards] == [1, 2]

    with DatasetStore.open(
        first_root,
        expected_build_manifest_fingerprint=dataset.manifest.fingerprint,
        expected_source_manifest_fingerprint=dataset.source_manifest.fingerprint,
        expected_storage_fingerprint=first.fingerprint,
    ) as stored:
        assert len(stored) == 2
        assert stored.source_manifest == dataset.source_manifest
        assert tuple(entry.record_id for entry in stored.record_index) == tuple(
            record.record_id for record in dataset.records
        )
        assert stored.split_manifest_for(SplitMode.COMPOSITION) == dataset.split_for(
            SplitMode.COMPOSITION
        ).manifest
        assert "Crystallography Open Database" in stored.notice_text
        assert not stored._mapped
        assert stored.record_ids == tuple(record.record_id for record in dataset.records)
        first_record = stored[0]
        source = dataset.records[0]
        target = source.canonical_record.target
        assert first_record.record_id == source.record_id
        assert first_record.aliases == source.aliases
        assert first_record.alias_scope_evidence == source.alias_scope_evidence
        assert first_record.identity.polymorph_key == source.identity.polymorph_key
        assert first_record.target_type == "exact-fixture"
        assert first_record.space_group == target.cover.space_group
        np.testing.assert_array_equal(first_record.atomic_numbers, target.composition.support)
        np.testing.assert_array_equal(
            first_record.composition_coefficients, target.composition.coefficients
        )
        np.testing.assert_array_equal(first_record.metric, target.metric)
        np.testing.assert_array_equal(first_record.lattice_latent, target.lattice_latent)
        np.testing.assert_array_equal(
            first_record.intrinsic_coordinates, target.intrinsic_coordinates
        )
        np.testing.assert_array_equal(first_record.period_basis, target.period_basis)
        np.testing.assert_array_equal(
            first_record.nu_legal_mask, target.cover_decisions.nu_legal_mask
        )
        assert first_record.nu_target_index == target.cover_decisions.nu_target_index
        assert first_record.group_target_index == target.cover_decisions.group_target_index
        assert isinstance(first_record.metric, np.memmap)
        assert not first_record.metric.flags.writeable
        assert set(stored._mapped) == {0}
        assert stored[first_record.record_id].record_id == first_record.record_id
        assert stored[-1].record_id == dataset.records[-1].record_id
        for actual, expected in zip(
            first_record.row_decisions,
            target.cover_decisions.row_decisions,
            strict=True,
        ):
            assert actual.row_label == expected.row.label
            assert actual.selected_action_index == expected.selected_action_index
            np.testing.assert_array_equal(
                actual.selected_action,
                expected.selected_action.counts,
            )
            np.testing.assert_array_equal(
                actual.legal_actions,
                [action.counts for action in expected.legal_actions],
            )

    for path in first_root.rglob("*.npy"):
        array = np.load(path, mmap_mode="r", allow_pickle=False)
        assert not array.dtype.hasobject


def test_reader_fails_closed_on_hash_tamper_and_wrong_build(tmp_path: Path) -> None:
    dataset = _compiled_dataset(tmp_path)
    root = tmp_path / "stored"
    manifest = storage_module.write_compiled_dataset(
        dataset, root, records_per_shard=2
    )
    with pytest.raises(DomainError, match="externally pinned manifest"):
        DatasetStore.open(root, expected_storage_fingerprint="f" * 64)
    with pytest.raises(DomainError, match="different source receipts"):
        DatasetStore.open(root, expected_source_manifest_fingerprint="f" * 64)
    with DatasetStore.open(
        root,
        expected_storage_fingerprint=manifest.fingerprint,
    ):
        pass
    metric_path = root / "shard-00000" / "metric.npy"
    metric_path.write_bytes(metric_path.read_bytes() + b"tamper")

    with pytest.raises(DomainError, match="byte count changed"):
        DatasetStore.open(root)
    with pytest.raises(DomainError, match="different build manifest"):
        DatasetStore.open(root, expected_build_manifest_fingerprint="f" * 64)


@pytest.mark.parametrize(
    "relative_path",
    (
        "source-manifest.json",
        "record-index.json",
        "NOTICE.txt",
        "splits/composition.json",
    ),
)
def test_reader_authenticates_detached_receipts(
    tmp_path: Path, relative_path: str
) -> None:
    dataset = _compiled_dataset(tmp_path)
    root = tmp_path / "stored"
    storage_module.write_compiled_dataset(dataset, root, records_per_shard=2)
    path = root / relative_path
    path.write_bytes(path.read_bytes() + b"tamper")

    with pytest.raises(DomainError, match="(byte count|hash) changed"):
        DatasetStore.open(root)


def test_reader_rejects_duplicate_json_keys(tmp_path: Path) -> None:
    dataset = _compiled_dataset(tmp_path)
    root = tmp_path / "stored"
    storage_module.write_compiled_dataset(dataset, root, records_per_shard=2)
    path = root / "storage-manifest.json"
    original = path.read_text(encoding="ascii")
    duplicated = (
        '{"schema":"crystalatlas.dataset-storage.v3",'
        '"schema":"crystalatlas.dataset-storage.v3",'
        + original[1:]
    )
    path.write_text(duplicated, encoding="ascii")

    with pytest.raises(DomainError, match="cannot read dataset storage manifest"):
        DatasetStore.open(root)


def test_build_manifest_binds_per_alias_scope_evidence(tmp_path: Path) -> None:
    dataset = _compiled_dataset(tmp_path)
    root = tmp_path / "stored"
    manifest = storage_module.write_compiled_dataset(
        dataset, root, records_per_shard=2
    )
    metadata_path = root / "shard-00000" / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="ascii"))
    metadata["records"][0]["alias_scope_evidence"][0]["attribution"] = [
        "Forged attribution"
    ]
    metadata_path.write_bytes(storage_module._canonical_json_bytes(metadata))
    replacement = StoredFile.from_path(
        metadata_path,
        relative_path="shard-00000/metadata.json",
    )
    shard = manifest.shards[0]
    files = tuple(
        replacement if item.relative_path == replacement.relative_path else item
        for item in shard.files
    )
    updated_shard = replace(shard, files=tuple(sorted(files)))
    updated = replace(
        manifest,
        shards=(updated_shard,),
        storage_fingerprint=None,
    )
    (root / "storage-manifest.json").write_bytes(
        storage_module._canonical_json_bytes(updated.to_dict())
    )

    with pytest.raises(DomainError, match="metadata disagrees with the build manifest"):
        DatasetStore.open(root)


def test_reader_semantically_audits_notice_and_record_index(tmp_path: Path) -> None:
    dataset = _compiled_dataset(tmp_path)

    notice_root = tmp_path / "notice-store"
    notice_manifest = storage_module.write_compiled_dataset(
        dataset, notice_root, records_per_shard=2
    )
    notice_path = notice_root / "NOTICE.txt"
    notice_path.write_bytes(b"forged notice\n")
    changed_notice = StoredFile.from_path(notice_path, relative_path="NOTICE.txt")
    changed_notice_manifest = replace(
        notice_manifest,
        notice=changed_notice,
        storage_fingerprint=None,
    )
    (notice_root / "storage-manifest.json").write_bytes(
        storage_module._canonical_json_bytes(changed_notice_manifest.to_dict())
    )
    with pytest.raises(DomainError, match="notice disagrees with source obligations"):
        DatasetStore.open(notice_root)

    index_root = tmp_path / "index-store"
    index_manifest = storage_module.write_compiled_dataset(
        dataset, index_root, records_per_shard=2
    )
    index_path = index_root / "record-index.json"
    index_payload = json.loads(index_path.read_text(encoding="ascii"))
    index_payload["records"][0]["target_fingerprint"] = "f" * 64
    unsigned = {
        key: value
        for key, value in index_payload.items()
        if key != "index_fingerprint"
    }
    index_payload["index_fingerprint"] = storage_module._fingerprint(unsigned)
    index_path.write_bytes(storage_module._canonical_json_bytes(index_payload))
    changed_index = StoredFile.from_path(
        index_path,
        relative_path="record-index.json",
    )
    changed_index_manifest = replace(
        index_manifest,
        record_index=changed_index,
        storage_fingerprint=None,
    )
    (index_root / "storage-manifest.json").write_bytes(
        storage_module._canonical_json_bytes(changed_index_manifest.to_dict())
    )
    with pytest.raises(DomainError, match="record index disagrees with the build manifest"):
        DatasetStore.open(index_root)


def test_build_manifest_binds_split_receipt(tmp_path: Path) -> None:
    dataset = _compiled_dataset(tmp_path)
    root = tmp_path / "stored"
    manifest = storage_module.write_compiled_dataset(
        dataset, root, records_per_shard=2
    )
    split_path = root / "splits" / "composition.json"
    payload = json.loads(split_path.read_text(encoding="ascii"))
    payload["seed"] += 1
    allocation = {
        "schema": "crystalatlas.dataset-split-allocation.v1",
        "mode": payload["mode"],
        "seed": payload["seed"],
        "fractions": payload["fractions"],
        "require_nonempty": payload["require_nonempty"],
        "dataset_fingerprint": payload["dataset_fingerprint"],
        "record_assignments": payload["record_assignments"],
        "group_assignments": payload["group_assignments"],
    }
    payload["allocation_fingerprint"] = storage_module._fingerprint(allocation)
    unsigned = {
        key: value
        for key, value in payload.items()
        if key != "manifest_fingerprint"
    }
    payload["manifest_fingerprint"] = storage_module._fingerprint(unsigned)
    split_path.write_bytes(storage_module._canonical_json_bytes(payload))
    changed_file = StoredFile.from_path(
        split_path,
        relative_path="splits/composition.json",
    )
    changed_split = replace(
        manifest.splits[0],
        manifest_fingerprint=payload["manifest_fingerprint"],
        file=changed_file,
    )
    changed_manifest = replace(
        manifest,
        splits=(changed_split,),
        storage_fingerprint=None,
    )
    (root / "storage-manifest.json").write_bytes(
        storage_module._canonical_json_bytes(changed_manifest.to_dict())
    )

    with pytest.raises(DomainError, match="split receipts disagree with the build manifest"):
        DatasetStore.open(root)


def test_reader_rejects_unexpected_files_directories_and_links(tmp_path: Path) -> None:
    dataset = _compiled_dataset(tmp_path)

    extra_file_root = tmp_path / "extra-file-store"
    storage_module.write_compiled_dataset(dataset, extra_file_root, records_per_shard=2)
    (extra_file_root / "unlisted.bin").write_bytes(b"not authenticated")
    with pytest.raises(DomainError, match="unauthenticated entries"):
        DatasetStore.open(extra_file_root)

    extra_directory_root = tmp_path / "extra-directory-store"
    storage_module.write_compiled_dataset(
        dataset, extra_directory_root, records_per_shard=2
    )
    (extra_directory_root / "unlisted").mkdir()
    with pytest.raises(DomainError, match="unauthenticated entries"):
        DatasetStore.open(extra_directory_root)

    link_root = tmp_path / "link-store"
    storage_module.write_compiled_dataset(dataset, link_root, records_per_shard=2)
    (link_root / "linked-notice").symlink_to(link_root / "NOTICE.txt")
    with pytest.raises(DomainError, match="symbolic link"):
        DatasetStore.open(link_root)


def test_reader_rejects_authenticated_array_schema_mismatch(tmp_path: Path) -> None:
    dataset = _compiled_dataset(tmp_path)
    root = tmp_path / "stored"
    manifest = storage_module.write_compiled_dataset(dataset, root, records_per_shard=2)
    metric_path = root / "shard-00000" / "metric.npy"
    with metric_path.open("wb") as stream:
        np.save(stream, np.zeros((2, 3, 3), dtype=np.int64), allow_pickle=False)

    old_shard = manifest.shards[0]
    replacement = StoredFile.from_path(
        metric_path,
        relative_path="shard-00000/metric.npy",
    )
    new_files = tuple(
        replacement if item.relative_path == replacement.relative_path else item
        for item in old_shard.files
    )
    new_shard = replace(old_shard, files=tuple(sorted(new_files)))
    updated = replace(
        manifest,
        shards=(new_shard,),
        storage_fingerprint=None,
    )
    (root / "storage-manifest.json").write_bytes(
        storage_module._canonical_json_bytes(updated.to_dict())
    )

    with pytest.raises(DomainError, match="metric has an unsupported dtype"):
        DatasetStore.open(root)


def test_writer_failure_is_atomic_and_existing_destination_is_untouched(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dataset = _compiled_dataset(tmp_path)
    destination = tmp_path / "failed-store"

    def fail_write(_path: Path, _array: np.ndarray) -> None:
        raise RuntimeError("injected shard failure")

    monkeypatch.setattr(storage_module, "_write_npy", fail_write)
    with pytest.raises(RuntimeError, match="injected shard failure"):
        storage_module.write_compiled_dataset(dataset, destination, records_per_shard=1)
    assert not destination.exists()
    assert not tuple(tmp_path.glob(".failed-store.tmp-*"))

    existing = tmp_path / "existing"
    existing.mkdir()
    sentinel = existing / "user-data"
    sentinel.write_text("preserve", encoding="utf-8")
    with pytest.raises(DomainError, match="already exists"):
        storage_module.write_compiled_dataset(dataset, existing)
    assert sentinel.read_text(encoding="utf-8") == "preserve"
