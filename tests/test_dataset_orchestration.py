from __future__ import annotations

import json
from dataclasses import replace
from fractions import Fraction
from hashlib import sha256
from pathlib import Path

import pytest
from test_dataset_builder import (
    _canonical_record,
    _compiler,
    _FixtureTargetBuilder,
)

from crystalatlas.catalogue import MINIMAL_CATALOGUE
from crystalatlas.composition import CompositionDomain, CompositionGrammar
from crystalatlas.dataset_orchestration import (
    compile_local_cif_dataset,
    load_local_cif_candidates,
)
from crystalatlas.dataset_provenance import (
    ArtifactChecksum,
    ArtifactRole,
    Citation,
    DatasetSnapshot,
    DatasetSourceDeclaration,
    DatasetSourceManifest,
    DatasetSourceReceipt,
    LicenseDeclaration,
    Permission,
    RecordScopePolicy,
    SourceRecordScopeEvidence,
    canonical_record_index_bytes,
    canonical_record_payload_index_bytes,
    canonical_record_scope_index_bytes,
    ordered_record_ids_sha256,
    ordered_record_payloads_sha256,
    ordered_record_scopes_sha256,
)
from crystalatlas.dataset_splits import SplitFractions, SplitMode
from crystalatlas.dataset_storage import DatasetStore
from crystalatlas.errors import DomainError
from crystalatlas.ingestion import CanonicalTrainingRecord, RawPeriodicStructure
from crystalatlas.tokenization import CompositionTokenCodec
from crystalatlas.training_data import (
    DeterministicBucketBatchSampler,
    StoredTrainingDataset,
    TrainingMeasure,
)
from crystalatlas.training_targets import ExactFixtureTrainingTarget

_SYNTHETIC_CIF = """data_crystalatlas_synthetic_fixture
_cell_length_a 4
_cell_length_b 5
_cell_length_c 6
_cell_angle_alpha 90
_cell_angle_beta 90
_cell_angle_gamma 90
_space_group_name_Hall 'P 1'
loop_
_atom_site_label
_atom_site_type_symbol
_atom_site_fract_x
_atom_site_fract_y
_atom_site_fract_z
_atom_site_occupancy
C1 C 0 0 0 1
N1 N 0.14285714285714285 0.2222222222222222 0.2727272727272727 1
O1 O 0.4 0.375 0.3076923076923077 1
"""


def _scope(record_id: str) -> SourceRecordScopeEvidence:
    return SourceRecordScopeEvidence(
        policy=RecordScopePolicy.SYNTHETIC_INTERNAL,
        claims=(("origin", "synthetic"), ("third_party_content", "false")),
        source_metadata_sha256=sha256(f"synthetic-metadata:{record_id}".encode()).hexdigest(),
        evidence_uri=f"https://fixtures.invalid/crystalatlas/{record_id}",
        attribution=("CrystalAtlas synthetic test fixture",),
    )


def _source_manifest(
    tmp_path: Path, *, direct_cif_data_artifact: bool = True
) -> tuple[DatasetSourceManifest, dict[tuple[str, str], Path]]:
    root = tmp_path / "synthetic-source"
    root.mkdir()
    record_id = "synthetic-row-0001"
    scope = _scope(record_id)
    cif = root / "synthetic-row-0001.cif"
    cif.write_bytes(_SYNTHETIC_CIF.encode("utf-8"))
    terms = root / "synthetic-fixture-terms.txt"
    terms.write_text(
        "Internal synthetic test fixture; no third-party database content.\n",
        encoding="utf-8",
    )
    selection = root / "selection.json"
    selection.write_bytes(
        b'{"generator":"crystalatlas-test-fixture-v1","third_party_content":false}'
    )
    record_index = root / "record-index.json"
    record_index.write_bytes(canonical_record_index_bytes((record_id,)))
    payload_rows = ((record_id, sha256(_SYNTHETIC_CIF.encode("utf-8")).hexdigest()),)
    payload_index = root / "record-payload-index.json"
    payload_index.write_bytes(canonical_record_payload_index_bytes(payload_rows))
    scope_rows = ((record_id, scope),)
    scope_index = root / "record-scope-index.json"
    scope_index.write_bytes(canonical_record_scope_index_bytes(scope_rows))
    if direct_cif_data_artifact:
        data_artifact = ArtifactChecksum.from_file(cif, role=ArtifactRole.DATA)
    else:
        archive = root / "unindexed-export.tar"
        archive.write_bytes(b"synthetic archive placeholder")
        data_artifact = ArtifactChecksum.from_file(archive, role=ArtifactRole.DATA)

    terms_artifact = ArtifactChecksum.from_file(terms, role=ArtifactRole.TERMS)
    source = DatasetSourceDeclaration(
        source_id="synthetic-internal-fixture",
        name="CrystalAtlas internal synthetic fixture",
        provider="CrystalAtlas test suite",
        homepage_uri="https://fixtures.invalid/crystalatlas",
        data_scope=(
            "Coordinates generated solely for internal integration tests; "
            "no third-party database content"
        ),
        record_scope_policy=RecordScopePolicy.SYNTHETIC_INTERNAL,
        license=LicenseDeclaration(
            spdx_id="LicenseRef-CrystalAtlas-Synthetic-Fixture",
            name="CrystalAtlas synthetic fixture declaration",
            license_uri="https://fixtures.invalid/crystalatlas/license",
            terms_uri="https://fixtures.invalid/crystalatlas/terms",
            reviewed_on="2026-08-29",
            terms_evidence_sha256=terms_artifact.sha256,
            training_use=Permission.ALLOWED,
            raw_redistribution=Permission.PROHIBITED,
            derived_redistribution=Permission.PROHIBITED,
            commercial_use=Permission.PROHIBITED,
            attribution_required=True,
            notice="Internal synthetic test data only; not a third-party licence grant.",
        ),
        citations=(
            Citation(
                key="crystalatlas-synthetic-fixture-v1",
                title="CrystalAtlas internal synthetic integration fixture",
                uri="https://fixtures.invalid/crystalatlas/fixture-v1",
            ),
        ),
    )
    selection_artifact = ArtifactChecksum.from_file(selection, role=ArtifactRole.SELECTION_SPEC)
    snapshot = DatasetSnapshot(
        source_id=source.source_id,
        snapshot_id="synthetic-fixture-v1",
        retrieved_utc="2026-08-29T00:00:00Z",
        retrieval_uri="https://fixtures.invalid/crystalatlas/synthetic-fixture-v1",
        selection_spec_sha256=selection_artifact.sha256,
        ordered_record_ids_sha256=ordered_record_ids_sha256((record_id,)),
        ordered_record_payloads_sha256=ordered_record_payloads_sha256(payload_rows),
        ordered_record_scopes_sha256=ordered_record_scopes_sha256(scope_rows),
        record_count=1,
        artifacts=(
            data_artifact,
            ArtifactChecksum.from_file(payload_index, role=ArtifactRole.PAYLOAD_INDEX),
            ArtifactChecksum.from_file(record_index, role=ArtifactRole.RECORD_INDEX),
            ArtifactChecksum.from_file(scope_index, role=ArtifactRole.SCOPE_INDEX),
            selection_artifact,
            terms_artifact,
        ),
    )
    manifest = DatasetSourceManifest((DatasetSourceReceipt(source, snapshot),))
    return manifest, {(source.source_id, snapshot.snapshot_id): root}


def test_local_cif_entry_point_reaches_authenticated_training_batch(
    tmp_path: Path,
) -> None:
    manifest, roots = _source_manifest(tmp_path)
    template = _canonical_record(shift=Fraction(0), source_id="synthetic-template")

    def build_target(
        raw: RawPeriodicStructure,
    ) -> CanonicalTrainingRecord[ExactFixtureTrainingTarget]:
        receipt = replace(
            template.canonicalized.receipt,
            source_fingerprint=raw.fingerprint,
        )
        return replace(
            template,
            raw_structure=raw,
            canonicalized=replace(template.canonicalized, receipt=receipt),
        )

    builder = _FixtureTargetBuilder(build_target)
    compiler = _compiler(manifest, builder)
    first_destination = tmp_path / "stored-synthetic-a"
    second_destination = tmp_path / "stored-synthetic-b"
    crystallography_fingerprint = compiler.crystallography_certification.receipt_fingerprint
    split_fractions = SplitFractions.from_values("1", "0", "0")
    first = compile_local_cif_dataset(
        compiler,
        destination=first_destination,
        expected_source_manifest_fingerprint=manifest.fingerprint,
        expected_crystallography_receipt_fingerprint=crystallography_fingerprint,
        artifact_roots=roots,
        split_modes=(SplitMode.COMPOSITION,),
        split_fractions=split_fractions,
        records_per_shard=1,
    )
    second = compile_local_cif_dataset(
        compiler,
        destination=second_destination,
        expected_source_manifest_fingerprint=manifest.fingerprint,
        expected_crystallography_receipt_fingerprint=crystallography_fingerprint,
        artifact_roots=roots,
        split_modes=(SplitMode.COMPOSITION,),
        split_fractions=split_fractions,
        records_per_shard=1,
    )

    assert first == second
    receipt_payload = json.loads(first.to_json())
    assert receipt_payload["receipt_fingerprint"] == first.fingerprint
    assert (
        receipt_payload["local_input_manifest"]["manifest_fingerprint"]
        == first.local_input_manifest_fingerprint
    )
    assert type(first).from_json(first.to_json()) == first
    assert (
        type(first.local_input_manifest).from_json(first.local_input_manifest.to_json())
        == first.local_input_manifest
    )
    assert first.input_count == 1
    assert first.output_count == 1
    assert first.rejection_count == 0
    source_text = json.dumps(manifest.to_dict())
    assert "synthetic" in source_text.lower()
    assert "third-party licence grant" in source_text
    assert "Crystallography Open Database" not in source_text
    assert "Materials Project" not in source_text

    with DatasetStore.open(
        first_destination,
        expected_build_manifest_fingerprint=first.build_manifest_fingerprint,
        expected_source_manifest_fingerprint=first.source_manifest_fingerprint,
        expected_storage_fingerprint=first.storage_manifest_fingerprint,
    ) as store:
        codec = CompositionTokenCodec(CompositionGrammar(CompositionDomain((6, 7, 8), 3, 8)))
        dataset = StoredTrainingDataset(store, codec, MINIMAL_CATALOGUE)
        sampler = DeterministicBucketBatchSampler(
            dataset,
            1,
            seed=17,
            measure=TrainingMeasure.STRUCTURE,
        )
        assert next(sampler).record_ids == store.record_ids


def test_local_loader_requires_external_source_manifest_pin(tmp_path: Path) -> None:
    manifest, roots = _source_manifest(tmp_path)
    with pytest.raises(DomainError, match="external trust anchor"):
        load_local_cif_candidates(
            manifest,
            expected_source_manifest_fingerprint="0" * 64,
            artifact_roots=roots,
        )


def test_local_loader_rechecks_semantic_snapshot_index_digests(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest, roots = _source_manifest(tmp_path)
    receipt = manifest.receipts[0]
    root = roots[(receipt.source.source_id, receipt.snapshot.snapshot_id)]
    payload_digest = sha256(_SYNTHETIC_CIF.encode("utf-8")).hexdigest()
    replacement_id = "synthetic-row-swapped"
    replacement_scope = _scope(replacement_id)
    (root / "record-index.json").write_bytes(
        canonical_record_index_bytes((replacement_id,))
    )
    (root / "record-payload-index.json").write_bytes(
        canonical_record_payload_index_bytes(((replacement_id, payload_digest),))
    )
    (root / "record-scope-index.json").write_bytes(
        canonical_record_scope_index_bytes(((replacement_id, replacement_scope),))
    )
    # Simulate a swap after the snapshot's first whole-file verification.  The
    # loader must still bind the rows it actually parsed to the semantic
    # digests in the immutable receipt.
    monkeypatch.setattr(DatasetSnapshot, "require_files", lambda self, root: None)
    with pytest.raises(DomainError, match="record index digest disagrees"):
        load_local_cif_candidates(
            manifest,
            expected_source_manifest_fingerprint=manifest.fingerprint,
            artifact_roots=roots,
        )


def test_local_loader_rejects_unindexed_archive_instead_of_inferring_extraction(
    tmp_path: Path,
) -> None:
    manifest, roots = _source_manifest(tmp_path, direct_cif_data_artifact=False)
    with pytest.raises(DomainError, match="exactly one pinned DATA artifact"):
        load_local_cif_candidates(
            manifest,
            expected_source_manifest_fingerprint=manifest.fingerprint,
            artifact_roots=roots,
        )
