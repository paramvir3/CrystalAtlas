from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import fields, replace
from fractions import Fraction
from hashlib import sha256
from pathlib import Path
from typing import cast

import numpy as np
import pytest

from crystalatlas.catalogue import MINIMAL_CATALOGUE
from crystalatlas.certification import certify_crystallography
from crystalatlas.dataset_builder import (
    CanonicalRecord,
    CompiledDataset,
    DatasetBuildManifest,
    DatasetCandidate,
    DatasetCompiler,
    DatasetRejection,
    RejectionStage,
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
from crystalatlas.dataset_splits import SplitFractions, SplitMode, audit_dataset_split
from crystalatlas.decoder import CrystalScaffold, ExactCrystalDecoder, OrbitAssignment
from crystalatlas.errors import DomainError, InvariantViolation
from crystalatlas.ingestion import (
    CanonicalizationReceipt,
    CanonicalizedStructure,
    CanonicalTrainingRecord,
    RawPeriodicStructure,
)
from crystalatlas.training_targets import ExactFixtureTrainingTarget, lift_exact_fixture_target


class _FixtureTargetBuilder:
    certified_preprocessing_only = True

    def __init__(
        self,
        function: Callable[
            [RawPeriodicStructure], CanonicalTrainingRecord[ExactFixtureTrainingTarget]
        ],
        *,
        builder_fingerprint: str = "f" * 64,
    ) -> None:
        self.function = function
        self.builder_fingerprint = builder_fingerprint

    def __call__(
        self, structure: RawPeriodicStructure
    ) -> CanonicalRecord:
        return cast(CanonicalRecord, self.function(structure))


def _canonical_record(
    *, shift: Fraction, source_id: str
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
    metric = np.diag([16.0, 25.0, 36.0])
    target = lift_exact_fixture_target(scaffold, metric, catalogue=MINIMAL_CATALOGUE)
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
    canonicalized = CanonicalizedStructure(raw, raw, target.canonical_scaffold, receipt)
    return CanonicalTrainingRecord(raw, canonicalized, target)


def _cod_scope(record_id: str) -> SourceRecordScopeEvidence:
    return SourceRecordScopeEvidence(
        policy=RecordScopePolicy.COD_MAIN_CC0,
        claims=(("collection", "cod-main"), ("record_license", "CC0-1.0")),
        source_metadata_sha256=sha256(record_id.encode("utf-8")).hexdigest(),
        evidence_uri=f"https://example.org/cod/metadata/{record_id}",
        attribution=("Crystallography Open Database",),
    )


def _manifest(
    tmp_path: Path, record_payloads: tuple[tuple[str, str], ...]
) -> tuple[DatasetSourceManifest, dict[tuple[str, str], Path]]:
    record_ids = tuple(record_id for record_id, _ in record_payloads)
    root = tmp_path / "cod"
    root.mkdir(parents=True)
    artifact = root / "selected-cifs.tar.gz"
    artifact.write_bytes(b"pinned selected fixture")
    terms = root / "terms.html"
    terms.write_bytes(b"archived COD CC0 terms")
    selection = root / "selection.json"
    selection.write_bytes(b'{"query":"fixture-release-1"}')
    record_index = root / "record-index.json"
    record_index.write_bytes(canonical_record_index_bytes(record_ids))
    payload_index = root / "record-payload-index.json"
    payload_index.write_bytes(canonical_record_payload_index_bytes(record_payloads))
    scope_rows = tuple((record_id, _cod_scope(record_id)) for record_id in record_ids)
    scope_index = root / "record-scope-index.json"
    scope_index.write_bytes(canonical_record_scope_index_bytes(scope_rows))
    terms_artifact = ArtifactChecksum.from_file(terms, role=ArtifactRole.TERMS)
    selection_artifact = ArtifactChecksum.from_file(
        selection, role=ArtifactRole.SELECTION_SPEC
    )
    source = crystallography_open_database_source(
        terms_evidence_sha256=terms_artifact.sha256,
        reviewed_on="2026-08-25",
    )
    snapshot = DatasetSnapshot(
        source_id=source.source_id,
        snapshot_id="fixture-release-1",
        retrieved_utc="2026-08-25T12:00:00Z",
        retrieval_uri="https://example.org/cod/fixture-release-1",
        selection_spec_sha256=selection_artifact.sha256,
        ordered_record_ids_sha256=ordered_record_ids_sha256(record_ids),
        ordered_record_payloads_sha256=ordered_record_payloads_sha256(record_payloads),
        ordered_record_scopes_sha256=ordered_record_scopes_sha256(scope_rows),
        record_count=len(record_ids),
        artifacts=(
            ArtifactChecksum.from_file(artifact, role=ArtifactRole.DATA),
            ArtifactChecksum.from_file(payload_index, role=ArtifactRole.PAYLOAD_INDEX),
            ArtifactChecksum.from_file(record_index, role=ArtifactRole.RECORD_INDEX),
            selection_artifact,
            ArtifactChecksum.from_file(scope_index, role=ArtifactRole.SCOPE_INDEX),
            terms_artifact,
        ),
    )
    return (
        DatasetSourceManifest((DatasetSourceReceipt(source, snapshot),)),
        {(source.source_id, snapshot.snapshot_id): root},
    )


def _compiler(
    manifest: DatasetSourceManifest,
    target_builder: _FixtureTargetBuilder,
    *,
    deduplication_level: DeduplicationLevel = DeduplicationLevel.POLYMORPH,
) -> DatasetCompiler:
    certification = certify_crystallography(
        MINIMAL_CATALOGUE,
        require_all_230=False,
        crosscheck_external_operations=False,
    )
    return DatasetCompiler(
        catalogue=MINIMAL_CATALOGUE,
        crystallography_certification=certification,
        source_manifest=manifest,
        use_request=DatasetUseRequest(train_model=True),
        target_builder=target_builder,
        identity_policy=IdentityPolicy(require_all_230=False),
        deduplication_level=deduplication_level,
        require_all_230=False,
        require_external_operation_database=False,
    )


def _bind_template(
    template: CanonicalTrainingRecord[ExactFixtureTrainingTarget],
    raw: RawPeriodicStructure,
) -> CanonicalTrainingRecord[ExactFixtureTrainingTarget]:
    receipt = replace(
        template.canonicalized.receipt,
        source_fingerprint=raw.fingerprint,
    )
    canonicalized = replace(
        template.canonicalized,
        conventional_structure=raw,
        primitive_structure=raw,
        receipt=receipt,
    )
    return replace(template, raw_structure=raw, canonicalized=canonicalized)


def _compile_single(
    tmp_path: Path,
    template: CanonicalTrainingRecord[ExactFixtureTrainingTarget],
    *,
    deduplication_level: DeduplicationLevel = DeduplicationLevel.POLYMORPH,
    builder_fingerprint: str = "f" * 64,
) -> CompiledDataset:
    candidate = DatasetCandidate(
        SourceRecordReference("cod", "fixture-release-1", "only"),
        template.raw_structure,
        _cod_scope("only"),
    )
    manifest, roots = _manifest(tmp_path, (("only", candidate.payload_sha256),))
    builder = _FixtureTargetBuilder(
        lambda raw: _bind_template(template, raw),
        builder_fingerprint=builder_fingerprint,
    )
    return _compiler(
        manifest,
        builder,
        deduplication_level=deduplication_level,
    ).compile((candidate,), artifact_roots=roots, split_modes=())


def test_compiler_deduplicates_aliases_preserves_polymorphs_and_builds_all_splits(
    tmp_path: Path,
) -> None:
    first = _canonical_record(shift=Fraction(0), source_id="template-a")
    second = _canonical_record(shift=Fraction(1, 17), source_id="template-b")
    templates = {
        first.raw_structure.fingerprint: first,
        second.raw_structure.fingerprint: second,
    }

    def target_builder(
        raw: RawPeriodicStructure,
    ) -> CanonicalTrainingRecord[ExactFixtureTrainingTarget]:
        template = templates[raw.fingerprint]
        receipt = replace(
            template.canonicalized.receipt,
            source_fingerprint=raw.fingerprint,
        )
        return replace(
            template,
            raw_structure=raw,
            canonicalized=replace(template.canonicalized, receipt=receipt),
        )

    record_ids = ("duplicate-a", "duplicate-b", "polymorph-b", "invalid")
    references = tuple(
        SourceRecordReference("cod", "fixture-release-1", record_id) for record_id in record_ids
    )
    candidates = (
        DatasetCandidate(references[0], first.raw_structure, _cod_scope(record_ids[0])),
        DatasetCandidate(references[1], first.raw_structure, _cod_scope(record_ids[1])),
        DatasetCandidate(references[2], second.raw_structure, _cod_scope(record_ids[2])),
        DatasetCandidate(
            references[3], "this is not a CIF data block", _cod_scope(record_ids[3])
        ),
    )
    manifest, roots = _manifest(
        tmp_path,
        tuple(
            (candidate.reference.source_record_id, candidate.payload_sha256)
            for candidate in candidates
        ),
    )
    certification = certify_crystallography(
        MINIMAL_CATALOGUE,
        require_all_230=False,
        crosscheck_external_operations=False,
    )
    compiler = DatasetCompiler(
        catalogue=MINIMAL_CATALOGUE,
        crystallography_certification=certification,
        source_manifest=manifest,
        use_request=DatasetUseRequest(train_model=True),
        target_builder=_FixtureTargetBuilder(target_builder),
        identity_policy=IdentityPolicy(require_all_230=False),
        deduplication_level=DeduplicationLevel.POLYMORPH,
        require_all_230=False,
        require_external_operation_database=False,
    )
    dataset = compiler.compile(
        candidates,
        artifact_roots=roots,
        split_modes=tuple(SplitMode),
        split_fractions=SplitFractions.from_values("1/2", "1/4", "1/4"),
        require_nonempty_splits=False,
    )

    assert dataset.manifest.input_count == 4
    assert dataset.manifest.accepted_pre_dedup_count == 3
    assert dataset.manifest.output_count == 2
    assert dataset.manifest.rejection_count == 1
    assert sorted(len(record.aliases) for record in dataset.records) == [1, 2]
    assert len({record.identity.composition_key for record in dataset.records}) == 1
    assert len({record.identity.polymorph_key for record in dataset.records}) == 2
    assert dataset.rejections[0].stage.value == "strict-parse"
    assert len(dataset.training_records) == 2
    split_records = tuple(record.split_record() for record in dataset.records)
    for _mode, split in dataset.splits:
        audit_dataset_split(split_records, split.manifest)
        assert set(split.train_ids + split.validation_ids + split.test_ids) == {
            record.record_id for record in dataset.records
        }


def test_compiler_default_dedup_preserves_volume_distinct_states() -> None:
    defaults = {item.name: item.default for item in fields(DatasetCompiler)}
    assert defaults["deduplication_level"] is DeduplicationLevel.STRUCTURE


def test_compiler_rejects_snapshot_subset_or_reordering_before_science(
    tmp_path: Path,
) -> None:
    raw = _canonical_record(shift=Fraction(0), source_id="raw").raw_structure
    manifest, roots = _manifest(
        tmp_path,
        (("a", raw.fingerprint), ("b", raw.fingerprint)),
    )
    certification = certify_crystallography(
        MINIMAL_CATALOGUE,
        require_all_230=False,
        crosscheck_external_operations=False,
    )

    def impossible_builder(
        raw: RawPeriodicStructure,
    ) -> CanonicalTrainingRecord[ExactFixtureTrainingTarget]:
        raise AssertionError("snapshot validation must precede target construction")

    compiler = DatasetCompiler(
        MINIMAL_CATALOGUE,
        certification,
        manifest,
        DatasetUseRequest(),
        _FixtureTargetBuilder(impossible_builder),
        IdentityPolicy(require_all_230=False),
        DeduplicationLevel.POLYMORPH,
        False,
        False,
    )
    reversed_candidates = (
        DatasetCandidate(
            SourceRecordReference("cod", "fixture-release-1", "b"), raw, _cod_scope("b")
        ),
        DatasetCandidate(
            SourceRecordReference("cod", "fixture-release-1", "a"), raw, _cod_scope("a")
        ),
    )
    with pytest.raises(DomainError, match="ordered record index hash"):
        compiler.compile(reversed_candidates, artifact_roots=roots, split_modes=())


def test_payload_mutation_fails_before_target_builder_is_called(tmp_path: Path) -> None:
    original = _canonical_record(shift=Fraction(0), source_id="original").raw_structure
    mutated_positions = original.fractional_positions.copy()
    mutated_positions[1, 0] += 1.0 / 37.0
    mutated = RawPeriodicStructure.from_arrays(
        original.lattice,
        mutated_positions,
        original.atomic_numbers,
        source_id="mutated",
    )
    manifest, roots = _manifest(tmp_path, (("only", original.fingerprint),))
    calls = 0

    def forbidden_builder(
        raw: RawPeriodicStructure,
    ) -> CanonicalTrainingRecord[ExactFixtureTrainingTarget]:
        nonlocal calls
        calls += 1
        raise AssertionError("payload validation must precede target construction")

    candidate = DatasetCandidate(
        SourceRecordReference("cod", "fixture-release-1", "only"),
        mutated,
        _cod_scope("only"),
    )
    with pytest.raises(DomainError, match="ordered record payload hash"):
        _compiler(manifest, _FixtureTargetBuilder(forbidden_builder)).compile(
            (candidate,), artifact_roots=roots, split_modes=()
        )
    assert calls == 0


def test_generic_domain_error_from_custom_builder_aborts_compilation(
    tmp_path: Path,
) -> None:
    raw = _canonical_record(shift=Fraction(0), source_id="raw").raw_structure
    candidate = DatasetCandidate(
        SourceRecordReference("cod", "fixture-release-1", "only"),
        raw,
        _cod_scope("only"),
    )
    manifest, roots = _manifest(tmp_path, (("only", candidate.payload_sha256),))

    def defective_builder(
        raw: RawPeriodicStructure,
    ) -> CanonicalTrainingRecord[ExactFixtureTrainingTarget]:
        raise DomainError("generic builder defect sentinel")

    with pytest.raises(DomainError, match="generic builder defect sentinel"):
        _compiler(manifest, _FixtureTargetBuilder(defective_builder)).compile(
            (candidate,), artifact_roots=roots, split_modes=()
        )


def test_target_builder_cannot_return_a_record_for_another_raw_structure(
    tmp_path: Path,
) -> None:
    supplied = _canonical_record(shift=Fraction(0), source_id="supplied")
    unrelated = _canonical_record(shift=Fraction(1, 17), source_id="unrelated")
    candidate = DatasetCandidate(
        SourceRecordReference("cod", "fixture-release-1", "only"),
        supplied.raw_structure,
        _cod_scope("only"),
    )
    manifest, roots = _manifest(tmp_path, (("only", candidate.payload_sha256),))

    with pytest.raises(InvariantViolation, match="different raw structure"):
        _compiler(
            manifest,
            _FixtureTargetBuilder(lambda _raw: unrelated),
        ).compile((candidate,), artifact_roots=roots, split_modes=())


def test_build_fingerprint_binds_exact_target_and_builder_configuration(
    tmp_path: Path,
) -> None:
    template = _canonical_record(shift=Fraction(0), source_id="template")
    candidate = DatasetCandidate(
        SourceRecordReference("cod", "fixture-release-1", "only"),
        template.raw_structure,
        _cod_scope("only"),
    )
    manifest, roots = _manifest(tmp_path, (("only", candidate.payload_sha256),))

    def compile_with(
        selected: CanonicalTrainingRecord[ExactFixtureTrainingTarget],
        builder_fingerprint: str,
    ) -> CompiledDataset:
        builder = _FixtureTargetBuilder(
            lambda raw: _bind_template(selected, raw),
            builder_fingerprint=builder_fingerprint,
        )
        return _compiler(manifest, builder).compile(
            (candidate,), artifact_roots=roots, split_modes=()
        )

    baseline = compile_with(template, "a" * 64)
    changed_builder = compile_with(template, "b" * 64)
    changed_latent = template.target.lattice_latent.copy()
    changed_latent[0] += 0.125
    changed_target = replace(
        template,
        target=replace(template.target, lattice_latent=changed_latent),
    )
    changed_scientific_target = compile_with(changed_target, "a" * 64)

    assert baseline.records[0].identity.exact_key == (
        changed_scientific_target.records[0].identity.exact_key
    )
    assert baseline.manifest.records[0].target_fingerprint != (
        changed_scientific_target.manifest.records[0].target_fingerprint
    )
    assert baseline.manifest.fingerprint != changed_builder.manifest.fingerprint
    assert baseline.manifest.fingerprint != changed_scientific_target.manifest.fingerprint


def test_exact_deduplication_binds_record_id_to_exact_identity(tmp_path: Path) -> None:
    dataset = _compile_single(
        tmp_path,
        _canonical_record(shift=Fraction(0), source_id="template"),
        deduplication_level=DeduplicationLevel.EXACT,
    )
    record = dataset.records[0]
    summary = dataset.manifest.records[0]
    assert record.record_id == record.identity.exact_key == summary.exact_key
    assert summary.exact_key != summary.structure_key

    wrong_summary = replace(summary, record_id=summary.structure_key)
    with pytest.raises(DomainError, match="selected duplicate quotient"):
        replace(
            dataset.manifest,
            records=(wrong_summary,),
            manifest_fingerprint=None,
        )


def test_build_manifest_roundtrip_tamper_detection_and_external_anchor(
    tmp_path: Path,
) -> None:
    dataset = _compile_single(
        tmp_path / "compiled",
        _canonical_record(shift=Fraction(0), source_id="template"),
    )
    path = tmp_path / "build-manifest.json"
    dataset.manifest.write_json(path)
    assert DatasetBuildManifest.read_json(path) == dataset.manifest
    assert (
        DatasetBuildManifest.read_json(
            path,
            expected_fingerprint=dataset.manifest.fingerprint,
        )
        == dataset.manifest
    )

    nested_payload = cast(dict[str, object], json.loads(path.read_text(encoding="utf-8")))
    nested_records = cast(list[dict[str, object]], nested_payload["records"])
    nested_records[0]["target_fingerprint"] = "0" * 64
    nested_path = tmp_path / "nested-tamper.json"
    nested_path.write_text(json.dumps(nested_payload), encoding="utf-8")
    with pytest.raises(DomainError, match="internally inconsistent"):
        DatasetBuildManifest.read_json(nested_path)

    recomputed_payload = cast(
        dict[str, object], json.loads(path.read_text(encoding="utf-8"))
    )
    recomputed_payload["target_builder_fingerprint"] = "1" * 64
    unsigned = dict(recomputed_payload)
    del unsigned["manifest_fingerprint"]
    recomputed_payload["manifest_fingerprint"] = sha256(
        json.dumps(
            unsigned,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("ascii")
    ).hexdigest()
    recomputed_path = tmp_path / "self-consistent-tamper.json"
    recomputed_path.write_text(json.dumps(recomputed_payload), encoding="utf-8")
    assert DatasetBuildManifest.read_json(recomputed_path).fingerprint == (
        recomputed_payload["manifest_fingerprint"]
    )
    with pytest.raises(DomainError, match="externally expected fingerprint"):
        DatasetBuildManifest.read_json(
            recomputed_path,
            expected_fingerprint=dataset.manifest.fingerprint,
        )

    duplicate_key_path = tmp_path / "duplicate-key.json"
    duplicate_key_path.write_text(
        '{"schema":"ignored",' + path.read_text(encoding="utf-8").lstrip()[1:],
        encoding="utf-8",
    )
    with pytest.raises(DomainError, match="repeats JSON key 'schema'"):
        DatasetBuildManifest.read_json(duplicate_key_path)

    nonfinite_path = tmp_path / "nonfinite.json"
    nonfinite_path.write_text('{"schema":NaN}\n', encoding="utf-8")
    with pytest.raises(DomainError, match="nonfinite JSON number"):
        DatasetBuildManifest.read_json(nonfinite_path)


def test_build_manifest_rejects_duplicate_aliases_and_partition_violations(
    tmp_path: Path,
) -> None:
    dataset = _compile_single(
        tmp_path,
        _canonical_record(shift=Fraction(0), source_id="template"),
    )
    manifest = dataset.manifest
    summary = manifest.records[0]
    alias = summary.aliases[0]
    scope = summary.alias_scope_evidence[0]

    with pytest.raises(DomainError, match="aliases must be nonempty and unique"):
        replace(
            summary,
            aliases=(alias, alias),
            alias_scope_evidence=(scope, scope),
        )

    duplicate_rejection = DatasetRejection(
        reference=alias,
        payload_sha256=summary.representative_payload_sha256,
        stage=RejectionStage.PARSE,
        exception_type="DomainError",
        reason="fixture rejection",
    )
    with pytest.raises(DomainError, match="appears more than once"):
        replace(
            manifest,
            input_count=2,
            rejection_count=1,
            rejections=(duplicate_rejection,),
            manifest_fingerprint=None,
        )

    with pytest.raises(DomainError, match="aliases do not cover every accepted"):
        replace(
            manifest,
            input_count=2,
            accepted_pre_dedup_count=2,
            manifest_fingerprint=None,
        )
