"""Assemble a pinned offline MP20 source bundle without authorizing training.

The source manifest is written last, after archive validation and an external
trust-pin write. Existing or partial destinations are never reused or deleted.
Original input files remain untouched. No network or GPU is used.
"""

from __future__ import annotations

import argparse
from hashlib import sha256
from pathlib import Path

from prepare_mp20_pilot import (
    FROZEN_FILES,
    REVISION,
    _canonical_bytes,
    _plain_path,
    _strict_json,
    _verify_acquisition,
    _write_exclusive,
)

from crystalatlas.benchmark_data import read_benchmark_cif_csv, select_training_pilot
from crystalatlas.dataset_orchestration import load_local_cif_candidates
from crystalatlas.dataset_provenance import (
    ArtifactChecksum,
    ArtifactRole,
    DatasetSnapshot,
    DatasetSourceManifest,
    DatasetSourceReceipt,
    DatasetUseRequest,
    canonical_record_scope_index_bytes,
    evaluate_dataset_use,
    historical_cdvae_mp20_scope,
    historical_cdvae_mp20_source,
    ordered_record_ids_sha256,
    ordered_record_payloads_sha256,
    ordered_record_scopes_sha256,
)
from crystalatlas.errors import DomainError
from crystalatlas.historical_mp20 import _read_bound

ACQUISITION_SHA = "3efeb934ea6fdc4800242c54b05c1211d89e432f92553533e1a6563632fbd095"
EXTRACTION_SHA = "512585859b2def771cb7cf31cd1df72884ba0fe17f225b8f371a3c6857932894"
EVIDENCE = {
    "SOURCE_REVIEW.md": "db9a1cc15d6dcc47d98ef6b83b48ae9ec1b3b4bd30973c645f063937adc586ed",
    "mp-provider-clarification-45437.json": (
        "67da73530a0444b2135dcc8aaae869349644a4567da45411cfbbcbb82acbd4b6"
    ),
    "CC-BY-4.0-legalcode.txt": (
        "9ba9550ad48438d0836ddab3da480b3b69ffa0aac7b7878b5a0039e7ab429411"
    ),
}


def _pinned(path: Path, digest: str, limit: int = 64 * 1024 * 1024) -> bytes:
    path = _plain_path(path, must_exist=True)
    artifact = ArtifactChecksum(ArtifactRole.AUXILIARY, path.name, digest, path.stat().st_size)
    return _read_bound(path.parent, artifact, limit)


def assemble(source: Path, pilot: Path, evidence: Path, destination: Path, trust_pin: Path):
    source, pilot, evidence = (
        _plain_path(path, must_exist=True) for path in (source, pilot, evidence)
    )
    destination = _plain_path(destination, must_exist=False)
    trust_pin = _plain_path(trust_pin, must_exist=False)
    if destination.exists() or trust_pin.exists():
        raise DomainError("bundle destination and external pin must both be new")
    if trust_pin == destination or destination in trust_pin.parents:
        raise DomainError("trust pin must be stored outside the source bundle")
    if any(trust_pin == p or p in trust_pin.parents for p in (source, pilot, evidence)):
        raise DomainError("trust pin must not modify an input tree")
    if any(
        destination == p or destination in p.parents or p in destination.parents
        for p in (source, pilot, evidence)
    ):
        raise DomainError("bundle output must not overlap any input tree")
    _verify_acquisition(source, ACQUISITION_SHA)
    acquisition = _strict_json(_pinned(source / "acquisition.json", ACQUISITION_SHA))
    extraction = _strict_json(_pinned(pilot / "extraction-receipt.json", EXTRACTION_SHA))
    if (
        extraction["selected_count"] != 512
        or extraction["seed"] != 20260913
        or extraction["source_acquisition_sha256"] != ACQUISITION_SHA
    ):
        raise DomainError("extraction does not describe the frozen pilot")
    rows = select_training_pilot(
        read_benchmark_cif_csv(
            source / "train.csv",
            split="train",
            expected_sha256=FROZEN_FILES["train.csv"][2],
            expected_count=27136,
        ),
        count=512,
        seed=20260913,
    )
    scopes = tuple(
        (
            row.material_id,
            historical_cdvae_mp20_scope(source_metadata_sha256=row.metadata_sha256),
        )
        for row in rows
    )
    scope_bytes = canonical_record_scope_index_bytes(scopes)
    declaration = historical_cdvae_mp20_source(reviewed_on="2026-09-14")
    # One linear copy plan. No directory traversal, implicit glob or copied symlink.
    plan: list[tuple[Path, ArtifactChecksum]] = []

    def plan_file(path: Path, name: str, digest: str, role: ArtifactRole):
        raw = _pinned(path, digest)
        plan.append((path, ArtifactChecksum(role, name, digest, len(raw))))

    roles = {
        "selected_cif": ArtifactRole.DATA,
        "record_index": ArtifactRole.RECORD_INDEX,
        "payload_index": ArtifactRole.PAYLOAD_INDEX,
        "selection_spec": ArtifactRole.SELECTION_SPEC,
    }
    for item in extraction["files"]:
        # ArtifactChecksum enforces normalized, relative, traversal-free paths.
        check = ArtifactChecksum(
            roles.get(item["role"], ArtifactRole.AUXILIARY),
            item["path"],
            item["sha256"],
            item["byte_count"],
        )
        plan_file(pilot / check.relative_path, check.relative_path, check.sha256, check.role)
        if plan[-1][1].byte_count != check.byte_count:
            raise DomainError("extracted artifact byte count differs from receipt")
    for name, (_, _, digest) in FROZEN_FILES.items():
        output_name = "upstream-README.md" if name == "README.md" else name
        plan_file(source / name, output_name, digest, ArtifactRole.AUXILIARY)
    plan_file(
        source / "acquisition.json", "acquisition.json", ACQUISITION_SHA, ArtifactRole.AUXILIARY
    )
    plan_file(
        pilot / "extraction-receipt.json",
        "extraction-receipt.json",
        EXTRACTION_SHA,
        ArtifactRole.AUXILIARY,
    )
    for name, digest in EVIDENCE.items():
        role = ArtifactRole.TERMS if name.startswith("mp-provider-") else ArtifactRole.AUXILIARY
        plan_file(evidence / name, name, digest, role)
    scope_artifact = ArtifactChecksum(
        ArtifactRole.SCOPE_INDEX,
        "record-scope-index.json",
        sha256(scope_bytes).hexdigest(),
        len(scope_bytes),
    )
    artifacts = tuple(artifact for _, artifact in plan) + (scope_artifact,)
    selection = next(a for a in artifacts if a.role is ArtifactRole.SELECTION_SPEC)
    snapshot = DatasetSnapshot(
        source_id=declaration.source_id,
        snapshot_id="cdvae-f857f598-train512-seed20260913",
        retrieved_utc=acquisition["retrieved_utc"],
        retrieval_uri=f"https://github.com/txie-93/cdvae/tree/{REVISION}/data/mp_20",
        selection_spec_sha256=selection.sha256,
        ordered_record_ids_sha256=ordered_record_ids_sha256(tuple(r.material_id for r in rows)),
        ordered_record_payloads_sha256=ordered_record_payloads_sha256(
            tuple((r.material_id, r.cif_sha256) for r in rows)
        ),
        ordered_record_scopes_sha256=ordered_record_scopes_sha256(scopes),
        record_count=len(rows),
        artifacts=artifacts,
    )
    manifest = DatasetSourceManifest((DatasetSourceReceipt(declaration, snapshot),))
    destination.parent.mkdir(parents=True, exist_ok=True)
    _plain_path(destination.parent, must_exist=True)
    destination.mkdir(exist_ok=False)
    for path, artifact in plan:
        raw = _pinned(path, artifact.sha256)
        output = destination / artifact.relative_path
        output.parent.mkdir(parents=True, exist_ok=True)
        _plain_path(output.parent, must_exist=True)
        _write_exclusive(output, raw)
    _write_exclusive(destination / scope_artifact.relative_path, scope_bytes)
    loaded = load_local_cif_candidates(
        manifest,
        expected_source_manifest_fingerprint=manifest.fingerprint,
        artifact_roots={(declaration.source_id, snapshot.snapshot_id): destination},
    )
    if tuple(c.reference.source_record_id for c in loaded.candidates) != tuple(
        r.material_id for r in rows
    ):
        raise DomainError("assembled candidate order changed")
    decision = evaluate_dataset_use(manifest, DatasetUseRequest(train_model=True))
    if decision.authorized:
        raise DomainError("inspection bundle must not authorize training")
    result = {
        "schema": "crystalatlas.mp20-source-bundle-pin.v1",
        "source_manifest_fingerprint": manifest.fingerprint,
        "manifest_sha256": sha256(manifest.to_json().encode()).hexdigest(),
        "input_manifest_fingerprint": loaded.manifest.fingerprint,
        "record_count": len(loaded.candidates),
        "artifact_count": len(artifacts),
        "training_authorized": False,
        "permission": "unresolved",
        "assembler_sha256": sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    trust_pin.parent.mkdir(parents=True, exist_ok=True)
    _plain_path(trust_pin.parent, must_exist=True)
    _write_exclusive(trust_pin, _canonical_bytes(result) + b"\n")
    # Sole completion marker. Failed/partial directories remain available for inspection.
    _write_exclusive(destination / "source-manifest.json", manifest.to_json().encode())
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("source", "pilot", "evidence", "destination", "trust-pin"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    result = assemble(args.source, args.pilot, args.evidence, args.destination, args.trust_pin)
    print(_canonical_bytes(result).decode())


if __name__ == "__main__":
    main()
