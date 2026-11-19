"""Compile the frozen 512-row MP20 pilot using a separately reviewed declaration.

This CPU-only runner grants no data-use permission and changes no scientific
tolerance. It retains every selected input in accepted/duplicate/rejected
accounting. A fresh external receipt is published only after authenticated
store readback. A failed/partial destination is never reused or removed.
"""

from __future__ import annotations

import argparse
import platform
import time
from collections import Counter
from dataclasses import asdict
from hashlib import sha256
from pathlib import Path

from prepare_mp20_pilot import _canonical_bytes, _plain_path, _strict_json, _write_exclusive
from prepare_mp20_source_bundle import _pinned

import crystalatlas
from crystalatlas.certification import CrystallographyCertification
from crystalatlas.config import DomainConfig
from crystalatlas.dataset_builder import CatalogueDatasetTargetBuilder, DatasetCompiler
from crystalatlas.dataset_identity import DeduplicationLevel
from crystalatlas.dataset_orchestration import compile_local_cif_dataset
from crystalatlas.dataset_provenance import (
    DatasetSourceManifest,
    DatasetUseRequest,
    RecordScopePolicy,
    evaluate_dataset_use,
)
from crystalatlas.dataset_splits import SplitMode
from crystalatlas.dataset_storage import DatasetStore
from crystalatlas.errors import DomainError
from crystalatlas.full_catalogue import build_full_catalogue, pinned_hall_numbers
from crystalatlas.ingestion import CatalogueTargetAssigner
from crystalatlas.standardization import SpglibPolicy, SpglibStandardizer

EXPECTED_BUILDER = "a6e24ded30b9d4a69f5ca07213e23359c49d3908290914bfb2abd9240855bf37"
FROZEN_SNAPSHOT = {
    "record_count": 512,
    "source_id": "cdvae-mp20-historical",
    "snapshot_id": "cdvae-f857f598-train512-seed20260913",
    "selection_spec_sha256": (
        "840091b9ebd2a1145474f87b2c644e1d3d054fe83171e4e342fd81dea6b73f54"
    ),
    "ordered_record_ids_sha256": (
        "2eb79dc6ddf532cb28d3591a60b909f104dfac1117c8231cf49e3e3e0e06d1f5"
    ),
    "ordered_record_payloads_sha256": (
        "6153746c8a3883cbdd1cf1e68c14eab2552967e080688a293d911e728fafc1da"
    ),
    "ordered_record_scopes_sha256": (
        "42e0c09370b59131dac576c05793da3a8c3b8ea8741bc0ba9be88c63b30572ff"
    ),
}


def _source_hashes() -> dict[str, str]:
    root = Path(crystalatlas.__file__).resolve().parent
    return {
        path.relative_to(root).as_posix(): sha256(path.read_bytes()).hexdigest()
        for path in sorted(root.rglob("*.py"))
    }


def _validate_paths(bundle, manifest, certification, destination, receipt):
    inputs = tuple(
        _plain_path(path, must_exist=True) for path in (bundle, manifest, certification)
    )
    outputs = tuple(_plain_path(path, must_exist=False) for path in (destination, receipt))
    if not inputs[0].is_dir():
        raise DomainError("bundle must be a directory")
    if any(path.exists() for path in outputs):
        raise DomainError("destination and external receipt must both be new")
    output, external = outputs
    if output == external or output in external.parents or external in output.parents:
        raise DomainError("external receipt must not overlap the output store")
    for path in outputs:
        if any(
            path == source or path in source.parents or source in path.parents
            for source in inputs
        ):
            raise DomainError("outputs must not overlap any input")
    return (*inputs, *outputs)


def _reviewed_manifest(path, expected_sha256, expected_fingerprint):
    manifest = DatasetSourceManifest.from_json(
        _pinned(path, expected_sha256, limit=4 * 1024 * 1024).decode("utf-8")
    )
    if manifest.fingerprint != expected_fingerprint:
        raise DomainError("reviewed source manifest disagrees with external fingerprint")
    if len(manifest.receipts) != 1:
        raise DomainError("pilot requires exactly one historical source snapshot")
    receipt = manifest.receipts[0]
    if receipt.source.record_scope_policy is not RecordScopePolicy.CDVAE_MP20_HISTORICAL:
        raise DomainError("pilot requires the historical MP20 scope policy")
    if any(getattr(receipt.snapshot, key) != value for key, value in FROZEN_SNAPSHOT.items()):
        raise DomainError("reviewed manifest changed the frozen 512-row selection")
    evaluate_dataset_use(manifest, DatasetUseRequest(train_model=True)).require_authorized()
    return manifest


def compile_pilot(
    *,
    bundle: Path,
    source_manifest: Path,
    expected_source_sha256: str,
    expected_source_fingerprint: str,
    certification: Path,
    expected_certification_sha256: str,
    expected_certification_fingerprint: str,
    destination: Path,
    receipt: Path,
    split_seed: int = 20261012,
):
    if type(split_seed) is not int or split_seed < 0:
        raise DomainError("split seed must be a nonnegative integer")
    bundle, source_manifest, certification, destination, receipt = _validate_paths(
        bundle, source_manifest, certification, destination, receipt
    )
    manifest = _reviewed_manifest(
        source_manifest, expected_source_sha256, expected_source_fingerprint
    )
    proof = CrystallographyCertification.from_dict(
        _strict_json(_pinned(certification, expected_certification_sha256, limit=1024 * 1024))
    )
    if proof.receipt_fingerprint != expected_certification_fingerprint:
        raise DomainError("crystallographic proof disagrees with external fingerprint")
    import numpy as np
    import spglib

    before = _source_hashes()
    started = time.perf_counter()
    print("Binding all-230 catalogue and frozen v6 target builder", flush=True)
    catalogue = build_full_catalogue()
    proof.require_for(catalogue)
    builder = CatalogueDatasetTargetBuilder(
        SpglibStandardizer(pinned_hall_numbers(), SpglibPolicy()),
        CatalogueTargetAssigner(catalogue),
        DomainConfig(lattice_chart="ball-v1"),  # frozen v6 contract
    )
    if builder.builder_fingerprint != EXPECTED_BUILDER:
        raise DomainError("target builder differs from the frozen v6 preflight contract")
    compiler = DatasetCompiler(
        catalogue=catalogue,
        crystallography_certification=proof,
        source_manifest=manifest,
        use_request=DatasetUseRequest(train_model=True),
        target_builder=builder,
        deduplication_level=DeduplicationLevel.STRUCTURE,
    )
    source = manifest.receipts[0]
    print("Compiling all 512 inputs; preprocessing rejects will not be replaced", flush=True)
    built = compile_local_cif_dataset(
        compiler,
        expected_source_manifest_fingerprint=expected_source_fingerprint,
        expected_crystallography_receipt_fingerprint=expected_certification_fingerprint,
        artifact_roots={(source.source.source_id, source.snapshot.snapshot_id): bundle},
        destination=destination,
        split_modes=tuple(SplitMode),
        split_seed=split_seed,
        require_nonempty_splits=True,
        records_per_shard=64,
    )
    if built.input_count != 512:
        raise DomainError("compiler lost the frozen input denominator")
    with DatasetStore.open(
        destination,
        expected_build_manifest_fingerprint=built.build_manifest_fingerprint,
        expected_source_manifest_fingerprint=expected_source_fingerprint,
        expected_storage_fingerprint=built.storage_manifest_fingerprint,
    ) as store:
        splits = {
            mode.value: {
                "fingerprint": store.split_manifest_for(mode).manifest_fingerprint,
                "record_counts": dict(store.split_manifest_for(mode).record_counts),
                "group_counts": dict(store.split_manifest_for(mode).group_counts),
            }
            for mode in SplitMode
        }
        rejection_counts = Counter(
            (item.stage.value, item.exception_type, item.reason)
            for item in store.build_manifest.rejections
        )
    if _source_hashes() != before:
        raise DomainError("package source changed during compilation; result is not certified")
    report = {
        "schema": "crystalatlas.mp20-research-pilot-compilation.v1",
        "build_receipt": built.to_dict(),
        "source_manifest_sha256": expected_source_sha256,
        "crystallography_receipt_sha256": expected_certification_sha256,
        "runner_sha256": sha256(Path(__file__).read_bytes()).hexdigest(),
        "package_sources": before,
        "source_tree_unchanged": True,
        "python": platform.python_version(),
        "numpy": np.__version__,
        "spglib": spglib.__version__,
        "domain": asdict(builder.domain),
        "standardizer_policy": asdict(builder.standardizer.policy),
        "split_seed": split_seed,
        "splits": splits,
        "rejections": [
            {"stage": stage, "exception": exception, "reason": reason, "count": count}
            for (stage, exception, reason), count in sorted(rejection_counts.items())
        ],
        "deduplicated_alias_count": built.accepted_pre_dedup_count - built.output_count,
        "total_seconds": time.perf_counter() - started,
        "scope": "controlled research corpus; no learning or generation-quality claim",
    }
    receipt.parent.mkdir(parents=True, exist_ok=True)
    _plain_path(receipt.parent, must_exist=True)
    _write_exclusive(receipt, _canonical_bytes(report) + b"\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("bundle", "source-manifest", "certification", "destination", "receipt"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    for name in (
        "source-sha256",
        "source-fingerprint",
        "certification-sha256",
        "certification-fingerprint",
    ):
        parser.add_argument(f"--expected-{name}", required=True)
    parser.add_argument("--split-seed", type=int, default=20261012)
    report = compile_pilot(**vars(parser.parse_args()))
    print(
        _canonical_bytes(
            {
                "receipt_sha256": sha256(_canonical_bytes(report) + b"\n").hexdigest(),
                "input_count": report["build_receipt"]["input_count"],
                "output_count": report["build_receipt"]["output_count"],
                "rejection_count": report["build_receipt"]["rejection_count"],
            }
        ).decode()
    )


if __name__ == "__main__":
    main()
