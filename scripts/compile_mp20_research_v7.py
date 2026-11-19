"""Compile the frozen 512-row MP20 pilot under the deployed v7 preprocessing policy.

Same reviewed source declaration, bundle, catalogue certification, split modes and
accounting as ``compile_mp20_research_pilot.py`` (the frozen v6 contract), but
with the policy the B1/B2 remediation deployed and the S2 study selected on
2026-10-20: dimensioned symmetry tolerance tau = 1e-4 A for classification and
idealization, scale-relative ranking quantum 1e-9, quantization-equivalent orbit
merging, the primitive atom bound (20 primitive / 80 conventional) and the
default ``loewner-v2`` chart.  At tau = 1e-4 the S2 arm accepted 431 of 512
records against 193 under v6.

This CPU-only runner grants no data-use permission, changes no tolerance beyond
the declared v7 values, retains every input in accepted/duplicate/rejected
accounting, and never reuses or removes a destination.  It is a research
compile: the builder fingerprint is recorded, not pinned.
"""

from __future__ import annotations

import argparse
import platform
import sys
import time
from collections import Counter
from dataclasses import asdict
from hashlib import sha256
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from compile_mp20_research_pilot import (  # noqa: E402
    _reviewed_manifest,
    _source_hashes,
    _validate_paths,
)
from prepare_mp20_pilot import (  # noqa: E402
    _canonical_bytes,
    _plain_path,
    _strict_json,
    _write_exclusive,
)
from prepare_mp20_source_bundle import _pinned  # noqa: E402

from crystalatlas.certification import CrystallographyCertification  # noqa: E402
from crystalatlas.config import DomainConfig  # noqa: E402
from crystalatlas.dataset_builder import (  # noqa: E402
    CatalogueDatasetTargetBuilder,
    DatasetCompiler,
)
from crystalatlas.dataset_identity import DeduplicationLevel  # noqa: E402
from crystalatlas.dataset_orchestration import compile_local_cif_dataset  # noqa: E402
from crystalatlas.dataset_provenance import DatasetUseRequest  # noqa: E402
from crystalatlas.dataset_splits import SplitMode  # noqa: E402
from crystalatlas.dataset_storage import DatasetStore  # noqa: E402
from crystalatlas.errors import DomainError  # noqa: E402
from crystalatlas.full_catalogue import build_full_catalogue, pinned_hall_numbers  # noqa: E402
from crystalatlas.ingestion import CatalogueTargetAssigner  # noqa: E402
from crystalatlas.standardization import SpglibPolicy, SpglibStandardizer  # noqa: E402

V7_TAU = 1.0e-4
V7_RELATIVE_METRIC_QUANTUM = 1.0e-9
V7_MAX_PRIMITIVE_ATOMS = 20


def compile_v7(
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
    tau: float = V7_TAU,
    relative_metric_quantum: float = V7_RELATIVE_METRIC_QUANTUM,
    max_primitive_atoms: int = V7_MAX_PRIMITIVE_ATOMS,
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
    print("Binding all-230 catalogue and the v7 target builder", flush=True)
    catalogue = build_full_catalogue()
    proof.require_for(catalogue)
    policy = SpglibPolicy(
        symprec=tau,
        idealization_tolerance_angstrom=tau,
        relative_metric_quantum=relative_metric_quantum,
    )
    domain = DomainConfig(
        max_primitive_atoms=max_primitive_atoms, max_raw_atoms=4 * max_primitive_atoms
    )
    builder = CatalogueDatasetTargetBuilder(
        SpglibStandardizer(pinned_hall_numbers(), policy),
        CatalogueTargetAssigner(catalogue, merge_quantization_equivalent_orbits=True),
        domain,
    )
    print(f"target builder fingerprint {builder.builder_fingerprint}", flush=True)
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
        "schema": "crystalatlas.mp20-research-v7-compilation.v1",
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
        "merge_quantization_equivalent_orbits": True,
        "target_builder_fingerprint": builder.builder_fingerprint,
        "split_seed": split_seed,
        "splits": splits,
        "rejections": [
            {"stage": stage, "exception": exception, "reason": reason, "count": count}
            for (stage, exception, reason), count in sorted(rejection_counts.items())
        ],
        "deduplicated_alias_count": built.accepted_pre_dedup_count - built.output_count,
        "total_seconds": time.perf_counter() - started,
        "scope": "controlled research corpus under the v7 policy; no learning or "
        "generation-quality claim",
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
    parser.add_argument("--tau", type=float, default=V7_TAU)
    parser.add_argument(
        "--relative-metric-quantum", type=float, default=V7_RELATIVE_METRIC_QUANTUM
    )
    parser.add_argument("--max-primitive-atoms", type=int, default=V7_MAX_PRIMITIVE_ATOMS)
    report = compile_v7(**vars(parser.parse_args()))
    print(
        _canonical_bytes(
            {
                "receipt_sha256": sha256(_canonical_bytes(report) + b"\n").hexdigest(),
                "input_count": report["build_receipt"]["input_count"],
                "output_count": report["build_receipt"]["output_count"],
                "rejection_count": report["build_receipt"]["rejection_count"],
                "total_seconds": round(report["total_seconds"], 1),
            }
        ).decode("utf-8"),
        flush=True,
    )


if __name__ == "__main__":
    main()
