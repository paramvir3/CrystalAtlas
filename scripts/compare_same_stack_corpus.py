"""Fail-closed comparison of the frozen MP20 corpus after a numerical-stack rebuild.

Only lattice-latent floating values and their downstream target/build/storage
digests may differ. This comparison does not certify the latent differences as
physically acceptable: target reconstruction and decoder audits on the training
stack remain mandatory. No input is replaced and no subset is newly selected.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping
from dataclasses import fields, is_dataclass
from hashlib import sha256
from pathlib import Path

import numpy as np
from compile_mp20_research_pilot import EXPECTED_BUILDER, FROZEN_SNAPSHOT
from prepare_mp20_pilot import _canonical_bytes, _plain_path, _strict_json, _write_exclusive
from prepare_mp20_source_bundle import _pinned

from crystalatlas.dataset_identity import DeduplicationLevel
from crystalatlas.dataset_orchestration import LocalDatasetBuildReceipt
from crystalatlas.dataset_splits import SplitMode
from crystalatlas.dataset_storage import DatasetStore
from crystalatlas.errors import DomainError

_COMPILATION_FIELDS = {
    "schema", "build_receipt", "source_manifest_sha256",
    "crystallography_receipt_sha256", "runner_sha256", "package_sources",
    "source_tree_unchanged", "python", "numpy", "spglib", "domain",
    "standardizer_policy", "split_seed", "splits", "rejections",
    "deduplicated_alias_count", "total_seconds", "scope",
}
_ALLOWED_RECEIPT_DIFFERENCES = {
    "build_manifest_fingerprint", "storage_manifest_fingerprint", "receipt_fingerprint",
}


def _exact(old, new, path):
    """Compare all fields, including float bits, without a numerical tolerance."""
    if type(old) is not type(new):
        raise DomainError(f"unexpected change at {path}: type")
    if isinstance(old, np.ndarray):
        if old.dtype != new.dtype or old.shape != new.shape or old.tobytes() != new.tobytes():
            raise DomainError(f"unexpected change at {path}: array")
    elif is_dataclass(old) and not isinstance(old, type):
        for field in fields(old):
            _exact(getattr(old, field.name), getattr(new, field.name), f"{path}.{field.name}")
    elif isinstance(old, Mapping):
        if old.keys() != new.keys():
            raise DomainError(f"unexpected change at {path}: fields")
        for key in old:
            _exact(old[key], new[key], f"{path}.{key}")
    elif isinstance(old, (tuple, list)):
        if len(old) != len(new):
            raise DomainError(f"unexpected change at {path}: length")
        for index, (left, right) in enumerate(zip(old, new, strict=True)):
            _exact(left, right, f"{path}[{index}]")
    elif isinstance(old, float):
        if not np.isfinite(old) or not np.isfinite(new) or old.hex() != new.hex():
            raise DomainError(f"unexpected change at {path}: float")
    elif old != new:
        raise DomainError(f"unexpected change at {path}: value")


def _read_compilation(path, digest):
    value = _strict_json(_pinned(path, digest, limit=16 * 1024 * 1024))
    if not isinstance(value, dict) or set(value) != _COMPILATION_FIELDS:
        raise DomainError("compilation receipt has missing or unexpected fields")
    if value["schema"] != "crystalatlas.mp20-research-pilot-compilation.v1":
        raise DomainError("unsupported compilation receipt schema")
    if value["source_tree_unchanged"] is not True:
        raise DomainError("compilation did not certify unchanged package source")
    receipt = LocalDatasetBuildReceipt.from_dict(value["build_receipt"])
    return value, receipt


def _compare_compilations(old, new):
    for name in _COMPILATION_FIELDS - {"build_receipt", "python", "total_seconds"}:
        _exact(old[name], new[name], f"compilation.{name}")
    for name in old["build_receipt"]:
        if name not in _ALLOWED_RECEIPT_DIFFERENCES:
            _exact(old["build_receipt"][name], new["build_receipt"][name], f"receipt.{name}")


def _open_store(path, receipt):
    return DatasetStore.open(
        path,
        expected_source_manifest_fingerprint=receipt.source_manifest_fingerprint,
        expected_build_manifest_fingerprint=receipt.build_manifest_fingerprint,
        expected_storage_fingerprint=receipt.storage_manifest_fingerprint,
    )


def _bind_store(store, receipt):
    build = store.build_manifest
    for name in (
        "source_manifest_fingerprint", "crystallography_receipt_fingerprint",
        "target_builder_fingerprint", "input_count", "accepted_pre_dedup_count",
        "output_count", "rejection_count", "split_manifest_fingerprints",
    ):
        _exact(getattr(build, name), getattr(receipt, name), f"store_receipt.{name}")
    _exact(build.policy_decision.fingerprint, receipt.policy_decision_fingerprint, "policy")
    _exact(store.manifest.records_per_shard, receipt.records_per_shard, "records_per_shard")
    if set(store.split_manifests) != set(SplitMode):
        raise DomainError("all four frozen split views are required")
    if build.deduplication_level is not DeduplicationLevel.STRUCTURE:
        raise DomainError("frozen corpus requires STRUCTURE deduplication")
    if not build.use_request.train_model:
        raise DomainError("corpus was not compiled for authorized training")
    build.policy_decision.require_authorized()


def _compare_records(old, new):
    """All stored semantics are exact except the explicit latent and digest fields."""
    for field in fields(old):
        if field.name not in {"lattice_latent", "target_fingerprint"}:
            _exact(getattr(old, field.name), getattr(new, field.name),
                   f"record[{old.record_id}].{field.name}")
    left, right = old.lattice_latent, new.lattice_latent
    if left.dtype != right.dtype or left.shape != right.shape:
        raise DomainError(f"lattice latent layout changed for {old.record_id}")
    if not np.all(np.isfinite(left)) or not np.all(np.isfinite(right)):
        raise DomainError("lattice latent values must be finite")
    with np.errstate(over="ignore", invalid="ignore"):
        difference = np.abs(left - right)
    if not np.all(np.isfinite(difference)):
        raise DomainError("lattice latent difference is not finite")
    return {
        "record_id": old.record_id,
        "bitwise_changed": left.tobytes() != right.tobytes(),
        "changed_value_count": int(np.count_nonzero(left != right)),
        "max_absolute_difference": float(np.max(difference, initial=0.0)),
        "old_target_fingerprint": old.target_fingerprint,
        "new_target_fingerprint": new.target_fingerprint,
    }


def _compare_stores(old, new):
    _exact(old.record_ids, new.record_ids, "accepted_record_ids")
    _exact(old.source_manifest.to_dict(), new.source_manifest.to_dict(), "source_manifest")
    left, right = old.build_manifest.to_dict(), new.build_manifest.to_dict()
    for value in (left, right):
        value.pop("manifest_fingerprint")
        for record in value["records"]:
            record.pop("target_fingerprint")
    _exact(left, right, "build_manifest")
    for mode in SplitMode:
        _exact(old.split_manifest_for(mode).to_dict(), new.split_manifest_for(mode).to_dict(),
               f"split.{mode.value}")
    return [_compare_records(old[record_id], new[record_id]) for record_id in old.record_ids]


def _check_frozen_corpus(store, receipt):
    if receipt.input_count != 512 or len(receipt.local_input_manifest.bindings) != 512:
        raise DomainError("frozen source denominator must remain 512")
    if receipt.target_builder_fingerprint != EXPECTED_BUILDER:
        raise DomainError("frozen target builder changed")
    if len(store.source_manifest.receipts) != 1:
        raise DomainError("frozen corpus requires one source snapshot")
    snapshot = store.source_manifest.receipts[0].snapshot
    for name, expected in FROZEN_SNAPSHOT.items():
        _exact(getattr(snapshot, name), expected, f"frozen_snapshot.{name}")


def _check_selection(experiment, old, new, old_receipt):
    if not isinstance(experiment, dict) or experiment.get("schema") != (
        "crystalatlas.bounded-research-pilot.v1"
    ):
        raise DomainError("invalid frozen experiment")
    contract = experiment.get("data_contract")
    settings = experiment.get("settings")
    if not isinstance(contract, dict) or not isinstance(settings, dict):
        raise DomainError("frozen experiment needs data contract and settings")
    for name in ("source_manifest_fingerprint", "build_manifest_fingerprint",
                 "storage_manifest_fingerprint", "crystallography_receipt_fingerprint",
                 "input_count"):
        _exact(contract.get(name), getattr(old_receipt, name), f"experiment.{name}")
    _exact(contract.get("preprocessing_rejection_count"), old_receipt.rejection_count,
           "experiment.preprocessing_rejection_count")
    _exact(contract.get("split_mode"), "composition", "experiment.split_mode")
    _exact(contract.get("partition"), "train", "experiment.partition")
    _exact(settings.get("records"), 8, "experiment.records")
    _exact(settings.get("seed"), 20261012, "experiment.seed")
    _exact(contract.get("selection_rule"),
           "sha256-rank-v1; no geometry/energy/model-result filtering", "selection_rule")
    selected = contract.get("record_ids")
    ranked = contract.get("selection_rank_order")
    if (not isinstance(selected, list) or len(selected) != 8
            or not all(isinstance(item, str) for item in selected)
            or len(set(selected)) != 8 or selected != sorted(selected)
            or not isinstance(ranked, list) or len(ranked) != 8
            or not all(isinstance(item, str) for item in ranked)
            or sorted(ranked) != selected):
        raise DomainError("experiment does not freeze eight unique IDs and rank order")
    split = old.split_manifest_for(SplitMode.COMPOSITION)
    _exact(contract.get("split_manifest_fingerprint"), split.manifest_fingerprint,
           "experiment.split_manifest_fingerprint")
    for store in (old, new):
        train = tuple(record_id for record_id, partition in
                      store.split_manifest_for(SplitMode.COMPOSITION).record_assignments
                      if partition == "train")
        if not set(selected).issubset(train):
            raise DomainError("a frozen selected record moved out of the train partition")

        def rank(record_id):
            payload = ["crystalatlas.research-pilot-subset.v1", settings["seed"], record_id]
            raw = (json.dumps(payload, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()
            return sha256(raw).hexdigest(), record_id

        # Verify the old recorded decision; never substitute a freshly chosen subset.
        _exact(list(sorted(train, key=rank)[:8]), ranked, "frozen_selection_rank_order")
    return selected, ranked


def compare_corpora(*, old_store, new_store, old_receipt, new_receipt,
                    old_receipt_sha256, new_receipt_sha256, experiment,
                    experiment_sha256, output):
    inputs = tuple(_plain_path(path, must_exist=True) for path in
                   (old_store, new_store, old_receipt, new_receipt, experiment))
    old_store, new_store, old_receipt, new_receipt, experiment = inputs
    output = _plain_path(output, must_exist=False)
    if output.exists():
        raise DomainError("comparison output must be new")
    if any(output == path or output in path.parents or path in output.parents for path in inputs):
        raise DomainError("comparison output must not overlap an input")
    old_report, old_build = _read_compilation(old_receipt, old_receipt_sha256)
    new_report, new_build = _read_compilation(new_receipt, new_receipt_sha256)
    frozen = _strict_json(_pinned(experiment, experiment_sha256, limit=1024 * 1024))
    _compare_compilations(old_report, new_report)
    with _open_store(old_store, old_build) as old, _open_store(new_store, new_build) as new:
        for store, receipt in ((old, old_build), (new, new_build)):
            _bind_store(store, receipt)
            _check_frozen_corpus(store, receipt)
        differences = _compare_stores(old, new)
        selected, ranked = _check_selection(frozen, old, new, old_build)
    report = {
        "schema": "crystalatlas.same-stack-corpus-comparison.v1",
        "corpus_comparison_passed": True,
        "training_stack_target_decoder_audit_required": True,
        "latent_numerical_equivalence_certified": False,
        "scope": "corpus and frozen-subset preservation only; not a training or physical-quality pass",
        "old_receipt_sha256": old_receipt_sha256,
        "new_receipt_sha256": new_receipt_sha256,
        "frozen_experiment_sha256": experiment_sha256,
        "comparator_sha256": sha256(Path(__file__).read_bytes()).hexdigest(),
        "old_build_receipt": old_build.to_dict(),
        "new_build_receipt": new_build.to_dict(),
        "old_python": old_report["python"], "new_python": new_report["python"],
        "input_count": old_build.input_count, "record_count": old_build.output_count,
        "rejection_count": old_build.rejection_count,
        "frozen_record_ids": selected, "frozen_selection_rank_order": ranked,
        "all_four_split_manifests_unchanged": True,
        "lattice_latent_changed_records": sum(item["bitwise_changed"] for item in differences),
        "max_lattice_latent_absolute_difference": max(
            (item["max_absolute_difference"] for item in differences), default=0.0),
        "record_differences": differences,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    _plain_path(output.parent, must_exist=True)
    _write_exclusive(output, _canonical_bytes(report) + b"\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("old-store", "new-store", "old-receipt", "new-receipt", "experiment", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    for name in ("old-receipt-sha256", "new-receipt-sha256", "experiment-sha256"):
        parser.add_argument(f"--{name}", required=True)
    report = compare_corpora(**vars(parser.parse_args()))
    print(_canonical_bytes({key: report[key] for key in (
        "corpus_comparison_passed", "record_count", "lattice_latent_changed_records",
        "max_lattice_latent_absolute_difference", "training_stack_target_decoder_audit_required",
    )}).decode())


if __name__ == "__main__":
    main()
