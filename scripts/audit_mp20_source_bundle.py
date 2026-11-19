"""Read-only audit of an unresolved MP20 source bundle against an external pin.

The pin SHA must come from a trusted location, not from the bundle under audit.
This checks archive integrity and permission status, not canonical targets.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from prepare_mp20_pilot import _canonical_bytes, _plain_path, _strict_json
from prepare_mp20_source_bundle import _pinned

from crystalatlas.dataset_orchestration import load_local_cif_candidates
from crystalatlas.dataset_provenance import (
    DatasetSourceManifest,
    DatasetUseRequest,
    Permission,
    RecordScopePolicy,
    evaluate_dataset_use,
)
from crystalatlas.errors import DomainError


def audit(bundle: Path, trust_pin: Path, expected_pin_sha256: str) -> dict:
    bundle = _plain_path(bundle, must_exist=True)
    trust_pin = _plain_path(trust_pin, must_exist=True)
    if bundle == trust_pin or bundle in trust_pin.parents:
        raise DomainError("trust pin must be outside the bundle")
    pin = _strict_json(_pinned(trust_pin, expected_pin_sha256, limit=16384))
    keys = {
        "schema",
        "source_manifest_fingerprint",
        "manifest_sha256",
        "input_manifest_fingerprint",
        "record_count",
        "artifact_count",
        "training_authorized",
        "permission",
        "assembler_sha256",
    }
    if not isinstance(pin, dict) or set(pin) != keys:
        raise DomainError("unexpected source-bundle pin fields")
    if (
        pin["schema"] != "crystalatlas.mp20-source-bundle-pin.v1"
        or pin["training_authorized"] is not False
        or pin["permission"] != "unresolved"
        or any(
            type(pin[k]) is not int or pin[k] <= 0 for k in ("record_count", "artifact_count")
        )
    ):
        raise DomainError("source-bundle pin does not describe an unresolved archive")
    for key in keys - {
        "schema",
        "record_count",
        "artifact_count",
        "training_authorized",
        "permission",
    }:
        value = pin[key]
        if (
            not isinstance(value, str)
            or len(value) != 64
            or any(c not in "0123456789abcdef" for c in value)
        ):
            raise DomainError("pin digest must be lowercase SHA-256")
    raw = _pinned(
        bundle / "source-manifest.json", pin["manifest_sha256"], limit=4 * 1024 * 1024
    )
    manifest = DatasetSourceManifest.from_json(raw.decode("utf-8"))
    if len(manifest.receipts) != 1:
        raise DomainError("expected one historical source receipt")
    receipt = manifest.receipts[0]
    if (
        receipt.source.record_scope_policy is not RecordScopePolicy.CDVAE_MP20_HISTORICAL
        or receipt.source.license.training_use is not Permission.UNRESOLVED
        or len(receipt.snapshot.artifacts) != pin["artifact_count"]
    ):
        raise DomainError("source declaration or artifact count disagrees with pin")
    loaded = load_local_cif_candidates(
        manifest,
        expected_source_manifest_fingerprint=pin["source_manifest_fingerprint"],
        artifact_roots={(receipt.source.source_id, receipt.snapshot.snapshot_id): bundle},
    )
    if (
        len(loaded.candidates) != pin["record_count"]
        or loaded.manifest.fingerprint != pin["input_manifest_fingerprint"]
    ):
        raise DomainError("loaded records disagree with external pin")
    if evaluate_dataset_use(manifest, DatasetUseRequest(train_model=True)).authorized:
        raise DomainError("unresolved inspection bundle unexpectedly authorizes training")
    return {
        "schema": "crystalatlas.mp20-source-bundle-audit.v1",
        "external_pin_sha256": expected_pin_sha256,
        "source_manifest_fingerprint": manifest.fingerprint,
        "input_manifest_fingerprint": loaded.manifest.fingerprint,
        "record_count": len(loaded.candidates),
        "artifact_count": len(receipt.snapshot.artifacts),
        "training_authorized": False,
        "canonical_targets_checked": False,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--trust-pin", type=Path, required=True)
    parser.add_argument("--expected-pin-sha256", required=True)
    args = parser.parse_args()
    print(
        _canonical_bytes(audit(args.bundle, args.trust_pin, args.expected_pin_sha256)).decode()
    )


if __name__ == "__main__":
    main()
