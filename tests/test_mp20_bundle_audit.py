"""Synthetic end-to-end archive audits; no real-data permission is granted."""

import importlib
import json
from dataclasses import replace
from hashlib import sha256
from pathlib import Path

import pytest
from test_historical_mp20 import _archive

from crystalatlas.dataset_orchestration import load_local_cif_candidates
from crystalatlas.dataset_provenance import (
    DatasetSourceManifest,
    DatasetSourceReceipt,
    Permission,
)
from crystalatlas.errors import DomainError


@pytest.fixture
def archive(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    auditor = importlib.import_module("audit_mp20_source_bundle")
    manifest, roots, _ = _archive(tmp_path, monkeypatch)
    receipt = manifest.receipts[0]
    source = replace(
        receipt.source,
        license=replace(receipt.source.license, training_use=Permission.UNRESOLVED),
    )
    manifest = DatasetSourceManifest((DatasetSourceReceipt(source, receipt.snapshot),))
    root = next(iter(roots.values()))
    loaded = load_local_cif_candidates(
        manifest,
        expected_source_manifest_fingerprint=manifest.fingerprint,
        artifact_roots=roots,
    )
    raw = manifest.to_json().encode()
    (root / "source-manifest.json").write_bytes(raw)
    pin = {
        "schema": "crystalatlas.mp20-source-bundle-pin.v1",
        "source_manifest_fingerprint": manifest.fingerprint,
        "manifest_sha256": sha256(raw).hexdigest(),
        "input_manifest_fingerprint": loaded.manifest.fingerprint,
        "record_count": 1,
        "artifact_count": len(receipt.snapshot.artifacts),
        "training_authorized": False,
        "permission": "unresolved",
        "assembler_sha256": "a" * 64,
    }
    return auditor, root, tmp_path / "pin.json", pin


def run(archive):
    auditor, root, path, pin = archive
    raw = json.dumps(pin).encode()
    path.write_bytes(raw)
    return auditor.audit(root, path, sha256(raw).hexdigest())


def test_valid_archive(archive):
    result = run(archive)
    assert result["record_count"] == 1
    assert result["training_authorized"] is False
    assert result["canonical_targets_checked"] is False


@pytest.mark.parametrize(
    "key,value",
    [
        ("record_count", True),
        ("record_count", 2),
        ("artifact_count", 1),
        ("training_authorized", 0),
        ("permission", "allowed"),
        ("assembler_sha256", "A" * 64),
        ("input_manifest_fingerprint", "0" * 64),
        ("source_manifest_fingerprint", "0" * 64),
        ("manifest_sha256", "0" * 64),
        ("extra", "unexpected"),
    ],
)
def test_altered_pin_rejected_even_with_matching_pin_hash(archive, key, value):
    archive[3][key] = value
    with pytest.raises(DomainError):
        run(archive)


def test_external_pin_hash_mismatch(archive):
    run(archive)
    auditor, root, path, _ = archive
    with pytest.raises(DomainError):
        auditor.audit(root, path, "0" * 64)


def test_missing_completion_marker(archive):
    (archive[1] / "source-manifest.json").rename(archive[1] / "incomplete.json")
    with pytest.raises((DomainError, FileNotFoundError)):
        run(archive)


def test_changed_archived_bytes(archive):
    (archive[1] / "train.csv").write_bytes(b"altered archived bytes")
    with pytest.raises(DomainError):
        run(archive)
