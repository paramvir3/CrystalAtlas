"""Pilot launcher boundary tests; no cached real data or permission grant needed."""

import importlib
import subprocess
import sys
from dataclasses import replace
from hashlib import sha256
from pathlib import Path

import pytest
from test_dataset_orchestration import _source_manifest

from crystalatlas.dataset_provenance import (
    ArtifactRole,
    DatasetSourceManifest,
    DatasetSourceReceipt,
    historical_cdvae_mp20_source,
)
from crystalatlas.errors import DomainError


@pytest.fixture
def runner(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    return importlib.import_module("compile_mp20_research_pilot")


@pytest.fixture
def paths(tmp_path):
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    manifest = tmp_path / "manifest.json"
    manifest.write_text("{}")
    certification = tmp_path / "certification.json"
    certification.write_text("{}")
    return bundle, manifest, certification, tmp_path / "store", tmp_path / "receipt.json"


def test_new_nonoverlapping_paths_allowed(runner, paths):
    assert runner._validate_paths(*paths) == paths
    assert not paths[3].exists() and not paths[4].exists()


@pytest.mark.parametrize("index", [3, 4])
def test_existing_output_fails_before_work(runner, paths, index):
    paths[index].write_text("preserve")
    with pytest.raises(DomainError, match="must both be new"):
        runner._validate_paths(*paths)
    assert paths[index].read_text() == "preserve"


@pytest.mark.parametrize("index", [3, 4])
def test_outputs_do_not_modify_bundle(runner, paths, index):
    changed = list(paths)
    changed[index] = paths[0] / "new"
    with pytest.raises(DomainError, match="overlap any input"):
        runner._validate_paths(*changed)


def test_external_receipt_outside_store(runner, paths):
    with pytest.raises(DomainError, match="external receipt"):
        runner._validate_paths(*paths[:4], paths[3] / "receipt.json")


def test_input_symlink_rejected(runner, paths, tmp_path):
    link = tmp_path / "linked-bundle"
    link.symlink_to(paths[0])
    with pytest.raises(DomainError, match="symlinks"):
        runner._validate_paths(link, *paths[1:])


def test_wrong_fingerprint_rejected(runner, tmp_path):
    manifest, _ = _source_manifest(tmp_path)
    raw = manifest.to_json().encode()
    path = tmp_path / "manifest.json"
    path.write_bytes(raw)
    with pytest.raises(DomainError, match="external fingerprint"):
        runner._reviewed_manifest(path, sha256(raw).hexdigest(), "0" * 64)


def test_no_synthetic_substitution(runner, tmp_path):
    manifest, _ = _source_manifest(tmp_path)
    raw = manifest.to_json().encode()
    path = tmp_path / "manifest.json"
    path.write_bytes(raw)
    with pytest.raises(DomainError, match="historical MP20 scope"):
        runner._reviewed_manifest(path, sha256(raw).hexdigest(), manifest.fingerprint)


def test_unresolved_source_never_granted_permission(runner, tmp_path, monkeypatch):
    manifest, _ = _source_manifest(tmp_path)
    source = historical_cdvae_mp20_source(reviewed_on="2026-09-14")
    # A one-row synthetic snapshot exercises the authorization gate only. It
    # never enters the archival loader or production compiler.
    snapshot = replace(
        manifest.receipts[0].snapshot,
        source_id=source.source_id,
        artifacts=tuple(
            replace(a, sha256=source.license.terms_evidence_sha256)
            if a.role is ArtifactRole.TERMS
            else a
            for a in manifest.receipts[0].snapshot.artifacts
        ),
    )
    manifest = DatasetSourceManifest((DatasetSourceReceipt(source, snapshot),))
    raw = manifest.to_json().encode()
    path = tmp_path / "unresolved.json"
    path.write_bytes(raw)
    monkeypatch.setattr(runner, "FROZEN_SNAPSHOT", {})
    with pytest.raises(DomainError):
        runner._reviewed_manifest(path, sha256(raw).hexdigest(), manifest.fingerprint)
    assert path.read_bytes() == raw


def test_frozen_selection_cannot_be_changed(runner, tmp_path):
    manifest, _ = _source_manifest(tmp_path)
    source = historical_cdvae_mp20_source(reviewed_on="2026-09-14")
    snapshot = replace(
        manifest.receipts[0].snapshot,
        source_id=source.source_id,
        artifacts=tuple(
            replace(a, sha256=source.license.terms_evidence_sha256)
            if a.role is ArtifactRole.TERMS
            else a
            for a in manifest.receipts[0].snapshot.artifacts
        ),
    )
    manifest = DatasetSourceManifest((DatasetSourceReceipt(source, snapshot),))
    raw = manifest.to_json().encode()
    path = tmp_path / "changed-selection.json"
    path.write_bytes(raw)
    with pytest.raises(DomainError, match="frozen 512-row selection"):
        runner._reviewed_manifest(path, sha256(raw).hexdigest(), manifest.fingerprint)


def test_cli_help_has_external_pins(runner):
    completed = subprocess.run(
        [sys.executable, runner.__file__, "--help"],
        capture_output=True,
        text=True,
        check=True,
    )
    for option in (
        "--expected-source-sha256",
        "--expected-source-fingerprint",
        "--expected-certification-sha256",
        "--expected-certification-fingerprint",
    ):
        assert option in completed.stdout


def test_builder_contract_matches_v6(runner):
    catalogue = runner.build_full_catalogue()
    builder = runner.CatalogueDatasetTargetBuilder(
        runner.SpglibStandardizer(runner.pinned_hall_numbers(), runner.SpglibPolicy()),
        runner.CatalogueTargetAssigner(catalogue),
        runner.DomainConfig(lattice_chart="ball-v1"),  # the frozen v6 contract
    )
    assert builder.builder_fingerprint == runner.EXPECTED_BUILDER
