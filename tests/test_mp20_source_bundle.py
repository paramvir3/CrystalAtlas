"""Fail-closed assembler checks that do not require the private local data cache."""

import importlib
from hashlib import sha256
from pathlib import Path

import pytest

from crystalatlas.dataset_provenance import (
    historical_cdvae_mp20_scope,
    historical_cdvae_mp20_source,
)
from crystalatlas.errors import DomainError


@pytest.fixture
def assembler(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    return importlib.import_module("prepare_mp20_source_bundle")


@pytest.mark.parametrize("target", ["source", "pilot", "evidence", "destination"])
def test_pin_cannot_be_inside_inputs_or_bundle(tmp_path, assembler, target):
    paths = {name: tmp_path / name for name in ("source", "pilot", "evidence", "destination")}
    for name in ("source", "pilot", "evidence"):
        paths[name].mkdir()
    with pytest.raises(DomainError, match="trust pin"):
        assembler.assemble(**paths, trust_pin=paths[target] / "pin.json")
    assert not paths["destination"].exists()


def test_existing_destination_is_not_reused(tmp_path, assembler):
    paths = {name: tmp_path / name for name in ("source", "pilot", "evidence", "destination")}
    for path in paths.values():
        path.mkdir()
    with pytest.raises(DomainError, match="must both be new"):
        assembler.assemble(**paths, trust_pin=tmp_path / "pin.json")


def test_pinned_reader_checks_bytes_and_size_limit(tmp_path, assembler):
    path = tmp_path / "input"
    path.write_bytes(b"original")
    digest = sha256(b"original").hexdigest()
    assert assembler._pinned(path, digest) == b"original"
    with pytest.raises(DomainError):
        assembler._pinned(path, digest, limit=2)
    path.write_bytes(b"modified")
    with pytest.raises(DomainError):
        assembler._pinned(path, digest)


def test_pinned_reader_rejects_symlink(tmp_path, assembler):
    path = tmp_path / "input"
    path.write_bytes(b"original")
    link = tmp_path / "link"
    link.symlink_to(path)
    with pytest.raises(DomainError):
        assembler._pinned(link, sha256(b"original").hexdigest())


def test_scope_factory_matches_historical_policy():
    digest = sha256(b"record metadata").hexdigest()
    scope = historical_cdvae_mp20_scope(source_metadata_sha256=digest)
    scope.require_for(historical_cdvae_mp20_source(reviewed_on="2026-09-14"))
    assert scope.source_metadata_sha256 == digest
    assert dict(scope.claims)["provider_signed"] == "false"
