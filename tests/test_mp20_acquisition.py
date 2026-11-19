from __future__ import annotations

import copy
import importlib.util
import io
import json
import os
from hashlib import sha1, sha256
from pathlib import Path

import pytest

_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "acquire_mp20.py"
_SPEC = importlib.util.spec_from_file_location("mp20_acquisition_script", _SCRIPT)
assert _SPEC is not None and _SPEC.loader is not None
acquisition = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(acquisition)


@pytest.fixture
def source(monkeypatch: pytest.MonkeyPatch) -> dict[str, bytes]:
    # Tiny arbitrary payloads test the acquisition protocol, not CIF science.
    payloads = {"README.md": b"notice\n", "train.csv": b"a", "val.csv": b"b,c\n"}
    monkeypatch.setattr(
        acquisition,
        "FILES",
        {
            name: (len(raw), sha1(f"blob {len(raw)}\0".encode() + raw).hexdigest())
            for name, raw in payloads.items()
        },
    )
    return payloads


def _network(monkeypatch: pytest.MonkeyPatch, payloads: dict[str, bytes]) -> list[str]:
    calls: list[str] = []

    class Response(io.BytesIO):
        def __init__(self, uri: str, raw: bytes) -> None:
            super().__init__(raw)
            self.url = uri

    def download(uri: str, *, timeout: int) -> Response:
        assert timeout == 30
        calls.append(uri)
        return Response(uri, payloads[uri.rsplit("/", 1)[1]])

    monkeypatch.setattr(acquisition, "urlopen", download)
    return calls


def _offline(monkeypatch: pytest.MonkeyPatch) -> None:
    def unexpected(*args: object, **kwargs: object) -> None:
        pytest.fail("completed acquisition verification must not use the network")

    monkeypatch.setattr(acquisition, "urlopen", unexpected)


def _completed(root: Path, payloads: dict[str, bytes]) -> dict[str, object]:
    root.mkdir()
    for name, raw in payloads.items():
        (root / name).write_bytes(raw)
    receipt = {
        "schema": "crystalatlas.mp20-byte-acquisition.v1",
        "revision": acquisition.REVISION,
        "retrieved_utc": "2026-09-14T00:00:00Z",
        "artifacts": [acquisition._artifact(name, raw) for name, raw in payloads.items()],
        "training_authorized_by_this_receipt": False,
        "notice": acquisition._NOTICE,
    }
    (root / "acquisition.json").write_text(json.dumps(receipt), encoding="utf-8")
    return receipt


def test_acquire_frozen_bytes_and_offline_verify_without_overwrite(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, source: dict[str, bytes]
) -> None:
    root = tmp_path / "download"
    calls = _network(monkeypatch, source)
    receipt = acquisition.acquire_mp20(root)
    assert calls == [f"{acquisition.BASE}/{name}" for name in source]
    assert receipt["training_authorized_by_this_receipt"] is False
    assert [entry["sha256"] for entry in receipt["artifacts"]] == [
        sha256(raw).hexdigest() for raw in source.values()
    ]
    before = {
        path.name: (path.read_bytes(), path.stat().st_mtime_ns) for path in root.iterdir()
    }
    _offline(monkeypatch)
    assert acquisition.acquire_mp20(root) == receipt
    assert {
        path.name: (path.read_bytes(), path.stat().st_mtime_ns) for path in root.iterdir()
    } == before


def test_symlink_ancestor_rejected_before_any_mkdir_or_download(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, source: dict[str, bytes]
) -> None:
    actual = tmp_path / "actual"
    actual.mkdir()
    linked = tmp_path / "linked"
    linked.symlink_to(actual, target_is_directory=True)
    _offline(monkeypatch)
    with pytest.raises(ValueError, match="symlinks"):
        acquisition.acquire_mp20(linked / "must-not-create" / "nested")
    assert list(actual.iterdir()) == []
    with pytest.raises(ValueError, match="symlinks"):
        acquisition.acquire_mp20(linked)


@pytest.mark.parametrize("kind", ["symlink", "fifo"])
def test_nonregular_existing_artifact_is_rejected_without_network_or_receipt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, source: dict[str, bytes], kind: str
) -> None:
    root = tmp_path / "download"
    root.mkdir()
    path = root / "README.md"
    if kind == "symlink":
        target = tmp_path / "elsewhere"
        target.write_bytes(source["README.md"])
        path.symlink_to(target)
    else:
        os.mkfifo(path)
    _offline(monkeypatch)
    with pytest.raises((OSError, ValueError)):
        acquisition.acquire_mp20(root)
    assert not (root / "acquisition.json").exists()


def test_exclusive_artifact_publication_race_never_emits_false_receipt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, source: dict[str, bytes]
) -> None:
    root = tmp_path / "download"
    _network(monkeypatch, source)
    publish = acquisition._exclusive_write

    def race(path: Path, raw: bytes) -> None:
        path.write_bytes(b"competing writer")
        publish(path, raw)

    monkeypatch.setattr(acquisition, "_exclusive_write", race)
    with pytest.raises(FileExistsError):
        acquisition.acquire_mp20(root)
    assert (root / "README.md").read_bytes() == b"competing writer"
    assert not (root / "acquisition.json").exists()


def test_exclusive_receipt_publication_race_does_not_overwrite_competitor(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, source: dict[str, bytes]
) -> None:
    root = tmp_path / "download"
    _network(monkeypatch, source)
    publish = acquisition._exclusive_write

    def race(path: Path, raw: bytes) -> None:
        if path.name == "acquisition.json":
            path.write_bytes(b"other receipt")
        publish(path, raw)

    monkeypatch.setattr(acquisition, "_exclusive_write", race)
    with pytest.raises(FileExistsError):
        acquisition.acquire_mp20(root)
    assert (root / "acquisition.json").read_bytes() == b"other receipt"


def test_existing_receipt_is_offline_and_missing_files_are_not_repaired(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, source: dict[str, bytes]
) -> None:
    root = tmp_path / "complete"
    _completed(root, source)
    # Only delete this test's explicitly created fixture, not a corpus artifact.
    (root / "train.csv").unlink()
    _offline(monkeypatch)
    before = (root / "acquisition.json").read_bytes()
    with pytest.raises(ValueError, match="missing artifact"):
        acquisition.acquire_mp20(root)
    assert not (root / "train.csv").exists()
    assert (root / "acquisition.json").read_bytes() == before


@pytest.mark.parametrize(
    "field", ["revision", "notice", "sha256", "byte_count", "extra", "order"]
)
def test_existing_receipt_wrong_contract_fails_without_network_or_changes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, source: dict[str, bytes], field: str
) -> None:
    root = tmp_path / "complete"
    receipt = copy.deepcopy(_completed(root, source))
    if field in {"revision", "notice"}:
        receipt[field] = "incorrect"
    elif field == "sha256":
        receipt["artifacts"][0]["sha256"] = "0" * 64
    elif field == "byte_count":
        # bool compares equal to 1; its JSON type must still be rejected.
        receipt["artifacts"][1]["byte_count"] = True
    elif field == "extra":
        receipt["artifacts"][0]["extra"] = "unexpected"
    else:
        receipt["artifacts"].reverse()
    before = json.dumps(receipt).encode()
    (root / "acquisition.json").write_bytes(before)
    _offline(monkeypatch)
    with pytest.raises(ValueError, match="receipt"):
        acquisition.acquire_mp20(root)
    assert (root / "acquisition.json").read_bytes() == before


@pytest.mark.parametrize(
    "payload",
    [
        b'{"schema":"a","schema":"b"}',
        b'{"artifacts":[{"sha256":"a","sha256":"b"}]}',
        b'{"bad":NaN}',
        b'{"bad":Infinity}',
        b'{"bad":-Infinity}',
        b"[]",
        b"null",
        b"\xff",
    ],
)
def test_existing_receipt_rejects_ambiguous_json(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, source: dict[str, bytes], payload: bytes
) -> None:
    root = tmp_path / "complete"
    root.mkdir()
    (root / "acquisition.json").write_bytes(payload)
    _offline(monkeypatch)
    with pytest.raises(ValueError):
        acquisition.acquire_mp20(root)
    assert list(root.iterdir()) == [root / "acquisition.json"]
    assert (root / "acquisition.json").read_bytes() == payload


def test_existing_file_mutation_during_final_verification_cannot_publish_receipt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, source: dict[str, bytes]
) -> None:
    root = tmp_path / "download"
    _network(monkeypatch, source)
    read = acquisition._read_regular
    count = 0

    def change(path: Path, maximum_bytes: int) -> bytes:
        nonlocal count
        if path.name == "README.md":
            count += 1
            if count == 2:
                path.write_bytes(b"altered")
        return read(path, maximum_bytes)

    monkeypatch.setattr(acquisition, "_read_regular", change)
    with pytest.raises(ValueError):
        acquisition.acquire_mp20(root)
    assert not (root / "acquisition.json").exists()


def test_interrupted_verified_download_is_reused_without_overwrite(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, source: dict[str, bytes]
) -> None:
    root = tmp_path / "partial"
    root.mkdir()
    (root / "README.md").write_bytes(source["README.md"])
    timestamp = (root / "README.md").stat().st_mtime_ns
    calls = _network(monkeypatch, source)
    acquisition.acquire_mp20(root)
    assert calls == [f"{acquisition.BASE}/{name}" for name in ("train.csv", "val.csv")]
    assert (root / "README.md").stat().st_mtime_ns == timestamp
