"""Acquire only the frozen, public CDVAE MP-20 benchmark; never run upstream code.

This is a byte-acquisition tool, not a licence approval or training launcher.
Git blob identities and sizes were independently read from the GitHub contents
API at the revision linked by CrystaLLM's BENCHMARKING.md. SHA-256 receipts are
also recorded. Existing files are verified, never overwritten.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import stat
from datetime import UTC, datetime
from pathlib import Path
from urllib.request import urlopen

REVISION = "f857f598d6f6cca5dc1ea0582d228f12dcc2c2ea"
BASE = f"https://raw.githubusercontent.com/txie-93/cdvae/{REVISION}/data/mp_20"
FILES = {
    "README.md": (1622, "498076c5663bc478e41a3405e8c2da530973f06a"),
    "train.csv": (34975028, "b1c9d498565ac2fd8d075a51ff2ef5e51ba6c21f"),
    "val.csv": (11617279, "7efe4d7ebeb4f460ceaad35e5326c42111d9c9a1"),
    "test.csv": (11599133, "5dc643041df53f5ce4970a79930ad262b4f07acd"),
}
_NOTICE = "Byte verification only. A reviewed source/terms receipt is separate."
_RECEIPT_LIMIT = 1024 * 1024
_DIGEST = re.compile(r"[0-9a-f]{64}\Z")


def _plain_destination(destination: str | Path) -> Path:
    candidate = Path(destination).absolute()
    # Check ancestors before mkdir can create anything through a symlink.
    if candidate.is_symlink() or candidate.resolve() != candidate:
        raise ValueError("destination must not traverse symlinks or '..'")
    return candidate


def _read_regular(path: Path, maximum_bytes: int) -> bytes:
    # O_NOFOLLOW closes the final-component symlink race; O_NONBLOCK avoids
    # blocking on a FIFO before fstat can reject it. Parent paths are checked
    # separately. This is not a sandbox against hostile ancestor replacement.
    descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    with os.fdopen(descriptor, "rb") as stream:
        if not stat.S_ISREG(os.fstat(stream.fileno()).st_mode):
            raise ValueError(f"artifact must be a regular file: {path.name}")
        payload = stream.read(maximum_bytes + 1)
    if len(payload) > maximum_bytes:
        raise ValueError(f"artifact exceeds its byte limit: {path.name}")
    return payload


def _exclusive_write(path: Path, payload: bytes) -> None:
    # Never replace a file that appeared during a download or receipt build.
    with path.open("xb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())


def _artifact(name: str, payload: bytes) -> dict[str, object]:
    size, expected_blob = FILES[name]
    observed = hashlib.sha1(f"blob {len(payload)}\0".encode() + payload).hexdigest()
    if len(payload) != size or observed != expected_blob:
        raise ValueError(f"frozen upstream byte identity mismatch: {name}")
    return {
        "path": name,
        "uri": f"{BASE}/{name}",
        "byte_count": len(payload),
        "git_blob_sha1": observed,
        "sha256": hashlib.sha256(payload).hexdigest(),
    }


def _parse_receipt(payload: bytes) -> dict[str, object]:
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("existing acquisition receipt repeats a key")
            result[key] = value
        return result

    def finite(value):
        raise ValueError(f"existing acquisition receipt contains {value}")

    receipt = json.loads(
        payload.decode("utf-8"), object_pairs_hook=unique, parse_constant=finite
    )
    fields = {
        "schema",
        "revision",
        "retrieved_utc",
        "artifacts",
        "training_authorized_by_this_receipt",
        "notice",
    }
    if (
        not isinstance(receipt, dict)
        or set(receipt) != fields
        or receipt["schema"] != "crystalatlas.mp20-byte-acquisition.v1"
        or receipt["revision"] != REVISION
        or receipt["training_authorized_by_this_receipt"] is not False
        or receipt["notice"] != _NOTICE
        or not isinstance(receipt["artifacts"], list)
        or not isinstance(receipt["retrieved_utc"], str)
    ):
        raise ValueError("existing acquisition receipt has a malformed contract")
    timestamp = datetime.strptime(receipt["retrieved_utc"], "%Y-%m-%dT%H:%M:%SZ")
    if timestamp.strftime("%Y-%m-%dT%H:%M:%SZ") != receipt["retrieved_utc"]:
        raise ValueError("existing acquisition receipt has a noncanonical UTC timestamp")
    artifacts = receipt["artifacts"]
    if len(artifacts) != len(FILES):
        raise ValueError("existing acquisition receipt has the wrong artifact count")
    for artifact, (name, (size, blob)) in zip(artifacts, FILES.items(), strict=True):
        if (
            not isinstance(artifact, dict)
            or set(artifact) != {"path", "uri", "byte_count", "git_blob_sha1", "sha256"}
            or artifact["path"] != name
            or artifact["uri"] != f"{BASE}/{name}"
            or type(artifact["byte_count"]) is not int
            or artifact["byte_count"] != size
            or artifact["git_blob_sha1"] != blob
            or not isinstance(artifact["sha256"], str)
            or not _DIGEST.fullmatch(artifact["sha256"])
        ):
            raise ValueError("existing acquisition receipt has a malformed artifact binding")
    return receipt


def acquire_mp20(destination: str | Path) -> dict[str, object]:
    """Acquire fixed bytes, or verify a completed acquisition without network use.

    A receipt marks completion: if it already exists, missing or altered files
    are errors, not an invitation to repair a supposedly immutable acquisition.
    Without a receipt, verified files from an interrupted acquisition can be
    reused. Neither case overwrites any existing path.
    """
    destination = _plain_destination(destination)
    destination.mkdir(parents=True, exist_ok=True)
    _plain_destination(destination)
    receipt_path = destination / "acquisition.json"
    try:
        original_receipt = _read_regular(receipt_path, _RECEIPT_LIMIT)
    except FileNotFoundError:
        original_receipt = None
    existing = None if original_receipt is None else _parse_receipt(original_receipt)
    artifacts = []
    for name, (size, _) in FILES.items():
        path = destination / name
        uri = f"{BASE}/{name}"
        try:
            payload = _read_regular(path, size)
        except FileNotFoundError as exc:
            if existing is not None:
                raise ValueError(f"completed acquisition is missing artifact: {name}") from exc
            with urlopen(uri, timeout=30) as response:
                if response.url != uri:
                    raise ValueError("unexpected dataset download redirect") from None
                payload = response.read(size + 1)
            _artifact(name, payload)  # Verify downloaded bytes before publishing.
            _exclusive_write(path, payload)
        artifact = _artifact(name, payload)
        artifacts.append(artifact)
        print(json.dumps(artifact, sort_keys=True), flush=True)
    # Bind the receipt to files actually present, not just download buffers.
    _plain_destination(destination)
    for artifact in artifacts:
        name = artifact["path"]
        if _artifact(name, _read_regular(destination / name, FILES[name][0])) != artifact:
            raise ValueError(f"artifact changed before receipt publication: {name}")
    if existing is not None:
        if existing["artifacts"] != artifacts:
            raise ValueError("existing acquisition receipt does not bind the frozen files")
        if _read_regular(receipt_path, _RECEIPT_LIMIT) != original_receipt:
            raise ValueError("existing acquisition receipt changed during verification")
        return existing
    receipt = {
        "schema": "crystalatlas.mp20-byte-acquisition.v1",
        "revision": REVISION,
        "retrieved_utc": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "artifacts": artifacts,
        "training_authorized_by_this_receipt": False,
        "notice": _NOTICE,
    }
    payload = (json.dumps(receipt, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()
    _exclusive_write(receipt_path, payload)
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("destination", type=Path)
    acquire_mp20(parser.parse_args().destination)


if __name__ == "__main__":
    main()
