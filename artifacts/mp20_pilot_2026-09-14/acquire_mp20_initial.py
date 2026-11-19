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
from datetime import datetime, timezone
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    destination = args.destination
    if destination.is_symlink():
        raise ValueError("destination cannot be a symlink")
    destination.mkdir(parents=True, exist_ok=True)
    if destination.resolve() != destination.absolute():
        raise ValueError("destination must not traverse symlinks or '..'")
    artifacts = []
    for name, (size, expected_blob) in FILES.items():
        path = destination / name
        uri = f"{BASE}/{name}"
        if path.is_symlink():
            raise ValueError(f"refusing symlink {path}")
        if path.exists():
            if not path.is_file() or path.stat().st_size != size:
                raise ValueError(f"existing artifact has wrong type/size: {path}")
            payload = path.read_bytes()
        else:
            with urlopen(uri, timeout=30) as response:
                if response.url != uri:
                    raise ValueError("unexpected dataset download redirect")
                payload = response.read(size + 1)
        observed = hashlib.sha1(f"blob {len(payload)}\0".encode() + payload).hexdigest()
        if len(payload) != size or observed != expected_blob:
            raise ValueError(f"frozen upstream byte identity mismatch: {name}")
        if not path.exists():
            with path.open("xb") as stream:
                stream.write(payload)
                stream.flush()
                os.fsync(stream.fileno())
        artifact = {
            "path": name,
            "uri": uri,
            "byte_count": len(payload),
            "git_blob_sha1": observed,
            "sha256": hashlib.sha256(payload).hexdigest(),
        }
        artifacts.append(artifact)
        print(json.dumps(artifact, sort_keys=True), flush=True)
    receipt = {
        "schema": "crystalatlas.mp20-byte-acquisition.v1",
        "revision": REVISION,
        "retrieved_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "artifacts": artifacts,
        "training_authorized_by_this_receipt": False,
        "notice": "Byte verification only. A reviewed source/terms receipt is separate.",
    }
    path = destination / "acquisition.json"
    if not path.exists():
        with path.open("x", encoding="utf-8") as stream:
            json.dump(receipt, stream, sort_keys=True, indent=2, allow_nan=False)
            stream.write("\n")


if __name__ == "__main__":
    main()
