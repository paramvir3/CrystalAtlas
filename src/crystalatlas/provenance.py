"""Reproducibility records for datasets, checkpoints, and generated batches."""

from __future__ import annotations

import json
import platform
import subprocess
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
from typing import Any

from .config import CrystalAtlasConfig


def sha256_file(path: str | Path, *, block_size: int = 1 << 20) -> str:
    """Hash a file without loading it into memory."""

    digest = sha256()
    with Path(path).open("rb") as handle:
        while block := handle.read(block_size):
            digest.update(block)
    return digest.hexdigest()


def _git_revision(worktree: Path) -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=worktree,
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    revision = result.stdout.strip()
    return revision or None


@dataclass(frozen=True, slots=True)
class ProvenanceRecord:
    """Immutable run identity suitable for JSON serialization."""

    created_utc: str
    config_fingerprint: str
    dataset_hashes: tuple[tuple[str, str], ...]
    catalogue_hash: str | None
    package_version: str
    python_version: str
    platform: str
    git_revision: str | None
    random_seed: int
    task: str

    @classmethod
    def create(
        cls,
        *,
        config: CrystalAtlasConfig,
        dataset_paths: Iterable[str | Path] = (),
        catalogue_path: str | Path | None = None,
        random_seed: int,
        task: str,
        package_version: str,
        worktree: str | Path = ".",
    ) -> ProvenanceRecord:
        if random_seed < 0:
            raise ValueError("random_seed must be nonnegative")
        hashes = tuple(
            sorted((str(Path(path).resolve()), sha256_file(path)) for path in dataset_paths)
        )
        catalogue_hash = sha256_file(catalogue_path) if catalogue_path else None
        return cls(
            created_utc=datetime.now(timezone.utc).isoformat(),  # noqa: UP017
            config_fingerprint=config.fingerprint,
            dataset_hashes=hashes,
            catalogue_hash=catalogue_hash,
            package_version=package_version,
            python_version=platform.python_version(),
            platform=platform.platform(),
            git_revision=_git_revision(Path(worktree)),
            random_seed=random_seed,
            task=task,
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def write_json(self, path: str | Path) -> None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        payload = json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"
        destination.write_text(payload, encoding="utf-8")


def verify_provenance(record: ProvenanceRecord) -> list[str]:
    """Return all file-integrity failures; an empty list means success."""

    failures: list[str] = []
    for path_text, expected in record.dataset_hashes:
        path = Path(path_text)
        if not path.is_file():
            failures.append(f"missing dataset file: {path}")
        elif sha256_file(path) != expected:
            failures.append(f"dataset hash mismatch: {path}")
    return failures
