"""Strict benchmark-container parsing and train-only pilot selection.

CSV is only a transport for original CIF text. This module does not reinterpret
coordinates, filter difficult structures, authorize data use, or turn a paper's
random split into a composition/prototype-disjoint split. Those are separate,
audited stages. Raw values are retained as text, including upstream property
metadata; no upstream pickle, Python literal, or formula is evaluated.
"""

from __future__ import annotations

import csv
import io
import json
import re
from collections.abc import Sequence
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path

from .errors import DomainError

_DIGEST = re.compile(r"[0-9a-f]{64}\Z")
_SPLITS = frozenset({"train", "val", "test"})
_MAX_CONTAINER_BYTES = 64 * 1024 * 1024


@dataclass(frozen=True, slots=True)
class BenchmarkCifRow:
    """One unmodified CIF and all its source-row metadata (zero-based row index)."""

    split: str
    row_index: int
    material_id: str
    cif: str
    metadata: tuple[tuple[str, str], ...]

    def __post_init__(self) -> None:
        if not isinstance(self.split, str) or self.split not in _SPLITS:
            raise DomainError("benchmark split must be train, val, or test")
        if (
            isinstance(self.row_index, bool)
            or not isinstance(self.row_index, int)
            or self.row_index < 0
        ):
            raise DomainError("benchmark row index must be a nonnegative integer")
        if (
            not isinstance(self.material_id, str)
            or not self.material_id
            or self.material_id.strip() != self.material_id
            or any(ord(character) < 32 for character in self.material_id)
        ):
            raise DomainError("benchmark material identifier is malformed")
        if not isinstance(self.cif, str) or not self.cif.strip() or "\0" in self.cif:
            raise DomainError("benchmark CIF must be nonempty text without NUL")
        if not isinstance(self.metadata, (tuple, list)):
            raise DomainError("benchmark metadata must be a sequence of text-field pairs")
        metadata = []
        for pair in self.metadata:
            if not isinstance(pair, (tuple, list)) or len(pair) != 2:
                raise DomainError("benchmark metadata entries must be text-field pairs")
            key, value = pair
            if not isinstance(key, str) or not key or not isinstance(value, str):
                raise DomainError("benchmark metadata entries must be named text fields")
            metadata.append((key, value))
        if len({key for key, _ in metadata}) != len(metadata):
            raise DomainError("benchmark metadata must contain uniquely named text fields")
        object.__setattr__(self, "metadata", tuple(metadata))

    @property
    def cif_sha256(self) -> str:
        return sha256(self.cif.encode("utf-8")).hexdigest()

    @property
    def metadata_sha256(self) -> str:
        payload = {
            "schema": "crystalatlas.benchmark-cif-row.v1",
            "split": self.split,
            "row_index": self.row_index,
            "material_id": self.material_id,
            "cif_sha256": self.cif_sha256,
            "metadata": dict(self.metadata),
        }
        return sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
        ).hexdigest()


def read_benchmark_cif_csv(
    path: str | Path,
    *,
    split: str,
    expected_sha256: str,
    expected_count: int,
) -> tuple[BenchmarkCifRow, ...]:
    """Verify exact container bytes before parsing every row; never drop a row.

    A hard container limit bounds memory use for the small MP-20-style pilot.
    The standard CSV parser's field-size limit remains in force: an oversized
    CIF is rejected explicitly, not truncated or silently skipped.
    """
    if not isinstance(split, str) or split not in _SPLITS:
        raise DomainError("benchmark split must be train, val, or test")
    if not isinstance(expected_sha256, str) or not _DIGEST.fullmatch(expected_sha256):
        raise DomainError("expected benchmark SHA-256 must be a lowercase digest")
    if (
        isinstance(expected_count, bool)
        or not isinstance(expected_count, int)
        or expected_count < 1
    ):
        raise DomainError("expected benchmark count must be a positive integer")
    source = Path(path)
    try:
        if source.is_symlink() or not source.is_file():
            raise DomainError("benchmark input must be a nonsymlink regular file")
        with source.open("rb") as stream:
            raw = stream.read(_MAX_CONTAINER_BYTES + 1)
    except OSError as exc:
        raise DomainError(f"cannot read benchmark container {source}") from exc
    if len(raw) > _MAX_CONTAINER_BYTES:
        raise DomainError("benchmark container exceeds the small-dataset size limit")
    if sha256(raw).hexdigest() != expected_sha256:
        raise DomainError("benchmark container SHA-256 disagrees with the external pin")
    try:
        text = raw.decode("utf-8", errors="strict")
        reader = csv.reader(io.StringIO(text, newline=""), strict=True)
        header = next(reader)
        # CDVAE's pandas export includes an unnamed first index column. Preserve
        # it under an explicit transport-only name; never use it as material ID.
        if header and header[0] == "":
            if "__source_index__" in header:
                raise DomainError("benchmark source-index metadata name collides")
            header[0] = "__source_index__"
        if (
            not header
            or any(not field or field.strip() != field for field in header)
            or len(set(header)) != len(header)
            or not {"material_id", "cif"}.issubset(header)
        ):
            raise DomainError("benchmark header must uniquely name material_id and cif")
        rows = []
        seen = set()
        id_column = header.index("material_id")
        cif_column = header.index("cif")
        for index, fields in enumerate(reader):
            if len(fields) != len(header):
                raise DomainError(f"benchmark row {index} has the wrong number of fields")
            row = BenchmarkCifRow(
                split=split,
                row_index=index,
                material_id=fields[id_column],
                cif=fields[cif_column],
                metadata=tuple(
                    (name, value)
                    for name, value in zip(header, fields, strict=True)
                    if name != "cif"
                ),
            )
            if row.material_id in seen:
                raise DomainError(f"benchmark repeats material identifier {row.material_id!r}")
            seen.add(row.material_id)
            rows.append(row)
            if len(rows) > expected_count:
                raise DomainError("benchmark contains more rows than the frozen count")
    except (UnicodeDecodeError, csv.Error, StopIteration) as exc:
        raise DomainError("benchmark must be strict UTF-8 CSV with a header") from exc
    if len(rows) != expected_count:
        raise DomainError("benchmark row count disagrees with the frozen count")
    return tuple(rows)


def select_training_pilot(
    rows: Sequence[BenchmarkCifRow], *, count: int, seed: int
) -> tuple[BenchmarkCifRow, ...]:
    """Select by SHA-256(seed, ID), before any physics/target success is known.

    The returned order is hash rank (ID as a deterministic tie breaker). The
    subset is independent of source iteration order and of any model outputs,
    energies, species, space groups, or preprocessing success. Never refill
    rejected rows: record the smaller accepted population and its denominator.
    """
    if isinstance(count, bool) or not isinstance(count, int) or not 1 <= count <= len(rows):
        raise DomainError("pilot count must lie between one and the number of source rows")
    if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 2**64:
        raise DomainError("pilot seed must be an unsigned 64-bit integer")
    if any(not isinstance(row, BenchmarkCifRow) or row.split != "train" for row in rows):
        raise DomainError("a training pilot may only select from the published training split")
    if len({row.material_id for row in rows}) != len(rows):
        raise DomainError("training pilot source identifiers must be unique")

    def rank(row: BenchmarkCifRow) -> tuple[str, str]:
        payload = json.dumps(
            ["crystalatlas.train-pilot-selection.v1", seed, row.material_id],
            separators=(",", ":"),
        ).encode("utf-8")
        return sha256(payload).hexdigest(), row.material_id

    return tuple(sorted(rows, key=rank)[:count])
