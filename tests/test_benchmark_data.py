from __future__ import annotations

import csv
import io
from dataclasses import replace
from hashlib import sha256
from pathlib import Path

import pytest

from crystalatlas.benchmark_data import (
    BenchmarkCifRow,
    read_benchmark_cif_csv,
    select_training_pilot,
)
from crystalatlas.errors import DomainError


def _csv(
    tmp_path: Path, rows: list[list[str]], header: list[str] | None = None
) -> tuple[Path, str]:
    buffer = io.StringIO(newline="")
    writer = csv.writer(buffer)
    writer.writerow(header if header is not None else ["", "material_id", "cif", "energy"])
    writer.writerows(rows)
    raw = buffer.getvalue().encode("utf-8")
    path = tmp_path / "transport.csv"
    path.write_bytes(raw)
    return path, sha256(raw).hexdigest()


def test_csv_transport_preserves_multiline_cif_and_untrusted_metadata(tmp_path: Path) -> None:
    cif = "data_fixture\r\n# quoted 'text', commas\r\n_cell_length_a 4\r\n"
    path, digest = _csv(tmp_path, [["123", "mp-1", cif, "=not_evaluated()"]])
    (row,) = read_benchmark_cif_csv(
        path, split="train", expected_sha256=digest, expected_count=1
    )
    assert row.cif == cif
    assert row.cif_sha256 == sha256(cif.encode()).hexdigest()
    assert dict(row.metadata) == {
        "__source_index__": "123",
        "material_id": "mp-1",
        "energy": "=not_evaluated()",
    }
    assert row.metadata_sha256 == replace(row).metadata_sha256
    assert row.metadata_sha256 != replace(row, split="test").metadata_sha256


def test_row_metadata_is_deeply_frozen_without_digest_change() -> None:
    mutable = [["material_id", "mp-1"], ["energy", "untrusted text"]]
    row = BenchmarkCifRow("train", 0, "mp-1", "data_one\n", mutable)
    expected = BenchmarkCifRow(
        "train",
        0,
        "mp-1",
        "data_one\n",
        (("material_id", "mp-1"), ("energy", "untrusted text")),
    )
    assert row == expected
    before = row.metadata_sha256
    mutable[0][1] = "changed"
    mutable.append(["new", "value"])
    assert row.metadata == expected.metadata
    assert row.metadata_sha256 == before == expected.metadata_sha256


@pytest.mark.parametrize(
    "metadata",
    [
        None,
        {},
        "ab",
        ["ab"],
        [["one"]],
        [["one", "two", "three"]],
        [[None, "text"]],
        [["name", []]],
        [["", "text"]],
        [["same", "first"], ["same", "second"]],
    ],
)
def test_malformed_row_metadata_raises_domain_error(metadata: object) -> None:
    with pytest.raises(DomainError, match="metadata"):
        BenchmarkCifRow("train", 0, "mp-1", "data_one\n", metadata)


@pytest.mark.parametrize("split", [None, [], {}, "validation"])
def test_malformed_split_raises_domain_error(split: object) -> None:
    with pytest.raises(DomainError, match="split"):
        BenchmarkCifRow(split, 0, "mp-1", "data_one\n", ())


@pytest.mark.parametrize(
    "header,rows",
    [
        (["material_id", "cif", "cif"], [["mp-1", "a", "b"]]),
        (["", "__source_index__", "material_id", "cif"], [["1", "2", "mp-1", "a"]]),
        (["material_id", "cif"], [["mp-1", "a", "extra"]]),
        (["material_id", "cif"], [["mp-1"]]),
        (["material_id", "cif"], [["mp-1", " "]]),
        (["material_id", "cif"], [["mp-1", "a\0b"]]),
        (["material_id", "cif"], [[" mp-1", "a"]]),
        (["material_id", "cif"], [["mp-1", "a"], ["mp-1", "b"]]),
    ],
)
def test_csv_rejects_ambiguous_container(
    tmp_path: Path, header: list[str], rows: list[list[str]]
) -> None:
    path, digest = _csv(tmp_path, rows, header)
    with pytest.raises(DomainError):
        read_benchmark_cif_csv(
            path, split="train", expected_sha256=digest, expected_count=len(rows)
        )


def test_csv_digest_count_dtype_and_symlink_gates(tmp_path: Path) -> None:
    path, digest = _csv(tmp_path, [["0", "mp-1", "data_one\n", "0"]])
    for changes in (
        {"expected_sha256": "0" * 64},
        {"expected_count": 2},
        {"expected_count": True},
        {"split": "validation"},
    ):
        options = dict(split="train", expected_sha256=digest, expected_count=1)
        options.update(changes)
        with pytest.raises(DomainError):
            read_benchmark_cif_csv(path, **options)
    link = tmp_path / "link.csv"
    link.symlink_to(path)
    with pytest.raises(DomainError, match="nonsymlink"):
        read_benchmark_cif_csv(link, split="train", expected_sha256=digest, expected_count=1)


def test_pilot_selection_is_replayable_train_only_and_geometry_independent(
    tmp_path: Path,
) -> None:
    path, digest = _csv(
        tmp_path, [[str(i), f"mp-{i}", f"data_{i}\n", str(i)] for i in range(20)]
    )
    rows = read_benchmark_cif_csv(
        path, split="train", expected_sha256=digest, expected_count=20
    )
    selected = select_training_pilot(rows, count=7, seed=20260913)
    assert selected == select_training_pilot(tuple(reversed(rows)), count=7, seed=20260913)
    changed = tuple(replace(row, cif="arbitrarily difficult geometry") for row in rows)
    assert [row.material_id for row in selected] == [
        row.material_id for row in select_training_pilot(changed, count=7, seed=20260913)
    ]
    assert selected != select_training_pilot(rows, count=7, seed=1)
    with pytest.raises(DomainError, match="published training"):
        select_training_pilot((replace(rows[0], split="test"),), count=1, seed=0)
    with pytest.raises(DomainError, match="unique"):
        select_training_pilot((rows[0], rows[0]), count=1, seed=0)
    for invalid in (0, 21, True):
        with pytest.raises(DomainError):
            select_training_pilot(rows, count=invalid, seed=0)
    with pytest.raises(DomainError):
        select_training_pilot(rows, count=1, seed=True)
