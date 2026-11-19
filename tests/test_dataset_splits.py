from __future__ import annotations

import json
from dataclasses import replace
from fractions import Fraction
from pathlib import Path

import pytest

from crystalatlas.dataset_splits import (
    DatasetSplitManifest,
    FrozenSplitKeys,
    SplitFractions,
    SplitMode,
    SplitRecord,
    assert_split_no_leakage,
    audit_dataset_split,
    build_dataset_split,
    dataset_split_fingerprint,
)
from crystalatlas.errors import DomainError


def _record(
    index: int,
    *,
    composition: str | None = None,
    polymorph: str | None = None,
    prototype: str | None = None,
    scaffold: str | None = None,
    elements: tuple[int, ...] | None = None,
) -> SplitRecord:
    return SplitRecord(
        record_id=f"record-{index:03d}",
        keys=FrozenSplitKeys(
            composition_key=composition or f"composition-{index}",
            polymorph_key=polymorph or f"polymorph-{index}",
            prototype_key=prototype or f"prototype-{index}",
            scaffold_key=scaffold or f"scaffold-{index}",
            element_set=elements or (index % 100 + 1,),
        ),
    )


def _independent_records(count: int = 24) -> tuple[SplitRecord, ...]:
    return tuple(_record(index) for index in range(count))


@pytest.mark.parametrize("mode", tuple(SplitMode))
def test_every_mode_is_deterministic_order_invariant_and_leakage_free(
    mode: SplitMode,
) -> None:
    records = _independent_records()
    fractions = SplitFractions.from_values("3/4", "1/8", "1/8")
    first = build_dataset_split(records, mode=mode, fractions=fractions, seed=1907)
    second = build_dataset_split(
        tuple(reversed(records)), mode=mode, fractions=fractions, seed=1907
    )

    assert first == second
    assert first.manifest.record_counts == (
        ("train", 18),
        ("validation", 3),
        ("test", 3),
    )
    assert len(first.train_ids + first.validation_ids + first.test_ids) == len(records)
    assert_split_no_leakage(records, first.manifest.record_assignments, mode=mode)
    audit_dataset_split(records, first.manifest)


def test_composition_split_preserves_both_polymorphs_without_collapsing_them() -> None:
    records = (
        _record(0, composition="SiO2", polymorph="quartz", elements=(8, 14)),
        _record(1, composition="SiO2", polymorph="stishovite", elements=(8, 14)),
        _record(2),
        _record(3),
        _record(4),
    )
    split = build_dataset_split(
        records,
        mode=SplitMode.COMPOSITION,
        fractions=SplitFractions.from_values("3/5", "1/5", "1/5"),
        seed=4,
    )

    assert split.manifest.partition_for("record-000") == split.manifest.partition_for(
        "record-001"
    )
    assigned = set(split.train_ids + split.validation_ids + split.test_ids)
    assert {"record-000", "record-001"}.issubset(assigned)


@pytest.mark.parametrize(
    ("mode", "first", "second", "error"),
    (
        (
            SplitMode.COMPOSITION,
            _record(0, composition="AB", elements=(1, 2)),
            _record(1, composition="AB", elements=(1, 2)),
            "composition leakage",
        ),
        (
            SplitMode.PROTOTYPE,
            _record(0, prototype="rocksalt"),
            _record(1, prototype="rocksalt"),
            "prototype leakage",
        ),
        (
            SplitMode.SCAFFOLD,
            _record(0, scaffold="hall-523|4a,4b"),
            _record(1, scaffold="hall-523|4a,4b"),
            "scaffold leakage",
        ),
        (
            SplitMode.ELEMENT_SET,
            _record(0, elements=(8, 14)),
            _record(1, elements=(8, 14)),
            "element-set leakage",
        ),
    ),
)
def test_adversarial_mode_key_leakage_is_rejected(
    mode: SplitMode,
    first: SplitRecord,
    second: SplitRecord,
    error: str,
) -> None:
    with pytest.raises(DomainError, match=error):
        assert_split_no_leakage(
            (first, second),
            ((first.record_id, "train"), (second.record_id, "test")),
            mode=mode,
        )


def test_polymorph_aliases_cannot_leak_even_under_an_unrelated_mode() -> None:
    keys = FrozenSplitKeys("AB", "same-physical-structure", "proto", "scaffold", (1, 8))
    records = (SplitRecord("source-a", keys), SplitRecord("source-b", keys))
    with pytest.raises(DomainError, match="polymorph leakage"):
        assert_split_no_leakage(
            records,
            (("source-a", "train"), ("source-b", "validation")),
            mode=SplitMode.PROTOTYPE,
        )


def test_conflicting_alias_and_composition_keys_fail_closed() -> None:
    first = _record(0, composition="AB", polymorph="same", elements=(1, 8))
    conflicting_polymorph = _record(
        1, composition="AB", polymorph="same", scaffold="different", elements=(1, 8)
    )
    with pytest.raises(DomainError, match="polymorph_key maps to conflicting"):
        dataset_split_fingerprint((first, conflicting_polymorph))

    conflicting_composition = _record(2, composition="AB", elements=(6, 8))
    with pytest.raises(DomainError, match="composition_key maps to conflicting"):
        dataset_split_fingerprint((first, conflicting_composition))


def test_manifest_round_trip_and_dataset_binding(tmp_path: Path) -> None:
    records = _independent_records(12)
    split = build_dataset_split(
        records,
        mode=SplitMode.PROTOTYPE,
        fractions=SplitFractions.from_values(Fraction(2, 3), Fraction(1, 6), Fraction(1, 6)),
        seed=99,
    )
    path = tmp_path / "split.json"
    split.manifest.write_json(path)
    restored = DatasetSplitManifest.read_json(
        path,
        expected_fingerprint=split.manifest.manifest_fingerprint,
    )

    assert restored == split.manifest
    assert restored.to_dict() == json.loads(path.read_text(encoding="utf-8"))
    audit_dataset_split(records, restored)
    with pytest.raises(DomainError, match="different frozen dataset"):
        audit_dataset_split(records[:-1] + (_record(200),), restored)
    with pytest.raises(DomainError, match="externally pinned"):
        DatasetSplitManifest.read_json(path, expected_fingerprint="0" * 64)


def test_manifest_reader_rejects_duplicate_json_keys(tmp_path: Path) -> None:
    path = tmp_path / "duplicate.json"
    path.write_text('{"schema":"first","schema":"second"}\n', encoding="utf-8")
    with pytest.raises(DomainError, match="duplicate JSON object key"):
        DatasetSplitManifest.read_json(path)

    nonfinite_path = tmp_path / "nonfinite.json"
    nonfinite_path.write_text('{"schema":Infinity}\n', encoding="utf-8")
    with pytest.raises(DomainError, match="nonfinite JSON number"):
        DatasetSplitManifest.read_json(nonfinite_path)


def test_manifest_tampering_and_non_total_assignment_are_rejected() -> None:
    records = _independent_records(9)
    split = build_dataset_split(
        records,
        mode=SplitMode.SCAFFOLD,
        fractions=SplitFractions.from_values("7/9", "1/9", "1/9"),
        seed=9,
    )
    with pytest.raises(DomainError, match="internally inconsistent"):
        replace(split.manifest, seed=10)
    with pytest.raises(DomainError, match="not total"):
        assert_split_no_leakage(
            records,
            split.manifest.record_assignments[:-1],
            mode=SplitMode.SCAFFOLD,
        )


def test_exact_fraction_and_nonempty_contracts() -> None:
    assert SplitFractions.from_values(0.8, 0.1, 0.1) == SplitFractions(
        Fraction(4, 5), Fraction(1, 10), Fraction(1, 10)
    )
    with pytest.raises(DomainError, match="sum exactly"):
        SplitFractions.from_values(0.8, 0.1, 0.099)
    with pytest.raises(DomainError, match="too few exact groups"):
        build_dataset_split(
            (
                _record(0, composition="same", elements=(1, 2)),
                _record(1, composition="same", elements=(1, 2)),
            ),
            mode=SplitMode.COMPOSITION,
            fractions=SplitFractions.from_values("4/5", "1/10", "1/10"),
            seed=0,
        )


def test_zero_mass_partitions_remain_empty() -> None:
    result = build_dataset_split(
        _independent_records(4),
        mode=SplitMode.COMPOSITION,
        fractions=SplitFractions.from_values("1", "0", "0"),
        seed=0,
    )
    assert len(result.train_ids) == 4
    assert result.validation_ids == ()
    assert result.test_ids == ()
