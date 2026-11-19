r"""Deterministic, leakage-resistant splits over frozen crystal identities.

The split layer deliberately does not infer identities from floating-point
structures.  Canonicalization and duplicate detection must first freeze the
five keys in :class:`FrozenSplitKeys`; this module then treats equality of
those keys as exact equality.

For a requested split mode, records are partitioned into indivisible groups
``G_k``.  If ``N`` is the number of records and ``f_s`` is the exact requested
fraction for partition ``s``, the allocator greedily minimizes the exact
Pearson discrepancy

.. math::

   L(n) = \sum_{s:f_s>0} \frac{(n_s - f_s N)^2}{f_s N}

after every whole-group assignment.  Groups are processed largest first;
cryptographic ranks derived from the seed resolve all ties.  Balanced graph
partitioning is NP-hard in general, so this is a deterministic, scalable
heuristic rather than a claim of globally optimal cardinalities.  Leakage
freedom, unlike cardinality optimality, is exact and is audited separately.
"""

from __future__ import annotations

import json
import math
import os
import tempfile
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from fractions import Fraction
from hashlib import sha256
from pathlib import Path
from typing import Any, Final

from .errors import DomainError

SPLIT_MANIFEST_SCHEMA: Final = "crystalatlas.dataset-split-manifest.v1"
_PARTITIONS: Final = ("train", "validation", "test")


def _canonical_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _fingerprint(value: object) -> str:
    return sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _reject_duplicate_json_keys(
    pairs: list[tuple[str, object]],
) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise DomainError(f"duplicate JSON object key {key!r}")
        result[key] = value
    return result


def _reject_nonfinite_json_number(value: str) -> object:
    raise DomainError(f"split manifest contains nonfinite JSON number {value!r}")


def _require_sha256(value: str, *, name: str) -> None:
    if len(value) != 64:
        raise DomainError(f"{name} must be a SHA-256 hexadecimal digest")
    try:
        int(value, 16)
    except ValueError as exc:
        raise DomainError(f"{name} must be a SHA-256 hexadecimal digest") from exc


def _require_key(value: str, *, name: str) -> None:
    if not isinstance(value, str) or not value:
        raise DomainError(f"{name} must be nonempty text")
    if value != value.strip() or any(ord(character) < 32 for character in value):
        raise DomainError(
            f"{name} must be a frozen canonical key without edge/control whitespace"
        )


class SplitMode(str, Enum):  # noqa: UP042 - importable by the Python-3.10 ML smoke image
    """The exact equivalence relation held disjoint across partitions."""

    COMPOSITION = "composition"
    PROTOTYPE = "prototype"
    SCAFFOLD = "scaffold"
    ELEMENT_SET = "element-set"


@dataclass(frozen=True, slots=True)
class FrozenSplitKeys:
    """Upstream-certified identities used by the splitter without inference.

    ``polymorph_key`` identifies one canonical physical structure after
    supercell, setting, and duplicate reduction.  Multiple records may retain
    that key when source aliases are intentionally preserved; they can never
    cross a split.  Distinct polymorph keys with the same composition are not
    collapsed.
    """

    composition_key: str
    polymorph_key: str
    prototype_key: str
    scaffold_key: str
    element_set: tuple[int, ...]

    def __post_init__(self) -> None:
        for name in ("composition_key", "polymorph_key", "prototype_key", "scaffold_key"):
            _require_key(getattr(self, name), name=name)
        if not self.element_set:
            raise DomainError("element_set must be nonempty")
        if self.element_set != tuple(sorted(set(self.element_set))):
            raise DomainError("element_set must be a sorted tuple of unique atomic numbers")
        if any(
            isinstance(number, bool) or not 1 <= number <= 118 for number in self.element_set
        ):
            raise DomainError("element_set contains an invalid atomic number")

    def to_dict(self) -> dict[str, object]:
        return {
            "composition": self.composition_key,
            "polymorph": self.polymorph_key,
            "prototype": self.prototype_key,
            "scaffold": self.scaffold_key,
            "element_set": list(self.element_set),
        }


@dataclass(frozen=True, slots=True)
class SplitRecord:
    """Stable record identifier together with its already frozen split keys."""

    record_id: str
    keys: FrozenSplitKeys

    def __post_init__(self) -> None:
        _require_key(self.record_id, name="record_id")
        if not isinstance(self.keys, FrozenSplitKeys):
            raise TypeError("keys must be FrozenSplitKeys")

    def to_dict(self) -> dict[str, object]:
        return {"record_id": self.record_id, "keys": self.keys.to_dict()}


def _as_fraction(value: Fraction | int | float | str, *, name: str) -> Fraction:
    if isinstance(value, bool):
        raise DomainError(f"{name} must be a finite rational number")
    if isinstance(value, float):
        if not math.isfinite(value):
            raise DomainError(f"{name} must be a finite rational number")
        result = Fraction(str(value))
    else:
        try:
            result = Fraction(value)
        except (TypeError, ValueError, ZeroDivisionError) as exc:
            raise DomainError(f"{name} must be a finite rational number") from exc
    return result


def _fraction_text(value: Fraction) -> str:
    if value.denominator == 1:
        return str(value.numerator)
    return f"{value.numerator}/{value.denominator}"


@dataclass(frozen=True, slots=True)
class SplitFractions:
    """Exact train/validation/test masses whose sum is exactly one."""

    train: Fraction
    validation: Fraction
    test: Fraction

    def __post_init__(self) -> None:
        values = tuple(
            _as_fraction(value, name=name)
            for name, value in (
                ("train", self.train),
                ("validation", self.validation),
                ("test", self.test),
            )
        )
        object.__setattr__(self, "train", values[0])
        object.__setattr__(self, "validation", values[1])
        object.__setattr__(self, "test", values[2])
        if any(value < 0 for value in values):
            raise DomainError("split fractions must be nonnegative")
        if sum(values, Fraction()) != 1:
            raise DomainError("split fractions must sum exactly to one")
        if values[0] == 0:
            raise DomainError("the training fraction must be positive")

    @classmethod
    def from_values(
        cls,
        train: Fraction | int | float | str,
        validation: Fraction | int | float | str,
        test: Fraction | int | float | str,
    ) -> SplitFractions:
        """Construct fractions, interpreting decimal floats by their printed decimal."""

        return cls(
            _as_fraction(train, name="train"),
            _as_fraction(validation, name="validation"),
            _as_fraction(test, name="test"),
        )

    def items(self) -> tuple[tuple[str, Fraction], ...]:
        return tuple(zip(_PARTITIONS, (self.train, self.validation, self.test), strict=True))

    def to_dict(self) -> dict[str, str]:
        return {name: _fraction_text(value) for name, value in self.items()}


def _mode_key(record: SplitRecord, mode: SplitMode) -> str | tuple[int, ...]:
    if mode is SplitMode.COMPOSITION:
        return record.keys.composition_key
    if mode is SplitMode.PROTOTYPE:
        return record.keys.prototype_key
    if mode is SplitMode.SCAFFOLD:
        return record.keys.scaffold_key
    if mode is SplitMode.ELEMENT_SET:
        return record.keys.element_set
    raise DomainError(f"unsupported split mode {mode!r}")


def _encoded_group_key(mode: SplitMode, value: str | tuple[int, ...]) -> str:
    encoded: object = list(value) if isinstance(value, tuple) else value
    return _canonical_json({"mode": mode.value, "value": encoded})


def _canonical_records(records: Sequence[SplitRecord]) -> tuple[SplitRecord, ...]:
    if not records:
        raise DomainError("at least one split record is required")
    if any(not isinstance(record, SplitRecord) for record in records):
        raise TypeError("every split input must be a SplitRecord")
    ordered = tuple(sorted(records, key=lambda record: record.record_id))
    record_ids = tuple(record.record_id for record in ordered)
    if len(set(record_ids)) != len(record_ids):
        raise DomainError("record_id values must be globally unique")

    # A physical polymorph may have source aliases, but its frozen scientific
    # identity cannot depend on which source supplied the alias.
    by_polymorph: dict[str, FrozenSplitKeys] = {}
    composition_elements: dict[str, tuple[int, ...]] = {}
    for record in ordered:
        previous = by_polymorph.setdefault(record.keys.polymorph_key, record.keys)
        if previous != record.keys:
            raise DomainError(
                "one polymorph_key maps to conflicting frozen "
                "composition/prototype/scaffold keys"
            )
        elements = composition_elements.setdefault(
            record.keys.composition_key, record.keys.element_set
        )
        if elements != record.keys.element_set:
            raise DomainError("one composition_key maps to conflicting element sets")
    return ordered


def dataset_split_fingerprint(records: Sequence[SplitRecord]) -> str:
    """Return an input-order-invariant fingerprint of all records and frozen keys."""

    ordered = _canonical_records(records)
    return _fingerprint(
        {
            "schema": "crystalatlas.frozen-split-input.v1",
            "records": [record.to_dict() for record in ordered],
        }
    )


def _seeded_rank(seed: int, *parts: str) -> str:
    return _fingerprint(
        {"schema": "crystalatlas.split-rank.v1", "seed": seed, "parts": list(parts)}
    )


def _require_seed(seed: int) -> None:
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise DomainError("split seed must be a nonnegative integer")


def _allocation_payload(
    *,
    mode: SplitMode,
    seed: int,
    fractions: SplitFractions,
    require_nonempty: bool,
    dataset_fingerprint: str,
    record_assignments: tuple[tuple[str, str], ...],
    group_assignments: tuple[tuple[str, str], ...],
) -> dict[str, object]:
    return {
        "schema": "crystalatlas.dataset-split-allocation.v1",
        "mode": mode.value,
        "seed": seed,
        "fractions": fractions.to_dict(),
        "require_nonempty": require_nonempty,
        "dataset_fingerprint": dataset_fingerprint,
        "record_assignments": [list(item) for item in record_assignments],
        "group_assignments": [list(item) for item in group_assignments],
    }


@dataclass(frozen=True, slots=True)
class DatasetSplitManifest:
    """Self-checking split receipt containing stable record/group assignments."""

    mode: SplitMode
    seed: int
    fractions: SplitFractions
    require_nonempty: bool
    dataset_fingerprint: str
    record_count: int
    group_count: int
    record_assignments: tuple[tuple[str, str], ...]
    group_assignments: tuple[tuple[str, str], ...]
    record_counts: tuple[tuple[str, int], ...]
    group_counts: tuple[tuple[str, int], ...]
    allocation_fingerprint: str
    manifest_fingerprint: str

    def __post_init__(self) -> None:
        if not isinstance(self.mode, SplitMode):
            raise TypeError("mode must be SplitMode")
        _require_seed(self.seed)
        if not isinstance(self.fractions, SplitFractions):
            raise TypeError("fractions must be SplitFractions")
        if not isinstance(self.require_nonempty, bool):
            raise TypeError("require_nonempty must be bool")
        _require_sha256(self.dataset_fingerprint, name="dataset_fingerprint")
        _require_sha256(self.allocation_fingerprint, name="allocation_fingerprint")
        _require_sha256(self.manifest_fingerprint, name="manifest_fingerprint")
        if self.record_count <= 0 or self.record_count != len(self.record_assignments):
            raise DomainError("record_count does not match record assignments")
        if self.group_count <= 0 or self.group_count != len(self.group_assignments):
            raise DomainError("group_count does not match group assignments")
        if tuple(sorted(self.record_assignments)) != self.record_assignments:
            raise DomainError("record assignments must be sorted canonically")
        if tuple(sorted(self.group_assignments)) != self.group_assignments:
            raise DomainError("group assignments must be sorted canonically")
        if len({record_id for record_id, _ in self.record_assignments}) != self.record_count:
            raise DomainError("record assignments contain duplicate identifiers")
        if len({key for key, _ in self.group_assignments}) != self.group_count:
            raise DomainError("group assignments contain duplicate keys")
        if any(split not in _PARTITIONS for _, split in self.record_assignments):
            raise DomainError("record assignment contains an unknown partition")
        if any(split not in _PARTITIONS for _, split in self.group_assignments):
            raise DomainError("group assignment contains an unknown partition")
        expected_record_counts = tuple(
            (split, sum(partition == split for _, partition in self.record_assignments))
            for split in _PARTITIONS
        )
        expected_group_counts = tuple(
            (split, sum(partition == split for _, partition in self.group_assignments))
            for split in _PARTITIONS
        )
        if self.record_counts != expected_record_counts:
            raise DomainError("record_counts do not match record assignments")
        if self.group_counts != expected_group_counts:
            raise DomainError("group_counts do not match group assignments")
        fraction_map = dict(self.fractions.items())
        for split, count in self.record_counts:
            if fraction_map[split] == 0 and count != 0:
                raise DomainError("a zero-mass partition contains assigned records")
            if self.require_nonempty and fraction_map[split] > 0 and count == 0:
                raise DomainError("a required nonempty partition contains no records")
        unsigned_allocation = _allocation_payload(
            mode=self.mode,
            seed=self.seed,
            fractions=self.fractions,
            require_nonempty=self.require_nonempty,
            dataset_fingerprint=self.dataset_fingerprint,
            record_assignments=self.record_assignments,
            group_assignments=self.group_assignments,
        )
        if self.allocation_fingerprint != _fingerprint(unsigned_allocation):
            raise DomainError("allocation fingerprint is internally inconsistent")
        if self.manifest_fingerprint != _fingerprint(self._unsigned_payload()):
            raise DomainError("split manifest fingerprint is internally inconsistent")

    def _unsigned_payload(self) -> dict[str, object]:
        return {
            **_allocation_payload(
                mode=self.mode,
                seed=self.seed,
                fractions=self.fractions,
                require_nonempty=self.require_nonempty,
                dataset_fingerprint=self.dataset_fingerprint,
                record_assignments=self.record_assignments,
                group_assignments=self.group_assignments,
            ),
            "schema": SPLIT_MANIFEST_SCHEMA,
            "record_count": self.record_count,
            "group_count": self.group_count,
            "record_counts": dict(self.record_counts),
            "group_counts": dict(self.group_counts),
            "allocation_fingerprint": self.allocation_fingerprint,
        }

    def to_dict(self) -> dict[str, object]:
        return {**self._unsigned_payload(), "manifest_fingerprint": self.manifest_fingerprint}

    def write_json(self, path: str | Path) -> None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        payload = (json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n").encode(
            "utf-8"
        )
        descriptor, temporary_name = tempfile.mkstemp(
            dir=destination.parent,
            prefix=f".{destination.name}.",
            suffix=".tmp",
        )
        try:
            with os.fdopen(descriptor, "wb") as stream:
                stream.write(payload)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary_name, destination)
        except BaseException:
            try:
                os.unlink(temporary_name)
            except FileNotFoundError:
                pass
            raise

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> DatasetSplitManifest:
        required = {
            "schema",
            "mode",
            "seed",
            "fractions",
            "require_nonempty",
            "dataset_fingerprint",
            "record_count",
            "group_count",
            "record_assignments",
            "group_assignments",
            "record_counts",
            "group_counts",
            "allocation_fingerprint",
            "manifest_fingerprint",
        }
        if set(payload) != required:
            raise DomainError("split manifest has missing or unexpected fields")
        if payload["schema"] != SPLIT_MANIFEST_SCHEMA:
            raise DomainError("unsupported split manifest schema")
        fractions_payload = payload["fractions"]
        if not isinstance(fractions_payload, Mapping) or set(fractions_payload) != set(
            _PARTITIONS
        ):
            raise DomainError("split manifest fractions are malformed")
        try:
            mode = SplitMode(payload["mode"])
            fractions = SplitFractions.from_values(
                fractions_payload["train"],
                fractions_payload["validation"],
                fractions_payload["test"],
            )
            record_assignments = tuple(
                (str(item[0]), str(item[1])) for item in payload["record_assignments"]
            )
            group_assignments = tuple(
                (str(item[0]), str(item[1])) for item in payload["group_assignments"]
            )
            record_counts_payload = payload["record_counts"]
            group_counts_payload = payload["group_counts"]
            if not isinstance(record_counts_payload, Mapping) or not isinstance(
                group_counts_payload, Mapping
            ):
                raise DomainError("split manifest counts are malformed")
            if set(record_counts_payload) != set(_PARTITIONS) or set(
                group_counts_payload
            ) != set(_PARTITIONS):
                raise DomainError("split manifest count partitions are malformed")
            return cls(
                mode=mode,
                seed=payload["seed"],
                fractions=fractions,
                require_nonempty=payload["require_nonempty"],
                dataset_fingerprint=payload["dataset_fingerprint"],
                record_count=payload["record_count"],
                group_count=payload["group_count"],
                record_assignments=record_assignments,
                group_assignments=group_assignments,
                record_counts=tuple(
                    (split, record_counts_payload[split]) for split in _PARTITIONS
                ),
                group_counts=tuple(
                    (split, group_counts_payload[split]) for split in _PARTITIONS
                ),
                allocation_fingerprint=payload["allocation_fingerprint"],
                manifest_fingerprint=payload["manifest_fingerprint"],
            )
        except (KeyError, TypeError, ValueError, IndexError) as exc:
            raise DomainError("split manifest values are malformed") from exc

    @classmethod
    def read_json(
        cls,
        path: str | Path,
        *,
        expected_fingerprint: str | None = None,
    ) -> DatasetSplitManifest:
        try:
            payload = json.loads(
                Path(path).read_text(encoding="utf-8"),
                object_pairs_hook=_reject_duplicate_json_keys,
                parse_constant=_reject_nonfinite_json_number,
            )
        except (OSError, json.JSONDecodeError) as exc:
            raise DomainError("could not read split manifest JSON") from exc
        if not isinstance(payload, Mapping):
            raise DomainError("split manifest JSON root must be an object")
        manifest = cls.from_dict(payload)
        if expected_fingerprint is not None:
            _require_sha256(expected_fingerprint, name="expected_fingerprint")
            if manifest.manifest_fingerprint != expected_fingerprint:
                raise DomainError(
                    "split manifest does not match the externally pinned fingerprint"
                )
        return manifest

    def partition_for(self, record_id: str) -> str:
        assignments = dict(self.record_assignments)
        try:
            return assignments[record_id]
        except KeyError as exc:
            raise DomainError(
                f"record {record_id!r} is absent from the split manifest"
            ) from exc


@dataclass(frozen=True, slots=True)
class DatasetSplit:
    """Convenient stable-ID view of a certified split manifest."""

    train_ids: tuple[str, ...]
    validation_ids: tuple[str, ...]
    test_ids: tuple[str, ...]
    manifest: DatasetSplitManifest

    def __post_init__(self) -> None:
        expected = {
            split: tuple(
                record_id
                for record_id, assigned in self.manifest.record_assignments
                if assigned == split
            )
            for split in _PARTITIONS
        }
        if self.train_ids != expected["train"]:
            raise DomainError("train_ids disagree with the split manifest")
        if self.validation_ids != expected["validation"]:
            raise DomainError("validation_ids disagree with the split manifest")
        if self.test_ids != expected["test"]:
            raise DomainError("test_ids disagree with the split manifest")


def _partition_loss(counts: Mapping[str, int], targets: Mapping[str, Fraction]) -> Fraction:
    loss = Fraction()
    for split, target in targets.items():
        if target > 0:
            delta = Fraction(counts[split]) - target
            loss += delta * delta / target
    return loss


def assert_split_no_leakage(
    records: Sequence[SplitRecord],
    assignments: Mapping[str, str] | Sequence[tuple[str, str]],
    *,
    mode: SplitMode,
) -> None:
    """Require a total partition with neither mode-key nor polymorph leakage."""

    if not isinstance(mode, SplitMode):
        raise TypeError("mode must be SplitMode")
    ordered = _canonical_records(records)
    assignment_items = (
        tuple(assignments.items()) if isinstance(assignments, Mapping) else tuple(assignments)
    )
    if len({record_id for record_id, _ in assignment_items}) != len(assignment_items):
        raise DomainError("split assignments repeat a record identifier")
    assignment = dict(assignment_items)
    expected_ids = {record.record_id for record in ordered}
    if set(assignment) != expected_ids:
        missing = sorted(expected_ids - set(assignment))
        extra = sorted(set(assignment) - expected_ids)
        raise DomainError(f"split assignment is not total; missing={missing}, extra={extra}")
    if any(split not in _PARTITIONS for split in assignment.values()):
        raise DomainError("split assignment contains an unknown partition")

    group_partition: dict[str, str] = {}
    polymorph_partition: dict[str, str] = {}
    for record in ordered:
        partition = assignment[record.record_id]
        previous_polymorph = polymorph_partition.setdefault(
            record.keys.polymorph_key, partition
        )
        if previous_polymorph != partition:
            raise DomainError("canonical polymorph leakage detected across dataset partitions")
        group_key = _encoded_group_key(mode, _mode_key(record, mode))
        previous_group = group_partition.setdefault(group_key, partition)
        if previous_group != partition:
            raise DomainError(f"{mode.value} leakage detected across dataset partitions")


def audit_dataset_split(records: Sequence[SplitRecord], manifest: DatasetSplitManifest) -> None:
    """Recompute every input-dependent invariant in a persisted split receipt."""

    if not isinstance(manifest, DatasetSplitManifest):
        raise TypeError("manifest must be DatasetSplitManifest")
    ordered = _canonical_records(records)
    if dataset_split_fingerprint(ordered) != manifest.dataset_fingerprint:
        raise DomainError("split manifest belongs to a different frozen dataset")
    assert_split_no_leakage(ordered, manifest.record_assignments, mode=manifest.mode)

    record_assignment = dict(manifest.record_assignments)
    reconstructed_groups: dict[str, str] = {}
    for record in ordered:
        key = _encoded_group_key(manifest.mode, _mode_key(record, manifest.mode))
        reconstructed_groups.setdefault(key, record_assignment[record.record_id])
    if tuple(sorted(reconstructed_groups.items())) != manifest.group_assignments:
        raise DomainError("manifest group assignments do not match the frozen record keys")


def build_dataset_split(
    records: Sequence[SplitRecord],
    *,
    mode: SplitMode,
    fractions: SplitFractions,
    seed: int,
    require_nonempty: bool = True,
) -> DatasetSplit:
    """Allocate exact identity groups and return a self-audited split receipt."""

    if not isinstance(mode, SplitMode):
        raise TypeError("mode must be SplitMode")
    if not isinstance(fractions, SplitFractions):
        raise TypeError("fractions must be SplitFractions")
    _require_seed(seed)
    if not isinstance(require_nonempty, bool):
        raise TypeError("require_nonempty must be bool")
    ordered = _canonical_records(records)
    groups: dict[str, list[SplitRecord]] = defaultdict(list)
    for record in ordered:
        key = _encoded_group_key(mode, _mode_key(record, mode))
        groups[key].append(record)

    positive_partitions = tuple(split for split, fraction in fractions.items() if fraction > 0)
    if require_nonempty and len(groups) < len(positive_partitions):
        raise DomainError("too few exact groups to make every positive-mass partition nonempty")
    targets = {split: fraction * len(ordered) for split, fraction in fractions.items()}
    counts = dict.fromkeys(_PARTITIONS, 0)
    groups_per_partition = dict.fromkeys(_PARTITIONS, 0)
    allocation: dict[str, str] = {}
    ordered_groups = sorted(
        groups.items(),
        key=lambda item: (-len(item[1]), _seeded_rank(seed, "group-order", item[0]), item[0]),
    )

    for group_index, (group_key, members) in enumerate(ordered_groups):
        remaining_after = len(ordered_groups) - group_index - 1
        candidates: list[tuple[Fraction, str, str]] = []
        for partition in positive_partitions:
            trial_counts = dict(counts)
            trial_counts[partition] += len(members)
            if require_nonempty:
                empty_after = sum(
                    groups_per_partition[split] + (split == partition) == 0
                    for split in positive_partitions
                )
                if empty_after > remaining_after:
                    continue
            candidates.append(
                (
                    _partition_loss(trial_counts, targets),
                    _seeded_rank(seed, "partition-tie", group_key, partition),
                    partition,
                )
            )
        if not candidates:
            raise DomainError("no feasible whole-group split assignment exists")
        _, _, chosen = min(candidates)
        allocation[group_key] = chosen
        counts[chosen] += len(members)
        groups_per_partition[chosen] += 1

    record_assignments = tuple(
        sorted(
            (record.record_id, allocation[_encoded_group_key(mode, _mode_key(record, mode))])
            for record in ordered
        )
    )
    group_assignments = tuple(sorted(allocation.items()))
    dataset_fingerprint = dataset_split_fingerprint(ordered)
    allocation_fingerprint = _fingerprint(
        _allocation_payload(
            mode=mode,
            seed=seed,
            fractions=fractions,
            require_nonempty=require_nonempty,
            dataset_fingerprint=dataset_fingerprint,
            record_assignments=record_assignments,
            group_assignments=group_assignments,
        )
    )
    unsigned = {
        **_allocation_payload(
            mode=mode,
            seed=seed,
            fractions=fractions,
            require_nonempty=require_nonempty,
            dataset_fingerprint=dataset_fingerprint,
            record_assignments=record_assignments,
            group_assignments=group_assignments,
        ),
        "schema": SPLIT_MANIFEST_SCHEMA,
        "record_count": len(ordered),
        "group_count": len(groups),
        "record_counts": counts,
        "group_counts": groups_per_partition,
        "allocation_fingerprint": allocation_fingerprint,
    }
    manifest = DatasetSplitManifest(
        mode=mode,
        seed=seed,
        fractions=fractions,
        require_nonempty=require_nonempty,
        dataset_fingerprint=dataset_fingerprint,
        record_count=len(ordered),
        group_count=len(groups),
        record_assignments=record_assignments,
        group_assignments=group_assignments,
        record_counts=tuple((split, counts[split]) for split in _PARTITIONS),
        group_counts=tuple((split, groups_per_partition[split]) for split in _PARTITIONS),
        allocation_fingerprint=allocation_fingerprint,
        manifest_fingerprint=_fingerprint(unsigned),
    )
    audit_dataset_split(ordered, manifest)
    ids = {
        split: tuple(
            record_id for record_id, assigned in record_assignments if assigned == split
        )
        for split in _PARTITIONS
    }
    return DatasetSplit(ids["train"], ids["validation"], ids["test"], manifest)


__all__ = [
    "SPLIT_MANIFEST_SCHEMA",
    "DatasetSplit",
    "DatasetSplitManifest",
    "FrozenSplitKeys",
    "SplitFractions",
    "SplitMode",
    "SplitRecord",
    "assert_split_no_leakage",
    "audit_dataset_split",
    "build_dataset_split",
    "dataset_split_fingerprint",
]
