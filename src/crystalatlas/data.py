"""Dataset identities, leakage-resistant splits, and declared training measures."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Hashable, Sequence
from dataclasses import dataclass
from enum import Enum
from hashlib import sha256

import numpy as np
import numpy.typing as npt

from .errors import DomainError


@dataclass(frozen=True, slots=True)
class CrystalRecord:
    """Canonical metadata required before neural tensorization."""

    record_id: str
    species: tuple[int, ...]
    composition_key: str
    structure_key: str
    source: str
    payload_index: int
    symmetry_ambiguous: bool = False

    def __post_init__(self) -> None:
        identity = (
            self.record_id,
            self.species,
            self.composition_key,
            self.structure_key,
            self.source,
        )
        if not all(identity):
            raise DomainError("dataset record identifiers and species must be nonempty")
        if tuple(sorted(set(self.species))) != self.species:
            raise DomainError("record species must be a sorted unique tuple")
        if self.payload_index < 0:
            raise DomainError("payload_index must be nonnegative")


class TrainingMeasure(str, Enum):  # noqa: UP042 - ML smoke runtime is Python 3.10
    RECORD = "record"
    UNIQUE_FORMULA = "unique-formula"
    SYSTEM_FORMULA_POLYMORPH = "system-formula-polymorph"


def _normalize(weights: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    if np.any(~np.isfinite(weights)) or np.any(weights < 0.0):
        raise DomainError("training weights must be finite and nonnegative")
    total = float(weights.sum())
    if total <= 0.0:
        raise DomainError("training weights must have positive total")
    return weights / total


def record_probabilities(
    records: Sequence[CrystalRecord], measure: TrainingMeasure
) -> npt.NDArray[np.float64]:
    """Return the exact declared record probability for equation (3.15a)."""

    if not records:
        raise DomainError("training records must be nonempty")
    count = len(records)
    if measure is TrainingMeasure.RECORD:
        return np.full(count, 1.0 / count, dtype=np.float64)

    by_formula: dict[str, list[int]] = defaultdict(list)
    by_system: dict[tuple[int, ...], dict[str, list[int]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for index, record in enumerate(records):
        by_formula[record.composition_key].append(index)
        by_system[record.species][record.composition_key].append(index)

    probabilities = np.zeros(count, dtype=np.float64)
    if measure is TrainingMeasure.UNIQUE_FORMULA:
        formula_mass = 1.0 / len(by_formula)
        for indices in by_formula.values():
            # Multiple canonical polymorphs share the formula mass uniformly.
            for index in indices:
                probabilities[index] = formula_mass / len(indices)
        return _normalize(probabilities)

    system_mass = 1.0 / len(by_system)
    for formulas in by_system.values():
        formula_mass = system_mass / len(formulas)
        for indices in formulas.values():
            polymorph_keys: dict[str, list[int]] = defaultdict(list)
            for index in indices:
                polymorph_keys[records[index].structure_key].append(index)
            polymorph_mass = formula_mass / len(polymorph_keys)
            for duplicates in polymorph_keys.values():
                # Exact duplicate records do not change physical polymorph mass.
                for index in duplicates:
                    probabilities[index] = polymorph_mass / len(duplicates)
    return _normalize(probabilities)


def deterministic_group_split(
    records: Sequence[CrystalRecord],
    *,
    group: str,
    train_fraction: float,
    validation_fraction: float,
    salt: str,
) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]:
    """Split complete formula, species-system, or structure groups without leakage."""

    if group not in {"composition", "species", "structure"}:
        raise DomainError("group must be composition, species, or structure")
    if not 0.0 < train_fraction < 1.0:
        raise DomainError("train_fraction must lie strictly between zero and one")
    if not 0.0 <= validation_fraction < 1.0 - train_fraction:
        raise DomainError("validation_fraction is incompatible with train_fraction")
    if not salt:
        raise DomainError("split salt must be nonempty")

    grouped: dict[Hashable, list[int]] = defaultdict(list)
    for index, record in enumerate(records):
        if group == "composition":
            key: Hashable = record.composition_key
        elif group == "species":
            key = record.species
        else:
            key = record.structure_key
        grouped[key].append(index)

    train: list[int] = []
    validation: list[int] = []
    test: list[int] = []
    for key, indices in grouped.items():
        digest = sha256(f"{salt}|{key!r}".encode()).digest()
        uniform = int.from_bytes(digest[:8], "big") / float(1 << 64)
        if uniform < train_fraction:
            train.extend(indices)
        elif uniform < train_fraction + validation_fraction:
            validation.extend(indices)
        else:
            test.extend(indices)
    return tuple(sorted(train)), tuple(sorted(validation)), tuple(sorted(test))


def assert_no_group_leakage(
    records: Sequence[CrystalRecord],
    splits: tuple[Sequence[int], Sequence[int], Sequence[int]],
    *,
    group: str,
) -> None:
    """Raise if a declared group key appears in more than one split."""

    def key(record: CrystalRecord) -> Hashable:
        if group == "composition":
            return record.composition_key
        if group == "species":
            return record.species
        if group == "structure":
            return record.structure_key
        raise DomainError("unknown leakage group")

    key_sets = [{key(records[index]) for index in split} for split in splits]
    if key_sets[0] & key_sets[1] or key_sets[0] & key_sets[2] or key_sets[1] & key_sets[2]:
        raise DomainError(f"{group} leakage detected across dataset splits")
