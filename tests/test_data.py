from __future__ import annotations

import unittest

from crystalatlas.data import (
    CrystalRecord,
    TrainingMeasure,
    assert_no_group_leakage,
    deterministic_group_split,
    record_probabilities,
)


def _record(
    index: int, species: tuple[int, ...], formula: str, structure: str
) -> CrystalRecord:
    return CrystalRecord(
        record_id=f"r{index}",
        species=species,
        composition_key=formula,
        structure_key=structure,
        source="fixture",
        payload_index=index,
    )


class DataTests(unittest.TestCase):
    def test_duplicate_records_do_not_change_polymorph_mass(self) -> None:
        records = (
            _record(0, (3, 8), "Li2O", "phase-a"),
            _record(1, (3, 8), "Li2O", "phase-a"),
            _record(2, (3, 8), "Li2O", "phase-b"),
            _record(3, (11, 17), "NaCl", "rocksalt"),
        )
        weights = record_probabilities(
            records, TrainingMeasure.SYSTEM_FORMULA_POLYMORPH
        )
        self.assertAlmostEqual(float(weights[0] + weights[1]), float(weights[2]))
        self.assertAlmostEqual(float(weights[:3].sum()), float(weights[3]))

    def test_group_split_is_reproducible_and_leakage_free(self) -> None:
        records = tuple(
            _record(index, (3, 8), f"formula-{index // 2}", f"structure-{index}")
            for index in range(20)
        )
        split_a = deterministic_group_split(
            records,
            group="composition",
            train_fraction=0.6,
            validation_fraction=0.2,
            salt="frozen-v1",
        )
        split_b = deterministic_group_split(
            records,
            group="composition",
            train_fraction=0.6,
            validation_fraction=0.2,
            salt="frozen-v1",
        )
        self.assertEqual(split_a, split_b)
        assert_no_group_leakage(records, split_a, group="composition")
        self.assertEqual(sum(len(split) for split in split_a), len(records))


if __name__ == "__main__":
    unittest.main()
