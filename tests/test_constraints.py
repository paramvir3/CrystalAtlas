from __future__ import annotations

import random
import unittest
from fractions import Fraction
from itertools import product
from time import perf_counter

from crystalatlas.composition import Composition
from crystalatlas.constraints import (
    ConstraintError,
    NoCompletionError,
    OccupationTable,
    RowAction,
    SuffixReachability,
    TerminalRowError,
    WyckoffRowSpec,
)


class FixedIndexRng:
    def __init__(self, index: int) -> None:
        self.index = index

    def randrange(self, stop: int, /) -> int:
        if not 0 <= self.index < stop:
            raise AssertionError(f"test index {self.index} is outside range({stop})")
        return self.index


def independent_local_actions(
    row: WyckoffRowSpec,
    residual: tuple[int, ...],
) -> tuple[tuple[int, ...], ...]:
    maxima = tuple(value // row.multiplicity for value in residual)
    candidates = product(*(range(maximum + 1) for maximum in maxima))
    if row.free_dimension == 0:
        return tuple(action for action in candidates if sum(action) <= 1)
    return tuple(candidates)


def independent_suffixes(
    rows: tuple[WyckoffRowSpec, ...],
    row_index: int,
    residual: tuple[int, ...],
) -> tuple[tuple[tuple[int, ...], ...], ...]:
    if row_index == len(rows):
        return ((),) if all(value == 0 for value in residual) else ()

    row = rows[row_index]
    completions: list[tuple[tuple[int, ...], ...]] = []
    for action in independent_local_actions(row, residual):
        child = tuple(
            value - row.multiplicity * count
            for value, count in zip(residual, action, strict=True)
        )
        for suffix in independent_suffixes(rows, row_index + 1, child):
            completions.append((action,) + suffix)
    return tuple(completions)


def table_counts(table: OccupationTable) -> tuple[tuple[int, ...], ...]:
    return tuple(action.counts for action in table.actions)


class SuffixReachabilityTests(unittest.TestCase):
    def setUp(self) -> None:
        self.composition = Composition(((1, 2), (8, 1)))
        self.rows = (
            WyckoffRowSpec("fixed-2", multiplicity=2, free_dimension=0),
            WyckoffRowSpec("free-a", multiplicity=1, free_dimension=1),
            WyckoffRowSpec("free-b", multiplicity=1, free_dimension=2),
        )
        self.oracle = SuffixReachability(self.composition, 1, self.rows)
        self.brute = independent_suffixes(self.rows, 0, (2, 1))

    def test_suffix_dp_is_sound_and_complete_against_independent_bruteforce(self) -> None:
        generated = tuple(self.oracle.iter_completions())
        self.assertEqual({table_counts(table) for table in generated}, set(self.brute))
        self.assertEqual(len(generated), len(self.brute))
        self.assertEqual(self.oracle.number_of_completions, len(self.brute))
        self.assertTrue(self.oracle.is_reachable())

        for table in generated:
            self.assertTrue(self.oracle.accepts(table))
            self.assertEqual(self.oracle.expanded_counts(table), self.oracle.initial_residual)
            self.assertLessEqual(table.actions[0].number_of_orbits, 1)

    def test_every_allowed_action_has_a_suffix_and_every_such_action_is_allowed(self) -> None:
        states: set[tuple[int, tuple[int, ...]]] = set()
        for raw_table in self.brute:
            residual = self.oracle.initial_residual
            states.add((0, residual))
            for row_index, raw_action in enumerate(raw_table):
                row = self.rows[row_index]
                residual = tuple(
                    value - row.multiplicity * count
                    for value, count in zip(residual, raw_action, strict=True)
                )
                states.add((row_index + 1, residual))

        for row_index, residual in states:
            self.assertTrue(self.oracle.is_reachable(row_index, residual))
            if row_index == len(self.rows):
                self.assertEqual(residual, (0, 0))
                continue
            expected = {
                action
                for action in independent_local_actions(self.rows[row_index], residual)
                if independent_suffixes(
                    self.rows,
                    row_index + 1,
                    tuple(
                        value - self.rows[row_index].multiplicity * count
                        for value, count in zip(residual, action, strict=True)
                    ),
                )
            }
            actual = {
                action.counts
                for action in self.oracle.allowed_actions(row_index, residual)
            }
            self.assertEqual(actual, expected)
            self.assertTrue(actual)

    def test_capacity_one_and_terminal_rules_cannot_be_bypassed(self) -> None:
        with self.assertRaises(ConstraintError):
            self.oracle.advance(0, self.oracle.initial_residual, RowAction((1, 1)))
        with self.assertRaises(ConstraintError):
            self.oracle.advance(0, self.oracle.initial_residual, RowAction((2, 0)))
        with self.assertRaises(TerminalRowError):
            self.oracle.allowed_actions(len(self.rows), (0, 0))
        self.assertTrue(self.oracle.is_reachable(len(self.rows), (0, 0)))
        self.assertFalse(self.oracle.is_reachable(len(self.rows), (1, 0)))

    def test_fixed_row_actions_are_exact_axes_for_many_species(self) -> None:
        species_count = 12
        fixed = WyckoffRowSpec("fixed-many", multiplicity=3, free_dimension=0)
        oracle = SuffixReachability(
            Composition(tuple((z, 1) for z in range(1, species_count + 1))),
            1,
            (fixed,),
        )
        residual = (0, 3, 8, 2, 99, 3, 5, 6, 1, 12, 0, 4)
        expected: list[tuple[int, ...]] = [(0,) * species_count]
        for index in reversed(range(species_count)):
            if residual[index] >= fixed.multiplicity:
                unit = [0] * species_count
                unit[index] = 1
                expected.append(tuple(unit))
        self.assertEqual(
            tuple(action.counts for action in oracle.local_actions(0, residual)),
            tuple(expected),
        )

    def test_impossible_formula_has_no_false_positive(self) -> None:
        impossible = SuffixReachability(
            Composition(((1, 1), (8, 1))),
            1,
            (
                WyckoffRowSpec("fixed-a", multiplicity=2, free_dimension=0),
                WyckoffRowSpec("fixed-b", multiplicity=2, free_dimension=0),
            ),
        )
        self.assertFalse(impossible.is_reachable())
        self.assertEqual(impossible.number_of_completions, 0)
        self.assertEqual(tuple(impossible.iter_completions()), ())
        with self.assertRaises(NoCompletionError):
            impossible.sample_uniform(random.Random(1))

    def test_boolean_reachability_does_not_enumerate_suffix_cardinalities(self) -> None:
        composition = Composition.from_counts(
            {6: 7, 8: 7, 14: 7, 16: 7, 26: 7, 29: 5}
        )
        rows = tuple(
            WyckoffRowSpec(f"free-{index}", multiplicity=2, free_dimension=1)
            for index in range(12)
        ) + (WyckoffRowSpec("general", multiplicity=4, free_dimension=3),)
        oracle = SuffixReachability(composition, 1, rows)
        started = perf_counter()
        self.assertFalse(oracle.is_reachable())  # odd residuals cannot use 2/4 rows
        self.assertLess(
            perf_counter() - started,
            1.0,
            "Boolean support masking regressed to exponential completion counting",
        )

    def test_exact_enumeration_unranking_and_uniform_normalization(self) -> None:
        enumerated = tuple(self.oracle.iter_completions())
        unranked = tuple(
            self.oracle.completion_at(index)
            for index in range(self.oracle.number_of_completions)
        )
        self.assertEqual(unranked, enumerated)
        for index, expected in enumerate(enumerated):
            self.assertEqual(self.oracle.sample_uniform(FixedIndexRng(index)), expected)

        exact_mass = sum(
            (Fraction(1, self.oracle.number_of_completions) for _ in enumerated),
            start=Fraction(0, 1),
        )
        self.assertEqual(exact_mass, Fraction(1, 1))

        rng_a = random.Random(20260825)
        rng_b = random.Random(20260825)
        draws_a = tuple(self.oracle.sample_uniform(rng_a) for _ in range(50))
        draws_b = tuple(self.oracle.sample_uniform(rng_b) for _ in range(50))
        self.assertEqual(draws_a, draws_b)

    def test_row_and_action_data_are_immutable_and_validated(self) -> None:
        action_source = [0, 1]
        action = RowAction(action_source)  # type: ignore[arg-type]
        action_source[1] = 9
        self.assertEqual(action.counts, (0, 1))
        with self.assertRaises(ValueError):
            WyckoffRowSpec("bad", multiplicity=1, free_dimension=4)
        with self.assertRaises(ConstraintError):
            SuffixReachability(
                self.composition,
                1,
                (
                    WyckoffRowSpec("same", 1, 0),
                    WyckoffRowSpec("same", 1, 1),
                ),
            )


if __name__ == "__main__":
    unittest.main()
