from __future__ import annotations

import random
import unittest
from itertools import combinations, permutations, product
from math import gcd

from crystalatlas.composition import (
    EOS_TOKEN,
    Composition,
    CompositionDomain,
    CompositionError,
    CompositionGrammar,
    CompositionToken,
    InvalidPrefixError,
    TerminalPrefixError,
    TokenKind,
)


class FixedIndexRng:
    def __init__(self, index: int) -> None:
        self.index = index

    def randrange(self, stop: int, /) -> int:
        if not 0 <= self.index < stop:
            raise AssertionError(f"test index {self.index} is outside range({stop})")
        return self.index


def coefficient_gcd(values: tuple[int, ...]) -> int:
    result = 0
    for value in values:
        result = gcd(result, value)
    return result


def independently_enumerate_counts(
    elements: tuple[int, ...],
    max_elements: int,
    max_atoms: int,
) -> set[tuple[tuple[int, int], ...]]:
    expected: set[tuple[tuple[int, int], ...]] = set()
    for support_size in range(1, min(max_elements, max_atoms, len(elements)) + 1):
        for support in combinations(elements, support_size):
            for coefficients in product(range(1, max_atoms + 1), repeat=support_size):
                if sum(coefficients) > max_atoms:
                    continue
                if coefficient_gcd(coefficients) != 1:
                    continue
                expected.add(tuple(zip(support, coefficients, strict=True)))
    return expected


class CompositionInvariantTests(unittest.TestCase):
    def test_raw_formula_is_sorted_combined_and_reduced(self) -> None:
        expected = Composition(((8, 2), (14, 1)))
        self.assertEqual(Composition.from_counts([(14, 2), (8, 4)]), expected)
        self.assertEqual(
            Composition.from_counts([(8, 1), (14, 2), (8, 3), (1, 0)]),
            expected,
        )
        self.assertEqual(expected.support, (8, 14))
        self.assertEqual(expected.coefficients, (2, 1))
        self.assertEqual(expected.total_atoms, 3)
        self.assertEqual(expected.coefficient(14), 1)
        self.assertEqual(expected.coefficient(6), 0)
        self.assertEqual(expected.scaled_counts(3), ((8, 6), (14, 3)))

    def test_strict_constructor_rejects_noncanonical_vectors(self) -> None:
        invalid = (
            (),
            ((8, 2), (14, 2)),
            ((14, 1), (8, 1)),
            ((8, 1), (8, 2)),
            ((8, 0),),
            ((0, 1),),
            ((119, 1),),
        )
        for counts in invalid:
            with self.subTest(counts=counts), self.assertRaises(CompositionError):
                Composition(counts)
        with self.assertRaises(CompositionError):
            Composition.from_counts({8: 0, 14: 0})
        with self.assertRaises(ValueError):
            Composition.from_counts({8: -1})
        with self.assertRaises(TypeError):
            Composition.from_counts({True: 1})

    def test_reduction_is_invariant_to_permutation_and_common_scale(self) -> None:
        grammar = CompositionGrammar(CompositionDomain((1, 6, 8), 2, 4))
        for composition in grammar.iter_compositions():
            for ordering in permutations(composition.counts):
                for multiplier in range(1, 5):
                    raw = [(z, multiplier * count) for z, count in ordering]
                    self.assertEqual(Composition.from_counts(raw), composition)


class CompositionGrammarTests(unittest.TestCase):
    def setUp(self) -> None:
        self.domain = CompositionDomain((1, 6, 8), max_elements=2, max_atoms=4)
        self.grammar = CompositionGrammar(self.domain)
        self.expected = independently_enumerate_counts((1, 6, 8), 2, 4)

    def test_language_equals_independent_reduced_formula_enumeration(self) -> None:
        generated = tuple(self.grammar.iter_compositions())
        self.assertEqual({composition.counts for composition in generated}, self.expected)
        self.assertEqual(len(generated), len(self.expected))
        self.assertEqual(self.grammar.completion_count(), len(self.expected))

    def test_serialization_is_support_first_typed_and_bijective(self) -> None:
        serializations = tuple(self.grammar.iter_serializations())
        self.assertEqual(len(serializations), len(set(serializations)))
        for tokens in serializations:
            composition = self.grammar.deserialize(tokens)
            k = len(composition)
            self.assertEqual(tokens[0], CompositionToken.support_size(k))
            self.assertTrue(all(token.kind is TokenKind.ELEMENT for token in tokens[1 : 1 + k]))
            self.assertTrue(
                all(token.kind is TokenKind.COEFFICIENT for token in tokens[1 + k : 1 + 2 * k])
            )
            self.assertEqual(tokens[-1], EOS_TOKEN)
            self.assertEqual(self.grammar.serialize(composition), tokens)

    def test_every_live_action_is_exactly_a_child_of_an_accepted_leaf(self) -> None:
        leaves = tuple(self.grammar.iter_serializations())
        prefixes: set[tuple[CompositionToken, ...]] = {()}
        for leaf in leaves:
            prefixes.update(leaf[:length] for length in range(1, len(leaf) + 1))

        for prefix in prefixes:
            matching = tuple(leaf for leaf in leaves if leaf[: len(prefix)] == prefix)
            self.assertEqual(self.grammar.completion_count(prefix), len(matching))
            if prefix in leaves:
                with self.assertRaises(TerminalPrefixError):
                    self.grammar.allowed_next(prefix)
                continue
            expected_children = {leaf[len(prefix)] for leaf in matching}
            self.assertEqual(set(self.grammar.allowed_next(prefix)), expected_children)

        dead = (CompositionToken.support_size(2), CompositionToken.element(8))
        self.assertFalse(self.grammar.is_live_prefix(dead))
        with self.assertRaises(InvalidPrefixError):
            self.grammar.allowed_next(dead)

    def test_clamped_support_is_exact_sdng_fibre(self) -> None:
        prefix = self.grammar.support_prefix([8, 6])
        self.assertEqual(
            prefix,
            (
                CompositionToken.support_size(2),
                CompositionToken.element(6),
                CompositionToken.element(8),
            ),
        )
        generated = tuple(self.grammar.iter_compositions((8, 6)))
        expected = {counts for counts in self.expected if tuple(z for z, _ in counts) == (6, 8)}
        self.assertEqual({composition.counts for composition in generated}, expected)
        self.assertTrue(all(composition.support == (6, 8) for composition in generated))
        self.assertEqual(self.grammar.completion_count(prefix), len(expected))

        after_first_two = prefix + (CompositionToken.coefficient(2),)
        self.assertEqual(
            self.grammar.allowed_next(after_first_two),
            (CompositionToken.coefficient(1),),
        )
        with self.assertRaises(CompositionError):
            self.grammar.support_prefix((6, 6))
        with self.assertRaises(CompositionError):
            self.grammar.support_prefix((6, 14))

    def test_unranking_and_rng_injection_are_exact(self) -> None:
        enumerated = tuple(self.grammar.iter_compositions())
        self.assertEqual(
            tuple(self.grammar.composition_at(index) for index in range(len(enumerated))),
            enumerated,
        )
        for index, expected in enumerate(enumerated):
            self.assertEqual(self.grammar.sample_uniform(FixedIndexRng(index)), expected)

        fibre = tuple(self.grammar.iter_compositions((6, 8)))
        for index, expected in enumerate(fibre):
            # A one-shot iterator catches accidental double consumption of S.
            support_iterator = iter((8, 6))
            self.assertEqual(
                self.grammar.sample_uniform(FixedIndexRng(index), support_iterator),
                expected,
            )

        rng_a = random.Random(20260825)
        rng_b = random.Random(20260825)
        draws_a = tuple(self.grammar.sample_uniform(rng_a) for _ in range(50))
        draws_b = tuple(self.grammar.sample_uniform(rng_b) for _ in range(50))
        self.assertEqual(draws_a, draws_b)


if __name__ == "__main__":
    unittest.main()
