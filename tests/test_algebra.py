from __future__ import annotations

import unittest
from fractions import Fraction

from crystalatlas.algebra import AffineOperation, fraction_mod_one, validate_finite_affine_group
from crystalatlas.errors import DomainError, InvariantViolation


class ExactAffineAlgebraTests(unittest.TestCase):
    def test_row_action_composition_and_inverse_are_exact(self) -> None:
        first = AffineOperation(
            ((0, 1, 0), (1, 0, 0), (0, 0, -1)),
            (Fraction(1, 3), Fraction(1, 4), Fraction(1, 5)),
        )
        second = AffineOperation(
            ((-1, 0, 0), (0, 1, 0), (0, 0, -1)),
            (Fraction(1, 7), Fraction(2, 7), Fraction(3, 7)),
        )
        coordinate = (Fraction(2, 11), Fraction(3, 11), Fraction(5, 11))
        sequential = second.apply_exact(first.apply_exact(coordinate))
        self.assertEqual(first.then(second).apply_exact(coordinate), sequential)
        self.assertEqual(first.then(first.inverse()), AffineOperation.identity())
        self.assertEqual(first.inverse().then(first), AffineOperation.identity())

    def test_modulo_one_has_canonical_exact_representative(self) -> None:
        self.assertEqual(fraction_mod_one(Fraction(-7, 3)), Fraction(2, 3))
        self.assertEqual(fraction_mod_one(Fraction(8, 3)), Fraction(2, 3))

    def test_exact_api_refuses_float_rationalization(self) -> None:
        with self.assertRaises(DomainError):
            AffineOperation.identity().apply_exact((0.0, 0, 0))

    def test_group_validator_detects_missing_product(self) -> None:
        inversion = AffineOperation(((-1, 0, 0), (0, -1, 0), (0, 0, -1)))
        validate_finite_affine_group((AffineOperation.identity(), inversion))
        half_translation = AffineOperation(
            ((1, 0, 0), (0, 1, 0), (0, 0, 1)),
            (Fraction(1, 2), Fraction(0), Fraction(0)),
        )
        with self.assertRaises(InvariantViolation):
            validate_finite_affine_group(
                (AffineOperation.identity(), inversion, half_translation)
            )


if __name__ == "__main__":
    unittest.main()
