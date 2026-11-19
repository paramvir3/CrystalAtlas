from __future__ import annotations

import unittest
from fractions import Fraction

import numpy as np

from crystalatlas.algebra import AffineOperation
from crystalatlas.catalogue import MINIMAL_CATALOGUE
from crystalatlas.decoder import (
    CrystalScaffold,
    ExactCrystalDecoder,
    OrbitAssignment,
    exact_group_containment,
)
from crystalatlas.errors import DomainError, InvariantViolation
from crystalatlas.lattice import InvariantSPDChart
from crystalatlas.wyckoff import (
    WyckoffPosition,
    stacked_cartesian_jacobian,
    translation_quotient_adjoint,
    translation_quotient_pullback,
)


class WyckoffMetricTests(unittest.TestCase):
    def test_pre_origin_pullback_matches_direct_coset_sum(self) -> None:
        position = MINIMAL_CATALOGUE.get("P-1").wyckoff("2i")
        metric = np.array([[2.0, 0.3, -0.1], [0.3, 1.5, 0.2], [-0.1, 0.2, 1.1]])
        np.testing.assert_allclose(
            position.pre_origin_pullback(metric),
            position.direct_pre_origin_pullback(metric),
            atol=1.0e-14,
        )
        np.testing.assert_allclose(position.pre_origin_pullback(metric), 2.0 * metric)

    def test_period_lattice_rejects_a_strict_sublattice(self) -> None:
        identity = AffineOperation.identity()
        with self.assertRaises(DomainError):
            WyckoffPosition(
                label="bad",
                multiplicity=1,
                representative_offset=(Fraction(0),) * 3,
                coordinate_matrix=(
                    (Fraction(1), Fraction(0), Fraction(0)),
                    (Fraction(0), Fraction(1), Fraction(0)),
                    (Fraction(0), Fraction(0), Fraction(1)),
                ),
                period_basis=(
                    (Fraction(2), Fraction(0), Fraction(0)),
                    (Fraction(0), Fraction(2), Fraction(0)),
                    (Fraction(0), Fraction(0), Fraction(2)),
                ),
                coset_representatives=(identity,),
                site_symmetry_order=1,
                generic_parameter=(Fraction(1, 7),) * 3,
            )

    def test_translation_quotient_adjoint_recovers_legal_velocity(self) -> None:
        position = MINIMAL_CATALOGUE.get("P-1").wyckoff("2i")
        lattice = np.array([[2.0, 0.0, 0.0], [0.2, 1.4, 0.0], [-0.1, 0.3, 1.1]])
        jacobian = stacked_cartesian_jacobian([position], [np.eye(3)], lattice)
        velocity = np.array([0.4, -0.2, 0.7])
        reference_length = 2.5
        common_translation = np.tile(np.array([1.2, -0.3, 0.8]), 2)
        field = (jacobian / reference_length) @ velocity + common_translation
        recovered = translation_quotient_adjoint(
            jacobian, field, reference_length=reference_length
        )
        np.testing.assert_allclose(recovered, velocity, atol=2.0e-14)
        recovered_in_nanometres = translation_quotient_adjoint(
            0.1 * jacobian,
            field,
            reference_length=0.1 * reference_length,
        )
        np.testing.assert_allclose(recovered_in_nanometres, velocity, atol=2.0e-14)
        eigenvalues = np.linalg.eigvalsh(translation_quotient_pullback(jacobian))
        self.assertTrue(np.all(eigenvalues > 0.0))

    def test_uncancelled_p1_origin_direction_fails_closed(self) -> None:
        position = MINIMAL_CATALOGUE.get("P1").wyckoff("1a")
        jacobian = stacked_cartesian_jacobian([position], [np.eye(3)], np.eye(3))
        with self.assertRaises(InvariantViolation):
            translation_quotient_adjoint(jacobian, np.ones(3))


class ExactDecoderTests(unittest.TestCase):
    def setUp(self) -> None:
        self.decoder = ExactCrystalDecoder()

    def test_p_minus_one_formula_and_group_containment(self) -> None:
        scaffold = CrystalScaffold.from_mapping(
            nu=2,
            space_group="P-1",
            reduced_composition={"C": 1},
            orbits=(
                OrbitAssignment(
                    "2i", "C", (Fraction(1, 7), Fraction(2, 7), Fraction(3, 7))
                ),
            ),
        )
        chart = InvariantSPDChart.from_setting(MINIMAL_CATALOGUE.get("P-1"))
        decoded = self.decoder.decode_from_chart(scaffold, chart, np.zeros(6))
        self.assertTrue(decoded.audit.passed)
        self.assertEqual(dict(decoded.conventional_counts), {"C": 2})
        self.assertEqual(decoded.translation_index, 1)
        self.assertTrue(
            exact_group_containment(
                decoded.setting,
                decoded.fractional_positions,
                decoded.species,
                decoded.metric,
            )
        )

    def test_hidden_translation_is_detected_without_changing_reduced_formula(self) -> None:
        scaffold = CrystalScaffold.from_mapping(
            nu=2,
            space_group="P1",
            reduced_composition={"X": 1},
            orbits=(
                OrbitAssignment("1a", "X", (Fraction(0), Fraction(0), Fraction(0))),
                OrbitAssignment("1a", "X", (Fraction(1, 2), Fraction(0), Fraction(0))),
            ),
        )
        decoded = self.decoder.decode(scaffold, np.diag([2.0, 1.5, 1.0]))
        self.assertEqual(decoded.translation_index, 2)
        self.assertEqual(dict(decoded.counts_after_translation_quotient), {"X": 1})
        self.assertTrue(decoded.audit.reduced_composition_matches)

    def test_formula_mismatch_and_cross_orbit_collision_raise(self) -> None:
        mismatch = CrystalScaffold.from_mapping(
            nu=1,
            space_group="P-1",
            reduced_composition={"C": 1},
            orbits=(
                OrbitAssignment(
                    "2i", "C", (Fraction(1, 7), Fraction(2, 7), Fraction(3, 7))
                ),
            ),
        )
        with self.assertRaises(InvariantViolation):
            self.decoder.decode(mismatch, np.eye(3))

        collision = CrystalScaffold.from_mapping(
            nu=2,
            space_group="P-1",
            reduced_composition={"C": 1},
            orbits=(
                OrbitAssignment("1a", "C", ()),
                OrbitAssignment("1a", "C", ()),
            ),
        )
        with self.assertRaises(InvariantViolation):
            self.decoder.decode(collision, np.eye(3))


if __name__ == "__main__":
    unittest.main()
