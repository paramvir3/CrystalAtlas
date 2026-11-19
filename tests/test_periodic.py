from __future__ import annotations

import unittest

import numpy as np

from crystalatlas.errors import (
    DomainError,
    NumericalError,
    RecoverableGenerationError,
    ResourceLimitError,
)
from crystalatlas.periodic import (
    closest_periodic_image,
    enumerate_pair_images,
    exhaustive_cube_images,
    images_within_cutoff,
    shortest_nonzero_lattice_image,
    smooth_cutoff,
)


class PeriodicGeometryTests(unittest.TestCase):
    def test_pair_iterator_uses_one_geometry_snapshot(self) -> None:
        for mutate_lattice in (False, True):
            coordinates = np.array([[0.0, 0.0, 0.0], [0.25, 0.125, 0.375]])
            lattice = np.eye(3)
            expected = tuple(enumerate_pair_images(coordinates, lattice, 1.1))
            iterator = enumerate_pair_images(coordinates, lattice, 1.1)
            first = next(iterator)
            if mutate_lattice:
                lattice *= 2.0
            else:
                coordinates[1] = [0.5, 0.5, 0.5]
            observed = (first, *iterator)
            self.assertEqual(len(observed), len(expected))
            for (a, b, image), (ea, eb, reference) in zip(observed, expected, strict=True):
                self.assertEqual((a, b, image.shift), (ea, eb, reference.shift))
                np.testing.assert_array_equal(
                    image.cartesian_displacement, reference.cartesian_displacement
                )
                self.assertEqual(image.squared_distance, reference.squared_distance)

    def test_empty_pair_iterator_still_validates_geometry(self) -> None:
        empty = np.empty((0, 3))
        self.assertEqual(tuple(enumerate_pair_images(empty, np.eye(3), 1.0)), ())
        for lattice, cutoff in ((np.zeros((3, 3)), 1.0), (np.eye(3), -1.0)):
            with self.assertRaises(DomainError):
                tuple(enumerate_pair_images(empty, lattice, cutoff))

    def test_nonfinite_later_atom_rejects_before_first_edge(self) -> None:
        iterator = enumerate_pair_images([[0.0, 0.0, 0.0], [np.nan, 0.0, 0.0]], np.eye(3), 1.1)
        with self.assertRaises(DomainError):
            next(iterator)

    def test_ordered_pair_edges_have_reverse_images(self) -> None:
        lattice = np.array([[1.0, 0.0, 0.0], [0.5, 1.0, 0.0], [0.25, 0.25, 1.0]])
        coordinates = np.array([[0.0, 0.0, 0.0], [0.25, 0.125, 0.375]])
        edges = {
            (a, b, image.shift): image
            for a, b, image in enumerate_pair_images(coordinates, lattice, 1.2)
        }
        for (a, b, shift), image in edges.items():
            reverse = edges[(b, a, tuple(-n for n in shift))]
            np.testing.assert_array_equal(
                reverse.cartesian_displacement, -image.cartesian_displacement
            )
            self.assertEqual(reverse.squared_distance, image.squared_distance)

    def test_candidate_budget_boundary_and_failure_classification(self) -> None:
        # Radius one yields the conservative 3 x 3 x 3 integer box.
        images = images_within_cutoff([0.0, 0.0, 0.0], np.eye(3), 1.0, max_search_points=27)
        self.assertEqual(len(images), 1)
        with self.assertRaises(ResourceLimitError) as caught:
            images_within_cutoff([0.0, 0.0, 0.0], np.eye(3), 1.0, max_search_points=26)
        self.assertNotIsInstance(caught.exception, RecoverableGenerationError)

    def test_extreme_anisotropy_refuses_before_iteration(self) -> None:
        with self.assertRaises(ResourceLimitError):
            images_within_cutoff([0.0, 0.0, 0.0], np.diag([1e-20, 1.0, 1e20]), 1.0)

    def test_closest_search_forwards_budget(self) -> None:
        with self.assertRaises(ResourceLimitError):
            closest_periodic_image([0.25, 0.25, 0.25], np.eye(3), max_search_points=1)

    def test_unrepresentable_squared_cutoffs_raise(self) -> None:
        for cutoff in (1e-200, 1e200):
            with self.assertRaises(NumericalError):
                smooth_cutoff(0.0, cutoff)
            with self.assertRaises(NumericalError):
                images_within_cutoff([0.0, 0.0, 0.0], np.eye(3), cutoff)

    def test_smooth_cutoff_does_not_divide_outside_support(self) -> None:
        with np.errstate(all="raise"):
            result = smooth_cutoff([0.0, 1e-200, 1e300], 1e-100)
        np.testing.assert_array_equal(result, [1.0, 0.0, 0.0])
        np.testing.assert_allclose(
            smooth_cutoff([0.0, 0.5, 1.0, 2.0], 1.0), [1.0, 0.125, 0.0, 0.0], rtol=0, atol=0
        )

    def test_nonfinite_lattice_determinant_rejects_before_enumeration(self) -> None:
        with self.assertRaises(NumericalError):
            images_within_cutoff([0.0, 0.0, 0.0], np.eye(3) * 1e200, 1.0)

    def test_image_enumerator_preserves_large_integer_translations(self) -> None:
        for lattice in (
            np.eye(3),
            np.array([[1.0, 0.0, 0.0], [0.5, 1.0, 0.0], [0.0, 0.0, 1.0]]),
        ):
            delta = np.array([0.0, 0.25, 0.0])
            cutoff = 1.1
            original = images_within_cutoff(delta, lattice, cutoff)
            oracle = exhaustive_cube_images(delta, lattice, cutoff, half_width=4)
            self.assertEqual({image.shift for image in original}, oracle)
            for sign in (-1, 1):
                moved = images_within_cutoff([sign * 1e20, 0.25, 0.0], lattice, cutoff)
                self.assertEqual(len(moved), len(original))
                for a, b in zip(original, moved, strict=True):
                    self.assertEqual(
                        b.shift,
                        (a.shift[0] - sign * 100000000000000000000, a.shift[1], a.shift[2]),
                    )
                    np.testing.assert_array_equal(
                        a.cartesian_displacement, b.cartesian_displacement
                    )
                    self.assertEqual(a.squared_distance, b.squared_distance)

    def test_translated_enumerator_retains_cutoff_boundary_policy(self) -> None:
        for include_boundary, expected_count in ((False, 1), (True, 7)):
            images = images_within_cutoff(
                [1e20, 0.0, 0.0], np.eye(3), 1.0, include_boundary=include_boundary
            )
            self.assertEqual(len(images), expected_count)

    def test_large_integral_translation_does_not_overflow_shift(self) -> None:
        for sign in (-1, 1):
            delta = np.array([sign * 1e20, 0.25, 0.0])
            with np.errstate(all="raise"):
                result = closest_periodic_image(delta, np.eye(3))
            self.assertEqual(result.shift, (-sign * 100000000000000000000, 0, 0))
            self.assertEqual(result.squared_distance, 0.0625)
            np.testing.assert_array_equal(result.fractional_displacement, [0.0, 0.25, 0.0])
        collision = closest_periodic_image([1e20, 0.0, 0.0], np.eye(3))
        self.assertEqual(collision.squared_distance, 0.0)
        self.assertEqual(collision.shift[0], -100000000000000000000)

    def test_skew_cell_closest_image_is_integer_translation_invariant(self) -> None:
        lattice = np.array([[1.0, 0.0, 0.0], [0.95, 0.2, 0.0], [0.0, 0.0, 1.0]])
        delta = np.array([0.375, 0.375, 0.125])
        translation = np.array([1024, -2048, 4096])
        original = closest_periodic_image(delta, lattice)
        moved = closest_periodic_image(delta + translation, lattice)
        self.assertEqual(
            moved.shift,
            tuple(a - int(b) for a, b in zip(original.shift, translation, strict=True)),
        )
        np.testing.assert_array_equal(
            moved.cartesian_displacement, original.cartesian_displacement
        )
        self.assertEqual(moved.squared_distance, original.squared_distance)

    def test_certified_enumeration_matches_large_cube_for_skew_cell(self) -> None:
        lattice = np.asarray(
            [[2.0, 0.0, 0.0], [1.8, 0.7, 0.0], [0.4, 0.3, 1.4]],
            dtype=np.float64,
        )
        delta = np.asarray([0.49, -0.47, 0.31])
        cutoff = 2.3
        certified = {item.shift for item in images_within_cutoff(delta, lattice, cutoff)}
        oracle = exhaustive_cube_images(delta, lattice, cutoff, half_width=8)
        self.assertEqual(certified, oracle)

    def test_closest_image_can_differ_from_componentwise_wrap(self) -> None:
        lattice = np.asarray(
            [[1.0, 0.0, 0.0], [0.95, 0.2, 0.0], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )
        delta = np.asarray([0.49, 0.49, 0.0])
        result = closest_periodic_image(delta, lattice)
        componentwise_shift = tuple(int(x) for x in -np.rint(delta).astype(int))
        componentwise = (delta + np.asarray(componentwise_shift)) @ lattice
        self.assertLessEqual(result.squared_distance, float(componentwise @ componentwise))
        self.assertNotEqual(result.shift, componentwise_shift)

    def test_nonzero_periodic_self_images_are_retained(self) -> None:
        lattice = np.eye(3)
        edges = tuple(enumerate_pair_images(np.zeros((1, 3)), lattice, cutoff=1.1))
        shifts = {edge.shift for _, _, edge in edges}
        self.assertEqual(len(shifts), 6)
        self.assertNotIn((0, 0, 0), shifts)
        shortest = shortest_nonzero_lattice_image(lattice)
        self.assertEqual(shortest.squared_distance, 1.0)
        self.assertNotEqual(shortest.shift, (0, 0, 0))

    def test_cutoff_value_and_first_derivative_vanish(self) -> None:
        cutoff = 2.0
        boundary = cutoff * cutoff
        self.assertEqual(float(smooth_cutoff(boundary, cutoff)), 0.0)
        epsilon = 1.0e-6
        left = float(smooth_cutoff(boundary - epsilon, cutoff))
        derivative = (0.0 - left) / epsilon
        self.assertLess(abs(derivative), 1.0e-9)


if __name__ == "__main__":
    unittest.main()
