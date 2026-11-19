"""Discarding nonprimitive integer rows preserves the SL(3,Z) search domain.

If g = gcd(v_1, v_2, v_3) > 1, then v = g w with w integral and
det(v, u, z) = g det(w, u, z). No completion of v can have determinant
+1 or -1. Its numerical norm therefore cannot affect the canonical minimizer.
This exact exclusion does not authorize omitting an ambiguous primitive row.
"""

from __future__ import annotations

from itertools import product
from math import gcd

import numpy as np
import pytest

from crystalatlas.errors import AmbiguousPreprocessingError
from crystalatlas.primitive_canonicalization import (
    _determinant,
    _guarded_bins,
    canonicalize_primitive_presentation,
)

_POSITIONS = np.array([[0.123, 0.234, 0.345], [0.372, 0.519, 0.783], [0.693, 0.816, 0.927]])
_SPECIES = (6, 8, 14)
_SHIFTED = np.array([[0.0, 0.0, 0.0], [0.249, 0.285, 0.438], [0.57, 0.582, 0.582]])


@pytest.mark.parametrize("row", [(2, 0, 0), (0, -3, 0), (2, -4, 6), (-3, 3, -6)])
def test_nonprimitive_row_cannot_complete_to_a_unimodular_basis(
    row: tuple[int, int, int],
) -> None:
    divisor = gcd(*row)
    assert divisor > 1
    primitive_row = tuple(value // divisor for value in row)
    completions = tuple(product(range(-1, 2), repeat=3))
    for second, third in product(completions, repeat=2):
        determinant = _determinant((row, second, third))
        assert determinant == divisor * _determinant((primitive_row, second, third))
        assert abs(determinant) != 1


def test_nonprimitive_wall_counterexample_keeps_the_existing_scalar_guard() -> None:
    # The short vector's squared length is safely within its bin, while its
    # irrelevant double is at a bin wall. Do not fix the latter by changing q/e.
    metric = np.diag([1.0 + 1.25e-11, 3.1, 4.5])
    lattice = np.linalg.cholesky(metric)
    actual_metric = lattice @ lattice.T
    allowance = 256.0 * np.finfo(np.float64).eps * np.linalg.norm(actual_metric, ord=2)
    quantum = 1.0e-10
    assert _guarded_bins(np.array([actual_metric[0, 0]]), quantum, allowance)
    with pytest.raises(AmbiguousPreprocessingError, match="ranking-bin boundary"):
        _guarded_bins(np.array([4.0 * actual_metric[0, 0]]), quantum, 4.0 * allowance)


@pytest.mark.parametrize("sheared", [False, True])
@pytest.mark.parametrize("rotated", [False, True])
def test_nonprimitive_bin_wall_does_not_reject_an_unambiguous_minimizer(
    sheared: bool, rotated: bool,
) -> None:
    # Before the exact row filter, v=(2,0,0) (or its rebased coordinates)
    # rejected this input before any basis comparison despite being ineligible.
    lattice = np.diag(np.sqrt([1.0 + 1.25e-11, 3.1, 4.5]))
    basis = np.array([[1, 1, 0], [0, 1, 0], [0, 0, 1]]) if sheared else np.eye(3)
    angle = 0.413
    rotation = (
        np.array(
            [[np.cos(angle), -np.sin(angle), 0], [np.sin(angle), np.cos(angle), 0], [0, 0, 1]]
        )
        if rotated
        else np.eye(3)
    )
    presented_lattice = basis @ lattice @ rotation
    presented_positions = np.mod(_POSITIONS @ np.linalg.inv(basis), 1.0)
    result = canonicalize_primitive_presentation(
        presented_lattice, presented_positions, _SPECIES
    )
    np.testing.assert_allclose(result.lattice, lattice, atol=2.0e-14, rtol=0.0)
    np.testing.assert_allclose(result.fractional_positions, _SHIFTED, atol=2.0e-14, rtol=0.0)
    assert result.species == _SPECIES
    assert result.permutation == (0, 1, 2)
    assert _determinant(result.basis_change) == 1
    np.testing.assert_allclose(
        np.asarray(result.basis_change) @ presented_lattice @ result.cartesian_rotation,
        result.lattice,
        atol=2.0e-14,
        rtol=0.0,
    )
    assert np.linalg.det(result.cartesian_rotation) == pytest.approx(1.0, abs=1.0e-14)


@pytest.mark.parametrize("off_diagonal_wall", [False, True])
def test_relevant_primitive_row_at_a_bin_wall_still_rejects(off_diagonal_wall: bool) -> None:
    metric = np.diag([1.0, 3.1, 4.5])
    if off_diagonal_wall:
        # v=(1,1,0) is primitive, and its norm contains 2 G_12.
        metric[0, 1] = metric[1, 0] = 2.5e-11
    else:
        # v=(1,0,0) is itself indispensable to the short basis.
        metric[0, 0] += 5.0e-11
    with pytest.raises(AmbiguousPreprocessingError, match="ranking-bin boundary"):
        canonicalize_primitive_presentation(np.linalg.cholesky(metric), _POSITIONS, _SPECIES)


@pytest.mark.parametrize("squared_lengths", [(1.0, 3.1, 4.5), (1.0, 1.3, 1.8)])
def test_previously_accepted_orthorhombic_representatives_are_unchanged(
    squared_lengths: tuple[float, float, float],
) -> None:
    # Captured from the unfiltered implementation; the first cell also contains
    # an irrelevant doubled row, whose norm was safely inside a bin.
    lattice = np.diag(np.sqrt(squared_lengths))
    result = canonicalize_primitive_presentation(lattice, _POSITIONS, _SPECIES)
    assert result.basis_change == ((1, 0, 0), (0, 1, 0), (0, 0, 1))
    assert result.inverse_basis_change == result.basis_change
    np.testing.assert_array_equal(result.lattice, lattice)
    np.testing.assert_array_equal(result.cartesian_rotation, np.eye(3))
    np.testing.assert_array_equal(result.subtracted_origin, _POSITIONS[0])
    np.testing.assert_allclose(result.fractional_positions, _SHIFTED, atol=2.0e-16, rtol=0.0)


def test_previously_accepted_skew_representative_is_unchanged() -> None:
    lattice = np.array([[2.0, 0.0, 0.0], [0.6, 2.2, 0.0], [0.3, 0.4, 2.7]])
    result = canonicalize_primitive_presentation(lattice, _POSITIONS, _SPECIES)
    assert result.basis_change == ((1, 0, 0), (0, -1, 0), (0, 0, -1))
    assert result.inverse_basis_change == result.basis_change
    np.testing.assert_array_equal(
        result.lattice, np.array([[2.0, 0.0, 0.0], [-0.6, 2.2, 0.0], [-0.3, 0.4, 2.7]])
    )
    np.testing.assert_array_equal(result.cartesian_rotation, np.diag([1.0, -1.0, -1.0]))
    np.testing.assert_allclose(
        result.subtracted_origin, [0.123, 0.766, 0.655], atol=2.0e-16, rtol=0.0
    )
    np.testing.assert_allclose(
        result.fractional_positions,
        [[0.0, 0.0, 0.0], [0.249, 0.715, 0.562], [0.57, 0.418, 0.418]],
        atol=2.0e-16,
        rtol=0.0,
    )
