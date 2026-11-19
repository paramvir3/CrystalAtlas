"""Input wrapping must not become approximate coordinate idealization."""

from __future__ import annotations

import numpy as np
import pytest

from crystalatlas.standardization import FloatingStructure


@pytest.mark.parametrize("gap", [9e-6, 1e-6, 1e-9, 1e-13, 1e-15, 2.0**-53])
def test_resolved_coordinate_below_one_is_preserved(gap: float) -> None:
    positions = np.array([[1.0 - gap, 0.2, 0.3], [0.1, 0.4, 0.7]])
    raw = positions.copy()
    structure = FloatingStructure(10.0 * np.eye(3), positions, (6, 8))
    np.testing.assert_array_equal(structure.fractional_positions, raw)
    np.testing.assert_array_equal(positions, raw)
    assert not structure.fractional_positions.flags.writeable


@pytest.mark.parametrize("value", [-2.0, -1.0, 0.0, 1.0, 2.0, 17.0])
def test_exact_integer_coordinates_wrap_to_zero(value: float) -> None:
    structure = FloatingStructure(np.eye(3), [[value, value, value]], (6,))
    np.testing.assert_array_equal(structure.fractional_positions, np.zeros((1, 3)))


@pytest.mark.parametrize("value", [-1e-6, -1e-13, -1e-15, -2.0**-53])
def test_resolved_negative_offsets_keep_their_periodic_residue(value: float) -> None:
    positions = np.array([[value, 0.0, 0.0]])
    structure = FloatingStructure(np.eye(3), positions, (6,))
    expected = np.mod(positions, 1.0)
    assert expected[0, 0] < 1.0
    np.testing.assert_array_equal(structure.fractional_positions, expected)


def test_remainder_rounded_to_one_maps_to_the_half_open_domain() -> None:
    # The nearest binary64 result of mod(-min_subnormal, 1) is exactly 1.
    # This *exact endpoint* must wrap; no neighborhood may be snapped to it.
    tiny_negative = np.nextafter(0.0, -1.0)
    assert np.mod(tiny_negative, 1.0) == 1.0
    structure = FloatingStructure(np.eye(3), [[tiny_negative, 0.0, 0.0]], (6,))
    np.testing.assert_array_equal(structure.fractional_positions, np.zeros((1, 3)))


def test_input_wrapping_is_idempotent_and_owns_its_arrays() -> None:
    positions = np.array([[0.999999, -0.000001, 1.234567]])
    lattice = np.diag([2.0, 3.0, 4.0])
    first = FloatingStructure(lattice, positions, (6,))
    second = FloatingStructure(first.lattice, first.fractional_positions, first.species)
    np.testing.assert_array_equal(first.fractional_positions, second.fractional_positions)
    positions[:] = 0.0
    lattice[:] = 0.0
    assert first.fractional_positions[0, 0] == 0.999999
    np.testing.assert_array_equal(first.lattice, np.diag([2.0, 3.0, 4.0]))
