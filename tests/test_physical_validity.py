"""Descriptive minimum-distance validity metric (review m3)."""

from __future__ import annotations

import numpy as np
import pytest

from crystalatlas.errors import DomainError
from crystalatlas.periodic import closest_periodic_image, shortest_nonzero_lattice_image
from crystalatlas.physical_validity import (
    distance_validity_summary,
    minimum_interatomic_distance,
)


def test_single_atom_distance_is_the_shortest_lattice_vector() -> None:
    lattice = np.array([[2.0, 0.0, 0.0], [0.0, 3.0, 0.0], [0.0, 0.0, 4.0]])
    assert minimum_interatomic_distance(lattice, [[0.1, 0.2, 0.3]]) == pytest.approx(2.0)


def test_pair_distance_uses_the_closest_periodic_image_in_a_skew_cell() -> None:
    lattice = np.array([[3.0, 0.0, 0.0], [2.9, 0.8, 0.0], [0.4, 0.3, 2.5]])
    rng = np.random.default_rng(1)
    points = rng.random((6, 3))
    expected = min(
        np.sqrt(closest_periodic_image(points[i] - points[j], lattice).squared_distance)
        for i in range(6)
        for j in range(i)
    )
    shortest = np.sqrt(shortest_nonzero_lattice_image(lattice).squared_distance)
    result = minimum_interatomic_distance(lattice, points)
    assert result == pytest.approx(min(expected, shortest), rel=1e-12)


def test_summary_counts_structures_at_or_above_the_threshold() -> None:
    cubic = 3.0 * np.eye(3)
    close = [[0.0, 0.0, 0.0], [0.1, 0.0, 0.0]]  # 0.3 A apart
    far = [[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]]
    summary = distance_validity_summary([(cubic, close), (cubic, far)])
    assert summary.structures == 2 and summary.valid == 1
    assert summary.rate == 0.5
    assert summary.smallest_distance_angstrom == pytest.approx(0.3)
    with pytest.raises(DomainError):
        distance_validity_summary([])
    with pytest.raises(DomainError):
        minimum_interatomic_distance(cubic @ np.diag([1.0, 1.0, -1.0]), far)
