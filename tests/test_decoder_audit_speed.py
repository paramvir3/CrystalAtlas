"""Decoder audits: exact semantics, vectorized (review finding m4).

Every audit asks only whether some periodic image lies within a tiny
tolerance.  If ``|x| <= tol`` for ``x = (f + n) L``, then
``|f_j + n_j| <= tol * sqrt((G^-1)_jj)``; when ``tol * max_j sqrt((G^-1)_jj) <
1/2`` componentwise rounding finds that ``n``.  The vectorized test therefore
reproduces the exhaustive closest-image adjacency exactly, and falls back to
the exhaustive search when the condition fails.
"""

from __future__ import annotations

import time

import numpy as np
import pytest

from crystalatlas.catalogue import CertifiedCatalogue
from crystalatlas.composition import Composition
from crystalatlas.full_catalogue import build_space_group_setting
from crystalatlas.geometry import (
    ContinuousCover,
    ContinuousGeometry,
    IntrinsicCoverLayout,
    OrbitType,
    _has_perfect_matching,
    _minimum_pair_distance,
    _numerical_group_containment,
    _pairs_within_tolerance,
)
from crystalatlas.periodic import closest_periodic_image


def _reference_adjacency(source, target, lattice, tolerance):  # type: ignore[no-untyped-def]
    adjacency = np.zeros((len(source), len(target)), dtype=bool)
    for left, point in enumerate(source):
        for right, candidate in enumerate(target):
            image = closest_periodic_image(point - candidate, lattice)
            adjacency[left, right] = image.squared_distance <= tolerance**2
    return adjacency


def _reference_containment(setting, fractional, numbers, lattice, tolerance):  # type: ignore[no-untyped-def]
    for operation in setting.operations:
        transformed = np.stack([operation.apply_float(point) for point in fractional], axis=0)
        for z in sorted(set(int(value) for value in numbers)):
            source = transformed[numbers == z]
            target = fractional[numbers == z]
            if source.shape != target.shape:
                return False
            adjacency = _reference_adjacency(source, target, lattice, tolerance)
            if not _has_perfect_matching(adjacency):
                return False
    return True


def test_vectorized_threshold_adjacency_equals_exact_closest_images() -> None:
    rng = np.random.default_rng(3)
    skew = np.array([[1.0, 0, 0], [0.97, 0.25, 0], [0.4, 0.3, 0.2]])
    lattices = [np.diag([3.0, 4.0, 5.0]), skew]
    lattices += [np.eye(3) * 4 + rng.standard_normal((3, 3)) for _ in range(4)]
    for lattice in lattices:
        if np.linalg.det(lattice) < 0:
            lattice = lattice[[1, 0, 2]]
        for tolerance in (1e-10, 1e-6, 1e-3):
            target = rng.random((12, 3))
            directions = rng.standard_normal((12, 3))
            directions /= np.linalg.norm(directions, axis=1, keepdims=True)
            scale = tolerance * rng.choice([0.5, 0.999, 1.001, 2.0], size=(12, 1))
            near = target + (directions * scale) @ np.linalg.inv(lattice)
            near += rng.integers(-2, 3, size=near.shape)
            source = np.vstack((near, rng.random((5, 3))))
            expected = _reference_adjacency(source, target, lattice, tolerance)
            vectorized = _pairs_within_tolerance(source, target, lattice, tolerance)
            assert np.array_equal(vectorized, expected)


def _fm3m_forty_atoms() -> tuple[IntrinsicCoverLayout, object]:
    setting = build_space_group_setting(225)
    catalogue = CertifiedCatalogue(
        {setting.key: setting},
        {setting.key: frozenset(setting.wyckoff_positions)},
        certification_scope="test fixture: Fm-3m only",
    )
    cover = ContinuousCover(
        4,
        "Fm-3m",
        Composition(((8, 8), (11, 1), (17, 1))),
        (OrbitType("4a", 11), OrbitType("4b", 17), OrbitType("32f", 8)),
    )
    layout = IntrinsicCoverLayout(cover, catalogue=catalogue)
    geometry = ContinuousGeometry(np.zeros(layout.lattice_dimension), np.array([0.137]))
    return layout, layout.decode(geometry)


def test_group_containment_equals_exhaustive_reference_and_is_fast() -> None:
    layout, decoded = _fm3m_forty_atoms()
    setting = layout.setting
    fractional = np.asarray(decoded.fractional_positions)
    numbers = np.asarray(decoded.atomic_numbers)
    lattice = np.asarray(decoded.lattice)
    started = time.perf_counter()
    contained = _numerical_group_containment(setting, fractional, numbers, lattice, 1e-10)
    elapsed = time.perf_counter() - started
    assert contained is _reference_containment(setting, fractional, numbers, lattice, 1e-10)
    assert contained
    assert elapsed < 2.0
    perturbed = fractional.copy()
    perturbed[5] += np.array([1e-6, 0.0, 0.0])
    assert not _numerical_group_containment(setting, perturbed, numbers, lattice, 1e-10)
    assert not _reference_containment(setting, perturbed, numbers, lattice, 1e-10)


def test_vectorized_minimum_pair_distance_equals_the_exhaustive_minimum() -> None:
    rng = np.random.default_rng(5)
    skew = np.array([[1.0, 0, 0], [0.97, 0.25, 0], [0.4, 0.3, 0.2]])
    lattices = [np.diag([3.0, 4.0, 5.0]), skew]
    lattices += [np.eye(3) * 4 + rng.standard_normal((3, 3)) for _ in range(4)]
    for lattice in lattices:
        if np.linalg.det(lattice) < 0:
            lattice = lattice[[1, 0, 2]]
        for count in (2, 7, 25):
            points = rng.random((count, 3))
            expected = min(
                np.sqrt(closest_periodic_image(points[i] - points[j], lattice).squared_distance)
                for i in range(count)
                for j in range(i)
            )
            # Same minimizing image; summation order may differ in the last ulps.
            assert _minimum_pair_distance(points, lattice) == pytest.approx(expected, rel=1e-12)
