"""Uniform-geometry-in-true-cover control (review m3)."""

from __future__ import annotations

from fractions import Fraction

import numpy as np

from crystalatlas.catalogue import MINIMAL_CATALOGUE
from crystalatlas.controls import uniform_cover_control_attempts, uniform_geometry_in_cover
from crystalatlas.decoder import CrystalScaffold, OrbitAssignment
from crystalatlas.geometry import ContinuousCover, IntrinsicCoverLayout


def _layout() -> IntrinsicCoverLayout:
    scaffold = CrystalScaffold.from_mapping(
        nu=2,
        space_group="P-1",
        reduced_composition={8: 1},
        orbits=(OrbitAssignment("2i", 8, (Fraction(1, 7), Fraction(2, 7), Fraction(3, 7))),),
    )
    return IntrinsicCoverLayout(
        ContinuousCover.from_exact_scaffold(scaffold),
        catalogue=MINIMAL_CATALOGUE,
        lattice_chart="ball-v1",  # this module tests the ball-v1 control law
    )


def test_control_is_uniform_in_the_chart_ball_and_on_the_torus() -> None:
    layout = _layout()
    rng = np.random.default_rng(5)
    radii = []
    units = []
    for _ in range(4000):
        geometry = uniform_geometry_in_cover(layout, rng)
        coefficients = layout.chart.bounded_coefficients(geometry.lattice_latent)
        radii.append(float(np.linalg.norm(coefficients)) / layout.chart.bound)
        units.append(np.linalg.solve(layout.period_basis.T, geometry.intrinsic_coordinates))
    radii_arr = np.asarray(radii)
    assert np.all(radii_arr < 1.0)
    # Uniform in a d-ball: (r/B)^d is U(0,1); compare its quartiles.
    scaled = radii_arr ** layout.lattice_dimension
    quartiles = np.quantile(scaled, [0.25, 0.5, 0.75])
    np.testing.assert_allclose(quartiles, [0.25, 0.5, 0.75], atol=0.03)
    units_arr = np.asarray(units)
    assert np.all((units_arr >= 0.0) & (units_arr < 1.0))
    np.testing.assert_allclose(units_arr.mean(axis=0), 0.5, atol=0.02)


def test_control_attempts_decode_and_keep_failures_in_place() -> None:
    attempts = uniform_cover_control_attempts(_layout(), samples=8, seed=3)
    assert len(attempts) == 8
    assert all((a.decoded is None) == (a.error is not None) for a in attempts)
    assert any(a.decoded is not None for a in attempts)
