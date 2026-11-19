from __future__ import annotations

import unittest

import numpy as np

from crystalatlas.catalogue import MINIMAL_CATALOGUE
from crystalatlas.lattice import InvariantSPDChart, scale_metric_to_volume


class InvariantSPDChartTests(unittest.TestCase):
    def test_p1_chart_is_complete_and_round_trips(self) -> None:
        chart = InvariantSPDChart.from_setting(MINIMAL_CATALOGUE.get("P1"))
        self.assertEqual(chart.dimension, 6)
        coefficients = np.array([0.3, -0.4, 0.2, 0.1, -0.15, 0.25])
        metric = chart.decode(coefficients)
        np.testing.assert_allclose(
            chart.encode(metric), coefficients, atol=2.0e-13, rtol=2.0e-13
        )
        np.testing.assert_allclose(chart.decode(chart.encode(metric)), metric, atol=2.0e-13)
        self.assertTrue(np.all(np.linalg.eigvalsh(metric) > 0.0))

    def test_independently_constructed_nearly_degenerate_spd_round_trip(self) -> None:
        chart = InvariantSPDChart.from_setting(MINIMAL_CATALOGUE.get("P-1"))
        orthogonal, _ = np.linalg.qr(
            np.array([[1.0, 2.0, 0.5], [0.25, 1.0, 3.0], [2.0, 0.1, 1.0]])
        )
        metric = (orthogonal * np.array([1.0e-8, 0.7, 4.0])) @ orthogonal.T
        reconstructed = chart.decode(chart.encode(metric))
        np.testing.assert_allclose(reconstructed, metric, atol=2.0e-14, rtol=2.0e-9)

    def test_bounded_radial_map_is_bijective(self) -> None:
        chart = InvariantSPDChart.from_setting(MINIMAL_CATALOGUE.get("P1"), bound=3.5)
        latent = np.array([2.0, -1.0, 0.2, 4.0, -0.3, 0.7])
        coefficients = chart.bounded_coefficients(latent)
        self.assertLess(np.linalg.norm(coefficients), 3.5)
        np.testing.assert_allclose(chart.unbounded_latent(coefficients), latent, atol=2.0e-14)

    def test_cholesky_section_obeys_row_metric_convention(self) -> None:
        chart = InvariantSPDChart.from_setting(MINIMAL_CATALOGUE.get("P1"))
        coefficients = np.array([0.1, 0.2, -0.1, 0.05, 0.02, -0.03])
        metric = chart.decode(coefficients)
        lattice = chart.cholesky_lattice(coefficients)
        np.testing.assert_allclose(lattice @ lattice.T, metric, atol=2.0e-14)
        self.assertGreater(np.linalg.det(lattice), 0.0)

    def test_volume_scaling_and_bounded_shortest_vector_certificate(self) -> None:
        reference = scale_metric_to_volume(np.eye(3), target_volume=40.0)
        self.assertAlmostEqual(float(np.sqrt(np.linalg.det(reference))), 40.0)
        chart = InvariantSPDChart(
            (np.eye(3),), reference_metric=reference, bound=2.0
        )
        expected = 40.0 ** (1.0 / 3.0) * np.exp(-1.0)
        self.assertAlmostEqual(
            chart.minimum_lattice_singular_value_bound, expected, places=12
        )
        for seed in range(20):
            latent = np.random.default_rng(seed).standard_normal(chart.dimension)
            lattice = np.linalg.cholesky(chart.decode_latent(latent))
            self.assertGreaterEqual(
                float(np.linalg.svd(lattice, compute_uv=False)[-1]),
                chart.minimum_lattice_singular_value_bound - 1.0e-12,
            )


if __name__ == "__main__":
    unittest.main()


def test_l1_chart_dimension_matches_the_crystal_system_for_all_230_settings() -> None:
    """Proof obligation L1 for every pinned setting, not only P1 and P-1.

    The point-group-invariant metrics form a cone whose dimension is fixed by
    the crystal system: 6/4/3/2/2/2/1 for triclinic through cubic (trigonal
    settings use hexagonal axes).  Each chart must also round-trip its
    reference metric and a nearby invariant metric.
    """

    from crystalatlas.full_catalogue import build_full_catalogue

    def expected(number: int) -> int:
        for upper, dimension in ((2, 6), (15, 4), (74, 3), (142, 2), (194, 2), (230, 1)):
            if number <= upper:
                return dimension
        raise AssertionError(number)

    catalogue = build_full_catalogue()
    assert len(catalogue.settings) == 230
    for setting in catalogue.settings.values():
        chart = InvariantSPDChart.from_setting(setting, bound=2.0)
        assert chart.dimension == expected(setting.international_number), setting.key
        coefficients = np.linspace(0.1, -0.2, chart.dimension)
        metric = chart.decode(coefficients)
        np.testing.assert_allclose(chart.encode(metric), coefficients, atol=1e-10)
        np.testing.assert_allclose(chart.encode(chart.reference_metric), 0.0, atol=1e-12)
