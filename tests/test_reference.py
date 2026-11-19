from __future__ import annotations

import unittest
from fractions import Fraction

import numpy as np

from crystalatlas.composition import Composition
from crystalatlas.config import CrystalAtlasConfig, DomainConfig
from crystalatlas.errors import InvariantViolation
from crystalatlas.pipeline import GenerationMode, GenerationRequest
from crystalatlas.reference import (
    NumpyRandrangeAdapter,
    ReferenceGenerator,
    ReferenceLawConfig,
)


def _small_config() -> CrystalAtlasConfig:
    return CrystalAtlasConfig(
        domain=DomainConfig(
            element_vocabulary=(1, 6, 8, 14),
            dng_element_vocabulary=(1, 6, 8),
            max_distinct_elements=3,
            max_raw_atoms=8,
            max_formula_units=4,
            metric_radius=1.25,
            lattice_chart="ball-v1",  # tests the ball-v1 reference law
            reference_length_angstrom=2.0,
            interaction_cutoff_angstrom=5.0,
        )
    )


class NumpyRandrangeAdapterTests(unittest.TestCase):
    def test_arbitrary_precision_draws_are_reproducible_and_in_range(self) -> None:
        stop = (1 << 137) + 12345
        first = NumpyRandrangeAdapter(np.random.default_rng(91))
        second = NumpyRandrangeAdapter(np.random.default_rng(91))
        draws_a = tuple(first.randrange(stop) for _ in range(20))
        draws_b = tuple(second.randrange(stop) for _ in range(20))
        self.assertEqual(draws_a, draws_b)
        self.assertTrue(all(type(draw) is int and 0 <= draw < stop for draw in draws_a))
        self.assertEqual(first.randrange(1), 0)

    def test_invalid_stop_fails_closed(self) -> None:
        adapter = NumpyRandrangeAdapter(np.random.default_rng(1))
        with self.assertRaises(ValueError):
            adapter.randrange(0)
        with self.assertRaises(TypeError):
            adapter.randrange(True)


class ReferenceGeneratorTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = _small_config()
        self.generator = ReferenceGenerator(
            self.config,
            ReferenceLawConfig(parameter_grid_denominator=3),
        )

    def test_all_three_input_contracts_generate_verified_crystals(self) -> None:
        requests = (
            GenerationRequest(mode=GenerationMode.DNG, samples=3, seed=10),
            GenerationRequest(
                mode=GenerationMode.SPECIES_DNG,
                species=(8, 1),
                samples=3,
                seed=11,
            ),
            GenerationRequest(
                mode=GenerationMode.CSP,
                composition=Composition.from_counts({14: 1, 8: 2}),
                samples=3,
                seed=12,
            ),
        )
        batches = tuple(self.generator.require_samples(request) for request in requests)
        self.assertTrue(all(sample.verification.valid for batch in batches for sample in batch))
        self.assertTrue(
            all(sample.composition.support == (1, 8) for sample in batches[1])
        )
        self.assertTrue(
            all(sample.composition.counts == ((8, 2), (14, 1)) for sample in batches[2])
        )
        self.assertTrue(
            all(
                sample.cover.raw_atom_count <= self.config.domain.max_raw_atoms
                for batch in batches
                for sample in batch
            )
        )

    def test_cover_sampler_exposes_only_exact_nonempty_fibres(self) -> None:
        composition = Composition.from_counts({6: 1, 8: 2})
        fibers = self.generator.cover_sampler.feasible_fibers(composition)
        self.assertTrue(fibers)
        for fiber in fibers:
            self.assertGreater(fiber.number_of_completions, 0)
            self.assertLessEqual(
                fiber.nu * composition.total_atoms,
                self.config.domain.max_raw_atoms,
            )
            for table in fiber.oracle.iter_completions():
                self.assertTrue(fiber.oracle.accepts(table))
                self.assertEqual(
                    fiber.oracle.expanded_counts(table),
                    fiber.oracle.initial_residual,
                )

    def test_seeded_reference_samples_are_bitwise_reproducible(self) -> None:
        request = GenerationRequest(
            mode=GenerationMode.CSP,
            composition=Composition.from_counts({6: 1, 8: 1}),
            samples=8,
            seed=2026,
        )
        self.assertEqual(
            self.generator.require_samples(request),
            self.generator.require_samples(request),
        )

    def test_parameters_are_exact_metric_is_invariant_and_audit_passes(self) -> None:
        request = GenerationRequest(
            mode=GenerationMode.CSP,
            composition=Composition.from_counts({6: 1}),
            samples=20,
            seed=45,
        )
        traces = self.generator.sample(request)
        for trace in traces:
            self.assertTrue(trace.succeeded)
            assert trace.state is not None
            self.assertLess(
                float(np.linalg.norm(trace.state.chart_coefficients)),
                self.config.domain.metric_radius,
            )
        samples = self.generator.require_samples(request)
        for sample in samples:
            self.assertTrue(sample.decoded.audit.passed)
            self.assertEqual(
                len(set(sample.decoded.fractional_positions)),
                len(sample.decoded.fractional_positions),
            )
            for orbit in sample.decoded.scaffold.orbits:
                self.assertTrue(all(isinstance(value, Fraction) for value in orbit.parameter))
                self.assertTrue(
                    all(3 % value.denominator == 0 for value in orbit.parameter)
                )
            for operation in sample.decoded.setting.operations:
                rotation = np.asarray(operation.rotation, dtype=np.float64)
                np.testing.assert_allclose(
                    rotation @ sample.metric @ rotation.T,
                    sample.metric,
                    atol=self.config.numerics.symmetry_atol,
                    rtol=self.config.numerics.symmetry_atol,
                )
            self.assertGreater(float(np.min(np.linalg.eigvalsh(sample.metric))), 0.0)
            metadata = dict(sample.verification.metadata)
            self.assertEqual(metadata["law"], "untrained-uniform-reference")
            self.assertEqual(metadata["physical_prediction"], "false")

    def test_invalid_csp_is_recorded_once_without_retry_or_repair(self) -> None:
        request = GenerationRequest(
            mode=GenerationMode.CSP,
            composition=Composition.from_counts({1: 9, 8: 1}),
            seed=3,
        )
        trace = self.generator.sample(request)[0]
        self.assertFalse(trace.succeeded)
        self.assertEqual(trace.failed_stage, "cover")
        self.assertIsNone(trace.cover)
        with self.assertRaises(InvariantViolation):
            self.generator.require_samples(request)


class ReferenceSupportValidationTests(unittest.TestCase):
    def test_grid_capacity_is_resolved_before_sampling(self) -> None:
        """P1 with 28 atoms needs 28 orbits of 1a; q = 3 gives 27, so q = 5.

        Before review m3 this configuration was refused; it is now resolved to
        the smallest sufficient odd denominator, and the cap is still enforced.
        """

        import crystalatlas.reference as reference

        config = CrystalAtlasConfig(
            domain=DomainConfig(
                element_vocabulary=(1,),
                dng_element_vocabulary=(1,),
                max_distinct_elements=1,
                max_raw_atoms=28,
            )
        )
        law = ReferenceLawConfig(space_groups=("P1",), parameter_grid_denominator=3)
        generator = ReferenceGenerator(config, law)
        self.assertEqual(
            dict(generator.geometry_sampler.parameter_grid_denominators), {("P1", "1a"): 5}
        )
        original = reference.MAX_PARAMETER_GRID_DENOMINATOR
        reference.MAX_PARAMETER_GRID_DENOMINATOR = 3
        try:
            with self.assertRaisesRegex(ValueError, "no odd grid denominator"):
                ReferenceGenerator(config, law)
        finally:
            reference.MAX_PARAMETER_GRID_DENOMINATOR = original


if __name__ == "__main__":
    unittest.main()
