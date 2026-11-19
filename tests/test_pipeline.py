from __future__ import annotations

import unittest

import numpy as np

from crystalatlas.pipeline import (
    GenerationEngine,
    GenerationMode,
    GenerationRequest,
    VerificationResult,
)


class _CompositionSampler:
    def sample_unconditional(self, rng: np.random.Generator) -> int:
        return int(rng.integers(1, 10))

    def sample_given_species(
        self, species: tuple[int, ...], rng: np.random.Generator
    ) -> int:
        return sum(species) + int(rng.integers(0, 2))


class _BuggyCompositionSampler(_CompositionSampler):
    def sample_unconditional(self, rng: np.random.Generator) -> int:
        del rng
        raise RuntimeError("simulated implementation bug")


class _CoverSampler:
    def sample_cover(self, composition: int, rng: np.random.Generator) -> int:
        return composition + int(rng.integers(1, 4))


class _GeometrySampler:
    def sample_geometry(
        self, composition: int, cover: int, rng: np.random.Generator
    ) -> float:
        return float(cover + rng.normal())


class _Decoder:
    def decode(self, cover: int, state: float) -> tuple[int, float]:
        return cover, state


class _Verifier:
    def verify(self, composition: int, crystal: tuple[int, float]) -> VerificationResult:
        return VerificationResult(valid=True, metadata=(("composition", str(composition)),))


class PipelineTests(unittest.TestCase):
    def setUp(self) -> None:
        self.engine = GenerationEngine(
            composition_sampler=_CompositionSampler(),
            cover_sampler=_CoverSampler(),
            geometry_sampler=_GeometrySampler(),
            decoder=_Decoder(),
            verifier=_Verifier(),
        )

    def test_all_three_modes_share_downstream_pipeline(self) -> None:
        dng = self.engine.sample(GenerationRequest(mode=GenerationMode.DNG, seed=3))[0]
        sdng = self.engine.sample(
            GenerationRequest(mode=GenerationMode.SPECIES_DNG, species=(3, 8), seed=3)
        )[0]
        csp = self.engine.sample(
            GenerationRequest(mode=GenerationMode.CSP, composition=11, seed=3)
        )[0]
        self.assertTrue(dng.succeeded and sdng.succeeded and csp.succeeded)

    def test_seeded_sampling_is_reproducible(self) -> None:
        request = GenerationRequest(mode=GenerationMode.DNG, samples=4, seed=42)
        self.assertEqual(self.engine.sample(request), self.engine.sample(request))

    def test_invalid_request_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            GenerationRequest(mode=GenerationMode.DNG, species=(8,))

    def test_dng_receives_neither_a_formula_nor_a_species_set(self) -> None:
        """S3 DNG input provenance (theory 4.9, mutation 11)."""

        with self.assertRaises(ValueError):
            GenerationRequest(mode=GenerationMode.DNG, composition=11)
        with self.assertRaises(ValueError):
            GenerationRequest(mode=GenerationMode.DNG, species=(8,))

    def test_unexpected_implementation_failures_are_not_model_invalidity(self) -> None:
        engine = GenerationEngine(
            composition_sampler=_BuggyCompositionSampler(),
            cover_sampler=_CoverSampler(),
            geometry_sampler=_GeometrySampler(),
            decoder=_Decoder(),
            verifier=_Verifier(),
        )
        with self.assertRaisesRegex(RuntimeError, "implementation bug"):
            engine.sample(GenerationRequest(mode=GenerationMode.DNG, seed=5))


if __name__ == "__main__":
    unittest.main()
