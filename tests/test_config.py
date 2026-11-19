from __future__ import annotations

import unittest

from crystalatlas.config import (
    CrystalAtlasConfig,
    DomainConfig,
    ModelConfig,
    NumericsConfig,
)
from crystalatlas.errors import DomainError


class ConfigTests(unittest.TestCase):
    def test_fingerprint_is_stable(self) -> None:
        first = CrystalAtlasConfig()
        second = CrystalAtlasConfig()
        self.assertEqual(first.fingerprint, second.fingerprint)
        self.assertEqual(len(first.fingerprint), 64)

    def test_dng_vocabulary_is_canonicalized(self) -> None:
        domain = DomainConfig(
            element_vocabulary=(8, 3, 1),
            dng_element_vocabulary=(8, 1),
            max_distinct_elements=2,
            max_raw_atoms=4,
        )
        self.assertEqual(domain.element_vocabulary, (1, 3, 8))
        self.assertEqual(domain.dng_element_vocabulary, (1, 8))

    def test_invalid_domain_fails_closed(self) -> None:
        with self.assertRaises(DomainError):
            DomainConfig(
                element_vocabulary=(1, 8),
                dng_element_vocabulary=(1, 6),
                max_distinct_elements=2,
            )

    def test_every_attention_width_must_match_heads(self) -> None:
        for field_name in (
            "composition_width",
            "selector_width",
            "geometry_scalar_width",
        ):
            with self.subTest(field_name=field_name), self.assertRaises(DomainError):
                ModelConfig(**{field_name: 31, "attention_heads": 8})
        with self.assertRaises(DomainError):
            ModelConfig(
                geometry_backbone="equivariant_local",
                geometry_vector_width=31,
                attention_heads=8,
            )

        # In the dense backbone this is a covariant readout-channel count, not
        # an attention width, so divisibility by the head count is irrelevant.
        self.assertEqual(
            ModelConfig(
                geometry_backbone="invariant_global",
                geometry_vector_width=31,
                attention_heads=8,
            ).geometry_vector_width,
            31,
        )

    def test_geometry_backbone_is_explicit_and_fail_closed(self) -> None:
        self.assertEqual(ModelConfig().geometry_backbone, "invariant_global")
        with self.assertRaises(DomainError):
            ModelConfig(geometry_backbone="non_equivariant_raw")
        with self.assertRaises(DomainError):
            ModelConfig(geometry_reciprocal_mode_radius=0)

    def test_model_integer_and_real_hyperparameters_are_type_strict(self) -> None:
        for field_name, invalid in (
            ("geometry_pair_width", 96.5),
            ("geometry_direct_harmonics", True),
            ("geometry_reciprocal_mode_radius", False),
        ):
            with self.subTest(field_name=field_name), self.assertRaises(DomainError):
                ModelConfig(**{field_name: invalid})
        for invalid_dropout in (True, float("nan"), float("inf")):
            with self.subTest(dropout=invalid_dropout), self.assertRaises(DomainError):
                ModelConfig(dropout=invalid_dropout)
        for updates in (
            {"geometry_heat_scales": (0.04, True, 0.64)},
            {"geometry_heat_scales": [0.04, 0.16, 0.64]},
            {"geometry_include_lattice_frame_carriers": 1},
            {"geometry_normalization_epsilon": float("nan")},
            {"geometry_maximum_lattice_condition_number": 0.5},
            {"geometry_readout_schema": "unversioned"},
        ):
            with self.subTest(updates=updates), self.assertRaises(DomainError):
                ModelConfig(**updates)  # type: ignore[arg-type]

    def test_domain_and_numerics_types_are_strict(self) -> None:
        for updates in (
            {"element_vocabulary": (1, True, 8)},
            {"element_vocabulary": (1, 8.0)},
            {"max_raw_atoms": 40.0},
            {"max_formula_units": True},
            {"metric_radius": True},
        ):
            with self.subTest(updates=updates), self.assertRaises(DomainError):
                DomainConfig(**updates)  # type: ignore[arg-type]
        for updates in (
            {"symmetry_atol": True},
            {"max_reciprocal_modes": 1.5},
            {"max_geometry_workspace_bytes": False},
            {"maximum_tangent_condition_number_float32": 0.5},
        ):
            with self.subTest(updates=updates), self.assertRaises(DomainError):
                NumericsConfig(**updates)  # type: ignore[arg-type]

    def test_architecture_defining_invariant_values_change_fingerprint(self) -> None:
        baseline = CrystalAtlasConfig().fingerprint
        variants = (
            ModelConfig(geometry_heat_scales=(0.05, 0.16, 0.64)),
            ModelConfig(geometry_normalization_epsilon=2.0e-4),
            ModelConfig(geometry_include_lattice_frame_carriers=False),
            ModelConfig(geometry_invariant_feedforward_multiplier=5),
        )
        for model in variants:
            with self.subTest(model=model):
                self.assertNotEqual(CrystalAtlasConfig(model=model).fingerprint, baseline)

    def test_physical_lattice_support_and_image_budgets_are_positive(self) -> None:
        domain = DomainConfig()
        self.assertGreater(domain.reference_atomic_volume_angstrom3, 0.0)
        self.assertGreater(domain.minimum_lattice_singular_value_angstrom, 0.0)
        with self.assertRaises(DomainError):
            DomainConfig(minimum_lattice_singular_value_angstrom=0.0)
        with self.assertRaises(DomainError):
            NumericsConfig(max_dense_pair_interactions=False)


if __name__ == "__main__":
    unittest.main()
