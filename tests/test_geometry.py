from __future__ import annotations

from fractions import Fraction

import numpy as np
import pytest

from crystalatlas.composition import Composition
from crystalatlas.decoder import CrystalScaffold, OrbitAssignment
from crystalatlas.errors import GeneratedSampleError, InvariantViolation
from crystalatlas.geometry import (
    ContinuousCover,
    ContinuousGeometry,
    IntrinsicCoverLayout,
    OrbitType,
)


def test_p1_origin_quotient_roundtrip_and_full_rank_jacobian() -> None:
    scaffold = CrystalScaffold.from_mapping(
        nu=1,
        space_group="P1",
        reduced_composition={8: 1, 14: 1},
        orbits=(
            OrbitAssignment("1a", 8, (Fraction(1, 5), Fraction(2, 5), Fraction(3, 5))),
            OrbitAssignment("1a", 14, (Fraction(4, 5), Fraction(1, 7), Fraction(2, 7))),
        ),
    )
    layout = IntrinsicCoverLayout(ContinuousCover.from_exact_scaffold(scaffold))
    eta = layout.encode_exact_parameters(scaffold)
    decoded = layout.decode(
        ContinuousGeometry(np.zeros(layout.lattice_dimension), eta)
    )
    assert decoded.audit.passed
    assert layout.intrinsic_dimension == 3
    centered = decoded.jacobian_cartesian - decoded.jacobian_cartesian.mean(
        axis=0, keepdims=True
    )
    assert np.linalg.matrix_rank(centered.reshape(-1, 3)) == 3


def test_identical_p1_orbit_relabelling_changes_anchor_but_not_crystal() -> None:
    scaffold = CrystalScaffold.from_mapping(
        nu=3,
        space_group="P1",
        reduced_composition={8: 1},
        orbits=(
            OrbitAssignment("1a", 8, (Fraction(1, 10),) * 3),
            OrbitAssignment("1a", 8, (Fraction(3, 10), Fraction(2, 10), Fraction(4, 10))),
            OrbitAssignment("1a", 8, (Fraction(7, 10), Fraction(6, 10), Fraction(8, 10))),
        ),
    )
    layout = IntrinsicCoverLayout(ContinuousCover.from_exact_scaffold(scaffold))
    eta = layout.encode_exact_parameters(scaffold)
    permutation = (1, 0, 2)
    action = layout.residual_action_matrix(permutation)
    assert abs(round(np.linalg.det(action))) == 1
    original = layout.decode(ContinuousGeometry(np.zeros(6), eta))
    transformed = layout.decode(
        ContinuousGeometry(np.zeros(6), layout.transform_residual_eta(eta, permutation))
    )
    shifted = np.mod(
        original.fractional_positions - original.fractional_positions[1], 1.0
    )
    np.testing.assert_allclose(
        np.asarray(sorted(map(tuple, shifted))),
        np.asarray(sorted(map(tuple, transformed.fractional_positions))),
        atol=1.0e-12,
    )


def test_p_minus_one_containment_and_specialization_failure() -> None:
    cover = ContinuousCover(
        nu=2,
        space_group="P-1",
        composition=Composition(((8, 1),)),
        orbits=(OrbitType("2i", 8),),
    )
    layout = IntrinsicCoverLayout(cover)
    good = layout.decode(
        ContinuousGeometry(
            np.zeros(layout.lattice_dimension),
            np.asarray([0.13, 0.27, 0.41]),
        )
    )
    assert good.audit.selected_group_is_contained
    sign_action = layout.residual_action_matrix((0,), orbit_signs=(-1,))
    np.testing.assert_allclose(sign_action, -np.eye(3))
    inverted = layout.decode(
        ContinuousGeometry(
            np.zeros(layout.lattice_dimension),
            layout.transform_residual_eta(
                np.asarray([0.13, 0.27, 0.41]),
                (0,),
                orbit_signs=(-1,),
            ),
        )
    )
    np.testing.assert_allclose(
        np.asarray(sorted(map(tuple, good.fractional_positions))),
        np.asarray(sorted(map(tuple, inverted.fractional_positions))),
        atol=1.0e-12,
    )
    with pytest.raises(GeneratedSampleError, match="specializes"):
        layout.decode(
            ContinuousGeometry(
                np.zeros(layout.lattice_dimension), np.zeros(layout.intrinsic_dimension)
            )
        )


def test_fixed_row_capacity_is_hard() -> None:
    cover = ContinuousCover(
        nu=2,
        space_group="P-1",
        composition=Composition(((8, 1),)),
        orbits=(OrbitType("1a", 8), OrbitType("1a", 8)),
    )
    with pytest.raises(InvariantViolation, match="cannot be instantiated twice"):
        IntrinsicCoverLayout(cover)


def test_continuous_decoder_reports_hidden_translation_index() -> None:
    cover = ContinuousCover(
        nu=2,
        space_group="P-1",
        composition=Composition(((8, 1),)),
        orbits=(OrbitType("1a", 8), OrbitType("1d", 8)),
    )
    layout = IntrinsicCoverLayout(cover)
    decoded = layout.decode(
        ContinuousGeometry(
            np.zeros(layout.lattice_dimension),
            np.empty(layout.intrinsic_dimension),
        )
    )
    assert decoded.audit.passed
    assert decoded.translation_index == 2
    primitive = decoded.full_primitive_reduction(cartesian_tolerance=1.0e-8)
    assert primitive.translation_index == 2
    assert primitive.atomic_numbers.tolist() == [8]
    assert primitive.counts == {8: 1}
    assert np.isclose(
        np.linalg.det(primitive.lattice) * 2,
        np.linalg.det(decoded.lattice),
        atol=1.0e-10,
        rtol=1.0e-10,
    )


def test_one_atom_cell_audits_periodic_self_distance_and_safe_support() -> None:
    cover = ContinuousCover(
        nu=1,
        space_group="P1",
        composition=Composition(((14, 1),)),
        orbits=(OrbitType("1a", 14),),
    )
    layout = IntrinsicCoverLayout(cover)
    decoded = layout.decode(
        ContinuousGeometry(
            np.zeros(layout.lattice_dimension),
            np.empty(layout.intrinsic_dimension),
        )
    )
    assert np.isfinite(decoded.audit.minimum_distance)
    assert (
        decoded.audit.minimum_distance
        >= layout.minimum_lattice_singular_value - 1.0e-12
    )
    assert layout.periodic_candidate_box_upper_bound(6.0) <= 32_768
    assert layout.periodic_edge_upper_bound(6.0) <= 500_000


def test_overbroad_tiny_cell_support_fails_before_model_construction() -> None:
    cover = ContinuousCover(
        nu=1,
        space_group="P1",
        composition=Composition(((14, 1),)),
        orbits=(OrbitType("1a", 14),),
    )
    with pytest.raises(ValueError, match="too broad"):
        IntrinsicCoverLayout(cover, metric_bound=5.0, lattice_chart="ball-v1")
