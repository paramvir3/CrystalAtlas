"""Proper affine witnesses for primitive-to-pinned conventional round trips."""

from __future__ import annotations

from dataclasses import replace
from typing import Any

import numpy as np
import pytest

import crystalatlas.standardization as standardization
from crystalatlas.errors import InvariantViolation
from crystalatlas.standardization import (
    FloatingStructure,
    SpglibPolicy,
    SpglibStandardizer,
    _CanonicalMotif,
    _verify_pinned_roundtrip,
)


def _motif(
    positions: np.ndarray,
    species: tuple[int, ...],
    shift: tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> _CanonicalMotif:
    return _CanonicalMotif(
        np.asarray(positions, dtype=np.float64),
        species,
        tuple(range(len(species))),
        np.asarray(shift, dtype=np.float64),
    )


def _polar_case() -> dict[str, Any]:
    lattice = np.array(
        [[4.2, 0.0, 0.0], [-2.1, 2.1 * np.sqrt(3.0), 0.0], [0.0, 0.0, 17.3]]
    )
    # Exact R-centred primitive embedding: det(B)=1/3 and B^-1 is integral.
    embedding = np.array([[2, 1, 1], [-1, 1, 1], [-1, -2, 1]]) / 3.0
    anchors = np.array([[0.07, 0.13, 0.0], [0.19, 0.29, 0.137], [0.23, 0.41, 0.629]])
    centring = np.array([[0, 0, 0], [2, 1, 1], [1, 2, 2]]) / 3.0
    positions = np.mod((anchors[:, None, :] + centring[None, :, :]).reshape(-1, 3), 1.0)
    species = (1, 1, 1, 2, 2, 2, 3, 3, 3)
    # Independently specified physical action: x'=y+.15, y'=x+.01,
    # z'=-z+.17. Do not construct this oracle using the helper's matrix formula.
    expected = np.column_stack(
        (positions[:, 1] + 0.15, positions[:, 0] + 0.01, -positions[:, 2] + 0.17)
    )
    permutation = np.array([8, 0, 5, 2, 6, 1, 4, 7, 3])
    return {
        "conventional": _motif(positions, species),
        "roundtrip": _motif(
            np.mod(expected[permutation], 1.0),
            tuple(species[index] for index in permutation),
            (0.2, 0.4, 0.6),
        ),
        "conventional_lattice": lattice,
        "roundtrip_lattice": lattice.copy(),
        "primitive_lattice": embedding @ lattice,
        "primitive_shift": np.array([0.3, 0.6, 0.9]),
        "transformation": np.array([[1, 1, -2], [2, -1, -1], [-1, -1, -1]]) / 3.0,
        "origin_shift": np.array([0.05, 0.11, 0.17]),
        "tolerance": 2.0e-8,
    }


def _identity_case() -> dict[str, Any]:
    # Distinct colours prevent an improper reflection from hiding in a permutation.
    positions = np.array([[0.11, 0.19, 0.23], [0.29, 0.37, 0.41], [0.47, 0.53, 0.61]])
    motif = _motif(positions, (1, 2, 3))
    return {
        "conventional": motif,
        "roundtrip": motif,
        "conventional_lattice": np.diag([2.0, 3.0, 4.0]),
        "roundtrip_lattice": np.diag([2.0, 3.0, 4.0]),
        "primitive_lattice": np.diag([2.0, 3.0, 4.0]),
        "primitive_shift": np.zeros(3),
        "transformation": np.eye(3),
        "origin_shift": np.zeros(3),
        "tolerance": 2.0e-8,
    }


def test_centered_polar_roundtrip_accepts_proper_map_and_nonzero_origins() -> None:
    case = _polar_case()
    assert np.linalg.det(case["conventional_lattice"]) / np.linalg.det(
        case["primitive_lattice"]
    ) == pytest.approx(3.0)
    _verify_pinned_roundtrip(**case)


def test_roundtrip_accepts_permutation_and_independent_integer_torus_wraps() -> None:
    case = _polar_case()
    roundtrip = case["roundtrip"]
    wraps = np.arange(27).reshape(9, 3) % 5 - 2
    case["roundtrip"] = replace(roundtrip, positions=roundtrip.positions + wraps)
    _verify_pinned_roundtrip(**case)


@pytest.mark.parametrize("wrong", ["origin_shift", "primitive_shift", "roundtrip_shift"])
def test_roundtrip_rejects_wrong_declared_origin_without_fitting_it(wrong: str) -> None:
    case = _polar_case()
    if wrong == "roundtrip_shift":
        motif = case["roundtrip"]
        case["roundtrip"] = replace(motif, shift=motif.shift + np.array([0.0, 0.0, 0.01]))
    else:
        case[wrong] = case[wrong] + np.array([0.0, 0.0, 0.01])
    with pytest.raises(InvariantViolation, match="changed the conventional motif"):
        _verify_pinned_roundtrip(**case)


def test_roundtrip_rejects_changed_motif() -> None:
    case = _polar_case()
    motif = case["roundtrip"]
    positions = motif.positions.copy()
    positions[2, 1] += 1.0e-5
    case["roundtrip"] = replace(motif, positions=positions)
    with pytest.raises(InvariantViolation, match="changed the conventional motif"):
        _verify_pinned_roundtrip(**case)


def test_roundtrip_rejects_changed_species_counts() -> None:
    case = _polar_case()
    motif = case["roundtrip"]
    case["roundtrip"] = replace(motif, encoded_types=(4,) + motif.encoded_types[1:])
    with pytest.raises(InvariantViolation, match="changed the conventional species"):
        _verify_pinned_roundtrip(**case)


def test_roundtrip_rejects_colour_swap_even_with_unchanged_counts() -> None:
    case = _identity_case()
    case["roundtrip"] = replace(case["roundtrip"], encoded_types=(2, 1, 3))
    with pytest.raises(InvariantViolation, match="changed the conventional motif"):
        _verify_pinned_roundtrip(**case)


@pytest.mark.parametrize("reflection", [np.diag([-1.0, 1.0, 1.0]), -np.eye(3)])
def test_roundtrip_rejects_improper_map_even_when_motif_and_metric_match(
    reflection: np.ndarray,
) -> None:
    case = _identity_case()
    motif = case["conventional"]
    case["roundtrip"] = replace(motif, positions=np.mod(motif.positions @ reflection, 1.0))
    case["transformation"] = reflection
    with pytest.raises(InvariantViolation, match="determinant"):
        _verify_pinned_roundtrip(**case)


def test_roundtrip_rejects_integer_unimodular_nonisometric_shear() -> None:
    case = _identity_case()
    shear = np.array([[1.0, 1.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    motif = case["conventional"]
    case["roundtrip"] = replace(motif, positions=np.mod(motif.positions @ shear, 1.0))
    case["transformation"] = shear.T
    with pytest.raises(InvariantViolation, match="not a proper Cartesian isometry"):
        _verify_pinned_roundtrip(**case)


@pytest.mark.parametrize("scale", [1.001, -1.0])
def test_roundtrip_rejects_changed_metric_or_left_handed_cartesian_map(scale: float) -> None:
    case = _identity_case()
    case["roundtrip_lattice"] = case["roundtrip_lattice"] * scale
    with pytest.raises(InvariantViolation, match="not a proper Cartesian isometry"):
        _verify_pinned_roundtrip(**case)


@pytest.mark.parametrize("entry", [0.01, 1.0])
def test_roundtrip_rejects_noninteger_or_nonunimodular_basis(entry: float) -> None:
    case = _identity_case()
    case["transformation"][0, 0] += entry
    with pytest.raises(InvariantViolation):
        _verify_pinned_roundtrip(**case)


@pytest.mark.parametrize(
    ("name", "bad_value"),
    [
        ("transformation", np.zeros((2, 3))),
        ("transformation", np.full((3, 3), np.nan)),
        ("conventional_lattice", np.full((3, 3), np.inf)),
        ("roundtrip_lattice", np.zeros((3, 2))),
        ("primitive_lattice", np.zeros((3, 3))),
        ("primitive_shift", np.zeros((1, 3))),
        ("origin_shift", np.full(3, np.nan)),
    ],
)
def test_roundtrip_rejects_malformed_basis_and_origin_maps(
    name: str, bad_value: np.ndarray,
) -> None:
    case = _identity_case()
    case[name] = bad_value
    with pytest.raises(InvariantViolation):
        _verify_pinned_roundtrip(**case)


def test_roundtrip_requires_a_bijection_not_independent_nearest_neighbours() -> None:
    case = _identity_case()
    tolerance = case["tolerance"]
    case["conventional"] = _motif(np.array([[0, 0, 0], [0.5 * tolerance, 0, 0]]), (1, 1))
    case["roundtrip"] = _motif(np.array([[0, 0, 0], [0.2, 0, 0]]), (1, 1))
    with pytest.raises(InvariantViolation, match="changed the conventional motif"):
        _verify_pinned_roundtrip(**case)


def test_roundtrip_matching_can_reassign_an_earlier_match() -> None:
    case = _identity_case()
    tolerance = case["tolerance"]
    case["conventional"] = _motif(np.array([[0.75 * tolerance, 0, 0], [0, 0, 0]]), (1, 1))
    case["roundtrip"] = _motif(
        np.array([[0.1 * tolerance, 0, 0], [1.5 * tolerance, 0, 0]]), (1, 1)
    )
    _verify_pinned_roundtrip(**case)


@pytest.mark.filterwarnings("ignore:Set OLD_ERROR_HANDLING.*:DeprecationWarning")
def test_ideal_polar_r3m_primitive_roundtrip_uses_proper_normalizer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spglib = pytest.importorskip("spglib")
    policy = SpglibPolicy(required_spglib_version=str(spglib.__version__))
    lattice = np.array(
        [[4.2, 0.0, 0.0], [-2.1, 2.1 * np.sqrt(3.0), 0.0], [0.0, 0.0, 17.3]]
    )
    embedding = np.array([[2, 1, 1], [-1, 1, 1], [-1, -2, 1]]) / 3.0
    # Synthetic, noncentrosymmetric axial motif; no benchmark-file dependency.
    axial = np.array([0.0, 0.137, 0.629, 0.383, 0.871])
    structure = FloatingStructure(
        embedding @ lattice, np.repeat(axial[:, None], 3, axis=1), (1, 2, 2, 3, 3)
    )
    coordinate_equalities: list[bool] = []
    original = standardization._verify_pinned_roundtrip

    def verify(
        conventional: _CanonicalMotif, roundtrip: _CanonicalMotif, **kwargs: Any,
    ) -> None:
        coordinate_equalities.append(bool(np.allclose(
            conventional.positions, roundtrip.positions,
            atol=policy.roundtrip_tolerance, rtol=0.0,
        )))
        original(conventional, roundtrip, **kwargs)

    monkeypatch.setattr(standardization, "_verify_pinned_roundtrip", verify)
    result = SpglibStandardizer({160: 450}, policy)._canonicalize_pinned(structure)
    assert result.international_number == 160
    assert result.hall_number == 450
    assert len(result.primitive_species) == 5
    assert len(result.conventional_species) == 15
    assert coordinate_equalities == [False]


def test_roundtrip_motif_drift_is_typed_only_under_the_dimensioned_policy() -> None:
    """At a large tolerance spglib's re-idealization depends on the presentation.

    mp-1179655 (C2/c) at tau = 1e-2 A: on the origin-re-anchored primitive
    cell, spglib returns the inversion translation (0, 0, 1/2) instead of the
    exact (0.001146, 0.998854, 1/2).  Re-idealizing moved sites by 1.15e-3
    (fractional) under an exactly integral affine map (S2 study, 2026-10-20).
    """

    from crystalatlas.errors import AmbiguousPreprocessingError

    case = _polar_case()
    motif = case["roundtrip"]
    positions = motif.positions.copy()
    positions[2, 1] += 1.0e-3
    case["roundtrip"] = replace(motif, positions=positions)
    with pytest.raises(InvariantViolation, match="changed the conventional motif"):
        _verify_pinned_roundtrip(**case)
    with pytest.raises(AmbiguousPreprocessingError, match=r"not idempotent.*0\.001"):
        _verify_pinned_roundtrip(**case, typed_motif_rejection=True)
    # A consistent round trip is unaffected by the dimensioned policy.
    _verify_pinned_roundtrip(**_polar_case(), typed_motif_rejection=True)
