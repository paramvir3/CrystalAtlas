"""Scale-relative metric quantum for primitive canonical ranking (review finding B2).

The legacy quantum is an absolute 1e-10 A^2.  Its resolution guard
``256 eps ||G|| < quantum / 8`` therefore rejects every cell with
``||G|| >= ~220 A^2`` (sigma_max(L) >= ~14.8 A), and its guard band is a sizable
fraction of a bin, so near-tied metric values often meet a bin wall.  A quantum
proportional to the largest diagonal metric entry makes ranking scale-free.
"""

from __future__ import annotations

import numpy as np
import pytest

from crystalatlas.errors import AmbiguousPreprocessingError, DomainError
from crystalatlas.primitive_canonicalization import canonicalize_primitive_presentation
from crystalatlas.standardization import SpglibPolicy, StandardizationProvenance

_POSITIONS = np.array([[0.0, 0.0, 0.0], [0.31, 0.27, 0.42], [0.66, 0.12, 0.81]])
_SPECIES = (8, 14, 14)


def _triclinic() -> np.ndarray:
    return np.array([[3.1, 0.0, 0.0], [0.7, 4.3, 0.0], [-0.9, 1.1, 5.2]])


def test_relative_quantum_ranking_is_exactly_scale_equivariant() -> None:
    lattice = _triclinic()
    reference = canonicalize_primitive_presentation(
        lattice, _POSITIONS, _SPECIES, relative_metric_quantum=1e-9
    )
    for scale in (0.5, 2.0, 4.0, 8.0):  # powers of two scale floats exactly
        scaled = canonicalize_primitive_presentation(
            scale * lattice, _POSITIONS, _SPECIES, relative_metric_quantum=1e-9
        )
        assert scaled.basis_change == reference.basis_change
        assert np.array_equal(scaled.fractional_positions, reference.fractional_positions)
        np.testing.assert_allclose(scaled.lattice, scale * reference.lattice, rtol=1e-14)


def test_large_cell_rejected_by_legacy_resolution_is_accepted_with_relative_quantum() -> None:
    large = 8.0 * _triclinic()
    assert np.linalg.svd(large, compute_uv=False)[0] > 14.8
    with pytest.raises(AmbiguousPreprocessingError, match="resolution"):
        canonicalize_primitive_presentation(large, _POSITIONS, _SPECIES)
    result = canonicalize_primitive_presentation(
        large, _POSITIONS, _SPECIES, relative_metric_quantum=1e-9
    )
    assert np.linalg.det(result.lattice) > 0.0


def test_relative_quantum_is_validated() -> None:
    for invalid in (0.0, -1e-9, float("nan"), float("inf"), 1e-14, 1e-2):
        with pytest.raises(DomainError):
            canonicalize_primitive_presentation(
                _triclinic(), _POSITIONS, _SPECIES, relative_metric_quantum=invalid
            )


def test_policy_and_provenance_record_the_quantum_only_when_set() -> None:
    assert "relative_metric_quantum" not in SpglibPolicy().fingerprint_payload()
    policy = SpglibPolicy(relative_metric_quantum=1e-9)
    assert policy.fingerprint_payload()["relative_metric_quantum"] == 1e-9
    with pytest.raises(DomainError):
        SpglibPolicy(relative_metric_quantum=0.0)
    common = {
        "backend": "spglib",
        "backend_version": "2.7.0",
        "hall_map_fingerprint": "a" * 64,
        "hall_number": 1,
        "symprec": 1e-5,
        "angle_tolerance": -1.0,
        "canonical_decimals": 12,
        "input_atom_count": 3,
        "primitive_atom_count": 3,
        "conventional_atom_count": 3,
        "stability_symprec_factors": (0.5, 2.0),
        "symmetry_stability_verified": True,
    }
    legacy = StandardizationProvenance(**common)  # type: ignore[arg-type]
    relative = StandardizationProvenance(
        **common, relative_metric_quantum=1e-9  # type: ignore[arg-type]
    )
    assert legacy.fingerprint != relative.fingerprint
