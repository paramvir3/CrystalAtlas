"""Topology certificates must not interpret metric idealization as a basis map."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from crystalatlas.algebra import exact_determinant, exact_inverse, exact_matmul
from crystalatlas.errors import AmbiguousPreprocessingError, InvariantViolation
from crystalatlas.standardization import (
    FloatingStructure,
    SpglibPolicy,
    SpglibStandardizer,
    _integer_basis_witness,
    _primitive_rebasis_before_idealization,
)

_IDENTITY = ((1, 0, 0), (0, 1, 0), (0, 0, 1))
_SHEAR = ((1, 2, -1), (0, 1, 1), (0, 0, 1))
_QUARTER_TURN = ((0, -1, 0), (1, 0, 0), (0, 0, 1))
_CENTERING_EMBEDDINGS = (
    _IDENTITY,
    ((1, 1, 0), (-1, 1, 0), (0, 0, 1)),
    ((1, -1, 0), (0, 1, -1), (1, 1, 1)),
    ((-1, 1, 1), (1, -1, 1), (1, 1, -1)),
)


@pytest.mark.parametrize("embedding", _CENTERING_EMBEDDINGS, ids=("P", "C", "R", "F"))
@pytest.mark.parametrize("rebasis", (_SHEAR, _QUARTER_TURN), ids=("shear", "proper-turn"))
@pytest.mark.parametrize("angle", (0.0, 0.37), ids=("unrotated", "rotated"))
def test_rebasis_recovers_exact_topology_despite_metric_idealization(
    embedding: tuple[tuple[int, ...], ...],
    rebasis: tuple[tuple[int, ...], ...],
    angle: float,
) -> None:
    # K = H U and P^T = C_w^-1 K^-1, with all topology identities exact.
    inverse_witness = ((2, -1, 0), (0, 1, 0), (0, 0, 1))
    pre_embedding = exact_matmul(embedding, rebasis)
    transformation_transpose = exact_matmul(inverse_witness, exact_inverse(pre_embedding))
    assert exact_matmul(exact_inverse(transformation_transpose), inverse_witness) == (
        pre_embedding
    )
    assert exact_matmul(exact_inverse(embedding), pre_embedding) == rebasis
    assert exact_determinant(rebasis) == 1

    # Metric idealization and a common proper Cartesian rotation do not alter
    # the index-J relation L_conv = H L_prim. The previous rounded metric
    # relation could not distinguish this strain from invalid integer topology.
    witness_lattice = np.asarray(((2.0, 0.1, 0.0), (0.0, 3.0, 0.2), (0.0, 0.0, 4.0)))
    raw_lattice = np.asarray(inverse_witness) @ witness_lattice
    pre_conventional = np.asarray(pre_embedding, dtype=np.float64) @ witness_lattice
    transformation = np.asarray(transformation_transpose, dtype=np.float64).T
    np.testing.assert_allclose(
        np.linalg.solve(transformation.T, raw_lattice), pre_conventional, atol=1.0e-13
    )
    rotation = np.asarray(
        ((np.cos(angle), -np.sin(angle), 0.0),
         (np.sin(angle), np.cos(angle), 0.0), (0.0, 0.0, 1.0))
    )
    strain = np.diag((1.003, 0.998, 1.002))
    primitive = np.asarray(rebasis) @ witness_lattice @ strain @ rotation
    conventional = np.asarray(embedding) @ primitive
    result = _primitive_rebasis_before_idealization(
        transformation,
        inverse_witness,
        conventional,
        primitive,
        centering_index=int(exact_determinant(embedding)),
        tolerance=1.0e-10,
    )
    assert result == rebasis


def test_row_convention_uses_inverse_transpose_not_inverse() -> None:
    transformation = np.asarray(((1.0, 0.0, 0.0), (-2.0, 1.0, 0.0), (0.0, 0.0, 1.0)))
    expected = ((1, 2, 0), (0, 1, 0), (0, 0, 1))
    actual = _primitive_rebasis_before_idealization(
        transformation, _IDENTITY, np.eye(3), np.eye(3),
        centering_index=1, tolerance=1.0e-10,
    )
    assert actual == expected
    # Both competing conventions produce proper integral matrices, so a
    # determinant-only test would miss this error.
    wrong_convention = np.linalg.inv(transformation)
    assert np.linalg.det(wrong_convention) == 1.0
    assert not np.array_equal(wrong_convention, np.asarray(expected))


@pytest.mark.parametrize(
    ("transformation", "message"),
    (
        (np.eye(2), "finite 3x3"),
        (np.ones(3), "finite 3x3"),
        (np.diag((np.nan, 1.0, 1.0)), "finite 3x3"),
        (np.diag((np.inf, 1.0, 1.0)), "finite 3x3"),
        (np.diag((0.0, 1.0, 1.0)), "singular"),
        (np.diag((0.8, 1.0, 1.0)), "pre-idealization conventional embedding is not integral"),
        (np.diag((0.5, 1.0, 1.0)), "pre-idealization.*incorrect exact determinant"),
        (np.diag((-1.0, 1.0, 1.0)), "pre-idealization.*incorrect exact determinant"),
    ),
)
def test_invalid_pre_idealization_transformation_is_fatal(
    transformation: np.ndarray, message: str,
) -> None:
    with pytest.raises(InvariantViolation, match=message):
        _primitive_rebasis_before_idealization(
            transformation, _IDENTITY, np.eye(3), np.eye(3),
            centering_index=1, tolerance=1.0e-10,
        )


@pytest.mark.parametrize(
    ("conventional", "primitive", "message"),
    (
        (np.diag((1.25, 1.0, 1.0)), np.eye(3), "idealized.*is not integral"),
        (np.diag((2.0, 1.0, 1.0)), np.eye(3), "idealized.*incorrect exact determinant"),
        (np.eye(3), np.diag((0.0, 1.0, 1.0)), "singular"),
    ),
)
def test_invalid_idealized_embedding_is_fatal(
    conventional: np.ndarray, primitive: np.ndarray, message: str,
) -> None:
    with pytest.raises(InvariantViolation, match=message):
        _primitive_rebasis_before_idealization(
            np.eye(3), _IDENTITY, conventional, primitive,
            centering_index=1, tolerance=1.0e-10,
        )


def test_equal_embedding_indices_do_not_prove_integral_primitive_rebasis() -> None:
    # K and H both have determinant 2, but H^-1 K = diag(1/2, 2, 1)
    # is not an automorphism of the primitive integer lattice.
    with pytest.raises(InvariantViolation, match="primitive rebasing is not integral"):
        _primitive_rebasis_before_idealization(
            np.diag((1.0, 0.5, 1.0)), _IDENTITY, np.diag((2.0, 1.0, 1.0)), np.eye(3),
            centering_index=2, tolerance=1.0e-10,
        )


@pytest.mark.parametrize(
    ("value", "message"),
    (
        (np.eye(2), "finite 3x3"),
        (np.diag((np.nan, 1.0, 1.0)), "finite 3x3"),
        (np.diag((np.inf, 1.0, 1.0)), "finite 3x3"),
        (np.diag((1.01, 1.0, 1.0)), "not integral"),
        (np.diag((0.0, 1.0, 1.0)), "incorrect exact determinant"),
        (np.diag((-1.0, 1.0, 1.0)), "incorrect exact determinant"),
    ),
)
def test_integer_basis_witness_rejects_invalid_certificate(
    value: np.ndarray, message: str,
) -> None:
    with pytest.raises(InvariantViolation, match=message):
        _integer_basis_witness(value, determinant=1, tolerance=1.0e-10, name="test basis")


def test_integer_basis_witness_certifies_python_integers_without_int64_overflow() -> None:
    huge_shear = ((1, 2**70, 0), (0, 1, 0), (0, 0, 1))
    result = _integer_basis_witness(
        np.asarray(huge_shear, dtype=np.float64),
        determinant=1, tolerance=1.0e-10, name="large shear",
    )
    assert result == huge_shear
    assert exact_determinant(result) == 1


class _IdealizingP1Backend:
    """Minimal backend fixture terminating at the raw-frame fidelity boundary."""

    __version__ = "2.7.0"

    def __init__(self, *, rotation: np.ndarray, raw_primitive: np.ndarray) -> None:
        self.dataset = SimpleNamespace(
            number=1, hall_number=1, wyckoffs=("a",), site_symmetry_symbols=("1",),
            crystallographic_orbits=(0,), mapping_to_primitive=(0,),
            std_mapping_to_primitive=(0,), std_types=(1,),
            std_positions=np.zeros((1, 3)), std_lattice=1.001 * np.eye(3),
            primitive_lattice=raw_primitive, std_rotation_matrix=rotation,
            transformation_matrix=np.eye(3), origin_shift=np.zeros(3),
            rotations=np.eye(3, dtype=np.int64)[None],
            translations=np.zeros((1, 3)),
        )

    def get_symmetry_dataset(self, *_args: object, **_kwargs: object) -> SimpleNamespace:
        return self.dataset

    def get_symmetry_from_database(self, _hall_number: int) -> dict[str, np.ndarray]:
        return {"rotations": np.eye(3, dtype=np.int64)[None], "translations": np.zeros((1, 3))}

    def find_primitive(
        self, *_args: object, **_kwargs: object,
    ) -> tuple[np.ndarray, np.ndarray, tuple[int, ...]]:
        return self.dataset.std_lattice, np.zeros((1, 3)), (1,)


@pytest.mark.parametrize(
    ("rotation", "raw_primitive", "error", "message"),
    (
        (np.eye(3), np.eye(3), AmbiguousPreprocessingError, "input-fidelity"),
        (np.diag((-1.0, 1.0, 1.0)), np.eye(3), InvariantViolation, "not a proper rotation"),
        (np.diag((1.1, 1.0, 1.0)), np.eye(3), InvariantViolation, "not a proper rotation"),
        (np.eye(3), 0.99 * np.eye(3), InvariantViolation, "raw-frame.*not integral"),
    ),
)
def test_invalid_backend_certificate_precedes_recoverable_idealization_rejection(
    monkeypatch: pytest.MonkeyPatch,
    rotation: np.ndarray,
    raw_primitive: np.ndarray,
    error: type[Exception],
    message: str,
) -> None:
    backend = _IdealizingP1Backend(rotation=rotation, raw_primitive=raw_primitive)
    monkeypatch.setattr(
        "crystalatlas.standardization.require_module", lambda *_args, **_kwargs: backend
    )
    standardizer = SpglibStandardizer({1: 1}, SpglibPolicy())
    structure = FloatingStructure(np.eye(3), np.zeros((1, 3)), (6,))
    with pytest.raises(error, match=message):
        standardizer._canonicalize_pinned(structure)


# mp-1305082 (Pc) at tau = 1e-2 A, spglib 2.7.0 (S2 study, 2026-10-20).  The
# lattice is pseudo-orthorhombic: c + a is perpendicular to a up to
# eps = -(c + a).x = 1.126e-3 A, so within tau the lattice alone has holohedry
# mmm while the crystal has 2/m.  In spglib's monoclinic orientation (a || x,
# b || y, c in xz) find_primitive's re-search returns the isometric cell
# (a, b, R_z(pi)(c + 2a)), i.e. the pseudo-equivalent basis M = (-a, -b, c + 2a)
# rotated by a proper 2-fold about z.  Its third vector is c + 2 eps x, which is
# not a vector of the conventional lattice, so H = I - (2 eps / a) e_31.
_A, _B, _CZ = 5.52297761, 5.575373, 7.825257398998706
_EPS = 5.524103537166411 - _A
_PSEUDO_CONVENTIONAL = np.asarray(((_A, 0.0, 0.0), (0.0, _B, 0.0), (-_A - _EPS, 0.0, _CZ)))
_RESEARCHED_PRIMITIVE = np.asarray(((_A, 0.0, 0.0), (0.0, _B, 0.0), (-_A + _EPS, 0.0, _CZ)))


def test_pseudo_symmetric_basis_flip_is_isometric_but_not_integer_related() -> None:
    unimodular = np.asarray(((-1, 0, 0), (0, -1, 0), (2, 0, 1)))
    rotation = np.diag((-1.0, -1.0, 1.0))
    np.testing.assert_allclose(
        unimodular @ _PSEUDO_CONVENTIONAL @ rotation, _RESEARCHED_PRIMITIVE, rtol=0, atol=1e-14
    )
    h = np.linalg.solve(_RESEARCHED_PRIMITIVE.T, _PSEUDO_CONVENTIONAL.T).T
    expected = np.eye(3)
    expected[2, 0] = -2.0 * _EPS / _A
    np.testing.assert_allclose(h, expected, rtol=0, atol=1e-14)
    assert 4.07e-4 < 2.0 * _EPS / _A < 4.08e-4


def test_non_integral_primitive_search_is_fatal_only_under_the_legacy_policy() -> None:
    arguments = dict(centering_index=1, tolerance=1.0e-10)
    with pytest.raises(InvariantViolation, match="idealized.*is not integral"):
        _primitive_rebasis_before_idealization(
            np.eye(3), _IDENTITY, _PSEUDO_CONVENTIONAL, _RESEARCHED_PRIMITIVE, **arguments
        )
    with pytest.raises(
        AmbiguousPreprocessingError,
        match=r"primitive search on the idealized conventional cell.*not integer-related"
        r".*0\.000408",
    ):
        _primitive_rebasis_before_idealization(
            np.eye(3), _IDENTITY, _PSEUDO_CONVENTIONAL, _RESEARCHED_PRIMITIVE,
            typed_idealization_rejection=True, **arguments,
        )
    # A consistent pair is unaffected by the typed-rejection policy.
    assert _primitive_rebasis_before_idealization(
        np.eye(3), _IDENTITY, _RESEARCHED_PRIMITIVE, _RESEARCHED_PRIMITIVE,
        typed_idealization_rejection=True, **arguments,
    ) == _IDENTITY
