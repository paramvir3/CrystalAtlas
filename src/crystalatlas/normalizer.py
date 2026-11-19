"""Euclidean normalizers of space-group settings for general metrics.

These supply the residual gauge of the intrinsic coordinates (the learned-F10
finding).  A proper normalizer element maps a crystal onto a properly congruent
copy of itself, so physically invariant models cannot separate the two
parameter values it relates.

Row convention, as :meth:`AffineOperation.apply_exact`: ``(R, t)`` acts as
``x -> x R + t``.  Conjugating it by ``g = (W, w)`` gives

    R' = W^-1 R W,        t' = t W + w - w R'   (mod 1).

``g`` normalizes ``G`` when every conjugate lies in ``G``.  Further conditions:
* general metric: ``W M W^T = M`` for every metric ``M`` that the point group
  preserves (``R M R^T = M``), i.e. the linear part is an isometry for every
  lattice of the family, not only special ones;
* proper: ``det W = +1``.  Under the SO(3) chirality quotient an improper element
  maps a crystal to its enantiomorph.  For a centrosymmetric group, an improper
  element times the inversion is proper, so nothing is lost.
  :func:`euclidean_normalizer` keeps the improper elements too (the racemic O(3)
  quotient of normalizer-v3, owner decision 2026-10-25): an equivariant field
  stagnates at achiral configurations, so it cannot move probability between
  enantiomorphic sectors of a chiral cover;
* translations along continuous-origin axes (rows of every R equal to the unit
  vector) are quotiented, matching the layout's origin gauge.

Linear parts are searched over integer matrices with entries in {-1, 0, 1}.
Every lattice holohedry has its conventional-basis matrices there.  Translations
are searched exactly on the grid ``(1/D) Z^3`` with ``D = 24`` by default.
Coset representatives modulo ``G`` and the continuous translations are canonical.
"""

from __future__ import annotations

from collections.abc import Sequence
from fractions import Fraction
from itertools import product

import numpy as np

from .algebra import AffineOperation, FractionMatrix, IntegerMatrix
from .catalogue import SpaceGroupSetting
from .errors import DomainError

_IDENTITY: IntegerMatrix = ((1, 0, 0), (0, 1, 0), (0, 0, 1))
_CACHE: dict[tuple[str, str, int, tuple[int, ...]], tuple[AffineOperation, ...]] = {}


def _int_matmul(left: Sequence[Sequence[int]], right: Sequence[Sequence[int]]) -> IntegerMatrix:
    return tuple(
        tuple(sum(left[i][k] * right[k][j] for k in range(3)) for j in range(3))
        for i in range(3)
    )


def _int_det(m: Sequence[Sequence[int]]) -> int:
    return (
        m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1])
        - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
        + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0])
    )


def _int_inverse_unimodular(m: Sequence[Sequence[int]]) -> IntegerMatrix:
    det = _int_det(m)
    if det not in (-1, 1):
        raise DomainError("matrix is not unimodular")
    cofactor = [[0] * 3 for _ in range(3)]
    for i in range(3):
        for j in range(3):
            minor = [[m[r][c] for c in range(3) if c != j] for r in range(3) if r != i]
            cofactor[i][j] = (-1) ** (i + j) * (
                minor[0][0] * minor[1][1] - minor[0][1] * minor[1][0]
            )
    return tuple(tuple(cofactor[j][i] * det for j in range(3)) for i in range(3))


def _transpose(m: Sequence[Sequence[object]]) -> tuple[tuple[object, ...], ...]:
    return tuple(tuple(m[j][i] for j in range(3)) for i in range(3))


def invariant_metric_basis(rotations: Sequence[IntegerMatrix]) -> tuple[FractionMatrix, ...]:
    """Exact basis of the symmetric ``M`` with ``R M R^T = M`` for every ``R``."""

    index = {(0, 0): 0, (0, 1): 1, (0, 2): 2, (1, 1): 3, (1, 2): 4, (2, 2): 5}

    def unknown(i: int, j: int) -> int:
        return index[(min(i, j), max(i, j))]

    rows: list[list[Fraction]] = []
    for rotation in rotations:
        for i in range(3):
            for j in range(i, 3):
                # (R M R^T)_{ij} - M_{ij} = sum_{k,l} R_ik R_jl M_kl - M_ij
                row = [Fraction(0)] * 6
                for k in range(3):
                    for el in range(3):
                        row[unknown(k, el)] += rotation[i][k] * rotation[j][el]
                row[unknown(i, j)] -= 1
                rows.append(row)
    # Exact reduced row echelon form, then nullspace.
    pivots: list[int] = []
    matrix = [row[:] for row in rows]
    rank = 0
    for column in range(6):
        pivot = next((r for r in range(rank, len(matrix)) if matrix[r][column] != 0), None)
        if pivot is None:
            continue
        matrix[rank], matrix[pivot] = matrix[pivot], matrix[rank]
        lead = matrix[rank][column]
        matrix[rank] = [value / lead for value in matrix[rank]]
        for r in range(len(matrix)):
            if r != rank and matrix[r][column] != 0:
                factor = matrix[r][column]
                matrix[r] = [
                    a - factor * b for a, b in zip(matrix[r], matrix[rank], strict=True)
                ]
        pivots.append(column)
        rank += 1
    free = [column for column in range(6) if column not in pivots]
    basis: list[FractionMatrix] = []
    for column in free:
        vector = [Fraction(0)] * 6
        vector[column] = Fraction(1)
        for r, pivot in enumerate(pivots):
            vector[pivot] = -matrix[r][column]
        basis.append(tuple(tuple(vector[unknown(i, j)] for j in range(3)) for i in range(3)))
    return tuple(basis)


def _preserves_metrics(w: IntegerMatrix, metrics: Sequence[FractionMatrix]) -> bool:
    wt = _transpose(w)
    for metric in metrics:
        left = tuple(
            tuple(sum(w[i][k] * metric[k][j] for k in range(3)) for j in range(3))
            for i in range(3)
        )
        image = tuple(
            tuple(sum(left[i][k] * wt[k][j] for k in range(3)) for j in range(3))  # type: ignore[operator]
            for i in range(3)
        )
        if image != tuple(tuple(row) for row in metric):
            return False
    return True


def free_origin_axes(setting: SpaceGroupSetting) -> tuple[int, ...]:
    """Axes along which every operation fixes the coordinate (continuous origin)."""

    return tuple(
        axis
        for axis in range(3)
        if all(
            tuple(rotation[axis]) == tuple(int(axis == column) for column in range(3))
            for rotation in setting.point_group_rotations
        )
    )


def compose(outer: AffineOperation, inner: AffineOperation) -> AffineOperation:
    """``outer o inner``: apply ``inner`` first (row convention)."""

    rotation = _int_matmul(inner.rotation, outer.rotation)
    translation = tuple(
        sum((inner.translation[k] * outer.rotation[k][j] for k in range(3)), Fraction(0))
        + outer.translation[j]
        for j in range(3)
    )
    return AffineOperation(rotation, translation)


def _conjugate(g: AffineOperation, operation: AffineOperation) -> AffineOperation:
    w_inverse = _int_inverse_unimodular(g.rotation)
    rotation = _int_matmul(_int_matmul(w_inverse, operation.rotation), g.rotation)
    translation = tuple(
        sum((operation.translation[k] * g.rotation[k][j] for k in range(3)), Fraction(0))
        + g.translation[j]
        - sum((g.translation[k] * rotation[k][j] for k in range(3)), Fraction(0))
        for j in range(3)
    )
    return AffineOperation(rotation, translation)


def normalizes(setting: SpaceGroupSetting, g: AffineOperation) -> bool:
    """Exactly: ``g G g^-1 = G`` (as operations modulo lattice translations)."""

    members = {(op.rotation, op.translation) for op in setting.operations}
    for operation in setting.operations:
        image = _conjugate(g, operation)
        if (image.rotation, image.translation) not in members:
            return False
    return True


def _coset(
    g: AffineOperation, setting: SpaceGroupSetting, free_axes: tuple[int, ...]
) -> list[AffineOperation]:
    """``g G`` with continuous-origin translation components set to zero."""

    members = []
    for operation in setting.operations:
        item = compose(g, operation)
        translation = tuple(
            Fraction(0) if axis in free_axes else value
            for axis, value in enumerate(item.translation)
        )
        members.append(AffineOperation(item.rotation, translation))
    return members


def _representative(coset: Sequence[AffineOperation]) -> AffineOperation:
    """Prefer a pure translation (so the identity coset is ``(I, 0)``)."""

    return min(
        coset, key=lambda item: (item.rotation != _IDENTITY, item.rotation, item.translation)
    )


def proper_euclidean_normalizer(
    setting: SpaceGroupSetting, *, translation_denominator: int = 24
) -> tuple[AffineOperation, ...]:
    """Coset representatives of the proper Euclidean normalizer modulo ``G``.

    The continuous-origin translations are quotiented as well.  The identity
    is always included.
    """

    return _euclidean_normalizer(setting, translation_denominator, (1,))


def euclidean_normalizer(
    setting: SpaceGroupSetting, *, translation_denominator: int = 24
) -> tuple[AffineOperation, ...]:
    """Coset representatives of the full Euclidean normalizer (``det W = +-1``).

    This is the racemic (O(3)) quotient of normalizer-v3: an improper element
    maps a crystal onto its enantiomorph.  It equals
    :func:`proper_euclidean_normalizer` unless ``G`` has proper operations only
    and is not one of the 22 enantiomorphic types; then it has twice as many
    cosets.  Every proper coset keeps its representative.
    """

    return _euclidean_normalizer(setting, translation_denominator, (1, -1))


_UNIMODULAR: list[tuple[IntegerMatrix, int]] = []


def _unimodular_candidates() -> list[tuple[IntegerMatrix, int]]:
    """Integer matrices with entries in {-1, 0, 1} and ``det = +-1`` (computed once)."""

    if not _UNIMODULAR:
        for entries in product((-1, 0, 1), repeat=9):
            w = (entries[0:3], entries[3:6], entries[6:9])
            determinant = _int_det(w)
            if determinant in (-1, 1):
                _UNIMODULAR.append((w, determinant))
    return _UNIMODULAR


def _euclidean_normalizer(
    setting: SpaceGroupSetting, translation_denominator: int, determinants: tuple[int, ...]
) -> tuple[AffineOperation, ...]:
    denominator = int(translation_denominator)
    if denominator < 1:
        raise DomainError("translation_denominator must be positive")
    key = (setting.key, setting.hall_symbol, denominator, determinants)
    cached = _CACHE.get(key)
    if cached is not None:
        return cached
    rotations = tuple(tuple(map(tuple, r)) for r in setting.point_group_rotations)
    rotation_set = set(rotations)
    metrics = invariant_metric_basis(rotations)
    free_axes = free_origin_axes(setting)
    # Translations of G by rotation, scaled to integers mod D.
    tables: dict[IntegerMatrix, np.ndarray] = {}
    for operation in setting.operations:
        scaled = []
        for value in operation.translation:
            product_value = value * denominator
            if product_value.denominator != 1:
                raise DomainError("a group translation is not on the (1/D) grid")
            scaled.append(int(product_value) % denominator)
        table = tables.setdefault(operation.rotation, np.zeros((denominator,) * 3, dtype=bool))
        table[tuple(scaled)] = True
    axes = [
        np.zeros(1, dtype=np.int64) if axis in free_axes else np.arange(denominator)
        for axis in range(3)
    ]
    grid = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, 3)
    survivors: list[AffineOperation] = []
    for w, determinant in _unimodular_candidates():
        if determinant not in determinants or not _preserves_metrics(w, metrics):
            continue
        w_inverse = _int_inverse_unimodular(w)
        conjugated = [_int_matmul(_int_matmul(w_inverse, r), w) for r in rotations]
        if any(item not in rotation_set for item in conjugated):
            continue
        keep = np.ones(len(grid), dtype=bool)
        w_array = np.asarray(w, dtype=np.int64)
        for operation in setting.operations:
            r_prime = _int_matmul(_int_matmul(w_inverse, operation.rotation), w)
            t_scaled = np.asarray(
                [int(value * denominator) for value in operation.translation], dtype=np.int64
            )
            shift = t_scaled @ w_array
            moved = (
                shift[None, :] + grid @ (np.eye(3, dtype=np.int64) - np.asarray(r_prime))
            ) % (denominator)
            table = tables[r_prime]
            keep &= table[moved[:, 0], moved[:, 1], moved[:, 2]]
            if not keep.any():
                break
        for point in grid[keep]:
            survivors.append(
                AffineOperation(w, tuple(Fraction(int(v), denominator) for v in point))
            )
    seen: set[tuple[object, object]] = set()
    representatives: list[AffineOperation] = []
    for item in survivors:
        zeroed = AffineOperation(
            item.rotation,
            tuple(
                Fraction(0) if axis in free_axes else value
                for axis, value in enumerate(item.translation)
            ),
        )
        if (zeroed.rotation, zeroed.translation) in seen:
            continue
        if not normalizes(setting, zeroed):  # exact confirmation of the vectorized test
            raise AssertionError(
                "vectorized normalizer test accepted a non-normalizing element"
            )
        coset = _coset(zeroed, setting, free_axes)
        seen.update((member.rotation, member.translation) for member in coset)
        representatives.append(_representative(coset))
    result = tuple(
        sorted(
            representatives,
            key=lambda item: (item.rotation != _IDENTITY, item.rotation, item.translation),
        )
    )
    if not any(item.rotation == _IDENTITY and not any(item.translation) for item in result):
        raise AssertionError("the identity is missing from the normalizer cosets")
    _CACHE[key] = result
    return result
