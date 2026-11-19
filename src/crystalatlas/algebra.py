"""Exact affine algebra for row-vector crystallographic coordinates.

The normative action is ``f -> f W + w (mod Z^3)``.  Rotations are integral
unimodular matrices and translations are stored as :class:`fractions.Fraction`.
Nothing in this module rounds a floating-point number into an "exact" value.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from fractions import Fraction
from itertools import combinations
from math import gcd
from numbers import Integral

import numpy as np

from .errors import DomainError, InvariantViolation

FractionVector = tuple[Fraction, ...]
FractionMatrix = tuple[tuple[Fraction, ...], ...]
IntegerMatrix = tuple[tuple[int, ...], ...]


def as_fraction(value: object) -> Fraction:
    """Convert an integer or ``Fraction`` without accepting inexact floats."""

    if isinstance(value, Fraction):
        return value
    if isinstance(value, Integral) and not isinstance(value, bool):
        return Fraction(int(value), 1)
    raise DomainError(
        f"exact crystallographic data require int or Fraction, got {type(value).__name__}"
    )


def fraction_mod_one(value: Fraction | int) -> Fraction:
    """Return the canonical representative in ``[0, 1)`` exactly."""

    q = as_fraction(value)
    return q - (q.numerator // q.denominator)


def as_fraction_vector(
    values: Sequence[object], *, length: int | None = None
) -> FractionVector:
    vector = tuple(as_fraction(value) for value in values)
    if length is not None and len(vector) != length:
        raise DomainError(f"expected vector length {length}, got {len(vector)}")
    return vector


def as_fraction_matrix(
    rows: Sequence[Sequence[object]], *, ncols: int | None = None
) -> FractionMatrix:
    matrix = tuple(as_fraction_vector(row) for row in rows)
    if not matrix:
        if ncols not in (None, 0):
            return ()
        return ()
    width = len(matrix[0])
    if any(len(row) != width for row in matrix):
        raise DomainError("matrix rows have inconsistent lengths")
    if ncols is not None and width != ncols:
        raise DomainError(f"expected {ncols} matrix columns, got {width}")
    return matrix


def as_integer_matrix(
    rows: Sequence[Sequence[object]], *, shape: tuple[int, int] | None = None
) -> IntegerMatrix:
    matrix: list[tuple[int, ...]] = []
    for row in rows:
        converted: list[int] = []
        for value in row:
            if not isinstance(value, Integral) or isinstance(value, bool):
                raise DomainError("an integral crystallographic rotation was required")
            converted.append(int(value))
        matrix.append(tuple(converted))
    result = tuple(matrix)
    if result and any(len(row) != len(result[0]) for row in result):
        raise DomainError("matrix rows have inconsistent lengths")
    if shape is not None:
        actual = (len(result), len(result[0]) if result else 0)
        if actual != shape:
            raise DomainError(f"expected matrix shape {shape}, got {actual}")
    return result


def identity_integer_matrix(size: int) -> IntegerMatrix:
    if size < 0:
        raise DomainError("matrix size cannot be negative")
    return tuple(tuple(int(i == j) for j in range(size)) for i in range(size))


def transpose(matrix: Sequence[Sequence[object]]) -> tuple[tuple[object, ...], ...]:
    if not matrix:
        return ()
    width = len(matrix[0])
    if any(len(row) != width for row in matrix):
        raise DomainError("matrix rows have inconsistent lengths")
    return tuple(tuple(matrix[i][j] for i in range(len(matrix))) for j in range(width))


def exact_matmul(
    left: Sequence[Sequence[Fraction | int]], right: Sequence[Sequence[Fraction | int]]
) -> FractionMatrix:
    if not left or not right:
        raise DomainError("empty matrices cannot be multiplied without explicit shapes")
    inner = len(left[0])
    if any(len(row) != inner for row in left):
        raise DomainError("left matrix rows have inconsistent lengths")
    if len(right) != inner:
        raise DomainError("incompatible exact matrix dimensions")
    out_width = len(right[0])
    if any(len(row) != out_width for row in right):
        raise DomainError("right matrix rows have inconsistent lengths")
    return tuple(
        tuple(
            sum(
                (as_fraction(left[i][k]) * as_fraction(right[k][j]) for k in range(inner)),
                Fraction(0),
            )
            for j in range(out_width)
        )
        for i in range(len(left))
    )


def exact_row_matmul(
    row: Sequence[Fraction | int], matrix: Sequence[Sequence[Fraction | int]]
) -> FractionVector:
    if len(row) != len(matrix):
        raise DomainError("incompatible exact row-vector/matrix dimensions")
    if not matrix:
        return ()
    width = len(matrix[0])
    if any(len(matrix_row) != width for matrix_row in matrix):
        raise DomainError("matrix rows have inconsistent lengths")
    return tuple(
        sum(
            (as_fraction(row[k]) * as_fraction(matrix[k][j]) for k in range(len(row))),
            Fraction(0),
        )
        for j in range(width)
    )


def exact_determinant(matrix: Sequence[Sequence[Fraction | int]]) -> Fraction:
    """Compute a small square determinant by fraction-preserving elimination."""

    n = len(matrix)
    if any(len(row) != n for row in matrix):
        raise DomainError("determinant requires a square matrix")
    if n == 0:
        return Fraction(1)
    work = [list(as_fraction_vector(row, length=n)) for row in matrix]
    determinant = Fraction(1)
    for column in range(n):
        pivot = next((row for row in range(column, n) if work[row][column] != 0), None)
        if pivot is None:
            return Fraction(0)
        if pivot != column:
            work[column], work[pivot] = work[pivot], work[column]
            determinant *= -1
        pivot_value = work[column][column]
        determinant *= pivot_value
        for j in range(column, n):
            work[column][j] /= pivot_value
        for row in range(column + 1, n):
            factor = work[row][column]
            if factor == 0:
                continue
            for j in range(column, n):
                work[row][j] -= factor * work[column][j]
    return determinant


def exact_rank(matrix: Sequence[Sequence[Fraction | int]], *, ncols: int | None = None) -> int:
    if not matrix:
        return 0
    width = len(matrix[0])
    if ncols is not None and width != ncols:
        raise DomainError(f"expected {ncols} columns, got {width}")
    if any(len(row) != width for row in matrix):
        raise DomainError("matrix rows have inconsistent lengths")
    work = [list(as_fraction_vector(row)) for row in matrix]
    rank = 0
    for column in range(width):
        pivot = next((row for row in range(rank, len(work)) if work[row][column] != 0), None)
        if pivot is None:
            continue
        work[rank], work[pivot] = work[pivot], work[rank]
        pivot_value = work[rank][column]
        for j in range(column, width):
            work[rank][j] /= pivot_value
        for row in range(len(work)):
            if row == rank:
                continue
            factor = work[row][column]
            if factor != 0:
                for j in range(column, width):
                    work[row][j] -= factor * work[rank][j]
        rank += 1
        if rank == len(work):
            break
    return rank


def exact_inverse(matrix: Sequence[Sequence[Fraction | int]]) -> FractionMatrix:
    n = len(matrix)
    if n == 0 or any(len(row) != n for row in matrix):
        raise DomainError("inverse requires a nonempty square matrix")
    work = [
        list(as_fraction_vector(row, length=n))
        + [Fraction(int(i == j)) for j in range(n)]
        for i, row in enumerate(matrix)
    ]
    for column in range(n):
        pivot = next((row for row in range(column, n) if work[row][column] != 0), None)
        if pivot is None:
            raise DomainError("matrix is singular")
        work[column], work[pivot] = work[pivot], work[column]
        scale = work[column][column]
        work[column] = [value / scale for value in work[column]]
        for row in range(n):
            if row == column:
                continue
            factor = work[row][column]
            if factor != 0:
                work[row] = [
                    work[row][j] - factor * work[column][j] for j in range(2 * n)
                ]
    return tuple(tuple(work[i][n:]) for i in range(n))


def unimodular_inverse(matrix: Sequence[Sequence[object]]) -> IntegerMatrix:
    integer = as_integer_matrix(matrix)
    determinant = exact_determinant(integer)
    if determinant not in (Fraction(-1), Fraction(1)):
        raise DomainError(f"matrix is not unimodular (determinant {determinant})")
    inverse = exact_inverse(integer)
    if any(value.denominator != 1 for row in inverse for value in row):
        raise InvariantViolation("a unimodular matrix produced a nonintegral inverse")
    return tuple(tuple(int(value) for value in row) for row in inverse)


def gcd_maximal_minors(integer_matrix: Sequence[Sequence[object]]) -> int:
    """Return the gcd of full-row-rank maximal minors.

    A rank-``d`` row lattice in ``Z^n`` is primitive exactly when this gcd is
    one.  This is used to certify an intrinsic period lattice without a
    floating-point search.
    """

    matrix = as_integer_matrix(integer_matrix)
    d = len(matrix)
    if d == 0:
        return 1
    n = len(matrix[0])
    if d > n:
        return 0
    result = 0
    for columns in combinations(range(n), d):
        minor = tuple(tuple(matrix[i][j] for j in columns) for i in range(d))
        result = gcd(result, abs(int(exact_determinant(minor))))
    return result


@dataclass(frozen=True, slots=True)
class AffineOperation:
    """An exact fractional affine operation acting on row coordinates."""

    rotation: IntegerMatrix
    translation: FractionVector = (Fraction(0), Fraction(0), Fraction(0))

    def __post_init__(self) -> None:
        rotation = as_integer_matrix(self.rotation, shape=(3, 3))
        if exact_determinant(rotation) not in (Fraction(-1), Fraction(1)):
            raise DomainError("fractional rotation must be integral unimodular")
        translation = tuple(
            fraction_mod_one(value) for value in as_fraction_vector(self.translation, length=3)
        )
        object.__setattr__(self, "rotation", rotation)
        object.__setattr__(self, "translation", translation)

    @classmethod
    def identity(cls) -> AffineOperation:
        return cls(identity_integer_matrix(3), (Fraction(0),) * 3)

    def apply_exact(self, coordinate: Sequence[Fraction | int]) -> FractionVector:
        transformed = exact_row_matmul(as_fraction_vector(coordinate, length=3), self.rotation)
        return tuple(
            fraction_mod_one(transformed[j] + self.translation[j]) for j in range(3)
        )

    def apply_float(self, coordinate: Sequence[float]) -> np.ndarray:
        array = np.asarray(coordinate, dtype=np.float64)
        if array.shape != (3,) or not np.all(np.isfinite(array)):
            raise DomainError("fractional coordinate must be a finite length-three vector")
        transformed = array @ np.asarray(self.rotation, dtype=np.float64)
        transformed += np.asarray([float(value) for value in self.translation])
        wrapped = np.mod(transformed, 1.0)
        # np.mod rounds tiny negative residues to exactly 1; only that endpoint
        # is 0 on the half-open torus (as in FloatingStructure).
        wrapped[wrapped == 1.0] = 0.0
        return wrapped

    def then(self, right: AffineOperation) -> AffineOperation:
        """Return the product that applies ``self`` and then ``right``.

        With row coordinates, ``(f · self) · right = f · self.then(right)``.
        """

        rotation_fraction = exact_matmul(self.rotation, right.rotation)
        rotation = tuple(tuple(int(value) for value in row) for row in rotation_fraction)
        carried = exact_row_matmul(self.translation, right.rotation)
        translation = tuple(carried[j] + right.translation[j] for j in range(3))
        return AffineOperation(rotation, translation)

    def inverse(self) -> AffineOperation:
        rotation_inverse = unimodular_inverse(self.rotation)
        translated = exact_row_matmul(self.translation, rotation_inverse)
        return AffineOperation(rotation_inverse, tuple(-value for value in translated))

    @property
    def is_pure_translation(self) -> bool:
        return self.rotation == identity_integer_matrix(3)


def validate_finite_affine_group(
    operations: Iterable[AffineOperation],
) -> tuple[AffineOperation, ...]:
    """Validate exact identity, inverse, and closure axioms and return a tuple."""

    group = tuple(operations)
    if not group:
        raise InvariantViolation("an affine operation group cannot be empty")
    if len(set(group)) != len(group):
        raise InvariantViolation("affine operation table contains duplicate operations")
    members = set(group)
    if AffineOperation.identity() not in members:
        raise InvariantViolation("affine operation table has no identity")
    for left in group:
        if left.inverse() not in members:
            raise InvariantViolation("affine operation table is not closed under inversion")
        for right in group:
            if left.then(right) not in members:
                raise InvariantViolation(
                    "affine operation table is not closed under composition"
                )
    return group
