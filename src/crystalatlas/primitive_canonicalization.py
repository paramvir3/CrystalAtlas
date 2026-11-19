"""Finite proper-basis canonicalization of a primitive coloured crystal.

The objective starts with the largest squared basis-vector length. Any global
minimizer is no worse than the supplied basis. Consequently all minimizing
bases occur in the finite lattice-vector ellipsoid bounded by that supplied
basis. Sorted diagonal Gram entries, off-diagonal entries, and the complete
origin-quotiented coloured motif break the remaining ties. This compares
different Gram matrices as well as automorphisms of one Gram matrix.

This is a numerical canonical section, not exact real arithmetic. Ranking uses
fixed quantization grids, rejects bin-wall ambiguity, and has explicit finite
resource limits. Only determinant-positive unimodular bases are enumerated;
the Cartesian section uses a proper rotation and retains enantiomers.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations, permutations, product
from math import ceil, gcd, prod, sqrt

import numpy as np

from .algebra import IntegerMatrix, unimodular_inverse
from .errors import AmbiguousPreprocessingError, DomainError, InvariantViolation

_MAX_VECTOR_BOX = 500_000
_MAX_UNORIENTED_VECTORS = 128
_MAX_BASIS_TRIPLES = 350_000


@dataclass(frozen=True, slots=True)
class PrimitiveCanonicalPresentation:
    lattice: np.ndarray
    fractional_positions: np.ndarray
    species: tuple[int, ...]
    basis_change: IntegerMatrix
    inverse_basis_change: IntegerMatrix
    cartesian_rotation: np.ndarray
    subtracted_origin: np.ndarray
    permutation: tuple[int, ...]


def _determinant(rows: IntegerMatrix) -> int:
    a, b, c = rows
    return (
        a[0] * (b[1] * c[2] - b[2] * c[1])
        - a[1] * (b[0] * c[2] - b[2] * c[0])
        + a[2] * (b[0] * c[1] - b[1] * c[0])
    )


def _guarded_bins(values: np.ndarray, quantum: float, uncertainty: float) -> tuple[int, ...]:
    scaled = np.asarray(values, dtype=np.float64).reshape(-1) / quantum
    if not np.all(np.isfinite(scaled)) or float(np.max(np.abs(scaled))) >= 2.0**50:
        raise AmbiguousPreprocessingError(
            "primitive canonical ranking exceeds float resolution"
        )
    bins = np.floor(scaled + 0.5)
    if np.any(0.5 - np.abs(scaled - bins) <= uncertainty / quantum):
        raise AmbiguousPreprocessingError(
            "primitive canonical representative intersects a ranking-bin boundary"
        )
    return tuple(int(value) for value in bins)


def canonicalize_primitive_presentation(
    lattice: np.ndarray,
    fractional_positions: np.ndarray,
    species: tuple[int, ...],
    *,
    decimals: int = 12,
    relative_metric_quantum: float | None = None,
) -> PrimitiveCanonicalPresentation:
    """Choose one boundedly certified numerical primitive representative.

    Callers must establish that the input translation lattice is primitive.
    The returned relations are ``L_new = B L Q`` and
    ``F_new = (F B^-1 - s)[permutation] mod 1``, with det(B)=det(Q)=+1.
    No partial vector search or fallback representative is accepted.

    ``relative_metric_quantum=None`` keeps the legacy absolute metric quantum
    ``10**-min(decimals, 10)`` A^2.  A value ``c`` uses ``c * max_i G_ii``
    instead (review B2).  Ranking is then exactly equivariant under scaling by
    powers of two, and the resolution guard cannot trigger:
    ``256 eps ||G||_2 <= 768 eps max_i G_ii < c max_i G_ii / 8`` for every
    admitted ``c >= 1e-11``.  The maximum diagonal entry is computed without
    LAPACK, so the quantum is bit-reproducible across numerical stacks.
    """

    cell = np.asarray(lattice, dtype=np.float64)
    positions = np.mod(np.asarray(fractional_positions, dtype=np.float64), 1.0)
    if (
        cell.shape != (3, 3)
        or not np.all(np.isfinite(cell))
        or np.linalg.det(cell) <= 0.0
        or positions.shape != (len(species), 3)
        or not species
        or not np.all(np.isfinite(positions))
        or not 6 <= decimals <= 15
    ):
        raise DomainError("invalid primitive canonicalization input")
    if relative_metric_quantum is not None and (
        isinstance(relative_metric_quantum, bool)
        or not np.isfinite(relative_metric_quantum)
        or not 1e-11 <= relative_metric_quantum <= 1e-4
    ):
        raise DomainError("relative_metric_quantum must be None or lie in [1e-11, 1e-4]")
    metric = cell @ cell.T
    if relative_metric_quantum is None:
        metric_quantum = 10.0 ** (-min(decimals, 10))
    else:
        metric_quantum = relative_metric_quantum * float(np.max(np.diag(metric)))
    coordinate_quantum = 10.0 ** (-min(decimals, 10))
    eps = np.finfo(np.float64).eps
    metric_error = 256.0 * eps * max(float(np.linalg.norm(metric, ord=2)), 1.0)
    if metric_error >= metric_quantum / 8.0:
        raise AmbiguousPreprocessingError(
            "primitive metric exceeds canonical numerical resolution"
        )
    # The extra two bins include every vector tied under the primary rounded
    # minimax objective, even when its true norm exceeds the supplied maximum.
    radius_squared = float(np.max(np.diag(metric))) + 2.0 * metric_quantum + metric_error
    inverse_metric = np.linalg.inv(metric)
    bounds = tuple(
        ceil(sqrt(radius_squared * max(float(inverse_metric[j, j]), 0.0))) + 1 for j in range(3)
    )
    if prod(2 * value + 1 for value in bounds) > _MAX_VECTOR_BOX:
        raise AmbiguousPreprocessingError(
            "primitive canonical vector enumeration exceeds resource limit"
        )

    vectors: list[tuple[int, tuple[int, int, int]]] = []
    for vector in product(*(range(-bound, bound + 1) for bound in bounds)):
        nonzero = next((value for value in vector if value), 0)
        if nonzero <= 0:  # one representative of each unoriented vector line
            continue
        # If v = g w with integer g > 1, g divides every determinant with
        # v as a row. Such a vector has no unimodular completion. Exclude it
        # exactly before numerical ranking; no potentially winning basis is
        # removed, and irrelevant bin walls cannot reject the whole input.
        if gcd(*vector) != 1:
            continue
        value = np.asarray(vector, dtype=np.float64)
        norm_squared = float(value @ metric @ value)
        error = metric_error * max(float(value @ value), 1.0)
        if norm_squared <= radius_squared + error:
            norm_bin = _guarded_bins(np.array([norm_squared]), metric_quantum, error)[0]
            vectors.append((norm_bin, vector))
    vectors.sort()
    if len(vectors) > _MAX_UNORIENTED_VECTORS:
        raise AmbiguousPreprocessingError(
            "primitive canonical vector set exceeds resource limit"
        )
    triples = len(vectors) * (len(vectors) - 1) * (len(vectors) - 2) // 6
    if triples > _MAX_BASIS_TRIPLES:
        raise AmbiguousPreprocessingError(
            "primitive canonical basis enumeration exceeds resource limit"
        )

    best_metric_key: tuple[int, ...] | None = None
    best_bases: list[IntegerMatrix] = []
    for triple in combinations(vectors, 3):
        diagonal = tuple(item[0] for item in triple)
        primary = (diagonal[-1], *diagonal)
        if best_metric_key is not None and primary > best_metric_key[:4]:
            continue
        unsigned = tuple(item[1] for item in triple)
        if abs(_determinant(unsigned)) != 1:
            continue
        # Row permutations need only be retained inside equal diagonal bins.
        # Sign changes allow either permutation parity while keeping det +1.
        for order in permutations(range(3)):
            if tuple(diagonal[index] for index in order) != diagonal:
                continue
            ordered = tuple(unsigned[index] for index in order)
            for signs in product((-1, 1), repeat=3):
                basis = tuple(
                    tuple(signs[index] * value for value in ordered[index])
                    for index in range(3)
                )
                if _determinant(basis) != 1:
                    continue
                array = np.asarray(basis, dtype=np.float64)
                gram = array @ metric @ array.T
                error = metric_error * max(float(np.linalg.norm(array, ord=2)) ** 2, 1.0)
                off_diagonal = _guarded_bins(gram[np.triu_indices(3, 1)], metric_quantum, error)
                key = (*primary, *off_diagonal)
                if best_metric_key is None or key < best_metric_key:
                    best_metric_key = key
                    best_bases = [basis]
                elif key == best_metric_key:
                    best_bases.append(basis)
    if not best_bases:
        raise InvariantViolation("complete primitive basis enumeration lost the supplied basis")

    best: (
        tuple[
            tuple[tuple[int, tuple[int, ...]], ...],
            IntegerMatrix,
            np.ndarray,
            np.ndarray,
            tuple[int, ...],
        ]
        | None
    ) = None
    least_species = min(species)
    for basis in best_bases:
        inverse = unimodular_inverse(basis)
        transformed = np.mod(positions @ np.asarray(inverse, dtype=np.float64), 1.0)
        coordinate_error = 128.0 * eps * max(float(np.linalg.norm(inverse, ord=1)), 1.0)
        for anchor in range(len(species)):
            if species[anchor] != least_species:
                continue
            shift = transformed[anchor]
            shifted = np.mod(transformed - shift, 1.0)
            keys = tuple(
                (
                    species[index],
                    tuple(
                        value % round(1.0 / coordinate_quantum)
                        for value in _guarded_bins(point, coordinate_quantum, coordinate_error)
                    ),
                )
                for index, point in enumerate(shifted)
            )
            order = tuple(sorted(range(len(species)), key=lambda index: keys[index]))
            key = tuple(keys[index] for index in order)
            candidate = (key, basis, shifted[np.asarray(order)], shift, order)
            if best is None or candidate[0] < best[0]:
                best = candidate
            elif candidate[0] == best[0]:
                # Ranking grids make numerical symmetry ties stable, but do
                # not authorize discarding physically resolved sub-bin data.
                # If the complete key cannot distinguish two outputs beyond
                # floating roundoff, refuse to choose by enumeration order.
                previous_basis = np.asarray(best[1], dtype=np.float64)
                current_basis = np.asarray(basis, dtype=np.float64)
                gram_error = (
                    2.0
                    * metric_error
                    * max(
                        float(np.linalg.norm(previous_basis, ord=2)) ** 2,
                        float(np.linalg.norm(current_basis, ord=2)) ** 2,
                        1.0,
                    )
                )
                previous_inverse = np.asarray(unimodular_inverse(best[1]), dtype=np.float64)
                position_error = 2.0 * max(
                    coordinate_error,
                    128.0 * eps * max(float(np.linalg.norm(previous_inverse, ord=1)), 1.0),
                )
                delta = candidate[2] - best[2]
                delta -= np.rint(delta)
                if not np.allclose(
                    current_basis @ metric @ current_basis.T,
                    previous_basis @ metric @ previous_basis.T,
                    atol=gram_error,
                    rtol=0.0,
                ) or np.any(np.abs(delta) > position_error):
                    raise AmbiguousPreprocessingError(
                        "primitive canonical key ties between resolved distinct representatives"
                    )
    assert best is not None
    _, basis, final_positions, shift, order = best
    transformed_cell = np.asarray(basis, dtype=np.float64) @ cell
    final_lattice = np.linalg.cholesky(transformed_cell @ transformed_cell.T)
    rotation = np.linalg.solve(transformed_cell, final_lattice)
    if not np.allclose(
        rotation @ rotation.T, np.eye(3), atol=1.0e-10, rtol=0.0
    ) or not np.isclose(np.linalg.det(rotation), 1.0, atol=1.0e-10, rtol=0.0):
        raise InvariantViolation(
            "primitive canonical section did not produce a proper rotation"
        )
    return PrimitiveCanonicalPresentation(
        final_lattice,
        final_positions,
        tuple(species[index] for index in order),
        basis,
        unimodular_inverse(basis),
        rotation,
        shift.copy(),
        order,
    )
