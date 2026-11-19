"""Certified tolerance-based periodic structure equivalence.

The strict canonical-partition matcher in :mod:`crystalatlas.canonical_matching`
is exact byte equality and makes no physical-equivalence claim.  This module
supplies the representation-invariant tolerance relation needed for geometry
Match@K, oracle-versus-predicted decompositions, and tiny-overfit gates.

Payloads are canonical ASCII records of a certified primitive cell (see
:func:`encode_physical_structure`).  The matcher is closure free, import free,
and uses only Python integers and IEEE-754 ``+ - * /`` and square root, so a
decision is a bit-reproducible function of the two payloads on every CPython
platform; no libm transcendental is evaluated.

Relation
--------
Both crystals are rescaled to unit volume per atom (volume-scaled comparison).
One crystal serves as the *reference*.  Its *reference bases* are all
right-handed bases whose row lengths attain the lattice's successive minima
(within a relative 1e-9); this set is a lattice invariant and includes every
inequivalent reduced form, such as the 60/60/60 and 90/60/60 degree cells of a
face-centred cubic lattice.  A *lattice correspondence* is an integer basis
``N`` of the other lattice with ``det N = +1`` (proper rotations only, so a
chiral crystal and its mirror image are distinct) whose row lengths lie within
a factor ``1 + length_tolerance`` of those of some reference basis ``b``, in
both directions, and whose pairwise row angles lie within
``angle_tolerance_degrees`` of those of ``b``.  Site displacements are
closest-image distances in the mean of the two normalized metrics.  A
reference-role search succeeds when some correspondence, translation, and
species-preserving bijection has every displacement at most
``site_tolerance``.  The pair is accepted when the search succeeds with either
crystal as the reference.  Apart from the tie cases listed under
*Invariance*, no step depends on the input basis, origin, Cartesian frame,
site order, or which argument is the candidate.

Guarantees
----------
*Soundness*: every accepted pair carries an explicit witness ``(N, t, pi)``
whose displacements were checked against the threshold, using an exact
closest-vector solution.

*Completeness at half tolerance*: translations map one anchor atom of the
least-populated species to every same-species atom, each followed by three
least-squares refinement steps.  The anchor is the atom with the smallest sorted
neighbour-distance signature, a crystal property.  If a witness exists with
every displacement at most ``site_tolerance / 2``, then by the triangle
inequality on the torus the anchor translation moves no displacement by more
than the anchor's own.  So a threshold-feasible bijection exists there, and the
constrained assignment finds it.  Between ``site_tolerance / 2`` and
``site_tolerance`` the search is heuristic, as in other anchor-and-refine
structure matchers.

*Invariance*: decisions are invariant under proper basis change, rotation,
origin shift, site permutation, supercell (through primitive reduction), and
argument order, with three documented exceptions:

- lattices whose successive-minimum vectors tie to within 1e-9 relative
  without being exact symmetry images;
- displacements lying on a threshold to within rounding;
- exact cost ties in an assignment, which can arise between two exactly
  symmetric structures.  There the least-squares refinement follows site
  order, which only matters in the heuristic band.

The relation is reflexive and symmetric but, like every tolerance relation,
not transitive.
"""

from __future__ import annotations

from collections.abc import Sequence
from math import ceil, floor, isfinite, sqrt
from typing import Final, TypeVar

import numpy as np
import numpy.typing as npt

from .diagnostics import (
    BootstrapProtocol,
    DiagnosticEvaluationContract,
    EquivalenceMatcherContract,
    GeometryDecompositionCase,
    GeometryDecompositionDiagnostics,
    evaluate_geometry_decomposition,
)
from .errors import DomainError
from .geometry import primitive_reduce_arrays

LENGTH_TOLERANCE: Final = 0.3
"""Lattice rows must agree in length within a factor ``1.3`` either way."""

SITE_TOLERANCE: Final = 0.5
"""Maximum displacement in units of the cube root of the volume per atom."""

ANGLE_TOLERANCE_DEGREES: Final = 10.0

_COS_ANGLE_TOLERANCE: Final = 0.984807753012208
"""cos(10 degrees), correctly rounded (derived from an 80-digit series)."""

_SIN_ANGLE_TOLERANCE: Final = 0.17364817766693036
"""sin(10 degrees), correctly rounded (derived from an 80-digit series)."""

PRIMITIVE_REDUCTION_TOLERANCE_ANGSTROM: Final = 1.0e-2
"""Cartesian tolerance of the certified primitive reduction in the encoder.

This is a declared evaluation policy, recorded in every payload header and in
the matcher contract.  It is not derived from data.
"""

_PAYLOAD_HEADER: Final = (
    "CRYSTALATLAS-PHYSICAL-STRUCTURE/1 primitive-tolerance-angstrom=0.01"
)
_Number = TypeVar("_Number", int, float)

_LLL_DELTA: Final = 0.99
_MINIMA_RELATIVE_TOLERANCE: Final = 1.0e-9
_REFINEMENT_STEPS: Final = 3
"""Least-squares translation steps after each anchor translation.

Each step reassigns sites at the current translation and moves it by the mean
displacement.  One step missed a witness at 0.45 stol that StructureMatcher
found (MP-20 mp-1185903); three recover it.  Extra steps only add certified
witnesses; they cannot cause a false acceptance.
"""
"""Relative slack identifying lattice vectors that attain a successive minimum."""
_LLL_STEP_LIMIT: Final = 10000
_CUBE_ROOT_GUARD_BITS: Final = 70


def _canonical_float(field: str) -> float:
    value = float(field)
    if not isfinite(value) or repr(value) != field or (value == 0.0 and field != "0.0"):
        raise ValueError(f"non-canonical floating field {field!r} in physical payload")
    return value


def _determinant3(matrix: list[list[float]] | list[list[int]]) -> float | int:
    return (
        matrix[0][0] * (matrix[1][1] * matrix[2][2] - matrix[1][2] * matrix[2][1])
        - matrix[0][1] * (matrix[1][0] * matrix[2][2] - matrix[1][2] * matrix[2][0])
        + matrix[0][2] * (matrix[1][0] * matrix[2][1] - matrix[1][1] * matrix[2][0])
    )


def _adjugate3(matrix: list[list[_Number]]) -> list[list[_Number]]:
    a, b, c = matrix[0]
    d, e, f = matrix[1]
    g, h, i = matrix[2]
    return [
        [e * i - f * h, c * h - b * i, b * f - c * e],
        [f * g - d * i, a * i - c * g, c * d - a * f],
        [d * h - e * g, b * g - a * h, a * e - b * d],
    ]


def _inverse3(matrix: list[list[float]]) -> list[list[float]]:
    determinant = _determinant3(matrix)
    if not determinant > 0.0:
        raise ValueError("metric is not positive definite")
    adjugate = _adjugate3(matrix)
    inverse = []
    for row in adjugate:
        inverse.append([row[0] / determinant, row[1] / determinant, row[2] / determinant])
    return inverse


def _parse_payload(payload: bytes) -> tuple[list[list[float]], list[list[float]], list[int]]:
    if not isinstance(payload, bytes):
        raise TypeError("physical-structure payloads must be canonical bytes")
    lines = payload.decode("ascii").split("\n")
    if len(lines) < 4 or lines[0] != _PAYLOAD_HEADER or lines[-1] != "":
        raise ValueError("payload is not a canonical CrystalAtlas physical-structure record")
    lattice_fields = lines[1].split(" ")
    if len(lattice_fields) != 10 or lattice_fields[0] != "lattice":
        raise ValueError("physical payload lattice record is malformed")
    entries = []
    for field in lattice_fields[1:]:
        entries.append(_canonical_float(field))
    lattice = [entries[0:3], entries[3:6], entries[6:9]]
    if not _determinant3(lattice) > 0.0:
        raise ValueError("physical payload lattice must be right handed")
    species: list[int] = []
    positions: list[list[float]] = []
    previous = None
    for line in lines[2:-1]:
        fields = line.split(" ")
        if len(fields) != 5 or fields[0] != "site":
            raise ValueError("physical payload site record is malformed")
        number = int(fields[1])
        if str(number) != fields[1] or not 1 <= number <= 118:
            raise ValueError("physical payload atomic number is not canonical")
        site = [
            _canonical_float(fields[2]),
            _canonical_float(fields[3]),
            _canonical_float(fields[4]),
        ]
        for value in site:
            if not 0.0 <= value < 1.0:
                raise ValueError("physical payload coordinates must lie in [0, 1)")
        key = (number, site[0], site[1], site[2])
        if previous is not None and not previous < key:
            raise ValueError("physical payload sites must be strictly sorted and distinct")
        previous = key
        species.append(number)
        positions.append(site)
    return lattice, positions, species


def _integer_cube_root(value: int) -> int:
    """Return ``floor(value ** (1/3))`` exactly by integer Newton iteration."""

    if value < 0:
        raise ValueError("integer cube root requires a nonnegative value")
    if value == 0:
        return 0
    root = 1 << ((value.bit_length() + 2) // 3)
    while True:
        smaller = (2 * root + value // (root * root)) // 3
        if smaller >= root:
            return root
        root = smaller


def _cube_root(value: float) -> float:
    """Platform-independent cube root of a positive finite float (error < 1 ulp)."""

    if not (isfinite(value) and value > 0.0):
        raise ValueError("cube root requires a positive finite value")
    numerator, denominator = value.as_integer_ratio()
    shift = _CUBE_ROOT_GUARD_BITS + denominator.bit_length() // 3
    root = _integer_cube_root((numerator << (3 * shift)) // denominator)
    return root / (1 << shift)


def _quadratic(vector: list[float] | list[int], metric: list[list[float]]) -> float:
    x0, x1, x2 = vector
    return (
        x0 * x0 * metric[0][0]
        + x1 * x1 * metric[1][1]
        + x2 * x2 * metric[2][2]
        + 2.0 * (x0 * x1 * metric[0][1] + x0 * x2 * metric[0][2] + x1 * x2 * metric[1][2])
    )


def _bilinear(left: list[int], right: list[int], metric: list[list[float]]) -> float:
    total = 0.0
    for a in range(3):
        for b in range(3):
            total += left[a] * metric[a][b] * right[b]
    return total


def _transform_gram(metric: list[list[float]], rows: list[list[int]]) -> list[list[float]]:
    """Return ``rows @ metric @ rows.T`` with exact symmetry."""

    result = [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]
    for i in range(3):
        for j in range(i, 3):
            value = _bilinear(rows[i], rows[j], metric)
            result[i][j] = value
            result[j][i] = value
    return result


def _gram_schmidt(metric: list[list[float]]) -> tuple[list[list[float]], list[float]]:
    mu = [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]
    norms = [0.0, 0.0, 0.0]
    for i in range(3):
        for j in range(i):
            value = metric[i][j]
            for k in range(j):
                value -= mu[j][k] * mu[i][k] * norms[k]
            mu[i][j] = value / norms[j]
        value = metric[i][i]
        for k in range(i):
            value -= mu[i][k] * mu[i][k] * norms[k]
        if not value > 0.0:
            raise ValueError("lattice metric is numerically singular")
        norms[i] = value
    return mu, norms


def _lll_reduce(metric: list[list[float]]) -> tuple[list[list[int]], list[list[float]]]:
    """Deterministic LLL reduction; returns rows ``U`` (det +1) and ``U G U^T``."""

    basis = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
    current = _transform_gram(metric, basis)
    k = 1
    steps = 0
    while k < 3:
        steps += 1
        if steps > _LLL_STEP_LIMIT:
            raise ValueError("lattice reduction exceeded its step budget")
        for j in range(k - 1, -1, -1):
            mu, _ = _gram_schmidt(current)
            multiple = round(mu[k][j])
            if multiple != 0:
                for a in range(3):
                    basis[k][a] -= multiple * basis[j][a]
                current = _transform_gram(metric, basis)
        mu, norms = _gram_schmidt(current)
        if norms[k] >= (_LLL_DELTA - mu[k][k - 1] * mu[k][k - 1]) * norms[k - 1]:
            k += 1
        else:
            basis[k], basis[k - 1] = basis[k - 1], basis[k]
            current = _transform_gram(metric, basis)
            k = max(k - 1, 1)
    if _determinant3(basis) < 0:
        basis[2] = [-basis[2][0], -basis[2][1], -basis[2][2]]
        current = _transform_gram(metric, basis)
    return basis, current


def _normalized_metric(lattice: list[list[float]], count: int) -> list[list[float]]:
    """Gram matrix rescaled to unit volume per atom."""

    side = _cube_root(_determinant3(lattice) / count)
    scale = side * side
    metric = [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]
    for i in range(3):
        for j in range(i, 3):
            value = (
                lattice[i][0] * lattice[j][0]
                + lattice[i][1] * lattice[j][1]
                + lattice[i][2] * lattice[j][2]
            ) / scale
            metric[i][j] = value
            metric[j][i] = value
    return metric


def _change_basis(positions: list[list[float]], rows: list[list[int]]) -> list[list[float]]:
    """Express fractional rows in the basis ``rows @ L`` (``det rows = +1``)."""

    if _determinant3(rows) != 1:
        raise ValueError("basis change must be unimodular and proper")
    inverse = _adjugate3(rows)
    result = []
    for point in positions:
        mapped = [0.0, 0.0, 0.0]
        for k in range(3):
            mapped[k] = (
                point[0] * inverse[0][k] + point[1] * inverse[1][k] + point[2] * inverse[2][k]
            )
        result.append(mapped)
    return result


def _short_vectors(
    metric: list[list[float]], radius_squared: float
) -> list[tuple[float, tuple[int, int, int]]]:
    """Every nonzero integer row ``n`` with ``n G n^T <= radius_squared``, sorted."""

    inverse = _inverse3(metric)
    bounds = [0, 0, 0]
    for j in range(3):
        bounds[j] = floor(sqrt(radius_squared * inverse[j][j])) + 1
    result = []
    for n0 in range(-bounds[0], bounds[0] + 1):
        for n1 in range(-bounds[1], bounds[1] + 1):
            for n2 in range(-bounds[2], bounds[2] + 1):
                if n0 == 0 and n1 == 0 and n2 == 0:
                    continue
                value = _quadratic([n0, n1, n2], metric)
                if value <= radius_squared:
                    result.append((value, (n0, n1, n2)))
    result.sort()
    return result


def _cosine_window(cosine: float) -> tuple[float, float]:
    """Cosine interval equivalent to an angle within the tolerance of ``acos(cosine)``.

    ``cos`` decreases on ``[0, pi]``, so ``|theta - phi| <= tau`` is
    ``cos(min(phi + tau, pi)) <= cos(theta) <= cos(max(phi - tau, 0))``,
    expanded with the angle-addition formulas.
    """

    sine = sqrt(max(0.0, 1.0 - cosine * cosine))
    if cosine >= _COS_ANGLE_TOLERANCE:
        upper = 1.0
    else:
        upper = cosine * _COS_ANGLE_TOLERANCE + sine * _SIN_ANGLE_TOLERANCE
    if cosine <= -_COS_ANGLE_TOLERANCE:
        lower = -1.0
    else:
        lower = cosine * _COS_ANGLE_TOLERANCE - sine * _SIN_ANGLE_TOLERANCE
    return lower, upper


def _correspondences_from(
    vectors: list[tuple[float, tuple[int, int, int]]],
    first: list[list[float]],
    second: list[list[float]],
) -> list[list[list[int]]]:
    """Proper integer bases of ``first`` whose rows match the rows of ``second``.

    ``vectors`` must contain every short vector of ``first`` up to
    ``(1 + length_tolerance)`` times the longest row of ``second``.
    """

    widen = (1.0 + LENGTH_TOLERANCE) * (1.0 + LENGTH_TOLERANCE)
    slots = []
    for i in range(3):
        target = second[i][i]
        slot = []
        for value, vector in vectors:
            if value <= target * widen and value * widen >= target:
                slot.append((value, list(vector)))
        slots.append(slot)
    windows = {}
    for i, j in ((0, 1), (0, 2), (1, 2)):
        windows[(i, j)] = _cosine_window(second[i][j] / sqrt(second[i][i] * second[j][j]))
    result = []
    for q0, n0 in slots[0]:
        for q1, n1 in slots[1]:
            lower, upper = windows[(0, 1)]
            cosine = _bilinear(n0, n1, first) / sqrt(q0 * q1)
            if not lower <= cosine <= upper:
                continue
            for q2, n2 in slots[2]:
                rows = [n0, n1, n2]
                if _determinant3(rows) != 1:
                    continue
                lower, upper = windows[(0, 2)]
                cosine = _bilinear(n0, n2, first) / sqrt(q0 * q2)
                if not lower <= cosine <= upper:
                    continue
                lower, upper = windows[(1, 2)]
                cosine = _bilinear(n1, n2, first) / sqrt(q1 * q2)
                if not lower <= cosine <= upper:
                    continue
                result.append([list(n0), list(n1), list(n2)])
    return result


def _integer_product(left: list[list[int]], right: list[list[int]]) -> list[list[int]]:
    result = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
    for i in range(3):
        for j in range(3):
            result[i][j] = (
                left[i][0] * right[0][j] + left[i][1] * right[1][j] + left[i][2] * right[2][j]
            )
    return result


def _signed_permutation_key(rows: list[list[int]]) -> tuple[int, ...]:
    """Smallest flattened image of ``rows`` under the 24 proper signed permutations.

    Permuting or sign-flipping reference rows (with determinant +1) relabels the
    row filter consistently, so all members of one class give one decision.
    """

    best = (
        rows[0][0], rows[0][1], rows[0][2],
        rows[1][0], rows[1][1], rows[1][2],
        rows[2][0], rows[2][1], rows[2][2],
    )
    for p0, p1, p2, parity in (
        (0, 1, 2, 1),
        (1, 2, 0, 1),
        (2, 0, 1, 1),
        (0, 2, 1, -1),
        (2, 1, 0, -1),
        (1, 0, 2, -1),
    ):
        for s0 in (1, -1):
            for s1 in (1, -1):
                s2 = parity * s0 * s1
                candidate = (
                    s0 * rows[p0][0],
                    s0 * rows[p0][1],
                    s0 * rows[p0][2],
                    s1 * rows[p1][0],
                    s1 * rows[p1][1],
                    s1 * rows[p1][2],
                    s2 * rows[p2][0],
                    s2 * rows[p2][1],
                    s2 * rows[p2][2],
                )
                if candidate < best:
                    best = candidate
    return best


def _reference_bases(metric: list[list[float]]) -> list[list[list[int]]]:
    """Every right-handed basis attaining the successive minima, per class.

    The successive minima are lattice invariants, so this set of bases does not
    depend on how the lattice was written down.  It contains several Gram forms
    for some lattices (for example 60/60/60 and 90/60/60 degree face-centred
    cubic cells) and every exact automorphic image.  One representative per
    proper signed-permutation class is returned, sorted.
    """

    longest = max(metric[0][0], metric[1][1], metric[2][2])
    vectors = _short_vectors(metric, longest * (1.0 + 2.0 * _MINIMA_RELATIVE_TOLERANCE))
    minima: list[float] = []
    chosen: list[list[int]] = []
    for value, vector in vectors:
        if len(chosen) == 0:
            independent = True
        elif len(chosen) == 1:
            a = chosen[0]
            independent = (
                a[1] * vector[2] - a[2] * vector[1] != 0
                or a[2] * vector[0] - a[0] * vector[2] != 0
                or a[0] * vector[1] - a[1] * vector[0] != 0
            )
        else:
            independent = _determinant3([chosen[0], chosen[1], list(vector)]) != 0
        if independent:
            chosen.append(list(vector))
            minima.append(value)
            if len(chosen) == 3:
                break
    if len(chosen) != 3:
        raise ValueError("successive minima enumeration did not span the lattice")
    slots = []
    for i in range(3):
        bound = minima[i] * (1.0 + _MINIMA_RELATIVE_TOLERANCE)
        slot = []
        for value, vector in vectors:
            if value <= bound:
                slot.append(list(vector))
        slots.append(slot)
    keys = set()
    for v0 in slots[0]:
        for v1 in slots[1]:
            for v2 in slots[2]:
                rows = [v0, v1, v2]
                if _determinant3(rows) == 1:
                    keys.add(_signed_permutation_key(rows))
    if not keys:
        raise ValueError("no basis attains the successive minima")
    result = []
    for key in sorted(keys):
        result.append(
            [[key[0], key[1], key[2]], [key[3], key[4], key[5]], [key[6], key[7], key[8]]]
        )
    return result


def _metric_frame(metric: list[list[float]]) -> list[float]:
    """Precomputed bounds for :func:`_closest_image` in one SPD metric.

    ``r_j = sqrt((G^-1)_jj)`` bounds ``|x_j| <= |x|_G r_j`` (Cauchy-Schwarz).
    ``S`` is the Schur complement of ``g22``, so ``x01 S x01^T`` is the exact
    minimum of ``|x|_G^2`` over real ``x2``, and ``p0 = S00 - S01^2/S11`` makes
    ``p0 x0^2`` the exact minimum over real ``(x1, x2)``.
    """

    inverse = _inverse3(metric)
    g02 = metric[0][2]
    g12 = metric[1][2]
    g22 = metric[2][2]
    s00 = metric[0][0] - g02 * g02 / g22
    s01 = metric[0][1] - g02 * g12 / g22
    s11 = metric[1][1] - g12 * g12 / g22
    return [
        metric[0][0],
        metric[0][1],
        g02,
        metric[1][1],
        g12,
        g22,
        sqrt(inverse[0][0]),
        sqrt(inverse[1][1]),
        s00 - s01 * s01 / s11,
        s00,
        s01,
        s11,
    ]


def _closest_image(delta: list[float], frame: list[float]) -> tuple[float, list[float]]:
    """Exact closest periodic image of a fractional displacement (sphere decoding).

    The box from ``frame`` contains every improving image.  Rows ``n0`` and
    pairs ``(n0, n1)`` whose exact lower bound (see :func:`_metric_frame`) is
    not below the incumbent cannot improve it and are skipped.  For fixed
    ``(n0, n1)`` the squared norm is a convex quadratic in ``n2``; its integer
    minimizer is one of the two integers bracketing the real minimizer.
    """

    g00, g01, g02, g11, g12, g22, r0, r1, p0, s00, s01, s11 = frame
    b0 = delta[0] - round(delta[0])
    b1 = delta[1] - round(delta[1])
    b2 = delta[2] - round(delta[2])
    best = (
        b0 * b0 * g00
        + b1 * b1 * g11
        + b2 * b2 * g22
        + 2.0 * (b0 * b1 * g01 + b0 * b2 * g02 + b1 * b2 * g12)
    )
    best_vector = [b0, b1, b2]
    radius = sqrt(best)
    for n0 in range(floor(-b0 - radius * r0), ceil(-b0 + radius * r0) + 1):
        x0 = b0 + n0
        if p0 * x0 * x0 >= best:
            continue
        for n1 in range(floor(-b1 - radius * r1), ceil(-b1 + radius * r1) + 1):
            x1 = b1 + n1
            if s00 * x0 * x0 + 2.0 * s01 * x0 * x1 + s11 * x1 * x1 >= best:
                continue
            lowest = floor(-b2 - (g02 * x0 + g12 * x1) / g22)
            for n2 in (lowest, lowest + 1):
                x2 = b2 + n2
                value = (
                    x0 * x0 * g00
                    + x1 * x1 * g11
                    + x2 * x2 * g22
                    + 2.0 * (x0 * x1 * g01 + x0 * x2 * g02 + x1 * x2 * g12)
                )
                if value < best:
                    best = value
                    best_vector = [x0, x1, x2]
    return best, best_vector


def _closest_image_squared(
    delta: list[float], metric: list[list[float]]
) -> tuple[float, list[float]]:
    return _closest_image(delta, _metric_frame(metric))


def _assignment(cost: list[list[float]]) -> list[int]:
    """Minimum-cost perfect matching (Hungarian algorithm with potentials)."""

    size = len(cost)
    infinity = float("inf")
    row_potential = [0.0] * (size + 1)
    column_potential = [0.0] * (size + 1)
    owner = [0] * (size + 1)
    way = [0] * (size + 1)
    for row in range(1, size + 1):
        owner[0] = row
        column = 0
        minimum = [infinity] * (size + 1)
        used = [False] * (size + 1)
        while True:
            used[column] = True
            current = owner[column]
            delta = infinity
            following = 0
            for j in range(1, size + 1):
                if not used[j]:
                    reduced = (
                        cost[current - 1][j - 1] - row_potential[current] - column_potential[j]
                    )
                    if reduced < minimum[j]:
                        minimum[j] = reduced
                        way[j] = column
                    if minimum[j] < delta:
                        delta = minimum[j]
                        following = j
            for j in range(size + 1):
                if used[j]:
                    row_potential[owner[j]] += delta
                    column_potential[j] -= delta
                else:
                    minimum[j] -= delta
            column = following
            if owner[column] == 0:
                break
        while True:
            previous = way[column]
            owner[column] = owner[previous]
            column = previous
            if column == 0:
                break
    result = [0] * size
    for j in range(1, size + 1):
        result[owner[j] - 1] = j - 1
    return result


def _anchor_atom(
    points: list[list[float]],
    species: list[int],
    block: tuple[int, int],
    frame: list[float],
) -> int:
    """Atom of ``block`` with the smallest sorted neighbour-distance signature.

    The signature depends only on the crystal, not on coordinate order; atoms
    with equal signatures are symmetry-equivalent up to rounding and give
    equivalent searches.
    """

    if block[1] - block[0] == 1:
        return block[0]
    best_index = block[0]
    best_signature = None
    for i in range(block[0], block[1]):
        signature = []
        for j in range(len(points)):
            if j != i:
                value, _ = _closest_image(
                    [
                        points[j][0] - points[i][0],
                        points[j][1] - points[i][1],
                        points[j][2] - points[i][2],
                    ],
                    frame,
                )
                signature.append((value, species[j]))
        signature.sort()
        if best_signature is None or signature < best_signature:
            best_signature = signature
            best_index = i
    return best_index


def _evaluate_translation(
    first: list[list[float]],
    second: list[list[float]],
    blocks: list[tuple[int, int]],
    translation: list[float],
    frame: list[float],
    limit_squared: float,
) -> tuple[bool, float, list[float], list[tuple[int, int]]]:
    """Feasibility, best feasible sum of squares, least-squares shift, and pairs."""

    feasible = True
    total = 0.0
    shift = [0.0, 0.0, 0.0]
    pairs = []
    for start, stop in blocks:
        size = stop - start
        squared = []
        vectors = []
        for i in range(start, stop):
            point = first[i]
            row_values = []
            row_vectors = []
            for j in range(start, stop):
                target = second[j]
                value, vector = _closest_image(
                    [
                        target[0] - point[0] - translation[0],
                        target[1] - point[1] - translation[1],
                        target[2] - point[2] - translation[2],
                    ],
                    frame,
                )
                row_values.append(value)
                row_vectors.append(vector)
            squared.append(row_values)
            vectors.append(row_vectors)
        free = _assignment(squared)
        free_valid = True
        free_total = 0.0
        for r in range(size):
            vector = vectors[r][free[r]]
            shift[0] += vector[0]
            shift[1] += vector[1]
            shift[2] += vector[2]
            free_total += squared[r][free[r]]
            if squared[r][free[r]] > limit_squared:
                free_valid = False
        if not feasible:
            continue
        if free_valid:
            # The unconstrained optimum is feasible, hence the feasible optimum.
            total += free_total
            for r in range(size):
                pairs.append((start + r, start + free[r]))
            continue
        forbidden = 1.0 + size * limit_squared
        constrained = []
        for r in range(size):
            row = []
            for c in range(size):
                value = squared[r][c]
                row.append(value if value <= limit_squared else forbidden)
            constrained.append(row)
        chosen = _assignment(constrained)
        for r in range(size):
            value = squared[r][chosen[r]]
            if value > limit_squared:
                feasible = False
                break
            total += value
            pairs.append((start + r, start + chosen[r]))
    count = blocks[-1][1]
    return feasible, total, [shift[0] / count, shift[1] / count, shift[2] / count], pairs


def _search_role(
    first: tuple[list[list[float]], list[list[float]], list[int]],
    second: tuple[list[list[float]], list[list[float]], list[int]],
    exhaustive: bool,
) -> tuple[float, list[list[int]], list[list[int]], list[float], list[tuple[int, int]]] | None:
    """Witness search with ``second`` as the reference lattice."""

    first_lattice, first_positions, species = first
    second_lattice, second_positions, _ = second
    count = len(species)
    blocks = []
    start = 0
    for index in range(1, count + 1):
        if index == count or species[index] != species[start]:
            blocks.append((start, index))
            start = index
    anchor_block = blocks[0]
    for block in blocks:
        if block[1] - block[0] < anchor_block[1] - anchor_block[0]:
            anchor_block = block
    first_basis, first_metric = _lll_reduce(_normalized_metric(first_lattice, count))
    second_reduction, second_reduced = _lll_reduce(_normalized_metric(second_lattice, count))
    references = _reference_bases(second_reduced)
    base = references[0]
    base_inverse = _adjugate3(base)
    second_basis = _integer_product(base, second_reduction)
    second_metric = _transform_gram(second_reduced, base)
    second_points = _change_basis(second_positions, second_basis)
    widen = (1.0 + LENGTH_TOLERANCE) * (1.0 + LENGTH_TOLERANCE)
    longest = max(second_metric[0][0], second_metric[1][1], second_metric[2][2])
    vectors = _short_vectors(
        first_metric, longest * (1.0 + 2.0 * _MINIMA_RELATIVE_TOLERANCE) * widen
    )
    correspondences = {}
    for reference in references:
        relative_inverse = _adjugate3(_integer_product(reference, base_inverse))
        for rows in _correspondences_from(
            vectors, first_metric, _transform_gram(second_reduced, reference)
        ):
            mapped_rows = _integer_product(relative_inverse, rows)
            key = (
                mapped_rows[0][0], mapped_rows[0][1], mapped_rows[0][2],
                mapped_rows[1][0], mapped_rows[1][1], mapped_rows[1][2],
                mapped_rows[2][0], mapped_rows[2][1], mapped_rows[2][2],
            )
            correspondences[key] = mapped_rows
    if not correspondences:
        return None
    anchor = _anchor_atom(
        _change_basis(first_positions, first_basis),
        species,
        anchor_block,
        _metric_frame(first_metric),
    )
    limit_squared = SITE_TOLERANCE * SITE_TOLERANCE
    best = None
    for key in sorted(correspondences):
        rows = correspondences[key]
        transform = _integer_product(rows, first_basis)
        first_points = _change_basis(first_positions, transform)
        mapped = _transform_gram(first_metric, rows)
        metric = [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]
        for i in range(3):
            for j in range(3):
                metric[i][j] = 0.5 * (mapped[i][j] + second_metric[i][j])
        frame = _metric_frame(metric)
        for partner in range(anchor_block[0], anchor_block[1]):
            translation = [
                second_points[partner][0] - first_points[anchor][0],
                second_points[partner][1] - first_points[anchor][1],
                second_points[partner][2] - first_points[anchor][2],
            ]
            feasible, total, shift, pairs = _evaluate_translation(
                first_points, second_points, blocks, translation, frame, limit_squared
            )
            if feasible and (best is None or total < best[0]):
                best = (total, transform, second_basis, translation, pairs)
                if not exhaustive:
                    return best
            refined = translation
            for _ in range(_REFINEMENT_STEPS):
                refined = [
                    refined[0] + shift[0],
                    refined[1] + shift[1],
                    refined[2] + shift[2],
                ]
                feasible, total, shift, pairs = _evaluate_translation(
                    first_points, second_points, blocks, refined, frame, limit_squared
                )
                if feasible and (best is None or total < best[0]):
                    best = (total, transform, second_basis, refined, pairs)
                    if not exhaustive:
                        return best
    return best


def _best_witness(
    left: bytes, right: bytes, exhaustive: bool
) -> (
    tuple[float, bool, list[list[int]], list[list[int]], list[float], list[tuple[int, int]]]
    | None
):
    """Best examined witness ``(sum_sq, swapped, T, U, t, pairs)`` or ``None``.

    Both lattices serve in turn as the reference (logical OR), so the relation
    is symmetric without a representation-dependent role choice.  ``T`` and
    ``U`` are proper integer bases of the witness's first and second payloads
    in their payload bases, ``t`` is the translation in the ``U`` basis, and
    ``pairs`` maps first-payload sites to second-payload sites.  ``swapped`` is
    true when the witness's first payload is the target.  ``sum_sq`` is the sum
    of squared normalized displacements.
    """

    ordered = right < left
    if ordered:
        left, right = right, left
    parsed_left = _parse_payload(left)
    parsed_right = _parse_payload(right)
    # Sites are sorted by atomic number, so equal lists mean equal compositions.
    if parsed_left[2] != parsed_right[2]:
        return None
    best = None
    for role in (0, 1):
        if role == 0:
            found = _search_role(parsed_left, parsed_right, exhaustive)
            swapped = ordered
        else:
            found = _search_role(parsed_right, parsed_left, exhaustive)
            swapped = not ordered
        if found is not None and (best is None or found[0] < best[0]):
            best = (found[0], swapped, found[1], found[2], found[3], found[4])
            if not exhaustive:
                return best
    return best


def physical_structure_match(candidate: bytes, target: bytes) -> bool:
    """Return the certified tolerance equivalence of two encoded structures."""

    return _best_witness(candidate, target, False) is not None


def physical_structure_rms(candidate: bytes, target: bytes) -> float | None:
    """Smallest RMS displacement over examined feasible witnesses, else ``None``.

    The value is in units of the cube root of the volume per atom.
    """

    best = _best_witness(candidate, target, True)
    if best is None:
        return None
    count = len(_parse_payload(candidate)[2])
    return sqrt(best[0] / count)


def physical_structure_witness(
    candidate: bytes, target: bytes
) -> (
    tuple[float, bool, list[list[int]], list[list[int]], list[float], list[tuple[int, int]]]
    | None
):
    """Minimum-RMS examined witness for independent audit, else ``None``.

    See ``_best_witness`` for the tuple layout.  When ``swapped`` is true the
    first payload of the witness is ``target``.
    """

    return _best_witness(candidate, target, True)


def encode_physical_structure(
    lattice: npt.ArrayLike,
    fractional_positions: npt.ArrayLike,
    atomic_numbers: npt.ArrayLike,
) -> bytes:
    """Canonical primitive-cell payload of one right-handed periodic structure.

    The input basis is first LLL-reduced (an exact proper change of basis), so
    the certified periodic searches of the primitive reduction stay bounded for
    arbitrarily skewed input cells.
    """

    cell = np.asarray(lattice, dtype=np.float64)
    if cell.shape != (3, 3) or not np.all(np.isfinite(cell)):
        raise DomainError("lattice must be a finite 3x3 matrix")
    positions = np.asarray(fractional_positions, dtype=np.float64)
    if positions.ndim != 2 or positions.shape[1:] != (3,) or not np.all(np.isfinite(positions)):
        raise DomainError("fractional positions must be a finite [N,3] array")
    try:
        reduction, _ = _lll_reduce((cell @ cell.T).tolist())
    except (ValueError, ZeroDivisionError) as exc:
        raise DomainError(f"lattice cannot be reduced: {exc}") from exc
    unimodular = np.asarray(reduction, dtype=np.int64)
    inverse = np.asarray(_adjugate3(reduction), dtype=np.int64)
    primitive = primitive_reduce_arrays(
        unimodular.astype(np.float64) @ cell,
        positions @ inverse.astype(np.float64),
        atomic_numbers,
        cartesian_tolerance=PRIMITIVE_REDUCTION_TOLERANCE_ANGSTROM,
    )
    entries = [repr(float(value) + 0.0) for value in primitive.lattice.reshape(-1)]
    sites = []
    for number, row in zip(
        primitive.atomic_numbers, primitive.fractional_positions, strict=True
    ):
        coordinates = []
        for value in row:
            wrapped = float(value) % 1.0
            if wrapped == 1.0 or wrapped == 0.0:
                wrapped = 0.0
            coordinates.append(wrapped)
        sites.append((int(number), coordinates[0], coordinates[1], coordinates[2]))
    sites.sort()
    lines = [_PAYLOAD_HEADER, "lattice " + " ".join(entries)]
    lines.extend(f"site {number} {x!r} {y!r} {z!r}" for number, x, y, z in sites)
    payload = ("\n".join(lines) + "\n").encode("ascii")
    try:
        _parse_payload(payload)
    except ValueError as exc:
        raise DomainError(f"structure cannot be encoded canonically: {exc}") from exc
    return payload


def validate_physical_payload(payload: bytes) -> None:
    """Raise :class:`DomainError` unless ``payload`` is a canonical record."""

    try:
        _parse_payload(payload)
    except (TypeError, ValueError) as exc:
        raise DomainError(f"invalid physical-structure payload: {exc}") from exc


def physical_structure_matcher_contract() -> EquivalenceMatcherContract:
    """Diagnostic contract binding the matcher code and every tolerance."""

    return EquivalenceMatcherContract.from_callable(
        physical_structure_match,
        algorithm="crystalatlas-certified-periodic-structure-equivalence",
        version="1",
        cell_policy="lll-then-certified-primitive-reduction-successive-minima-reference-bases",
        parameters={
            "anchor": "least-populated-species-atom-with-smallest-neighbour-signature",
            "angle_tolerance_cosine": _COS_ANGLE_TOLERANCE,
            "angle_tolerance_degrees": ANGLE_TOLERANCE_DEGREES,
            "angle_tolerance_sine": _SIN_ANGLE_TOLERANCE,
            "arithmetic": "python-int-and-ieee754-basic-operations-and-sqrt-only",
            "completeness": "exact-for-witnesses-within-half-site-tolerance",
            "length_tolerance": LENGTH_TOLERANCE,
            "length_tolerance_semantics": "row-length-ratio-within-1-plus-tolerance-both-ways",
            "payload_header": _PAYLOAD_HEADER,
            "primitive_reduction_tolerance_angstrom": PRIMITIVE_REDUCTION_TOLERANCE_ANGSTROM,
            "reference_bases": "all-right-handed-bases-attaining-successive-minima",
            "reference_minima_relative_tolerance": _MINIMA_RELATIVE_TOLERANCE,
            "reference_roles": "either-crystal-logical-or",
            "representation_invariant_physical_equivalence_claim": True,
            "rotation_group": "SO(3)",
            "site_metric": "mean-of-unit-volume-per-atom-metrics",
            "site_tolerance": SITE_TOLERANCE,
            "site_tolerance_unit": "cube-root-volume-per-atom",
            "translation_search": (
                "anchor-to-every-same-species-atom-plus-three-least-squares-steps"
            ),
            "volume_scaled": True,
        },
    )


def evaluate_physical_geometry_decomposition(
    cases: Sequence[GeometryDecompositionCase[bytes]],
    *,
    ks: Sequence[int],
    evaluation_contract: DiagnosticEvaluationContract,
    bootstrap: BootstrapProtocol | None = None,
) -> GeometryDecompositionDiagnostics:
    """Oracle-versus-predicted geometry Match@K under the physical matcher.

    Every target and valid-attempt payload is validated before any decision so
    a malformed record fails closed instead of counting as a non-match.
    """

    for case in cases:
        for target in case.acceptable_targets:
            validate_physical_payload(target.payload)
        for attempt in (*case.oracle_scaffold_attempts, *case.predicted_scaffold_attempts):
            if attempt.valid:
                assert attempt.payload is not None
                validate_physical_payload(attempt.payload)
    return evaluate_geometry_decomposition(
        cases,
        ks=ks,
        matcher=physical_structure_match,
        matcher_contract=physical_structure_matcher_contract(),
        evaluation_contract=evaluation_contract,
        bootstrap=bootstrap,
    )


__all__ = [
    "ANGLE_TOLERANCE_DEGREES",
    "LENGTH_TOLERANCE",
    "PRIMITIVE_REDUCTION_TOLERANCE_ANGSTROM",
    "SITE_TOLERANCE",
    "encode_physical_structure",
    "evaluate_physical_geometry_decomposition",
    "physical_structure_match",
    "physical_structure_matcher_contract",
    "physical_structure_rms",
    "physical_structure_witness",
    "validate_physical_payload",
]
