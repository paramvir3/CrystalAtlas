"""Certified residual gauge of one cover's intrinsic coordinates (option (a)).

The intrinsic coordinate ``eta`` of a cover names a crystal non-uniquely.
Three kinds of parameter maps leave the crystal unchanged, or map it onto a
properly congruent copy, which the SO(3) chirality quotient identifies:

1. Per-orbit in-group gauges.  An operation ``h`` of ``G`` that maps a Wyckoff
   row's parametrized subspace ``x(p) = p C + c0`` onto itself acts as
   ``x(p) -> x(p A + a)``.  The orbits at ``p`` and ``p A + a`` are the same
   point set, so the gauge acts on each orbit independently.  P-1's sign flip
   ``p -> -p`` is such a gauge.
2. Global automorphisms from the proper Euclidean normalizer
   (:mod:`crystalatlas.normalizer`).  They map every occupied (row, species) onto
   an occupied one, possibly a different row of the same species.  One element
   acts on all orbits together.  With ``improper=True`` (normalizer-v3) the
   improper elements are included as well.  They map a crystal onto its
   enantiomorph, so the sampled law becomes racemic within the cover.
3. Identical-orbit relabellings (``IntrinsicCoverLayout.residual_action_matrix``).

Each map is derived exactly in raw Wyckoff parameters, certified exactly on
decoded point sets with species, and conjugated to ``eta`` through the layout's
integer encoder.  The result is an affine torus map ``eta -> eta M + m``.

Uniform sampling.  The group is ``(K_1 x ... x K_n) S N``.  The per-orbit gauges
form a normal subgroup.  The relabellings permute identical orbits, and the
normalizer automorphisms conjugate the in-group gauges, so every partial product
is a group.  Independent uniform draws of the three factors therefore give the
uniform law on the product, even where the factors overlap.
"""

from __future__ import annotations

import math
from collections import OrderedDict
from collections.abc import Sequence
from dataclasses import dataclass
from fractions import Fraction
from itertools import product

import numpy as np

from .algebra import AffineOperation
from .catalogue import SpaceGroupSetting
from .errors import DomainError, InvariantViolation
from .normalizer import compose, euclidean_normalizer, proper_euclidean_normalizer

FracMatrix = tuple[tuple[Fraction, ...], ...]
FracVector = tuple[Fraction, ...]


def _mat(rows: Sequence[Sequence[object]]) -> FracMatrix:
    return tuple(tuple(Fraction(v) for v in row) for row in rows)  # type: ignore[arg-type]


def _matmul(left: FracMatrix, right: FracMatrix) -> FracMatrix:
    inner = len(right)
    return tuple(
        tuple(
            sum((left[i][k] * right[k][j] for k in range(inner)), Fraction(0))
            for j in range(len(right[0]))
        )
        for i in range(len(left))
    )


def _vecmat(vector: FracVector, matrix: FracMatrix) -> FracVector:
    return tuple(
        sum((vector[k] * matrix[k][j] for k in range(len(vector))), Fraction(0))
        for j in range(len(matrix[0]))
    )


def _inverse(matrix: FracMatrix) -> FracMatrix:
    size = len(matrix)
    work = [
        list(row) + [Fraction(int(i == j)) for j in range(size)] for i, row in enumerate(matrix)
    ]
    for column in range(size):
        pivot = next((r for r in range(column, size) if work[r][column] != 0), None)
        if pivot is None:
            raise DomainError("matrix is singular")
        work[column], work[pivot] = work[pivot], work[column]
        lead = work[column][column]
        work[column] = [value / lead for value in work[column]]
        for r in range(size):
            if r != column and work[r][column] != 0:
                factor = work[r][column]
                work[r] = [a - factor * b for a, b in zip(work[r], work[column], strict=True)]
    return tuple(tuple(row[size:]) for row in work)


def _right_pseudo_inverse(c: FracMatrix) -> FracMatrix:
    """``C^T (C C^T)^-1`` for a full-row-rank ``C`` (d x 3)."""

    ct = tuple(tuple(c[i][j] for i in range(len(c))) for j in range(3))
    return _matmul(ct, _inverse(_matmul(c, ct)))


def _det(matrix: FracMatrix) -> Fraction:
    size = len(matrix)
    if size == 0:
        return Fraction(1)
    if size == 1:
        return matrix[0][0]
    return sum(
        (
            (-1) ** j * matrix[0][j] * _det(tuple(row[:j] + row[j + 1 :] for row in matrix[1:]))
            for j in range(size)
        ),
        Fraction(0),
    )


def _mod_one(vector: FracVector) -> FracVector:
    return tuple(value - (value.numerator // value.denominator) for value in vector)


def _solve_offset(c: FracMatrix, c_pinv: FracMatrix, target: FracVector) -> list[FracVector]:
    """All ``a`` (mod 1) with ``a C = target + n`` for small integer ``n``."""

    solutions = set()
    for shift in product(range(-2, 3), repeat=3):
        value = tuple(t + n for t, n in zip(target, shift, strict=True))
        a = _vecmat(value, c_pinv)
        if _vecmat(a, c) == value:
            solutions.add(_mod_one(a))
    return sorted(solutions)


def _row_affine(
    row: object, image_row: object, operation: AffineOperation
) -> list[tuple[FracMatrix, FracVector]]:
    """Maps ``p -> p A + a`` with ``operation(x_row(p)) = x_image(p A + a)`` (mod Z^3)."""

    if len(row.coordinate_matrix) == 0:  # type: ignore[attr-defined]
        # A fixed row: the operation must map its orbit point set onto the image row's.
        moved_points = frozenset(
            tuple(v % 1 for v in operation.apply_exact(point))
            for point in row.expand_exact(())  # type: ignore[attr-defined]
        )
        return [((), ())] if moved_points == _orbit_points(image_row, ()) else []
    c = _mat(row.coordinate_matrix)  # type: ignore[attr-defined]
    c_image = _mat(image_row.coordinate_matrix)  # type: ignore[attr-defined]
    offset = tuple(Fraction(v) for v in row.representative_offset)  # type: ignore[attr-defined]
    image_offset = tuple(Fraction(v) for v in image_row.representative_offset)  # type: ignore[attr-defined]
    rotation = _mat(operation.rotation)
    cr = _matmul(c, rotation)
    c_pinv = _right_pseudo_inverse(c_image)
    a_matrix = _matmul(cr, c_pinv)
    if _matmul(a_matrix, c_image) != cr:
        return []
    if any(v.denominator != 1 for r in a_matrix for v in r) or abs(_det(a_matrix)) != 1:
        return []
    moved = tuple(
        v + t - o
        for v, t, o in zip(
            _vecmat(offset, rotation), operation.translation, image_offset, strict=True
        )
    )
    return [(a_matrix, a) for a in _solve_offset(c_image, c_pinv, moved)]


def row_parameter_gauges(
    setting: SpaceGroupSetting, row: object
) -> tuple[tuple[FracMatrix, FracVector], ...]:
    """Per-orbit in-group gauges ``p -> p A + a`` of one positive-dimensional row."""

    dimension = len(row.coordinate_matrix)  # type: ignore[attr-defined]
    if dimension == 0:
        return (((), ()),)
    identity = tuple(
        tuple(Fraction(int(i == j)) for j in range(dimension)) for i in range(dimension)
    )
    found: dict[tuple[FracMatrix, FracVector], None] = {
        (identity, (Fraction(0),) * dimension): None
    }
    for operation in setting.operations:
        for item in _row_affine(row, row, operation):
            found.setdefault(item, None)
    return tuple(found)


_ROW_GAUGES: dict[tuple[str, str, str], tuple[tuple[FracMatrix, FracVector], ...]] = {}


def certified_row_parameter_gauges(
    setting: SpaceGroupSetting, row: object
) -> tuple[tuple[FracMatrix, FracVector], ...]:
    """:func:`row_parameter_gauges`, certified exactly once per (setting, row).

    For two generic rational parameters, every gauge must map the orbit point
    set onto itself exactly.
    """

    key = (setting.key, setting.hall_symbol, row.label)  # type: ignore[attr-defined]
    cached = _ROW_GAUGES.get(key)
    if cached is not None:
        return cached
    gauges = row_parameter_gauges(setting, row)
    dimension = len(row.coordinate_matrix)  # type: ignore[attr-defined]
    for k in range(2 if dimension else 0):
        parameter = tuple(Fraction(7 * i + 3 * k + 1, 101 + 2 * k) for i in range(dimension))
        reference = _orbit_points(row, parameter)
        for a_matrix, a_shift in gauges:
            moved = _mod_one(
                tuple(v + t for v, t in zip(_vecmat(parameter, a_matrix), a_shift, strict=True))
            )
            if _orbit_points(row, moved) != reference:
                raise InvariantViolation(
                    f"{setting.key}:{row.label}: an orbit gauge changed the orbit"  # type: ignore[attr-defined]
                )
    _ROW_GAUGES[key] = gauges
    return gauges


def _orbit_points(row: object, parameter: FracVector) -> frozenset[FracVector]:
    return frozenset(
        tuple(v % 1 for v in point)
        for point in row.expand_exact(parameter)  # type: ignore[attr-defined]
    )


@dataclass(frozen=True)
class RawAffineMap:
    """``raw -> raw M + m`` on the concatenated raw Wyckoff parameters."""

    matrix: FracMatrix
    shift: FracVector


class CoverResidualGauge:
    """The certified residual gauge of one cover, with eta-level actions."""

    def __init__(self, layout: object, *, improper: bool = False) -> None:
        self.layout = layout
        self.improper = bool(improper)
        setting: SpaceGroupSetting = layout.setting  # type: ignore[attr-defined]
        cover = layout.cover  # type: ignore[attr-defined]
        rows = layout.positions  # type: ignore[attr-defined]
        self._slices = layout._raw_parameter_slices  # type: ignore[attr-defined]
        self._raw_dimension = sum(len(r.coordinate_matrix) for r in rows)
        self.orbit_choices = tuple(certified_row_parameter_gauges(setting, row) for row in rows)
        self.normalizer_maps = self._normalizer_maps(setting, cover, rows)
        reconstruction = np.zeros(
            (layout.intrinsic_dimension, self._raw_dimension),
            dtype=np.float64,  # type: ignore[attr-defined]
        )
        for parameter_slice, mapping in zip(self._slices, layout.parameter_maps, strict=True):  # type: ignore[attr-defined]
            reconstruction[:, parameter_slice] = mapping
        self._reconstruction = reconstruction
        self._encoder = np.asarray(layout._parameter_encoder, dtype=np.float64)  # type: ignore[attr-defined]
        self.certified = self._certify(setting, cover, rows)

    # -- construction ---------------------------------------------------------------
    def _normalizer(self, setting: SpaceGroupSetting) -> tuple[AffineOperation, ...]:
        if self.improper:
            return euclidean_normalizer(setting)
        return proper_euclidean_normalizer(setting)

    def _block_map(
        self, blocks: Sequence[tuple[int, int, FracMatrix, FracVector]]
    ) -> RawAffineMap:
        """Raw map from (source slot, target slot, A, a) blocks."""

        size = self._raw_dimension
        matrix = [[Fraction(0)] * size for _ in range(size)]
        shift = [Fraction(0)] * size
        for source, target, a_matrix, a_shift in blocks:
            src, dst = self._slices[source], self._slices[target]
            for i in range(src.stop - src.start):
                for j in range(dst.stop - dst.start):
                    matrix[src.start + i][dst.start + j] = a_matrix[i][j]
            for j in range(dst.stop - dst.start):
                shift[dst.start + j] = a_shift[j]
        return RawAffineMap(tuple(map(tuple, matrix)), tuple(shift))

    def _normalizer_maps(self, setting, cover, rows) -> tuple[RawAffineMap, ...]:  # type: ignore[no-untyped-def]
        identity = self._block_map(
            [
                (
                    k,
                    k,
                    row_parameter_gauges(setting, row)[0][0],
                    (Fraction(0),) * len(row.coordinate_matrix),
                )
                for k, row in enumerate(rows)
            ]
        )
        maps: list[RawAffineMap] = [identity]
        all_rows = list(setting.wyckoff_positions.values())
        for g in self._normalizer(setting):
            if g.rotation == ((1, 0, 0), (0, 1, 0), (0, 0, 1)) and not any(g.translation):
                continue
            images: list[tuple[object, FracMatrix, FracVector]] = []
            for row in rows:
                found = None
                for image_row in all_rows:
                    if image_row.multiplicity != row.multiplicity or len(
                        image_row.coordinate_matrix
                    ) != len(row.coordinate_matrix):
                        continue
                    for operation in setting.operations:
                        candidates = _row_affine(row, image_row, compose(operation, g))
                        if candidates:
                            found = (image_row, *candidates[0])
                            break
                    if found is not None:
                        break
                if found is None:
                    raise InvariantViolation(f"{setting.key}: a normalizer element lost a row")
                images.append(found)
            # The element is a cover automorphism iff it preserves the (row, species) multiset.
            image_types = sorted(
                (image[0].label, orbit.atomic_number)
                for image, orbit in zip(images, cover.orbits, strict=True)
            )
            source_types = sorted(
                (orbit.wyckoff_label, orbit.atomic_number) for orbit in cover.orbits
            )
            if image_types != source_types:
                continue
            free = list(range(len(cover.orbits)))
            blocks = []
            for k, (image, orbit) in enumerate(zip(images, cover.orbits, strict=True)):
                target = next(
                    j
                    for j in free
                    if cover.orbits[j].wyckoff_label == image[0].label
                    and cover.orbits[j].atomic_number == orbit.atomic_number
                )
                free.remove(target)
                blocks.append((k, target, image[1], image[2]))
            maps.append(self._block_map(blocks))
        return tuple(maps)

    # -- certification ---------------------------------------------------------------
    def _raw_crystal(self, rows, cover, raw: FracVector) -> frozenset:  # type: ignore[no-untyped-def]
        points = set()
        for parameter_slice, row, orbit in zip(self._slices, rows, cover.orbits, strict=True):
            parameter = raw[parameter_slice.start : parameter_slice.stop]
            for point in _orbit_points(row, parameter):
                points.add((point, orbit.atomic_number))
        return frozenset(points)

    def _certify(self, setting, cover, rows) -> bool:  # type: ignore[no-untyped-def]
        samples = [
            tuple(Fraction(7 * i + 3 * k + 1, 101 + 2 * k) for i in range(self._raw_dimension))
            for k in range(2)
        ]
        cosets = self._normalizer(setting)
        for raw in samples:
            crystal = self._raw_crystal(rows, cover, raw)
            for raw_map in self.normalizer_maps:
                image = (
                    _mod_one(
                        tuple(
                            v + s
                            for v, s in zip(
                                _vecmat(raw, raw_map.matrix), raw_map.shift, strict=True
                            )
                        )
                    )
                    if self._raw_dimension
                    else ()
                )
                target = self._raw_crystal(rows, cover, image)
                if not any(
                    frozenset((tuple(v % 1 for v in g.apply_exact(p)), z) for p, z in crystal)
                    == target
                    for g in cosets
                ):
                    raise InvariantViolation(
                        f"{setting.key}: a normalizer map is not a congruence of the crystal"
                    )
        return True

    # -- eta actions -------------------------------------------------------------------
    def _to_eta(self, matrix: np.ndarray, shift: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        eta_matrix = self._reconstruction @ matrix @ self._encoder
        eta_shift = np.mod(shift @ self._encoder, 1.0)
        if eta_matrix.size and not np.isclose(abs(np.linalg.det(eta_matrix)), 1.0, atol=1e-12):
            raise InvariantViolation("residual gauge action is not torus-unimodular")
        return eta_matrix, eta_shift

    def _raw_numpy(self, raw_map: RawAffineMap) -> tuple[np.ndarray, np.ndarray]:
        size = self._raw_dimension
        return (
            np.asarray(
                [[float(v) for v in row] for row in raw_map.matrix], dtype=np.float64
            ).reshape(size, size),
            np.asarray([float(v) for v in raw_map.shift], dtype=np.float64).reshape(size),
        )

    def eta_action(
        self, orbit_indices: Sequence[int], permutation: Sequence[int], normalizer_index: int
    ) -> tuple[np.ndarray, np.ndarray]:
        """Affine ``eta -> eta M + m`` of (per-orbit gauges, relabelling, normalizer)."""

        size = self._raw_dimension
        k_matrix = np.zeros((size, size))
        k_shift = np.zeros(size)  # noqa: E702
        for slot, index in enumerate(orbit_indices):
            sl = self._slices[slot]
            a_matrix, a_shift = self.orbit_choices[slot][index]
            width = sl.stop - sl.start
            if width:
                k_matrix[sl, sl] = np.asarray(
                    [[float(v) for v in r] for r in a_matrix]
                ).reshape(width, width)
                k_shift[sl] = [float(v) for v in a_shift]
        p_matrix = np.zeros((size, size))
        for new_slot, old_slot in enumerate(permutation):
            out, old = self._slices[new_slot], self._slices[old_slot]
            if out.stop > out.start:
                p_matrix[old, out] = np.eye(out.stop - out.start)
        n_matrix, n_shift = self._raw_numpy(self.normalizer_maps[normalizer_index])
        matrix = k_matrix @ p_matrix @ n_matrix
        shift = (k_shift @ p_matrix) @ n_matrix + n_shift
        return self._to_eta(matrix, shift)

    def all_eta_actions(self) -> tuple[tuple[np.ndarray, np.ndarray], ...]:
        """Every distinct action of the group (for oracles on small covers)."""

        groups = self.layout.residual_orbit_groups  # type: ignore[attr-defined]
        permutations = [tuple(range(len(self.orbit_choices)))]
        for group in groups:
            extended = []
            for base in permutations:
                for order in __import__("itertools").permutations(group):
                    item = list(base)
                    for slot, source in zip(group, order, strict=True):
                        item[slot] = base[source]
                    extended.append(tuple(item))
            permutations = extended
        seen: dict[tuple[float, ...], tuple[np.ndarray, np.ndarray]] = {}
        for orbit_indices in product(*[range(len(c)) for c in self.orbit_choices]):
            for permutation in permutations:
                for normalizer_index in range(len(self.normalizer_maps)):
                    matrix, shift = self.eta_action(
                        orbit_indices, permutation, normalizer_index
                    )
                    key = tuple(
                        np.round(np.concatenate([matrix.ravel(), np.mod(shift, 1.0)]), 12)
                    )
                    seen.setdefault(key, (matrix, np.mod(shift, 1.0)))
        return tuple(seen.values())

    def sample_eta_action(
        self, rng: np.random.Generator, *, permutation: Sequence[int]
    ) -> tuple[np.ndarray, np.ndarray]:
        orbit_indices = [int(rng.integers(len(choices))) for choices in self.orbit_choices]
        normalizer_index = int(rng.integers(len(self.normalizer_maps)))
        return self.eta_action(orbit_indices, permutation, normalizer_index)


_GAUGES: dict[tuple[object, ...], CoverResidualGauge] = {}


def cover_residual_gauge(layout: object, *, improper: bool = False) -> CoverResidualGauge:
    """The cached certified residual gauge: normalizer-v2, or normalizer-v3 if ``improper``."""

    key = (layout.cover, layout.setting.key, layout.setting.hall_symbol, bool(improper))  # type: ignore[attr-defined]
    gauge = _GAUGES.get(key)
    if gauge is None:
        gauge = CoverResidualGauge(layout, improper=improper)
        _GAUGES[key] = gauge
    return gauge


RESIDUAL_GAUGE_KINDS = ("relabel-v1", "normalizer-v2", "normalizer-v3")
RESIDUAL_ACTION_CACHE_SIZE = 4096
RESIDUAL_ENUMERATION_LIMIT = 65536


class ResidualActionTable:
    """Digit coding of one cover's residual actions, for batched uniform sampling.

    One action is named by a row of digits, one digit per column.  The columns,
    in order: for every identical-orbit group of size ``n`` the Fisher-Yates
    digits with moduli ``n, n-1, ..., 2`` (digit ``t`` picks one of the orbits not
    yet placed, so the rows of a group are in bijection with its ``n!``
    relabellings); then for ``relabel-v1`` one bit per P-1 row with a free
    parameter, and for ``normalizer-v2/v3`` one index per orbit that has more
    than one certified per-orbit gauge and one index into the normalizer maps
    when there is more than one.  Columns of modulus one are not stored.

    Independent uniform digits are independent uniform draws of the factors,
    which is the law of the per-record sampler
    (``LearnedCrystalGenerator._sample_residual_torus_action``).  The
    multiplication map of the factors onto the group has fibres of one size (the
    module docstring: every partial product is a group), so the action is uniform
    on the group, and :meth:`all_rows` lists every group element equally often.
    """

    def __init__(self, layout: object, kind: str) -> None:
        if kind not in RESIDUAL_GAUGE_KINDS:
            raise DomainError(
                "residual gauge must be one of " + ", ".join(RESIDUAL_GAUGE_KINDS)
            )
        self.layout = layout
        self.kind = kind
        self.dimension = int(layout.intrinsic_dimension)  # type: ignore[attr-defined]
        self._orbit_count = len(layout.cover.orbits)  # type: ignore[attr-defined]
        self._groups = tuple(
            tuple(int(slot) for slot in group)
            for group in layout.residual_orbit_groups  # type: ignore[attr-defined]
        )
        moduli: list[int] = []
        for group in self._groups:
            moduli.extend(range(len(group), 1, -1))
        self._gauge: CoverResidualGauge | None = None
        self._sign_rows: tuple[int, ...] = ()
        self._orbit_slots: tuple[int, ...] = ()
        self._normalizer_column = False
        if kind == "relabel-v1":
            if layout.setting.key == "P-1":  # type: ignore[attr-defined]
                self._sign_rows = tuple(
                    index
                    for index, row in enumerate(layout.positions)  # type: ignore[attr-defined]
                    if row.dimension
                )
            moduli.extend(2 for _ in self._sign_rows)
        else:
            gauge = cover_residual_gauge(layout, improper=kind == "normalizer-v3")
            if any(not choices for choices in gauge.orbit_choices) or not gauge.normalizer_maps:
                raise InvariantViolation("a residual gauge factor has no element")
            self._gauge = gauge
            self._orbit_slots = tuple(
                slot for slot, choices in enumerate(gauge.orbit_choices) if len(choices) > 1
            )
            moduli.extend(len(gauge.orbit_choices[slot]) for slot in self._orbit_slots)
            self._normalizer_column = len(gauge.normalizer_maps) > 1
            if self._normalizer_column:
                moduli.append(len(gauge.normalizer_maps))
        self.moduli = tuple(moduli)
        self.cardinality = math.prod(moduli)
        self.has_shift = kind != "relabel-v1"
        self._cache: OrderedDict[tuple[int, ...], tuple[np.ndarray, np.ndarray | None]] = (
            OrderedDict()
        )

    def permutation(self, digits: Sequence[int]) -> tuple[int, ...]:
        """The orbit relabelling named by the leading (Fisher-Yates) digits.

        ``permutation[new_slot]`` is the old slot placed there, the convention of
        ``IntrinsicCoverLayout.residual_action_matrix``.
        """

        cursor = 0
        permutation = list(range(self._orbit_count))
        for group in self._groups:
            pool = list(range(len(group)))
            order = []
            for _ in range(len(group) - 1):
                order.append(pool.pop(int(digits[cursor])))
                cursor += 1
            order.append(pool[0])
            for new_slot, old_offset in zip(group, order, strict=True):
                permutation[new_slot] = group[old_offset]
        return tuple(permutation)

    def action(self, digits: Sequence[int]) -> tuple[np.ndarray, np.ndarray | None]:
        """The affine action ``eta -> eta M + m`` of one digit row (``m`` is None
        for ``relabel-v1``, which has no translation part)."""

        key = tuple(int(value) for value in digits)
        if len(key) != len(self.moduli) or any(
            not 0 <= value < modulus for value, modulus in zip(key, self.moduli, strict=True)
        ):
            raise DomainError("residual action digits lie outside the table")
        cached = self._cache.get(key)
        if cached is not None:
            self._cache.move_to_end(key)
            return cached
        permutation = self.permutation(key)
        cursor = sum(len(group) - 1 for group in self._groups)
        if self._gauge is None:
            signs = [1] * self._orbit_count
            for index in self._sign_rows:
                signs[index] = 1 if key[cursor] == 0 else -1
                cursor += 1
            matrix = np.asarray(
                self.layout.residual_action_matrix(permutation, orbit_signs=signs),  # type: ignore[attr-defined]
                dtype=np.float64,
            )
            result: tuple[np.ndarray, np.ndarray | None] = (matrix, None)
        else:
            orbit_indices = [0] * len(self._gauge.orbit_choices)
            for slot in self._orbit_slots:
                orbit_indices[slot] = key[cursor]
                cursor += 1
            normalizer_index = key[cursor] if self._normalizer_column else 0
            matrix, shift = self._gauge.eta_action(orbit_indices, permutation, normalizer_index)
            result = (
                np.asarray(matrix, dtype=np.float64),
                np.asarray(shift, dtype=np.float64),
            )
        for array in result:
            if array is not None:
                array.setflags(write=False)
        self._cache[key] = result
        if len(self._cache) > RESIDUAL_ACTION_CACHE_SIZE:
            self._cache.popitem(last=False)
        return result

    def actions(self, rows: np.ndarray) -> tuple[np.ndarray, np.ndarray | None]:
        """Actions of many digit rows: ``([count, d, d], [count, d] or None)``."""

        rows = np.asarray(rows)
        if (
            rows.ndim != 2
            or rows.shape[1] != len(self.moduli)
            or not np.issubdtype(rows.dtype, np.integer)
        ):
            raise DomainError("residual action rows must be an integer [count, columns] array")
        count = rows.shape[0]
        if rows.shape[1]:
            unique, inverse = np.unique(rows, axis=0, return_inverse=True)
            inverse = np.asarray(inverse).reshape(-1)
        else:
            unique = np.zeros((1, 0), dtype=np.int64)
            inverse = np.zeros(count, dtype=np.int64)
        table = [self.action(row) for row in unique.tolist()]
        matrices = np.stack([matrix for matrix, _ in table])[inverse]
        if not self.has_shift:
            return matrices, None
        shifts = np.stack([shift for _, shift in table])[inverse]  # type: ignore[misc]
        return matrices, shifts

    def all_rows(self) -> np.ndarray:
        """Every digit row once, ``[cardinality, columns]`` (small tables only)."""

        if self.cardinality > RESIDUAL_ENUMERATION_LIMIT:
            raise DomainError("the residual action table is too large to enumerate")
        if not self.moduli:
            return np.zeros((1, 0), dtype=np.int64)
        return np.indices(self.moduli, dtype=np.int64).reshape(len(self.moduli), -1).T
