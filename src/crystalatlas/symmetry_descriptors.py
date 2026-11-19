"""Operation-derived symmetry descriptors for the cover selector (review m1).

Every descriptor is computed from the certified operations of a setting, so it
cannot depend on Wyckoff letters or row indices.  Rotation types are classified
by ``(det, trace)``, which conjugation preserves, so the descriptors do not
depend on the basis.

* :func:`rotation_type_histogram`: fraction of the elements of a point group in
  each of the ten crystallographic rotation types.  Over the 230 pinned
  settings it separates exactly the 32 geometric point-group classes (tested).
* :func:`group_descriptor`: that histogram for the space group's point group,
  then ``log(1 + |PG|) / log(49)``, a centring one-hot (P, A/B/C, I, F, R), a
  polar flag (some vector is fixed by every rotation), and a chiral (Sohncke)
  flag (every rotation is proper).
* :func:`site_stabilizer_rotations`: the exact stabilizer of a row's certified
  generic representative.  Normalizer-equivalent rows share a site group and
  therefore a descriptor.
* :func:`row_geometry_descriptor` (review 2026-10-29, M1): the exact chart of
  the row's canonical representative under the full Euclidean normalizer,
  with the canonical row indices.  Rotation-type histograms cannot separate
  Pm-3m 12h ``(x, 1/2, 0)`` from 12i ``(0, y, y)``, or I4/mmm 8g from 8h, which
  are different crystal chemistry; the chart does, while normalizer-equivalent
  rows (Pm-3m 1a and 1b) still share it.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from fractions import Fraction

import numpy as np
import numpy.typing as npt

from .catalogue import SpaceGroupSetting
from .cover import canonical_rows
from .wyckoff import WyckoffPosition

ROTATION_TYPES: tuple[tuple[int, int], ...] = (
    (1, 3),  # 1
    (1, -1),  # 2
    (1, 0),  # 3
    (1, 1),  # 4
    (1, 2),  # 6
    (-1, -3),  # -1
    (-1, 1),  # m
    (-1, 0),  # -3
    (-1, -1),  # -4
    (-1, -2),  # -6
)
GROUP_DESCRIPTOR_DIM = len(ROTATION_TYPES) + 1 + 5 + 2
ROW_SITE_DESCRIPTOR_DIM = len(ROTATION_TYPES)
ROW_GEOMETRY_DESCRIPTOR_DIM = 3 + 9 + 2
"""Canonical representative offset (3), coordinate matrix rows padded to 3 x 3
(9), the canonical representative's row index and the row's own index, both
divided by 32 (the catalogue's largest row count is 27)."""
_ROW_INDEX_SCALE = 32.0
_ROW_CLASSES: dict[tuple[str, str], dict[str, tuple[str, ...]]] = {}
_ROW_GEOMETRY: dict[tuple[str, str], dict[str, npt.NDArray[np.float64]]] = {}


def rotation_type_histogram(rotations: Sequence[npt.ArrayLike]) -> npt.NDArray[np.float64]:
    counts: npt.NDArray[np.float64] = np.zeros(len(ROTATION_TYPES), dtype=np.float64)
    for rotation in rotations:
        matrix = np.rint(np.asarray(rotation, dtype=np.float64)).astype(int)
        key = (round(float(np.linalg.det(matrix))), int(np.trace(matrix)))
        counts[ROTATION_TYPES.index(key)] += 1.0
    return counts / counts.sum()


def _distinct_rotations(setting: SpaceGroupSetting) -> list[np.ndarray]:
    return [np.asarray(rotation, dtype=np.int64) for rotation in setting.point_group_rotations]


def site_stabilizer_rotations(
    setting: SpaceGroupSetting, row: WyckoffPosition
) -> tuple[np.ndarray, ...]:
    """Distinct rotation parts of the operations fixing the generic representative."""

    point = row.representative_exact(row.generic_parameter)
    rotations: dict[bytes, np.ndarray] = {}
    for operation in setting.operations:
        image = operation.apply_exact(point)
        if all(
            (Fraction(a) - Fraction(b)).denominator == 1
            for a, b in zip(image, point, strict=True)
        ):
            matrix = np.asarray(operation.rotation, dtype=np.int64)
            rotations[matrix.tobytes()] = matrix
    return tuple(rotations.values())


def _centring_one_hot(setting: SpaceGroupSetting) -> list[float]:
    translations = {tuple(t) for t in setting.centering_translations}
    index = len(translations)
    half = Fraction(1, 2)
    if index == 1:
        position = 0
    elif index == 4:
        position = 3
    elif index == 3:
        position = 4
    elif (half, half, half) in translations:
        position = 2
    else:
        position = 1
    return [1.0 if slot == position else 0.0 for slot in range(5)]


def group_descriptor(setting: SpaceGroupSetting) -> npt.NDArray[np.float64]:
    rotations = _distinct_rotations(setting)
    stacked = np.stack([rotation - np.eye(3, dtype=np.int64) for rotation in rotations])
    fixed_dimension = 3 - int(np.linalg.matrix_rank(stacked.reshape(-1, 3).astype(float)))
    polar = 1.0 if fixed_dimension > 0 else 0.0
    chiral = 1.0 if all(round(np.linalg.det(r)) == 1 for r in rotations) else 0.0
    values = [
        *rotation_type_histogram(rotations),
        math.log1p(len(rotations)) / math.log(49.0),
        *_centring_one_hot(setting),
        polar,
        chiral,
    ]
    return np.asarray(values, dtype=np.float64)


def row_site_descriptor(
    setting: SpaceGroupSetting, row: WyckoffPosition
) -> npt.NDArray[np.float64]:
    return rotation_type_histogram(site_stabilizer_rotations(setting, row))


def row_normalizer_classes(setting: SpaceGroupSetting) -> dict[str, tuple[str, ...]]:
    """Wyckoff rows grouped by the full Euclidean normalizer, keyed by representative.

    Two rows are in one class when some normalizer element composed with a group
    operation maps one row's parametrized chart onto the other's (the exact
    ``_row_affine`` test of :mod:`crystalatlas.residual_gauge`).  The
    representative is the class member that comes first in
    :func:`crystalatlas.cover.canonical_rows`.  Computed once per setting.
    """

    from .normalizer import compose, euclidean_normalizer
    from .residual_gauge import _row_affine

    key = (setting.key, setting.hall_symbol)
    cached = _ROW_CLASSES.get(key)
    if cached is not None:
        return cached
    order = [spec.label for spec in canonical_rows(setting)]
    rows = {label: setting.wyckoff(label) for label in order}
    identity = ((1, 0, 0), (0, 1, 0), (0, 0, 1))
    cosets = [
        g
        for g in euclidean_normalizer(setting)
        if not (g.rotation == identity and not any(g.translation))
    ]
    parent = {label: label for label in order}

    def find(label: str) -> str:
        while parent[label] != label:
            label = parent[label]
        return label

    for index, source in enumerate(order):
        row = rows[source]
        for target in order[index + 1 :]:
            other = rows[target]
            if other.multiplicity != row.multiplicity or other.dimension != row.dimension:
                continue
            if find(source) == find(target):
                continue
            if any(
                _row_affine(row, other, compose(op, g))
                for g in cosets
                for op in setting.operations
            ):
                parent[find(target)] = find(source)
    classes: dict[str, list[str]] = {}
    for label in order:
        classes.setdefault(find(label), []).append(label)
    result = {
        representative: tuple(members) for representative, members in classes.items()
    }
    _ROW_CLASSES[key] = result
    return result


def row_geometry_descriptor(
    setting: SpaceGroupSetting, row: WyckoffPosition
) -> npt.NDArray[np.float64]:
    """Exact chart of the row's normalizer-canonical representative and row indices.

    Entries 0-2: the representative offset of the canonical row; 3-11: its
    coordinate matrix, one row per free parameter, padded with zeros to three
    rows; 12: the canonical row's index in :func:`canonical_rows` over 32; 13:
    this row's own index over 32.  Entries 0-12 are invariant under the full
    Euclidean normalizer; entry 13 lets the selector reproduce the data's
    canonical choice among gauge-equivalent rows.
    """

    key = (setting.key, setting.hall_symbol)
    table = _ROW_GEOMETRY.get(key)
    if table is None:
        order = [spec.label for spec in canonical_rows(setting)]
        representative_of = {
            member: representative
            for representative, members in row_normalizer_classes(setting).items()
            for member in members
        }
        table = {}
        for label in order:
            canonical = setting.wyckoff(representative_of[label])
            matrix = np.zeros((3, 3), dtype=np.float64)
            for i, coordinate_row in enumerate(canonical.coordinate_matrix):
                matrix[i] = [float(value) for value in coordinate_row]
            values = [
                *(float(value) for value in canonical.representative_offset),
                *matrix.reshape(-1),
                order.index(canonical.label) / _ROW_INDEX_SCALE,
                order.index(label) / _ROW_INDEX_SCALE,
            ]
            table[label] = np.asarray(values, dtype=np.float64)
        _ROW_GEOMETRY[key] = table
    try:
        return table[row.label].copy()
    except KeyError as exc:
        raise ValueError(f"row {row.label!r} is not a Wyckoff row of {setting.key}") from exc


__all__ = [
    "GROUP_DESCRIPTOR_DIM",
    "ROTATION_TYPES",
    "ROW_GEOMETRY_DESCRIPTOR_DIM",
    "ROW_SITE_DESCRIPTOR_DIM",
    "group_descriptor",
    "rotation_type_histogram",
    "row_geometry_descriptor",
    "row_normalizer_classes",
    "row_site_descriptor",
    "site_stabilizer_rotations",
]
