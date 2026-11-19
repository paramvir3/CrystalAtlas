"""Bounded unimodular basis-change augmentation of the CFM endpoint (review 2026-10-29, C2).

Theorem 2.3's chart does not quotient GL(3, Z) basis changes.  Preprocessing
canonicalizes every training presentation, so for P1, P-1 and the monoclinic
groups the target law lives on a fundamental domain with sharp boundaries while
the base ``N(0, I)`` covers all of ``z`` (probe D: ``|z_1 - z_2| = 1.54`` for one
P-1 lattice in two reduced bases).  ``shear-v1`` draws, with a declared
probability, one admissible elementary shear ``W = I +- E_ij`` and moves the
record to the neighbouring presentation: ``G' = W^-1 G W^-T`` and ``eta'`` from
the exact row maps (row convention ``x -> x W``).  The target becomes a mixture
over neighbouring presentations of the same crystals, which reduces the jump
across the domain boundary.  A shear is admissible when ``(W, 0)`` normalizes
the setting exactly and maps every row of the cover onto itself (not every
shear preserves the fixed sites of P-1).  An image outside the chart's Loewner
interval is refused and the record is kept.  A sample generated in a
non-canonical basis decodes to the same physical crystal; the primitive
reduction canonicalizes it.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction

import numpy as np

from .algebra import AffineOperation, IntegerMatrix
from .errors import DomainError, InvariantViolation
from .normalizer import compose, normalizes
from .residual_gauge import _row_affine

BASIS_AUGMENTATIONS: tuple[str, ...] = ("none", "shear-v1")


def elementary_shears() -> tuple[IntegerMatrix, ...]:
    """The twelve unimodular shears ``I +- E_ij`` (``i != j``), ``det = +1``."""

    shears: list[IntegerMatrix] = []
    for i in range(3):
        for j in range(3):
            if i == j:
                continue
            for sign in (1, -1):
                rows = [[int(r == c) for c in range(3)] for r in range(3)]
                rows[i][j] = sign
                shears.append(tuple(tuple(row) for row in rows))
    return tuple(shears)


@dataclass(frozen=True)
class BasisChange:
    """One admissible change ``x -> x W`` with its exact ``eta -> eta M + m``."""

    matrix: np.ndarray
    inverse: np.ndarray
    eta_matrix: np.ndarray
    eta_shift: np.ndarray

    def move_metric(self, metric: np.ndarray) -> np.ndarray:
        """``G' = W^-1 G W^-T``: the metric of the same lattice in the new basis."""

        moved = self.inverse @ np.asarray(metric, dtype=np.float64) @ self.inverse.T
        return 0.5 * (moved + moved.T)


class CoverBasisAugmentation:
    """The admissible elementary shears of one cover and their torus actions."""

    def __init__(self, layout: object) -> None:
        self.layout = layout
        setting = layout.setting  # type: ignore[attr-defined]
        rows = layout.positions  # type: ignore[attr-defined]
        slices = layout._raw_parameter_slices  # type: ignore[attr-defined]
        raw_dimension = sum(len(row.coordinate_matrix) for row in rows)
        reconstruction = np.zeros(
            (layout.intrinsic_dimension, raw_dimension),  # type: ignore[attr-defined]
            dtype=np.float64,
        )
        for parameter_slice, mapping in zip(
            slices,
            layout.parameter_maps,  # type: ignore[attr-defined]
            strict=True,
        ):
            reconstruction[:, parameter_slice] = mapping
        encoder = np.asarray(layout._parameter_encoder, dtype=np.float64)  # type: ignore[attr-defined]
        changes: list[BasisChange] = []
        for w in elementary_shears():
            g = AffineOperation(w, (Fraction(0), Fraction(0), Fraction(0)))
            if not normalizes(setting, g):
                continue
            raw_matrix = np.zeros((raw_dimension, raw_dimension), dtype=np.float64)
            raw_shift = np.zeros(raw_dimension, dtype=np.float64)
            admissible = True
            for row, parameter_slice in zip(rows, slices, strict=True):
                found = None
                for operation in setting.operations:
                    candidates = _row_affine(row, row, compose(operation, g))
                    if candidates:
                        found = candidates[0]
                        break
                if found is None:
                    admissible = False
                    break
                width = parameter_slice.stop - parameter_slice.start
                if width:
                    raw_matrix[parameter_slice, parameter_slice] = np.asarray(
                        [[float(v) for v in r] for r in found[0]], dtype=np.float64
                    ).reshape(width, width)
                    raw_shift[parameter_slice] = [float(v) for v in found[1]]
            if not admissible:
                continue
            eta_matrix = reconstruction @ raw_matrix @ encoder
            eta_shift = np.mod(raw_shift @ encoder, 1.0)
            if eta_matrix.size and not np.isclose(
                abs(np.linalg.det(eta_matrix)), 1.0, atol=1e-12
            ):
                raise InvariantViolation("basis change is not torus-unimodular")
            matrix = np.asarray(w, dtype=np.int64)
            inverse = np.rint(np.linalg.inv(matrix)).astype(np.int64)
            if not np.array_equal(matrix @ inverse, np.eye(3, dtype=np.int64)):
                raise InvariantViolation("shear inverse is not integral")
            changes.append(BasisChange(matrix, inverse, eta_matrix, eta_shift))
        self.changes: tuple[BasisChange, ...] = tuple(changes)

    def __len__(self) -> int:
        return len(self.changes)

    def transform(
        self, index: int, z: np.ndarray, eta: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray] | None:
        """``(z', eta')`` of the record in basis ``index``, or ``None`` outside the chart."""

        change = self.changes[index]
        chart = self.layout.chart  # type: ignore[attr-defined]
        moved = change.move_metric(chart.decode_latent(np.asarray(z, dtype=np.float64)))
        try:
            z_new = np.asarray(chart.encode_latent(moved), dtype=np.float64)
        except DomainError:
            return None
        eta_value = np.asarray(eta, dtype=np.float64)
        eta_new = self.layout.canonicalize_eta(  # type: ignore[attr-defined]
            eta_value @ change.eta_matrix + change.eta_shift
        )
        return z_new, eta_new


_CACHE: dict[tuple[object, ...], CoverBasisAugmentation] = {}


def cover_basis_augmentation(layout: object) -> CoverBasisAugmentation:
    """The cached admissible shears of a layout's cover."""

    key = (layout.cover, layout.setting.key, layout.setting.hall_symbol)  # type: ignore[attr-defined]
    cached = _CACHE.get(key)
    if cached is None:
        cached = CoverBasisAugmentation(layout)
        _CACHE[key] = cached
    return cached


__all__ = [
    "BASIS_AUGMENTATIONS",
    "BasisChange",
    "CoverBasisAugmentation",
    "cover_basis_augmentation",
    "elementary_shears",
]
