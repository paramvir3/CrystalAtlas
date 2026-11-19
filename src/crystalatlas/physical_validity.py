"""Descriptive physical-validity metrics for generated structures (review m3).

The decoder certifies crystallographic validity: catalogue arithmetic, group
containment, and collisions at 1e-8 A.  These metrics add the conventional
*descriptive* screen of the generative-crystal literature: the fraction of
structures whose smallest interatomic distance, including periodic self-images,
reaches a declared threshold (0.5 A by default).  It is a data-policy report,
not a physical acceptance criterion.

Charge neutrality is deliberately not computed here.  It requires a vetted
oxidation-state table (for example the SMACT or pymatgen defaults), which must
be bound as an external, versioned dependency rather than transcribed.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from .errors import DomainError
from .periodic import minimum_periodic_pair_distance, shortest_nonzero_lattice_image

DEFAULT_MINIMUM_DISTANCE_ANGSTROM = 0.5


def minimum_interatomic_distance(
    lattice: npt.ArrayLike, fractional_positions: npt.ArrayLike
) -> float:
    """Smallest distance between any two atoms or an atom and its own image (A)."""

    cell = np.asarray(lattice, dtype=np.float64)
    positions = np.asarray(fractional_positions, dtype=np.float64)
    if cell.shape != (3, 3) or not np.all(np.isfinite(cell)) or np.linalg.det(cell) <= 0.0:
        raise DomainError("lattice must be a finite right-handed 3x3 matrix")
    if positions.ndim != 2 or positions.shape[1:] != (3,) or not len(positions):
        raise DomainError("fractional positions must have shape [N, 3] with N > 0")
    if not np.all(np.isfinite(positions)):
        raise DomainError("fractional positions must be finite")
    self_image = math.sqrt(shortest_nonzero_lattice_image(cell).squared_distance)
    if len(positions) == 1:
        return self_image
    return float(
        min(self_image, minimum_periodic_pair_distance(positions, cell, upper_bound=self_image))
    )


@dataclass(frozen=True, slots=True)
class DistanceValiditySummary:
    threshold_angstrom: float
    structures: int
    valid: int
    smallest_distance_angstrom: float

    @property
    def rate(self) -> float:
        return self.valid / self.structures


def distance_validity_summary(
    structures: Sequence[tuple[npt.ArrayLike, npt.ArrayLike]],
    *,
    threshold_angstrom: float = DEFAULT_MINIMUM_DISTANCE_ANGSTROM,
) -> DistanceValiditySummary:
    """Fraction of ``(lattice, fractional_positions)`` pairs at or above the threshold."""

    if not math.isfinite(threshold_angstrom) or threshold_angstrom <= 0.0:
        raise DomainError("threshold must be finite and positive")
    if not structures:
        raise DomainError("at least one structure is required")
    distances = [minimum_interatomic_distance(*structure) for structure in structures]
    return DistanceValiditySummary(
        threshold_angstrom=float(threshold_angstrom),
        structures=len(distances),
        valid=sum(distance >= threshold_angstrom for distance in distances),
        smallest_distance_angstrom=min(distances),
    )


__all__ = [
    "DEFAULT_MINIMUM_DISTANCE_ANGSTROM",
    "DistanceValiditySummary",
    "distance_validity_summary",
    "minimum_interatomic_distance",
]
