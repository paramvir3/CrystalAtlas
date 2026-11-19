"""Strict raw-structure ingestion and the certified training-target boundary.

The CIF reader in this module intentionally implements a small, auditable
subset of CIF 1.1: one data block, scalar cell parameters, fractional atom-site
coordinates, explicit occupancies, and (unless the block declares P1) an
explicit complete symmetry-operation loop.  Unsupported or ambiguous CIF
semantics are rejected rather than silently interpreted.

Parsing is separate from crystallographic standardization.  A caller must
provide a :class:`CanonicalizationBackend` which removes hidden translations,
chooses the pinned Hall setting and origin, assigns exact Wyckoff charts, and
returns a proof receipt.  Training-target lifting is not attempted unless every
receipt gate passes and the primitive coloured composition is an exact quotient
of the parsed input.
"""

from __future__ import annotations

import json
import math
import re
import shlex
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from fractions import Fraction
from hashlib import sha256
from itertools import product
from math import gcd
from numbers import Integral
from pathlib import Path
from typing import Generic, Protocol, TypeVar, runtime_checkable

import numpy as np
import numpy.typing as npt

from .algebra import (
    AffineOperation,
    FractionMatrix,
    FractionVector,
    exact_determinant,
    exact_inverse,
    fraction_mod_one,
    validate_finite_affine_group,
)
from .catalogue import CertifiedCatalogue
from .config import DomainConfig
from .decoder import CrystalScaffold, ExactCrystalDecoder, OrbitAssignment
from .elements import atomic_number
from .errors import (
    AmbiguousPreprocessingError,
    CatalogueError,
    DomainError,
    InvariantViolation,
)
from .full_catalogue import (
    SPGLIB_SOURCE_VERSION,
    build_full_catalogue,
    pinned_hall_numbers,
    source_rows_for_type,
)
from .periodic import closest_periodic_image
from .standardization import (
    CanonicalizedStructure as NumericallyCanonicalizedStructure,
)
from .standardization import (
    FloatingStructure,
    _require_lattice_fidelity,
    _species_key,
    idealized_volume_matches,
)
from .training_targets import (
    CanonicalCatalogueTrainingTarget,
    ExactFixtureTrainingTarget,
    lift_canonical_catalogue_target,
    lift_exact_fixture_target,
)
from .wyckoff import WyckoffPosition

FloatArray = npt.NDArray[np.float64]
CIFLoop = tuple[tuple[str, ...], tuple[tuple[str, ...], ...]]
TargetT = TypeVar("TargetT")

# Bind this verification convention in dataset-builder receipts.  Species are
# ranked before spglib encoding by the numerical producer's type/repr key, not
# by numeric atomic number (for example, 16 precedes 8).
CANONICAL_ORIGIN_VERIFIER_ID = "crystalatlas.canonical-origin-verifier.type-repr-order.v2"

_CIF_NUMBER = re.compile(
    r"^([+-]?(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+))(?:\([0-9]+\))?"
    r"([Ee][+-]?[0-9]+)?$"
)
_ELEMENT_TYPE = re.compile(r"^([A-Z][a-z]?)(?:[0-9]*[+-])?$")
_ELEMENT_LABEL = re.compile(r"^([A-Z][a-z]?)")
_SYMMETRY_VARIABLES = {"x": 0, "y": 1, "z": 2}


def _readonly_float_array(
    value: npt.ArrayLike, shape: tuple[int, ...], *, name: str
) -> FloatArray:
    result = np.asarray(value, dtype=np.float64).copy()
    if result.shape != shape or not np.all(np.isfinite(result)):
        raise DomainError(f"{name} must be a finite array with shape {shape}")
    result.setflags(write=False)
    return result


@dataclass(frozen=True, slots=True)
class RawPeriodicStructure:
    """One fully expanded, fully occupied periodic coloured point set.

    Lattice vectors are rows, so Cartesian coordinates are ``f @ lattice`` and
    the fractional metric is ``lattice @ lattice.T``.  Fractional coordinates
    are reduced modulo one on construction.  Labels and ``source_id`` are audit
    metadata and do not enter the physical structure fingerprint.
    """

    lattice: FloatArray
    fractional_positions: FloatArray
    atomic_numbers: tuple[int, ...]
    occupancies: FloatArray
    labels: tuple[str, ...] = ()
    source_id: str | None = None

    def __post_init__(self) -> None:
        lattice = _readonly_float_array(self.lattice, (3, 3), name="lattice")
        determinant = float(np.linalg.det(lattice))
        scale = max(float(np.linalg.norm(lattice, ord=2)) ** 3, 1.0)
        if abs(determinant) <= 64.0 * np.finfo(np.float64).eps * scale:
            raise DomainError("raw lattice is singular or numerically rank deficient")

        raw_numbers = tuple(self.atomic_numbers)
        normalized_numbers: list[int] = []
        for raw_value in raw_numbers:
            value: object = raw_value
            if isinstance(value, bool) or not isinstance(value, Integral):
                raise DomainError("atomic numbers must be integers in the IUPAC element range")
            normalized_numbers.append(int(value))
        numbers = tuple(normalized_numbers)
        if not numbers:
            raise DomainError("raw structure must contain at least one atom")
        if any(isinstance(value, bool) or atomic_number(value) != value for value in numbers):
            raise DomainError("atomic numbers must be integers in the IUPAC element range")
        positions = _readonly_float_array(
            self.fractional_positions,
            (len(numbers), 3),
            name="fractional_positions",
        ).copy()
        positions %= 1.0
        positions[np.isclose(positions, 1.0, atol=0.0, rtol=0.0)] = 0.0
        if len({tuple(float(value) for value in row) for row in positions}) != len(positions):
            raise DomainError("raw structure contains coincident fractional sites")
        positions.setflags(write=False)

        occupancies = _readonly_float_array(
            self.occupancies, (len(numbers),), name="occupancies"
        )
        # A strict equality test is deliberate: partial occupancy is a different
        # state space and must never be rounded into this package's integer model.
        if not np.all(occupancies == 1.0):
            raise DomainError("only exactly unit-occupancy atom sites are supported")

        labels = tuple(self.labels)
        if labels and (len(labels) != len(numbers) or any(not value for value in labels)):
            raise DomainError("atom labels must be nonempty and align with all atoms")
        if len(set(labels)) != len(labels):
            raise DomainError("atom labels must be unique when supplied")
        if self.source_id is not None and not self.source_id.strip():
            raise DomainError("source_id must be nonempty when supplied")

        object.__setattr__(self, "lattice", lattice)
        object.__setattr__(self, "fractional_positions", positions)
        object.__setattr__(self, "atomic_numbers", numbers)
        object.__setattr__(self, "occupancies", occupancies)
        object.__setattr__(self, "labels", labels)

    @classmethod
    def from_arrays(
        cls,
        lattice: npt.ArrayLike,
        fractional_positions: npt.ArrayLike,
        atomic_numbers: Sequence[int],
        *,
        occupancies: npt.ArrayLike | None = None,
        labels: Sequence[str] = (),
        source_id: str | None = None,
    ) -> RawPeriodicStructure:
        """Validate raw arrays without inferring chemistry or symmetry."""

        numbers = tuple(atomic_numbers)
        occupancy_values = (
            np.ones(len(numbers), dtype=np.float64)
            if occupancies is None
            else np.asarray(occupancies, dtype=np.float64)
        )
        return cls(
            np.asarray(lattice, dtype=np.float64),
            np.asarray(fractional_positions, dtype=np.float64),
            numbers,
            occupancy_values,
            tuple(labels),
            source_id,
        )

    @property
    def metric(self) -> FloatArray:
        result = self.lattice @ self.lattice.T
        result.setflags(write=False)
        return result

    @property
    def volume(self) -> float:
        return abs(float(np.linalg.det(self.lattice)))

    @property
    def species_counts(self) -> Mapping[int, int]:
        return dict(Counter(self.atomic_numbers))

    @property
    def fingerprint(self) -> str:
        """Stable identity of the physical arrays, excluding labels and source name."""

        payload = {
            "schema": "crystalatlas.raw-periodic-structure.v1",
            "lattice": [[float(value).hex() for value in row] for row in self.lattice],
            "fractional_positions": [
                [float(value).hex() for value in row] for row in self.fractional_positions
            ],
            "atomic_numbers": list(self.atomic_numbers),
            "occupancies": [float(value).hex() for value in self.occupancies],
        }
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return sha256(encoded.encode("ascii")).hexdigest()


@dataclass(frozen=True, slots=True)
class CanonicalizationReceipt:
    """Auditable certificate required at the raw-to-training boundary."""

    source_fingerprint: str
    standardization_fingerprint: str
    catalogue_fingerprint: str
    hall_number: int
    hall_symbol: str
    source_to_primitive: FractionMatrix
    origin_shift: FractionVector
    free_origin_shift: FractionVector
    translation_index: int
    symmetry_tolerance: float
    roundtrip_max_cartesian_error: float
    symmetry_stability_verified: bool
    catalogue_closure_verified: bool
    wyckoff_multiplicity_verified: bool
    symmetry_verified: bool
    canonical_origin_verified: bool
    scaffold_roundtrip_verified: bool
    idealization_lattice_displacement_angstrom: float | None = None
    """Raw-to-ideal displacement of the primitive cell (dimensioned policy only)."""
    idealization_site_displacement_angstrom: float | None = None
    """Raw-to-ideal site displacement in the idealized metric (dimensioned policy only)."""

    def __post_init__(self) -> None:
        metadata = (
            self.idealization_lattice_displacement_angstrom,
            self.idealization_site_displacement_angstrom,
        )
        if (metadata[0] is None) != (metadata[1] is None):
            raise DomainError("idealization displacements must be recorded together")
        if any(
            value is not None
            and (isinstance(value, bool) or not math.isfinite(value) or value < 0.0)
            for value in metadata
        ):
            raise DomainError("idealization displacements must be finite and nonnegative")
        if not re.fullmatch(r"[0-9a-f]{64}", self.source_fingerprint):
            raise DomainError("source_fingerprint must be a lowercase SHA-256 digest")
        if not re.fullmatch(r"[0-9a-f]{64}", self.standardization_fingerprint):
            raise DomainError("standardization_fingerprint must be a lowercase SHA-256 digest")
        if not re.fullmatch(r"[0-9a-f]{64}", self.catalogue_fingerprint):
            raise DomainError("catalogue_fingerprint must be a lowercase SHA-256 digest")
        if (
            isinstance(self.hall_number, bool)
            or not isinstance(self.hall_number, int)
            or not 1 <= self.hall_number <= 530
            or not self.hall_symbol
        ):
            raise DomainError("receipt needs a valid pinned Hall number and symbol")
        matrix = tuple(tuple(value for value in row) for row in self.source_to_primitive)
        if len(matrix) != 3 or any(len(row) != 3 for row in matrix):
            raise DomainError("source_to_primitive must be an exact 3x3 matrix")
        if any(not isinstance(value, Fraction) for row in matrix for value in row):
            raise DomainError("source_to_primitive entries must be exact Fractions")
        origin = tuple(self.origin_shift)
        if len(origin) != 3 or any(not isinstance(value, Fraction) for value in origin):
            raise DomainError("origin_shift must be an exact length-three Fraction vector")
        free_origin = tuple(self.free_origin_shift)
        if len(free_origin) != 3 or any(
            not isinstance(value, Fraction) for value in free_origin
        ):
            raise DomainError("free_origin_shift must be an exact length-three Fraction vector")
        object.__setattr__(self, "source_to_primitive", matrix)
        object.__setattr__(
            self, "origin_shift", tuple(fraction_mod_one(value) for value in origin)
        )
        object.__setattr__(
            self,
            "free_origin_shift",
            tuple(fraction_mod_one(value) for value in free_origin),
        )
        if (
            isinstance(self.translation_index, bool)
            or not isinstance(self.translation_index, int)
            or self.translation_index < 1
        ):
            raise DomainError("translation_index must be a positive integer")
        determinant = exact_determinant(matrix)
        if determinant != Fraction(1, self.translation_index):
            raise DomainError(
                "source_to_primitive determinant disagrees with translation_index"
            )
        inverse = exact_inverse(matrix)
        if any(value.denominator != 1 for row in inverse for value in row):
            raise DomainError("source_to_primitive inverse must be integral")
        numerical = (self.symmetry_tolerance, self.roundtrip_max_cartesian_error)
        if any(not math.isfinite(value) for value in numerical):
            raise DomainError("receipt tolerances and errors must be finite")
        if self.symmetry_tolerance <= 0.0 or self.roundtrip_max_cartesian_error < 0.0:
            raise DomainError("receipt has an invalid tolerance or round-trip error")
        gates = (
            self.symmetry_stability_verified,
            self.catalogue_closure_verified,
            self.wyckoff_multiplicity_verified,
            self.symmetry_verified,
            self.canonical_origin_verified,
            self.scaffold_roundtrip_verified,
        )
        if any(not isinstance(value, bool) for value in gates):
            raise DomainError("canonicalization receipt gates must be booleans")

    @property
    def passed(self) -> bool:
        return all(
            (
                self.catalogue_closure_verified,
                self.symmetry_stability_verified,
                self.wyckoff_multiplicity_verified,
                self.symmetry_verified,
                self.canonical_origin_verified,
                self.scaffold_roundtrip_verified,
                self.roundtrip_max_cartesian_error <= self.symmetry_tolerance,
            )
        )


@dataclass(frozen=True, slots=True)
class CanonicalizedStructure:
    """Pinned conventional target, primitive quotient, and exact scaffold.

    Wyckoff multiplicities and ``scaffold`` coordinates refer to
    ``conventional_structure``.  Hidden-translation removal and the receipt's
    ``translation_index`` refer to ``primitive_structure``.  These cells are
    identical only for primitive Hall settings.
    """

    conventional_structure: RawPeriodicStructure
    primitive_structure: RawPeriodicStructure
    scaffold: CrystalScaffold
    receipt: CanonicalizationReceipt

    def __post_init__(self) -> None:
        if not isinstance(self.conventional_structure, RawPeriodicStructure):
            raise TypeError("conventional_structure must be a RawPeriodicStructure")
        if not isinstance(self.primitive_structure, RawPeriodicStructure):
            raise TypeError("primitive_structure must be a RawPeriodicStructure")
        if not isinstance(self.scaffold, CrystalScaffold):
            raise TypeError("scaffold must be a CrystalScaffold")
        if not isinstance(self.receipt, CanonicalizationReceipt):
            raise TypeError("receipt must be a CanonicalizationReceipt")
        if (
            float(np.linalg.det(self.conventional_structure.lattice)) <= 0.0
            or float(np.linalg.det(self.primitive_structure.lattice)) <= 0.0
        ):
            raise InvariantViolation("canonical lattices must be right handed")


@runtime_checkable
class CanonicalizationBackend(Protocol):
    """Backend contract for FullPrim, pinned setting, origin, and orbit assignment."""

    def canonicalize(self, structure: RawPeriodicStructure) -> CanonicalizedStructure:
        """Return the unique certified primitive representative of ``structure``."""


@runtime_checkable
class NumericalStandardizationBackend(Protocol):
    """Finite-precision FullPrim and pinned-Hall standardization contract."""

    def canonicalize(self, structure: FloatingStructure) -> NumericallyCanonicalizedStructure:
        """Return pinned conventional and primitive numerical representatives."""


@dataclass(frozen=True, slots=True)
class StandardizedRawStructure:
    """Raw input bound to the verified numerical standardizer output."""

    raw_structure: RawPeriodicStructure
    numerical_result: NumericallyCanonicalizedStructure
    conventional_structure: RawPeriodicStructure
    primitive_structure: RawPeriodicStructure


@runtime_checkable
class CatalogueTargetAssignmentBackend(Protocol):
    """Exact catalogue-backed inversion of standardized atoms to a scaffold."""

    def assign(self, standardized: StandardizedRawStructure) -> CanonicalizedStructure:
        """Return exact Wyckoff parameters and all proof gates for training."""


@dataclass(frozen=True, slots=True)
class ComposedCanonicalizer:
    """Compose numerical standardization with exact catalogue target assignment."""

    standardizer: NumericalStandardizationBackend
    target_assigner: CatalogueTargetAssignmentBackend

    def canonicalize(self, structure: RawPeriodicStructure) -> CanonicalizedStructure:
        standardized = standardize_raw_structure(structure, standardizer=self.standardizer)
        assigned = self.target_assigner.assign(standardized)
        if not isinstance(assigned, CanonicalizedStructure):
            raise TypeError("catalogue target assigner returned an invalid result type")
        receipt = assigned.receipt
        numerical = standardized.numerical_result
        if (
            receipt.standardization_fingerprint != numerical.provenance.fingerprint
            or receipt.hall_number != numerical.hall_number
            or receipt.hall_symbol != numerical.hall_symbol
            or receipt.translation_index != numerical.raw_translation_index
        ):
            raise InvariantViolation(
                "exact target receipt disagrees with numerical standardization provenance"
            )
        if (
            assigned.conventional_structure.fingerprint
            != standardized.conventional_structure.fingerprint
            or assigned.primitive_structure.fingerprint
            != standardized.primitive_structure.fingerprint
        ):
            raise InvariantViolation(
                "exact target assigner changed a standardized structure array"
            )
        return assigned


def _quantized_fraction(value: float, *, decimals: int) -> Fraction:
    if not math.isfinite(value):
        raise DomainError("cannot quantize a nonfinite crystallographic coordinate")
    scale = 10**decimals
    numerator = int(np.rint(float(value) * scale))
    return fraction_mod_one(Fraction(numerator, scale))


def _quantized_point(point: npt.ArrayLike, *, decimals: int) -> FractionVector:
    value = np.asarray(point, dtype=np.float64)
    if value.shape != (3,) or not np.all(np.isfinite(value)):
        raise DomainError("a crystallographic point must have three coordinates")
    return tuple(
        _quantized_fraction(float(component), decimals=decimals) for component in value
    )


def _origin_key(
    positions: np.ndarray,
    species: Sequence[int],
    *,
    decimals: int,
) -> tuple[tuple[tuple[str, str], tuple[float, float, float]], ...]:
    entries: list[tuple[tuple[str, str], tuple[float, float, float]]] = []
    for point, atom_species in zip(positions, species, strict=True):
        rounded = np.round(point, decimals=decimals)
        entries.append(
            (
                _species_key(int(atom_species)),
                (float(rounded[0]), float(rounded[1]), float(rounded[2])),
            )
        )
    return tuple(sorted(entries))


def _verify_canonical_free_origin(
    standardized: StandardizedRawStructure, *, decimals: int
) -> None:
    numerical = standardized.numerical_result
    positions = numerical.conventional_fractional_positions
    species = tuple(int(value) for value in numerical.conventional_species)
    axes = numerical.free_origin_axes
    current_key = _origin_key(positions, species, decimals=decimals)
    if not axes:
        if current_key != tuple(
            (
                _species_key(int(atom_species)),
                tuple(float(value) for value in np.round(point, decimals=decimals)),
            )
            for point, atom_species in zip(positions, species, strict=True)
        ):
            raise InvariantViolation(
                "fixed-origin conventional motif is not canonically sorted"
            )
        return
    least_species = min(species, key=_species_key)
    candidates: list[
        tuple[tuple[tuple[str, str], tuple[float, float, float]], ...]
    ] = []
    for point, atom_species in zip(positions, species, strict=True):
        if atom_species != least_species:
            continue
        shift = np.zeros(3, dtype=np.float64)
        shift[list(axes)] = point[list(axes)]
        shifted = np.mod(positions - shift[None, :], 1.0)
        shifted[np.isclose(shifted, 1.0, atol=10.0 ** (-decimals), rtol=0.0)] = 0.0
        candidates.append(_origin_key(shifted, species, decimals=decimals))
    if not candidates or current_key != min(candidates):
        raise InvariantViolation("continuous free-origin section is not canonical")


def _chart_parameter_candidates(
    row: WyckoffPosition,
    point: np.ndarray,
    *,
    decimals: int,
    fractional_tolerance: float,
) -> tuple[FractionVector, ...]:
    offset = np.asarray(row.representative_offset, dtype=np.float64)
    if row.dimension == 0:
        residual = point - offset
        residual -= np.rint(residual)
        if float(np.linalg.norm(residual)) <= fractional_tolerance:
            return ((),)
        return ()
    matrix = np.asarray(row.coordinate_matrix, dtype=np.float64)
    minimum = np.minimum(matrix, 0.0).sum(axis=0)
    maximum = np.maximum(matrix, 0.0).sum(axis=0)
    delta = point - offset
    ranges = tuple(
        range(
            math.floor(float(minimum[axis] - delta[axis])) - 1,
            math.ceil(float(maximum[axis] - delta[axis])) + 2,
        )
        for axis in range(3)
    )
    candidates: set[FractionVector] = set()
    for shift in product(*ranges):
        target = delta + np.asarray(shift, dtype=np.float64)
        try:
            parameter, _, rank, _ = np.linalg.lstsq(matrix.T, target, rcond=None)
        except np.linalg.LinAlgError as exc:
            raise InvariantViolation("Wyckoff chart inversion linear solve failed") from exc
        if rank != row.dimension:
            raise CatalogueError(f"Wyckoff chart {row.label} is numerically rank deficient")
        parameter = np.mod(parameter, 1.0)
        residual = point - np.mod(offset + parameter @ matrix, 1.0)
        residual -= np.rint(residual)
        if float(np.linalg.norm(residual)) > fractional_tolerance:
            continue
        exact = tuple(
            _quantized_fraction(float(value), decimals=decimals) for value in parameter
        )
        candidates.add(exact)
    return tuple(sorted(candidates))


def _invert_wyckoff_orbit(
    row: WyckoffPosition,
    orbit_positions: np.ndarray,
    lattice: np.ndarray,
    *,
    atomic_species: int,
    decimals: int,
    tolerance: float,
    merge_quantization_equivalent: bool = False,
) -> FractionVector:
    sigma_min = float(np.linalg.svd(lattice, compute_uv=False)[-1])
    if not math.isfinite(sigma_min) or sigma_min <= 0.0:
        raise InvariantViolation("standardized conventional lattice is singular")
    fractional_tolerance = tolerance / sigma_min
    exact_orbits: dict[frozenset[FractionVector], list[FractionVector]] = {}
    actual = RawPeriodicStructure.from_arrays(
        lattice,
        orbit_positions,
        [atomic_species] * len(orbit_positions),
    )
    for point in orbit_positions:
        for parameter in _chart_parameter_candidates(
            row,
            point,
            decimals=decimals,
            fractional_tolerance=fractional_tolerance,
        ):
            try:
                expanded = row.expand_exact(parameter, require_declared_multiplicity=True)
            except InvariantViolation:
                continue
            try:
                error = _coloured_periodic_roundtrip_error(
                    actual,
                    np.asarray(
                        [
                            [float(value) for value in expanded_point]
                            for expanded_point in expanded
                        ],
                        dtype=np.float64,
                    ),
                    (atomic_species,) * len(expanded),
                    tolerance=tolerance,
                )
            except InvariantViolation:
                continue
            if error <= tolerance:
                exact_orbits.setdefault(frozenset(expanded), []).append(parameter)
    if not exact_orbits:
        raise AmbiguousPreprocessingError(
            f"standardized orbit has no exact inverse in Wyckoff chart {row.label}"
        )
    if len(exact_orbits) != 1 and merge_quantization_equivalent:
        # Representatives quantized to ``decimals`` digits need not be exactly
        # group-consistent (e.g. thirds in rhombohedral settings), so one orbit
        # can expand to several exact point sets that differ only at the
        # quantization scale.  They are one orbit iff every pair coincides
        # within 100 * 10**-decimals * sigma_max(L), far below ``tolerance``.
        resolution = 100.0 * 10.0 ** (-decimals) * float(
            np.linalg.svd(lattice, compute_uv=False)[0]
        )
        if resolution >= tolerance:
            raise InvariantViolation("orbit quantization resolution exceeds the tolerance")
        orbits = list(exact_orbits)
        if all(
            _exact_orbit_distance(left, right, lattice) <= resolution
            for i, left in enumerate(orbits)
            for right in orbits[i + 1 :]
        ):
            merged: list[FractionVector] = []
            for values in exact_orbits.values():
                merged.extend(values)
            return min(merged)
    if len(exact_orbits) != 1:
        raise AmbiguousPreprocessingError(
            f"standardized orbit is ambiguous in Wyckoff chart {row.label}"
        )
    parameters = next(iter(exact_orbits.values()))
    return min(parameters)


def _exact_orbit_distance(
    left: frozenset[FractionVector], right: frozenset[FractionVector], lattice: np.ndarray
) -> float:
    """Symmetric periodic Hausdorff distance between two exact point sets (A)."""

    def as_array(points: frozenset[FractionVector]) -> list[np.ndarray]:
        return [np.asarray([float(value) for value in point]) for point in points]

    def directed(source: list[np.ndarray], target: list[np.ndarray]) -> float:
        worst = 0.0
        for point in source:
            nearest = min(
                closest_periodic_image(point - other, lattice).squared_distance
                for other in target
            )
            worst = max(worst, float(np.sqrt(nearest)))
        return worst

    first, second = as_array(left), as_array(right)
    return max(directed(first, second), directed(second, first))


class CatalogueTargetAssigner:
    """Invert pinned standardized orbits into exact full-catalogue scaffolds."""

    def __init__(
        self,
        catalogue: CertifiedCatalogue | None = None,
        *,
        rational_decimals: int = 12,
        roundtrip_tolerance: float = 2.0e-8,
        merge_quantization_equivalent_orbits: bool = False,
    ) -> None:
        """``merge_quantization_equivalent_orbits=False`` is the legacy v6 rule.

        When true, exact orbits that coincide up to the quantization scale of
        ``rational_decimals`` are one orbit rather than an ambiguity.
        """

        selected = build_full_catalogue() if catalogue is None else catalogue
        if not isinstance(selected, CertifiedCatalogue):
            raise TypeError("catalogue must be a CertifiedCatalogue")
        selected.validate(require_all_230=True)
        if not 6 <= rational_decimals <= 15:
            raise DomainError("rational_decimals must lie between 6 and 15")
        if not math.isfinite(roundtrip_tolerance) or roundtrip_tolerance <= 0.0:
            raise DomainError("roundtrip_tolerance must be finite and positive")
        self.catalogue = selected
        self.rational_decimals = rational_decimals
        self.roundtrip_tolerance = float(roundtrip_tolerance)
        if not isinstance(merge_quantization_equivalent_orbits, bool):
            raise DomainError("merge_quantization_equivalent_orbits must be Boolean")
        self.merge_quantization_equivalent_orbits = merge_quantization_equivalent_orbits
        encoded_halls = json.dumps(
            sorted(pinned_hall_numbers().items()), separators=(",", ":")
        ).encode()
        self.hall_map_fingerprint = sha256(encoded_halls).hexdigest()

    def _exact_primitive_basis(self, standardized: StandardizedRawStructure) -> FractionMatrix:
        raw = standardized.raw_structure
        numerical = standardized.numerical_result
        exact_translations = numerical.exact_raw_translation_subgroup
        basis = numerical.exact_input_to_raw_frame_primitive_basis
        inverse = numerical.exact_inverse_input_to_raw_frame_primitive_basis
        max_error = numerical.exact_translation_max_cartesian_error
        raw_frame_lattice = numerical.raw_frame_primitive_lattice
        frame_rotation = numerical.raw_frame_to_standard_rotation
        if (
            exact_translations is None
            or basis is None
            or inverse is None
            or max_error is None
            or raw_frame_lattice is None
            or frame_rotation is None
        ):
            raise InvariantViolation(
                "numerical FullPrim result lacks an exact translation certificate"
            )
        if len(exact_translations) != numerical.raw_translation_index:
            raise InvariantViolation(
                "exact raw translation subgroup disagrees with the FullPrim index"
            )
        if exact_determinant(basis) != Fraction(1, len(exact_translations)):
            raise InvariantViolation(
                "exact input-to-raw-frame-primitive basis violates the subgroup index"
            )
        exact_identity = tuple(
            tuple(
                sum(basis[row][inner] * inverse[inner][column] for inner in range(3))
                for column in range(3)
            )
            for row in range(3)
        )
        if exact_identity != (
            (Fraction(1), Fraction(0), Fraction(0)),
            (Fraction(0), Fraction(1), Fraction(0)),
            (Fraction(0), Fraction(0), Fraction(1)),
        ):
            raise InvariantViolation(
                "exact input-to-raw-frame-primitive basis and integer inverse disagree"
            )
        if max_error > numerical.provenance.symprec:
            raise InvariantViolation(
                "exact raw translation certificate exceeds the standardization tolerance"
            )
        action_error = 0.0
        for translation in exact_translations:
            translated = np.mod(
                raw.fractional_positions
                + np.asarray([float(value) for value in translation], dtype=np.float64)[
                    None, :
                ],
                1.0,
            )
            action_error = max(
                action_error,
                _coloured_periodic_roundtrip_error(
                    raw,
                    translated,
                    raw.atomic_numbers,
                    tolerance=numerical.provenance.symprec,
                ),
            )
        certificate_slack = (
            64.0
            * np.finfo(np.float64).eps
            * max(float(np.linalg.norm(raw.lattice, ord=2)), 1.0)
        )
        if action_error > max_error + certificate_slack:
            raise InvariantViolation(
                "exact raw translation certificate understates its motif-action error"
            )
        witness_basis = np.asarray(
            [[float(value) for value in row] for row in basis], dtype=np.float64
        )
        witness_lattice = witness_basis @ raw.lattice
        message = "exact raw-frame primitive basis/rotation certificate is inconsistent"
        if not np.allclose(
            witness_lattice,
            raw_frame_lattice,
            atol=self.roundtrip_tolerance,
            rtol=self.roundtrip_tolerance,
        ):
            raise InvariantViolation(message)
        # Raw against idealized: bounded by the standardizer's declared length.
        _require_lattice_fidelity(
            raw_frame_lattice @ frame_rotation,
            numerical.primitive_lattice,
            tolerance_angstrom=numerical.provenance.idealization_tolerance_angstrom,
            legacy_tolerance=self.roundtrip_tolerance,
            error=InvariantViolation,
            legacy_message=message,
            message=message,
        )
        if not idealized_volume_matches(
            abs(float(np.linalg.det(witness_lattice))),
            standardized.primitive_structure.volume,
            raw_primitive_lattice=witness_lattice,
            lattice_displacement_angstrom=_recorded_displacement(numerical),
        ):
            raise InvariantViolation(
                "exact primitive witness and standardized primitive volumes disagree"
            )
        return basis

    def assign(self, standardized: StandardizedRawStructure) -> CanonicalizedStructure:
        """Return an exact scaffold, rejecting every ambiguous chart inverse."""

        if not isinstance(standardized, StandardizedRawStructure):
            raise TypeError("standardized must be a StandardizedRawStructure")
        numerical = standardized.numerical_result
        provenance = numerical.provenance
        if (
            provenance.backend != "spglib"
            or provenance.backend_version != SPGLIB_SOURCE_VERSION
            or provenance.hall_map_fingerprint != self.hall_map_fingerprint
            or provenance.canonical_decimals != self.rational_decimals
        ):
            raise InvariantViolation(
                "numerical standardization identity disagrees with the exact catalogue"
            )
        setting = self.catalogue.get(numerical.international_number)
        if (
            numerical.hall_number != pinned_hall_numbers()[setting.international_number]
            or numerical.hall_symbol != setting.hall_symbol
        ):
            raise CatalogueError("standardized Hall setting disagrees with pinned catalogue")
        _verify_canonical_free_origin(standardized, decimals=self.rational_decimals)
        source_rows = source_rows_for_type(setting.international_number)

        orbit_ids = numerical.crystallographic_orbits
        first_indices: dict[int, int] = {}
        for index, orbit_id in enumerate(orbit_ids):
            first_indices.setdefault(orbit_id, index)
        assignments: list[OrbitAssignment] = []
        row_rank = {
            label: index
            for index, label in enumerate(
                sorted(
                    setting.wyckoff_positions,
                    key=lambda value: (
                        setting.wyckoff(value).dimension != 0,
                        value,
                    ),
                )
            )
        }
        for orbit_id, _ in sorted(first_indices.items(), key=lambda item: item[1]):
            indices = tuple(index for index, value in enumerate(orbit_ids) if value == orbit_id)
            letters = {numerical.wyckoff_letters[index] for index in indices}
            species = {numerical.conventional_species[index] for index in indices}
            site_symbols = {numerical.site_symmetry_symbols[index] for index in indices}
            if len(letters) != 1 or len(species) != 1 or len(site_symbols) != 1:
                raise InvariantViolation(
                    "one crystallographic orbit has inconsistent aligned metadata"
                )
            atom_species = next(iter(species))
            if isinstance(atom_species, bool) or not isinstance(atom_species, int):
                raise DomainError("catalogue target assignment requires atomic numbers")
            # Wyckoff letters are case-sensitive once a group has more than 26
            # rows: spglib uses uppercase ``A`` for the 27th letter (SG 47).
            reported_label = f"{len(indices)}{next(iter(letters))}"
            if reported_label not in setting.wyckoff_positions:
                raise CatalogueError(
                    "spglib reported a Wyckoff row absent from the pinned catalogue: "
                    f"{reported_label}"
                )
            source_row = source_rows.get(reported_label)
            if source_row is None:
                raise CatalogueError(
                    "the pinned upstream source is missing the reported Wyckoff row: "
                    f"{reported_label}"
                )

            # Whitespace has no crystallographic content in the printed
            # site-symmetry symbol; no other normalization is permitted.
            reported_site_symbol = "".join(next(iter(site_symbols)).split())
            source_site_symbol = "".join(source_row.site_symmetry_symbol.split())
            if reported_site_symbol != source_site_symbol:
                raise CatalogueError(
                    "spglib site-symmetry metadata disagrees with the pinned source "
                    f"for {reported_label}: {reported_site_symbol!r} != "
                    f"{source_site_symbol!r}"
                )
            row = setting.wyckoff(reported_label)
            parameter = _invert_wyckoff_orbit(
                row,
                numerical.conventional_fractional_positions[np.asarray(indices)],
                numerical.conventional_lattice,
                atomic_species=atom_species,
                decimals=self.rational_decimals,
                tolerance=self.roundtrip_tolerance,
                merge_quantization_equivalent=self.merge_quantization_equivalent_orbits,
            )
            assignments.append(OrbitAssignment(reported_label, atom_species, parameter))
        assignments.sort(
            key=lambda item: (
                row_rank[item.wyckoff_label],
                int(item.species),
                item.parameter,
            )
        )

        counts = Counter(standardized.conventional_structure.atomic_numbers)
        formula_units = 0
        for count in counts.values():
            formula_units = gcd(formula_units, count)
        if formula_units < 1:
            raise InvariantViolation("cannot reduce an empty conventional composition")
        reduced_composition: dict[int | str, int] = {
            species_number: count // formula_units for species_number, count in counts.items()
        }
        scaffold = CrystalScaffold.from_mapping(
            nu=formula_units,
            space_group=setting.key,
            reduced_composition=reduced_composition,
            orbits=assignments,
        )
        decoded = ExactCrystalDecoder(
            self.catalogue, metric_tolerance=self.roundtrip_tolerance
        ).decode(scaffold, standardized.conventional_structure.metric)
        if decoded.hidden_translation_index != 1:
            raise InvariantViolation(
                "assigned conventional scaffold retains an undeclared hidden translation"
            )
        roundtrip_error = _coloured_periodic_roundtrip_error(
            standardized.conventional_structure,
            decoded.fractional_positions_float,
            tuple(int(value) for value in decoded.species),
            tolerance=self.roundtrip_tolerance,
        )
        source_to_primitive = self._exact_primitive_basis(standardized)
        origin_shift = _quantized_point(
            numerical.spglib_origin_shift,
            decimals=self.rational_decimals,
        )
        free_origin_shift = _quantized_point(
            numerical.subtracted_free_origin,
            decimals=self.rational_decimals,
        )
        receipt = CanonicalizationReceipt(
            source_fingerprint=standardized.raw_structure.fingerprint,
            standardization_fingerprint=provenance.fingerprint,
            catalogue_fingerprint=self.catalogue.fingerprint,
            hall_number=numerical.hall_number,
            hall_symbol=numerical.hall_symbol,
            source_to_primitive=source_to_primitive,
            origin_shift=origin_shift,
            free_origin_shift=free_origin_shift,
            translation_index=numerical.raw_translation_index,
            symmetry_tolerance=self.roundtrip_tolerance,
            roundtrip_max_cartesian_error=roundtrip_error,
            symmetry_stability_verified=provenance.symmetry_stability_verified,
            catalogue_closure_verified=True,
            wyckoff_multiplicity_verified=True,
            symmetry_verified=decoded.audit.selected_group_is_contained,
            canonical_origin_verified=True,
            scaffold_roundtrip_verified=True,
            # Recorded only under the dimensioned policy, so legacy receipts
            # and their fingerprints stay bit-identical.
            idealization_lattice_displacement_angstrom=(
                numerical.idealization_lattice_displacement_angstrom
                if provenance.idealization_tolerance_angstrom is not None
                else None
            ),
            idealization_site_displacement_angstrom=(
                numerical.idealization_site_displacement_angstrom
                if provenance.idealization_tolerance_angstrom is not None
                else None
            ),
        )
        return CanonicalizedStructure(
            standardized.conventional_structure,
            standardized.primitive_structure,
            scaffold,
            receipt,
        )


@dataclass(frozen=True, slots=True)
class CanonicalTrainingRecord(Generic[TargetT]):
    """Raw input, standardization proof, and deterministic supervised target."""

    raw_structure: RawPeriodicStructure
    canonicalized: CanonicalizedStructure
    target: TargetT


def _recorded_displacement(
    numerical: NumericallyCanonicalizedStructure,
) -> float | None:
    """Measured raw-to-ideal cell displacement, only under the dimensioned policy."""

    if numerical.provenance.idealization_tolerance_angstrom is None:
        return None
    return numerical.idealization_lattice_displacement_angstrom


def _validate_primitive_quotient(
    raw: RawPeriodicStructure, canonicalized: CanonicalizedStructure
) -> None:
    receipt = canonicalized.receipt
    primitive = canonicalized.primitive_structure
    if receipt.source_fingerprint != raw.fingerprint:
        raise InvariantViolation("canonicalization receipt belongs to a different raw input")
    if not receipt.passed:
        raise InvariantViolation("canonicalization proof gates did not all pass")
    index = receipt.translation_index
    if len(raw.atomic_numbers) != index * len(primitive.atomic_numbers):
        raise InvariantViolation(
            "translation index does not relate raw and primitive atom counts"
        )
    raw_counts = Counter(raw.atomic_numbers)
    primitive_counts = Counter(primitive.atomic_numbers)
    expected = Counter({species: index * count for species, count in primitive_counts.items()})
    if raw_counts != expected:
        raise InvariantViolation(
            "primitive reduction changed the coloured composition quotient"
        )
    conventional = canonicalized.conventional_structure
    if len(conventional.atomic_numbers) % len(primitive.atomic_numbers):
        raise InvariantViolation(
            "primitive atom count does not divide the conventional atom count"
        )
    centering_index = len(conventional.atomic_numbers) // len(primitive.atomic_numbers)
    _require_indexed_composition_quotient(
        conventional.atomic_numbers,
        primitive.atomic_numbers,
        index=centering_index,
        name="conventional-to-primitive canonicalization",
    )
    if not np.isclose(
        conventional.volume,
        centering_index * primitive.volume,
        atol=1.0e-10,
        rtol=1.0e-8,
    ):
        raise InvariantViolation("conventional-to-primitive volume/index equation failed")


def _integer_atomic_numbers(values: Sequence[object], *, name: str) -> tuple[int, ...]:
    result: list[int] = []
    for value in values:
        if isinstance(value, bool) or not isinstance(value, int):
            raise InvariantViolation(f"{name} contains a non-integer chemical species")
        try:
            result.append(atomic_number(value))
        except DomainError as exc:
            raise InvariantViolation(f"{name} contains an invalid atomic number") from exc
    return tuple(result)


def _require_indexed_composition_quotient(
    larger: Sequence[int], smaller: Sequence[int], *, index: int, name: str
) -> None:
    expected = Counter({species: index * count for species, count in Counter(smaller).items()})
    if Counter(larger) != expected:
        raise InvariantViolation(f"{name} changed a coloured composition quotient")


def standardize_raw_structure(
    structure: RawPeriodicStructure,
    *,
    standardizer: NumericalStandardizationBackend,
) -> StandardizedRawStructure:
    """Run the pinned numerical standardizer and revalidate both output cells."""

    if not isinstance(structure, RawPeriodicStructure):
        raise TypeError("structure must be a RawPeriodicStructure")
    floating = FloatingStructure(
        structure.lattice,
        structure.fractional_positions,
        structure.atomic_numbers,
    )
    numerical = standardizer.canonicalize(floating)
    if not isinstance(numerical, NumericallyCanonicalizedStructure):
        raise TypeError("numerical standardizer returned an invalid result type")
    conventional_numbers = _integer_atomic_numbers(
        numerical.conventional_species, name="conventional standardization"
    )
    primitive_numbers = _integer_atomic_numbers(
        numerical.primitive_species, name="primitive standardization"
    )
    conventional = RawPeriodicStructure.from_arrays(
        numerical.conventional_lattice,
        numerical.conventional_fractional_positions,
        conventional_numbers,
        source_id=f"standardized:{numerical.provenance.fingerprint}:conventional",
    )
    primitive = RawPeriodicStructure.from_arrays(
        numerical.primitive_lattice,
        numerical.primitive_fractional_positions,
        primitive_numbers,
        source_id=f"standardized:{numerical.provenance.fingerprint}:primitive",
    )

    raw_index = numerical.raw_translation_index
    if raw_index < 1:
        raise InvariantViolation("numerical standardizer returned an empty translation group")
    _require_indexed_composition_quotient(
        structure.atomic_numbers,
        primitive.atomic_numbers,
        index=raw_index,
        name="raw-to-primitive standardization",
    )
    if len(conventional.atomic_numbers) % len(primitive.atomic_numbers):
        raise InvariantViolation("primitive atom count does not divide conventional count")
    centering_index = len(conventional.atomic_numbers) // len(primitive.atomic_numbers)
    _require_indexed_composition_quotient(
        conventional.atomic_numbers,
        primitive.atomic_numbers,
        index=centering_index,
        name="conventional-to-primitive standardization",
    )
    # Raw against idealized: bounded by the recorded idealization displacement.
    if not idealized_volume_matches(
        structure.volume,
        raw_index * primitive.volume,
        raw_primitive_lattice=numerical.raw_frame_primitive_lattice,
        lattice_displacement_angstrom=_recorded_displacement(numerical),
    ):
        raise InvariantViolation("raw-to-primitive volume/index equation failed")
    # Idealized against idealized: exact up to float rounding.
    if not np.isclose(
        conventional.volume, centering_index * primitive.volume, atol=1.0e-10, rtol=1.0e-8
    ):
        raise InvariantViolation("conventional-to-primitive volume/index equation failed")
    provenance = numerical.provenance
    if (
        provenance.input_atom_count != len(structure.atomic_numbers)
        or provenance.primitive_atom_count != len(primitive.atomic_numbers)
        or provenance.conventional_atom_count != len(conventional.atomic_numbers)
    ):
        raise InvariantViolation("standardization provenance atom counts are inconsistent")
    return StandardizedRawStructure(structure, numerical, conventional, primitive)


def structure_to_training_target(
    structure: RawPeriodicStructure,
    *,
    canonicalizer: CanonicalizationBackend,
    target_lifter: Callable[[CrystalScaffold, npt.ArrayLike], TargetT],
) -> CanonicalTrainingRecord[TargetT]:
    """Canonicalize one validated structure and lift it only after all proof gates."""

    if not isinstance(structure, RawPeriodicStructure):
        raise TypeError("structure must be a RawPeriodicStructure")
    canonicalized = canonicalizer.canonicalize(structure)
    if not isinstance(canonicalized, CanonicalizedStructure):
        raise TypeError("canonicalizer returned an invalid result type")
    _validate_primitive_quotient(structure, canonicalized)
    target = target_lifter(
        canonicalized.scaffold,
        canonicalized.conventional_structure.metric,
    )
    return CanonicalTrainingRecord(structure, canonicalized, target)


def structure_to_exact_fixture_target(
    structure: RawPeriodicStructure,
    *,
    canonicalizer: CanonicalizationBackend,
) -> CanonicalTrainingRecord[ExactFixtureTrainingTarget]:
    """Use the current certified P1/P-1 post-standardization target lift."""

    record = structure_to_training_target(
        structure,
        canonicalizer=canonicalizer,
        target_lifter=lift_exact_fixture_target,
    )
    # This wrapper can independently check the current exact fixture lift.  A
    # future all-group lifter must provide the same comparison for its catalogue.
    decoded = ExactCrystalDecoder().decode(
        record.target.canonical_scaffold, record.target.metric
    )
    _assert_same_coloured_periodic_set(
        record.canonicalized.conventional_structure,
        decoded.fractional_positions_float,
        tuple(int(value) for value in decoded.species),
        tolerance=record.canonicalized.receipt.symmetry_tolerance,
    )
    return record


def structure_to_catalogue_training_target(
    structure: RawPeriodicStructure,
    *,
    standardizer: NumericalStandardizationBackend,
    target_assigner: CatalogueTargetAssigner | None = None,
    domain: DomainConfig | None = None,
) -> CanonicalTrainingRecord[CanonicalCatalogueTrainingTarget]:
    """Run FullPrim, exact orbit inversion, and the complete catalogue lift."""

    assigner = CatalogueTargetAssigner() if target_assigner is None else target_assigner
    canonicalizer = ComposedCanonicalizer(standardizer, assigner)
    canonicalized = canonicalizer.canonicalize(structure)
    _validate_primitive_quotient(structure, canonicalized)
    target = lift_canonical_catalogue_target(
        canonicalized.scaffold,
        canonicalized.conventional_structure.metric,
        catalogue=assigner.catalogue,
        domain=domain,
    )
    decoded = ExactCrystalDecoder(assigner.catalogue).decode(
        target.canonical_scaffold, target.metric
    )
    _assert_same_coloured_periodic_set(
        canonicalized.conventional_structure,
        decoded.fractional_positions_float,
        tuple(int(value) for value in decoded.species),
        tolerance=canonicalized.receipt.symmetry_tolerance,
    )
    return CanonicalTrainingRecord(structure, canonicalized, target)


def _assert_same_coloured_periodic_set(
    structure: RawPeriodicStructure,
    expected_positions: npt.ArrayLike,
    expected_species: Sequence[int],
    *,
    tolerance: float,
) -> None:
    """Require a species-preserving perfect matching under periodic distance."""

    _coloured_periodic_roundtrip_error(
        structure,
        expected_positions,
        expected_species,
        tolerance=tolerance,
    )


def _coloured_periodic_roundtrip_error(
    structure: RawPeriodicStructure,
    expected_positions: npt.ArrayLike,
    expected_species: Sequence[int],
    *,
    tolerance: float,
) -> float:
    """Return the largest distance in a valid coloured periodic matching."""

    expected = np.asarray(expected_positions, dtype=np.float64)
    species = tuple(expected_species)
    if expected.shape != (len(species), 3) or not np.all(np.isfinite(expected)):
        raise InvariantViolation("decoded scaffold returned invalid point arrays")
    if Counter(structure.atomic_numbers) != Counter(species):
        raise InvariantViolation("decoded scaffold changed primitive species counts")
    maximum_error = 0.0
    for atomic_species in sorted(set(species)):
        actual_indices = [
            index
            for index, value in enumerate(structure.atomic_numbers)
            if value == atomic_species
        ]
        expected_indices = [
            index for index, value in enumerate(species) if value == atomic_species
        ]
        adjacency: list[list[int]] = []
        distances: list[list[float]] = []
        for actual_index in actual_indices:
            candidates: list[int] = []
            candidate_distances: list[float] = []
            for local_expected, expected_index in enumerate(expected_indices):
                image = closest_periodic_image(
                    structure.fractional_positions[actual_index] - expected[expected_index],
                    structure.lattice,
                )
                distance = math.sqrt(image.squared_distance)
                candidate_distances.append(distance)
                if image.squared_distance <= tolerance * tolerance:
                    candidates.append(local_expected)
            adjacency.append(candidates)
            distances.append(candidate_distances)
        matching = _perfect_matching(adjacency)
        if matching is None:
            raise InvariantViolation(
                "canonical scaffold failed the coloured periodic round-trip"
            )
        for expected_local, actual_local in enumerate(matching):
            maximum_error = max(maximum_error, distances[actual_local][expected_local])
    return maximum_error


def _perfect_matching(adjacency: Sequence[Sequence[int]]) -> tuple[int, ...] | None:
    """Return a right-to-left perfect matching for a square bipartite graph."""

    matched_left = [-1] * len(adjacency)

    def augment(left: int, visited: set[int]) -> bool:
        for right in adjacency[left]:
            if right in visited:
                continue
            visited.add(right)
            previous = matched_left[right]
            if previous < 0 or augment(previous, visited):
                matched_left[right] = left
                return True
        return False

    if not all(augment(index, set()) for index in range(len(adjacency))):
        return None
    return tuple(matched_left)


def _tokenize_cif(text: str) -> list[str]:
    if not isinstance(text, str) or not text.strip():
        raise DomainError("CIF input must be a nonempty string")
    if "\x00" in text:
        raise DomainError("CIF input contains a NUL byte")
    tokens: list[str] = []
    for line_number, line in enumerate(text.splitlines(), start=1):
        if line.startswith(";"):
            raise DomainError(f"CIF semicolon text fields are unsupported (line {line_number})")
        lexer = shlex.shlex(line, posix=True)
        lexer.whitespace_split = True
        lexer.commenters = "#"
        try:
            tokens.extend(lexer)
        except ValueError as exc:
            raise DomainError(f"malformed quoted CIF value on line {line_number}") from exc
    return tokens


def _parse_cif_tables(
    tokens: Sequence[str],
) -> tuple[str, dict[str, str], tuple[CIFLoop, ...]]:
    scalars: dict[str, str] = {}
    loops: list[CIFLoop] = []
    block_name: str | None = None
    index = 0
    while index < len(tokens):
        token = tokens[index]
        lowered = token.lower()
        if lowered.startswith("data_"):
            if block_name is not None:
                raise DomainError("exactly one CIF data block is required")
            block_name = token[5:]
            if not block_name:
                raise DomainError("CIF data block name cannot be empty")
            index += 1
            continue
        if block_name is None:
            raise DomainError("CIF content must begin with a data_ block")
        if lowered == "loop_":
            index += 1
            tags: list[str] = []
            while index < len(tokens) and tokens[index].startswith("_"):
                tags.append(tokens[index].lower())
                index += 1
            if not tags or len(set(tags)) != len(tags):
                raise DomainError("CIF loop tags must be nonempty and unique")
            values: list[str] = []
            while index < len(tokens):
                candidate = tokens[index]
                candidate_lower = candidate.lower()
                if (
                    candidate.startswith("_")
                    or candidate_lower == "loop_"
                    or candidate_lower.startswith("data_")
                    or candidate_lower.startswith("save_")
                ):
                    break
                values.append(candidate)
                index += 1
            if not values or len(values) % len(tags):
                raise DomainError("CIF loop value count is not a multiple of its tag count")
            rows = tuple(
                tuple(values[start : start + len(tags)])
                for start in range(0, len(values), len(tags))
            )
            loops.append((tuple(tags), rows))
            continue
        if token.startswith("_"):
            tag = lowered
            if tag in scalars:
                raise DomainError(f"duplicate CIF scalar tag {tag}")
            if index + 1 >= len(tokens):
                raise DomainError(f"CIF scalar tag {tag} has no value")
            value = tokens[index + 1]
            value_lower = value.lower()
            if (
                value.startswith("_")
                or value_lower == "loop_"
                or value_lower.startswith("data_")
                or value_lower.startswith("save_")
            ):
                raise DomainError(f"CIF scalar tag {tag} has no value")
            scalars[tag] = value
            index += 2
            continue
        raise DomainError(f"unsupported free CIF token {token!r}")
    if block_name is None:
        raise DomainError("CIF has no data block")
    return block_name, scalars, tuple(loops)


def _cif_fraction(value: str, *, field: str) -> Fraction:
    if value in {".", "?"}:
        raise DomainError(f"CIF field {field} is unknown")
    match = _CIF_NUMBER.fullmatch(value)
    if match is None:
        raise DomainError(f"CIF field {field} is not a finite decimal number")
    exponent = match.group(2) or ""
    try:
        result = Fraction(match.group(1) + exponent)
    except (ValueError, ZeroDivisionError) as exc:
        raise DomainError(f"CIF field {field} is not a valid number") from exc
    return result


def _required_cell_lattice(scalars: Mapping[str, str]) -> FloatArray:
    field_names = (
        "_cell_length_a",
        "_cell_length_b",
        "_cell_length_c",
        "_cell_angle_alpha",
        "_cell_angle_beta",
        "_cell_angle_gamma",
    )
    missing = [name for name in field_names if name not in scalars]
    if missing:
        raise DomainError(f"CIF is missing cell fields: {missing}")
    values = [float(_cif_fraction(scalars[name], field=name)) for name in field_names]
    a, b, c, alpha_degrees, beta_degrees, gamma_degrees = values
    if min(a, b, c) <= 0.0:
        raise DomainError("CIF cell lengths must be positive")
    if any(not 0.0 < angle < 180.0 for angle in values[3:]):
        raise DomainError("CIF cell angles must lie strictly between 0 and 180 degrees")
    alpha, beta, gamma = map(math.radians, values[3:])
    cos_alpha, cos_beta, cos_gamma = map(math.cos, (alpha, beta, gamma))
    sin_gamma = math.sin(gamma)
    if abs(sin_gamma) <= 64.0 * np.finfo(np.float64).eps:
        raise DomainError("CIF cell has a degenerate gamma angle")
    c_x = c * cos_beta
    c_y = c * (cos_alpha - cos_beta * cos_gamma) / sin_gamma
    c_z_squared = c * c - c_x * c_x - c_y * c_y
    tolerance = 256.0 * np.finfo(np.float64).eps * c * c
    if c_z_squared <= tolerance:
        raise DomainError("CIF cell parameters do not define a positive volume")
    lattice = np.asarray(
        (
            (a, 0.0, 0.0),
            (b * cos_gamma, b * sin_gamma, 0.0),
            (c_x, c_y, math.sqrt(c_z_squared)),
        ),
        dtype=np.float64,
    )
    if not np.all(np.linalg.eigvalsh(lattice @ lattice.T) > 0.0):
        raise DomainError("CIF cell metric is not positive definite")
    return lattice


def _find_unique_loop(
    loops: Sequence[CIFLoop],
    required_tags: frozenset[str],
    *,
    description: str,
) -> tuple[tuple[str, ...], tuple[tuple[str, ...], ...]] | None:
    matches = [loop for loop in loops if required_tags.issubset(loop[0])]
    if len(matches) > 1:
        raise DomainError(f"CIF contains multiple {description} loops")
    return matches[0] if matches else None


def _parse_symmetry_component(component: str) -> tuple[tuple[int, int, int], Fraction]:
    expression = component.lower().replace(" ", "")
    if not expression:
        raise DomainError("empty component in CIF symmetry operation")
    if expression[0] not in "+-":
        expression = "+" + expression
    terms = re.findall(r"[+-][^+-]+", expression)
    if "".join(terms) != expression:
        raise DomainError(f"unsupported CIF symmetry expression {component!r}")
    coefficients = [0, 0, 0]
    translation = Fraction(0)
    for term in terms:
        sign = 1 if term[0] == "+" else -1
        body = term[1:]
        if body in _SYMMETRY_VARIABLES:
            variable = _SYMMETRY_VARIABLES[body]
            coefficients[variable] += sign
            if abs(coefficients[variable]) > 1:
                raise DomainError(f"repeated variable in CIF symmetry expression {component!r}")
        else:
            try:
                translation += sign * Fraction(body)
            except (ValueError, ZeroDivisionError) as exc:
                raise DomainError(f"unsupported CIF symmetry expression {component!r}") from exc
    return (coefficients[0], coefficients[1], coefficients[2]), translation


def _parse_symmetry_operation(value: str) -> AffineOperation:
    components = value.split(",")
    if len(components) != 3:
        raise DomainError(f"CIF symmetry operation must have three components: {value!r}")
    parsed = tuple(_parse_symmetry_component(component) for component in components)
    # CIF lists output-coordinate expressions.  AffineOperation uses row
    # coordinates f' = f W + w, so expression coefficient vectors are columns.
    rotation = tuple(
        tuple(parsed[output][0][source] for output in range(3)) for source in range(3)
    )
    translation = tuple(item[1] for item in parsed)
    return AffineOperation(rotation, translation)


def _declares_p1(scalars: Mapping[str, str]) -> bool:
    tags = (
        "_space_group_name_hall",
        "_symmetry_space_group_name_hall",
        "_space_group_name_h-m_alt",
        "_symmetry_space_group_name_h-m",
    )
    values = [scalars[tag] for tag in tags if tag in scalars]
    if not values:
        return False
    compact = {re.sub(r"[ _]", "", value).lower() for value in values}
    if "p1" in compact and compact != {"p1"}:
        raise DomainError("CIF space-group name fields contradict a P1 declaration")
    return compact.issubset({"p1"})


def _symmetry_operations(
    scalars: Mapping[str, str],
    loops: Sequence[CIFLoop],
) -> tuple[AffineOperation, ...]:
    operations: tuple[AffineOperation, ...]
    operation_tags = (
        "_space_group_symop_operation_xyz",
        "_symmetry_equiv_pos_as_xyz",
    )
    matches: list[tuple[tuple[str, ...], tuple[tuple[str, ...], ...], str]] = []
    for tags, rows in loops:
        for operation_tag in operation_tags:
            if operation_tag in tags:
                matches.append((tags, rows, operation_tag))
    if len(matches) > 1:
        raise DomainError("CIF contains multiple symmetry-operation loops")
    if not matches:
        scalar_values = [scalars[tag] for tag in operation_tags if tag in scalars]
        if len(scalar_values) > 1:
            raise DomainError("CIF contains duplicate scalar symmetry operations")
        if scalar_values:
            operations = (_parse_symmetry_operation(scalar_values[0]),)
        elif _declares_p1(scalars):
            operations = (AffineOperation.identity(),)
        else:
            raise DomainError(
                "non-P1 or undeclared CIF requires an explicit symmetry-operation loop"
            )
    else:
        tags, rows, operation_tag = matches[0]
        operation_index = tags.index(operation_tag)
        operations = tuple(_parse_symmetry_operation(row[operation_index]) for row in rows)
    try:
        validated = validate_finite_affine_group(operations)
    except InvariantViolation as exc:
        raise DomainError("CIF symmetry operations are not an exact finite group") from exc
    if _declares_p1(scalars) and validated != (AffineOperation.identity(),):
        raise DomainError("CIF P1 declaration contradicts its symmetry operations")
    return validated


def _atom_species(value: str, *, label_fallback: bool) -> int:
    matcher = _ELEMENT_LABEL if label_fallback else _ELEMENT_TYPE
    match = matcher.match(value) if label_fallback else matcher.fullmatch(value)
    if match is None:
        source = "label" if label_fallback else "type symbol"
        raise DomainError(f"cannot infer an unambiguous element from CIF atom-site {source}")
    return atomic_number(match.group(1))


def parse_cif_structure(text: str, *, source_id: str | None = None) -> RawPeriodicStructure:
    """Parse and exactly symmetry-expand the supported, strict CIF subset.

    Every atom row must state occupancy exactly equal to one.  This prevents a
    missing value or a nearly-one disordered occupancy from entering an integer
    atom-count model.  Cartesian-only atom loops, magnetic operations, multiple
    data blocks, and semicolon text fields are outside this reader's certified
    subset and fail closed.
    """

    block_name, scalars, loops = _parse_cif_tables(_tokenize_cif(text))
    lattice = _required_cell_lattice(scalars)
    operations = _symmetry_operations(scalars, loops)
    # Affine closure alone is insufficient: the declared operations must also
    # be Euclidean isometries of the supplied cell.  In row convention their
    # Cartesian linear part is L^-1 W L.  Testing its orthogonality avoids an
    # Angstrom-dependent threshold and catches contradictory CIF cell/group
    # declarations before their asymmetric units are expanded.
    for operation in operations:
        cartesian_rotation = np.linalg.solve(
            lattice, np.asarray(operation.rotation, dtype=np.float64) @ lattice
        )
        if not np.allclose(
            cartesian_rotation @ cartesian_rotation.T,
            np.eye(3),
            atol=1.0e-8,
            rtol=0.0,
        ):
            raise DomainError(
                "CIF symmetry operation is incompatible with the cell metric "
                "(dimensionless isometry tolerance 1e-8)"
            )
    coordinate_tags = frozenset(
        {
            "_atom_site_fract_x",
            "_atom_site_fract_y",
            "_atom_site_fract_z",
            "_atom_site_occupancy",
        }
    )
    atom_loop = _find_unique_loop(loops, coordinate_tags, description="fractional atom-site")
    if atom_loop is None:
        raise DomainError("CIF needs one fractional atom-site loop with explicit occupancies")
    tags, rows = atom_loop
    type_tag = "_atom_site_type_symbol"
    label_tag = "_atom_site_label"
    if type_tag not in tags and label_tag not in tags:
        raise DomainError("CIF atom sites need a type symbol or label")
    disorder_tags = {
        "_atom_site_disorder_assembly",
        "_atom_site_disorder_group",
    }.intersection(tags)

    x_index = tags.index("_atom_site_fract_x")
    y_index = tags.index("_atom_site_fract_y")
    z_index = tags.index("_atom_site_fract_z")
    occupancy_index = tags.index("_atom_site_occupancy")
    type_index = tags.index(type_tag) if type_tag in tags else tags.index(label_tag)
    label_index = tags.index(label_tag) if label_tag in tags else None

    positions: list[FractionVector] = []
    numbers: list[int] = []
    labels: list[str] = []
    occupied_points: dict[FractionVector, tuple[int, str]] = {}
    for site_index, row in enumerate(rows):
        if any(row[tags.index(tag)] not in {".", "?"} for tag in disorder_tags):
            raise DomainError("CIF disorder groups are outside the integer-site domain")
        occupancy = _cif_fraction(row[occupancy_index], field="_atom_site_occupancy")
        if occupancy != 1:
            raise DomainError("only exactly unit-occupancy CIF atom sites are supported")
        coordinate = (
            fraction_mod_one(_cif_fraction(row[x_index], field="_atom_site_fract_x")),
            fraction_mod_one(_cif_fraction(row[y_index], field="_atom_site_fract_y")),
            fraction_mod_one(_cif_fraction(row[z_index], field="_atom_site_fract_z")),
        )
        species = _atom_species(row[type_index], label_fallback=type_tag not in tags)
        base_label = row[label_index] if label_index is not None else f"site{site_index + 1}"
        if not base_label or base_label in {".", "?"}:
            raise DomainError("CIF atom-site labels must be known when supplied")
        orbit = tuple(sorted({operation.apply_exact(coordinate) for operation in operations}))
        for orbit_index, point in enumerate(orbit, start=1):
            expanded_label = f"{base_label}#{orbit_index}"
            previous = occupied_points.get(point)
            if previous is not None:
                raise DomainError(
                    "distinct full-occupancy CIF sites overlap after symmetry expansion: "
                    f"{previous[1]!r} and {expanded_label!r}"
                )
            occupied_points[point] = (species, expanded_label)
            positions.append(point)
            numbers.append(species)
            labels.append(expanded_label)
    if not positions:
        raise DomainError("CIF atom-site loop is empty")
    fractional = np.asarray(
        [[float(component) for component in point] for point in positions],
        dtype=np.float64,
    )
    return RawPeriodicStructure.from_arrays(
        lattice,
        fractional,
        numbers,
        occupancies=np.ones(len(numbers), dtype=np.float64),
        labels=labels,
        source_id=source_id or block_name,
    )


def read_cif_structure(
    path: str | Path, *, max_bytes: int = 16 * 1024 * 1024
) -> RawPeriodicStructure:
    """Read one bounded UTF-8 CIF file and parse the certified subset."""

    if isinstance(max_bytes, bool) or not isinstance(max_bytes, int) or max_bytes <= 0:
        raise DomainError("max_bytes must be a positive integer")
    source = Path(path)
    try:
        size = source.stat().st_size
    except OSError as exc:
        raise DomainError(f"cannot inspect CIF file {source}") from exc
    if size > max_bytes:
        raise DomainError(f"CIF file exceeds the {max_bytes}-byte safety limit")
    try:
        text = source.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as exc:
        raise DomainError(f"cannot read CIF file {source} as UTF-8") from exc
    return parse_cif_structure(text, source_id=str(source.resolve()))


def cif_to_training_target(
    text: str,
    *,
    canonicalizer: CanonicalizationBackend,
    target_lifter: Callable[[CrystalScaffold, npt.ArrayLike], TargetT],
    source_id: str | None = None,
) -> CanonicalTrainingRecord[TargetT]:
    """Parse, standardize, verify, and lift one CIF without silent repair."""

    structure = parse_cif_structure(text, source_id=source_id)
    return structure_to_training_target(
        structure,
        canonicalizer=canonicalizer,
        target_lifter=target_lifter,
    )


def cif_to_catalogue_training_target(
    text: str,
    *,
    standardizer: NumericalStandardizationBackend,
    target_assigner: CatalogueTargetAssigner | None = None,
    domain: DomainConfig | None = None,
    source_id: str | None = None,
) -> CanonicalTrainingRecord[CanonicalCatalogueTrainingTarget]:
    """End-to-end strict CIF to complete-catalogue supervised target."""

    structure = parse_cif_structure(text, source_id=source_id)
    return structure_to_catalogue_training_target(
        structure,
        standardizer=standardizer,
        target_assigner=target_assigner,
        domain=domain,
    )


__all__ = [
    "CanonicalTrainingRecord",
    "CanonicalizationBackend",
    "CanonicalizationReceipt",
    "CanonicalizedStructure",
    "CatalogueTargetAssigner",
    "CatalogueTargetAssignmentBackend",
    "ComposedCanonicalizer",
    "NumericalStandardizationBackend",
    "RawPeriodicStructure",
    "StandardizedRawStructure",
    "cif_to_training_target",
    "cif_to_catalogue_training_target",
    "parse_cif_structure",
    "read_cif_structure",
    "standardize_raw_structure",
    "structure_to_exact_fixture_target",
    "structure_to_catalogue_training_target",
    "structure_to_training_target",
]
