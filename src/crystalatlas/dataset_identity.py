"""Certified canonical identities and lossless duplicate accounting.

The identity boundary is deliberately *after* crystallographic
standardization.  A raw cell is first reduced by its complete coloured
translation subgroup, moved to the pinned Hall setting and origin, inverted
to exact Wyckoff coordinates, and decoded again.  Only that certified object
may enter the functions below.  Consequently a supercell, an equivalent
integer basis and another presentation of the same setting have no direct
role in the resulting key.

There is no transitive, tolerance-based equality relation on floating point
structures.  CrystalAtlas therefore uses a versioned partition instead.  For
a scalar ``x`` and bin width ``q`` the integer coordinate is

``Q_q(x) = floor(x / q + 1/2)``.

Records whose uncertainty guard intersects a bin wall are rejected.  This
makes every accepted record a member of exactly one cell and prevents the
non-transitive ``A close to B, B close to C`` clustering failure.  Exact
Wyckoff fractions are partitioned on the torus with integer arithmetic.  The
full metric ``G`` defines the finite-resolution structure key, while

``G_shape = G / det(G)^(1/3)``

defines a scale-free polymorph key.  Reflections are never quotiented: the
package's proper-rotation/SO(3) chirality convention remains in force.
"""

from __future__ import annotations

import json
import math
from collections import Counter, defaultdict
from collections.abc import Sequence
from dataclasses import dataclass, field
from enum import Enum
from fractions import Fraction
from hashlib import sha256
from typing import TypeAlias

import numpy as np
import numpy.typing as npt

from .catalogue import CertifiedCatalogue, SpaceGroupSetting
from .cover import canonical_rows
from .decoder import CrystalScaffold, ExactCrystalDecoder
from .errors import DomainError, InvariantViolation
from .full_catalogue import pinned_hall_numbers
from .ingestion import CanonicalTrainingRecord, RawPeriodicStructure
from .periodic import closest_periodic_image
from .standardization import idealized_volume_matches
from .training_targets import (
    CanonicalCatalogueTrainingTarget,
    ExactFixtureTrainingTarget,
)

FloatArray = npt.NDArray[np.float64]
CanonicalTarget: TypeAlias = ExactFixtureTrainingTarget | CanonicalCatalogueTrainingTarget
CanonicalRecord: TypeAlias = CanonicalTrainingRecord[CanonicalTarget]

_SCHEMA = "crystalatlas.canonical-structure-identity.v2"
_METRIC_COMPONENTS = ((0, 0), (1, 1), (2, 2), (0, 1), (0, 2), (1, 2))


class AmbiguousIdentityError(DomainError):
    """A certified value intersects an identity-cell boundary."""


class DeduplicationLevel(str, Enum):  # noqa: UP042 - retained for Python-3.10 smoke jobs
    """The explicitly selected physical quotient used for duplicate removal."""

    EXACT = "exact"
    STRUCTURE = "structure"
    POLYMORPH = "polymorph"


@dataclass(frozen=True, slots=True)
class IdentityPolicy:
    """Frozen numerical partition and audit policy for dataset identities.

    ``STRUCTURE`` retains isotropic scale and quantizes the six independent
    components of the pinned conventional metric in square Angstroms.
    ``POLYMORPH`` instead quantizes its determinant-one shape, so uniformly
    scaled measurements share a scale-free *grouping* key.  That quotient can
    also merge real isostructural volume-collapse or pressure-dependent states;
    it is therefore opt-in for irreversible deduplication.  Fractional
    coordinates are treated on ``R/Z`` using exactly
    ``fractional_bin_count`` cells.
    """

    metric_quantum_angstrom2: float = 1.0e-6
    metric_boundary_guard_angstrom2: float = 1.0e-10
    shape_quantum: float = 1.0e-8
    shape_boundary_guard: float = 1.0e-12
    fractional_bin_count: int = 10_000_000
    fractional_boundary_guard: Fraction = Fraction(1, 1_000_000_000)
    max_roundtrip_tolerance_angstrom: float = 1.0e-6
    require_all_230: bool = True

    def __post_init__(self) -> None:
        scalar_pairs = (
            (
                self.metric_quantum_angstrom2,
                self.metric_boundary_guard_angstrom2,
                "metric",
            ),
            (self.shape_quantum, self.shape_boundary_guard, "shape"),
        )
        for quantum, guard, name in scalar_pairs:
            if not math.isfinite(quantum) or quantum <= 0.0:
                raise DomainError(f"{name} quantum must be finite and positive")
            if not math.isfinite(guard) or guard < 0.0 or 2.0 * guard >= quantum:
                raise DomainError(
                    f"{name} boundary guard must be nonnegative and less than half its quantum"
                )
        if (
            isinstance(self.fractional_bin_count, bool)
            or not isinstance(self.fractional_bin_count, int)
            or self.fractional_bin_count < 2
        ):
            raise DomainError("fractional_bin_count must be an integer of at least two")
        fractional_guard = self.fractional_boundary_guard
        if not isinstance(fractional_guard, Fraction) or fractional_guard < 0:
            raise DomainError("fractional_boundary_guard must be a nonnegative Fraction")
        if 2 * fractional_guard * self.fractional_bin_count >= 1:
            raise DomainError("fractional boundary guard must be less than half a torus cell")
        if (
            not math.isfinite(self.max_roundtrip_tolerance_angstrom)
            or self.max_roundtrip_tolerance_angstrom <= 0.0
        ):
            raise DomainError("maximum round-trip tolerance must be finite and positive")
        if not isinstance(self.require_all_230, bool):
            raise DomainError("require_all_230 must be Boolean")

    @property
    def fingerprint(self) -> str:
        payload = {
            "schema": f"{_SCHEMA}.policy.v2",
            "composition_projection": "validated-atomic-number-order.v1",
            "fractional_bin_count": self.fractional_bin_count,
            "fractional_boundary_guard": _fraction_payload(self.fractional_boundary_guard),
            "max_roundtrip_tolerance_angstrom": float(
                self.max_roundtrip_tolerance_angstrom
            ).hex(),
            "metric_boundary_guard_angstrom2": float(
                self.metric_boundary_guard_angstrom2
            ).hex(),
            "metric_quantum_angstrom2": float(self.metric_quantum_angstrom2).hex(),
            "require_all_230": self.require_all_230,
            "shape_boundary_guard": float(self.shape_boundary_guard).hex(),
            "shape_quantum": float(self.shape_quantum).hex(),
        }
        return _digest(payload)


DEFAULT_IDENTITY_POLICY = IdentityPolicy()


@dataclass(frozen=True, slots=True)
class CanonicalStructureIdentity:
    """All leakage and duplicate keys for one audited canonical record."""

    exact_key: str
    structure_key: str
    polymorph_key: str
    scaffold_key: str
    prototype_key: str
    composition_key: str
    element_set_key: str
    policy_fingerprint: str
    catalogue_fingerprint: str
    metric_bins: tuple[int, int, int, int, int, int]
    shape_bins: tuple[int, int, int, int, int, int]
    exact_payload: str = field(repr=False)
    structure_payload: str = field(repr=False)
    polymorph_payload: str = field(repr=False)

    def __post_init__(self) -> None:
        digests = (
            self.exact_key,
            self.structure_key,
            self.polymorph_key,
            self.scaffold_key,
            self.prototype_key,
            self.composition_key,
            self.element_set_key,
            self.policy_fingerprint,
            self.catalogue_fingerprint,
        )
        if any(not _is_digest(value) for value in digests):
            raise DomainError("canonical identity fields must be lowercase SHA-256 digests")
        if len(self.metric_bins) != 6 or len(self.shape_bins) != 6:
            raise DomainError("metric and shape identities need six independent components")
        if not self.exact_payload or not self.structure_payload or not self.polymorph_payload:
            raise DomainError("canonical identity payloads cannot be empty")

    def key_for(self, level: DeduplicationLevel) -> str:
        """Return the selected, explicit duplicate quotient key."""

        if not isinstance(level, DeduplicationLevel):
            raise TypeError("level must be a DeduplicationLevel")
        if level is DeduplicationLevel.EXACT:
            return self.exact_key
        if level is DeduplicationLevel.STRUCTURE:
            return self.structure_key
        return self.polymorph_key

    def payload_for(self, level: DeduplicationLevel) -> str:
        """Return collision-checkable canonical serialization for ``level``."""

        if not isinstance(level, DeduplicationLevel):
            raise TypeError("level must be a DeduplicationLevel")
        if level is DeduplicationLevel.EXACT:
            return self.exact_payload
        if level is DeduplicationLevel.STRUCTURE:
            return self.structure_payload
        return self.polymorph_payload


@dataclass(frozen=True, slots=True)
class DuplicateGroup:
    """One physical identity and every source row mapped to it."""

    key: str
    representative_index: int
    member_indices: tuple[int, ...]

    def __post_init__(self) -> None:
        if not _is_digest(self.key):
            raise DomainError("duplicate-group key must be a SHA-256 digest")
        members = tuple(self.member_indices)
        object.__setattr__(self, "member_indices", members)
        if (
            not members
            or tuple(sorted(set(members))) != members
            or any(index < 0 for index in members)
            or self.representative_index not in members
        ):
            raise DomainError("duplicate group indices must be sorted, unique, and nonempty")


@dataclass(frozen=True, slots=True)
class DeduplicationResult:
    """Duplicate groups plus a deterministic representative projection."""

    level: DeduplicationLevel
    identities: tuple[CanonicalStructureIdentity, ...]
    groups: tuple[DuplicateGroup, ...]
    representative_indices: tuple[int, ...]
    discarded_indices: tuple[int, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.level, DeduplicationLevel):
            raise TypeError("level must be a DeduplicationLevel")
        count = len(self.identities)
        members = tuple(index for group in self.groups for index in group.member_indices)
        if tuple(sorted(members)) != tuple(range(count)):
            raise DomainError("duplicate groups must partition every input record exactly once")
        representatives = tuple(sorted(group.representative_index for group in self.groups))
        discarded = tuple(sorted(set(range(count)).difference(representatives)))
        if (
            self.representative_indices != representatives
            or self.discarded_indices != discarded
        ):
            raise DomainError("deduplication projections disagree with the duplicate groups")


def _canonical_json(payload: object) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _digest(payload: object) -> str:
    return sha256(_canonical_json(payload).encode("ascii")).hexdigest()


def _is_digest(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _fraction_payload(value: Fraction) -> tuple[int, int]:
    return value.numerator, value.denominator


def _float_hex(value: float) -> str:
    # Signed zero is not a distinct physical metric component.
    return float(0.0 if value == 0.0 else value).hex()


def _guarded_float_bin(
    value: float,
    *,
    quantum: float,
    guard: float,
    name: str,
) -> int:
    if not math.isfinite(value):
        raise InvariantViolation(f"{name} is not finite")
    scaled = value / quantum
    if not math.isfinite(scaled) or abs(scaled) >= float(1 << 51):
        raise AmbiguousIdentityError(
            f"{name} cannot be resolved with the configured floating-point quantum"
        )
    nearest = math.floor(scaled + 0.5)
    wall_distance = 0.5 - abs(scaled - nearest)
    numerical_guard = guard / quantum + 8.0 * math.ulp(scaled)
    if wall_distance <= numerical_guard:
        raise AmbiguousIdentityError(
            f"{name} lies within its certified guard of an identity-cell boundary"
        )
    return nearest


def _guarded_fractional_bin(
    value: Fraction,
    *,
    policy: IdentityPolicy,
    name: str,
    uncertainty_bound: float,
) -> int:
    if not isinstance(value, Fraction) or not Fraction(0) <= value < Fraction(1):
        raise InvariantViolation(f"{name} is not a canonical exact point on R/Z")
    scaled = value * policy.fractional_bin_count
    nearest = (scaled + Fraction(1, 2)).numerator // (scaled + Fraction(1, 2)).denominator
    wall_distance = Fraction(1, 2) - abs(scaled - nearest)
    if not math.isfinite(uncertainty_bound) or uncertainty_bound < 0.0:
        raise InvariantViolation("fractional uncertainty bound must be finite and nonnegative")
    effective_guard = max(
        policy.fractional_boundary_guard,
        Fraction.from_float(uncertainty_bound),
    )
    if 2 * effective_guard * policy.fractional_bin_count >= 1:
        raise AmbiguousIdentityError(
            "certified coordinate uncertainty exceeds half an identity cell"
        )
    if wall_distance <= effective_guard * policy.fractional_bin_count:
        raise AmbiguousIdentityError(
            f"{name} lies within its exact guard of a torus-cell boundary"
        )
    return nearest % policy.fractional_bin_count


def _metric_components(metric: FloatArray) -> tuple[float, float, float, float, float, float]:
    return tuple(float(metric[row, column]) for row, column in _METRIC_COMPONENTS)  # type: ignore[return-value]


def _metric_bins(
    metric: FloatArray,
    *,
    quantum: float,
    guard: float,
    name: str,
) -> tuple[int, int, int, int, int, int]:
    return tuple(
        _guarded_float_bin(
            value,
            quantum=quantum,
            guard=guard,
            name=f"{name}[{index}]",
        )
        for index, value in enumerate(_metric_components(metric))
    )  # type: ignore[return-value]


def _perfect_matching(adjacency: Sequence[Sequence[int]]) -> bool:
    matched_right = [-1] * len(adjacency)

    def augment(left: int, seen: set[int]) -> bool:
        for right in adjacency[left]:
            if right in seen:
                continue
            seen.add(right)
            previous = matched_right[right]
            if previous < 0 or augment(previous, seen):
                matched_right[right] = left
                return True
        return False

    return all(augment(left, set()) for left in range(len(adjacency)))


def _require_same_coloured_periodic_set(
    structure: RawPeriodicStructure,
    expected_positions: npt.ArrayLike,
    expected_species: Sequence[int],
    *,
    tolerance: float,
) -> None:
    expected = np.asarray(expected_positions, dtype=np.float64)
    species = tuple(expected_species)
    if expected.shape != (len(species), 3) or not np.all(np.isfinite(expected)):
        raise InvariantViolation("exact decoder returned malformed positions")
    if Counter(structure.atomic_numbers) != Counter(species):
        raise InvariantViolation("exact decoder changed the conventional composition")
    tolerance2 = tolerance * tolerance
    for atomic_number in sorted(set(species)):
        actual_indices = tuple(
            index
            for index, value in enumerate(structure.atomic_numbers)
            if value == atomic_number
        )
        expected_indices = tuple(
            index for index, value in enumerate(species) if value == atomic_number
        )
        adjacency = tuple(
            tuple(
                local_expected
                for local_expected, expected_index in enumerate(expected_indices)
                if closest_periodic_image(
                    structure.fractional_positions[actual_index] - expected[expected_index],
                    structure.lattice,
                ).squared_distance
                <= tolerance2
            )
            for actual_index in actual_indices
        )
        if not _perfect_matching(adjacency):
            raise InvariantViolation(
                "canonical target fails the species-preserving exact-decoder round trip"
            )


def _require_species_quotient(
    larger: Sequence[int],
    smaller: Sequence[int],
    *,
    index: int,
    name: str,
) -> None:
    expected = Counter({species: index * count for species, count in Counter(smaller).items()})
    if Counter(larger) != expected:
        raise InvariantViolation(f"{name} violates its exact coloured count quotient")


def _audit_record(
    record: CanonicalRecord,
    *,
    catalogue: CertifiedCatalogue,
    policy: IdentityPolicy,
) -> tuple[CrystalScaffold, SpaceGroupSetting, FloatArray, float]:
    if not isinstance(record, CanonicalTrainingRecord):
        raise TypeError("record must be a CanonicalTrainingRecord")
    catalogue.validate(require_all_230=policy.require_all_230)
    canonicalized = record.canonicalized
    receipt = canonicalized.receipt
    target = record.target
    if receipt.catalogue_fingerprint != catalogue.fingerprint:
        raise InvariantViolation("record receipt belongs to a different exact catalogue")
    if receipt.source_fingerprint != record.raw_structure.fingerprint:
        raise InvariantViolation("record receipt belongs to a different raw structure")
    if not receipt.passed:
        raise InvariantViolation("canonicalization receipt did not pass every proof gate")
    if receipt.symmetry_tolerance > policy.max_roundtrip_tolerance_angstrom:
        raise AmbiguousIdentityError(
            "canonicalization tolerance exceeds the identity policy's accepted resolution"
        )
    scaffold = target.canonical_scaffold
    if canonicalized.scaffold != scaffold:
        raise InvariantViolation("canonicalizer and lifted target disagree on the scaffold")
    setting = catalogue.get(scaffold.space_group)
    pinned_hall = pinned_hall_numbers()[setting.international_number]
    if receipt.hall_number != pinned_hall or receipt.hall_symbol != setting.hall_symbol:
        raise InvariantViolation("record does not use the catalogue's pinned Hall setting")

    metric = np.asarray(target.metric, dtype=np.float64)
    canonical_metric = canonicalized.conventional_structure.metric
    if metric.shape != (3, 3) or not np.array_equal(metric, canonical_metric):
        raise InvariantViolation("target metric is not the canonical conventional metric")
    if not np.array_equal(metric, metric.T):
        raise InvariantViolation("canonical target metric is not exactly symmetric")
    determinant = float(np.linalg.det(metric))
    if not math.isfinite(determinant) or determinant <= 0.0:
        raise InvariantViolation("canonical target metric is not positive definite")

    raw = record.raw_structure
    primitive = canonicalized.primitive_structure
    conventional = canonicalized.conventional_structure
    raw_index = receipt.translation_index
    if len(raw.atomic_numbers) != raw_index * len(primitive.atomic_numbers):
        raise InvariantViolation("raw-to-primitive atom index equation failed")
    _require_species_quotient(
        raw.atomic_numbers,
        primitive.atomic_numbers,
        index=raw_index,
        name="raw-to-primitive reduction",
    )
    centering_index = setting.centering_index
    if len(conventional.atomic_numbers) != centering_index * len(primitive.atomic_numbers):
        raise InvariantViolation("conventional centring atom index equation failed")
    _require_species_quotient(
        conventional.atomic_numbers,
        primitive.atomic_numbers,
        index=centering_index,
        name="conventional centring reduction",
    )
    volume_scale = max(raw.volume, conventional.volume, primitive.volume, 1.0)
    displacement = receipt.idealization_lattice_displacement_angstrom
    if displacement is None:
        raw_volume_ok = math.isclose(
            raw.volume,
            raw_index * primitive.volume,
            rel_tol=1.0e-8,
            abs_tol=1.0e-10 * volume_scale,
        )
    else:
        # Raw against idealized under the dimensioned policy (review B1).
        source_basis = np.asarray(
            [[float(value) for value in row] for row in receipt.source_to_primitive],
            dtype=np.float64,
        )
        raw_volume_ok = idealized_volume_matches(
            raw.volume,
            raw_index * primitive.volume,
            raw_primitive_lattice=source_basis @ raw.lattice,
            lattice_displacement_angstrom=displacement,
        )
    if not raw_volume_ok:
        raise InvariantViolation("raw-to-primitive volume/index equation failed")
    if not math.isclose(
        conventional.volume,
        centering_index * primitive.volume,
        rel_tol=1.0e-8,
        abs_tol=1.0e-10 * volume_scale,
    ):
        raise InvariantViolation("centring volume/index equation failed")

    decoded = ExactCrystalDecoder(
        catalogue,
        metric_tolerance=receipt.symmetry_tolerance,
    ).decode(scaffold, metric)
    if decoded.hidden_translation_index != 1:
        raise InvariantViolation("canonical target retains an undeclared translation")
    decoded_species: list[int] = []
    for species in decoded.species:
        if isinstance(species, bool) or not isinstance(species, int):
            raise InvariantViolation("dataset target contains a non-atomic-number species")
        decoded_species.append(species)
    _require_same_coloured_periodic_set(
        conventional,
        decoded.fractional_positions_float,
        decoded_species,
        tolerance=receipt.symmetry_tolerance,
    )
    sigma_min = float(np.linalg.svd(conventional.lattice, compute_uv=False)[-1])
    if not math.isfinite(sigma_min) or sigma_min <= 0.0:
        raise InvariantViolation("canonical conventional lattice is singular")
    numerical_floor = float(
        64.0
        * np.finfo(np.float64).eps
        * max(float(np.linalg.norm(conventional.lattice, ord=2)), 1.0)
    )
    fractional_uncertainty = (
        max(float(receipt.symmetry_tolerance), numerical_floor) / sigma_min
    )
    return scaffold, setting, metric, fractional_uncertainty


def _atomic_number_composition(scaffold: CrystalScaffold) -> tuple[tuple[int, int], ...]:
    """Project the generic decoder composition into the dataset convention.

    ``CrystalScaffold`` orders generic species by ``(type name, repr)``.
    Dataset identities use integer atomic numbers instead, so they must sort
    a validated copy rather than require or mutate the decoder's ordering.
    """

    composition: list[tuple[int, int]] = []
    for species, count in scaffold.reduced_composition:
        if (
            isinstance(species, bool)
            or not isinstance(species, int)
            or not 1 <= species <= 118
        ):
            raise InvariantViolation("canonical dataset composition must use atomic numbers")
        if isinstance(count, bool) or not isinstance(count, int) or count <= 0:
            raise InvariantViolation(
                "canonical dataset composition needs positive integer counts"
            )
        composition.append((species, count))
    if not composition or len({species for species, _ in composition}) != len(composition):
        raise InvariantViolation("canonical dataset composition must be nonempty and unique")
    if math.gcd(*(count for _, count in composition)) != 1:
        raise InvariantViolation("canonical dataset composition must be reduced")
    return tuple(sorted(composition))


def _scaffold_payloads(
    scaffold: CrystalScaffold,
    setting: SpaceGroupSetting,
    *,
    policy: IdentityPolicy,
    fractional_uncertainty: float,
) -> tuple[
    dict[str, object],
    dict[str, object],
    dict[str, object],
]:
    composition = _atomic_number_composition(scaffold)

    row_rank = {row.label: index for index, row in enumerate(canonical_rows(setting))}
    canonical_orbits = tuple(
        sorted(
            scaffold.orbits,
            key=lambda orbit: (
                row_rank[orbit.wyckoff_label],
                int(orbit.species),
                orbit.parameter,
            ),
        )
    )
    if scaffold.orbits != canonical_orbits:
        raise AmbiguousIdentityError("scaffold orbit instances are not canonically ordered")

    exact_orbits: list[dict[str, object]] = []
    binned_orbit_rows: list[tuple[int, int, tuple[int, ...], dict[str, object]]] = []
    discrete_orbit_rows: list[tuple[int, int, dict[str, object]]] = []
    for orbit_index, orbit in enumerate(scaffold.orbits):
        if isinstance(orbit.species, bool) or not isinstance(orbit.species, int):
            raise InvariantViolation("canonical dataset orbit species must be atomic numbers")
        if any(value < 0 or value >= 1 for value in orbit.parameter):
            raise AmbiguousIdentityError(
                "Wyckoff parameters are not in the canonical torus cell"
            )
        parameter_bins = tuple(
            _guarded_fractional_bin(
                value,
                policy=policy,
                name=f"orbit[{orbit_index}].parameter[{coordinate}]",
                uncertainty_bound=fractional_uncertainty,
            )
            for coordinate, value in enumerate(orbit.parameter)
        )
        exact_orbits.append(
            {
                "parameter": [_fraction_payload(value) for value in orbit.parameter],
                "species": orbit.species,
                "wyckoff": orbit.wyckoff_label,
            }
        )
        binned_orbit = {
            "parameter_bins": parameter_bins,
            "species": orbit.species,
            "wyckoff": orbit.wyckoff_label,
        }
        binned_orbit_rows.append(
            (
                row_rank[orbit.wyckoff_label],
                int(orbit.species),
                parameter_bins,
                binned_orbit,
            )
        )
        discrete_orbit = {
            "species": orbit.species,
            "wyckoff": orbit.wyckoff_label,
        }
        discrete_orbit_rows.append(
            (
                row_rank[orbit.wyckoff_label],
                int(orbit.species),
                discrete_orbit,
            )
        )
    # The exact scaffold is ordered by exact torus coordinates.  Its binned
    # quotient needs its own canonical ordering because a coordinate close to
    # one can round to torus bin zero.  Sorting the quotient after binning makes
    # its byte representation independently checkable without access to the
    # discarded exact coordinates.
    binned_orbits = [row[3] for row in sorted(binned_orbit_rows, key=lambda row: row[:3])]
    discrete_orbits = [row[2] for row in sorted(discrete_orbit_rows, key=lambda row: row[:2])]
    setting_payload = {
        "hall_symbol": setting.hall_symbol,
        "international_number": setting.international_number,
        "key": setting.key,
    }
    exact = {
        "composition": composition,
        "nu": scaffold.nu,
        "orbits": exact_orbits,
        "setting": setting_payload,
    }
    decorated = {
        "composition": composition,
        "nu": scaffold.nu,
        "orbits": binned_orbits,
        "setting": setting_payload,
    }
    discrete = {
        "composition": composition,
        "nu": scaffold.nu,
        "orbits": discrete_orbits,
        "setting": setting_payload,
    }
    return exact, decorated, discrete


def _anonymous_protostructure(
    scaffold: CrystalScaffold,
    setting: SpaceGroupSetting,
) -> dict[str, object]:
    """Return the species-erased crystallographic protostructure.

    A chemical species is represented only by the multiset of Wyckoff rows it
    occupies.  Sorting these role blocks quotients every permutation of the
    element labels without a factorial search.  Free Wyckoff coordinates and
    lattice parameters are deliberately absent: a prototype-disjoint split is
    a conservative holdout of the discrete symmetry/orbit template, not a
    near-duplicate geometry test.
    """

    rows_by_species: dict[int, list[str]] = defaultdict(list)
    for orbit in scaffold.orbits:
        if isinstance(orbit.species, bool) or not isinstance(orbit.species, int):
            raise InvariantViolation("prototype identity requires atomic-number species")
        setting.wyckoff(orbit.wyckoff_label)
        rows_by_species[orbit.species].append(orbit.wyckoff_label)
    if not rows_by_species:
        raise InvariantViolation("an anonymous prototype cannot have an empty motif")
    role_blocks = tuple(
        sorted(tuple(sorted(labels)) for labels in rows_by_species.values())
    )
    return {
        "formula_units_in_pinned_cell": scaffold.nu,
        "anonymous_wyckoff_roles": role_blocks,
        "setting": {
            "hall_symbol": setting.hall_symbol,
            "international_number": setting.international_number,
            "key": setting.key,
        },
    }


def build_canonical_identity(
    record: CanonicalRecord,
    *,
    catalogue: CertifiedCatalogue,
    policy: IdentityPolicy = DEFAULT_IDENTITY_POLICY,
) -> CanonicalStructureIdentity:
    """Audit and identify one post-decoder canonical training record.

    No raw-cell property is serialized into the physical keys.  The raw cell
    is used only to recheck the exact translation-index and coloured
    composition equations, which is why supercells and equivalent settings
    collapse only after their proof receipts have passed.
    """

    if not isinstance(catalogue, CertifiedCatalogue):
        raise TypeError("catalogue must be a CertifiedCatalogue")
    if not isinstance(policy, IdentityPolicy):
        raise TypeError("policy must be an IdentityPolicy")
    scaffold, setting, metric, fractional_uncertainty = _audit_record(
        record,
        catalogue=catalogue,
        policy=policy,
    )
    exact_scaffold, decorated_geometry, discrete_scaffold = _scaffold_payloads(
        scaffold,
        setting,
        policy=policy,
        fractional_uncertainty=fractional_uncertainty,
    )
    anonymous_prototype = _anonymous_protostructure(scaffold, setting)

    metric_bins = _metric_bins(
        metric,
        quantum=policy.metric_quantum_angstrom2,
        guard=policy.metric_boundary_guard_angstrom2,
        name="metric",
    )
    determinant = float(np.linalg.det(metric))
    metric_scale = determinant ** (1.0 / 3.0)
    if not math.isfinite(metric_scale) or metric_scale <= 0.0:
        raise InvariantViolation("canonical metric has an invalid determinant scale")
    shape = np.asarray(metric / metric_scale, dtype=np.float64)
    shape_bins = _metric_bins(
        shape,
        quantum=policy.shape_quantum,
        guard=policy.shape_boundary_guard,
        name="unit-determinant metric shape",
    )

    composition = _atomic_number_composition(scaffold)
    elements = tuple(species for species, _ in composition)
    composition_key = _digest(
        {"composition": composition, "schema": f"{_SCHEMA}.composition.v1"}
    )
    element_set_key = _digest({"elements": elements, "schema": f"{_SCHEMA}.element-set.v1"})
    common = {
        "catalogue_fingerprint": catalogue.fingerprint,
        "chirality_quotient": "SO(3)",
        "coordinate_convention": "fractional-row-v1",
        "schema": _SCHEMA,
    }
    policy_common = {**common, "policy_fingerprint": policy.fingerprint}
    geometry_payload = {**policy_common, "scaffold": decorated_geometry}
    scaffold_payload = {**common, "scaffold": discrete_scaffold}
    structure_payload_object = {
        **geometry_payload,
        "metric_bins_angstrom2": metric_bins,
    }
    polymorph_payload_object = {
        **geometry_payload,
        "unit_determinant_metric_shape_bins": shape_bins,
    }
    exact_payload_object = {
        **common,
        "metric_hex_angstrom2": tuple(
            _float_hex(value) for value in _metric_components(metric)
        ),
        "scaffold": exact_scaffold,
    }
    prototype_payload = {
        **common,
        "anonymous_crystallographic_protostructure": anonymous_prototype,
    }
    exact_payload = _canonical_json(exact_payload_object)
    structure_payload = _canonical_json(structure_payload_object)
    polymorph_payload = _canonical_json(polymorph_payload_object)
    return CanonicalStructureIdentity(
        exact_key=sha256(exact_payload.encode("ascii")).hexdigest(),
        structure_key=sha256(structure_payload.encode("ascii")).hexdigest(),
        polymorph_key=sha256(polymorph_payload.encode("ascii")).hexdigest(),
        scaffold_key=_digest(scaffold_payload),
        prototype_key=_digest(prototype_payload),
        composition_key=composition_key,
        element_set_key=element_set_key,
        policy_fingerprint=policy.fingerprint,
        catalogue_fingerprint=catalogue.fingerprint,
        metric_bins=metric_bins,
        shape_bins=shape_bins,
        exact_payload=exact_payload,
        structure_payload=structure_payload,
        polymorph_payload=polymorph_payload,
    )


def _representative_order(
    index: int,
    record: CanonicalRecord,
) -> tuple[bool, str, str, int]:
    source_id = record.raw_structure.source_id
    return source_id is None, source_id or "", record.raw_structure.fingerprint, index


def deduplicate_canonical_records(
    records: Sequence[CanonicalRecord],
    *,
    catalogue: CertifiedCatalogue,
    policy: IdentityPolicy = DEFAULT_IDENTITY_POLICY,
    level: DeduplicationLevel,
) -> DeduplicationResult:
    """Partition records by an explicit canonical identity quotient.

    The result retains every member index, so provenance from discarded source
    rows is never lost.  The representative is deterministic by source ID and
    raw fingerprint, independent of input order except for physically and
    provenance-identical ties.
    """

    if not isinstance(level, DeduplicationLevel):
        raise TypeError("level must be a DeduplicationLevel")
    frozen_records = tuple(records)
    if not frozen_records:
        raise DomainError("deduplication needs at least one canonical record")
    identities = tuple(
        build_canonical_identity(record, catalogue=catalogue, policy=policy)
        for record in frozen_records
    )
    members_by_key: dict[str, list[int]] = defaultdict(list)
    payload_by_key: dict[str, str] = {}
    for index, identity in enumerate(identities):
        key = identity.key_for(level)
        payload = identity.payload_for(level)
        previous = payload_by_key.setdefault(key, payload)
        if previous != payload:
            raise InvariantViolation("SHA-256 collision between canonical identity payloads")
        members_by_key[key].append(index)

    groups: list[DuplicateGroup] = []
    for key in sorted(members_by_key):
        members = tuple(sorted(members_by_key[key]))
        representative = min(
            members,
            key=lambda index: _representative_order(index, frozen_records[index]),
        )
        groups.append(DuplicateGroup(key, representative, members))
    group_tuple = tuple(groups)
    representatives = tuple(sorted(group.representative_index for group in group_tuple))
    discarded = tuple(sorted(set(range(len(frozen_records))).difference(representatives)))
    return DeduplicationResult(
        level,
        identities,
        group_tuple,
        representatives,
        discarded,
    )


__all__ = [
    "AmbiguousIdentityError",
    "CanonicalStructureIdentity",
    "DEFAULT_IDENTITY_POLICY",
    "DeduplicationLevel",
    "DeduplicationResult",
    "DuplicateGroup",
    "IdentityPolicy",
    "build_canonical_identity",
    "deduplicate_canonical_records",
]
