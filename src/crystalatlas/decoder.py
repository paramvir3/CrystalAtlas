"""Fail-closed exact decoding for the certified crystallographic fixtures."""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from fractions import Fraction
from math import gcd
from numbers import Integral
from types import MappingProxyType
from typing import TypeAlias

import numpy as np

from .algebra import FractionVector, as_fraction_vector, fraction_mod_one
from .catalogue import MINIMAL_CATALOGUE, CertifiedCatalogue, SpaceGroupSetting
from .errors import DomainError, InvariantViolation, NumericalError
from .lattice import InvariantSPDChart

Species: TypeAlias = int | str


def _validate_species(species: object) -> Species:
    if isinstance(species, bool):
        raise DomainError("Boolean values are not chemical species")
    if isinstance(species, Integral) and species > 0:
        return int(species)
    if isinstance(species, str) and species.strip():
        return species.strip()
    raise DomainError("species must be a positive atomic number or nonempty symbol")


def _composition_tuple(composition: Mapping[Species, int]) -> tuple[tuple[Species, int], ...]:
    if not composition:
        raise DomainError("reduced composition cannot be empty")
    normalized: list[tuple[Species, int]] = []
    for species, count in composition.items():
        value = _validate_species(species)
        if isinstance(count, bool) or not isinstance(count, Integral) or count <= 0:
            raise DomainError("reduced composition counts must be positive integers")
        normalized.append((value, int(count)))
    if len({species for species, _ in normalized}) != len(normalized):
        raise DomainError("reduced composition repeats a species")
    common = 0
    for _, count in normalized:
        common = gcd(common, count)
    if common != 1:
        raise DomainError("composition must be reduced (gcd equal to one)")
    normalized.sort(key=lambda item: (type(item[0]).__name__, repr(item[0])))
    return tuple(normalized)


@dataclass(frozen=True, slots=True)
class OrbitAssignment:
    """A species-decorated independent Wyckoff orbit and exact parameter."""

    wyckoff_label: str
    species: Species
    parameter: FractionVector

    def __post_init__(self) -> None:
        if not self.wyckoff_label:
            raise DomainError("Wyckoff label cannot be empty")
        object.__setattr__(self, "species", _validate_species(self.species))
        object.__setattr__(self, "parameter", as_fraction_vector(self.parameter))


@dataclass(frozen=True, slots=True)
class CrystalScaffold:
    """The exact raw scaffold ``alpha=(nu,g,h)`` with explicit orbit instances."""

    nu: int
    space_group: str
    reduced_composition: tuple[tuple[Species, int], ...]
    orbits: tuple[OrbitAssignment, ...]

    def __post_init__(self) -> None:
        if isinstance(self.nu, bool) or not isinstance(self.nu, Integral) or self.nu <= 0:
            raise DomainError("formula-unit multiplier nu must be a positive integer")
        object.__setattr__(self, "nu", int(self.nu))
        if not self.space_group:
            raise DomainError("scaffold needs an explicit pinned space-group key")
        if len(dict(self.reduced_composition)) != len(self.reduced_composition):
            raise DomainError("reduced composition repeats a species")
        composition = _composition_tuple(dict(self.reduced_composition))
        orbits = tuple(self.orbits)
        if not orbits:
            raise DomainError("a scaffold must advertise at least one orbit")
        supported_species = {species for species, _ in composition}
        if any(orbit.species not in supported_species for orbit in orbits):
            raise DomainError("an orbit uses a species outside the exact formula support")
        object.__setattr__(self, "reduced_composition", composition)
        object.__setattr__(self, "orbits", orbits)

    @classmethod
    def from_mapping(
        cls,
        *,
        nu: int,
        space_group: str,
        reduced_composition: Mapping[Species, int],
        orbits: Sequence[OrbitAssignment],
    ) -> CrystalScaffold:
        return cls(nu, space_group, tuple(reduced_composition.items()), tuple(orbits))

    @property
    def composition(self) -> Mapping[Species, int]:
        return MappingProxyType(dict(self.reduced_composition))


@dataclass(frozen=True, slots=True)
class DecoderAudit:
    positive_volume: bool
    formula_matches: bool
    orbit_multiplicities_match: bool
    no_collisions: bool
    metric_is_invariant: bool
    selected_group_is_contained: bool
    translation_index: int
    reduced_composition_matches: bool

    @property
    def passed(self) -> bool:
        return all(
            (
                self.positive_volume,
                self.formula_matches,
                self.orbit_multiplicities_match,
                self.no_collisions,
                self.metric_is_invariant,
                self.selected_group_is_contained,
                self.reduced_composition_matches,
            )
        )


@dataclass(frozen=True, slots=True)
class DecodedCrystal:
    """An exact conventional motif plus its frozen right-handed lattice section.

    ``translation_subgroup`` includes the setting's declared centring subgroup.
    Any quotient beyond ``setting.centering_index`` is an undeclared hidden
    translation and therefore a nonprimitive training representation.
    """

    scaffold: CrystalScaffold
    setting: SpaceGroupSetting
    lattice: np.ndarray
    metric: np.ndarray
    fractional_positions: tuple[FractionVector, ...]
    species: tuple[Species, ...]
    orbit_slices: tuple[slice, ...]
    translation_subgroup: tuple[FractionVector, ...]
    audit: DecoderAudit

    def __post_init__(self) -> None:
        lattice = np.array(self.lattice, dtype=np.float64, copy=True)
        metric = np.array(self.metric, dtype=np.float64, copy=True)
        if lattice.shape != (3, 3) or metric.shape != (3, 3):
            raise DomainError("decoded lattice and metric must be 3x3")
        lattice.setflags(write=False)
        metric.setflags(write=False)
        object.__setattr__(self, "lattice", lattice)
        object.__setattr__(self, "metric", metric)

    @property
    def conventional_counts(self) -> Mapping[Species, int]:
        return MappingProxyType(dict(Counter(self.species)))

    @property
    def translation_index(self) -> int:
        return len(self.translation_subgroup)

    @property
    def hidden_translation_index(self) -> int:
        centering_index = self.setting.centering_index
        if self.translation_index % centering_index:
            raise InvariantViolation(
                "complete translation group is not divisible by declared centring"
            )
        return self.translation_index // centering_index

    @property
    def counts_after_translation_quotient(self) -> Mapping[Species, int]:
        index = self.translation_index
        counts = Counter(self.species)
        if any(count % index for count in counts.values()):
            raise InvariantViolation(
                "translation index does not divide every coloured atom count"
            )
        return MappingProxyType({species: count // index for species, count in counts.items()})

    @property
    def fractional_positions_float(self) -> np.ndarray:
        return np.asarray(
            [[float(component) for component in point] for point in self.fractional_positions],
            dtype=np.float64,
        )

    @property
    def cartesian_positions(self) -> np.ndarray:
        return self.fractional_positions_float @ self.lattice


def exact_formula_matches(
    species: Sequence[Species], composition: Mapping[Species, int], nu: int
) -> bool:
    expected = Counter({key: nu * value for key, value in composition.items()})
    return Counter(species) == expected


def exact_group_containment(
    setting: SpaceGroupSetting,
    positions: Sequence[FractionVector],
    species: Sequence[Species],
    metric: np.ndarray,
    *,
    metric_tolerance: float = 1.0e-10,
) -> bool:
    """Check selected operations on the coloured motif and metric explicitly."""

    if len(positions) != len(species):
        raise DomainError("position/species lengths disagree")
    coloured = Counter(zip(positions, species, strict=True))
    g = np.asarray(metric, dtype=np.float64)
    if g.shape != (3, 3):
        raise DomainError("metric must be 3x3")
    for operation in setting.operations:
        transformed = Counter(
            (operation.apply_exact(position), atom_species)
            for position, atom_species in zip(positions, species, strict=True)
        )
        if transformed != coloured:
            return False
        rotation = np.asarray(operation.rotation, dtype=np.float64)
        if not np.allclose(
            rotation @ g @ rotation.T,
            g,
            atol=metric_tolerance,
            rtol=metric_tolerance,
        ):
            return False
    return True


def exact_translation_subgroup(
    positions: Sequence[FractionVector], species: Sequence[Species]
) -> tuple[FractionVector, ...]:
    """Enumerate every species-preserving torus translation from pair differences."""

    if not positions or len(positions) != len(species):
        raise DomainError("translation enumeration needs a nonempty coloured motif")
    coloured = Counter(zip(positions, species, strict=True))
    species_counts = Counter(species)
    anchor_species = min(species_counts, key=lambda value: species_counts[value])
    anchor_index = next(i for i, value in enumerate(species) if value == anchor_species)
    anchor = positions[anchor_index]
    candidates = {
        tuple(fraction_mod_one(target[j] - anchor[j]) for j in range(3))
        for target, value in zip(positions, species, strict=True)
        if value == anchor_species
    }
    accepted: list[FractionVector] = []
    for translation in candidates:
        translated = Counter(
            (
                tuple(fraction_mod_one(point[j] + translation[j]) for j in range(3)),
                atom_species,
            )
            for point, atom_species in zip(positions, species, strict=True)
        )
        if translated == coloured:
            accepted.append(translation)
    accepted.sort()
    identity = (Fraction(0), Fraction(0), Fraction(0))
    if identity not in accepted:
        raise InvariantViolation("translation enumeration lost the identity")
    # Verify the returned candidates really form the complete finite subgroup.
    accepted_set = set(accepted)
    for left in accepted:
        inverse = tuple(fraction_mod_one(-value) for value in left)
        if inverse not in accepted_set:
            raise InvariantViolation("translation set is not closed under inversion")
        for right in accepted:
            total = tuple(fraction_mod_one(left[j] + right[j]) for j in range(3))
            if total not in accepted_set:
                raise InvariantViolation("translation set is not closed under addition")
    return tuple(accepted)


def _reduced_counts(counts: Mapping[Species, int]) -> Mapping[Species, int]:
    common = 0
    for count in counts.values():
        common = gcd(common, count)
    if common <= 0:
        raise InvariantViolation("cannot reduce an empty composition")
    return MappingProxyType({species: count // common for species, count in counts.items()})


class ExactCrystalDecoder:
    """Decode and verify exact Fraction-valued conventional Hall covers."""

    def __init__(
        self,
        catalogue: CertifiedCatalogue = MINIMAL_CATALOGUE,
        *,
        metric_tolerance: float = 1.0e-10,
    ) -> None:
        if not np.isfinite(metric_tolerance) or metric_tolerance <= 0.0:
            raise DomainError("metric tolerance must be finite and positive")
        catalogue.validate()
        self.catalogue = catalogue
        self.metric_tolerance = float(metric_tolerance)

    def decode(self, scaffold: CrystalScaffold, metric: np.ndarray) -> DecodedCrystal:
        setting = self.catalogue.get(scaffold.space_group)
        value = np.asarray(metric, dtype=np.float64)
        if value.shape != (3, 3) or not np.all(np.isfinite(value)):
            raise DomainError("metric must be a finite 3x3 matrix")
        if not np.allclose(value, value.T, atol=1.0e-12, rtol=1.0e-12):
            raise DomainError("metric must be symmetric")
        value = (value + value.T) / 2.0
        if float(np.min(np.linalg.eigvalsh(value))) <= 0.0:
            raise DomainError("metric must be positive definite")
        try:
            lattice = np.linalg.cholesky(value)
        except np.linalg.LinAlgError as error:
            raise NumericalError("right-handed Cholesky decoder section failed") from error
        if float(np.linalg.det(lattice)) <= 0.0:
            raise InvariantViolation("decoded lattice volume is not positive")

        formula_counts: Counter[Species] = Counter()
        for orbit in scaffold.orbits:
            row = setting.wyckoff(orbit.wyckoff_label)
            if len(orbit.parameter) != row.dimension:
                raise DomainError(
                    f"orbit {orbit.wyckoff_label} needs {row.dimension} intrinsic parameters"
                )
            formula_counts[orbit.species] += row.multiplicity
        expected_counts = Counter(
            {species: scaffold.nu * count for species, count in scaffold.composition.items()}
        )
        if formula_counts != expected_counts:
            raise InvariantViolation(
                f"scaffold violates exact formula equations: got {dict(formula_counts)}, "
                f"expected {dict(expected_counts)}"
            )

        positions: list[FractionVector] = []
        species: list[Species] = []
        orbit_slices: list[slice] = []
        orbit_multiplicities_match = True
        for orbit in scaffold.orbits:
            row = setting.wyckoff(orbit.wyckoff_label)
            start = len(positions)
            expanded = row.expand_exact(orbit.parameter, require_declared_multiplicity=True)
            orbit_multiplicities_match &= len(expanded) == row.multiplicity
            positions.extend(expanded)
            species.extend([orbit.species] * row.multiplicity)
            orbit_slices.append(slice(start, len(positions)))

        no_collisions = len(set(positions)) == len(positions)
        if not no_collisions:
            raise InvariantViolation(
                "distinct advertised orbits collide in the conventional cell"
            )
        formula_matches = exact_formula_matches(species, scaffold.composition, scaffold.nu)
        metric_is_invariant = all(
            np.allclose(
                np.asarray(operation.rotation, dtype=np.float64)
                @ value
                @ np.asarray(operation.rotation, dtype=np.float64).T,
                value,
                atol=self.metric_tolerance,
                rtol=self.metric_tolerance,
            )
            for operation in setting.operations
        )
        if not metric_is_invariant:
            raise InvariantViolation("decoded metric violates selected point-group invariance")
        group_contained = exact_group_containment(
            setting,
            positions,
            species,
            value,
            metric_tolerance=self.metric_tolerance,
        )
        if not group_contained:
            raise InvariantViolation(
                "selected Hall operations do not preserve the decoded crystal"
            )
        translations = exact_translation_subgroup(positions, species)
        translation_index = len(translations)
        if not set(setting.centering_translations).issubset(translations):
            raise InvariantViolation(
                "decoded motif lost a declared conventional centring translation"
            )
        if translation_index % setting.centering_index:
            raise InvariantViolation(
                "complete translation group is not divisible by declared centring"
            )
        if any(count % translation_index for count in formula_counts.values()):
            raise InvariantViolation(
                "complete translation index does not divide species counts"
            )
        quotient_counts = {
            atom_species: count // translation_index
            for atom_species, count in formula_counts.items()
        }
        reduced_composition_matches = dict(_reduced_counts(quotient_counts)) == dict(
            scaffold.composition
        )
        if not reduced_composition_matches:
            raise InvariantViolation("translation quotient changed the reduced composition")
        audit = DecoderAudit(
            positive_volume=float(np.linalg.det(lattice)) > 0.0,
            formula_matches=formula_matches,
            orbit_multiplicities_match=orbit_multiplicities_match,
            no_collisions=no_collisions,
            metric_is_invariant=metric_is_invariant,
            selected_group_is_contained=group_contained,
            translation_index=translation_index,
            reduced_composition_matches=reduced_composition_matches,
        )
        if not audit.passed:
            raise InvariantViolation("decoded crystal failed its exact theorem audit")
        return DecodedCrystal(
            scaffold=scaffold,
            setting=setting,
            lattice=lattice,
            metric=value,
            fractional_positions=tuple(positions),
            species=tuple(species),
            orbit_slices=tuple(orbit_slices),
            translation_subgroup=translations,
            audit=audit,
        )

    def decode_from_chart(
        self,
        scaffold: CrystalScaffold,
        chart: InvariantSPDChart,
        coefficients: Sequence[float],
    ) -> DecodedCrystal:
        setting = self.catalogue.get(scaffold.space_group)
        setting_rotations = {
            tuple(tuple(int(value) for value in row) for row in rotation)
            for rotation in setting.point_group_rotations
        }
        chart_rotations = set()
        for rotation in chart.rotations:
            rounded = np.rint(rotation)
            if not np.allclose(
                rotation,
                rounded,
                atol=self.metric_tolerance,
                rtol=self.metric_tolerance,
            ):
                raise DomainError("lattice chart rotation is not an exact integral operation")
            chart_rotations.add(tuple(tuple(int(value) for value in row) for row in rounded))
        if chart_rotations != setting_rotations:
            raise DomainError("lattice chart point group does not match scaffold setting")
        return self.decode(scaffold, chart.decode(coefficients))
