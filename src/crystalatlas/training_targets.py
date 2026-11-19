"""Canonical supervised targets for exact crystallographic covers.

This module lifts an *already symmetry-resolved* exact scaffold and invariant
metric into the categorical and continuous targets consumed by the current
models.  It is intentionally not a CIF parser, a symmetry finder, or a
230-space-group structure standardizer.  The strict raw boundary and its
receipts live in :mod:`crystalatlas.ingestion`; this module accepts only their
already symmetry-resolved exact scaffold.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from fractions import Fraction
from typing import TypeAlias

import numpy as np
import numpy.typing as npt

from .algebra import FractionVector, fraction_mod_one
from .catalogue import MINIMAL_CATALOGUE, CertifiedCatalogue, SpaceGroupSetting
from .composition import Composition
from .config import DomainConfig
from .constraints import OccupationTable, RowAction, SuffixReachability, WyckoffRowSpec
from .cover import canonical_rows
from .cover_legality import AtomBound, feasible_formula_units, legal_settings
from .decoder import CrystalScaffold, ExactCrystalDecoder, OrbitAssignment
from .errors import CatalogueError, DomainError, InvariantViolation
from .geometry import (
    ContinuousCover,
    ContinuousGeometry,
    IntrinsicCoverLayout,
)
from .lattice import InvariantSPDChart
from .periodic import closest_periodic_image

FloatArray: TypeAlias = npt.NDArray[np.float64]

_CERTIFIED_LIFT_KEYS = frozenset({"P1", "P-1"})
_IDENTITY_PERIOD_3 = (
    (Fraction(1), Fraction(0), Fraction(0)),
    (Fraction(0), Fraction(1), Fraction(0)),
    (Fraction(0), Fraction(0), Fraction(1)),
)


@dataclass(frozen=True, slots=True)
class RowDecisionTarget:
    """One exact teacher-forced suffix decision and its complete legal set."""

    row_index: int
    row: WyckoffRowSpec
    residual_before: tuple[int, ...]
    legal_actions: tuple[RowAction, ...]
    selected_action_index: int
    selected_action: RowAction
    residual_after: tuple[int, ...]

    def __post_init__(self) -> None:
        legal = tuple(self.legal_actions)
        before = tuple(self.residual_before)
        after = tuple(self.residual_after)
        object.__setattr__(self, "legal_actions", legal)
        object.__setattr__(self, "residual_before", before)
        object.__setattr__(self, "residual_after", after)
        if (
            isinstance(self.row_index, bool)
            or not isinstance(self.row_index, int)
            or self.row_index < 0
        ):
            raise DomainError("row_index must be a nonnegative integer")
        if not legal:
            raise DomainError("a supervised row decision needs a legal action")
        if not 0 <= self.selected_action_index < len(legal):
            raise DomainError("selected row-action index lies outside its legal set")
        if legal[self.selected_action_index] != self.selected_action:
            raise DomainError("selected row action disagrees with its target index")
        residual_values = (*before, *after)
        if len(before) != len(after) or any(
            isinstance(value, bool) or not isinstance(value, int) or value < 0
            for value in residual_values
        ):
            raise DomainError("row residuals must be equal-length nonnegative vectors")


@dataclass(frozen=True, slots=True)
class CoverDecisionTarget:
    """Exact selector masks and targets in neural vocabulary coordinates."""

    nu_legal_mask: tuple[bool, ...]
    nu_target_index: int
    group_legal_mask: tuple[bool, ...]
    group_target_index: int
    rows: tuple[WyckoffRowSpec, ...]
    occupation: OccupationTable
    row_decisions: tuple[RowDecisionTarget, ...]

    def __post_init__(self) -> None:
        nu_mask = tuple(self.nu_legal_mask)
        group_mask = tuple(self.group_legal_mask)
        rows = tuple(self.rows)
        decisions = tuple(self.row_decisions)
        object.__setattr__(self, "nu_legal_mask", nu_mask)
        object.__setattr__(self, "group_legal_mask", group_mask)
        object.__setattr__(self, "rows", rows)
        object.__setattr__(self, "row_decisions", decisions)
        if not nu_mask or not 0 <= self.nu_target_index < len(nu_mask):
            raise DomainError("nu target lies outside its selector vocabulary")
        if not nu_mask[self.nu_target_index]:
            raise DomainError("nu target is absent from its exact legal mask")
        if not group_mask or not 0 <= self.group_target_index < len(group_mask):
            raise DomainError("group target lies outside its selector vocabulary")
        if not group_mask[self.group_target_index]:
            raise DomainError("group target is absent from its exact legal mask")
        if len(rows) != len(self.occupation) or len(decisions) != len(rows):
            raise DomainError("row decisions, rows, and occupation table must align")
        if tuple(item.row for item in decisions) != rows:
            raise DomainError("row decision metadata disagrees with canonical row order")
        if tuple(item.selected_action for item in decisions) != self.occupation.actions:
            raise DomainError("row decision targets disagree with the occupation table")


@dataclass(frozen=True, slots=True)
class ExactFixtureTrainingTarget:
    """One deterministic discrete/continuous target in the certified domain."""

    composition: Composition
    canonical_scaffold: CrystalScaffold
    cover: ContinuousCover
    cover_decisions: CoverDecisionTarget
    layout: IntrinsicCoverLayout
    metric: FloatArray
    lattice_latent: FloatArray
    intrinsic_coordinates: FloatArray
    period_basis: FloatArray

    def __post_init__(self) -> None:
        arrays = {
            "metric": (self.metric, (3, 3)),
            "lattice_latent": (
                self.lattice_latent,
                (self.layout.lattice_dimension,),
            ),
            "intrinsic_coordinates": (
                self.intrinsic_coordinates,
                (self.layout.intrinsic_dimension,),
            ),
            "period_basis": (
                self.period_basis,
                (self.layout.intrinsic_dimension, self.layout.intrinsic_dimension),
            ),
        }
        for name, (value, shape) in arrays.items():
            array = np.asarray(value, dtype=np.float64).copy()
            if array.shape != shape or not np.all(np.isfinite(array)):
                raise DomainError(f"{name} has the wrong shape or nonfinite entries")
            array.setflags(write=False)
            object.__setattr__(self, name, array)
        if ContinuousCover.from_exact_scaffold(self.canonical_scaffold) != self.cover:
            raise DomainError("canonical scaffold and continuous cover disagree")
        if self.cover.composition != self.composition:
            raise DomainError("target composition and cover composition disagree")


@dataclass(frozen=True, slots=True)
class CanonicalCatalogueTrainingTarget:
    """Post-standardization target valid for the complete exact catalogue.

    The shared :class:`IntrinsicCoverLayout` removes continuous origin freedom
    with the exact axiswise chart quotient certified for all 230 pinned types.
    """

    composition: Composition
    canonical_scaffold: CrystalScaffold
    cover: ContinuousCover
    cover_decisions: CoverDecisionTarget
    layout: IntrinsicCoverLayout
    metric: FloatArray
    lattice_latent: FloatArray
    intrinsic_coordinates: FloatArray
    period_basis: FloatArray

    def __post_init__(self) -> None:
        arrays = {
            "metric": (self.metric, (3, 3)),
            "lattice_latent": (
                self.lattice_latent,
                (self.layout.lattice_dimension,),
            ),
            "intrinsic_coordinates": (
                self.intrinsic_coordinates,
                (self.layout.intrinsic_dimension,),
            ),
            "period_basis": (
                self.period_basis,
                (self.layout.intrinsic_dimension, self.layout.intrinsic_dimension),
            ),
        }
        for name, (payload, shape) in arrays.items():
            array = np.asarray(payload, dtype=np.float64).copy()
            if array.shape != shape or not np.all(np.isfinite(array)):
                raise DomainError(f"{name} has the wrong shape or nonfinite entries")
            array.setflags(write=False)
            object.__setattr__(self, name, array)
        if self.cover != ContinuousCover.from_exact_scaffold(self.canonical_scaffold):
            raise DomainError("canonical scaffold and catalogue cover disagree")
        if self.cover.composition != self.composition:
            raise DomainError("target composition and cover composition disagree")
        if self.layout.cover != self.cover:
            raise DomainError("target layout and catalogue cover disagree")

    @property
    def lattice_chart(self) -> InvariantSPDChart:
        return self.layout.chart


def _canonical_parameter(parameter: FractionVector) -> FractionVector:
    return tuple(fraction_mod_one(value) for value in parameter)


def _perfect_matching(adjacency: np.ndarray) -> bool:
    """Return whether a square Boolean bipartite graph has a perfect match."""

    size = adjacency.shape[0]
    matched_right = [-1] * size

    def augment(left: int, seen: list[bool]) -> bool:
        for right in range(size):
            if not adjacency[left, right] or seen[right]:
                continue
            seen[right] = True
            if matched_right[right] < 0 or augment(matched_right[right], seen):
                matched_right[right] = left
                return True
        return False

    return all(augment(left, [False] * size) for left in range(size))


def _require_origin_equivalent_roundtrip(
    reference_positions: npt.ArrayLike,
    reference_species: tuple[object, ...],
    decoded_positions: npt.ArrayLike,
    decoded_species: npt.ArrayLike,
    lattice: npt.ArrayLike,
    *,
    tolerance: float,
) -> None:
    """Verify equality of two coloured motifs modulo one global origin shift."""

    reference = np.asarray(reference_positions, dtype=np.float64)
    decoded = np.asarray(decoded_positions, dtype=np.float64)
    decoded_z = tuple(int(value) for value in np.asarray(decoded_species).tolist())
    reference_z = tuple(int(value) for value in reference_species)
    lattice_value = np.asarray(lattice, dtype=np.float64)
    if reference.shape != decoded.shape or reference.shape[1:] != (3,):
        raise InvariantViolation("continuous target round-trip changed the atom count")
    if Counter(reference_z) != Counter(decoded_z):
        raise InvariantViolation("continuous target round-trip changed coloured counts")

    anchor_species = min(reference_z)
    anchor_index = reference_z.index(anchor_species)
    candidate_indices = [
        index for index, value in enumerate(decoded_z) if value == anchor_species
    ]
    for decoded_anchor in candidate_indices:
        shift = reference[anchor_index] - decoded[decoded_anchor]
        valid = True
        for atomic_number in sorted(set(reference_z)):
            left = np.asarray(
                [
                    np.mod(point + shift, 1.0)
                    for point, value in zip(decoded, decoded_z, strict=True)
                    if value == atomic_number
                ]
            )
            right = np.asarray(
                [
                    point
                    for point, value in zip(reference, reference_z, strict=True)
                    if value == atomic_number
                ]
            )
            adjacency = np.zeros((len(left), len(right)), dtype=bool)
            for left_index, point in enumerate(left):
                for right_index, target in enumerate(right):
                    image = closest_periodic_image(point - target, lattice_value)
                    adjacency[left_index, right_index] = (
                        image.squared_distance <= tolerance**2
                    )
            if not _perfect_matching(adjacency):
                valid = False
                break
        if valid:
            return
    raise InvariantViolation(
        "continuous target failed its coloured periodic round-trip modulo origin"
    )


def _require_fixture_parameter_chart(
    setting: SpaceGroupSetting, assignment: OrbitAssignment
) -> None:
    row = setting.wyckoff(assignment.wyckoff_label)
    if row.dimension == 0:
        if assignment.parameter:
            raise DomainError("fixed fixture rows cannot carry intrinsic parameters")
        return
    if row.dimension != 3 or row.period_basis != _IDENTITY_PERIOD_3:
        raise CatalogueError(
            "the exact target lift has no certificate for this intrinsic parameter chart"
        )


def _p1_canonical_assignments(
    scaffold: CrystalScaffold, setting: SpaceGroupSetting
) -> tuple[tuple[OrbitAssignment, ...], FractionVector]:
    normalized: list[OrbitAssignment] = []
    for assignment in scaffold.orbits:
        _require_fixture_parameter_chart(setting, assignment)
        if assignment.wyckoff_label != "1a":
            raise CatalogueError("the certified P1 lift expects only row 1a")
        normalized.append(
            OrbitAssignment(
                assignment.wyckoff_label,
                assignment.species,
                _canonical_parameter(assignment.parameter),
            )
        )
    integer_species = [int(item.species) for item in normalized]
    minimum_species = min(integer_species)
    candidates: list[
        tuple[
            tuple[tuple[int, FractionVector], ...],
            tuple[OrbitAssignment, ...],
            FractionVector,
        ]
    ] = []
    for anchor_assignment in normalized:
        if int(anchor_assignment.species) != minimum_species:
            continue
        anchor = anchor_assignment.parameter
        shifted = tuple(
            OrbitAssignment(
                item.wyckoff_label,
                item.species,
                tuple(
                    fraction_mod_one(value - origin)
                    for value, origin in zip(item.parameter, anchor, strict=True)
                ),
            )
            for item in normalized
        )
        ordered = tuple(sorted(shifted, key=lambda item: (int(item.species), item.parameter)))
        key = tuple((int(item.species), item.parameter) for item in ordered)
        candidates.append((key, ordered, anchor))
    _, assignments, anchor = min(candidates, key=lambda item: item[0])
    return assignments, anchor


def _p_minus_one_canonical_assignments(
    scaffold: CrystalScaffold,
    setting: SpaceGroupSetting,
    row_rank: dict[str, int],
) -> tuple[OrbitAssignment, ...]:
    normalized: list[OrbitAssignment] = []
    for assignment in scaffold.orbits:
        _require_fixture_parameter_chart(setting, assignment)
        parameter = _canonical_parameter(assignment.parameter)
        if assignment.wyckoff_label == "2i":
            inverse = tuple(fraction_mod_one(-value) for value in parameter)
            parameter = min(parameter, inverse)
        elif setting.wyckoff(assignment.wyckoff_label).dimension != 0:
            raise CatalogueError("the certified P-1 lift recognizes only free row 2i")
        normalized.append(
            OrbitAssignment(
                assignment.wyckoff_label,
                assignment.species,
                parameter,
            )
        )
    return tuple(
        sorted(
            normalized,
            key=lambda item: (
                row_rank[item.wyckoff_label],
                int(item.species),
                item.parameter,
            ),
        )
    )


def _coloured_orbit_points(
    assignments: tuple[OrbitAssignment, ...], setting: SpaceGroupSetting
) -> Counter[tuple[FractionVector, int]]:
    result: Counter[tuple[FractionVector, int]] = Counter()
    for assignment in assignments:
        for point in setting.wyckoff(assignment.wyckoff_label).expand_exact(
            assignment.parameter, require_declared_multiplicity=True
        ):
            result[(point, int(assignment.species))] += 1
    return result


def _canonical_scaffold(
    scaffold: CrystalScaffold, setting: SpaceGroupSetting
) -> CrystalScaffold:
    rows = canonical_rows(setting)
    row_rank = {row.label: index for index, row in enumerate(rows)}
    if setting.key == "P1":
        assignments, anchor = _p1_canonical_assignments(scaffold, setting)
    elif setting.key == "P-1":
        assignments = _p_minus_one_canonical_assignments(scaffold, setting, row_rank)
        anchor = (Fraction(0), Fraction(0), Fraction(0))
    else:  # pragma: no cover - guarded by the public scope check
        raise CatalogueError("unsupported target-lift setting")

    canonical = CrystalScaffold(
        scaffold.nu,
        scaffold.space_group,
        scaffold.reduced_composition,
        assignments,
    )
    expected = _coloured_orbit_points(tuple(scaffold.orbits), setting)
    if setting.key == "P1":
        expected = Counter(
            {
                (
                    tuple(
                        fraction_mod_one(value - origin)
                        for value, origin in zip(point, anchor, strict=True)
                    ),
                    species,
                ): count
                for (point, species), count in expected.items()
            }
        )
    if _coloured_orbit_points(assignments, setting) != expected:
        raise InvariantViolation("canonical orbit ordering changed the coloured crystal")
    return canonical


def _occupation_for_scaffold(
    scaffold: CrystalScaffold,
    composition: Composition,
    rows: tuple[WyckoffRowSpec, ...],
) -> OccupationTable:
    species_index = {z: index for index, z in enumerate(composition.support)}
    row_index = {row.label: index for index, row in enumerate(rows)}
    counts = [[0] * len(composition) for _ in rows]
    for assignment in scaffold.orbits:
        try:
            target_row = row_index[assignment.wyckoff_label]
            target_species = species_index[int(assignment.species)]
            counts[target_row][target_species] += 1
        except KeyError as exc:  # decoder validation should already make this impossible
            raise InvariantViolation(
                "canonical orbit lies outside its row/species axes"
            ) from exc
    return OccupationTable(tuple(RowAction(tuple(value)) for value in counts))


def _reachability(
    composition: Composition, nu: int, setting: SpaceGroupSetting
) -> SuffixReachability:
    return SuffixReachability(composition, nu, canonical_rows(setting))


def _cover_decision_target(
    scaffold: CrystalScaffold,
    composition: Composition,
    setting: SpaceGroupSetting,
    occupation: OccupationTable,
    rows: tuple[WyckoffRowSpec, ...],
    *,
    catalogue: CertifiedCatalogue,
    max_formula_units: int,
    bound: AtomBound,
    num_space_groups: int,
) -> CoverDecisionTarget:
    settings = tuple(
        sorted(catalogue.settings.values(), key=lambda value: value.international_number)
    )
    # Shared with NeuralCoverSampler, so both normalize over one support.  For
    # the legacy bound the admission test is group independent and the target's
    # own nu passed it, so these masks equal the historical ones bit for bit.
    feasible_nu = set(
        feasible_formula_units(
            composition,
            settings,
            max_formula_units=max_formula_units,
            bound=bound,
            oracle=_reachability,
        )
    )
    nu_mask = tuple(multiplier in feasible_nu for multiplier in range(1, max_formula_units + 1))

    group_mask_values = [False] * num_space_groups
    for candidate in legal_settings(
        composition, scaffold.nu, settings, bound=bound, oracle=_reachability
    ):
        group_mask_values[candidate.international_number - 1] = True

    oracle = SuffixReachability(composition, scaffold.nu, rows)
    if not oracle.accepts(occupation):
        raise InvariantViolation("canonical scaffold is absent from the exact suffix fibre")
    residual = oracle.initial_residual
    decisions: list[RowDecisionTarget] = []
    for index, (row, selected) in enumerate(zip(rows, occupation.actions, strict=True)):
        allowed = oracle.allowed_actions(index, residual)
        try:
            selected_index = allowed.index(selected)
        except ValueError as exc:
            raise InvariantViolation(
                "canonical occupation is absent from its completion-safe action set"
            ) from exc
        after = oracle.advance(index, residual, selected, require_completion_safe=True)
        decisions.append(
            RowDecisionTarget(
                row_index=index,
                row=row,
                residual_before=residual,
                legal_actions=allowed,
                selected_action_index=selected_index,
                selected_action=selected,
                residual_after=after,
            )
        )
        residual = after
    if any(residual):
        raise InvariantViolation("teacher-forced cover trace did not terminate at zero")
    return CoverDecisionTarget(
        nu_legal_mask=nu_mask,
        nu_target_index=scaffold.nu - 1,
        group_legal_mask=tuple(group_mask_values),
        group_target_index=setting.international_number - 1,
        rows=rows,
        occupation=occupation,
        row_decisions=tuple(decisions),
    )


def lift_exact_fixture_target(
    scaffold: CrystalScaffold,
    metric: npt.ArrayLike,
    *,
    catalogue: CertifiedCatalogue = MINIMAL_CATALOGUE,
    domain: DomainConfig | None = None,
    num_space_groups: int = 230,
    collision_tolerance: float = 1.0e-8,
    symmetry_tolerance: float = 1.0e-8,
) -> ExactFixtureTrainingTarget:
    """Lift one validated exact fixture into deterministic ML targets.

    The input must already contain exact Wyckoff labels, exact intrinsic
    parameters, a pinned P1/P-1 setting, and an invariant SPD metric.  The
    function validates the exact crystal, removes certified representation
    redundancies, constructs the same suffix masks used at sampling time, and
    returns ``z=chart.encode_latent(G)`` and the gauge-reduced ``eta``.

    Raw Cartesian/CIF structures and inferred symmetries are intentionally not
    accepted: this function is the post-standardization lift, not a symmetry
    discovery algorithm.
    """

    if not isinstance(scaffold, CrystalScaffold):
        raise TypeError("scaffold must be a CrystalScaffold")
    if not isinstance(catalogue, CertifiedCatalogue):
        raise TypeError("catalogue must be a CertifiedCatalogue")
    selected_domain = DomainConfig() if domain is None else domain
    if not isinstance(selected_domain, DomainConfig):
        raise TypeError("domain must be a DomainConfig")
    if (
        isinstance(num_space_groups, bool)
        or not isinstance(num_space_groups, int)
        or num_space_groups < 1
    ):
        raise DomainError("num_space_groups must be a positive integer")
    if (
        not np.isfinite(collision_tolerance)
        or collision_tolerance <= 0.0
        or not np.isfinite(symmetry_tolerance)
        or symmetry_tolerance <= 0.0
    ):
        raise DomainError("continuous decoder tolerances must be positive")
    catalogue.validate()
    if any(key not in _CERTIFIED_LIFT_KEYS for key in catalogue.supported_keys):
        raise CatalogueError(
            "the exact target lift is certified only for P1/P-1 fixture catalogues"
        )
    if max(catalogue.supported_international_numbers) > num_space_groups:
        raise DomainError("selector group vocabulary cannot represent the catalogue")
    setting = catalogue.get(scaffold.space_group)
    if any(
        isinstance(assignment.species, bool) or not isinstance(assignment.species, int)
        for assignment in scaffold.orbits
    ):
        raise DomainError("catalogue ML targets require integer atomic-number species")
    if setting.key not in _CERTIFIED_LIFT_KEYS:
        raise CatalogueError("the exact target lift is certified only for P1/P-1")

    value = np.asarray(metric, dtype=np.float64)
    decoded = ExactCrystalDecoder(catalogue).decode(scaffold, value)
    if decoded.translation_index != 1:
        raise InvariantViolation(
            "training targets must be translation-primitive; hidden translations remain"
        )
    # Fail with the public domain error before any fixture-specific integer
    # species ordering is attempted.
    ContinuousCover.from_exact_scaffold(scaffold)
    canonical = _canonical_scaffold(scaffold, setting)
    canonical_decoded = ExactCrystalDecoder(catalogue).decode(canonical, value)
    if canonical_decoded.translation_index != 1:
        raise InvariantViolation("canonicalization introduced a hidden translation")

    cover = ContinuousCover.from_exact_scaffold(canonical)
    composition = cover.composition
    if not set(composition.support).issubset(selected_domain.element_vocabulary):
        raise DomainError("composition contains an element outside the model vocabulary")
    if len(composition) > selected_domain.max_distinct_elements:
        raise DomainError("composition exceeds the configured distinct-element limit")
    if scaffold.nu > selected_domain.max_formula_units:
        raise DomainError("nu exceeds the selector vocabulary")
    if not selected_domain.atom_bound.admits(
        scaffold.nu, composition.total_atoms, setting.centering_index
    ):
        raise DomainError("target exceeds the configured raw atom bound")

    rows = canonical_rows(setting)
    occupation = _occupation_for_scaffold(canonical, composition, rows)
    decisions = _cover_decision_target(
        canonical,
        composition,
        setting,
        occupation,
        rows,
        catalogue=catalogue,
        max_formula_units=selected_domain.max_formula_units,
        bound=selected_domain.atom_bound,
        num_space_groups=num_space_groups,
    )
    layout = IntrinsicCoverLayout(
        cover, catalogue=catalogue, **selected_domain.layout_options()
    )
    lattice_latent = layout.chart.encode_latent(value)
    intrinsic = layout.encode_exact_parameters(canonical)
    continuous = layout.decode(
        ContinuousGeometry(lattice_latent, intrinsic),
        collision_tolerance=collision_tolerance,
        symmetry_tolerance=symmetry_tolerance,
    )
    if continuous.translation_index != 1:
        raise InvariantViolation(
            "continuous target round-trip has an unresolved hidden translation"
        )
    if not np.allclose(continuous.metric, value, atol=1.0e-9, rtol=1.0e-9):
        raise InvariantViolation("continuous lattice target failed its metric round-trip")
    _require_origin_equivalent_roundtrip(
        canonical_decoded.fractional_positions_float,
        canonical_decoded.species,
        continuous.fractional_positions,
        continuous.atomic_numbers,
        continuous.lattice,
        tolerance=symmetry_tolerance,
    )
    return ExactFixtureTrainingTarget(
        composition=composition,
        canonical_scaffold=canonical,
        cover=cover,
        cover_decisions=decisions,
        layout=layout,
        metric=value,
        lattice_latent=lattice_latent,
        intrinsic_coordinates=intrinsic,
        period_basis=layout.period_basis,
    )


def lift_canonical_catalogue_target(
    scaffold: CrystalScaffold,
    metric: npt.ArrayLike,
    *,
    catalogue: CertifiedCatalogue,
    domain: DomainConfig | None = None,
    num_space_groups: int = 230,
) -> CanonicalCatalogueTrainingTarget:
    """Lift one already canonicalized all-catalogue scaffold into ML targets.

    This function is deliberately post-standardization.  It checks the exact
    Hall action, Wyckoff multiplicities, formula equations, declared centring,
    absence of any *additional* hidden translation, selector masks, and the
    invariant SPD chart.  Origin and orbit-chart ambiguity must have been
    resolved by a certified raw-structure assigner before this function is
    called.
    """

    if not isinstance(scaffold, CrystalScaffold):
        raise TypeError("scaffold must be a CrystalScaffold")
    if not isinstance(catalogue, CertifiedCatalogue):
        raise TypeError("catalogue must be a CertifiedCatalogue")
    catalogue.validate(require_all_230=True)
    selected_domain = DomainConfig() if domain is None else domain
    if not isinstance(selected_domain, DomainConfig):
        raise TypeError("domain must be a DomainConfig")
    if (
        isinstance(num_space_groups, bool)
        or not isinstance(num_space_groups, int)
        or num_space_groups < 230
    ):
        raise DomainError("the complete catalogue requires a 230-group selector vocabulary")
    setting = catalogue.get(scaffold.space_group)
    row_rank = {row.label: index for index, row in enumerate(canonical_rows(setting))}
    normalized_assignments = tuple(
        OrbitAssignment(
            assignment.wyckoff_label,
            assignment.species,
            _canonical_parameter(assignment.parameter),
        )
        for assignment in scaffold.orbits
    )
    ordered_assignments = tuple(
        sorted(
            normalized_assignments,
            key=lambda item: (
                row_rank[item.wyckoff_label],
                int(item.species),
                item.parameter,
            ),
        )
    )
    canonical = CrystalScaffold(
        scaffold.nu,
        scaffold.space_group,
        scaffold.reduced_composition,
        ordered_assignments,
    )
    value = np.asarray(metric, dtype=np.float64)
    decoded = ExactCrystalDecoder(catalogue).decode(canonical, value)
    if decoded.hidden_translation_index != 1:
        raise InvariantViolation(
            "canonical target contains translations beyond declared centring"
        )

    cover = ContinuousCover.from_exact_scaffold(canonical)
    composition = cover.composition
    if not set(composition.support).issubset(selected_domain.element_vocabulary):
        raise DomainError("composition contains an element outside the model vocabulary")
    if len(composition) > selected_domain.max_distinct_elements:
        raise DomainError("composition exceeds the configured distinct-element limit")
    if canonical.nu > selected_domain.max_formula_units:
        raise DomainError("nu exceeds the selector vocabulary")
    if not selected_domain.atom_bound.admits(
        canonical.nu, composition.total_atoms, setting.centering_index
    ):
        raise DomainError("target exceeds the configured conventional atom bound")

    rows = canonical_rows(setting)
    occupation = _occupation_for_scaffold(canonical, composition, rows)
    decisions = _cover_decision_target(
        canonical,
        composition,
        setting,
        occupation,
        rows,
        catalogue=catalogue,
        max_formula_units=selected_domain.max_formula_units,
        bound=selected_domain.atom_bound,
        num_space_groups=num_space_groups,
    )
    layout = IntrinsicCoverLayout(
        cover, catalogue=catalogue, **selected_domain.layout_options()
    )
    lattice_latent = layout.chart.encode_latent(value)
    if not np.allclose(
        layout.chart.decode_latent(lattice_latent.tolist()),
        value,
        atol=1.0e-9,
        rtol=1.0e-9,
    ):
        raise InvariantViolation("catalogue lattice target failed its chart round-trip")

    intrinsic = layout.encode_exact_parameters(canonical)
    continuous = layout.decode(
        ContinuousGeometry(lattice_latent, intrinsic),
        collision_tolerance=1.0e-8,
        symmetry_tolerance=1.0e-8,
    )
    if continuous.hidden_translation_index != 1:
        raise InvariantViolation(
            "continuous catalogue target retains an undeclared hidden translation"
        )
    if not np.allclose(continuous.metric, value, atol=1.0e-9, rtol=1.0e-9):
        raise InvariantViolation("continuous catalogue metric failed its round-trip")
    _require_origin_equivalent_roundtrip(
        decoded.fractional_positions_float,
        decoded.species,
        continuous.fractional_positions,
        continuous.atomic_numbers,
        continuous.lattice,
        tolerance=1.0e-8,
    )
    return CanonicalCatalogueTrainingTarget(
        composition=composition,
        canonical_scaffold=canonical,
        cover=cover,
        cover_decisions=decisions,
        layout=layout,
        metric=value,
        lattice_latent=lattice_latent,
        intrinsic_coordinates=intrinsic,
        period_basis=layout.period_basis,
    )


__all__ = [
    "CanonicalCatalogueTrainingTarget",
    "CoverDecisionTarget",
    "ExactFixtureTrainingTarget",
    "RowDecisionTarget",
    "lift_exact_fixture_target",
    "lift_canonical_catalogue_target",
]
