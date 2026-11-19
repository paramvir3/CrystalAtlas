"""Finite, untrained reference law over any certified catalogue groups (review m3).

This module is an executable integration reference for the exact factorization;
it is **not** the learned CrystalAtlas model and it is not a physical energy or
stability predictor.  Its probability law is deliberately simple and explicit:

* compositions are uniform over the exact grammar leaves (or one S-DNG fibre),
* discrete covers are uniform over all feasible ``(nu, g, h)`` leaves, and
* lattice-chart coefficients are uniform in the bounded invariant Euclidean ball
  (``loewner-v2``: on the Loewner support within the domain's volumetric window).

Orbit parameters are drawn without replacement from a finite exact rational
grid after special-position and cross-row collision classes have been removed.

Validity holds for every group, not only P1/P-1.  A pooled parameter's orbit has
the row's declared multiplicity, so each of its points has exactly the row's
site-symmetry group.  Orbits of one row are therefore equal or disjoint.  Orbits
of different rows are disjoint because the Wyckoff positions partition the cell
and the grids exclude specializations: a point of a generic orbit of one row
would otherwise lie on another row's locus, which the pool construction removes
(stabilizers alone do not decide this; Pm-3m 3c and 3d have the same site group
and are distinct orbits).  Every pool of every requested group is checked
exactly (:func:`certify_disjoint_orbit_pools`).
Each row's grid denominator is the smallest odd ``q`` at least
``parameter_grid_denominator`` that supplies the ``capacity // multiplicity``
distinct orbits a cover can need.  Every feasible cover leaf is therefore
realizable, which keeps the law exact without retry.  The default
``("P1", "P-1")`` law is unchanged, because its declared grid already suffices.
There is no candidate retry, repair, relaxation, or post-generation filtering.
The generic :mod:`crystalatlas.pipeline` records any failed stage as a first-class
attempt.
"""

from __future__ import annotations

from collections import Counter, OrderedDict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from fractions import Fraction
from itertools import product
from math import isfinite
from types import MappingProxyType
from typing import Final, TypeAlias

import numpy as np

from .algebra import FractionVector, exact_row_matmul
from .catalogue import MINIMAL_CATALOGUE, CertifiedCatalogue, SpaceGroupSetting
from .composition import Composition, CompositionDomain, CompositionGrammar
from .config import CrystalAtlasConfig
from .constraints import OccupationTable, SuffixReachability, WyckoffRowSpec
from .decoder import (
    CrystalScaffold,
    DecodedCrystal,
    ExactCrystalDecoder,
    OrbitAssignment,
)
from .errors import DomainError, InvariantViolation, NumericalError
from .lattice import (
    InvariantSPDChart,
    LoewnerSPDChart,
    loewner_lower_bound,
    scale_metric_to_volume,
)
from .pipeline import (
    GenerationEngine,
    GenerationMode,
    GenerationRequest,
    GenerationTrace,
    VerificationResult,
)


@dataclass(slots=True)
class NumpyRandrangeAdapter:
    """Expose an exact ``randrange(stop)`` interface over a NumPy generator.

    Arbitrarily large Python integers are supported.  Rejection here operates
    only on raw entropy words to remove modulo bias; it is not a crystal-level
    retry and cannot condition the generated scientific outcome on validity.
    """

    generator: np.random.Generator

    def __post_init__(self) -> None:
        if not isinstance(self.generator, np.random.Generator):
            raise TypeError("generator must be a numpy.random.Generator")

    def randrange(self, stop: int, /) -> int:
        if isinstance(stop, bool) or not isinstance(stop, int):
            raise TypeError("stop must be a positive integer")
        if stop <= 0:
            raise ValueError("stop must be a positive integer")
        if stop == 1:
            return 0

        words = max(1, (stop.bit_length() + 63) // 64)
        modulus = 1 << (64 * words)
        acceptance_limit = modulus - modulus % stop
        while True:
            value = 0
            for word_index in range(words):
                word = int(self.generator.bit_generator.random_raw())
                value |= word << (64 * word_index)
            if value < acceptance_limit:
                return value % stop


@dataclass(frozen=True, slots=True)
class ReferenceLawConfig:
    """Finite support choices specific to the untrained reference law."""

    space_groups: tuple[str, ...] = ("P1", "P-1")
    parameter_grid_denominator: int = 5

    def __post_init__(self) -> None:
        groups = tuple(self.space_groups)
        object.__setattr__(self, "space_groups", groups)
        if not groups or any(not isinstance(group, str) or not group for group in groups):
            raise DomainError("reference law needs nonempty explicit space-group keys")
        if len(groups) != len(set(groups)):
            raise DomainError("reference-law space-group keys must be unique")
        denominator = self.parameter_grid_denominator
        if (
            isinstance(denominator, bool)
            or not isinstance(denominator, int)
            or denominator < 3
            or denominator % 2 == 0
        ):
            raise DomainError("parameter_grid_denominator must be an odd integer >= 3")


@dataclass(frozen=True, slots=True)
class ReferenceCover:
    """One exact arithmetic cover ``(c, nu, g, h)`` without geometry."""

    composition: Composition
    nu: int
    space_group: str
    rows: tuple[WyckoffRowSpec, ...]
    occupation: OccupationTable

    def __post_init__(self) -> None:
        if not isinstance(self.composition, Composition):
            raise TypeError("composition must be a Composition")
        if isinstance(self.nu, bool) or not isinstance(self.nu, int) or self.nu <= 0:
            raise DomainError("nu must be a positive integer")
        if not isinstance(self.space_group, str) or not self.space_group:
            raise DomainError("space_group must be a nonempty certified key")
        rows = tuple(self.rows)
        object.__setattr__(self, "rows", rows)
        if any(not isinstance(row, WyckoffRowSpec) for row in rows):
            raise TypeError("rows must contain WyckoffRowSpec values")
        if not isinstance(self.occupation, OccupationTable):
            raise TypeError("occupation must be an OccupationTable")
        oracle = SuffixReachability(self.composition, self.nu, rows)
        if not oracle.accepts(self.occupation):
            raise InvariantViolation("reference cover violates the exact occupation equations")

    @property
    def raw_atom_count(self) -> int:
        return self.nu * self.composition.total_atoms


@dataclass(frozen=True, slots=True)
class ReferenceCoverFiber:
    """A nonempty ``(c, nu, g)`` fibre and its exact leaf count."""

    space_group: str
    oracle: SuffixReachability

    def __post_init__(self) -> None:
        if not isinstance(self.space_group, str) or not self.space_group:
            raise DomainError("cover fibre needs a nonempty space-group key")
        if not isinstance(self.oracle, SuffixReachability):
            raise TypeError("cover fibre oracle must be a SuffixReachability")
        if self.oracle.number_of_completions <= 0:
            raise DomainError("a ReferenceCoverFiber must be arithmetic-feasible")

    @property
    def nu(self) -> int:
        return self.oracle.multiplier

    @property
    def number_of_completions(self) -> int:
        return self.oracle.number_of_completions


@dataclass(frozen=True, slots=True, eq=False)
class ReferenceGeometry:
    """Exact orbit scaffold plus an invariant floating-point SPD metric."""

    scaffold: CrystalScaffold
    chart_coefficients: tuple[float, ...]
    metric: np.ndarray

    def __post_init__(self) -> None:
        if not isinstance(self.scaffold, CrystalScaffold):
            raise TypeError("scaffold must be a CrystalScaffold")
        coefficients = tuple(float(value) for value in self.chart_coefficients)
        if any(not isfinite(value) for value in coefficients):
            raise DomainError("chart coefficients must be finite")
        metric = np.array(self.metric, dtype=np.float64, copy=True)
        if metric.shape != (3, 3) or not np.all(np.isfinite(metric)):
            raise DomainError("metric must be a finite 3x3 matrix")
        metric.setflags(write=False)
        object.__setattr__(self, "chart_coefficients", coefficients)
        object.__setattr__(self, "metric", metric)

    def __eq__(self, other: object) -> bool:
        return (
            isinstance(other, ReferenceGeometry)
            and self.scaffold == other.scaffold
            and self.chart_coefficients == other.chart_coefficients
            and np.array_equal(self.metric, other.metric)
        )


ReferenceTrace: TypeAlias = GenerationTrace[
    Composition, ReferenceCover, ReferenceGeometry, DecodedCrystal
]


@dataclass(frozen=True, slots=True, eq=False)
class ReferenceSample:
    """Flattened successful reference attempt for examples and command-line use."""

    mode: GenerationMode
    sample_index: int
    seed: int
    composition: Composition
    cover: ReferenceCover
    metric: np.ndarray
    decoded: DecodedCrystal
    verification: VerificationResult

    def __post_init__(self) -> None:
        metric = np.array(self.metric, dtype=np.float64, copy=True)
        if metric.shape != (3, 3) or not np.all(np.isfinite(metric)):
            raise DomainError("reference-sample metric must be finite and 3x3")
        metric.setflags(write=False)
        object.__setattr__(self, "metric", metric)

    @classmethod
    def from_trace(
        cls, mode: GenerationMode, trace: ReferenceTrace
    ) -> ReferenceSample:
        if not trace.succeeded:
            raise InvariantViolation(
                "cannot flatten a failed reference attempt; inspect its GenerationTrace"
            )
        assert trace.composition is not None
        assert trace.cover is not None
        assert trace.state is not None
        assert trace.crystal is not None
        assert trace.verification is not None
        return cls(
            mode=mode,
            sample_index=trace.sample_index,
            seed=trace.seed,
            composition=trace.composition,
            cover=trace.cover,
            metric=trace.state.metric,
            decoded=trace.crystal,
            verification=trace.verification,
        )

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, ReferenceSample):
            return False
        return (
            self.mode == other.mode
            and self.sample_index == other.sample_index
            and self.seed == other.seed
            and self.composition == other.composition
            and self.cover == other.cover
            and np.array_equal(self.metric, other.metric)
            and self.decoded.scaffold == other.decoded.scaffold
            and np.array_equal(self.decoded.metric, other.decoded.metric)
            and np.array_equal(self.decoded.lattice, other.decoded.lattice)
            and self.decoded.fractional_positions == other.decoded.fractional_positions
            and self.decoded.species == other.decoded.species
            and self.decoded.translation_subgroup == other.decoded.translation_subgroup
            and self.decoded.audit == other.decoded.audit
            and self.verification == other.verification
        )


def _row_specs(setting: SpaceGroupSetting) -> tuple[WyckoffRowSpec, ...]:
    rows = tuple(
        WyckoffRowSpec(
            label=label,
            multiplicity=position.multiplicity,
            free_dimension=position.dimension,
        )
        for label, position in setting.wyckoff_positions.items()
    )
    # Bounded capacity-one branching comes first.  This does not alter the set
    # of occupation tables, but avoids exposing the DP to the enormous action
    # rectangle of a repeatable general row before the bounded residual is known.
    return tuple(sorted(rows, key=lambda row: (row.free_dimension != 0, row.label)))


def _ordered_sample_without_replacement(
    population: Sequence[FractionVector],
    sample_size: int,
    rng: NumpyRandrangeAdapter,
) -> tuple[FractionVector, ...]:
    """Draw an ordered injection using O(sample_size) auxiliary memory."""

    if sample_size < 0 or sample_size > len(population):
        raise DomainError("sample size lies outside the exact parameter pool")
    redirects: dict[int, int] = {}
    selected: list[FractionVector] = []
    for offset in range(sample_size):
        remaining = len(population) - offset
        draw = rng.randrange(remaining)
        selected_index = redirects.get(draw, draw)
        final_index = remaining - 1
        redirects[draw] = redirects.get(final_index, final_index)
        redirects.pop(final_index, None)
        selected.append(population[selected_index])
    return tuple(selected)


class ReferenceCompositionSampler:
    """Exact uniform composition-leaf sampler for DNG and S-DNG."""

    def __init__(self, config: CrystalAtlasConfig) -> None:
        domain = config.domain
        self.grammar = CompositionGrammar(
            CompositionDomain(
                elements=domain.dng_element_vocabulary,
                max_elements=domain.max_distinct_elements,
                max_atoms=domain.composition_atom_bound,
            )
        )

    def sample_unconditional(self, rng: np.random.Generator) -> Composition:
        return self.grammar.sample_uniform(NumpyRandrangeAdapter(rng))

    def sample_given_species(
        self, species: tuple[int, ...], rng: np.random.Generator
    ) -> Composition:
        return self.grammar.sample_uniform(NumpyRandrangeAdapter(rng), species)


class ReferenceCoverSampler:
    """Uniform sampler over exact feasible cover leaves, never local logits."""

    def __init__(
        self,
        config: CrystalAtlasConfig,
        law: ReferenceLawConfig,
        catalogue: CertifiedCatalogue = MINIMAL_CATALOGUE,
    ) -> None:
        catalogue.validate()
        for group in law.space_groups:
            catalogue.get(group)
        self.config = config
        self.law = law
        self.catalogue = catalogue
        self._fiber_cache: OrderedDict[
            Composition, tuple[ReferenceCoverFiber, ...]
        ] = OrderedDict()

    def _validate_composition(self, composition: Composition) -> None:
        if not isinstance(composition, Composition):
            raise TypeError("composition must be a Composition")
        domain = self.config.domain
        if not set(composition.support).issubset(domain.element_vocabulary):
            raise DomainError("composition contains species outside the package vocabulary")
        if composition.total_atoms > domain.composition_atom_bound:
            raise DomainError("even nu=1 exceeds the configured raw-cell atom bound")

    def feasible_fibers(self, composition: Composition) -> tuple[ReferenceCoverFiber, ...]:
        """Enumerate only nonempty ``(nu, g)`` fibres in normative order."""

        self._validate_composition(composition)
        cached = self._fiber_cache.get(composition)
        if cached is not None:
            self._fiber_cache.move_to_end(composition)
            return cached
        domain = self.config.domain
        largest_nu = min(
            domain.max_formula_units,
            domain.max_raw_atoms // composition.total_atoms,
        )
        fibers: list[ReferenceCoverFiber] = []
        for nu in range(1, largest_nu + 1):
            for group in self.law.space_groups:
                setting = self.catalogue.get(group)
                if not domain.atom_bound.admits(
                    nu, composition.total_atoms, setting.centering_index
                ):
                    continue
                oracle = SuffixReachability(composition, nu, _row_specs(setting))
                if oracle.number_of_completions > 0:
                    fibers.append(ReferenceCoverFiber(group, oracle))
        if not fibers:
            raise DomainError("composition has no arithmetic-feasible certified cover")
        result = tuple(fibers)
        self._fiber_cache[composition] = result
        if len(self._fiber_cache) > 1024:
            self._fiber_cache.popitem(last=False)
        return result

    def sample_cover(
        self, composition: Composition, rng: np.random.Generator
    ) -> ReferenceCover:
        fibers = self.feasible_fibers(composition)
        total = sum(fiber.number_of_completions for fiber in fibers)
        rank = NumpyRandrangeAdapter(rng).randrange(total)
        for fiber in fibers:
            if rank < fiber.number_of_completions:
                occupation = fiber.oracle.completion_at(rank)
                return ReferenceCover(
                    composition=composition,
                    nu=fiber.nu,
                    space_group=fiber.space_group,
                    rows=fiber.oracle.rows,
                    occupation=occupation,
                )
            rank -= fiber.number_of_completions
        raise AssertionError("exact cover unranking exhausted a positive leaf set")


MAX_PARAMETER_GRID_DENOMINATOR: Final = 99
"""Largest odd per-row grid denominator; all 230 groups need at most 41 at the
default domain (`tests/test_reference_all_groups.py`)."""


def certify_disjoint_orbit_pools(
    setting: SpaceGroupSetting, pools: Mapping[str, Sequence[FractionVector]]
) -> None:
    """Exactly check that pooled orbits have their multiplicity and share no point.

    Fixed (dimension-0) sites are included, so a pooled orbit can never meet a
    fixed site either.
    """

    owner: dict[FractionVector, tuple[str, int]] = {}
    rows = {label: setting.wyckoff(label) for label in pools}
    for row in setting.wyckoff_positions.values():
        if row.dimension == 0:
            rows.setdefault(row.label, row)
    for label, row in rows.items():
        parameters: Sequence[FractionVector] = pools.get(label, ((),))
        for index, parameter in enumerate(parameters):
            points = row.expand_exact(parameter, require_declared_multiplicity=True)
            for point in points:
                previous = owner.setdefault(point, (label, index))
                if previous != (label, index):
                    raise InvariantViolation(
                        f"{setting.key}: pooled orbits {previous} and {(label, index)} "
                        "share a point"
                    )


class ReferenceGeometrySampler:
    """Collision-free rational orbit and invariant-SPD reference sampler."""

    def __init__(
        self,
        config: CrystalAtlasConfig,
        law: ReferenceLawConfig,
        catalogue: CertifiedCatalogue = MINIMAL_CATALOGUE,
    ) -> None:
        self.config = config
        self.law = law
        self.catalogue = catalogue
        self._parameter_pools: dict[tuple[str, str], tuple[FractionVector, ...]] = {}
        denominators: dict[tuple[str, str], int] = {}
        for group in law.space_groups:
            setting = catalogue.get(group)
            group_pools: dict[str, tuple[FractionVector, ...]] = {}
            for row in setting.wyckoff_positions.values():
                if row.dimension == 0:
                    continue
                maximum_needed = (
                    config.domain.atom_bound.capacity(setting.centering_index)
                    // row.multiplicity
                )
                if maximum_needed == 0:
                    # No feasible cover can occupy a row larger than the capacity.
                    continue
                denominator = law.parameter_grid_denominator
                pool = self._build_parameter_pool(setting, row.label, denominator)
                while len(pool) < maximum_needed:
                    denominator += 2
                    if denominator > MAX_PARAMETER_GRID_DENOMINATOR:
                        raise DomainError(
                            f"no odd grid denominator <= {MAX_PARAMETER_GRID_DENOMINATOR} "
                            f"supplies the {maximum_needed} safe {group}:{row.label} orbits "
                            "that the support requires"
                        )
                    pool = self._build_parameter_pool(setting, row.label, denominator)
                self._parameter_pools[(group, row.label)] = pool
                denominators[(group, row.label)] = denominator
                group_pools[row.label] = pool
            certify_disjoint_orbit_pools(setting, group_pools)
        self.parameter_grid_denominators: Mapping[tuple[str, str], int] = MappingProxyType(
            denominators
        )

    def _build_parameter_pool(
        self, setting: SpaceGroupSetting, row_label: str, denominator: int | None = None
    ) -> tuple[FractionVector, ...]:
        row = setting.wyckoff(row_label)
        if row.dimension == 0:
            return ((),)
        fixed_sites = {
            point
            for other in setting.wyckoff_positions.values()
            if other.dimension == 0
            for point in other.expand_exact(())
        }
        if denominator is None:
            denominator = self.law.parameter_grid_denominator
        seen_orbits: set[tuple[FractionVector, ...]] = set()
        accepted: list[FractionVector] = []
        for numerators in product(range(denominator), repeat=row.dimension):
            unit = tuple(Fraction(value, denominator) for value in numerators)
            parameter = exact_row_matmul(unit, row.period_basis)
            try:
                expanded = row.expand_exact(parameter, require_declared_multiplicity=True)
            except InvariantViolation:
                # Exact support construction removes special-position defect strata.
                continue
            orbit_key = tuple(sorted(expanded))
            if any(point in fixed_sites for point in expanded) or orbit_key in seen_orbits:
                continue
            seen_orbits.add(orbit_key)
            accepted.append(parameter)
        return tuple(accepted)

    def _sample_scaffold(
        self,
        composition: Composition,
        cover: ReferenceCover,
        rng: NumpyRandrangeAdapter,
    ) -> CrystalScaffold:
        if cover.composition != composition:
            raise InvariantViolation("cover and geometry composition keys disagree")
        setting = self.catalogue.get(cover.space_group)
        expected_rows = _row_specs(setting)
        if cover.rows != expected_rows:
            raise InvariantViolation("cover row manifest disagrees with the pinned catalogue")

        assignments: list[OrbitAssignment] = []
        for row_spec, action in zip(cover.rows, cover.occupation.actions, strict=True):
            row = setting.wyckoff(row_spec.label)
            number_orbits = action.number_of_orbits
            if number_orbits == 0:
                continue
            if row.dimension == 0:
                if number_orbits != 1:
                    raise InvariantViolation("a fixed Wyckoff row was occupied more than once")
                parameters: tuple[FractionVector, ...] = ((),)
            else:
                parameters = _ordered_sample_without_replacement(
                    self._parameter_pools[(setting.key, row.label)], number_orbits, rng
                )
            cursor = 0
            for species, count in zip(
                composition.support, action.counts, strict=True
            ):
                for _ in range(count):
                    assignments.append(
                        OrbitAssignment(row.label, species, parameters[cursor])
                    )
                    cursor += 1
            if cursor != number_orbits:
                raise AssertionError("orbit-parameter assignment lost an occupation")

        return CrystalScaffold.from_mapping(
            nu=cover.nu,
            space_group=cover.space_group,
            reduced_composition=composition.as_dict(),
            orbits=assignments,
        )

    def _sample_metric(
        self,
        setting: SpaceGroupSetting,
        raw_atom_count: int,
        rng: np.random.Generator,
    ) -> tuple[tuple[float, ...], np.ndarray]:
        domain = self.config.domain
        rotations = [
            np.asarray(rotation, dtype=np.float64)
            for rotation in setting.point_group_rotations
        ]
        reference = scale_metric_to_volume(
            np.asarray(setting.reference_metric, dtype=np.float64),
            domain.reference_atomic_volume_angstrom3 * raw_atom_count,
        )
        if domain.lattice_chart == "loewner-v2":
            # Uniform in log-metric coefficients over the Loewner support (review B2).
            loewner = LoewnerSPDChart(
                rotations,
                reference_metric=reference,
                lower=loewner_lower_bound(
                    reference, domain.minimum_lattice_singular_value_angstrom
                ),
                upper=float(np.exp(domain.loewner_upper_log_eigenvalue)),
                tolerance=self.config.numerics.symmetry_atol,
            )
            sample = loewner.uniform_coefficients(
                rng, log_volume_window=domain.loewner_control_log_volume_window
            )
            return tuple(float(value) for value in sample), loewner.decode(sample)
        chart = InvariantSPDChart(
            rotations,
            reference_metric=reference,
            bound=domain.metric_radius,
            tolerance=self.config.numerics.symmetry_atol,
        )
        if (
            chart.minimum_lattice_singular_value_bound
            < domain.minimum_lattice_singular_value_angstrom
        ):
            raise DomainError(
                "reference lattice law violates the configured shortest-vector support"
            )
        radial_uniform = float(rng.random())
        if radial_uniform == 0.0:
            coefficients = np.zeros(chart.dimension, dtype=np.float64)
        else:
            direction = np.asarray(rng.standard_normal(chart.dimension), dtype=np.float64)
            norm = float(np.linalg.norm(direction))
            if not np.isfinite(norm) or norm == 0.0:
                raise NumericalError("could not construct a finite direction in lattice chart")
            ideal_radius = domain.metric_radius * radial_uniform ** (
                1.0 / chart.dimension
            )
            # ``random() < 1`` mathematically, but the power and product may
            # round to the closed boundary.  Preserve the declared open ball.
            radius = min(
                ideal_radius,
                float(np.nextafter(domain.metric_radius, 0.0)),
            )
            coefficients = radius * direction / norm
            coefficient_norm = float(np.linalg.norm(coefficients))
            if not np.isfinite(coefficient_norm):
                raise NumericalError("lattice-chart coefficient norm is not finite")
            if coefficient_norm >= domain.metric_radius:
                coefficients *= (
                    float(np.nextafter(domain.metric_radius, 0.0))
                    / coefficient_norm
                )
        return tuple(float(value) for value in coefficients), chart.decode(coefficients)

    def sample_geometry(
        self,
        composition: Composition,
        cover: ReferenceCover,
        rng: np.random.Generator,
    ) -> ReferenceGeometry:
        adapter = NumpyRandrangeAdapter(rng)
        scaffold = self._sample_scaffold(composition, cover, adapter)
        coefficients, metric = self._sample_metric(
            self.catalogue.get(cover.space_group), cover.raw_atom_count, rng
        )
        return ReferenceGeometry(scaffold, coefficients, metric)


class ReferenceDecoder:
    """Pipeline adapter around :class:`ExactCrystalDecoder`."""

    def __init__(self, catalogue: CertifiedCatalogue = MINIMAL_CATALOGUE) -> None:
        self.decoder = ExactCrystalDecoder(catalogue)

    def decode(
        self, cover: ReferenceCover, state: ReferenceGeometry
    ) -> DecodedCrystal:
        scaffold = state.scaffold
        if (
            scaffold.nu != cover.nu
            or scaffold.space_group != cover.space_group
            or dict(scaffold.composition) != cover.composition.as_dict()
        ):
            raise InvariantViolation("reference geometry does not belong to its cover")
        observed = Counter(
            (orbit.wyckoff_label, orbit.species) for orbit in scaffold.orbits
        )
        expected: Counter[tuple[str, int]] = Counter()
        for row, action in zip(cover.rows, cover.occupation.actions, strict=True):
            for species, count in zip(
                cover.composition.support, action.counts, strict=True
            ):
                expected[(row.label, species)] = count
        if observed != expected:
            raise InvariantViolation("scaffold orbit instances disagree with occupation table")
        return self.decoder.decode(scaffold, state.metric)


class ReferenceVerifier:
    """Audit adapter; it never substitutes a physical-validity claim."""

    def verify(
        self, composition: Composition, crystal: DecodedCrystal
    ) -> VerificationResult:
        reasons: list[str] = []
        if dict(crystal.scaffold.composition) != composition.as_dict():
            reasons.append("decoded reduced composition differs from conditioning formula")
        if not crystal.audit.passed:
            reasons.append("exact decoder audit did not pass")
        expected_counts = {
            species: crystal.scaffold.nu * count
            for species, count in composition.counts
        }
        if dict(crystal.conventional_counts) != expected_counts:
            reasons.append("decoded conventional counts violate nu*c")
        return VerificationResult(
            valid=not reasons,
            reasons=tuple(reasons),
            metadata=(
                ("law", "untrained-uniform-reference"),
                ("space_group_constraint", crystal.setting.key),
                ("nu", str(crystal.scaffold.nu)),
                ("translation_index", str(crystal.translation_index)),
                ("physical_prediction", "false"),
            ),
        )


class ReferenceGenerator:
    """End-to-end DNG/S-DNG/CSP reference generator with stage traces."""

    def __init__(
        self,
        config: CrystalAtlasConfig | None = None,
        law: ReferenceLawConfig | None = None,
        catalogue: CertifiedCatalogue = MINIMAL_CATALOGUE,
    ) -> None:
        config = CrystalAtlasConfig() if config is None else config
        law = ReferenceLawConfig() if law is None else law
        catalogue.validate()
        self.config = config
        self.law = law
        self.catalogue = catalogue
        self.composition_sampler = ReferenceCompositionSampler(config)
        self.cover_sampler = ReferenceCoverSampler(config, law, catalogue)
        self.geometry_sampler = ReferenceGeometrySampler(config, law, catalogue)
        self.decoder = ReferenceDecoder(catalogue)
        self.verifier = ReferenceVerifier()
        self.engine: GenerationEngine[
            Composition, ReferenceCover, ReferenceGeometry, DecodedCrystal
        ] = GenerationEngine(
            composition_sampler=self.composition_sampler,
            cover_sampler=self.cover_sampler,
            geometry_sampler=self.geometry_sampler,
            decoder=self.decoder,
            verifier=self.verifier,
        )

    def sample(self, request: GenerationRequest[Composition]) -> tuple[ReferenceTrace, ...]:
        """Return every raw attempt, including a possible failed-stage trace."""

        return self.engine.sample(request)

    def require_samples(
        self, request: GenerationRequest[Composition]
    ) -> tuple[ReferenceSample, ...]:
        """Flatten successful attempts; raise on a recorded failure without retrying."""

        traces = self.sample(request)
        failed = next((trace for trace in traces if not trace.succeeded), None)
        if failed is not None:
            raise InvariantViolation(
                f"reference attempt {failed.sample_index} failed at {failed.failed_stage}: "
                f"{failed.error}"
            )
        return tuple(ReferenceSample.from_trace(request.mode, trace) for trace in traces)


def build_reference_engine(
    config: CrystalAtlasConfig | None = None,
    law: ReferenceLawConfig | None = None,
    catalogue: CertifiedCatalogue = MINIMAL_CATALOGUE,
) -> GenerationEngine[Composition, ReferenceCover, ReferenceGeometry, DecodedCrystal]:
    """Build the generic pipeline configured with the untrained reference law."""

    return ReferenceGenerator(config, law, catalogue).engine


__all__ = [
    "NumpyRandrangeAdapter",
    "ReferenceCompositionSampler",
    "ReferenceCover",
    "ReferenceCoverFiber",
    "ReferenceCoverSampler",
    "ReferenceDecoder",
    "ReferenceGenerator",
    "ReferenceGeometry",
    "ReferenceGeometrySampler",
    "ReferenceLawConfig",
    "ReferenceSample",
    "ReferenceTrace",
    "ReferenceVerifier",
    "build_reference_engine",
]
