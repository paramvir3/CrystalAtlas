"""Continuous crystallographic cover coordinates and numerical decoding.

This module is the bridge between a learned real-valued flow state and the
exact catalogue.  Integer formula arithmetic and catalogue operations remain
exact; only lattice/coordinate evaluation and tolerance-based collision/group
checks are floating point.  Continuous origin freedom is removed axis by axis
using the exact fixed subspace and the unique ``+/- e_j`` chart parameter
certified for every pinned Wyckoff row.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from fractions import Fraction
from types import MappingProxyType

import numpy as np
import numpy.typing as npt

from .catalogue import MINIMAL_CATALOGUE, CertifiedCatalogue, SpaceGroupSetting
from .composition import Composition
from .decoder import CrystalScaffold
from .errors import (
    CatalogueError,
    DomainError,
    GeneratedSampleError,
    InvariantViolation,
)
from .lattice import (
    InvariantSPDChart,
    LoewnerSPDChart,
    loewner_lower_bound,
    scale_metric_to_volume,
)
from .periodic import (
    minimum_periodic_pair_distance as _minimum_pair_distance,
)
from .periodic import (
    periodic_pairs_within,
    shortest_nonzero_lattice_image,
)
from .standardization import (
    FloatingStructure,
    certify_floating_translation_subgroup,
)
from .wyckoff import stacked_cartesian_jacobian

FloatArray = npt.NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class OrbitType:
    """One species-decorated Wyckoff-orbit instance without its coordinates."""

    wyckoff_label: str
    atomic_number: int

    def __post_init__(self) -> None:
        if not self.wyckoff_label:
            raise DomainError("Wyckoff label cannot be empty")
        if isinstance(self.atomic_number, bool) or not 1 <= self.atomic_number <= 118:
            raise DomainError("atomic_number must lie in [1,118]")


@dataclass(frozen=True, slots=True)
class ContinuousCover:
    """Arithmetic cover used by a real-valued conditional geometry model."""

    nu: int
    space_group: str
    composition: Composition
    orbits: tuple[OrbitType, ...]

    def __post_init__(self) -> None:
        if isinstance(self.nu, bool) or not isinstance(self.nu, int) or self.nu <= 0:
            raise DomainError("nu must be a positive integer")
        if not self.space_group:
            raise DomainError("space_group must be an explicit catalogue key")
        if not isinstance(self.composition, Composition):
            raise TypeError("composition must be a Composition")
        orbits = tuple(self.orbits)
        if not orbits or any(not isinstance(orbit, OrbitType) for orbit in orbits):
            raise DomainError("orbits must be a nonempty tuple of OrbitType values")
        if any(orbit.atomic_number not in self.composition.support for orbit in orbits):
            raise DomainError("an orbit species lies outside the reduced formula")
        object.__setattr__(self, "orbits", orbits)

    @classmethod
    def from_exact_scaffold(cls, scaffold: CrystalScaffold) -> ContinuousCover:
        composition: dict[int, int] = {}
        for species, count in scaffold.reduced_composition:
            if not isinstance(species, int):
                raise DomainError("continuous ML covers require integer atomic numbers")
            composition[species] = count
        return cls(
            nu=scaffold.nu,
            space_group=scaffold.space_group,
            composition=Composition(tuple(sorted(composition.items()))),
            orbits=tuple(
                OrbitType(orbit.wyckoff_label, int(orbit.species))
                for orbit in scaffold.orbits
            ),
        )


@dataclass(frozen=True, slots=True)
class ContinuousGeometry:
    """One real state ``q=(z,eta)`` on a fixed arithmetic cover."""

    lattice_latent: FloatArray
    intrinsic_coordinates: FloatArray

    def __post_init__(self) -> None:
        z = np.asarray(self.lattice_latent, dtype=np.float64).copy()
        eta = np.asarray(self.intrinsic_coordinates, dtype=np.float64).copy()
        if z.ndim != 1 or eta.ndim != 1 or not np.all(np.isfinite(z)) or not np.all(
            np.isfinite(eta)
        ):
            raise DomainError("continuous geometry blocks must be finite vectors")
        z.setflags(write=False)
        eta.setflags(write=False)
        object.__setattr__(self, "lattice_latent", z)
        object.__setattr__(self, "intrinsic_coordinates", eta)


@dataclass(frozen=True, slots=True)
class ContinuousDecoderAudit:
    formula_matches: bool
    no_collisions: bool
    metric_is_invariant: bool
    selected_group_is_contained: bool
    minimum_distance: float
    translation_index: int

    @property
    def passed(self) -> bool:
        return (
            self.formula_matches
            and self.no_collisions
            and self.metric_is_invariant
            and self.selected_group_is_contained
        )


@dataclass(frozen=True, slots=True)
class ContinuousDecodedCrystal:
    """A floating conventional motif decoded from one continuous cover."""

    cover: ContinuousCover
    setting: SpaceGroupSetting
    geometry: ContinuousGeometry
    metric: FloatArray
    lattice: FloatArray
    fractional_positions: FloatArray
    atomic_numbers: npt.NDArray[np.int64]
    orbit_slices: tuple[slice, ...]
    jacobian_cartesian: FloatArray
    audit: ContinuousDecoderAudit

    def __post_init__(self) -> None:
        arrays = {
            "metric": (self.metric, np.float64),
            "lattice": (self.lattice, np.float64),
            "fractional_positions": (self.fractional_positions, np.float64),
            "atomic_numbers": (self.atomic_numbers, np.int64),
            "jacobian_cartesian": (self.jacobian_cartesian, np.float64),
        }
        for name, (value, dtype) in arrays.items():
            array = np.asarray(value, dtype=dtype).copy()
            array.setflags(write=False)
            object.__setattr__(self, name, array)

    @property
    def cartesian_positions(self) -> FloatArray:
        return self.fractional_positions @ self.lattice

    @property
    def conventional_counts(self) -> Mapping[int, int]:
        return MappingProxyType(dict(Counter(int(z) for z in self.atomic_numbers)))

    @property
    def translation_index(self) -> int:
        return self.audit.translation_index

    @property
    def hidden_translation_index(self) -> int:
        centering_index = self.setting.centering_index
        if self.translation_index % centering_index:
            raise InvariantViolation(
                "complete translation group is not divisible by declared centring"
            )
        return self.translation_index // centering_index

    def full_primitive_reduction(
        self,
        *,
        cartesian_tolerance: float,
    ) -> PrimitivePhysicalCrystal:
        """Remove every certified species-preserving motif translation.

        The returned cell remains in the sampled Cartesian frame.  This is a
        complete primitive reduction, not an actual-maximal-space-group or
        pinned-Hall classification; those are separate metadata operations.
        """

        return full_primitive_reduction(
            self,
            cartesian_tolerance=cartesian_tolerance,
        )


@dataclass(frozen=True, slots=True)
class PrimitivePhysicalCrystal:
    """A certified primitive quotient of a floating decoded cover motif."""

    lattice: FloatArray
    fractional_positions: FloatArray
    atomic_numbers: npt.NDArray[np.int64]
    basis_change: tuple[tuple[Fraction, ...], ...]
    inverse_basis_change: tuple[tuple[int, ...], ...]
    translation_subgroup: tuple[tuple[Fraction, ...], ...]
    translation_action_max_cartesian_error: float

    def __post_init__(self) -> None:
        lattice = np.asarray(self.lattice, dtype=np.float64).copy()
        positions = np.asarray(self.fractional_positions, dtype=np.float64).copy()
        numbers = np.asarray(self.atomic_numbers, dtype=np.int64).copy()
        if (
            lattice.shape != (3, 3)
            or float(np.linalg.det(lattice)) <= 0.0
            or positions.ndim != 2
            or positions.shape[1:] != (3,)
            or numbers.shape != (len(positions),)
            or not len(positions)
            or not np.all(np.isfinite(lattice))
            or not np.all(np.isfinite(positions))
        ):
            raise DomainError("primitive physical crystal arrays are invalid")
        if not np.isfinite(self.translation_action_max_cartesian_error) or (
            self.translation_action_max_cartesian_error < 0.0
        ):
            raise DomainError("translation action error must be finite and nonnegative")
        lattice.setflags(write=False)
        positions.setflags(write=False)
        numbers.setflags(write=False)
        object.__setattr__(self, "lattice", lattice)
        object.__setattr__(self, "fractional_positions", positions)
        object.__setattr__(self, "atomic_numbers", numbers)

    @property
    def translation_index(self) -> int:
        return len(self.translation_subgroup)

    @property
    def cartesian_positions(self) -> FloatArray:
        return self.fractional_positions @ self.lattice

    @property
    def counts(self) -> Mapping[int, int]:
        return MappingProxyType(dict(Counter(int(z) for z in self.atomic_numbers)))


def full_primitive_reduction(
    decoded: ContinuousDecodedCrystal,
    *,
    cartesian_tolerance: float,
) -> PrimitivePhysicalCrystal:
    """Certify and quotient the complete floating translation subgroup.

    Candidate translations are generated from the least-populated species,
    exactly as in the finite-group count audit.  Their reported floating
    values are then reconstructed as rationals with denominator dividing the
    proved group order, and every exact translation is rechecked against the
    coloured motif by perfect matching.  Only this certified exact subgroup is
    used to construct the primitive basis.
    """

    if not isinstance(decoded, ContinuousDecodedCrystal):
        raise TypeError("decoded must be a ContinuousDecodedCrystal")
    return primitive_reduce_arrays(
        decoded.lattice,
        decoded.fractional_positions,
        decoded.atomic_numbers,
        cartesian_tolerance=cartesian_tolerance,
    )


def primitive_reduce_arrays(
    lattice: npt.ArrayLike,
    fractional_positions: npt.ArrayLike,
    atomic_numbers: npt.ArrayLike,
    *,
    cartesian_tolerance: float,
) -> PrimitivePhysicalCrystal:
    """Certified complete primitive reduction of one floating coloured motif.

    This is the array form of :func:`full_primitive_reduction`; it is used for
    any right-handed periodic structure, including evaluation payloads.
    """

    if not np.isfinite(cartesian_tolerance) or cartesian_tolerance <= 0.0:
        raise DomainError("primitive-reduction tolerance must be finite and positive")
    structure = FloatingStructure(
        np.asarray(lattice, dtype=np.float64),
        np.asarray(fractional_positions, dtype=np.float64),
        tuple(int(value) for value in np.asarray(atomic_numbers).reshape(-1)),
    )
    cell = structure.lattice
    positions = structure.fractional_positions
    numbers = np.asarray(structure.species, dtype=np.int64)
    reported = _numerical_translation_subgroup(
        positions,
        numbers,
        cell,
        cartesian_tolerance,
    )
    certificate = certify_floating_translation_subgroup(
        structure,
        reported,
        len(reported),
        cartesian_tolerance=cartesian_tolerance,
    )
    basis = np.asarray(
        [[float(value) for value in row] for row in certificate.basis_change],
        dtype=np.float64,
    )
    inverse = np.asarray(certificate.inverse_basis_change, dtype=np.int64)
    primitive_lattice = basis @ cell
    transformed = np.mod(positions @ inverse, 1.0)
    index = len(certificate.translation_subgroup)

    # Translation fibres must become complete same-species cliques in the
    # primitive torus.  Connected-component-only clustering would accept
    # tolerance chains, so every resulting component is checked pairwise.
    within = _pairs_within_tolerance(
        transformed, transformed, primitive_lattice, cartesian_tolerance
    )
    adjacency = (within & (numbers[:, None] == numbers[None, :])) | np.eye(
        len(transformed), dtype=bool
    )
    components: list[tuple[int, ...]] = []
    unseen = set(range(len(transformed)))
    while unseen:
        seed = min(unseen)
        stack = [seed]
        component: set[int] = set()
        while stack:
            current = stack.pop()
            if current in component:
                continue
            component.add(current)
            stack.extend(int(value) for value in np.flatnonzero(adjacency[current]))
        unseen.difference_update(component)
        ordered = tuple(sorted(component))
        if len(ordered) != index or not bool(adjacency[np.ix_(ordered, ordered)].all()):
            raise InvariantViolation(
                "certified translation quotient did not form complete primitive fibres"
            )
        components.append(ordered)

    primitive_rows: list[tuple[int, np.ndarray]] = []
    for component in components:
        species = {int(numbers[value]) for value in component}
        if len(species) != 1:
            raise InvariantViolation("primitive translation fibre mixed chemical species")
        anchor = transformed[component[0]]
        lifted = []
        for value in component:
            delta = transformed[value] - anchor
            delta -= np.rint(delta)
            lifted.append(anchor + delta)
        representative = np.mod(np.mean(lifted, axis=0), 1.0)
        primitive_rows.append((species.pop(), representative))
    primitive_rows.sort(
        key=lambda item: (item[0], tuple(float(value) for value in item[1]))
    )
    primitive_numbers = np.asarray([item[0] for item in primitive_rows], dtype=np.int64)
    primitive_positions = np.stack([item[1] for item in primitive_rows])

    if len(primitive_positions) * index != len(positions):
        raise InvariantViolation("primitive quotient atom count violates translation index")
    expected_counts = Counter(int(value) for value in numbers)
    if any(count % index for count in expected_counts.values()) or Counter(
        int(value) for value in primitive_numbers
    ) != Counter({species: count // index for species, count in expected_counts.items()}):
        raise InvariantViolation("primitive quotient violates species-resolved counts")
    if not np.isclose(
        np.linalg.det(primitive_lattice) * index,
        np.linalg.det(cell),
        atol=cartesian_tolerance,
        rtol=cartesian_tolerance,
    ):
        raise InvariantViolation("primitive quotient volume violates translation index")
    return PrimitivePhysicalCrystal(
        lattice=primitive_lattice,
        fractional_positions=primitive_positions,
        atomic_numbers=primitive_numbers,
        basis_change=certificate.basis_change,
        inverse_basis_change=certificate.inverse_basis_change,
        translation_subgroup=certificate.translation_subgroup,
        translation_action_max_cartesian_error=certificate.action_max_cartesian_error,
    )


def _block_diagonal(blocks: Sequence[np.ndarray]) -> FloatArray:
    size = sum(block.shape[0] for block in blocks)
    output = np.zeros((size, size), dtype=np.float64)
    start = 0
    for block in blocks:
        width = block.shape[0]
        output[start : start + width, start : start + width] = block
        start += width
    return output


INACTIVE_CHART_PARAMETER_DEFAULTS: Mapping[str, Mapping[str, object]] = MappingProxyType(
    {
        # Equal to the corresponding DomainConfig field defaults.
        "loewner-v2": MappingProxyType({"metric_bound": 2.0}),
        "ball-v1": MappingProxyType(
            {
                "loewner_upper_log_eigenvalue": 4.5,
                "loewner_control_log_volume_window": (-1.5, 2.0),
            }
        ),
    }
)


class IntrinsicCoverLayout:
    """Certified gauge reduction and decoder constants for one cover.

    For each continuous origin axis, the first orbit is used as an anchor and
    every other parameter is represented relative to that anchor.  The exact
    catalogue gate proves that every row has one and only one parameter with
    coordinate direction ``+/- e_j`` for each such axis, and that every stored
    intrinsic period basis is the identity.  Settings with no continuous
    origin freedom use the direct product of their Wyckoff tori.
    """

    def __init__(
        self,
        cover: ContinuousCover,
        *,
        catalogue: CertifiedCatalogue = MINIMAL_CATALOGUE,
        metric_bound: float = 2.0,
        reference_atomic_volume: float = 20.0,
        minimum_lattice_singular_value: float = 0.75,
        lattice_chart: str = "loewner-v2",
        loewner_upper_log_eigenvalue: float = 4.5,
        loewner_control_log_volume_window: tuple[float, float] = (-1.5, 2.0),
    ) -> None:
        if not np.isfinite(metric_bound) or metric_bound <= 0.0:
            raise DomainError("metric_bound must be finite and positive")
        if not np.isfinite(reference_atomic_volume) or reference_atomic_volume <= 0.0:
            raise DomainError("reference_atomic_volume must be finite and positive")
        if (
            not np.isfinite(minimum_lattice_singular_value)
            or minimum_lattice_singular_value <= 0.0
        ):
            raise DomainError(
                "minimum_lattice_singular_value must be finite and positive"
            )
        catalogue.validate()
        self.cover = cover
        self.catalogue = catalogue
        self.setting = catalogue.get(cover.space_group)
        self.positions = tuple(
            self.setting.wyckoff(orbit.wyckoff_label) for orbit in cover.orbits
        )
        self._validate_arithmetic()
        self.reference_atomic_volume = float(reference_atomic_volume)
        self.minimum_lattice_singular_value = float(
            minimum_lattice_singular_value
        )
        target_volume = self.reference_atomic_volume * self.expanded_atom_count
        reference_metric = scale_metric_to_volume(
            np.asarray(self.setting.reference_metric, dtype=np.float64),
            target_volume,
        )
        rotations = [
            np.asarray(rotation, dtype=np.float64)
            for rotation in self.setting.point_group_rotations
        ]
        # Volumetric window of the untrained uniform laws; None on the ball.
        self.control_log_volume_window: tuple[float, float] | None = None
        if lattice_chart == "ball-v1":
            self.chart = InvariantSPDChart(
                rotations,
                reference_metric=reference_metric,
                bound=float(metric_bound),
            )
        elif lattice_chart == "loewner-v2":
            self.control_log_volume_window = (
                float(loewner_control_log_volume_window[0]),
                float(loewner_control_log_volume_window[1]),
            )
            self.chart = LoewnerSPDChart(
                rotations,
                reference_metric=reference_metric,
                lower=loewner_lower_bound(
                    reference_metric, self.minimum_lattice_singular_value
                ),
                upper=float(np.exp(loewner_upper_log_eigenvalue)),
            )
        else:
            raise DomainError("lattice_chart must be ball-v1 or loewner-v2")
        # Parameters of the inactive chart have no effect; refuse non-defaults.
        supplied: dict[str, object] = {
            "metric_bound": float(metric_bound),
            "loewner_upper_log_eigenvalue": float(loewner_upper_log_eigenvalue),
            "loewner_control_log_volume_window": tuple(
                float(value) for value in loewner_control_log_volume_window
            ),
        }
        for name, default in INACTIVE_CHART_PARAMETER_DEFAULTS[lattice_chart].items():
            if supplied[name] != default:
                raise DomainError(
                    f"{name} has no effect under lattice_chart={lattice_chart!r}; "
                    "leave it at its default or select the chart it belongs to"
                )
        certified_floor = self.chart.minimum_lattice_singular_value_bound
        if certified_floor < self.minimum_lattice_singular_value:
            raise DomainError(
                "bounded lattice chart is too broad for the declared shortest-vector "
                f"support: certified sigma_min >= {certified_floor:.6g} Angstrom, "
                f"required >= {self.minimum_lattice_singular_value:.6g} Angstrom"
            )
        (
            self.parameter_maps,
            self.period_basis,
            self._parameter_encoder,
            self._raw_parameter_slices,
            self._free_axis_components,
        ) = self._build_gauge()

    @property
    def residual_orbit_groups(self) -> tuple[tuple[int, ...], ...]:
        """Identical-orbit index groups whose relabellings are physical gauges."""

        grouped: dict[tuple[str, int], list[int]] = {}
        for index, orbit in enumerate(self.cover.orbits):
            grouped.setdefault(
                (orbit.wyckoff_label, orbit.atomic_number), []
            ).append(index)
        return tuple(
            tuple(indices)
            for _, indices in sorted(grouped.items())
            if len(indices) > 1
        )

    def residual_action_matrix(
        self,
        permutation: Sequence[int],
        *,
        orbit_signs: Sequence[int] | None = None,
    ) -> FloatArray:
        """Return ``T`` for the gauge action ``eta' = eta @ T``.

        ``permutation[new_slot]`` names the old, physically identical orbit
        placed into that slot.  Only permutations within equal
        ``(Wyckoff-row, species)`` classes are legal.  For P1 this includes the
        nontrivial change of origin when the anchor orbit is exchanged.
        """

        orbit_count = len(self.cover.orbits)
        source = tuple(int(value) for value in permutation)
        if len(source) != orbit_count or set(source) != set(range(orbit_count)):
            raise DomainError("residual permutation must be a bijection of orbit slots")
        for new_index, old_index in enumerate(source):
            if self.cover.orbits[new_index] != self.cover.orbits[old_index]:
                raise DomainError(
                    "residual permutation may exchange only identical orbit types"
                )
        signs = (
            (1,) * orbit_count
            if orbit_signs is None
            else tuple(int(value) for value in orbit_signs)
        )
        if len(signs) != orbit_count or any(value not in (-1, 1) for value in signs):
            raise DomainError("residual orbit signs must contain one +/-1 per orbit")
        dimension = self.intrinsic_dimension
        action = np.zeros((dimension, dimension), dtype=np.float64)
        if dimension == 0:
            return action
        if self.setting.key != "P-1" and any(value != 1 for value in signs):
            raise DomainError(
                "per-orbit sign gauges are certified only for P-1"
            )
        raw_dimension = sum(row.dimension for row in self.positions)
        reconstruction = np.zeros((dimension, raw_dimension), dtype=np.float64)
        for parameter_slice, mapping in zip(
            self._raw_parameter_slices, self.parameter_maps, strict=True
        ):
            reconstruction[:, parameter_slice] = mapping
        raw_action = np.zeros((raw_dimension, raw_dimension), dtype=np.float64)
        for new_index, old_index in enumerate(source):
            output = self._raw_parameter_slices[new_index]
            old = self._raw_parameter_slices[old_index]
            width = output.stop - output.start
            if width:
                raw_action[old, output] = signs[new_index] * np.eye(width)
        action = reconstruction @ raw_action @ self._parameter_encoder
        determinant = float(np.linalg.det(action))
        if not np.isclose(abs(determinant), 1.0, atol=1.0e-12, rtol=1.0e-12):
            raise InvariantViolation("residual orbit action is not torus-unimodular")
        return action

    def transform_residual_eta(
        self,
        eta: Sequence[float],
        permutation: Sequence[int],
        *,
        orbit_signs: Sequence[int] | None = None,
    ) -> FloatArray:
        """Apply and canonically wrap one certified identical-orbit gauge action."""

        value = np.asarray(eta, dtype=np.float64)
        if value.shape != (self.intrinsic_dimension,):
            raise DomainError("eta has the wrong intrinsic-cover dimension")
        return self.canonicalize_eta(
            value
            @ self.residual_action_matrix(
                permutation, orbit_signs=orbit_signs
            )
        )

    def _validate_arithmetic(self) -> None:
        actual: Counter[int] = Counter()
        fixed_rows: set[str] = set()
        for orbit, row in zip(self.cover.orbits, self.positions, strict=True):
            actual[orbit.atomic_number] += row.multiplicity
            if row.dimension == 0:
                if row.label in fixed_rows:
                    raise InvariantViolation(
                        f"fixed Wyckoff row {row.label} cannot be instantiated twice"
                    )
                fixed_rows.add(row.label)
        expected = Counter(
            {z: self.cover.nu * count for z, count in self.cover.composition.counts}
        )
        if actual != expected:
            raise InvariantViolation(
                f"cover violates exact formula arithmetic: got {dict(actual)}, "
                f"expected {dict(expected)}"
            )

    def _build_gauge(
        self,
    ) -> tuple[
        tuple[FloatArray, ...],
        FloatArray,
        FloatArray,
        tuple[slice, ...],
        tuple[tuple[int, tuple[tuple[int, int], ...]], ...],
    ]:
        """Construct an exact axiswise origin section and its linear encoder."""

        raw_slices: list[slice] = []
        raw_start = 0
        for row in self.positions:
            raw_slices.append(slice(raw_start, raw_start + row.dimension))
            raw_start += row.dimension
            expected_period = np.eye(row.dimension, dtype=np.float64)
            stored_period = np.asarray(row.period_basis, dtype=np.float64).reshape(
                row.dimension, row.dimension
            )
            if not np.array_equal(stored_period, expected_period):
                raise CatalogueError(
                    f"Wyckoff row {self.setting.key}:{row.label} needs a "
                    "nonidentity quotient-period implementation"
                )

        free_axes = tuple(
            axis
            for axis in range(3)
            if all(
                rotation[axis][column] == int(axis == column)
                for rotation in self.setting.point_group_rotations
                for column in range(3)
            )
        )
        axis_components: list[tuple[int, tuple[tuple[int, int], ...]]] = []
        dropped: dict[int, tuple[int, int]] = {}
        for axis in free_axes:
            positive = tuple(int(column == axis) for column in range(3))
            negative = tuple(-value for value in positive)
            components: list[tuple[int, int]] = []
            for orbit_index, (row, raw_slice) in enumerate(
                zip(self.positions, raw_slices, strict=True)
            ):
                matches = [
                    (raw_slice.start + parameter_index, 1 if vector == positive else -1)
                    for parameter_index, vector in enumerate(row.coordinate_matrix)
                    if vector == positive or vector == negative
                ]
                if len(matches) != 1:
                    raise CatalogueError(
                        f"{self.setting.key}:{row.label} does not have exactly one "
                        f"+/-e_{axis} continuous-origin parameter"
                    )
                components.append(matches[0])
                if orbit_index == 0:
                    dropped[axis] = matches[0]
            axis_components.append((axis, tuple(components)))

        dropped_raw = {value[0] for value in dropped.values()}
        retained_raw = tuple(
            index for index in range(raw_start) if index not in dropped_raw
        )
        quotient_dimension = len(retained_raw)
        eta_index = {raw: index for index, raw in enumerate(retained_raw)}
        maps: list[FloatArray] = []
        for row, raw_slice in zip(self.positions, raw_slices, strict=True):
            mapping = np.zeros((quotient_dimension, row.dimension), dtype=np.float64)
            for local in range(row.dimension):
                raw = raw_slice.start + local
                if raw in eta_index:
                    mapping[eta_index[raw], local] = 1.0
            maps.append(mapping)

        encoder = np.zeros((raw_start, quotient_dimension), dtype=np.float64)
        component_lookup = {
            raw: (axis, sign)
            for axis, components in axis_components
            for raw, sign in components
        }
        for raw, output in eta_index.items():
            encoder[raw, output] = 1.0
            if raw in component_lookup:
                axis, sign = component_lookup[raw]
                anchor_raw, anchor_sign = dropped[axis]
                encoder[anchor_raw, output] -= sign * anchor_sign

        period = np.eye(quotient_dimension, dtype=np.float64)
        return (
            tuple(maps),
            period,
            encoder,
            tuple(raw_slices),
            tuple(axis_components),
        )

    @property
    def lattice_dimension(self) -> int:
        return self.chart.dimension

    @property
    def intrinsic_dimension(self) -> int:
        return self.period_basis.shape[0]

    @property
    def expanded_atom_count(self) -> int:
        return sum(row.multiplicity for row in self.positions)

    def periodic_edge_upper_bound(self, cutoff: float) -> int:
        """Conservative packing bound for all directed periodic pair images.

        Distinct translated lattice points are separated by at least the
        certified lower singular-value bound ``s``.  Radius-``s/2`` balls are
        disjoint, so a cutoff ball contains at most ``(1+2 r/s)^3`` points.
        The result includes central self images and is therefore an upper
        bound for the Transformer's edge list.
        """

        if not np.isfinite(cutoff) or cutoff <= 0.0:
            raise DomainError("cutoff must be finite and positive")
        sigma = self.chart.minimum_lattice_singular_value_bound
        per_pair = int(np.ceil((1.0 + 2.0 * float(cutoff) / sigma) ** 3))
        return self.expanded_atom_count**2 * per_pair

    def periodic_candidate_box_upper_bound(self, cutoff: float) -> int:
        """Upper bound on the integer box materialized for any atom pair."""

        if not np.isfinite(cutoff) or cutoff <= 0.0:
            raise DomainError("cutoff must be finite and positive")
        sigma = self.chart.minimum_lattice_singular_value_bound
        side = 2 * int(np.ceil(float(cutoff) / sigma)) + 3
        return side**3

    def canonicalize_eta(self, eta: Sequence[float]) -> FloatArray:
        value = np.asarray(eta, dtype=np.float64)
        if value.shape != (self.intrinsic_dimension,) or not np.all(np.isfinite(value)):
            raise DomainError(
                f"eta must contain {self.intrinsic_dimension} finite coordinates"
            )
        if self.intrinsic_dimension == 0:
            return value.copy()
        coefficients = np.linalg.solve(self.period_basis.T, value)
        coefficients -= np.floor(coefficients)
        return coefficients @ self.period_basis

    def reconstruct_parameters(self, eta: Sequence[float]) -> tuple[FloatArray, ...]:
        canonical = self.canonicalize_eta(eta)
        return tuple(canonical @ mapping for mapping in self.parameter_maps)

    def encode_exact_parameters(self, scaffold: CrystalScaffold) -> FloatArray:
        """Encode an exact target scaffold into this gauge for dequantization."""

        if ContinuousCover.from_exact_scaffold(scaffold) != self.cover:
            raise DomainError("exact scaffold does not represent this continuous cover")
        parameters = [
            np.asarray([float(value) for value in orbit.parameter], dtype=np.float64)
            for orbit in scaffold.orbits
        ]
        raw = (
            np.concatenate([value for value in parameters if value.size])
            if any(value.size for value in parameters)
            else np.empty(0)
        )
        eta = raw @ self._parameter_encoder
        return self.canonicalize_eta(eta)

    def decode(
        self,
        geometry: ContinuousGeometry,
        *,
        collision_tolerance: float = 1.0e-8,
        symmetry_tolerance: float = 1.0e-8,
    ) -> ContinuousDecodedCrystal:
        """Decode one real state and run formula/collision/containment audits."""

        if geometry.lattice_latent.shape != (self.lattice_dimension,):
            raise DomainError("lattice latent has the wrong cover dimension")
        if geometry.intrinsic_coordinates.shape != (self.intrinsic_dimension,):
            raise DomainError("intrinsic coordinate vector has the wrong cover dimension")
        if collision_tolerance <= 0.0 or symmetry_tolerance <= 0.0:
            raise DomainError("decoder tolerances must be positive")
        metric = self.chart.decode_latent(geometry.lattice_latent)
        lattice = np.linalg.cholesky(metric)
        actual_sigma = float(np.linalg.svd(lattice, compute_uv=False)[-1])
        if actual_sigma < self.minimum_lattice_singular_value:
            raise InvariantViolation(
                "decoded lattice left the certified shortest-vector support"
            )
        parameters = self.reconstruct_parameters(geometry.intrinsic_coordinates)
        positions: list[np.ndarray] = []
        atomic_numbers: list[int] = []
        orbit_slices: list[slice] = []
        for orbit, row, parameter in zip(
            self.cover.orbits, self.positions, parameters, strict=True
        ):
            start = len(positions)
            try:
                expanded = row.expand_float(
                    parameter,
                    require_declared_multiplicity=True,
                )
            except InvariantViolation as exc:
                # Wyckoff specialization is a genuine endpoint defect, not a
                # catalogue/program invariant failure.  Preserve it as one
                # failed model attempt so evaluators keep it in the denominator.
                raise GeneratedSampleError(str(exc)) from exc
            positions.extend(expanded)
            atomic_numbers.extend([orbit.atomic_number] * row.multiplicity)
            orbit_slices.append(slice(start, len(positions)))
        fractional = np.asarray(positions, dtype=np.float64)
        numbers = np.asarray(atomic_numbers, dtype=np.int64)
        minimum_distance = shortest_nonzero_lattice_image(
            lattice
        ).squared_distance**0.5
        if len(fractional) > 1:
            minimum_distance = min(
                minimum_distance,
                _minimum_pair_distance(fractional, lattice, upper_bound=minimum_distance),
            )
        no_collisions = minimum_distance > collision_tolerance
        formula_matches = Counter(int(z) for z in numbers) == Counter(
            {z: self.cover.nu * count for z, count in self.cover.composition.counts}
        )
        metric_is_invariant = _metric_is_invariant(
            self.setting, metric, symmetry_tolerance
        )
        group_contained = metric_is_invariant and _numerical_group_containment(
            self.setting,
            fractional,
            numbers,
            lattice,
            symmetry_tolerance,
        )
        translation_index = _numerical_translation_index(
            fractional,
            numbers,
            lattice,
            symmetry_tolerance,
        )
        audit = ContinuousDecoderAudit(
            formula_matches=formula_matches,
            no_collisions=no_collisions,
            metric_is_invariant=metric_is_invariant,
            selected_group_is_contained=group_contained,
            minimum_distance=minimum_distance,
            translation_index=translation_index,
        )
        if not audit.no_collisions:
            raise GeneratedSampleError(
                "continuous decode produced a crystallographic collision: "
                f"minimum_distance={audit.minimum_distance:.17g}"
            )
        if not (
            audit.formula_matches
            and audit.metric_is_invariant
            and audit.selected_group_is_contained
        ):
            # These properties are structural consequences of a certified
            # cover decoder.  Their failure is evidence of corrupted catalogue
            # data or an implementation bug, not an inconvenient model sample.
            raise InvariantViolation(f"continuous decode failed invariant audit: {audit}")
        cartesian_jacobian = stacked_cartesian_jacobian(
            self.positions, self.parameter_maps, lattice
        ).reshape(self.expanded_atom_count, 3, self.intrinsic_dimension)
        return ContinuousDecodedCrystal(
            cover=self.cover,
            setting=self.setting,
            geometry=ContinuousGeometry(
                geometry.lattice_latent,
                self.canonicalize_eta(geometry.intrinsic_coordinates),
            ),
            metric=metric,
            lattice=lattice,
            fractional_positions=fractional,
            atomic_numbers=numbers,
            orbit_slices=tuple(orbit_slices),
            jacobian_cartesian=cartesian_jacobian,
            audit=audit,
        )


def _metric_is_invariant(
    setting: SpaceGroupSetting, metric: np.ndarray, tolerance: float
) -> bool:
    return all(
        np.allclose(
            np.asarray(operation.rotation, dtype=np.float64)
            @ metric
            @ np.asarray(operation.rotation, dtype=np.float64).T,
            metric,
            atol=tolerance,
            rtol=tolerance,
        )
        for operation in setting.operations
    )


def _has_perfect_matching(adjacency: np.ndarray) -> bool:
    """Small deterministic augmenting-path bipartite matcher."""

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


def _numerical_group_containment(
    setting: SpaceGroupSetting,
    fractional: np.ndarray,
    atomic_numbers: np.ndarray,
    lattice: np.ndarray,
    tolerance: float,
) -> bool:
    for operation in setting.operations:
        transformed = np.stack(
            [operation.apply_float(point) for point in fractional], axis=0
        )
        for z in sorted(set(int(value) for value in atomic_numbers)):
            source = transformed[atomic_numbers == z]
            target = fractional[atomic_numbers == z]
            if source.shape != target.shape:
                return False
            adjacency = _pairs_within_tolerance(source, target, lattice, tolerance)
            if not _has_perfect_matching(adjacency):
                return False
    return True


def _pairs_within_tolerance(
    source: np.ndarray, target: np.ndarray, lattice: np.ndarray, tolerance: float
) -> npt.NDArray[np.bool_]:
    """Exact ``[closest periodic image of s - t] <= tolerance`` for all pairs."""

    return periodic_pairs_within(source, target, lattice, tolerance)[0]


def _numerical_translation_index(
    fractional: np.ndarray,
    atomic_numbers: np.ndarray,
    lattice: np.ndarray,
    tolerance: float,
) -> int:
    """Count every motif-preserving torus translation at a frozen tolerance."""

    return len(
        _numerical_translation_subgroup(
            fractional,
            atomic_numbers,
            lattice,
            tolerance,
        )
    )


def _numerical_translation_subgroup(
    fractional: np.ndarray,
    atomic_numbers: np.ndarray,
    lattice: np.ndarray,
    tolerance: float,
) -> FloatArray:
    """Enumerate every reported species-preserving torus translation."""

    counts = Counter(int(value) for value in atomic_numbers)
    anchor_species = min(counts, key=lambda value: (counts[value], value))
    anchor_index = int(np.flatnonzero(atomic_numbers == anchor_species)[0])
    anchor = fractional[anchor_index]
    raw_candidates = [
        np.mod(point - anchor, 1.0)
        for point, species in zip(fractional, atomic_numbers, strict=True)
        if int(species) == anchor_species
    ]
    candidates: list[np.ndarray] = []
    for candidate in raw_candidates:
        if candidates and bool(
            np.any(
                _pairs_within_tolerance(
                    candidate[None, :], np.asarray(candidates), lattice, tolerance
                )
            )
        ):
            continue
        candidates.append(candidate)
    accepted: list[np.ndarray] = []
    for translation in candidates:
        transformed = np.mod(fractional + translation, 1.0)
        preserves = True
        for species in sorted(counts):
            source = transformed[atomic_numbers == species]
            target = fractional[atomic_numbers == species]
            adjacency = _pairs_within_tolerance(source, target, lattice, tolerance)
            if not _has_perfect_matching(adjacency):
                preserves = False
                break
        if preserves:
            accepted.append(translation)
    if not accepted:
        raise InvariantViolation("floating translation audit lost the identity")
    result = np.asarray(accepted, dtype=np.float64)
    result.setflags(write=False)
    return result


__all__ = [
    "ContinuousCover",
    "ContinuousDecodedCrystal",
    "ContinuousDecoderAudit",
    "ContinuousGeometry",
    "IntrinsicCoverLayout",
    "OrbitType",
    "PrimitivePhysicalCrystal",
    "full_primitive_reduction",
]
