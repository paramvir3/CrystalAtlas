"""Primitive reduction and pinned-setting standardization.

Two deliberately separate paths live here.

``exact_full_primitive_reduction`` is a finite oracle for rational motifs.  It
enumerates the complete species-preserving translation group and constructs a
primitive basis from the corresponding rational superlattice.  No tolerance
or float-to-rational conversion is used.

``SpglibStandardizer`` is the production boundary for finite-precision input.
It pins the backend version, the Hall setting for every encountered space-group
type, and all numerical policy.  The result is accepted only after a primitive
round trip reproduces the same pinned conventional representative.  This
module does not silently guess a Hall setting or repair a left-handed cell.

Lattices use row vectors throughout: ``x_cart = f @ lattice``.  If ``C`` is
the conventional-to-primitive basis map, then

``lattice_primitive = C @ lattice_input`` and ``f_primitive = f_input @ C^-1``.

Consequently ``det(C) = 1 / |T|``, where ``T`` is the complete finite
translation subgroup of the coloured motif.
"""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass, replace
from fractions import Fraction
from functools import lru_cache
from hashlib import sha256
from itertools import combinations, product
from math import lcm
from numbers import Integral
from types import MappingProxyType, ModuleType
from typing import Any, TypeAlias, cast

import numpy as np
import numpy.typing as npt

from ._optional import require_module
from .algebra import (
    FractionMatrix,
    FractionVector,
    IntegerMatrix,
    as_fraction_matrix,
    as_fraction_vector,
    as_integer_matrix,
    exact_determinant,
    exact_inverse,
    exact_matmul,
    exact_rank,
    exact_row_matmul,
    fraction_mod_one,
    unimodular_inverse,
)
from .catalogue import CertifiedCatalogue, SpaceGroupSetting
from .decoder import Species, exact_translation_subgroup
from .errors import (
    AmbiguousPreprocessingError,
    CatalogueError,
    DomainError,
    InvariantViolation,
    NumericalError,
)
from .periodic import closest_periodic_image, periodic_pairs_within

FloatArray: TypeAlias = npt.NDArray[np.float64]
IntArray: TypeAlias = npt.NDArray[np.int64]

_IDENTITY_3 = ((1, 0, 0), (0, 1, 0), (0, 0, 1))
STANDARDIZATION_ALGORITHM_ID = "fullprim-minimax-proper-primitive-pinned-hall-v6"


def _integer_basis_witness(
    value: FloatArray, *, determinant: int, tolerance: float, name: str
) -> IntegerMatrix:
    """Recover a finite integer basis and check its determinant exactly."""

    if value.shape != (3, 3) or not np.all(np.isfinite(value)):
        raise InvariantViolation(f"{name} is not a finite 3x3 basis")
    nearest = np.rint(value)
    if not np.allclose(value, nearest, atol=tolerance, rtol=0.0):
        raise InvariantViolation(f"{name} is not integral")
    # Python integers avoid a silent int64 overflow during certification.
    result = as_integer_matrix(
        tuple(tuple(int(item) for item in row) for row in nearest), shape=(3, 3)
    )
    if exact_determinant(result) != determinant:
        raise InvariantViolation(f"{name} has an incorrect exact determinant")
    return result


def _primitive_rebasis_before_idealization(
    transformation: FloatArray,
    inverse_witness_basis: IntegerMatrix,
    conventional_lattice: FloatArray,
    primitive_lattice: FloatArray,
    *,
    centering_index: int,
    tolerance: float,
    typed_idealization_rejection: bool = False,
) -> IntegerMatrix:
    """Recover topology without interpreting idealization strain as a basis.

    In row convention, L_pre = P^-T L_raw. If L_w = C_w L_raw,
    K = P^-T C_w^-1 maps the exact primitive witness to the conventional
    basis. Independently, L_conv = H L_prim after idealization. Both K and H
    are integer index-J embeddings. Thus U = H^-1 K must be an exact proper
    unimodular matrix. The metric distortion is audited separately; spglib's
    orthogonal rotation does not undo its lattice idealization.

    ``find_primitive`` searches symmetry again on the idealized conventional
    cell.  At a large dimensioned tolerance that search can choose another
    basis and orientation, so ``H`` is not integral.  In all five such S2 cases
    (tau = 1e-2 and 1e-1 A) the returned cell was isometric to ``std_lattice``
    (it equals ``M L_conv R`` with ``M`` unimodular and ``R`` a proper
    rotation), and the lattice alone had a higher holohedry within tau than
    the crystal (mmm or 4/mmm against 2/m).  For example, mp-1305082 at
    tau = 1e-2 A gives ``H = I - 4.08e-4 e_31``.  The converse does not hold:
    accepted records can also have pseudo-symmetric lattices.  Such a pair
    cannot be related exactly, and later site-mapping invariants assume it can.
    It means the conventional basis depends on how the crystal is presented at
    this tolerance.  With ``typed_idealization_rejection`` (the dimensioned
    policy) the pair is a recorded preprocessing rejection, not a fatal
    invariant.  The legacy policy keeps the fatal check.  No accepted output
    changes under either policy.
    """

    if transformation.shape != (3, 3) or not np.all(np.isfinite(transformation)):
        raise InvariantViolation("spglib transformation is not a finite 3x3 matrix")
    try:
        conventional_from_witness = np.linalg.solve(
            transformation.T, np.asarray(inverse_witness_basis, dtype=np.float64)
        )
        conventional_from_primitive = np.linalg.solve(
            primitive_lattice.T, conventional_lattice.T
        ).T
    except np.linalg.LinAlgError as exc:
        raise InvariantViolation("spglib basis transformation is singular") from exc
    k = _integer_basis_witness(
        conventional_from_witness,
        determinant=centering_index,
        tolerance=tolerance,
        name="pre-idealization conventional embedding",
    )
    if (
        typed_idealization_rejection
        and np.all(np.isfinite(conventional_from_primitive))
        and not np.allclose(
            conventional_from_primitive,
            np.rint(conventional_from_primitive),
            atol=tolerance,
            rtol=0.0,
        )
    ):
        deviation = float(
            np.max(np.abs(conventional_from_primitive - np.rint(conventional_from_primitive)))
        )
        raise AmbiguousPreprocessingError(
            "spglib's primitive search on the idealized conventional cell returns a "
            "basis that is not integer-related to it at this tolerance "
            f"(max |H - round(H)| = {deviation:.3g})"
        )
    h = _integer_basis_witness(
        conventional_from_primitive,
        determinant=centering_index,
        tolerance=tolerance,
        name="idealized conventional-to-primitive embedding",
    )
    rebasis = exact_matmul(exact_inverse(h), k)
    if any(value.denominator != 1 for row in rebasis for value in row):
        raise InvariantViolation("spglib primitive rebasing is not integral")
    result = as_integer_matrix(
        tuple(tuple(int(value) for value in row) for row in rebasis), shape=(3, 3)
    )
    if exact_determinant(result) != 1:
        raise InvariantViolation("spglib primitive rebasing is not proper unimodular")
    return result


def _species_key(species: Species) -> tuple[str, str]:
    return type(species).__name__, repr(species)


def _validate_species(species: object) -> Species:
    if isinstance(species, bool):
        raise DomainError("Boolean values are not chemical species")
    if isinstance(species, Integral) and int(species) > 0:
        return int(species)
    if isinstance(species, str) and species.strip():
        return species.strip()
    raise DomainError("species must be a positive atomic number or nonempty symbol")


def _extended_gcd(left: int, right: int) -> tuple[int, int, int]:
    """Return nonnegative ``g`` and Bezout coefficients ``x, y``."""

    old_r, remainder = left, right
    old_x, x = 1, 0
    old_y, y = 0, 1
    while remainder:
        quotient = old_r // remainder
        old_r, remainder = remainder, old_r - quotient * remainder
        old_x, x = x, old_x - quotient * x
        old_y, y = y, old_y - quotient * y
    if old_r < 0:
        return -old_r, -old_x, -old_y
    return old_r, old_x, old_y


def _integer_row_lattice_basis(generators: Sequence[Sequence[int]]) -> IntegerMatrix:
    """Return a triangular basis for a full-rank row lattice in ``Z^3``.

    Only unimodular row operations are used.  The three ``D e_i`` generators
    supplied by the caller make full rank an invariant, so three nonzero rows
    must remain.  This is the row analogue of Hermite reduction; canonical HNF
    is unnecessary because the input generator order is itself canonical.
    """

    work = [list(map(int, row)) for row in generators]
    if not work or any(len(row) != 3 for row in work):
        raise DomainError("integer lattice generators must be nonempty length-three rows")
    pivot_row = 0
    for column in range(3):
        candidate = next(
            (row for row in range(pivot_row, len(work)) if work[row][column] != 0),
            None,
        )
        if candidate is None:
            continue
        work[pivot_row], work[candidate] = work[candidate], work[pivot_row]
        for row in range(pivot_row + 1, len(work)):
            if work[row][column] == 0:
                continue
            a = work[pivot_row][column]
            b = work[row][column]
            common, x, y = _extended_gcd(a, b)
            old_pivot = work[pivot_row][:]
            old_row = work[row][:]
            work[pivot_row] = [
                x * first + y * second for first, second in zip(old_pivot, old_row, strict=True)
            ]
            work[row] = [
                -(b // common) * first + (a // common) * second
                for first, second in zip(old_pivot, old_row, strict=True)
            ]
        if work[pivot_row][column] < 0:
            work[pivot_row] = [-value for value in work[pivot_row]]
        pivot = work[pivot_row][column]
        for row in range(pivot_row):
            quotient = work[row][column] // pivot
            work[row] = [
                first - quotient * second
                for first, second in zip(work[row], work[pivot_row], strict=True)
            ]
        pivot_row += 1
    nonzero = tuple(tuple(row) for row in work if any(row))
    if len(nonzero) != 3 or exact_determinant(nonzero) <= 0:
        raise InvariantViolation("integer translation lattice did not yield a positive basis")
    return nonzero


@dataclass(frozen=True, slots=True)
class ExactOriginCanonicalization:
    """Canonical representative modulo translations along declared free axes."""

    fractional_positions: tuple[FractionVector, ...]
    species: tuple[Species, ...]
    subtracted_origin: FractionVector
    free_axes: tuple[int, ...]


def canonicalize_exact_origin(
    fractional_positions: Sequence[Sequence[Fraction | int]],
    species: Sequence[Species],
    *,
    free_axes: Sequence[int],
) -> ExactOriginCanonicalization:
    """Quotient an axis-aligned continuous origin subtorus exactly.

    Every least-species atom is considered as a possible anchor.  Sorting the
    complete shifted coloured motif and taking the lexicographic minimum makes
    atom permutation and anchor ties harmless.  The procedure is a finite
    canonical section, not a data-dependent greedy choice.
    """

    positions = tuple(as_fraction_vector(point, length=3) for point in fractional_positions)
    normalized_species = tuple(_validate_species(value) for value in species)
    axes = tuple(sorted(set(free_axes)))
    if not positions or len(positions) != len(normalized_species):
        raise DomainError("origin canonicalization needs a nonempty aligned coloured motif")
    if len(axes) != len(tuple(free_axes)) or any(axis not in range(3) for axis in axes):
        raise DomainError("free origin axes must be distinct members of {0,1,2}")
    if not axes:
        order = tuple(
            sorted(
                range(len(positions)),
                key=lambda index: (_species_key(normalized_species[index]), positions[index]),
            )
        )
        return ExactOriginCanonicalization(
            tuple(positions[index] for index in order),
            tuple(normalized_species[index] for index in order),
            (Fraction(0),) * 3,
            axes,
        )

    least_species = min(normalized_species, key=_species_key)
    candidates: list[
        tuple[
            tuple[tuple[tuple[str, str], FractionVector], ...],
            tuple[FractionVector, ...],
            tuple[Species, ...],
            FractionVector,
        ]
    ] = []
    for anchor, atom_species in zip(positions, normalized_species, strict=True):
        if atom_species != least_species:
            continue
        shift = tuple(anchor[axis] if axis in axes else Fraction(0) for axis in range(3))
        shifted = tuple(
            tuple(fraction_mod_one(point[axis] - shift[axis]) for axis in range(3))
            for point in positions
        )
        order = tuple(
            sorted(
                range(len(shifted)),
                key=lambda index: (_species_key(normalized_species[index]), shifted[index]),
            )
        )
        ordered_positions = tuple(shifted[index] for index in order)
        ordered_species = tuple(normalized_species[index] for index in order)
        key = tuple(
            (_species_key(value), point)
            for point, value in zip(ordered_positions, ordered_species, strict=True)
        )
        candidates.append((key, ordered_positions, ordered_species, shift))
    _, result_positions, result_species, result_shift = min(
        candidates, key=lambda item: item[0]
    )
    return ExactOriginCanonicalization(
        result_positions,
        result_species,
        result_shift,
        axes,
    )


@dataclass(frozen=True, slots=True)
class ExactTranslationLattice:
    """Exact finite translation group and its primitive-lattice basis.

    ``basis_change`` is a row basis of ``Z^3 + T`` in input fractional
    coordinates.  Its inverse is integral, its positive determinant is
    ``1 / |T|``, and every member of ``T`` maps to an integral row under that
    inverse.  These identities are checked by the constructor function below,
    rather than trusted from a floating crystallography backend.
    """

    basis_change: FractionMatrix
    inverse_basis_change: IntegerMatrix
    translation_subgroup: tuple[FractionVector, ...]

    @property
    def translation_index(self) -> int:
        return len(self.translation_subgroup)


def exact_primitive_basis_from_translations(
    translation_subgroup: Sequence[Sequence[Fraction | int]],
) -> ExactTranslationLattice:
    """Construct and certify a primitive basis from an exact finite subgroup.

    Let ``D`` clear all denominators in ``T``.  The lattice generated in
    ``Z^3`` by ``D I`` and ``D t`` (``t in T``) has a row basis ``H``.  Then
    ``C = H / D`` is a basis of ``Z^3 + T``.  Exact closure and determinant
    checks prove both that the input is a group and that the quotient has the
    claimed complete index.
    """

    translations = tuple(
        sorted(
            {
                tuple(fraction_mod_one(value) for value in as_fraction_vector(row, length=3))
                for row in translation_subgroup
            }
        )
    )
    if not translations:
        raise DomainError("translation subgroup cannot be empty")
    zero = (Fraction(0),) * 3
    translation_set = set(translations)
    if zero not in translation_set:
        raise InvariantViolation("translation subgroup does not contain the identity")
    for left in translations:
        inverse = tuple(fraction_mod_one(-value) for value in left)
        if inverse not in translation_set:
            raise InvariantViolation("translation subgroup is not closed under inverses")
        for right in translations:
            product_translation = tuple(
                fraction_mod_one(first + second)
                for first, second in zip(left, right, strict=True)
            )
            if product_translation not in translation_set:
                raise InvariantViolation("translation subgroup is not closed under addition")

    denominator = 1
    for translation in translations:
        for value in translation:
            denominator = lcm(denominator, value.denominator)
    generators: list[tuple[int, int, int]] = [
        (denominator, 0, 0),
        (0, denominator, 0),
        (0, 0, denominator),
    ]
    generators.extend(
        (
            int(translation[0] * denominator),
            int(translation[1] * denominator),
            int(translation[2] * denominator),
        )
        for translation in translations
    )
    integer_basis = _integer_row_lattice_basis(generators)
    basis = as_fraction_matrix(
        tuple(tuple(Fraction(value, denominator) for value in row) for row in integer_basis),
        ncols=3,
    )
    index = len(translations)
    determinant = exact_determinant(basis)
    if determinant != Fraction(1, index) or determinant <= 0:
        raise InvariantViolation(
            "translation superlattice has the wrong exact determinant/index"
        )
    inverse_fraction = exact_inverse(basis)
    if any(value.denominator != 1 for row in inverse_fraction for value in row):
        raise InvariantViolation("primitive basis inverse is unexpectedly nonintegral")
    inverse: IntegerMatrix = tuple(
        tuple(int(value) for value in row) for row in inverse_fraction
    )
    for translation in translations:
        image = exact_row_matmul(translation, inverse)
        if any(value.denominator != 1 for value in image):
            raise InvariantViolation("enumerated translation is outside the quotient kernel")
    return ExactTranslationLattice(basis, inverse, translations)


@dataclass(frozen=True, slots=True)
class ExactPrimitiveCell:
    """Certified quotient of an exact coloured motif by all translations."""

    basis_change: FractionMatrix
    inverse_basis_change: IntegerMatrix
    translation_subgroup: tuple[FractionVector, ...]
    fractional_positions: tuple[FractionVector, ...]
    species: tuple[Species, ...]
    input_fractional_positions: tuple[FractionVector, ...]
    input_species: tuple[Species, ...]

    @property
    def translation_index(self) -> int:
        return len(self.translation_subgroup)

    def lattice_from_input(self, input_lattice: npt.ArrayLike) -> FloatArray:
        """Apply the exact positive-determinant primitive basis map."""

        lattice = np.asarray(input_lattice, dtype=np.float64)
        if lattice.shape != (3, 3) or not np.all(np.isfinite(lattice)):
            raise DomainError("input lattice must be a finite 3x3 row-basis matrix")
        if float(np.linalg.det(lattice)) <= 0.0:
            raise DomainError("input lattice must be right handed")
        basis = np.asarray(
            [[float(value) for value in row] for row in self.basis_change],
            dtype=np.float64,
        )
        result = basis @ lattice
        if float(np.linalg.det(result)) <= 0.0:
            raise InvariantViolation("exact primitive map failed to preserve handedness")
        result.setflags(write=False)
        return result

    def expand_to_input_cell(self) -> tuple[tuple[FractionVector, Species], ...]:
        """Expand the primitive motif through the exact quotient kernel."""

        expanded: list[tuple[FractionVector, Species]] = []
        for point, atom_species in zip(self.fractional_positions, self.species, strict=True):
            base = exact_row_matmul(point, self.basis_change)
            for translation in self.translation_subgroup:
                expanded.append(
                    (
                        tuple(
                            fraction_mod_one(base[axis] + translation[axis])
                            for axis in range(3)
                        ),
                        atom_species,
                    )
                )
        expanded.sort(key=lambda item: (_species_key(item[1]), item[0]))
        return tuple(expanded)


def exact_full_primitive_reduction(
    fractional_positions: Sequence[Sequence[Fraction | int]],
    species: Sequence[Species],
) -> ExactPrimitiveCell:
    """Remove every exact species-preserving hidden translation.

    The finite coloured-motif oracle first enumerates ``T`` without tolerance.
    :func:`exact_primitive_basis_from_translations` then constructs the exact
    quotient lattice.  The fibre and expansion checks below additionally prove
    that the quotient acts on this particular coloured motif as claimed.
    """

    positions = tuple(as_fraction_vector(point, length=3) for point in fractional_positions)
    normalized_species = tuple(_validate_species(value) for value in species)
    if not positions or len(positions) != len(normalized_species):
        raise DomainError("primitive reduction needs a nonempty aligned coloured motif")
    if len(set(positions)) != len(positions):
        raise DomainError("input motif contains coincident sites")
    translations = exact_translation_subgroup(positions, normalized_species)
    translation_lattice = exact_primitive_basis_from_translations(translations)
    basis = translation_lattice.basis_change
    inverse = translation_lattice.inverse_basis_change
    index = translation_lattice.translation_index

    transformed = tuple(
        tuple(fraction_mod_one(value) for value in exact_row_matmul(point, inverse))
        for point in positions
    )
    coloured_counts = Counter(zip(transformed, normalized_species, strict=True))
    if any(multiplicity != index for multiplicity in coloured_counts.values()):
        raise InvariantViolation(
            "translation quotient fibres do not all have the complete group index"
        )
    if len({point for point, _ in coloured_counts}) != len(coloured_counts):
        raise DomainError("different species collide after primitive reduction")
    ordered = sorted(coloured_counts, key=lambda item: (_species_key(item[1]), item[0]))
    primitive_positions = tuple(point for point, _ in ordered)
    primitive_species = tuple(value for _, value in ordered)
    result = ExactPrimitiveCell(
        basis_change=basis,
        inverse_basis_change=inverse,
        translation_subgroup=translation_lattice.translation_subgroup,
        fractional_positions=primitive_positions,
        species=primitive_species,
        input_fractional_positions=positions,
        input_species=normalized_species,
    )
    expected = Counter(zip(positions, normalized_species, strict=True))
    if Counter(result.expand_to_input_cell()) != expected:
        raise InvariantViolation("exact primitive expansion failed its round trip")
    return result


def _readonly_float_array(
    value: npt.ArrayLike, *, shape: tuple[int, ...], name: str
) -> FloatArray:
    array = np.asarray(value, dtype=np.float64).copy()
    if array.shape != shape or not np.all(np.isfinite(array)):
        raise DomainError(f"{name} must be finite with shape {shape}")
    array.setflags(write=False)
    return array


@dataclass(frozen=True, slots=True)
class FloatingStructure:
    """A right-handed finite-precision periodic structure in row convention."""

    lattice: FloatArray
    fractional_positions: FloatArray
    species: tuple[Species, ...]

    def __post_init__(self) -> None:
        positions = np.asarray(self.fractional_positions, dtype=np.float64)
        if positions.ndim != 2 or positions.shape[1:] != (3,) or not len(positions):
            raise DomainError("fractional positions must have shape [N,3] with N>0")
        lattice = _readonly_float_array(self.lattice, shape=(3, 3), name="lattice")
        if float(np.linalg.det(lattice)) <= 0.0:
            raise DomainError(
                "lattice must be right handed; orientation is not silently reflected"
            )
        if not np.all(np.isfinite(positions)):
            raise DomainError("fractional positions must be finite")
        normalized_species = tuple(_validate_species(value) for value in self.species)
        if len(normalized_species) != len(positions):
            raise DomainError("fractional-position and species counts disagree")
        canonical_positions = np.mod(positions, 1.0)
        # This boundary wraps the supplied data, it does not idealize them.
        # np.mod can round a tiny negative residue to exactly 1; only that
        # endpoint is equivalent to 0 in the stored half-open representation.
        # An approximate comparison would erase resolved offsets near 1.
        canonical_positions[canonical_positions == 1.0] = 0.0
        canonical_positions.setflags(write=False)
        object.__setattr__(self, "lattice", lattice)
        object.__setattr__(self, "fractional_positions", canonical_positions)
        object.__setattr__(self, "species", normalized_species)


@dataclass(frozen=True, slots=True)
class CertifiedFloatingTranslationSubgroup:
    """Exact certificate reconstructed from finite-precision pure translations."""

    exact_lattice: ExactTranslationLattice
    denominator_bound: int
    reported_max_cartesian_error: float
    action_max_cartesian_error: float

    @property
    def translation_subgroup(self) -> tuple[FractionVector, ...]:
        return self.exact_lattice.translation_subgroup

    @property
    def basis_change(self) -> FractionMatrix:
        return self.exact_lattice.basis_change

    @property
    def inverse_basis_change(self) -> IntegerMatrix:
        return self.exact_lattice.inverse_basis_change


def _has_perfect_matching(adjacency: Sequence[Sequence[int]]) -> tuple[int, ...] | None:
    """Return a deterministic left-to-right perfect matching, if one exists."""

    size = len(adjacency)
    if any(any(index not in range(size) for index in row) for row in adjacency):
        raise DomainError("matching adjacency contains an invalid right-hand index")
    matched_right = [-1] * size

    def augment(left: int, visited: set[int]) -> bool:
        for right in adjacency[left]:
            if right in visited:
                continue
            visited.add(right)
            previous = matched_right[right]
            if previous < 0 or augment(previous, visited):
                matched_right[right] = left
                return True
        return False

    if not all(augment(left, set()) for left in range(size)):
        return None
    matched_left = [-1] * size
    for right, left in enumerate(matched_right):
        matched_left[left] = right
    if any(right < 0 for right in matched_left):  # pragma: no cover - defensive
        raise InvariantViolation("perfect-matching reconstruction is incomplete")
    return tuple(matched_left)


def certify_floating_translation_subgroup(
    structure: FloatingStructure,
    reported_translations: npt.ArrayLike,
    expected_index: int,
    *,
    cartesian_tolerance: float,
) -> CertifiedFloatingTranslationSubgroup:
    """Reconstruct and certify the exact raw translation subgroup.

    A translation in a finite group ``T`` of order ``J`` has order dividing
    ``J`` (Lagrange), so every fractional component has denominator dividing
    ``J``.  Thus ``J`` is a *proved* reconstruction bound, not a decimal-grid
    guess.  Each reconstructed vector is accepted only if it is the unique
    bounded-denominator candidate near the reported float, agrees in Cartesian
    norm, forms an exact group of order ``J``, and acts on the raw motif by a
    species-preserving perfect matching within ``cartesian_tolerance``.
    """

    if (
        isinstance(expected_index, bool)
        or not isinstance(expected_index, Integral)
        or int(expected_index) <= 0
    ):
        raise DomainError("expected translation index must be a positive integer")
    index = int(expected_index)
    if len(structure.species) % index:
        raise InvariantViolation("translation index does not divide the raw atom count")
    if not np.isfinite(cartesian_tolerance) or cartesian_tolerance <= 0.0:
        raise DomainError("translation certification tolerance must be finite and positive")
    reported = np.asarray(reported_translations, dtype=np.float64)
    if reported.ndim != 2 or reported.shape[1:] != (3,) or not len(reported):
        raise NumericalError("reported pure translations must have shape [K,3] with K>0")
    if not np.all(np.isfinite(reported)):
        raise NumericalError("reported pure translations contain nonfinite values")

    exact_to_reported: dict[FractionVector, FloatArray] = {}
    reported_max_error = 0.0
    uniqueness_radius = Fraction(1, 2 * index * index)
    for raw_translation in reported:
        exact = tuple(
            fraction_mod_one(Fraction(float(fraction_mod)).limit_denominator(index))
            for fraction_mod in np.mod(raw_translation, 1.0)
        )
        if any(index % value.denominator for value in exact):
            raise InvariantViolation(
                "reconstructed translation denominator does not divide the group index"
            )
        exact_float = np.asarray([float(value) for value in exact], dtype=np.float64)
        fractional_delta = exact_float - raw_translation
        fractional_delta -= np.rint(fractional_delta)
        if any(
            Fraction.from_float(abs(float(value))) >= uniqueness_radius
            for value in fractional_delta
        ):
            raise NumericalError(
                "a reported translation has no unique rational reconstruction at the "
                "group-index denominator bound"
            )
        cartesian_error = float(
            np.sqrt(
                closest_periodic_image(
                    exact_float - raw_translation,
                    structure.lattice,
                ).squared_distance
            )
        )
        if cartesian_error > cartesian_tolerance:
            raise NumericalError(
                "rationally reconstructed translation exceeds the Cartesian tolerance"
            )
        reported_max_error = max(reported_max_error, cartesian_error)
        exact_to_reported.setdefault(exact, raw_translation.copy())

    if len(exact_to_reported) != index:
        raise InvariantViolation(
            "reconstructed pure translations disagree with the primitive count index"
        )
    exact_lattice = exact_primitive_basis_from_translations(tuple(exact_to_reported))

    positions = np.asarray(structure.fractional_positions, dtype=np.float64)
    species_groups: dict[Species, list[int]] = {}
    for atom_index, atom_species in enumerate(structure.species):
        species_groups.setdefault(atom_species, []).append(atom_index)
    action_max_error = 0.0
    for exact_translation in exact_lattice.translation_subgroup:
        translation_float = np.asarray(
            [float(value) for value in exact_translation], dtype=np.float64
        )
        for indices in species_groups.values():
            # Exact adjacency, and exact distances on every adjacent pair
            # (``periodic_pairs_within``); only adjacent pairs can be matched.
            within, pair_distances = periodic_pairs_within(
                positions[indices] + translation_float,
                positions[indices],
                structure.lattice,
                cartesian_tolerance,
            )
            adjacency = [
                [int(target) for target in np.flatnonzero(row)] for row in within
            ]
            distances = pair_distances.tolist()
            matching = _has_perfect_matching(adjacency)
            if matching is None:
                raise InvariantViolation(
                    "an exact reconstructed translation does not preserve the coloured motif"
                )
            for source_local, target_local in enumerate(matching):
                action_max_error = max(
                    action_max_error,
                    distances[source_local][target_local],
                )
    return CertifiedFloatingTranslationSubgroup(
        exact_lattice=exact_lattice,
        denominator_bound=index,
        reported_max_cartesian_error=reported_max_error,
        action_max_cartesian_error=action_max_error,
    )


def _require_optional_length(value: float | None, *, name: str) -> None:
    if value is not None and (
        isinstance(value, bool) or not np.isfinite(value) or value <= 0.0
    ):
        raise DomainError(f"{name} must be None or a finite positive length in angstrom")


def lattice_idealization_displacement(raw: npt.ArrayLike, ideal: npt.ArrayLike) -> float:
    """Largest Cartesian displacement of any cell point when ``raw`` becomes ``ideal``.

    Both row lattices must share one Cartesian frame and basis.  A point with
    fractional coordinates ``f`` in the unit cube moves by ``f (ideal - raw)``;
    its norm is convex in ``f``, so the maximum over the cube is attained at
    one of the eight vertices and is computed exactly there.
    """

    delta = np.asarray(ideal, dtype=np.float64) - np.asarray(raw, dtype=np.float64)
    if delta.shape != (3, 3) or not np.all(np.isfinite(delta)):
        raise DomainError("idealization displacement needs two finite 3x3 lattices")
    vertices = np.array(
        [[i, j, k] for i in (0.0, 1.0) for j in (0.0, 1.0) for k in (0.0, 1.0)]
    )
    return float(np.max(np.linalg.norm(vertices @ delta, axis=1)))


def _require_lattice_fidelity(
    raw: FloatArray,
    ideal: FloatArray,
    *,
    tolerance_angstrom: float | None,
    legacy_tolerance: float,
    error: type[Exception],
    legacy_message: str,
    message: str,
) -> None:
    """Raw-to-ideal lattice gate: legacy componentwise or dimensioned in angstrom.

    The legacy branch and its message are byte-for-byte the v6 behaviour.
    """

    if tolerance_angstrom is None:
        if not np.allclose(raw, ideal, atol=legacy_tolerance, rtol=legacy_tolerance):
            raise error(legacy_message)
        return
    displacement = lattice_idealization_displacement(raw, ideal)
    if not displacement <= tolerance_angstrom:
        raise error(
            f"{message}: lattice idealization moves a cell point by {displacement:.3e} A, "
            f"more than the declared {tolerance_angstrom:.3e} A"
        )


def idealized_volume_matches(
    raw_volume: float,
    ideal_volume: float,
    *,
    raw_primitive_lattice: npt.ArrayLike | None,
    lattice_displacement_angstrom: float | None,
) -> bool:
    """Raw-cell volume against index times the idealized primitive volume.

    ``None`` keeps the legacy relation ``isclose(atol=1e-10, rtol=1e-8)``.
    Otherwise every vertex of the raw primitive cell ``L`` moves by at most
    ``d``, so ``||Delta||_2 <= ||Delta||_F <= sqrt(3) d`` and
    ``L' = L (I + X)`` with ``||X||_2 <= x = sqrt(3) d / sigma_min(L)``.  All
    singular values of ``I + X`` lie in ``[1 - x, 1 + x]`` (Weyl), hence
    ``|V'/V - 1| <= (1 + x)^3 - 1``.  The legacy float slack is retained.
    """

    slack = 1.0e-10 + 1.0e-8 * abs(ideal_volume)
    if lattice_displacement_angstrom is None:
        return bool(np.isclose(raw_volume, ideal_volume, atol=1.0e-10, rtol=1.0e-8))
    if raw_primitive_lattice is None:
        return False
    lattice = np.asarray(raw_primitive_lattice, dtype=np.float64)
    sigma_min = float(np.linalg.svd(lattice, compute_uv=False)[-1])
    x = np.sqrt(3.0) * float(lattice_displacement_angstrom) / sigma_min
    if not 0.0 <= x < 1.0:
        return False
    bound = ((1.0 + x) ** 3 - 1.0) * abs(ideal_volume) + slack
    return abs(raw_volume - ideal_volume) <= bound


@dataclass(frozen=True, slots=True)
class SpglibPolicy:
    """Frozen numerical and backend identity for one standardization corpus."""

    symprec: float = 1.0e-5
    angle_tolerance: float = -1.0
    required_spglib_version: str = "2.7.0"
    canonical_decimals: int = 12
    roundtrip_tolerance: float = 2.0e-8
    stability_symprec_factors: tuple[float, ...] = (0.5, 2.0)
    relative_metric_quantum: float | None = None
    """Primitive-ranking metric quantum relative to ``max_i G_ii`` (review B2).

    ``None`` keeps the legacy absolute ``10**-min(canonical_decimals, 10)`` A^2,
    whose resolution guard rejects cells with sigma_max(L) >= ~14.8 A.
    """
    idealization_tolerance_angstrom: float | None = None
    """Maximum raw-to-ideal Cartesian displacement of any primitive-cell point.

    ``None`` keeps the legacy v6 gate: raw and idealized lattices compared
    componentwise with the dimensionless ``roundtrip_tolerance``.  A length
    replaces that gate by :func:`lattice_idealization_displacement`, so symmetry
    classification and idealization are bounded in the same unit (review B1).
    ``roundtrip_tolerance`` then governs only internal consistency checks.
    """

    def fingerprint_payload(self) -> dict[str, object]:
        """Identity fields; the legacy policy keeps its historical payload."""

        payload = asdict(self)
        for name in ("relative_metric_quantum", "idealization_tolerance_angstrom"):
            if payload[name] is None:
                del payload[name]
        return payload

    def __post_init__(self) -> None:
        if not np.isfinite(self.symprec) or self.symprec <= 0.0:
            raise DomainError("symprec must be finite and positive")
        if not np.isfinite(self.angle_tolerance) or (
            self.angle_tolerance != -1.0 and self.angle_tolerance <= 0.0
        ):
            raise DomainError("angle_tolerance must be -1 or finite and positive")
        if not self.required_spglib_version.strip():
            raise DomainError("the spglib version must be pinned explicitly")
        if not 6 <= self.canonical_decimals <= 15:
            raise DomainError("canonical_decimals must lie between 6 and 15")
        if not np.isfinite(self.roundtrip_tolerance) or self.roundtrip_tolerance <= 0.0:
            raise DomainError("roundtrip_tolerance must be finite and positive")
        factors = tuple(float(value) for value in self.stability_symprec_factors)
        if (
            not factors
            or any(not np.isfinite(value) or value <= 0.0 for value in factors)
            or tuple(sorted(set(factors))) != factors
            or not any(value < 1.0 for value in factors)
            or not any(value > 1.0 for value in factors)
        ):
            raise DomainError(
                "stability_symprec_factors must be sorted, unique, positive, and straddle one"
            )
        object.__setattr__(self, "stability_symprec_factors", factors)
        _require_optional_length(
            self.idealization_tolerance_angstrom, name="idealization_tolerance_angstrom"
        )
        if self.relative_metric_quantum is not None and (
            isinstance(self.relative_metric_quantum, bool)
            or not np.isfinite(self.relative_metric_quantum)
            or not 1e-11 <= self.relative_metric_quantum <= 1e-4
        ):
            raise DomainError("relative_metric_quantum must be None or lie in [1e-11, 1e-4]")


@dataclass(frozen=True, slots=True)
class StandardizationProvenance:
    """Deterministic identity of a crystallographic standardization decision."""

    backend: str
    backend_version: str
    hall_map_fingerprint: str
    hall_number: int
    symprec: float
    angle_tolerance: float
    canonical_decimals: int
    input_atom_count: int
    primitive_atom_count: int
    conventional_atom_count: int
    stability_symprec_factors: tuple[float, ...]
    symmetry_stability_verified: bool
    algorithm: str = STANDARDIZATION_ALGORITHM_ID
    roundtrip_tolerance: float = 2.0e-8
    idealization_tolerance_angstrom: float | None = None
    relative_metric_quantum: float | None = None

    def __post_init__(self) -> None:
        if self.relative_metric_quantum is not None and (
            isinstance(self.relative_metric_quantum, bool)
            or not np.isfinite(self.relative_metric_quantum)
            or self.relative_metric_quantum <= 0.0
        ):
            raise DomainError("provenance relative_metric_quantum must be positive")
        _require_optional_length(
            self.idealization_tolerance_angstrom,
            name="provenance idealization_tolerance_angstrom",
        )
        if not self.algorithm:
            raise DomainError("standardization provenance needs an algorithm identity")
        if not np.isfinite(self.roundtrip_tolerance) or self.roundtrip_tolerance <= 0.0:
            raise DomainError("provenance roundtrip_tolerance must be finite and positive")
        factors = tuple(float(value) for value in self.stability_symprec_factors)
        if not factors or any(not np.isfinite(value) or value <= 0.0 for value in factors):
            raise DomainError("provenance stability factors must be finite and positive")
        object.__setattr__(self, "stability_symprec_factors", factors)
        if not isinstance(self.symmetry_stability_verified, bool):
            raise DomainError("symmetry_stability_verified must be Boolean")

    @property
    def fingerprint(self) -> str:
        payload = {
            "angle_tolerance": self.angle_tolerance,
            "algorithm": self.algorithm,
            "backend": self.backend,
            "backend_version": self.backend_version,
            "canonical_decimals": self.canonical_decimals,
            "conventional_atom_count": self.conventional_atom_count,
            "hall_map_fingerprint": self.hall_map_fingerprint,
            "hall_number": self.hall_number,
            "input_atom_count": self.input_atom_count,
            "primitive_atom_count": self.primitive_atom_count,
            "roundtrip_tolerance": self.roundtrip_tolerance,
            "stability_symprec_factors": list(self.stability_symprec_factors),
            "symmetry_stability_verified": self.symmetry_stability_verified,
            "symprec": self.symprec,
        }
        if self.idealization_tolerance_angstrom is not None:
            payload["idealization_tolerance_angstrom"] = self.idealization_tolerance_angstrom
        if self.relative_metric_quantum is not None:
            payload["relative_metric_quantum"] = self.relative_metric_quantum
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        return sha256(encoded).hexdigest()


@dataclass(frozen=True, slots=True)
class CanonicalizedStructure:
    """Pinned conventional target plus its fully reduced primitive witness.

    The returned primitive cell is in spglib's standardized Cartesian frame.
    The exact rational basis is necessarily defined in the raw Cartesian
    frame, and the certified relation is

    ``primitive_lattice = C_raw @ input_lattice @ Q``

    where ``C_raw`` is ``exact_input_to_raw_frame_primitive_basis`` and ``Q``
    is ``raw_frame_to_standard_rotation``.  Keeping both factors explicit
    avoids representing a Cartesian rotation as a rational fractional basis.
    """

    international_number: int
    hall_number: int
    hall_symbol: str
    international_symbol: str
    setting_choice: str
    conventional_lattice: FloatArray
    conventional_fractional_positions: FloatArray
    conventional_species: tuple[Species, ...]
    wyckoff_letters: tuple[str, ...]
    site_symmetry_symbols: tuple[str, ...]
    crystallographic_orbits: tuple[int, ...]
    primitive_lattice: FloatArray
    primitive_fractional_positions: FloatArray
    primitive_species: tuple[Species, ...]
    raw_translation_subgroup: FloatArray
    input_mapping_to_primitive: tuple[int, ...]
    transformation_matrix: FloatArray
    spglib_origin_shift: FloatArray
    free_origin_axes: tuple[int, ...]
    subtracted_free_origin: FloatArray
    provenance: StandardizationProvenance
    input_to_raw_frame_primitive_basis: FloatArray | None = None
    exact_raw_translation_subgroup: tuple[FractionVector, ...] | None = None
    exact_input_to_raw_frame_primitive_basis: FractionMatrix | None = None
    exact_inverse_input_to_raw_frame_primitive_basis: IntegerMatrix | None = None
    exact_translation_max_cartesian_error: float | None = None
    raw_frame_primitive_lattice: FloatArray | None = None
    raw_frame_to_standard_rotation: FloatArray | None = None
    idealization_lattice_displacement_angstrom: float | None = None
    """Largest raw-to-ideal displacement of a primitive-cell point (metadata)."""
    idealization_site_displacement_angstrom: float | None = None
    """Largest raw-to-ideal site displacement in the idealized metric (metadata)."""

    def __post_init__(self) -> None:
        for name in (
            "idealization_lattice_displacement_angstrom",
            "idealization_site_displacement_angstrom",
        ):
            value = getattr(self, name)
            if value is not None and (not np.isfinite(value) or value < 0.0):
                raise DomainError(f"{name} must be finite and nonnegative")
        if self.international_number not in range(1, 231):
            raise DomainError("standardized structure has an invalid space-group type")
        if self.hall_number not in range(1, 531) or not self.hall_symbol:
            raise DomainError("standardized structure has an invalid Hall setting")
        if self.provenance.hall_number != self.hall_number:
            raise DomainError("standardization provenance names a different Hall setting")
        conventional_count = len(self.conventional_species)
        primitive_count = len(self.primitive_species)
        if not conventional_count or not primitive_count:
            raise DomainError("standardized motifs cannot be empty")
        arrays = {
            "conventional_lattice": (self.conventional_lattice, (3, 3)),
            "conventional_fractional_positions": (
                self.conventional_fractional_positions,
                (conventional_count, 3),
            ),
            "primitive_lattice": (self.primitive_lattice, (3, 3)),
            "primitive_fractional_positions": (
                self.primitive_fractional_positions,
                (primitive_count, 3),
            ),
            "transformation_matrix": (self.transformation_matrix, (3, 3)),
            "spglib_origin_shift": (self.spglib_origin_shift, (3,)),
            "subtracted_free_origin": (self.subtracted_free_origin, (3,)),
        }
        for name, (value, shape) in arrays.items():
            object.__setattr__(
                self,
                name,
                _readonly_float_array(value, shape=shape, name=name),
            )
        if self.input_to_raw_frame_primitive_basis is not None:
            primitive_basis = _readonly_float_array(
                self.input_to_raw_frame_primitive_basis,
                shape=(3, 3),
                name="input_to_raw_frame_primitive_basis",
            )
            object.__setattr__(self, "input_to_raw_frame_primitive_basis", primitive_basis)
        if self.raw_frame_primitive_lattice is not None:
            object.__setattr__(
                self,
                "raw_frame_primitive_lattice",
                _readonly_float_array(
                    self.raw_frame_primitive_lattice,
                    shape=(3, 3),
                    name="raw_frame_primitive_lattice",
                ),
            )
        if self.raw_frame_to_standard_rotation is not None:
            object.__setattr__(
                self,
                "raw_frame_to_standard_rotation",
                _readonly_float_array(
                    self.raw_frame_to_standard_rotation,
                    shape=(3, 3),
                    name="raw_frame_to_standard_rotation",
                ),
            )
        translations = np.asarray(self.raw_translation_subgroup, dtype=np.float64)
        if translations.ndim != 2 or translations.shape[1:] != (3,) or not len(translations):
            raise DomainError("raw translation subgroup must have shape [J,3] with J>0")
        if not np.all(np.isfinite(translations)):
            raise DomainError("raw translation subgroup must be finite")
        translations = np.mod(translations, 1.0)
        translations.setflags(write=False)
        object.__setattr__(self, "raw_translation_subgroup", translations)
        exact_fields = (
            self.exact_raw_translation_subgroup,
            self.exact_input_to_raw_frame_primitive_basis,
            self.exact_inverse_input_to_raw_frame_primitive_basis,
            self.exact_translation_max_cartesian_error,
            self.raw_frame_primitive_lattice,
            self.raw_frame_to_standard_rotation,
        )
        if any(value is not None for value in exact_fields):
            if any(value is None for value in exact_fields):
                raise DomainError(
                    "exact translation certification fields must be supplied together"
                )
            exact_translations = tuple(
                as_fraction_vector(row, length=3)
                for row in cast(tuple[FractionVector, ...], self.exact_raw_translation_subgroup)
            )
            exact_certificate = exact_primitive_basis_from_translations(exact_translations)
            if exact_certificate.translation_subgroup != exact_translations:
                raise InvariantViolation(
                    "exact raw translation subgroup is not in canonical order"
                )
            exact_basis = as_fraction_matrix(
                cast(
                    FractionMatrix,
                    self.exact_input_to_raw_frame_primitive_basis,
                ),
                ncols=3,
            )
            if len(exact_basis) != 3:
                raise DomainError(
                    "exact input-to-raw-frame-primitive basis must have shape [3,3]"
                )
            exact_inverse = as_integer_matrix(
                cast(
                    IntegerMatrix,
                    self.exact_inverse_input_to_raw_frame_primitive_basis,
                ),
                shape=(3, 3),
            )
            identity = as_fraction_matrix(_IDENTITY_3, ncols=3)
            if (
                exact_matmul(exact_basis, exact_inverse) != identity
                or exact_matmul(exact_inverse, exact_basis) != identity
            ):
                raise InvariantViolation(
                    "exact input-to-raw-frame-primitive basis and inverse disagree"
                )
            if exact_determinant(exact_basis) != Fraction(1, len(exact_translations)):
                raise InvariantViolation(
                    "exact input-to-raw-frame-primitive basis violates the translation index"
                )
            for translation in exact_translations:
                if any(
                    value.denominator != 1
                    for value in exact_row_matmul(translation, exact_inverse)
                ):
                    raise InvariantViolation(
                        "exact translation lies outside the supplied primitive kernel"
                    )
            exact_translation_array = np.asarray(
                [[float(value) for value in row] for row in exact_translations],
                dtype=np.float64,
            )
            if translations.shape != exact_translation_array.shape:
                raise InvariantViolation(
                    "floating and exact raw translation subgroup sizes disagree"
                )
            translation_delta = translations - exact_translation_array
            translation_delta -= np.rint(translation_delta)
            if float(np.max(np.abs(translation_delta))) > self.provenance.roundtrip_tolerance:
                raise InvariantViolation(
                    "floating and exact raw translation subgroup values disagree"
                )
            exact_basis_array = np.asarray(
                [[float(value) for value in row] for row in exact_basis],
                dtype=np.float64,
            )
            if self.input_to_raw_frame_primitive_basis is not None and not np.allclose(
                self.input_to_raw_frame_primitive_basis,
                exact_basis_array,
                atol=self.provenance.roundtrip_tolerance,
                rtol=self.provenance.roundtrip_tolerance,
            ):
                raise InvariantViolation(
                    "floating and exact raw-frame primitive bases disagree"
                )
            max_error = cast(float, self.exact_translation_max_cartesian_error)
            if (
                not np.isfinite(max_error)
                or max_error < 0.0
                or max_error > self.provenance.symprec
            ):
                raise InvariantViolation(
                    "exact translation certificate exceeds the symmetry tolerance"
                )
            object.__setattr__(self, "exact_raw_translation_subgroup", exact_translations)
            object.__setattr__(self, "exact_input_to_raw_frame_primitive_basis", exact_basis)
            object.__setattr__(
                self,
                "exact_inverse_input_to_raw_frame_primitive_basis",
                exact_inverse,
            )
            raw_frame_lattice = cast(FloatArray, self.raw_frame_primitive_lattice)
            frame_rotation = cast(FloatArray, self.raw_frame_to_standard_rotation)
            tolerance = self.provenance.roundtrip_tolerance
            if float(np.linalg.det(raw_frame_lattice)) <= 0.0:
                raise InvariantViolation("raw-frame primitive lattice is left handed")
            if not np.allclose(
                frame_rotation @ frame_rotation.T,
                np.eye(3),
                atol=tolerance,
                rtol=tolerance,
            ) or not np.isclose(
                np.linalg.det(frame_rotation),
                1.0,
                atol=tolerance,
                rtol=tolerance,
            ):
                raise InvariantViolation(
                    "raw-to-standard Cartesian frame map is not a proper rotation"
                )
            _require_lattice_fidelity(
                raw_frame_lattice @ frame_rotation,
                self.primitive_lattice,
                tolerance_angstrom=self.provenance.idealization_tolerance_angstrom,
                legacy_tolerance=tolerance,
                error=InvariantViolation,
                legacy_message=(
                    "raw-frame basis and Cartesian rotation do not reproduce the "
                    "returned primitive lattice"
                ),
                message=(
                    "raw-frame basis and Cartesian rotation do not reproduce the "
                    "returned primitive lattice"
                ),
            )
        if (
            float(np.linalg.det(self.conventional_lattice)) <= 0.0
            or float(np.linalg.det(self.primitive_lattice)) <= 0.0
        ):
            raise InvariantViolation("standardized cells must remain right handed")
        metadata_lengths = {
            len(self.wyckoff_letters),
            len(self.site_symmetry_symbols),
            len(self.crystallographic_orbits),
            conventional_count,
        }
        if len(metadata_lengths) != 1:
            raise DomainError("conventional atom metadata are not aligned")
        if len(self.input_mapping_to_primitive) != self.provenance.input_atom_count:
            raise DomainError("input-to-primitive mapping has the wrong length")
        mapping = tuple(self.input_mapping_to_primitive)
        if any(
            isinstance(value, bool)
            or not isinstance(value, int)
            or value not in range(primitive_count)
            for value in mapping
        ):
            raise DomainError("input-to-primitive mapping contains an invalid output index")
        if set(mapping) != set(range(primitive_count)):
            raise DomainError("input-to-primitive mapping does not cover the primitive motif")
        axes = tuple(self.free_origin_axes)
        if tuple(sorted(set(axes))) != axes or any(axis not in range(3) for axis in axes):
            raise DomainError("free origin axes must be sorted distinct members of {0,1,2}")
        if (
            self.provenance.primitive_atom_count != primitive_count
            or self.provenance.conventional_atom_count != conventional_count
            or self.provenance.input_atom_count
            != primitive_count * len(self.raw_translation_subgroup)
        ):
            raise DomainError("standardization provenance atom/index counts disagree")
        if self.input_to_raw_frame_primitive_basis is not None and not np.isclose(
            np.linalg.det(self.input_to_raw_frame_primitive_basis),
            1.0 / len(self.raw_translation_subgroup),
            atol=self.provenance.roundtrip_tolerance,
            rtol=self.provenance.roundtrip_tolerance,
        ):
            raise InvariantViolation(
                "input-to-raw-frame-primitive basis determinant violates the translation index"
            )
        if conventional_count % primitive_count:
            raise InvariantViolation(
                "primitive atom count does not divide the conventional motif"
            )
        centering_index = conventional_count // primitive_count
        expected_conventional = Counter(
            {
                atom_species: centering_index * count
                for atom_species, count in Counter(self.primitive_species).items()
            }
        )
        if Counter(self.conventional_species) != expected_conventional:
            raise InvariantViolation(
                "conventional and primitive coloured counts violate the centring index"
            )
        if not np.isclose(
            np.linalg.det(self.conventional_lattice),
            centering_index * np.linalg.det(self.primitive_lattice),
            atol=self.provenance.roundtrip_tolerance,
            rtol=self.provenance.roundtrip_tolerance,
        ):
            raise InvariantViolation(
                "conventional and primitive volumes violate the centring index"
            )

    @property
    def raw_translation_index(self) -> int:
        return len(self.raw_translation_subgroup)

    @property
    def centering_index(self) -> int:
        return len(self.conventional_species) // len(self.primitive_species)

    @property
    def conventional_metric(self) -> FloatArray:
        """Fractional metric for Wyckoff/scaffold training targets."""

        metric = self.conventional_lattice @ self.conventional_lattice.T
        metric.setflags(write=False)
        return metric

    @property
    def primitive_metric(self) -> FloatArray:
        """Fractional metric of the FullPrim physical witness."""

        metric = self.primitive_lattice @ self.primitive_lattice.T
        metric.setflags(write=False)
        return metric


@dataclass(frozen=True, slots=True)
class _CanonicalMotif:
    positions: FloatArray
    encoded_types: tuple[int, ...]
    permutation: tuple[int, ...]
    shift: FloatArray


def _verify_pinned_roundtrip(
    conventional: _CanonicalMotif,
    roundtrip: _CanonicalMotif,
    *,
    conventional_lattice: FloatArray,
    roundtrip_lattice: FloatArray,
    primitive_lattice: FloatArray,
    primitive_shift: FloatArray,
    transformation: FloatArray,
    origin_shift: FloatArray,
    tolerance: float,
    typed_motif_rejection: bool = False,
) -> None:
    """Audit the declared proper affine map, not identical Hall coordinates.

    A pinned Hall number does not fix every proper basis normalizer (a polar
    hexagonal setting can reverse z while swapping x and y). For row vectors,
    W = L_conv L_prim^-1 P^T and s = -u_prim P^T + p - u_roundtrip.
    This checks that known map only: no nearest-structure search, mirror
    quotient, or alternative fitted origin is permitted. Coordinates retain
    the existing componentwise fractional tolerance of the motif round trip.

    With ``typed_motif_rejection`` (the dimensioned policy), a motif that moves
    under an otherwise valid map is a recorded rejection rather than a fatal
    invariant.  At a large tolerance spglib's re-idealization depends on how
    the cell is presented.  The round trip re-anchors the primitive origin on
    an atom.  On that exactly symmetric cell spglib returns, at symprec tau, an
    operation that holds only within tau.  For mp-1179655 (C2/c) at
    tau = 1e-2 A the inversion translation is (0, 0, 1/2) rather than the exact
    (0.001146, 0.998854, 1/2) found at symprec <= 1e-6.  Re-idealizing about
    that centre moves six sites by 1.15e-3 (fractional).  Malformed maps,
    non-isometries and species changes stay fatal under every policy.
    """

    for value in (
        conventional_lattice, roundtrip_lattice, primitive_lattice, transformation
    ):
        if value.shape != (3, 3) or not np.all(np.isfinite(value)):
            raise InvariantViolation("primitive round trip has a malformed basis map")
    for value in (primitive_shift, origin_shift, roundtrip.shift):
        if value.shape != (3,) or not np.all(np.isfinite(value)):
            raise InvariantViolation("primitive round trip has a malformed origin map")
    try:
        row_map = np.linalg.solve(primitive_lattice.T, conventional_lattice.T).T
        basis = _integer_basis_witness(
            row_map @ transformation.T,
            determinant=1,
            tolerance=tolerance,
            name="primitive round-trip conventional rebasing",
        )
        rotation = np.linalg.solve(
            conventional_lattice, np.asarray(basis, dtype=np.float64) @ roundtrip_lattice
        )
    except np.linalg.LinAlgError as exc:
        raise InvariantViolation("primitive round trip has a singular basis map") from exc
    if not np.allclose(
        rotation @ rotation.T, np.eye(3), atol=tolerance, rtol=tolerance
    ) or not np.isclose(np.linalg.det(rotation), 1.0, atol=tolerance, rtol=tolerance):
        raise InvariantViolation("primitive round-trip map is not a proper Cartesian isometry")
    if Counter(conventional.encoded_types) != Counter(roundtrip.encoded_types):
        raise InvariantViolation("primitive round trip changed the conventional species")
    shift = -primitive_shift @ transformation.T + origin_shift - roundtrip.shift
    mapped = conventional.positions @ np.asarray(basis, dtype=np.float64) + shift
    adjacency: list[tuple[int, ...]] = []
    for point, species in zip(mapped, conventional.encoded_types, strict=True):
        delta = point[None, :] - roundtrip.positions
        delta -= np.rint(delta)
        adjacency.append(tuple(
            index for index, other in enumerate(roundtrip.encoded_types)
            if species == other and float(np.max(np.abs(delta[index]))) <= tolerance
        ))
    if _has_perfect_matching(adjacency) is None:
        if typed_motif_rejection:
            worst = 0.0
            for point, species in zip(mapped, conventional.encoded_types, strict=True):
                delta = point[None, :] - roundtrip.positions
                delta -= np.rint(delta)
                misses = [
                    float(np.max(np.abs(delta[index])))
                    for index, other in enumerate(roundtrip.encoded_types)
                    if species == other
                ]
                worst = max(worst, min(misses))
            raise AmbiguousPreprocessingError(
                "spglib's standardization is not idempotent at this tolerance: "
                f"re-standardizing moves a site by up to {worst:.3g} (fractional)"
            )
        raise InvariantViolation("primitive round trip changed the conventional motif")


def _float_position_key(point: npt.NDArray[np.float64], decimals: int) -> tuple[float, ...]:
    return tuple(float(value) for value in np.round(point, decimals=decimals))


def _canonicalize_float_origin(
    positions: npt.ArrayLike,
    encoded_types: Sequence[int],
    free_axes: Sequence[int],
    *,
    decimals: int,
    discrete_shifts: Sequence[FractionVector] = ((Fraction(0),) * 3,),
) -> _CanonicalMotif:
    values = np.mod(np.asarray(positions, dtype=np.float64), 1.0)
    types = tuple(int(value) for value in encoded_types)
    axes = tuple(sorted(set(free_axes)))
    if values.ndim != 2 or values.shape != (len(types), 3) or not len(values):
        raise DomainError("floating origin canonicalization received an invalid motif")
    if len(axes) != len(tuple(free_axes)) or any(axis not in range(3) for axis in axes):
        raise DomainError("free origin axes must be distinct members of {0,1,2}")
    exact_shifts = tuple(as_fraction_vector(shift, length=3) for shift in discrete_shifts)
    if not exact_shifts:
        raise DomainError("the discrete origin gauge must contain the identity")
    candidates: list[
        tuple[
            tuple[tuple[int, tuple[float, ...]], ...],
            FloatArray,
            tuple[int, ...],
            tuple[int, ...],
            FloatArray,
        ]
    ] = []
    least_type = min(types)
    for exact_discrete in exact_shifts:
        discrete = np.asarray([float(value) for value in exact_discrete], dtype=np.float64)
        discretely_shifted = np.mod(values - discrete[None, :], 1.0)
        anchors: Iterable[int] = (
            (index for index, value in enumerate(types) if value == least_type)
            if axes
            else range(1)
        )
        for anchor_index in anchors:
            free_shift: FloatArray = np.zeros(3, dtype=np.float64)
            if axes:
                free_shift[list(axes)] = discretely_shifted[anchor_index, list(axes)]
            shift = np.mod(discrete + free_shift, 1.0)
            shifted = np.mod(values - shift[None, :], 1.0)
            close_to_boundary = np.isclose(shifted, 1.0, atol=10.0 ** (-decimals), rtol=0.0)
            shifted[close_to_boundary] = 0.0
            order = tuple(
                sorted(
                    range(len(types)),
                    key=lambda index: (
                        types[index],
                        _float_position_key(shifted[index], decimals),
                    ),
                )
            )
            ordered_positions = shifted[np.asarray(order, dtype=np.int64)]
            ordered_types = tuple(types[index] for index in order)
            key = tuple(
                (atom_type, _float_position_key(point, decimals))
                for atom_type, point in zip(ordered_types, ordered_positions, strict=True)
            )
            candidates.append((key, ordered_positions, ordered_types, order, shift))
    _, result_positions, result_types, result_order, result_shift = min(
        candidates, key=lambda item: item[0]
    )
    result_positions = result_positions.copy()
    result_positions.setflags(write=False)
    result_shift = result_shift.copy()
    result_shift.setflags(write=False)
    return _CanonicalMotif(
        result_positions,
        result_types,
        result_order,
        result_shift,
    )


def _dataset_field(dataset: object, name: str) -> Any:
    if hasattr(dataset, name):
        return getattr(dataset, name)
    if isinstance(dataset, Mapping):
        return dataset[name]
    try:
        return dataset[name]  # type: ignore[index]
    except (KeyError, TypeError) as error:
        raise NumericalError(f"spglib dataset omits required field {name!r}") from error


def _classification_signature(
    dataset: object,
    encoded_input_types: Sequence[int],
) -> tuple[object, ...]:
    """Return a label-invariant signature of a spglib classification.

    The signature detects tolerance-driven changes in space-group type,
    primitive/conventional counts, Wyckoff/site-symmetry assignments, orbit
    partitions, or species membership.  Arbitrary integer orbit identifiers
    are removed by serializing each orbit as a sorted block.
    """

    wyckoffs = tuple(str(value) for value in _dataset_field(dataset, "wyckoffs"))
    sites = tuple(
        "".join(str(value).split())
        for value in _dataset_field(dataset, "site_symmetry_symbols")
    )
    orbits = tuple(int(value) for value in _dataset_field(dataset, "crystallographic_orbits"))
    types = tuple(int(value) for value in encoded_input_types)
    if not (len(wyckoffs) == len(sites) == len(orbits) == len(types)):
        raise NumericalError("spglib tolerance audit returned misaligned site metadata")
    orbit_blocks: list[tuple[object, ...]] = []
    for orbit in sorted(set(orbits)):
        members = tuple(index for index, value in enumerate(orbits) if value == orbit)
        orbit_blocks.append(
            (
                tuple(sorted(types[index] for index in members)),
                tuple(sorted(wyckoffs[index] for index in members)),
                tuple(sorted(sites[index] for index in members)),
            )
        )
    mapping = tuple(int(value) for value in _dataset_field(dataset, "mapping_to_primitive"))
    return (
        int(_dataset_field(dataset, "number")),
        len(set(mapping)),
        len(tuple(_dataset_field(dataset, "std_types"))),
        tuple(sorted(orbit_blocks)),
    )


def _origin_free_axes(rotations: npt.ArrayLike) -> tuple[int, ...]:
    """Find the exact axis-aligned identity subspace of all pinned rotations."""

    values = np.asarray(rotations)
    if values.ndim != 3 or values.shape[1:] != (3, 3):
        raise NumericalError("spglib database returned malformed rotations")
    if not np.issubdtype(values.dtype, np.integer):
        if not np.all(values == np.rint(values)):
            raise NumericalError("spglib database rotations are not integral")
        values = np.rint(values).astype(np.int64)
    integer_rotations = tuple(
        tuple(tuple(int(component) for component in row) for row in rotation)
        for rotation in values
    )
    constraints: list[tuple[int, int, int]] = []
    for rotation in integer_rotations:
        for row in range(3):
            constraints.append(
                (
                    rotation[row][0] - int(row == 0),
                    rotation[row][1] - int(row == 1),
                    rotation[row][2] - int(row == 2),
                )
            )
    dimension = 3 - exact_rank(constraints, ncols=3)
    axes = tuple(
        axis
        for axis in range(3)
        if all(
            rotation[row][axis] == int(row == axis)
            for rotation in integer_rotations
            for row in range(3)
        )
    )
    if len(axes) != dimension:
        raise CatalogueError(
            "pinned Hall setting has an oblique continuous-origin subspace; "
            "an explicit exact origin chart is required"
        )
    return axes


def exact_continuous_origin_axes(setting: SpaceGroupSetting) -> tuple[int, ...]:
    """Return and certify the axis-aligned continuous origin subtorus.

    In row convention an origin displacement ``s`` leaves the pinned affine
    operation table unchanged continuously exactly when ``s(W-I)=0`` for every
    point-group rotation ``W``.  The pinned standard settings are accepted only
    when this common fixed space is spanned by crystallographic coordinate
    axes; oblique origin charts fail closed.
    """

    rotations = setting.point_group_rotations
    constraints = tuple(
        tuple(rotation[row][column] - int(row == column) for column in range(3))
        for rotation in rotations
        for row in range(3)
    )
    dimension = 3 - exact_rank(constraints, ncols=3)
    axes = tuple(
        axis
        for axis in range(3)
        if all(
            rotation[axis][column] == int(axis == column)
            for rotation in rotations
            for column in range(3)
        )
    )
    if len(axes) != dimension:
        raise CatalogueError(
            f"{setting.key} has an oblique continuous-origin subspace; "
            "an explicit exact origin chart is required"
        )
    return axes


def validate_origin_chart_compatibility(
    catalogue: CertifiedCatalogue,
    *,
    require_all_230: bool = True,
) -> Mapping[str, tuple[int, ...]]:
    """Certify the parameter-chart structure required by origin quotienting.

    For every Wyckoff row and every continuous free origin axis, exactly one
    intrinsic parameter must translate the representative by ``+e_j`` or
    ``-e_j``.  All compiler period bases must be the identity in those
    parameters.  These exact conditions allow the P1 anchor construction to
    generalize axiswise without a pseudoinverse or numerical lattice search.
    """

    catalogue.validate(require_all_230=require_all_230)
    result: dict[str, tuple[int, ...]] = {}
    for setting in catalogue.settings.values():
        axes = exact_continuous_origin_axes(setting)
        for row in setting.wyckoff_positions.values():
            dimension = row.dimension
            identity_period = tuple(
                tuple(Fraction(int(i == j)) for j in range(dimension)) for i in range(dimension)
            )
            if row.period_basis != identity_period:
                raise CatalogueError(
                    f"{setting.key}:{row.label} has a nonidentity compiler period basis"
                )
            for axis in axes:
                unit = tuple(Fraction(int(column == axis)) for column in range(3))
                negative_unit = tuple(-value for value in unit)
                matches = tuple(
                    index
                    for index, parameter_row in enumerate(row.coordinate_matrix)
                    if parameter_row in (unit, negative_unit)
                )
                if len(matches) != 1:
                    raise CatalogueError(
                        f"{setting.key}:{row.label} does not have exactly one "
                        f"+/-e_{axis} free-origin parameter row"
                    )
        result[setting.key] = axes
    return MappingProxyType(result)


def _exact_database_operations(
    rotations: npt.ArrayLike,
    translations: npt.ArrayLike,
) -> tuple[tuple[IntegerMatrix, FractionVector], ...]:
    rotation_values = np.asarray(rotations)
    translation_values = np.asarray(translations, dtype=np.float64)
    if (
        rotation_values.ndim != 3
        or rotation_values.shape[1:] != (3, 3)
        or translation_values.shape != (len(rotation_values), 3)
    ):
        raise NumericalError("spglib database returned malformed affine operations")
    result: list[tuple[IntegerMatrix, FractionVector]] = []
    for rotation_value, translation_value in zip(
        rotation_values, translation_values, strict=True
    ):
        if not np.all(rotation_value == np.rint(rotation_value)):
            raise NumericalError("spglib database returned a nonintegral rotation")
        rotation: IntegerMatrix = tuple(
            tuple(int(value) for value in row) for row in rotation_value
        )
        translation: list[Fraction] = []
        for value in translation_value:
            exact = Fraction(float(value)).limit_denominator(24)
            if abs(float(exact) - float(value)) > 1.0e-12:
                raise NumericalError(
                    "spglib database returned a non-crystallographic translation"
                )
            translation.append(fraction_mod_one(exact))
        result.append((rotation, tuple(translation)))
    if len(result) != len(set(result)):
        raise InvariantViolation("spglib database operation table contains duplicates")
    return tuple(result)


@lru_cache(maxsize=530)
def _origin_gauge_shifts(
    operations: tuple[tuple[IntegerMatrix, FractionVector], ...],
    free_axes: tuple[int, ...],
) -> tuple[FractionVector, ...]:
    """Enumerate the finite pinned-origin gauge exactly.

    For a column-convention operation ``(R,t)``, shifting the origin by ``s``
    sends ``t`` to ``t + (R-I)s``.  The operation table is unchanged as a set
    precisely for the shifts retained below.  Continuous fixed axes are set to
    zero here and handled separately by the atom-anchor quotient.

    Completeness is finite rather than grid-assumed.  If ``c`` clears the
    centring-translation denominators, every valid shift satisfies
    ``c(R-I)s in Z^3``.  For a nonsingular minor ``M`` of ``R-I``, this gives
    ``s=M^-1(t+k)`` with ``ct`` integral; hence every component denominator
    divides ``c |det M|``.  We enumerate that proved finite superset and then
    test exact group equality.
    """

    constrained_axes = tuple(axis for axis in range(3) if axis not in free_axes)
    if not constrained_axes:
        return ((Fraction(0),) * 3,)
    identity = _IDENTITY_3
    centering = tuple(
        translation for rotation, translation in operations if rotation == identity
    )
    if not centering:
        raise InvariantViolation("pinned operation table has no identity translation")
    centering_denominator = 1
    for translation in centering:
        for value in translation:
            centering_denominator = lcm(centering_denominator, value.denominator)

    constraint_rows = {
        tuple(rotation[row][axis] - int(row == axis) for axis in constrained_axes)
        for rotation, _ in operations
        for row in range(3)
    }
    nonzero_rows = tuple(sorted(row for row in constraint_rows if any(row)))
    rank = len(constrained_axes)
    if exact_rank(nonzero_rows, ncols=rank) != rank:
        raise InvariantViolation("declared constrained origin axes are not full rank")
    determinant_candidates: list[tuple[int, tuple[tuple[int, ...], ...]]] = []
    for rows in combinations(nonzero_rows, rank):
        determinant = exact_determinant(rows)
        if determinant:
            determinant_candidates.append((abs(int(determinant)), rows))
    if not determinant_candidates:
        raise InvariantViolation("origin constraint lattice has no nonsingular minor")
    minor_determinant, _ = min(determinant_candidates, key=lambda item: item[0])
    # If M is the chosen integral minor and M s equals a centring vector
    # modulo Z^r, then s=M^-1(t+k).  adj(M) is integral, so every component
    # denominator divides c*|det(M)|.  This bound is much tighter than taking
    # a determinant only after scaling all r rows by c.
    denominator = centering_denominator * minor_determinant
    if denominator <= 0:
        raise InvariantViolation("origin-gauge denominator is not positive")

    operation_set = set(operations)
    candidates: list[FractionVector] = []
    for numerators in product(range(denominator), repeat=rank):
        constrained = tuple(Fraction(value, denominator) for value in numerators)
        if any(
            (
                centering_denominator
                * sum(Fraction(row[index]) * constrained[index] for index in range(rank))
            ).denominator
            != 1
            for row in nonzero_rows
        ):
            continue
        shift = tuple(
            Fraction(0) if axis in free_axes else constrained[constrained_axes.index(axis)]
            for axis in range(3)
        )
        conjugated: set[tuple[IntegerMatrix, FractionVector]] = set()
        for rotation, translation in operations:
            delta = tuple(
                sum(
                    Fraction(rotation[row][column] - int(row == column)) * shift[column]
                    for column in range(3)
                )
                for row in range(3)
            )
            conjugated.add(
                (
                    rotation,
                    tuple(fraction_mod_one(translation[row] + delta[row]) for row in range(3)),
                )
            )
        if conjugated == operation_set:
            candidates.append(shift)
    candidates = sorted(set(candidates))
    if (Fraction(0),) * 3 not in candidates:
        raise InvariantViolation("finite origin gauge lost the identity")
    return tuple(candidates)


def _canonical_orbit_ids(values: Sequence[int]) -> tuple[int, ...]:
    remapping: dict[int, int] = {}
    result: list[int] = []
    for value in values:
        old = int(value)
        if old not in remapping:
            remapping[old] = len(remapping)
        result.append(remapping[old])
    return tuple(result)


class SpglibStandardizer:
    """Fail-closed full primitive reduction and pinned Hall standardization."""

    @property
    def algorithm_id(self) -> str:
        return STANDARDIZATION_ALGORITHM_ID

    def __init__(
        self,
        pinned_hall_numbers: Mapping[int, int],
        policy: SpglibPolicy,
    ) -> None:
        hall_map = dict(pinned_hall_numbers)
        if not hall_map:
            raise CatalogueError(
                "a nonempty international-number to Hall-number map is required"
            )
        for international_number, hall_number in hall_map.items():
            if international_number not in range(1, 231):
                raise CatalogueError("pinned map contains an invalid international number")
            if isinstance(hall_number, bool):
                raise CatalogueError("pinned Hall numbers must be integers")
            if int(hall_number) not in range(1, 531):
                raise CatalogueError("pinned map contains an invalid Hall number")
        self.pinned_hall_numbers = MappingProxyType(
            {int(number): int(hall) for number, hall in hall_map.items()}
        )
        self.policy = policy
        encoded = json.dumps(
            sorted(self.pinned_hall_numbers.items()), separators=(",", ":")
        ).encode()
        self.hall_map_fingerprint = sha256(encoded).hexdigest()

    def _load_spglib(self) -> ModuleType:
        module = cast(ModuleType, require_module("spglib", extra="crystallography"))
        installed = str(getattr(module, "__version__", ""))
        if installed != self.policy.required_spglib_version:
            raise NumericalError(
                "spglib version mismatch: "
                f"required {self.policy.required_spglib_version!r}, found {installed!r}"
            )
        return module

    def validate_catalogue_preflight(self, *, require_all_types: bool = True) -> None:
        """Validate every pinned Hall setting and its complete origin gauge.

        Dataset builders should call this once before processing training data.
        Per-structure canonicalization still checks the encountered setting, but
        that cannot detect a deleted entry elsewhere in a nominally complete
        230-type mapping.
        """

        if require_all_types and set(self.pinned_hall_numbers) != set(range(1, 231)):
            missing = sorted(set(range(1, 231)).difference(self.pinned_hall_numbers))
            extra = sorted(set(self.pinned_hall_numbers).difference(range(1, 231)))
            raise CatalogueError(
                f"pinned Hall map is not an exact 230-type bijection; "
                f"missing={missing}, extra={extra}"
            )
        spglib = self._load_spglib()
        observed_halls: set[int] = set()
        for international_number, hall_number in self.pinned_hall_numbers.items():
            if hall_number in observed_halls:
                raise CatalogueError("two space-group types share one pinned Hall serial")
            observed_halls.add(hall_number)
            group_type = spglib.get_spacegroup_type(hall_number)
            symmetry = spglib.get_symmetry_from_database(hall_number)
            if group_type is None or symmetry is None:
                raise CatalogueError(f"spglib rejected pinned Hall number {hall_number}")
            if (
                int(group_type.number) != international_number
                or int(group_type.hall_number) != hall_number
            ):
                raise CatalogueError(
                    f"pinned Hall {hall_number} does not realize type {international_number}"
                )
            free_axes = _origin_free_axes(symmetry["rotations"])
            operations = _exact_database_operations(
                symmetry["rotations"], symmetry["translations"]
            )
            shifts = _origin_gauge_shifts(operations, free_axes)
            if not shifts:
                raise InvariantViolation(f"Hall {hall_number} has an empty pinned-origin gauge")

    def _dataset(
        self,
        spglib: ModuleType,
        cell: tuple[FloatArray, FloatArray, Sequence[int]],
        *,
        hall_number: int = 0,
    ) -> object:
        dataset = spglib.get_symmetry_dataset(
            cell,
            symprec=self.policy.symprec,
            angle_tolerance=self.policy.angle_tolerance,
            hall_number=hall_number,
        )
        if dataset is None:
            raise NumericalError(
                "spglib could not classify the structure at the pinned tolerance"
            )
        return dataset

    @staticmethod
    def _encode_species(
        species: Sequence[Species],
    ) -> tuple[tuple[int, ...], Mapping[int, Species]]:
        ordered_species = sorted(set(species), key=_species_key)
        encode = {value: index + 1 for index, value in enumerate(ordered_species)}
        decode = MappingProxyType({identifier: value for value, identifier in encode.items()})
        return tuple(encode[value] for value in species), decode

    @staticmethod
    def _decode_species(
        encoded_types: Sequence[int], decode: Mapping[int, Species]
    ) -> tuple[Species, ...]:
        try:
            return tuple(decode[int(value)] for value in encoded_types)
        except KeyError as error:
            raise InvariantViolation("spglib changed an encoded chemical type") from error

    def _pinned_conventional(
        self,
        dataset: object,
        *,
        free_axes: tuple[int, ...],
        origin_gauge: tuple[FractionVector, ...],
    ) -> _CanonicalMotif:
        return _canonicalize_float_origin(
            _dataset_field(dataset, "std_positions"),
            tuple(int(value) for value in _dataset_field(dataset, "std_types")),
            free_axes,
            decimals=self.policy.canonical_decimals,
            discrete_shifts=origin_gauge,
        )

    def _metadata_in_final_origin(
        self,
        spglib: ModuleType,
        *,
        international_number: int,
        hall_number: int,
        lattice: FloatArray,
        motif: _CanonicalMotif,
    ) -> tuple[tuple[str, ...], tuple[str, ...], tuple[int, ...]]:
        """Reclassify sites after the final finite/free origin quotient.

        Dataset metadata obtained before origin canonicalization cannot be
        carried through a finite normalizer shift: such a shift may permute
        Wyckoff letters (for example P-1 ``1h -> 1a``).  Reclassification is
        accepted only when spglib returns the already-canonical conventional
        cell in exactly the same pinned gauge, modulo atom sorting.  This makes
        its input-aligned site metadata valid for ``motif.positions``.
        """

        classified = self._dataset(
            spglib,
            (lattice, motif.positions, motif.encoded_types),
            hall_number=hall_number,
        )
        if (
            int(_dataset_field(classified, "number")) != international_number
            or int(_dataset_field(classified, "hall_number")) != hall_number
        ):
            raise InvariantViolation(
                "final-origin site reclassification changed the pinned space group"
            )
        returned_lattice = np.asarray(
            _dataset_field(classified, "std_lattice"), dtype=np.float64
        )
        tolerance = self.policy.roundtrip_tolerance
        if not np.allclose(
            returned_lattice,
            lattice,
            atol=tolerance,
            rtol=tolerance,
        ):
            raise InvariantViolation(
                "final-origin site reclassification changed the conventional lattice"
            )
        returned_motif = _canonicalize_float_origin(
            _dataset_field(classified, "std_positions"),
            tuple(int(value) for value in _dataset_field(classified, "std_types")),
            (),
            decimals=self.policy.canonical_decimals,
        )
        if returned_motif.encoded_types != motif.encoded_types or not np.allclose(
            returned_motif.positions,
            motif.positions,
            atol=tolerance,
            rtol=0.0,
        ):
            raise InvariantViolation(
                "final-origin site reclassification returned a different origin gauge"
            )
        wyckoffs = tuple(str(value) for value in _dataset_field(classified, "wyckoffs"))
        sites = tuple(
            str(value) for value in _dataset_field(classified, "site_symmetry_symbols")
        )
        raw_orbits = tuple(
            int(value) for value in _dataset_field(classified, "crystallographic_orbits")
        )
        if not (len(wyckoffs) == len(sites) == len(raw_orbits) == len(motif.encoded_types)):
            raise InvariantViolation(
                "final-origin crystallographic metadata are not atom aligned"
            )
        for orbit in set(raw_orbits):
            members = tuple(index for index, value in enumerate(raw_orbits) if value == orbit)
            if (
                len({wyckoffs[index] for index in members}) != 1
                or len({sites[index] for index in members}) != 1
            ):
                raise InvariantViolation(
                    "one final-origin crystallographic orbit has inconsistent metadata"
                )
        return wyckoffs, sites, _canonical_orbit_ids(raw_orbits)

    def canonicalize(self, structure: FloatingStructure) -> CanonicalizedStructure:
        """Return a proper-basis canonical primitive-to-pinned target.

        Spglib's pinned Hall output alone need not choose a unique metric or
        motif at lattice-reduction boundaries. A complete finite primitive
        basis search supplies one numerical physical representative before a
        second pinned-setting pass. All source-frame certificates are then
        composed back to the original input; the source is never replaced in
        the returned provenance or primitive atom map.
        """

        from .primitive_canonicalization import canonicalize_primitive_presentation

        initial = self._canonicalize_pinned(structure)
        encoded, decode = self._encode_species(initial.primitive_species)
        canonical = canonicalize_primitive_presentation(
            initial.primitive_lattice,
            initial.primitive_fractional_positions,
            encoded,
            decimals=self.policy.canonical_decimals,
            relative_metric_quantum=self.policy.relative_metric_quantum,
        )
        # Validate the internal presentation map before another backend call
        # can turn a corrupted intermediate into a recoverable input failure.
        primitive_basis = np.asarray(canonical.basis_change, dtype=np.float64)
        primitive_rotation = canonical.cartesian_rotation
        tolerance = self.policy.roundtrip_tolerance
        if (
            exact_determinant(canonical.basis_change) != 1
            or exact_matmul(canonical.basis_change, canonical.inverse_basis_change)
            != _IDENTITY_3
            or not np.allclose(
                primitive_rotation @ primitive_rotation.T,
                np.eye(3), atol=tolerance, rtol=tolerance,
            )
            or not np.isclose(
                np.linalg.det(primitive_rotation), 1.0, atol=tolerance, rtol=tolerance
            )
            or not np.allclose(
                primitive_basis @ initial.primitive_lattice @ primitive_rotation,
                canonical.lattice, atol=tolerance, rtol=tolerance,
            )
        ):
            raise InvariantViolation("internal primitive presentation map is inconsistent")
        canonical_species = self._decode_species(canonical.species, decode)
        final = self._canonicalize_pinned(
            FloatingStructure(
                canonical.lattice, canonical.fractional_positions, canonical_species
            )
        )
        if final.raw_translation_index != 1:
            raise InvariantViolation(
                "canonical primitive presentation gained hidden translations"
            )
        initial_basis = initial.exact_input_to_raw_frame_primitive_basis
        initial_inverse = initial.exact_inverse_input_to_raw_frame_primitive_basis
        final_basis = final.exact_input_to_raw_frame_primitive_basis
        if (
            initial_basis is None
            or initial_inverse is None
            or final_basis is None
            or initial.raw_frame_to_standard_rotation is None
            or final.raw_frame_to_standard_rotation is None
            or initial.raw_frame_primitive_lattice is None
        ):
            raise InvariantViolation(
                "primitive repinning lacks exact source-frame certificates"
            )
        composed_basis = exact_matmul(
            exact_matmul(final_basis, canonical.basis_change), initial_basis
        )
        inverse_fraction = exact_inverse(composed_basis)
        if any(value.denominator != 1 for row in inverse_fraction for value in row):
            raise InvariantViolation("repinned primitive basis inverse is not integral")
        composed_inverse: IntegerMatrix = tuple(
            tuple(int(value) for value in row) for row in inverse_fraction
        )
        composed_rotation = (
            initial.raw_frame_to_standard_rotation
            @ canonical.cartesian_rotation
            @ final.raw_frame_to_standard_rotation
        )
        basis_array = np.asarray(composed_basis, dtype=np.float64)
        raw_frame_lattice = basis_array @ structure.lattice
        # Audit the composition itself separately from its fidelity to the
        # twice-idealized endpoint. Left fractional-basis maps and right
        # Cartesian rotations compose independently. This staged expression
        # uses the independently certified initial raw-frame primitive rather
        # than the idealized initial metric.
        staged_lattice = (
            np.asarray(final_basis, dtype=np.float64)
            @ primitive_basis
            @ initial.raw_frame_primitive_lattice
            @ initial.raw_frame_to_standard_rotation
            @ primitive_rotation
            @ final.raw_frame_to_standard_rotation
        )
        if not np.allclose(
            raw_frame_lattice @ composed_rotation,
            staged_lattice, atol=tolerance, rtol=tolerance,
        ):
            raise InvariantViolation("composed primitive witness disagrees with staged maps")
        _require_lattice_fidelity(
            raw_frame_lattice @ composed_rotation,
            final.primitive_lattice,
            tolerance_angstrom=self.policy.idealization_tolerance_angstrom,
            legacy_tolerance=self.policy.roundtrip_tolerance,
            error=AmbiguousPreprocessingError,
            legacy_message=(
                "composed idealization exceeds the frozen final-frame input-fidelity "
                "tolerance (unchanged componentwise absolute/relative checks)"
            ),
            message="composed idealization exceeds the final-frame idealization bound",
        )
        lattice_displacement = lattice_idealization_displacement(
            raw_frame_lattice @ composed_rotation, final.primitive_lattice
        )

        # Initial primitive coordinates equal f_raw C^-1 + delta, modulo one.
        # Recover and verify that common origin using the certified atom map.
        initial_fractional = np.mod(
            structure.fractional_positions @ np.asarray(initial_inverse, dtype=np.float64),
            1.0,
        )
        initial_map = initial.input_mapping_to_primitive
        delta = np.mod(
            initial.primitive_fractional_positions[initial_map[0]] - initial_fractional[0],
            1.0,
        )
        site_displacement_squared = 0.0
        for raw_index, primitive_index in enumerate(initial_map):
            error = closest_periodic_image(
                initial_fractional[raw_index]
                + delta
                - initial.primitive_fractional_positions[primitive_index],
                initial.primitive_lattice,
            )
            if error.squared_distance > self.policy.symprec**2:
                raise InvariantViolation(
                    "source-to-primitive origin composition is inconsistent"
                )
            site_displacement_squared = max(site_displacement_squared, error.squared_distance)
        primitive_to_canonical = np.asarray(canonical.inverse_basis_change, dtype=np.float64)
        canonical_origin = delta @ primitive_to_canonical - canonical.subtracted_origin
        source_to_canonical = (
            np.asarray(initial_inverse, dtype=np.float64) @ primitive_to_canonical
        )
        # Spglib's column convention is x_std = P x_input + p. The row
        # fractional map is therefore f_std = f_input P^T + p.
        composed_transformation = (source_to_canonical @ final.transformation_matrix.T).T
        composed_origin = np.mod(
            canonical_origin @ final.transformation_matrix.T + final.spglib_origin_shift,
            1.0,
        )
        old_to_canonical = {
            old_index: new_index for new_index, old_index in enumerate(canonical.permutation)
        }
        composed_map = tuple(
            final.input_mapping_to_primitive[old_to_canonical[index]] for index in initial_map
        )
        provenance = replace(
            final.provenance,
            input_atom_count=len(structure.species),
            algorithm=self.algorithm_id,
        )
        return replace(
            final,
            raw_translation_subgroup=initial.raw_translation_subgroup,
            input_mapping_to_primitive=composed_map,
            transformation_matrix=composed_transformation,
            spglib_origin_shift=composed_origin,
            provenance=provenance,
            input_to_raw_frame_primitive_basis=basis_array,
            exact_raw_translation_subgroup=initial.exact_raw_translation_subgroup,
            exact_input_to_raw_frame_primitive_basis=composed_basis,
            exact_inverse_input_to_raw_frame_primitive_basis=composed_inverse,
            exact_translation_max_cartesian_error=(
                initial.exact_translation_max_cartesian_error
            ),
            raw_frame_primitive_lattice=raw_frame_lattice,
            raw_frame_to_standard_rotation=composed_rotation,
            idealization_lattice_displacement_angstrom=lattice_displacement,
            idealization_site_displacement_angstrom=float(np.sqrt(site_displacement_squared)),
        )

    def _canonicalize_pinned(self, structure: FloatingStructure) -> CanonicalizedStructure:
        """Reduce and pin one presentation without asserting basis uniqueness."""

        spglib = self._load_spglib()
        encoded_types, decode = self._encode_species(structure.species)
        cell = (
            np.asarray(structure.lattice, dtype=np.float64),
            np.asarray(structure.fractional_positions, dtype=np.float64),
            encoded_types,
        )
        detected = self._dataset(spglib, cell)
        reference_signature = _classification_signature(detected, encoded_types)
        for factor in self.policy.stability_symprec_factors:
            audited = spglib.get_symmetry_dataset(
                cell,
                symprec=self.policy.symprec * factor,
                angle_tolerance=self.policy.angle_tolerance,
            )
            if audited is None:
                raise AmbiguousPreprocessingError(
                    "symmetry classification disappears inside the frozen tolerance window"
                )
            if _classification_signature(audited, encoded_types) != reference_signature:
                raise AmbiguousPreprocessingError(
                    "symmetry/Wyckoff classification changes inside the frozen tolerance window"
                )
        international_number = int(_dataset_field(detected, "number"))
        try:
            pinned_hall = self.pinned_hall_numbers[international_number]
        except KeyError as error:
            raise CatalogueError(
                f"space-group type {international_number} has no pinned Hall setting"
            ) from error
        dataset = self._dataset(spglib, cell, hall_number=pinned_hall)
        if int(_dataset_field(dataset, "number")) != international_number:
            raise InvariantViolation(
                "pinned Hall request changed the detected space-group type"
            )
        if int(_dataset_field(dataset, "hall_number")) != pinned_hall:
            raise InvariantViolation("spglib did not honor the pinned Hall setting")
        input_primitive_lattice = np.asarray(
            _dataset_field(dataset, "primitive_lattice"), dtype=np.float64
        )
        if input_primitive_lattice.shape != (3, 3) or not np.all(
            np.isfinite(input_primitive_lattice)
        ):
            raise NumericalError("spglib returned an invalid raw-frame primitive lattice")
        if float(np.linalg.det(input_primitive_lattice)) <= 0.0:
            raise InvariantViolation("spglib returned a left-handed raw-frame primitive cell")
        database_symmetry = spglib.get_symmetry_from_database(pinned_hall)
        if database_symmetry is None:
            raise CatalogueError(f"spglib has no operation table for Hall number {pinned_hall}")
        free_axes = _origin_free_axes(database_symmetry["rotations"])
        database_operations = _exact_database_operations(
            database_symmetry["rotations"], database_symmetry["translations"]
        )
        origin_gauge = _origin_gauge_shifts(database_operations, free_axes)
        conventional = self._pinned_conventional(
            dataset,
            free_axes=free_axes,
            origin_gauge=origin_gauge,
        )
        conventional_lattice = np.asarray(
            _dataset_field(dataset, "std_lattice"), dtype=np.float64
        )
        if conventional_lattice.shape != (3, 3) or not np.all(
            np.isfinite(conventional_lattice)
        ):
            raise NumericalError("spglib returned an invalid standardized lattice")
        if float(np.linalg.det(conventional_lattice)) <= 0.0:
            raise InvariantViolation(
                "spglib returned a left-handed conventional cell; refusing a mirror repair"
            )

        primitive_cell = spglib.find_primitive(
            (
                conventional_lattice,
                conventional.positions,
                conventional.encoded_types,
            ),
            symprec=self.policy.symprec,
            angle_tolerance=self.policy.angle_tolerance,
        )
        if primitive_cell is None:
            raise NumericalError("spglib failed to reduce the pinned conventional cell")
        primitive_lattice = np.asarray(primitive_cell[0], dtype=np.float64)
        primitive = _canonicalize_float_origin(
            primitive_cell[1],
            primitive_cell[2],
            (0, 1, 2),
            decimals=self.policy.canonical_decimals,
        )
        if primitive_lattice.shape != (3, 3) or not np.all(np.isfinite(primitive_lattice)):
            raise NumericalError("spglib returned an invalid primitive lattice")
        if float(np.linalg.det(primitive_lattice)) <= 0.0:
            raise InvariantViolation(
                "spglib returned a left-handed primitive cell; refusing a mirror repair"
            )

        input_count = len(structure.species)
        primitive_count = len(primitive.encoded_types)
        if input_count % primitive_count:
            raise InvariantViolation("primitive atom count does not divide the raw atom count")
        raw_index = input_count // primitive_count
        rotations = np.asarray(_dataset_field(dataset, "rotations"), dtype=np.int64)
        translations = np.asarray(_dataset_field(dataset, "translations"), dtype=np.float64)
        identity_mask = np.all(rotations == np.eye(3, dtype=np.int64)[None, :, :], axis=(1, 2))
        pure_translations = np.mod(translations[identity_mask], 1.0)
        translation_certificate = certify_floating_translation_subgroup(
            structure,
            pure_translations,
            raw_index,
            cartesian_tolerance=self.policy.symprec,
        )
        exact_raw_translations = translation_certificate.translation_subgroup
        raw_translation_subgroup = np.asarray(
            [[float(value) for value in row] for row in exact_raw_translations],
            dtype=np.float64,
        )
        if len(raw_translation_subgroup) != raw_index:
            raise InvariantViolation(
                "raw pure-translation subgroup disagrees with the primitive atom-count index"
            )

        # Rotate the returned standardized-frame primitive back into the raw
        # Cartesian frame.  Only in that common frame can the relation to the
        # raw cell be an exact rational fractional-basis transformation.
        standard_to_raw_rotation = np.asarray(
            _dataset_field(dataset, "std_rotation_matrix"), dtype=np.float64
        )
        rebasis_tolerance = max(self.policy.roundtrip_tolerance, 1.0e-10)
        if (
            standard_to_raw_rotation.shape != (3, 3)
            or not np.all(np.isfinite(standard_to_raw_rotation))
            or not np.allclose(
                standard_to_raw_rotation @ standard_to_raw_rotation.T,
                np.eye(3),
                atol=rebasis_tolerance,
                rtol=rebasis_tolerance,
            )
            or not np.isclose(
                np.linalg.det(standard_to_raw_rotation),
                1.0,
                atol=rebasis_tolerance,
                rtol=rebasis_tolerance,
            )
        ):
            raise InvariantViolation("spglib standard-frame map is not a proper rotation")
        raw_frame_to_standard_rotation = standard_to_raw_rotation.T
        raw_frame_primitive_target = primitive_lattice @ standard_to_raw_rotation

        # Recover basis topology from the pre-idealization transformation and
        # the idealized conventional/primitive embedding. Directly rounding
        # L_prim R L_w^-1 would wrongly treat idealization strain as a failure
        # of the exact integer lattice certificate.
        witness_basis = translation_certificate.basis_change
        conventional_count = len(conventional.encoded_types)
        if conventional_count % primitive_count:
            raise InvariantViolation("primitive atom count does not divide conventional count")
        primitive_rebasis = _primitive_rebasis_before_idealization(
            np.asarray(_dataset_field(dataset, "transformation_matrix"), dtype=np.float64),
            translation_certificate.inverse_basis_change,
            conventional_lattice,
            primitive_lattice,
            centering_index=conventional_count // primitive_count,
            tolerance=rebasis_tolerance,
            typed_idealization_rejection=(
                self.policy.idealization_tolerance_angstrom is not None
            ),
        )
        primitive_rebasis_inverse = unimodular_inverse(primitive_rebasis)
        exact_input_to_raw_frame_primitive_basis = exact_matmul(
            primitive_rebasis, witness_basis
        )
        exact_inverse_fraction = exact_matmul(
            translation_certificate.inverse_basis_change,
            primitive_rebasis_inverse,
        )
        if any(value.denominator != 1 for row in exact_inverse_fraction for value in row):
            raise InvariantViolation(
                "exact input-to-raw-frame-primitive inverse became nonintegral"
            )
        exact_inverse_input_to_raw_frame_primitive_basis: IntegerMatrix = tuple(
            tuple(int(value) for value in row) for row in exact_inverse_fraction
        )
        input_to_raw_frame_primitive_basis = np.asarray(
            [
                [float(value) for value in row]
                for row in exact_input_to_raw_frame_primitive_basis
            ],
            dtype=np.float64,
        )
        raw_frame_primitive_lattice = input_to_raw_frame_primitive_basis @ structure.lattice

        # A malformed raw-frame certificate must still stop the run, even if
        # the input also needs too much idealization. Check this before the
        # recoverable numerical-input boundary below.
        backend_rebasis = np.linalg.solve(
            input_primitive_lattice.T, raw_frame_primitive_lattice.T
        ).T
        _integer_basis_witness(
            backend_rebasis,
            determinant=1,
            tolerance=rebasis_tolerance,
            name="returned-to-backend raw-frame primitive rebasing",
        )
        input_mapping = tuple(
            int(value) for value in _dataset_field(dataset, "mapping_to_primitive")
        )
        if len(input_mapping) != input_count or len(set(input_mapping)) != primitive_count:
            raise InvariantViolation("spglib returned an incomplete input-to-primitive map")
        for primitive_index in set(input_mapping):
            types_for_fibre = {
                encoded_types[index]
                for index, mapped in enumerate(input_mapping)
                if mapped == primitive_index
            }
            if len(types_for_fibre) != 1:
                raise InvariantViolation("primitive map merged different chemical species")

        standard_mapping = tuple(
            int(value) for value in _dataset_field(dataset, "std_mapping_to_primitive")
        )
        if len(standard_mapping) != len(conventional.encoded_types):
            raise InvariantViolation("standardized primitive mapping has the wrong length")
        if set(standard_mapping) != set(input_mapping):
            raise InvariantViolation(
                "raw and standardized cells disagree on primitive-site identifiers"
            )

        # Reindex spglib's raw primitive identifiers onto the *returned*,
        # origin-quotiented and sorted primitive motif.  The conventional cell
        # and find_primitive output share one Cartesian frame, so this matching
        # is only a torus-coordinate comparison, not a structure matcher.
        canonical_standard_mapping = tuple(
            standard_mapping[index] for index in conventional.permutation
        )
        primitive_input_positions = np.mod(np.asarray(primitive_cell[1], dtype=np.float64), 1.0)
        primitive_input_types = tuple(int(value) for value in primitive_cell[2])
        primitive_lattice_inverse = np.linalg.inv(primitive_lattice)
        primitive_id_to_input_index: dict[int, int] = {}
        matching_tolerance = max(self.policy.roundtrip_tolerance, 1.0e-10)
        for point, atom_type, primitive_identifier in zip(
            conventional.positions,
            conventional.encoded_types,
            canonical_standard_mapping,
            strict=True,
        ):
            primitive_coordinate = np.mod(
                point @ conventional_lattice @ primitive_lattice_inverse,
                1.0,
            )
            candidates: list[int] = []
            for index, (candidate, candidate_type) in enumerate(
                zip(primitive_input_positions, primitive_input_types, strict=True)
            ):
                if candidate_type != atom_type:
                    continue
                delta = primitive_coordinate - candidate
                delta -= np.rint(delta)
                if float(np.max(np.abs(delta))) <= matching_tolerance:
                    candidates.append(index)
            if len(candidates) != 1:
                raise InvariantViolation(
                    "could not map a conventional site uniquely into the primitive motif"
                )
            previous_primitive_index = primitive_id_to_input_index.setdefault(
                primitive_identifier, candidates[0]
            )
            if previous_primitive_index != candidates[0]:
                raise InvariantViolation(
                    "one primitive identifier maps to inconsistent primitive sites"
                )
        if set(primitive_id_to_input_index) != set(input_mapping):
            raise InvariantViolation("primitive-site reindexing is incomplete")
        input_to_output_index = {
            input_index: output_index
            for output_index, input_index in enumerate(primitive.permutation)
        }
        canonical_input_mapping = tuple(
            input_to_output_index[primitive_id_to_input_index[identifier]]
            for identifier in input_mapping
        )
        wyckoffs, sites, canonical_orbits = self._metadata_in_final_origin(
            spglib,
            international_number=international_number,
            hall_number=pinned_hall,
            lattice=conventional_lattice,
            motif=conventional,
        )

        # Independent primitive -> pinned-conventional round trip.  This catches
        # missed translations, setting drift, and origin handling errors.
        primitive_dataset = self._dataset(
            spglib,
            (primitive_lattice, primitive.positions, primitive.encoded_types),
            hall_number=pinned_hall,
        )
        if int(_dataset_field(primitive_dataset, "number")) != international_number:
            raise InvariantViolation("primitive round trip changed the space-group type")
        roundtrip = self._pinned_conventional(
            primitive_dataset,
            free_axes=free_axes,
            origin_gauge=origin_gauge,
        )
        roundtrip_lattice = np.asarray(
            _dataset_field(primitive_dataset, "std_lattice"), dtype=np.float64
        )
        tolerance = self.policy.roundtrip_tolerance
        _verify_pinned_roundtrip(
            conventional, roundtrip,
            conventional_lattice=conventional_lattice,
            roundtrip_lattice=roundtrip_lattice,
            primitive_lattice=primitive_lattice,
            primitive_shift=primitive.shift,
            transformation=np.asarray(
                _dataset_field(primitive_dataset, "transformation_matrix"), dtype=np.float64
            ),
            origin_shift=np.asarray(
                _dataset_field(primitive_dataset, "origin_shift"), dtype=np.float64
            ),
            tolerance=tolerance,
            typed_motif_rejection=self.policy.idealization_tolerance_angstrom is not None,
        )

        # Check backend topology, atom maps and its own idealized-cell round
        # trip before classifying input fidelity as a recoverable rejection.
        # These are the existing mixed absolute/relative predicates, not a new
        # absolute Cartesian norm bound and not an increased tolerance.
        _require_lattice_fidelity(
            raw_frame_primitive_lattice,
            raw_frame_primitive_target,
            tolerance_angstrom=self.policy.idealization_tolerance_angstrom,
            legacy_tolerance=rebasis_tolerance,
            error=AmbiguousPreprocessingError,
            legacy_message=(
                "reported idealized lattice exceeds the frozen raw-frame input-fidelity "
                "tolerance (unchanged componentwise absolute/relative checks)"
            ),
            message="reported idealized lattice exceeds the raw-frame idealization bound",
        )
        _require_lattice_fidelity(
            raw_frame_primitive_lattice @ raw_frame_to_standard_rotation,
            primitive_lattice,
            tolerance_angstrom=self.policy.idealization_tolerance_angstrom,
            legacy_tolerance=rebasis_tolerance,
            error=AmbiguousPreprocessingError,
            legacy_message=(
                "reported idealized lattice exceeds the frozen standardized-frame "
                "primitive-lattice tolerance (unchanged absolute/relative checks)"
            ),
            message="reported idealized lattice exceeds the standardized-frame bound",
        )

        conventional_species = self._decode_species(conventional.encoded_types, decode)
        primitive_species = self._decode_species(primitive.encoded_types, decode)
        provenance = StandardizationProvenance(
            backend="spglib",
            backend_version=self.policy.required_spglib_version,
            hall_map_fingerprint=self.hall_map_fingerprint,
            hall_number=pinned_hall,
            symprec=self.policy.symprec,
            angle_tolerance=self.policy.angle_tolerance,
            canonical_decimals=self.policy.canonical_decimals,
            input_atom_count=input_count,
            primitive_atom_count=primitive_count,
            conventional_atom_count=len(conventional_species),
            stability_symprec_factors=self.policy.stability_symprec_factors,
            symmetry_stability_verified=True,
            algorithm=self.algorithm_id,
            roundtrip_tolerance=self.policy.roundtrip_tolerance,
            idealization_tolerance_angstrom=self.policy.idealization_tolerance_angstrom,
            relative_metric_quantum=self.policy.relative_metric_quantum,
        )
        return CanonicalizedStructure(
            international_number=international_number,
            hall_number=pinned_hall,
            hall_symbol=str(_dataset_field(dataset, "hall")),
            international_symbol=str(_dataset_field(dataset, "international")),
            setting_choice=str(_dataset_field(dataset, "choice")),
            conventional_lattice=conventional_lattice,
            conventional_fractional_positions=conventional.positions,
            conventional_species=conventional_species,
            wyckoff_letters=wyckoffs,
            site_symmetry_symbols=sites,
            crystallographic_orbits=canonical_orbits,
            primitive_lattice=primitive_lattice,
            primitive_fractional_positions=primitive.positions,
            primitive_species=primitive_species,
            raw_translation_subgroup=raw_translation_subgroup,
            input_mapping_to_primitive=canonical_input_mapping,
            transformation_matrix=np.asarray(
                _dataset_field(dataset, "transformation_matrix"), dtype=np.float64
            ),
            spglib_origin_shift=np.asarray(
                _dataset_field(dataset, "origin_shift"), dtype=np.float64
            ),
            free_origin_axes=free_axes,
            subtracted_free_origin=conventional.shift,
            provenance=provenance,
            input_to_raw_frame_primitive_basis=input_to_raw_frame_primitive_basis,
            exact_raw_translation_subgroup=exact_raw_translations,
            exact_input_to_raw_frame_primitive_basis=(exact_input_to_raw_frame_primitive_basis),
            exact_inverse_input_to_raw_frame_primitive_basis=(
                exact_inverse_input_to_raw_frame_primitive_basis
            ),
            exact_translation_max_cartesian_error=max(
                translation_certificate.reported_max_cartesian_error,
                translation_certificate.action_max_cartesian_error,
            ),
            raw_frame_primitive_lattice=raw_frame_primitive_lattice,
            raw_frame_to_standard_rotation=raw_frame_to_standard_rotation,
        )


__all__ = [
    "CanonicalizedStructure",
    "CertifiedFloatingTranslationSubgroup",
    "ExactOriginCanonicalization",
    "ExactPrimitiveCell",
    "ExactTranslationLattice",
    "FloatingStructure",
    "SpglibPolicy",
    "SpglibStandardizer",
    "StandardizationProvenance",
    "canonicalize_exact_origin",
    "certify_floating_translation_subgroup",
    "exact_continuous_origin_axes",
    "exact_full_primitive_reduction",
    "exact_primitive_basis_from_translations",
    "validate_origin_chart_compatibility",
]
