"""Exact intrinsic Wyckoff charts and fixed-cell tangent mathematics."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from fractions import Fraction

import numpy as np

from .algebra import (
    AffineOperation,
    FractionMatrix,
    FractionVector,
    as_fraction_matrix,
    as_fraction_vector,
    exact_determinant,
    exact_matmul,
    exact_rank,
    exact_row_matmul,
    fraction_mod_one,
    gcd_maximal_minors,
)
from .errors import DomainError, InvariantViolation, NumericalError


@dataclass(frozen=True, slots=True)
class WyckoffPosition:
    """One explicit Wyckoff row in a pinned Hall setting.

    ``representative(xi) = offset + xi @ coordinate_matrix`` uses row-vector
    coordinates.  ``period_basis`` is a row basis of the full intrinsic period
    lattice, not merely a collection of known periods.
    """

    label: str
    multiplicity: int
    representative_offset: FractionVector
    coordinate_matrix: FractionMatrix
    period_basis: FractionMatrix
    coset_representatives: tuple[AffineOperation, ...]
    site_symmetry_order: int
    generic_parameter: FractionVector

    def __post_init__(self) -> None:
        if not self.label:
            raise DomainError("a Wyckoff label cannot be empty")
        if self.multiplicity <= 0:
            raise DomainError("Wyckoff multiplicity must be positive")
        if self.site_symmetry_order <= 0:
            raise DomainError("site-symmetry order must be positive")
        offset = as_fraction_vector(self.representative_offset, length=3)
        coordinate_matrix = as_fraction_matrix(self.coordinate_matrix, ncols=3)
        dimension = len(coordinate_matrix)
        if exact_rank(coordinate_matrix, ncols=3) != dimension:
            raise DomainError("the intrinsic coordinate matrix must have full row rank")
        period_basis = as_fraction_matrix(self.period_basis, ncols=dimension)
        if len(period_basis) != dimension:
            raise DomainError("the intrinsic period basis must be square and full rank")
        if dimension and exact_determinant(period_basis) == 0:
            raise DomainError("the intrinsic period basis is singular")
        generic_parameter = as_fraction_vector(self.generic_parameter, length=dimension)
        cosets = tuple(self.coset_representatives)
        if len(cosets) != self.multiplicity:
            raise DomainError("the number of coset representatives must equal multiplicity")
        if len(set(cosets)) != len(cosets):
            raise DomainError("Wyckoff coset representative list contains duplicates")

        # A stored B is the complete Lambda={lambda: lambda A in Z^3} exactly
        # iff M=B A is integral and its row lattice is primitive in Z^3.
        if dimension:
            image = exact_matmul(period_basis, coordinate_matrix)
            if any(value.denominator != 1 for row in image for value in row):
                raise DomainError("a declared intrinsic period does not decode to Z^3")
            integer_image = tuple(tuple(int(value) for value in row) for row in image)
            if gcd_maximal_minors(integer_image) != 1:
                raise DomainError(
                    "the period basis is a strict sublattice of the full "
                    "intrinsic period lattice"
                )

        object.__setattr__(self, "representative_offset", offset)
        object.__setattr__(self, "coordinate_matrix", coordinate_matrix)
        object.__setattr__(self, "period_basis", period_basis)
        object.__setattr__(self, "coset_representatives", cosets)
        object.__setattr__(self, "generic_parameter", generic_parameter)

    @property
    def dimension(self) -> int:
        return len(self.coordinate_matrix)

    def representative_exact(self, parameter: Sequence[Fraction | int]) -> FractionVector:
        xi = as_fraction_vector(parameter, length=self.dimension)
        if self.dimension:
            displacement = exact_row_matmul(xi, self.coordinate_matrix)
        else:
            displacement = (Fraction(0), Fraction(0), Fraction(0))
        return tuple(
            fraction_mod_one(self.representative_offset[j] + displacement[j])
            for j in range(3)
        )

    def expand_exact(
        self,
        parameter: Sequence[Fraction | int],
        *,
        require_declared_multiplicity: bool = True,
    ) -> tuple[FractionVector, ...]:
        representative = self.representative_exact(parameter)
        expanded = tuple(
            operation.apply_exact(representative)
            for operation in self.coset_representatives
        )
        if require_declared_multiplicity and len(set(expanded)) != self.multiplicity:
            raise InvariantViolation(
                f"parameter specializes Wyckoff row {self.label} below multiplicity "
                f"{self.multiplicity}"
            )
        return expanded

    def expand_float(
        self, parameter: Sequence[float], *, require_declared_multiplicity: bool = False
    ) -> np.ndarray:
        xi = np.asarray(parameter, dtype=np.float64)
        if xi.shape != (self.dimension,) or not np.all(np.isfinite(xi)):
            raise DomainError(f"Wyckoff parameter for {self.label} has the wrong shape")
        offset = np.asarray([float(value) for value in self.representative_offset])
        if self.dimension:
            coordinate_matrix = np.asarray(self.coordinate_matrix, dtype=np.float64)
            representative = np.mod(offset + xi @ coordinate_matrix, 1.0)
        else:
            representative = offset
        expanded = np.stack(
            [operation.apply_float(representative) for operation in self.coset_representatives]
        )
        if require_declared_multiplicity:
            for i in range(self.multiplicity):
                for j in range(i):
                    delta = expanded[i] - expanded[j]
                    delta -= np.rint(delta)
                    if float(np.linalg.norm(delta)) <= 1.0e-12:
                        raise InvariantViolation(
                            f"parameter specializes Wyckoff row {self.label} "
                            "below its multiplicity"
                        )
        return expanded

    def pre_origin_pullback(self, metric: np.ndarray, *, atol: float = 1.0e-10) -> np.ndarray:
        """Return ``mu A G A^T`` from Proposition 2.10.

        The equality is refused unless every stored coset rotation preserves
        the supplied metric to the declared numerical tolerance.
        """

        g = _validated_metric(metric)
        for operation in self.coset_representatives:
            rotation = np.asarray(operation.rotation, dtype=np.float64)
            if not np.allclose(rotation @ g @ rotation.T, g, atol=atol, rtol=atol):
                raise InvariantViolation(
                    f"metric is not invariant under a coset rotation for row {self.label}"
                )
        a = np.asarray(self.coordinate_matrix, dtype=np.float64).reshape(self.dimension, 3)
        return self.multiplicity * (a @ g @ a.T)

    def direct_pre_origin_pullback(self, metric: np.ndarray) -> np.ndarray:
        """Direct coset sum used as an independent check of Proposition 2.10."""

        g = _validated_metric(metric)
        a = np.asarray(self.coordinate_matrix, dtype=np.float64).reshape(self.dimension, 3)
        result = np.zeros((self.dimension, self.dimension), dtype=np.float64)
        for operation in self.coset_representatives:
            rotation = np.asarray(operation.rotation, dtype=np.float64)
            result += a @ rotation @ g @ rotation.T @ a.T
        return result


def _validated_metric(metric: np.ndarray) -> np.ndarray:
    g = np.asarray(metric, dtype=np.float64)
    if g.shape != (3, 3) or not np.all(np.isfinite(g)):
        raise DomainError("metric must be a finite 3x3 matrix")
    if not np.allclose(g, g.T, atol=1.0e-12, rtol=1.0e-12):
        raise DomainError("metric must be symmetric")
    if float(np.min(np.linalg.eigvalsh(g))) <= 0.0:
        raise DomainError("metric must be positive definite")
    return (g + g.T) / 2.0


def stacked_fractional_jacobian(
    positions: Sequence[WyckoffPosition], parameter_maps: Sequence[np.ndarray]
) -> np.ndarray:
    """Build the advertised-slot Jacobian after an exact origin gauge choice.

    For orbit ``i``, ``parameter_maps[i]`` has shape ``(d_eta, d_i)`` and
    represents ``delta xi_i = delta eta @ C_i``.  Coordinates are column-stacked
    in the returned array, whose shape is ``(3 N, d_eta)``.
    """

    if len(positions) != len(parameter_maps):
        raise DomainError("one intrinsic reconstruction map is required per orbit")
    if not positions:
        raise DomainError("at least one advertised orbit is required")
    maps = [np.asarray(mapping, dtype=np.float64) for mapping in parameter_maps]
    total_dimension = maps[0].shape[0] if maps[0].ndim == 2 else -1
    blocks: list[np.ndarray] = []
    for position, mapping in zip(positions, maps, strict=True):
        if mapping.shape != (total_dimension, position.dimension):
            raise DomainError("an intrinsic reconstruction map has the wrong shape")
        if not np.all(np.isfinite(mapping)):
            raise DomainError("intrinsic reconstruction maps must be finite")
        a = np.asarray(position.coordinate_matrix, dtype=np.float64).reshape(
            position.dimension, 3
        )
        for operation in position.coset_representatives:
            rotation = np.asarray(operation.rotation, dtype=np.float64)
            blocks.append((mapping @ a @ rotation).T)
    return np.concatenate(blocks, axis=0)


def stacked_cartesian_jacobian(
    positions: Sequence[WyckoffPosition],
    parameter_maps: Sequence[np.ndarray],
    lattice: np.ndarray,
) -> np.ndarray:
    """Map gauge-reduced intrinsic velocities to row-convention Cartesian slots."""

    lattice_matrix = np.asarray(lattice, dtype=np.float64)
    if (
        lattice_matrix.shape != (3, 3)
        or not np.all(np.isfinite(lattice_matrix))
        or np.linalg.det(lattice_matrix) <= 0.0
    ):
        raise DomainError("lattice must be a finite right-handed invertible 3x3 matrix")
    fractional = stacked_fractional_jacobian(positions, parameter_maps)
    number_atoms = fractional.shape[0] // 3
    dimension = fractional.shape[1]
    blocks = fractional.reshape(number_atoms, 3, dimension)
    # For row coordinates, delta r^T = L^T delta f^T.
    cartesian = np.einsum("ij,ajk->aik", lattice_matrix.T, blocks)
    return cartesian.reshape(3 * number_atoms, dimension)


def translation_projector(number_atoms: int) -> np.ndarray:
    if number_atoms <= 0:
        raise DomainError("translation quotient requires at least one atom")
    atom_projector = np.eye(number_atoms) - np.ones((number_atoms, number_atoms)) / number_atoms
    return np.kron(atom_projector, np.eye(3))


def center_stacked_cartesian(array: np.ndarray) -> np.ndarray:
    """Apply ``P_N`` without materializing the Kronecker matrix."""

    value = np.asarray(array, dtype=np.float64)
    if value.ndim not in (1, 2) or value.shape[0] == 0 or value.shape[0] % 3:
        raise DomainError("Cartesian data must have leading dimension 3N")
    if not np.all(np.isfinite(value)):
        raise DomainError("Cartesian data must be finite")
    number_atoms = value.shape[0] // 3
    if value.ndim == 1:
        reshaped = value.reshape(number_atoms, 3)
        return (reshaped - reshaped.mean(axis=0, keepdims=True)).reshape(-1)
    reshaped = value.reshape(number_atoms, 3, value.shape[1])
    return (reshaped - reshaped.mean(axis=0, keepdims=True)).reshape(value.shape)


def translation_quotient_pullback(cartesian_jacobian: np.ndarray) -> np.ndarray:
    """Return the exact fixed-cell pullback ``J^T P_N J``."""

    jacobian = np.asarray(cartesian_jacobian, dtype=np.float64)
    if jacobian.ndim != 2:
        raise DomainError("Cartesian Jacobian must be a matrix")
    centered = center_stacked_cartesian(jacobian)
    return centered.T @ centered


def translation_quotient_adjoint(
    cartesian_jacobian: np.ndarray,
    dimensionless_cartesian_field: np.ndarray,
    *,
    reference_length: float = 1.0,
    rank_rtol: float = 1.0e-12,
) -> np.ndarray:
    """Reduce expanded vectors with the ridge-free metric adjoint (3.9).

    ``dimensionless_cartesian_field`` is ``s_hat`` and the physical Jacobian is
    divided by ``reference_length`` before solving.  Rank failure is an invalid
    origin-gauge certificate and raises; it is never hidden with a ridge.
    """

    jacobian = np.asarray(cartesian_jacobian, dtype=np.float64)
    field = np.asarray(dimensionless_cartesian_field, dtype=np.float64)
    if jacobian.ndim != 2 or field.shape != (jacobian.shape[0],):
        raise DomainError("field and Cartesian Jacobian dimensions are inconsistent")
    if not np.all(np.isfinite(jacobian)) or not np.all(np.isfinite(field)):
        raise DomainError("adjoint inputs must be finite")
    if not np.isfinite(reference_length) or reference_length <= 0.0:
        raise DomainError("reference length must be finite and positive")
    dimension = jacobian.shape[1]
    if dimension == 0:
        return np.empty((0,), dtype=np.float64)
    j_hat = jacobian / reference_length
    centered_j = center_stacked_cartesian(j_hat)
    centered_field = center_stacked_cartesian(field)
    normal = centered_j.T @ centered_j
    eigenvalues = np.linalg.eigvalsh((normal + normal.T) / 2.0)
    scale = float(eigenvalues[-1])
    if scale <= 0.0 or float(eigenvalues[0]) <= rank_rtol * scale:
        raise InvariantViolation(
            "translation-quotient Jacobian is rank deficient; origin gauge is uncertified"
        )
    try:
        return np.linalg.solve(normal, centered_j.T @ centered_field)
    except np.linalg.LinAlgError as error:
        raise NumericalError("translation-quotient adjoint solve failed") from error
