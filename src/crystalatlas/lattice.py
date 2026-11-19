"""Complete whitened charts for point-group-invariant SPD lattice metrics."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from .catalogue import SpaceGroupSetting
from .errors import DomainError, InvariantViolation, NumericalError


def _symmetric_basis_3() -> tuple[np.ndarray, ...]:
    """Canonical Frobenius-orthonormal basis of ``Sym(3)``."""

    basis: list[np.ndarray] = []
    for index in range(3):
        matrix = np.zeros((3, 3), dtype=np.float64)
        matrix[index, index] = 1.0
        basis.append(matrix)
    scale = 1.0 / np.sqrt(2.0)
    for i, j in ((0, 1), (0, 2), (1, 2)):
        matrix = np.zeros((3, 3), dtype=np.float64)
        matrix[i, j] = scale
        matrix[j, i] = scale
        basis.append(matrix)
    return tuple(basis)


def symmetric_matrix_exp(matrix: np.ndarray) -> np.ndarray:
    """Principal exponential of a finite real symmetric matrix."""

    value = np.asarray(matrix, dtype=np.float64)
    if value.shape != (3, 3) or not np.all(np.isfinite(value)):
        raise DomainError("symmetric exponential requires a finite 3x3 matrix")
    if not np.allclose(value, value.T, atol=1.0e-12, rtol=1.0e-12):
        raise DomainError("matrix exponential input must be symmetric")
    eigenvalues, eigenvectors = np.linalg.eigh((value + value.T) / 2.0)
    if float(np.max(eigenvalues)) > np.log(np.finfo(np.float64).max) - 2.0:
        raise NumericalError("symmetric matrix exponential would overflow float64")
    result = (eigenvectors * np.exp(eigenvalues)) @ eigenvectors.T
    return (result + result.T) / 2.0


def spd_matrix_log(metric: np.ndarray) -> np.ndarray:
    """Principal symmetric logarithm of an SPD matrix."""

    value = _validate_spd(metric)
    eigenvalues, eigenvectors = np.linalg.eigh(value)
    if float(eigenvalues[0]) <= 0.0:
        raise DomainError("matrix logarithm requires strict positive definiteness")
    result = (eigenvectors * np.log(eigenvalues)) @ eigenvectors.T
    return (result + result.T) / 2.0


def spd_matrix_sqrt(metric: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return the symmetric positive square root and its inverse."""

    value = _validate_spd(metric)
    eigenvalues, eigenvectors = np.linalg.eigh(value)
    root_values = np.sqrt(eigenvalues)
    root = (eigenvectors * root_values) @ eigenvectors.T
    inverse = (eigenvectors * (1.0 / root_values)) @ eigenvectors.T
    return (root + root.T) / 2.0, (inverse + inverse.T) / 2.0


def _validate_spd(metric: np.ndarray) -> np.ndarray:
    value = np.asarray(metric, dtype=np.float64)
    if value.shape != (3, 3) or not np.all(np.isfinite(value)):
        raise DomainError("metric must be a finite 3x3 matrix")
    if not np.allclose(value, value.T, atol=1.0e-12, rtol=1.0e-12):
        raise DomainError("metric must be symmetric")
    value = (value + value.T) / 2.0
    try:
        eigenvalues = np.linalg.eigvalsh(value)
    except np.linalg.LinAlgError as error:
        raise NumericalError("SPD eigenvalue calculation failed") from error
    if float(eigenvalues[0]) <= 0.0:
        raise DomainError("metric must be strictly positive definite")
    return value


def invariant_reference_metric(
    rotations: Sequence[np.ndarray], seed_metric: np.ndarray | None = None
) -> np.ndarray:
    """Construct ``G0 = |P|^-1 sum_W W H W^T``."""

    group = _validated_rotations(rotations)
    h = np.eye(3) if seed_metric is None else _validate_spd(seed_metric)
    average = sum((rotation @ h @ rotation.T for rotation in group), np.zeros((3, 3)))
    average /= len(group)
    average = _validate_spd((average + average.T) / 2.0)
    for rotation in group:
        if not np.allclose(
            rotation @ average @ rotation.T,
            average,
            atol=1.0e-10,
            rtol=1.0e-10,
        ):
            raise InvariantViolation(
                "group average did not produce an invariant reference metric"
            )
    return average


def scale_metric_to_volume(metric: np.ndarray, target_volume: float) -> np.ndarray:
    """Scale an SPD metric to a prescribed right-handed cell volume.

    For the row-lattice convention ``G = L L^T``, the cell volume is
    ``det(L) = sqrt(det(G))``.  Multiplying ``G`` by ``a`` multiplies this
    volume by ``a^(3/2)``.  The scalar rescaling preserves every point-group
    invariance of the input metric.
    """

    value = _validate_spd(metric)
    if not np.isfinite(target_volume) or target_volume <= 0.0:
        raise DomainError("target_volume must be finite and positive")
    current_volume = float(np.sqrt(np.linalg.det(value)))
    scale = (float(target_volume) / current_volume) ** (2.0 / 3.0)
    result = value * scale
    return _validate_spd((result + result.T) / 2.0)


def _validated_rotations(rotations: Sequence[np.ndarray]) -> tuple[np.ndarray, ...]:
    if not rotations:
        raise DomainError("an invariant metric chart requires a nonempty point group")
    converted: list[np.ndarray] = []
    for rotation in rotations:
        value = np.asarray(rotation, dtype=np.float64)
        if value.shape != (3, 3) or not np.all(np.isfinite(value)):
            raise DomainError("point-group rotations must be finite 3x3 matrices")
        determinant = float(np.linalg.det(value))
        if not np.isclose(abs(determinant), 1.0, atol=1.0e-10, rtol=1.0e-10):
            raise DomainError("point-group rotation is not unimodular")
        if not any(
            np.allclose(value, existing, atol=1.0e-10, rtol=1.0e-10)
            for existing in converted
        ):
            converted.append(value)
    group = tuple(converted)
    identity = np.eye(3)
    if not any(np.allclose(value, identity, atol=1.0e-10, rtol=1.0e-10) for value in group):
        raise DomainError("point-group table has no identity")
    for left in group:
        try:
            inverse = np.linalg.inv(left)
        except np.linalg.LinAlgError as error:
            raise DomainError("point-group operation is singular") from error
        if not any(
            np.allclose(inverse, candidate, atol=1.0e-9, rtol=1.0e-9)
            for candidate in group
        ):
            raise DomainError("point-group table is not closed under inversion")
        for right in group:
            product = left @ right
            if not any(
                np.allclose(product, candidate, atol=1.0e-9, rtol=1.0e-9)
                for candidate in group
            ):
                raise DomainError("point-group table is not closed under composition")
    return group


def symmetric_invariant_basis(
    whitened_rotations: Sequence[np.ndarray], *, tolerance: float = 1.0e-10
) -> tuple[np.ndarray, ...]:
    """Compute a Frobenius-orthonormal basis of the symmetric commutant."""

    rotations = _validated_rotations(whitened_rotations)
    canonical = _symmetric_basis_3()
    constraints = np.concatenate(
        [
            np.stack(
                [(rotation @ basis @ rotation.T - basis).reshape(-1) for basis in canonical],
                axis=1,
            )
            for rotation in rotations
        ],
        axis=0,
    )
    if float(np.linalg.norm(constraints, ord=2)) <= tolerance:
        return canonical
    try:
        _, singular_values, vh = np.linalg.svd(constraints, full_matrices=True)
    except np.linalg.LinAlgError as error:
        raise NumericalError("symmetric-commutant SVD failed") from error
    scale = max(float(singular_values[0]), 1.0)
    rank = int(np.count_nonzero(singular_values > tolerance * scale))
    null_coefficients = vh[rank:, :]
    if null_coefficients.shape[0] == 0:
        raise InvariantViolation("point group has no invariant symmetric metric direction")
    basis_matrices: list[np.ndarray] = []
    for coefficients in null_coefficients:
        matrix = sum(
            (
                coefficient * basis
                for coefficient, basis in zip(coefficients, canonical, strict=True)
            ),
            np.zeros((3, 3), dtype=np.float64),
        )
        matrix = (matrix + matrix.T) / 2.0
        # Fix the otherwise arbitrary SVD sign for reproducible serialization.
        nonzero = np.flatnonzero(np.abs(matrix.reshape(-1)) > tolerance)
        if nonzero.size and matrix.reshape(-1)[nonzero[0]] < 0.0:
            matrix *= -1.0
        basis_matrices.append(matrix)
    for i, basis in enumerate(basis_matrices):
        for rotation in rotations:
            if not np.allclose(
                rotation @ basis @ rotation.T,
                basis,
                atol=tolerance,
                rtol=tolerance,
            ):
                raise InvariantViolation("computed basis leaves the symmetric commutant")
        for j, other in enumerate(basis_matrices):
            expected = float(i == j)
            if not np.isclose(np.sum(basis * other), expected, atol=tolerance, rtol=tolerance):
                raise InvariantViolation("computed commutant basis is not orthonormal")
    return tuple(basis_matrices)


class InvariantSPDChart:
    """The bijection ``b <-> S exp(sum_j b_j E_j) S`` from Theorem 2.3."""

    def __init__(
        self,
        rotations: Sequence[np.ndarray],
        *,
        reference_metric: np.ndarray | None = None,
        bound: float | None = None,
        tolerance: float = 1.0e-10,
    ) -> None:
        if tolerance <= 0.0 or not np.isfinite(tolerance):
            raise DomainError("chart tolerance must be finite and positive")
        self.rotations = _validated_rotations(rotations)
        self.tolerance = float(tolerance)
        if reference_metric is None:
            self.reference_metric = invariant_reference_metric(self.rotations)
        else:
            self.reference_metric = _validate_spd(reference_metric)
        for rotation in self.rotations:
            if not np.allclose(
                rotation @ self.reference_metric @ rotation.T,
                self.reference_metric,
                atol=self.tolerance,
                rtol=self.tolerance,
            ):
                raise DomainError("reference metric is not invariant under the point group")
        self.sqrt_reference, self.inverse_sqrt_reference = spd_matrix_sqrt(
            self.reference_metric
        )
        self.whitened_rotations = tuple(
            self.inverse_sqrt_reference @ rotation @ self.sqrt_reference
            for rotation in self.rotations
        )
        for whitened in self.whitened_rotations:
            if not np.allclose(
                whitened @ whitened.T,
                np.eye(3),
                atol=self.tolerance,
                rtol=self.tolerance,
            ):
                raise InvariantViolation("whitened point-group operation is not orthogonal")
        self.basis = symmetric_invariant_basis(
            self.whitened_rotations, tolerance=self.tolerance
        )
        if bound is not None and (not np.isfinite(bound) or bound <= 0.0):
            raise DomainError("bounded chart radius must be finite and positive")
        self.bound = None if bound is None else float(bound)

    @classmethod
    def from_setting(
        cls,
        setting: SpaceGroupSetting,
        *,
        bound: float | None = None,
        tolerance: float = 1.0e-10,
    ) -> InvariantSPDChart:
        rotations = [
            np.asarray(rotation, dtype=np.float64)
            for rotation in setting.point_group_rotations
        ]
        reference = np.asarray(setting.reference_metric, dtype=np.float64)
        return cls(rotations, reference_metric=reference, bound=bound, tolerance=tolerance)

    @property
    def dimension(self) -> int:
        return len(self.basis)

    @property
    def canonical_commutant_basis(self) -> np.ndarray:
        """``C[k, j] = <S_k, E_j>``: the whitened basis in canonical Sym(3) coordinates.

        ``S_k`` is the fixed Frobenius-orthonormal basis of symmetric 3 x 3
        matrices, identical for every setting, so ``C`` (6 x d, orthonormal
        columns) gives every crystal family one shared coordinate frame.
        """

        canonical = _symmetric_basis_3()
        return np.asarray(
            [[float(np.sum(row * matrix)) for matrix in self.basis] for row in canonical],
            dtype=np.float64,
        ).reshape(6, self.dimension)

    @property
    def minimum_lattice_singular_value_bound(self) -> float:
        """Analytic lower bound over this chart's bounded coefficient ball.

        If ``G=S exp(A) S`` and the invariant basis is Frobenius orthonormal,
        ``||A||_2 <= ||A||_F < B``.  Therefore

        ``lambda_min(G) >= lambda_min(G0) exp(-B)``.

        Taking square roots gives a lower bound on ``sigma_min(L)`` for every
        right-handed factor satisfying ``L L^T = G``.  An unbounded chart has
        no positive uniform lower bound and returns zero.
        """

        if self.bound is None:
            return 0.0
        reference_floor = float(np.linalg.eigvalsh(self.reference_metric)[0])
        return float(np.sqrt(reference_floor) * np.exp(-0.5 * self.bound))

    def algebra_matrix(self, coefficients: Sequence[float]) -> np.ndarray:
        b = np.asarray(coefficients, dtype=np.float64)
        if b.shape != (self.dimension,) or not np.all(np.isfinite(b)):
            raise DomainError(f"expected {self.dimension} finite invariant coefficients")
        return sum(
            (coefficient * basis for coefficient, basis in zip(b, self.basis, strict=True)),
            np.zeros((3, 3), dtype=np.float64),
        )

    def decode(self, coefficients: Sequence[float]) -> np.ndarray:
        algebra = self.algebra_matrix(coefficients)
        whitened_metric = symmetric_matrix_exp(algebra)
        metric = self.sqrt_reference @ whitened_metric @ self.sqrt_reference
        metric = _validate_spd((metric + metric.T) / 2.0)
        if not self.is_invariant(metric):
            raise InvariantViolation("decoded SPD metric violates point-group invariance")
        return metric

    def encode(self, metric: np.ndarray) -> np.ndarray:
        value = _validate_spd(metric)
        if not self.is_invariant(value):
            raise DomainError("metric lies outside this point-group-invariant SPD cone")
        whitened = self.inverse_sqrt_reference @ value @ self.inverse_sqrt_reference
        algebra = spd_matrix_log((whitened + whitened.T) / 2.0)
        coefficients = np.asarray([np.sum(algebra * basis) for basis in self.basis])
        reconstructed = self.algebra_matrix(coefficients)
        residual = float(np.linalg.norm(algebra - reconstructed, ord="fro"))
        scale = max(float(np.linalg.norm(algebra, ord="fro")), 1.0)
        if residual > self.tolerance * scale * 10.0:
            raise InvariantViolation("whitened logarithm is outside the symmetric commutant")
        return coefficients

    def is_invariant(self, metric: np.ndarray) -> bool:
        value = np.asarray(metric, dtype=np.float64)
        return value.shape == (3, 3) and all(
            np.allclose(
                rotation @ value @ rotation.T,
                value,
                atol=self.tolerance,
                rtol=self.tolerance,
            )
            for rotation in self.rotations
        )

    def bounded_coefficients(self, latent: Sequence[float]) -> np.ndarray:
        """Apply ``kappa(z)=B z/sqrt(1+||z||^2)``."""

        if self.bound is None:
            raise DomainError("this chart has no declared bounded radius")
        z = np.asarray(latent, dtype=np.float64)
        if z.shape != (self.dimension,) or not np.all(np.isfinite(z)):
            raise DomainError(f"expected {self.dimension} finite latent coordinates")
        return self.bound * z / np.sqrt(1.0 + float(z @ z))

    def unbounded_latent(self, coefficients: Sequence[float]) -> np.ndarray:
        """Invert the bounded radial map exactly up to floating-point arithmetic."""

        if self.bound is None:
            raise DomainError("this chart has no declared bounded radius")
        b = np.asarray(coefficients, dtype=np.float64)
        if b.shape != (self.dimension,) or not np.all(np.isfinite(b)):
            raise DomainError(f"expected {self.dimension} finite bounded coefficients")
        squared_norm = float(b @ b)
        denominator_squared = self.bound * self.bound - squared_norm
        if denominator_squared <= 0.0:
            raise DomainError("coefficient lies outside the open bounded chart ball")
        return b / np.sqrt(denominator_squared)

    def decode_latent(self, latent: Sequence[float]) -> np.ndarray:
        return self.decode(self.bounded_coefficients(latent))

    def encode_latent(self, metric: np.ndarray) -> np.ndarray:
        return self.unbounded_latent(self.encode(metric))

    def cholesky_lattice(self, coefficients: Sequence[float]) -> np.ndarray:
        """Return the smooth right-handed lower-triangular section ``L L^T=G``."""

        try:
            lattice = np.linalg.cholesky(self.decode(coefficients))
        except np.linalg.LinAlgError as error:
            raise NumericalError("Cholesky lattice section failed") from error
        if float(np.linalg.det(lattice)) <= 0.0:
            raise InvariantViolation("Cholesky section was not right handed")
        return lattice


def loewner_lower_bound(reference_metric: np.ndarray, minimum_singular_value: float) -> float:
    """``l = s_min^2 / lambda_min(G0)``, raised by whole ulps so the floor is exact.

    It guarantees ``sqrt(lambda_min(G0) l) >= s_min`` in float64, which is the
    Loewner chart's shortest-vector certificate.  A reference cell already
    shorter than ``s_min`` admits no interval and is a domain error.
    """

    if not np.isfinite(minimum_singular_value) or minimum_singular_value <= 0.0:
        raise DomainError("the declared shortest-vector support must be finite and positive")
    reference_floor = float(np.linalg.eigvalsh(_validate_spd(reference_metric))[0])
    lower = float(minimum_singular_value) ** 2 / reference_floor
    while float(np.sqrt(reference_floor * lower)) < minimum_singular_value:
        lower = float(np.nextafter(lower, np.inf))
    if not lower < 1.0:
        raise DomainError(
            "reference cell is too small for the declared shortest-vector support"
        )
    return lower


class LoewnerSPDChart(InvariantSPDChart):
    """Point-group-invariant chart onto the open Loewner interval ``l G0 < G < u G0``.

    Every lattice direction's length lies within ``[sqrt(l), sqrt(u)]`` of the
    reference.  For ``Z`` in the commutant, the latent map is

    ``W(Z) = (u e^Z + c l I)(e^Z + c I)^-1 = l I + (u - l) sigma(Z - ln(c) I)``,

    with ``c = (u - 1)/(1 - l)``, so that ``W(0) = I``, and ``G = S W S``.  Here
    ``sigma`` is the logistic function.  ``W`` is a monotone scalar function of
    the symmetric matrix ``Z``.  It therefore maps the spectrum bijectively onto
    ``(l, u)``, is well defined at repeated eigenvalues, and preserves the
    commutant: ``R W R^T = W`` whenever ``R Z R^T = Z``.  Point-group invariance
    is exact.

    Unlike the radial ball ``||log W||_F < B``, which bounds the shortest and
    longest directions together, the shortest-vector certificate
    ``sigma_min(L) >= sqrt(lambda_min(G0) l)`` depends only on ``l``.  Long-axis
    (layered) lattices can thus be admitted through ``u`` without lowering the
    physical floor (review B2).  ``bounded_coefficients`` and ``decode`` keep the
    log-metric meaning of the inherited chart, with ``b`` the coefficients of
    ``log W``.
    """

    def __init__(
        self,
        rotations: Sequence[np.ndarray],
        *,
        reference_metric: np.ndarray | None = None,
        lower: float,
        upper: float,
        tolerance: float = 1.0e-10,
    ) -> None:
        values = (float(lower), float(upper))
        if not all(np.isfinite(values)) or not 0.0 < values[0] < 1.0 < values[1]:
            raise DomainError("Loewner interval requires finite 0 < lower < 1 < upper")
        super().__init__(
            rotations, reference_metric=reference_metric, bound=None, tolerance=tolerance
        )
        self.lower, self.upper = values
        self._log_mobius = float(np.log((values[1] - 1.0) / (1.0 - values[0])))

    @property
    def minimum_lattice_singular_value_bound(self) -> float:
        """``sigma_min(L) >= sqrt(lambda_min(G0) l)`` over the whole support."""

        reference_floor = float(np.linalg.eigvalsh(self.reference_metric)[0])
        return float(np.sqrt(reference_floor * self.lower))

    @property
    def maximum_lattice_singular_value_bound(self) -> float:
        """``sigma_max(L) <= sqrt(lambda_max(G0) u)`` over the whole support."""

        reference_ceiling = float(np.linalg.eigvalsh(self.reference_metric)[-1])
        return float(np.sqrt(reference_ceiling * self.upper))

    def _spectral(self, matrix: np.ndarray, function) -> np.ndarray:  # type: ignore[no-untyped-def]
        eigenvalues, eigenvectors = np.linalg.eigh((matrix + matrix.T) / 2.0)
        result = (eigenvectors * function(eigenvalues)) @ eigenvectors.T
        return (result + result.T) / 2.0

    def _whitened_from_latent(self, latent: Sequence[float]) -> np.ndarray:
        algebra = self.algebra_matrix(latent)
        span = self.upper - self.lower
        # sigma(x) = (1 + tanh(x/2))/2 is exact and overflow-free for all x.
        return self._spectral(
            algebra,
            lambda mu: self.lower
            + span * 0.5 * (1.0 + np.tanh(0.5 * (mu - self._log_mobius))),
        )

    def decode_latent(self, latent: Sequence[float]) -> np.ndarray:
        whitened = self._whitened_from_latent(latent)
        metric = _validate_spd(self.sqrt_reference @ whitened @ self.sqrt_reference)
        metric = (metric + metric.T) / 2.0
        if not self.is_invariant(metric):
            raise InvariantViolation("Loewner-chart metric violates point-group invariance")
        return metric

    def encode_latent(self, metric: np.ndarray) -> np.ndarray:
        value = _validate_spd(metric)
        if not self.is_invariant(value):
            raise DomainError("metric lies outside this point-group-invariant SPD cone")
        whitened = self.inverse_sqrt_reference @ value @ self.inverse_sqrt_reference
        whitened = (whitened + whitened.T) / 2.0
        eigenvalues = np.linalg.eigvalsh(whitened)
        if not (self.lower < float(eigenvalues[0]) and float(eigenvalues[-1]) < self.upper):
            raise DomainError("metric lies outside the open Loewner interval of this chart")
        span = self.upper - self.lower
        algebra = self._spectral(
            whitened,
            lambda w: self._log_mobius
            + np.log((w - self.lower) / span)
            - np.log((self.upper - w) / span),
        )
        coefficients = np.asarray([np.sum(algebra * basis) for basis in self.basis])
        residual = float(np.linalg.norm(algebra - self.algebra_matrix(coefficients), ord="fro"))
        scale = max(float(np.linalg.norm(algebra, ord="fro")), 1.0)
        if residual > self.tolerance * scale * 10.0:
            raise InvariantViolation("Loewner latent is outside the symmetric commutant")
        return coefficients

    def uniform_coefficients(
        self,
        rng: np.random.Generator,
        *,
        log_volume_window: tuple[float, float] | None = None,
        batch: int = 4096,
        max_batches: int = 10_000,
    ) -> np.ndarray:
        """One draw of ``b`` uniform (Lebesgue) on the open support.

        The support is ``{b : spectrum of A(b) in (ln l, ln u)}``.  Batched
        rejection from the cube ``|b_j| <= sqrt(3) max(|ln l|, ln u)``
        contains it, because ``|b_j| <= ||A||_F <= sqrt(3) ||A||_2``.  The
        first accepted proposal of an i.i.d. stream is exactly uniform on the
        support.

        ``log_volume_window = (lo, hi)`` further restricts the draw to
        ``lo < tr(A)/2 < hi``.  Since ``tr(A)/2 = ln(V/V0)`` is the Hencky
        volumetric strain against the reference, this keeps the cell volume
        within ``(e^lo, e^hi)`` of the reference volume.  The intersection is
        still inside the cube, so the draw stays exactly uniform on it.  The
        window must contain the reference volume (``lo < 0 < hi``).
        """

        low, high = float(np.log(self.lower)), float(np.log(self.upper))
        volume_low, volume_high = -np.inf, np.inf
        if log_volume_window is not None:
            volume_low, volume_high = (float(value) for value in log_volume_window)
            if not (
                np.isfinite(volume_low) and np.isfinite(volume_high)
                and volume_low < 0.0 < volume_high
            ):
                raise DomainError(
                    "log_volume_window must be finite with lo < 0 < hi "
                    "(it must contain the reference volume)"
                )
        radius = float(np.sqrt(3.0)) * max(abs(low), high)
        basis = np.stack(self.basis)
        half_traces = np.trace(basis, axis1=1, axis2=2) / 2.0
        for _ in range(max_batches):
            proposals = rng.uniform(-radius, radius, size=(batch, self.dimension))
            spectra = np.linalg.eigvalsh(np.einsum("ni,ijk->njk", proposals, basis))
            volumetric = proposals @ half_traces
            accepted = np.flatnonzero(
                (spectra[:, 0] > low)
                & (spectra[:, -1] < high)
                & (volumetric > volume_low)
                & (volumetric < volume_high)
            )
            if accepted.size:
                return proposals[int(accepted[0])].copy()
        raise NumericalError("uniform Loewner-support rejection exceeded its draw budget")

    def bounded_coefficients(self, latent: Sequence[float]) -> np.ndarray:
        """Log-metric coefficients ``b`` of ``log W(Z)``, with spectrum in ``(ln l, ln u)``."""

        return self.encode(self.decode_latent(latent))

    def unbounded_latent(self, coefficients: Sequence[float]) -> np.ndarray:
        """Invert :meth:`bounded_coefficients`; refuses ``b`` outside the open interval."""

        return self.encode_latent(self.decode(coefficients))


def affine_invariant_spd_inner_product(
    metric: np.ndarray, left: np.ndarray, right: np.ndarray
) -> float:
    """Evaluate ``tr(G^-1 U G^-1 V)`` from equation (1.36)."""

    g = _validate_spd(metric)
    u = np.asarray(left, dtype=np.float64)
    v = np.asarray(right, dtype=np.float64)
    if u.shape != (3, 3) or v.shape != (3, 3):
        raise DomainError("SPD tangent matrices must be 3x3")
    solved_u = np.linalg.solve(g, u)
    solved_v = np.linalg.solve(g, v)
    return float(np.trace(solved_u @ solved_v))
