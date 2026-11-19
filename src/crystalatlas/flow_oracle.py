"""Exact tiny-distribution flow oracle on one cover (theory F8b and F10; review M1).

The oracle is the law that the production geometry objective samples inside
one fixed cover with ``M`` data records, written in closed form:

1. choose record ``m`` with probability ``w_m``;
2. draw ``q_0 = (z_0, eta_0)`` from :func:`~crystalatlas.flow.sample_product_base`,
   ``N(0, I) x Haar(R^D / Lambda)``;
3. draw ``q_1`` from :func:`~crystalatlas.flow.sample_dequantized_endpoint`,
   ``N(z_m, sigma_z^2 I) x`` the torus heat kernel at ``eta_m`` with Cartesian
   scale ``sigma_eta``;
4. choose a residual action ``A_g`` uniformly from a finite group ``H`` of
   lattice automorphisms, and build the path with
   :func:`~crystalatlas.flow.conditional_flow_path` (``torus_action=A_g``).

Derivation, in lattice coefficients ``a = eta P^-1`` with period basis ``P``
(rows), Gram matrix ``G = P P^T``, ``s = 1 - t`` and ``alpha_m = eta_m P^-1``.

* The record selects independent ``z`` and ``eta`` factors.  For a Gaussian
  record, ``z_t | m ~ N(t z_m, v_t I)`` with ``v_t = s^2 + t^2 sigma_z^2``, and
  ``E[z_1 - z_0 | z_t, m] = z_m + (t sigma_z^2 - s) (z_t - t z_m) / v_t``.
* Because ``a_0`` is Haar, the coordinate lift ``delta = wrap(a_1 - a_0)`` is
  uniform on ``[-1/2, 1/2)^D`` and independent of ``a_1``, and
  ``a_t = a_1 - s delta``.  Hence
  ``p_t(a) = int p_1(a + s delta) d delta`` and the current is
  ``J_t(a) = int delta p_1(a + s delta) d delta``.  The heat kernel has
  coefficients ``exp(-2 pi^2 sigma_eta^2 n^T G^-1 n - 2 pi i n.alpha_m)``, so the
  box average multiplies coefficient ``n`` by ``prod_j sinc(n_j s)``.  The
  current multiplies it by ``i tau(n_j s) prod_{l != j} sinc(n_l s)``, with
  ``tau(x) = (sin(pi x) - pi x cos(pi x)) / (2 pi^2 x^2)``.
* The action is applied after the lift, so with ``M_g = P A_g P^-1`` (an integer
  unimodular matrix) each intermediate law is the pushforward
  ``p_t^g(a) = p_t(a M_g^-1)``, with current ``J_t^g(a) = J_t(a M_g^-1) M_g``.
  This is not the law of a new half-open lift between transformed endpoints
  when ``M_g`` is not a signed permutation.
* The marginal velocity is ``v_t = J_t / p_t`` for the mixture over ``(m, g)``,
  with the record's ``z`` density as the posterior weight.  The Cartesian torus
  velocity is ``v^a P``.

Every density is evaluated by its truncated Fourier series.  The constructor
certifies a bound on the relative error of each component's density and
current, which is valid for every ``t`` because the box average of ``p_1`` is
bounded below by the density floor of ``p_1``.  An ill-conditioned toy is
refused rather than evaluated.  The bound covers the truncated tail plus a
conservative summation-rounding term.  The floor uses the nearest-plane
covering radius of the given basis.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

from ._optional import require_torch
from .errors import DomainError, InvariantViolation
from .flow import (
    CFMPathSample,
    FlowVelocity,
    conditional_flow_path,
    sample_dequantized_endpoint,
    sample_product_base,
)

torch = require_torch()
Tensor = torch.Tensor

_EPSILON = 2.0**-52
_TAU_SERIES_LIMIT = 1.0e-2
_CHUNK = 2048


@dataclass(frozen=True)
class FlowOracleConditioning:
    """Certified Fourier truncation and rounding bounds for one toy."""

    fourier_radius: int
    fourier_terms: int
    fourier_tail_bound: float
    component_density_floor: float
    relative_error_bound: float


def _tau(x: Tensor) -> Tensor:
    """``int_{-1/2}^{1/2} delta exp(2 pi i x delta) d delta = i tau(x)``."""

    y = math.pi * x
    small = y.abs() < _TAU_SERIES_LIMIT
    safe = torch.where(small, torch.ones_like(y), y)
    direct = (torch.sin(safe) - safe * torch.cos(safe)) / (2.0 * safe * safe)
    y2 = y * y
    series = y * (1.0 / 6.0 - y2 * (1.0 / 60.0 - y2 * (1.0 / 1680.0 - y2 / 90720.0)))
    return torch.where(small, series, direct)


def _lll_unimodular(basis: Tensor, delta: float = 0.99) -> Tensor:
    """Integer unimodular ``U`` with ``U @ basis`` LLL-reduced (rows are vectors).

    Used only to choose the Fourier evaluation frame: the series and its
    certificate are valid for any unimodular ``U``, which is verified here.
    """

    rows = [list(map(float, row)) for row in basis.tolist()]
    dimension = len(rows)
    unimodular = [[1 if i == j else 0 for j in range(dimension)] for i in range(dimension)]

    def dot(u: list[float], v: list[float]) -> float:
        return sum(a * b for a, b in zip(u, v, strict=True))

    def orthogonalized() -> list[list[float]]:
        result: list[list[float]] = []
        for row in rows:
            vector = list(row)
            for previous in result:
                scale = dot(row, previous) / dot(previous, previous)
                vector = [a - scale * b for a, b in zip(vector, previous, strict=True)]
            result.append(vector)
        return result

    index = 1
    while index < dimension:
        for j in range(index - 1, -1, -1):
            star = orthogonalized()[j]
            quotient = round(dot(rows[index], star) / dot(star, star))
            if quotient:
                rows[index] = [
                    a - quotient * b for a, b in zip(rows[index], rows[j], strict=True)
                ]
                unimodular[index] = [
                    a - quotient * b
                    for a, b in zip(unimodular[index], unimodular[j], strict=True)
                ]
        star = orthogonalized()
        mu = dot(rows[index], star[index - 1]) / dot(star[index - 1], star[index - 1])
        if dot(star[index], star[index]) >= (delta - mu * mu) * dot(
            star[index - 1], star[index - 1]
        ):
            index += 1
        else:
            rows[index - 1], rows[index] = rows[index], rows[index - 1]
            unimodular[index - 1], unimodular[index] = unimodular[index], unimodular[index - 1]
            index = max(index - 1, 1)
    matrix = torch.tensor(unimodular, dtype=torch.float64)
    if abs(round(float(torch.linalg.det(matrix)))) != 1:  # pragma: no cover - LLL invariant
        raise InvariantViolation("lattice reduction lost unimodularity")
    return matrix


def _integer_matrix(matrix: Tensor, name: str) -> Tensor:
    rounded = torch.round(matrix)
    if not bool(torch.all((matrix - rounded).abs() <= 1.0e-9).item()):
        raise DomainError(f"{name} is not an integer matrix in lattice coordinates")
    determinant = round(float(torch.linalg.det(rounded)))
    if abs(determinant) != 1:
        raise DomainError(f"{name} is not unimodular in lattice coordinates")
    return rounded


class TinyFlowOracle:
    """Exact marginal density, current, and velocity of the production CFM law."""

    def __init__(
        self,
        z_data: Tensor,
        eta_data: Tensor,
        weights: Tensor,
        period_basis: Tensor,
        *,
        sigma_z: float,
        sigma_eta: float | None = None,
        torus_actions: Sequence[Tensor] | None = None,
        torus_translations: Sequence[Tensor] | None = None,
        max_relative_error: float = 1.0e-9,
        eta_covariances: Sequence[Tensor] | None = None,
    ) -> None:
        """``sigma_eta`` gives the flat kernel ``sigma_eta^2 I`` in ``eta`` coordinates
        (the production ``fractional-v1`` kernel).  ``eta_covariances`` gives one
        ``[D, D]`` covariance per record instead (the ``cartesian-v2`` kernel, whose
        covariance depends on the record's lattice).  Exactly one must be given.
        """

        dtype = torch.float64
        z_data = torch.as_tensor(z_data, dtype=dtype)
        eta_data = torch.as_tensor(eta_data, dtype=dtype)
        weights = torch.as_tensor(weights, dtype=dtype)
        period_basis = torch.as_tensor(period_basis, dtype=dtype)
        if z_data.ndim != 2 or eta_data.ndim != 2 or weights.ndim != 1:
            raise DomainError("z_data and eta_data are [M, d] and weights is [M]")
        records = len(weights)
        if records < 1 or len(z_data) != records or len(eta_data) != records:
            raise DomainError("z_data, eta_data, and weights must describe the same records")
        dimension = eta_data.shape[1]
        if dimension < 1 or period_basis.shape != (dimension, dimension):
            raise DomainError("period_basis must be a nonsingular [D, D] matrix with D >= 1")
        if not all(bool(torch.isfinite(x).all().item()) for x in (z_data, eta_data, weights)):
            raise DomainError("toy data must be finite")
        if bool((weights <= 0.0).any().item()) or abs(float(weights.sum()) - 1.0) > 1.0e-12:
            raise DomainError("weights must be positive and sum to one")
        if not (math.isfinite(sigma_z) and sigma_z > 0.0):
            raise DomainError("sigma_z must be finite and positive")
        if (sigma_eta is None) == (eta_covariances is None):
            raise DomainError("give exactly one of sigma_eta and eta_covariances")
        if sigma_eta is not None:
            if not (math.isfinite(sigma_eta) and sigma_eta > 0.0):
                raise DomainError("sigma_eta must be finite and positive")
            covariances = (sigma_eta * sigma_eta) * torch.eye(dimension, dtype=dtype).expand(
                records, -1, -1
            ).clone()
        else:
            assert eta_covariances is not None
            items = [torch.as_tensor(item, dtype=dtype) for item in eta_covariances]
            if len(items) != records or any(
                item.shape != (dimension, dimension) for item in items
            ):
                raise DomainError("eta_covariances must hold one [D, D] matrix per record")
            covariances = torch.stack(items)
            if not bool(torch.isfinite(covariances).all().item()) or not torch.allclose(
                covariances, covariances.transpose(-1, -2), rtol=1e-12, atol=1e-14
            ):
                raise DomainError("eta_covariances must be finite symmetric matrices")
            if bool((torch.linalg.eigvalsh(covariances)[:, 0] <= 0.0).any().item()):
                raise DomainError("eta_covariances must be positive definite")
        determinant = float(torch.linalg.det(period_basis))
        if not math.isfinite(determinant) or determinant == 0.0:
            raise DomainError("period_basis must be nonsingular")
        inverse_basis = torch.linalg.inv(period_basis)
        cartesian_actions = (
            [torch.eye(dimension, dtype=dtype)]
            if torus_actions is None
            else [torch.as_tensor(action, dtype=dtype) for action in torus_actions]
        )
        cartesian_translations = (
            [torch.zeros(dimension, dtype=dtype) for _ in cartesian_actions]
            if torus_translations is None
            else [torch.as_tensor(item, dtype=dtype) for item in torus_translations]
        )
        if len(cartesian_translations) != len(cartesian_actions):
            raise DomainError("torus translations must align with the torus actions")
        lattice_actions = []
        lattice_translations = []
        for index, (action, shift) in enumerate(
            zip(cartesian_actions, cartesian_translations, strict=True)
        ):
            if action.shape != (dimension, dimension) or shift.shape != (dimension,):
                raise DomainError("every torus action must be [D, D] with a [D] translation")
            lattice_actions.append(
                _integer_matrix(period_basis @ action @ inverse_basis, f"torus action {index}")
            )
            lattice_translations.append(torch.remainder(shift @ inverse_basis, 1.0))

        def _key(matrix: Tensor, shift: Tensor) -> tuple[float, ...]:
            wrapped = torch.remainder(shift, 1.0)
            wrapped = torch.where(
                (wrapped - 1.0).abs() < 1e-12, torch.zeros_like(wrapped), wrapped
            )
            return tuple(matrix.flatten().tolist()) + tuple(
                round(v, 12) for v in wrapped.tolist()
            )

        keys = {_key(m, t) for m, t in zip(lattice_actions, lattice_translations, strict=True)}
        if len(keys) != len(lattice_actions):
            raise DomainError("torus actions must be distinct")
        identity_key = _key(
            torch.eye(dimension, dtype=dtype), torch.zeros(dimension, dtype=dtype)
        )
        if identity_key not in keys:
            raise DomainError("torus actions must contain the identity")
        # Affine composition in row convention: (M1, m1) then (M2, m2) = (M1 M2, m1 M2 + m2).
        for left_m, left_t in zip(lattice_actions, lattice_translations, strict=True):
            for right_m, right_t in zip(lattice_actions, lattice_translations, strict=True):
                if _key(left_m @ right_m, left_t @ right_m + right_t) not in keys:
                    raise DomainError("torus actions must be closed under composition")

        self.z_data = z_data
        self.eta_data = eta_data
        self.weights = weights
        self.period_basis = period_basis
        self.sigma_z = float(sigma_z)
        self.sigma_eta = None if sigma_eta is None else float(sigma_eta)
        self.eta_covariances = covariances
        # Covariances in lattice coefficients a = eta P^-1: P^-T Sigma P^-1.
        self._coefficient_covariances = (
            inverse_basis.transpose(-1, -2)[None] @ covariances @ inverse_basis[None]
        )
        self._noise_transforms = torch.linalg.cholesky(covariances).transpose(-1, -2)
        self.cartesian_actions = torch.stack(cartesian_actions)
        self.cartesian_translations = torch.stack(cartesian_translations)
        self.lattice_actions = torch.stack(lattice_actions)
        self.lattice_translations = torch.stack(lattice_translations)
        self.inverse_lattice_actions = torch.round(torch.linalg.inv(self.lattice_actions))
        self._inverse_basis = inverse_basis
        self._log_cell_volume = math.log(abs(determinant))
        self._alpha = eta_data @ inverse_basis
        # The series is evaluated in an LLL-reduced frame, a' = a U^-1 and
        # n' = U n, which needs the fewest modes for a given accuracy.  The
        # coordinate-lift box stays in the given basis, through n = U^-1 n'.
        reduction = _lll_unimodular(period_basis)
        self._to_reduced = torch.round(torch.linalg.inv(reduction))
        self.conditioning, reduced_frequencies, decay = self._certify(
            reduction @ period_basis, max_relative_error
        )
        radius = self.conditioning.fourier_radius
        # Every coefficient set is Hermitian (c_-n = conj c_n): the box factors
        # are even and i*tau is odd and imaginary.  Re sum_n is therefore the
        # n'_0 = 0 slab plus twice the n'_0 > 0 half, so only n'_0 >= 0 is kept.
        half = reduced_frequencies[:, 0] >= 0
        multiplicity = torch.where(reduced_frequencies[half, 0] > 0, 2.0, 1.0).to(dtype)
        self._modes = [
            torch.arange(0 if axis == 0 else -radius, radius + 1, dtype=dtype)
            for axis in range(dimension)
        ]
        frequencies = reduced_frequencies[half] @ self._to_reduced.T
        phase = -2.0 * math.pi * (self._alpha @ frequencies.T)  # [M, T_half]
        self._frequencies = frequencies
        self._record_coefficients = (multiplicity * decay[:, half]).to(
            torch.complex128
        ) * torch.exp(1j * phase)

    @property
    def lattice_dimension(self) -> int:
        return self.z_data.shape[1]

    @property
    def intrinsic_dimension(self) -> int:
        return self.eta_data.shape[1]

    def _certify(
        self, reduced_basis: Tensor, max_relative_error: float
    ) -> tuple[FlowOracleConditioning, Tensor, Tensor]:
        """Choose the Fourier radius and bound the relative error of every component.

        Frequencies are returned in the reduced frame.  The density floor uses the
        nearest-plane covering-radius bound of the reduced basis.  With per-record
        covariances ``Sigma_m``, the decay rate uses the smallest eigenvalue of any
        reduced covariance and the floor uses ``|v|^2_{Sigma^-1} <= |v|^2 / lambda_min``,
        so every bound below holds for every record.
        """

        dimension = self.intrinsic_dimension
        inverse_reduced = torch.linalg.inv(reduced_basis)
        reduced_covariances = (
            inverse_reduced.transpose(-1, -2)[None]
            @ self.eta_covariances
            @ inverse_reduced[None]
        )
        reduced_covariances = 0.5 * (
            reduced_covariances + reduced_covariances.transpose(-1, -2)
        )
        rate = 2.0 * math.pi**2 * float(torch.linalg.eigvalsh(reduced_covariances)[:, 0].min())
        orthogonal = torch.linalg.qr(reduced_basis.T).R.diagonal().abs()
        covering_radius = 0.5 * float(torch.sqrt((orthogonal**2).sum()))
        eigenvalues = torch.linalg.eigvalsh(self.eta_covariances)  # [M, D]
        floors = torch.exp(
            self._log_cell_volume
            - 0.5 * torch.log(2.0 * math.pi * eigenvalues).sum(-1)
            - covering_radius**2 / (2.0 * eigenvalues[:, 0])
        )
        floor = float(floors.min())

        def shell_tail(radius: int) -> float:
            total, shell = 0.0, radius + 1
            while True:
                term = (
                    2
                    * dimension
                    * (2 * shell + 1) ** (dimension - 1)
                    * math.exp(-rate * shell * shell)
                )
                total += term
                if term < 1.0e-30 * max(total, 1.0e-300) or term == 0.0:
                    return total
                shell += 1

        radius = 1
        while shell_tail(radius) > _EPSILON * floor:
            radius += 1
            if radius > 64:
                raise DomainError(
                    "toy is ill-conditioned: its torus kernel needs more than 64 Fourier modes"
                )
        modes = torch.arange(-radius, radius + 1, dtype=torch.float64)
        grids = torch.meshgrid(*([modes] * dimension), indexing="ij")
        frequencies = torch.stack([grid.reshape(-1) for grid in grids], dim=-1)
        quadratic = torch.einsum("ti,mij,tj->mt", frequencies, reduced_covariances, frequencies)
        decay = torch.exp(-2.0 * math.pi**2 * quadratic)  # [M, T]
        terms = len(frequencies)
        tail = shell_tail(radius)
        # |sinc| <= 1 and |tau| <= 1/4, so the density coefficients dominate.
        coefficient_sum = float(decay.sum(-1).max()) + tail
        bound = (tail + (terms + 16) * _EPSILON * coefficient_sum) / floor
        if not bound <= max_relative_error:
            raise DomainError(
                f"toy is ill-conditioned: certified relative error {bound:.3e} exceeds "
                f"{max_relative_error:.3e}; increase sigma_eta"
            )
        return FlowOracleConditioning(radius, terms, tail, floor, bound), frequencies, decay

    def _fourier(self, coefficients: Tensor, points: Tensor) -> Tensor:
        """``Re sum_n c_kn exp(2 pi i n.x_b)`` for coefficient sets ``k``: ``[B, K]``."""

        dimension = self.intrinsic_dimension
        widths = [len(modes) for modes in self._modes]
        sets = coefficients.shape[0]
        flat = coefficients.reshape(-1, widths[-1]).T  # [W_last, sets * prod(W_rest)]
        values = []
        for chunk in torch.split(points, _CHUNK):
            waves = [
                torch.exp(
                    2j * math.pi * chunk[:, j, None].to(torch.complex128) * self._modes[j]
                )
                for j in range(dimension)
            ]
            partial = waves[-1] @ flat
            for axis in range(dimension - 2, -1, -1):
                partial = partial.reshape(len(chunk), -1, widths[axis])
                partial = (partial * waves[axis][:, None, :]).sum(-1)
            values.append(partial.reshape(len(chunk), sets).real)
        return torch.cat(values)

    def _torus_components(self, time: Tensor, points: Tensor) -> tuple[Tensor, Tensor]:
        """Per-record density ``[B, M]`` and current ``[B, M, D]`` in coefficients."""

        dimension = self.intrinsic_dimension
        shrink = 1.0 - time
        box = torch.sinc(self._frequencies * shrink)  # [T, D]
        first = _tau(self._frequencies * shrink)
        density_factor = box.prod(dim=-1)
        sets = [self._record_coefficients * density_factor]
        for j in range(dimension):
            others = torch.cat((box[:, :j], box[:, j + 1 :]), dim=-1).prod(dim=-1)
            # i * tau multiplies the complex coefficient; its real part is the current.
            sets.append(self._record_coefficients * (1j * first[:, j] * others))
        records = len(self.weights)
        values = self._fourier(torch.cat(sets), points @ self._to_reduced)  # [B, (1 + D) M]
        density = values[:, :records]
        current = values[:, records:].reshape(len(points), dimension, records).transpose(1, 2)
        return density, current

    @staticmethod
    def _push_current(current: Tensor, lattice_action: Tensor) -> Tensor:
        return current @ lattice_action

    def _mixture(self, time: Tensor, z: Tensor, eta: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """Log density, lattice velocity, and torus coefficient velocity."""

        time = torch.as_tensor(time, dtype=z.dtype)
        if time.ndim != 0:
            raise DomainError("the oracle evaluates one time at a time")
        if z.ndim != 2 or eta.ndim != 2 or len(z) != len(eta):
            raise DomainError("the oracle evaluates batches z [B, d_z] and eta [B, D]")
        shrink = 1.0 - time
        variance = shrink * shrink + time * time * self.sigma_z**2
        offset = z[:, None, :] - time * self.z_data[None, :, :]  # [B, M, d_z]
        log_z = -0.5 * (
            (offset * offset).sum(-1) / variance
            + self.lattice_dimension * torch.log(2.0 * math.pi * variance)
        )
        component_z = self.z_data[None] + (time * self.sigma_z**2 - shrink) / variance * offset
        scale = log_z.max(dim=1, keepdim=True).values
        prior = self.weights * torch.exp(log_z - scale) / len(self.lattice_actions)
        coefficients = eta @ self._inverse_basis
        total = torch.zeros(len(z), dtype=z.dtype)
        lattice_flux = torch.zeros_like(z)
        torus_flux = torch.zeros_like(eta)
        for action, inverse, shift in zip(
            self.lattice_actions,
            self.inverse_lattice_actions,
            self.lattice_translations,
            strict=True,
        ):
            # Pushforward by a -> a M + m: evaluate the base law at (a - m) M^-1.
            density, current = self._torus_components(time, (coefficients - shift) @ inverse)
            if not bool((density > 0.0).all().item()):
                raise InvariantViolation("certified toy density lost positivity")
            weight = prior * density
            total = total + weight.sum(1)
            lattice_flux = lattice_flux + (weight[..., None] * component_z).sum(1)
            pushed = self._push_current((prior[..., None] * current).sum(1), action)
            torus_flux = torus_flux + pushed
        log_density = scale.squeeze(1) + torch.log(total) - self._log_cell_volume
        return log_density, lattice_flux / total[:, None], torus_flux / total[:, None]

    def log_density(self, time: Tensor, z: Tensor, eta: Tensor) -> Tensor:
        """Cartesian log density of ``q_t`` (Lebesgue in ``z`` and ``eta``)."""

        return self._mixture(time, z, eta)[0]

    def velocity(self, time: Tensor, z: Tensor, eta: Tensor) -> FlowVelocity:
        """``E[u | q_t]``; the signature of the production integrators' field."""

        _, lattice, torus = self._mixture(time, z, eta)
        return FlowVelocity(z=lattice, eta=torus @ self.period_basis)

    def base_log_density(self, z: Tensor, eta: Tensor) -> Tensor:
        del eta
        return (
            -0.5 * (z * z).sum(-1)
            - 0.5 * z.shape[-1] * math.log(2.0 * math.pi)
            - self._log_cell_volume
        )

    def moment(
        self,
        time: float,
        frequency: Sequence[int],
        *,
        z_index: int | None = None,
        z_power: int = 0,
    ) -> complex:
        """Exact ``E[z_i^p exp(2 pi i n.a_t)]`` of the action-averaged law at ``t``.

        Record ``m`` and action ``g`` contribute ``E[z_t^p | m]`` times the
        heat-kernel characteristic function at ``k = M_g n``, damped by the box
        average ``prod_j sinc(s k_j)``.  This uses the coefficient formula of the
        derivation, but none of the evaluation code of :meth:`log_density`.
        """

        if not (math.isfinite(time) and 0.0 <= time <= 1.0):
            raise DomainError("time must lie in [0, 1]")
        if z_power not in (0, 1, 2) or (z_power and z_index is None):
            raise DomainError("z_power must be 0, 1, or 2 with a z_index")
        n = torch.as_tensor(frequency, dtype=torch.float64)
        if n.shape != (self.intrinsic_dimension,):
            raise DomainError("frequency must have one integer per torus dimension")
        shrink = 1.0 - time
        variance = shrink * shrink + time * time * self.sigma_z**2
        total = 0.0 + 0.0j
        for action, shift in zip(self.lattice_actions, self.lattice_translations, strict=True):
            pulled = action @ n
            translation_phase = 2.0 * math.pi * float(shift @ n)
            box = float(torch.sinc(shrink * pulled).prod())
            for record in range(len(self.weights)):
                decay = box * math.exp(
                    -2.0
                    * math.pi**2
                    * float(pulled @ self._coefficient_covariances[record] @ pulled)
                )
                phase = 2.0 * math.pi * float(self._alpha[record] @ pulled) + translation_phase
                if z_power == 0:
                    lattice = 1.0
                else:
                    centre = time * float(self.z_data[record, z_index])
                    lattice = centre if z_power == 1 else centre**2 + variance
                total += (
                    float(self.weights[record])
                    * lattice
                    * decay
                    * complex(math.cos(phase), math.sin(phase))
                )
        return total / len(self.lattice_actions)

    def sample_paths(self, time: Tensor, *, generator: torch.Generator) -> CFMPathSample:
        """Production CFM samples: base, dequantized endpoint, uniform action, path."""

        time = torch.as_tensor(time, dtype=torch.float64)
        if time.ndim != 1 or not len(time):
            raise DomainError("time must be a nonempty batch of path times")
        batch = len(time)
        records = torch.multinomial(self.weights, batch, replacement=True, generator=generator)
        base = sample_product_base(
            self.lattice_dimension, self.period_basis, batch_shape=(batch,), generator=generator
        )
        if self.sigma_eta is not None:
            endpoint = sample_dequantized_endpoint(
                self.z_data[records],
                self.eta_data[records],
                self.period_basis,
                sigma_z=self.sigma_z,
                sigma_eta=self.sigma_eta,
                generator=generator,
            )
        else:
            endpoint = sample_dequantized_endpoint(
                self.z_data[records],
                self.eta_data[records],
                self.period_basis,
                sigma_z=self.sigma_z,
                sigma_eta=1.0,
                generator=generator,
                eta_noise_transform=self._noise_transforms[records],
            )
        actions = torch.randint(len(self.cartesian_actions), (batch,), generator=generator)
        return conditional_flow_path(
            base.z,
            base.eta,
            endpoint.z,
            endpoint.eta,
            time,
            self.period_basis,
            torus_action=self.cartesian_actions[actions],
            torus_translation=self.cartesian_translations[actions],
        )


__all__ = ["FlowOracleConditioning", "TinyFlowOracle"]
