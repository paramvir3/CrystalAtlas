"""Differentiable exact-catalogue expansion for a fixed continuous cover.

The catalogue and gauge are certified by :mod:`crystalatlas.geometry`.  This
module converts those immutable constants into PyTorch buffers so coordinates,
lattices, and the advertised-slot Jacobian remain differentiable functions of
``(z, eta)`` during conditional-flow training and ODE sampling.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from .._optional import require_torch
from ..algebra import AffineOperation
from ..errors import DomainError, InvariantViolation
from ..flow import canonicalize_torus_validated, validate_period_basis
from ..geometry import IntrinsicCoverLayout
from ..lattice import LoewnerSPDChart, _symmetric_basis_3
from .field import (
    IntrinsicFieldOutput,
    IntrinsicGeometryField,
    translation_quotient_normal_matrix,
)

torch = require_torch()
Tensor = torch.Tensor
nn = torch.nn


def geometry_bridge_numerical_contract() -> dict[str, str]:
    """Numerical semantics that must change identity when resume kernels change."""

    return {
        "schema": "crystalatlas.geometry-bridge-numerics.v1",
        "lattice_chart": "canonical-float64-matrix-exp-then-master-cast.v1",
        "chart_constant_storage": "float64-bits-in-int64-buffers.v1",
        "matrix_exponential": "torch.matrix_exp",
        "published_dtype": "input-float32-or-float64",
        "cholesky_and_pullback_solve": "master-dtype-outside-autocast",
    }


def _log_cosh(value: Tensor) -> Tensor:
    magnitude = value.abs()
    return magnitude + torch.log1p(torch.exp(-2.0 * magnitude)) - math.log(2.0)


def _loewner_divided_differences(shifted: Tensor, span: float) -> Tensor:
    """Divided differences of ``w(x) = l + span sigma(x)`` at shifted eigenvalues.

    ``[sigma(a) - sigma(b)]/(a - b) = (sinh h / h) / (4 cosh(a/2) cosh(b/2))`` with
    ``h = (a - b)/2``, in log-cosh form; the diagonal is the derivative.
    """

    a, b = shifted[..., :, None], shifted[..., None, :]
    half_gap = 0.5 * (a - b)
    small = half_gap.abs() < 1.0e-4
    safe = torch.where(small, torch.ones_like(half_gap), half_gap)
    squared = half_gap * half_gap
    sinhc = torch.where(
        small, 1.0 + squared / 6.0 + squared * squared / 120.0, torch.sinh(safe) / safe
    )
    return span * sinhc * torch.exp(-math.log(4.0) - _log_cosh(0.5 * a) - _log_cosh(0.5 * b))


class _LoewnerSpectralMap(torch.autograd.Function):
    """``W = l I + (u - l) sigma(Z - ln c I)`` and ``log W`` for symmetric ``Z``.

    The backward is the exact Daleckii--Krein Frechet adjoint,
    ``U (D o (U^T G U)) U^T``.  Its divided differences come in closed form, so
    eigenvectors are never differentiated, and the formula is exact at repeated
    eigenvalues (always present in cubic charts):

    ``[sigma(a) - sigma(b)]/(a - b) = (sinh h / h) / (4 cosh(a/2) cosh(b/2))`` with
    ``h = (a - b)/2``, evaluated in log-cosh form so it never overflows; and
    ``[log w_a - log w_b]/(a - b) = (D_w / w_b) log1p(r)/r`` with
    ``r = D_w (a - b) / w_b``.
    """

    @staticmethod
    def forward(  # type: ignore[override]
        ctx: Any, algebra: Tensor, lower: float, upper: float
    ) -> tuple[Tensor, Tensor]:
        log_mobius = math.log((upper - 1.0) / (1.0 - lower))
        eigenvalues, vectors = torch.linalg.eigh(0.5 * (algebra + algebra.transpose(-1, -2)))
        shifted = eigenvalues - log_mobius
        weights = lower + (upper - lower) * 0.5 * (1.0 + torch.tanh(0.5 * shifted))
        whitened = (vectors * weights[..., None, :]) @ vectors.transpose(-1, -2)
        logarithm = (vectors * torch.log(weights)[..., None, :]) @ vectors.transpose(-1, -2)
        ctx.save_for_backward(eigenvalues, vectors, weights, shifted)
        ctx.span = upper - lower
        return (
            0.5 * (whitened + whitened.transpose(-1, -2)),
            0.5 * (logarithm + logarithm.transpose(-1, -2)),
        )

    @staticmethod
    def backward(  # type: ignore[override]
        ctx: Any, grad_whitened: Tensor, grad_logarithm: Tensor
    ) -> tuple[Tensor, None, None]:
        # Grad mode is on inside a backward only for create_graph=True.  This
        # adjoint is first-order; torch's once_differentiable would silently drop
        # the dependence on the saved spectrum and return zero second derivatives.
        if torch.is_grad_enabled():
            raise RuntimeError(
                "the loewner-v2 chart provides first derivatives only; second "
                "derivatives (create_graph=True) through it are not implemented"
            )
        eigenvalues, vectors, weights, shifted = ctx.saved_tensors
        divided_w = _loewner_divided_differences(shifted, ctx.span)
        gap = eigenvalues[..., :, None] - eigenvalues[..., None, :]
        ratio = divided_w * gap / weights[..., None, :]
        tiny = ratio.abs() < 1.0e-8
        safe_ratio = torch.where(tiny, torch.ones_like(ratio), ratio)
        log1p_ratio = torch.where(
            tiny, 1.0 - ratio / 2.0 + ratio * ratio / 3.0, torch.log1p(safe_ratio) / safe_ratio
        )
        divided_log = divided_w / weights[..., None, :] * log1p_ratio
        divided_log = 0.5 * (divided_log + divided_log.transpose(-1, -2))
        rotated_w = vectors.transpose(-1, -2) @ grad_whitened @ vectors
        rotated_log = vectors.transpose(-1, -2) @ grad_logarithm @ vectors
        gradient = vectors @ (divided_w * rotated_w + divided_log * rotated_log) @ (
            vectors.transpose(-1, -2)
        )
        return 0.5 * (gradient + gradient.transpose(-1, -2)), None, None


def loewner_whitened_metric(
    algebra: Tensor, lower: float, upper: float
) -> tuple[Tensor, Tensor]:
    """Whitened Loewner-chart metric ``W(Z)`` and ``log W(Z)`` (review B2), float64."""

    if not 0.0 < lower < 1.0 < upper or not math.isfinite(upper):
        raise DomainError("Loewner interval requires finite 0 < lower < 1 < upper")
    result: tuple[Tensor, Tensor] = _LoewnerSpectralMap.apply(algebra, lower, upper)
    return result


EXPANSION_CELLS: tuple[str, ...] = ("conventional-v1", "primitive-v2")


def _primitive_coset_representatives(
    row: object, setting: object
) -> tuple[AffineOperation, ...]:
    """One coset representative per class modulo the centring translations.

    Two representatives give centring-translated copies of each other for every
    parameter iff their rotations agree and their translations differ by a
    centring vector; the kept set has ``multiplicity / centering_index`` members.
    """

    centring = {
        tuple(value % 1 for value in translation)
        for translation in setting.centering_translations  # type: ignore[attr-defined]
    }
    chosen: list[AffineOperation] = []
    for operation in row.coset_representatives:  # type: ignore[attr-defined]
        duplicate = any(
            operation.rotation == kept.rotation
            and tuple(
                (a - b) % 1
                for a, b in zip(operation.translation, kept.translation, strict=True)
            )
            in centring
            for kept in chosen
        )
        if not duplicate:
            chosen.append(operation)
    index = setting.centering_index  # type: ignore[attr-defined]
    if len(chosen) * index != len(row.coset_representatives):  # type: ignore[attr-defined]
        raise InvariantViolation("coset representatives do not split evenly over the centring")
    return tuple(chosen)


@dataclass(frozen=True)
class TorchExpandedGeometry:
    bounded_lattice_coordinates: Tensor
    metric: Tensor
    lattice: Tensor
    intrinsic_coordinates: Tensor
    fractional_coordinates: Tensor
    atomic_numbers: Tensor
    atom_features: Tensor
    jacobian_cartesian: Tensor
    lattice_commutant_basis: Tensor
    """Chart commutant basis in canonical Sym(3) coordinates, [6, d] (review M3)."""


class TorchCoverGeometry(nn.Module):
    """Differentiable decoder constants for one :class:`ContinuousCover`.

    The feature vector contains only operation-derived invariant scalars:
    ``[log(1+mu), d/3, log(1+|G_u|), 1[d=0]]``.  It contains no orbit index,
    coordinate component, or arbitrary Wyckoff-letter embedding.
    """

    atom_feature_dim = 4

    def __init__(
        self, layout: IntrinsicCoverLayout, *, expansion_cell: str = "conventional-v1"
    ) -> None:
        super().__init__()
        if not isinstance(layout, IntrinsicCoverLayout):
            raise TypeError("layout must be an IntrinsicCoverLayout")
        if expansion_cell not in EXPANSION_CELLS:
            raise DomainError("expansion_cell must be conventional-v1 or primitive-v2")
        self.layout = layout
        self.expansion_cell = expansion_cell
        setting = layout.setting
        primitive = expansion_cell == "primitive-v2"
        # Review 2026-10-29, M3: under ``primitive-v2`` the field sees the
        # primitive cell.  Each orbit keeps one coset representative per class
        # modulo the centring translations, the lattice handed to the field is
        # ``P L`` with ``P`` the catalogue's centring reduction (rows: primitive
        # basis in conventional fractional coordinates), and fractional
        # coordinates are ``x P^-1``.  Cartesian positions and Jacobians are the
        # same geometry, so the translation-quotient metric of the conventional
        # expansion is ``metric_copies`` times the primitive one.
        self.metric_copies = setting.centering_index if primitive else 1
        self.lattice_dimension = layout.lattice_dimension
        self.intrinsic_dimension = layout.intrinsic_dimension
        reduction = np.asarray(
            [[float(value) for value in row] for row in setting.centering_reduction],
            dtype=np.float64,
        )
        if primitive and abs(abs(np.linalg.det(reduction)) * self.metric_copies - 1.0) > 1e-12:
            raise InvariantViolation("centring reduction volume disagrees with the index")
        primitive_basis = reduction if primitive else np.eye(3)
        self.register_buffer(
            "_primitive_basis_bits",
            torch.tensor(primitive_basis, dtype=torch.float64).view(torch.int64),
            persistent=False,
        )
        self.register_buffer(
            "_primitive_inverse_bits",
            torch.tensor(np.linalg.inv(primitive_basis), dtype=torch.float64).view(torch.int64),
            persistent=False,
        )
        if isinstance(layout.chart, LoewnerSPDChart):
            self.chart_kind = "loewner-v2"
            self.loewner_lower = float(layout.chart.lower)
            self.loewner_upper = float(layout.chart.upper)
            self.metric_bound = math.nan
        elif layout.chart.bound is None:
            raise DomainError("Torch flow expansion requires a bounded lattice chart")
        else:
            self.chart_kind = "ball-v1"
            self.metric_bound = float(layout.chart.bound)
        # Preserve the canonical FP64 values through Module.float()/half()/to().
        # Integer buffers move with the module but are never floating-point
        # downcasts. Views recover the original FP64 bits without allocating or
        # copying from NumPy on each ODE evaluation. In particular, .double()
        # after .float() must not merely promote already-rounded constants.
        self.register_buffer(
            "_sqrt_reference_bits",
            torch.tensor(layout.chart.sqrt_reference, dtype=torch.float64).view(torch.int64),
        )
        self.register_buffer(
            "_invariant_basis_bits",
            torch.tensor(np.stack(layout.chart.basis), dtype=torch.float64).view(torch.int64),
        )
        self.register_load_state_dict_pre_hook(self._validate_chart_buffer_state)
        # Non-persistent: checkpoints keep their state-dict keys.  The bits view
        # keeps exact FP64 values through Module.float()/to().
        self.register_buffer(
            "_canonical_sym3_bits",
            torch.tensor(np.stack(_symmetric_basis_3()), dtype=torch.float64).view(torch.int64),
            persistent=False,
        )
        period_basis = torch.tensor(layout.period_basis, dtype=torch.float64)
        # Validated once here, not at every RK4 stage (review 2026-10-29, H1).
        validate_period_basis(period_basis, layout.intrinsic_dimension)
        self.register_buffer(
            "period_basis",
            # A copy: layouts are cached and shared, so a buffer must never alias them.
            period_basis,
        )

        atomic_numbers: list[int] = []
        atom_features: list[list[float]] = []
        orbit_constants: list[dict[str, Tensor | tuple[tuple[Tensor, Tensor], ...]]] = []
        for index, (orbit, row, parameter_map) in enumerate(
            zip(layout.cover.orbits, layout.positions, layout.parameter_maps, strict=True)
        ):
            offset = torch.as_tensor(
                [float(value) for value in row.representative_offset],
                dtype=torch.float64,
            )
            coordinate = torch.as_tensor(
                np.asarray(row.coordinate_matrix, dtype=np.float64).reshape(row.dimension, 3),
                dtype=torch.float64,
            )
            mapping = torch.as_tensor(parameter_map, dtype=torch.float64)
            representatives = (
                _primitive_coset_representatives(row, setting)
                if primitive
                else row.coset_representatives
            )
            operations = tuple(
                (
                    torch.as_tensor(operation.rotation, dtype=torch.float64),
                    torch.as_tensor(
                        [float(value) for value in operation.translation],
                        dtype=torch.float64,
                    ),
                )
                for operation in representatives
            )
            # Register every tensor so .to(device/dtype) is exact and visible in
            # state_dict; the metadata tuple stores only their buffer names.
            prefix = f"orbit_{index}"
            self.register_buffer(f"{prefix}_offset", offset)
            self.register_buffer(f"{prefix}_coordinate", coordinate)
            self.register_buffer(f"{prefix}_mapping", mapping)
            operation_names: list[tuple[str, str]] = []
            for operation_index, (rotation, translation) in enumerate(operations):
                rotation_name = f"{prefix}_rotation_{operation_index}"
                translation_name = f"{prefix}_translation_{operation_index}"
                self.register_buffer(rotation_name, rotation)
                self.register_buffer(translation_name, translation)
                operation_names.append((rotation_name, translation_name))
            orbit_constants.append(
                {
                    "offset_name": f"{prefix}_offset",
                    "coordinate_name": f"{prefix}_coordinate",
                    "mapping_name": f"{prefix}_mapping",
                    "operations": tuple(operation_names),
                }
            )
            feature = [
                math.log1p(row.multiplicity),
                row.dimension / 3.0,
                math.log1p(row.site_symmetry_order),
                float(row.dimension == 0),
            ]
            for _ in representatives:
                atomic_numbers.append(orbit.atomic_number)
                atom_features.append(feature)
        self._orbit_constants = tuple(orbit_constants)
        self.register_buffer(
            "atomic_numbers", torch.as_tensor(atomic_numbers, dtype=torch.long)
        )
        self.register_buffer(
            "atom_features", torch.as_tensor(atom_features, dtype=torch.float64)
        )
        # Fused per-atom expansion constants (review 2026-10-29, H2).  For the
        # expanded atom n = (orbit o, coset operation j), with A[n] = M_o C_o R_j
        # and c[n] = o_o R_j + t_j, the fractional position is eta A[n] + c[n]
        # (mod 1) and the Cartesian Jacobian is L^T A[n]^T.  Derived exactly from
        # the buffers above in float64 and non-persistent: checkpoints are
        # unchanged.  :func:`expand_padded` expands a mixed batch with them.
        matrices: list[Tensor] = []
        offsets: list[Tensor] = []
        inverse = self._primitive_inverse_bits.view(torch.float64)
        for constants in self._orbit_constants:
            offset = self._buffer(constants["offset_name"])
            coordinate = self._buffer(constants["coordinate_name"])
            mapping = self._buffer(constants["mapping_name"])
            operations = constants["operations"]
            if not isinstance(operations, tuple):
                raise RuntimeError("invalid immutable operation metadata")
            for rotation_name, translation_name in operations:
                rotation = self._buffer(rotation_name)
                translation = self._buffer(translation_name)
                matrices.append(mapping @ coordinate @ rotation @ inverse)
                offsets.append((offset @ rotation + translation) @ inverse)
        self.register_buffer(
            "_expansion_matrix",
            torch.stack(matrices)
            if matrices
            else torch.zeros(0, self.intrinsic_dimension, 3, dtype=torch.float64),
            persistent=False,
        )
        self.register_buffer(
            "_expansion_offset",
            torch.stack(offsets) if offsets else torch.zeros(0, 3, dtype=torch.float64),
            persistent=False,
        )

    @property
    def atom_count(self) -> int:
        """Atoms the field sees: conventional, or primitive under ``primitive-v2``."""

        return int(self.atomic_numbers.numel())

    def primitive_basis(self, *, dtype: torch.dtype, device: torch.device) -> Tensor:
        """``P`` with ``L_field = P L`` (the identity under ``conventional-v1``)."""

        return self._primitive_basis_bits.view(torch.float64).to(dtype=dtype, device=device)

    def _to_primitive_cell(self, positions: Tensor, lattice: Tensor) -> tuple[Tensor, Tensor]:
        """``(x P^-1 mod 1, P L)`` for conventional positions ``[..., N, 3]`` and ``L``."""

        basis = self._primitive_basis_bits.view(torch.float64).to(
            dtype=lattice.dtype, device=lattice.device
        )
        inverse = self._primitive_inverse_bits.view(torch.float64).to(
            dtype=lattice.dtype, device=lattice.device
        )
        return torch.remainder(positions @ inverse, 1.0), basis @ lattice

    def _buffer(self, name: object) -> Tensor:
        if not isinstance(name, str):
            raise RuntimeError("invalid immutable orbit buffer reference")
        return getattr(self, name)

    @staticmethod
    def _validate_chart_buffer_state(
        module: Any,
        state: Any,
        prefix: str,
        local_metadata: Any,
        strict: bool,
        missing_keys: Any,
        unexpected_keys: Any,
        error_msgs: Any,
    ) -> None:
        """Forbid implicit casts or replacement of this layout's canonical bits."""

        del local_metadata, strict, missing_keys, unexpected_keys, error_msgs
        for name in ("_sqrt_reference_bits", "_invariant_basis_bits"):
            value = state.get(prefix + name)
            if value is None:  # Standard strict loading diagnoses missing keys.
                continue
            expected = module._buffer(name)
            if (
                not isinstance(value, Tensor)
                or value.dtype != torch.int64
                or value.shape != expected.shape
                or not torch.equal(value.to(device=expected.device), expected)
            ):
                raise RuntimeError(
                    f"checkpoint changes canonical float64 chart constant {name}"
                )

    @property
    def sqrt_reference(self) -> Tensor:
        """Device-resident canonical FP64 chart reference, regardless of master dtype."""

        return self._sqrt_reference_bits.view(torch.float64)

    @property
    def invariant_basis(self) -> Tensor:
        """Device-resident canonical FP64 invariant basis, without recasting loss."""

        return self._invariant_basis_bits.view(torch.float64)

    def _metric_from_latent(self, z: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        if z.ndim != 1 or z.shape[0] != self.lattice_dimension:
            raise ValueError("z has the wrong lattice-chart dimension")
        if z.dtype not in (torch.float32, torch.float64):
            raise ValueError(
                "lattice-chart states must use float32 or float64 master precision"
            )
        if z.device.type == "mps":
            raise DomainError(
                "the canonical float64 lattice-chart policy is unsupported on MPS"
            )
        # Matrix exponentiation of the invariant lattice chart is deterministic
        # crystallographic algebra.  It must never inherit an enclosing neural
        # AMP policy.
        with torch.autocast(device_type=z.device.type, enabled=False):
            sqrt_reference = self.sqrt_reference.to(device=z.device)
            invariant_basis = self.invariant_basis.to(device=z.device)
            # Preserve autograd through both casts. Matrix exponentiation is
            # smooth even at repeated eigenvalues; do not replace it with
            # differentiation through nonunique eigenvectors.
            chart_z = z.to(dtype=torch.float64)
            if self.chart_kind == "loewner-v2":
                latent_algebra = torch.einsum("i,ijk->jk", chart_z, invariant_basis)
                whitened, logarithm = loewner_whitened_metric(
                    latent_algebra, self.loewner_lower, self.loewner_upper
                )
                # Log-metric coefficients b: the bounded coordinates of this chart.
                bounded = torch.einsum("jk,ijk->i", logarithm, invariant_basis)
                metric = sqrt_reference @ whitened @ sqrt_reference
                return (
                    bounded.to(dtype=z.dtype),
                    (0.5 * (metric + metric.transpose(-1, -2))).to(dtype=z.dtype),
                    sqrt_reference.to(dtype=z.dtype),
                )
            bounded = (
                self.metric_bound * chart_z / torch.sqrt(1.0 + torch.dot(chart_z, chart_z))
            )
            algebra = torch.einsum("i,ijk->jk", bounded, invariant_basis)
            metric = sqrt_reference @ torch.matrix_exp(algebra) @ sqrt_reference
            return (
                bounded.to(dtype=z.dtype),
                (0.5 * (metric + metric.transpose(-1, -2))).to(dtype=z.dtype),
                sqrt_reference.to(dtype=z.dtype),
            )

    def lattice_pullback_metric(self, z: Tensor, *, create_graph: bool = False) -> Tensor:
        """Pull back the affine-invariant SPD metric to unbounded ``z``.

        If ``U_i = dG/dz_i``, this returns

        ``H_ij = tr(G^-1 U_i G^-1 U_j)``.

        It is the current-state lattice weight required by the CFM objective,
        not an endpoint-dependent surrogate.  ``create_graph=False`` is enough
        when paths are sampled independently of trainable field parameters.
        """

        if z.ndim != 1 or z.shape[0] != self.lattice_dimension:
            raise ValueError("z has the wrong lattice-chart dimension")
        if self.chart_kind == "loewner-v2" and not create_graph:
            return self._loewner_pullback_metric(z)
        return self._lattice_pullback_metric_autograd(z, create_graph=create_graph)

    def _loewner_pullback_metric(self, z: Tensor) -> Tensor:
        """Closed form ``H_ij = sum_ab Gamma_ab^2 M_i,ab M_j,ab / (w_a w_b)``.

        With ``G = S W S`` the trace is similarity invariant, so
        ``tr(G^-1 U_i G^-1 U_j) = tr(W^-1 dW_i W^-1 dW_j)``.  In the eigenbasis Q of
        Z, ``dW_i = Q (Gamma o M_i) Q^T`` with ``M_i = Q^T B_i Q`` (Daleckii--Krein),
        which gives the formula exactly, including at repeated eigenvalues.
        """

        with torch.no_grad(), torch.autocast(device_type=z.device.type, enabled=False):
            basis = self.invariant_basis.to(device=z.device)
            algebra = torch.einsum("i,ijk->jk", z.detach().to(dtype=torch.float64), basis)
            symmetric = 0.5 * (algebra + algebra.transpose(-1, -2))
            eigenvalues, vectors = torch.linalg.eigh(symmetric)
            lower, upper = self.loewner_lower, self.loewner_upper
            shifted = eigenvalues - math.log((upper - 1.0) / (1.0 - lower))
            weights = lower + (upper - lower) * 0.5 * (1.0 + torch.tanh(0.5 * shifted))
            gamma = _loewner_divided_differences(shifted, upper - lower)
            rotated = vectors.transpose(-1, -2) @ basis @ vectors
            kernel = gamma * gamma / (weights[:, None] * weights[None, :])
            pullback = torch.einsum("ab,iab,jab->ij", kernel, rotated, rotated)
            return (0.5 * (pullback + pullback.transpose(-1, -2))).to(dtype=z.dtype)

    def _lattice_pullback_metric_autograd(
        self, z: Tensor, *, create_graph: bool = False
    ) -> Tensor:
        """Reference pullback through autograd (every chart; one pass per entry)."""

        with torch.autocast(device_type=z.device.type, enabled=False):
            if not z.requires_grad:
                probe = z.detach().requires_grad_(True)
            else:
                probe = z

            def metric_only(value: Tensor) -> Tensor:
                return self._metric_from_latent(value)[1]

            jacobian = torch.autograd.functional.jacobian(
                metric_only,
                probe,
                create_graph=create_graph,
                # The Loewner map's custom backward is not vmap-registered; its
                # d <= 6 columns are cheap to evaluate one at a time.
                vectorize=self.chart_kind == "ball-v1",
            )
            metric = metric_only(probe)
            tangents = jacobian.permute(2, 0, 1)
            solved = torch.linalg.solve(metric, tangents)
            pullback = torch.einsum("iab,jba->ij", solved, solved)
            return 0.5 * (pullback + pullback.transpose(-1, -2))

    def lattice_pullback_metric_batch(self, z: Tensor) -> Tensor:
        """:meth:`lattice_pullback_metric` for ``z[B, d]``, without a second-order graph.

        The ``loewner-v2`` closed form is evaluated batched.  ``ball-v1`` falls back
        to one autograd pullback per row.
        """

        if z.ndim != 2 or z.shape[1] != self.lattice_dimension:
            raise ValueError("z must have shape [B, lattice dimension]")
        if self.chart_kind != "loewner-v2":
            return torch.stack(
                [self._lattice_pullback_metric_autograd(item) for item in z]
            ).detach()
        with torch.no_grad(), torch.autocast(device_type=z.device.type, enabled=False):
            basis = self.invariant_basis.to(device=z.device)
            algebra = torch.einsum("bi,ijk->bjk", z.detach().to(dtype=torch.float64), basis)
            symmetric = 0.5 * (algebra + algebra.transpose(-1, -2))
            eigenvalues, vectors = torch.linalg.eigh(symmetric)
            lower, upper = self.loewner_lower, self.loewner_upper
            shifted = eigenvalues - math.log((upper - 1.0) / (1.0 - lower))
            weights = lower + (upper - lower) * 0.5 * (1.0 + torch.tanh(0.5 * shifted))
            gamma = _loewner_divided_differences(shifted, upper - lower)
            rotated = torch.einsum("bpa,ipq,bqc->biac", vectors, basis, vectors)
            kernel = gamma * gamma / (weights[:, :, None] * weights[:, None, :])
            pullback = torch.einsum("bac,biac,bjac->bij", kernel, rotated, rotated)
            return (0.5 * (pullback + pullback.transpose(-1, -2))).to(dtype=z.dtype)

    def _metric_from_latent_batch(self, z: Tensor) -> tuple[Tensor, Tensor]:
        """Batched ``(bounded, metric)`` of :meth:`_metric_from_latent`, autocast disabled."""

        sqrt_reference = self.sqrt_reference.to(device=z.device)
        invariant_basis = self.invariant_basis.to(device=z.device)
        chart_z = z.to(dtype=torch.float64)
        if self.chart_kind == "loewner-v2":
            latent_algebra = torch.einsum("bi,ijk->bjk", chart_z, invariant_basis)
            whitened, logarithm = loewner_whitened_metric(
                latent_algebra, self.loewner_lower, self.loewner_upper
            )
            bounded = torch.einsum("bjk,ijk->bi", logarithm, invariant_basis)
            metric = sqrt_reference @ whitened @ sqrt_reference
        else:
            bounded = (
                self.metric_bound
                * chart_z
                / torch.sqrt(1.0 + (chart_z * chart_z).sum(dim=-1, keepdim=True))
            )
            algebra = torch.einsum("bi,ijk->bjk", bounded, invariant_basis)
            metric = sqrt_reference @ torch.matrix_exp(algebra) @ sqrt_reference
        return (
            bounded.to(dtype=z.dtype),
            (0.5 * (metric + metric.transpose(-1, -2))).to(dtype=z.dtype),
        )

    def forward_batch(self, z: Tensor, eta: Tensor) -> TorchExpandedGeometry:
        """:meth:`forward` for ``z[B, d]`` and ``eta[B, k]`` in one pass.

        Every field gains a leading batch axis, except ``lattice_commutant_basis``,
        which is a cover constant.  The values equal stacked :meth:`forward` calls up
        to rounding (``tests/test_nn_bridge.py``).  The production objectives keep the
        per-record :meth:`forward`.
        """

        if z.ndim != 2 or z.shape[1] != self.lattice_dimension:
            raise ValueError("z must have shape [B, lattice dimension]")
        if eta.ndim != 2 or eta.shape != (z.shape[0], self.intrinsic_dimension):
            raise ValueError("eta must have shape [B, intrinsic dimension]")
        if z.dtype != eta.dtype or z.device != eta.device:
            raise ValueError("z and eta must share dtype and device")
        if z.dtype not in (torch.float32, torch.float64):
            raise ValueError(
                "lattice-chart states must use float32 or float64 master precision"
            )
        if z.device.type == "mps":
            raise DomainError(
                "the canonical float64 lattice-chart policy is unsupported on MPS"
            )
        batch = z.shape[0]
        with torch.autocast(device_type=z.device.type, enabled=False):
            bounded, metric = self._metric_from_latent_batch(z)
            lattice = torch.linalg.cholesky(metric)
            period_basis = self.period_basis.to(dtype=eta.dtype, device=eta.device)
            canonical_eta = canonicalize_torus_validated(eta, period_basis)
            fractional: list[Tensor] = []
            jacobian: list[Tensor] = []
            for constants in self._orbit_constants:
                offset = self._buffer(constants["offset_name"]).to(
                    dtype=eta.dtype, device=eta.device
                )
                coordinate = self._buffer(constants["coordinate_name"]).to(
                    dtype=eta.dtype, device=eta.device
                )
                mapping = self._buffer(constants["mapping_name"]).to(
                    dtype=eta.dtype, device=eta.device
                )
                representative = torch.remainder(
                    offset + canonical_eta @ mapping @ coordinate, 1.0
                )
                operations = constants["operations"]
                if not isinstance(operations, tuple):
                    raise RuntimeError("invalid immutable operation metadata")
                for rotation_name, translation_name in operations:
                    rotation = self._buffer(rotation_name).to(
                        dtype=eta.dtype, device=eta.device
                    )
                    translation = self._buffer(translation_name).to(
                        dtype=eta.dtype, device=eta.device
                    )
                    fractional.append(
                        torch.remainder(representative @ rotation + translation, 1.0)
                    )
                    fractional_block = (mapping @ coordinate @ rotation).transpose(0, 1)
                    jacobian.append(lattice.transpose(-1, -2) @ fractional_block)
            positions = torch.stack(fractional, dim=1)
            if self.expansion_cell == "primitive-v2":
                positions, lattice = self._to_primitive_cell(positions, lattice)
            return TorchExpandedGeometry(
                bounded_lattice_coordinates=bounded,
                metric=metric,
                lattice=lattice,
                intrinsic_coordinates=canonical_eta,
                fractional_coordinates=positions,
                atomic_numbers=self.atomic_numbers.to(device=z.device).expand(batch, -1),
                atom_features=self.atom_features.to(dtype=z.dtype, device=z.device).expand(
                    batch, -1, -1
                ),
                jacobian_cartesian=torch.stack(jacobian, dim=1),
                lattice_commutant_basis=torch.einsum(
                    "kab,jab->kj",
                    self._canonical_sym3_bits.view(torch.float64),
                    self._invariant_basis_bits.view(torch.float64),
                ).to(dtype=z.dtype, device=z.device),
            )

    def forward(self, z: Tensor, eta: Tensor) -> TorchExpandedGeometry:
        if eta.ndim != 1 or eta.shape[0] != self.intrinsic_dimension:
            raise ValueError("eta has the wrong intrinsic-cover dimension")
        if z.dtype != eta.dtype or z.device != eta.device:
            raise ValueError("z and eta must share dtype and device")
        if not torch.is_floating_point(z):
            raise ValueError("continuous states must use floating tensors")
        with torch.autocast(device_type=z.device.type, enabled=False):
            return self._forward_master(z, eta)

    def _forward_master(self, z: Tensor, eta: Tensor) -> TorchExpandedGeometry:
        """Expand one state with autocast already disabled by :meth:`forward`."""

        bounded, metric, _ = self._metric_from_latent(z)
        lattice = torch.linalg.cholesky(metric)
        period_basis = self.period_basis.to(dtype=eta.dtype, device=eta.device)
        canonical_eta = canonicalize_torus_validated(eta, period_basis)

        fractional: list[Tensor] = []
        jacobian: list[Tensor] = []
        for constants in self._orbit_constants:
            offset = self._buffer(constants["offset_name"]).to(
                dtype=eta.dtype, device=eta.device
            )
            coordinate = self._buffer(constants["coordinate_name"]).to(
                dtype=eta.dtype, device=eta.device
            )
            mapping = self._buffer(constants["mapping_name"]).to(
                dtype=eta.dtype, device=eta.device
            )
            xi = canonical_eta @ mapping
            representative = torch.remainder(offset + xi @ coordinate, 1.0)
            operations = constants["operations"]
            if not isinstance(operations, tuple):
                raise RuntimeError("invalid immutable operation metadata")
            for rotation_name, translation_name in operations:
                rotation = self._buffer(rotation_name).to(dtype=eta.dtype, device=eta.device)
                translation = self._buffer(translation_name).to(
                    dtype=eta.dtype, device=eta.device
                )
                fractional.append(torch.remainder(representative @ rotation + translation, 1.0))
                fractional_block = (mapping @ coordinate @ rotation).transpose(0, 1)
                jacobian.append(lattice.transpose(0, 1) @ fractional_block)
        positions = torch.stack(fractional)
        if self.expansion_cell == "primitive-v2":
            positions, lattice = self._to_primitive_cell(positions, lattice)
        return TorchExpandedGeometry(
            bounded_lattice_coordinates=bounded,
            metric=metric,
            lattice=lattice,
            intrinsic_coordinates=canonical_eta,
            fractional_coordinates=positions,
            atomic_numbers=self.atomic_numbers.to(device=z.device),
            atom_features=self.atom_features.to(dtype=z.dtype, device=z.device),
            jacobian_cartesian=torch.stack(jacobian),
            lattice_commutant_basis=torch.einsum(
                "kab,jab->kj",
                self._canonical_sym3_bits.view(torch.float64),
                self._invariant_basis_bits.view(torch.float64),
            ).to(dtype=z.dtype, device=z.device),
        )


@dataclass(frozen=True)
class PaddedExpandedGeometry:
    """A mixed-cover batch expanded with its bridges' fused constants (review H2).

    ``fractional_coordinates[B, N, 3]``, ``jacobian_cartesian[B, N, 3, D]``,
    ``atomic_numbers[B, N]``, ``atom_mask[B, N]`` and ``atom_features[B, N, F]``
    are padded to the widest crystal; padded atom slots repeat atom 0's
    position (finite data for masked attention) and have zero Jacobian and
    atomic number 0.  ``lattice[B, 3, 3]`` is each crystal's expansion-cell
    basis (``P L`` under ``primitive-v2``) and ``metric[B, 3, 3]`` the chart's
    conventional metric.  The per-crystal ``bounded_lattice_coordinates`` and
    ``lattice_commutant_basis`` keep their own widths.
    """

    bounded_lattice_coordinates: tuple[Tensor, ...]
    metric: Tensor
    lattice: Tensor
    fractional_coordinates: Tensor
    jacobian_cartesian: Tensor
    atomic_numbers: Tensor
    atom_mask: Tensor
    atom_features: Tensor
    lattice_commutant_basis: tuple[Tensor, ...]


def expand_padded(
    bridges: Sequence[TorchCoverGeometry],
    z_values: Sequence[Tensor],
    eta_values: Sequence[Tensor],
) -> PaddedExpandedGeometry:
    """Expand crystals of different covers in two ``einsum`` calls (review H2).

    Each crystal's metric comes from its own chart (one small call per
    crystal); the orbit-and-operation expansion, which the per-record bridge
    runs as a Python loop over every coset operation, is done once for the
    whole batch with the bridges' fused constants.  Values equal the
    per-record :meth:`TorchCoverGeometry.forward` to rounding
    (``tests/test_padded_expansion.py``).
    """

    batch = len(bridges)
    if batch == 0 or len(z_values) != batch or len(eta_values) != batch:
        raise ValueError("expand_padded needs one bridge, z and eta per crystal")
    for bridge, z, eta in zip(bridges, z_values, eta_values, strict=True):
        if not isinstance(bridge, TorchCoverGeometry):
            raise TypeError("bridges must be TorchCoverGeometry modules")
        if z.ndim != 1 or z.shape[0] != bridge.lattice_dimension:
            raise ValueError("z has the wrong lattice-chart dimension")
        if eta.ndim != 1 or eta.shape[0] != bridge.intrinsic_dimension:
            raise ValueError("eta has the wrong intrinsic-cover dimension")
        if z.dtype != eta.dtype or z.device != eta.device:
            raise ValueError("z and eta must share dtype and device")
        if z.dtype != z_values[0].dtype or z.device != z_values[0].device:
            raise ValueError("every crystal must share one dtype and device")
    dtype, device = z_values[0].dtype, z_values[0].device
    width_atoms = max(int(bridge.atomic_numbers.numel()) for bridge in bridges)
    width_eta = max(bridge.intrinsic_dimension for bridge in bridges)
    feature_dim = int(bridges[0].atom_features.shape[-1])
    with torch.autocast(device_type=device.type, enabled=False):
        bounded: list[Tensor] = []
        metrics: list[Tensor] = []
        etas: list[Tensor] = []
        matrices = torch.zeros(batch, width_atoms, width_eta, 3, dtype=dtype, device=device)
        offsets = torch.zeros(batch, width_atoms, 3, dtype=dtype, device=device)
        mask = torch.zeros(batch, width_atoms, dtype=torch.bool, device=device)
        numbers = torch.zeros(batch, width_atoms, dtype=torch.long, device=device)
        features = torch.zeros(batch, width_atoms, feature_dim, dtype=dtype, device=device)
        commutants: list[Tensor] = []
        for b, (bridge, z, eta) in enumerate(zip(bridges, z_values, eta_values, strict=True)):
            bounded_b, metric_b, _ = bridge._metric_from_latent(z)
            bounded.append(bounded_b)
            metrics.append(metric_b)
            period = bridge.period_basis.to(dtype=dtype, device=device)
            canonical = canonicalize_torus_validated(eta, period)
            etas.append(torch.nn.functional.pad(canonical, (0, width_eta - canonical.numel())))
            n, k = int(bridge.atomic_numbers.numel()), bridge.intrinsic_dimension
            matrices[b, :n, :k] = bridge._expansion_matrix.to(dtype=dtype, device=device)
            offsets[b, :n] = bridge._expansion_offset.to(dtype=dtype, device=device)
            mask[b, :n] = True
            numbers[b, :n] = bridge.atomic_numbers.to(device=device)
            features[b, :n] = bridge.atom_features.to(dtype=dtype, device=device)
            commutants.append(
                torch.einsum(
                    "kab,jab->kj",
                    bridge._canonical_sym3_bits.view(torch.float64),
                    bridge._invariant_basis_bits.view(torch.float64),
                ).to(dtype=dtype, device=device)
            )
        metric = torch.stack(metrics)
        lattice = torch.linalg.cholesky(metric)
        if any(bridge.expansion_cell == "primitive-v2" for bridge in bridges):
            bases = torch.stack(
                [bridge.primitive_basis(dtype=dtype, device=device) for bridge in bridges]
            )
            lattice = bases @ lattice
        eta_padded = torch.stack(etas)
        fractional = torch.remainder(
            torch.einsum("bk,bnkc->bnc", eta_padded, matrices) + offsets, 1.0
        )
        fractional = torch.where(mask[:, :, None], fractional, fractional[:, :1, :])
        jacobian = torch.einsum("bdc,bnkd->bnck", lattice, matrices)
    return PaddedExpandedGeometry(
        bounded_lattice_coordinates=tuple(bounded),
        metric=metric,
        lattice=lattice,
        fractional_coordinates=fractional,
        jacobian_cartesian=jacobian,
        atomic_numbers=numbers,
        atom_mask=mask,
        atom_features=features,
        lattice_commutant_basis=tuple(commutants),
    )


@dataclass(frozen=True)
class PaddedCoverExpansion:
    """One state of a :class:`PaddedCoverPlan`, every field with a leading batch axis.

    ``bounded_lattice_coordinates[B, Z]`` and ``intrinsic_coordinates[B, E]`` are
    zero in padded slots.  ``lattice[B, 3, 3]`` is each crystal's expansion-cell
    basis (``P L`` under ``primitive-v2``), ``metric[B, 3, 3]`` the chart's
    conventional metric.  Padded atoms of ``fractional_coordinates[B, N, 3]``
    repeat atom 0 and have zero ``jacobian_cartesian[B, N, 3, E]``.
    """

    bounded_lattice_coordinates: Tensor
    metric: Tensor
    lattice: Tensor
    intrinsic_coordinates: Tensor
    fractional_coordinates: Tensor
    jacobian_cartesian: Tensor


class PaddedCoverPlan:
    """State-independent constants of one mixed-cover ODE solve, padded once.

    :func:`expand_padded` rebuilds its padded constants and runs one chart call
    per crystal on every invocation, which is right for a training step whose
    batch changes every time.  An ODE solve expands the same ``B`` covers at
    every Runge--Kutta stage, so this plan pads the constants once and
    :meth:`expand` is then a fixed number of batched tensor operations with no
    Python loop over crystals, orbits or coset operations:

    * crystal ``b`` owns the leading ``lattice_dimensions[b]`` slots of
      ``z[b]``, the leading ``intrinsic_dimensions[b]`` slots of ``eta[b]`` and
      the atoms where ``atom_mask[b]`` is true;
    * ``period_basis[B, E, E]`` is each cover's period basis completed by the
      identity, so a padded coordinate stays at zero under the row-wise torus
      canonicalization of :func:`crystalatlas.flow.canonicalize_torus_validated`;
    * the invariant chart basis is completed by zero matrices, which add
      nothing to the chart algebra, and the chart keeps the canonical float64
      policy of :meth:`TorchCoverGeometry._metric_from_latent`.

    Values equal the per-record :meth:`TorchCoverGeometry.forward` to rounding
    (``tests/test_mixed_sampler.py``).
    """

    def __init__(
        self,
        bridges: Sequence[TorchCoverGeometry],
        *,
        dtype: torch.dtype,
        device: torch.device,
    ) -> None:
        if len(bridges) == 0:
            raise ValueError("a padded cover plan needs at least one bridge")
        if any(not isinstance(bridge, TorchCoverGeometry) for bridge in bridges):
            raise TypeError("bridges must be TorchCoverGeometry modules")
        if dtype not in (torch.float32, torch.float64):
            raise ValueError(
                "lattice-chart states must use float32 or float64 master precision"
            )
        device = torch.device(device)
        if device.type == "mps":
            raise DomainError(
                "the canonical float64 lattice-chart policy is unsupported on MPS"
            )
        batch = len(bridges)
        lattice_dims = [bridge.lattice_dimension for bridge in bridges]
        intrinsic_dims = [bridge.intrinsic_dimension for bridge in bridges]
        atom_counts = [bridge.atom_count for bridge in bridges]
        if min(atom_counts) < 1:
            raise ValueError("every padded crystal must keep at least one atom")
        width_z, width_eta, width_atoms = (
            max(lattice_dims),
            max(intrinsic_dims),
            max(atom_counts),
        )
        feature_dim = TorchCoverGeometry.atom_feature_dim
        self.batch = batch
        self.dtype = dtype
        self.device = device
        self.lattice_width = width_z
        self.intrinsic_width = width_eta
        self.atom_width = width_atoms
        self.atom_counts = tuple(atom_counts)
        self.lattice_dimensions = torch.tensor(lattice_dims, dtype=torch.long, device=device)
        self.intrinsic_dimensions = torch.tensor(
            intrinsic_dims, dtype=torch.long, device=device
        )
        self._lattice_columns = (
            torch.arange(width_z, device=device)[None, :] < self.lattice_dimensions[:, None]
        ).to(dtype=dtype)
        self._intrinsic_columns = (
            torch.arange(width_eta, device=device)[None, :] < self.intrinsic_dimensions[:, None]
        ).to(dtype=dtype)
        matrices = torch.zeros(batch, width_atoms, width_eta, 3, dtype=dtype, device=device)
        offsets = torch.zeros(batch, width_atoms, 3, dtype=dtype, device=device)
        mask = torch.zeros(batch, width_atoms, dtype=torch.bool, device=device)
        numbers = torch.zeros(batch, width_atoms, dtype=torch.long, device=device)
        features = torch.zeros(batch, width_atoms, feature_dim, dtype=dtype, device=device)
        commutant = torch.zeros(batch, 6, width_z, dtype=dtype, device=device)
        period = torch.eye(width_eta, dtype=dtype, device=device).repeat(batch, 1, 1)
        sqrt_reference = torch.zeros(batch, 3, 3, dtype=torch.float64, device=device)
        invariant_basis = torch.zeros(batch, width_z, 3, 3, dtype=torch.float64, device=device)
        charts: dict[tuple[object, ...], list[int]] = {}
        for b, bridge in enumerate(bridges):
            n, k, d = atom_counts[b], intrinsic_dims[b], lattice_dims[b]
            matrices[b, :n, :k] = bridge._expansion_matrix.to(dtype=dtype, device=device)
            offsets[b, :n] = bridge._expansion_offset.to(dtype=dtype, device=device)
            mask[b, :n] = True
            numbers[b, :n] = bridge.atomic_numbers.to(device=device)
            features[b, :n] = bridge.atom_features.to(dtype=dtype, device=device)
            period[b, :k, :k] = bridge.period_basis.to(dtype=dtype, device=device)
            sqrt_reference[b] = bridge.sqrt_reference.to(device=device)
            basis = bridge.invariant_basis.to(device=device)
            invariant_basis[b, :d] = basis
            commutant[b, :, :d] = torch.einsum(
                "kab,jab->kj",
                bridge._canonical_sym3_bits.view(torch.float64).to(device=device),
                basis,
            ).to(dtype=dtype)
            key: tuple[object, ...] = (
                ("loewner-v2", bridge.loewner_lower, bridge.loewner_upper)
                if bridge.chart_kind == "loewner-v2"
                else ("ball-v1", bridge.metric_bound)
            )
            charts.setdefault(key, []).append(b)
        self.atom_mask = mask
        self.atomic_numbers = numbers
        self.atom_features = features
        self.lattice_commutant_basis = commutant
        self.period_basis = period
        self._matrices = matrices
        self._offsets = offsets
        self._sqrt_reference = sqrt_reference
        self._invariant_basis = invariant_basis
        # One batched chart call per distinct chart; a model has one in practice.
        self._charts: tuple[tuple[tuple[object, ...], Tensor | None], ...] = tuple(
            (
                key,
                None
                if len(charts) == 1
                else torch.tensor(rows, dtype=torch.long, device=device),
            )
            for key, rows in charts.items()
        )
        self._primitive_bases = (
            torch.stack(
                [bridge.primitive_basis(dtype=dtype, device=device) for bridge in bridges]
            )
            if any(bridge.expansion_cell == "primitive-v2" for bridge in bridges)
            else None
        )

    def _chart(
        self, key: tuple[object, ...], z: Tensor, rows: Tensor | None
    ) -> tuple[Tensor, Tensor]:
        """Float64 ``(bounded, metric)`` of one chart for its rows of ``z``."""

        sqrt_reference = self._sqrt_reference
        invariant_basis = self._invariant_basis
        if rows is not None:
            z = z[rows]
            sqrt_reference = sqrt_reference[rows]
            invariant_basis = invariant_basis[rows]
        if key[0] == "loewner-v2":
            _, lower, upper = key
            if not isinstance(lower, float) or not isinstance(upper, float):
                raise RuntimeError("invalid immutable Loewner chart metadata")
            latent_algebra = torch.einsum("bi,bijk->bjk", z, invariant_basis)
            whitened, logarithm = loewner_whitened_metric(latent_algebra, lower, upper)
            bounded = torch.einsum("bjk,bijk->bi", logarithm, invariant_basis)
            metric = sqrt_reference @ whitened @ sqrt_reference
        else:
            bound = key[1]
            if not isinstance(bound, float):
                raise RuntimeError("invalid immutable ball chart metadata")
            bounded = bound * z / torch.sqrt(1.0 + (z * z).sum(dim=-1, keepdim=True))
            algebra = torch.einsum("bi,bijk->bjk", bounded, invariant_basis)
            metric = sqrt_reference @ torch.matrix_exp(algebra) @ sqrt_reference
        return bounded, 0.5 * (metric + metric.transpose(-1, -2))

    def expand(self, z: Tensor, eta: Tensor) -> PaddedCoverExpansion:
        """Expand padded states ``z[B, Z]`` and ``eta[B, E]`` of the plan's covers.

        Padded slots of ``z`` and ``eta`` are masked first, so their values
        cannot leak into any crystal.
        """

        if z.shape != (self.batch, self.lattice_width):
            raise ValueError("z must have shape [B, widest lattice dimension]")
        if eta.shape != (self.batch, self.intrinsic_width):
            raise ValueError("eta must have shape [B, widest intrinsic dimension]")
        if z.dtype != self.dtype or eta.dtype != self.dtype:
            raise ValueError("z and eta must use the plan's master dtype")
        if z.device != self.device or eta.device != self.device:
            raise ValueError("z and eta must live on the plan's device")
        # Chart, Cholesky factor and torus wrap are deterministic crystallographic
        # algebra and never inherit an enclosing neural AMP policy.
        with torch.autocast(device_type=self.device.type, enabled=False):
            chart_z = (z * self._lattice_columns).to(dtype=torch.float64)
            if len(self._charts) == 1:
                bounded64, metric64 = self._chart(self._charts[0][0], chart_z, None)
            else:
                bounded64 = torch.zeros_like(chart_z)
                metric64 = chart_z.new_zeros(self.batch, 3, 3)
                for key, rows in self._charts:
                    if rows is None:
                        raise RuntimeError("invalid immutable chart row metadata")
                    bounded_rows, metric_rows = self._chart(key, chart_z, rows)
                    bounded64 = bounded64.index_copy(0, rows, bounded_rows)
                    metric64 = metric64.index_copy(0, rows, metric_rows)
            bounded = bounded64.to(dtype=self.dtype)
            metric = metric64.to(dtype=self.dtype)
            lattice = torch.linalg.cholesky(metric)
            if self._primitive_bases is not None:
                lattice = self._primitive_bases @ lattice
            canonical_eta = canonicalize_torus_validated(
                eta * self._intrinsic_columns, self.period_basis
            )
            fractional = torch.remainder(
                torch.einsum("bk,bnkc->bnc", canonical_eta, self._matrices) + self._offsets,
                1.0,
            )
            fractional = torch.where(
                self.atom_mask[:, :, None], fractional, fractional[:, :1, :]
            )
            jacobian = torch.einsum("bdc,bnkd->bnck", lattice, self._matrices)
        return PaddedCoverExpansion(
            bounded_lattice_coordinates=bounded,
            metric=metric,
            lattice=lattice,
            intrinsic_coordinates=canonical_eta,
            fractional_coordinates=fractional,
            jacobian_cartesian=jacobian,
        )


def lattice_pullback_metrics(
    bridges: Sequence[TorchCoverGeometry], z_values: Sequence[Tensor]
) -> tuple[Tensor, ...]:
    """Detached current-state lattice metrics, batched per shared bridge (review H2)."""

    if len(bridges) != len(z_values):
        raise ValueError("one z per bridge is required")
    groups: dict[int, list[int]] = {}
    for index, bridge in enumerate(bridges):
        groups.setdefault(id(bridge), []).append(index)
    results: list[Tensor | None] = [None] * len(bridges)
    for indices in groups.values():
        bridge = bridges[indices[0]]
        if len(indices) == 1:
            results[indices[0]] = bridge.lattice_pullback_metric(z_values[indices[0]]).detach()
            continue
        stacked = bridge.lattice_pullback_metric_batch(
            torch.stack([z_values[index].detach() for index in indices])
        )
        for row, index in enumerate(indices):
            results[index] = stacked[row]
    return tuple(item for item in results if item is not None)


class CoverConditionedGeometryField(nn.Module):
    """Bind one shared neural field to deterministic cover expansion constants."""

    def __init__(
        self,
        geometry: TorchCoverGeometry,
        field: IntrinsicGeometryField,
    ) -> None:
        super().__init__()
        if field.config.atom_feature_dim != geometry.atom_feature_dim:
            raise ValueError(
                f"field expects {field.config.atom_feature_dim} atom features; "
                f"cover bridge provides {geometry.atom_feature_dim}"
            )
        if field.config.backbone == "equivariant_local":
            certified_edge_bound = geometry.layout.periodic_edge_upper_bound(
                field.config.cutoff
            )
            certified_pair_bound = geometry.layout.periodic_candidate_box_upper_bound(
                field.config.cutoff
            )
            if certified_pair_bound > field.config.max_candidates_per_pair:
                raise DomainError(
                    "cover/cutoff support can exceed max_candidates_per_pair "
                    f"({certified_pair_bound} > {field.config.max_candidates_per_pair}); "
                    "tighten the metric support or raise the explicit budget"
                )
            if certified_edge_bound > field.config.max_periodic_edges:
                raise DomainError(
                    "cover/cutoff support can exceed the Transformer's "
                    "max_periodic_edges "
                    f"({certified_edge_bound} > {field.config.max_periodic_edges}); "
                    "tighten the metric support or raise the explicit budget"
                )
        self.geometry = geometry
        self.field = field
        composition = geometry.layout.cover.composition
        self.register_buffer(
            "composition_atomic_numbers",
            torch.as_tensor(composition.support, dtype=torch.long),
        )
        self.register_buffer(
            "composition_counts",
            torch.as_tensor(composition.coefficients, dtype=torch.float64),
        )

    def forward(self, time: Tensor, z: Tensor, eta: Tensor) -> IntrinsicFieldOutput:
        expanded = self.geometry(z, eta)
        context = self.field.encode_composition(
            self.composition_atomic_numbers.to(device=z.device),
            self.composition_counts.to(dtype=z.dtype, device=z.device),
        )
        return self.field(
            time,
            z,
            eta,
            context,
            atomic_numbers=expanded.atomic_numbers,
            atom_features=expanded.atom_features,
            fractional_coordinates=expanded.fractional_coordinates,
            lattice=expanded.lattice,
            bounded_lattice_coordinates=expanded.bounded_lattice_coordinates,
            jacobian_cartesian=expanded.jacobian_cartesian,
            geometry_is_certified=True,
            lattice_commutant_basis=expanded.lattice_commutant_basis,
        )


def intrinsic_noise_covariance(
    bridge: TorchCoverGeometry,
    z: Tensor,
    eta: Tensor,
    sigma: float,
    *,
    reference_length: float,
) -> Tensor:
    """``sigma^2 (J_hat^T P_N J_hat)^-1`` at ``(z, eta)``, in float64 (review C1).

    This is the torus covariance of the ``cartesian-v2`` dequantization kernel.
    Gaussian noise with this covariance has translation-quotient Cartesian RMS
    displacement ``sigma`` per free coordinate, in units of ``reference_length``,
    for every cell shape: ``tr(M Sigma) = sigma^2 d`` with ``M`` the quotient
    metric of the loss.  The legacy ``fractional-v1`` kernel is ``sigma^2 I`` in
    the stored fractional coordinates, which is anisotropic in space for any
    non-cubic cell.  The bridge may hold float32 buffers; the chart constants are
    canonical float64 bits and the integer orbit data are exact, so evaluating it
    on float64 states gives the float64 Jacobian.
    """

    if not (math.isfinite(sigma) and sigma > 0.0):
        raise DomainError("the dequantization scale must be finite and positive")
    if not (math.isfinite(reference_length) and reference_length > 0.0):
        raise DomainError("reference_length must be finite and positive")
    with torch.no_grad(), torch.autocast(device_type=z.device.type, enabled=False):
        if eta.shape[-1] == 0:
            return torch.zeros((0, 0), dtype=torch.float64, device=z.device)
        expanded = bridge(
            z.detach().to(dtype=torch.float64), eta.detach().to(dtype=torch.float64)
        )
        # (1.30d) uses the conventional-cell metric; a primitive expansion holds
        # each atom once instead of ``metric_copies`` times (review M3).
        metric = bridge.metric_copies * translation_quotient_normal_matrix(
            expanded.jacobian_cartesian.to(dtype=torch.float64),
            reference_length=reference_length,
        )
        covariance = (sigma * sigma) * torch.linalg.inv(metric)
        return 0.5 * (covariance + covariance.transpose(-1, -2))


def intrinsic_noise_transform(covariance: Tensor) -> Tensor:
    """``T`` with ``T^T T = covariance``, so ``epsilon @ T`` has that covariance."""

    if covariance.shape[-1] == 0:
        return covariance.clone()
    return torch.linalg.cholesky(covariance).transpose(-1, -2)


__all__ = [
    "CoverConditionedGeometryField",
    "TorchCoverGeometry",
    "TorchExpandedGeometry",
    "geometry_bridge_numerical_contract",
    "intrinsic_noise_covariance",
    "intrinsic_noise_transform",
    "PaddedCoverExpansion",
    "PaddedCoverPlan",
    "PaddedExpandedGeometry",
    "expand_padded",
    "lattice_pullback_metrics",
    "EXPANSION_CELLS",
]
