"""Expand--perceive--reduce intrinsic crystal velocity field.

Exact orbit expansion, lattice-chart decoding, and the advertised-slot
Jacobian are crystallographic operations and remain outside this neural module.
The field consumes their current-state outputs, applies the configured periodic
Transformer, and performs the exact no-ridge metric-adjoint reduction in
equations (3.9)--(3.10).  The default backbone has scalar learned hidden
channels, dense all-pairs periodic attention, and a deterministic covariant
readout; the former sparse local equivariant backbone is an explicit ablation.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from functools import partial

from .._optional import require_torch
from .composition import CompositionEncoder, make_element_embedding
from .equivariant_transformer import (
    PeriodicEdges,
    PeriodicSE3Transformer,
    PeriodicTransformerConfig,
    VectorLinear,
    enumerate_packed_periodic_edges,
    enumerate_periodic_edges,
)
from .host_sync import DeviceChecks, check
from .invariant_transformer import (
    InvariantPeriodicTransformer,
    InvariantPeriodicTransformerConfig,
)

torch = require_torch()
Tensor = torch.Tensor
nn = torch.nn


class TangentRankError(RuntimeError):
    """Raised when a supplied advertised-slot Jacobian violates its certificate."""


def _transformer_edges(edges: PeriodicEdges, master: Tensor) -> PeriodicEdges:
    """Return an activation-precision view of certified master-precision edges.

    Edge membership and displacements are constructed in master precision.
    The Transformer is a learned map, however, so its private radial/directional
    feature copy must match the active AMP dtype (``torch.cross`` in particular
    requires identical operand dtypes).  The master edge list remains the
    public/certified value returned by the packed field.
    """

    # Autocast intentionally leaves Float64 operations alone.  Mirror that
    # policy here so a Float64 research run never receives a forced BF16 edge
    # operand merely because an outer context was enabled.
    dtype = master.dtype
    if master.dtype != torch.float64 and torch.is_autocast_enabled(master.device.type):
        dtype = torch.get_autocast_dtype(master.device.type)
    if edges.displacement.dtype == dtype:
        return edges
    return PeriodicEdges(
        destination=edges.destination,
        source=edges.source,
        shifts=edges.shifts,
        displacement=edges.displacement.to(dtype=dtype),
        squared_distance=edges.squared_distance.to(dtype=dtype),
    )


def translation_quotient_normal_matrix(
    jacobian_cartesian: Tensor,
    *,
    reference_length: float = 1.0,
    checks: DeviceChecks | None = None,
) -> Tensor:
    """Return ``J_hat^T P_N J_hat`` from equation (3.9).

    ``jacobian_cartesian[n, :, j]`` is the Cartesian velocity of expanded atom
    ``n`` induced by unit intrinsic velocity ``j``.  No atoms are deduplicated,
    including at an advertised-slot collision.  ``checks`` defers the
    finiteness read to the caller's single synchronization (review H1).
    """

    if jacobian_cartesian.ndim != 3 or jacobian_cartesian.shape[1] != 3:
        raise ValueError("jacobian_cartesian must have shape [N,3,d_eta]")
    if jacobian_cartesian.shape[0] == 0:
        raise ValueError("the advertised expanded cell must contain atoms")
    if (
        isinstance(reference_length, bool)
        or not isinstance(reference_length, (int, float))
        or not math.isfinite(float(reference_length))
        or float(reference_length) <= 0.0
    ):
        raise ValueError("reference_length must be finite and positive")
    if not jacobian_cartesian.is_floating_point() or jacobian_cartesian.dtype not in {
        torch.float32,
        torch.float64,
    }:
        raise ValueError("jacobian_cartesian must use float32 or float64 master precision")
    check(
        checks,
        ~torch.isfinite(jacobian_cartesian).all(),
        partial(ValueError, "jacobian_cartesian entries must be finite"),
    )
    # This small matrix is crystallographic algebra, not a learned activation.
    # Always accumulate it in Float64: normal equations square the Jacobian's
    # condition number, so Float32 accumulation is not an acceptable rank or
    # metric contract even when the neural master parameters are Float32.
    with torch.autocast(device_type=jacobian_cartesian.device.type, enabled=False):
        jacobian = jacobian_cartesian.to(dtype=torch.float64) / reference_length
        centered = jacobian - jacobian.mean(dim=0, keepdim=True)
        return torch.einsum("nci,ncj->ij", centered, centered)


def reduce_cartesian_to_intrinsic(
    cartesian_vectors: Tensor,
    jacobian_cartesian: Tensor,
    *,
    reference_length: float = 1.0,
    check_rank: bool = True,
    maximum_condition_number_float32: float = 1.0e4,
    maximum_condition_number_float64: float = 1.0e6,
    checks: DeviceChecks | None = None,
) -> Tensor:
    """Solve the exact translation-quotient least-squares problem (3.10).

    The Cartesian network vectors and ``J / reference_length`` are both
    dimensionless.  A singular normal matrix is a failed tangent-kernel
    certificate and raises ``TangentRankError``; no numerical ridge is added.
    With ``checks`` the finiteness, rank and definiteness flags join the
    caller's single host read (review H1); without it they are read here.
    """

    if cartesian_vectors.ndim != 2 or cartesian_vectors.shape[-1] != 3:
        raise ValueError("cartesian_vectors must have shape [N,3]")
    if jacobian_cartesian.ndim != 3 or jacobian_cartesian.shape[1] != 3:
        raise ValueError("jacobian_cartesian must have shape [N,3,d_eta]")
    if jacobian_cartesian.shape[:2] != cartesian_vectors.shape:
        raise ValueError("vectors and Jacobian must describe the same atoms")
    if (
        isinstance(reference_length, bool)
        or not isinstance(reference_length, (int, float))
        or not math.isfinite(float(reference_length))
        or float(reference_length) <= 0.0
    ):
        raise ValueError("reference_length must be finite and positive")
    if not isinstance(check_rank, bool):
        raise ValueError("check_rank must be boolean")
    for name, value in (
        ("maximum_condition_number_float32", maximum_condition_number_float32),
        ("maximum_condition_number_float64", maximum_condition_number_float64),
    ):
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or float(value) < 1.0
        ):
            raise ValueError(f"{name} must be finite and at least one")
    if cartesian_vectors.device != jacobian_cartesian.device:
        raise ValueError("vectors and Jacobian must share one device")
    if not cartesian_vectors.is_floating_point() or not jacobian_cartesian.is_floating_point():
        raise ValueError("vectors and Jacobian must be floating-point tensors")
    if cartesian_vectors.dtype != jacobian_cartesian.dtype:
        raise ValueError("vectors and Jacobian must share one master dtype")
    if cartesian_vectors.dtype not in {torch.float32, torch.float64}:
        raise ValueError("vectors and Jacobian must use float32 or float64 master precision")
    own = checks is None
    if checks is None:
        checks = DeviceChecks()
    checks.add(
        ~(torch.isfinite(cartesian_vectors).all() & torch.isfinite(jacobian_cartesian).all()),
        partial(ValueError, "vectors and Jacobian entries must be finite"),
    )
    if own:
        checks.raise_first()  # keep non-finite data out of the LAPACK calls below
    intrinsic_dim = jacobian_cartesian.shape[-1]
    if intrinsic_dim == 0:
        return cartesian_vectors.new_empty(0)
    # v_eta = (J_hat^T P_N J_hat)^-1 J_hat^T P_N v_cart is an
    # exact metric-adjoint reduction.  The neural Cartesian vectors may have
    # been produced by AMP, but the solve and its rank certificate must execute
    # in Float64 crystallographic precision.  The final legal velocity is cast
    # back to the neural master dtype; `.to` keeps its gradient path intact.
    with torch.autocast(device_type=jacobian_cartesian.device.type, enabled=False):
        jacobian = jacobian_cartesian.to(dtype=torch.float64) / reference_length
        vectors = cartesian_vectors.to(dtype=torch.float64)
        centered_jacobian = jacobian - jacobian.mean(dim=0, keepdim=True)
        centered_vectors = vectors - vectors.mean(dim=0, keepdim=True)
        matrix = centered_jacobian.reshape(-1, intrinsic_dim)
        normal = torch.einsum("nci,ncj->ij", centered_jacobian, centered_jacobian)
        right_hand_side = torch.einsum("nci,nc->i", centered_jacobian, centered_vectors)
        if check_rank:
            maximum_condition = (
                float(maximum_condition_number_float64)
                if jacobian_cartesian.dtype == torch.float64
                else float(maximum_condition_number_float32)
            )
            _certify_batched_tangent_rank(
                torch.linalg.svdvals(matrix.detach()).unsqueeze(0),
                torch.full((1,), intrinsic_dim, dtype=torch.long),
                torch.full((1,), matrix.shape[0] // 3, dtype=torch.long),
                maximum_condition,
                checks=checks,
            )
        try:
            factor, info = torch.linalg.cholesky_ex(normal)
            checks.add(
                (info != 0).any(),
                partial(
                    TangentRankError, "J^T P_N J is not positive definite in Float64 algebra"
                ),
            )
            solution = torch.cholesky_solve(
                right_hand_side.unsqueeze(-1), factor
            ).squeeze(-1)
        except RuntimeError as exc:
            if own:
                checks.raise_first()
            raise TangentRankError(
                "intrinsic tangent reduction failed without a ridge"
            ) from exc
        if own:
            checks.raise_first()
        return solution.to(dtype=cartesian_vectors.dtype)


def _certify_batched_tangent_rank(
    singular_values: Tensor,
    dimensions: Tensor,
    atom_counts: Tensor,
    maximum_condition: float,
    *,
    checks: DeviceChecks | None = None,
) -> None:
    """The per-crystal certificate of (3.10) with one host synchronization per batch.

    Crystal ``b`` reads singular values ``0 .. d_b - 1``.  It is rank deficient
    when ``3 n_b < d_b`` or ``s_{d_b - 1} <= 0``, and refused when
    ``s_0 / s_{d_b - 1}`` exceeds the bound.  The first refused crystal raises the
    unbatched audit's message.  With ``checks`` the read joins the caller's.
    """

    if singular_values.shape[-1] == 0:
        return
    dimensions = dimensions.to(device=singular_values.device)
    atom_counts = atom_counts.to(device=singular_values.device)
    active = dimensions > 0
    last = (dimensions - 1).clamp(min=0, max=singular_values.shape[-1] - 1)
    smallest = singular_values.gather(-1, last[:, None]).squeeze(-1)
    deficient = active & ((3 * atom_counts < dimensions) | (smallest <= 0.0))
    usable = active & ~deficient
    condition = singular_values[:, 0] / torch.where(usable, smallest, torch.ones_like(smallest))
    refused = deficient | (usable & (condition > maximum_condition))

    def error() -> Exception:
        first = int(torch.nonzero(refused)[0, 0].item())
        if bool(deficient[first].item()):
            return TangentRankError(
                "P_N J is rank deficient; the deterministic "
                "tangent-kernel certificate or supplied Jacobian is invalid"
            )
        return TangentRankError(
            "P_N J exceeds the declared condition-number support "
            f"({float(condition[first].item()):.6g} > {maximum_condition:.6g})"
        )

    check(checks, refused.any(), error)


def _validate_packed_jacobian(
    jacobian_cartesian: Tensor,
    reference_length: float,
    checks: DeviceChecks | None = None,
) -> None:
    if jacobian_cartesian.ndim != 4 or jacobian_cartesian.shape[2] != 3:
        raise ValueError("packed jacobian_cartesian must have shape [B,N,3,d_eta]")
    if jacobian_cartesian.shape[0] == 0 or jacobian_cartesian.shape[1] == 0:
        raise ValueError("a packed batch needs crystals with atoms")
    if (
        isinstance(reference_length, bool)
        or not isinstance(reference_length, (int, float))
        or not math.isfinite(float(reference_length))
        or float(reference_length) <= 0.0
    ):
        raise ValueError("reference_length must be finite and positive")
    if jacobian_cartesian.dtype not in {torch.float32, torch.float64}:
        raise ValueError("jacobian_cartesian must use float32 or float64 master precision")
    check(
        checks,
        ~torch.isfinite(jacobian_cartesian).all(),
        partial(ValueError, "jacobian_cartesian entries must be finite"),
    )


def translation_quotient_normal_matrix_packed(
    jacobian_cartesian: Tensor,
    *,
    reference_length: float = 1.0,
    checks: DeviceChecks | None = None,
) -> Tensor:
    """:func:`translation_quotient_normal_matrix` of a topology-homogeneous batch.

    ``jacobian_cartesian`` is ``[B, N, 3, d]``; the result is ``[B, d, d]`` in Float64.
    """

    _validate_packed_jacobian(jacobian_cartesian, reference_length, checks)
    with torch.autocast(device_type=jacobian_cartesian.device.type, enabled=False):
        jacobian = jacobian_cartesian.to(dtype=torch.float64) / reference_length
        centered = jacobian - jacobian.mean(dim=1, keepdim=True)
        return torch.einsum("bnci,bncj->bij", centered, centered)


def reduce_cartesian_to_intrinsic_packed(
    cartesian_vectors: Tensor,
    jacobian_cartesian: Tensor,
    *,
    reference_length: float = 1.0,
    maximum_condition_number_float32: float = 1.0e4,
    maximum_condition_number_float64: float = 1.0e6,
    checks: DeviceChecks | None = None,
) -> Tensor:
    """:func:`reduce_cartesian_to_intrinsic` of a topology-homogeneous batch.

    ``cartesian_vectors`` is ``[B, N, 3]`` and ``jacobian_cartesian`` is
    ``[B, N, 3, d]``.  Each crystal gets the same Float64 solve and the same rank
    and condition-number certificate, with no ridge.  All flags cost one host
    synchronization per batch, or none here when ``checks`` collects them.
    """

    own = checks is None
    if checks is None:
        checks = DeviceChecks()
    _validate_packed_jacobian(jacobian_cartesian, reference_length, checks)
    if cartesian_vectors.ndim != 3 or jacobian_cartesian.shape[:3] != cartesian_vectors.shape:
        raise ValueError("packed vectors and Jacobian must describe the same atoms")
    if cartesian_vectors.device != jacobian_cartesian.device:
        raise ValueError("vectors and Jacobian must share one device")
    if cartesian_vectors.dtype != jacobian_cartesian.dtype:
        raise ValueError("vectors and Jacobian must share one master dtype")
    for name, value in (
        ("maximum_condition_number_float32", maximum_condition_number_float32),
        ("maximum_condition_number_float64", maximum_condition_number_float64),
    ):
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or float(value) < 1.0
        ):
            raise ValueError(f"{name} must be finite and at least one")
    checks.add(
        ~torch.isfinite(cartesian_vectors).all(),
        partial(ValueError, "vectors and Jacobian entries must be finite"),
    )
    if own:
        checks.raise_first()  # keep non-finite data out of the LAPACK calls below
    batch, atoms, _, intrinsic_dim = jacobian_cartesian.shape
    if intrinsic_dim == 0:
        return cartesian_vectors.new_empty(batch, 0)
    with torch.autocast(device_type=jacobian_cartesian.device.type, enabled=False):
        jacobian = jacobian_cartesian.to(dtype=torch.float64) / reference_length
        vectors = cartesian_vectors.to(dtype=torch.float64)
        centered_jacobian = jacobian - jacobian.mean(dim=1, keepdim=True)
        centered_vectors = vectors - vectors.mean(dim=1, keepdim=True)
        normal = torch.einsum("bnci,bncj->bij", centered_jacobian, centered_jacobian)
        right_hand_side = torch.einsum("bnci,bnc->bi", centered_jacobian, centered_vectors)
        _certify_batched_tangent_rank(
            torch.linalg.svdvals(centered_jacobian.detach().reshape(batch, -1, intrinsic_dim)),
            torch.full((batch,), intrinsic_dim, dtype=torch.long),
            torch.full((batch,), atoms, dtype=torch.long),
            float(maximum_condition_number_float64)
            if jacobian_cartesian.dtype == torch.float64
            else float(maximum_condition_number_float32),
            checks=checks,
        )
        try:
            factor, info = torch.linalg.cholesky_ex(normal)
            checks.add(
                (info != 0).any(),
                partial(
                    TangentRankError, "J^T P_N J is not positive definite in Float64 algebra"
                ),
            )
            solution = torch.cholesky_solve(
                right_hand_side.unsqueeze(-1), factor
            ).squeeze(-1)
        except RuntimeError as exc:
            if own:
                checks.raise_first()
            raise TangentRankError(
                "intrinsic tangent reduction failed without a ridge"
            ) from exc
        if own:
            checks.raise_first()
        return solution.to(dtype=cartesian_vectors.dtype)


def _padded_dimension_mask(
    dimensions: Tensor, width: int, *, name: str, checks: DeviceChecks | None = None
) -> Tensor:
    if dimensions.dtype == torch.bool or dimensions.is_floating_point() or dimensions.ndim != 1:
        raise ValueError(f"{name} must be a one-dimensional integer tensor")
    check(
        checks,
        ((dimensions < 0) | (dimensions > width)).any(),
        partial(ValueError, f"{name} must lie in [0, {width}]"),
    )
    return torch.arange(width, device=dimensions.device)[None, :] < dimensions[:, None]


def translation_quotient_normal_matrix_padded(
    jacobian_cartesian: Tensor,
    atom_mask: Tensor,
    intrinsic_dimensions: Tensor,
    *,
    reference_length: float = 1.0,
    checks: DeviceChecks | None = None,
) -> Tensor:
    """Per-crystal ``J_hat^T P_N J_hat`` of a padded batch, identity-padded.

    ``jacobian_cartesian`` is ``[B, N, 3, D]``.  Each crystal is centred over
    its real atoms only, and its ``D - d_b`` padded coordinates get the identity
    block.  A padded velocity error is exactly zero, so it contributes nothing
    to the loss, and the metric stays positive definite.
    """

    if jacobian_cartesian.ndim != 4 or jacobian_cartesian.shape[2] != 3:
        raise ValueError("padded jacobian_cartesian must have shape [B,N,3,D]")
    batch, atoms, _, width = jacobian_cartesian.shape
    if atom_mask.dtype != torch.bool or atom_mask.shape != (batch, atoms):
        raise ValueError("atom_mask must be a bool tensor of shape [B, N]")
    columns = _padded_dimension_mask(
        intrinsic_dimensions, width, name="intrinsic_dimensions", checks=checks
    )
    with torch.autocast(device_type=jacobian_cartesian.device.type, enabled=False):
        real = atom_mask.to(dtype=torch.float64)
        jacobian = jacobian_cartesian.to(dtype=torch.float64) / reference_length
        jacobian = jacobian * real[:, :, None, None] * columns[:, None, None, :]
        mean = jacobian.sum(dim=1, keepdim=True) / real.sum(dim=1)[:, None, None, None]
        centered = (jacobian - mean) * real[:, :, None, None]
        normal = torch.einsum("bnci,bncj->bij", centered, centered)
        return normal + torch.diag_embed((~columns).to(dtype=torch.float64))


def reduce_cartesian_to_intrinsic_padded(
    cartesian_vectors: Tensor,
    jacobian_cartesian: Tensor,
    atom_mask: Tensor,
    intrinsic_dimensions: Tensor,
    *,
    reference_length: float = 1.0,
    maximum_condition_number_float32: float = 1.0e4,
    maximum_condition_number_float64: float = 1.0e6,
    checks: DeviceChecks | None = None,
) -> Tensor:
    """Batched exact translation-quotient solves (3.10) for a padded batch.

    Each crystal is solved on its real atoms and its own ``d_b`` coordinates.
    Zero-padded rows and columns only add zero singular values, so crystal
    ``b``'s rank and condition-number certificate reads singular values
    ``0 .. d_b - 1`` of the padded matrix, exactly as the unpadded audit does.
    Identity padding makes the Cholesky factor block diagonal, so padded
    coordinates solve to zero.  No ridge is added.
    """

    if cartesian_vectors.ndim != 3 or cartesian_vectors.shape[-1] != 3:
        raise ValueError("padded cartesian_vectors must have shape [B,N,3]")
    if jacobian_cartesian.shape[:3] != cartesian_vectors.shape:
        raise ValueError("padded vectors and Jacobian must describe the same atoms")
    if cartesian_vectors.dtype not in {torch.float32, torch.float64} or (
        jacobian_cartesian.dtype != cartesian_vectors.dtype
    ):
        raise ValueError("padded vectors and Jacobian must share a float32/float64 dtype")
    own = checks is None
    if checks is None:
        checks = DeviceChecks()
    checks.add(
        ~(torch.isfinite(cartesian_vectors).all() & torch.isfinite(jacobian_cartesian).all()),
        partial(ValueError, "padded vectors and Jacobian entries must be finite"),
    )
    batch, atoms, _, width = jacobian_cartesian.shape
    if atom_mask.dtype != torch.bool or atom_mask.shape != (batch, atoms):
        raise ValueError("atom_mask must be a bool tensor of shape [B, N]")
    columns = _padded_dimension_mask(
        intrinsic_dimensions, width, name="intrinsic_dimensions", checks=checks
    )
    if own:
        checks.raise_first()  # keep non-finite data out of the LAPACK calls below
    if width == 0:
        return cartesian_vectors.new_zeros(batch, 0)
    with torch.autocast(device_type=jacobian_cartesian.device.type, enabled=False):
        real = atom_mask.to(dtype=torch.float64)
        counts = real.sum(dim=1)
        jacobian = jacobian_cartesian.to(dtype=torch.float64) / reference_length
        jacobian = jacobian * real[:, :, None, None] * columns[:, None, None, :]
        vectors = cartesian_vectors.to(dtype=torch.float64) * real[:, :, None]
        centered_jacobian = (
            jacobian - jacobian.sum(dim=1, keepdim=True) / counts[:, None, None, None]
        ) * real[:, :, None, None]
        centered_vectors = (
            vectors - vectors.sum(dim=1, keepdim=True) / counts[:, None, None]
        ) * real[:, :, None]
        _certify_batched_tangent_rank(
            torch.linalg.svdvals(centered_jacobian.detach().reshape(batch, -1, width)),
            intrinsic_dimensions,
            counts,
            float(maximum_condition_number_float64)
            if jacobian_cartesian.dtype == torch.float64
            else float(maximum_condition_number_float32),
            checks=checks,
        )
        normal = torch.einsum(
            "bnci,bncj->bij", centered_jacobian, centered_jacobian
        ) + torch.diag_embed((~columns).to(dtype=torch.float64))
        right_hand_side = torch.einsum("bnci,bnc->bi", centered_jacobian, centered_vectors)
        try:
            factor, info = torch.linalg.cholesky_ex(normal)
            checks.add(
                (info != 0).any(),
                partial(
                    TangentRankError, "J^T P_N J is not positive definite in Float64 algebra"
                ),
            )
            solution = torch.cholesky_solve(right_hand_side.unsqueeze(-1), factor).squeeze(-1)
        except RuntimeError as exc:
            if own:
                checks.raise_first()
            raise TangentRankError(
                "intrinsic tangent reduction failed without a ridge"
            ) from exc
        if own:
            checks.raise_first()
        return solution.to(dtype=cartesian_vectors.dtype)


class _TimeFeatures(nn.Module):
    def __init__(self, num_frequencies: int) -> None:
        super().__init__()
        if (
            isinstance(num_frequencies, bool)
            or not isinstance(num_frequencies, int)
            or num_frequencies < 1
        ):
            raise ValueError("num_time_frequencies must be a positive integer")
        frequencies = (
            2.0 * math.pi * torch.pow(torch.tensor(2.0), torch.arange(num_frequencies))
        )
        self.register_buffer("frequencies", frequencies)

    @property
    def output_dim(self) -> int:
        return 1 + 2 * int(self.frequencies.numel())

    def forward(
        self,
        time: Tensor,
        *,
        dtype: torch.dtype,
        device: torch.device,
        checks: DeviceChecks | None = None,
    ) -> Tensor:
        value = torch.as_tensor(time, dtype=dtype, device=device)
        if value.numel() != 1:
            raise ValueError("a single-structure field evaluation requires scalar time")
        value = value.reshape(1)
        check(
            checks,
            ~torch.isfinite(value).all(),
            partial(ValueError, "flow time must be finite"),
        )
        phase = value * self.frequencies.to(dtype=dtype, device=device)
        return torch.cat((value, torch.sin(phase), torch.cos(phase)), dim=0)

    def batched(
        self,
        times: Tensor,
        *,
        dtype: torch.dtype,
        device: torch.device,
        checks: DeviceChecks | None = None,
    ) -> Tensor:
        """Rows of :meth:`forward` for the times ``[B]``, in one pass."""

        values = torch.as_tensor(times, dtype=dtype, device=device)
        if values.ndim != 1:
            raise ValueError("batched flow times must have shape [B]")
        check(
            checks,
            ~torch.isfinite(values).all(),
            partial(ValueError, "flow time must be finite"),
        )
        phase = values[:, None] * self.frequencies.to(dtype=dtype, device=device)
        return torch.cat((values[:, None], torch.sin(phase), torch.cos(phase)), dim=-1)


@dataclass(frozen=True)
class IntrinsicFieldConfig:
    atom_feature_dim: int
    composition_dim: int = 128
    species_embedding_dim: int = 64
    max_lattice_dim: int = 6
    max_atomic_number: int = 118
    num_time_frequencies: int = 8
    backbone: str = "invariant_global"
    scalar_dim: int = 192
    pair_dim: int = 96
    vector_dim: int = 48
    num_heads: int = 6
    num_layers: int = 6
    radial_basis_dim: int = 24
    invariant_feedforward_multiplier: int = 4
    reciprocal_mode_radius: int = 2
    heat_scales: tuple[float, ...] = (0.04, 0.16, 0.64)
    direct_harmonics: int = 2
    include_lattice_frame_carriers: bool = True
    normalization_epsilon: float = 1.0e-4
    minimum_dimensionless_lattice_singular_value: float = 1.0e-6
    maximum_dimensionless_lattice_singular_value: float = 1.0e6
    maximum_lattice_condition_number: float = 1.0e6
    cutoff: float = 6.0
    reference_length: float = 1.0
    max_candidates_per_pair: int = 32_768
    max_periodic_edges: int = 500_000
    max_dense_pair_interactions: int = 500_000
    max_reciprocal_modes: int = 4_096
    max_readout_coefficient_channels: int = 4_096
    max_geometry_workspace_bytes: int = 2_000_000_000
    maximum_tangent_condition_number_float32: float = 1.0e4
    maximum_tangent_condition_number_float64: float = 1.0e6
    lattice_linear_growth_bound: float = 8.0
    lattice_residual_bound: float = 8.0
    radial_gaussian_widths: tuple[float, ...] = ()
    pair_carrier_smoothing_length: float | None = None
    """Dense backbone only: coincidence smoothing length of the pair carriers
    (:class:`InvariantPeriodicTransformerConfig`); ``None`` is the legacy carrier."""
    element_embedding: str = "learned-v1"
    """``learned-v1`` (free tables) or ``periodic-v1`` for both element embeddings."""
    """Metric-aware periodic radial pair scalars for the dense backbone (review M2)."""
    local_vector_normalization: str = "rms-v1"
    """``equivariant_local`` vector normalization: ``rms-v1`` (legacy) or ``soft-v2``.

    ``rms-v1`` amplifies cutoff-gated vanishing vectors (review F6 defect).
    ``soft-v2`` is 1-Lipschitz, so the field is C1 across the cutoff.
    """
    lattice_head: str = "truncated-v1"
    """``truncated-v1`` (legacy) or ``commutant-v2`` (basis-free, review M3).

    ``commutant-v2`` reads the lattice state as ``C b`` and writes
    ``C^T A C z + C^T r`` in canonical Sym(3) coordinates, where ``C`` (6 x d) is
    the chart's orthonormal commutant basis; it requires six slots.
    """


def _commutant_basis(
    basis: Tensor | None,
    *,
    dimension: int,
    dtype: torch.dtype,
    device: torch.device,
    checks: DeviceChecks | None = None,
) -> Tensor:
    if basis is None:
        raise ValueError("the commutant-v2 lattice head requires lattice_commutant_basis")
    if basis.shape != (6, dimension):
        raise ValueError("lattice_commutant_basis must have shape [6, d]")
    if not basis.is_floating_point():
        raise ValueError("lattice_commutant_basis must be finite and floating point")
    check(
        checks,
        ~torch.isfinite(basis).all(),
        partial(ValueError, "lattice_commutant_basis must be finite and floating point"),
    )
    return basis.to(dtype=dtype, device=device)


@dataclass(frozen=True)
class IntrinsicFieldOutput:
    lattice_velocity: Tensor
    intrinsic_velocity: Tensor
    cartesian_vectors: Tensor
    atom_scalar_channels: Tensor
    coordinate_metric: Tensor
    interaction_count: int
    edges: PeriodicEdges | None = None


@dataclass(frozen=True)
class PaddedIntrinsicFieldOutput:
    """Mixed-topology padded field output (review M5).

    Shapes are ``lattice_velocity[B, Z]``, ``intrinsic_velocity[B, D]``,
    ``cartesian_vectors[B, N, 3]`` and ``coordinate_metric[B, D, D]``.  Every
    padded entry is exactly zero, except the metric's padded diagonal, which is
    one.
    """

    lattice_velocity: Tensor
    intrinsic_velocity: Tensor
    cartesian_vectors: Tensor
    atom_scalar_channels: Tensor
    coordinate_metric: Tensor
    interaction_count: int
    atom_mask: Tensor
    lattice_dimensions: Tensor
    intrinsic_dimensions: Tensor


@dataclass(frozen=True)
class PackedIntrinsicFieldOutput:
    """Topology-homogeneous block-diagonal field output.

    The leading dimension is the number of crystals.  Every crystal in this
    path has equal lattice, intrinsic, and expanded-atom dimensions; batching
    heterogeneous topology is deliberately handled by the outer bucket
    sampler rather than padding crystallographic tangent spaces.
    """

    lattice_velocity: Tensor
    intrinsic_velocity: Tensor
    cartesian_vectors: Tensor
    atom_scalar_channels: Tensor
    coordinate_metric: Tensor
    interaction_count: int
    edges: PeriodicEdges | None


class IntrinsicGeometryField(nn.Module):
    """Shared cover-conditioned flow field with an exact tangent reduction.

    Stable forward signature::

        field(
            time, z, eta, composition_context,
            atomic_numbers=..., atom_features=...,
            fractional_coordinates=..., lattice=...,
            bounded_lattice_coordinates=..., jacobian_cartesian=...,
            geometry_is_certified=False,
        ) -> IntrinsicFieldOutput

    ``atom_features`` are deterministic invariant orbit/site features expanded
    to atoms, including intrinsic-torus Fourier characters when applicable.
    Their construction and residual-group closure cannot be certified by this
    neural class.  Dependence on ``eta`` enters through those current expanded
    coordinates/features, never through a nonperiodic raw-coordinate embedding.
    ``geometry_is_certified=True`` is retained as provenance metadata for
    tensors produced immediately by :class:`TorchCoverGeometry`.  It never
    waives current-value finite, handedness, conditioning, dtype, shape, or
    resource checks; a bounded chart is not a proof that an arbitrary ODE state
    supplied at runtime is finite.
    """

    def __init__(self, config: IntrinsicFieldConfig) -> None:
        super().__init__()
        nonnegative_integers = {"atom_feature_dim": config.atom_feature_dim}
        positive_integers = {
            "composition_dim": config.composition_dim,
            "species_embedding_dim": config.species_embedding_dim,
            "max_lattice_dim": config.max_lattice_dim,
            "max_atomic_number": config.max_atomic_number,
            "num_time_frequencies": config.num_time_frequencies,
            "scalar_dim": config.scalar_dim,
            "pair_dim": config.pair_dim,
            "vector_dim": config.vector_dim,
            "num_heads": config.num_heads,
            "num_layers": config.num_layers,
            "radial_basis_dim": config.radial_basis_dim,
            "invariant_feedforward_multiplier": (
                config.invariant_feedforward_multiplier
            ),
            "direct_harmonics": config.direct_harmonics,
            "max_candidates_per_pair": config.max_candidates_per_pair,
            "max_periodic_edges": config.max_periodic_edges,
            "max_dense_pair_interactions": config.max_dense_pair_interactions,
            "max_reciprocal_modes": config.max_reciprocal_modes,
            "max_readout_coefficient_channels": (
                config.max_readout_coefficient_channels
            ),
            "max_geometry_workspace_bytes": config.max_geometry_workspace_bytes,
        }
        for name, value in nonnegative_integers.items():
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a nonnegative integer")
        for name, value in positive_integers.items():
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if (
            isinstance(config.reciprocal_mode_radius, bool)
            or not isinstance(config.reciprocal_mode_radius, int)
            or config.reciprocal_mode_radius < 0
        ):
            raise ValueError("reciprocal_mode_radius must be a nonnegative integer")
        for name in (
            "cutoff",
            "reference_length",
            "lattice_linear_growth_bound",
            "lattice_residual_bound",
            "maximum_tangent_condition_number_float32",
            "maximum_tangent_condition_number_float64",
        ):
            value = getattr(config, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                or float(value) <= 0.0
            ):
                raise ValueError(f"{name} must be finite and positive")
        if config.maximum_tangent_condition_number_float32 < 1.0 or (
            config.maximum_tangent_condition_number_float64 < 1.0
        ):
            raise ValueError("tangent condition-number bounds must be at least one")
        if config.lattice_head not in {"truncated-v1", "commutant-v2"}:
            raise ValueError("lattice_head must be 'truncated-v1' or 'commutant-v2'")
        if config.local_vector_normalization not in {"rms-v1", "soft-v2"}:
            raise ValueError("local_vector_normalization must be 'rms-v1' or 'soft-v2'")
        if config.lattice_head == "commutant-v2" and config.max_lattice_dim != 6:
            raise ValueError("the commutant-v2 lattice head needs six canonical Sym(3) slots")
        if config.backbone not in {"invariant_global", "equivariant_local"}:
            raise ValueError(
                "backbone must be 'invariant_global' or 'equivariant_local'"
            )
        if config.backbone == "invariant_global" and config.reciprocal_mode_radius < 1:
            raise ValueError("invariant_global requires reciprocal_mode_radius >= 1")
        if type(config.include_lattice_frame_carriers) is not bool:
            raise ValueError("include_lattice_frame_carriers must be boolean")
        if not isinstance(config.heat_scales, tuple) or not config.heat_scales or any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or float(value) <= 0.0
            for value in config.heat_scales
        ):
            raise ValueError("heat_scales must be a nonempty tuple of positive values")
        for name in (
            "normalization_epsilon",
            "minimum_dimensionless_lattice_singular_value",
            "maximum_dimensionless_lattice_singular_value",
            "maximum_lattice_condition_number",
        ):
            value = getattr(config, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                or float(value) <= 0.0
            ):
                raise ValueError(f"{name} must be finite and positive")
        if (
            config.maximum_dimensionless_lattice_singular_value
            <= config.minimum_dimensionless_lattice_singular_value
        ):
            raise ValueError("maximum lattice singular value must exceed its minimum")
        if config.maximum_lattice_condition_number < 1.0:
            raise ValueError("maximum_lattice_condition_number must be at least one")
        self.config = config
        self.flow_composition_encoder = CompositionEncoder(
            config.composition_dim,
            max_atomic_number=config.max_atomic_number,
            element_dim=config.species_embedding_dim,
            hidden_dim=max(config.composition_dim, config.species_embedding_dim),
            element_embedding=config.element_embedding,
        )
        self.species_embedding = make_element_embedding(
            config.element_embedding,
            config.max_atomic_number + 1,
            config.species_embedding_dim,
        )
        self.time_features = _TimeFeatures(config.num_time_frequencies)
        transformer_input_dim = (
            config.atom_feature_dim
            + config.species_embedding_dim
            + config.composition_dim
            + self.time_features.output_dim
            + config.max_lattice_dim
        )
        if config.backbone == "invariant_global":
            invariant_config = InvariantPeriodicTransformerConfig(
                input_scalar_dim=transformer_input_dim,
                model_dim=config.scalar_dim,
                pair_dim=config.pair_dim,
                num_heads=config.num_heads,
                num_layers=config.num_layers,
                feedforward_multiplier=config.invariant_feedforward_multiplier,
                reciprocal_mode_radius=config.reciprocal_mode_radius,
                heat_scales=config.heat_scales,
                direct_harmonics=config.direct_harmonics,
                include_lattice_frame_carriers=(
                    config.include_lattice_frame_carriers
                ),
                output_vector_dim=config.vector_dim,
                reference_length=config.reference_length,
                normalization_epsilon=config.normalization_epsilon,
                max_pair_interactions=config.max_dense_pair_interactions,
                max_reciprocal_modes=config.max_reciprocal_modes,
                max_readout_coefficient_channels=(
                    config.max_readout_coefficient_channels
                ),
                max_geometry_workspace_bytes=config.max_geometry_workspace_bytes,
                minimum_dimensionless_lattice_singular_value=(
                    config.minimum_dimensionless_lattice_singular_value
                ),
                maximum_dimensionless_lattice_singular_value=(
                    config.maximum_dimensionless_lattice_singular_value
                ),
                maximum_lattice_condition_number=(
                    config.maximum_lattice_condition_number
                ),
                radial_gaussian_widths=tuple(config.radial_gaussian_widths),
                pair_carrier_smoothing_length=config.pair_carrier_smoothing_length,
            )
            self.transformer = InvariantPeriodicTransformer(invariant_config)
        else:
            equivariant_config = PeriodicTransformerConfig(
                input_scalar_dim=transformer_input_dim,
                scalar_dim=config.scalar_dim,
                vector_dim=config.vector_dim,
                num_heads=config.num_heads,
                num_layers=config.num_layers,
                radial_basis_dim=config.radial_basis_dim,
                cutoff=config.cutoff,
                reference_length=config.reference_length,
                max_candidates_per_pair=config.max_candidates_per_pair,
                max_edges=config.max_periodic_edges,
                vector_normalization=config.local_vector_normalization,
            )
            self.transformer = PeriodicSE3Transformer(equivariant_config)
        self.cartesian_head = VectorLinear(config.vector_dim, 1)
        lattice_input = (
            config.scalar_dim
            + config.composition_dim
            + self.time_features.output_dim
            + config.max_lattice_dim
        )
        self.lattice_residual_head = nn.Sequential(
            nn.Linear(lattice_input, config.scalar_dim),
            nn.SiLU(),
            nn.Linear(config.scalar_dim, config.max_lattice_dim),
            nn.Tanh(),
        )
        self.lattice_linear_head = nn.Sequential(
            nn.Linear(lattice_input, config.scalar_dim),
            nn.SiLU(),
            nn.Linear(
                config.scalar_dim,
                config.max_lattice_dim * config.max_lattice_dim,
            ),
            nn.Tanh(),
        )

    def encode_composition(
        self,
        atomic_numbers: Tensor,
        reduced_counts: Tensor,
        descriptors: Tensor | None = None,
    ) -> Tensor:
        return self.flow_composition_encoder(
            atomic_numbers, reduced_counts, descriptors=descriptors
        )

    def forward(
        self,
        time: Tensor,
        z: Tensor,
        eta: Tensor,
        composition_context: Tensor | None,
        *,
        atomic_numbers: Tensor,
        atom_features: Tensor,
        fractional_coordinates: Tensor,
        lattice: Tensor,
        bounded_lattice_coordinates: Tensor,
        jacobian_cartesian: Tensor,
        composition_atomic_numbers: Tensor | None = None,
        composition_counts: Tensor | None = None,
        geometry_is_certified: bool = False,
        lattice_commutant_basis: Tensor | None = None,
    ) -> IntrinsicFieldOutput | PackedIntrinsicFieldOutput:
        if not isinstance(geometry_is_certified, bool):
            raise ValueError("geometry_is_certified must be boolean")
        if z.ndim == 2:
            return self.forward_packed(
                time,
                z,
                eta,
                composition_context,
                atomic_numbers=atomic_numbers,
                atom_features=atom_features,
                fractional_coordinates=fractional_coordinates,
                lattice=lattice,
                bounded_lattice_coordinates=bounded_lattice_coordinates,
                jacobian_cartesian=jacobian_cartesian,
                composition_atomic_numbers=composition_atomic_numbers,
                composition_counts=composition_counts,
                geometry_is_certified=geometry_is_certified,
                lattice_commutant_basis=lattice_commutant_basis,
            )
        checks = DeviceChecks()
        with checks.deferred():
            return self._forward_single(
                checks,
                time,
                z,
                eta,
                composition_context,
                atomic_numbers=atomic_numbers,
                atom_features=atom_features,
                fractional_coordinates=fractional_coordinates,
                lattice=lattice,
                bounded_lattice_coordinates=bounded_lattice_coordinates,
                jacobian_cartesian=jacobian_cartesian,
                composition_atomic_numbers=composition_atomic_numbers,
                composition_counts=composition_counts,
                geometry_is_certified=geometry_is_certified,
                lattice_commutant_basis=lattice_commutant_basis,
            )

    def _forward_single(
        self,
        checks: DeviceChecks,
        time: Tensor,
        z: Tensor,
        eta: Tensor,
        composition_context: Tensor | None,
        *,
        atomic_numbers: Tensor,
        atom_features: Tensor,
        fractional_coordinates: Tensor,
        lattice: Tensor,
        bounded_lattice_coordinates: Tensor,
        jacobian_cartesian: Tensor,
        composition_atomic_numbers: Tensor | None = None,
        composition_counts: Tensor | None = None,
        geometry_is_certified: bool = False,
        lattice_commutant_basis: Tensor | None = None,
    ) -> IntrinsicFieldOutput:
        """One structure; every device check is read once by :meth:`forward`."""

        if z.ndim != 1 or eta.ndim != 1:
            raise ValueError("z and eta must be one-dimensional intrinsic states")
        if not z.is_floating_point() or z.dtype not in {torch.float32, torch.float64}:
            raise ValueError("z must use float32 or float64 master precision")
        if eta.dtype != z.dtype or eta.device != z.device:
            raise ValueError("z and eta must share one master dtype and device")
        checks.add(
            ~(torch.isfinite(z).all() & torch.isfinite(eta).all()),
            partial(ValueError, "z and eta entries must be finite"),
        )
        if z.numel() > self.config.max_lattice_dim:
            raise ValueError("z exceeds configured max_lattice_dim")
        if bounded_lattice_coordinates.shape != z.shape:
            raise ValueError("bounded lattice coordinates must have the shape of z")
        if composition_context is None:
            if composition_atomic_numbers is None or composition_counts is None:
                raise ValueError(
                    "composition values are required when no encoded context is supplied"
                )
            composition_context = self.encode_composition(
                composition_atomic_numbers, composition_counts
            )
        elif composition_atomic_numbers is not None or composition_counts is not None:
            raise ValueError(
                "supply either composition_context or raw composition values, not both"
            )
        if composition_context.shape != (self.config.composition_dim,):
            raise ValueError("composition_context has the wrong shape")
        if not composition_context.is_floating_point():
            raise ValueError("composition_context must be finite and floating point")
        checks.add(
            ~torch.isfinite(composition_context).all(),
            partial(ValueError, "composition_context must be finite and floating point"),
        )
        if composition_context.device != z.device:
            raise ValueError("composition_context must share the continuous-state device")
        if fractional_coordinates.ndim != 2 or fractional_coordinates.shape[-1] != 3:
            raise ValueError("fractional_coordinates must have shape [N,3]")
        if lattice.shape != (3, 3):
            raise ValueError("lattice must have shape [3,3]")
        atom_count = fractional_coordinates.shape[0]
        if atomic_numbers.shape != (atom_count,):
            raise ValueError("atomic_numbers must have shape [N]")
        if atom_features.shape != (atom_count, self.config.atom_feature_dim):
            raise ValueError("atom_features have the wrong shape")
        if jacobian_cartesian.shape != (atom_count, 3, eta.numel()):
            raise ValueError("jacobian_cartesian must have shape [N,3,len(eta)]")
        if atomic_numbers.dtype == torch.bool or atomic_numbers.is_floating_point():
            raise ValueError("atomic_numbers must use an integer tensor dtype")
        if atomic_numbers.device != z.device:
            raise ValueError("atomic_numbers must share the continuous-state device")
        checks.add(
            ((atomic_numbers <= 0) | (atomic_numbers > self.config.max_atomic_number)).any(),
            partial(ValueError, "expanded atoms must carry valid atomic numbers"),
        )

        # z/eta and every deterministic bridge tensor are master
        # crystallographic state.  Only learned feature transforms are allowed
        # to follow the enclosing autocast policy.
        dtype = z.dtype
        device = z.device
        deterministic_tensors = {
            "atom_features": atom_features,
            "fractional_coordinates": fractional_coordinates,
            "lattice": lattice,
            "bounded_lattice_coordinates": bounded_lattice_coordinates,
            "jacobian_cartesian": jacobian_cartesian,
        }
        for name, value in deterministic_tensors.items():
            if not value.is_floating_point() or value.dtype != dtype:
                raise ValueError(f"{name} must share the float32/float64 master dtype")
            if value.device != device:
                raise ValueError(f"{name} must share the continuous-state device")
            checks.add(
                ~torch.isfinite(value).all(),
                partial(ValueError, f"{name} entries must be finite"),
            )
        parameter = next(self.parameters())
        if parameter.dtype != dtype or parameter.device != device:
            raise ValueError(
                "field parameters and continuous states must share one master dtype/device"
            )
        composition_context = composition_context.to(dtype=dtype, device=device)
        time_features = self.time_features(time, dtype=dtype, device=device, checks=checks)
        commutant = None
        if self.config.lattice_head == "commutant-v2":
            commutant = _commutant_basis(
                lattice_commutant_basis,
                dimension=z.numel(),
                dtype=dtype,
                device=device,
                checks=checks,
            )
            bounded_padded = commutant @ bounded_lattice_coordinates.to(dtype=dtype)
        else:
            bounded_padded = torch.zeros(
                self.config.max_lattice_dim, dtype=dtype, device=device
            )
            bounded_padded[: z.numel()] = bounded_lattice_coordinates.to(
                dtype=dtype, device=device
            )
        global_features = torch.cat(
            (
                composition_context.to(dtype=dtype, device=device),
                time_features,
                bounded_padded,
            )
        )
        scalar_input = torch.cat(
            (
                self.species_embedding(atomic_numbers.to(torch.long)).to(dtype=dtype),
                atom_features.to(dtype=dtype),
                global_features.unsqueeze(0).expand(atom_count, -1),
            ),
            dim=-1,
        )
        edges: PeriodicEdges | None
        if self.config.backbone == "invariant_global":
            edges = None
            interaction_count = atom_count * atom_count
            if interaction_count > self.config.max_dense_pair_interactions:
                raise ValueError(
                    "dense geometry interaction budget exceeded before allocation "
                    f"({interaction_count} > "
                    f"{self.config.max_dense_pair_interactions})"
                )
            transformed = self.transformer(
                scalar_input,
                fractional_coordinates,
                lattice,
                validate_geometry=True,
                checks=checks,
            )
        else:
            with torch.autocast(device_type=device.type, enabled=False):
                edges = enumerate_periodic_edges(
                    fractional_coordinates,
                    lattice,
                    self.config.cutoff,
                    max_candidates_per_pair=self.config.max_candidates_per_pair,
                    max_edges=self.config.max_periodic_edges,
                )
            transformed = self.transformer(
                scalar_input,
                fractional_coordinates,
                lattice,
                edges=_transformer_edges(edges, z),
            )
            interaction_count = edges.num_edges
        # The target intrinsic torus lift is bounded, but its Cartesian image
        # scales with the current lattice.  An arbitrary unit-norm squash would
        # therefore make valid large-cell CFM targets unreachable.  The chart
        # has bounded physical geometry, so a finite checkpoint still yields a
        # bounded Cartesian branch without this destructive hard cap.
        neural_cartesian_vectors = self.cartesian_head(transformed.vector_channels).squeeze(-2)
        lattice_features = torch.cat(
            (transformed.pooled_scalars.to(dtype=dtype), global_features), dim=-1
        )
        neural_residual = self.lattice_residual_head(lattice_features)
        neural_linear = self.lattice_linear_head(lattice_features).reshape(
            self.config.max_lattice_dim, self.config.max_lattice_dim
        )
        # Normalize AMP neural outputs back to master precision before the
        # metric adjoint and chart-vector algebra.  Casting is differentiable;
        # it does not detach either head from the loss.
        with torch.autocast(device_type=device.type, enabled=False):
            cartesian_vectors = neural_cartesian_vectors.to(dtype=dtype)
            master_jacobian = jacobian_cartesian.to(dtype=dtype)
            intrinsic_velocity = reduce_cartesian_to_intrinsic(
                cartesian_vectors,
                master_jacobian,
                reference_length=self.config.reference_length,
                maximum_condition_number_float32=(
                    self.config.maximum_tangent_condition_number_float32
                ),
                maximum_condition_number_float64=(
                    self.config.maximum_tangent_condition_number_float64
                ),
                checks=checks,
            )
            residual = self.config.lattice_residual_bound * neural_residual.to(dtype=dtype)
            linear = (
                self.config.lattice_linear_growth_bound
                * neural_linear.to(dtype=dtype)
                / float(self.config.max_lattice_dim)
            )
            dimension = z.numel()
            if commutant is None:
                lattice_velocity = (
                    linear[:dimension, :dimension] @ z.to(dtype=dtype) + residual[:dimension]
                )
            else:
                lattice_velocity = commutant.T @ (
                    linear @ (commutant @ z.to(dtype=dtype)) + residual
                )
            coordinate_metric = translation_quotient_normal_matrix(
                master_jacobian,
                reference_length=self.config.reference_length,
                checks=checks,
            )
        return IntrinsicFieldOutput(
            lattice_velocity=lattice_velocity,
            intrinsic_velocity=intrinsic_velocity,
            cartesian_vectors=cartesian_vectors,
            atom_scalar_channels=transformed.scalar_channels.to(dtype=dtype),
            coordinate_metric=coordinate_metric,
            interaction_count=interaction_count,
            edges=edges,
        )

    def forward_packed(
        self,
        time: Tensor,
        z: Tensor,
        eta: Tensor,
        composition_context: Tensor | None,
        *,
        atomic_numbers: Tensor,
        atom_features: Tensor,
        fractional_coordinates: Tensor,
        lattice: Tensor,
        bounded_lattice_coordinates: Tensor,
        jacobian_cartesian: Tensor,
        composition_atomic_numbers: Tensor | None = None,
        composition_counts: Tensor | None = None,
        geometry_is_certified: bool = False,
        lattice_commutant_basis: Tensor | None = None,
    ) -> PackedIntrinsicFieldOutput:
        """Evaluate one topology-homogeneous batch with one Transformer call.

        Inputs use shapes ``z[B,d_z]``, ``eta[B,d_eta]``, atoms ``[B,N,...]``
        and lattices ``[B,3,3]``.  The dense backbone keeps an explicit batch
        axis and therefore cannot attend across crystals.  The local ablation
        enumerates each lattice separately and concatenates only certified
        block-diagonal edges.  Both paths are algebraically equivalent to
        independent field calls in evaluation mode while sharing kernels.
        """

        checks = DeviceChecks()
        with checks.deferred():
            return self._forward_packed(
                checks,
                time,
                z,
                eta,
                composition_context,
                atomic_numbers=atomic_numbers,
                atom_features=atom_features,
                fractional_coordinates=fractional_coordinates,
                lattice=lattice,
                bounded_lattice_coordinates=bounded_lattice_coordinates,
                jacobian_cartesian=jacobian_cartesian,
                composition_atomic_numbers=composition_atomic_numbers,
                composition_counts=composition_counts,
                geometry_is_certified=geometry_is_certified,
                lattice_commutant_basis=lattice_commutant_basis,
            )

    def _forward_packed(
        self,
        checks: DeviceChecks,
        time: Tensor,
        z: Tensor,
        eta: Tensor,
        composition_context: Tensor | None,
        *,
        atomic_numbers: Tensor,
        atom_features: Tensor,
        fractional_coordinates: Tensor,
        lattice: Tensor,
        bounded_lattice_coordinates: Tensor,
        jacobian_cartesian: Tensor,
        composition_atomic_numbers: Tensor | None = None,
        composition_counts: Tensor | None = None,
        geometry_is_certified: bool = False,
        lattice_commutant_basis: Tensor | None = None,
    ) -> PackedIntrinsicFieldOutput:
        if not isinstance(geometry_is_certified, bool):
            raise ValueError("geometry_is_certified must be boolean")
        if z.ndim != 2 or eta.ndim != 2:
            raise ValueError("packed z and eta must have shapes [B,d_z] and [B,d_eta]")
        if not z.is_floating_point() or z.dtype not in {torch.float32, torch.float64}:
            raise ValueError("packed z must use float32 or float64 master precision")
        if eta.dtype != z.dtype or eta.device != z.device:
            raise ValueError("packed z and eta must share one master dtype and device")
        checks.add(
            ~(torch.isfinite(z).all() & torch.isfinite(eta).all()),
            partial(ValueError, "packed z and eta entries must be finite"),
        )
        batch, lattice_dimension = z.shape
        if batch < 1 or eta.shape[0] != batch:
            raise ValueError("packed continuous states must have one nonempty batch")
        intrinsic_dimension = eta.shape[1]
        if lattice_dimension > self.config.max_lattice_dim:
            raise ValueError("packed z exceeds configured max_lattice_dim")
        if bounded_lattice_coordinates.shape != z.shape:
            raise ValueError("packed bounded lattice coordinates must match z")
        if composition_context is None:
            if composition_atomic_numbers is None or composition_counts is None:
                raise ValueError(
                    "packed composition values are required without encoded contexts"
                )
            composition_context = self.encode_composition(
                composition_atomic_numbers, composition_counts
            )
        elif composition_atomic_numbers is not None or composition_counts is not None:
            raise ValueError(
                "supply either packed composition contexts or raw values, not both"
            )
        if composition_context.shape != (batch, self.config.composition_dim):
            raise ValueError("packed composition_context has the wrong shape")
        dtype = z.dtype
        if not composition_context.is_floating_point():
            raise ValueError(
                "packed composition_context must be finite and floating point"
            )
        checks.add(
            ~torch.isfinite(composition_context).all(),
            partial(ValueError, "packed composition_context must be finite and floating point"),
        )
        if composition_context.device != z.device:
            raise ValueError(
                "packed composition_context must share the continuous-state device"
            )
        composition_context = composition_context.to(dtype=dtype)
        if atomic_numbers.ndim != 2:
            raise ValueError("packed atomic_numbers must have shape [B,N]")
        atom_count = atomic_numbers.shape[1]
        expected_shapes = {
            "atom_features": (batch, atom_count, self.config.atom_feature_dim),
            "fractional_coordinates": (batch, atom_count, 3),
            "lattice": (batch, 3, 3),
            "jacobian_cartesian": (
                batch,
                atom_count,
                3,
                intrinsic_dimension,
            ),
        }
        actual = {
            "atom_features": tuple(atom_features.shape),
            "fractional_coordinates": tuple(fractional_coordinates.shape),
            "lattice": tuple(lattice.shape),
            "jacobian_cartesian": tuple(jacobian_cartesian.shape),
        }
        for name, expected in expected_shapes.items():
            if actual[name] != expected:
                raise ValueError(f"packed {name} must have shape {expected}")
        if atomic_numbers.dtype == torch.bool or atomic_numbers.is_floating_point():
            raise ValueError("packed atomic_numbers must use an integer tensor dtype")
        if atomic_numbers.device != z.device:
            raise ValueError(
                "packed atomic_numbers must share the continuous-state device"
            )
        if atomic_numbers.shape != (batch, atom_count):
            raise ValueError("packed atoms must carry valid atomic numbers")
        checks.add(
            ((atomic_numbers <= 0) | (atomic_numbers > self.config.max_atomic_number)).any(),
            partial(ValueError, "packed atoms must carry valid atomic numbers"),
        )
        deterministic_tensors = {
            "atom_features": atom_features,
            "fractional_coordinates": fractional_coordinates,
            "lattice": lattice,
            "bounded_lattice_coordinates": bounded_lattice_coordinates,
            "jacobian_cartesian": jacobian_cartesian,
        }
        for name, value in deterministic_tensors.items():
            if not value.is_floating_point() or value.dtype != dtype:
                raise ValueError(
                    f"packed {name} must share the float32/float64 master dtype"
                )
            if value.device != z.device:
                raise ValueError(
                    f"packed {name} must share the continuous-state device"
                )
            checks.add(
                ~torch.isfinite(value).all(),
                partial(ValueError, f"packed {name} entries must be finite"),
            )
        parameter = next(self.parameters())
        if parameter.dtype != dtype or parameter.device != z.device:
            raise ValueError(
                "field parameters and packed states must share one master dtype/device"
            )

        # Establish an explicit master-precision boundary.  Inputs from the
        # exact cover bridge are FP32/FP64 even while the learned Transformer
        # and heads below participate in an outer AMP context.
        time_value = torch.as_tensor(time, dtype=z.dtype, device=z.device)
        if time_value.numel() == 1:
            time_value = time_value.reshape(1).expand(batch)
        elif time_value.shape != (batch,):
            raise ValueError("packed time must be scalar or have shape [B]")
        checks.add(
            ~torch.isfinite(time_value).all(),
            partial(ValueError, "packed flow times must be finite"),
        )
        time_features = self.time_features.batched(
            time_value, dtype=z.dtype, device=z.device, checks=checks
        )
        packed_commutant = None
        if self.config.lattice_head == "commutant-v2":
            packed_commutant = _commutant_basis(
                lattice_commutant_basis,
                dimension=lattice_dimension,
                dtype=z.dtype,
                device=z.device,
                checks=checks,
            )
            bounded_padded = bounded_lattice_coordinates @ packed_commutant.T
        else:
            bounded_padded = z.new_zeros(batch, self.config.max_lattice_dim)
            bounded_padded[:, :lattice_dimension] = bounded_lattice_coordinates
        global_features = torch.cat(
            (composition_context, time_features, bounded_padded), dim=-1
        )
        scalar_input = torch.cat(
            (
                self.species_embedding(atomic_numbers.to(torch.long)).to(dtype=z.dtype),
                atom_features,
                global_features[:, None, :].expand(batch, atom_count, -1),
            ),
            dim=-1,
        )
        packed_edges: PeriodicEdges | None
        if self.config.backbone == "invariant_global":
            packed_edges = None
            interaction_count = batch * atom_count * atom_count
            if interaction_count > self.config.max_dense_pair_interactions:
                raise ValueError(
                    "packed dense geometry interaction budget exceeded before "
                    f"allocation ({interaction_count} > "
                    f"{self.config.max_dense_pair_interactions})"
                )
            transformed = self.transformer(
                scalar_input,
                fractional_coordinates,
                lattice,
                validate_geometry=True,
                checks=checks,
            )
            neural_cartesian_vectors = self.cartesian_head(
                transformed.vector_channels
            ).squeeze(-2)
        else:
            with torch.autocast(device_type=z.device.type, enabled=False):
                packed_edges = enumerate_packed_periodic_edges(
                    tuple(fractional_coordinates[index] for index in range(batch)),
                    tuple(lattice[index] for index in range(batch)),
                    self.config.cutoff,
                    max_candidates_per_pair=self.config.max_candidates_per_pair,
                    max_edges=self.config.max_periodic_edges,
                )
            batch_index = torch.arange(
                batch, device=z.device, dtype=torch.long
            ).repeat_interleave(atom_count)
            transformed = self.transformer(
                scalar_input.reshape(batch * atom_count, -1),
                fractional_coordinates.reshape(batch * atom_count, 3),
                lattice[0],
                edges=_transformer_edges(packed_edges, z),
                batch_index=batch_index,
                num_graphs=batch,
            )
            neural_cartesian_vectors = (
                self.cartesian_head(transformed.vector_channels)
                .squeeze(-2)
                .reshape(batch, atom_count, 3)
            )
            interaction_count = packed_edges.num_edges
        if transformed.pooled_scalars.shape != (batch, self.config.scalar_dim):
            raise RuntimeError("packed Transformer returned a malformed pooled state")
        lattice_features = torch.cat(
            (transformed.pooled_scalars.to(dtype=dtype), global_features), dim=-1
        )
        neural_residual = self.lattice_residual_head(lattice_features)
        neural_linear = self.lattice_linear_head(lattice_features).reshape(
            batch, self.config.max_lattice_dim, self.config.max_lattice_dim
        )
        with torch.autocast(device_type=z.device.type, enabled=False):
            cartesian_vectors = neural_cartesian_vectors.to(dtype=dtype)
            intrinsic_velocity = reduce_cartesian_to_intrinsic_packed(
                cartesian_vectors,
                jacobian_cartesian,
                reference_length=self.config.reference_length,
                maximum_condition_number_float32=(
                    self.config.maximum_tangent_condition_number_float32
                ),
                maximum_condition_number_float64=(
                    self.config.maximum_tangent_condition_number_float64
                ),
                checks=checks,
            )
            scalar_channels = transformed.scalar_channels.to(dtype=dtype).reshape(
                batch, atom_count, self.config.scalar_dim
            )
            residual = self.config.lattice_residual_bound * neural_residual.to(dtype=dtype)
            linear = (
                self.config.lattice_linear_growth_bound
                * neural_linear.to(dtype=dtype)
                / float(self.config.max_lattice_dim)
            )
            if packed_commutant is None:
                lattice_velocity = (
                    torch.bmm(
                        linear[:, :lattice_dimension, :lattice_dimension],
                        z.unsqueeze(-1),
                    ).squeeze(-1)
                    + residual[:, :lattice_dimension]
                )
            else:
                canonical_z = z.to(dtype=dtype) @ packed_commutant.T
                canonical_velocity = (
                    torch.bmm(linear, canonical_z.unsqueeze(-1)).squeeze(-1) + residual
                )
                lattice_velocity = canonical_velocity @ packed_commutant
            coordinate_metric = translation_quotient_normal_matrix_packed(
                jacobian_cartesian, reference_length=self.config.reference_length, checks=checks
            )
        return PackedIntrinsicFieldOutput(
            lattice_velocity=lattice_velocity,
            intrinsic_velocity=intrinsic_velocity,
            cartesian_vectors=cartesian_vectors,
            atom_scalar_channels=scalar_channels,
            coordinate_metric=coordinate_metric,
            interaction_count=interaction_count,
            edges=packed_edges,
        )

    def forward_padded(
        self,
        time: Tensor,
        z: Tensor,
        eta: Tensor,
        composition_context: Tensor | None,
        *,
        atomic_numbers: Tensor,
        atom_mask: Tensor,
        atom_features: Tensor,
        fractional_coordinates: Tensor,
        lattice: Tensor,
        bounded_lattice_coordinates: Tensor,
        jacobian_cartesian: Tensor,
        lattice_dimensions: Tensor,
        intrinsic_dimensions: Tensor,
        composition_atomic_numbers: Tensor | None = None,
        composition_counts: Tensor | None = None,
        lattice_commutant_basis: Tensor | None = None,
    ) -> PaddedIntrinsicFieldOutput:
        """Evaluate crystals of different topologies in one dense call (review M5).

        Crystal ``b`` has ``lattice_dimensions[b]`` lattice coordinates (the
        leading slots of ``z[b]``), ``intrinsic_dimensions[b]`` torus coordinates
        (the leading slots of ``eta[b]``), and the atoms where ``atom_mask[b]`` is
        true.  Padded slots are masked inside, so arbitrary padding values
        cannot leak.  Real outputs equal independent :meth:`forward` calls to
        floating-point rounding, and the tangent rank certificate is still
        applied per crystal.  Only the dense ``invariant_global`` backbone is
        supported.
        """

        checks = DeviceChecks()
        with checks.deferred():
            return self._forward_padded(
                checks,
                time,
                z,
                eta,
                composition_context,
                atomic_numbers=atomic_numbers,
                atom_mask=atom_mask,
                atom_features=atom_features,
                fractional_coordinates=fractional_coordinates,
                lattice=lattice,
                bounded_lattice_coordinates=bounded_lattice_coordinates,
                jacobian_cartesian=jacobian_cartesian,
                lattice_dimensions=lattice_dimensions,
                intrinsic_dimensions=intrinsic_dimensions,
                composition_atomic_numbers=composition_atomic_numbers,
                composition_counts=composition_counts,
                lattice_commutant_basis=lattice_commutant_basis,
            )

    def _forward_padded(
        self,
        checks: DeviceChecks,
        time: Tensor,
        z: Tensor,
        eta: Tensor,
        composition_context: Tensor | None,
        *,
        atomic_numbers: Tensor,
        atom_mask: Tensor,
        atom_features: Tensor,
        fractional_coordinates: Tensor,
        lattice: Tensor,
        bounded_lattice_coordinates: Tensor,
        jacobian_cartesian: Tensor,
        lattice_dimensions: Tensor,
        intrinsic_dimensions: Tensor,
        composition_atomic_numbers: Tensor | None = None,
        composition_counts: Tensor | None = None,
        lattice_commutant_basis: Tensor | None = None,
    ) -> PaddedIntrinsicFieldOutput:
        if self.config.backbone != "invariant_global":
            raise ValueError("padded batching requires the dense invariant_global backbone")
        if z.ndim != 2 or eta.ndim != 2 or z.shape[0] != eta.shape[0] or z.shape[0] < 1:
            raise ValueError("padded z and eta must have shapes [B,Z] and [B,D] with B >= 1")
        if z.dtype not in {torch.float32, torch.float64} or eta.dtype != z.dtype:
            raise ValueError("padded z and eta must share a float32/float64 master dtype")
        if eta.device != z.device:
            raise ValueError("padded z and eta must share one device")
        batch, lattice_width = z.shape
        intrinsic_width = eta.shape[1]
        if lattice_width > self.config.max_lattice_dim:
            raise ValueError("padded z exceeds configured max_lattice_dim")
        lattice_columns = _padded_dimension_mask(
            lattice_dimensions.to(device=z.device),
            lattice_width,
            name="lattice_dimensions",
            checks=checks,
        )
        intrinsic_columns = _padded_dimension_mask(
            intrinsic_dimensions.to(device=z.device),
            intrinsic_width,
            name="intrinsic_dimensions",
            checks=checks,
        )
        if lattice_dimensions.shape != (batch,) or intrinsic_dimensions.shape != (batch,):
            raise ValueError("padded dimension tensors must have shape [B]")
        if atomic_numbers.ndim != 2 or atomic_numbers.shape[0] != batch:
            raise ValueError("padded atomic_numbers must have shape [B,N]")
        atom_count = atomic_numbers.shape[1]
        if atom_mask.dtype != torch.bool or atom_mask.shape != (batch, atom_count):
            raise ValueError("atom_mask must be a bool tensor of shape [B,N]")
        checks.add(
            ~atom_mask.any(dim=1).all(),
            partial(ValueError, "every padded crystal must keep at least one atom"),
        )
        if atomic_numbers.dtype == torch.bool or atomic_numbers.is_floating_point():
            raise ValueError("padded atomic_numbers must use an integer tensor dtype")
        invalid_numbers = (atomic_numbers <= 0) | (
            atomic_numbers > self.config.max_atomic_number
        )
        checks.add(
            (invalid_numbers & atom_mask).any(),
            partial(ValueError, "real padded atoms must carry valid atomic numbers"),
        )
        expected_shapes = {
            "atom_features": (batch, atom_count, self.config.atom_feature_dim),
            "fractional_coordinates": (batch, atom_count, 3),
            "lattice": (batch, 3, 3),
            "bounded_lattice_coordinates": (batch, lattice_width),
            "jacobian_cartesian": (batch, atom_count, 3, intrinsic_width),
        }
        values = {
            "atom_features": atom_features,
            "fractional_coordinates": fractional_coordinates,
            "lattice": lattice,
            "bounded_lattice_coordinates": bounded_lattice_coordinates,
            "jacobian_cartesian": jacobian_cartesian,
        }
        for name, expected in expected_shapes.items():
            value = values[name]
            if tuple(value.shape) != expected:
                raise ValueError(f"padded {name} must have shape {expected}")
            if value.dtype != z.dtype or value.device != z.device:
                raise ValueError(f"padded {name} must share the master dtype and device")
            checks.add(
                ~torch.isfinite(value).all(),
                partial(ValueError, f"padded {name} entries must be finite"),
            )
        checks.add(
            ~(torch.isfinite(z).all() & torch.isfinite(eta).all()),
            partial(ValueError, "padded z and eta entries must be finite"),
        )
        parameter = next(self.parameters())
        if parameter.dtype != z.dtype or parameter.device != z.device:
            raise ValueError(
                "field parameters and padded states must share one master dtype/device"
            )
        interaction_count = batch * atom_count * atom_count
        if interaction_count > self.config.max_dense_pair_interactions:
            raise ValueError(
                "padded dense geometry interaction budget exceeded before allocation "
                f"({interaction_count} > {self.config.max_dense_pair_interactions})"
            )
        if composition_context is None:
            if composition_atomic_numbers is None or composition_counts is None:
                raise ValueError("padded composition values are required without contexts")
            composition_context = self.encode_composition(
                composition_atomic_numbers, composition_counts
            )
        elif composition_atomic_numbers is not None or composition_counts is not None:
            raise ValueError("supply either padded composition contexts or raw values")
        if composition_context.shape != (batch, self.config.composition_dim):
            raise ValueError("padded composition_context has the wrong shape")
        dtype = z.dtype
        composition_context = composition_context.to(dtype=dtype)

        lattice_mask = lattice_columns.to(dtype=dtype)
        z_real = z * lattice_mask
        bounded_real = bounded_lattice_coordinates * lattice_mask
        time_value = torch.as_tensor(time, dtype=dtype, device=z.device)
        if time_value.numel() == 1:
            time_value = time_value.reshape(1).expand(batch)
        elif time_value.shape != (batch,):
            raise ValueError("padded time must be scalar or have shape [B]")
        checks.add(
            ~torch.isfinite(time_value).all(),
            partial(ValueError, "padded flow times must be finite"),
        )
        time_features = self.time_features.batched(
            time_value, dtype=dtype, device=z.device, checks=checks
        )
        commutant = None
        if self.config.lattice_head == "commutant-v2":
            if lattice_commutant_basis is None:
                raise ValueError(
                    "the commutant-v2 lattice head requires lattice_commutant_basis"
                )
            if lattice_commutant_basis.shape != (batch, 6, lattice_width):
                raise ValueError("padded lattice_commutant_basis must have shape [B,6,Z]")
            checks.add(
                ~torch.isfinite(lattice_commutant_basis).all(),
                partial(ValueError, "lattice_commutant_basis must be finite"),
            )
            commutant = lattice_commutant_basis.to(dtype=dtype) * lattice_mask[:, None, :]
            bounded_padded = torch.einsum("bkd,bd->bk", commutant, bounded_real)
        else:
            bounded_padded = z.new_zeros(batch, self.config.max_lattice_dim)
            bounded_padded[:, :lattice_width] = bounded_real
        global_features = torch.cat(
            (composition_context, time_features, bounded_padded), dim=-1
        )
        species = torch.where(atom_mask, atomic_numbers, torch.zeros_like(atomic_numbers))
        scalar_input = torch.cat(
            (
                self.species_embedding(species.to(torch.long)).to(dtype=dtype),
                atom_features,
                global_features[:, None, :].expand(batch, atom_count, -1),
            ),
            dim=-1,
        )
        transformed = self.transformer(
            scalar_input,
            fractional_coordinates,
            lattice,
            validate_geometry=True,
            atom_mask=atom_mask,
            checks=checks,
        )
        neural_cartesian_vectors = self.cartesian_head(transformed.vector_channels).squeeze(-2)
        lattice_features = torch.cat(
            (transformed.pooled_scalars.to(dtype=dtype), global_features), dim=-1
        )
        neural_residual = self.lattice_residual_head(lattice_features)
        neural_linear = self.lattice_linear_head(lattice_features).reshape(
            batch, self.config.max_lattice_dim, self.config.max_lattice_dim
        )
        with torch.autocast(device_type=z.device.type, enabled=False):
            real_atoms = atom_mask.to(dtype=dtype)
            cartesian_vectors = neural_cartesian_vectors.to(dtype=dtype) * real_atoms[..., None]
            intrinsic_velocity = reduce_cartesian_to_intrinsic_padded(
                cartesian_vectors,
                jacobian_cartesian,
                atom_mask,
                intrinsic_dimensions.to(device=z.device),
                reference_length=self.config.reference_length,
                maximum_condition_number_float32=(
                    self.config.maximum_tangent_condition_number_float32
                ),
                maximum_condition_number_float64=(
                    self.config.maximum_tangent_condition_number_float64
                ),
                checks=checks,
            )
            residual = self.config.lattice_residual_bound * neural_residual.to(dtype=dtype)
            linear = (
                self.config.lattice_linear_growth_bound
                * neural_linear.to(dtype=dtype)
                / float(self.config.max_lattice_dim)
            )
            if commutant is None:
                lattice_velocity = (
                    torch.bmm(
                        linear[:, :lattice_width, :lattice_width], z_real.unsqueeze(-1)
                    ).squeeze(-1)
                    + residual[:, :lattice_width]
                ) * lattice_mask
            else:
                canonical_z = torch.einsum("bkd,bd->bk", commutant, z_real)
                canonical_velocity = (
                    torch.bmm(linear, canonical_z.unsqueeze(-1)).squeeze(-1) + residual
                )
                lattice_velocity = torch.einsum("bkd,bk->bd", commutant, canonical_velocity)
            coordinate_metric = translation_quotient_normal_matrix_padded(
                jacobian_cartesian,
                atom_mask,
                intrinsic_dimensions.to(device=z.device),
                reference_length=self.config.reference_length,
                checks=checks,
            ).to(dtype=dtype)
            scalar_channels = transformed.scalar_channels.to(dtype=dtype)
        return PaddedIntrinsicFieldOutput(
            lattice_velocity=lattice_velocity,
            intrinsic_velocity=intrinsic_velocity * intrinsic_columns.to(dtype=dtype),
            cartesian_vectors=cartesian_vectors,
            atom_scalar_channels=scalar_channels,
            coordinate_metric=coordinate_metric,
            interaction_count=interaction_count,
            atom_mask=atom_mask,
            lattice_dimensions=lattice_dimensions,
            intrinsic_dimensions=intrinsic_dimensions,
        )
