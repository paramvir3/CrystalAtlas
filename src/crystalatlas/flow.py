"""Intrinsic conditional-flow-matching paths, loss, and ODE integration.

The functions implement equations (1.31)--(1.33), (1.37)/(3.14), and (3.16)
on ``R^d_g x (R^d_eta / Lambda)``.  Discrete composition and cover variables
are intentionally absent: one flow call always remains inside one fixed cover.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass

from ._optional import require_torch

torch = require_torch()
Tensor = torch.Tensor


@dataclass(frozen=True)
class FlowState:
    z: Tensor
    eta: Tensor


@dataclass(frozen=True)
class FlowVelocity:
    z: Tensor
    eta: Tensor


@dataclass(frozen=True)
class CFMPathSample:
    state: FlowState
    target_velocity: FlowVelocity
    torus_delta: Tensor


def validate_period_basis(period_basis: Tensor, dimension: int) -> None:
    """Refuse a period basis that is not ``[d, d]``, finite and nonsingular (two host reads)."""

    if period_basis.shape != (dimension, dimension):
        raise ValueError(f"period_basis must have shape [{dimension},{dimension}]")
    if dimension and not bool(torch.isfinite(period_basis).all().item()):
        raise ValueError("period_basis must be finite")
    if dimension and not bool((torch.linalg.det(period_basis).abs() > 0).item()):
        raise ValueError("period_basis must be nonsingular")


_validate_basis = validate_period_basis


def validate_period_basis_rows(period_basis: Tensor, batch: int, dimension: int) -> None:
    """Refuse row-wise period bases that are not ``[B, d, d]``, finite and nonsingular.

    Row ``b`` of a state ``eta[B, d]`` then lives on its own torus
    ``R^d / (Z^d B_b)``.  A mixed-cover solve pads each cover's basis with the
    identity, which leaves the padded coordinates at zero (two host reads).
    """

    if period_basis.shape != (batch, dimension, dimension):
        raise ValueError(f"period_basis must have shape [{batch},{dimension},{dimension}]")
    if not (batch and dimension):
        return
    if not bool(torch.isfinite(period_basis).all().item()):
        raise ValueError("period_basis must be finite")
    if not bool((torch.linalg.det(period_basis).abs() > 0).all().item()):
        raise ValueError("period_basis must be nonsingular")


def _right_coordinates(values: Tensor, period_basis: Tensor) -> Tensor:
    """Return row coefficients ``a`` satisfying ``a @ B == values``.

    A three-dimensional ``period_basis`` holds one basis per row of ``values``.
    """

    dimension = values.shape[-1]
    if dimension == 0:
        return values.clone()
    if period_basis.ndim == 3:
        if values.ndim != 2 or period_basis.shape != (values.shape[0], dimension, dimension):
            raise ValueError("row-wise period bases need values [B,d] and bases [B,d,d]")
        return torch.linalg.solve(
            period_basis.transpose(-1, -2), values.unsqueeze(-1)
        ).squeeze(-1)
    flat = values.reshape(-1, dimension)
    coefficients = torch.linalg.solve(period_basis.transpose(-1, -2), flat.T).T
    return coefficients.reshape_as(values)


def wrap_half_open(values: Tensor) -> Tensor:
    """Coordinatewise wrap to the deterministic interval ``[-1/2, 1/2)``."""

    return values - torch.floor(values + 0.5)


def canonicalize_torus(eta: Tensor, period_basis: Tensor) -> Tensor:
    """Return the representative with lattice coefficients in ``[0,1)``."""

    with torch.autocast(device_type=eta.device.type, enabled=False):
        _validate_basis(period_basis, eta.shape[-1])
    return canonicalize_torus_validated(eta, period_basis)


def canonicalize_torus_validated(eta: Tensor, period_basis: Tensor) -> Tensor:
    """:func:`canonicalize_torus` for a basis :func:`validate_period_basis` accepted.

    The integrator validates its period basis once per call and the bridge once
    at construction; re-reading finiteness and determinant at every RK4 stage
    cost two host synchronizations per stage (review 2026-10-29, H1).  Row-wise
    bases ``[B, d, d]`` (:func:`validate_period_basis_rows`) wrap row ``b`` of
    ``eta[B, d]`` on its own torus.
    """

    dimension = eta.shape[-1]
    # Lattice-coordinate solves and half-open wrapping define the state space;
    # they are not neural kernels and must retain their input/master dtype.
    with torch.autocast(device_type=eta.device.type, enabled=False):
        if dimension == 0:
            return eta.clone()
        coefficients = _right_coordinates(eta, period_basis)
        coefficients = coefficients - torch.floor(coefficients)
        if period_basis.ndim == 3:
            return torch.einsum("bi,bij->bj", coefficients, period_basis)
        return coefficients @ period_basis


def shortest_coordinate_lift(
    eta_start: Tensor, eta_end: Tensor, period_basis: Tensor
) -> Tensor:
    """Equation (1.32), which is a coordinate lift, not a Cartesian geodesic."""

    if eta_start.shape != eta_end.shape:
        raise ValueError("torus endpoints must have identical shapes")
    dimension = eta_start.shape[-1]
    with torch.autocast(device_type=eta_start.device.type, enabled=False):
        _validate_basis(period_basis, dimension)
        if dimension == 0:
            return eta_start.clone()
        coefficients = _right_coordinates(eta_end - eta_start, period_basis)
        return wrap_half_open(coefficients) @ period_basis


def _time_for_state(time: Tensor, state: Tensor) -> Tensor:
    value = torch.as_tensor(time, dtype=state.dtype, device=state.device)
    while value.ndim < state.ndim:
        value = value.unsqueeze(-1)
    try:
        return torch.broadcast_to(value, state.shape[:-1] + (1,))
    except RuntimeError as exc:
        raise ValueError("time is not broadcastable over the state batch") from exc


def _validate_torus_action(action: Tensor, eta: Tensor) -> None:
    """Validate tensor metadata; the caller certifies the lattice automorphism."""

    dimension = eta.shape[-1]
    if action.shape[-2:] != (dimension, dimension):
        raise ValueError("torus_action has the wrong terminal dimensions")
    if action.dtype != eta.dtype or action.device != eta.device:
        raise ValueError("torus_action must share the state dtype and device")
    if not bool(torch.isfinite(action).all().item()):
        raise ValueError("torus_action must be finite")
    try:
        action_batch = torch.broadcast_shapes(eta.shape[:-1], action.shape[:-2])
    except RuntimeError as exc:
        raise ValueError("torus_action is not broadcastable over the state batch") from exc
    if action_batch != eta.shape[:-1]:
        raise ValueError("torus_action must preserve the state batch shape")


def conditional_flow_path(
    z_start: Tensor,
    eta_start: Tensor,
    z_end: Tensor,
    eta_end: Tensor,
    time: Tensor,
    period_basis: Tensor,
    *,
    torus_action: Tensor | None = None,
    torus_translation: Tensor | None = None,
) -> CFMPathSample:
    """Construct and optionally push forward the complete sampled CFM path.

    A separately certified row action transforms both the path state and its
    tangent, after choosing the half-open lift. Choosing a new half-open lift
    between already transformed endpoints is not equivalent for nonorthogonal
    actions. Uniform finite-group path augmentation makes every intermediate
    marginal and its probability current invariant/covariant, not just t=1.
    The transformed tangent is never wrapped: it may leave the original lift
    parallelepiped while remaining a valid velocity on the torus.

    ``torus_translation`` (``[..., D]``, Cartesian ``eta`` units) completes the
    affine action ``eta -> eta A + m`` of a residual normalizer gauge.  It shifts
    the state after the linear part and leaves the tangent unchanged.
    """

    if z_start.shape != z_end.shape:
        raise ValueError("z endpoints must have identical shapes")
    if eta_start.shape != eta_end.shape:
        raise ValueError("eta endpoints must have identical shapes")
    if z_start.shape[:-1] != eta_start.shape[:-1]:
        raise ValueError("z and eta endpoints must have the same batch shape")
    if torus_action is not None:
        _validate_torus_action(torus_action, eta_start)
    with torch.autocast(device_type=z_start.device.type, enabled=False):
        torus_delta = shortest_coordinate_lift(eta_start, eta_end, period_basis)
        z_time = _time_for_state(time, z_start)
        eta_time = _time_for_state(time, eta_start)
        z_current = (1.0 - z_time) * z_start + z_time * z_end
        eta_current = canonicalize_torus(eta_start + eta_time * torus_delta, period_basis)
        if torus_action is not None:
            eta_current = canonicalize_torus(
                torch.matmul(eta_current.unsqueeze(-2), torus_action).squeeze(-2), period_basis
            )
            torus_delta = torch.matmul(torus_delta.unsqueeze(-2), torus_action).squeeze(-2)
        if torus_translation is not None:
            shift = torus_translation.to(dtype=eta_current.dtype, device=eta_current.device)
            if shift.shape[-1:] != eta_current.shape[-1:]:
                raise ValueError("torus_translation must end in the torus dimension")
            eta_current = canonicalize_torus(eta_current + shift, period_basis)
        return CFMPathSample(
            state=FlowState(z=z_current, eta=eta_current),
            target_velocity=FlowVelocity(z=z_end - z_start, eta=torus_delta),
            torus_delta=torus_delta,
        )


def sample_product_base(
    z_dimension: int,
    period_basis: Tensor,
    *,
    batch_shape: tuple[int, ...] = (),
    dtype: torch.dtype | None = None,
    device: torch.device | None = None,
    generator: torch.Generator | None = None,
) -> FlowState:
    """Sample ``N(0,I) x Haar(R^d/Lambda)`` before residual-group averaging."""

    if z_dimension < 0:
        raise ValueError("z_dimension must be nonnegative")
    dtype = dtype or period_basis.dtype
    device = device or period_basis.device
    target_device = torch.device(device)
    eta_dimension = period_basis.shape[-1]
    with torch.autocast(device_type=target_device.type, enabled=False):
        _validate_basis(period_basis, eta_dimension)
        z = torch.randn(
            (*batch_shape, z_dimension),
            dtype=dtype,
            device=target_device,
            generator=generator,
        )
        coefficients = torch.rand(
            (*batch_shape, eta_dimension),
            dtype=dtype,
            device=target_device,
            generator=generator,
        )
        eta = coefficients @ period_basis.to(dtype=dtype, device=target_device)
        return FlowState(z=z, eta=eta)


def sample_dequantized_endpoint(
    z_data: Tensor,
    eta_data: Tensor,
    period_basis: Tensor,
    *,
    sigma_z: float,
    sigma_eta: float,
    generator: torch.Generator | None = None,
    torus_action: Tensor | None = None,
    eta_noise_transform: Tensor | None = None,
) -> FlowState:
    """Sample the Gaussian--wrapped-Gaussian (torus heat) endpoint kernel.

    Without ``eta_noise_transform`` the torus kernel is ``sigma_eta^2 I`` in the
    stored ``eta`` coordinates, which are fractional; in space that noise is
    anisotropic and scales with the cell (the legacy ``fractional-v1`` kernel).
    With ``eta_noise_transform`` ``T`` (``[..., D, D]``, broadcast over the batch)
    the noise is ``epsilon @ T`` with ``epsilon ~ N(0, I)``, so its covariance is
    ``T^T T``.  :func:`crystalatlas.nn.bridge.intrinsic_noise_transform` builds
    ``T`` for the Cartesian-metric kernel ``cartesian-v2``.  ``sigma_eta`` is
    then unused but must still be positive.  The random stream (``z`` noise, then
    ``eta`` noise) is the same in both cases.

    ``torus_action`` is a separately certified lattice automorphism acting on
    row coordinates.  It acts on the entire noisy endpoint, not only its mean.
    This distinction is necessary when an origin-anchor exchange is
    unimodular but nonorthogonal: isotropic noise is not invariant under that
    action.  Uniformly sampling the action therefore samples the exact finite
    group average of the dequantization kernel.  Batched actions may have
    shape ``[...,d_eta,d_eta]`` and broadcast over the endpoint batch.
    This option alone does not symmetrize intermediate CFM marginals. The
    production objective instead samples this kernel without an action and
    supplies the action once to ``conditional_flow_path``.
    """

    if sigma_z <= 0.0 or sigma_eta <= 0.0:
        raise ValueError("the theorem-compatible dequantization scales are positive")
    if z_data.shape[:-1] != eta_data.shape[:-1]:
        raise ValueError("z_data and eta_data must have the same batch shape")
    if torus_action is not None:
        _validate_torus_action(torus_action, eta_data)
    with torch.autocast(device_type=z_data.device.type, enabled=False):
        z_noise = torch.randn(
            z_data.shape,
            dtype=z_data.dtype,
            device=z_data.device,
            generator=generator,
        )
        eta_noise = torch.randn(
            eta_data.shape,
            dtype=eta_data.dtype,
            device=eta_data.device,
            generator=generator,
        )
        if eta_noise_transform is None:
            noisy_eta = eta_data + sigma_eta * eta_noise
        else:
            transform = eta_noise_transform.to(dtype=eta_data.dtype, device=eta_data.device)
            dimension = eta_data.shape[-1]
            if transform.shape[-2:] != (dimension, dimension):
                raise ValueError(
                    "eta_noise_transform must end in [D, D] for the torus dimension"
                )
            if not bool(torch.isfinite(transform).all().item()):
                raise ValueError("eta_noise_transform must be finite")
            noisy_eta = eta_data + torch.matmul(eta_noise.unsqueeze(-2), transform).squeeze(-2)
        if torus_action is not None:
            noisy_eta = torch.matmul(noisy_eta.unsqueeze(-2), torus_action).squeeze(-2)
        return FlowState(
            z=z_data + sigma_z * z_noise,
            eta=canonicalize_torus(noisy_eta, period_basis),
        )


def _quadratic(vector: Tensor, matrix: Tensor) -> Tensor:
    if vector.shape[-1] == 0:
        return vector.new_zeros(vector.shape[:-1])
    try:
        return torch.einsum("...i,...ij,...j->...", vector, matrix, vector)
    except RuntimeError as exc:
        raise ValueError("metric is not broadcastable over its velocity block") from exc


def _validate_positive_metric(matrix: Tensor, dimension: int, name: str) -> None:
    if matrix.shape[-2:] != (dimension, dimension):
        raise ValueError(f"{name} has the wrong terminal dimensions")
    if dimension == 0:
        return
    if not torch.allclose(matrix, matrix.transpose(-1, -2), rtol=1e-5, atol=1e-7):
        raise ValueError(f"{name} must be symmetric")
    _, info = torch.linalg.cholesky_ex(matrix)
    if bool((info != 0).any().item()):
        raise ValueError(f"{name} must be positive definite at the current state")


@dataclass(frozen=True)
class CFMLossTerms:
    total: Tensor
    lattice: Tensor
    coordinates: Tensor
    per_sample: Tensor


def conditional_flow_matching_loss(
    predicted: FlowVelocity,
    target: FlowVelocity,
    *,
    current_lattice_metric: Tensor,
    current_coordinate_metric: Tensor,
    atom_count: int | Tensor,
    reduction: str = "mean",
    validate_metrics: bool = True,
    return_terms: bool = False,
    lattice_weight: float = 1.0,
    coordinate_weight: float = 1.0,
) -> Tensor | CFMLossTerms:
    """Current-state block-metric CFM loss from equation (3.14).

    The argument names intentionally say ``current_*``.  Passing an endpoint-
    dependent metric changes the population minimizer and is not ordinary CFM.
    ``current_coordinate_metric`` is the dimensionless
    ``J_hat^T P_N J_hat`` and is divided by the expanded advertised atom count.
    ``lattice_weight`` and ``coordinate_weight`` are the block weights
    ``lambda_z, lambda_eta`` of (3.14) (review 2026-10-29, m2): current-state
    constants, so Theorem 2.9 keeps the population minimizer.  They scale
    ``total`` and ``per_sample``; the returned ``lattice`` and ``coordinates``
    terms stay unweighted.  Unit weights reproduce the legacy loss bitwise.
    """

    for name, value in (
        ("lattice_weight", lattice_weight),
        ("coordinate_weight", coordinate_weight),
    ):
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or float(value) <= 0.0
        ):
            raise ValueError(f"{name} must be a finite positive number")

    if predicted.z.shape != target.z.shape or predicted.eta.shape != target.eta.shape:
        raise ValueError("predicted and target velocity blocks must agree")
    if predicted.z.shape[:-1] != predicted.eta.shape[:-1]:
        raise ValueError("velocity blocks must have the same batch shape")
    if validate_metrics:
        _validate_positive_metric(
            current_lattice_metric, predicted.z.shape[-1], "current_lattice_metric"
        )
        _validate_positive_metric(
            current_coordinate_metric,
            predicted.eta.shape[-1],
            "current_coordinate_metric",
        )
    z_error = predicted.z - target.z
    eta_error = predicted.eta - target.eta
    lattice_term = _quadratic(z_error, current_lattice_metric)
    coordinate_term = _quadratic(eta_error, current_coordinate_metric)
    count = torch.as_tensor(
        atom_count, dtype=coordinate_term.dtype, device=coordinate_term.device
    )
    if torch.any(count <= 0):
        raise ValueError("atom_count must be positive")
    coordinate_term = coordinate_term / count
    per_sample = (
        float(lattice_weight) * lattice_term + float(coordinate_weight) * coordinate_term
    )
    if reduction == "none":
        total = per_sample
        lattice_reduced = lattice_term
        coordinate_reduced = coordinate_term
    elif reduction == "sum":
        total = per_sample.sum()
        lattice_reduced = lattice_term.sum()
        coordinate_reduced = coordinate_term.sum()
    elif reduction == "mean":
        total = per_sample.mean()
        lattice_reduced = lattice_term.mean()
        coordinate_reduced = coordinate_term.mean()
    else:
        raise ValueError("reduction must be 'none', 'sum', or 'mean'")
    if return_terms:
        return CFMLossTerms(
            total=total,
            lattice=lattice_reduced,
            coordinates=coordinate_reduced,
            per_sample=per_sample,
        )
    return total


def _coerce_velocity(value: object) -> FlowVelocity:
    if isinstance(value, FlowVelocity):
        return value
    if isinstance(value, tuple) and len(value) == 2:
        return FlowVelocity(z=value[0], eta=value[1])
    if hasattr(value, "lattice_velocity") and hasattr(value, "intrinsic_velocity"):
        return FlowVelocity(
            z=value.lattice_velocity,
            eta=value.intrinsic_velocity,
        )
    raise TypeError(
        "field must return FlowVelocity, (z_velocity, eta_velocity), or an "
        "object with lattice_velocity/intrinsic_velocity"
    )


def _wrapped_add(
    state: FlowState,
    velocity: FlowVelocity,
    scale: Tensor,
    period_basis: Tensor,
) -> FlowState:
    with torch.autocast(device_type=state.z.device.type, enabled=False):
        return FlowState(
            z=state.z + scale * velocity.z,
            eta=canonicalize_torus_validated(state.eta + scale * velocity.eta, period_basis),
        )


def rk4_step(
    field: Callable[[Tensor, Tensor, Tensor], object],
    state: FlowState,
    time: Tensor,
    step_size: Tensor,
    period_basis: Tensor,
) -> FlowState:
    """One differentiable classical fourth-order Runge--Kutta step."""

    _validate_basis(period_basis, state.eta.shape[-1])
    return _rk4_step(field, state, time, step_size, period_basis)


def _rk4_step(
    field: Callable[[Tensor, Tensor, Tensor], object],
    state: FlowState,
    time: Tensor,
    step_size: Tensor,
    period_basis: Tensor,
) -> FlowState:
    """:func:`rk4_step` on a period basis the integrator has validated."""

    k1 = _coerce_velocity(field(time, state.z, state.eta))
    half = step_size * 0.5
    second = _wrapped_add(state, k1, half, period_basis)
    k2 = _coerce_velocity(field(time + half, second.z, second.eta))
    third = _wrapped_add(state, k2, half, period_basis)
    k3 = _coerce_velocity(field(time + half, third.z, third.eta))
    fourth = _wrapped_add(state, k3, step_size, period_basis)
    k4 = _coerce_velocity(field(time + step_size, fourth.z, fourth.eta))
    combined = FlowVelocity(
        z=(k1.z + 2.0 * k2.z + 2.0 * k3.z + k4.z) / 6.0,
        eta=(k1.eta + 2.0 * k2.eta + 2.0 * k3.eta + k4.eta) / 6.0,
    )
    return _wrapped_add(state, combined, step_size, period_basis)


def integrate_with_endpoint_estimate(
    field: Callable[[Tensor, Tensor, Tensor], object],
    initial_state: FlowState,
    period_basis: Tensor,
    *,
    num_steps: int,
    terminal_time: float = 1.0,
) -> FlowState:
    """Integrate to ``terminal_time`` and return the CFM endpoint estimate.

    For straight conditional paths the target velocity is ``u = x_1 - x_0``, so
    ``x_1 = x_t + (1 - t) u`` and ``E[x_1 | x_t] = x_t + (1 - t) E[u | x_t]``.  With
    ``terminal_time = 1`` this is plain RK4 to one; otherwise the stiff end of
    the trajectory is replaced by that estimate (review M4).  Torus coordinates
    use the same wrapped addition as the integrator.
    """

    if not (math.isfinite(terminal_time) and 0.0 < terminal_time <= 1.0):
        raise ValueError("terminal_time must lie in (0, 1]")
    result = integrate_rk4(
        field, initial_state, period_basis, num_steps=num_steps, t_end=terminal_time
    )
    if terminal_time == 1.0:
        return result.final_state
    state = result.final_state
    velocity = _coerce_velocity(field(state.z.new_tensor(terminal_time), state.z, state.eta))
    return _wrapped_add(state, velocity, state.z.new_tensor(1.0 - terminal_time), period_basis)


def base_log_density(z: Tensor, period_basis: Tensor) -> Tensor:
    """Log density of the product base ``N(0, I) x Haar`` (Lebesgue in ``z`` and ``eta``).

    The residual-group average (1.30b) has the same density because the raw base
    is invariant.  The torus factor is ``1 / |det B|`` for the period basis ``B``.
    """

    if z.ndim < 1:
        raise ValueError("z must have a trailing lattice dimension")
    dimension = period_basis.shape[-1]
    _validate_basis(period_basis, dimension)
    log_volume = (
        torch.linalg.slogdet(period_basis.to(dtype=torch.float64)).logabsdet
        if dimension
        else torch.zeros((), dtype=torch.float64)
    )
    value = -0.5 * (z.to(dtype=torch.float64) ** 2).sum(-1)
    value = value - 0.5 * z.shape[-1] * math.log(2.0 * math.pi) - log_volume.to(z.device)
    return value.to(dtype=z.dtype)


EXACT_DIVERGENCES: tuple[str, ...] = ("exact", "exact-batched")
DIVERGENCES: tuple[str, ...] = (*EXACT_DIVERGENCES, "hutchinson")


def _replicated_velocity_and_divergence(
    field: Callable[[Tensor, Tensor, Tensor], object],
    time: Tensor,
    z: Tensor,
    eta: Tensor,
    *,
    row_chunk: int | None,
) -> tuple[FlowVelocity, Tensor]:
    """``(v, div v)`` with the exact trace taken on a replicated batch.

    Member ``b`` is copied ``c`` times and copy ``(b, k)`` contributes velocity
    coordinate ``k`` to one scalar ``S = sum_{b,k} v_k(q^{(b,k)})``.  Because the
    field acts on every row independently, ``dS/dq^{(b,k)}`` is row ``k`` of the
    Jacobian of member ``b``, so ``div v(q^{(b)}) = sum_k [dS/dq^{(b,k)}]_k`` from
    one reverse pass.  This is the same exact trace as the coordinate loop of
    ``divergence="exact"`` (equal up to rounding) at one forward and one reverse
    pass per ``row_chunk`` coordinates, instead of one reverse pass per
    coordinate.  ``field`` must accept any number of rows.
    """

    batch, lattice_dimension = z.shape
    dimension = lattice_dimension + eta.shape[-1]
    if dimension == 0:
        with torch.no_grad():
            plain = _coerce_velocity(field(time, z.detach(), eta.detach()))
        zero = torch.zeros(batch, dtype=z.dtype, device=z.device)
        return FlowVelocity(plain.z.detach(), plain.eta.detach()), zero
    chunk = dimension if row_chunk is None else min(row_chunk, dimension)
    total = torch.zeros(batch, dtype=z.dtype, device=z.device)
    published: FlowVelocity | None = None
    for start in range(0, dimension, chunk):
        rows = torch.arange(start, min(start + chunk, dimension), device=z.device)
        copies = int(rows.numel())
        z_probe = z.detach().repeat_interleave(copies, dim=0).requires_grad_(True)
        eta_probe = eta.detach().repeat_interleave(copies, dim=0).requires_grad_(True)
        row_time = time if time.ndim == 0 else time.repeat_interleave(copies, dim=0)
        # Row (b, k) of the replicated batch selects velocity coordinate rows[k].
        selector = rows.repeat(batch)[:, None]
        with torch.enable_grad():
            velocity = _coerce_velocity(field(row_time, z_probe, eta_probe))
            if velocity.z.shape != z_probe.shape or velocity.eta.shape != eta_probe.shape:
                raise ValueError(
                    "exact-batched divergence needs a field that maps any number of "
                    "rows to velocities of the same shape"
                )
            stacked = torch.cat((velocity.z, velocity.eta), dim=-1)
            grads = torch.autograd.grad(
                stacked.gather(1, selector).sum(), (z_probe, eta_probe), allow_unused=True
            )
        jacobian_rows = torch.cat(
            [
                torch.zeros_like(probe) if grad is None else grad
                for grad, probe in zip(grads, (z_probe, eta_probe), strict=True)
            ],
            dim=-1,
        )
        total = total + jacobian_rows.gather(1, selector).reshape(batch, copies).sum(dim=1)
        if published is None:
            published = FlowVelocity(
                velocity.z.detach().reshape(batch, copies, -1)[:, 0],
                velocity.eta.detach().reshape(batch, copies, -1)[:, 0],
            )
    if published is None:  # the loop runs at least once because dimension >= 1
        raise RuntimeError("replicated divergence produced no velocity")
    return published, total.detach()


def _velocity_and_divergence(
    field: Callable[[Tensor, Tensor, Tensor], object],
    time: Tensor,
    z: Tensor,
    eta: Tensor,
    *,
    divergence: str,
    probes: int,
    generator: torch.Generator | None,
    row_chunk: int | None = None,
) -> tuple[FlowVelocity, Tensor]:
    """``(v, div v)`` at ``(t, z, eta)`` for a batch; exact trace or Hutchinson."""

    if divergence == "exact-batched":
        return _replicated_velocity_and_divergence(
            field, time, z, eta, row_chunk=row_chunk
        )
    z_probe = z.detach().requires_grad_(True)
    eta_probe = eta.detach().requires_grad_(True)
    inputs = (z_probe, eta_probe)
    with torch.enable_grad():
        velocity = _coerce_velocity(field(time, z_probe, eta_probe))
        batch = z.shape[0]
        total = torch.zeros(batch, dtype=z.dtype, device=z.device)
        blocks = ((velocity.z, 0), (velocity.eta, 1))
        if divergence == "exact":
            for block, which in blocks:
                for k in range(block.shape[-1]):
                    grads = torch.autograd.grad(
                        block[:, k].sum(), inputs, retain_graph=True, allow_unused=True
                    )
                    grad = grads[which]
                    if grad is not None:
                        total = total + grad[:, k]
        elif divergence == "hutchinson":
            for _ in range(probes):
                noise_z = torch.randint(
                    0, 2, z.shape, generator=generator, device=z.device, dtype=z.dtype
                ) * 2.0 - 1.0
                noise_eta = torch.randint(
                    0, 2, eta.shape, generator=generator, device=z.device, dtype=z.dtype
                ) * 2.0 - 1.0
                grads = torch.autograd.grad(
                    (velocity.z * noise_z).sum() + (velocity.eta * noise_eta).sum(),
                    inputs,
                    retain_graph=True,
                    allow_unused=True,
                )
                if grads[0] is not None:
                    total = total + (grads[0] * noise_z).sum(-1)
                if grads[1] is not None:
                    total = total + (grads[1] * noise_eta).sum(-1)
            total = total / probes
        else:
            raise ValueError(f"divergence must be one of {DIVERGENCES}")
    return FlowVelocity(velocity.z.detach(), velocity.eta.detach()), total.detach()


def log_density_transport(
    field: Callable[[Tensor, Tensor, Tensor], object],
    final_state: FlowState,
    period_basis: Tensor,
    *,
    num_steps: int,
    divergence: str = "exact",
    probes: int = 8,
    generator: torch.Generator | None = None,
    row_chunk: int | None = None,
) -> tuple[FlowState, Tensor]:
    """Backward RK4 from ``t = 1`` with the divergence integral (research R2).

    Returns the base point ``q_0`` reached from ``final_state`` and
    ``int_0^1 div v(t, q_t) dt`` along that trajectory, so that the flow's log
    density at the final state is ``base_log_density(q_0) - integral``
    (instantaneous change of variables on the flat torus times ``R^{d_z}``).
    ``divergence="exact"`` differentiates every coordinate in its own reverse
    pass; ``"exact-batched"`` takes the same exact trace on a replicated batch
    (``row_chunk`` coordinates per pass, all of them by default) and needs a
    field that accepts any number of rows; ``"hutchinson"`` uses ``probes``
    Rademacher probes.  The RK4 stages are the production integrator's, run on
    the time-reversed field.
    """

    if isinstance(num_steps, bool) or not isinstance(num_steps, int) or num_steps < 1:
        raise ValueError("num_steps must be a positive integer")
    if isinstance(probes, bool) or not isinstance(probes, int) or probes < 1:
        raise ValueError("probes must be a positive integer")
    if divergence not in DIVERGENCES:
        raise ValueError(f"divergence must be one of {DIVERGENCES}")
    if row_chunk is not None and (
        isinstance(row_chunk, bool) or not isinstance(row_chunk, int) or row_chunk < 1
    ):
        raise ValueError("row_chunk must be a positive integer or None")
    if final_state.z.ndim != 2 or final_state.eta.ndim != 2:
        raise ValueError("log_density_transport needs batched states [B, d]")
    _validate_basis(period_basis, final_state.eta.shape[-1])
    canonical = canonicalize_torus_validated(final_state.eta, period_basis)
    state = FlowState(final_state.z, canonical)
    step = state.z.new_tensor(1.0 / num_steps)
    half = step * 0.5
    integral = torch.zeros(state.z.shape[0], dtype=state.z.dtype, device=state.z.device)

    def stage(point: FlowState, elapsed: Tensor) -> tuple[FlowVelocity, Tensor]:
        velocity, div = _velocity_and_divergence(
            field,
            1.0 - elapsed,
            point.z,
            point.eta,
            divergence=divergence,
            probes=probes,
            generator=generator,
            row_chunk=row_chunk,
        )
        return FlowVelocity(-velocity.z, -velocity.eta), div

    elapsed = state.z.new_tensor(0.0)
    for _ in range(num_steps):
        k1, d1 = stage(state, elapsed)
        second = _wrapped_add(state, k1, half, period_basis)
        k2, d2 = stage(second, elapsed + half)
        third = _wrapped_add(state, k2, half, period_basis)
        k3, d3 = stage(third, elapsed + half)
        fourth = _wrapped_add(state, k3, step, period_basis)
        k4, d4 = stage(fourth, elapsed + step)
        combined = FlowVelocity(
            z=(k1.z + 2.0 * k2.z + 2.0 * k3.z + k4.z) / 6.0,
            eta=(k1.eta + 2.0 * k2.eta + 2.0 * k3.eta + k4.eta) / 6.0,
        )
        state = _wrapped_add(state, combined, step, period_basis)
        integral = integral + step * (d1 + 2.0 * d2 + 2.0 * d3 + d4) / 6.0
        elapsed = elapsed + step
    return state, integral


@dataclass(frozen=True)
class RK4Result:
    final_state: FlowState
    times: Tensor | None = None
    z_trajectory: Tensor | None = None
    eta_trajectory: Tensor | None = None


def integrate_rk4(
    field: Callable[[Tensor, Tensor, Tensor], object],
    initial_state: FlowState,
    period_basis: Tensor,
    *,
    num_steps: int,
    t_start: float = 0.0,
    t_end: float = 1.0,
    return_trajectory: bool = False,
) -> RK4Result:
    """Integrate ``dq/dt = field(t,q)`` without detaching autograd.

    Fixed-step RK4 is deterministic and differentiable almost everywhere,
    including through model parameters and the initial state.  Torus wrapping
    uses ``floor`` and therefore has the correct unit derivative away from its
    measure-zero representative boundaries.  Numerical RK4 accuracy is not a
    proof of the global-existence hypothesis for the learned vector field.
    """

    if num_steps <= 0:
        raise ValueError("num_steps must be positive")
    if not t_end > t_start:
        raise ValueError("t_end must be greater than t_start")
    if initial_state.z.shape[:-1] != initial_state.eta.shape[:-1]:
        raise ValueError("z and eta must have the same batch shape")
    if period_basis.ndim == 3:
        if initial_state.eta.ndim != 2:
            raise ValueError("row-wise period bases need a state of shape [B,d]")
        validate_period_basis_rows(period_basis, *initial_state.eta.shape)
    else:
        _validate_basis(period_basis, initial_state.eta.shape[-1])
    time = initial_state.z.new_tensor(t_start)
    step_size = initial_state.z.new_tensor((t_end - t_start) / num_steps)
    state = FlowState(
        z=initial_state.z,
        eta=canonicalize_torus_validated(initial_state.eta, period_basis),
    )
    times = [time]
    z_values = [state.z]
    eta_values = [state.eta]
    for _ in range(num_steps):
        state = _rk4_step(field, state, time, step_size, period_basis)
        time = time + step_size
        if return_trajectory:
            times.append(time)
            z_values.append(state.z)
            eta_values.append(state.eta)
    if not return_trajectory:
        return RK4Result(final_state=state)
    return RK4Result(
        final_state=state,
        times=torch.stack(times),
        z_trajectory=torch.stack(z_values),
        eta_trajectory=torch.stack(eta_values),
    )
