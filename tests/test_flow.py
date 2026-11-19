from __future__ import annotations

import math
from itertools import permutations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from crystalatlas.composition import Composition  # noqa: E402
from crystalatlas.flow import (  # noqa: E402
    FlowState,
    FlowVelocity,
    canonicalize_torus,
    conditional_flow_matching_loss,
    conditional_flow_path,
    integrate_rk4,
    sample_dequantized_endpoint,
    shortest_coordinate_lift,
    wrap_half_open,
)
from crystalatlas.geometry import ContinuousCover, IntrinsicCoverLayout, OrbitType  # noqa: E402

pytestmark = pytest.mark.ml


def test_half_open_tie_rule_and_torus_path() -> None:
    values = torch.tensor([-0.5, 0.5, 1.5, -1.5], dtype=torch.float64)
    torch.testing.assert_close(
        wrap_half_open(values),
        torch.tensor([-0.5, -0.5, -0.5, -0.5], dtype=torch.float64),
    )
    basis = torch.tensor([[2.0, 0.0], [0.5, 1.5]], dtype=torch.float64)
    eta0 = torch.tensor([0.1, 0.2], dtype=torch.float64)
    eta1 = eta0 + torch.tensor([0.75, -0.8], dtype=torch.float64) @ basis
    delta = shortest_coordinate_lift(eta0, eta1, basis)
    expected = torch.tensor([-0.25, 0.2], dtype=torch.float64) @ basis
    torch.testing.assert_close(delta, expected, rtol=1.0e-12, atol=1.0e-12)
    path0 = conditional_flow_path(
        torch.tensor([1.0]), eta0, torch.tensor([3.0]), eta1, torch.tensor(0.0), basis
    )
    path1 = conditional_flow_path(
        torch.tensor([1.0]), eta0, torch.tensor([3.0]), eta1, torch.tensor(1.0), basis
    )
    torch.testing.assert_close(path0.state.eta, canonicalize_torus(eta0, basis))
    torch.testing.assert_close(path1.state.eta, canonicalize_torus(eta1, basis))
    torch.testing.assert_close(path1.target_velocity.eta, expected)


def test_current_state_block_metric_loss_matches_equation() -> None:
    predicted = FlowVelocity(
        z=torch.tensor([[2.0, -1.0]], dtype=torch.float64),
        eta=torch.tensor([[0.5]], dtype=torch.float64),
    )
    target = FlowVelocity(
        z=torch.tensor([[1.0, 1.0]], dtype=torch.float64),
        eta=torch.tensor([[-0.5]], dtype=torch.float64),
    )
    lattice_metric = torch.tensor([[2.0, 0.3], [0.3, 1.0]], dtype=torch.float64)
    coordinate_metric = torch.tensor([[6.0]], dtype=torch.float64)
    loss = conditional_flow_matching_loss(
        predicted,
        target,
        current_lattice_metric=lattice_metric,
        current_coordinate_metric=coordinate_metric,
        atom_count=3,
    )
    z_error = predicted.z[0] - target.z[0]
    expected = z_error @ lattice_metric @ z_error + 6.0 / 3.0
    torch.testing.assert_close(loss, expected)


def test_rk4_is_accurate_and_differentiable() -> None:
    parameter = torch.tensor(0.7, dtype=torch.float64, requires_grad=True)
    initial_z = torch.tensor([1.2], dtype=torch.float64, requires_grad=True)
    initial = FlowState(z=initial_z, eta=torch.empty(0, dtype=torch.float64))
    basis = torch.empty(0, 0, dtype=torch.float64)

    def field(time: torch.Tensor, z: torch.Tensor, eta: torch.Tensor) -> FlowVelocity:
        del time
        return FlowVelocity(z=parameter * z, eta=torch.zeros_like(eta))

    result = integrate_rk4(field, initial, basis, num_steps=32)
    expected = initial_z * torch.exp(parameter)
    torch.testing.assert_close(result.final_state.z, expected, rtol=2.0e-8, atol=2.0e-8)
    result.final_state.z.sum().backward()
    assert initial_z.grad is not None and torch.isfinite(initial_z.grad).all()
    assert parameter.grad is not None and torch.isfinite(parameter.grad)
    assert math.isclose(parameter.grad.item(), expected.item(), rel_tol=3.0e-8)


def _three_identical_p1_layout() -> IntrinsicCoverLayout:
    return IntrinsicCoverLayout(
        ContinuousCover(
            nu=3,
            space_group="P1",
            composition=Composition.from_counts({6: 1}),
            orbits=(OrbitType("1a", 6),) * 3,
        ),
        metric_bound=1.0,
        lattice_chart="ball-v1",
    )


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("batched", [False, True])
def test_dequantization_pushes_forward_noise_and_preserves_rng(dtype, batched) -> None:
    layout = _three_identical_p1_layout()
    action = torch.as_tensor(layout.residual_action_matrix((1, 0, 2)), dtype=dtype)
    assert not torch.allclose(action.T @ action, torch.eye(6, dtype=dtype))
    period = torch.as_tensor(layout.period_basis, dtype=dtype)
    eta = torch.tensor([0.17, 0.23, 0.31, 0.41, 0.43, 0.47], dtype=dtype)
    z = torch.zeros(6, dtype=dtype)
    if batched:
        eta = eta.repeat(2, 1)
        z = z.repeat(2, 1)
        action = torch.stack((torch.eye(6, dtype=dtype), action))
    eta.requires_grad_(True)
    plain_rng = torch.Generator().manual_seed(471)
    action_rng = torch.Generator().manual_seed(471)
    plain = sample_dequantized_endpoint(
        z, eta, period, sigma_z=0.05, sigma_eta=0.1, generator=plain_rng
    )
    transformed = sample_dequantized_endpoint(
        z, eta, period, sigma_z=0.05, sigma_eta=0.1,
        generator=action_rng, torus_action=action,
    )
    expected = canonicalize_torus(
        torch.matmul(plain.eta.unsqueeze(-2), action).squeeze(-2), period
    )
    torch.testing.assert_close(transformed.eta, expected)
    torch.testing.assert_close(transformed.z, plain.z, rtol=0, atol=0)
    torch.testing.assert_close(plain_rng.get_state(), action_rng.get_state(), rtol=0, atol=0)
    transformed.eta.sum().backward()
    expected_gradient = action.sum(dim=-1)
    torch.testing.assert_close(eta.grad, expected_gradient)


def test_symmetrized_three_orbit_heat_kernel_has_invariant_fourier_coefficients() -> None:
    layout = _three_identical_p1_layout()
    actions = [layout.residual_action_matrix(p) for p in permutations(range(3))]
    mean = np.array([0.17, 0.23, 0.31, 0.41, 0.43, 0.47])
    sigma = 0.1
    mode = np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0])

    def coefficient(k, *, transform_noise):
        # Characteristic function of (mean + Gaussian noise) @ action, modulo
        # Z^6.  Integer Fourier characters remove the wrapping exactly.
        return sum(
            np.exp(
                2j * np.pi * (mean @ action @ k)
                - 2 * np.pi**2 * sigma**2
                * np.sum((action @ k if transform_noise else k) ** 2)
            )
            for action in actions
        ) / len(actions)

    for action in actions:
        np.testing.assert_allclose(
            coefficient(mode, transform_noise=True),
            coefficient(action @ mode, transform_noise=True),
            atol=1e-15, rtol=1e-13,
        )
    anchor_exchange = layout.residual_action_matrix((1, 0, 2))
    assert abs(
        coefficient(mode, transform_noise=False)
        - coefficient(anchor_exchange @ mode, transform_noise=False)
    ) > 0.01


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("batched", [False, True])
def test_complete_path_action_transforms_tangent_without_rewrapping(dtype, batched) -> None:
    layout = _three_identical_p1_layout()
    action = torch.as_tensor(layout.residual_action_matrix((1, 0, 2)), dtype=dtype)
    period = torch.as_tensor(layout.period_basis, dtype=dtype)
    eta0 = torch.full((6,), 0.1, dtype=dtype)
    delta = torch.tensor([0.4, 0, 0, -0.4, 0, 0], dtype=dtype)
    eta1 = canonicalize_torus(eta0 + delta, period)
    z0 = torch.zeros(6, dtype=dtype)
    z1 = torch.full((6,), 0.2, dtype=dtype)
    if batched:
        eta0, eta1, z0, z1 = (value.repeat(2, 1) for value in (eta0, eta1, z0, z1))
        action = torch.stack((torch.eye(6, dtype=dtype), action))
    eta1.requires_grad_(True)
    for t in (0.0, 0.37, 1.0):
        time = torch.tensor(t, dtype=dtype)
        plain = conditional_flow_path(z0, eta0, z1, eta1, time, period)
        transformed = conditional_flow_path(
            z0, eta0, z1, eta1, time, period, torus_action=action
        )
        expected_state = canonicalize_torus(
            torch.matmul(plain.state.eta.unsqueeze(-2), action).squeeze(-2), period
        )
        expected_velocity = torch.matmul(
            plain.target_velocity.eta.unsqueeze(-2), action
        ).squeeze(-2)
        torch.testing.assert_close(transformed.state.eta, expected_state)
        torch.testing.assert_close(transformed.target_velocity.eta, expected_velocity)
        torch.testing.assert_close(transformed.torus_delta, expected_velocity)
        torch.testing.assert_close(transformed.state.z, plain.state.z, rtol=0, atol=0)
        assert transformed.target_velocity.eta.abs().max() > 0.5
    # Rebuilding a half-open lift between transformed endpoints loses winding.
    changed_endpoints = conditional_flow_path(
        z0, canonicalize_torus(torch.matmul(eta0.unsqueeze(-2), action).squeeze(-2), period),
        z1, canonicalize_torus(torch.matmul(eta1.unsqueeze(-2), action).squeeze(-2), period),
        torch.tensor(0.37, dtype=dtype), period,
    )
    assert not torch.allclose(changed_endpoints.target_velocity.eta, expected_velocity)
    derivative = torch.autograd.functional.jacobian(
        lambda time: conditional_flow_path(
            z0, eta0, z1, eta1, time, period, torus_action=action
        ).state.eta,
        torch.tensor(0.37, dtype=dtype, requires_grad=True),
    )
    torch.testing.assert_close(derivative, expected_velocity)
    transformed.state.eta.sum().backward()
    torch.testing.assert_close(eta1.grad, action.sum(dim=-1))


def test_group_averaged_path_has_invariant_interior_fourier_coefficients() -> None:
    layout = _three_identical_p1_layout()
    actions = [layout.residual_action_matrix(p) for p in permutations(range(3))]
    mean = np.array([0.17, 0.23, 0.31, 0.41, 0.43, 0.47])
    sigma = 0.1
    mode = np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0])

    def endpoint(k):
        return np.exp(2j * np.pi * mean @ k - 2 * np.pi**2 * sigma**2 * np.sum(k**2))

    def coefficient(k, time, *, whole_path):
        # Haar base implies eta_t = eta_1 - (1-t) U, with U independent uniform
        # on the half-open unit cube. Its Fourier factor is prod sinc((1-t)k).
        return sum(
            endpoint(action @ k)
            * np.prod(np.sinc((1 - time) * (action @ k if whole_path else k)))
            for action in actions
        ) / len(actions)

    for time in (0.0, 0.25, 0.5, 0.75, 1.0):
        for action in actions:
            np.testing.assert_allclose(
                coefficient(mode, time, whole_path=True),
                coefficient(action @ mode, time, whole_path=True),
                atol=1e-15, rtol=1e-13,
            )
    anchor_exchange = layout.residual_action_matrix((1, 0, 2))
    assert abs(
        coefficient(mode, 0.5, whole_path=False)
        - coefficient(anchor_exchange @ mode, 0.5, whole_path=False)
    ) > 0.019
