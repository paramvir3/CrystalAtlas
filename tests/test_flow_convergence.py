"""Analytic FP64 RK4-order regressions; not learned-field accuracy certificates."""

from __future__ import annotations

import math

import pytest

torch = pytest.importorskip("torch")

from crystalatlas.flow import FlowState, FlowVelocity, integrate_rk4  # noqa: E402

pytestmark = pytest.mark.ml


def _fixture(*, batched, skew):
    dtype = torch.float64
    basis = torch.tensor([[1.25, 0.0], [0.375, 0.875]], dtype=dtype) if skew else torch.eye(
        2, dtype=dtype
    )
    z0 = torch.tensor([0.7, -0.4], dtype=dtype)
    x0 = torch.tensor([0.82, 0.12], dtype=dtype)
    if batched:
        z0 = torch.stack((z0, torch.tensor([-0.3, 0.9], dtype=dtype)))
        x0 = torch.stack((x0, torch.tensor([0.31, 0.91], dtype=dtype)))
    stretch = torch.tensor([1.4, 0.7], dtype=dtype)
    omega = torch.tensor([0.7, -0.6], dtype=dtype)
    rate = torch.tensor([0.25, -0.15], dtype=dtype)
    initial_phase = torch.atan2(torch.sin(torch.pi * x0) / stretch,
                                torch.cos(torch.pi * x0)) / torch.pi
    matrix = torch.tensor([[0.3, -1.1], [1.1, 0.3]], dtype=dtype)

    def exact(time):
        # The lift u(t)=u0+omega*expm1(rate*t)/rate is mapped to the circle
        # through tan(pi*x)=stretch*tan(pi*u). atan2 fixes the quadrants;
        # modulo one chooses the canonical representative, not a new path.
        phase = initial_phase + omega * torch.expm1(rate * time) / rate
        x = torch.atan2(stretch * torch.sin(torch.pi * phase),
                        torch.cos(torch.pi * phase)) / torch.pi
        x = x - torch.floor(x)
        angle = 1.1 * time
        cosine, sine = torch.cos(angle), torch.sin(angle)
        z = torch.exp(0.3 * time) * torch.stack(
            (z0[..., 0] * cosine - z0[..., 1] * sine,
             z0[..., 0] * sine + z0[..., 1] * cosine), dim=-1,
        )
        return FlowState(z, x @ basis)

    def field(time, z, eta):
        x = torch.linalg.solve(basis.T, eta.unsqueeze(-1)).squeeze(-1)
        # dx/dt=omega*exp(rate*t)*(s*cos(pi*x)^2+sin(pi*x)^2/s).
        # This is smooth and one-periodic for every positive stretch s;
        # it is not a constant-velocity field or linear coordinate field.
        velocity_x = omega * torch.exp(rate * time) * (
            stretch * torch.cos(torch.pi * x).square()
            + torch.sin(torch.pi * x).square() / stretch
        )
        return FlowVelocity(z @ matrix.T, velocity_x @ basis)

    return basis, FlowState(z0, x0 @ basis), exact, field


@pytest.mark.parametrize("batched", [False, True])
@pytest.mark.parametrize("skew", [False, True])
def test_rk4_fourth_order_for_lattice_and_periodic_state_with_wrap_crossings(batched, skew):
    basis, initial, exact, field = _fixture(batched=batched, skew=skew)
    expected = exact(torch.tensor(1.0, dtype=torch.float64))
    errors_z, errors_eta = [], []
    for steps in (16, 32, 64):
        field_times = []

        def counted(time, z, eta, field_times=field_times):
            field_times.append(float(time))
            return field(time, z, eta)

        result = integrate_rk4(
            counted, initial, basis, num_steps=steps, return_trajectory=True
        )
        expected_times = [
            time for index in range(steps)
            for time in (index / steps, (index + 0.5) / steps,
                         (index + 0.5) / steps, (index + 1) / steps)
        ]
        assert field_times == expected_times
        assert len(field_times) == 4 * steps
        assert result.times.shape == (steps + 1,)
        assert result.z_trajectory.dtype == result.eta_trajectory.dtype == torch.float64
        coefficient_path = torch.linalg.solve(
            basis.T, result.eta_trajectory.unsqueeze(-1)
        ).squeeze(-1)
        assert bool(((coefficient_path >= -1e-14) & (coefficient_path < 1 + 1e-14)).all())
        # First trajectory wraps both its positive-velocity first coordinate
        # and negative-velocity second coordinate during the interval.
        first_path = coefficient_path[:, 0] if batched else coefficient_path
        differences = torch.diff(first_path, dim=0)
        assert differences[:, 0].min() < -0.8
        assert differences[:, 1].max() > 0.8
        delta_eta = result.final_state.eta - expected.eta
        coefficients = torch.linalg.solve(basis.T, delta_eta.unsqueeze(-1)).squeeze(-1)
        periodic_delta = (coefficients - torch.round(coefficients)) @ basis
        errors_z.append(float(torch.linalg.vector_norm(result.final_state.z - expected.z)))
        errors_eta.append(float(torch.linalg.vector_norm(periodic_delta)))
    for errors in (errors_z, errors_eta):
        # The asymptotic ratio is16. This predeclared finite-step band excludes
        # ordinary first-, second-, and third-order behavior without treating
        # a finite-resolution test as a formal convergence proof.
        for coarse, fine in zip(errors, errors[1:], strict=False):
            assert fine > 100 * torch.finfo(torch.float64).eps
            assert 10.0 <= coarse / fine <= 24.0, errors
        assert errors[-1] < 1.0e-6, errors


@pytest.mark.parametrize("batched", [False, True])
def test_analytic_solution_derivative_and_periodic_field_identity(batched):
    basis, initial, exact, field = _fixture(batched=batched, skew=True)
    start = exact(torch.tensor(0.0, dtype=torch.float64))
    torch.testing.assert_close(start.z, initial.z, rtol=2e-14, atol=2e-14)
    torch.testing.assert_close(start.eta, initial.eta, rtol=2e-14, atol=2e-14)
    shifted = torch.tensor([2.0, -3.0], dtype=torch.float64) @ basis
    for value in (0.17, 0.43, 0.91):
        time = torch.tensor(value, dtype=torch.float64, requires_grad=True)
        state = exact(time)
        velocity = field(time, state.z, state.eta)
        lattice_derivative = torch.autograd.functional.jacobian(lambda t: exact(t).z, time)
        torus_derivative = torch.autograd.functional.jacobian(lambda t: exact(t).eta, time)
        torch.testing.assert_close(lattice_derivative, velocity.z, rtol=2e-13, atol=2e-13)
        torch.testing.assert_close(torus_derivative, velocity.eta, rtol=2e-13, atol=2e-13)
        shifted_velocity = field(time, state.z, state.eta + shifted)
        torch.testing.assert_close(shifted_velocity.eta, velocity.eta, rtol=2e-13, atol=2e-13)
    # The time dependence really changes the flow's speed, independently of
    # its state dependence; a constant-vector implementation cannot pass.
    velocity0 = field(torch.tensor(0.0), initial.z, initial.eta)
    velocity1 = field(torch.tensor(1.0), initial.z, initial.eta)
    assert not torch.allclose(velocity0.eta, velocity1.eta)
    assert math.isfinite(float(torch.linalg.vector_norm(velocity1.eta)))
