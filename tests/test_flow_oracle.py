"""Theory F8b and the deterministic half of F10 on the production flow (review M1).

The oracle (``crystalatlas.flow_oracle``) is the exact law sampled by the
production geometry objective on one fixed cover: the Haar product base, the
Gaussian--torus-heat dequantized endpoint, a uniformly chosen residual action,
and the half-open coordinate-lift path.  Two toys follow theory F10: a
one-dimensional lattice chart with a one-parameter orbit torus, and with a
two-parameter torus whose period basis is skew and whose action is not a
signed permutation in lattice coordinates.

* The oracle's endpoint laws are the base and an independent image-sum
  wrapped-Gaussian target, and its density has the exact moments at every t.
* The marginal and the record-conditional continuity equations hold
  (autograd).  Conditional on the endpoint itself the law is a uniform box,
  so that equation holds only distributionally and is not tested pointwise.
* Production samples have the oracle's law and E[u - v(q_t) | q_t] = 0.
* Production RK4 at the production default step count reproduces DOP853
  (rtol 1e-9, atol 1e-11) forward and backward within 1e-7 (theory 4.10).
* The deterministic-quadrature endpoint gate of theory 4.10: relative moment
  error and density L1 error both at most 1e-3.  The pushforward density at a
  quadrature node y is p_0(x)|det dx/dy|, where x is the production RK4
  integral of the reversed field from y and the Jacobian is exact forward-mode
  AD through that integration.

None of this trains a network.  The learned half of F10 needs its own
thresholds (open item M1 in docs/PEER_REVIEW_REMEDIATION_2026-10.md).
"""

from __future__ import annotations

import itertools
import math

import pytest

torch = pytest.importorskip("torch")
scipy_integrate = pytest.importorskip("scipy.integrate")

import torch.autograd.forward_ad as forward_ad  # noqa: E402

from crystalatlas.config import NumericsConfig  # noqa: E402
from crystalatlas.errors import DomainError  # noqa: E402
from crystalatlas.flow import FlowState, FlowVelocity, integrate_rk4  # noqa: E402
from crystalatlas.flow_oracle import TinyFlowOracle  # noqa: E402

pytestmark = pytest.mark.ml

F64 = torch.float64
PRODUCTION_STEPS = NumericsConfig().ode_steps
ODE_STATE_TOLERANCE = 1.0e-7  # theory 4.10, analytic-toy forward/backward
QUADRATURE_TOLERANCE = 1.0e-3  # theory 4.10, deterministic-quadrature endpoint
MOMENT_FLOOR = 0.05  # relative errors are reported only for |moment| >= this


def _tensor(values):  # type: ignore[no-untyped-def]
    return torch.tensor(values, dtype=F64)


def one_orbit_toy(**overrides) -> TinyFlowOracle:  # type: ignore[no-untyped-def]
    """d_z = 1 cubic-like chart; one orbit parameter on R/Z with the sign gauge."""

    arguments = dict(
        z_data=_tensor([[-0.3], [0.5], [1.2]]),
        eta_data=_tensor([[0.12], [0.30], [0.71]]),
        weights=_tensor([0.5, 0.3, 0.2]),
        period_basis=_tensor([[1.0]]),
        sigma_z=0.25,
        sigma_eta=0.125,
        torus_actions=[_tensor([[1.0]]), _tensor([[-1.0]])],
    )
    arguments.update(overrides)
    return TinyFlowOracle(**arguments)


def two_parameter_toy(**overrides) -> TinyFlowOracle:  # type: ignore[no-untyped-def]
    """A skew (unreduced) checkerboard basis and the coordinate swap.

    In lattice coordinates the swap is M = [[1, 0], [2, -1]], which is not a
    signed permutation.  A new half-open lift between transformed endpoints
    therefore differs from the production path's transformed lift.
    """

    arguments = dict(
        z_data=_tensor([[-0.2], [0.4], [1.0]]),
        eta_data=_tensor([[0.10, 0.35], [0.40, 0.20], [0.25, 0.80]]),
        weights=_tensor([0.45, 0.35, 0.20]),
        period_basis=_tensor([[0.5, 0.5], [0.0, 1.0]]),
        sigma_z=0.25,
        sigma_eta=0.125,
        torus_actions=[torch.eye(2, dtype=F64), _tensor([[0.0, 1.0], [1.0, 0.0]])],
    )
    arguments.update(overrides)
    return TinyFlowOracle(**arguments)


TOYS = {"one-orbit": one_orbit_toy, "two-parameter": two_parameter_toy}


def _random_states(oracle: TinyFlowOracle, count: int, seed: int):  # type: ignore[no-untyped-def]
    generator = torch.Generator().manual_seed(seed)
    z = 1.5 * torch.randn(count, oracle.lattice_dimension, generator=generator, dtype=F64)
    unit = torch.rand(count, oracle.intrinsic_dimension, generator=generator, dtype=F64)
    return z, unit @ oracle.period_basis


def _lattice_coordinates(oracle: TinyFlowOracle, eta):  # type: ignore[no-untyped-def]
    return eta @ torch.linalg.inv(oracle.period_basis)


def _torus_distance(oracle: TinyFlowOracle, left, right):  # type: ignore[no-untyped-def]
    difference = _lattice_coordinates(oracle, left - right)
    difference = difference - torch.floor(difference + 0.5)
    return (difference @ oracle.period_basis).norm(dim=-1)


def test_oracle_refuses_invalid_or_ill_conditioned_toys() -> None:
    with pytest.raises(DomainError, match="sum to one"):
        one_orbit_toy(weights=_tensor([0.5, 0.3, 0.3]))
    with pytest.raises(DomainError, match="unimodular"):
        one_orbit_toy(torus_actions=[_tensor([[1.0]]), _tensor([[2.0]])])
    with pytest.raises(DomainError, match="integer"):
        one_orbit_toy(torus_actions=[_tensor([[1.0]]), _tensor([[0.5]])])
    with pytest.raises(DomainError, match="identity"):
        one_orbit_toy(torus_actions=[_tensor([[-1.0]])])
    # A lattice shear is an automorphism of infinite order: {I, A} is not a group.
    basis = two_parameter_toy().period_basis
    shear = torch.linalg.inv(basis) @ _tensor([[1.0, 0.0], [1.0, 1.0]]) @ basis
    with pytest.raises(DomainError, match="closed"):
        two_parameter_toy(torus_actions=[torch.eye(2, dtype=F64), shear])
    with pytest.raises(DomainError, match="ill-conditioned: certified relative error"):
        one_orbit_toy(sigma_eta=0.06)
    with pytest.raises(DomainError, match="ill-conditioned: its torus kernel"):
        one_orbit_toy(sigma_eta=0.02)


def _image_sum_target(oracle: TinyFlowOracle, z, eta, images: int = 4):  # type: ignore[no-untyped-def]
    """Independent spatial evaluation: Gaussian x pushed, wrapped Gaussian."""

    shifts = (
        _tensor(
            list(
                itertools.product(range(-images, images + 1), repeat=oracle.intrinsic_dimension)
            )
        )
        @ oracle.period_basis
    )
    total = torch.zeros(len(z), dtype=F64)
    for action in oracle.cartesian_actions:
        covariance = oracle.sigma_eta**2 * action.T @ action
        precision = torch.linalg.inv(covariance)
        log_normalizer = 0.5 * torch.logdet(2.0 * math.pi * covariance)
        for record, weight in enumerate(oracle.weights):
            offset = eta[:, None, :] - oracle.eta_data[record] @ action - shifts[None]
            torus = torch.exp(
                -0.5 * torch.einsum("bki,ij,bkj->bk", offset, precision, offset)
                - log_normalizer
            ).sum(1)
            lattice = torch.exp(
                -0.5 * ((z - oracle.z_data[record]) ** 2).sum(-1) / oracle.sigma_z**2
            ) / (2.0 * math.pi * oracle.sigma_z**2) ** (0.5 * oracle.lattice_dimension)
            total = total + weight * lattice * torus / len(oracle.cartesian_actions)
    return total


@pytest.mark.parametrize("name", TOYS)
def test_endpoint_laws_are_the_base_and_an_independent_image_sum_target(name: str) -> None:
    oracle = TOYS[name]()
    assert oracle.conditioning.relative_error_bound <= 1.0e-9
    z, eta = _random_states(oracle, 512, seed=1)
    base = oracle.log_density(_tensor(0.0), z, eta)
    assert torch.allclose(base, oracle.base_log_density(z, eta), rtol=0.0, atol=1.0e-13)
    target = oracle.log_density(_tensor(1.0), z, eta).exp()
    reference = _image_sum_target(oracle, z, eta)
    certified = oracle.conditioning.relative_error_bound
    assert float(((target - reference).abs() / reference).max()) <= certified + 1.0e-13


def _grid_quadrature(
    oracle: TinyFlowOracle, *, z_step: float, z_margin: float, torus_points: int
) -> tuple[torch.Tensor, torch.Tensor, float]:
    """Trapezoid nodes and Cartesian weights over a z interval times the torus."""

    low = float(oracle.z_data.min()) - z_margin
    high = float(oracle.z_data.max()) + z_margin
    z_values = torch.arange(math.floor(low / z_step), math.ceil(high / z_step) + 1, dtype=F64)
    unit = (torch.arange(torus_points, dtype=F64) + 0.5) / torus_points
    cells = torch.stack(
        torch.meshgrid(*([unit] * oracle.intrinsic_dimension), indexing="ij"), dim=-1
    ).reshape(-1, oracle.intrinsic_dimension)
    z = (z_step * z_values).repeat_interleave(len(cells))[:, None]
    eta = cells.repeat(len(z_values), 1) @ oracle.period_basis
    weight = z_step * abs(float(torch.linalg.det(oracle.period_basis))) / len(cells)
    return z, eta, weight


def _all_moments(oracle: TinyFlowOracle, time: float):  # type: ignore[no-untyped-def]
    """``E[z^p exp(2 pi i n.a_t)]`` for |n_j| <= 2 and p <= 2, except the trivial one."""

    moments = []
    for frequency in itertools.product(range(-2, 3), repeat=oracle.intrinsic_dimension):
        for power in (0, 1, 2):
            if not any(frequency) and power == 0:
                continue
            exact = oracle.moment(time, frequency, z_index=0 if power else None, z_power=power)
            moments.append((frequency, power, exact))
    return moments


def _test_moments(oracle: TinyFlowOracle, time: float):  # type: ignore[no-untyped-def]
    """The moments with exact |m| >= MOMENT_FLOOR, for which relative error is meaningful."""

    return [moment for moment in _all_moments(oracle, time) if abs(moment[2]) >= MOMENT_FLOOR]


def _estimate(oracle, z, eta, weights, frequency, power):  # type: ignore[no-untyped-def]
    phase = 2.0 * math.pi * (_lattice_coordinates(oracle, eta) @ _tensor(frequency))
    values = z[:, 0] ** power * torch.exp(1j * phase)
    return complex((weights * values).sum())


@pytest.mark.parametrize("name", TOYS)
def test_evaluated_density_has_the_exact_moments_at_every_time(name: str) -> None:
    oracle = TOYS[name]()
    # Every q_t has z-standard deviation between 0.24 and 1, so the step-1/8
    # z-trapezoid is exact to e^-70.  The evaluated series has given-basis
    # frequencies |n_j| <= K, and the test moments have |n_j| <= 2, so a torus
    # grid of K + 3 points integrates every product exactly.
    largest = int(oracle._frequencies.abs().max())
    z, eta, weight = _grid_quadrature(
        oracle, z_step=0.125, z_margin=9.0, torus_points=largest + 3
    )
    certified = oracle.conditioning.relative_error_bound
    for time in (0.0, 0.3, 0.7, 1.0):
        density = oracle.log_density(_tensor(time), z, eta).exp()
        assert abs(float(weight * density.sum()) - 1.0) <= certified + 1.0e-13
        for frequency, power, exact in _all_moments(oracle, time):
            estimate = _estimate(oracle, z, eta, weight * density, frequency, power)
            scale = oracle.moment(time, (0,) * oracle.intrinsic_dimension, z_index=0, z_power=2)
            bound = certified * (1.0 + abs(scale)) + 1.0e-13
            assert abs(estimate - exact) <= bound, (time, frequency, power)


def _continuity_residual(oracle: TinyFlowOracle, time: float, z, eta):  # type: ignore[no-untyped-def]
    """``(d_t p + div J, |d_t p| + sum |d_i J_i|)`` per point by forward-mode AD."""

    def density_and_current(t, zz, ee):  # type: ignore[no-untyped-def]
        density = oracle.log_density(t, zz, ee).exp()
        velocity = oracle.velocity(t, zz, ee)
        return density, torch.cat(
            (density[:, None] * velocity.z, density[:, None] * velocity.eta), 1
        )

    columns = oracle.lattice_dimension + oracle.intrinsic_dimension
    with forward_ad.dual_level():
        dual_time = forward_ad.make_dual(_tensor(time), _tensor(1.0))
        density, _ = density_and_current(dual_time, z, eta)
        time_derivative = forward_ad.unpack_dual(density).tangent
        divergence = torch.zeros_like(time_derivative)
        scale = time_derivative.abs()
        for column in range(columns):
            tangent = torch.zeros(len(z), columns, dtype=F64)
            tangent[:, column] = 1.0
            dual_z = forward_ad.make_dual(
                z, tangent[:, : oracle.lattice_dimension].contiguous()
            )
            dual_eta = forward_ad.make_dual(
                eta, tangent[:, oracle.lattice_dimension :].contiguous()
            )
            _, current = density_and_current(_tensor(time), dual_z, dual_eta)
            partial = forward_ad.unpack_dual(current).tangent[:, column]
            divergence = divergence + partial
            scale = scale + partial.abs()
    return time_derivative + divergence, scale


@pytest.mark.parametrize("name", TOYS)
def test_marginal_and_record_conditional_continuity_equations_hold(name: str) -> None:
    oracle = TOYS[name]()
    single_record = TOYS[name](
        z_data=oracle.z_data[:1], eta_data=oracle.eta_data[:1], weights=_tensor([1.0])
    )
    z, eta = _random_states(oracle, 256, seed=2)
    for model in (oracle, single_record):
        for time in (0.02, 0.25, 0.5, 0.75, 0.98):
            residual, scale = _continuity_residual(model, time, z, eta)
            assert float((residual.abs() / scale).max()) <= 1.0e-9, time


@pytest.mark.parametrize("name", TOYS)
def test_production_paths_sample_the_oracle_law_and_its_conditional_mean(name: str) -> None:
    oracle = TOYS[name]()
    samples = 1 << 17
    generator = torch.Generator().manual_seed(3)
    for time in (0.3, 0.8, 1.0):
        path = oracle.sample_paths(torch.full((samples,), time, dtype=F64), generator=generator)
        z, eta = path.state.z, path.state.eta
        uniform = torch.full((samples,), 1.0 / samples, dtype=F64)
        for frequency, power, exact in _test_moments(oracle, time):
            phase = 2.0 * math.pi * (_lattice_coordinates(oracle, eta) @ _tensor(frequency))
            values = z[:, 0] ** power * torch.exp(1j * phase)
            error = abs(complex(values.mean()) - exact)
            variance = float(values.real.var() + values.imag.var())
            standard_error = math.sqrt(variance / samples)
            assert error <= 5.0 * standard_error, (time, frequency, power)
        velocity = oracle.velocity(_tensor(time), z, eta)
        residual = torch.cat(
            (path.target_velocity.z - velocity.z, path.target_velocity.eta - velocity.eta), 1
        )
        coordinates = _lattice_coordinates(oracle, eta)
        tests = [torch.ones(samples, dtype=F64), z[:, 0]]
        for j in range(oracle.intrinsic_dimension):
            tests += [torch.cos(2.0 * math.pi * coordinates[:, j])]
            tests += [torch.sin(2.0 * math.pi * coordinates[:, j])]
        if oracle.intrinsic_dimension == 2:
            tests += [torch.cos(2.0 * math.pi * (coordinates[:, 0] - coordinates[:, 1]))]
        for test in tests:
            for column in range(residual.shape[1]):
                values = residual[:, column] * test
                mean = float((uniform * values).sum())
                standard_error = float(values.std()) / math.sqrt(samples)
                assert abs(mean) <= 5.0 * standard_error, (time, column)


def _reversed(oracle: TinyFlowOracle):  # type: ignore[no-untyped-def]
    def field(time, z, eta):  # type: ignore[no-untyped-def]
        velocity = oracle.velocity(1.0 - time, z, eta)
        return FlowVelocity(z=-velocity.z, eta=-velocity.eta)

    return field


def _reference_solution(oracle: TinyFlowOracle, z, eta, span):  # type: ignore[no-untyped-def]
    """DOP853 on the unwrapped state with the 4.10 reference tolerances."""

    count = len(z)

    def rhs(time, flat):  # type: ignore[no-untyped-def]
        state = torch.as_tensor(flat, dtype=F64).reshape(count, -1)
        velocity = oracle.velocity(
            _tensor(time),
            state[:, : oracle.lattice_dimension].contiguous(),
            state[:, oracle.lattice_dimension :].contiguous(),
        )
        return torch.cat((velocity.z, velocity.eta), 1).reshape(-1).numpy()

    solution = scipy_integrate.solve_ivp(
        rhs,
        span,
        torch.cat((z, eta), 1).reshape(-1).numpy(),
        method="DOP853",
        rtol=1.0e-9,
        atol=1.0e-11,
    )
    assert solution.status == 0
    final = torch.as_tensor(solution.y[:, -1], dtype=F64).reshape(count, -1)
    return final[:, : oracle.lattice_dimension], final[:, oracle.lattice_dimension :]


@pytest.mark.parametrize("name", TOYS)
def test_production_rk4_forward_and_backward_match_the_reference_solver(name: str) -> None:
    oracle = TOYS[name]()
    z0, eta0 = _random_states(oracle, 6, seed=4)
    z1, eta1 = _reference_solution(oracle, z0, eta0, (0.0, 1.0))
    forward = integrate_rk4(
        oracle.velocity, FlowState(z0, eta0), oracle.period_basis, num_steps=PRODUCTION_STEPS
    ).final_state
    assert float((forward.z - z1).abs().max()) <= ODE_STATE_TOLERANCE
    assert float(_torus_distance(oracle, forward.eta, eta1).max()) <= ODE_STATE_TOLERANCE
    zb, etab = _reference_solution(oracle, z1, eta1, (1.0, 0.0))
    backward = integrate_rk4(
        _reversed(oracle), FlowState(z1, eta1), oracle.period_basis, num_steps=PRODUCTION_STEPS
    ).final_state
    assert float((backward.z - zb).abs().max()) <= ODE_STATE_TOLERANCE
    assert float(_torus_distance(oracle, backward.eta, etab).max()) <= ODE_STATE_TOLERANCE
    # The reference itself returns to the start (a check on the reference).
    assert float((zb - z0).abs().max()) <= 1.0e-8
    assert float(_torus_distance(oracle, etab, eta0).max()) <= 1.0e-8


def _pushforward_density(oracle: TinyFlowOracle, z, eta, steps: int):  # type: ignore[no-untyped-def]
    """``p_0(x) |det dx/dy|`` with x the production RK4 back-integral of y."""

    dimension = oracle.lattice_dimension + oracle.intrinsic_dimension
    columns = []
    with forward_ad.dual_level():
        for column in range(dimension):
            tangent = torch.zeros(len(z), dimension, dtype=F64)
            tangent[:, column] = 1.0
            start = FlowState(
                forward_ad.make_dual(z, tangent[:, : oracle.lattice_dimension].contiguous()),
                forward_ad.make_dual(eta, tangent[:, oracle.lattice_dimension :].contiguous()),
            )
            end = integrate_rk4(
                _reversed(oracle), start, oracle.period_basis, num_steps=steps
            ).final_state
            base_z, tangent_z = forward_ad.unpack_dual(end.z)
            base_eta, tangent_eta = forward_ad.unpack_dual(end.eta)
            columns.append(torch.cat((tangent_z, tangent_eta), 1))
    determinant = torch.linalg.det(torch.stack(columns, dim=-1)).abs()
    return oracle.base_log_density(base_z, base_eta).exp() * determinant


def _endpoint_gate(oracle: TinyFlowOracle, *, z_step: float, torus_points: int):  # type: ignore[no-untyped-def]
    # A z-margin of 6 sigma_z leaves 2e-9 of the target mass outside, and step
    # sigma_z keeps the z-trapezoid error near 2 exp(-2 pi^2) = 5e-9.  The torus
    # grid is chosen per toy (below) so that every aliased target mode is below
    # 2e-11 of the peak.
    z, eta, weight = _grid_quadrature(
        oracle, z_step=z_step, z_margin=6.0 * oracle.sigma_z, torus_points=torus_points
    )
    flow = _pushforward_density(oracle, z, eta, PRODUCTION_STEPS)
    target = oracle.log_density(_tensor(1.0), z, eta).exp()
    l1 = float(weight * (flow - target).abs().sum())
    target_mass = float(weight * target.sum())
    errors = []
    for frequency, power, exact in _test_moments(oracle, 1.0):
        estimate = _estimate(oracle, z, eta, weight * flow, frequency, power)
        errors.append(abs(estimate - exact) / abs(exact))
    return l1, target_mass, errors


# One orbit: on 16 points the nearest alias of a test moment is mode 14, with
# amplitude exp(-2 pi^2 sigma^2 14^2) = 5.6e-27.  Two parameters: on 10 points
# every aliased mode n = 10k - n_test has n^T G^-1 n >= 80 (computed over
# |k_j| <= 3; larger k exceed 1000 because lambda_min(G^-1) = 0.76), giving
# 1.9e-11.
@pytest.mark.parametrize(
    ("name", "z_step", "torus_points"),
    [
        ("one-orbit", 0.25, 16),
        ("two-parameter", 0.25, 10),
    ],
)
def test_deterministic_quadrature_endpoint_moments_and_density_l1(
    name: str, z_step: float, torus_points: int
) -> None:
    oracle = TOYS[name]()
    l1, target_mass, errors = _endpoint_gate(oracle, z_step=z_step, torus_points=torus_points)
    assert abs(target_mass - 1.0) <= 1.0e-7  # the quadrature itself is accurate
    assert len(errors) >= 4
    assert max(errors) <= QUADRATURE_TOLERANCE
    assert l1 <= QUADRATURE_TOLERANCE


def test_endpoint_gate_detects_an_action_current_that_is_not_pushed_forward(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Mutation: dropping ``J -> J M_g`` breaks transport by far more than 1e-3."""

    oracle = one_orbit_toy()
    monkeypatch.setattr(
        TinyFlowOracle, "_push_current", staticmethod(lambda current, _: current)
    )
    l1, _, errors = _endpoint_gate(oracle, z_step=0.25, torus_points=16)
    assert l1 > 100.0 * QUADRATURE_TOLERANCE
    assert max(errors) > 10.0 * QUADRATURE_TOLERANCE


def test_dequantization_kernel_has_the_full_gaussian_support() -> None:
    """F1/F8b: the production endpoint kernel is the untruncated Gaussian law.

    With period 20 and sigma 0.1, wrapping moves no sample, so the
    standardized noise must be exactly N(0, 1) in every coordinate.  Tail masses
    beyond 1, 2, 2.5, 3 and 3.5 sigma must match 2(1 - Phi(t)) within 5 SE.  A
    kernel truncated anywhere inside 3.5 sigma fails this test (theory 4.9,
    mutation 5).
    """

    from crystalatlas.flow import sample_dequantized_endpoint

    samples = 1 << 18
    sigma = 0.1
    period = 20.0 * torch.eye(2, dtype=F64)
    centre_z = torch.full((samples, 1), 0.3, dtype=F64)
    centre_eta = torch.full((samples, 2), 10.0, dtype=F64)  # mid-cell: no wrap
    endpoint = sample_dequantized_endpoint(
        centre_z,
        centre_eta,
        period,
        sigma_z=sigma,
        sigma_eta=sigma,
        generator=torch.Generator().manual_seed(41),
    )
    noise = torch.cat(
        ((endpoint.z - centre_z) / sigma, (endpoint.eta - centre_eta) / sigma), dim=1
    )
    assert float(noise.abs().max()) < 100.0  # no sample wrapped
    for threshold in (1.0, 2.0, 2.5, 3.0, 3.5):
        exact = math.erfc(threshold / math.sqrt(2.0))
        error = math.sqrt(exact * (1.0 - exact) / samples)
        for column in range(noise.shape[1]):
            observed = float((noise[:, column].abs() > threshold).to(F64).mean())
            assert abs(observed - exact) <= 5.0 * error, (threshold, column)
