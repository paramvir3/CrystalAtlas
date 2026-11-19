"""Proof obligation F8 against the production CFM loss (review finding M1).

At one current state ``q_t``, the conditional law of the endpoint is a finite
mixture with probabilities ``p_k`` and target velocities ``u_k``.  The expected
production loss ``sum_k p_k L(v; u_k, W_k)`` is quadratic in the predicted
velocity ``v``, so one Newton step from any point is its exact minimizer.

* Current-state weighting (``W_k = W(q_t)`` for every ``k``) must return the
  conditional mean ``E[u | q_t]``, whatever the weight.
* Endpoint-dependent weighting ``W_k = W(q_1^k)`` returns
  ``(sum p_k W_k)^-1 sum p_k W_k u_k`` instead, which is not ordinary CFM.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from crystalatlas.flow import FlowVelocity, conditional_flow_matching_loss  # noqa: E402

_LATTICE_DIM, _COORD_DIM, _ATOMS = 2, 3, 4


def _spd(generator, size):  # type: ignore[no-untyped-def]
    a = torch.randn(size, size, generator=generator, dtype=torch.float64)
    return a @ a.T + 0.5 * torch.eye(size, dtype=torch.float64)


def _branches(seed: int):  # type: ignore[no-untyped-def]
    generator = torch.Generator().manual_seed(seed)
    count = 5
    probabilities = torch.rand(count, generator=generator, dtype=torch.float64)
    probabilities /= probabilities.sum()
    u_z = torch.randn(count, _LATTICE_DIM, generator=generator, dtype=torch.float64)
    u_eta = torch.randn(count, _COORD_DIM, generator=generator, dtype=torch.float64)
    return generator, probabilities, u_z, u_eta


def _minimizer(probabilities, u_z, u_eta, lattice_metrics, coordinate_metrics):  # type: ignore[no-untyped-def]
    count = len(probabilities)

    def expected_loss(flat):  # type: ignore[no-untyped-def]
        v_z = flat[:_LATTICE_DIM].expand(count, _LATTICE_DIM)
        v_eta = flat[_LATTICE_DIM:].expand(count, _COORD_DIM)
        per_sample = conditional_flow_matching_loss(
            FlowVelocity(v_z, v_eta),
            FlowVelocity(u_z, u_eta),
            current_lattice_metric=lattice_metrics,
            current_coordinate_metric=coordinate_metrics,
            atom_count=_ATOMS,
            reduction="none",
        )
        return (probabilities * per_sample).sum()

    start = torch.zeros(_LATTICE_DIM + _COORD_DIM, dtype=torch.float64, requires_grad=True)
    gradient = torch.autograd.grad(expected_loss(start), start)[0]
    hessian = torch.autograd.functional.hessian(expected_loss, start.detach())
    return start.detach() - torch.linalg.solve(hessian, gradient)


def _conditional_mean(probabilities, u_z, u_eta):  # type: ignore[no-untyped-def]
    weights = probabilities[:, None]
    return torch.cat(((weights * u_z).sum(0), (weights * u_eta).sum(0)))


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_current_state_weighting_returns_the_conditional_mean(seed: int) -> None:
    generator, probabilities, u_z, u_eta = _branches(seed)
    count = len(probabilities)
    lattice = _spd(generator, _LATTICE_DIM).expand(count, -1, -1)
    coordinate = _spd(generator, _COORD_DIM).expand(count, -1, -1)
    minimizer = _minimizer(probabilities, u_z, u_eta, lattice, coordinate)
    mean = _conditional_mean(probabilities, u_z, u_eta)
    torch.testing.assert_close(minimizer, mean, rtol=1e-10, atol=1e-12)


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_endpoint_dependent_weighting_returns_the_weighted_counterexample(seed: int) -> None:
    generator, probabilities, u_z, u_eta = _branches(seed)
    count = len(probabilities)
    lattice = torch.stack([_spd(generator, _LATTICE_DIM) for _ in range(count)])
    coordinate = torch.stack([_spd(generator, _COORD_DIM) for _ in range(count)])
    minimizer = _minimizer(probabilities, u_z, u_eta, lattice, coordinate)

    def weighted(metrics, targets, scale):  # type: ignore[no-untyped-def]
        p = probabilities[:, None, None]
        weight = (p * metrics).sum(0) / scale
        moment = (p * (metrics @ targets[:, :, None])).sum(0)[:, 0] / scale
        return torch.linalg.solve(weight, moment)

    expected = torch.cat((weighted(lattice, u_z, 1.0), weighted(coordinate, u_eta, _ATOMS)))
    torch.testing.assert_close(minimizer, expected, rtol=1e-10, atol=1e-12)
    mean = _conditional_mean(probabilities, u_z, u_eta)
    assert float(torch.max(torch.abs(minimizer - mean))) > 1e-3
