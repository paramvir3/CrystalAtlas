"""Affine residual torus actions eta -> eta A + m on the CFM path and the oracle.

The residual normalizer gauge contains translations (e.g. P2_13's x -> x + 1/2),
so the production path action and the exact oracle need a translation part.
* Path: a translation shifts the whole sampled path and leaves the tangent unchanged;
  for a pure translation this equals the path between translated endpoints,
  because the shortest coordinate lift is translation invariant.
* Oracle: with lattice coefficients a = eta P^-1 the action is
  a -> a (P A P^-1) + m P^-1; the pushforward density is p((a - m) M^-1), the
  current transforms by M, and moments gain the phase exp(2 pi i n . m).
  Cross-check: the affine group {(I, 0), (I, 1/2)} equals the independently exact
  construction that duplicates every record at eta + 1/2 with half weight.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from crystalatlas.errors import DomainError  # noqa: E402
from crystalatlas.flow import canonicalize_torus, conditional_flow_path  # noqa: E402
from crystalatlas.flow_oracle import TinyFlowOracle  # noqa: E402

F64 = torch.float64


def _endpoints(seed: int):  # type: ignore[no-untyped-def]
    g = torch.Generator().manual_seed(seed)
    return (
        torch.randn(8, 1, generator=g, dtype=F64),
        torch.rand(8, 2, generator=g, dtype=F64),
        torch.randn(8, 1, generator=g, dtype=F64),
        torch.rand(8, 2, generator=g, dtype=F64),
        torch.rand(8, generator=g, dtype=F64),
    )


def test_translation_shifts_the_path_and_keeps_the_tangent() -> None:
    period = torch.eye(2, dtype=F64)
    z0, e0, z1, e1, t = _endpoints(1)
    shift = torch.tensor([0.5, 0.25], dtype=F64)
    plain = conditional_flow_path(z0, e0, z1, e1, t, period)
    moved = conditional_flow_path(z0, e0, z1, e1, t, period, torus_translation=shift)
    torch.testing.assert_close(
        moved.state.eta, canonicalize_torus(plain.state.eta + shift, period), rtol=0, atol=1e-15
    )
    torch.testing.assert_close(
        moved.target_velocity.eta, plain.target_velocity.eta, rtol=0, atol=0
    )
    translated = conditional_flow_path(
        z0,
        canonicalize_torus(e0 + shift, period),
        z1,
        canonicalize_torus(e1 + shift, period),
        t,
        period,
    )
    torch.testing.assert_close(moved.state.eta, translated.state.eta, rtol=0, atol=1e-14)
    zero = conditional_flow_path(
        z0, e0, z1, e1, t, period, torus_translation=torch.zeros(2, dtype=F64)
    )
    torch.testing.assert_close(zero.state.eta, plain.state.eta, rtol=0, atol=0)


def _toy(**overrides):  # type: ignore[no-untyped-def]
    arguments = dict(
        z_data=torch.tensor([[-0.3], [0.5], [1.2]], dtype=F64),
        eta_data=torch.tensor([[0.12], [0.30], [0.41]], dtype=F64),
        weights=torch.tensor([0.5, 0.3, 0.2], dtype=F64),
        period_basis=torch.eye(1, dtype=F64),
        sigma_z=0.25,
        sigma_eta=0.125,
    )
    arguments.update(overrides)
    return TinyFlowOracle(**arguments)


def test_affine_oracle_equals_the_record_duplication_construction() -> None:
    affine = _toy(
        torus_actions=[torch.eye(1, dtype=F64), torch.eye(1, dtype=F64)],
        torus_translations=[torch.zeros(1, dtype=F64), torch.tensor([0.5], dtype=F64)],
    )
    base = _toy()
    duplicated = _toy(
        z_data=torch.cat([base.z_data, base.z_data]),
        eta_data=torch.remainder(torch.cat([base.eta_data, base.eta_data + 0.5]), 1.0),
        weights=torch.cat([base.weights, base.weights]) / 2.0,
    )
    z = torch.linspace(-1.0, 1.5, 7, dtype=F64)[:, None]
    eta = torch.linspace(0.03, 0.97, 7, dtype=F64)[:, None]
    for time in (0.0, 0.35, 0.8, 1.0):
        t = torch.tensor(time, dtype=F64)
        torch.testing.assert_close(
            affine.log_density(t, z, eta),
            duplicated.log_density(t, z, eta),
            rtol=1e-12,
            atol=1e-12,
        )
        left, right = affine.velocity(t, z, eta), duplicated.velocity(t, z, eta)
        torch.testing.assert_close(left.eta, right.eta, rtol=1e-10, atol=1e-12)
        torch.testing.assert_close(left.z, right.z, rtol=1e-10, atol=1e-12)
        for frequency in ((1,), (2,), (-3,)):
            for power in (0, 1):
                a = affine.moment(time, frequency, z_index=0 if power else None, z_power=power)
                b = duplicated.moment(
                    time, frequency, z_index=0 if power else None, z_power=power
                )
                assert abs(a - b) <= 1e-12


def test_affine_actions_must_form_a_group_with_the_identity() -> None:
    with pytest.raises(DomainError, match="closed"):
        _toy(
            torus_actions=[torch.eye(1, dtype=F64), torch.eye(1, dtype=F64)],
            torus_translations=[
                torch.zeros(1, dtype=F64),
                torch.tensor([1.0 / 3.0], dtype=F64),
            ],
        )
    with pytest.raises(DomainError, match="align"):
        _toy(torus_translations=[torch.zeros(1, dtype=F64), torch.zeros(1, dtype=F64)])


def test_affine_oracle_sampling_uses_the_translations() -> None:
    oracle = _toy(
        torus_actions=[torch.eye(1, dtype=F64), torch.eye(1, dtype=F64)],
        torus_translations=[torch.zeros(1, dtype=F64), torch.tensor([0.5], dtype=F64)],
    )
    generator = torch.Generator().manual_seed(3)
    paths = oracle.sample_paths(torch.full((4000,), 1.0, dtype=F64), generator=generator)
    eta = paths.state.eta[:, 0]
    # At t = 1 the law is the record heat kernels and their +1/2 images, equally weighted.
    near_shift = ((eta - 0.62).abs() < 0.1).double().mean()
    near_record = ((eta - 0.12).abs() < 0.1).double().mean()
    assert abs(float(near_shift - near_record)) < 0.05
