"""CFM-consistent endpoint estimate (review finding M4).

For straight conditional paths ``x_t = (1 - t) x_0 + t x_1`` the target velocity
is ``u = x_1 - x_0``, so ``x_1 = x_t + (1 - t) u`` and the posterior-mean endpoint
is ``E[x_1 | x_t] = x_t + (1 - t) v(t, x_t)`` for the exact field ``v = E[u | x_t]``.
Integrating to ``t_stop < 1`` and applying this estimate avoids the stiff end of
the trajectory; ``t_stop = 1`` is plain RK4.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from crystalatlas.config import CrystalAtlasConfig, NumericsConfig  # noqa: E402
from crystalatlas.errors import DomainError  # noqa: E402
from crystalatlas.flow import (  # noqa: E402
    FlowState,
    FlowVelocity,
    integrate_rk4,
    integrate_with_endpoint_estimate,
)

_PERIOD = torch.eye(2, dtype=torch.float64)


def test_terminal_time_one_is_plain_rk4() -> None:
    def field(t, z, eta):  # type: ignore[no-untyped-def]
        return FlowVelocity(torch.sin(z) * t, torch.cos(eta) + 0.1)

    start = FlowState(
        torch.tensor([0.3, -0.2], dtype=torch.float64),
        torch.tensor([0.1, 0.7], dtype=torch.float64),
    )
    plain = integrate_rk4(field, start, _PERIOD, num_steps=8).final_state
    estimate = integrate_with_endpoint_estimate(
        field, start, _PERIOD, num_steps=8, terminal_time=1.0
    )
    torch.testing.assert_close(estimate.z, plain.z, rtol=0.0, atol=0.0)
    torch.testing.assert_close(estimate.eta, plain.eta, rtol=0.0, atol=0.0)


@pytest.mark.parametrize("terminal_time", [0.5, 0.9, 0.99])
def test_estimate_recovers_the_endpoint_of_the_exact_single_target_field(
    terminal_time: float,
) -> None:
    target_z = torch.tensor([1.5, -0.25], dtype=torch.float64)
    target_eta = torch.tensor([0.8, 0.35], dtype=torch.float64)

    def field(t, z, eta):  # type: ignore[no-untyped-def]
        # Straight paths towards one endpoint: v = (x_1 - x) / (1 - t).
        delta = target_eta - eta
        delta = delta - torch.round(delta)
        return FlowVelocity((target_z - z) / (1.0 - t), delta / (1.0 - t))

    start = FlowState(
        torch.tensor([0.0, 0.4], dtype=torch.float64),
        torch.tensor([0.1, 0.9], dtype=torch.float64),
    )
    estimate = integrate_with_endpoint_estimate(
        field, start, _PERIOD, num_steps=16, terminal_time=terminal_time
    )
    torch.testing.assert_close(estimate.z, target_z, rtol=1e-12, atol=1e-12)
    torch.testing.assert_close(estimate.eta, target_eta, rtol=1e-12, atol=1e-12)


def test_terminal_time_is_validated_and_fingerprinted() -> None:
    for invalid in (0.0, -0.5, 1.5, float("nan")):
        with pytest.raises(DomainError):
            NumericsConfig(ode_terminal_time=invalid)
    legacy = CrystalAtlasConfig()
    assert "ode_terminal_time" not in legacy.to_dict()["numerics"]
    changed = CrystalAtlasConfig(numerics=NumericsConfig(ode_terminal_time=0.95))
    assert changed.to_dict()["numerics"]["ode_terminal_time"] == 0.95
    assert changed.fingerprint != legacy.fingerprint
