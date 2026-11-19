"""Pair carriers that vanish linearly at a coincidence (research 2026-11-12).

The legacy carrier ``m / sqrt(|m|^2 + epsilon)`` is a unit vector down to a
separation of ``sqrt(epsilon) / 2 pi`` reference lengths, so the field is close
to a step function of the separation of two atoms at their coincidence.
``pair_carrier_smoothing_length = s`` multiplies every pair carrier by
``sqrt((|m_1|^2 + epsilon) / (|m_1|^2 + (2 pi s)^2))``; ``None`` is the legacy
carrier.  See ``artifacts/research_2026-10-30/R1_LOG.md``, "Orbit collapse".
"""

from __future__ import annotations

import math
from dataclasses import replace

import pytest

torch = pytest.importorskip("torch")

from crystalatlas.config import CrystalAtlasConfig, ModelConfig  # noqa: E402
from crystalatlas.errors import DomainError  # noqa: E402
from crystalatlas.nn.invariant_transformer import (  # noqa: E402
    InvariantPeriodicTransformer,
    InvariantPeriodicTransformerConfig,
)

pytestmark = pytest.mark.ml

F64 = torch.float64
SMOOTHING = 0.2
HARMONICS = 2
SCALES = (0.04, 0.16, 0.64)


def _transformer(smoothing: float | None, **overrides) -> InvariantPeriodicTransformer:  # type: ignore[no-untyped-def]
    torch.manual_seed(5)
    arguments = dict(
        input_scalar_dim=5,
        model_dim=12,
        pair_dim=6,
        num_heads=2,
        num_layers=2,
        output_vector_dim=3,
        direct_harmonics=HARMONICS,
        heat_scales=SCALES,
        pair_carrier_smoothing_length=smoothing,
    )
    arguments.update(overrides)
    config = InvariantPeriodicTransformerConfig(**arguments)
    return InvariantPeriodicTransformer(config).double()


LATTICE = torch.tensor([[4.1, 0.0, 0.0], [0.7, 3.8, 0.0], [0.3, -0.5, 5.2]], dtype=F64)


def _pair(offset: float) -> torch.Tensor:
    """Two atoms ``offset`` angstrom apart along a fixed skew direction, plus a third."""

    direction = torch.tensor([0.48, -0.6, 0.64], dtype=F64)  # unit length
    first = torch.tensor([0.21, 0.37, 0.58], dtype=F64)
    second = first + (offset * direction) @ torch.linalg.inv(LATTICE)
    return torch.stack((first, second, torch.tensor([0.77, 0.11, 0.42], dtype=F64)))


def _bases(transformer, fractional):  # type: ignore[no-untyped-def]
    return transformer._periodic_geometry(fractional[None], LATTICE[None]).covariant_bases[0]


def test_none_is_the_legacy_carrier_and_the_legacy_configuration() -> None:
    fractional = _pair(0.9)
    legacy = InvariantPeriodicTransformer(
        InvariantPeriodicTransformerConfig(
            input_scalar_dim=5, model_dim=12, pair_dim=6, num_heads=2, num_layers=2,
            output_vector_dim=3,
        )
    ).double()
    assert legacy.config.pair_carrier_smoothing_length is None
    assert torch.equal(_bases(legacy, fractional), _bases(_transformer(None), fractional))
    assert set(_transformer(None).state_dict()) == set(_transformer(SMOOTHING).state_dict())
    base = CrystalAtlasConfig()
    assert base.model.geometry_pair_carrier_smoothing_length is None
    assert "geometry_pair_carrier_smoothing_length" not in base.to_dict()["model"]
    smoothed = replace(
        base, model=replace(base.model, geometry_pair_carrier_smoothing_length=SMOOTHING)
    )
    assert smoothed.to_dict()["model"]["geometry_pair_carrier_smoothing_length"] == SMOOTHING
    assert smoothed.fingerprint != base.fingerprint


@pytest.mark.parametrize("offset", [1.0e-4, 3.0e-3, 0.05, 0.2, 0.9, 1.7])
def test_every_pair_carrier_is_the_legacy_carrier_times_the_declared_switch(offset) -> None:  # type: ignore[no-untyped-def]
    fractional = _pair(offset)
    legacy = _bases(_transformer(None), fractional)
    smoothed = _bases(_transformer(SMOOTHING), fractional)
    epsilon = _transformer(None).config.normalization_epsilon
    delta = fractional[None, :, :] - fractional[:, None, :]
    first = torch.sin(2.0 * math.pi * delta) @ LATTICE  # m_1[i, j]
    squared = first.square().sum(dim=-1)
    switch = torch.sqrt((squared + epsilon) / (squared + (2.0 * math.pi * SMOOTHING) ** 2))
    primary = HARMONICS + len(SCALES)
    cross = HARMONICS * len(SCALES)
    assert legacy.shape[-2] == primary + cross + 3
    torch.testing.assert_close(
        smoothed[..., :primary, :], legacy[..., :primary, :] * switch[..., None, None],
        rtol=1.0e-13, atol=1.0e-15,
    )
    torch.testing.assert_close(
        smoothed[..., primary : primary + cross, :],
        legacy[..., primary : primary + cross, :] * switch.square()[..., None, None],
        rtol=1.0e-13, atol=1.0e-15,
    )
    # The lattice-frame carriers are not pair carriers and stay as they are.
    assert torch.equal(smoothed[..., primary + cross :, :], legacy[..., primary + cross :, :])
    # The first direct carrier is exactly m_1 / sqrt(|m_1|^2 + (2 pi s)^2).
    torch.testing.assert_close(
        smoothed[..., 0, :],
        first / torch.sqrt(squared + (2.0 * math.pi * SMOOTHING) ** 2)[..., None],
        rtol=1.0e-13, atol=1.0e-15,
    )
    # In separation units: rho / sqrt(rho^2 + s^2).  rho is the smooth periodic
    # separation: it equals r far below the cell and does not exceed it in this
    # lattice (not a general property of skew bases; 0.82 for
    # r = 0.9 in this 4 to 5 angstrom cell).
    rho = float(torch.sqrt(squared[0, 1])) / (2.0 * math.pi)
    if offset <= 0.05:
        assert rho == pytest.approx(offset, rel=2.0e-3)
    assert 0.7 * offset < rho <= offset
    assert float(smoothed[0, 1, 0].norm()) == pytest.approx(
        rho / math.sqrt(rho * rho + SMOOTHING * SMOOTHING), rel=1.0e-12
    )


def test_carriers_are_linear_below_the_length_and_unchanged_far_above_it() -> None:
    primary = HARMONICS + len(SCALES)
    near = 1.0e-2  # six legacy layers out: the legacy direct carrier is a unit vector
    legacy_near = _bases(_transformer(None), _pair(near))[0, 1, :primary].norm(dim=-1)
    smooth_near = _bases(_transformer(SMOOTHING), _pair(near))[0, 1, :primary].norm(dim=-1)
    assert float(legacy_near[0]) > 0.98
    assert float(smooth_near.max()) < 1.05 * near / SMOOTHING
    # Halving the separation halves the first direct carrier (linear regime).  A
    # carrier whose own moment is already below sqrt(epsilon) falls faster.
    half = _bases(_transformer(SMOOTHING), _pair(near / 2))[0, 1, :primary].norm(dim=-1)
    assert float(half[0] / smooth_near[0]) == pytest.approx(0.5, rel=2.0e-3)
    assert float((half / smooth_near).max()) < 0.51
    # Far above the length the change is second order in s / rho.
    far = _bases(_transformer(SMOOTHING), _pair(1.7))[0, 1, :primary]
    legacy_far = _bases(_transformer(None), _pair(1.7))[0, 1, :primary]
    assert float((far - legacy_far).norm(dim=-1).max()) < 0.02


def test_the_readout_jump_across_a_coincidence_is_removed() -> None:
    """Same weights, opposite sides of a coincidence 0.002 angstrom apart."""

    scalars = torch.randn(3, 5, generator=torch.Generator().manual_seed(3), dtype=F64)
    scalars[1] = scalars[0]  # identical tokens: the symmetric-orbit case
    jumps = {}
    for label, smoothing in (("legacy", None), ("smoothed", SMOOTHING)):
        transformer = _transformer(smoothing).eval()
        plus = transformer(scalars, _pair(+1.0e-3), LATTICE).vector_channels
        minus = transformer(scalars, _pair(-1.0e-3), LATTICE).vector_channels
        relative = (plus[0] - plus[1]) - (minus[0] - minus[1])
        jumps[label] = float(relative.detach().norm())
    assert jumps["legacy"] > 0.0
    assert jumps["smoothed"] < 0.05 * jumps["legacy"]


def test_equivariance_and_padding_hold_with_the_switch() -> None:
    transformer = _transformer(SMOOTHING).eval()
    generator = torch.Generator().manual_seed(11)
    scalars = torch.randn(4, 5, generator=generator, dtype=F64)
    fractional = torch.rand(4, 3, generator=generator, dtype=F64)
    fractional[2] = fractional[0] + torch.tensor([0.004, -0.002, 0.001], dtype=F64)
    reference = transformer(scalars, fractional, LATTICE)
    permutation = torch.tensor([2, 0, 3, 1])
    permuted = transformer(scalars[permutation], fractional[permutation], LATTICE)
    torch.testing.assert_close(
        permuted.vector_channels, reference.vector_channels[permutation], rtol=1e-11, atol=1e-11
    )
    rotation, _ = torch.linalg.qr(
        torch.tensor([[0.3, -0.7, 0.2], [0.4, 0.1, -0.8], [0.6, 0.5, 0.3]], dtype=F64)
    )
    if torch.linalg.det(rotation) < 0:
        rotation[:, 0] *= -1
    rotated = transformer(scalars, fractional, LATTICE @ rotation)
    torch.testing.assert_close(
        rotated.vector_channels, reference.vector_channels @ rotation, rtol=2e-11, atol=2e-11
    )
    torch.testing.assert_close(
        rotated.scalar_channels, reference.scalar_channels, rtol=2e-11, atol=2e-11
    )
    # One padded slot on top of a real atom must contribute exact zeros.
    padded_scalars = torch.cat((scalars, torch.full((1, 5), 7.0, dtype=F64)))[None]
    padded_fractional = torch.cat((fractional, fractional[:1]))[None]
    mask = torch.tensor([[True, True, True, True, False]])
    padded = transformer(padded_scalars, padded_fractional, LATTICE[None], atom_mask=mask)
    torch.testing.assert_close(
        padded.vector_channels[0, :4], reference.vector_channels, rtol=1e-12, atol=1e-12
    )
    assert bool((padded.vector_channels[0, 4:] == 0.0).all())


def test_collision_values_and_gradients_are_finite_with_the_switch() -> None:
    transformer = _transformer(SMOOTHING)
    scalars = torch.randn(3, 5, generator=torch.Generator().manual_seed(37), dtype=F64)
    fractional = torch.tensor(
        [[0.2, 0.2, 0.2], [0.2, 0.2, 0.2], [0.71, 0.43, 0.86]], dtype=F64, requires_grad=True
    )
    lattice = LATTICE.clone().requires_grad_(True)
    output = transformer(scalars, fractional, lattice)
    (output.vector_channels.square().mean() + output.scalar_channels.square().mean()).backward()
    assert torch.isfinite(output.vector_channels).all()
    assert fractional.grad is not None and torch.isfinite(fractional.grad).all()
    assert lattice.grad is not None and torch.isfinite(lattice.grad).all()


@pytest.mark.parametrize("bad", [0.0, -0.2, 1.0e-3, float("nan"), float("inf"), True, "0.2"])
def test_the_length_is_validated(bad) -> None:  # type: ignore[no-untyped-def]
    # 1e-3 is refused: (2 pi s)^2 must exceed normalization_epsilon = 1e-4.
    with pytest.raises(ValueError, match="pair_carrier_smoothing_length"):
        _transformer(bad)
    with pytest.raises(DomainError, match="geometry_pair_carrier_smoothing_length"):
        ModelConfig(geometry_pair_carrier_smoothing_length=bad)


def test_the_length_needs_the_dense_backbone() -> None:
    with pytest.raises(DomainError, match="invariant_global"):
        ModelConfig(
            geometry_backbone="equivariant_local", geometry_pair_carrier_smoothing_length=0.2
        )


def test_the_switch_also_acts_at_the_half_lattice_separations() -> None:
    """``m_1`` vanishes on ``{0, 1/2}^3``: the documented scope (review 2026-11-14)."""

    transformer = _transformer(SMOOTHING)
    epsilon = transformer.config.normalization_epsilon
    cubic = 4.0 * torch.eye(3, dtype=F64)
    floor = (2.0 * math.pi * SMOOTHING) ** 2

    def first_direct_norm(delta, smoothing):  # type: ignore[no-untyped-def]
        fractional = torch.tensor([[0.0, 0.0, 0.0], delta], dtype=F64)
        bases = _transformer(smoothing)._periodic_geometry(fractional[None], cubic[None])
        return float(bases.covariant_bases[0, 0, 1, 0].norm())

    for delta in ([0.5, 0.0, 0.0], [0.5, 0.5, 0.0], [0.5, 0.5, 0.5]):
        # Exactly at a half-lattice separation both carriers vanish (sin(pi) is 1.2e-16
        # in float64, so the legacy carrier is 4.9e-16 / sqrt(epsilon) = 4.9e-14).
        assert first_direct_norm(delta, None) < 1e-12
        assert first_direct_norm(delta, SMOOTHING) < 1e-14
    # Near it, at 1.92 angstrom: the legacy carrier is a unit vector, the switched one is
    # damped exactly as a pair whose smooth separation rho is 0.08 angstrom.
    near = [0.48, 0.0, 0.0]
    squared = (4.0 * math.sin(2.0 * math.pi * 0.48)) ** 2
    assert first_direct_norm(near, None) == pytest.approx(
        math.sqrt(squared / (squared + epsilon)), rel=1e-12
    )
    assert first_direct_norm(near, SMOOTHING) == pytest.approx(
        math.sqrt(squared / (squared + floor)), rel=1e-12
    )
    assert first_direct_norm(near, SMOOTHING) < 0.4 < 0.99 < first_direct_norm(near, None)
    # Away from every point of {0, 1/2}^3 the switch is close to one.
    assert first_direct_norm([0.25, 0.25, 0.25], SMOOTHING) > 0.98


def test_the_model_forwards_the_length_to_the_field_and_the_transformer() -> None:
    from test_learned import _tiny_config

    from crystalatlas.catalogue import MINIMAL_CATALOGUE
    from crystalatlas.learned import CrystalAtlasModel

    base = _tiny_config()
    legacy = CrystalAtlasModel(base, catalogue=MINIMAL_CATALOGUE)
    assert legacy.geometry_field.config.pair_carrier_smoothing_length is None
    assert legacy.geometry_field.transformer.config.pair_carrier_smoothing_length is None
    config = replace(
        base, model=replace(base.model, geometry_pair_carrier_smoothing_length=SMOOTHING)
    )
    model = CrystalAtlasModel(config, catalogue=MINIMAL_CATALOGUE)
    assert model.geometry_field.config.pair_carrier_smoothing_length == SMOOTHING
    assert model.geometry_field.transformer.config.pair_carrier_smoothing_length == SMOOTHING
    assert set(model.state_dict()) == set(legacy.state_dict())
    assert config.fingerprint != base.fingerprint
