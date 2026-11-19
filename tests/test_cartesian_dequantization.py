"""Cartesian-metric dequantization kernel (review 2026-10-29, finding C1).

The legacy kernel ``fractional-v1`` adds ``sigma_eta N(0, I)`` in the stored
torus coordinates, which are fractional: for a 3 x 3 x 30 A cell that is 0.09 A
in-plane and 0.9 A along c.  ``cartesian-v2`` draws the torus noise with
covariance ``sigma^2 (J_hat^T P_N J_hat)^-1`` at the record, so ``sigma`` is the
RMS Cartesian displacement per free coordinate in the translation-quotient
metric, in angstrom, for every cell shape.  The exact oracle carries one
covariance per record, so the production objective remains checkable.
"""

from __future__ import annotations

import math
from fractions import Fraction

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from crystalatlas.catalogue import MINIMAL_CATALOGUE  # noqa: E402
from crystalatlas.composition import Composition  # noqa: E402
from crystalatlas.config import CrystalAtlasConfig, DomainConfig, ModelConfig  # noqa: E402
from crystalatlas.decoder import CrystalScaffold, OrbitAssignment  # noqa: E402
from crystalatlas.errors import DomainError  # noqa: E402
from crystalatlas.flow import sample_dequantized_endpoint  # noqa: E402
from crystalatlas.flow_oracle import TinyFlowOracle  # noqa: E402
from crystalatlas.geometry import ContinuousCover, IntrinsicCoverLayout, OrbitType  # noqa: E402
from crystalatlas.learned import CrystalAtlasModel, LearnedCrystalGenerator  # noqa: E402
from crystalatlas.nn.bridge import (  # noqa: E402
    TorchCoverGeometry,
    intrinsic_noise_covariance,
    intrinsic_noise_transform,
)
from crystalatlas.nn.field import translation_quotient_normal_matrix  # noqa: E402
from crystalatlas.training_objectives import CFMDequantization  # noqa: E402

pytestmark = pytest.mark.ml
F64 = torch.float64


def test_noise_transform_sets_the_endpoint_covariance_and_legacy_is_unchanged() -> None:
    period = torch.eye(2, dtype=F64)
    eta = torch.tensor([[0.3, 0.7]], dtype=F64).expand(20000, -1)
    z = torch.zeros(20000, 1, dtype=F64)
    # Small against the unit torus, so wrapping does not truncate the variance.
    sigma = torch.tensor([[0.0016, 0.0006], [0.0006, 0.0036]], dtype=F64)
    transform = torch.linalg.cholesky(sigma).transpose(-1, -2)  # T^T T = Sigma
    out = sample_dequantized_endpoint(
        z, eta, period, sigma_z=0.1, sigma_eta=1.0,
        generator=torch.Generator().manual_seed(1), eta_noise_transform=transform,
    )
    delta = out.eta - eta
    delta -= torch.round(delta)
    covariance = delta.T @ delta / len(delta)
    torch.testing.assert_close(covariance, sigma, rtol=0.05, atol=5e-5)

    legacy = sample_dequantized_endpoint(
        z[:4], eta[:4], period, sigma_z=0.1, sigma_eta=0.05,
        generator=torch.Generator().manual_seed(2),
    )
    g = torch.Generator().manual_seed(2)
    torch.randn(z[:4].shape, dtype=F64, generator=g)  # the z draw comes first
    noise = torch.randn(eta[:4].shape, dtype=F64, generator=g)
    expected = torch.remainder(eta[:4] + 0.05 * noise, 1.0)
    torch.testing.assert_close(legacy.eta, expected, rtol=0, atol=0)
    with pytest.raises(ValueError, match="eta_noise_transform"):
        sample_dequantized_endpoint(
            z[:4], eta[:4], period, sigma_z=0.1, sigma_eta=1.0,
            eta_noise_transform=torch.eye(3, dtype=F64),
        )


def _p1_layout() -> IntrinsicCoverLayout:
    cover = ContinuousCover(
        nu=1,
        space_group="P1",
        composition=Composition(((8, 1), (14, 1))),
        orbits=(OrbitType("1a", 8), OrbitType("1a", 14)),
    )
    return IntrinsicCoverLayout(cover)


def test_covariance_is_the_inverse_quotient_metric_scaled_by_sigma_squared() -> None:
    layout = _p1_layout()
    bridge = TorchCoverGeometry(layout).double()
    z = torch.tensor(layout.chart.encode_latent(np.diag([9.0, 9.0, 900.0])), dtype=F64)
    eta = torch.tensor([0.21, 0.63, 0.37], dtype=F64)
    covariance = intrinsic_noise_covariance(bridge, z, eta, 0.2, reference_length=1.0)
    metric = translation_quotient_normal_matrix(
        bridge(z, eta).jacobian_cartesian, reference_length=1.0
    )
    # tr(M Sigma) = sigma^2 d: the RMS quotient displacement is sigma per free coordinate.
    assert float(torch.trace(metric @ covariance)) == pytest.approx(0.04 * 3, rel=1e-10)
    assert torch.linalg.eigvalsh(covariance).min() > 0.0
    transform = intrinsic_noise_transform(covariance)
    torch.testing.assert_close(transform.T @ transform, covariance, rtol=1e-12, atol=1e-14)
    empty_cover = ContinuousCover(
        nu=1, space_group="P1", composition=Composition(((6, 1),)), orbits=(OrbitType("1a", 6),)
    )
    empty = TorchCoverGeometry(IntrinsicCoverLayout(empty_cover)).double()
    assert intrinsic_noise_covariance(
        empty, torch.zeros(6, dtype=F64), torch.zeros(0, dtype=F64), 0.2, reference_length=1.0
    ).shape == (0, 0)


def _model(**model_options) -> CrystalAtlasModel:  # type: ignore[no-untyped-def]
    torch.manual_seed(0)
    config = CrystalAtlasConfig(
        domain=DomainConfig(
            element_vocabulary=(8, 14),
            dng_element_vocabulary=(8, 14),
            max_distinct_elements=2,
            max_raw_atoms=8,
            max_formula_units=4,
        ),
        model=ModelConfig(
            composition_width=16, composition_layers=1, selector_width=16, selector_layers=1,
            geometry_scalar_width=16, geometry_vector_width=8, geometry_layers=1,
            attention_heads=2, radial_features=4, **model_options,
        ),
    )
    return CrystalAtlasModel(config, catalogue=MINIMAL_CATALOGUE).double()


@pytest.mark.parametrize(
    ("kernel", "isotropic"), [("fractional-v1", False), ("cartesian-v2", True)]
)
def test_production_endpoint_noise_is_isotropic_in_space_only_under_cartesian_v2(
    kernel: str, isotropic: bool
) -> None:
    """A 3 x 3 x 30 A P1 cell: the relative Cartesian displacement of the two atoms."""

    model = _model()
    generator = LearnedCrystalGenerator(model, allow_untrained=True)
    layout = model.layout(_p1_layout().cover)
    scaffold = CrystalScaffold.from_mapping(
        nu=1,
        space_group="P1",
        reduced_composition={8: 1, 14: 1},
        orbits=(
            OrbitAssignment("1a", 8, (0, 0, 0)),
            OrbitAssignment(
                "1a", 14, (Fraction(21, 100), Fraction(63, 100), Fraction(37, 100))
            ),
        ),
    )
    metric = np.diag([9.0, 9.0, 900.0])
    count = 2048
    g = torch.Generator().manual_seed(7)
    result = generator.geometry_cfm_objective_batch(
        (scaffold,) * count,
        (metric,) * count,
        time=torch.ones(count, dtype=F64),
        sigma_z=0.05,
        sigma_eta=0.1,
        generator=g,
        geometry_field=model.geometry_field,
        intrinsic_kernel=kernel,
    )
    eta_data = torch.tensor(layout.encode_exact_parameters(scaffold), dtype=F64)
    delta = result.path.state.eta - eta_data
    delta -= torch.round(delta)
    cartesian = delta @ torch.tensor(np.linalg.cholesky(metric), dtype=F64)
    variance = cartesian.var(dim=0)
    ratio = float(variance[2] / variance[0])
    if isotropic:
        assert 0.85 < ratio < 1.18, variance
        # Two atoms: the quotient metric is half the single-atom metric, so the
        # relative displacement has variance 2 sigma^2 per axis.
        torch.testing.assert_close(
            variance, torch.full((3,), 2 * 0.1**2, dtype=F64), rtol=0.12, atol=0.0
        )
    else:
        assert ratio > 60.0, variance


def test_dequantization_payload_keeps_the_legacy_identity() -> None:
    legacy = CFMDequantization()
    assert legacy.intrinsic_kernel == "fractional-v1"
    assert "intrinsic_kernel" not in legacy.to_dict()
    cartesian = CFMDequantization(intrinsic_kernel="cartesian-v2")
    assert cartesian.to_dict()["intrinsic_kernel"] == "cartesian-v2"
    assert set(cartesian.to_dict()) == set(legacy.to_dict()) | {"intrinsic_kernel"}
    with pytest.raises(DomainError, match="intrinsic_kernel"):
        CFMDequantization(intrinsic_kernel="cartesian")


def test_oracle_requires_exactly_one_kernel_description() -> None:
    z = torch.zeros(1, 1, dtype=F64)
    eta = torch.zeros(1, 1, dtype=F64)
    w = torch.ones(1, dtype=F64)
    period = torch.eye(1, dtype=F64)
    with pytest.raises(DomainError, match="exactly one"):
        TinyFlowOracle(z, eta, w, period, sigma_z=0.2)
    with pytest.raises(DomainError, match="exactly one"):
        TinyFlowOracle(
            z, eta, w, period, sigma_z=0.2, sigma_eta=0.125,
            eta_covariances=[torch.eye(1, dtype=F64) * 0.125**2],
        )
    with pytest.raises(DomainError, match="positive definite"):
        TinyFlowOracle(
            z, eta, w, period, sigma_z=0.2, eta_covariances=[torch.zeros(1, 1, dtype=F64)]
        )
    legacy = TinyFlowOracle(z, eta, w, period, sigma_z=0.2, sigma_eta=0.125)
    explicit = TinyFlowOracle(
        z, eta, w, period, sigma_z=0.2, eta_covariances=[torch.eye(1, dtype=F64) * 0.125**2]
    )
    point_z = torch.zeros(3, 1, dtype=F64)
    point_eta = torch.linspace(0, 0.9, 3, dtype=F64)[:, None]
    torch.testing.assert_close(
        legacy.log_density(torch.tensor(0.5, dtype=F64), point_z, point_eta),
        explicit.log_density(torch.tensor(0.5, dtype=F64), point_z, point_eta),
        rtol=1e-12,
        atol=1e-12,
    )
    assert legacy.moment(1.0, (2,)) == pytest.approx(explicit.moment(1.0, (2,)), abs=1e-14)


def test_oracle_with_record_covariances_matches_its_own_sampler_and_the_objective() -> None:
    """Per-record Cartesian covariances: exact moments against production samples."""

    import test_flow_oracle_learned as toy_module

    from crystalatlas.flow_oracle_learned import toy_oracle

    model = toy_module._model()
    toy = toy_module._toy(model)
    # The oracle certifies fractional scales of about 0.125 and above with its
    # 64-mode budget.  Choose the Cartesian scale so that the sharpest record (the
    # largest quotient metric) has exactly that fractional width: a test scale,
    # not a physical recommendation.
    layout = model.layout(toy.cover)
    bridge = TorchCoverGeometry(layout).double()
    metrics = []
    for scaffold, metric in zip(toy.scaffolds, toy.metrics, strict=True):
        z_m = torch.tensor(layout.chart.encode_latent(np.asarray(metric)), dtype=F64)
        eta_m = torch.tensor(layout.encode_exact_parameters(scaffold), dtype=F64)
        metrics.append(
            float(
                translation_quotient_normal_matrix(bridge(z_m, eta_m).jacobian_cartesian)[0, 0]
            )
        )
    sigma = 0.125 * math.sqrt(max(metrics))
    toy = type(toy)(toy.scaffolds, toy.metrics, sigma_z=toy.sigma_z, sigma_eta=sigma)
    oracle = toy_oracle(model, toy, intrinsic_kernel="cartesian-v2")
    assert oracle.sigma_eta is None
    covariances = oracle.eta_covariances
    assert covariances.shape == (3, 1, 1)
    assert len({round(float(c), 12) for c in covariances.flatten()}) == 3  # record dependent
    # (i) moments of the oracle's own production sampler at t = 1.
    paths = oracle.sample_paths(
        torch.ones(40000, dtype=F64), generator=torch.Generator().manual_seed(3)
    )
    a = (paths.state.eta @ torch.linalg.inv(oracle.period_basis))[:, 0]
    for n in (1, 2, 3):
        estimate = complex(torch.exp(2j * math.pi * n * a).mean())
        assert abs(estimate - oracle.moment(1.0, (n,))) < 4.5 / math.sqrt(40000), n
    # (ii) the production objective under cartesian-v2 has the oracle's velocity law.
    generator = LearnedCrystalGenerator(model, allow_untrained=True)
    g = torch.Generator().manual_seed(17)
    states, targets, zs = [], [], []
    for _ in range(24):
        indices = torch.randint(len(toy.scaffolds), (64,), generator=g).tolist()
        result = generator.geometry_cfm_objective_batch(
            tuple(toy.scaffolds[i] for i in indices),
            tuple(toy.metrics[i] for i in indices),
            time=torch.full((64,), 0.6, dtype=F64),
            sigma_z=toy.sigma_z,
            sigma_eta=toy.sigma_eta,
            generator=g,
            geometry_field=model.geometry_field,
            intrinsic_kernel="cartesian-v2",
        )
        states.append(result.path.state.eta.detach())
        targets.append(result.path.target_velocity.eta.detach())
        zs.append(result.path.state.z.detach())
    eta = torch.cat(states)
    exact = oracle.velocity(torch.tensor(0.6, dtype=F64), torch.cat(zs), eta).eta
    residual = (torch.cat(targets) - exact)[:, 0]
    scores = []
    for phi in (
        torch.sin(2 * math.pi * eta[:, 0]),
        torch.cos(2 * math.pi * eta[:, 0]),
        torch.cos(4 * math.pi * eta[:, 0]),
    ):
        values = residual * phi
        scores.append(float(values.mean() / (values.std() / math.sqrt(len(values)))))
    assert max(abs(s) for s in scores) < 4.0, scores
