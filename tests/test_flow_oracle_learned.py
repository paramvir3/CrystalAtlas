"""Harness for the learned half of F10 (review M1): validation, not the gate.

The toy is P2_13 (cubic, d_z = 1) with one 4a orbit (x, x, x) of carbon, so
d_eta = 1.  It has no identical-orbit relabelling and no P-1 sign gauge, so the
production residual action is the identity.  Three records with distinct
(z, x) give a Gaussian-mixture times wrapped-heat-kernel endpoint law that the
exact oracle evaluates in closed form.

These tests fix the harness before any learned threshold exists (theory 4.10):
* with the oracle's own velocity, the velocity error is exactly 0, and the
  endpoint density L1 and moment errors meet the deterministic 1e-3 bounds;
* the reverse-mode pushforward Jacobian agrees with the forward-mode one of the
  deterministic gate;
* an untrained production field is measured as far from the oracle; and
* the production objective trains on the toy.
"""

from __future__ import annotations

from fractions import Fraction
from functools import cache

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from crystalatlas.config import (  # noqa: E402
    CrystalAtlasConfig,
    DomainConfig,
    ModelConfig,
    NumericsConfig,
)
from crystalatlas.decoder import CrystalScaffold, OrbitAssignment  # noqa: E402
from crystalatlas.flow_oracle_learned import (  # noqa: E402
    F10Toy,
    LearnedToyField,
    endpoint_errors,
    endpoint_quadrature_grid,
    pushforward_density_reverse_mode,
    toy_oracle,
    train_on_toy,
    velocity_relative_error,
)
from crystalatlas.full_catalogue import build_space_group_setting  # noqa: E402
from crystalatlas.learned import CrystalAtlasModel  # noqa: E402

pytestmark = pytest.mark.ml

PRODUCTION_STEPS = NumericsConfig().ode_steps
Z_DATA = (-0.3, 0.5, 1.2)
X_DATA = (Fraction(3, 25), Fraction(3, 10), Fraction(41, 100))


@cache
def _catalogue():  # type: ignore[no-untyped-def]
    from crystalatlas.catalogue import (
        FIXTURE_EXPECTED_ROWS,
        MINIMAL_CATALOGUE,
        CertifiedCatalogue,
    )

    settings = {key: MINIMAL_CATALOGUE.get(key) for key in ("P1", "P-1")}
    rows = dict(FIXTURE_EXPECTED_ROWS)
    setting = build_space_group_setting(198)
    settings[setting.key] = setting
    rows[setting.key] = frozenset(setting.wyckoff_positions)
    return CertifiedCatalogue(settings, rows, certification_scope="F10 learned toy")


def _model(seed: int = 0) -> CrystalAtlasModel:
    torch.manual_seed(seed)
    config = CrystalAtlasConfig(
        domain=DomainConfig(
            element_vocabulary=(6, 8),
            dng_element_vocabulary=(6, 8),
            max_distinct_elements=2,
            max_raw_atoms=8,
            max_formula_units=8,
            interaction_cutoff_angstrom=3.0,
        ),
        model=ModelConfig(
            composition_width=16,
            composition_layers=1,
            selector_width=16,
            selector_layers=1,
            geometry_scalar_width=24,
            geometry_vector_width=8,
            geometry_layers=2,
            attention_heads=2,
            radial_features=6,
        ),
    )
    return CrystalAtlasModel(config, catalogue=_catalogue()).double()


def _toy(model: CrystalAtlasModel) -> F10Toy:
    scaffolds = tuple(
        CrystalScaffold.from_mapping(
            nu=4,
            space_group="P2_13",
            reduced_composition={6: 1},
            orbits=(OrbitAssignment("4a", 6, (x,)),),
        )
        for x in X_DATA
    )
    from crystalatlas.geometry import ContinuousCover

    layout = model.layout(ContinuousCover.from_exact_scaffold(scaffolds[0]))
    metrics = tuple(np.asarray(layout.chart.decode_latent([z])) for z in Z_DATA)
    return F10Toy(scaffolds, metrics, sigma_z=0.25, sigma_eta=0.125)


def test_oracle_matches_the_records_and_the_production_layout() -> None:
    model = _model()
    toy = _toy(model)
    oracle = toy_oracle(model, toy)
    np.testing.assert_allclose(oracle.z_data[:, 0].numpy(), Z_DATA, atol=1e-10)
    np.testing.assert_allclose(
        oracle.eta_data[:, 0].numpy(), [float(x) for x in X_DATA], atol=1e-12
    )
    assert oracle.lattice_dimension == 1 and oracle.intrinsic_dimension == 1


def test_oracle_velocity_has_zero_velocity_error_and_meets_the_deterministic_gate() -> None:
    model = _model()
    toy = _toy(model)
    oracle = toy_oracle(model, toy)
    field = LearnedToyField(model, toy.cover)
    zero = velocity_relative_error(
        field, oracle, times=(0.2, 0.6), samples_per_time=16, seed=3,
        velocity_override=oracle.velocity,
    )
    assert zero["relative_velocity_error"] == 0.0
    assert zero["lattice_relative_velocity_error"] == 0.0
    assert zero["torus_relative_velocity_error"] == 0.0
    gate = endpoint_errors(oracle.velocity, oracle, z_step=0.25, torus_points=16,
                           steps=PRODUCTION_STEPS)
    assert abs(gate["target_mass"] - 1.0) <= 1e-7
    assert gate["density_l1"] <= 1e-3
    assert gate["max_relative_moment_error"] <= 1e-3


def test_reverse_mode_pushforward_matches_the_forward_mode_gate() -> None:
    import test_flow_oracle as deterministic

    model = _model()
    oracle = toy_oracle(model, _toy(model))
    z, eta, _ = endpoint_quadrature_grid(oracle, z_step=0.5, z_margin=1.0, torus_points=4)
    reverse = pushforward_density_reverse_mode(oracle.velocity, oracle, z, eta, steps=16)
    forward = deterministic._pushforward_density(oracle, z, eta, 16)
    torch.testing.assert_close(reverse, forward, rtol=1e-10, atol=1e-14)


def test_untrained_field_is_measured_far_from_the_oracle() -> None:
    model = _model()
    toy = _toy(model)
    oracle = toy_oracle(model, toy)
    field = LearnedToyField(model, toy.cover)
    error = velocity_relative_error(
        field, oracle, times=(0.3, 0.7), samples_per_time=16, seed=5
    )
    assert error["relative_velocity_error"] > 0.5
    coarse = endpoint_errors(field, oracle, z_step=0.5, torus_points=6, steps=8)
    assert np.isfinite(coarse["density_l1"]) and coarse["density_l1"] > 0.1


def test_production_objective_trains_on_the_toy() -> None:
    model = _model()
    toy = _toy(model)
    losses = train_on_toy(model, toy, steps=3, batch_size=4, learning_rate=1e-3, seed=7)
    assert len(losses) == 3 and all(np.isfinite(losses))


def test_model_layout_is_cached_exactly_per_cover() -> None:
    """``model.layout`` is a deterministic function of the cover (fixed domain and
    catalogue), so a bounded cache is exact.  Rebuilding it per record made 77% of
    a training step (point-group closure re-validation)."""

    import crystalatlas.learned as learned_module
    from crystalatlas.composition import Composition
    from crystalatlas.errors import DomainError
    from crystalatlas.geometry import ContinuousCover, IntrinsicCoverLayout, OrbitType

    model = _model()
    toy = _toy(model)
    first = model.layout(toy.cover)
    assert model.layout(toy.cover) is first
    fresh = IntrinsicCoverLayout(
        toy.cover, catalogue=model.catalogue, **model.scientific_config.domain.layout_options()
    )
    np.testing.assert_array_equal(first.period_basis, fresh.period_basis)
    np.testing.assert_array_equal(first.chart.reference_metric, fresh.chart.reference_metric)
    other = ContinuousCover(
        nu=4,
        space_group="P2_13",
        composition=Composition(((8, 1),)),
        orbits=(OrbitType("4a", 8),),
    )
    assert model.layout(other) is not first
    # Validation still runs before the cache: an oversized cover is refused.
    with pytest.raises(DomainError):
        model.layout(
            ContinuousCover(
                nu=12,
                space_group="P2_13",
                composition=Composition(((6, 1),)),
                orbits=(OrbitType("4a", 6),) * 3,
            )
        )
    from crystalatlas.nn.bridge import TorchCoverGeometry

    bridge = TorchCoverGeometry(first)
    assert not np.shares_memory(bridge.period_basis.numpy(), first.period_basis)
    assert learned_module.LAYOUT_CACHE_SIZE >= 1
    model._layout_cache.clear()
    assert model.layout(toy.cover) is not first  # rebuilt after eviction


# --- residual Euclidean-normalizer gauge (finding from the learned F10 run) ---------
# P2_13's Euclidean normalizer is Ia-3; its proper part beyond the group is the
# centring translation (1/2, 1/2, 1/2), which maps the 4a orbit at x onto the one at
# x + 1/2.  The two parameters are the same crystal, so a physically invariant field
# cannot separate them.  The production objective symmetrizes only identical-orbit
# relabellings and P-1 signs, so the learnable target is the gauge-symmetrized one.

HALF = Fraction(1, 2)


def test_residual_gauge_translation_is_verified_exactly() -> None:
    from crystalatlas.errors import DomainError
    from crystalatlas.flow_oracle_learned import ResidualTranslation

    model = _model()
    toy = _toy(model)
    gauge = ResidualTranslation((HALF, HALF, HALF), ((HALF,),))
    symmetric = toy_oracle(model, toy, gauge=(gauge,))
    assert len(symmetric.weights) == 2 * len(toy.scaffolds)
    np.testing.assert_allclose(symmetric.weights.sum().item(), 1.0, rtol=0, atol=1e-15)
    shifted = np.sort(np.mod(symmetric.eta_data[:, 0].numpy(), 1.0))
    expected = np.sort(np.mod([float(x) + d for x in X_DATA for d in (0.0, 0.5)], 1.0))
    np.testing.assert_allclose(shifted, expected, atol=1e-12)
    zero = Fraction(0)
    # Closed (twice it is a lattice vector) but it does not map (x, x, x) onto itself.
    with pytest.raises(DomainError, match="not a residual gauge"):
        toy_oracle(model, toy, gauge=(ResidualTranslation((HALF, zero, zero), ((HALF,),)),))
    with pytest.raises(DomainError, match="closed"):
        toy_oracle(
            model,
            toy,
            gauge=(ResidualTranslation((Fraction(1, 3),) * 3, ((Fraction(1, 3),),)),),
        )


def test_field_is_invariant_under_the_residual_gauge() -> None:
    model = _model()
    toy = _toy(model)
    field = LearnedToyField(model, toy.cover)
    z = torch.tensor([[0.3], [0.3], [-0.2]], dtype=torch.float64)
    eta = torch.tensor([[0.12], [0.37], [0.71]], dtype=torch.float64)
    with torch.no_grad():
        left = field(torch.tensor(0.4, dtype=torch.float64), z, eta)
        shifted = torch.remainder(eta + 0.5, 1.0)
        right = field(torch.tensor(0.4, dtype=torch.float64), z, shifted)
    torch.testing.assert_close(left.eta, right.eta, rtol=1e-9, atol=1e-12)
    torch.testing.assert_close(left.z, right.z, rtol=1e-9, atol=1e-12)


def test_symmetrized_oracle_meets_the_deterministic_gate_and_is_gauge_invariant() -> None:
    from crystalatlas.flow_oracle_learned import ResidualTranslation

    model = _model()
    toy = _toy(model)
    gauge = ResidualTranslation((HALF, HALF, HALF), ((HALF,),))
    oracle = toy_oracle(model, toy, gauge=(gauge,))
    z = torch.tensor([[0.1], [0.9]], dtype=torch.float64)
    eta = torch.tensor([[0.2], [0.55]], dtype=torch.float64)
    t = torch.tensor(0.6, dtype=torch.float64)
    left = oracle.velocity(t, z, eta)
    right = oracle.velocity(t, z, torch.remainder(eta + 0.5, 1.0))
    torch.testing.assert_close(left.eta, right.eta, rtol=1e-10, atol=1e-12)
    gate = endpoint_errors(
        oracle.velocity, oracle, z_step=0.25, torus_points=16, steps=PRODUCTION_STEPS
    )
    assert gate["density_l1"] <= 1e-3 and gate["max_relative_moment_error"] <= 1e-3


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_cuda_field_matches_the_cpu_field() -> None:
    """GPU runs use float32 (production precision; consumer FP64 runs at ~1/64 rate).

    Each property is checked where it is well-posed:
    * device plumbing: CUDA float64 equals CPU float64 in the pushforward density
      (measured 1.4e-11);
    * field precision: float32 velocities agree with float64 to float32 level;
    * rho_v agrees.
    The pushforward of an untrained random field over 8 coarse RK4 steps is too
    ill-conditioned to compare across precisions (on this toy its mass is about
    20).  On the trained development field the float32/float64 L1 gap is 1.2e-6.
    """

    model = _model()
    toy = _toy(model)
    oracle = toy_oracle(model, toy, layout_gauge=True)
    cpu_field = LearnedToyField(model, toy.cover)
    z, eta, _ = endpoint_quadrature_grid(oracle, z_step=0.5, z_margin=1.0, torus_points=4)
    reference = pushforward_density_reverse_mode(cpu_field, oracle, z, eta, steps=8)
    cuda64 = LearnedToyField(_model().to(device="cuda"), toy.cover)
    density = pushforward_density_reverse_mode(cuda64, oracle, z, eta, steps=8, device="cuda")
    assert density.dtype == torch.float64 and density.device.type == "cpu"
    torch.testing.assert_close(density, reference, rtol=1e-9, atol=1e-12)
    cuda32 = LearnedToyField(_model().to(device="cuda", dtype=torch.float32), toy.cover)
    assert cuda32.device.type == "cuda" and cuda32.dtype == torch.float32
    g = torch.Generator().manual_seed(1)
    zz = torch.randn(64, 1, generator=g, dtype=torch.float64)
    ee = torch.rand(64, 1, generator=g, dtype=torch.float64)
    with torch.no_grad():
        exact = cpu_field(torch.tensor(0.5, dtype=torch.float64), zz, ee)
        single = cuda32(
            torch.tensor(0.5, dtype=torch.float32),
            zz.to("cuda", torch.float32),
            ee.to("cuda", torch.float32),
        )
    for left, right in ((exact.z, single.z), (exact.eta, single.eta)):
        scale = float(left.abs().max())
        assert float((left - right.double().cpu()).abs().max()) <= 1e-4 * scale
    cpu_rho = velocity_relative_error(
        cpu_field, oracle, times=(0.5,), samples_per_time=32, seed=3
    )
    cuda_rho = velocity_relative_error(
        cuda32, oracle, times=(0.5,), samples_per_time=32, seed=3
    )
    assert abs(cpu_rho["relative_velocity_error"] - cuda_rho["relative_velocity_error"]) < 1e-3
