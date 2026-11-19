"""CUDA regressions for algebraic contracts, not materials-quality benchmarks.

CPU/CUDA comparisons copy the same tensors and parameters; they never assume
the two devices implement identical random streams. FP64 algebra tolerances
are O(1e-12); neural solves/reductions permit O(1e-9). FP32 neural comparisons
permit O(1e-4) relative error with TF32 disabled. These small, well-conditioned
fixtures do not establish those bounds for every catalogue chart or GPU.
"""

from __future__ import annotations

import os
from fractions import Fraction
from itertools import permutations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from crystalatlas.catalogue import MINIMAL_CATALOGUE  # noqa: E402
from crystalatlas.composition import Composition  # noqa: E402
from crystalatlas.config import (  # noqa: E402
    CrystalAtlasConfig,
    DomainConfig,
    ModelConfig,
    NumericsConfig,
)
from crystalatlas.decoder import (  # noqa: E402
    CrystalScaffold,
    ExactCrystalDecoder,
    OrbitAssignment,
)
from crystalatlas.flow import (  # noqa: E402
    FlowState,
    canonicalize_torus,
    conditional_flow_path,
    sample_dequantized_endpoint,
)
from crystalatlas.geometry import (  # noqa: E402
    ContinuousCover,
    ContinuousGeometry,
    IntrinsicCoverLayout,
    OrbitType,
)
from crystalatlas.lattice import affine_invariant_spd_inner_product  # noqa: E402
from crystalatlas.learned import CrystalAtlasModel, LearnedCrystalGenerator  # noqa: E402
from crystalatlas.nn.bridge import (  # noqa: E402
    CoverConditionedGeometryField,
    TorchCoverGeometry,
)
from crystalatlas.nn.invariant_transformer import (  # noqa: E402
    InvariantPeriodicTransformer,
    InvariantPeriodicTransformerConfig,
)

pytestmark = pytest.mark.ml


@pytest.fixture(scope="module", autouse=True)
def strict_cuda_policy():
    """Restore precision, deterministic flags, environment, and both RNGs."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA scientific regression requires a CUDA-enabled PyTorch device")
    patch = pytest.MonkeyPatch()
    workspace = os.environ.get("CUBLAS_WORKSPACE_CONFIG")
    if workspace not in {":4096:8", ":16:8"}:
        if torch.cuda.is_initialized():
            pytest.fail(
                "Set CUBLAS_WORKSPACE_CONFIG=:4096:8 before CUDA initialization "
                "for the strict scientific regression process"
            )
        patch.setenv("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    previous = (
        torch.are_deterministic_algorithms_enabled(),
        torch.is_deterministic_algorithms_warn_only_enabled(),
        torch.get_float32_matmul_precision(),
        torch.backends.cuda.matmul.allow_tf32,
        torch.backends.cudnn.allow_tf32,
        torch.backends.cudnn.benchmark,
        torch.backends.cudnn.deterministic,
    )
    try:
        torch.use_deterministic_algorithms(True, warn_only=False)
        torch.set_float32_matmul_precision("highest")
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
            torch.set_rng_state(torch.Generator(device="cpu").manual_seed(7123).get_state())
            torch.cuda.manual_seed(7123)
            yield
            torch.cuda.synchronize()
    finally:
        torch.use_deterministic_algorithms(previous[0], warn_only=previous[1])
        torch.set_float32_matmul_precision(previous[2])
        torch.backends.cuda.matmul.allow_tf32 = previous[3]
        torch.backends.cudnn.allow_tf32 = previous[4]
        torch.backends.cudnn.benchmark = previous[5]
        torch.backends.cudnn.deterministic = previous[6]
        patch.undo()


@pytest.fixture(autouse=True)
def fixed_case_rng(strict_cuda_policy):
    """Keep selected-case results independent of test collection/order."""
    with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
        torch.set_rng_state(torch.Generator(device="cpu").manual_seed(7123).get_state())
        torch.cuda.manual_seed(7123)
        yield


def _layout() -> IntrinsicCoverLayout:
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


def _algebra_tolerance(dtype):
    return dict(rtol=2.0e-12, atol=2.0e-12) if dtype == torch.float64 else dict(
        rtol=4.0e-6, atol=4.0e-6
    )


def _neural_tolerance(dtype):
    return dict(rtol=5.0e-9, atol=5.0e-10) if dtype == torch.float64 else dict(
        rtol=2.0e-4, atol=2.0e-5
    )


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("batched", [False, True])
def test_cuda_complete_noisy_path_pushforward_and_cpu_agreement(dtype, batched):
    layout = _layout()
    action = torch.tensor(layout.residual_action_matrix((1, 0, 2)), dtype=dtype)
    assert not torch.equal(action.T @ action, torch.eye(6, dtype=dtype))
    period = torch.tensor(layout.period_basis, dtype=dtype)
    eta0 = torch.full((6,), 0.1, dtype=dtype)
    eta1 = canonicalize_torus(
        eta0 + torch.tensor([0.4, 0.01, -0.02, -0.4, 0.03, 0.01], dtype=dtype), period
    )
    z0, z1 = torch.zeros(6, dtype=dtype), torch.full((6,), 0.2, dtype=dtype)
    if batched:
        eta0, eta1, z0, z1 = (v.repeat(2, 1) for v in (eta0, eta1, z0, z1))
        action = torch.stack((torch.eye(6, dtype=dtype), action))
    supplied = (z0, eta0, z1, eta1, period, action)
    for time in (0.0, 0.37, 1.0):
        results = []
        for device in ("cpu", "cuda"):
            z_start, eta_start, z_end, eta_end, basis, row_action = (
                v.to(device) for v in supplied
            )
            t = torch.tensor(time, dtype=dtype, device=device, requires_grad=True)
            path = conditional_flow_path(
                z_start, eta_start, z_end, eta_end, t, basis, torus_action=row_action
            )
            plain = conditional_flow_path(z_start, eta_start, z_end, eta_end, t, basis)
            expected_velocity = torch.matmul(
                plain.target_velocity.eta.unsqueeze(-2), row_action
            ).squeeze(-2)
            torch.testing.assert_close(
                path.target_velocity.eta, expected_velocity, **_algebra_tolerance(dtype)
            )
            torch.testing.assert_close(
                path.state.eta,
                canonicalize_torus(
                    torch.matmul(plain.state.eta.unsqueeze(-2), row_action).squeeze(-2), basis
                ),
                **_algebra_tolerance(dtype),
            )
            assert path.target_velocity.eta.abs().max().item() > 0.5
            if time == 0.37:
                tangent = torch.autograd.functional.jacobian(
                    lambda value, z_start=z_start, eta_start=eta_start, z_end=z_end,
                    eta_end=eta_end, basis=basis, row_action=row_action: conditional_flow_path(
                        z_start, eta_start, z_end, eta_end, value, basis,
                        torus_action=row_action,
                    ).state.eta,
                    t,
                )
                torch.testing.assert_close(
                    tangent, expected_velocity, **_algebra_tolerance(dtype)
                )
            results.append(path)
        for attribute in ("state", "target_velocity"):
            for block in ("z", "eta"):
                torch.testing.assert_close(
                    getattr(getattr(results[1], attribute), block).cpu(),
                    getattr(getattr(results[0], attribute), block),
                    **_algebra_tolerance(dtype),
                )


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_cuda_endpoint_pushes_forward_whole_noise_and_keeps_cuda_rng(dtype):
    layout = _layout()
    period = torch.tensor(layout.period_basis, dtype=dtype, device="cuda")
    action = torch.tensor(
        layout.residual_action_matrix((1, 0, 2)), dtype=dtype, device="cuda"
    )
    z = torch.zeros(6, dtype=dtype, device="cuda")
    eta = torch.tensor([0.17, 0.23, 0.31, 0.41, 0.43, 0.47], dtype=dtype, device="cuda")
    eta.requires_grad_(True)
    plain_rng = torch.Generator(device="cuda").manual_seed(171)
    action_rng = torch.Generator(device="cuda").manual_seed(171)
    plain = sample_dequantized_endpoint(
        z, eta, period, sigma_z=0.05, sigma_eta=0.1, generator=plain_rng
    )
    changed = sample_dequantized_endpoint(
        z, eta, period, sigma_z=0.05, sigma_eta=0.1,
        generator=action_rng, torus_action=action,
    )
    torch.testing.assert_close(changed.z, plain.z, rtol=0, atol=0)
    torch.testing.assert_close(
        changed.eta, canonicalize_torus(plain.eta @ action, period),
        **_algebra_tolerance(dtype),
    )
    assert torch.equal(plain_rng.get_state(), action_rng.get_state())
    changed.eta.sum().backward()
    torch.testing.assert_close(eta.grad, action.sum(dim=-1), **_algebra_tolerance(dtype))


def test_cuda_group_averaged_interior_path_fourier_invariance():
    """Finite group averaging is an identity at every t, not a Monte Carlo claim."""
    layout = _layout()
    actions = torch.tensor(
        np.stack([layout.residual_action_matrix(p) for p in permutations(range(3))]),
        dtype=torch.float64, device="cuda",
    )
    basis = torch.tensor(layout.period_basis, dtype=torch.float64, device="cuda")
    z0 = torch.zeros(6, 6, dtype=torch.float64, device="cuda")
    eta0 = torch.full((6, 6), 0.1, dtype=torch.float64, device="cuda")
    eta1 = torch.tensor(
        [0.17, 0.23, 0.31, 0.41, 0.43, 0.47], dtype=torch.float64, device="cuda"
    ).repeat(6, 1)
    mode = torch.tensor([0, 0, 0, 1, 0, 0], dtype=torch.float64, device="cuda")
    for time in (0.0, 0.25, 0.5, 0.75, 1.0):
        path = conditional_flow_path(
            z0, eta0, z0 + 0.2, eta1, torch.tensor(time, device="cuda"), basis,
            torus_action=actions,
        )
        reference = torch.exp(2j * torch.pi * (path.state.eta @ mode)).mean()
        for action in actions:
            coefficient = torch.exp(2j * torch.pi * (path.state.eta @ action @ mode)).mean()
            torch.testing.assert_close(coefficient, reference, rtol=2.0e-12, atol=2.0e-12)


@pytest.mark.parametrize("space_group", ["P1", "P-1"])
def test_cuda_fp64_crystallographic_expansion_survives_neural_autocast(space_group):
    # Float64 is not autocast-eligible. This test checks FP64 numerics, not
    # sensitivity to removal of AMP guards; the FP32 test below checks that.
    if space_group == "P1":
        layout = _layout()
    else:
        layout = IntrinsicCoverLayout(
            ContinuousCover(
                nu=2, space_group="P-1", composition=Composition.from_counts({6: 1}),
                orbits=(OrbitType("2i", 6),),
            ), metric_bound=1.0, lattice_chart="ball-v1",
        )
    z = torch.linspace(-0.13, 0.17, layout.lattice_dimension, dtype=torch.float64)
    eta = torch.tensor(
        [0.17, 0.23, 0.31, 0.41, 0.43, 0.47][:layout.intrinsic_dimension],
        dtype=torch.float64,
    )
    expected = layout.decode(
        ContinuousGeometry(z.numpy(), eta.numpy()), symmetry_tolerance=1.0e-10
    )
    assert expected.audit.passed
    bridge = TorchCoverGeometry(layout).to(device="cuda", dtype=torch.float64)
    with torch.autocast("cuda", dtype=torch.float16):
        expanded = bridge(z.cuda(), eta.cuda())
        pullback = bridge.lattice_pullback_metric(z.cuda())
    for name in ("metric", "lattice", "fractional_coordinates", "jacobian_cartesian"):
        assert getattr(expanded, name).dtype == torch.float64
    assert pullback.dtype == torch.float64
    assert torch.linalg.eigvalsh(pullback).min().item() > 0
    torch.testing.assert_close(
        expanded.metric.cpu(), torch.tensor(expected.metric), rtol=2.0e-12, atol=2.0e-12
    )
    torch.testing.assert_close(
        expanded.fractional_coordinates.cpu(), torch.tensor(expected.fractional_positions),
        rtol=2.0e-12, atol=2.0e-12,
    )
    torch.testing.assert_close(
        expanded.jacobian_cartesian.cpu(), torch.tensor(expected.jacobian_cartesian),
        rtol=2.0e-12, atol=2.0e-12,
    )
    cpu_pullback = TorchCoverGeometry(layout).double().lattice_pullback_metric(z)
    torch.testing.assert_close(pullback.cpu(), cpu_pullback, rtol=2.0e-11, atol=2.0e-11)
    # An independent NumPy chart derivative checks the affine-invariant
    # contraction H_ij=tr(G^-1 d_iG G^-1 d_jG), not merely symmetry/positivity.
    # Five-point differentiation at h=1e-4 has O(h^4) truncation error; the
    # 2e-8 tolerance also allows float64 eigensolver error amplified by 1/h.
    formula_pullback = _finite_difference_pullback(layout, z.numpy())
    torch.testing.assert_close(
        pullback.cpu(), torch.tensor(formula_pullback), rtol=2.0e-8, atol=2.0e-8
    )
    # Test the exact rational scaffold decoder independently of floating orbit
    # expansion. Rationalization is used only to construct this test fixture.
    parameters = layout.reconstruct_parameters(eta.numpy())
    scaffold = CrystalScaffold.from_mapping(
        nu=layout.cover.nu, space_group=space_group, reduced_composition={6: 1},
        orbits=tuple(
            OrbitAssignment(
                orbit.wyckoff_label, 6,
                tuple(Fraction(float(x)).limit_denominator(10000) for x in parameter),
            )
            for orbit, parameter in zip(layout.cover.orbits, parameters, strict=True)
        ),
    )
    decoded = ExactCrystalDecoder(MINIMAL_CATALOGUE, metric_tolerance=1.0e-10).decode(
        scaffold, expanded.metric.cpu().numpy()
    )
    assert decoded.audit.passed and decoded.hidden_translation_index == 1


def _finite_difference_pullback(layout, latent):
    step = 1.0e-4
    directions = np.eye(layout.lattice_dimension) * step
    tangents = [
        (
            -layout.chart.decode_latent(latent + 2 * direction)
            + 8 * layout.chart.decode_latent(latent + direction)
            - 8 * layout.chart.decode_latent(latent - direction)
            + layout.chart.decode_latent(latent - 2 * direction)
        ) / (12 * step)
        for direction in directions
    ]
    metric = layout.chart.decode_latent(latent)
    return np.array([
        [affine_invariant_spd_inner_product(metric, left, right) for right in tangents]
        for left in tangents
    ])


@pytest.mark.parametrize("activation_dtype", [torch.float16, torch.bfloat16])
def test_cuda_fp32_bridge_guards_autocast_eligible_algebra(activation_dtype):
    layout = _layout()
    bridge = TorchCoverGeometry(layout).to(device="cuda", dtype=torch.float32)
    z = torch.linspace(-0.13, 0.17, layout.lattice_dimension, device="cuda")
    eta = torch.tensor([0.17, 0.23, 0.31, 0.41, 0.43, 0.47], device="cuda")
    with torch.autocast("cuda", enabled=False):
        reference = bridge(z, eta)
        reference_pullback = bridge.lattice_pullback_metric(z)
    with torch.autocast("cuda", dtype=activation_dtype):
        expanded = bridge(z, eta)
        pullback = bridge.lattice_pullback_metric(z)
        # Negative control intentionally bypasses only the public forward
        # guard, demonstrating that this fixture really exercises eligible
        # matrix products. The chart's own nested guard remains in effect.
        unguarded = bridge._forward_master(z, eta)
    assert unguarded.jacobian_cartesian.dtype == activation_dtype
    assert not torch.allclose(
        unguarded.jacobian_cartesian.float(), reference.jacobian_cartesian,
        rtol=2.0e-6, atol=2.0e-6,
    )
    for name in (
        "bounded_lattice_coordinates", "metric", "lattice", "intrinsic_coordinates",
        "fractional_coordinates", "jacobian_cartesian", "atom_features",
    ):
        value = getattr(expanded, name)
        assert value.dtype == torch.float32
        torch.testing.assert_close(value, getattr(reference, name), rtol=0, atol=0)
    assert pullback.dtype == torch.float32
    torch.testing.assert_close(pullback, reference_pullback, rtol=0, atol=0)


def _tiny_model(dtype):
    config = CrystalAtlasConfig(
        domain=DomainConfig(
            element_vocabulary=(6, 8), dng_element_vocabulary=(6, 8),
            max_distinct_elements=2, max_raw_atoms=4, max_formula_units=3,
            metric_radius=1.0, interaction_cutoff_angstrom=2.5, lattice_chart="ball-v1",
        ),
        numerics=NumericsConfig(ode_steps=2, max_periodic_edges=20000),
        model=ModelConfig(
            composition_width=24, composition_layers=1,
            selector_width=24, selector_layers=1,
            geometry_scalar_width=24, geometry_vector_width=12,
            geometry_layers=1, attention_heads=3, radial_features=6,
        ),
    )
    return CrystalAtlasModel(config, catalogue=MINIMAL_CATALOGUE).to(dtype=dtype).eval()


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_cuda_real_geometry_field_nonorthogonal_anchor_covariance(dtype):
    model = _tiny_model(dtype).cuda()
    layout = _layout()
    field = CoverConditionedGeometryField(
        TorchCoverGeometry(layout).to(device="cuda", dtype=dtype), model.geometry_field
    ).eval()
    z = torch.linspace(-0.13, 0.17, 6, dtype=dtype, device="cuda")
    eta = torch.tensor([0.17, 0.23, 0.31, 0.41, 0.43, 0.47], dtype=dtype, device="cuda")
    action = torch.tensor(
        layout.residual_action_matrix((1, 0, 2)), dtype=dtype, device="cuda"
    )
    basis = torch.tensor(layout.period_basis, dtype=dtype, device="cuda")
    time = torch.tensor(0.37, dtype=dtype, device="cuda")
    original = field(time, z, eta)
    changed = field(time, z, canonicalize_torus(eta @ action, basis))
    torch.testing.assert_close(
        changed.lattice_velocity, original.lattice_velocity, **_neural_tolerance(dtype)
    )
    torch.testing.assert_close(
        changed.intrinsic_velocity, original.intrinsic_velocity @ action,
        **_neural_tolerance(dtype),
    )


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("batched", [False, True])
def test_cuda_actual_cfm_objective_matches_cpu_with_identical_supplied_noise(
    monkeypatch, dtype, batched
):
    import crystalatlas.learned as learned_module

    cpu_model = _tiny_model(dtype)
    # The exact catalogue deliberately contains immutable mapping proxies,
    # which are not pickle/deepcopy payloads. Copy the actual learned state
    # through the model's supported state-dictionary contract instead.
    gpu_model = _tiny_model(dtype)
    gpu_model.load_state_dict(cpu_model.state_dict(), strict=True)
    gpu_model = gpu_model.cuda()
    cpu_state, gpu_state = cpu_model.state_dict(), gpu_model.state_dict()
    assert cpu_state.keys() == gpu_state.keys()
    for name, cpu_tensor in cpu_state.items():
        torch.testing.assert_close(gpu_state[name].cpu(), cpu_tensor, rtol=0, atol=0, msg=name)
    scaffold = CrystalScaffold.from_mapping(
        nu=3, space_group="P1", reduced_composition={6: 1},
        orbits=tuple(
            OrbitAssignment("1a", 6, tuple(Fraction(v, 100) for v in position))
            for position in ((0, 0, 0), (17, 23, 31), (41, 43, 47))
        ),
    )
    layout = cpu_model.layout(ContinuousCover.from_exact_scaffold(scaffold))
    metric = layout.chart.decode_latent(np.linspace(-0.13, 0.17, 6))
    noise_z = torch.tensor([0.1, -0.2, 0.3, -0.4, 0.5, -0.6], dtype=dtype)
    noise_eta = torch.tensor([-0.2, 0.3, -0.1, 0.4, -0.5, 0.6], dtype=dtype)
    real_path = learned_module.conditional_flow_path
    calls = []

    def supplied_base(dimension, period, *, batch_shape=(), dtype=None, device=None, **_):
        assert dimension == 6
        shape = (*batch_shape, dimension)
        return FlowState(
            noise_z.to(device=device, dtype=dtype).expand(shape),
            torch.full(shape, 0.1, device=device, dtype=dtype),
        )

    def supplied_endpoint(z, eta, period, *, sigma_z, sigma_eta, **kwargs):
        assert kwargs.get("torus_action") is None
        return FlowState(
            z + sigma_z * noise_z.to(z),
            canonicalize_torus(eta + sigma_eta * noise_eta.to(eta), period),
        )

    def capture_path(z0, eta0, z1, eta1, time, period, **kwargs):
        action = kwargs["torus_action"]
        expected_action = torch.tensor(
            layout.residual_action_matrix((1, 0, 2)), dtype=dtype, device=z0.device
        ).expand_as(action)
        torch.testing.assert_close(action, expected_action, rtol=0, atol=0)
        result = real_path(z0, eta0, z1, eta1, time, period, **kwargs)
        plain = real_path(z0, eta0, z1, eta1, time, period)
        torch.testing.assert_close(
            result.target_velocity.eta,
            torch.matmul(plain.target_velocity.eta.unsqueeze(-2), action).squeeze(-2),
            **_algebra_tolerance(dtype),
        )
        calls.append(z0.device.type)
        return result

    monkeypatch.setattr(learned_module, "sample_product_base", supplied_base)
    monkeypatch.setattr(learned_module, "sample_dequantized_endpoint", supplied_endpoint)
    monkeypatch.setattr(learned_module, "conditional_flow_path", capture_path)
    results = []
    for model in (cpu_model, gpu_model):
        generator = LearnedCrystalGenerator(model, allow_untrained=True)
        monkeypatch.setattr(
            generator, "_sample_residual_permutation", lambda *a, **k: (1, 0, 2)
        )
        kwargs = dict(time=torch.tensor(0.37, dtype=dtype), sigma_z=0.05, sigma_eta=0.1)
        if batched:
            result = generator.geometry_cfm_objective_batch(
                (scaffold, scaffold), (metric, metric), **kwargs
            )
        else:
            result = generator.geometry_cfm_objective(scaffold, metric, **kwargs)
        result.loss.total.backward()
        results.append(result)
    assert calls == ["cpu", "cuda"]
    torch.testing.assert_close(
        results[1].loss.total.cpu(), results[0].loss.total, **_neural_tolerance(dtype)
    )
    for attribute in ("state", "target_velocity"):
        for block in ("z", "eta"):
            torch.testing.assert_close(
                getattr(getattr(results[1].path, attribute), block).cpu(),
                getattr(getattr(results[0].path, attribute), block),
                **_algebra_tolerance(dtype),
            )
    compared = 0
    for (name, cpu_parameter), (gpu_name, gpu_parameter) in zip(
        cpu_model.named_parameters(), gpu_model.named_parameters(), strict=True
    ):
        assert name == gpu_name
        assert (cpu_parameter.grad is None) == (gpu_parameter.grad is None)
        if cpu_parameter.grad is not None:
            assert torch.isfinite(cpu_parameter.grad).all()
            assert torch.isfinite(gpu_parameter.grad).all()
            tolerance = _neural_tolerance(dtype) if dtype == torch.float64 else dict(
                rtol=5.0e-4, atol=5.0e-5
            )
            torch.testing.assert_close(
                gpu_parameter.grad.cpu(), cpu_parameter.grad, msg=name, **tolerance
            )
            compared += 1
    assert compared > 0


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_cuda_transformer_rigid_motion_and_atom_permutation_contract(dtype):
    model = InvariantPeriodicTransformer(
        InvariantPeriodicTransformerConfig(
            input_scalar_dim=3, model_dim=12, pair_dim=8, num_heads=3,
            num_layers=1, reciprocal_mode_radius=1, direct_harmonics=2,
            output_vector_dim=3,
        )
    ).to(device="cuda", dtype=dtype).eval()
    features = torch.tensor(
        [[0.2, -0.7, 0.4], [0.8, 0.1, -0.9], [-0.4, 0.6, 0.3]],
        dtype=dtype, device="cuda",
    )
    fractional = torch.tensor(
        [[0.07, 0.11, 0.19], [0.31, 0.23, 0.63], [0.76, 0.41, 0.29]],
        dtype=dtype, device="cuda",
    )
    lattice = torch.tensor(
        [[3.7, 0.2, -0.1], [0.6, 3.2, 0.3], [0.2, -0.4, 3.5]],
        dtype=dtype, device="cuda",
    )
    rotation = torch.tensor(
        [[0.6, -0.8, 0], [0.8, 0.6, 0], [0, 0, 1]], dtype=dtype, device="cuda"
    )
    permutation = torch.tensor([2, 0, 1], device="cuda")
    origin = torch.tensor([0.13, -0.22, 0.37], dtype=dtype, device="cuda")
    image_labels = torch.tensor(
        [[2, -3, 1], [-1, 2, 0], [0, 1, -2]], dtype=dtype, device="cuda"
    )
    reference = model(features, fractional, lattice)
    changed = model(
        features[permutation], (fractional + origin + image_labels)[permutation],
        lattice @ rotation,
    )
    torch.testing.assert_close(
        changed.scalar_channels, reference.scalar_channels[permutation],
        **_neural_tolerance(dtype),
    )
    torch.testing.assert_close(
        changed.pooled_scalars, reference.pooled_scalars, **_neural_tolerance(dtype)
    )
    torch.testing.assert_close(
        changed.vector_channels, reference.vector_channels[permutation] @ rotation,
        **_neural_tolerance(dtype),
    )
