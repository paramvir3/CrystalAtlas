from __future__ import annotations

import weakref

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from crystalatlas.composition import Composition  # noqa: E402
from crystalatlas.errors import DomainError  # noqa: E402
from crystalatlas.geometry import (  # noqa: E402
    ContinuousCover,
    ContinuousGeometry,
    IntrinsicCoverLayout,
    OrbitType,
)
from crystalatlas.nn.bridge import (  # noqa: E402
    CoverConditionedGeometryField,
    TorchCoverGeometry,
)
from crystalatlas.nn.field import IntrinsicFieldConfig, IntrinsicGeometryField  # noqa: E402

pytestmark = pytest.mark.ml


def _p1_layout() -> IntrinsicCoverLayout:
    cover = ContinuousCover(
        nu=1,
        space_group="P1",
        composition=Composition(((8, 1), (14, 1))),
        orbits=(OrbitType("1a", 8), OrbitType("1a", 14)),
    )
    # This module tests the ball-v1 bridge (loewner-v2: tests/test_loewner_chart.py).
    return IntrinsicCoverLayout(cover, metric_bound=2.0, lattice_chart="ball-v1")


@pytest.mark.parametrize("atoms", (8, 16, 40))
@pytest.mark.parametrize("dtype", (torch.float32, torch.float64))
def test_chart_island_preserves_original_large_p1_decoder_tolerances(atoms, dtype) -> None:
    cover = ContinuousCover(
        nu=1,
        space_group="P1",
        composition=Composition.from_counts({6: atoms - 1, 7: 1}),
        orbits=(OrbitType("1a", 6),) * (atoms - 1) + (OrbitType("1a", 7),),
    )
    layout = IntrinsicCoverLayout(cover)
    bridge = TorchCoverGeometry(layout).to(dtype=dtype)
    z = (torch.arange(layout.lattice_dimension, dtype=dtype) - 3) / 32
    eta = ((17 * torch.arange(layout.intrinsic_dimension, dtype=dtype)) % 251 + 1) / 256
    actual = bridge(z, eta)
    expected = layout.decode(ContinuousGeometry(z.double().numpy(), eta.double().numpy()))
    tolerance = 4e-6 if dtype == torch.float32 else 2e-12
    for value, reference in (
        (actual.metric, expected.metric),
        (actual.lattice, expected.lattice),
        (actual.fractional_coordinates, expected.fractional_positions),
        (actual.jacobian_cartesian, expected.jacobian_cartesian),
    ):
        assert value.dtype == dtype
        np.testing.assert_allclose(
            value.detach().double().numpy(), reference, rtol=tolerance, atol=tolerance
        )


def test_canonical_chart_constants_survive_dtype_and_state_round_trips() -> None:
    layout = _p1_layout()
    bridge = TorchCoverGeometry(layout)
    expected_reference = torch.tensor(layout.chart.sqrt_reference, dtype=torch.float64)
    expected_basis = torch.tensor(np.stack(layout.chart.basis), dtype=torch.float64)
    # P1 already contains non-dyadic reference and invariant-basis constants.
    assert not torch.equal(expected_reference, expected_reference.float().double())
    assert not torch.equal(expected_basis, expected_basis.float().double())
    for dtype in (torch.float32, torch.float16, torch.bfloat16, torch.float64):
        bridge.to(dtype=dtype)
        assert bridge.sqrt_reference.dtype == bridge.invariant_basis.dtype == torch.float64
        torch.testing.assert_close(bridge.sqrt_reference, expected_reference, rtol=0, atol=0)
        torch.testing.assert_close(bridge.invariant_basis, expected_basis, rtol=0, atol=0)
        for key in ("_sqrt_reference_bits", "_invariant_basis_bits"):
            assert bridge.state_dict()[key].dtype == torch.int64
    restored = TorchCoverGeometry(layout).float()
    restored.load_state_dict(bridge.float().state_dict())
    restored.double()
    z = torch.tensor([0.1] * layout.lattice_dimension, dtype=torch.float64)
    direct = TorchCoverGeometry(layout).double()._metric_from_latent(z)[1]
    torch.testing.assert_close(restored._metric_from_latent(z)[1], direct, rtol=0, atol=0)


def test_chart_state_hook_does_not_retain_ephemeral_bridge() -> None:
    bridge = TorchCoverGeometry(_p1_layout())
    reference = weakref.ref(bridge)
    del bridge
    assert reference() is None


@pytest.mark.parametrize("dtype", (torch.float32, torch.float64))
@pytest.mark.parametrize("space_group", ("P1", "P-1"))
def test_zero_intrinsic_dimension_retains_master_dtype_and_empty_jacobian(
    dtype,
    space_group,
) -> None:
    cover = ContinuousCover(
        nu=1,
        space_group=space_group,
        composition=Composition.from_counts({6: 1}),
        orbits=(OrbitType("1a", 6),),
    )
    layout = IntrinsicCoverLayout(cover)
    assert layout.intrinsic_dimension == 0
    bridge = TorchCoverGeometry(layout).to(dtype=dtype)
    z = torch.zeros(layout.lattice_dimension, dtype=dtype, requires_grad=True)
    eta = torch.empty(0, dtype=dtype, requires_grad=True)
    expanded = bridge(z, eta)
    assert expanded.intrinsic_coordinates.shape == (0,)
    assert expanded.jacobian_cartesian.shape == (1, 3, 0)
    assert expanded.jacobian_cartesian.dtype == expanded.metric.dtype == dtype
    assert expanded.fractional_coordinates.shape == (1, 3)
    assert expanded.fractional_coordinates.dtype == dtype
    expanded.metric.sum().backward()
    assert z.grad is not None and z.grad.dtype == dtype and torch.isfinite(z.grad).all()
    expected = layout.chart.decode_latent(z.detach().double().numpy())
    tolerance = 4e-6 if dtype == torch.float32 else 2e-12
    np.testing.assert_allclose(
        expanded.metric.detach().double().numpy(), expected, rtol=tolerance, atol=tolerance
    )


@pytest.mark.parametrize("dtype", (torch.float16, torch.bfloat16, torch.int64, torch.complex64))
def test_chart_rejects_nonmaster_state_dtypes(dtype) -> None:
    bridge = TorchCoverGeometry(_p1_layout())
    with pytest.raises(ValueError, match="float32 or float64 master precision"):
        bridge._metric_from_latent(torch.zeros(bridge.lattice_dimension, dtype=dtype))


@pytest.mark.parametrize("corruption", ("rounded-dtype", "changed-bits"))
def test_chart_state_rejects_changed_canonical_constant_bits(corruption) -> None:
    bridge = TorchCoverGeometry(_p1_layout()).float()
    state = {key: value.clone() for key, value in bridge.state_dict().items()}
    original = bridge.sqrt_reference.clone()
    key = "_sqrt_reference_bits"
    if corruption == "rounded-dtype":
        state[key] = state[key].float()
    else:
        state[key].view(-1)[0] += 1
    with pytest.raises(RuntimeError, match="changes canonical float64 chart constant"):
        bridge.load_state_dict(state)
    torch.testing.assert_close(bridge.sqrt_reference, original, rtol=0, atol=0)


@pytest.mark.parametrize(
    "diagonal", ((0.0, 0.0, 0.0), (0.25, 0.25, -0.1), (0.25, 0.25000001, -0.1))
)
def test_chart_gradient_and_double_gradient_at_repeated_eigenvalues(diagonal) -> None:
    layout = _p1_layout()
    bridge = TorchCoverGeometry(layout).double()
    algebra = np.diag(diagonal)
    bounded = np.asarray([np.sum(algebra * basis) for basis in layout.chart.basis])
    z = torch.tensor(
        layout.chart.unbounded_latent(bounded), dtype=torch.float64, requires_grad=True
    )
    reference_scale = float(np.linalg.norm(layout.chart.reference_metric))

    def normalized_metric(value):
        return bridge._metric_from_latent(value)[1] / reference_scale

    assert torch.autograd.gradcheck(normalized_metric, (z,), eps=1e-6, atol=2e-7, rtol=1e-5)
    assert torch.autograd.gradgradcheck(normalized_metric, (z,), eps=1e-6, atol=2e-7, rtol=1e-5)


@pytest.mark.parametrize("dtype", (torch.float32, torch.float64))
def test_chart_jacobian_and_pullback_at_origin_have_analytic_values(dtype) -> None:
    layout = _p1_layout()
    bridge = TorchCoverGeometry(layout).to(dtype=dtype)
    z = torch.zeros(layout.lattice_dimension, dtype=dtype, requires_grad=True)
    reference = bridge.sqrt_reference
    basis = bridge.invariant_basis
    expected_jacobian = torch.stack(
        [layout.chart.bound * reference @ matrix @ reference for matrix in basis], dim=-1
    )
    jacobian = torch.autograd.functional.jacobian(
        lambda value: bridge._metric_from_latent(value)[1], z
    )
    # dG/dz_i = bound*S*E_i*S and tr(G^-1 U_i G^-1 U_j) = bound² delta_ij.
    tolerance = 3e-6 if dtype == torch.float32 else 2e-11
    torch.testing.assert_close(
        jacobian.double(), expected_jacobian, rtol=tolerance, atol=tolerance
    )
    pullback = bridge.lattice_pullback_metric(z, create_graph=True)
    assert pullback.dtype == dtype and pullback.requires_grad
    torch.testing.assert_close(
        pullback,
        layout.chart.bound**2 * torch.eye(layout.lattice_dimension, dtype=dtype),
        rtol=tolerance,
        atol=tolerance,
    )
    derivative = torch.autograd.grad(pullback.square().sum(), z)[0]
    assert derivative.dtype == dtype and torch.isfinite(derivative).all()


def test_float32_chart_island_ignores_enclosing_neural_autocast(monkeypatch) -> None:
    bridge = TorchCoverGeometry(_p1_layout()).float()
    z = torch.tensor([0.1] * bridge.lattice_dimension, dtype=torch.float32, requires_grad=True)
    eta = torch.tensor([0.23, 0.31, 0.47], dtype=torch.float32, requires_grad=True)
    observed_dtypes = []
    real_exp = torch.matrix_exp

    def checked_exp(value):
        observed_dtypes.append(value.dtype)
        assert value.dtype == torch.float64
        return real_exp(value)

    monkeypatch.setattr(torch, "matrix_exp", checked_exp)
    baseline = bridge(z, eta)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        actual = bridge(z, eta)
    assert observed_dtypes == [torch.float64, torch.float64]
    for name in (
        "bounded_lattice_coordinates",
        "metric",
        "lattice",
        "intrinsic_coordinates",
        "fractional_coordinates",
        "atom_features",
        "jacobian_cartesian",
    ):
        value = getattr(actual, name)
        assert value.dtype == torch.float32
        torch.testing.assert_close(value, getattr(baseline, name), rtol=0, atol=0)
    (actual.metric.square().sum() + actual.fractional_coordinates.square().sum()).backward()
    assert z.grad is not None and z.grad.dtype == torch.float32 and torch.isfinite(z.grad).all()
    assert (
        eta.grad is not None
        and eta.grad.dtype == torch.float32
        and torch.isfinite(eta.grad).all()
    )


def test_torch_expansion_matches_numpy_decoder_and_is_differentiable() -> None:
    layout = _p1_layout()
    bridge = TorchCoverGeometry(layout).double()
    z = torch.tensor([0.1] * layout.lattice_dimension, dtype=torch.float64, requires_grad=True)
    eta = torch.tensor([0.2, 0.3, 0.4], dtype=torch.float64, requires_grad=True)
    output = bridge(z, eta)
    expected = layout.decode(ContinuousGeometry(z.detach().numpy(), eta.detach().numpy()))
    torch.testing.assert_close(
        output.metric,
        torch.from_numpy(expected.metric.copy()),
        rtol=2.0e-12,
        atol=2.0e-12,
    )
    torch.testing.assert_close(
        output.fractional_coordinates,
        torch.from_numpy(expected.fractional_positions.copy()),
        rtol=2.0e-12,
        atol=2.0e-12,
    )
    objective = output.metric.square().sum() + output.fractional_coordinates.square().sum()
    objective.backward()
    assert z.grad is not None and torch.isfinite(z.grad).all()
    assert eta.grad is not None and torch.isfinite(eta.grad).all()
    assert eta.grad.abs().max().item() > 0.0

    pullback = bridge.lattice_pullback_metric(z.detach())
    assert pullback.shape == (layout.lattice_dimension, layout.lattice_dimension)
    torch.testing.assert_close(pullback, pullback.transpose(0, 1), rtol=1.0e-12, atol=1.0e-12)
    assert torch.linalg.eigvalsh(pullback).min().item() > 0.0


def test_cover_bound_field_closes_expand_perceive_reduce_path() -> None:
    torch.manual_seed(43)
    layout = _p1_layout()
    bridge = TorchCoverGeometry(layout).double()
    config = IntrinsicFieldConfig(
        atom_feature_dim=bridge.atom_feature_dim,
        composition_dim=12,
        species_embedding_dim=8,
        max_lattice_dim=6,
        num_time_frequencies=2,
        scalar_dim=24,
        vector_dim=12,
        num_heads=3,
        num_layers=2,
        radial_basis_dim=6,
        cutoff=3.0,
    )
    neural = IntrinsicGeometryField(config).double()
    field = CoverConditionedGeometryField(bridge, neural)
    z = torch.zeros(layout.lattice_dimension, dtype=torch.float64, requires_grad=True)
    eta = torch.tensor([0.23, 0.31, 0.47], dtype=torch.float64, requires_grad=True)
    output = field(torch.tensor(0.3, dtype=torch.float64), z, eta)
    assert output.lattice_velocity.shape == z.shape
    assert output.intrinsic_velocity.shape == eta.shape
    loss = output.lattice_velocity.square().sum() + output.intrinsic_velocity.square().sum()
    loss.backward()
    assert z.grad is not None and torch.isfinite(z.grad).all()
    assert eta.grad is not None and torch.isfinite(eta.grad).all()


def test_local_edge_budgets_do_not_constrain_dense_backend_support() -> None:
    layout = _p1_layout()
    bridge = TorchCoverGeometry(layout).double()
    common = dict(
        atom_feature_dim=bridge.atom_feature_dim,
        composition_dim=8,
        species_embedding_dim=4,
        max_lattice_dim=6,
        num_time_frequencies=1,
        scalar_dim=8,
        pair_dim=8,
        vector_dim=4,
        num_heads=1,
        num_layers=1,
        radial_basis_dim=2,
        cutoff=3.0,
        max_candidates_per_pair=1,
        max_periodic_edges=1,
    )
    dense = IntrinsicGeometryField(
        IntrinsicFieldConfig(backbone="invariant_global", **common)
    ).double()
    CoverConditionedGeometryField(bridge, dense)

    local = IntrinsicGeometryField(
        IntrinsicFieldConfig(backbone="equivariant_local", **common)
    ).double()
    with pytest.raises(DomainError, match="max_candidates_per_pair"):
        CoverConditionedGeometryField(bridge, local)


def test_field_is_covariant_to_identical_p1_anchor_exchange() -> None:
    torch.manual_seed(47)
    cover = ContinuousCover(
        nu=2,
        space_group="P1",
        composition=Composition(((8, 1),)),
        orbits=(OrbitType("1a", 8), OrbitType("1a", 8)),
    )
    layout = IntrinsicCoverLayout(cover, metric_bound=1.0, lattice_chart="ball-v1")
    bridge = TorchCoverGeometry(layout).double()
    neural = (
        IntrinsicGeometryField(
            IntrinsicFieldConfig(
                atom_feature_dim=bridge.atom_feature_dim,
                composition_dim=12,
                species_embedding_dim=8,
                max_lattice_dim=6,
                num_time_frequencies=2,
                scalar_dim=24,
                vector_dim=12,
                num_heads=3,
                num_layers=2,
                radial_basis_dim=6,
                cutoff=3.0,
            )
        )
        .double()
        .eval()
    )
    field = CoverConditionedGeometryField(bridge, neural).eval()
    z = torch.zeros(layout.lattice_dimension, dtype=torch.float64)
    eta = torch.tensor([0.23, 0.31, 0.47], dtype=torch.float64)
    action = torch.as_tensor(layout.residual_action_matrix((1, 0)), dtype=torch.float64)
    transformed_eta = eta @ action
    first = field(torch.tensor(0.37, dtype=torch.float64), z, eta)
    second = field(torch.tensor(0.37, dtype=torch.float64), z, transformed_eta)
    torch.testing.assert_close(
        second.lattice_velocity,
        first.lattice_velocity,
        rtol=2.0e-10,
        atol=2.0e-10,
    )
    torch.testing.assert_close(
        second.intrinsic_velocity,
        first.intrinsic_velocity @ action,
        rtol=2.0e-9,
        atol=2.0e-9,
    )


def test_field_is_covariant_to_p_minus_one_representative_inversion() -> None:
    torch.manual_seed(53)
    cover = ContinuousCover(
        nu=2,
        space_group="P-1",
        composition=Composition(((8, 1),)),
        orbits=(OrbitType("2i", 8),),
    )
    layout = IntrinsicCoverLayout(cover, metric_bound=1.0, lattice_chart="ball-v1")
    bridge = TorchCoverGeometry(layout).double()
    neural = (
        IntrinsicGeometryField(
            IntrinsicFieldConfig(
                atom_feature_dim=bridge.atom_feature_dim,
                composition_dim=12,
                species_embedding_dim=8,
                max_lattice_dim=6,
                num_time_frequencies=2,
                scalar_dim=24,
                vector_dim=12,
                num_heads=3,
                num_layers=2,
                radial_basis_dim=6,
                cutoff=3.0,
            )
        )
        .double()
        .eval()
    )
    field = CoverConditionedGeometryField(bridge, neural).eval()
    z = torch.zeros(layout.lattice_dimension, dtype=torch.float64)
    eta = torch.tensor([0.17, 0.29, 0.41], dtype=torch.float64, requires_grad=True)
    action = torch.as_tensor(
        layout.residual_action_matrix((0,), orbit_signs=(-1,)),
        dtype=torch.float64,
    )
    first = field(torch.tensor(0.43, dtype=torch.float64), z, eta)
    second = field(torch.tensor(0.43, dtype=torch.float64), z, eta @ action)
    torch.testing.assert_close(
        second.lattice_velocity,
        first.lattice_velocity,
        rtol=2.0e-10,
        atol=2.0e-10,
    )
    torch.testing.assert_close(
        second.intrinsic_velocity,
        first.intrinsic_velocity @ action,
        rtol=2.0e-9,
        atol=2.0e-9,
    )
    expanded = bridge(z, eta)
    normal = torch.einsum(
        "nci,ncj->ij",
        expanded.jacobian_cartesian - expanded.jacobian_cartesian.mean(dim=0, keepdim=True),
        expanded.jacobian_cartesian - expanded.jacobian_cartesian.mean(dim=0, keepdim=True),
    )
    assert torch.linalg.eigvalsh(normal).min().item() > 1.0e-8
    assert first.intrinsic_velocity.abs().max().item() > 1.0e-10
    first.intrinsic_velocity.square().sum().backward()
    assert eta.grad is not None and torch.isfinite(eta.grad).all()
    assert eta.grad.abs().max().item() > 1.0e-12


def _batched_cover(space_group: str) -> ContinuousCover:
    if space_group == "P1":
        return ContinuousCover(
            nu=1,
            space_group="P1",
            composition=Composition(((8, 1), (14, 1))),
            orbits=(OrbitType("1a", 8), OrbitType("1a", 14)),
        )
    return ContinuousCover(
        nu=2,
        space_group="P-1",
        composition=Composition(((8, 1),)),
        orbits=(OrbitType("2i", 8),),
    )


@pytest.mark.parametrize("lattice_chart", ("ball-v1", "loewner-v2"))
@pytest.mark.parametrize("space_group", ("P1", "P-1"))
@pytest.mark.parametrize("dtype", (torch.float32, torch.float64))
def test_batched_expansion_equals_stacked_per_record_expansion(
    lattice_chart, space_group, dtype
) -> None:
    """``forward_batch`` and ``lattice_pullback_metric_batch`` equal per-record calls.

    The gradients of a smooth scalar agree too, which checks the batched Loewner
    adjoint.  Row 0 is the chart origin, where every eigenvalue is repeated.
    """

    layout = IntrinsicCoverLayout(_batched_cover(space_group), lattice_chart=lattice_chart)
    bridge = TorchCoverGeometry(layout).to(dtype=dtype)
    generator = torch.Generator().manual_seed(5)
    z_values = 1.5 * torch.randn(7, layout.lattice_dimension, generator=generator)
    z_values[0] = 0.0
    eta_values = 3.0 * torch.randn(7, layout.intrinsic_dimension, generator=generator)
    tolerance = 1.0e-12 if dtype == torch.float64 else 5.0e-6

    def loss(expanded) -> torch.Tensor:  # type: ignore[no-untyped-def]
        return (
            expanded.metric.square().sum()
            + torch.sin(2.0 * torch.pi * expanded.fractional_coordinates).sum()
            + expanded.jacobian_cartesian.square().sum()
            + expanded.bounded_lattice_coordinates.square().sum()
        )

    z = z_values.to(dtype=dtype).requires_grad_(True)
    eta = eta_values.to(dtype=dtype).requires_grad_(True)
    batched = bridge.forward_batch(z, eta)
    loss(batched).backward()
    z_single = z_values.to(dtype=dtype).requires_grad_(True)
    eta_single = eta_values.to(dtype=dtype).requires_grad_(True)
    singles = [bridge(z_single[index], eta_single[index]) for index in range(7)]
    sum(loss(item) for item in singles).backward()  # type: ignore[union-attr]
    for name in (
        "bounded_lattice_coordinates",
        "metric",
        "lattice",
        "intrinsic_coordinates",
        "fractional_coordinates",
        "atom_features",
        "jacobian_cartesian",
    ):
        expected = torch.stack([getattr(item, name) for item in singles]).detach()
        actual = getattr(batched, name).detach()
        assert actual.dtype == expected.dtype == dtype, name
        scale = max(1.0, float(expected.abs().max()))
        torch.testing.assert_close(
            actual, expected, rtol=tolerance, atol=tolerance * scale, msg=name
        )
    assert torch.equal(batched.atomic_numbers, torch.stack([s.atomic_numbers for s in singles]))
    assert torch.equal(batched.lattice_commutant_basis, singles[0].lattice_commutant_basis)
    for actual, expected in ((z.grad, z_single.grad), (eta.grad, eta_single.grad)):
        assert actual is not None and expected is not None
        scale = max(1.0, float(expected.abs().max()))
        torch.testing.assert_close(
            actual, expected, rtol=10 * tolerance, atol=10 * tolerance * scale
        )

    pullbacks = bridge.lattice_pullback_metric_batch(z.detach())
    expected = torch.stack([bridge.lattice_pullback_metric(item) for item in z.detach()])
    assert pullbacks.dtype == dtype and not pullbacks.requires_grad
    scale = float(expected.abs().max())
    torch.testing.assert_close(pullbacks, expected, rtol=tolerance, atol=tolerance * scale)
    with pytest.raises(ValueError, match="shape"):
        bridge.forward_batch(z[0], eta[0])
    with pytest.raises(ValueError, match="shape"):
        bridge.lattice_pullback_metric_batch(z[0])
