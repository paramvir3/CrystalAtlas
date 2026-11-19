from __future__ import annotations

import itertools

import pytest

torch = pytest.importorskip("torch")

from crystalatlas.nn.equivariant_transformer import (  # noqa: E402
    PeriodicImageBudgetError,
    PeriodicSE3Transformer,
    PeriodicTransformerConfig,
    enumerate_periodic_edges,
    oriented_triple_product,
    regular_solid_harmonics,
    smooth_whole_edge_cutoff,
)

pytestmark = pytest.mark.ml


def _proper_rotation(dtype: torch.dtype) -> torch.Tensor:
    matrix = torch.tensor(
        [[0.3, -0.7, 0.2], [0.4, 0.1, -0.8], [0.6, 0.5, 0.3]], dtype=dtype
    )
    q, _ = torch.linalg.qr(matrix)
    if torch.linalg.det(q) < 0:
        q[:, 0] *= -1
    return q


def test_cutoff_value_and_first_derivative_vanish() -> None:
    squared = torch.tensor(4.0, dtype=torch.float64, requires_grad=True)
    value = smooth_whole_edge_cutoff(squared, 4.0)
    derivative = torch.autograd.grad(value, squared)[0]
    assert value.item() == 0.0
    assert derivative.item() == 0.0


def test_regular_features_are_collision_safe_and_oriented_path_is_so3_only() -> None:
    collision = torch.zeros(3, dtype=torch.float64, requires_grad=True)
    features = regular_solid_harmonics(collision, max_degree=2)
    assert torch.isfinite(features).all()
    assert features[0].item() == 1.0
    assert torch.count_nonzero(features[1:]).item() == 0
    features.sum().backward()
    assert torch.isfinite(collision.grad).all()

    first = torch.tensor([1.0, 2.0, -0.5], dtype=torch.float64)
    second = torch.tensor([-0.3, 0.4, 1.2], dtype=torch.float64)
    third = torch.tensor([0.7, -1.1, 0.2], dtype=torch.float64)
    rotation = _proper_rotation(torch.float64)
    reflected = torch.diag(torch.tensor([-1.0, 1.0, 1.0], dtype=torch.float64))
    value = oriented_triple_product(first, second, third)
    rotated = oriented_triple_product(first @ rotation, second @ rotation, third @ rotation)
    mirror = oriented_triple_product(first @ reflected, second @ reflected, third @ reflected)
    torch.testing.assert_close(rotated, value, rtol=1.0e-12, atol=1.0e-12)
    torch.testing.assert_close(mirror, -value, rtol=1.0e-12, atol=1.0e-12)


def test_periodic_enumeration_is_complete_for_a_skew_cell() -> None:
    dtype = torch.float64
    lattice = torch.tensor(
        [[2.1, 0.0, 0.0], [1.7, 0.55, 0.0], [0.2, 0.15, 1.8]], dtype=dtype
    )
    fractional = torch.tensor([[0.0, 0.0, 0.0], [0.37, 0.21, 0.46]], dtype=dtype)
    cutoff = 1.15
    edges = enumerate_periodic_edges(fractional, lattice, cutoff)
    actual = {
        tuple(int(x) for x in edges.shifts[index].tolist())
        for index in range(edges.num_edges)
        if edges.destination[index].item() == 0 and edges.source[index].item() == 1
    }
    delta = fractional[1] - fractional[0]
    expected = set()
    for shift in itertools.product(range(-6, 7), repeat=3):
        displacement = (delta + torch.tensor(shift, dtype=dtype)) @ lattice
        if displacement.square().sum().item() < cutoff**2:
            expected.add(shift)
    assert actual == expected
    # The central self edge is absent, but close noncentral self images survive.
    assert not any(
        edges.destination[i] == edges.source[i]
        and torch.all(edges.shifts[i] == 0)
        for i in range(edges.num_edges)
    )


def test_periodic_enumeration_is_complete_for_a_strongly_skewed_cell() -> None:
    """F7 skew-cell exhaustive oracle beyond any small fixed image cube (mutation 14).

    With a_2 - a_1 = (-0.15, 0.2, 0), the vector 4(a_2 - a_1) has length 1.0, inside
    r_c = 1.15.  A complete enumeration therefore needs shifts with |n_j| = 4.
    """

    dtype = torch.float64
    lattice = torch.tensor(
        [[2.1, 0.0, 0.0], [1.95, 0.2, 0.0], [0.0, 0.0, 2.0]], dtype=dtype
    )
    fractional = torch.tensor([[0.0, 0.0, 0.0], [0.31, 0.62, 0.5]], dtype=dtype)
    cutoff = 1.15
    edges = enumerate_periodic_edges(fractional, lattice, cutoff)
    for destination, source in ((0, 0), (0, 1), (1, 0), (1, 1)):
        actual = {
            tuple(int(x) for x in edges.shifts[index].tolist())
            for index in range(edges.num_edges)
            if edges.destination[index].item() == destination
            and edges.source[index].item() == source
        }
        delta = fractional[source] - fractional[destination]
        expected = set()
        for shift in itertools.product(range(-12, 13), repeat=3):
            if destination == source and not any(shift):
                continue
            displacement = (delta + torch.tensor(shift, dtype=dtype)) @ lattice
            if displacement.square().sum().item() < cutoff**2:
                expected.add(shift)
        assert actual == expected, (destination, source)
    assert max(abs(value) for value in edges.shifts.flatten().tolist()) >= 4


def test_periodic_enumeration_preserves_public_shifts_after_cell_relabelling() -> None:
    dtype = torch.float64
    lattice = torch.tensor(
        [[2.3, 0.0, 0.0], [0.8, 1.7, 0.0], [-0.4, 0.3, 2.1]], dtype=dtype
    )
    fractional = torch.tensor(
        [[0.13, 0.27, 0.41], [0.77, 0.04, 0.58], [0.32, 0.91, 0.16]],
        dtype=dtype,
    )
    cell_offsets = torch.tensor([[12, -9, 3], [-7, 5, 11], [4, 8, -6]])
    original = enumerate_periodic_edges(fractional, lattice, 2.4)
    relabelled = enumerate_periodic_edges(
        fractional + cell_offsets.to(dtype), lattice, 2.4
    )

    torch.testing.assert_close(relabelled.destination, original.destination)
    torch.testing.assert_close(relabelled.source, original.source)
    offset_delta = (
        cell_offsets[original.source] - cell_offsets[original.destination]
    )
    torch.testing.assert_close(relabelled.shifts, original.shifts - offset_delta)
    torch.testing.assert_close(
        relabelled.displacement, original.displacement, rtol=1.0e-13, atol=1.0e-13
    )
    torch.testing.assert_close(
        relabelled.squared_distance,
        original.squared_distance,
        rtol=1.0e-13,
        atol=1.0e-13,
    )


def test_periodic_enumeration_keeps_value_gradients() -> None:
    dtype = torch.float64
    lattice = torch.tensor(
        [[2.8, 0.0, 0.0], [0.5, 2.4, 0.0], [0.2, -0.3, 2.6]],
        dtype=dtype,
        requires_grad=True,
    )
    fractional = torch.tensor(
        [[0.11, 0.23, 0.37], [0.63, 0.42, 0.79]],
        dtype=dtype,
        requires_grad=True,
    )
    edges = enumerate_periodic_edges(fractional, lattice, 2.2)
    weights = torch.linspace(
        0.5, 1.5, edges.num_edges, dtype=dtype, device=lattice.device
    )
    objective = (weights * edges.squared_distance).sum()
    objective.backward()

    assert fractional.grad is not None
    assert lattice.grad is not None
    assert torch.isfinite(fractional.grad).all()
    assert torch.isfinite(lattice.grad).all()


def test_periodic_enumeration_fails_before_unsafe_allocation() -> None:
    lattice = torch.eye(3, dtype=torch.float64) * 0.1
    fractional = torch.zeros(1, 3, dtype=torch.float64)
    with pytest.raises(PeriodicImageBudgetError, match="max_candidates_per_pair"):
        enumerate_periodic_edges(
            fractional,
            lattice,
            6.0,
            max_candidates_per_pair=1_000,
            max_edges=10_000,
        )


def test_float32_image_box_rounding_cannot_omit_an_in_cutoff_image() -> None:
    """Topology is decided in conservative FP64, not rounded inward in FP32."""

    lattice = torch.diag(
        torch.tensor([5.956221580505371, 10.0, 10.0], dtype=torch.float32)
    )
    fractional = torch.tensor(
        [[0.0, 0.0, 0.0], [0.007350000087171793, 0.0, 0.0]],
        dtype=torch.float32,
    )
    edges = enumerate_periodic_edges(fractional, lattice, 6.0)
    pair = (edges.destination == 0) & (edges.source == 1)
    shifts = {tuple(value) for value in edges.shifts[pair].tolist()}

    # In FP64 evaluation of the represented FP32 inputs this image has
    # d^2=35.9999977157 < 36.  Computing ceil/floor bounds in FP32 used to
    # round the upper endpoint inward and omit it.
    assert (1, 0, 0) in shifts


def _small_transformer(vector_normalization: str = "rms-v1") -> PeriodicSE3Transformer:
    config = PeriodicTransformerConfig(
        input_scalar_dim=7,
        scalar_dim=24,
        vector_dim=12,
        num_heads=3,
        num_layers=3,
        radial_basis_dim=8,
        cutoff=3.2,
        reference_length=1.0,
        vector_normalization=vector_normalization,
    )
    return PeriodicSE3Transformer(config).double().eval()


@pytest.mark.parametrize("normalization", ["rms-v1", "soft-v2"])
def test_transformer_is_proper_se3_equivariant_and_chirality_sensitive(
    normalization: str,
) -> None:
    torch.manual_seed(23)
    model = _small_transformer(normalization)
    lattice = torch.tensor(
        [[3.8, 0.0, 0.0], [0.4, 3.5, 0.0], [0.2, 0.3, 3.3]], dtype=torch.float64
    )
    fractional = torch.tensor(
        [[0.07, 0.13, 0.19], [0.33, 0.21, 0.61], [0.72, 0.46, 0.28], [0.55, 0.82, 0.77]],
        dtype=torch.float64,
    )
    scalars = torch.randn(4, 7, dtype=torch.float64)
    rotation = _proper_rotation(torch.float64)
    original = model(scalars, fractional, lattice)
    rotated = model(scalars, fractional, lattice @ rotation)
    torch.testing.assert_close(
        rotated.scalar_channels,
        original.scalar_channels,
        rtol=2.0e-9,
        atol=2.0e-9,
    )
    torch.testing.assert_close(
        rotated.vector_channels,
        original.vector_channels @ rotation,
        rtol=2.0e-9,
        atol=2.0e-9,
    )

    # Mirror the physical structure, then restore a right-handed row lattice
    # using an orientation-reversing integer basis description.  An O(3)
    # network would return exactly mirrored polar vectors and equal scalars.
    reflection = torch.diag(torch.tensor([-1.0, 1.0, 1.0], dtype=torch.float64))
    basis = reflection
    mirror_fractional = fractional @ basis
    mirror_lattice = basis @ lattice @ reflection
    mirrored = model(scalars, mirror_fractional, mirror_lattice)
    scalar_difference = (mirrored.scalar_channels - original.scalar_channels).abs().max()
    vector_difference = (
        mirrored.vector_channels - original.vector_channels @ reflection
    ).abs().max()
    assert max(scalar_difference.item(), vector_difference.item()) > 1.0e-7


def test_transformer_is_finite_at_distinct_atom_collision() -> None:
    torch.manual_seed(3)
    model = _small_transformer()
    lattice = torch.eye(3, dtype=torch.float64) * 4.0
    fractional = torch.tensor(
        [[0.2, 0.2, 0.2], [0.2, 0.2, 0.2]], dtype=torch.float64
    )
    scalars = torch.randn(2, 7, dtype=torch.float64, requires_grad=True)
    output = model(scalars, fractional, lattice)
    objective = output.scalar_channels.square().sum() + output.vector_channels.square().sum()
    objective.backward()
    assert torch.isfinite(output.scalar_channels).all()
    assert torch.isfinite(output.vector_channels).all()
    assert torch.isfinite(scalars.grad).all()


@pytest.mark.xfail(
    strict=True,
    reason=(
        "documented F6 defect of the legacy rms-v1 vector normalization: "
        "v / sqrt(<v^2> + eps), with eps = 1e-8 (the field does not forward its own "
        "1e-4), has Jacobian I/sqrt(eps) at v = 0. Cutoff-gated vanishing vectors are "
        "amplified about 1e8 per block, and the output jumps by O(1) at r_c. The "
        "opt-in soft-v2 normalization fixes this (C1, gap ~ delta^2); see "
        "test_soft_normalized_local_transformer_is_c1_across_the_cutoff."
    ),
)
def test_local_transformer_is_continuous_across_the_cutoff() -> None:
    """F6 cutoff value test (theory 4.9, mutation 12).

    An edge that appears just inside r_c carries weight (1 - r^2/r_c^2)^2, about
    4e-15 here.  The output must therefore not jump when an atom crosses the
    cutoff.
    """

    torch.manual_seed(31)
    model = _small_transformer()  # cutoff 3.2
    lattice = torch.eye(3, dtype=torch.float64) * 20.0  # no other image in reach
    scalars = torch.randn(2, 7, dtype=torch.float64)
    outputs = []
    for distance in (3.2 - 1.0e-7, 3.2 + 1.0e-7):
        fractional = torch.tensor(
            [[0.1, 0.1, 0.1], [0.1 + distance / 20.0, 0.1, 0.1]], dtype=torch.float64
        )
        outputs.append(model(scalars, fractional, lattice))
    inside, outside = outputs
    for name in ("scalar_channels", "vector_channels"):
        difference = (getattr(inside, name) - getattr(outside, name)).abs().max()
        assert float(difference) <= 1.0e-9, name


def test_edge_budget_overflow_fails_closed_instead_of_truncating() -> None:
    """F7 resource-bound test (theory 4.9, mutation 17)."""

    fractional = torch.zeros(1, 3, dtype=torch.float64)
    lattice = torch.eye(3, dtype=torch.float64)
    assert enumerate_periodic_edges(fractional, lattice, 1.01, max_edges=6).num_edges == 6
    with pytest.raises(PeriodicImageBudgetError, match="max_edges"):
        enumerate_periodic_edges(fractional, lattice, 1.01, max_edges=5)



def test_soft_vector_normalization_is_one_lipschitz_bounded_and_o3_equivariant() -> None:
    """N(v) = v / sqrt(1 + <v^2>) (soft-v2).

    With r = ||v|| and n components, the radial Jacobian eigenvalue is
    (1 + r^2/n)^-3/2 and the tangential one is (1 + r^2/n)^-1/2.  Both are at most
    one, so N is 1-Lipschitz, and its Jacobian at v = 0 is exactly I: no
    amplification of vanishing signals.  The output RMS is below 1.
    """

    from crystalatlas.nn.equivariant_transformer import _vector_soft_normalize

    generator = torch.Generator().manual_seed(5)
    zero = torch.zeros(4, 3, dtype=torch.float64)
    jacobian = torch.autograd.functional.jacobian(_vector_soft_normalize, zero)
    torch.testing.assert_close(
        jacobian.reshape(12, 12), torch.eye(12, dtype=torch.float64), rtol=0.0, atol=0.0
    )
    for scale in (1.0e-6, 0.3, 1.0, 10.0, 1.0e4):
        v = scale * torch.randn(4, 3, generator=generator, dtype=torch.float64)
        u = v + 0.01 * scale * torch.randn(4, 3, generator=generator, dtype=torch.float64)
        assert float(_vector_soft_normalize(v).square().mean()) < 1.0
        gap = (_vector_soft_normalize(u) - _vector_soft_normalize(v)).norm()
        assert float(gap) <= float((u - v).norm()) * (1.0 + 1.0e-12)
        spectral = torch.linalg.matrix_norm(
            torch.autograd.functional.jacobian(_vector_soft_normalize, v).reshape(12, 12), 2
        )
        assert float(spectral) <= 1.0 + 1.0e-12
        for orthogonal in (_proper_rotation(torch.float64), -_proper_rotation(torch.float64)):
            torch.testing.assert_close(
                _vector_soft_normalize(v @ orthogonal),
                _vector_soft_normalize(v) @ orthogonal,
                rtol=1.0e-13,
                atol=1.0e-13,
            )


def _cutoff_crossing_gap(model: PeriodicSE3Transformer, delta: float) -> float:
    lattice = torch.eye(3, dtype=torch.float64) * 20.0
    torch.manual_seed(31)
    scalars = torch.randn(2, 7, dtype=torch.float64)
    outputs = []
    for distance in (3.2 - delta, 3.2 + delta):
        fractional = torch.tensor(
            [[0.1, 0.1, 0.1], [0.1 + distance / 20.0, 0.1, 0.1]], dtype=torch.float64
        )
        outputs.append(model(scalars, fractional, lattice))
    inside, outside = outputs
    return max(
        float((getattr(inside, name) - getattr(outside, name)).abs().max())
        for name in ("scalar_channels", "vector_channels")
    )


def test_soft_normalized_local_transformer_is_c1_across_the_cutoff() -> None:
    """F6 (theory 4.9, mutation 12): the gap at r_c scales like chi ~ delta^2.

    chi(s) = (1 - s/r_c^2)^2 vanishes with its first derivative at r_c.  Every
    soft-v2 block map has a bounded derivative, so the jump across r_c +- delta
    is O(delta^2).  A discontinuous (gain-amplified) network gives O(1) at every
    delta.
    """

    torch.manual_seed(31)
    model = _small_transformer("soft-v2")
    coarse = _cutoff_crossing_gap(model, 1.0e-4)
    fine = _cutoff_crossing_gap(model, 1.0e-5)
    assert fine <= 1.0e-6
    assert coarse / fine > 50.0  # quadratic: about 100; discontinuous: about 1


def _local_field_velocity_gap(normalization: str) -> float:
    from crystalatlas.nn.field import IntrinsicFieldConfig, IntrinsicGeometryField

    torch.manual_seed(3)
    config = IntrinsicFieldConfig(
        atom_feature_dim=4,
        backbone="equivariant_local",
        cutoff=6.0,
        local_vector_normalization=normalization,
    )
    field = IntrinsicGeometryField(config).double().eval()
    assert field.transformer.config.vector_normalization == normalization
    lattice = torch.eye(3, dtype=torch.float64) * 30.0
    jacobian = torch.zeros(2, 3, 1, dtype=torch.float64)
    jacobian[0, 0, 0], jacobian[1, 0, 0] = 1.0, -1.0
    velocities = []
    for distance in (6.0 - 1.0e-7, 6.0 + 1.0e-7):
        fractional = torch.tensor(
            [[0.1, 0.1, 0.1], [0.1 + distance / 30.0, 0.1, 0.1]], dtype=torch.float64
        )
        output = field(
            torch.tensor(0.5, dtype=torch.float64),
            torch.zeros(6, dtype=torch.float64),
            torch.zeros(1, dtype=torch.float64),
            None,
            composition_atomic_numbers=torch.tensor([8]),
            composition_counts=torch.tensor([1.0], dtype=torch.float64),
            atomic_numbers=torch.tensor([8, 8]),
            atom_features=torch.zeros(2, 4, dtype=torch.float64),
            fractional_coordinates=fractional,
            lattice=lattice,
            bounded_lattice_coordinates=torch.zeros(6, dtype=torch.float64),
            jacobian_cartesian=jacobian,
        )
        velocities.append(torch.cat((output.intrinsic_velocity, output.lattice_velocity)))
    return float((velocities[0] - velocities[1]).abs().max())


def test_production_local_field_is_continuous_only_with_soft_normalization() -> None:
    """Production widths and depth (6 layers): the legacy jump is O(1), soft-v2 ~ chi."""

    assert _local_field_velocity_gap("rms-v1") > 1.0e-1
    assert _local_field_velocity_gap("soft-v2") <= 1.0e-9


def test_local_vector_normalization_is_a_versioned_model_option() -> None:
    from dataclasses import replace

    from crystalatlas.config import CrystalAtlasConfig, ModelConfig
    from crystalatlas.errors import DomainError

    legacy = CrystalAtlasConfig()
    assert "geometry_local_vector_normalization" not in legacy.to_dict()["model"]
    soft = replace(
        legacy, model=replace(legacy.model, geometry_local_vector_normalization="soft-v2")
    )
    assert soft.to_dict()["model"]["geometry_local_vector_normalization"] == "soft-v2"
    assert soft.fingerprint != legacy.fingerprint
    with pytest.raises(DomainError, match="local_vector_normalization"):
        ModelConfig(geometry_local_vector_normalization="layer-norm")
