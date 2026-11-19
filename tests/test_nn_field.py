from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from crystalatlas.nn.field import (  # noqa: E402
    IntrinsicFieldConfig,
    IntrinsicGeometryField,
    TangentRankError,
    reduce_cartesian_to_intrinsic,
    translation_quotient_normal_matrix,
)

pytestmark = pytest.mark.ml


def _certified_jacobian() -> torch.Tensor:
    jacobian = torch.zeros(3, 3, 2, dtype=torch.float64)
    jacobian[:, 0, 0] = torch.tensor([-1.0, 1.0, 0.0], dtype=torch.float64)
    jacobian[:, 1, 1] = torch.tensor([0.0, -1.0, 1.0], dtype=torch.float64)
    return jacobian


def _proper_rotation(dtype: torch.dtype) -> torch.Tensor:
    matrix = torch.tensor(
        [[0.3, -0.7, 0.2], [0.4, 0.1, -0.8], [0.6, 0.5, 0.3]],
        dtype=dtype,
    )
    rotation, _ = torch.linalg.qr(matrix)
    if torch.linalg.det(rotation) < 0:
        rotation[:, 0] *= -1
    return rotation


def test_exact_no_ridge_translation_quotient_reduction() -> None:
    jacobian = _certified_jacobian()
    true_velocity = torch.tensor([0.4, -0.7], dtype=torch.float64)
    translation = torch.tensor([3.0, -2.0, 0.5], dtype=torch.float64)
    expanded = torch.einsum("nci,i->nc", jacobian, true_velocity) + translation
    recovered = reduce_cartesian_to_intrinsic(expanded, jacobian)
    torch.testing.assert_close(recovered, true_velocity, rtol=1.0e-12, atol=1.0e-12)
    normal = translation_quotient_normal_matrix(jacobian)
    assert torch.linalg.eigvalsh(normal).min().item() > 0.0

    invalid = torch.ones(3, 3, 1, dtype=torch.float64)
    with pytest.raises(TangentRankError):
        reduce_cartesian_to_intrinsic(torch.zeros(3, 3, dtype=torch.float64), invalid)


def test_float32_tangent_reduction_uses_float64_algebra_and_condition_gate() -> None:
    base = torch.tensor([-1.0, 0.0, 1.0], dtype=torch.float32)
    perturbation = torch.tensor([-1.0, 1.0, 0.0], dtype=torch.float32)
    jacobian = torch.zeros(3, 3, 2, dtype=torch.float32)
    jacobian[:, 0, 0] = base
    jacobian[:, 0, 1] = base
    jacobian[:, 1, 1] = 1.0e-3 * perturbation
    true_velocity = torch.tensor([1.0, -0.7], dtype=torch.float32)
    expanded = torch.einsum("nci,i->nc", jacobian, true_velocity)

    singular_values = torch.linalg.svdvals(
        (jacobian - jacobian.mean(dim=0, keepdim=True)).reshape(-1, 2)
    )
    condition = float((singular_values[0] / singular_values[-1]).item())
    assert 1.0e3 < condition < 1.0e4
    recovered = reduce_cartesian_to_intrinsic(expanded, jacobian)
    assert recovered.dtype == torch.float32
    torch.testing.assert_close(recovered, true_velocity, rtol=2.0e-5, atol=2.0e-5)
    assert translation_quotient_normal_matrix(jacobian).dtype == torch.float64

    near_collinear = jacobian.clone()
    near_collinear[:, 1, 1] = 1.0e-5 * perturbation
    with pytest.raises(TangentRankError, match="condition-number support"):
        reduce_cartesian_to_intrinsic(expanded, near_collinear)


@pytest.mark.parametrize(
    ("vectors", "jacobian", "message"),
    [
        (
            torch.zeros(3, 3, dtype=torch.float64),
            torch.zeros(3, 3, dtype=torch.float64),
            "shape",
        ),
        (
            torch.zeros(3, 3, dtype=torch.float32),
            torch.zeros(3, 3, 1, dtype=torch.float64),
            "master dtype",
        ),
        (
            torch.zeros(3, 3, dtype=torch.int64),
            torch.zeros(3, 3, 1, dtype=torch.int64),
            "floating-point",
        ),
    ],
)
def test_tangent_helpers_reject_malformed_dtype_contracts(
    vectors: torch.Tensor,
    jacobian: torch.Tensor,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        reduce_cartesian_to_intrinsic(vectors, jacobian)


def test_tangent_helpers_reject_nonfinite_values() -> None:
    jacobian = _certified_jacobian()
    jacobian[0, 0, 0] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        translation_quotient_normal_matrix(jacobian)
    with pytest.raises(ValueError, match="finite"):
        reduce_cartesian_to_intrinsic(
            torch.zeros(3, 3, dtype=torch.float64), jacobian
        )


def test_intrinsic_field_smoke_and_gradients() -> None:
    torch.manual_seed(31)
    config = IntrinsicFieldConfig(
        atom_feature_dim=5,
        composition_dim=12,
        species_embedding_dim=8,
        max_lattice_dim=3,
        num_time_frequencies=2,
        scalar_dim=24,
        vector_dim=12,
        num_heads=3,
        num_layers=2,
        radial_basis_dim=6,
        cutoff=2.8,
    )
    field = IntrinsicGeometryField(config).double().eval()
    z = torch.tensor([0.2, -0.3], dtype=torch.float64, requires_grad=True)
    eta = torch.tensor([0.1, 0.7], dtype=torch.float64)
    context = torch.randn(12, dtype=torch.float64)
    atomic_numbers = torch.tensor([8, 14, 26])
    atom_features = torch.randn(3, 5, dtype=torch.float64, requires_grad=True)
    fractional = torch.tensor(
        [[0.1, 0.2, 0.3], [0.4, 0.1, 0.6], [0.7, 0.6, 0.2]], dtype=torch.float64
    )
    lattice = torch.eye(3, dtype=torch.float64) * 3.2
    output = field(
        torch.tensor(0.4, dtype=torch.float64),
        z,
        eta,
        context,
        atomic_numbers=atomic_numbers,
        atom_features=atom_features,
        fractional_coordinates=fractional,
        lattice=lattice,
        bounded_lattice_coordinates=torch.tanh(z),
        jacobian_cartesian=_certified_jacobian(),
    )
    assert output.lattice_velocity.shape == z.shape
    assert output.intrinsic_velocity.shape == eta.shape
    assert output.cartesian_vectors.shape == (3, 3)
    assert torch.isfinite(output.lattice_velocity).all()
    assert torch.isfinite(output.intrinsic_velocity).all()
    objective = (
        output.lattice_velocity.square().sum()
        + output.intrinsic_velocity.square().sum()
    )
    objective.backward()
    assert z.grad is not None and torch.isfinite(z.grad).all()
    assert atom_features.grad is not None and torch.isfinite(atom_features.grad).all()


def test_certified_provenance_flag_never_waives_runtime_value_checks() -> None:
    field = IntrinsicGeometryField(
        IntrinsicFieldConfig(
            atom_feature_dim=1,
            composition_dim=4,
            species_embedding_dim=4,
            max_lattice_dim=1,
            num_time_frequencies=1,
            scalar_dim=8,
            pair_dim=6,
            vector_dim=4,
            num_heads=1,
            num_layers=1,
            radial_basis_dim=2,
            reciprocal_mode_radius=1,
            direct_harmonics=1,
        )
    ).double()
    common = dict(
        time=torch.tensor(0.5, dtype=torch.float64),
        z=torch.zeros(1, dtype=torch.float64),
        eta=torch.empty(0, dtype=torch.float64),
        composition_context=torch.zeros(4, dtype=torch.float64),
        atomic_numbers=torch.tensor([8]),
        atom_features=torch.zeros(1, 1, dtype=torch.float64),
        fractional_coordinates=torch.zeros(1, 3, dtype=torch.float64),
        lattice=torch.eye(3, dtype=torch.float64),
        bounded_lattice_coordinates=torch.zeros(1, dtype=torch.float64),
        jacobian_cartesian=torch.empty(1, 3, 0, dtype=torch.float64),
        geometry_is_certified=True,
    )
    invalid = dict(common)
    invalid["z"] = torch.tensor([float("nan")], dtype=torch.float64)
    with pytest.raises(ValueError, match="finite"):
        field(**invalid)

    invalid = dict(common)
    invalid["atomic_numbers"] = torch.tensor([8.0], dtype=torch.float64)
    with pytest.raises(ValueError, match="integer"):
        field(**invalid)


def test_lattice_velocity_has_a_linear_growth_path_for_gaussian_tails() -> None:
    config = IntrinsicFieldConfig(
        atom_feature_dim=1,
        composition_dim=4,
        species_embedding_dim=4,
        max_lattice_dim=1,
        num_time_frequencies=1,
        scalar_dim=8,
        vector_dim=4,
        num_heads=1,
        num_layers=1,
        radial_basis_dim=3,
        cutoff=1.5,
        lattice_linear_growth_bound=2.0,
        lattice_residual_bound=1.0,
    )
    field = IntrinsicGeometryField(config).double().eval()
    with torch.no_grad():
        for parameter in field.lattice_residual_head.parameters():
            parameter.zero_()
        for parameter in field.lattice_linear_head.parameters():
            parameter.zero_()
        field.lattice_linear_head[2].bias.fill_(0.5)
    common = {
        "time": torch.tensor(0.4, dtype=torch.float64),
        "eta": torch.empty(0, dtype=torch.float64),
        "composition_context": torch.zeros(4, dtype=torch.float64),
        "atomic_numbers": torch.tensor([8]),
        "atom_features": torch.zeros(1, 1, dtype=torch.float64),
        "fractional_coordinates": torch.zeros(1, 3, dtype=torch.float64),
        "lattice": torch.eye(3, dtype=torch.float64) * 3.0,
        "jacobian_cartesian": torch.empty(1, 3, 0, dtype=torch.float64),
    }

    def evaluate(value: float) -> torch.Tensor:
        z = torch.tensor([value], dtype=torch.float64)
        return field(
            common["time"],
            z,
            common["eta"],
            common["composition_context"],
            atomic_numbers=common["atomic_numbers"],
            atom_features=common["atom_features"],
            fractional_coordinates=common["fractional_coordinates"],
            lattice=common["lattice"],
            bounded_lattice_coordinates=torch.tanh(z),
            jacobian_cartesian=common["jacobian_cartesian"],
        ).lattice_velocity

    small = evaluate(1.0)
    large = evaluate(100.0)
    torch.testing.assert_close(large, 100.0 * small, rtol=1.0e-12, atol=1.0e-12)
    assert large.abs().item() > 1.0


def test_full_invariant_field_is_unchanged_by_consistent_angstrom_to_nm_units() -> None:
    torch.manual_seed(71)
    common = dict(
        atom_feature_dim=5,
        composition_dim=12,
        species_embedding_dim=8,
        max_lattice_dim=3,
        num_time_frequencies=2,
        scalar_dim=24,
        pair_dim=20,
        vector_dim=12,
        num_heads=3,
        num_layers=2,
        radial_basis_dim=6,
        reciprocal_mode_radius=1,
        direct_harmonics=2,
    )
    angstrom_field = IntrinsicGeometryField(
        IntrinsicFieldConfig(reference_length=1.0, **common)
    ).double().eval()
    nanometre_field = IntrinsicGeometryField(
        IntrinsicFieldConfig(reference_length=0.1, **common)
    ).double().eval()
    nanometre_field.load_state_dict(angstrom_field.state_dict())

    z = torch.tensor([0.2, -0.3], dtype=torch.float64)
    eta = torch.tensor([0.1, 0.7], dtype=torch.float64)
    context = torch.randn(12, dtype=torch.float64)
    atomic_numbers = torch.tensor([8, 14, 26])
    atom_features = torch.randn(3, 5, dtype=torch.float64)
    fractional = torch.tensor(
        [[0.1, 0.2, 0.3], [0.4, 0.1, 0.6], [0.7, 0.6, 0.2]],
        dtype=torch.float64,
    )
    lattice_angstrom = torch.eye(3, dtype=torch.float64) * 3.2
    jacobian_angstrom = _certified_jacobian()

    def evaluate(
        field: IntrinsicGeometryField,
        lattice: torch.Tensor,
        jacobian: torch.Tensor,
    ):  # type: ignore[no-untyped-def]
        return field(
            torch.tensor(0.4, dtype=torch.float64),
            z,
            eta,
            context,
            atomic_numbers=atomic_numbers,
            atom_features=atom_features,
            fractional_coordinates=fractional,
            lattice=lattice,
            bounded_lattice_coordinates=torch.tanh(z),
            jacobian_cartesian=jacobian,
        )

    angstrom = evaluate(angstrom_field, lattice_angstrom, jacobian_angstrom)
    nanometre = evaluate(
        nanometre_field,
        0.1 * lattice_angstrom,
        0.1 * jacobian_angstrom,
    )
    for actual, expected in (
        (nanometre.lattice_velocity, angstrom.lattice_velocity),
        (nanometre.intrinsic_velocity, angstrom.intrinsic_velocity),
        (nanometre.cartesian_vectors, angstrom.cartesian_vectors),
        (nanometre.atom_scalar_channels, angstrom.atom_scalar_channels),
        (nanometre.coordinate_metric, angstrom.coordinate_metric),
    ):
        torch.testing.assert_close(actual, expected, rtol=3.0e-11, atol=3.0e-11)


@pytest.mark.parametrize(
    ("dtype", "rtol", "atol"),
    (
        (torch.float64, 3.0e-10, 3.0e-10),
        (torch.float32, 8.0e-5, 8.0e-6),
    ),
)
def test_full_invariant_field_obeys_proper_so3_contract(
    dtype: torch.dtype,
    rtol: float,
    atol: float,
) -> None:
    """Test the complete field, including the metric-adjoint reduction.

    With row lattices and row Cartesian vectors, a proper Cartesian rotation
    acts as ``L' = L Q`` and ``v' = v Q``.  Each Jacobian column is a Cartesian
    vector, hence ``J'[n,:,a] = Q^T J[n,:,a]``.  Consequently
    ``J'^T P_N J' = J^T P_N J`` and the reduced intrinsic velocity is invariant.
    """

    torch.manual_seed(79)
    field = IntrinsicGeometryField(
        IntrinsicFieldConfig(
            atom_feature_dim=5,
            composition_dim=12,
            species_embedding_dim=8,
            max_lattice_dim=3,
            num_time_frequencies=2,
            backbone="invariant_global",
            scalar_dim=24,
            pair_dim=20,
            vector_dim=12,
            num_heads=3,
            num_layers=2,
            radial_basis_dim=6,
            reciprocal_mode_radius=1,
            direct_harmonics=2,
        )
    ).to(dtype=dtype).eval()

    z = torch.tensor([0.2, -0.3], dtype=dtype)
    eta = torch.tensor([0.1, 0.7], dtype=dtype)
    context = torch.randn(12, dtype=dtype)
    atomic_numbers = torch.tensor([8, 14, 26, 6])
    atom_features = torch.randn(4, 5, dtype=dtype)
    fractional = torch.tensor(
        [
            [0.07, 0.11, 0.19],
            [0.31, 0.23, 0.63],
            [0.76, 0.41, 0.29],
            [0.52, 0.84, 0.73],
        ],
        dtype=dtype,
    )
    lattice = torch.tensor(
        [[3.7, 0.2, -0.1], [0.6, 3.2, 0.3], [0.2, -0.4, 3.5]],
        dtype=dtype,
    )
    rotation = _proper_rotation(dtype)

    # Start from a fractional-coordinate Jacobian so the Cartesian Jacobian is
    # physically consistent with the row lattice: J[:, :, a] = L^T D[:, :, a].
    fractional_jacobian = torch.tensor(
        [
            [[1.0, 0.0], [0.0, 0.5], [0.2, -0.3]],
            [[-1.0, 0.0], [0.1, -0.5], [-0.2, 0.4]],
            [[0.3, 0.8], [-0.6, 0.2], [0.5, -0.1]],
            [[-0.3, -0.8], [0.5, -0.2], [-0.5, 0.0]],
        ],
        dtype=dtype,
    )
    jacobian = torch.einsum("cd,ndk->nck", lattice.transpose(0, 1), fractional_jacobian)
    rotated_lattice = lattice @ rotation
    rotated_jacobian = torch.einsum(
        "ac,nci->nai", rotation.transpose(0, 1), jacobian
    )

    pooled_states: list[torch.Tensor] = []

    def capture_pooled(_module, _inputs, output) -> None:  # type: ignore[no-untyped-def]
        pooled_states.append(output.pooled_scalars.detach().clone())

    hook = field.transformer.register_forward_hook(capture_pooled)
    try:
        common = {
            "time": torch.tensor(0.4, dtype=dtype),
            "z": z,
            "eta": eta,
            "composition_context": context,
            "atomic_numbers": atomic_numbers,
            "atom_features": atom_features,
            "fractional_coordinates": fractional,
            "bounded_lattice_coordinates": torch.tanh(z),
        }
        with torch.no_grad():
            reference = field(
                **common,
                lattice=lattice,
                jacobian_cartesian=jacobian,
            )
            rotated = field(
                **common,
                lattice=rotated_lattice,
                jacobian_cartesian=rotated_jacobian,
            )
    finally:
        hook.remove()

    assert len(pooled_states) == 2
    torch.testing.assert_close(
        rotated.cartesian_vectors,
        reference.cartesian_vectors @ rotation,
        rtol=rtol,
        atol=atol,
    )
    for actual, expected in (
        (rotated.atom_scalar_channels, reference.atom_scalar_channels),
        (pooled_states[1], pooled_states[0]),
        (rotated.lattice_velocity, reference.lattice_velocity),
        (rotated.intrinsic_velocity, reference.intrinsic_velocity),
        (rotated.coordinate_metric, reference.coordinate_metric),
    ):
        torch.testing.assert_close(actual, expected, rtol=rtol, atol=atol)


def _random_jacobians(batch: int, atoms: int, dimension: int, seed: int) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    return torch.randn(batch, atoms, 3, dimension, generator=generator, dtype=torch.float64)


@pytest.mark.parametrize("dtype", (torch.float32, torch.float64))
def test_packed_reduction_equals_per_crystal_reduction(dtype) -> None:
    from crystalatlas.nn.field import (
        reduce_cartesian_to_intrinsic_packed,
        translation_quotient_normal_matrix_packed,
    )

    jacobian = _random_jacobians(6, 4, 3, seed=3).to(dtype=dtype)
    generator = torch.Generator().manual_seed(4)
    vectors = torch.randn(6, 4, 3, generator=generator, dtype=torch.float64).to(dtype=dtype)
    packed = reduce_cartesian_to_intrinsic_packed(vectors, jacobian, reference_length=1.7)
    alone = torch.stack(
        [
            reduce_cartesian_to_intrinsic(vectors[b], jacobian[b], reference_length=1.7)
            for b in range(6)
        ]
    )
    assert packed.dtype == dtype
    tolerance = 1.0e-12 if dtype == torch.float64 else 1.0e-6
    torch.testing.assert_close(packed, alone, rtol=tolerance, atol=tolerance)
    metric = translation_quotient_normal_matrix_packed(jacobian, reference_length=1.7)
    assert metric.dtype == torch.float64
    torch.testing.assert_close(
        metric,
        torch.stack(
            [
                translation_quotient_normal_matrix(jacobian[b], reference_length=1.7)
                for b in range(6)
            ]
        ),
        rtol=1.0e-12,
        atol=1.0e-12,
    )
    empty = reduce_cartesian_to_intrinsic_packed(vectors, jacobian[..., :0])
    assert empty.shape == (6, 0) and empty.dtype == dtype


def test_packed_reduction_refuses_the_first_uncertified_crystal_like_the_audit() -> None:
    """Same certificate and message as the per-crystal audit, first refusal first."""

    from crystalatlas.nn.field import reduce_cartesian_to_intrinsic_packed

    jacobian = _random_jacobians(4, 4, 2, seed=5)
    vectors = torch.zeros(4, 4, 3, dtype=torch.float64)
    ill_conditioned = jacobian.clone()
    ill_conditioned[1, :, :, 1] = ill_conditioned[1, :, :, 0] + 1.0e-8 * jacobian[1, :, :, 1]
    ill_conditioned[2] = 1.0  # every atom moves alike: P_N J = 0 exactly
    with pytest.raises(TangentRankError) as alone:
        reduce_cartesian_to_intrinsic(vectors[1], ill_conditioned[1])
    with pytest.raises(TangentRankError) as packed:
        reduce_cartesian_to_intrinsic_packed(vectors, ill_conditioned)
    assert "condition-number support" in str(alone.value)
    assert str(packed.value) == str(alone.value)

    deficient = ill_conditioned.clone()
    deficient[1] = jacobian[1]
    with pytest.raises(TangentRankError) as alone:
        reduce_cartesian_to_intrinsic(vectors[2], deficient[2])
    with pytest.raises(TangentRankError) as packed:
        reduce_cartesian_to_intrinsic_packed(vectors, deficient)
    assert "rank deficient" in str(alone.value)
    assert str(packed.value) == str(alone.value)


def test_batched_time_features_equal_per_time_features() -> None:
    field = IntrinsicGeometryField(IntrinsicFieldConfig(atom_feature_dim=4)).double()
    times = torch.tensor([0.0, 0.13, 0.5, 0.97, 1.0], dtype=torch.float64)
    batched = field.time_features.batched(times, dtype=torch.float64, device=times.device)
    expected = torch.stack(
        [field.time_features(t, dtype=torch.float64, device=times.device) for t in times]
    )
    torch.testing.assert_close(batched, expected, rtol=1.0e-15, atol=1.0e-15)
    with pytest.raises(ValueError, match="finite"):
        field.time_features.batched(
            torch.tensor([0.1, float("nan")]), dtype=torch.float64, device=times.device
        )
