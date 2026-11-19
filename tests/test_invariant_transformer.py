from __future__ import annotations

import itertools

import pytest

torch = pytest.importorskip("torch")

from crystalatlas.nn.invariant_transformer import (  # noqa: E402
    InvariantPeriodicTransformer,
    InvariantPeriodicTransformerConfig,
    _smooth_unit_vectors,
)

pytestmark = pytest.mark.ml


def _model() -> InvariantPeriodicTransformer:
    config = InvariantPeriodicTransformerConfig(
        input_scalar_dim=5,
        model_dim=24,
        pair_dim=20,
        num_heads=3,
        num_layers=2,
        reciprocal_mode_radius=1,
        heat_scales=(0.025, 0.11, 0.7),
        direct_harmonics=2,
        output_vector_dim=5,
        reference_length=1.0,
    )
    return InvariantPeriodicTransformer(config).double().eval()


def _structure() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    features = torch.tensor(
        [
            [0.2, -0.7, 0.4, 1.1, -0.3],
            [0.8, 0.1, -0.9, 0.2, 0.5],
            [-0.4, 0.6, 0.3, -0.8, 0.7],
            [0.1, 0.9, -0.2, 0.4, -1.0],
            [-0.5, 0.2, 0.8, -0.1, 0.6],
        ],
        dtype=torch.float64,
    )
    fractional = torch.tensor(
        [
            [0.07, 0.11, 0.19],
            [0.31, 0.23, 0.63],
            [0.76, 0.41, 0.29],
            [0.52, 0.84, 0.73],
            [0.19, 0.68, 0.46],
        ],
        dtype=torch.float64,
    )
    lattice = torch.tensor(
        [[3.7, 0.2, -0.1], [0.6, 3.2, 0.3], [0.2, -0.4, 3.5]],
        dtype=torch.float64,
    )
    return features, fractional, lattice


def _proper_rotation(dtype: torch.dtype) -> torch.Tensor:
    matrix = torch.tensor(
        [[0.3, -0.7, 0.2], [0.4, 0.1, -0.8], [0.6, 0.5, 0.3]],
        dtype=dtype,
    )
    rotation, _ = torch.linalg.qr(matrix)
    if torch.linalg.det(rotation) < 0:
        rotation[:, 0] *= -1
    return rotation


def test_mode_buffer_is_the_complete_symmetric_integer_cube() -> None:
    model = _model()
    expected = set(itertools.product(range(-1, 2), repeat=3))
    actual = {tuple(int(component) for component in row) for row in model.reciprocal_modes}
    assert actual == expected
    assert model.reciprocal_modes.shape == (27, 3)
    # Raw phase features keep exactly one representative of each +/- pair,
    # plus the zero cosine and no zero sine.
    assert model.phase_feature_mode_indices.shape == (14,)
    assert int(model.phase_feature_nonzero.sum().item()) == 13
    assert model.pair_input[0].in_features == 14 + 13 + 3 + 3


@pytest.mark.parametrize(
    "updates",
    [
        {"reciprocal_mode_radius": 0},
        {"heat_scales": (0.1, float("nan"), 0.4)},
        {"heat_scales": (0.1, float("inf"), 0.4)},
        {"heat_scales": (0.1, True, 0.4)},
        {"heat_scales": [0.1, 0.2, 0.4]},
        {"reference_length": float("nan")},
        {"reference_length": float("inf")},
        {"normalization_epsilon": float("nan")},
        {"max_pair_interactions": False},
        {"include_lattice_frame_carriers": 1},
        {"max_reciprocal_modes": 1},
        {"max_readout_coefficient_channels": 1},
        {"max_geometry_workspace_bytes": False},
        {"maximum_lattice_condition_number": float("nan")},
        {
            "minimum_dimensionless_lattice_singular_value": 2.0,
            "maximum_dimensionless_lattice_singular_value": 1.0,
        },
    ],
)
def test_invalid_geometry_hyperparameters_fail_closed(
    updates: dict[str, object],
) -> None:
    values: dict[str, object] = {
        "input_scalar_dim": 5,
        "model_dim": 24,
        "pair_dim": 20,
        "num_heads": 3,
        "num_layers": 2,
        "reciprocal_mode_radius": 1,
        "heat_scales": (0.025, 0.11, 0.7),
        "direct_harmonics": 2,
        "output_vector_dim": 5,
    }
    values.update(updates)
    with pytest.raises(ValueError):
        InvariantPeriodicTransformer(
            InvariantPeriodicTransformerConfig(**values)  # type: ignore[arg-type]
        )


def test_smooth_covariant_normalization_has_the_declared_collision_jacobian_bound() -> None:
    epsilon = InvariantPeriodicTransformerConfig(input_scalar_dim=1).normalization_epsilon
    vector = torch.zeros(3, dtype=torch.float64, requires_grad=True)
    jacobian = torch.autograd.functional.jacobian(
        lambda value: _smooth_unit_vectors(value, epsilon), vector
    )
    expected = 1.0 / epsilon**0.5
    torch.testing.assert_close(
        torch.linalg.matrix_norm(jacobian, ord=2),
        torch.tensor(expected, dtype=torch.float64),
        rtol=1.0e-12,
        atol=1.0e-12,
    )
    assert expected == 100.0


def test_dense_transformer_is_atom_permutation_equivariant() -> None:
    torch.manual_seed(17)
    model = _model()
    features, fractional, lattice = _structure()
    permutation = torch.tensor([3, 0, 4, 1, 2])

    reference = model(features, fractional, lattice)
    permuted = model(features[permutation], fractional[permutation], lattice)

    torch.testing.assert_close(
        permuted.scalar_channels,
        reference.scalar_channels[permutation],
        rtol=1.0e-11,
        atol=1.0e-11,
    )
    torch.testing.assert_close(
        permuted.vector_channels,
        reference.vector_channels[permutation],
        rtol=1.0e-11,
        atol=1.0e-11,
    )
    torch.testing.assert_close(
        permuted.pooled_scalars,
        reference.pooled_scalars,
        rtol=1.0e-11,
        atol=1.0e-11,
    )


def test_fourier_geometry_is_invariant_to_origins_and_integer_cell_labels() -> None:
    torch.manual_seed(19)
    model = _model()
    features, fractional, lattice = _structure()
    integer_offsets = torch.tensor(
        [[2, -3, 1], [-4, 1, 5], [0, 2, -2], [7, 0, 3], [-1, -1, 4]],
        dtype=torch.float64,
    )
    common_origin = torch.tensor([0.123, -0.44, 0.81], dtype=torch.float64)

    reference = model(features, fractional, lattice)
    relabelled = model(
        features,
        fractional + integer_offsets + common_origin,
        lattice,
    )

    torch.testing.assert_close(
        relabelled.scalar_channels,
        reference.scalar_channels,
        rtol=2.0e-11,
        atol=2.0e-11,
    )
    torch.testing.assert_close(
        relabelled.vector_channels,
        reference.vector_channels,
        rtol=2.0e-11,
        atol=2.0e-11,
    )
    torch.testing.assert_close(
        relabelled.pooled_scalars,
        reference.pooled_scalars,
        rtol=2.0e-11,
        atol=2.0e-11,
    )


def test_lattice_right_rotation_leaves_scalars_and_rotates_vector_readout() -> None:
    torch.manual_seed(23)
    model = _model()
    features, fractional, lattice = _structure()
    rotation = _proper_rotation(torch.float64)

    reference = model(features, fractional, lattice)
    rotated = model(features, fractional, lattice @ rotation)

    torch.testing.assert_close(
        rotated.scalar_channels,
        reference.scalar_channels,
        rtol=2.0e-11,
        atol=2.0e-11,
    )
    torch.testing.assert_close(
        rotated.pooled_scalars,
        reference.pooled_scalars,
        rtol=2.0e-11,
        atol=2.0e-11,
    )
    torch.testing.assert_close(
        rotated.vector_channels,
        reference.vector_channels @ rotation,
        rtol=2.0e-11,
        atol=2.0e-11,
    )


def test_consistent_length_unit_change_is_exactly_dimensionless() -> None:
    torch.manual_seed(27)
    model_angstrom = _model()
    config_nm = InvariantPeriodicTransformerConfig(
        input_scalar_dim=5,
        model_dim=24,
        pair_dim=20,
        num_heads=3,
        num_layers=2,
        reciprocal_mode_radius=1,
        heat_scales=(0.025, 0.11, 0.7),
        direct_harmonics=2,
        output_vector_dim=5,
        reference_length=0.1,
    )
    model_nm = InvariantPeriodicTransformer(config_nm).double().eval()
    model_nm.load_state_dict(model_angstrom.state_dict())
    features, fractional, lattice_angstrom = _structure()

    angstrom = model_angstrom(features, fractional, lattice_angstrom)
    nanometre = model_nm(features, fractional, 0.1 * lattice_angstrom)

    torch.testing.assert_close(
        nanometre.scalar_channels,
        angstrom.scalar_channels,
        rtol=2.0e-11,
        atol=2.0e-11,
    )
    torch.testing.assert_close(
        nanometre.vector_channels,
        angstrom.vector_channels,
        rtol=2.0e-11,
        atol=2.0e-11,
    )
    torch.testing.assert_close(
        nanometre.pooled_scalars,
        angstrom.pooled_scalars,
        rtol=2.0e-11,
        atol=2.0e-11,
    )


def test_homogeneous_packed_evaluation_equals_independent_evaluation() -> None:
    torch.manual_seed(29)
    model = _model()
    features_a, fractional_a, lattice_a = _structure()
    features_b = 0.7 * features_a.flip(0)
    fractional_b = torch.remainder(0.83 * fractional_a.flip(0) + 0.17, 1.0)
    lattice_b = torch.tensor(
        [[2.9, -0.3, 0.4], [0.2, 3.6, 0.1], [-0.5, 0.4, 3.1]],
        dtype=torch.float64,
    )

    independent_a = model(features_a, fractional_a, lattice_a)
    independent_b = model(features_b, fractional_b, lattice_b)
    packed = model(
        torch.stack((features_a, features_b)),
        torch.stack((fractional_a, fractional_b)),
        torch.stack((lattice_a, lattice_b)),
    )

    torch.testing.assert_close(
        packed.scalar_channels,
        torch.stack(
            (independent_a.scalar_channels, independent_b.scalar_channels)
        ),
        rtol=1.0e-11,
        atol=1.0e-11,
    )
    torch.testing.assert_close(
        packed.vector_channels,
        torch.stack(
            (independent_a.vector_channels, independent_b.vector_channels)
        ),
        rtol=1.0e-11,
        atol=1.0e-11,
    )
    torch.testing.assert_close(
        packed.pooled_scalars,
        torch.stack(
            (independent_a.pooled_scalars, independent_b.pooled_scalars)
        ),
        rtol=1.0e-11,
        atol=1.0e-11,
    )


def test_pair_values_preserve_geometry_for_identical_atom_tokens() -> None:
    torch.manual_seed(31)
    model = _model()
    _, fractional, lattice = _structure()
    identical = torch.ones(5, 5, dtype=torch.float64)

    output = model(identical, fractional, lattice)

    # With token-only values every identical token would remain identical.
    assert torch.std(output.scalar_channels, dim=0).max().item() > 1.0e-6
    assert output.vector_channels.abs().max().item() > 1.0e-6


def test_collision_forward_and_coordinate_lattice_gradients_are_finite() -> None:
    torch.manual_seed(37)
    model = _model()
    features = torch.randn(3, 5, dtype=torch.float64, requires_grad=True)
    fractional = torch.tensor(
        [[0.2, 0.2, 0.2], [0.2, 0.2, 0.2], [0.71, 0.43, 0.86]],
        dtype=torch.float64,
        requires_grad=True,
    )
    lattice = torch.tensor(
        [[3.2, 0.1, -0.2], [0.4, 3.5, 0.3], [0.2, -0.1, 3.0]],
        dtype=torch.float64,
        requires_grad=True,
    )

    output = model(features, fractional, lattice)
    objective = (
        output.scalar_channels.square().mean()
        + output.vector_channels.square().mean()
        + output.pooled_scalars.square().mean()
    )
    objective.backward()

    assert torch.isfinite(output.scalar_channels).all()
    assert torch.isfinite(output.vector_channels).all()
    assert features.grad is not None and torch.isfinite(features.grad).all()
    assert fractional.grad is not None and torch.isfinite(fractional.grad).all()
    assert lattice.grad is not None and torch.isfinite(lattice.grad).all()


def test_lattice_frame_carriers_remove_unphysical_all_species_collision_zero() -> None:
    model = _model()
    features, _, lattice = _structure()
    fractional = torch.full((5, 3), 0.25, dtype=torch.float64)
    geometry = model._periodic_geometry(  # noqa: SLF001 - equation regression
        fractional.unsqueeze(0), lattice.unsqueeze(0)
    )
    pair_only_basis_count = (
        model.config.direct_harmonics
        + len(model.config.heat_scales)
        + model.config.direct_harmonics * len(model.config.heat_scales)
    )
    torch.testing.assert_close(
        geometry.covariant_bases[..., :pair_only_basis_count, :],
        torch.zeros_like(geometry.covariant_bases[..., :pair_only_basis_count, :]),
        rtol=0.0,
        atol=1.0e-14,
    )
    assert geometry.covariant_bases[..., pair_only_basis_count:, :].abs().max() > 0.1

    # Distinct scalar/species-like tokens can now receive different Cartesian
    # vectors at the collision. Identical tokens remain exchange symmetric and
    # reduce to common translation, as they must.
    torch.manual_seed(73)
    output = model(features, fractional, lattice)
    relative = output.vector_channels - output.vector_channels.mean(
        dim=0, keepdim=True
    )
    assert relative.abs().max().item() > 1.0e-7


def test_internal_carriers_have_exact_polar_axial_reflection_parities() -> None:
    model = _model()
    _, fractional, lattice = _structure()
    reflection = torch.diag(torch.tensor([-1.0, 1.0, 1.0], dtype=torch.float64))
    reference = model._periodic_geometry(  # noqa: SLF001 - equation regression
        fractional.unsqueeze(0), lattice.unsqueeze(0)
    )
    mirrored = model._periodic_geometry(  # noqa: SLF001 - bypass public handedness gate
        fractional.unsqueeze(0), (lattice @ reflection).unsqueeze(0)
    )
    direct_count = model.config.direct_harmonics
    reciprocal_count = len(model.config.heat_scales)
    primary_count = direct_count + reciprocal_count
    cross_count = direct_count * reciprocal_count
    pair_scalar_without_chirality = reference.pair_features.shape[-1] - 3

    torch.testing.assert_close(
        mirrored.pair_features[..., :pair_scalar_without_chirality],
        reference.pair_features[..., :pair_scalar_without_chirality],
        rtol=2.0e-12,
        atol=2.0e-12,
    )
    torch.testing.assert_close(
        mirrored.node_chirality,
        -reference.node_chirality,
        rtol=2.0e-11,
        atol=2.0e-11,
    )
    torch.testing.assert_close(
        mirrored.covariant_bases[..., :primary_count, :],
        reference.covariant_bases[..., :primary_count, :] @ reflection,
        rtol=2.0e-11,
        atol=2.0e-11,
    )
    torch.testing.assert_close(
        mirrored.covariant_bases[
            ..., primary_count : primary_count + cross_count, :
        ],
        -reference.covariant_bases[
            ..., primary_count : primary_count + cross_count, :
        ]
        @ reflection,
        rtol=2.0e-11,
        atol=2.0e-11,
    )
    torch.testing.assert_close(
        mirrored.covariant_bases[..., primary_count + cross_count :, :],
        reference.covariant_bases[..., primary_count + cross_count :, :]
        @ reflection,
        rtol=2.0e-11,
        atol=2.0e-11,
    )


def test_full_model_is_not_forced_to_follow_o3_polar_parity_on_mirror_witness() -> None:
    torch.manual_seed(41)
    model = _model()
    features, fractional, lattice = _structure()
    reflection = torch.diag(torch.tensor([-1.0, 1.0, 1.0], dtype=torch.float64))
    orientation_reversing_basis = torch.diag(
        torch.tensor([-1.0, 1.0, 1.0], dtype=torch.float64)
    )
    mirrored_lattice = orientation_reversing_basis @ lattice @ reflection
    mirrored_fractional = fractional @ torch.linalg.inv(
        orientation_reversing_basis
    )
    assert torch.linalg.det(mirrored_lattice).item() > 0.0

    reference = model(features, fractional, lattice)
    mirrored = model(features, mirrored_fractional, mirrored_lattice)

    # This is a valid-domain witness that the complete model is not constrained
    # to polar O(3) parity. It deliberately includes an orientation-reversing
    # fractional-basis change and therefore is not, by itself, an enantiomorph-
    # pipeline or basis-invariance test. The internal carrier test above
    # separately isolates the polar/axial equations.
    assert (
        mirrored.scalar_channels - reference.scalar_channels
    ).abs().max().item() > 1.0e-6
    assert (
        mirrored.vector_channels - reference.vector_channels @ reflection
    ).abs().max().item() > 1.0e-6


@pytest.mark.parametrize("bad_input", ["features", "fractional", "lattice"])
def test_nonfinite_inputs_fail_closed(bad_input: str) -> None:
    model = _model()
    features, fractional, lattice = _structure()
    values = {
        "features": features.clone(),
        "fractional": fractional.clone(),
        "lattice": lattice.clone(),
    }
    values[bad_input].reshape(-1)[0] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        model(values["features"], values["fractional"], values["lattice"])


def test_lattice_domain_is_right_handed_and_well_conditioned() -> None:
    model = _model()
    features, fractional, lattice = _structure()
    left_handed = lattice.clone()
    left_handed[0] *= -1.0
    with pytest.raises(ValueError, match="right-handed"):
        model(features, fractional, left_handed)

    ill_conditioned = torch.diag(
        torch.tensor([1.0, 1.0, 1.0e-10], dtype=torch.float64)
    )
    with pytest.raises(ValueError, match="supported nonsingular domain"):
        model(features, fractional, ill_conditioned)


def test_direct_transformer_pair_budget_fails_before_geometry_allocation() -> None:
    config = InvariantPeriodicTransformerConfig(
        input_scalar_dim=5,
        model_dim=24,
        pair_dim=20,
        num_heads=3,
        num_layers=1,
        reciprocal_mode_radius=1,
        direct_harmonics=2,
        output_vector_dim=5,
        max_pair_interactions=24,
    )
    model = InvariantPeriodicTransformer(config).double()
    features, fractional, lattice = _structure()
    with pytest.raises(ValueError, match=r"\(25 > 24\)"):
        model(features, fractional, lattice)


def test_workspace_budget_fails_before_dense_pair_geometry_allocation() -> None:
    config = InvariantPeriodicTransformerConfig(
        input_scalar_dim=5,
        model_dim=24,
        pair_dim=20,
        num_heads=3,
        num_layers=1,
        reciprocal_mode_radius=1,
        direct_harmonics=2,
        output_vector_dim=5,
        max_pair_interactions=100,
        max_geometry_workspace_bytes=1,
    )
    model = InvariantPeriodicTransformer(config).double()
    features, fractional, lattice = _structure()
    with pytest.raises(ValueError, match="workspace budget"):
        model(features, fractional, lattice)


def test_direct_low_precision_inputs_require_master_inputs_and_autocast() -> None:
    model = _model().float()
    features, fractional, lattice = _structure()
    with pytest.raises(ValueError, match="master dtype"):
        model(features.bfloat16(), fractional.bfloat16(), lattice.bfloat16())


def test_validate_geometry_false_never_waives_nonfinite_or_handedness_checks() -> None:
    model = _model()
    features, fractional, lattice = _structure()
    invalid = fractional.clone()
    invalid[0, 0] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        model(features, invalid, lattice, validate_geometry=False)
    left_handed = lattice.clone()
    left_handed[0] *= -1.0
    with pytest.raises(ValueError, match="right-handed"):
        model(features, fractional, left_handed, validate_geometry=False)


def test_bfloat16_attention_uses_stable_finite_accumulation_under_large_logits() -> None:
    torch.manual_seed(59)
    model = _model().float().train()
    with torch.no_grad():
        for block in model.blocks:
            block.query.weight.mul_(1.0e3)
            block.key.weight.mul_(1.0e3)
            block.pair_bias.weight.mul_(1.0e3)
    features, fractional, lattice = (
        value.float() for value in _structure()
    )
    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        output = model(features, fractional, lattice)
        loss = (
            output.scalar_channels.float().square().mean()
            + output.vector_channels.float().square().mean()
        )
    loss.backward()

    assert torch.isfinite(output.scalar_channels).all()
    assert torch.isfinite(output.vector_channels).all()
    gradients = [
        parameter.grad
        for parameter in model.parameters()
        if parameter.grad is not None
    ]
    assert gradients
    assert all(bool(torch.isfinite(gradient).all().item()) for gradient in gradients)
