from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from crystalatlas.flow import FlowState, integrate_rk4  # noqa: E402
from crystalatlas.nn.equivariant_transformer import (  # noqa: E402
    PeriodicEdges,
    PeriodicImageBudgetError,
    concatenate_periodic_edges,
    enumerate_packed_periodic_edges,
    enumerate_periodic_edges,
)
from crystalatlas.nn.field import (  # noqa: E402
    IntrinsicFieldConfig,
    IntrinsicGeometryField,
)

pytestmark = pytest.mark.ml


def _field() -> IntrinsicGeometryField:
    return IntrinsicGeometryField(
        IntrinsicFieldConfig(
            atom_feature_dim=4,
            composition_dim=8,
            species_embedding_dim=4,
            max_lattice_dim=3,
            num_time_frequencies=2,
            scalar_dim=12,
            vector_dim=6,
            num_heads=3,
            num_layers=2,
            radial_basis_dim=4,
            cutoff=1.5,
            reference_length=1.0,
            max_candidates_per_pair=256,
            max_periodic_edges=2_000,
        )
    ).double()


def _inputs(field: IntrinsicGeometryField) -> dict[str, torch.Tensor]:
    dtype = torch.float64
    z = torch.tensor([[0.1, -0.2], [-0.3, 0.25]], dtype=dtype)
    eta = torch.tensor([[0.2], [-0.1]], dtype=dtype)
    numbers = torch.tensor([[6, 8], [6, 8]], dtype=torch.long)
    atom_features = torch.tensor(
        [
            [[0.2, 0.0, 0.1, 1.0], [0.3, 1.0, 0.4, 0.0]],
            [[0.2, 0.0, 0.1, 1.0], [0.3, 1.0, 0.4, 0.0]],
        ],
        dtype=dtype,
    )
    fractional = torch.tensor(
        [
            [[0.0, 0.0, 0.0], [0.2, 0.1, 0.0]],
            [[0.1, 0.0, 0.2], [0.3, 0.2, 0.2]],
        ],
        dtype=dtype,
    )
    lattice = torch.stack(
        (
            torch.diag(torch.tensor([3.0, 3.2, 3.4], dtype=dtype)),
            torch.diag(torch.tensor([3.1, 3.3, 3.5], dtype=dtype)),
        )
    )
    jacobian = torch.zeros(2, 2, 3, 1, dtype=dtype)
    jacobian[:, 0, 0, 0] = 1.0
    jacobian[:, 1, 0, 0] = -1.0
    contexts = torch.stack(
        [
            field.encode_composition(
                numbers[index], torch.tensor([1.0, 1.0], dtype=dtype)
            )
            for index in range(2)
        ]
    )
    return {
        "z": z,
        "eta": eta,
        "composition_context": contexts,
        "atomic_numbers": numbers,
        "atom_features": atom_features,
        "fractional_coordinates": fractional,
        "lattice": lattice,
        "bounded_lattice_coordinates": 0.5 * z,
        "jacobian_cartesian": jacobian,
    }


def test_concatenated_periodic_edges_are_block_diagonal() -> None:
    dtype = torch.float64
    fractional = (
        torch.tensor([[0.0, 0.0, 0.0], [0.2, 0.0, 0.0]], dtype=dtype),
        torch.tensor([[0.0, 0.0, 0.0]], dtype=dtype),
    )
    lattices = (
        torch.eye(3, dtype=dtype) * 3.0,
        torch.eye(3, dtype=dtype) * 2.5,
    )
    components = tuple(
        enumerate_periodic_edges(points, lattice, 1.2, max_edges=100)
        for points, lattice in zip(fractional, lattices, strict=True)
    )
    packed = concatenate_periodic_edges(components, (2, 1), max_edges=200)
    assert packed.num_edges == sum(item.num_edges for item in components)
    graph = torch.tensor([0, 0, 1], dtype=torch.long)
    assert torch.equal(graph[packed.destination], graph[packed.source])
    if components[1].num_edges:
        offset = components[0].num_edges
        torch.testing.assert_close(
            packed.destination[offset:], components[1].destination + 2
        )


def test_per_crystal_edge_support_is_invariant_to_batch_size() -> None:
    fractional = torch.zeros(1, 3, dtype=torch.float64)
    lattice = torch.eye(3, dtype=torch.float64)

    individual = enumerate_periodic_edges(
        fractional,
        lattice,
        1.01,
        max_edges=7,
    )
    assert individual.num_edges == 6
    packed = enumerate_packed_periodic_edges(
        (fractional, fractional),
        (lattice, lattice),
        1.01,
        max_edges=7,
    )
    assert packed.num_edges == 2 * individual.num_edges

    # A separately declared batch-memory cap may still reject the proposed
    # packing, but it is not confused with the per-crystal scientific support.
    with pytest.raises(PeriodicImageBudgetError, match="packed edge count"):
        enumerate_packed_periodic_edges(
            (fractional, fractional),
            (lattice, lattice),
            1.01,
            max_edges=7,
            max_total_edges=10,
        )


def test_packed_field_equals_independent_block_evaluations() -> None:
    torch.manual_seed(47)
    field = _field().eval()
    values = _inputs(field)
    time = torch.tensor(0.37, dtype=torch.float64)
    independent = []
    for index in range(2):
        independent.append(
            field(
                time,
                values["z"][index],
                values["eta"][index],
                values["composition_context"][index],
                atomic_numbers=values["atomic_numbers"][index],
                atom_features=values["atom_features"][index],
                fractional_coordinates=values["fractional_coordinates"][index],
                lattice=values["lattice"][index],
                bounded_lattice_coordinates=values[
                    "bounded_lattice_coordinates"
                ][index],
                jacobian_cartesian=values["jacobian_cartesian"][index],
            )
        )
    packed = field.forward_packed(time, **values)
    torch.testing.assert_close(
        packed.lattice_velocity,
        torch.stack([item.lattice_velocity for item in independent]),
        rtol=1.0e-11,
        atol=1.0e-11,
    )
    torch.testing.assert_close(
        packed.intrinsic_velocity,
        torch.stack([item.intrinsic_velocity for item in independent]),
        rtol=1.0e-11,
        atol=1.0e-11,
    )
    torch.testing.assert_close(
        packed.cartesian_vectors,
        torch.stack([item.cartesian_vectors for item in independent]),
        rtol=1.0e-11,
        atol=1.0e-11,
    )
    torch.testing.assert_close(
        packed.atom_scalar_channels,
        torch.stack([item.atom_scalar_channels for item in independent]),
        rtol=1.0e-11,
        atol=1.0e-11,
    )
    torch.testing.assert_close(
        packed.coordinate_metric,
        torch.stack([item.coordinate_metric for item in independent]),
        rtol=1.0e-11,
        atol=1.0e-11,
    )


def test_packed_transformer_path_backpropagates_finite_gradients() -> None:
    torch.manual_seed(53)
    field = _field().train()
    values = _inputs(field)
    packed = field.forward_packed(torch.tensor([0.2, 0.8]), **values)
    loss = packed.lattice_velocity.square().mean() + packed.intrinsic_velocity.square().mean()
    loss.backward()
    gradients = [
        parameter.grad for parameter in field.parameters() if parameter.grad is not None
    ]
    assert gradients
    assert all(bool(torch.isfinite(gradient).all().item()) for gradient in gradients)


def test_packed_dense_interaction_budget_fails_before_attention_allocation() -> None:
    field = IntrinsicGeometryField(
        IntrinsicFieldConfig(
            atom_feature_dim=4,
            composition_dim=8,
            species_embedding_dim=4,
            max_lattice_dim=3,
            num_time_frequencies=2,
            scalar_dim=12,
            pair_dim=8,
            vector_dim=6,
            num_heads=3,
            num_layers=1,
            reciprocal_mode_radius=1,
            direct_harmonics=2,
            max_dense_pair_interactions=7,
        )
    ).double()
    values = _inputs(field)
    with pytest.raises(ValueError, match=r"\(8 > 7\)"):
        field.forward_packed(torch.tensor(0.5), **values)


def test_concatenation_rejects_cross_block_component_indices() -> None:
    bad = PeriodicEdges(
        destination=torch.tensor([0]),
        source=torch.tensor([1]),
        shifts=torch.zeros(1, 3, dtype=torch.long),
        displacement=torch.zeros(1, 3),
        squared_distance=torch.zeros(1),
    )
    with pytest.raises(ValueError, match="outside its atom block"):
        concatenate_periodic_edges((bad,), (1,), max_edges=3)


def test_rk4_evaluates_a_topology_batch_in_one_field_call_per_stage() -> None:
    torch.manual_seed(61)
    field = _field().eval()
    values = _inputs(field)
    period = torch.eye(1, dtype=torch.float64)
    batched_calls = 0

    def batched_velocity(time, z, eta):  # type: ignore[no-untyped-def]
        nonlocal batched_calls
        batched_calls += 1
        return field.forward_packed(
            time,
            z,
            eta,
            values["composition_context"],
            atomic_numbers=values["atomic_numbers"],
            atom_features=values["atom_features"],
            fractional_coordinates=values["fractional_coordinates"],
            lattice=values["lattice"],
            bounded_lattice_coordinates=values["bounded_lattice_coordinates"],
            jacobian_cartesian=values["jacobian_cartesian"],
        )

    steps = 3
    packed = integrate_rk4(
        batched_velocity,
        FlowState(values["z"], values["eta"]),
        period,
        num_steps=steps,
    ).final_state
    assert batched_calls == 4 * steps

    sequential = []
    for index in range(2):
        def velocity(time, z, eta, *, sample=index):  # type: ignore[no-untyped-def]
            return field(
                time,
                z,
                eta,
                values["composition_context"][sample],
                atomic_numbers=values["atomic_numbers"][sample],
                atom_features=values["atom_features"][sample],
                fractional_coordinates=values["fractional_coordinates"][sample],
                lattice=values["lattice"][sample],
                bounded_lattice_coordinates=values[
                    "bounded_lattice_coordinates"
                ][sample],
                jacobian_cartesian=values["jacobian_cartesian"][sample],
            )

        sequential.append(
            integrate_rk4(
                velocity,
                FlowState(values["z"][index], values["eta"][index]),
                period,
                num_steps=steps,
            ).final_state
        )
    torch.testing.assert_close(
        packed.z,
        torch.stack([item.z for item in sequential]),
        rtol=1.0e-11,
        atol=1.0e-11,
    )
    torch.testing.assert_close(
        packed.eta,
        torch.stack([item.eta for item in sequential]),
        rtol=1.0e-11,
        atol=1.0e-11,
    )
