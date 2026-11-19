"""Basis-free lattice-velocity head (review finding M3).

The legacy head writes ``v_z = A z + r`` into the first ``d`` of six slots, so
slot ``j`` means a different metric direction per crystal family and chart
basis.  The ``commutant-v2`` head reads and writes canonical Sym(3)
coordinates through the chart's commutant basis ``C`` (6 x d, orthonormal):
input ``C b`` and output ``C^T A C z + C^T r``.  It keeps the linear-growth bound
``|v_z| <= ||A|| |z| + ||r||`` (theory F9) and is exactly equivariant under any
orthogonal change of commutant basis ``C -> C Q``.
"""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from crystalatlas.full_catalogue import build_space_group_setting  # noqa: E402
from crystalatlas.lattice import InvariantSPDChart, _symmetric_basis_3  # noqa: E402
from crystalatlas.nn.field import IntrinsicFieldConfig, IntrinsicGeometryField  # noqa: E402

pytestmark = pytest.mark.ml


def _field(head: str) -> IntrinsicGeometryField:
    torch.manual_seed(7)
    config = IntrinsicFieldConfig(
        atom_feature_dim=1,
        composition_dim=4,
        species_embedding_dim=4,
        max_lattice_dim=6,
        num_time_frequencies=1,
        scalar_dim=8,
        vector_dim=4,
        num_heads=1,
        num_layers=1,
        radial_basis_dim=3,
        cutoff=1.5,
        lattice_head=head,
    )
    return IntrinsicGeometryField(config).double().eval()


def _evaluate(field, z, bounded, basis):  # type: ignore[no-untyped-def]
    return field(
        torch.tensor(0.4, dtype=torch.float64),
        z,
        torch.empty(0, dtype=torch.float64),
        torch.zeros(4, dtype=torch.float64),
        atomic_numbers=torch.tensor([8]),
        atom_features=torch.zeros(1, 1, dtype=torch.float64),
        fractional_coordinates=torch.zeros(1, 3, dtype=torch.float64),
        lattice=torch.eye(3, dtype=torch.float64) * 3.0,
        bounded_lattice_coordinates=bounded,
        jacobian_cartesian=torch.empty(1, 3, 0, dtype=torch.float64),
        lattice_commutant_basis=basis,
    ).lattice_velocity


def test_canonical_commutant_basis_is_orthonormal_and_reconstructs_the_chart() -> None:
    canonical = _symmetric_basis_3()
    for number in (1, 12, 47, 123, 166, 194, 225):
        chart = InvariantSPDChart.from_setting(build_space_group_setting(number), bound=2.0)
        basis = chart.canonical_commutant_basis
        assert basis.shape == (6, chart.dimension)
        np.testing.assert_allclose(basis.T @ basis, np.eye(chart.dimension), atol=1e-12)
        for j, matrix in enumerate(chart.basis):
            rebuilt = sum(basis[k, j] * canonical[k] for k in range(6))
            np.testing.assert_allclose(rebuilt, matrix, atol=1e-12)


def test_commutant_head_reduces_to_the_legacy_head_on_the_first_coordinates() -> None:
    legacy = _field("truncated-v1")
    commutant = _field("commutant-v2")
    commutant.load_state_dict(legacy.state_dict())
    z = torch.tensor([0.7, -1.3], dtype=torch.float64)
    bounded = torch.tanh(z)
    first = torch.eye(6, dtype=torch.float64)[:, :2]
    torch.testing.assert_close(
        _evaluate(commutant, z, bounded, first),
        _evaluate(legacy, z, bounded, None),
        rtol=0.0,
        atol=0.0,
    )


def test_commutant_head_is_exactly_equivariant_under_a_change_of_basis() -> None:
    field = _field("commutant-v2")
    generator = torch.Generator().manual_seed(3)
    basis, _ = torch.linalg.qr(torch.randn(6, 2, generator=generator, dtype=torch.float64))
    rotation, _ = torch.linalg.qr(torch.randn(2, 2, generator=generator, dtype=torch.float64))
    z = torch.tensor([0.7, -1.3], dtype=torch.float64)
    bounded = torch.tanh(z)
    velocity = _evaluate(field, z, bounded, basis)
    rotated = _evaluate(field, rotation.T @ z, rotation.T @ bounded, basis @ rotation)
    torch.testing.assert_close(rotated, rotation.T @ velocity, rtol=1e-12, atol=1e-12)


def test_commutant_head_keeps_the_linear_growth_bound() -> None:
    field = _field("commutant-v2")
    basis, _ = torch.linalg.qr(
        torch.randn(6, 2, generator=torch.Generator().manual_seed(4), dtype=torch.float64)
    )
    config = field.config
    for scale in (1.0, 10.0, 1e3):
        z = scale * torch.tensor([0.6, -0.8], dtype=torch.float64)
        velocity = _evaluate(field, z, torch.tanh(z), basis)
        bound = config.lattice_linear_growth_bound * float(z.norm()) + (
            config.lattice_residual_bound * 6**0.5
        )
        assert float(velocity.norm()) <= bound


def test_commutant_head_requires_a_basis_and_six_slots() -> None:
    with pytest.raises(ValueError, match="six canonical"):
        IntrinsicGeometryField(
            IntrinsicFieldConfig(
                atom_feature_dim=1, max_lattice_dim=4, lattice_head="commutant-v2"
            )
        )
    field = _field("commutant-v2")
    z = torch.tensor([0.7, -1.3], dtype=torch.float64)
    with pytest.raises(ValueError, match="requires lattice_commutant_basis"):
        _evaluate(field, z, torch.tanh(z), None)


def test_commutant_head_trains_through_the_single_and_packed_objectives() -> None:
    from dataclasses import replace
    from fractions import Fraction

    from test_learned import _tiny_config

    from crystalatlas.catalogue import MINIMAL_CATALOGUE
    from crystalatlas.decoder import CrystalScaffold, OrbitAssignment
    from crystalatlas.geometry import ContinuousCover
    from crystalatlas.learned import CrystalAtlasModel, LearnedCrystalGenerator

    base = _tiny_config()
    assert base.model.geometry_lattice_head == "commutant-v2"  # the default since 2026-10-29
    config = replace(base, model=replace(base.model, geometry_lattice_head="commutant-v2"))
    assert config.fingerprint == base.fingerprint
    legacy = replace(base, model=replace(base.model, geometry_lattice_head="truncated-v1"))
    assert legacy.fingerprint != base.fingerprint
    assert "geometry_lattice_head" not in legacy.to_dict()["model"]  # historical payload
    assert config.to_dict()["model"]["geometry_lattice_head"] == "commutant-v2"
    torch.manual_seed(29)
    model = CrystalAtlasModel(config, catalogue=MINIMAL_CATALOGUE).double()
    assert model.geometry_field.config.lattice_head == "commutant-v2"
    generator = LearnedCrystalGenerator(model, allow_untrained=True)
    scaffolds = tuple(
        CrystalScaffold.from_mapping(
            nu=2,
            space_group="P-1",
            reduced_composition={8: 1},
            orbits=(OrbitAssignment("2i", 8, parameter),),
        )
        for parameter in (
            (Fraction(1, 7), Fraction(2, 7), Fraction(3, 7)),
            (Fraction(2, 9), Fraction(1, 9), Fraction(4, 9)),
        )
    )
    layout = model.layout(ContinuousCover.from_exact_scaffold(scaffolds[0]))
    metrics = (
        layout.chart.decode_latent(np.zeros(layout.lattice_dimension)),
        layout.chart.decode_latent(np.full(layout.lattice_dimension, 0.05)),
    )
    single = generator.geometry_cfm_objective(
        scaffolds[0],
        metrics[0],
        time=torch.tensor(0.37, dtype=torch.float64),
        sigma_z=0.05,
        sigma_eta=0.03,
        generator=torch.Generator().manual_seed(101),
    )
    packed = generator.geometry_cfm_objective_batch(
        scaffolds,
        metrics,
        time=torch.tensor([0.25, 0.75], dtype=torch.float64),
        sigma_z=0.05,
        sigma_eta=0.03,
        generator=torch.Generator().manual_seed(113),
    )
    (single.loss.total + packed.loss.total).backward()
    gradients = [p.grad for p in model.geometry_field.lattice_linear_head.parameters()]
    assert all(g is not None and bool(torch.isfinite(g).all().item()) for g in gradients)
