"""Explicit CFM block weights ``lambda_z, lambda_eta`` (review 2026-10-29, m2).

The coordinate block of (3.14) is in A^2 per atom and the lattice block is a
dimensionless squared strain; their trade-off is a hyperparameter.  It is now
explicit, validated and recorded in the objective contract.  Unit weights are
the legacy loss bitwise, and current-state constants keep the population
minimizer (Theorem 2.9).
"""

from __future__ import annotations

from fractions import Fraction

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from test_learned import _tiny_config  # noqa: E402

from crystalatlas.catalogue import MINIMAL_CATALOGUE  # noqa: E402
from crystalatlas.decoder import CrystalScaffold, OrbitAssignment  # noqa: E402
from crystalatlas.errors import DomainError  # noqa: E402
from crystalatlas.flow import FlowVelocity, conditional_flow_matching_loss  # noqa: E402
from crystalatlas.geometry import ContinuousCover  # noqa: E402
from crystalatlas.learned import CrystalAtlasModel, LearnedCrystalGenerator  # noqa: E402
from crystalatlas.training_objectives import CFMBlockWeights  # noqa: E402


def test_loss_weights_scale_the_blocks_and_unity_is_bitwise_legacy() -> None:
    generator = torch.Generator().manual_seed(4)
    predicted = FlowVelocity(
        torch.randn(5, 2, generator=generator, dtype=torch.float64),
        torch.randn(5, 3, generator=generator, dtype=torch.float64),
    )
    target = FlowVelocity(
        torch.randn(5, 2, generator=generator, dtype=torch.float64),
        torch.randn(5, 3, generator=generator, dtype=torch.float64),
    )
    lattice_metric = torch.eye(2, dtype=torch.float64) * 1.7
    coordinate_metric = torch.eye(3, dtype=torch.float64) * 0.4
    kwargs = dict(
        current_lattice_metric=lattice_metric,
        current_coordinate_metric=coordinate_metric,
        atom_count=4,
        return_terms=True,
        reduction="none",
    )
    legacy = conditional_flow_matching_loss(predicted, target, **kwargs)
    unity = conditional_flow_matching_loss(
        predicted, target, lattice_weight=1.0, coordinate_weight=1.0, **kwargs
    )
    assert torch.equal(unity.per_sample, legacy.per_sample)
    weighted = conditional_flow_matching_loss(
        predicted, target, lattice_weight=2.0, coordinate_weight=0.5, **kwargs
    )
    torch.testing.assert_close(
        weighted.per_sample, 2.0 * legacy.lattice + 0.5 * legacy.coordinates, rtol=0, atol=1e-15
    )
    assert torch.equal(weighted.lattice, legacy.lattice)  # the reported terms stay unweighted
    assert torch.equal(weighted.coordinates, legacy.coordinates)
    for bad in (0.0, -1.0, float("inf"), float("nan"), True):
        with pytest.raises(ValueError, match="finite positive"):
            conditional_flow_matching_loss(predicted, target, lattice_weight=bad, **kwargs)


def test_block_weights_are_validated_and_recorded_only_when_non_unity() -> None:
    assert CFMBlockWeights().is_unity and CFMBlockWeights().as_tuple() == (1.0, 1.0)
    assert not CFMBlockWeights(lattice=2.0).is_unity
    assert CFMBlockWeights(1.0, 0.5).to_dict() == {
        "lattice_hex": (1.0).hex(),
        "coordinates_hex": (0.5).hex(),
    }
    for bad in (0.0, -1.0, float("inf"), True):
        with pytest.raises(DomainError, match="finite and positive"):
            CFMBlockWeights(lattice=bad)


def test_objective_weights_reach_the_per_record_loss() -> None:
    torch.manual_seed(5)
    model = CrystalAtlasModel(_tiny_config(), catalogue=MINIMAL_CATALOGUE).double()
    generator = LearnedCrystalGenerator(model, allow_untrained=True)
    scaffold = CrystalScaffold.from_mapping(
        nu=2,
        space_group="P-1",
        reduced_composition={8: 1},
        orbits=(OrbitAssignment("2i", 8, (Fraction(1, 7), Fraction(2, 7), Fraction(3, 7))),),
    )
    layout = model.layout(ContinuousCover.from_exact_scaffold(scaffold))
    target_metric = layout.chart.decode_latent(np.zeros(layout.lattice_dimension))

    def run(weights: tuple[float, float]):  # type: ignore[no-untyped-def]
        return generator.geometry_cfm_objective(
            scaffold,
            target_metric,
            time=torch.tensor(0.37, dtype=torch.float64),
            sigma_z=0.05,
            sigma_eta=0.03,
            generator=torch.Generator().manual_seed(101),
            block_weights=weights,
        )

    legacy = run((1.0, 1.0))
    weighted = run((3.0, 0.25))
    assert torch.equal(weighted.loss.lattice, legacy.loss.lattice)
    assert torch.equal(weighted.loss.coordinates, legacy.loss.coordinates)
    torch.testing.assert_close(
        weighted.loss.total,
        3.0 * legacy.loss.lattice + 0.25 * legacy.loss.coordinates,
        rtol=1e-14,
        atol=0,
    )
