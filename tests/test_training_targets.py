from __future__ import annotations

from fractions import Fraction

import numpy as np
import pytest

from crystalatlas.decoder import CrystalScaffold, OrbitAssignment
from crystalatlas.errors import InvariantViolation
from crystalatlas.geometry import ContinuousCover, IntrinsicCoverLayout
from crystalatlas.training_targets import lift_exact_fixture_target


def _metric_for(scaffold: CrystalScaffold, latent: np.ndarray) -> np.ndarray:
    layout = IntrinsicCoverLayout(ContinuousCover.from_exact_scaffold(scaffold))
    return layout.chart.decode_latent(latent)


def _shift(
    parameter: tuple[Fraction, Fraction, Fraction],
    displacement: tuple[Fraction, Fraction, Fraction],
) -> tuple[Fraction, Fraction, Fraction]:
    return tuple(
        (value + delta) % 1
        for value, delta in zip(parameter, displacement, strict=True)
    )


def test_p1_lift_is_translation_and_orbit_permutation_canonical() -> None:
    parameters = (
        (Fraction(2, 11), Fraction(3, 13), Fraction(5, 17)),
        (Fraction(7, 19), Fraction(11, 23), Fraction(13, 29)),
        (Fraction(17, 31), Fraction(19, 37), Fraction(23, 41)),
    )
    scaffold = CrystalScaffold.from_mapping(
        nu=1,
        space_group="P1",
        reduced_composition={8: 2, 14: 1},
        orbits=(
            OrbitAssignment("1a", 14, parameters[2]),
            OrbitAssignment("1a", 8, parameters[1]),
            OrbitAssignment("1a", 8, parameters[0]),
        ),
    )
    metric = _metric_for(scaffold, np.linspace(-0.12, 0.13, 6))
    target = lift_exact_fixture_target(scaffold, metric)

    displacement = (Fraction(1, 7), Fraction(2, 9), Fraction(3, 10))
    shifted = CrystalScaffold.from_mapping(
        nu=1,
        space_group="P1",
        reduced_composition={14: 1, 8: 2},
        orbits=(
            OrbitAssignment("1a", 8, _shift(parameters[0], displacement)),
            OrbitAssignment("1a", 14, _shift(parameters[2], displacement)),
            OrbitAssignment("1a", 8, _shift(parameters[1], displacement)),
        ),
    )
    equivalent = lift_exact_fixture_target(shifted, metric)

    assert target.canonical_scaffold == equivalent.canonical_scaffold
    assert tuple(orbit.species for orbit in target.canonical_scaffold.orbits) == (
        8,
        8,
        14,
    )
    assert target.canonical_scaffold.orbits[0].parameter == (Fraction(0),) * 3
    np.testing.assert_allclose(target.lattice_latent, equivalent.lattice_latent)
    np.testing.assert_allclose(
        target.intrinsic_coordinates, equivalent.intrinsic_coordinates
    )
    assert not target.metric.flags.writeable
    assert not target.lattice_latent.flags.writeable
    assert not target.intrinsic_coordinates.flags.writeable


def test_p_minus_one_lift_uses_sampler_rows_and_exact_teacher_forcing() -> None:
    first = (Fraction(2, 11), Fraction(3, 13), Fraction(5, 17))
    second = (Fraction(7, 19), Fraction(11, 23), Fraction(13, 29))
    scaffold = CrystalScaffold.from_mapping(
        nu=2,
        space_group="P-1",
        reduced_composition={8: 1, 14: 1},
        orbits=(
            OrbitAssignment("2i", 14, second),
            OrbitAssignment("2i", 8, first),
        ),
    )
    metric = _metric_for(scaffold, np.linspace(-0.1, 0.1, 6))
    target = lift_exact_fixture_target(scaffold, metric)
    inverted = CrystalScaffold.from_mapping(
        nu=2,
        space_group="P-1",
        reduced_composition={14: 1, 8: 1},
        orbits=(
            OrbitAssignment("2i", 8, tuple((-value) % 1 for value in first)),
            OrbitAssignment("2i", 14, tuple((-value) % 1 for value in second)),
        ),
    )
    equivalent = lift_exact_fixture_target(inverted, metric)

    assert target.canonical_scaffold == equivalent.canonical_scaffold
    assert tuple(row.label for row in target.cover_decisions.rows) == (
        "1a",
        "1b",
        "1c",
        "1d",
        "1e",
        "1f",
        "1g",
        "1h",
        "2i",
    )
    trace = target.cover_decisions.row_decisions
    assert trace[0].residual_before == (2, 2)
    assert trace[-1].selected_action == target.cover_decisions.occupation.actions[-1]
    assert trace[-1].selected_action.counts == (1, 1)
    assert trace[-1].residual_after == (0, 0)
    assert all(
        decision.legal_actions[decision.selected_action_index]
        == decision.selected_action
        for decision in trace
    )
    assert target.cover_decisions.nu_legal_mask[target.cover_decisions.nu_target_index]
    assert target.cover_decisions.group_legal_mask[
        target.cover_decisions.group_target_index
    ]
    np.testing.assert_allclose(
        target.layout.chart.decode_latent(target.lattice_latent), metric
    )


def test_lift_rejects_nonprimitive_hidden_translation_target() -> None:
    scaffold = CrystalScaffold.from_mapping(
        nu=2,
        space_group="P1",
        reduced_composition={8: 1},
        orbits=(
            OrbitAssignment("1a", 8, (Fraction(0), Fraction(0), Fraction(0))),
            OrbitAssignment(
                "1a", 8, (Fraction(1, 2), Fraction(0), Fraction(0))
            ),
        ),
    )
    metric = _metric_for(scaffold, np.zeros(6))
    with pytest.raises(InvariantViolation, match="hidden translations"):
        lift_exact_fixture_target(scaffold, metric)


def test_lift_does_not_accept_raw_structure_arrays() -> None:
    with pytest.raises(TypeError, match="CrystalScaffold"):
        lift_exact_fixture_target(np.eye(3), np.eye(3))  # type: ignore[arg-type]
