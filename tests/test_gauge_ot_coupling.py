"""Gauge-aligned minibatch OT coupling of the batched CFM objective (research R1).

The CFM loss is dominated by the conditional variance of its target velocity.
``coupling="gauge-ot-v1"`` pairs the batch's base samples with the records by an
assignment on the gauge-quotient cost and pushes every path by the residual
action that minimizes it.  What must hold exactly: the legacy path is bitwise
unchanged under ``"independent"``; both endpoint marginals are preserved (the
base samples are only permuted, so every pushed path still starts at a drawn
base sample, and each record's chosen action is uniform over the group, so the
pushed endpoints keep the symmetrized target law); and the assigned cost is at
most the identity pairing's.

Proposition 2.9d is a statement about the records of one cover, the variable the
geometry field is conditioned on.  A batch that mixes covers is therefore
coupled block by block, and every block permutes only its own base samples.
The candidate actions come from the digit-coded action table
(``ResidualActionTable``): the whole group when it has at most
``coupling_candidates`` digit rows, i.i.d. uniform draws otherwise.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import replace
from fractions import Fraction
from itertools import permutations

import pytest

torch = pytest.importorskip("torch")

import numpy as np  # noqa: E402
from test_learned import _tiny_config  # noqa: E402

from crystalatlas.catalogue import MINIMAL_CATALOGUE  # noqa: E402
from crystalatlas.decoder import CrystalScaffold, OrbitAssignment  # noqa: E402
from crystalatlas.errors import DomainError  # noqa: E402
from crystalatlas.flow import FlowState, canonicalize_torus, sample_product_base  # noqa: E402
from crystalatlas.geometry import ContinuousCover  # noqa: E402
from crystalatlas.learned import (  # noqa: E402
    COUPLINGS,
    CrystalAtlasModel,
    LearnedCrystalGenerator,
)
from crystalatlas.residual_gauge import (  # noqa: E402
    ResidualActionTable,
    cover_residual_gauge,
)

F64 = torch.float64


def _records():  # type: ignore[no-untyped-def]
    return tuple(
        CrystalScaffold.from_mapping(
            nu=2,
            space_group="P-1",
            reduced_composition={8: 1},
            orbits=(OrbitAssignment("2i", 8, parameter),),
        )
        for parameter in (
            (Fraction(1, 7), Fraction(2, 7), Fraction(3, 7)),
            (Fraction(5, 11), Fraction(3, 11), Fraction(9, 11)),
            (Fraction(1, 5), Fraction(4, 5), Fraction(2, 5)),
        )
    )


def _setup():  # type: ignore[no-untyped-def]
    torch.manual_seed(11)
    model = CrystalAtlasModel(_tiny_config(), catalogue=MINIMAL_CATALOGUE).double()
    generator = LearnedCrystalGenerator(model, allow_untrained=True)
    scaffolds = _records()
    layout = model.layout(ContinuousCover.from_exact_scaffold(scaffolds[0]))
    metrics = tuple(
        layout.chart.decode_latent(np.full(layout.lattice_dimension, 0.05 * (k + 1)))
        for k in range(len(scaffolds))
    )
    return generator, scaffolds, metrics, layout


def _objective(generator, scaffolds, metrics, seed, **kwargs):  # type: ignore[no-untyped-def]
    batch = 8
    picks = [k % len(scaffolds) for k in range(batch)]
    return generator.geometry_cfm_objective_batch(
        tuple(scaffolds[k] for k in picks),
        tuple(metrics[k] for k in picks),
        time=torch.linspace(0.05, 0.95, batch, dtype=F64),
        sigma_z=0.05,
        sigma_eta=0.03,
        generator=torch.Generator().manual_seed(seed),
        **kwargs,
    )


def _torus_multiset_equal(left: torch.Tensor, right: torch.Tensor, tol: float = 1e-9) -> bool:
    unmatched = list(range(right.shape[0]))
    for row in left:
        for candidate in unmatched:
            gap = torch.remainder(row - right[candidate] + 0.5, 1.0) - 0.5
            if float(gap.abs().max()) <= tol:
                unmatched.remove(candidate)
                break
        else:
            return False
    return not unmatched


def test_independent_coupling_is_bitwise_legacy_and_options_are_validated() -> None:
    generator, scaffolds, metrics, _ = _setup()
    legacy = _objective(generator, scaffolds, metrics, 3)
    explicit = _objective(generator, scaffolds, metrics, 3, coupling="independent")
    assert torch.equal(legacy.loss.per_sample, explicit.loss.per_sample)
    assert torch.equal(legacy.path.state.eta, explicit.path.state.eta)
    assert legacy.coupling_stats is None and explicit.coupling_stats is None
    assert COUPLINGS == ("independent", "gauge-ot-v1")
    with pytest.raises(DomainError, match="coupling must be"):
        _objective(generator, scaffolds, metrics, 3, coupling="sinkhorn-v9")
    with pytest.raises(DomainError, match="positive integer"):
        _objective(
            generator, scaffolds, metrics, 3, coupling="gauge-ot-v1", coupling_candidates=0
        )


def test_aligned_objective_is_finite_differentiable_and_cheaper_than_identity() -> None:
    generator, scaffolds, metrics, _ = _setup()
    aligned = _objective(generator, scaffolds, metrics, 5, coupling="gauge-ot-v1")
    assert aligned.coupling_stats is not None
    assert aligned.coupling_stats["assigned_mean_cost"] <= (
        aligned.coupling_stats["identity_mean_cost"] + 1e-12
    )
    # One cover, and its group {+I, -I} has 2 <= 8 digit rows: one block, exact quotient.
    assert aligned.coupling_stats["block_count"] == 1.0
    assert aligned.coupling_stats["exact_quotient_fraction"] == 1.0
    assert torch.isfinite(aligned.loss.total)
    aligned.loss.total.backward()
    state = aligned.path.state.eta
    assert bool((state >= 0).all()) and bool((state < 1).all())
    assert bool((aligned.path.torus_delta.abs() <= 0.5 + 1e-12).all())


@pytest.mark.parametrize(("candidates", "exhaustive"), [(3, True), (1, False)])
def test_coupling_permutes_bases_and_chooses_uniform_actions(
    candidates: int, exhaustive: bool
) -> None:
    """P-1 2i carries the sign gauge {+I, -I}: the chosen action must be a fair coin.

    The pushed start of every path must be one of the drawn base samples (the
    base marginal is a permutation), and the chosen actions must be uniform over
    the group (the endpoint marginal stays the symmetrized law).  With three
    candidates the two-element group is enumerated; with one the candidate is a
    uniform draw.
    """

    generator, scaffolds, metrics, layout = _setup()
    assert generator.model.scientific_config.model.geometry_residual_gauge == "relabel-v1"
    device, dtype = torch.device("cpu"), F64
    bridge = generator._bridge(layout, device, dtype)
    period = bridge.period_basis.to(dtype=dtype)
    batch = 12
    flips = 0
    total = 0
    cheaper = 0
    identity = torch.eye(layout.intrinsic_dimension, dtype=dtype)
    for seed in range(120):
        draw = torch.Generator().manual_seed(500 + seed)
        base = sample_product_base(
            layout.lattice_dimension, period, batch_shape=(batch,), dtype=dtype, generator=draw
        )
        endpoint = FlowState(
            0.1 * torch.randn(batch, layout.lattice_dimension, generator=draw, dtype=dtype),
            torch.rand(batch, layout.intrinsic_dimension, generator=draw, dtype=dtype),
        )
        actions, shifts, all_rows = generator._residual_action_candidates(
            layout, batch, candidates, generator=draw, device=device
        )
        assert all_rows is exhaustive and shifts is None
        assert actions.shape[:2] == ((1, 2) if exhaustive else (batch, candidates))
        order, eta_base, chosen, chosen_shifts, sums = generator._gauge_aligned_coupling(
            bridge, base, endpoint, actions, shifts, period
        )
        assert chosen_shifts is None
        assert sorted(order.tolist()) == list(range(batch))
        for action in chosen:
            assert torch.equal(action, identity) or torch.equal(action, -identity)
            flips += int(torch.equal(action, -identity))
            total += 1
        pushed = torch.stack(
            [canonicalize_torus(eta_base[j] @ chosen[j], period) for j in range(batch)]
        )
        # Record j starts at exactly the base sample it was paired with.
        gap = torch.remainder(pushed - base.eta[order] + 0.5, 1.0) - 0.5
        assert float(gap.abs().max()) <= 1e-9
        assert _torus_multiset_equal(pushed, base.eta)
        cheaper += int(sums[0] <= sums[1] + 1e-12)
    assert cheaper == 120
    fraction = flips / total
    assert 0.44 <= fraction <= 0.56, fraction  # 1440 draws: SE 0.013


def _three_orbit_setup():  # type: ignore[no-untyped-def]
    """P1 with three identical 1a orbits: the relabelling group S_3 (six actions)."""

    config = _tiny_config()
    config = replace(config, domain=replace(config.domain, max_formula_units=3))
    torch.manual_seed(3)
    model = CrystalAtlasModel(config, catalogue=MINIMAL_CATALOGUE).double()
    generator = LearnedCrystalGenerator(model, allow_untrained=True)
    scaffold = CrystalScaffold.from_mapping(
        nu=3,
        space_group="P1",
        reduced_composition={6: 1},
        orbits=tuple(
            OrbitAssignment("1a", 6, tuple(Fraction(value, 100) for value in position))
            for position in ((0, 0, 0), (17, 23, 31), (41, 43, 47))
        ),
    )
    layout = model.layout(ContinuousCover.from_exact_scaffold(scaffold))
    return generator, layout


def _action_key(matrix, shift=None):  # type: ignore[no-untyped-def]
    parts = [np.asarray(matrix, dtype=np.float64).ravel()]
    if shift is not None:
        parts.append(np.mod(np.asarray(shift, dtype=np.float64), 1.0))
    return tuple(np.round(np.concatenate(parts), 9) + 0.0)


def test_fisher_yates_digits_are_a_bijection_onto_the_relabellings() -> None:
    _, layout = _three_orbit_setup()
    table = ResidualActionTable(layout, "relabel-v1")
    assert table.moduli == (3, 2) and table.cardinality == 6 and not table.has_shift
    rows = table.all_rows()
    assert rows.shape == (6, 2)
    assert {table.permutation(row) for row in rows.tolist()} == set(permutations(range(3)))
    matrices, shifts = table.actions(rows)
    assert shifts is None and matrices.shape == (6, 6, 6)
    for row, matrix in zip(rows.tolist(), matrices, strict=True):
        np.testing.assert_array_equal(
            matrix, layout.residual_action_matrix(table.permutation(row))
        )
    assert len({_action_key(matrix) for matrix in matrices}) == 6
    with pytest.raises(DomainError, match="outside the table"):
        table.action((3, 0))
    with pytest.raises(DomainError, match="integer"):
        table.actions(np.zeros((2, 3), dtype=np.int64))
    with pytest.raises(DomainError, match="residual gauge must be"):
        ResidualActionTable(layout, "relabel-v0")


@pytest.mark.parametrize(
    ("gauge", "improper", "order"), [("normalizer-v2", False, 4), ("normalizer-v3", True, 8)]
)
def test_action_table_covers_the_certified_group_with_equal_multiplicity(
    gauge: str, improper: bool, order: int
) -> None:
    """All digit rows give exactly the certified group, each element equally often,
    and i.i.d. digit rows give the uniform law on it (the per-record sampler's law)."""

    from test_residual_gauge_objective import _models

    model, toy = _models(gauge)
    generator = LearnedCrystalGenerator(model, allow_untrained=True)
    layout = model.layout(toy.cover)
    group = cover_residual_gauge(layout, improper=improper).all_eta_actions()
    assert len(group) == order
    keys = {_action_key(matrix, shift): index for index, (matrix, shift) in enumerate(group)}
    table = generator._residual_action_table(layout)
    assert table is generator._residual_action_table(layout) and table.kind == gauge
    matrices, shifts = table.actions(table.all_rows())
    assert shifts is not None and table.has_shift
    counts = Counter(keys[_action_key(m, s)] for m, s in zip(matrices, shifts, strict=True))
    assert set(counts) == set(range(order))
    assert set(counts.values()) == {table.cardinality // order}

    device = torch.device("cpu")
    # At least as many candidates as digit rows: the whole table, no random draw.
    draw = torch.Generator().manual_seed(9)
    state = draw.get_state()
    actions, action_shifts, exhaustive = generator._residual_action_candidates(
        layout, 5, table.cardinality, generator=draw, device=device
    )
    assert exhaustive and torch.equal(draw.get_state(), state)
    assert actions.shape == (1, table.cardinality, table.dimension, table.dimension)
    assert action_shifts is not None and actions.dtype == torch.float64
    np.testing.assert_array_equal(actions[0].numpy(), matrices)
    # Fewer candidates: i.i.d. uniform draws.
    actions, action_shifts, exhaustive = generator._residual_action_candidates(
        layout, 400, 2, generator=draw, device=device
    )
    assert not exhaustive and actions.shape[:2] == (400, 2)
    assert action_shifts is not None
    sampled = Counter(
        keys[_action_key(m, s)]
        for m, s in zip(
            actions.reshape(800, table.dimension, table.dimension).numpy(),
            action_shifts.reshape(800, table.dimension).numpy(),
            strict=True,
        )
    )
    assert set(sampled) == set(range(order))
    chi_square = sum((value - 800 / order) ** 2 / (800 / order) for value in sampled.values())
    # 99.9% quantiles of chi^2 with 3 and 7 degrees of freedom.
    assert chi_square < {4: 16.27, 8: 24.32}[order]


def test_iid_candidates_keep_the_chosen_action_uniform_on_a_larger_group() -> None:
    """S_3 with two i.i.d. candidates per record: the kept action is still uniform."""

    generator, layout = _three_orbit_setup()
    device, dtype = torch.device("cpu"), F64
    bridge = generator._bridge(layout, device, dtype)
    period = bridge.period_basis.to(dtype=dtype)
    table = generator._residual_action_table(layout)
    group = {_action_key(matrix): index for index, matrix in enumerate(
        table.actions(table.all_rows())[0]
    )}
    assert len(group) == 6
    batch = 10
    counts: Counter[int] = Counter()
    for seed in range(120):
        draw = torch.Generator().manual_seed(900 + seed)
        base = sample_product_base(
            layout.lattice_dimension, period, batch_shape=(batch,), dtype=dtype, generator=draw
        )
        endpoint = FlowState(
            0.1 * torch.randn(batch, layout.lattice_dimension, generator=draw, dtype=dtype),
            canonicalize_torus(
                torch.rand(batch, layout.intrinsic_dimension, generator=draw, dtype=dtype)
                @ period,
                period,
            ),
        )
        actions, shifts, exhaustive = generator._residual_action_candidates(
            layout, batch, 2, generator=draw, device=device
        )
        assert not exhaustive and shifts is None
        order, eta_base, chosen, _, sums = generator._gauge_aligned_coupling(
            bridge, base, endpoint, actions, shifts, period
        )
        assert sums[0] <= sums[1] + 1e-12
        pushed = torch.stack(
            [canonicalize_torus(eta_base[j] @ chosen[j], period) for j in range(batch)]
        )
        torch.testing.assert_close(
            canonicalize_torus(pushed - base.eta[order] + 0.5 * period.sum(0), period),
            (0.5 * period.sum(0)).expand_as(pushed),
            rtol=0,
            atol=1e-9,
        )
        counts.update(group[_action_key(action.numpy())] for action in chosen)
    assert set(counts) == set(range(6))
    chi_square = sum((value - 200.0) ** 2 / 200.0 for value in counts.values())
    assert chi_square < 20.52  # 99.9% quantile of chi^2 with 5 degrees of freedom


def test_mixed_cover_batches_are_coupled_within_each_cover() -> None:
    """Two covers of one topology (2i oxygen, 2i carbon): each block permutes only
    its own base samples, so the base marginal of every cover is the drawn base."""

    generator, scaffolds, metrics, layout = _setup()
    carbon = CrystalScaffold.from_mapping(
        nu=2,
        space_group="P-1",
        reduced_composition={6: 1},
        orbits=(OrbitAssignment("2i", 6, (Fraction(2, 9), Fraction(4, 9), Fraction(7, 9))),),
    )
    carbon_layout = generator.model.layout(ContinuousCover.from_exact_scaffold(carbon))
    device, dtype = torch.device("cpu"), F64
    period = generator._bridge(layout, device, dtype).period_basis.to(dtype=dtype)
    pattern = [0, 1, 1, 0, 0, 1, 0, 1, 1, 1]
    positions = {
        kind: [i for i, value in enumerate(pattern) if value == kind] for kind in (0, 1)
    }
    batch = len(pattern)
    for seed in range(20):
        draw = torch.Generator().manual_seed(40 + seed)
        base = sample_product_base(
            layout.lattice_dimension, period, batch_shape=(batch,), dtype=dtype, generator=draw
        )
        endpoint = FlowState(
            0.1 * torch.randn(batch, layout.lattice_dimension, generator=draw, dtype=dtype),
            torch.rand(batch, layout.intrinsic_dimension, generator=draw, dtype=dtype),
        )
        blocks = [
            (
                positions[kind],
                generator._bridge(block_layout, device, dtype),
                *generator._residual_action_candidates(
                    block_layout, len(positions[kind]), 8, generator=draw, device=device
                ),
            )
            for kind, block_layout in ((0, layout), (1, carbon_layout))
        ]
        new_base, actions, shifts, stats = generator._gauge_aligned_batch(
            blocks, base, endpoint, period
        )
        assert shifts is None and stats["block_count"] == 2.0
        assert stats["exact_quotient_fraction"] == 1.0
        assert stats["assigned_mean_cost"] <= stats["identity_mean_cost"] + 1e-12
        for index in positions.values():
            assert torch.equal(
                torch.sort(new_base.z[index], dim=0).values,
                torch.sort(base.z[index], dim=0).values,
            )
            pushed = torch.stack(
                [canonicalize_torus(new_base.eta[j] @ actions[j], period) for j in index]
            )
            assert _torus_multiset_equal(pushed, base.eta[index])
    with pytest.raises(ValueError, match="partition"):
        generator._gauge_aligned_batch(blocks[:1], base, endpoint, period)

    # The objective forms the same blocks from the covers of its records.
    picks = [scaffolds[0], carbon, scaffolds[1], carbon, scaffolds[2], carbon]
    result = generator.geometry_cfm_objective_batch(
        tuple(picks),
        tuple(metrics[k % len(metrics)] for k in range(len(picks))),
        time=torch.linspace(0.1, 0.9, len(picks), dtype=F64),
        sigma_z=0.05,
        sigma_eta=0.03,
        generator=torch.Generator().manual_seed(2),
        coupling="gauge-ot-v1",
    )
    assert result.coupling_stats is not None
    assert result.coupling_stats["block_count"] == 2.0
    assert torch.isfinite(result.loss.total)


def test_toy_training_runs_with_the_aligned_coupling() -> None:
    from types import SimpleNamespace

    from crystalatlas.flow_oracle_learned import train_on_toy

    generator, scaffolds, metrics, _ = _setup()
    toy = SimpleNamespace(scaffolds=scaffolds, metrics=metrics, sigma_z=0.05, sigma_eta=0.03)
    losses = train_on_toy(
        generator.model,
        toy,
        steps=3,
        batch_size=6,
        learning_rate=1e-3,
        seed=1,
        coupling="gauge-ot-v1",
        coupling_candidates=4,
    )
    assert len(losses) == 3 and all(np.isfinite(losses))
