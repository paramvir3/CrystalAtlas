"""Exact model log density by the divergence integral (research R2).

``log_density_transport`` integrates the learned ODE backward from ``t = 1`` with
the divergence of the field and returns ``log rho_0(q_0) - int div v``, the
instantaneous change of variables on ``R^{d_z} x`` the flat torus.  The exact
Fourier oracle has the exact density of the same law, so the transport run on
the oracle's own velocity field must reproduce ``oracle.log_density(1, q_1)`` to
quadrature accuracy.  This is the certificate behind likelihood ranking without
an energy model.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

import numpy as np  # noqa: E402
from test_flow_oracle import one_orbit_toy, two_parameter_toy  # noqa: E402

from crystalatlas.flow import (  # noqa: E402
    FlowState,
    base_log_density,
    log_density_transport,
)

F64 = torch.float64


@pytest.mark.parametrize("toy", [one_orbit_toy, two_parameter_toy])
def test_transport_on_the_oracle_field_reproduces_the_exact_density(toy) -> None:  # type: ignore[no-untyped-def]
    oracle = toy()
    generator = torch.Generator().manual_seed(5)
    batch = 24
    paths = oracle.sample_paths(torch.ones(batch, dtype=F64), generator=generator)
    final = FlowState(paths.state.z, paths.state.eta)
    exact = oracle.log_density(torch.tensor(1.0, dtype=F64), final.z, final.eta)

    def field(time, z, eta):  # type: ignore[no-untyped-def]
        return oracle.velocity(torch.as_tensor(time, dtype=F64).reshape(()), z, eta)

    start, integral = log_density_transport(field, final, oracle.period_basis, num_steps=192)
    transported = base_log_density(start.z, oracle.period_basis) - integral
    torch.testing.assert_close(transported, exact, rtol=0, atol=2e-3)
    # The base point reached backward has the base's log density by construction.
    torch.testing.assert_close(
        base_log_density(start.z, oracle.period_basis),
        oracle.base_log_density(start.z, start.eta),
        rtol=0,
        atol=1e-12,
    )
    # Hutchinson is unbiased: averaged over probes it agrees within its noise.
    _, estimate = log_density_transport(
        field,
        final,
        oracle.period_basis,
        num_steps=64,
        divergence="hutchinson",
        probes=48,
        generator=torch.Generator().manual_seed(1),
    )
    assert abs(float((estimate - integral).mean())) < 0.08


@pytest.mark.parametrize("toy", [one_orbit_toy, two_parameter_toy])
def test_replicated_trace_equals_the_coordinate_loop(toy) -> None:  # type: ignore[no-untyped-def]
    """``exact-batched`` is the same trace as ``exact``: row k of the Jacobian of member
    b is read from copy (b, k), because a row's velocity depends on that row only."""

    oracle = toy()
    paths = oracle.sample_paths(
        torch.ones(5, dtype=F64), generator=torch.Generator().manual_seed(11)
    )
    final = FlowState(paths.state.z, paths.state.eta)
    dimension = final.z.shape[1] + final.eta.shape[1]

    def field(time, z, eta):  # type: ignore[no-untyped-def]
        return oracle.velocity(torch.as_tensor(time, dtype=F64).reshape(()), z, eta)

    start, integral = log_density_transport(field, final, oracle.period_basis, num_steps=8)
    assert float(integral.abs().max()) > 1e-3  # the comparison is not vacuous
    # None: every coordinate in one pass; 1: one pass per coordinate; an uneven
    # split when the dimension allows one; a chunk larger than the dimension.
    for row_chunk in (None, 1, max(1, dimension - 1), dimension + 3):
        batched_start, batched = log_density_transport(
            field,
            final,
            oracle.period_basis,
            num_steps=8,
            divergence="exact-batched",
            row_chunk=row_chunk,
        )
        torch.testing.assert_close(batched, integral, rtol=0, atol=1e-11)
        torch.testing.assert_close(batched_start.z, start.z, rtol=0, atol=1e-12)
        torch.testing.assert_close(batched_start.eta, start.eta, rtol=0, atol=1e-12)


def test_transport_validates_its_inputs() -> None:
    oracle = one_orbit_toy()
    state = FlowState(torch.zeros(2, 1, dtype=F64), torch.zeros(2, 1, dtype=F64))

    def field(time, z, eta):  # type: ignore[no-untyped-def]
        return oracle.velocity(torch.as_tensor(time, dtype=F64).reshape(()), z, eta)

    with pytest.raises(ValueError, match="num_steps"):
        log_density_transport(field, state, oracle.period_basis, num_steps=0)
    with pytest.raises(ValueError, match="divergence"):
        log_density_transport(field, state, oracle.period_basis, num_steps=2, divergence="x")
    with pytest.raises(ValueError, match="batched"):
        log_density_transport(
            field, FlowState(torch.zeros(1, dtype=F64), torch.zeros(1, dtype=F64)),
            oracle.period_basis, num_steps=2,
        )
    for row_chunk in (0, -1, 1.5, True):
        with pytest.raises(ValueError, match="row_chunk"):
            log_density_transport(
                field,
                state,
                oracle.period_basis,
                num_steps=2,
                divergence="exact-batched",
                row_chunk=row_chunk,  # type: ignore[arg-type]
            )


def test_generator_log_density_is_finite_deterministic_and_cover_checked(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from fractions import Fraction

    from test_learned import _tiny_config

    from crystalatlas.catalogue import MINIMAL_CATALOGUE
    from crystalatlas.decoder import CrystalScaffold, OrbitAssignment
    from crystalatlas.errors import DomainError
    from crystalatlas.geometry import ContinuousCover, ContinuousGeometry
    from crystalatlas.learned import CrystalAtlasModel, LearnedCrystalGenerator

    torch.manual_seed(2)
    model = CrystalAtlasModel(_tiny_config(), catalogue=MINIMAL_CATALOGUE).double()
    generator = LearnedCrystalGenerator(model, allow_untrained=True)
    scaffold = CrystalScaffold.from_mapping(
        nu=2,
        space_group="P-1",
        reduced_composition={8: 1},
        orbits=(OrbitAssignment("2i", 8, (Fraction(1, 7), Fraction(2, 7), Fraction(3, 7))),),
    )
    cover = ContinuousCover.from_exact_scaffold(scaffold)
    layout = model.layout(cover)
    geometries = [
        ContinuousGeometry(
            np.full(layout.lattice_dimension, 0.1 * (k + 1)),
            np.linspace(0.1, 0.4, layout.intrinsic_dimension) + 0.05 * k,
        )
        for k in range(3)
    ]
    model.eval()
    values = generator.geometry_log_density(cover, geometries, num_steps=4)
    assert values.shape == (3,) and np.all(np.isfinite(values))
    again = generator.geometry_log_density(cover, geometries, num_steps=4)
    np.testing.assert_array_equal(values, again)
    hutch = generator.geometry_log_density(
        cover, geometries, num_steps=4, divergence="hutchinson", probes=4, seed=3
    )
    assert hutch.shape == (3,) and np.all(np.isfinite(hutch))
    # The replicated-row trace on the batched expansion is the trace of the
    # reference coordinate loop on the per-member expansion.
    reference = generator.geometry_log_density(
        cover, geometries, num_steps=4, divergence="exact-loop"
    )
    np.testing.assert_allclose(values, reference, rtol=0, atol=1e-9)
    assert float(np.ptp(reference)) > 1e-3  # distinct geometries, distinct densities
    # A candidate budget below members x dimension changes the chunking only:
    # 2 -> two member chunks, one coordinate per pass; 7 -> one chunk, two per pass.
    dimension = layout.lattice_dimension + layout.intrinsic_dimension
    assert dimension == 9
    for capacity in (1, 2, 7):
        monkeypatch.setattr(
            LearnedCrystalGenerator,
            "_candidate_capacity",
            lambda self, bridge, dtype, capacity=capacity: capacity,
        )
        for mode in ("exact", "exact-loop"):
            chunked = generator.geometry_log_density(
                cover, geometries, num_steps=4, divergence=mode
            )
            np.testing.assert_allclose(chunked, reference, rtol=0, atol=1e-9)
    monkeypatch.undo()
    with pytest.raises(DomainError, match="divergence"):
        generator.geometry_log_density(cover, geometries, num_steps=4, divergence="x")
    with pytest.raises(DomainError, match="dimension"):
        generator.geometry_log_density(
            cover, [ContinuousGeometry(np.zeros(1), geometries[0].intrinsic_coordinates)]
        )


def test_rank_samples_scores_generated_candidates_per_cover() -> None:
    from test_learned import _tiny_config

    from crystalatlas.catalogue import MINIMAL_CATALOGUE
    from crystalatlas.composition import Composition
    from crystalatlas.learned import CrystalAtlasModel, LearnedCrystalGenerator, SampleScore
    from crystalatlas.pipeline import GenerationMode, GenerationRequest

    torch.manual_seed(7)
    model = CrystalAtlasModel(_tiny_config(), catalogue=MINIMAL_CATALOGUE).double()
    generator = LearnedCrystalGenerator(model, allow_untrained=True)
    request = GenerationRequest(
        mode=GenerationMode.CSP,
        composition=Composition.from_counts({6: 1, 8: 1}),
        samples=4,
        seed=29,
    )
    samples = generator.require_samples(request)
    scores = generator.rank_samples(samples, num_steps=4)
    assert len(scores) == 4 and all(isinstance(score, SampleScore) for score in scores)
    assert [score.sample_index for score in scores] == [0, 1, 2, 3]
    for score, sample in zip(scores, samples, strict=True):
        assert np.isfinite(score.log_geometry)
        assert score.log_cover == sample.cover_sample.log_probability
        assert score.log_joint == score.log_cover + score.log_geometry
        layout = model.layout(sample.cover_sample.cover)
        assert score.dimension == layout.lattice_dimension + layout.intrinsic_dimension
    # Batched per-cover scoring equals scoring each sample alone.
    alone = [generator.rank_samples([sample], num_steps=4)[0] for sample in samples]
    for score, single in zip(scores, alone, strict=True):
        assert abs(score.log_geometry - single.log_geometry) < 1e-9
    assert generator.rank_samples([]) == ()


def test_select_map_sample_takes_the_cover_mass_first_and_the_density_second(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from dataclasses import replace

    from test_learned import _tiny_config

    from crystalatlas.catalogue import MINIMAL_CATALOGUE
    from crystalatlas.composition import Composition
    from crystalatlas.errors import DomainError, InvariantViolation
    from crystalatlas.learned import CrystalAtlasModel, LearnedCrystalGenerator, MapSelection
    from crystalatlas.pipeline import GenerationMode, GenerationRequest

    torch.manual_seed(7)
    model = CrystalAtlasModel(_tiny_config(), catalogue=MINIMAL_CATALOGUE).double()
    generator = LearnedCrystalGenerator(model, allow_untrained=True)
    request = GenerationRequest(
        mode=GenerationMode.CSP,
        composition=Composition.from_counts({6: 1, 8: 1}),
        samples=6,
        seed=29,
    )
    samples = list(generator.require_samples(request))
    covers = [sample.cover_sample.cover for sample in samples]
    distinct = list(dict.fromkeys(covers))
    assert len(distinct) >= 2  # the rule is exercised across covers

    # On the model's own numbers: the cover of largest selector mass, then the
    # member of largest exact density; only that cover is integrated.
    selection = generator.select_map_sample(samples, num_steps=4)
    assert isinstance(selection, MapSelection)
    masses = [sample.cover_sample.log_probability for sample in samples]
    top = covers[int(np.argmax(masses))]
    assert selection.members == tuple(k for k, cover in enumerate(covers) if cover == top)
    assert selection.log_cover == max(masses)
    direct = generator.geometry_log_density(
        top, [samples[k].geometry for k in selection.members], num_steps=4
    )
    np.testing.assert_array_equal(np.asarray(selection.log_geometry), direct)
    assert selection.sample_index == selection.members[int(np.argmax(direct))]

    # Controlled numbers: the last distinct cover gets the largest mass although
    # its densities are the smallest of all; inside it the last member is densest.
    def with_mass(sample, mass):  # type: ignore[no-untyped-def]
        return replace(sample, cover_sample=replace(sample.cover_sample, log_probability=mass))

    rank = {cover: position for position, cover in enumerate(distinct)}
    staged = [
        with_mass(sample, -5.0 + rank[cover])
        for sample, cover in zip(samples, covers, strict=True)
    ]
    integrated: list[object] = []

    def fake_density(self, cover, geometries, **_):  # type: ignore[no-untyped-def]
        integrated.append(cover)
        return np.arange(len(geometries), dtype=np.float64) - 100.0 * rank[cover]

    monkeypatch.setattr(LearnedCrystalGenerator, "geometry_log_density", fake_density)
    staged_selection = generator.select_map_sample(staged)
    last = distinct[-1]
    members = tuple(k for k, cover in enumerate(covers) if cover == last)
    assert integrated == [last]
    assert staged_selection.members == members
    assert staged_selection.sample_index == members[-1]
    assert staged_selection.log_cover == -5.0 + rank[last]
    # Ties: the first cover in input order, and the first member of equal density.
    tied = [with_mass(sample, -1.0) for sample in samples]
    monkeypatch.setattr(
        LearnedCrystalGenerator,
        "geometry_log_density",
        lambda self, cover, geometries, **_: np.zeros(len(geometries)),
    )
    assert generator.select_map_sample(tied).sample_index == 0
    # One cover is one table of the selector law: unequal masses are a defect.
    with pytest.raises(InvariantViolation, match="disagree"):
        generator.select_map_sample([tied[0], with_mass(samples[0], -2.0)])
    monkeypatch.setattr(
        LearnedCrystalGenerator,
        "geometry_log_density",
        lambda self, cover, geometries, **_: np.full(len(geometries), np.nan),
    )
    with pytest.raises(InvariantViolation, match="not finite"):
        generator.select_map_sample(tied)
    with pytest.raises(DomainError, match="at least one"):
        generator.select_map_sample([])
