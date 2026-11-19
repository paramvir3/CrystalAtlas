"""Opt-in start rejection of the geometry ODE (``start_min_separation``, 2026-11-16).

A base draw whose two closest distinct atoms are at most ``s`` apart is drawn
again from the attempt's own generator until they are farther apart.  These tests
check the exact separation search against a brute-force image sum, that the option
is off by default and changes nothing when it rejects nothing, that every
integrated start of a treated solve is far, that a far first draw is kept bitwise
and a near one is replaced by the first far draw of the same generator's stream,
the draw cap, the bucket path, and the likelihood guards.
"""

from __future__ import annotations

import itertools
import math

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from test_learned import _manifest, _provenance, _tiny_config  # noqa: E402
from test_mixed_sampler import _assert_same_traces, _request  # noqa: E402
from test_padded_batching import mixed_model  # noqa: E402,F401 - pytest fixture

import crystalatlas.learned as learned  # noqa: E402
from crystalatlas.catalogue import MINIMAL_CATALOGUE  # noqa: E402
from crystalatlas.errors import DomainError, RecoverableGenerationError  # noqa: E402
from crystalatlas.flow import canonicalize_torus_validated  # noqa: E402
from crystalatlas.learned import (  # noqa: E402
    CrystalAtlasModel,
    LearnedCrystalGenerator,
    closest_distinct_pair_within,
)
from crystalatlas.training import save_checkpoint  # noqa: E402

pytestmark = pytest.mark.ml

F64 = torch.float64


def _brute_force_d0(lattice: np.ndarray, fractional: np.ndarray, reach: int) -> float:
    """Smallest distinct-atom image distance over the box ``[-reach, reach]^3``."""

    shifts = np.array(list(itertools.product(range(-reach, reach + 1), repeat=3)), dtype=float)
    best = math.inf
    for i, j in itertools.combinations(range(len(fractional)), 2):
        delta = fractional[j] - fractional[i]
        cartesian = (delta[None, :] + shifts) @ lattice
        best = min(best, float(np.sqrt((cartesian * cartesian).sum(axis=-1).min())))
    return best


def test_the_separation_search_reaches_far_images_of_a_sheared_cell() -> None:
    # The minimizing translation is several images away along b.
    lattice = np.array([[1.0, 0.0, 0.0], [3.9, 0.3, 0.0], [0.0, 0.0, 3.0]])
    fractional = np.array([[0.0, 0.0, 0.0], [0.0, 0.5, 0.0]])
    exact = _brute_force_d0(lattice, fractional, reach=8)
    assert exact == pytest.approx(math.sqrt(0.025), rel=1e-12)
    found = closest_distinct_pair_within(torch.tensor(lattice), torch.tensor(fractional), 0.5)
    assert found == pytest.approx(exact, rel=1e-12)


@pytest.mark.parametrize("seed", range(6))
def test_the_separation_search_is_exact(seed: int) -> None:
    rng = np.random.default_rng(seed)
    # A cell with lengths about 2 to 4 and a moderate shear, and 2 to 7 atoms.
    lattice = np.diag(rng.uniform(2.0, 4.0, 3)) + rng.uniform(-0.8, 0.8, (3, 3))
    fractional = rng.uniform(-1.5, 2.5, (int(rng.integers(2, 8)), 3))
    exact = _brute_force_d0(lattice, fractional, reach=5)
    assert exact == pytest.approx(_brute_force_d0(lattice, fractional, reach=7), abs=0.0)
    for radius in (0.5 * exact, exact * (1.0 - 1e-9), exact * (1.0 + 1e-9), 1.5 * exact):
        found = closest_distinct_pair_within(
            torch.tensor(lattice), torch.tensor(fractional), radius
        )
        if radius < exact * (1.0 - 1e-12):
            assert found is None
        else:
            assert found == pytest.approx(exact, rel=1e-12)


def test_the_separation_search_ignores_images_of_one_atom_and_refuses_bad_input() -> None:
    lattice = torch.eye(3, dtype=F64) * 0.3  # images of one atom are 0.3 apart
    one = torch.tensor([[0.1, 0.2, 0.3]], dtype=F64)
    assert closest_distinct_pair_within(lattice * 10.0, one, 1.0) is None
    two = torch.tensor([[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]], dtype=F64)
    found = closest_distinct_pair_within(torch.eye(3, dtype=F64) * 4.0, two, 3.5)
    assert found == pytest.approx(2.0 * math.sqrt(3.0), rel=1e-14)
    assert closest_distinct_pair_within(torch.eye(3, dtype=F64) * 4.0, two, 3.4) is None
    # Each atom's own image is 1.0 away along a; the distinct pair is 2 sqrt(2) apart.
    short_a = torch.diag(torch.tensor([1.0, 4.0, 4.0], dtype=F64))
    face = torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.5, 0.5]], dtype=F64)
    found = closest_distinct_pair_within(short_a, face, 3.0)
    assert found == pytest.approx(2.0 * math.sqrt(2.0), rel=1e-14)
    for radius in (0.0, -1.0, math.inf, math.nan):
        with pytest.raises(DomainError, match="positive and finite"):
            closest_distinct_pair_within(torch.eye(3, dtype=F64), two, radius)
    # A plane spacing far below the radius needs too many images: the attempt fails.
    flat = torch.diag(torch.tensor([4.0, 4.0, 0.01], dtype=F64))
    with pytest.raises(RecoverableGenerationError, match="images per axis"):
        closest_distinct_pair_within(flat, two, 1.0)


def test_the_option_is_off_by_default_and_validated(mixed_model) -> None:  # type: ignore[no-untyped-def]  # noqa: F811,E501
    generator = LearnedCrystalGenerator(mixed_model, allow_untrained=True)
    assert generator.start_min_separation is None
    generator.start_min_separation = 1
    assert generator.start_min_separation == 1.0
    assert isinstance(generator.start_min_separation, float)
    for value in (0, -0.1, math.inf, math.nan, True, "0.2"):
        with pytest.raises(DomainError, match="start_min_separation"):
            generator.start_min_separation = value  # type: ignore[assignment]
    generator.start_min_separation = None
    assert generator.start_min_separation is None


class _Spy:
    """Every base draw (with its generator) and every integrated initial state."""

    def __init__(self, monkeypatch) -> None:  # type: ignore[no-untyped-def]
        self.draws: list[tuple[int, torch.Tensor, torch.Tensor]] = []
        self.seeds: list[int] = []
        self.solves: list[dict[str, object]] = []
        self.plans: list[object] = []
        sample = learned.sample_product_base
        integrate = learned.integrate_with_endpoint_estimate
        plan_class = learned.PaddedCoverPlan
        spy = self

        def recording_sample(*args, **kwargs):  # type: ignore[no-untyped-def]
            base = sample(*args, **kwargs)
            spy.draws.append((id(kwargs["generator"]), base.z.clone(), base.eta.clone()))
            spy.seeds.append(int(kwargs["generator"].initial_seed()))
            return base

        class RecordingPlan(plan_class):  # type: ignore[misc,valid-type]
            def __init__(self, *args, **kwargs) -> None:  # type: ignore[no-untyped-def]
                super().__init__(*args, **kwargs)
                spy.plans.append(self)

        def recording_integrate(field, initial_state, period_basis, **kwargs):  # type: ignore[no-untyped-def]
            spy.solves.append(
                {
                    "z": initial_state.z.clone(),
                    "eta": initial_state.eta.clone(),
                    "plan": spy.plans[-1] if spy.plans else None,
                    "draws": len(spy.draws),
                }
            )
            return integrate(field, initial_state, period_basis, **kwargs)

        monkeypatch.setattr(learned, "sample_product_base", recording_sample)
        monkeypatch.setattr(learned, "PaddedCoverPlan", RecordingPlan)
        monkeypatch.setattr(learned, "integrate_with_endpoint_estimate", recording_integrate)


def _row_d0s(plan, z: torch.Tensor, eta: torch.Tensor) -> list[float]:  # type: ignore[no-untyped-def]
    """Brute-force d0 of every row of a padded initial state, as the field sees it."""

    canonical = canonicalize_torus_validated(eta, plan.period_basis)
    expanded = plan.expand(z, canonical)
    return [
        _brute_force_d0(
            expanded.lattice[row].double().numpy(),
            expanded.fractional_coordinates[row, : plan.atom_counts[row]].double().numpy(),
            reach=4,
        )
        if plan.atom_counts[row] > 1
        else math.inf
        for row in range(plan.batch)
    ]


def _first_draw_d0s(model, monkeypatch) -> list[float]:  # type: ignore[no-untyped-def]
    with monkeypatch.context() as patch:
        spy = _Spy(patch)
        LearnedCrystalGenerator(
            model, allow_untrained=True, geometry_sampler="mixed-v1"
        ).sample(_request())
    assert len(spy.solves) == 1 and spy.solves[0]["plan"] is not None
    solve = spy.solves[0]
    return _row_d0s(solve["plan"], solve["z"], solve["eta"])


def test_a_threshold_that_rejects_nothing_changes_nothing(mixed_model, monkeypatch) -> None:  # type: ignore[no-untyped-def]  # noqa: F811,E501
    smallest = min(_first_draw_d0s(mixed_model, monkeypatch))
    off = LearnedCrystalGenerator(
        mixed_model, allow_untrained=True, geometry_sampler="mixed-v1"
    )
    on = LearnedCrystalGenerator(
        mixed_model,
        allow_untrained=True,
        geometry_sampler="mixed-v1",
        start_min_separation=0.5 * smallest,
    )
    _assert_same_traces(mixed_model, on.sample(_request()), off.sample(_request()), 0.0)


def test_near_starts_are_redrawn_from_their_own_stream(mixed_model, monkeypatch) -> None:  # type: ignore[no-untyped-def]  # noqa: F811,E501
    first = _first_draw_d0s(mixed_model, monkeypatch)
    threshold = float(np.median([value for value in first if math.isfinite(value)]))
    near = [row for row, value in enumerate(first) if value <= threshold]
    assert near and len(near) < len(first)
    with monkeypatch.context() as patch:
        baseline_spy = _Spy(patch)
        LearnedCrystalGenerator(
            mixed_model, allow_untrained=True, geometry_sampler="mixed-v1"
        ).sample(_request())
    with monkeypatch.context() as patch:
        spy = _Spy(patch)
        generator = LearnedCrystalGenerator(
            mixed_model,
            allow_untrained=True,
            geometry_sampler="mixed-v1",
            start_min_separation=threshold,
        )
        traces = generator.sample(_request())
    assert all(trace.succeeded for trace in traces)
    assert len(spy.solves) == 1
    solve, reference = spy.solves[0], baseline_spy.solves[0]
    plan = solve["plan"]
    used = _row_d0s(plan, solve["z"], solve["eta"])
    assert all(value > threshold for value in used)
    # Rows in solve order own one generator each; the first draws come first.
    owners = [owner for owner, _, _ in spy.draws[: plan.batch]]
    assert len(set(owners)) == plan.batch
    for row in range(plan.batch):
        stream = [(z, eta) for owner, z, eta in spy.draws if owner == owners[row]]
        d = plan.lattice_dimensions[row].item()
        k = plan.intrinsic_dimensions[row].item()
        # The first draw is the untreated first draw.
        assert torch.equal(stream[0][0], reference["z"][row, :d])
        assert torch.equal(stream[0][1], reference["eta"][row, :k])
        if row not in near:
            assert len(stream) == 1
        else:
            assert len(stream) > 1
        # The integrated row is the last draw of its own stream.
        assert torch.equal(solve["z"][row, :d], stream[-1][0])
        assert torch.equal(solve["eta"][row, :k], stream[-1][1])
        # Every earlier draw of the stream was near.
        for z, eta in stream[:-1]:
            trial_z, trial_eta = solve["z"].clone(), solve["eta"].clone()
            trial_z[row, :d], trial_eta[row, :k] = z, eta
            assert _row_d0s(plan, trial_z, trial_eta)[row] <= threshold


def test_an_attempt_that_never_starts_far_fails_alone(mixed_model, monkeypatch) -> None:  # type: ignore[no-untyped-def]  # noqa: F811,E501
    first = _first_draw_d0s(mixed_model, monkeypatch)
    finite = [value for value in first if math.isfinite(value)]
    monkeypatch.setattr(learned, "START_REJECTION_MAX_DRAWS", 4)
    calls: list[int] = []

    def always_near(lattice, fractional, radius):  # type: ignore[no-untyped-def]
        calls.append(int(fractional.shape[0]))
        return 0.0 if fractional.shape[0] > 1 else None

    monkeypatch.setattr(learned, "closest_distinct_pair_within", always_near)
    generator = LearnedCrystalGenerator(
        mixed_model,
        allow_untrained=True,
        geometry_sampler="mixed-v1",
        start_min_separation=0.2,
    )
    traces = generator.sample(_request())
    failed = [trace for trace in traces if not trace.succeeded]
    assert calls and len(failed) == len(finite)
    assert {trace.failed_stage for trace in failed} == {"geometry/ode"}
    assert {trace.error for trace in failed} == {
        "RecoverableGenerationError: every one of 4 base draws has two distinct atoms "
        "within 0.2 angstrom"
    }


def test_a_capped_attempt_fails_and_its_siblings_keep_their_outcomes(mixed_model, monkeypatch) -> None:  # type: ignore[no-untyped-def]  # noqa: F811,E501
    first = _first_draw_d0s(mixed_model, monkeypatch)
    threshold = float(np.median([value for value in first if math.isfinite(value)]))
    near = {row for row, value in enumerate(first) if value <= threshold}
    assert near and len(near) < len(first)
    # Row b of the padded solve belongs to the attempt whose geometry generator is
    # seeded with word 3 of its child of the request's seed sequence.
    request = _request()
    geometry_seeds = [
        int(child.generate_state(4, dtype=np.uint64)[3])
        for child in np.random.SeedSequence(request.seed).spawn(request.samples)
    ]
    generator = LearnedCrystalGenerator(
        mixed_model, allow_untrained=True, geometry_sampler="mixed-v1"
    )
    with monkeypatch.context() as patch:
        spy = _Spy(patch)
        untreated = generator.sample(request)
    batch = spy.solves[0]["plan"].batch
    near_attempts = {geometry_seeds.index(spy.seeds[row]) for row in near}
    assert batch == len(geometry_seeds) and len(near_attempts) == len(near)
    monkeypatch.setattr(learned, "START_REJECTION_MAX_DRAWS", 1)
    generator.start_min_separation = threshold
    treated = generator.sample(_request())
    failed = {trace.sample_index for trace in treated if not trace.succeeded}
    assert failed == near_attempts
    assert {trace.failed_stage for trace in treated if not trace.succeeded} == {"geometry/ode"}
    assert all(trace.sample is None for trace in treated if not trace.succeeded)
    survivors = [trace for trace in treated if trace.succeeded]
    references = [trace for trace in untreated if trace.sample_index not in near_attempts]
    _assert_same_traces(mixed_model, survivors, references, 1e-10)


def test_the_separation_is_in_angstrom_in_any_domain(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    from dataclasses import replace

    from test_padded_batching import mixed_model as fixture

    base = fixture.__wrapped__()
    config = base.scientific_config
    torch.manual_seed(31)  # as the fixture, so the untrained selector draws the same covers
    scaled = CrystalAtlasModel(
        replace(config, domain=replace(config.domain, reference_length_angstrom=2.0)),
        catalogue=base.catalogue,
    ).double()
    first = _first_draw_d0s(scaled, monkeypatch)
    assert first == _first_draw_d0s(base, monkeypatch)  # the starts do not depend on it
    threshold = float(np.median([value for value in first if math.isfinite(value)]))
    with monkeypatch.context() as patch:
        spy = _Spy(patch)
        LearnedCrystalGenerator(
            scaled, allow_untrained=True, geometry_sampler="mixed-v1",
            start_min_separation=threshold,
        ).sample(_request())
    solve = spy.solves[0]
    owners = [owner for owner, _, _ in spy.draws[: solve["plan"].batch]]
    redrawn = {row for row, owner in enumerate(owners)
               if sum(1 for item, _, _ in spy.draws if item == owner) > 1}
    assert redrawn == {row for row, value in enumerate(first) if value <= threshold}


def test_the_bucket_sampler_is_treated_the_same_way(mixed_model, monkeypatch) -> None:  # type: ignore[no-untyped-def]  # noqa: F811,E501
    first = _first_draw_d0s(mixed_model, monkeypatch)
    threshold = float(np.median([value for value in first if math.isfinite(value)]))
    mixed = LearnedCrystalGenerator(
        mixed_model,
        allow_untrained=True,
        geometry_sampler="mixed-v1",
        start_min_separation=threshold,
    ).sample(_request())
    with monkeypatch.context() as patch:
        spy = _Spy(patch)
        bucket = LearnedCrystalGenerator(
            mixed_model, allow_untrained=True, start_min_separation=threshold
        ).sample(_request())
    assert len(spy.solves) > 1
    for solve in spy.solves:
        used = _row_d0s(solve["plan"], solve["z"], solve["eta"])
        assert all(value > threshold for value in used)
    _assert_same_traces(mixed_model, mixed, bucket, 1e-10)


def test_likelihood_guards(mixed_model, monkeypatch) -> None:  # type: ignore[no-untyped-def]  # noqa: F811,E501
    first = _first_draw_d0s(mixed_model, monkeypatch)
    threshold = float(np.median([value for value in first if math.isfinite(value)]))
    generator = LearnedCrystalGenerator(
        mixed_model,
        allow_untrained=True,
        geometry_sampler="mixed-v1",
        start_min_separation=threshold,
    )
    traces = generator.sample(_request(samples=4))
    samples = [trace.sample for trace in traces if trace.succeeded]
    with pytest.raises(DomainError, match=r"1 / B\(A \| c\)"):
        generator.rank_samples(samples)
    chosen = generator.select_map_sample(samples, num_steps=4)
    generator.start_min_separation = None
    assert generator.select_map_sample(samples, num_steps=4) == chosen
    assert len(generator.rank_samples(samples, num_steps=4)) == len(samples)


def test_a_checkpoint_bound_generator_takes_the_option(tmp_path) -> None:  # type: ignore[no-untyped-def]
    config = _tiny_config()
    model = CrystalAtlasModel(config, catalogue=MINIMAL_CATALOGUE).double()
    info = save_checkpoint(
        tmp_path / "model.pt",
        model=model,
        optimizer=torch.optim.Adam(model.parameters(), lr=1.0e-3),
        config=config,
        scientific_manifest=_manifest(model),
        provenance=_provenance(config),
        step=0,
    )
    bound = LearnedCrystalGenerator.from_checkpoint(
        model,
        tmp_path / "model.pt",
        scientific_manifest=_manifest(model),
        expected_sha256=info.sha256,
        start_min_separation=0.25,
    )
    assert bound.start_min_separation == 0.25 and bound.is_trained_inference
