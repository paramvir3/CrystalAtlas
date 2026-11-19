"""Selection rules and bookkeeping of the research CSP measurement (program R3).

The measurement module lives in ``scripts/csp_evaluation_v7.py``.  These tests
fix what it counts: a failed first attempt is a miss, every attempt stays in the
denominators, the consensus rule is the first maximum of the off-diagonal vote
count of the certified matcher relation, no rule reads the target, and a resumed
evaluation reuses finished cases without recomputing them.  A decoded attempt
the matcher cannot encode is a miss labelled by its cause, and the shortest
interatomic distance counts the periodic images of one site.
"""

from __future__ import annotations

import math
from dataclasses import replace
from fractions import Fraction
from pathlib import Path

import numpy as np
import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"


@pytest.fixture
def csp(monkeypatch: pytest.MonkeyPatch):  # type: ignore[no-untyped-def]
    pytest.importorskip("torch")
    monkeypatch.syspath_prepend(str(SCRIPTS))
    import csp_evaluation_v7

    return csp_evaluation_v7


def _relation(size: int, pairs: set[tuple[int, int]]) -> list[list[bool]]:
    return [
        [i == j or (i, j) in pairs or (j, i) in pairs for j in range(size)] for i in range(size)
    ]


def test_consensus_is_the_first_maximum_of_the_off_diagonal_votes(csp) -> None:  # type: ignore[no-untyped-def]
    from crystalatlas.errors import DomainError

    # Classes {0}, {1, 3}, {2, 4}: two classes tie; the lowest index wins.
    assert csp.consensus_selection(_relation(5, {(1, 3), (2, 4)})) == (1, (0, 1, 1, 1, 1))
    # Classes {0, 2, 4}, {1, 3}.
    assert csp.consensus_selection(_relation(5, {(0, 2), (0, 4), (2, 4), (1, 3)})) == (
        0,
        (2, 1, 2, 1, 2),
    )
    # A tolerance relation need not be transitive: 0 ~ 1 ~ 2 without 0 ~ 2.
    assert csp.consensus_selection(_relation(3, {(0, 1), (1, 2)})) == (1, (1, 2, 1))
    # No agreement at all: the first valid attempt, with zero votes.
    assert csp.consensus_selection(_relation(3, set())) == (0, (0, 0, 0))
    assert csp.consensus_selection([[True]]) == (0, (0,))
    # The diagonal carries no vote, whatever it holds.
    assert csp.consensus_selection([[False, True], [True, False]]) == (0, (1, 1))
    with pytest.raises(DomainError, match="at least one"):
        csp.consensus_selection([])
    with pytest.raises(DomainError, match="square"):
        csp.consensus_selection([[True, False], [False]])
    with pytest.raises(DomainError, match="symmetric"):
        csp.consensus_selection([[True, True], [False, True]])


def test_mutual_equivalence_matches_every_unordered_pair_once(csp) -> None:  # type: ignore[no-untyped-def]
    calls: list[tuple[bytes, bytes]] = []

    def match(left: bytes, right: bytes) -> bool:
        calls.append((left, right))
        return left[:1] == right[:1]

    payloads = [b"a1", b"b1", b"a2", b"c1", b"b2"]
    relation = csp.mutual_equivalence(payloads, match)
    assert len(calls) == 10 and len(set(calls)) == 10
    assert all(payloads.index(left) < payloads.index(right) for left, right in calls)
    assert relation == _relation(5, {(0, 2), (1, 4)})
    assert csp.mutual_equivalence([], match) == []


def test_wilson_interval_closed_forms_and_symmetry(csp) -> None:  # type: ignore[no-untyped-def]
    from crystalatlas.errors import DomainError

    z2 = 1.959963984540054**2
    low, high = csp.wilson_interval(0, 50)
    assert low == 0.0 and high == pytest.approx(z2 / (50 + z2), rel=1e-12)
    low, high = csp.wilson_interval(50, 50)
    assert high == 1.0 and low == pytest.approx(50 / (50 + z2), rel=1e-12)
    for successes, trials in ((1, 50), (17, 50), (25, 50), (3, 7)):
        low, high = csp.wilson_interval(successes, trials)
        mirror_low, mirror_high = csp.wilson_interval(trials - successes, trials)
        assert low < successes / trials < high
        assert low == pytest.approx(1.0 - mirror_high, abs=1e-14)
        assert high == pytest.approx(1.0 - mirror_low, abs=1e-14)
        # The endpoints solve (p - rate)^2 = z^2 p (1 - p) / n.
        for endpoint in (low, high):
            assert (endpoint - successes / trials) ** 2 == pytest.approx(
                z2 * endpoint * (1.0 - endpoint) / trials, rel=1e-10
            )
    for successes, trials in ((1, 0), (-1, 5), (6, 5), (True, 5), (1, True), (1.0, 5)):
        with pytest.raises(DomainError):
            csp.wilson_interval(successes, trials)


def test_case_seed_depends_on_the_run_seed_and_the_record_only(csp) -> None:  # type: ignore[no-untyped-def]
    from crystalatlas.errors import DomainError

    seeds = {csp.case_seed(7, record) for record in ("mp-1", "mp-2", "mp-10", "mp-01")}
    assert len(seeds) == 4 and all(0 <= value < 2**32 for value in seeds)
    assert csp.case_seed(7, "mp-1") == csp.case_seed(7, "mp-1")
    assert csp.case_seed(8, "mp-1") != csp.case_seed(7, "mp-1")
    for bad in (-1, True, 1.5):
        with pytest.raises(DomainError):
            csp.case_seed(bad, "mp-1")


def _brute_force_shortest(lattice, positions, reach: int = 4) -> float:  # type: ignore[no-untyped-def]
    """Shortest distance between distinct atoms by enumerating a cube of images."""

    cell = np.asarray(lattice, dtype=float)
    points = np.asarray(positions, dtype=float)
    span = range(-reach, reach + 1)
    best = math.inf
    for n0 in span:
        for n1 in span:
            for n2 in span:
                shift = np.array([n0, n1, n2], dtype=float)
                for i in range(len(points)):
                    for j in range(i, len(points)):
                        if i == j and not shift.any():
                            continue
                        best = min(
                            best, float(np.linalg.norm((points[j] - points[i] + shift) @ cell))
                        )
    return best


def test_shortest_distance_counts_periodic_images_in_any_cell(csp) -> None:  # type: ignore[no-untyped-def]
    cubic = 3.0 * np.eye(3)
    # One site: the shortest lattice vector.
    assert csp.shortest_interatomic_distance(cubic, [[0.2, 0.3, 0.4]]) == pytest.approx(3.0)
    body = [[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]]
    assert csp.shortest_interatomic_distance(4.0 * np.eye(3), body) == pytest.approx(
        2.0 * math.sqrt(3.0)
    )
    # A skew cell whose shortest lattice vector b - a is no row of the basis.
    skew = np.array([[4.0, 0.0, 0.0], [3.6, 1.0, 0.0], [0.0, 0.0, 5.0]])
    assert csp.shortest_interatomic_distance(skew, [[0.1, 0.7, 0.2]]) == pytest.approx(
        math.hypot(0.4, 1.0)
    )
    # The closest image of this pair is not the componentwise-rounded one.
    pair = [[0.0, 0.0, 0.0], [0.5, 0.5, 0.0]]
    assert csp.shortest_interatomic_distance(skew, pair) == pytest.approx(math.hypot(0.2, 0.5))
    # A long thin cell: the lattice vector is shorter than every pair distance.
    thin = np.diag([0.4, 6.0, 6.0])
    assert csp.shortest_interatomic_distance(thin, body) == pytest.approx(0.4)
    assert csp.shortest_interatomic_distance(cubic, [[0.25, 0.5, 0.75]] * 2) == 0.0
    rng = np.random.default_rng(5)
    for _ in range(20):
        lattice = np.eye(3) * rng.uniform(2.0, 5.0, 3) + rng.uniform(-1.5, 1.5, (3, 3))
        if np.linalg.det(lattice) <= 0.5:
            continue
        positions = rng.uniform(-1.0, 2.0, (int(rng.integers(1, 6)), 3))
        assert csp.shortest_interatomic_distance(lattice, positions) == pytest.approx(
            _brute_force_shortest(lattice, positions), rel=1e-12, abs=1e-12
        )


def test_coincident_pairs_are_same_species_closest_images(csp) -> None:  # type: ignore[no-untyped-def]
    lattice = 4.0 * np.eye(3)
    positions = [
        [0.0, 0.0, 0.0],
        [0.001, 0.0, 0.0],  # 0.004 from site 0
        [0.0, 0.0, 0.0005],  # another species on top of site 0
        [0.999, 0.0, 0.0],  # 0.004 from site 0 through the cell face, 0.008 from site 1
        [0.5, 0.5, 0.5],
    ]
    numbers = [8, 8, 1, 8, 8]
    assert csp.coincident_same_species_pairs(lattice, positions, numbers, 0.01) == 3
    assert csp.coincident_same_species_pairs(lattice, positions, numbers, 0.005) == 2
    assert csp.coincident_same_species_pairs(lattice, positions, numbers, 0.001) == 0
    assert csp.coincident_same_species_pairs(lattice, positions, [8, 7, 1, 6, 8], 0.01) == 0


def _case(  # type: ignore[no-untyped-def]
    csp, record, *, matches, valid, first, chosen_map, consensus, table=(0, 0), sound=0,
    stages=None,
):
    rms = {index: 0.01 * (index + 1) for index in matches}
    match = [index in matches for index in range(4)]

    def selection(attempt):  # type: ignore[no-untyped-def]
        return csp._selection(attempt, match, rms)

    return {
        "record_id": record,
        "track": "reduced-formula",
        "attempts": 4,
        "seed": 1,
        "request_error": None,
        "valid": valid,
        "matches": sorted(matches),
        "match_count": len(matches),
        "first": selection(first),
        "map": {**selection(chosen_map), "error": None},
        "consensus": {**selection(consensus), "votes": 0},
        "best_rms": min(rms.values()) if rms else None,
        "distinct_tables": 2,
        "target_table_valid": table[0],
        "target_table_matches": table[1],
        "target_primitive_atoms": 2,
        "target_cover_primitive_atoms": 2,
        "target_payload_atoms": 2,
        "structurally_valid": sound,
        "failed_stages": dict(stages or {}),
        "encode_errors": [],
        "seconds": {"sample": 1.0, "match": 0.5, "map": 2.0, "consensus": 0.25},
    }


def test_summary_keeps_every_case_and_attempt_in_the_denominators(csp) -> None:  # type: ignore[no-untyped-def]
    from crystalatlas.errors import DomainError

    cases = [
        # First attempt matches; MAP picks a miss; consensus picks the hit.
        _case(csp, "a", matches={0, 2}, valid=4, first=0, chosen_map=1, consensus=2,
              table=(3, 2), sound=4),
        # Only attempt 3 matches; MAP finds it.
        _case(csp, "b", matches={3}, valid=3, first=0, chosen_map=3, consensus=1,
              table=(1, 1), sound=2, stages={"evaluation/coincident-atoms": 1}),
        # Nothing matches.
        _case(csp, "c", matches=set(), valid=2, first=0, chosen_map=2, consensus=2,
              sound=1, stages={"geometry/ode": 1, "evaluation/coincident-atoms": 1}),
        # No valid attempt at all: no selection exists and every rule misses.
        _case(csp, "d", matches=set(), valid=0, first=0, chosen_map=None, consensus=None,
              stages={"geometry/decode": 3, "evaluation/encode-invariant": 1}),
    ]
    cases[3]["map"]["error"] = "ValueError: staged"
    cases[3]["encode_errors"] = ["InvariantViolation: staged"]
    cases[2]["target_payload_atoms"] = 1
    summary = csp.summarize(cases, track="reduced-formula", candidates=4)
    assert summary["cases"] == 4
    assert summary["match_at_1"]["successes"] == 1 and summary["match_at_1"]["rate"] == 0.25
    assert summary["match_map"]["successes"] == 1
    assert summary["match_consensus"]["successes"] == 1
    assert summary["match_at_k"]["successes"] == 2 and summary["match_at_k"]["trials"] == 4
    # RMS means run over the matched selections only.
    assert summary["mean_rms_first"] == pytest.approx(0.01)
    assert summary["mean_rms_map"] == pytest.approx(0.04)
    assert summary["mean_rms_consensus"] == pytest.approx(0.03)
    assert summary["mean_best_rms_at_k"] == pytest.approx((0.01 + 0.04) / 2)
    fractions = [0.5, 0.25, 0.0, 0.0]
    mean = sum(fractions) / 4
    assert summary["match_per_attempt"]["rate"] == pytest.approx(mean)
    assert summary["match_per_attempt"]["standard_error"] == pytest.approx(
        math.sqrt(sum((value - mean) ** 2 for value in fractions) / 3) / 2
    )
    assert summary["valid_fraction"] == pytest.approx(9 / 16)
    sound = [1.0, 0.5, 0.25, 0.0]
    assert summary["structurally_valid_fraction"]["rate"] == pytest.approx(7 / 16)
    assert summary["structurally_valid_fraction"]["standard_error"] == pytest.approx(
        math.sqrt(sum((value - 7 / 16) ** 2 for value in sound) / 3) / 2
    )
    assert summary["failed_stages"] == {
        "evaluation/coincident-atoms": 2,
        "evaluation/encode-invariant": 1,
        "geometry/decode": 3,
        "geometry/ode": 1,
    }
    assert list(summary["failed_stages"]) == sorted(summary["failed_stages"])
    assert summary["encode_invariant_errors"] == ["InvariantViolation: staged"]
    assert summary["encode_invariant_cases"] == 1
    assert summary["target_table_sampled"]["successes"] == 2
    assert summary["match_given_target_table"]["successes"] == 3
    assert summary["match_given_target_table"]["trials"] == 4
    assert summary["count_inconsistent_targets"] == 1
    assert summary["map_error_cases"] == 1 and summary["map_errors"] == ["ValueError: staged"]
    assert summary["request_error_cases"] == 0
    assert summary["seconds"] == {"sample": 4.0, "match": 2.0, "map": 8.0, "consensus": 1.0}
    assert summary["match_at_1"]["wilson_95"] == list(csp.wilson_interval(1, 4))
    with pytest.raises(DomainError, match="no case"):
        csp.summarize(cases, track="primitive-count", candidates=4)
    with pytest.raises(DomainError, match="candidate count"):
        csp.summarize(cases, track="reduced-formula", candidates=5)


def _tiny_generator():  # type: ignore[no-untyped-def]
    import torch
    from test_learned import _tiny_config

    from crystalatlas.catalogue import MINIMAL_CATALOGUE
    from crystalatlas.learned import CrystalAtlasModel, LearnedCrystalGenerator

    torch.manual_seed(7)
    model = CrystalAtlasModel(_tiny_config(), catalogue=MINIMAL_CATALOGUE).double()
    return LearnedCrystalGenerator(model, allow_untrained=True)


def _stored_target(csp, generator, record_id="mp-test"):  # type: ignore[no-untyped-def]
    from crystalatlas.decoder import CrystalScaffold, OrbitAssignment
    from crystalatlas.geometry import ContinuousCover

    scaffold = CrystalScaffold.from_mapping(
        nu=2,
        space_group="P-1",
        reduced_composition={8: 1},
        orbits=(OrbitAssignment("2i", 8, (Fraction(1, 7), Fraction(2, 7), Fraction(3, 7))),),
    )
    layout = generator.model.layout(ContinuousCover.from_exact_scaffold(scaffold))
    metric = layout.chart.decode_latent(np.zeros(layout.lattice_dimension))
    return csp.target_from_record(generator.model, record_id, scaffold, metric)


@pytest.mark.ml
def test_target_and_case_on_a_tiny_model(csp, monkeypatch: pytest.MonkeyPatch) -> None:  # type: ignore[no-untyped-def]
    from crystalatlas.errors import DomainError, InvariantViolation
    from crystalatlas.learned import LearnedCrystalGenerator
    from crystalatlas.physical_matching import (
        encode_physical_structure,
        physical_structure_match,
    )
    from crystalatlas.pipeline import GenerationMode, GenerationRequest

    generator = _tiny_generator()
    stored = _stored_target(csp, generator)
    assert stored.primitive_atoms == stored.cover_primitive_atoms == stored.payload_atoms == 2
    assert stored.composition.counts == ((8, 1),)
    assert csp.table_key(stored.cover) == (2, "P-1", stored.composition, (("2i", 8),))

    # The attempts of the case are those of the plain request with the same seed.
    candidates, seed = 6, 29
    request = GenerationRequest(
        mode=GenerationMode.CSP, composition=stored.composition, samples=candidates, seed=seed
    )
    traces = generator.sample(request)
    valid = [index for index, trace in enumerate(traces) if trace.succeeded]
    assert len(valid) >= 3
    payloads = {
        index: encode_physical_structure(
            traces[index].sample.physical_primitive.lattice,
            traces[index].sample.physical_primitive.fractional_positions,
            traces[index].sample.physical_primitive.atomic_numbers,
        )
        for index in valid
    }
    # Target: the last valid attempt itself, so the matcher is exercised on a hit.
    hit = valid[-1]
    target = replace(
        stored,
        cover=traces[hit].sample.cover_sample.cover,
        payload=payloads[hit],
        primitive_atoms=len(traces[hit].sample.physical_primitive.atomic_numbers),
    )
    case = csp.evaluate_case(
        generator,
        target,
        track="reduced-formula",
        candidates=candidates,
        seed=seed,
        likelihood_steps=4,
    )
    expected = [
        index for index in valid if physical_structure_match(payloads[index], target.payload)
    ]
    assert hit in expected
    assert case["matches"] == expected and case["match_count"] == len(expected)
    assert case["valid"] == len(valid) and case["attempts"] == candidates
    assert sum(case["failed_stages"].values()) == candidates - len(valid)
    assert case["first"] == {
        "attempt": 0,
        "match": 0 in expected,
        "rms": case["first"]["rms"],
    }
    assert (case["first"]["rms"] is not None) == (0 in expected)
    assert case["best_rms"] is not None and case["best_rms"] < 1e-9
    assert case["primitive_atom_count"] is None and case["request_error"] is None
    distances = [
        None
        if not trace.succeeded
        else _brute_force_shortest(
            trace.sample.physical_primitive.lattice,
            trace.sample.physical_primitive.fractional_positions,
        )
        for trace in traces
    ]
    assert [value is None for value in case["shortest_distances"]] == [
        value is None for value in distances
    ]
    for value, expected_distance in zip(case["shortest_distances"], distances, strict=True):
        if expected_distance is not None:
            assert value == pytest.approx(expected_distance, rel=1e-10, abs=1e-10)
    assert case["structurally_valid"] == sum(
        value is not None and value > 0.5 for value in distances
    )
    assert case["encode_errors"] == []

    # The MAP rule is the library rule on the valid attempts; it never sees the target.
    selection = generator.select_map_sample(
        [traces[index].sample for index in valid], num_steps=4
    )
    assert case["map"]["attempt"] == valid[selection.sample_index]
    assert case["map"]["match"] == (case["map"]["attempt"] in expected)
    assert case["map"]["cover_members"] == [valid[index] for index in selection.members]
    assert case["map"]["log_cover"] == selection.log_cover and case["map"]["error"] is None

    # The consensus rule is recomputed from the matcher relation of the valid attempts.
    relation = csp.mutual_equivalence([payloads[index] for index in valid])
    chosen, votes = csp.consensus_selection(relation)
    assert case["consensus"]["attempt"] == valid[chosen]
    assert case["consensus"]["votes"] == votes[chosen]
    assert case["consensus"]["match"] == (valid[chosen] in expected)

    tables = [csp.table_key(traces[index].sample.cover_sample.cover) for index in valid]
    wanted = csp.table_key(target.cover)
    assert case["distinct_tables"] == len(set(tables))
    assert case["target_table_valid"] == sum(key == wanted for key in tables) >= 1
    assert case["target_table_matches"] == sum(
        index in expected for index, key in zip(valid, tables, strict=True) if key == wanted
    )

    # Another target changes the matches and nothing else that a rule selects.
    other = csp.evaluate_case(
        generator, stored, track="reduced-formula", candidates=candidates, seed=seed,
        likelihood_steps=4,
    )
    for rule in ("first", "map", "consensus"):
        assert other[rule]["attempt"] == case[rule]["attempt"]
    assert other["valid"] == case["valid"]

    # Primitive-count track: the request carries the target's primitive atom count.
    counted = csp.evaluate_case(
        generator, stored, track="primitive-count", candidates=candidates, seed=seed,
        likelihood_steps=2,
    )
    assert counted["primitive_atom_count"] == 2 and counted["track"] == "primitive-count"
    conditioned = generator.sample(replace(request, primitive_atom_count=2))
    assert all(
        len(trace.sample.physical_primitive.atomic_numbers) == 2
        for trace in conditioned
        if trace.succeeded
    )
    assert counted["valid"] == sum(trace.succeeded for trace in conditioned)

    # A failed likelihood integration is a recorded miss of the MAP rule only.
    def broken(self, samples, **_):  # type: ignore[no-untyped-def]
        raise ValueError("staged failure")

    with monkeypatch.context() as patch:
        patch.setattr(LearnedCrystalGenerator, "select_map_sample", broken)
        failed = csp.evaluate_case(
            generator, target, track="reduced-formula", candidates=candidates, seed=seed
        )
    assert failed["map"] == {
        "attempt": None,
        "match": False,
        "rms": None,
        "error": "ValueError: staged failure",
    }
    assert failed["consensus"] == case["consensus"] and failed["matches"] == case["matches"]

    # A motif with two sites of one species on top of each other decodes (the
    # collision tolerance of the decoder is far smaller) but has no matcher payload:
    # the encoder's primitive reduction refuses it.  It is a labelled miss.
    victim = next(
        index
        for index in valid
        if index != hit and len(traces[index].sample.physical_primitive.atomic_numbers) >= 2
    )
    physical = traces[victim].sample.physical_primitive
    assert physical.atomic_numbers[0] == physical.atomic_numbers[1]
    collapsed = np.array(physical.fractional_positions)
    collapsed[1] = collapsed[0] + np.array([1.0e-7, 0.0, 0.0])
    doctored = replace(physical, fractional_positions=collapsed)
    with pytest.raises(InvariantViolation, match="complete primitive fibres"):
        encode_physical_structure(
            doctored.lattice, doctored.fractional_positions, doctored.atomic_numbers
        )
    staged = list(traces)
    staged[victim] = replace(
        traces[victim], sample=replace(traces[victim].sample, physical_primitive=doctored)
    )
    with monkeypatch.context() as patch:
        patch.setattr(LearnedCrystalGenerator, "sample", lambda self, request: tuple(staged))
        collided = csp.evaluate_case(
            generator, target, track="reduced-formula", candidates=candidates, seed=seed,
            likelihood_steps=4,
        )
    assert collided["valid"] == case["valid"] - 1
    assert collided["failed_stages"] == {
        **case["failed_stages"],
        "evaluation/coincident-atoms": 1,
    }
    assert collided["encode_errors"] == []
    assert victim not in collided["matches"]
    assert collided["matches"] == [index for index in case["matches"] if index != victim]
    assert collided["shortest_distances"][victim] < 1.0e-5
    assert collided["structurally_valid"] == case["structurally_valid"] - (
        case["shortest_distances"][victim] > 0.5
    )
    assert victim not in (collided["map"]["attempt"], collided["consensus"]["attempt"])

    # Any other invariant failure of the encoder is a miss too, but it keeps its
    # message and is never merged into the declared causes.
    real_encode = csp.encode_physical_structure
    calls = iter(range(candidates))

    def unexplained(lattice, positions, numbers):  # type: ignore[no-untyped-def]
        if next(calls) == 0:
            raise InvariantViolation("staged defect")
        return real_encode(lattice, positions, numbers)

    with monkeypatch.context() as patch:
        patch.setattr(csp, "encode_physical_structure", unexplained)
        defect = csp.evaluate_case(
            generator, target, track="reduced-formula", candidates=candidates, seed=seed,
            likelihood_steps=4,
        )
    assert defect["failed_stages"] == {
        **case["failed_stages"],
        "evaluation/encode-invariant": 1,
    }
    assert defect["encode_errors"] == ["InvariantViolation: staged defect"]
    assert defect["valid"] == case["valid"] - 1 and valid[0] not in defect["matches"]
    assert defect["structurally_valid"] == case["structurally_valid"]
    summary = csp.summarize([collided, defect], track="reduced-formula", candidates=candidates)
    assert summary["failed_stages"]["evaluation/coincident-atoms"] == 1
    assert summary["failed_stages"]["evaluation/encode-invariant"] == 1
    assert summary["encode_invariant_errors"] == ["InvariantViolation: staged defect"]
    assert summary["encode_invariant_cases"] == 1

    # A request the cover sampler cannot serve: every attempt of the case is a miss.
    def refused(self, request):  # type: ignore[no-untyped-def]
        raise InvariantViolation("no formula-unit multiplier has an arithmetic cover")

    with monkeypatch.context() as patch:
        patch.setattr(LearnedCrystalGenerator, "sample", refused)
        missed = csp.evaluate_case(
            generator, target, track="reduced-formula", candidates=candidates, seed=seed
        )
    assert missed["request_error"].startswith("InvariantViolation: no formula-unit")
    assert missed["valid"] == 0 and missed["match_count"] == 0
    assert missed["failed_stages"] == {"request": candidates}
    assert missed["shortest_distances"] == [None] * candidates
    assert missed["structurally_valid"] == 0 and missed["encode_errors"] == []
    assert not missed["first"]["match"] and not missed["map"]["match"]
    assert not missed["consensus"]["match"]
    summary = csp.summarize([case, missed], track="reduced-formula", candidates=candidates)
    assert summary["cases"] == 2 and summary["request_error_cases"] == 1
    assert summary["match_at_k"]["successes"] == 1
    assert summary["valid_fraction"] == len(valid) / (2 * candidates)

    with pytest.raises(DomainError, match="track"):
        csp.evaluate_case(generator, target, track="x", candidates=candidates, seed=seed)


@pytest.mark.ml
def test_resumed_evaluation_reuses_finished_cases(csp, monkeypatch: pytest.MonkeyPatch) -> None:  # type: ignore[no-untyped-def]
    import json

    from crystalatlas.errors import DomainError
    from crystalatlas.learned import LearnedCrystalGenerator

    generator = _tiny_generator()
    targets = (_stored_target(csp, generator, "mp-a"), _stored_target(csp, generator, "mp-b"))
    generator.model.train()
    seen: list[dict[str, object]] = []
    first = csp.evaluate_csp(
        generator, targets, candidates=3, seed=11, likelihood_steps=2, on_case=seen.append
    )
    assert generator.model.training  # the caller's mode is restored
    assert [(case["record_id"], case["track"]) for case in seen] == [
        ("mp-a", "reduced-formula"),
        ("mp-b", "reduced-formula"),
        ("mp-a", "primitive-count"),
        ("mp-b", "primitive-count"),
    ]
    assert seen[0]["seed"] == csp.case_seed(11, "mp-a") != seen[1]["seed"]
    assert first["cases"] == seen and set(first["summary"]) == set(csp.TRACKS)
    assert first["protocol"]["schema"] == csp.SCHEMA
    json.dumps(first, allow_nan=False)  # the whole result is finite JSON

    # Resume from the JSON round trip of three finished cases: only the fourth is computed.
    done = {
        (case["record_id"], case["track"]): json.loads(json.dumps(case)) for case in seen[:3]
    }
    calls: list[object] = []
    original = LearnedCrystalGenerator.sample

    def counting(self, request):  # type: ignore[no-untyped-def]
        calls.append(request)
        return original(self, request)

    monkeypatch.setattr(LearnedCrystalGenerator, "sample", counting)
    fresh: list[dict[str, object]] = []
    second = csp.evaluate_csp(
        generator, targets, candidates=3, seed=11, likelihood_steps=2, done=done,
        on_case=fresh.append,
    )
    assert len(calls) == 1 and len(fresh) == 1
    strip = lambda case: {k: v for k, v in case.items() if k != "seconds"}  # noqa: E731
    assert [strip(case) for case in second["cases"]] == [strip(case) for case in seen]
    for track in csp.TRACKS:
        one, two = dict(first["summary"][track]), dict(second["summary"][track])
        one.pop("seconds"), two.pop("seconds")
        assert one == two

    with pytest.raises(DomainError, match="another protocol"):
        csp.evaluate_csp(generator, targets, candidates=3, seed=12, done=done)
    with pytest.raises(DomainError, match="another protocol"):
        csp.evaluate_csp(generator, targets, candidates=4, seed=11, done=done)
    with pytest.raises(DomainError, match="two candidates"):
        csp.evaluate_csp(generator, targets, candidates=1, seed=11)
    with pytest.raises(DomainError, match="at least one target"):
        csp.evaluate_csp(generator, (), candidates=3, seed=11)
    with pytest.raises(DomainError, match="tracks"):
        csp.evaluate_csp(generator, targets, candidates=3, seed=11, tracks=("x",))


def test_replay_distance_counts_use_the_declared_thresholds(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    pytest.importorskip("torch")
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    import diagnose_field_replay_v7 as replay

    rows = [
        {"d": [None, 0.0, 0.01, 0.0100001, 0.1]},
        {"d": [0.5, 0.5000001, 2.3, None]},
    ]
    assert replay.distance_counts(rows, "d") == {
        "attempts": 9,
        "without_structure": 2,
        "at_most_0.01": 2,
        "at_most_0.1": 4,
        "at_most_0.5": 5,
        "structurally_valid": 2,
    }


def test_oracle_case_integrates_the_targets_own_cover(csp, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.syspath_prepend(str(SCRIPTS))
    import diagnose_field_replay_v7 as replay

    generator = _tiny_generator()
    generator.model.eval()
    stored = _stored_target(csp, generator)
    draws, seed = 5, 41
    first = replay.oracle_case(generator, stored, draws=draws, seed=seed)
    assert first == replay.oracle_case(generator, stored, draws=draws, seed=seed)
    assert first["record_id"] == stored.record_id and first["draws"] == draws
    for key in ("shortest_distances", "matches", "rms"):
        assert len(first[key]) == draws
    assert [value is not None for value in first["rms"]] == first["matches"]
    without = sum(value is None for value in first["shortest_distances"])
    assert sum(first["failed_stages"].values()) >= without
    # The geometries are the generator's own integration of the declared seeds.
    seeds = np.random.SeedSequence([seed, replay.ORACLE_STREAM]).generate_state(
        draws, dtype=np.uint64
    )
    device, _ = generator._device_dtype()
    outcomes = generator._integrate_members_mixed(
        (stored.cover,) * draws,
        tuple(generator._torch_generator(int(value), device) for value in seeds),
    )
    numerics = generator.model.scientific_config.numerics
    layout = generator.model.layout(stored.cover)
    checked = 0
    for outcome, distance in zip(outcomes, first["shortest_distances"], strict=True):
        if distance is None:
            continue
        physical = layout.decode(
            outcome,
            collision_tolerance=numerics.collision_atol_angstrom,
            symmetry_tolerance=numerics.symmetry_atol,
        ).full_primitive_reduction(cartesian_tolerance=numerics.symmetry_atol)
        assert distance == pytest.approx(
            _brute_force_shortest(physical.lattice, physical.fractional_positions),
            rel=1e-10,
            abs=1e-10,
        )
        checked += 1
    assert checked >= 3
    other = replay.oracle_case(generator, stored, draws=draws, seed=seed + 1)
    assert other["shortest_distances"] != first["shortest_distances"]
