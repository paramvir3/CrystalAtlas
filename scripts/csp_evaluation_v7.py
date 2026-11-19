"""Composition-conditioned CSP measurement with the certified physical matcher.

Research protocol of program R3 (2026-11-10).  For every target record the
generator draws ``candidates`` independent attempts for the target's reduced
composition, and every reported quantity is read from those same attempts:

``match_at_1``
    the first attempt matches the target; a failed first attempt is a miss.
``match_per_attempt``
    the mean over all attempts of "this attempt matches".  The attempts of one
    request are exchangeable, so this estimates the same probability as
    ``match_at_1`` with a smaller variance.
``match_at_k``
    at least one attempt matches.
``match_map``
    the attempt chosen by the hierarchical model MAP rule matches
    (``LearnedCrystalGenerator.select_map_sample``: the cover of largest exact
    selector mass first, the largest exact geometry log density within that
    cover second, Proposition 2.9e).
``match_consensus``
    the attempt chosen by the consensus rule matches: the valid attempt that is
    equivalent to the largest number of other valid attempts under the same
    certified matcher, the lowest attempt index winning ties.  Given attempt
    ``i``, the other ``K - 1`` attempts are i.i.d. draws of the model, so
    ``votes_i / (K - 1)`` is an unbiased estimate of the model probability of
    the matcher neighbourhood of attempt ``i``.  The rule therefore selects a
    plug-in mode of the generated law over physical structures; it needs no
    density and no parametrization.

No selection rule reads the target.  No energy model, relaxation or filter is
used, and every attempt stays in every denominator.  An attempt is *valid* when
it decoded and its structure has a canonical matcher payload.  A decoded attempt
without a payload is a miss, labelled by cause:

``evaluation/coincident-atoms``
    two sites of one species lie within the matcher's primitive-reduction
    tolerance (0.01 angstrom) of each other, so the tolerance-based primitive
    reduction is undefined.  This is a generated orbit that collapsed onto a
    special position; the decoder's collision tolerance is 1e-8 angstrom, so
    such a motif still decodes.
``evaluation/encode``
    a declared domain failure of the encoder.
``evaluation/encode-invariant``
    any other invariant failure of the encoder; the message is kept on the case
    and listed in the summary.

``structurally_valid`` counts the decoded attempts whose shortest distance
between two distinct atoms of the infinite crystal, periodic images of one site
included, is larger than 0.5 angstrom (the distance criterion of the CDVAE
benchmark, Xie et al. 2021).  It is reported next to *valid*, which says only
that the matcher can read the structure.  Two tracks are
labelled separately: ``reduced-formula`` (the composition alone) and
``primitive-count`` (the composition and the atom count of the target's
primitive cell, the standard MP-20 protocol).

Matcher: ``crystalatlas.physical_matching`` (length 0.3, site 0.5, angle 10
degrees, volume-normalized, proper rotations only, so a chiral structure and its
mirror image do not match; this last point is stricter than matchers that admit
improper lattice maps).  RMS values are in units of the cube root of the volume
per atom and are averaged over the matched selections only.

The target is the stored canonical record of the dataset (symmetrized by the
compilation policy), decoded through the model layout; it is read by the
matcher only.  This is a bounded research measurement on a small corpus, not a
benchmark claim.
"""

from __future__ import annotations

import json
import math
import time
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass

import numpy as np

from crystalatlas.composition import Composition
from crystalatlas.errors import DomainError, InvariantViolation, RecoverableGenerationError
from crystalatlas.geometry import ContinuousCover, ContinuousGeometry
from crystalatlas.learned import LearnedCrystalGenerator
from crystalatlas.nn.field import TangentRankError
from crystalatlas.periodic import (
    minimum_periodic_pair_distance,
    periodic_pairs_within,
    shortest_nonzero_lattice_image,
)
from crystalatlas.physical_matching import (
    PRIMITIVE_REDUCTION_TOLERANCE_ANGSTROM,
    encode_physical_structure,
    physical_structure_match,
    physical_structure_matcher_contract,
    physical_structure_rms,
)
from crystalatlas.pipeline import GenerationMode, GenerationRequest

SCHEMA = "crystalatlas.research-v7-csp.v3"
TRACKS = ("reduced-formula", "primitive-count")
RULES = ("first", "map", "consensus")
STRUCTURAL_VALIDITY_DISTANCE_ANGSTROM = 0.5
"""A decoded attempt is structurally valid when its shortest distance exceeds this."""
_NORMAL_975 = 1.959963984540054
"""Standard normal 0.975 quantile (95 % two-sided Wilson interval)."""

_SELECTION_ERRORS = (
    RecoverableGenerationError,
    TangentRankError,
    ValueError,
    InvariantViolation,
)
"""Failures of the backward likelihood integration of already generated samples.

They are recorded on the case and counted as a miss of the MAP rule, never
dropped, and the summary lists every distinct message.
"""


@dataclass(frozen=True, slots=True)
class CspTarget:
    """One evaluation target; only the matcher and the count condition read it."""

    record_id: str
    composition: Composition
    cover: ContinuousCover
    payload: bytes
    primitive_atoms: int
    """Atom count of the full primitive reduction of the decoded target."""
    cover_primitive_atoms: int
    """``nu |c| / I_G``: primitive atoms implied by the target's cover."""
    payload_atoms: int
    """Atom count of the matcher payload (its own primitive-reduction tolerance)."""


def table_key(cover: ContinuousCover) -> tuple[object, ...]:
    """Order-free identity of a Wyckoff occupation table."""

    return (
        cover.nu,
        cover.space_group,
        cover.composition,
        tuple(sorted((orbit.wyckoff_label, orbit.atomic_number) for orbit in cover.orbits)),
    )


def target_from_record(model, record_id, scaffold, metric) -> CspTarget:  # type: ignore[no-untyped-def]
    """Decode one stored record through the model layout and encode it for the matcher."""

    cover = ContinuousCover.from_exact_scaffold(scaffold)
    layout = model.layout(cover)
    z = np.asarray(layout.chart.encode_latent(np.asarray(metric, dtype=np.float64)))
    eta = np.asarray(layout.encode_exact_parameters(scaffold))
    numerics = model.scientific_config.numerics
    decoded = layout.decode(
        ContinuousGeometry(z, eta),
        collision_tolerance=numerics.collision_atol_angstrom,
        symmetry_tolerance=numerics.symmetry_atol,
    )
    physical = decoded.full_primitive_reduction(cartesian_tolerance=numerics.symmetry_atol)
    payload = encode_physical_structure(
        physical.lattice, physical.fractional_positions, physical.atomic_numbers
    )
    centering = model.catalogue.get(cover.space_group).centering_index
    conventional = cover.nu * cover.composition.total_atoms
    if conventional % centering:
        raise InvariantViolation("a target cover has a fractional primitive atom count")
    return CspTarget(
        record_id=record_id,
        composition=cover.composition,
        cover=cover,
        payload=payload,
        primitive_atoms=len(physical.atomic_numbers),
        cover_primitive_atoms=conventional // centering,
        payload_atoms=payload.count(b"\nsite "),
    )


def collect_targets(model, batches: Iterable[object]) -> tuple[CspTarget, ...]:  # type: ignore[no-untyped-def]
    """Targets of every record of the batches, in record-id order.

    The order, and with it the per-case seed, does not depend on the batching.
    """

    targets = {}
    for batch in batches:
        for index, record_id in enumerate(batch.record_ids):  # type: ignore[attr-defined]
            if record_id in targets:
                raise InvariantViolation("a record occurs twice in the evaluation view")
            targets[record_id] = target_from_record(
                model,
                record_id,
                batch.geometry.scaffolds[index],  # type: ignore[attr-defined]
                batch.geometry.metric[index],  # type: ignore[attr-defined]
            )
    return tuple(targets[record_id] for record_id in sorted(targets))


def case_seed(seed: int, record_id: str) -> int:
    """Request seed of one record: a function of the run seed and the record id only."""

    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise DomainError("seed must be a nonnegative integer")
    entropy = [seed, *record_id.encode("utf-8")]
    return int(np.random.SeedSequence(entropy).generate_state(1, dtype=np.uint32)[0])


def wilson_interval(successes: int, trials: int) -> tuple[float, float]:
    """95 % Wilson score interval of a binomial proportion."""

    if isinstance(trials, bool) or not isinstance(trials, int) or trials < 1:
        raise DomainError("trials must be a positive integer")
    if isinstance(successes, bool) or not isinstance(successes, int):
        raise DomainError("successes must be an integer")
    if not 0 <= successes <= trials:
        raise DomainError("successes must lie in [0, trials]")
    z2 = _NORMAL_975 * _NORMAL_975
    rate = successes / trials
    denominator = 1.0 + z2 / trials
    centre = (rate + z2 / (2.0 * trials)) / denominator
    half = (
        _NORMAL_975
        * math.sqrt(rate * (1.0 - rate) / trials + z2 / (4.0 * trials * trials))
        / denominator
    )
    # At the boundary counts the closed endpoints are exact: the lower limit of 0/n is 0
    # and the upper limit of n/n is 1; rounding of centre -+ half must not move them.
    low = 0.0 if successes == 0 else max(0.0, centre - half)
    high = 1.0 if successes == trials else min(1.0, centre + half)
    return low, high


def shortest_interatomic_distance(lattice, fractional_positions) -> float:  # type: ignore[no-untyped-def]
    """Shortest distance between two distinct atoms of the infinite crystal, in angstrom.

    The periodic images of one site are distinct atoms, so the shortest nonzero
    lattice vector is an upper bound, and it is the answer for a one-site cell.
    ``minimum_periodic_pair_distance`` is exact whenever some pair lies within
    the supplied bound and returns a value above the bound otherwise, so the
    minimum of the two is exact in both cases.
    """

    cell = np.asarray(lattice, dtype=np.float64)
    points = np.asarray(fractional_positions, dtype=np.float64)
    shortest = math.sqrt(shortest_nonzero_lattice_image(cell).squared_distance)
    if len(points) < 2:
        return shortest
    return min(shortest, minimum_periodic_pair_distance(points, cell, upper_bound=shortest))


def coincident_same_species_pairs(  # type: ignore[no-untyped-def]
    lattice, fractional_positions, atomic_numbers, tolerance: float
) -> int:
    """Pairs of distinct sites of one species whose closest images lie within ``tolerance``."""

    points = np.asarray(fractional_positions, dtype=np.float64)
    numbers = np.asarray(atomic_numbers, dtype=np.int64).reshape(-1)
    within, _ = periodic_pairs_within(points, points, lattice, tolerance)
    same = numbers[:, None] == numbers[None, :]
    return int(np.count_nonzero(np.triu(within & same, k=1)))


def consensus_selection(equivalent: Sequence[Sequence[bool]]) -> tuple[int, tuple[int, ...]]:
    """First row with the most off-diagonal hits of a symmetric relation, and all votes."""

    size = len(equivalent)
    if size == 0:
        raise DomainError("consensus_selection needs at least one candidate")
    for i, row in enumerate(equivalent):
        if len(row) != size:
            raise DomainError("the equivalence relation must be a square matrix")
        for j in range(i):
            if bool(row[j]) != bool(equivalent[j][i]):
                raise DomainError("the equivalence relation must be symmetric")
    votes = tuple(
        sum(1 for j in range(size) if j != i and equivalent[i][j]) for i in range(size)
    )
    # ``max`` returns the first maximum: the lowest index wins a tie.
    return max(range(size), key=votes.__getitem__), votes


def mutual_equivalence(
    payloads: Sequence[bytes],
    match: Callable[[bytes, bytes], bool] = physical_structure_match,
) -> list[list[bool]]:
    """Symmetric matcher relation of the payloads; each unordered pair is matched once."""

    size = len(payloads)
    relation = [[False] * size for _ in range(size)]
    for i in range(size):
        relation[i][i] = True
        for j in range(i + 1, size):
            relation[i][j] = relation[j][i] = bool(match(payloads[i], payloads[j]))
    return relation


def _selection(attempt: int | None, match: Sequence[bool], rms: Mapping[int, float]):  # type: ignore[no-untyped-def]
    if attempt is None:
        return {"attempt": None, "match": False, "rms": None}
    return {"attempt": attempt, "match": bool(match[attempt]), "rms": rms.get(attempt)}


def evaluate_case(
    generator: LearnedCrystalGenerator,
    target: CspTarget,
    *,
    track: str,
    candidates: int,
    seed: int,
    likelihood_steps: int | None = None,
) -> dict[str, object]:
    """Draw the attempts of one target on one track and apply every selection rule."""

    if track not in TRACKS:
        raise DomainError(f"track must be one of {TRACKS}")
    count = target.primitive_atoms if track == "primitive-count" else None
    request = GenerationRequest(
        mode=GenerationMode.CSP,
        composition=target.composition,
        samples=candidates,
        seed=seed,
        primitive_atom_count=count,
    )
    case: dict[str, object] = {
        "record_id": target.record_id,
        "track": track,
        "composition": [list(pair) for pair in target.composition.counts],
        "space_group": target.cover.space_group,
        "nu": target.cover.nu,
        "target_primitive_atoms": target.primitive_atoms,
        "target_cover_primitive_atoms": target.cover_primitive_atoms,
        "target_payload_atoms": target.payload_atoms,
        "primitive_atom_count": count,
        "seed": seed,
        "attempts": candidates,
    }
    seconds: dict[str, float] = {}
    clock = time.perf_counter
    started = clock()
    try:
        traces = generator.sample(request)
    except InvariantViolation as exc:
        # The cover sampler found no admissible table for the request.  Every
        # attempt of the case is a miss; the message is kept and summarized.
        case.update(
            request_error=f"{type(exc).__name__}: {exc}",
            valid=0,
            failed_stages={"request": candidates},
            matches=[],
            match_count=0,
            first=_selection(None, (), {}),
            map={**_selection(None, (), {}), "error": None},
            consensus={**_selection(None, (), {}), "votes": None},
            best_rms=None,
            distinct_tables=0,
            target_table_valid=0,
            target_table_matches=0,
            shortest_distances=[None] * candidates,
            structurally_valid=0,
            encode_errors=[],
            seconds={"sample": round(clock() - started, 2)},
        )
        return case
    seconds["sample"] = clock() - started
    if len(traces) != candidates:
        raise InvariantViolation("the generator returned a wrong number of attempts")

    started = clock()
    payloads: list[bytes | None] = []
    stages: dict[str, int] = {}
    shortest: list[float | None] = []
    encode_errors: set[str] = set()
    for trace in traces:
        if not trace.succeeded:
            stage = str(trace.failed_stage)
            stages[stage] = stages.get(stage, 0) + 1
            payloads.append(None)
            shortest.append(None)
            continue
        physical = trace.sample.physical_primitive
        shortest.append(
            shortest_interatomic_distance(physical.lattice, physical.fractional_positions)
        )
        payload, failure = None, None
        try:
            payload = encode_physical_structure(
                physical.lattice, physical.fractional_positions, physical.atomic_numbers
            )
        except DomainError:
            failure = "evaluation/encode"
        except InvariantViolation as exc:
            # The encoder's primitive reduction presupposes that distinct sites of one
            # species are farther apart than its tolerance.  A motif that breaks this
            # premise has no payload and is a miss; any other invariant failure is
            # kept with its message, never merged into the declared causes.
            if coincident_same_species_pairs(
                physical.lattice,
                physical.fractional_positions,
                physical.atomic_numbers,
                PRIMITIVE_REDUCTION_TOLERANCE_ANGSTROM,
            ):
                failure = "evaluation/coincident-atoms"
            else:
                failure = "evaluation/encode-invariant"
                encode_errors.add(f"{type(exc).__name__}: {exc}")
        if failure is not None:
            stages[failure] = stages.get(failure, 0) + 1
        payloads.append(payload)
    valid = [index for index, payload in enumerate(payloads) if payload is not None]
    match = [
        payload is not None and bool(physical_structure_match(payload, target.payload))
        for payload in payloads
    ]
    rms: dict[int, float] = {}
    for index in valid:
        if match[index]:
            value = physical_structure_rms(payloads[index], target.payload)
            if value is None or not math.isfinite(value):
                raise InvariantViolation("a matched pair has no finite RMS witness")
            rms[index] = float(value)
    seconds["match"] = clock() - started

    started = clock()
    map_attempt = None
    map_detail: dict[str, object] = {"error": None}
    if valid:
        try:
            chosen = generator.select_map_sample(
                [traces[index].sample for index in valid], num_steps=likelihood_steps
            )
        except _SELECTION_ERRORS as exc:
            map_detail["error"] = f"{type(exc).__name__}: {exc}"
        else:
            map_attempt = valid[chosen.sample_index]
            map_detail.update(
                log_cover=chosen.log_cover,
                cover_members=[valid[index] for index in chosen.members],
                log_geometry=list(chosen.log_geometry),
            )
    seconds["map"] = clock() - started

    started = clock()
    consensus_attempt = None
    votes = None
    if valid:
        chosen_index, tally = consensus_selection(
            mutual_equivalence([payloads[index] for index in valid])  # type: ignore[misc]
        )
        consensus_attempt = valid[chosen_index]
        votes = tally[chosen_index]
    seconds["consensus"] = clock() - started

    tables = [table_key(traces[index].sample.cover_sample.cover) for index in valid]
    wanted = table_key(target.cover)
    on_target = [index for index, key in zip(valid, tables, strict=True) if key == wanted]
    case.update(
        request_error=None,
        valid=len(valid),
        failed_stages=dict(sorted(stages.items())),
        matches=[index for index in range(candidates) if match[index]],
        match_count=int(sum(match)),
        first=_selection(0, match, rms),
        map={**_selection(map_attempt, match, rms), **map_detail},
        consensus={**_selection(consensus_attempt, match, rms), "votes": votes},
        best_rms=min(rms.values()) if rms else None,
        distinct_tables=len(set(tables)),
        target_table_valid=len(on_target),
        target_table_matches=int(sum(match[index] for index in on_target)),
        shortest_distances=shortest,
        structurally_valid=sum(
            value is not None and value > STRUCTURAL_VALIDITY_DISTANCE_ANGSTROM
            for value in shortest
        ),
        encode_errors=sorted(encode_errors),
        seconds={name: round(value, 2) for name, value in seconds.items()},
    )
    return case


def _rate(successes: int, trials: int) -> dict[str, object]:
    low, high = wilson_interval(successes, trials)
    return {
        "successes": successes,
        "trials": trials,
        "rate": successes / trials,
        "wilson_95": [low, high],
    }


def _mean(values: Sequence[float]) -> float | None:
    return math.fsum(values) / len(values) if values else None


def _case_mean(fractions: Sequence[float]) -> dict[str, object]:
    """Mean of one fraction per case and its standard error over the cases.

    The attempts of one case share its composition: the case is the unit.
    """

    total = len(fractions)
    mean = math.fsum(fractions) / total
    return {
        "rate": mean,
        "standard_error": (
            math.sqrt(math.fsum((value - mean) ** 2 for value in fractions) / (total - 1))
            / math.sqrt(total)
            if total > 1
            else None
        ),
    }


def summarize(cases: Sequence[Mapping[str, object]], *, track: str, candidates: int):  # type: ignore[no-untyped-def]
    """Rates over the cases of one track; every case and attempt is in the denominator."""

    rows = [case for case in cases if case["track"] == track]
    total = len(rows)
    if total == 0:
        raise DomainError(f"no case on track {track}")
    if any(case["attempts"] != candidates for case in rows):
        raise DomainError("cases of one summary must share the candidate count")
    summary: dict[str, object] = {"track": track, "cases": total, "candidates": candidates}
    for rule in RULES:
        name = "match_at_1" if rule == "first" else f"match_{rule}"
        summary[name] = _rate(sum(bool(case[rule]["match"]) for case in rows), total)
        summary[f"mean_rms_{rule}"] = _mean(
            [case[rule]["rms"] for case in rows if case[rule]["match"]]
        )
    summary["match_at_k"] = _rate(sum(case["match_count"] > 0 for case in rows), total)
    summary["mean_best_rms_at_k"] = _mean(
        [case["best_rms"] for case in rows if case["best_rms"] is not None]
    )
    summary["match_per_attempt"] = _case_mean(
        [case["match_count"] / candidates for case in rows]
    )
    summary["valid_fraction"] = sum(case["valid"] for case in rows) / (total * candidates)
    summary["structurally_valid_fraction"] = _case_mean(
        [case["structurally_valid"] / candidates for case in rows]
    )
    stages: dict[str, int] = {}
    for case in rows:
        for stage, number in case["failed_stages"].items():
            stages[stage] = stages.get(stage, 0) + int(number)
    summary["failed_stages"] = dict(sorted(stages.items()))
    summary["encode_invariant_errors"] = sorted(
        {str(message) for case in rows for message in case["encode_errors"]}
    )
    summary["encode_invariant_cases"] = sum(bool(case["encode_errors"]) for case in rows)
    summary["target_table_sampled"] = _rate(
        sum(case["target_table_valid"] > 0 for case in rows), total
    )
    on_target = sum(case["target_table_valid"] for case in rows)
    summary["match_given_target_table"] = (
        _rate(sum(case["target_table_matches"] for case in rows), on_target)
        if on_target
        else None
    )
    summary["mean_distinct_tables"] = _mean([float(case["distinct_tables"]) for case in rows])
    summary["count_inconsistent_targets"] = sum(
        not (
            case["target_primitive_atoms"]
            == case["target_cover_primitive_atoms"]
            == case["target_payload_atoms"]
        )
        for case in rows
    )
    summary["request_errors"] = sorted(
        {str(case["request_error"]) for case in rows if case["request_error"]}
    )
    summary["request_error_cases"] = sum(bool(case["request_error"]) for case in rows)
    summary["map_errors"] = sorted(
        {str(case["map"]["error"]) for case in rows if case["map"]["error"]}
    )
    summary["map_error_cases"] = sum(bool(case["map"]["error"]) for case in rows)
    summary["seconds"] = {
        name: round(math.fsum(case["seconds"].get(name, 0.0) for case in rows), 1)
        for name in ("sample", "match", "map", "consensus")
    }
    return summary


def protocol(*, candidates: int, seed: int, likelihood_steps: int | None, tracks):  # type: ignore[no-untyped-def]
    """The declared measurement, written next to every result."""

    return {
        "schema": SCHEMA,
        "candidates": candidates,
        "seed": seed,
        "case_seed": "SeedSequence([seed, *utf8(record_id)]) first uint32",
        "likelihood_steps": likelihood_steps,
        "tracks": list(tracks),
        "rules": {
            "first": "attempt 0; a failed attempt is a miss",
            "map": "largest selector mass among sampled covers, then largest exact "
            "geometry log density within that cover; first occurrence wins ties",
            "consensus": "valid attempt matching the most other valid attempts under "
            "the certified matcher; lowest attempt index wins ties",
        },
        "matcher": physical_structure_matcher_contract().to_manifest(),
        "unencodable": {
            "evaluation/coincident-atoms": "two sites of one species within the "
            "matcher's primitive-reduction tolerance; a miss",
            "evaluation/encode": "declared domain failure of the encoder; a miss",
            "evaluation/encode-invariant": "other invariant failure of the encoder; "
            "a miss, message kept",
        },
        "structural_validity": {
            "distance_angstrom": STRUCTURAL_VALIDITY_DISTANCE_ANGSTROM,
            "rule": "decoded attempt whose shortest distance between two distinct "
            "atoms, periodic images of one site included, is larger than the distance",
        },
        "rms_unit": "cube root of the volume per atom; mean over matched selections",
        "scope": "small research corpus; no relaxation, energy model or filter; "
        "every attempt is in every denominator",
    }


def evaluate_csp(
    generator: LearnedCrystalGenerator,
    targets: Sequence[CspTarget],
    *,
    candidates: int,
    seed: int,
    tracks: Sequence[str] = TRACKS,
    likelihood_steps: int | None = None,
    done: Mapping[tuple[str, str], Mapping[str, object]] | None = None,
    on_case: Callable[[Mapping[str, object]], None] | None = None,
) -> dict[str, object]:
    """Evaluate every target on every track; ``done`` holds cases of a resumed run.

    A resumed case is reused only when its seed and candidate count equal the
    ones this call would use; ``on_case`` receives every newly computed case.
    """

    if candidates < 2:
        raise DomainError("the consensus rule needs at least two candidates")
    if not targets:
        raise DomainError("evaluate_csp needs at least one target")
    if len(set(tracks)) != len(tracks) or any(track not in TRACKS for track in tracks):
        raise DomainError(f"tracks must be distinct members of {TRACKS}")
    was_training = generator.model.training
    generator.model.eval()
    cases: list[Mapping[str, object]] = []
    try:
        for track in tracks:
            for target in targets:
                request_seed = case_seed(seed, target.record_id)
                previous = None if done is None else done.get((target.record_id, track))
                if previous is not None:
                    if previous["seed"] != request_seed or previous["attempts"] != candidates:
                        raise DomainError("a resumed case was computed under another protocol")
                    cases.append(previous)
                    continue
                case = evaluate_case(
                    generator,
                    target,
                    track=track,
                    candidates=candidates,
                    seed=request_seed,
                    likelihood_steps=likelihood_steps,
                )
                cases.append(case)
                if on_case is not None:
                    on_case(case)
    finally:
        generator.model.train(was_training)
    return {
        "protocol": protocol(
            candidates=candidates, seed=seed, likelihood_steps=likelihood_steps, tracks=tracks
        ),
        "summary": {
            track: summarize(cases, track=track, candidates=candidates) for track in tracks
        },
        "cases": list(cases),
    }


def print_case(case: Mapping[str, object]) -> None:
    print(json.dumps(case, allow_nan=False), flush=True)
