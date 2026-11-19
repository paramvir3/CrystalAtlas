"""Measurement M4: the opt-in start-rejection sampler (``start_min_separation``).

Registered in R1_LOG.md, "Measurement M4" (2026-11-16).  The paired field replay of
``diagnose_field_replay_v7`` (covers from the cover run, geometry from the field run,
the protocol seed of the cover run, the fused sampler, the likelihood rule off) on one
partition, with the package option ``LearnedCrystalGenerator.start_min_separation``
set to ``--threshold`` (``none`` is the untreated sampler).  The option does the
treatment; this script only observes, with arithmetic-neutral wrappers:

* every padded solve: its plan, covers and generators, the number of base draws each
  member's generator made in that solve, and, after the solve, M2's start block
  (``m2_starts.start_summary``, radius 4 angstrom) of the state that was integrated;
  for a block that search leaves unresolved, an exact search at the threshold (at
  ``UNTREATED_SEARCH`` angstrom in an untreated run); and the unrounded d0 up to
  ``UNTREATED_SEARCH`` angstrom by the option's own search
  (``learned.closest_distinct_pair_within``), on which the option decides;
* every replay attempt's trace: its failed stage and error, and the decoded volume per
  atom of its structure;
* every single-member fallback solve and every padded solve that raised.

Each attempt is checked against a reference: its cover and, when it was drawn once,
its start block and outcome equal the reference's exactly; when it was drawn more
than once, the reference start is near (d0 <= threshold); the integrated start is far.
References: on the test records (the selection), M2's start blocks (the first draws,
shared by every field) and this field's stored replay; on the validation records (the
confirmation), this script's untreated run of the same field.  Research diagnostic: it
reads private attributes of the generator.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from pathlib import Path

import diagnose_field_replay_v7 as replay
import m1_contacts as m1
import m2_starts as m2
import numpy as np
import torch
import train_research_v7 as runner
from csp_evaluation_v7 import case_seed, collect_targets, evaluate_case
from evaluate_csp_research_v7 import (
    add_run_arguments,
    code_fingerprint,
    configure_torch,
    load_run,
    open_research_store,
    partition_record_ids,
)

import crystalatlas.learned as learned
from crystalatlas.errors import DomainError, InvariantViolation, RecoverableGenerationError
from crystalatlas.flow import canonicalize_torus_validated
from crystalatlas.trainer import TrainingPhase
from crystalatlas.training_data import StoredTrainingDataset

SCHEMA = "crystalatlas.research-v7-m4-rejection.v1"
RADIUS = 4.0
MAX_REACH = 8
"""M2's start block: exact image distances up to RADIUS, at most MAX_REACH images per axis."""
UNTREATED_SEARCH = 0.5
"""Angstrom; the exact search of an unresolved block in an untreated run (the largest
threshold of the registered sequence), so that later treated runs can read it."""
INVALID_DISTANCE = 0.5
M1_SHA256 = "1f8332a87fc8c707c71ad35a745862665bc2b4c7cd19ebd71465950e0d3ca432"
M2_SHA256 = "5adfad80bc2d570fa08b53a5434549e8d5fe4c87218e7082aed48c4120220bd9"
M2_RESULT_SCHEMA = "crystalatlas.research-v7-m2-starts.v1"
M2_RESULT_SHA256 = "60ccf1fefafa78929b64c9b7a18cfb0bc40ef6aeb405d0430668d07d6614f844"
STORED_SCHEMA = "crystalatlas.research-v7-field-replay.v1"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class ObserverError(RuntimeError):
    """A defect of this script's wrappers.

    Not an ``InvariantViolation``, which ``evaluate_case`` would record as a request
    error, and not a member error of the sampler: it ends the run.
    """


class Observer:
    """What the wrappers saw in the case in progress, and counters of the run."""

    def __init__(self) -> None:
        self.created: list[tuple[int, torch.Generator]] = []
        self.rows: dict[int, dict[str, object]] = {}
        self.pending_plan = None
        self.in_chunk = False
        self.draws: dict[int, int] = {}
        self.current: list[object] = []
        self.traces: tuple[object, ...] | None = None
        self.bucket_solves: list[list[object]] = []
        self.failed_chunks: list[dict[str, object]] = []
        self.unresolved_searches = 0


def nearness(attempt: dict[str, object], threshold: float) -> bool | None:
    """d0 <= threshold of an attempt's integrated start; None when it cannot be decided.

    The unrounded d0 of the option's own search decides when it was recorded (it is
    None when no pair is within UNTREATED_SEARCH).  Otherwise (M2's records) a resolved
    block decides on its 7-digit r, and an unresolved one by the stored exact search
    when its radius is at least the threshold.
    """

    if (
        "d0_exact" in attempt
        and attempt["d0_exact"] != "unresolved"
        and threshold <= UNTREATED_SEARCH
    ):
        value = attempt["d0_exact"]
        return value is not None and float(value) <= threshold  # type: ignore[arg-type]
    block = attempt["start"]
    if block["resolved"]:  # type: ignore[index]
        closest = block["closest"]  # type: ignore[index]
        return closest is not None and float(closest["r"]) <= threshold
    search = attempt.get("threshold_search")
    if search is None or float(search["radius"]) < threshold:  # type: ignore[index]
        return None
    value = search["d0"]  # type: ignore[index]
    return value is not None and float(value) <= threshold


def install(generator, model, observer: Observer, threshold: float | None) -> None:  # type: ignore[no-untyped-def]
    plan_class = learned.PaddedCoverPlan

    class ObservedPlan(plan_class):  # type: ignore[misc,valid-type]
        def __init__(self, *args, **kwargs) -> None:  # type: ignore[no-untyped-def]
            super().__init__(*args, **kwargs)
            observer.pending_plan = self

    learned.PaddedCoverPlan = ObservedPlan  # type: ignore[misc]

    sample = learned.sample_product_base

    def counting_sample(*args, **kwargs):  # type: ignore[no-untyped-def]
        base = sample(*args, **kwargs)
        key = id(kwargs["generator"])
        observer.draws[key] = observer.draws.get(key, 0) + 1
        return base

    learned.sample_product_base = counting_sample  # type: ignore[assignment]

    integrate = learned.integrate_with_endpoint_estimate
    solve_inputs: dict[str, object] = {}
    search_radius = UNTREATED_SEARCH if threshold is None else threshold

    def observed_integrator(field, initial_state, period_basis, **kwargs):  # type: ignore[no-untyped-def]
        plan, observer.pending_plan = observer.pending_plan, None
        if not observer.in_chunk:
            # A single-member fallback solve (bucket path); not a padded solve.
            observer.bucket_solves.append(list(observer.current))
            return integrate(field, initial_state, period_basis, **kwargs)
        if plan is None or "covers" not in solve_inputs:
            raise ObserverError("a padded solve has no plan or no members")
        final = integrate(field, initial_state, period_basis, **kwargs)
        try:
            observe(plan, initial_state, period_basis)
        except ObserverError:
            raise
        except Exception as exc:
            # Not a member error of the sampler: a fault of this observation ends the run.
            raise ObserverError(f"observation failed: {type(exc).__name__}: {exc}") from exc
        return final

    def observe(plan, initial_state, period_basis) -> None:  # type: ignore[no-untyped-def]
        covers, generators = solve_inputs["covers"], solve_inputs["generators"]
        if not (len(covers) == len(generators) == plan.batch):  # type: ignore[arg-type]
            raise ObserverError("the padded solve's members are not its plan's rows")
        with torch.no_grad():
            canonical = canonicalize_torus_validated(initial_state.eta, period_basis)
            pseudo = {
                "plan": plan,
                "covers": covers,
                "times": (0.0,),
                "z": (initial_state.z,),
                "eta": (canonical,),
            }
            expanded_all = plan.expand(initial_state.z, canonical)
            for row, item in enumerate(generators):  # type: ignore[arg-type]
                block = m2.start_summary(pseudo, row, model, radius=RADIUS, max_reach=MAX_REACH)
                entry: dict[str, object] = {
                    "cover": covers[row],  # type: ignore[index]
                    "draws": observer.draws.get(id(item), 0),
                    "start": block,
                }
                # The unrounded d0 by the option's own search, as the option decides it.
                try:
                    entry["d0_exact"] = learned.closest_distinct_pair_within(
                        expanded_all.lattice[row],
                        expanded_all.fractional_coordinates[row, : plan.atom_counts[row]],
                        UNTREATED_SEARCH,
                    )
                except RecoverableGenerationError:
                    entry["d0_exact"] = "unresolved"
                if not block["resolved"]:
                    observer.unresolved_searches += 1
                    count = plan.atom_counts[row]
                    expanded = plan.expand(initial_state.z, canonical)
                    found = m1.pair_geometry(
                        expanded.lattice[row].to(dtype=torch.float64),
                        expanded.fractional_coordinates[row, :count].to(dtype=torch.float64),
                        search_radius,
                        MAX_REACH,
                    )
                    if found is None:
                        raise ObserverError("the exact search at the threshold is unresolved")
                    distance = found[0]
                    values = [
                        float(distance[i, j])
                        for i in range(count)
                        for j in range(i + 1, count)
                        if distance[i, j] <= search_radius
                    ]
                    entry["threshold_search"] = {
                        "radius": search_radius,
                        "d0": m1.significant(min(values)) if values else None,
                    }
                observer.rows[id(item)] = entry

    learned.integrate_with_endpoint_estimate = observed_integrator  # type: ignore[assignment]

    chunk = generator._integrate_mixed_chunk

    def observed_chunk(covers, generators):  # type: ignore[no-untyped-def]
        observer.pending_plan = None
        observer.draws = {}
        solve_inputs.clear()
        solve_inputs.update(covers=tuple(covers), generators=tuple(generators))
        observer.in_chunk = True
        try:
            return chunk(covers, generators)
        except Exception as exc:
            observer.failed_chunks.append(
                {
                    "case": list(observer.current),
                    "error": f"{type(exc).__name__}: {exc}",
                    "batch": len(covers),
                    "space_groups": [cover.space_group for cover in covers],
                }
            )
            raise
        finally:
            observer.in_chunk = False
            solve_inputs.clear()

    generator._integrate_mixed_chunk = observed_chunk

    make = generator._torch_generator

    def observed_generator(seed, device):  # type: ignore[no-untyped-def]
        made = make(seed, device)
        observer.created.append((int(seed), made))
        return made

    generator._torch_generator = observed_generator

    draw = generator.sample

    def observed_sample(request):  # type: ignore[no-untyped-def]
        traces = draw(request)
        observer.traces = tuple(traces)
        return traces

    generator.sample = observed_sample


def attempt_entry(observer: Observer, model, item, index: int, shortest, matched: bool):  # type: ignore[no-untyped-def]
    entry: dict[str, object] = {"index": index, "shortest": shortest, "matched": matched}
    row = None if item is None else observer.rows.get(id(item))
    if row is None:
        entry["solved"] = False
        return entry
    cover = row["cover"]
    _, _, orbits = m1.cell_atoms(model, cover)
    entry.update(
        solved=True,
        space_group=cover.space_group,  # type: ignore[attr-defined]
        nu=cover.nu,  # type: ignore[attr-defined]
        orbits=orbits,
        draws=row["draws"],
        start=row["start"],
    )
    if "threshold_search" in row:
        entry["threshold_search"] = row["threshold_search"]
    entry["d0_exact"] = row["d0_exact"]
    return entry


def trace_fields(trace) -> dict[str, object]:  # type: ignore[no-untyped-def]
    """Failed stage, error and decoded volume per atom of one replay attempt."""

    fields: dict[str, object] = {"failed_stage": trace.failed_stage, "error": trace.error}
    if trace.succeeded:
        physical = trace.sample.physical_primitive
        lattice = np.asarray(physical.lattice, dtype=float)
        atoms = len(physical.atomic_numbers)
        fields["decoded_volume_per_atom"] = m1.significant(abs(float(np.linalg.det(lattice))) / atoms)
    return fields


def invalid(value: float | None) -> bool:
    return value is None or value <= INVALID_DISTANCE


def checks(result: dict[str, object], starts: dict[str, object] | None,
           outcomes: dict[str, object] | None) -> dict[str, int]:
    """Every attempt against the reference starts and outcomes (see the module docstring).

    The registered analysis recomputes these from the raw files; they are written here
    so that a run can be read on its own.
    """

    threshold = result["threshold"]
    counts = {
        "attempts": 0, "solved": 0, "unsolved": 0, "unsolved_reference_near": 0,
        "reference_unsolved": 0, "redrawn": 0, "redrawn_untreated": 0, "cover_differs": 0,
        "redrawn_reference_not_near": 0, "reference_near_not_redrawn": 0,
        "reference_undecided": 0, "used_near": 0, "used_undecided": 0,
        "kept_start_differs": 0, "kept_outcome_differs": 0,
        "kept_outcome_differs_in_failed_solve_cases": 0, "draws_zero": 0,
    }
    # Kept attempts in a case with a failed padded solve, in this run or in the reference
    # run, were integrated in different solves and agree only to rounding.
    failed_cases = {tuple(item["case"]) for item in result["failed_padded_solves"]}  # type: ignore[union-attr]
    for reference in (starts, outcomes):
        if reference is not None:
            failed_cases |= {tuple(item["case"]) for item in reference.get("failed_padded_solves", [])}
    reference_starts = (
        {case["record_id"]: case for case in starts["records"]} if starts is not None else None
    )
    stored_replay = stored_oracle = None
    if outcomes is not None:
        stored_replay = {case["record_id"]: case for case in outcomes["replay_cases"]}
        stored_oracle = {case["record_id"]: case for case in outcomes["oracle_cases"]}
    for case in result["records"]:  # type: ignore[union-attr]
        record_id = case["record_id"]
        for kind in ("replay", "oracle"):
            for attempt in case[kind]:
                counts["attempts"] += 1
                index = attempt["index"]
                previous = reference_starts[record_id][kind][index] if reference_starts else None
                if previous is not None and not previous.get("solved"):
                    counts["reference_unsolved"] += 1
                    previous = None
                was_near = None
                if previous is not None and threshold is not None:
                    was_near = nearness(previous, threshold)
                    counts["reference_undecided"] += was_near is None
                if not attempt["solved"]:
                    counts["unsolved"] += 1
                    counts["unsolved_reference_near"] += bool(was_near)
                    continue
                counts["solved"] += 1
                counts["draws_zero"] += attempt["draws"] < 1
                redrawn = attempt["draws"] > 1
                counts["redrawn"] += redrawn
                if threshold is None:
                    counts["redrawn_untreated"] += redrawn
                else:
                    used = nearness(attempt, threshold)
                    counts["used_undecided"] += used is None
                    counts["used_near"] += bool(used)
                if previous is None:
                    continue
                counts["cover_differs"] += any(
                    attempt[name] != previous.get(name) for name in ("space_group", "nu", "orbits")
                )
                if was_near is not None:
                    counts["redrawn_reference_not_near"] += redrawn and not was_near
                    counts["reference_near_not_redrawn"] += was_near and not redrawn
                if redrawn:
                    continue
                counts["kept_start_differs"] += attempt["start"] != previous["start"]
                if stored_replay is None:
                    continue
                if kind == "replay":
                    old_case = stored_replay[record_id]
                    old = (old_case["shortest_distances"][index], index in old_case["matches"])
                else:
                    old_case = stored_oracle[record_id]  # type: ignore[index]
                    old = (old_case["shortest_distances"][index], old_case["matches"][index])
                if (attempt["shortest"], attempt["matched"]) != old:
                    key = ("kept_outcome_differs_in_failed_solve_cases"
                           if (record_id, kind) in failed_cases else "kept_outcome_differs")
                    counts[key] += 1
    return counts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_run_arguments(parser)  # --run is the cover run
    parser.add_argument("--field-run", type=Path, required=True)
    parser.add_argument("--field-label", default="after-geometry")
    parser.add_argument("--partition", choices=("validation", "test"), required=True)
    parser.add_argument("--threshold", required=True, help="angstrom, or 'none' (untreated)")
    parser.add_argument("--reference-starts", type=Path, default=None)
    parser.add_argument("--reference-outcomes", type=Path, default=None)
    parser.add_argument("--output", type=Path, required=True, help="result JSON (new file)")
    parser.add_argument("--candidates", type=int, default=20)
    parser.add_argument("--oracle-draws", type=int, default=20)
    parser.add_argument("--max-cases", type=int, default=None)
    args = parser.parse_args()

    for module, expected in ((m1, M1_SHA256), (m2, M2_SHA256)):
        found = sha256(Path(module.__file__))
        if found != expected:
            raise InvariantViolation(f"{module.__name__} has sha256 {found}, not {expected}")
    threshold = None if args.threshold == "none" else float(args.threshold)
    if threshold is not None and not (math.isfinite(threshold) and threshold > 0.0):
        raise DomainError("--threshold must be 'none' or a positive length")
    if threshold is not None and threshold > UNTREATED_SEARCH:
        raise DomainError(f"--threshold above {UNTREATED_SEARCH} is not registered")

    cover_run = args.run.absolute()
    field_run = args.field_run.absolute()
    arguments = json.loads((cover_run / "training-contract.json").read_text())["arguments"]
    seed = int(arguments["seed"]) + 500
    device = configure_torch(args.device, args.cpu_threads, seed)
    pins = dict(
        certification=args.certification,
        certification_fingerprint=args.certification_fingerprint,
        device=device,
        geometry_sampler="mixed-v1",
    )
    covers = load_run(cover_run, args.checkpoint_label, **pins)
    field = load_run(field_run, args.field_label, catalogue=covers.catalogue, **pins)
    code = code_fingerprint()

    # The references, checked before any work.
    starts = outcomes = None
    references: dict[str, object] = {}
    if args.partition == "test":
        if args.reference_starts is None or args.reference_outcomes is None:
            raise DomainError("the test partition needs M2's starts and the stored replay")
        if sha256(args.reference_starts) != M2_RESULT_SHA256:
            raise InvariantViolation("--reference-starts is not the registered M2 result")
        starts = json.loads(args.reference_starts.read_text())
        outcomes = json.loads(args.reference_outcomes.read_text())
        if starts["schema"] != M2_RESULT_SCHEMA or outcomes["schema"] != STORED_SCHEMA:
            raise InvariantViolation("the test references have the wrong schemas")
        if Path(outcomes["field_run"]).resolve() != field_run.resolve():
            raise DomainError("--reference-outcomes belongs to another field run")
        if outcomes["field_checkpoint_sha256"] != field.checkpoint_sha256:
            raise DomainError("the field checkpoint differs from the stored replay's")
    elif threshold is not None:
        if args.reference_starts is None or args.reference_outcomes != args.reference_starts:
            raise DomainError(
                "a treated validation run needs the untreated run as both references"
            )
        starts = outcomes = json.loads(args.reference_starts.read_text())
        if not (
            starts["schema"] == SCHEMA
            and starts["threshold"] is None
            and starts["partition"] == "validation"
            and starts["field_checkpoint_sha256"] == field.checkpoint_sha256
            and starts["code"] == code
            and starts["m4_script_sha256"] == sha256(Path(__file__))
            and starts["cases"] == 50
        ):
            raise InvariantViolation("--reference-starts is not this field's untreated run")
    elif args.reference_starts is not None or args.reference_outcomes is not None:
        raise DomainError("the untreated validation run has no reference")
    for name, path in (("starts", args.reference_starts), ("outcomes", args.reference_outcomes)):
        if path is not None:
            references[name] = {"path": str(path.absolute()), "sha256": sha256(path)}
    for reference in (starts, outcomes):
        if reference is None:
            continue
        if reference["protocol_seed"] != seed:
            raise DomainError("a reference has another protocol seed")
        if reference["cover_checkpoint_sha256"] != covers.checkpoint_sha256:
            raise DomainError("a reference has another cover checkpoint")
        if (reference["candidates"], reference["oracle_draws"]) != (
            args.candidates, args.oracle_draws
        ):
            raise DomainError("a reference has other candidate or oracle draw counts")

    generator = field.generator
    generator._cover_sampler = covers.generator._cover_sampler
    generator.select_map_sample = replay._no_likelihood  # type: ignore[method-assign]
    generator.start_min_separation = threshold
    model = generator.model

    with open_research_store(args) as store:
        record_ids = partition_record_ids(covers, store, args.partition)
        selected = tuple(sorted(record_ids))
        for reference in (starts, outcomes):
            if reference is not None and tuple(reference["record_ids"]) != selected:
                raise DomainError("a reference has other records")
        if args.max_cases is not None:
            selected = selected[: args.max_cases]
        dataset = StoredTrainingDataset(
            store,
            covers.model.composition_codec,
            covers.catalogue,
            domain=covers.model.scientific_config.domain,
            record_ids=selected,
        )
        targets = collect_targets(
            covers.model,
            runner._sampler(
                dataset,
                TrainingPhase.GEOMETRY,
                int(arguments["seed"]),
                int(arguments["batch_size"]),
            ),
        )
    if tuple(target.record_id for target in targets) != selected:
        raise DomainError("the evaluation targets are not the selected records")

    observer = Observer()
    install(generator, model, observer, threshold)
    started = time.perf_counter()
    replay_cases: list[dict[str, object]] = []
    oracle_cases: list[dict[str, object]] = []
    records: list[dict[str, object]] = []
    generator.model.eval()
    for target in targets:
        clock = time.perf_counter()
        request_seed = case_seed(seed, target.record_id)
        observer.created.clear()
        observer.rows.clear()
        observer.traces = None
        observer.current = [target.record_id, "replay"]
        case = evaluate_case(
            generator, target, track=replay.TRACK, candidates=args.candidates, seed=request_seed
        )
        # A request error (the cover sampler found no table) is an outcome: every attempt
        # of the case is a miss without a structure.  The analysis requires the same
        # request errors in a run and its reference.  Defects of this script are
        # ObserverError and end the run instead.
        traces = observer.traces
        if case.get("request_error") is None and (
            traces is None or len(traces) != args.candidates
        ):
            raise ObserverError("the replay traces were not observed")
        replay_generators = list(observer.created)
        observer.current = [target.record_id, "oracle"]
        own = replay.oracle_case(generator, target, draws=args.oracle_draws, seed=request_seed)
        oracle_generators = observer.created[len(replay_generators) :]
        # The geometry generator of attempt i is seeded with word 3 of child i of the
        # request's seed sequence (LearnedCrystalGenerator._sample_in_eval_mode); the
        # oracle draws with the words of [case seed, ORACLE_STREAM].
        made: dict[int, list[torch.Generator]] = {}
        for value, item in replay_generators:
            made.setdefault(value, []).append(item)
        geometry_seeds = [
            int(child.generate_state(4, dtype=np.uint64)[3])
            for child in np.random.SeedSequence(request_seed).spawn(args.candidates)
        ]
        replay_items = []
        for value in geometry_seeds:
            found = made.get(value, [])
            if len(found) > 1:
                raise ObserverError("two generators of one case share a geometry seed")
            # None: the attempt failed before its geometry generator was made.
            replay_items.append(found[0] if found else None)
        oracle_seeds = [
            int(value)
            for value in np.random.SeedSequence(
                [request_seed, replay.ORACLE_STREAM]
            ).generate_state(args.oracle_draws, dtype=np.uint64)
        ]
        if [value for value, _ in oracle_generators] != oracle_seeds:
            raise ObserverError("the oracle generators are not the oracle draws' seeds")
        for reference in (starts, outcomes):
            if reference is None:
                continue
            seeds = {item["record_id"]: item["seed"] for item in reference.get("records", [])}
            seeds.update(
                {item["record_id"]: item["seed"] for item in reference.get("replay_cases", [])}
            )
            if seeds.get(target.record_id) != request_seed:
                raise InvariantViolation("a reference has another request seed")
        entry: dict[str, object] = {"record_id": target.record_id, "seed": request_seed}
        entry["replay"] = []
        for index, item in enumerate(replay_items):
            attempt = attempt_entry(
                observer, model, item, index,
                case["shortest_distances"][index], index in case["matches"],
            )
            if traces is not None:
                if traces[index].sample_index != index:
                    raise ObserverError("the replay traces are out of order")
                attempt.update(trace_fields(traces[index]))
            entry["replay"].append(attempt)  # type: ignore[union-attr]
        entry["oracle"] = [
            attempt_entry(
                observer, model, item, index, own["shortest_distances"][index], own["matches"][index]
            )
            for index, (_, item) in enumerate(oracle_generators)
        ]
        entry["seconds"] = round(time.perf_counter() - clock, 2)
        replay_cases.append(case)
        oracle_cases.append(own)
        records.append(entry)
        drawn = [a["draws"] for kind in ("replay", "oracle") for a in entry[kind] if a["solved"]]
        print(json.dumps({
            "record_id": target.record_id[:12],
            "redrawn": sum(value > 1 for value in drawn),
            "draws_max": max(drawn, default=0),
            "replay_invalid": sum(invalid(value) for value in case["shortest_distances"]),
            "oracle_invalid": sum(invalid(value) for value in own["shortest_distances"]),
            "request_error": case.get("request_error"),
            "seconds": entry["seconds"],
        }), flush=True)
    result: dict[str, object] = {
        "schema": SCHEMA,
        "partition": args.partition,
        "threshold": threshold,
        "max_draws": learned.START_REJECTION_MAX_DRAWS,
        "max_images": learned.START_REJECTION_MAX_IMAGES,
        "generator_start_min_separation": generator.start_min_separation,
        "cover_run": str(cover_run),
        "cover_checkpoint_sha256": covers.checkpoint_sha256,
        "field_run": str(field_run),
        "field_checkpoint_sha256": field.checkpoint_sha256,
        "field_options": field.arguments.get("options", "default"),
        "protocol_seed": seed,
        "candidates": args.candidates,
        "oracle_draws": args.oracle_draws,
        "record_ids": [target.record_id for target in targets],
        "cases": len(records),
        "radius": RADIUS,
        "max_reach": MAX_REACH,
        "untreated_search": UNTREATED_SEARCH,
        "code": code,
        "m4_script_sha256": sha256(Path(__file__)),
        "m1_script_sha256": M1_SHA256,
        "m2_script_sha256": M2_SHA256,
        "replay_module_sha256": sha256(Path(replay.__file__)),
        "references": references,
        "device": str(device),
        "torch": torch.__version__,
        "bucket_solves": observer.bucket_solves,
        "failed_padded_solves": observer.failed_chunks,
        "unresolved_searches": observer.unresolved_searches,
        "wall_seconds": round(time.perf_counter() - started, 1),
        "replay_cases": replay_cases,
        "oracle_cases": oracle_cases,
        "records": records,
    }
    try:
        result["checks"] = checks(result, starts, outcomes)
    except Exception:
        # Keep the finished run for diagnosis, then stop.
        runner._write_new(Path(f"{args.output.absolute()}.unchecked.json"), result)
        raise
    runner._write_new(args.output.absolute(), result)
    print(json.dumps({"cases": len(records), "checks": result["checks"],
                      "bucket_solves": len(observer.bucket_solves),
                      "failed_padded_solves": len(observer.failed_chunks)}), flush=True)


if __name__ == "__main__":
    main()
