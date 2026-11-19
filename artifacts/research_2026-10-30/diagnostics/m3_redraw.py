"""Measurement M3: the paired redraw of near-coincident starts.

Registered in R1_LOG.md, "Measurement M3" (2026-11-15).  Runs ``m1_contacts.main``
unchanged, with the arguments of M2, after wrappers that change one thing.  Inside every
padded solve, a start whose closest two distinct atoms are at most ``THRESHOLD`` angstrom
apart is drawn again.  That distance is d0 of M2, read from ``m2_starts.start_summary``
of the state the integrator starts from.  The new draw comes from the same attempt's
geometry generator, which continues its own stream, and the draw is repeated until d0
exceeds the threshold or ``MAX_DRAWS`` draws are used.  Every other start is the first
draw, as in M2.  The cover, the field, the members and order of the padded solve, and the
integrator are unchanged.

Recorded for every attempt solved in a padded solve: ``start``, the start that was
integrated (the M2 block); ``redraw``, the number of draws and whether the first and the
used start are near; for a redrawn attempt, ``first_start``, the M2 block of its first
draw.  Recorded for every case: this run's full replay case and oracle case, and a check
of every attempt that was not redrawn against the stored replay.  Research diagnostic: it
reads private attributes of the generator.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import m1_contacts as m1
import m2_starts as m2
import torch

import crystalatlas.learned as learned
from crystalatlas.errors import CrystalAtlasError, InvariantViolation
from crystalatlas.flow import FlowState, canonicalize_torus_validated

SCHEMA = "crystalatlas.research-v7-m3-redraw.v1"
THRESHOLD = 0.1
"""Angstrom; a start with d0 <= THRESHOLD is near-coincident (M2) and is drawn again."""
MAX_DRAWS = 1000
"""Draws of one attempt at most; an attempt still near after them is integrated as is."""
RADIUS = 4.0
MAX_REACH = 8
"""The image search of M2's start block; the run must use these values."""
M2_SHA256 = "5adfad80bc2d570fa08b53a5434549e8d5fe4c87218e7082aed48c4120220bd9"
M2_RESULT_SCHEMA = "crystalatlas.research-v7-m2-starts.v1"
M2_RESULT_SHA256 = "60ccf1fefafa78929b64c9b7a18cfb0bc40ef6aeb405d0430668d07d6614f844"
INVALID_DISTANCE = 0.5
"""An attempt is invalid when it has no structure or its shortest distance is at most this."""


class State:
    """Inputs of the padded solve in progress and counters of the whole run."""

    def __init__(self) -> None:
        self.covers: tuple[object, ...] = ()
        self.generators: tuple[torch.Generator, ...] = ()
        self.calls: list[tuple[tuple[object, ...], dict[str, object], FlowState]] = []
        self.sample = None
        self.bucket_solves = 0
        self.failed_solves: list[dict[str, object]] = []
        self.current: list[object] = []
        self.block_changes = 0
        self.cases: list[dict[str, object]] = []
        self.oracles: list[dict[str, object]] = []


def is_near(summary: dict[str, object], plan, z, eta, row: int) -> tuple[bool, float | None]:  # type: ignore[no-untyped-def]
    """d0 <= THRESHOLD from M2's start block, and the exact search's d0 if the block is unresolved.

    For an unresolved block the second value is the smallest distinct-atom distance found by
    an exact image search at radius THRESHOLD (None when no pair is that close).
    """

    if summary["resolved"]:
        closest = summary["closest"]
        return closest is not None and float(closest["r"]) <= THRESHOLD, None  # type: ignore[index]
    count = plan.atom_counts[row]
    expanded = plan.expand(z, eta)
    lattice = expanded.lattice[row].to(dtype=torch.float64)
    fractional = expanded.fractional_coordinates[row, :count].to(dtype=torch.float64)
    geometry = m1.pair_geometry(lattice, fractional, THRESHOLD, MAX_REACH)
    if geometry is None:
        raise InvariantViolation("the image search at the threshold is unresolved")
    distance = geometry[0]
    values = [float(distance[i, j]) for i in range(count) for j in range(i + 1, count)
              if distance[i, j] <= THRESHOLD]
    return bool(values), m1.significant(min(values)) if values else None


def start_blocks(plan, covers, model, z, eta, period_basis, rows):  # type: ignore[no-untyped-def]
    """M2's start block of ``rows`` of a padded initial state, whether each is near, and the
    exact threshold search's d0 for an unresolved block.

    The state is canonicalized on the torus as ``flow.integrate_rk4`` does before its first
    step, so the block equals the one M2 read from step 0 of the trajectory.
    """

    canonical = canonicalize_torus_validated(eta, period_basis)
    pseudo = {"plan": plan, "covers": covers, "times": (0.0,), "z": (z,), "eta": (canonical,)}
    result = {}
    for row in rows:
        summary = m2.start_summary(pseudo, row, model, radius=RADIUS, max_reach=MAX_REACH)
        result[row] = (summary, *is_near(summary, plan, z, canonical, row))
    return result


def redraw(state: State, plan, model, initial_state, period_basis):  # type: ignore[no-untyped-def]
    """The initial state with every near row drawn again, and the record of each row."""

    calls, covers, generators = state.calls, state.covers, state.generators
    if not (len(calls) == len(covers) == len(generators) == plan.batch):
        raise InvariantViolation("the base draws are not one per member of the padded solve")
    z = initial_state.z.clone()
    eta = initial_state.eta.clone()
    for row, (args, kwargs, base) in enumerate(calls):
        if kwargs.get("generator") is not generators[row]:
            raise InvariantViolation("a base draw used another member's generator")
        dimension = int(args[0])  # type: ignore[call-overload]
        width = base.eta.shape[-1]
        if not (torch.equal(z[row, :dimension], base.z) and torch.equal(eta[row, :width], base.eta)):
            raise InvariantViolation("row of the initial state is not its member's base draw")
    with torch.no_grad():
        blocks = start_blocks(plan, covers, model, z, eta, period_basis, range(plan.batch))
    first = dict(blocks)
    draws = [1] * plan.batch
    pending = [row for row in range(plan.batch) if blocks[row][1]]
    while pending:
        for row in pending:
            args, kwargs, _ = calls[row]
            base = state.sample(*args, **kwargs)  # type: ignore[misc]
            z[row, : int(args[0])] = base.z  # type: ignore[call-overload]
            eta[row, : base.eta.shape[-1]] = base.eta
            draws[row] += 1
        with torch.no_grad():
            blocks.update(start_blocks(plan, covers, model, z, eta, period_basis, pending))
        pending = [row for row in pending if blocks[row][1] and draws[row] < MAX_DRAWS]
    # The blocks of the exact state that is integrated.  They equal the per-iteration
    # blocks unless batched kernels couple rows, which is counted, not assumed.
    if any(count > 1 for count in draws):
        with torch.no_grad():
            final = start_blocks(plan, covers, model, z, eta, period_basis, range(plan.batch))
        state.block_changes += sum(final[row] != blocks[row] for row in range(plan.batch))
    else:
        final = blocks
    rows = []
    for row in range(plan.batch):
        item: dict[str, object] = {
            "draws": draws[row],
            "near_first": first[row][1],
            "near_used": final[row][1],
            "used_start": final[row][0],
        }
        if draws[row] > 1:
            item["first_start"] = first[row][0]
        if not first[row][0]["resolved"] or not final[row][0]["resolved"]:
            # [first draw, used start]; None: resolved block, or no pair within THRESHOLD.
            item["threshold_search_d0"] = [first[row][2], final[row][2]]
        rows.append(item)
    return FlowState(z, eta), rows


def install(state: State) -> None:
    install_m1 = m1.install

    def install_with_redraw(generator, recorder) -> None:  # type: ignore[no-untyped-def]
        install_m1(generator, recorder)
        model = generator.model
        inner = learned.integrate_with_endpoint_estimate  # the M1 recording integrator

        def redraw_integrator(  # type: ignore[no-untyped-def]
            field, initial_state, period_basis, *, num_steps, terminal_time=1.0
        ):
            plan = recorder.pending_plan
            if plan is None:
                # A single-member fallback solve: not recorded and not redrawn; counted.
                state.calls = []
                state.bucket_solves += 1
                return inner(
                    field, initial_state, period_basis,
                    num_steps=num_steps, terminal_time=terminal_time,
                )
            try:
                start, rows = redraw(state, plan, model, initial_state, period_basis)
            except InvariantViolation:
                raise
            except (CrystalAtlasError, ValueError, RuntimeError) as exc:
                # A defect here must stop the run, not pass for a member's ODE failure.
                raise InvariantViolation(f"the redraw failed: {type(exc).__name__}: {exc}") from exc
            state.calls = []
            before = len(recorder.records)
            try:
                final = inner(
                    field, start, period_basis, num_steps=num_steps, terminal_time=terminal_time
                )
            except Exception as exc:
                # A member error halves the solve (learned._integrate_mixed_members); a member
                # that fails alone ends in the untreated single-member fallback.  Recorded.
                state.failed_solves.append({
                    "case": state.current,
                    "error": f"{type(exc).__name__}: {exc}",
                    "batch": plan.batch,
                    "space_groups": [cover.space_group for cover in state.covers],
                    "draws": [item["draws"] for item in rows],
                })
                raise
            if len(recorder.records) != before + 1 or recorder.records[-1]["plan"] is not plan:
                raise InvariantViolation("the padded solve was not recorded once")
            recorder.records[-1]["redraws"] = rows
            return final

        learned.integrate_with_endpoint_estimate = redraw_integrator  # type: ignore[assignment]

        chunk = generator._integrate_mixed_chunk  # the M1 recording chunk

        def chunk_with_inputs(covers, generators):  # type: ignore[no-untyped-def]
            state.covers, state.generators, state.calls = tuple(covers), tuple(generators), []
            return chunk(covers, generators)

        generator._integrate_mixed_chunk = chunk_with_inputs

    m1.install = install_with_redraw

    sample = learned.sample_product_base
    state.sample = sample

    def capturing_sample(*args, **kwargs):  # type: ignore[no-untyped-def]
        base = sample(*args, **kwargs)
        state.calls.append((args, kwargs, base))
        return base

    learned.sample_product_base = capturing_sample  # type: ignore[assignment]

    describe = m1.describe_row

    def describe_with_redraw(record, row, model, *, steps, radius, max_reach, **rest):  # type: ignore[no-untyped-def]
        if radius != RADIUS or max_reach != MAX_REACH:
            raise InvariantViolation("M3 runs with the image search of M2")
        entry = describe(record, row, model, steps=steps, radius=radius, max_reach=max_reach, **rest)
        entry["start"] = m2.start_summary(record, row, model, radius=radius, max_reach=max_reach)
        item = record["redraws"][row]
        entry["redraw"] = {
            "draws": item["draws"],
            "near_first": item["near_first"],
            "near_used": item["near_used"],
            # Step 0 of the recorded trajectory against the block the redraw accepted; the
            # analysis stops on False.  Not raised here, so the run's output is kept.
            "start_equals_accepted": entry["start"] == item["used_start"],
        }
        if "first_start" in item:
            entry["first_start"] = item["first_start"]
        if "threshold_search_d0" in item:
            entry["redraw"]["threshold_search_d0"] = item["threshold_search_d0"]
        return entry

    m1.describe_row = describe_with_redraw

    evaluate = m1.evaluate_case

    def keeping_case(generator, target, **kwargs):  # type: ignore[no-untyped-def]
        state.current = [target.record_id, "replay"]
        case = evaluate(generator, target, **kwargs)
        # evaluate_case turns an InvariantViolation raised inside sampling (for example by
        # the redraw) into a request error; the stored replay has none, so stop with it.
        if case.get("request_error") is not None:
            raise InvariantViolation(f"request error in {case['record_id']}: {case['request_error']}")
        state.cases.append(case)
        return case

    m1.evaluate_case = keeping_case

    oracle = m1.replay.oracle_case

    def keeping_oracle(generator, target, **kwargs):  # type: ignore[no-untyped-def]
        state.current = [target.record_id, "oracle"]
        own = oracle(generator, target, **kwargs)
        state.oracles.append(own)
        return own

    m1.replay.oracle_case = keeping_oracle


def invalid(value: float | None) -> bool:
    return value is None or value <= INVALID_DISTANCE


def checked_result(state: State, result: dict[str, object], m2_path: Path) -> dict[str, object]:
    """The M1 result with this run's cases and the checks against the stored replay and M2."""

    if any(attempt["traced"] for case in result["records"] for kind in ("replay", "oracle")
           for attempt in case[kind]):
        raise InvariantViolation("M3 runs with no traced attempt")
    records = result["records"]
    if not (len(state.cases) == len(state.oracles) == len(records)):
        raise InvariantViolation("one replay case and one oracle case per record are required")
    stored = json.loads(Path(str(result["stored"])).read_text())
    stored_replay = {case["record_id"]: case for case in stored["replay_cases"]}
    stored_oracle = {case["record_id"]: case for case in stored["oracle_cases"]}
    m2_text = m2_path.read_bytes()
    reference = json.loads(m2_text)
    m2_cases = {case["record_id"]: case for case in reference["records"]}
    counts = {
        "redrawn_replay": 0, "redrawn_oracle": 0, "capped": 0,
        "unsolved_m2_near": 0, "unsolved_other": 0,
        "redrawn_not_m2_near": 0, "m2_near_not_redrawn": 0,
        "first_start_differs_from_m2": 0, "start_differs_from_m2": 0,
        "start_not_equal_accepted": 0, "cover_differs_from_m2": 0,
        "not_redrawn": 0, "not_redrawn_outcome_differs": 0,
        "not_redrawn_validity_or_match_differs": 0,
    }
    for entry, case, own in zip(records, state.cases, state.oracles, strict=True):
        record_id = entry["record_id"]
        if case["record_id"] != record_id or own["record_id"] != record_id:
            raise InvariantViolation("the kept cases are not in record order")
        entry["replay_case"] = case
        entry["oracle_case"] = own
        old_case, old_oracle, old_starts = (
            stored_replay[record_id], stored_oracle[record_id], m2_cases[record_id]
        )
        for kind in ("replay", "oracle"):
            for attempt in entry[kind]:
                index = attempt["index"]
                previous = old_starts[kind][index]
                closest = previous.get("start", {}).get("closest")
                m2_near = closest is not None and float(closest["r"]) <= THRESHOLD
                if not attempt["solved"]:
                    counts["unsolved_m2_near" if m2_near else "unsolved_other"] += 1
                    continue
                redrawn = attempt["redraw"]["draws"] > 1
                counts[f"redrawn_{kind}"] += redrawn
                counts["capped"] += bool(attempt["redraw"]["near_used"])
                counts["start_not_equal_accepted"] += not attempt["redraw"]["start_equals_accepted"]
                counts["cover_differs_from_m2"] += any(
                    attempt[name] != previous.get(name) for name in ("space_group", "nu", "orbits")
                )
                counts["redrawn_not_m2_near"] += redrawn and not m2_near
                counts["m2_near_not_redrawn"] += m2_near and not redrawn
                if redrawn:
                    counts["first_start_differs_from_m2"] += attempt["first_start"] != previous["start"]
                    continue
                counts["start_differs_from_m2"] += attempt["start"] != previous["start"]
                if kind == "replay":
                    new = (case["shortest_distances"][index], index in case["matches"])
                    old = (old_case["shortest_distances"][index], index in old_case["matches"])
                else:
                    new = (own["shortest_distances"][index], own["matches"][index])
                    old = (old_oracle["shortest_distances"][index], old_oracle["matches"][index])
                counts["not_redrawn"] += 1
                counts["not_redrawn_outcome_differs"] += new != old
                counts["not_redrawn_validity_or_match_differs"] += (
                    (invalid(new[0]), new[1]) != (invalid(old[0]), old[1])
                )
    return {
        **result,
        "schema": SCHEMA,
        "m1_schema": m1.SCHEMA,
        "m1_script_sha256": result["script_sha256"],
        "m2_script_sha256": M2_SHA256,
        "m3_script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "m2_result": str(m2_path.absolute()),
        "m2_result_sha256": hashlib.sha256(m2_text).hexdigest(),
        "threshold": THRESHOLD,
        "max_draws": MAX_DRAWS,
        "bucket_solves": state.bucket_solves,
        "failed_padded_solves": state.failed_solves,
        "block_changes": state.block_changes,
        "checks": counts,
    }


def take_m2_result(argv: list[str]) -> Path:
    """Remove ``--m2-result PATH`` from ``argv`` (M1's parser does not know it) and check it."""

    if argv.count("--m2-result") != 1 or argv.index("--m2-result") + 1 >= len(argv):
        raise InvariantViolation("--m2-result PATH is required once")
    position = argv.index("--m2-result")
    path = Path(argv[position + 1])
    del argv[position : position + 2]
    found = hashlib.sha256(path.read_bytes()).hexdigest()
    if found != M2_RESULT_SHA256:
        raise InvariantViolation(f"the M2 result has sha256 {found}, not {M2_RESULT_SHA256}")
    if json.loads(path.read_text())["schema"] != M2_RESULT_SCHEMA:
        raise InvariantViolation("--m2-result is not an M2 result")
    for token in argv[1:]:
        name = token.split("=", 1)[0]
        # argparse accepts unambiguous abbreviations, so refuse those too.
        if (len(name) >= 4 and "--radius".startswith(name)) or (
            len(name) >= 7 and "--max-reach".startswith(name)
        ):
            raise InvariantViolation(f"M3 runs with the M2 defaults of --radius and --max-reach: {token}")
    return path


def check_scripts() -> None:
    found = hashlib.sha256(Path(m2.__file__).read_bytes()).hexdigest()
    if found != M2_SHA256:
        raise InvariantViolation(f"m2_starts.py has sha256 {found}, not the registered {M2_SHA256}")


if __name__ == "__main__":
    m2_result = take_m2_result(sys.argv)
    check_scripts()
    m2.check_inputs(sys.argv)  # the M1 script's sha256 and an empty targets file
    run = State()
    install(run)
    write = m1.runner._write_new

    def write_checked(path, result):  # type: ignore[no-untyped-def]
        try:
            checked = checked_result(run, result, m2_result)
        except Exception:
            # Keep the finished run for diagnosis, then stop.
            write(Path(f"{path}.unchecked.json"), {
                **result, "schema": f"{SCHEMA}-unchecked", "replay_cases": run.cases,
                "oracle_cases": run.oracles, "bucket_solves": run.bucket_solves,
                "failed_padded_solves": run.failed_solves, "block_changes": run.block_changes,
            })
            raise
        write(path, checked)

    m1.runner._write_new = write_checked
    m1.main()
