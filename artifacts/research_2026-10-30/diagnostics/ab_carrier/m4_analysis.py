"""Registered analysis of measurement M4 (R1_LOG.md, "Measurement M4", 2026-11-16).

The opt-in start-rejection sampler.  The threshold is chosen on the test records (the
records of M1 to M3) in field control_s20261030 by the registered sequential rule, then
confirmed once on the validation records, which no geometry replay had used, in the
three control fields against new untreated runs.  Pure Python; no torch.

Modes (exit 0 on success; a failed check prints "STOP: ..." and exits 2):
  smoke-off M4 STORED M2       untreated, 3 test records: reproduces the stored replay
  smoke-m3 M4 M3               0.1 angstrom, 3 test records: equals M3 on its records
  smoke-val M4                 untreated, 3 validation records: the validation path runs
  equals-m3 M4 M3              the selection run at 0.1 angstrom equals M3 on 50 records
  check M4 STARTS OUTCOMES     every stop of one treated run against its references
  check-untreated M4           every stop of one untreated validation run
  same-starts M4 M4            two validation runs of one threshold in two fields share
                               covers, draws and starts
  step PREVIOUS NEXT           one selection step: prints "ACCEPT <next>" or "REJECT";
                               PREVIOUS is the stored replay for the step to 0.1
  verdict OFF M3SMOKE VAL -- S01 [S02 ...] -- U30 U31 U32 -- T30 T31 T32 -- STORED M2 M3
                               the printed selection, decision and reported numbers
"""

from __future__ import annotations

import hashlib
import json
import math
import random
import statistics
import sys
from pathlib import Path

SCHEMA = "crystalatlas.research-v7-m4-rejection.v1"
UNPINNED = "PINNED_AFTER_REVIEW"
M4_SHA256 = "6a7984de418727d7be6dce0b2b01019444f1adbac3b6bfaa4ce46737c491cc27"
"""sha256 of m4_rejection.py."""
CODE_SHA256 = "941fbdd1d5105dec54c56ee01d96a0648b3de04bcdb5fc4df67bd352c4db74b1"
"""The harness code fingerprint of the M1 to M3 snapshot with the registered learned.py."""
M2_RESULT_SHA256 = "60ccf1fefafa78929b64c9b7a18cfb0bc40ef6aeb405d0430668d07d6614f844"
M3_RESULT_SHA256 = "b5fd5622fd7c1b55247d77bd402d499bef383a42a2f20bb49111c08b0e4a9318"
REPLAY_MODULE_SHA256 = "a5c825bdfeb2192600c4185146a606ad13a15399363a4bdec44e94447d673436"
STORED_SHA256 = "d2d6c00cba6d07bc71b483ae0c2ad48932726c01d5ec5f775a5a1fb021719075"
"""The stored untreated replay of the selection field on the test records."""
FIELD_CHECKPOINTS = {
    "control_s20261030": "c2a99ffb18e88e4cf5909ee1404503f1ca383192da80aed49303c97d74a52d48",
    "control_s20261031": "25c3e7037ed3c0b39e74a95cce5d69ede24a382070ded1a7824a39ac34b8299b",
    "control_s20261032": "368afe9c0ae516dea2c3c029afee0165d6358ffa703c68c8b6e75add707f47ca",
}
FIELDS = tuple(sorted(FIELD_CHECKPOINTS))
COVER_CHECKPOINT = "119e249ac6cb54d7cd20ae861c364088c99980e97298765fee4fc8dbd84ba756"
PROTOCOL_SEED = 20261530
SELECTION_FIELD = "control_s20261030"
SEQUENCE = (None, 0.1, 0.2, 0.3, 0.5)
"""Thresholds in angstrom, tried in this order on the test records."""
GAIN_ALPHA = 0.05
HARM_ALPHA = 0.10
CONFIRM_GAIN_ALPHA = 0.01
CONFIRM_HARM_ALPHA = 0.05
INVALID_DISTANCE = 0.5
SENSITIVITY_DISTANCES = (0.6, 0.7, 0.8)
MAX_DRAWS = 1000
UNTREATED_SEARCH = 0.5
ROUNDING = 1.0e-6
"""Relative agreement required between a block's 7-digit r and the unrounded d0."""
REBATCHED = 1.0e-5
"""Relative agreement required of a start rebuilt in a smaller solve after a failed padded
solve (batched kernels need not be bitwise equal across batch sizes)."""
REPLAY_FIELDS = (
    "seed", "attempts", "valid", "failed_stages", "matches", "match_count", "first",
    "consensus", "best_rms", "distinct_tables", "target_table_valid",
    "target_table_matches", "shortest_distances", "structurally_valid",
)
ORACLE_FIELDS = ("seed", "draws", "failed_stages", "shortest_distances", "matches", "rms")
STOP_CHECKS = (
    "cover_differs", "redrawn_reference_not_near", "reference_near_not_redrawn",
    "reference_undecided", "used_near", "used_undecided", "exact_inconsistent",
    "kept_start_differs", "kept_outcome_differs", "draws_zero", "redrawn_untreated",
    "unsolved_unexplained", "bucket_unexplained", "request_errors_differ",
)


class Stop(Exception):  # noqa: N818 - a registered stop, not an error
    pass


def stop(condition: bool, message: str) -> None:
    if not condition:
        raise Stop(message)


def load(path: str) -> tuple[dict, str]:
    data = Path(path).read_bytes()
    return json.loads(data), hashlib.sha256(data).hexdigest()


def binomial_upper(k: int, n: int) -> float:
    """P(X >= k) for X ~ Binomial(n, 1/2); 1 for k <= 0 or n = 0."""

    if n == 0 or k <= 0:
        return 1.0
    return sum(math.comb(n, i) for i in range(k, n + 1)) / 2.0**n


def two_sided(a: int, b: int) -> float:
    return min(1.0, 2.0 * binomial_upper(max(a, b), a + b)) if a + b else 1.0


def invalid(value: float | None, cutoff: float = INVALID_DISTANCE) -> bool:
    return value is None or value <= cutoff


def nearness(attempt: dict, threshold: float) -> bool | None:
    """d0 <= threshold of an attempt's start; None when the record cannot decide it.

    The unrounded d0 of the option's own search decides when the run recorded it (an M4
    file); otherwise (M2's records) a resolved block decides on its 7-digit r, and an
    unresolved one its stored exact search, when that reached the threshold.
    """

    exact = attempt.get("d0_exact", "absent")
    if exact != "absent" and exact != "unresolved" and threshold <= UNTREATED_SEARCH:
        return exact is not None and float(exact) <= threshold
    block = attempt["start"]
    if block["resolved"]:
        closest = block["closest"]
        return closest is not None and float(closest["r"]) <= threshold
    search = attempt.get("threshold_search")
    if search is None or float(search["radius"]) < threshold:
        return None
    return search["d0"] is not None and float(search["d0"]) <= threshold


def exact_consistent(attempt: dict) -> bool:
    """The unrounded d0 agrees with the independent radius-4 block (7-digit r)."""

    exact = attempt.get("d0_exact", "absent")
    block = attempt["start"]
    if exact in ("absent", "unresolved") or not block["resolved"]:
        return True
    closest = block["closest"]
    if exact is None:
        return closest is None or float(closest["r"]) > UNTREATED_SEARCH * (1.0 - ROUNDING)
    return closest is not None and abs(float(closest["r"]) - float(exact)) <= ROUNDING * float(exact)


def close(a: float | None, b: float | None, tolerance: float = REBATCHED) -> bool:
    if a is None or b is None or isinstance(a, str) or isinstance(b, str):
        return a == b
    return abs(float(a) - float(b)) <= tolerance * max(abs(float(a)), abs(float(b)), 1e-12)


def starts_close(one: dict, two: dict) -> bool:
    """Two start blocks of one draw agree to REBATCHED in d0 and volume per atom."""

    if one["resolved"] != two["resolved"] or one["atoms"] != two["atoms"]:
        return False
    if not close(one["volume_per_atom"], two["volume_per_atom"]):
        return False
    if not one["resolved"]:
        return True
    a, b = one["closest"], two["closest"]
    if (a is None) != (b is None):
        return False
    return a is None or (a["class"] == b["class"] and close(a["r"], b["r"]))


def d0(attempt: dict) -> float | None:
    """d0 of an attempt's start; infinity beyond 4 angstrom; None when unresolved."""

    block = attempt["start"]
    if not block["resolved"]:
        exact = attempt.get("d0_exact", "absent")
        if exact not in ("absent", "unresolved"):
            # The option's own search reaches 0.5 angstrom; None means farther.
            return float(exact) if exact is not None else None
        return None
    closest = block["closest"]
    return math.inf if closest is None else float(closest["r"])


def field_name(m4: dict) -> str:
    return Path(m4["field_run"]).name


def raw_outcomes(cases: dict) -> dict[tuple[str, str, int], tuple[float | None, bool]]:
    """(decoded shortest distance, matched) of every replay attempt and oracle draw."""

    found = {}
    for case in cases["replay_cases"]:
        for index, value in enumerate(case["shortest_distances"]):
            found[(case["record_id"], "replay", index)] = (value, index in case["matches"])
    for case in cases["oracle_cases"]:
        for index, value in enumerate(case["shortest_distances"]):
            found[(case["record_id"], "oracle", index)] = (value, bool(case["matches"][index]))
    return found


def outcomes(cases: dict, kinds=("replay", "oracle"), cutoff: float = INVALID_DISTANCE):  # type: ignore[no-untyped-def]
    """(invalid, matched) of every attempt of the given kinds."""

    return {
        key: (invalid(value, cutoff), matched)
        for key, (value, matched) in raw_outcomes(cases).items()
        if key[1] in kinds
    }


def failed_solve_cases(*files: dict | None) -> set[tuple[str, str]]:
    """(record, kind) of every case with a failed padded solve in any of the files."""

    found: set[tuple[str, str]] = set()
    for item in files:
        if item is not None:
            found |= {(f["case"][0], f["case"][1]) for f in item.get("failed_padded_solves", [])}
    return found


def verify_file(m4: dict, *, partition: str, threshold, field: str, cases: int) -> None:  # type: ignore[no-untyped-def]
    """Provenance of one M4 file, including the pinned code fingerprint."""

    stop(M4_SHA256 != UNPINNED and CODE_SHA256 != UNPINNED, "the analysis hash pins are not set")
    stop(m4["schema"] == SCHEMA, "not an M4 file")
    stop(m4["m4_script_sha256"] == M4_SHA256,
         f"m4_rejection.py sha256 {m4['m4_script_sha256']} is not the registered one")
    stop(m4["code"]["sha256"] == CODE_SHA256,
         f"code fingerprint {m4['code']['sha256']} is not the registered snapshot's")
    stop(m4["replay_module_sha256"] == REPLAY_MODULE_SHA256, "the replay module differs from M2's")
    stop(m4["partition"] == partition, f"partition {m4['partition']}, expected {partition}")
    stop(m4["threshold"] == threshold, f"threshold {m4['threshold']}, expected {threshold}")
    stop(m4["generator_start_min_separation"] == threshold, "the generator ran another threshold")
    stop(m4["max_draws"] == MAX_DRAWS and m4["radius"] == 4.0 and m4["max_reach"] == 8
         and m4["untreated_search"] == UNTREATED_SEARCH, "draw cap or start-block search differs")
    stop(field_name(m4) == field, f"field {field_name(m4)}, expected {field}")
    stop(m4["field_checkpoint_sha256"] == FIELD_CHECKPOINTS[field], "field checkpoint differs")
    stop(m4["cover_checkpoint_sha256"] == COVER_CHECKPOINT, "cover checkpoint differs")
    stop(m4["protocol_seed"] == PROTOCOL_SEED, "protocol seed differs")
    stop(m4["candidates"] == 20 and m4["oracle_draws"] == 20, "candidate counts differ")
    stop(m4["cases"] == cases == len(m4["records"]) == len(m4["replay_cases"])
         == len(m4["oracle_cases"]), f"expected {cases} cases")
    ids = [record["record_id"] for record in m4["records"]]
    stop(ids == m4["record_ids"] == [case["record_id"] for case in m4["replay_cases"]]
         == [case["record_id"] for case in m4["oracle_cases"]], "records are out of order")


def recheck(m4: dict, starts: dict | None, reference: dict | None) -> dict[str, int]:
    """The per-attempt checks, recomputed from the raw files (not the run's own counts).

    ``starts`` holds the reference first draws (M2's records or an untreated M4 run),
    ``reference`` the reference outcomes (the stored replay or an untreated M4 run).
    """

    threshold = m4["threshold"]
    counts = {name: 0 for name in STOP_CHECKS}
    counts.update(redrawn=0, kept=0, kept_outcome_differs_in_failed_solve_cases=0,
                  kept_start_rebatched_in_failed_solve_cases=0,
                  unsolved=0, solved=0, reference_unsolved=0, unsolved_without_structure=0)
    first = {case["record_id"]: case for case in starts["records"]} if starts else None
    failed = failed_solve_cases(m4, starts, reference)
    own_failed = failed_solve_cases(m4)
    request_errors = {c["record_id"] for c in m4["replay_cases"] if c.get("request_error")}
    if reference is not None:
        old_errors = {c["record_id"] for c in reference["replay_cases"] if c.get("request_error")}
        counts["request_errors_differ"] = len(request_errors ^ old_errors)
    reference_buckets = (
        {(item[0], item[1]) for item in starts.get("bucket_solves", [])} if starts else set()
    )
    for item in m4["bucket_solves"]:
        counts["bucket_unexplained"] += (
            (item[0], item[1]) not in own_failed and (item[0], item[1]) not in reference_buckets
        )
    old = raw_outcomes(reference) if reference is not None else None
    new = raw_outcomes(m4)
    for record in m4["records"]:
        rid = record["record_id"]
        for kind in ("replay", "oracle"):
            for attempt in record[kind]:
                index = attempt["index"]
                previous = first[rid][kind][index] if first else None
                if previous is not None and not previous.get("solved"):
                    counts["reference_unsolved"] += 1
                if not attempt["solved"]:
                    counts["unsolved"] += 1
                    without_structure = new[(rid, kind, index)][0] is None
                    counts["unsolved_without_structure"] += without_structure
                    explained = (
                        (rid, kind) in own_failed
                        or (kind == "replay" and rid in request_errors)
                        or (previous is not None and not previous.get("solved")
                            and previous.get("failed_stage") == attempt.get("failed_stage"))
                        # An untreated run without a reference: a failure before any
                        # padded solve (a cover it cannot expand) leaves no structure.
                        or (threshold is None and first is None and without_structure)
                    )
                    counts["unsolved_unexplained"] += not explained
                    continue
                counts["solved"] += 1
                counts["draws_zero"] += attempt["draws"] < 1
                counts["exact_inconsistent"] += not exact_consistent(attempt)
                redrawn = attempt["draws"] > 1
                counts["redrawn"] += redrawn
                if threshold is None:
                    counts["redrawn_untreated"] += redrawn
                else:
                    used = nearness(attempt, threshold)
                    counts["used_undecided"] += used is None
                    counts["used_near"] += bool(used)
                if previous is None or not previous.get("solved"):
                    continue
                counts["cover_differs"] += any(
                    attempt[name] != previous.get(name) for name in ("space_group", "nu", "orbits")
                )
                if threshold is not None:
                    was_near = nearness(previous, threshold)
                    if was_near is None:
                        counts["reference_undecided"] += 1
                    else:
                        counts["redrawn_reference_not_near"] += redrawn and not was_near
                        counts["reference_near_not_redrawn"] += was_near and not redrawn
                if redrawn:
                    continue
                counts["kept"] += 1
                if attempt["start"] != previous["start"]:
                    if (rid, kind) in failed and starts_close(attempt["start"], previous["start"]):
                        counts["kept_start_rebatched_in_failed_solve_cases"] += 1
                    else:
                        counts["kept_start_differs"] += 1
                if old is not None and new[(rid, kind, index)] != old[(rid, kind, index)]:
                    if (rid, kind) in failed:
                        counts["kept_outcome_differs_in_failed_solve_cases"] += 1
                    else:
                        counts["kept_outcome_differs"] += 1
    return counts


def require_clean(counts: dict[str, int], label: str) -> None:
    bad = {name: counts[name] for name in STOP_CHECKS if counts[name]}
    stop(not bad, f"{label}: failed checks {bad}")


def equals_m3(m4: dict, m3: dict) -> int:
    """Stops unless every case, draw count and integrated start equals M3's; redrawn count."""

    stop(m4["bucket_solves"] == [] and m4["failed_padded_solves"] == [], "a fallback solve ran")
    m3_records = {record["record_id"]: record for record in m3["records"]}
    redrawn = 0
    for record, case, own in zip(m4["records"], m4["replay_cases"], m4["oracle_cases"], strict=True):
        reference = m3_records[record["record_id"]]
        differing = [f for f in REPLAY_FIELDS if case.get(f) != reference["replay_case"].get(f)]
        differing += [f"oracle/{f}" for f in ORACLE_FIELDS
                      if own.get(f) != reference["oracle_case"].get(f)]
        stop(not differing, f"{record['record_id']} differs from M3 in {differing}")
        for kind in ("replay", "oracle"):
            for attempt, old in zip(record[kind], reference[kind], strict=True):
                stop(attempt["solved"] and old["solved"], "an attempt was not solved")
                stop(attempt["draws"] == old["redraw"]["draws"], "draw counts differ from M3")
                stop(attempt["start"] == old["start"], "integrated starts differ from M3")
                redrawn += attempt["draws"] > 1
    return redrawn


def smoke_off_data(m4: dict, stored: dict, m2: dict) -> int:
    verify_file(m4, partition="test", threshold=None, field=SELECTION_FIELD, cases=3)
    stop(m4["references"]["starts"]["sha256"] == M2_RESULT_SHA256
         and m4["references"]["outcomes"]["sha256"] == STORED_SHA256,
         "smoke-off used other references")
    counts = recheck(m4, m2, stored)
    require_clean(counts, "smoke-off")
    stop(counts["redrawn"] == 0 and counts["unsolved"] == 0, "the untreated smoke redrew or lost")
    stop(m4["bucket_solves"] == [] and m4["failed_padded_solves"] == [], "a fallback solve ran")
    old_replay = {c["record_id"]: c for c in stored["replay_cases"]}
    old_oracle = {c["record_id"]: c for c in stored["oracle_cases"]}
    for case, own in zip(m4["replay_cases"], m4["oracle_cases"], strict=True):
        differing = [f for f in REPLAY_FIELDS if case.get(f) != old_replay[case["record_id"]].get(f)]
        differing += [f"oracle/{f}" for f in ORACLE_FIELDS
                      if own.get(f) != old_oracle[own["record_id"]].get(f)]
        stop(not differing, f"smoke-off: {case['record_id']} differs in {differing}")
    return counts["solved"]


def smoke_m3_data(m4: dict, m3: dict, cases: int) -> int:
    verify_file(m4, partition="test", threshold=0.1, field=SELECTION_FIELD, cases=cases)
    stop(m4["references"]["starts"]["sha256"] == M2_RESULT_SHA256
         and m4["references"]["outcomes"]["sha256"] == STORED_SHA256,
         "the run used other references")
    redrawn = equals_m3(m4, m3)
    stop(redrawn > 0, "the run redrew nothing")
    return redrawn


def smoke_val_data(m4: dict) -> int:
    verify_file(m4, partition="validation", threshold=None, field=SELECTION_FIELD, cases=3)
    stop(not m4["references"], "the validation smoke has references")
    counts = recheck(m4, None, None)
    require_clean(counts, "smoke-val")
    return counts["solved"]


def smoke_off(m4_path: str, stored_path: str, m2_path: str) -> None:
    m4, _ = load(m4_path)
    stored, stored_sha = load(stored_path)
    m2, m2_sha = load(m2_path)
    stop(stored_sha == STORED_SHA256, "the stored replay is not the registered one")
    stop(m2_sha == M2_RESULT_SHA256, "the M2 result is not the registered one")
    solved = smoke_off_data(m4, stored, m2)
    print(f"smoke-off passed: {solved} attempts equal the stored replay and M2")


def smoke_m3(m4_path: str, m3_path: str, cases: int = 3) -> None:
    m4, _ = load(m4_path)
    m3, m3_sha = load(m3_path)
    stop(m3_sha == M3_RESULT_SHA256, "the M3 result is not the registered one")
    redrawn = smoke_m3_data(m4, m3, cases)
    print(f"equal to M3 in {cases} cases: cases, draws and starts ({redrawn} redrawn)")


def smoke_val(m4_path: str) -> None:
    m4, _ = load(m4_path)
    solved = smoke_val_data(m4)
    print(f"smoke-val passed: {solved} attempts solved on 3 validation records")


def check(m4_path: str, starts_path: str, outcomes_path: str) -> dict[str, int]:
    m4, _ = load(m4_path)
    starts, starts_sha = load(starts_path)
    reference, reference_sha = load(outcomes_path)
    field = field_name(m4)
    stop(m4["threshold"] in SEQUENCE[1:], "a treated run has an unregistered threshold")
    if m4["partition"] == "test":
        stop(field == SELECTION_FIELD, "the selection runs in the selection field only")
        stop(starts_sha == M2_RESULT_SHA256, "the M2 result is not the registered one")
        stop(reference_sha == STORED_SHA256, "the stored replay is not the registered one")
    else:
        stop(starts_sha == reference_sha, "a validation run has one reference")
        verify_file(starts, partition="validation", threshold=None, field=field, cases=50)
    verify_file(m4, partition=m4["partition"], threshold=m4["threshold"], field=field, cases=50)
    stop(m4["references"]["starts"]["sha256"] == starts_sha
         and m4["references"]["outcomes"]["sha256"] == reference_sha,
         "the run used other reference files")
    counts = recheck(m4, starts, reference)
    require_clean(counts, Path(m4_path).name)
    print(f"check passed: {Path(m4_path).name}: {counts['redrawn']} redrawn, {counts['kept']} kept, "
          f"{counts['kept_outcome_differs_in_failed_solve_cases']} kept outcomes and "
          f"{counts['kept_start_rebatched_in_failed_solve_cases']} kept starts differ in cases "
          f"with a failed padded solve, {counts['unsolved']} unsolved")
    return counts


def check_untreated(m4_path: str) -> None:
    m4, _ = load(m4_path)
    stop(m4["threshold"] is None and not m4["references"], "not an untreated run without references")
    verify_file(m4, partition="validation", threshold=None, field=field_name(m4), cases=50)
    counts = recheck(m4, None, None)
    require_clean(counts, Path(m4_path).name)
    errors = {c["record_id"][:8]: c["request_error"] for c in m4["replay_cases"]
              if c.get("request_error")}
    print(f"check passed: {Path(m4_path).name}: {counts['solved']} solved, "
          f"{counts['unsolved']} unsolved ({counts['unsolved_without_structure']} without a "
          f"structure), none redrawn; request errors {errors}")


def shared_starts(one: dict, two: dict) -> int:
    """Stops unless two runs of one threshold in two fields share covers, draws and starts."""

    stop(one["record_ids"] == two["record_ids"], "the runs have other records")
    failed = failed_solve_cases(one, two)
    compared = 0
    for a, b in zip(one["records"], two["records"], strict=True):
        stop(a["record_id"] == b["record_id"] and a["seed"] == b["seed"],
             "records or request seeds differ between fields")
        for kind in ("replay", "oracle"):
            for x, y in zip(a[kind], b[kind], strict=True):
                if not (x["solved"] and y["solved"]):
                    continue
                rebatched = (a["record_id"], kind) in failed
                for name in ("space_group", "nu", "orbits", "draws", "start", "d0_exact"):
                    same = x[name] == y[name]
                    if not same and rebatched and name == "start":
                        same = starts_close(x[name], y[name])
                    if not same and rebatched and name == "d0_exact":
                        same = close(x[name], y[name])
                    stop(same, f"{a['record_id']} {kind} {x['index']}: {name} differs")
                compared += 1
    return compared


def same_starts(first_path: str, second_path: str) -> None:
    one, _ = load(first_path)
    two, _ = load(second_path)
    stop(one["threshold"] == two["threshold"], "same-starts compares runs of one threshold")
    stop(field_name(one) != field_name(two), "same-starts compares two fields")
    for item in (one, two):
        verify_file(item, partition="validation", threshold=one["threshold"],
                    field=field_name(item), cases=50)
    compared = shared_starts(one, two)
    print(f"same-starts passed: {compared} solved attempts share covers, draws and starts")


def compare(previous: dict, nxt: dict, kinds=("replay", "oracle"), cutoff=INVALID_DISTANCE):  # type: ignore[no-untyped-def]
    old, new = outcomes(previous, kinds, cutoff), outcomes(nxt, kinds, cutoff)
    stop(old.keys() == new.keys(), "the two runs have different attempts")
    result = {"fixed": 0, "broken": 0, "gained": 0, "lost": 0, "invalid_old": 0,
              "invalid_new": 0, "matched_old": 0, "matched_new": 0}
    for key in old:
        (was_invalid, was_matched), (is_invalid, is_matched) = old[key], new[key]
        result["fixed"] += was_invalid and not is_invalid
        result["broken"] += is_invalid and not was_invalid
        result["gained"] += is_matched and not was_matched
        result["lost"] += was_matched and not is_matched
        result["invalid_old"] += was_invalid
        result["invalid_new"] += is_invalid
        result["matched_old"] += was_matched
        result["matched_new"] += is_matched
    return result


def step_decision(previous: dict, nxt: dict) -> tuple[bool, dict]:
    """The registered selection step: robust gain, no harm overall and in the replay."""

    pair = compare(previous, nxt)
    replay = compare(previous, nxt, kinds=("replay",))
    pair["p_gain"] = binomial_upper(pair["fixed"], pair["fixed"] + pair["broken"])
    pair["p_harm"] = binomial_upper(pair["lost"], pair["gained"] + pair["lost"])
    pair["replay_gained"], pair["replay_lost"] = replay["gained"], replay["lost"]
    pair["p_harm_replay"] = binomial_upper(replay["lost"], replay["gained"] + replay["lost"])
    old, new = outcomes(previous), outcomes(nxt)
    per_record: dict[str, list[int]] = {}
    for key in old:
        item = per_record.setdefault(key[0], [0, 0, 0, 0])
        (was_invalid, was_matched), (is_invalid, is_matched) = old[key], new[key]
        item[0] += was_invalid and not is_invalid
        item[1] += is_invalid and not was_invalid
        item[2] += is_matched and not was_matched
        item[3] += was_matched and not is_matched
    pair["records_with_validity_change"] = sum(1 for v in per_record.values() if v[0] or v[1])
    pair["records_with_match_change"] = sum(1 for v in per_record.values() if v[2] or v[3])
    # Every record holding a discordant pair: [fixed, broken, gained, lost].
    pair["discordant_records"] = {
        rid[:8]: v for rid, v in sorted(per_record.items()) if any(v)
    }
    # The gain must hold with every single record left out (and with none left out).
    pair["p_gain_robust"], pair["robust_limited_by"] = max(
        [(pair["p_gain"], None)]
        + [(binomial_upper(pair["fixed"] - v[0], pair["fixed"] - v[0] + pair["broken"] - v[1]),
            rid[:8]) for rid, v in sorted(per_record.items())],
        key=lambda item: item[0],
    )
    accept = (pair["p_gain_robust"] <= GAIN_ALPHA and pair["p_harm"] > HARM_ALPHA
              and pair["p_harm_replay"] > HARM_ALPHA)
    return accept, pair


def selection_arm(path: str) -> tuple[dict, str]:
    """A selection file: the stored replay (threshold none) or an M4 test run."""

    data, digest = load(path)
    if data.get("schema") == SCHEMA:
        return data, digest
    stop(digest == STORED_SHA256, "the untreated selection arm is not the registered stored replay")
    return {**data, "threshold": None}, digest


def step(previous_path: str, next_path: str) -> None:
    previous, _ = selection_arm(previous_path)
    nxt, _ = selection_arm(next_path)
    stop(nxt["threshold"] in SEQUENCE[1:], "the next arm has an unregistered threshold")
    position = SEQUENCE.index(nxt["threshold"])
    stop(previous["threshold"] == SEQUENCE[position - 1],
         "a step compares consecutive thresholds of the registered sequence")
    for item in (previous, nxt):
        if item.get("schema") == SCHEMA:
            verify_file(item, partition="test", threshold=item["threshold"],
                        field=SELECTION_FIELD, cases=50)
    accept, pair = step_decision(previous, nxt)
    print(json.dumps({"from": previous["threshold"], "to": nxt["threshold"], **pair}))
    print(f"ACCEPT {nxt['threshold']}" if accept else "REJECT")


def fmt(value: float) -> str:
    return f"{value:.3g}" if value >= 1e-3 else f"{value:.2e}"


def summary(cases: dict) -> dict[str, object]:
    replay, oracle = cases["replay_cases"], cases["oracle_cases"]
    values = [v for c in replay for v in c["shortest_distances"]]
    oracle_values = [v for c in oracle for v in c["shortest_distances"]]
    best = [c["best_rms"] for c in replay if c["best_rms"] is not None]
    return {
        "replay structurally valid": sum(not invalid(v) for v in values),
        "replay without structure": sum(v is None for v in values),
        "replay matched attempts": sum(c["match_count"] for c in replay),
        "replay records matched (match@20)": sum(c["match_count"] > 0 for c in replay),
        "replay first attempt matched (match@1)": sum(bool(c["first"]["match"]) for c in replay),
        "replay consensus matched": sum(bool(c["consensus"]["match"]) for c in replay),
        "replay mean best RMS of matched records": round(statistics.fmean(best), 5) if best else None,
        "oracle structurally valid": sum(not invalid(v) for v in oracle_values),
        "oracle without structure": sum(v is None for v in oracle_values),
        "oracle matched draws": sum(sum(c["matches"]) for c in oracle),
        "oracle records with a match": sum(any(c["matches"]) for c in oracle),
    }


def draw_report(m4: dict) -> str:
    draws = [a["draws"] for r in m4["records"] for k in ("replay", "oracle") for a in r[k]
             if a["solved"]]
    redrawn = [value for value in draws if value > 1]
    # A capped attempt fails once alone in a padded solve before its fallback fails too.
    alone = [item for item in m4["failed_padded_solves"] if item["batch"] == 1]
    capped = sum("base draws" in item["error"] for item in alone)
    images = sum("images per axis" in item["error"] for item in alone)
    replay_records = sum(
        any(a["solved"] and a["draws"] > 1 for a in r["replay"]) for r in m4["records"]
    )
    return (f"solved {len(draws)}, redrawn {len(redrawn)} ({len(redrawn) / max(1, len(draws)):.3f}), "
            f"draws mean {statistics.fmean(draws) if draws else 0:.3f}, median of redrawn "
            f"{statistics.median(redrawn) if redrawn else 0}, max {max(draws, default=0)}; "
            f"records with a redrawn replay attempt {replay_records}; attempts failed at the "
            f"draw cap {capped}, at the image search {images}; failed padded solves "
            f"{len(m4['failed_padded_solves'])}, fallback solves {len(m4['bucket_solves'])}")


BINS = (0.0, 0.01, 0.03, 0.1, 0.15, 0.2, 0.3, 0.5, 1.0, 4.0, math.inf)


def by_bin(pairs: list[tuple[float | None, bool]]) -> str:
    cells = []
    for low, high in zip(BINS[:-1], BINS[1:], strict=True):
        members = [bad for value, bad in pairs
                   if value is not None and (low < value or (low == 0.0 and value == 0.0))
                   and value <= high]
        if members:
            cells.append(f"({low:g}, {high:g}] {sum(members)}/{len(members)}")
    unresolved = [bad for value, bad in pairs if value is None]
    if unresolved:
        cells.append(f"unresolved {sum(unresolved)}/{len(unresolved)}")
    return ", ".join(cells)


def bins_report(m4: dict, first_records: dict, reference: dict) -> tuple[str, str]:
    """Invalidity by the d0 of the reference first draw (every attempt the reference solved)
    and by the d0 of the integrated start (every attempt this run solved)."""

    old, new = outcomes(reference), outcomes(m4)
    before, used = [], []
    for record in m4["records"]:
        rid = record["record_id"]
        for kind in ("replay", "oracle"):
            for attempt in record[kind]:
                key = (rid, kind, attempt["index"])
                previous = first_records[rid][kind][attempt["index"]]
                if previous.get("solved"):
                    before.append((d0(previous), old[key][0]))
                if attempt["solved"]:
                    used.append((d0(attempt), new[key][0]))
    return by_bin(before), by_bin(used)


def shift_report(m4: dict, first_records: dict, cases: dict | None = None) -> str:
    """Over redrawn attempts whose reference is solved: median ratios of the start volume
    per atom (used / first), of the decoded volume per atom (treated / untreated), and the
    median decoded shortest distance of redrawn against kept replay attempts."""

    start_ratio, decoded_ratio, skipped = [], [], 0
    for record in m4["records"]:
        rid = record["record_id"]
        for kind in ("replay", "oracle"):
            for attempt in record[kind]:
                if not attempt["solved"] or attempt["draws"] <= 1:
                    continue
                previous = first_records[rid][kind][attempt["index"]]
                if not previous.get("solved"):
                    skipped += 1
                    continue
                start_ratio.append(attempt["start"]["volume_per_atom"]
                                   / previous["start"]["volume_per_atom"])
                a, b = attempt.get("decoded_volume_per_atom"), previous.get("decoded_volume_per_atom")
                if a is not None and b is not None:
                    decoded_ratio.append(a / b)
    parts = [f"start volume per atom, used / first: median "
             f"{statistics.median(start_ratio):.3f} over {len(start_ratio)}" if start_ratio else
             "no redrawn attempt with a solved reference"]
    if decoded_ratio:
        parts.append(f"decoded volume per atom, treated / untreated: median "
                     f"{statistics.median(decoded_ratio):.3f} over {len(decoded_ratio)}")
    if skipped:
        parts.append(f"{skipped} redrawn attempts with an unsolved reference left out")
    shortest = {(c["record_id"], i): v for c in m4["replay_cases"]
                for i, v in enumerate(c["shortest_distances"])}
    redrawn = [shortest[(r["record_id"], a["index"])] for r in m4["records"] for a in r["replay"]
               if a["solved"] and a["draws"] > 1 and shortest[(r["record_id"], a["index"])] is not None]
    kept = [shortest[(r["record_id"], a["index"])] for r in m4["records"] for a in r["replay"]
            if a["solved"] and a["draws"] == 1 and shortest[(r["record_id"], a["index"])] is not None]
    if redrawn and kept:
        parts.append(f"median decoded shortest distance, redrawn {statistics.median(redrawn):.3f} "
                     f"(n {len(redrawn)}) against kept {statistics.median(kept):.3f} "
                     f"(n {len(kept)}) replay attempts")
    return "; ".join(parts)


def split(argv: list[str], parts: int) -> list[list[str]]:
    groups: list[list[str]] = [[]]
    for token in argv:
        if token == "--":
            groups.append([])
        else:
            groups[-1].append(token)
    stop(len(groups) == parts, f"verdict needs {parts} groups separated by --")
    return groups


def sign_test(values: list[int]) -> tuple[int, int, float]:
    """Records with a positive and a negative value, and P(Bin(n, 1/2) >= positives)."""

    positive = sum(value > 0 for value in values)
    negative = sum(value < 0 for value in values)
    return positive, negative, binomial_upper(positive, positive + negative)


def bootstrap(per_record: dict[str, tuple[float, float]], seed: int) -> tuple[float, float]:
    """Record bootstrap 95 % interval of sum(new) - sum(old) (2000 resamples)."""

    rng = random.Random(seed)
    records = sorted(per_record)
    values = []
    for _ in range(2000):
        draw = rng.choices(records, k=len(records))
        values.append(sum(per_record[r][1] - per_record[r][0] for r in draw))
    values.sort()
    return values[int(0.025 * len(values))], values[int(0.975 * len(values)) - 1]


def verdict(argv: list[str]) -> None:
    smokes, selection, untreated_paths, treated_paths, rest = split(argv, 5)
    stop(len(smokes) == 3 and len(untreated_paths) == len(treated_paths) and len(rest) == 3,
         "verdict: 3 smokes, the selection runs, 3 untreated and 3 treated validation runs "
         "(none if nothing is selected), the stored replay, M2 and M3")
    stored_path, m2_path, m3_path = rest
    stored, stored_sha = load(stored_path)
    m2, m2_sha = load(m2_path)
    m3, m3_sha = load(m3_path)
    stop(stored_sha == STORED_SHA256 and m2_sha == M2_RESULT_SHA256 and m3_sha == M3_RESULT_SHA256,
         "a reference file is not the registered one")
    smoke_off_data(load(smokes[0])[0], stored, m2)
    smoke_m3_data(load(smokes[1])[0], m3, 3)
    smoke_val_data(load(smokes[2])[0])

    # Selection on the test records.
    sel = [load(path)[0] for path in selection]
    stop(bool(sel), "no selection run")
    arms = [{**stored, "threshold": None}]
    for position, item in enumerate(sel, start=1):
        stop(position < len(SEQUENCE) and item["threshold"] == SEQUENCE[position],
             "selection runs are out of sequence")
        verify_file(item, partition="test", threshold=SEQUENCE[position], field=SELECTION_FIELD,
                    cases=50)
        stop(item["references"]["starts"]["sha256"] == M2_RESULT_SHA256
             and item["references"]["outcomes"]["sha256"] == STORED_SHA256,
             f"selection {item['threshold']} used other references")
        require_clean(recheck(item, m2, stored), f"selection {item['threshold']}")
        arms.append(item)
    redrawn_m3 = smoke_m3_data(sel[0], m3, 50)
    chosen, steps = None, []
    for position in range(1, len(arms)):
        accept, pair = step_decision(arms[position - 1], arms[position])
        steps.append({"from": SEQUENCE[position - 1], "to": SEQUENCE[position],
                      "accept": accept, **pair})
        if not accept:
            stop(position == len(arms) - 1, "a selection run exists after the first rejected step")
            break
        chosen = SEQUENCE[position]
    else:
        stop(len(arms) == len(SEQUENCE), "the selection stopped without a rejected step")
    print("Measurement M4: opt-in start-rejection sampler (registered 2026-11-16)")
    print(f"smokes passed; the selection run at 0.1 equals M3 in all 50 cases ({redrawn_m3} redrawn)")
    print(f"SELECTION on the test records, field {SELECTION_FIELD}, sequence {SEQUENCE}; a step "
          f"is accepted when the largest P(fixed | fixed + broken) with no record or any one "
          f"record left out is <= {GAIN_ALPHA}, and P(lost | gained + lost) > {HARM_ALPHA} over "
          f"all 2000 attempts and over the 1000 replay attempts")
    print("step | invalid | fixed / broken | p_gain | p_gain robust | records changed | matched "
          "| gained / lost | p_harm | replay gained / lost | p_harm_replay | accepted")
    for item in steps:
        print(f"{item['from']} -> {item['to']} | {item['invalid_old']} -> {item['invalid_new']} | "
              f"{item['fixed']} / {item['broken']} | {fmt(item['p_gain'])} | "
              f"{fmt(item['p_gain_robust'])} | {item['records_with_validity_change']} | "
              f"{item['matched_old']} -> {item['matched_new']} | {item['gained']} / "
              f"{item['lost']} | {fmt(item['p_harm'])} | {item['replay_gained']} / "
              f"{item['replay_lost']} | {fmt(item['p_harm_replay'])} | {item['accept']}")
    for item in steps:
        print(f"  {item['from']} -> {item['to']}: records holding discordant pairs, "
              f"[fixed, broken, gained, lost]: {item['discordant_records']}; the robust p is "
              f"set by leaving out {item['robust_limited_by']}")
    print(f"SELECTED THRESHOLD: {chosen}")
    report_failed = None
    try:
        first_test = {r["record_id"]: r for r in m2["records"]}
        before, _ = bins_report(sel[0], first_test, stored)
        print(f"  untreated (stored replay): invalid by d0 of the first draw (M2): {before}")
        for item in sel:
            _, used = bins_report(item, first_test, stored)
            print(f"  selection {item['threshold']}: {draw_report(item)}; wall "
                  f"{item['wall_seconds']} s")
            print(f"    invalid by d0 of the integrated start: {used}")
            print(f"    {shift_report(item, first_test)}")
    except Exception as exc:  # reported only; the selection above stands
        report_failed = exc
        print(f"REPORTED SELECTION LINES FAILED: {type(exc).__name__}: {exc}")
    if chosen is None:
        stop(not untreated_paths, "validation runs exist although no threshold was selected")
        print("DECISION: (V) no threshold passes the first selection step; not supported.")
        return
    stop(len(untreated_paths) == 3, "three untreated and three treated validation runs are needed")

    # Confirmation on the validation records.
    untreated = [load(path) for path in untreated_paths]
    treated = [load(path) for path in treated_paths]
    ids = untreated[0][0]["record_ids"]
    for field, (u, u_sha), (t, _) in zip(FIELDS, untreated, treated, strict=True):
        verify_file(u, partition="validation", threshold=None, field=field, cases=50)
        verify_file(t, partition="validation", threshold=chosen, field=field, cases=50)
        stop(u["record_ids"] == t["record_ids"] == ids, "the validation runs have other records")
        stop(not u["references"], f"{field}: the untreated run has references")
        stop(t["references"]["starts"]["sha256"] == u_sha
             and t["references"]["outcomes"]["sha256"] == u_sha,
             f"{field}: the treated run used another untreated run")
        require_clean(recheck(u, None, None), f"untreated {field}")
        require_clean(recheck(t, u, u), f"treated {field}")
    for runs in (untreated, treated):
        for other in runs[1:]:
            shared_starts(runs[0][0], other[0])
    records = sorted(ids)
    gain = {rid: 0 for rid in records}
    harm = {rid: 0 for rid in records}
    harm_replay = {rid: 0 for rid in records}
    per_field = []
    for field, (u, _), (t, _) in zip(FIELDS, untreated, treated, strict=True):
        validity = compare(u, t, kinds=("replay",))
        accuracy = compare(u, t)
        replay_accuracy = compare(u, t, kinds=("replay",))
        old, new = outcomes(u), outcomes(t)
        for key in old:
            rid, kind, _ = key
            (was_invalid, was_matched), (is_invalid, is_matched) = old[key], new[key]
            if kind == "replay":
                gain[rid] += int(was_invalid) - int(is_invalid)
                harm_replay[rid] += int(was_matched) - int(is_matched)
            harm[rid] += int(was_matched) - int(is_matched)
        per_field.append((field, u, t, validity, accuracy, replay_accuracy))
    n_gain, n_loss, p_gain = sign_test(list(gain.values()))
    h_pos, h_neg, p_harm = sign_test(list(harm.values()))
    r_pos, r_neg, p_harm_replay = sign_test(list(harm_replay.values()))
    every_field = all(item[3]["invalid_new"] < item[3]["invalid_old"] for item in per_field)
    print(f"CONFIRMATION at {chosen} angstrom on the 50 validation records, three control fields, "
          "against new untreated runs (the fields share covers, first draws and redraws)")
    print("field | replay invalid untreated -> treated | fixed / broken | McNemar two-sided p "
          "(descriptive) | matched, replay + oracle | gained / lost | replay gained / lost")
    for field, _, _, validity, accuracy, replay_accuracy in per_field:
        print(f"{field} | {validity['invalid_old']} -> {validity['invalid_new']} | "
              f"{validity['fixed']} / {validity['broken']} | "
              f"{fmt(two_sided(validity['fixed'], validity['broken']))} | "
              f"{accuracy['matched_old']} -> {accuracy['matched_new']} | "
              f"{accuracy['gained']} / {accuracy['lost']} | "
              f"{replay_accuracy['gained']} / {replay_accuracy['lost']}")
    print(f"record sign tests (unit: record, summed over the three fields): validity, fewer "
          f"invalid in {n_gain} records, more in {n_loss}, one-sided p_gain {fmt(p_gain)} "
          f"(needs <= {CONFIRM_GAIN_ALPHA}); matches, net loss in {h_pos} records, net gain in "
          f"{h_neg}, p_harm {fmt(p_harm)}; replay matches, net loss in {r_pos}, net gain in "
          f"{r_neg}, p_harm_replay {fmt(p_harm_replay)} (harm when either <= "
          f"{CONFIRM_HARM_ALPHA}); fewer invalid replay attempts in every field (a consistency "
          f"check, not a replication): {every_field}")
    if p_harm <= CONFIRM_HARM_ALPHA or p_harm_replay <= CONFIRM_HARM_ALPHA:
        decision = "(H) harmful: matches fall; not supported"
    elif p_gain <= CONFIRM_GAIN_ALPHA and every_field:
        decision = (f"(S) supported as an opt-in sampler setting at {chosen} angstrom; no harm "
                    "to matches was detected at the power of the harm tests")
    elif p_gain > CONFIRM_GAIN_ALPHA:
        decision = "(N) a validity gain is not shown on the validation records; not supported"
    else:
        decision = "(I) a validity gain over records but not in every field; owner decision"
    print(f"DECISION: {decision}")
    for field, (u, _), (t, _) in zip(FIELDS, untreated, treated, strict=True):
        for label, item in (("untreated", u), ("treated", t)):
            errors = {c["record_id"][:8]: c["request_error"] for c in item["replay_cases"]
                      if c.get("request_error")}
            if errors:
                print(f"  request errors, {label} {field}: {errors}")
    try:
        reported(per_field, records, chosen)
    except Exception as exc:  # the decision above stands; the report must not hide it
        print(f"REPORTED SECTION FAILED: {type(exc).__name__}: {exc}")
        raise SystemExit(3) from exc
    if report_failed is not None:
        raise SystemExit(3)


def reported(per_field, records, chosen) -> None:  # type: ignore[no-untyped-def]
    print("Reported, not part of the decision:")
    pooled = {"fixed": 0, "broken": 0}
    matched = {rid: [0, 0] for rid in records}
    match20 = {rid: [0, 0] for rid in records}
    invalid_counts = {rid: [0, 0] for rid in records}
    for field, u, t, validity, _, _ in per_field:
        pooled["fixed"] += validity["fixed"]
        pooled["broken"] += validity["broken"]
        old, new = summary(u), summary(t)
        print(f"  {field}: " + "; ".join(f"{name} {old[name]} -> {new[name]}" for name in old))
        first_validation = {r["record_id"]: r for r in u["records"]}
        before, used = bins_report(t, first_validation, u)
        counts = recheck(t, u, u)
        print(f"  {field}: {draw_report(t)}; in cases with a failed padded solve, kept outcomes "
              f"that differ {counts['kept_outcome_differs_in_failed_solve_cases']} and kept starts "
              f"rebuilt to rounding {counts['kept_start_rebatched_in_failed_solve_cases']}; wall "
              f"{t['wall_seconds']} s, untreated {u['wall_seconds']} s")
        print(f"  {field}: invalid by d0 of the first draw: {before}")
        print(f"  {field}: invalid by d0 of the integrated start: {used}")
        print(f"  {field}: {shift_report(t, first_validation)}")
        cells = []
        for cutoff in SENSITIVITY_DISTANCES:
            item = compare(u, t, kinds=("replay",), cutoff=cutoff)
            cells.append(f"at {cutoff} angstrom {item['invalid_old']} -> {item['invalid_new']}")
        print(f"  {field}: replay invalid by other validity distances: " + ", ".join(cells))
        for side, cases in ((0, u["replay_cases"]), (1, t["replay_cases"])):
            for case in cases:
                matched[case["record_id"]][side] += case["match_count"]
                match20[case["record_id"]][side] += case["match_count"] > 0
                invalid_counts[case["record_id"]][side] += sum(
                    invalid(v) for v in case["shortest_distances"])
    print(f"  pooled replay pairs (descriptive): fixed {pooled['fixed']}, broken "
          f"{pooled['broken']}, McNemar two-sided p {fmt(two_sided(pooled['fixed'], pooled['broken']))}")
    for cutoff in SENSITIVITY_DISTANCES:
        values = {rid: 0 for rid in records}
        for _, u, t, _, _, _ in per_field:
            old, new = outcomes(u, ("replay",), cutoff), outcomes(t, ("replay",), cutoff)
            for key in old:
                values[key[0]] += int(old[key][0]) - int(new[key][0])
        positive, negative, p = sign_test(list(values.values()))
        print(f"  record sign test at a validity distance of {cutoff} angstrom: fewer invalid in "
              f"{positive} records, more in {negative}, one-sided p {fmt(p)}")
    old_total = sum(v[0] for v in invalid_counts.values())
    new_total = sum(v[1] for v in invalid_counts.values())
    rng = random.Random(20261116)
    ratios = []
    for _ in range(2000):
        draw = rng.choices(records, k=len(records))
        before_total = sum(invalid_counts[rid][0] for rid in draw)
        if before_total:
            ratios.append(sum(invalid_counts[rid][1] for rid in draw) / before_total)
    ratios.sort()
    if old_total and ratios:
        print(f"  pooled replay invalid {old_total} -> {new_total}, ratio {new_total / old_total:.3f};"
              f" record bootstrap 95 % {ratios[int(0.025 * len(ratios))]:.3f} to "
              f"{ratios[int(0.975 * len(ratios)) - 1]:.3f}")
    else:
        print(f"  pooled replay invalid {old_total} -> {new_total}")
    for label, values, seed in (("replay matched attempts", matched, 20261117),
                                ("records matched (match@20), summed over fields", match20, 20261118)):
        low, high = bootstrap({rid: tuple(v) for rid, v in values.items()}, seed)
        change = sum(v[1] - v[0] for v in values.values())
        print(f"  change in {label}: {change:+d}; record bootstrap 95 % {low:+d} to {high:+d}")
    del chosen


def main(argv: list[str]) -> int:
    mode, rest = argv[1], argv[2:]
    try:
        if mode == "smoke-off":
            smoke_off(*rest)
        elif mode == "smoke-m3":
            smoke_m3(*rest)
        elif mode == "smoke-val":
            smoke_val(*rest)
        elif mode == "equals-m3":
            smoke_m3(*rest, cases=50)
        elif mode == "check":
            check(*rest)
        elif mode == "check-untreated":
            check_untreated(*rest)
        elif mode == "same-starts":
            same_starts(*rest)
        elif mode == "step":
            step(*rest)
        elif mode == "verdict":
            verdict(rest)
        else:
            raise Stop(f"unknown mode {mode}")
    except Stop as exc:
        print(f"STOP: {exc}")
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
