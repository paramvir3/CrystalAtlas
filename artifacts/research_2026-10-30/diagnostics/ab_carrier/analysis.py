"""Registered analysis of the pair-carrier smoothing A/B (R1_LOG.md, 2026-11-12).

Run in r3/ab_carrier.  Reads replay_{control,smooth}_s<seed>.json, the runs'
selection-geometry.json, run 2's selection, and the scan dumps of seed 20261030.
"""
import json
import math
import os
import sys

import numpy as np

REGISTERED = (20261030, 20261031, 20261032)
OPTIONS = {"control": "default", "smooth": "carrier-smoothing"}
SEEDS = [s for s in REGISTERED
         if all(os.path.exists(f"replay_{arm}_s{s}.json") for arm in ("control", "smooth"))]
ARMS = ("control", "smooth")
if not SEEDS:
    sys.exit("no complete seed yet")
PARTIAL = SEEDS != list(REGISTERED)
if PARTIAL and "--partial" not in sys.argv:
    sys.exit(f"only seeds {SEEDS} are complete; the registered verdict needs {list(REGISTERED)} "
             "(pass --partial for a progress reading that is not the verdict)")
print("seeds analysed:", SEEDS, "PARTIAL: NOT THE REGISTERED VERDICT" if PARTIAL else "(all registered seeds)")


def load(arm, seed):
    return json.load(open(f"replay_{arm}_s{seed}.json"))


def invalid(case):
    return sum(v is None or v <= 0.5 for v in case["shortest_distances"])


def collapsed(case):
    return sum(v is None or v <= 0.01 for v in case["shortest_distances"])


data = {(arm, seed): load(arm, seed) for arm in ARMS for seed in SEEDS}
records = data[("control", SEEDS[0])]["record_ids"]
for key, value in data.items():
    assert value["record_ids"] == records, key
    assert value["candidates"] == 20 and value["oracle_draws"] == 20, key
    assert value["protocol_seed"] == 20261530, key
    arm, seed = key
    assert value["field_options"] == OPTIONS[arm], (key, value["field_options"])
    assert value["field_run"].rstrip("/").endswith(f"{arm}_s{seed}"), (key, value["field_run"])
    summary = value["replay_summary"]
    if summary["request_error_cases"] or summary["failed_stages"].get("request"):
        print(f"WARNING {key}: request-level failures {summary['request_errors']} "
              f"count toward N for all 20 attempts of the case")
print("records", len(records))
print("model fields differing from the cover run:")
for arm in ARMS:
    print("  ", arm, data[(arm, SEEDS[0])]["model_fields_differing"], "options", data[(arm, SEEDS[0])]["field_options"])

print("\nper arm and seed: N (no structure), S01, S5, C = N + S01, I = N + S5, failed stages")
totals = {arm: {"C": 0, "I": 0} for arm in ARMS}
for seed in SEEDS:
    for arm in ARMS:
        d = data[(arm, seed)]
        dist = d["replay_distances"]
        n, s01, s5 = dist["without_structure"], dist["at_most_0.01"], dist["at_most_0.5"]
        totals[arm]["C"] += n + s01
        totals[arm]["I"] += n + s5
        summary = d["replay_summary"]
        oracle = d["oracle_summary"]
        selection = json.load(open(f"{arm}_s{seed}/selection-geometry.json"))["selected"]
        print(f"  s{seed} {arm:7s} N {n:3d} S01 {s01:3d} S5 {s5:3d} C {n + s01:3d} I {n + s5:3d} "
              f"{summary['failed_stages']} | replay match/attempt {summary['match_per_attempt']['rate']:.4f} "
              f"@20 {summary['match_at_k']['successes']} @1 {summary['match_at_1']['successes']} "
              f"table-given {summary['match_given_target_table']} | oracle match/draw {oracle['match_per_draw']:.4f} "
              f"any {oracle['cases_any_draw_matches']} first {oracle['cases_first_draw_matches']} "
              f"rms {oracle['mean_rms_of_matches']} oracle dist {oracle['distances']} | selected {selection}")
print("\nP1: C summed over seeds:", {arm: totals[arm]["C"] for arm in ARMS},
      "| quarter of control:", totals["control"]["C"] / 4,
      "| holds:", totals["smooth"]["C"] <= totals["control"]["C"] / 4)
print("    I summed over seeds:", {arm: totals[arm]["I"] for arm in ARMS})


def paired(function, key):
    """Per-record difference treatment minus control, averaged over the seeds."""
    diffs = np.zeros(len(records))
    for seed in SEEDS:
        control = {c["record_id"]: function(c) for c in data[("control", seed)][key]}
        treated = {c["record_id"]: function(c) for c in data[("smooth", seed)][key]}
        diffs += np.array([treated[r] - control[r] for r in records], dtype=float)
    diffs /= len(SEEDS)
    mean = float(diffs.mean())
    se = float(diffs.std(ddof=1) / math.sqrt(len(diffs)))
    return mean, se, (mean / se if se > 0 else float("nan")), int((diffs != 0).sum())


for label, function, key in (
    ("P2 invalid attempts per record (of 20)", invalid, "replay_cases"),
    ("   collapsed attempts per record (of 20)", collapsed, "replay_cases"),
    ("P3 oracle match fraction", lambda c: sum(c["matches"]) / len(c["matches"]), "oracle_cases"),
    ("P3 replay match fraction", lambda c: c["match_count"] / c["attempts"], "replay_cases"),
    ("   oracle invalid draws per record (of 20)", invalid, "oracle_cases"),
):
    mean, se, ratio, nonzero = paired(function, key)
    print(f"{label}: mean difference {mean:+.4f}, standard error {se:.4f}, ratio {ratio:+.2f}, "
          f"records with a difference {nonzero}")

# Consistency check against run 2.
run2 = json.load(open("../v7_medium_run2/selection-geometry.json"))
if os.path.exists("control_s20261030/selection-geometry.json"):
    mine = json.load(open("control_s20261030/selection-geometry.json"))
    theirs = {(h["phase_epoch"], h["weights"]): h["cfm_regression"] for h in run2["history"]}
    worst = 0.0
    shared = 0
    for h in mine["history"]:
        ref = theirs.get((h["phase_epoch"], h["weights"]))
        if ref is not None:
            shared += 1
            worst = max(worst, abs(ref - h["cfm_regression"]))
    if shared == 0:
        sys.exit("consistency check has no shared validation point")
    print("consistency: shared validation points", shared)
    print("\nconsistency: largest |control s20261030 - run 2| over shared validation points:", worst)
    print("  control selected", mine["selected"], "| run 2 selected", run2["selected"])
    print("  checkpoints: control", data[("control", 20261030)]["field_checkpoint_sha256"][:16],
          "run 2 119e249ac6cb54d7")

# P4: slopes of the diagnosed coordinates.
for case in ("1518540c", "8733cd61"):
    for arm in ARMS:
        path = f"scan_{case}_{arm}.bucket-v1.json"
        if not os.path.exists(path):
            continue
        report = json.load(open(path))
        for scan in report["scans"]:
            slopes = []
            for step in scan["steps"]:
                value = [p[1] for p in step["points"] if abs(p[0] - 1e-5) < 1e-9]
                slopes.append((step["time"], value[0] / 1e-5))
            largest = max(slopes, key=lambda item: abs(item[1]))
            attempt = report["attempts"][scan["attempt"]]
            print(f"P4 {case} {arm}: attempt {scan['attempt']} coordinate {scan['coordinate']} "
                  f"largest |slope| {abs(largest[1]):.1f} at t {largest[0]:.4f}; "
                  f"stage {attempt['failed_stage']} final eta {attempt['final_eta'][scan['coordinate']]:.6e} "
                  f"shortest {attempt.get('shortest_distance')}")
            print("     slopes:", " ".join(f"{t:.3f}:{s:+.0f}" for t, s in slopes))
