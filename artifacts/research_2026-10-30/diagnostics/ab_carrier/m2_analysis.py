"""Registered analysis of measurement M2 (R1_LOG.md, "Measurement M2", 2026-11-15).

usage: m2_analysis.py M2_FILE M1_DIRECTORY   (Python 3.10 or newer; run with 2>&1)
M2_FILE is the starts file of the carrier control seed 20261030; M1_DIRECTORY holds the six
files m1_{control,smooth}_s{20261030,20261031,20261032}.json.  Written before the M2 run;
standard library only.
"""

import json
import math
import random
import sys
from pathlib import Path

SEEDS = (20261030, 20261031, 20261032)
ARMS = ("control", "smooth")
KINDS = ("replay", "oracle")
VALID = 0.5
NEAR = 0.1
OTHER_THRESHOLDS = (0.01, 0.03, 0.3, 0.5)
EDGES = (0.01, 0.03, 0.1, 0.3, 0.5, 1.0, 4.0)  # upper bin edges in angstrom; then above 4
RADIUS = 4.0
BOOTSTRAP = 2000
BOOTSTRAP_SEED = 20261115
FIELD = "control_s20261030"
SCHEMA = "crystalatlas.research-v7-m2-starts.v1"
M1_SHA256 = "1f8332a87fc8c707c71ad35a745862665bc2b4c7cd19ebd71465950e0d3ca432"
TRACED_PER_FILE = {"replay": 131, "oracle": 111}  # M1 registration, m1_targets.json


def stop(message):
    sys.exit(f"STOP: {message}")


m2 = json.loads(Path(sys.argv[1]).read_text())
m1_root = Path(sys.argv[2])
m1 = {(arm, seed): json.loads((m1_root / f"m1_{arm}_s{seed}.json").read_text()) for arm in ARMS for seed in SEEDS}

# Validity of the inputs.
if m2.get("schema") != SCHEMA:
    stop(f"schema {m2.get('schema')}")
if m2["cases"] != 50:
    stop(f"M2 holds {m2['cases']} cases")
if Path(m2["field_run"]).name != FIELD:
    stop(f"M2 field run {m2['field_run']}")
if m2["m1_script_sha256"] != M1_SHA256 or m2["radius"] != RADIUS:
    stop(f"M2 ran M1 script {m2['m1_script_sha256']} with radius {m2['radius']}")
for (arm, seed), value in m1.items():
    if value["cases"] != 50 or Path(value["field_run"]).name != f"{arm}_s{seed}":
        stop(f"M1 file {arm} {seed} holds {value['cases']} cases of field {value['field_run']}")
    if value["script_sha256"] != M1_SHA256 or value["keep_min"] < RADIUS or value["radius"] != RADIUS:
        stop(f"M1 file {arm} {seed}: script, radius or keep_min differ")
    for name in ("protocol_seed", "record_ids", "cover_checkpoint_sha256", "candidates", "oracle_draws", "code"):
        if value[name] != m2[name]:
            stop(f"M1 {arm} {seed} differs from M2 in {name}")
if len({value["field_checkpoint_sha256"] for value in m1.values()}) != 6:
    stop("the six M1 files are not six different fields")
if m1[("control", SEEDS[0])]["field_checkpoint_sha256"] != m2["field_checkpoint_sha256"]:
    stop("M2 is not the field of m1_control_s20261030")

# Cases count only when they reproduce in M2 and in every M1 file (registration).
reproduced = {case["record_id"] for case in m2["records"] if case["reproduced"]}
for value in m1.values():
    reproduced &= {case["record_id"] for case in value["records"] if case["reproduced"]}
left_out = sorted({case["record_id"][:8] for case in m2["records"]} - {r[:8] for r in reproduced})
print(f"M2: field {FIELD}, m2 script {m2['m2_script_sha256'][:12]}, m1 script {m2['m1_script_sha256'][:12]}; "
      f"cases used {len(reproduced)} of 50; left out (not reproduced somewhere): {left_out}")
if not reproduced:
    stop("no case reproduced")

starts = {}
unsolved = {kind: [] for kind in KINDS}
unresolved = {kind: [] for kind in KINDS}
for case in m2["records"]:
    if case["record_id"] not in reproduced:
        continue
    for kind in KINDS:
        expected = m2["candidates"] if kind == "replay" else m2["oracle_draws"]
        if len(case[kind]) != expected:
            stop(f"{case['record_id'][:8]} {kind}: {len(case[kind])} attempts")
        for position, attempt in enumerate(case[kind]):
            if attempt["index"] != position or attempt["traced"]:
                stop(f"{case['record_id'][:8]} {kind} {position}: index or traced flag")
            key = (case["record_id"], kind, position)
            if attempt["solved"]:
                starts[key] = attempt["start"]
                if not attempt["start"]["resolved"]:
                    unresolved[kind].append(key)
            else:
                unsolved[kind].append(key)
print(f"starts recorded: {len(starts)}; unsolved, no start: "
      f"{ {k: len(v) for k, v in unsolved.items()} }; unresolved start: { {k: len(v) for k, v in unresolved.items()} }")
for kind in KINDS:
    for key in unsolved[kind] + unresolved[kind]:
        print(f"   without a defined start: {kind} {key[0][:8]} attempt {key[2]}")

# M2 replays the field of m1_control_s20261030: every attempt must be the same.
same_field = m1[("control", SEEDS[0])]
outcome_mismatch = []
for case, other in zip(m2["records"], same_field["records"], strict=True):
    if case["record_id"] != other["record_id"]:
        stop("record order differs from m1_control_s20261030")
    if case["record_id"] not in reproduced:
        continue
    for kind in KINDS:
        for a, b in zip(case[kind], other[kind], strict=True):
            if a["shortest"] != b["shortest"] or a["solved"] != b["solved"] or a.get("end") != b.get("end"):
                outcome_mismatch.append((case["record_id"][:8], kind, a["index"]))
if outcome_mismatch:
    stop(f"{len(outcome_mismatch)} attempts differ from m1_control_s20261030: {outcome_mismatch[:10]}")
print("every attempt equals m1_control_s20261030 (decoded distance, solved flag, end state)")


def overall(start):
    values = [v for v in (start["lattice_min"], (start["closest"] or {}).get("r")) if v is not None]
    return min(values) if values else None


# Cross-check: the start equals step 0 of every traced M1 trajectory, in all six files.
checked = 0
bad = []
for key, value in m1.items():
    for case in value["records"]:
        if case["record_id"] not in reproduced:
            continue
        for kind in KINDS:
            for attempt in case[kind]:
                if not attempt["traced"]:
                    continue
                checked += 1
                where = (key, case["record_id"][:8], kind, attempt["index"])
                start = starts.get((case["record_id"], kind, attempt["index"]))
                if not attempt["solved"]:
                    if start is not None:
                        bad.append((*where, "solved in M2 only"))
                    continue
                if start is None:
                    bad.append((*where, "no start in M2"))
                    continue
                trajectory = attempt["trajectory"]
                step0_unresolved = 0 in trajectory["unresolved_steps"]
                if not start["resolved"]:
                    if not step0_unresolved:
                        bad.append((*where, "M2 unresolved, M1 resolved"))
                    continue
                if step0_unresolved:
                    bad.append((*where, "M2 resolved, M1 unresolved"))
                    continue
                if trajectory["closest_r"][0] != overall(start) or trajectory["lattice_min"][0] != start["lattice_min"]:
                    bad.append((*where, "closest"))
                if start["closest"] is not None:
                    low, high = start["closest"]["class"]
                    track = trajectory["classes"].get(f"{low}-{high}")
                    if track is None or track["r"][0] != start["closest"]["r"] or track["rho"][0] != start["closest"]["rho"]:
                        bad.append((*where, "class track"))
expected_checks = 6 * sum(TRACED_PER_FILE.values()) if len(reproduced) == 50 else None
if bad:
    stop(f"{len(bad)} of {checked} traced attempts differ from the start: {bad[:10]}")
if expected_checks is not None and checked != expected_checks:
    stop(f"{checked} traced attempts compared, expected {expected_checks}")
print(f"cross-check: start equals step 0 of the M1 trajectory for {checked} traced attempts in six files")


def d0(start, with_lattice=False):
    """Smallest image distance between distinct atoms at step 0; inf beyond the radius; None if undefined."""

    if start is None or not start["resolved"]:
        return None
    value = math.inf if start["closest"] is None else start["closest"]["r"]
    if with_lattice and start["lattice_min"] is not None:
        value = min(value, start["lattice_min"])
    return value


def invalid(attempt):
    return attempt["shortest"] is None or attempt["shortest"] <= VALID


def outcomes(arm, kind, seeds=SEEDS, measure=d0):
    """(record_id, separation, invalid, attempt, start) per outcome, and the invalid outcomes left out."""

    rows, missing_invalid = [], 0
    for seed in seeds:
        for case in m1[(arm, seed)]["records"]:
            if case["record_id"] not in reproduced:
                continue
            for attempt in case[kind]:
                start = starts.get((case["record_id"], kind, attempt["index"]))
                value = measure(start)
                if value is None:
                    missing_invalid += invalid(attempt)
                    continue
                rows.append((case["record_id"], value, invalid(attempt), attempt, start))
    return rows, missing_invalid


def ratio(a, b):
    return a / b if b else math.nan


def statistics_of(rows, threshold):
    near = [r for r in rows if r[1] <= threshold]
    far = [r for r in rows if r[1] > threshold]
    bad = sum(r[2] for r in rows)
    bad_near = sum(r[2] for r in near)
    bad_far = sum(r[2] for r in far)
    rate_near, rate_far = ratio(bad_near, len(near)), ratio(bad_far, len(far))
    if math.isnan(rate_near) or math.isnan(rate_far):
        rr = math.nan
    elif rate_far == 0.0:
        rr = math.inf if rate_near > 0.0 else math.nan
    else:
        rr = rate_near / rate_far
    return {"outcomes": len(rows), "near": len(near), "invalid": bad, "invalid_near": bad_near,
            "F": ratio(bad_near, bad), "rate_near": rate_near, "rate_far": rate_far, "RR": rr}


def bootstrap(rows, threshold):
    by_record = {}
    for row in rows:
        by_record.setdefault(row[0], []).append(row)
    records = sorted(by_record)
    generator = random.Random(BOOTSTRAP_SEED)
    shares, ratios, undefined = [], [], 0
    for _ in range(BOOTSTRAP):
        sample = [row for record in generator.choices(records, k=len(records)) for row in by_record[record]]
        result = statistics_of(sample, threshold)
        if math.isnan(result["F"]) or math.isnan(result["RR"]):
            undefined += 1
            continue
        shares.append(result["F"])
        ratios.append(result["RR"])
    shares.sort()
    ratios.sort()

    def interval(values):
        if not values:
            return (math.nan, math.nan)
        return (values[int(0.025 * (len(values) - 1))], values[int(0.975 * (len(values) - 1))])

    return interval(shares), interval(ratios), undefined


def line(label, result):
    return (f"{label}: outcomes {result['outcomes']}, near {result['near']}, invalid {result['invalid']} "
            f"({result['invalid_near']} near); F {result['F']:.4f}; invalid rate near {result['rate_near']:.4f}, "
            f"other {result['rate_far']:.4f}; RR {result['RR']:.3f}")


# Registered statistic: control arms, replay, three seeds pooled.
control, control_missing = outcomes("control", "replay")
states = {(r[0], r[3]["index"]): r[1] for r in control}
total_states = len(reproduced) * m2["candidates"]
print(f"\nreplay states with a defined d0: {len(states)} of {total_states}; control outcomes {len(control)}; "
      f"invalid control outcomes without a defined start (left out): {control_missing}")
if not states:
    stop("no replay state has a defined d0")
near_states = sum(value <= NEAR for value in states.values())
mass = near_states / len(states)
print(f"mass of near-coincident starts (d0 <= {NEAR}): {near_states} of {len(states)} states = {mass:.4f}")
registered = statistics_of(control, NEAR)
(f_low, f_high), (rr_low, rr_high), undefined = bootstrap(control, NEAR)
print(line(f"REGISTERED, control, d0 <= {NEAR}", registered))
print(f"   bootstrap 95 % over records: F [{f_low:.4f}, {f_high:.4f}], RR [{rr_low:.3f}, {rr_high:.3f}], "
      f"undefined resamples {undefined}")
if control_missing:
    bad_all = registered["invalid"] + control_missing
    print(f"   bounds with the {control_missing} left-out invalid outcomes counted as near / as far: "
          f"F {(registered['invalid_near'] + control_missing) / bad_all:.4f} / {registered['invalid_near'] / bad_all:.4f}")
F, RR = registered["F"], registered["RR"]
if math.isnan(F) or math.isnan(RR):
    print(f"REGISTERED DECISION: none, statistic undefined (F {F}, RR {RR})")
elif F >= 0.5 and RR >= 5.0:
    print(f"REGISTERED DECISION: (P) F {F:.4f} >= 0.5 and RR {RR:.3f} >= 5")
elif F < 0.25:
    print(f"REGISTERED DECISION: (N) F {F:.4f} < 0.25")
else:
    print(f"REGISTERED DECISION: (I) F {F:.4f}, RR {RR:.3f}")

# Reported, not part of the decision.
print("\n== invalid rate by d0 bin (replay, three seeds pooled)")
arm_rows = {arm: outcomes(arm, "replay")[0] for arm in ARMS}
lower = 0.0
for upper in (*EDGES, math.inf):
    def inside(value, lower=lower, upper=upper):
        return lower < value <= upper or (lower == 0.0 and value == 0.0)

    cells = []
    for arm in ARMS:
        rows = [r for r in arm_rows[arm] if inside(r[1])]
        bad_count = sum(r[2] for r in rows)
        cells.append(f"{arm} {bad_count}/{len(rows)} = {ratio(bad_count, len(rows)):.4f}")
    print(f"   ({lower}, {upper}]: states {sum(inside(v) for v in states.values())}; " + "; ".join(cells))
    lower = upper

print("\n== other thresholds, control")
for threshold in OTHER_THRESHOLDS:
    print("   " + line(f"d0 <= {threshold}", statistics_of(control, threshold)))
print("\n== per seed, control, d0 <= 0.1")
for seed in SEEDS:
    print("   " + line(f"seed {seed}", statistics_of(outcomes("control", "replay", seeds=(seed,))[0], NEAR)))
print("\n== treatment arms, d0 <= 0.1")
print("   " + line("smooth", statistics_of(arm_rows["smooth"], NEAR)))
print("\n== d0 including the shortest lattice vector, control, d0 <= 0.1")
print("   " + line("with lattice", statistics_of(outcomes("control", "replay", measure=lambda s: d0(s, True))[0], NEAR)))


def normalized(start):
    value = d0(start)
    volume = None if start is None else start.get("volume_per_atom")
    if value is None or volume is None or volume <= 0.0:
        return None
    return value / volume ** (1.0 / 3.0)


print("\n== d0 / (starting volume per atom)^(1/3), control")
scaled = outcomes("control", "replay", measure=normalized)[0]
scaled_by_state = {(r[0], r[3]["index"]): r[1] for r in scaled}
if set(scaled_by_state) != set(states):
    print(f"   note: {len(set(states) - set(scaled_by_state))} states have no starting volume")
scaled_states = sorted(scaled_by_state.values())
if scaled_states:
    matched = scaled_states[near_states - 1] if near_states else 0.0
    print(f"   threshold with the same number of states as d0 <= {NEAR} ({near_states}): {matched:.6g}")
    print("   " + line("normalized", statistics_of(scaled, matched)))
    deciles = [scaled_states[int(q * (len(scaled_states) - 1))] for q in (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)]
    lower = -math.inf
    for upper in (*deciles, math.inf):
        rows = [r for r in scaled if lower < r[1] <= upper]
        bad_count = sum(r[2] for r in rows)
        print(f"   ({lower:.4g}, {upper:.4g}]: outcomes {len(rows)}, invalid {bad_count} = {ratio(bad_count, len(rows)):.4f}")
        lower = upper

print("\n== closest class at the start, control")
near_rows = [r for r in control if r[1] <= NEAR]
bad_rows = [r for r in control if r[2]]
mates_near = sum(r[4]["closest"]["mates"] for r in near_rows if r[4]["closest"] is not None)
mates_bad = sum(r[4]["closest"]["mates"] for r in bad_rows if r[4]["closest"] is not None)
print(f"   mates among near-coincident outcomes: {mates_near} of {len(near_rows)}; among invalid outcomes: "
      f"{mates_bad} of {len(bad_rows)}")
same = comparable = 0
for r in bad_rows:
    end = r[3].get("end")
    if r[4]["closest"] is None or end is None or end["closest"] is None:
        continue
    comparable += 1
    same += end["closest"]["class"] == r[4]["closest"]["class"]
print(f"   invalid outcomes whose closest class at the end is the closest class at the start: {same} of {comparable}")

print("\n== oracle draws, control, d0 <= 0.1")
oracle, oracle_missing = outcomes("control", "oracle")
print("   " + line("oracle", statistics_of(oracle, NEAR)) + f"; invalid outcomes without a start {oracle_missing}")
oracle_states = {(r[0], r[3]["index"]): r[1] for r in oracle}
print(f"   oracle mass of near-coincident starts: {sum(v <= NEAR for v in oracle_states.values())} of {len(oracle_states)}")
