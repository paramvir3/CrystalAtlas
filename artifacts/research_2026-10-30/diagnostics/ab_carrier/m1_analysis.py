"""Registered analysis of measurement M1 (R1_LOG.md, "Measurement M1", 2026-11-15).

usage: m1_analysis.py DIRECTORY
DIRECTORY holds m1_{control,smooth}_s{20261030,20261031,20261032}.json.  Written
before any M1 result existed; reads only those six files.
"""

import json
import math
import statistics
import sys
from pathlib import Path

SEEDS = (20261030, 20261031, 20261032)
ARMS = ("control", "smooth")
VALID = 0.5

root = Path(sys.argv[1])
data = {(arm, seed): json.loads((root / f"m1_{arm}_s{seed}.json").read_text()) for arm in ARMS for seed in SEEDS}


def rho_at_switch(value, smoothing, epsilon, reference):
    floor = (2.0 * math.pi * smoothing) ** 2
    return reference * math.sqrt((value * value * floor - epsilon) / (1.0 - value * value)) / (2.0 * math.pi)


transformer = data[("smooth", SEEDS[0])]["transformer"]
assert transformer["pair_carrier_smoothing_length"] == 0.2, transformer
OWN_CARRIER = rho_at_switch(0.90, 0.2, transformer["normalization_epsilon"], transformer["reference_length"])
FAR = rho_at_switch(0.95, 0.2, transformer["normalization_epsilon"], transformer["reference_length"])
print(f"thresholds: switch 0.90 at rho {OWN_CARRIER:.5f}, switch 0.95 at rho {FAR:.5f}")
for key, value in data.items():
    if value["cases"] != 50 or value["cases_reproduced"] != 50 or value["mismatches"]:
        print(f"WARNING {key}: reproduced {value['cases_reproduced']} of {value['cases']}, mismatches {value['mismatches']}")
    print(key, "largest plan minus decoded", value["largest_plan_minus_decoded"],
          "options", value["field_options"], "cell", value["expansion_cell"])
reproduced = {
    key: {case["record_id"]: case for case in value["records"] if case["reproduced"]}
    for key, value in data.items()
}


def attempts(arm, seed, kind):
    for record_id, case in reproduced[(arm, seed)].items():
        for item in case[kind]:
            yield record_id, item


def lookup(arm, seed, kind, record_id, index):
    case = reproduced[(arm, seed)].get(record_id)
    return None if case is None else case[kind][index]


def invalid(value):
    return value is None or value <= VALID


def k_half(lam):
    """Smallest k with lam[k'] <= lam_end / 2 for every k' >= k; undefined steps fail."""

    end = lam[-1]
    found = None
    for k in range(len(lam) - 1, -1, -1):
        if lam[k] is not None and lam[k] <= end / 2.0:
            found = k
        else:
            break
    return found


def member(seed, kind, record_id, index):
    control = lookup("control", seed, kind, record_id, index)
    treated = lookup("smooth", seed, kind, record_id, index)
    if control is None or treated is None:
        return {"missing": "case not reproduced"}
    if not treated["solved"] or not control["solved"]:
        return {"missing": "attempt not recorded (single-member fallback solve)"}
    if treated.get("end") is None or treated["end"]["closest"] is None:
        return {"missing": "treatment end state unresolved"}
    low, high = treated["end"]["closest"]["class"]
    name = f"{low}-{high}"
    if "trajectory" not in treated or "trajectory" not in control:
        return {"missing": "not traced"}
    t_track = treated["trajectory"]["classes"].get(name)
    c_track = control["trajectory"]["classes"].get(name)
    if t_track is None or c_track is None:
        return {"missing": f"class {name} not kept ({'treatment' if t_track is None else 'control'})"}
    lam = [
        None if a is None or b is None or a <= 0.0 or b <= 0.0 else math.log(a / b)
        for a, b in zip(t_track["r"], c_track["r"], strict=True)
    ]
    if lam[-1] is None:
        return {"missing": "Lambda_end undefined"}
    half = k_half(lam)
    unresolved = sorted(
        set(treated["trajectory"]["unresolved_steps"]) | set(control["trajectory"]["unresolved_steps"])
    )
    rho = None if half is None else t_track["rho"][half]
    rho_max = None if half is None else t_track["rho_max"][half]
    return {
        "rho_max_at_half": rho_max,
        "on_special_point": rho is not None and rho < 1.0e-5,
        "class": name,
        "mates": low == high,
        "wyckoff": treated["end"]["closest"]["wyckoff"],
        "atomic_numbers": treated["end"]["closest"]["atomic_numbers"],
        "lambda_end": lam[-1],
        "k_half": half,
        "rho_at_half": rho,
        "r_treatment_at_half": None if half is None else t_track["r"][half],
        "r_control_at_half": None if half is None else c_track["r"][half],
        "unresolved_at_or_after_half": [k for k in unresolved if half is not None and k >= half],
        "half_lattice_at_half": None if half is None else treated["trajectory"]["half_lattice_counts"][half],
        "d_control": control["shortest"],
        "d_treatment": treated["shortest"],
    }


def report(label, members):
    rows = [m for m in members if "missing" not in m and m["rho_at_half"] is not None]
    missing = [m for m in members if "missing" in m or m.get("rho_at_half") is None]
    print(f"\n== {label}: {len(members)} members, {len(rows)} with a statistic, {len(missing)} without")
    for m in missing:
        print("   without:", m)
    if not rows:
        return None
    values = sorted(m["rho_at_half"] for m in rows)
    median = statistics.median(values)
    quantiles = [values[int(q * (len(values) - 1))] for q in (0.1, 0.25, 0.5, 0.75, 0.9)]
    print(f"   median rho_T,c*(k_half) {median:.4f}; q10 q25 q50 q75 q90 {[round(v, 4) for v in quantiles]}")
    print(f"   at most {OWN_CARRIER}: {sum(v <= OWN_CARRIER for v in values)}; at least {FAR}: {sum(v >= FAR for v in values)}")
    print(f"   mates among c*: {sum(m['mates'] for m in rows)} of {len(rows)}")
    print(f"   k_half quantiles {[sorted(m['k_half'] for m in rows)[int(q * (len(rows) - 1))] for q in (0.1, 0.5, 0.9)]}")
    unresolved = [m for m in rows if m["unresolved_at_or_after_half"]]
    print(f"   members with an unresolved step at or after k_half: {len(unresolved)}")
    special = [m for m in rows if m["on_special_point"]]
    print(f"   members whose c* pair sits on {{0, 1/2}}^3 at k_half (carriers zero in both arms): {len(special)}")
    spread = sorted(m["rho_max_at_half"] - m["rho_at_half"] for m in rows)
    upper = statistics.median(m["rho_max_at_half"] for m in rows)
    print(f"   tie spread rho_max - rho at k_half: largest {spread[-1]:.4g}; median of rho_max {upper:.4f}")
    return median


v_members = []
for seed in SEEDS:
    for record_id, control in attempts("control", seed, "replay"):
        treated = lookup("smooth", seed, "replay", record_id, control["index"])
        if treated is None:
            continue
        if not invalid(control["shortest"]) and invalid(treated["shortest"]):
            v_members.append({"seed": seed, "record": record_id[:8], "attempt": control["index"],
                              **member(seed, "replay", record_id, control["index"])})
print("\nset V (replay, valid in control, invalid in treatment):", len(v_members))
for m in v_members:
    print("  ", {k: (round(v, 4) if isinstance(v, float) else v) for k, v in m.items()})
median = report("V (registered)", v_members)
if median is None:
    print("REGISTERED DECISION: no statistic")
elif median <= OWN_CARRIER:
    print(f"REGISTERED DECISION: (a) median {median:.4f} <= {OWN_CARRIER}")
elif median >= FAR:
    print(f"REGISTERED DECISION: (b) median {median:.4f} >= {FAR}")
else:
    print(f"REGISTERED DECISION: (c) inconclusive, median {median:.4f}")

band = []
for seed in SEEDS:
    for record_id, control in attempts("control", seed, "replay"):
        treated = lookup("smooth", seed, "replay", record_id, control["index"])
        if treated is None or control["shortest"] is None or treated["shortest"] is None:
            continue
        geometric = math.sqrt(control["shortest"] * treated["shortest"])
        if 0.1 < geometric <= 0.5:
            m = member(seed, "replay", record_id, control["index"])
            if "missing" in m or m["lambda_end"] <= -0.1:
                band.append(m)
report("band set (reported, not registered)", band)

oracle = []
for seed in SEEDS:
    for record_id, control in attempts("control", seed, "oracle"):
        treated = lookup("smooth", seed, "oracle", record_id, control["index"])
        if treated is None:
            continue
        if not invalid(control["shortest"]) and invalid(treated["shortest"]):
            oracle.append(member(seed, "oracle", record_id, control["index"]))
report("oracle draws valid in control and invalid in treatment (reported)", oracle)
