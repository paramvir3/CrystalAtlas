"""Registered analysis of measurement M1b (R1_LOG.md, "Measurement M1b", 2026-11-15).

usage: m1b_analysis.py DIRECTORY   (Python 3.10 or newer; run with 2>&1 to keep STOP lines)
DIRECTORY holds m1_{control,smooth}_s{20261030,20261031,20261032}.json, the six M1 result
files.  Written before any path of a member of G was read; reads only those six files.
Standard library only.
"""

import json
import math
import statistics
import sys
from pathlib import Path

SEEDS = (20261030, 20261031, 20261032)
ARMS = ("control", "smooth")
VALID = 0.5
COLLAPSE = 0.01
SENSITIVITY = (0.001, 0.1)
HARM_MEDIAN = 0.03056363  # registered M1 statistic, R1_LOG.md "Result of measurement M1"
REGISTERED_S_STAR = 0.014908
REGISTERED_BOUND = 0.014737  # rule (A): m <= 0.014737 angstrom, as registered
EXPECTED_COLLAPSES = {"control": (5, 3, 4), "smooth": (2, 2, 0)}  # carrier A/B table
EXPECTED_G = (5, 3, 4)  # implied by the A/B result; checked and reported, not a stop
LAST = 64

root = Path(sys.argv[1])
data = {(arm, seed): json.loads((root / f"m1_{arm}_s{seed}.json").read_text()) for arm in ARMS for seed in SEEDS}

# Constants of the switch, read from the files.
smooth_constants = {data[("smooth", seed)]["transformer"]["pair_carrier_smoothing_length"] for seed in SEEDS}
epsilons = {data[key]["transformer"]["normalization_epsilon"] for key in data}
references = {data[key]["transformer"]["reference_length"] for key in data}
assert smooth_constants == {0.2}, smooth_constants
assert all(data[("control", seed)]["transformer"]["pair_carrier_smoothing_length"] is None for seed in SEEDS)
assert len(epsilons) == 1 and len(references) == 1, (epsilons, references)
EPSILON = epsilons.pop()
REFERENCE = references.pop()
assert REFERENCE == 1.0, "the thresholds below are in angstrom only for l_0 = 1 angstrom"
RHO_0 = math.sqrt(EPSILON) * REFERENCE / (2.0 * math.pi)


def switch(rho, smoothing):
    return math.sqrt((rho * rho + RHO_0 * RHO_0) / (rho * rho + smoothing * smoothing))


S_STAR = math.sqrt((HARM_MEDIAN**2 + RHO_0**2) / 0.81 - HARM_MEDIAN**2)
EXACT_BOUND = math.sqrt(S_STAR**2 - 2.0 * RHO_0**2)
assert abs(switch(HARM_MEDIAN, S_STAR) - 0.9) < 1e-12
assert abs(switch(EXACT_BOUND, S_STAR) - 1.0 / math.sqrt(2.0)) < 1e-12
assert round(S_STAR, 6) == REGISTERED_S_STAR and round(EXACT_BOUND, 6) == REGISTERED_BOUND, (S_STAR, EXACT_BOUND)
print(f"rho_0 {RHO_0!r}; s* {S_STAR!r} (switch at h {switch(HARM_MEDIAN, S_STAR):.6f}); "
      f"exact 1/sqrt(2) point {EXACT_BOUND!r}; registered bound of (A) {REGISTERED_BOUND}; "
      f"switch at h for s = 0.015: {switch(HARM_MEDIAN, 0.015):.5f}")

# Validity: every case reproduced, and the collapse counts of the carrier A/B table.
for key, value in data.items():
    if value["cases"] != 50 or value["cases_reproduced"] != 50 or value["mismatches"]:
        sys.exit(f"STOP {key}: reproduced {value['cases_reproduced']} of {value['cases']}, "
                 f"mismatches {value['mismatches']}")
cases = {key: {case["record_id"]: case for case in value["records"]} for key, value in data.items()}


def collapsed(attempt):
    return attempt["shortest"] is None or attempt["shortest"] <= COLLAPSE


def invalid(attempt):
    return attempt["shortest"] is None or attempt["shortest"] <= VALID


def collapse_counts(arm, kind):
    return tuple(sum(collapsed(a) for case in cases[(arm, seed)].values() for a in case[kind]) for seed in SEEDS)


for arm in ARMS:
    counts = collapse_counts(arm, "replay")
    print(f"replay collapses {arm}: {counts}")
    if counts != EXPECTED_COLLAPSES[arm]:
        sys.exit(f"STOP: collapse counts {counts} differ from the A/B table {EXPECTED_COLLAPSES[arm]}")
for arm in ARMS:
    print(f"oracle collapses {arm} (reported): {collapse_counts(arm, 'oracle')}")


def pairs(seed, kind):
    """Yield (record_id, control attempt, treatment attempt), paired by record and index."""

    for record_id, control_case in cases[("control", seed)].items():
        treated_case = cases[("smooth", seed)][record_id]
        assert len(control_case[kind]) == len(treated_case[kind])
        for position, (control, treated) in enumerate(zip(control_case[kind], treated_case[kind], strict=True)):
            assert control["index"] == treated["index"] == position
            yield record_id, control, treated


def defined_rho(track, k):
    """rho of the class at step k, or None when its distance is undefined there.

    m1_contacts stores r = None when no pair of the class is within the radius or the step
    is unresolved; rho is then not the minimizing pair's and is not used.
    """

    return None if track["r"][k] is None else track["rho"][k]


def tracks(control, treated):
    """The class c-dagger and both arms' tracks of it, or a reason why there are none."""

    for arm, attempt in (("control", control), ("treatment", treated)):
        if not attempt["solved"]:
            return f"{arm} attempt not recorded (solved false, failed stage {attempt.get('failed_stage')})"
        if "trajectory" not in attempt:
            return f"{arm} attempt not traced"
    end = control.get("end")
    if end is None:
        return "control end state unresolved (image search failed at t = 1)"
    if end["closest"] is None:
        return f"control end state has no atom pair within the radius (lattice_min {end['lattice_min']})"
    low, high = end["closest"]["class"]
    name = f"{low}-{high}"
    c_track = control["trajectory"]["classes"].get(name)
    t_track = treated["trajectory"]["classes"].get(name)
    if c_track is None or t_track is None:
        return f"class {name} not kept ({'control' if c_track is None else 'treatment'})"
    assert len(c_track["r"]) == len(t_track["r"]) == LAST + 1
    return name, end, c_track, t_track


def k_cap(r_control, threshold):
    """Smallest k with r_control[k'] <= threshold for every k' >= k; undefined steps fail."""

    found = None
    for k in range(LAST, -1, -1):
        if r_control[k] is not None and r_control[k] <= threshold:
            found = k
        else:
            break
    return found


def gain_member(seed, record_id, control, treated, threshold=COLLAPSE):
    head = {"seed": seed, "record": record_id[:8], "attempt": control["index"],
            "d_control": control["shortest"], "failed_stage_control": control.get("failed_stage"),
            "d_treatment": treated["shortest"]}
    found = tracks(control, treated)
    if isinstance(found, str):
        return {**head, "missing": found}
    name, end, c_track, t_track = found
    closest = end["closest"]
    rho_t = [(k, defined_rho(t_track, k)) for k in range(LAST + 1)]
    rho_t = [(k, v) for k, v in rho_t if v is not None]
    smallest = min(rho_t, key=lambda item: item[1]) if rho_t else (None, None)
    described = {
        **head,
        "class": name,
        "mates": closest["mates"],
        "wyckoff": closest["wyckoff"],
        "atomic_numbers": closest["atomic_numbers"],
        "control_end_r": closest["r"],
        "control_end_lattice_min": end["lattice_min"],
        "rho_0_control": defined_rho(c_track, 0),
        "rho_0_treatment": defined_rho(t_track, 0),
        "rho_treatment_min": smallest[1],
        "k_rho_treatment_min": smallest[0],
        "rho_treatment_end": defined_rho(t_track, LAST),
        "other_classes_within_threshold": [
            f"{lo}-{hi}" for lo, hi, r, _, _ in end["classes"] if r <= threshold and f"{lo}-{hi}" != name
        ],
    }
    cap = k_cap(c_track["r"], threshold)
    if cap is None:
        return {**described, "missing": f"control pair above {threshold} at t = 1 or undefined there"}
    gamma = defined_rho(t_track, cap)
    if gamma is None:
        return {**described, "k_cap": cap, "missing": "treatment pair beyond 4 angstrom or unresolved at k_cap"}
    return {
        **described,
        "k_cap": cap,
        "gamma": gamma,
        "r_control_before_cap": None if cap == 0 else c_track["r"][cap - 1],
        "g_0.2": switch(gamma, 0.2),
        "g_s*": switch(gamma, S_STAR),
    }


def show(member):
    print("  ", {k: (round(v, 7) if isinstance(v, float) else v) for k, v in member.items()})


def summary(label, members, key):
    rows = [m for m in members if "missing" not in m]
    print(f"\n== {label}: {len(members)} members, {len(rows)} with a statistic, {len(members) - len(rows)} without")
    if not rows:
        return None, rows
    values = sorted(m[key] for m in rows)
    median = statistics.median(values)
    print(f"   median {key} {median!r}; sorted {values}")
    return median, rows


def median_of(rows, key):
    values = [m[key] for m in rows if m.get(key) is not None]
    text = f"{statistics.median(values):.7f}" if values else "none"
    return f"{text} ({len(values)} of {len(rows)} defined)"


# Set G (registered).
gain = []
for seed in SEEDS:
    for record_id, control, treated in pairs(seed, "replay"):
        if collapsed(control) and not collapsed(treated):
            gain.append(gain_member(seed, record_id, control, treated))
per_seed = tuple(sum(m["seed"] == seed for m in gain) for seed in SEEDS)
print(f"\nset G (replay, collapse in control, none in treatment): {len(gain)}, per seed {per_seed}")
if per_seed != EXPECTED_G:
    print(f"WARNING: G per seed {per_seed} differs from the {EXPECTED_G} the A/B result implies")
decoded = [m for m in gain if m["d_control"] is not None]
if decoded:
    print(f"WARNING: {len(decoded)} members of G decoded in the control (registration: all no structure)")
for m in gain:
    show(m)
m_statistic, rows = summary("G (registered)", gain, "gamma")
if m_statistic is None or 2 * len(rows) < len(gain):
    print("REGISTERED DECISION: (C) inconclusive, fewer than half of G have gamma")
elif m_statistic <= REGISTERED_BOUND:
    print(f"REGISTERED DECISION: (A) m {m_statistic!r} <= {REGISTERED_BOUND}")
else:
    print(f"REGISTERED DECISION: (B) m {m_statistic!r} > {REGISTERED_BOUND}")
if rows:
    print(f"   at most the bound {REGISTERED_BOUND}: {sum(m['gamma'] <= REGISTERED_BOUND for m in rows)} of {len(rows)}")
    print(f"   median g_0.2(gamma) {median_of(rows, 'g_0.2')}; median g_s*(gamma) {median_of(rows, 'g_s*')}")
    print(f"   mates among c-dagger: {sum(m['mates'] for m in rows)} of {len(rows)}")
    print(f"   median rho(0) {median_of(rows, 'rho_0_control')}; rho(0) equal in both arms: "
          f"{all(m['rho_0_control'] == m['rho_0_treatment'] for m in rows)}")
    print(f"   median smallest treatment rho {median_of(rows, 'rho_treatment_min')}; "
          f"median k_cap {statistics.median(m['k_cap'] for m in rows)}")

# Reported: other collapse thresholds in k_cap (same set G).
for threshold in SENSITIVITY:
    variant = []
    for seed in SEEDS:
        for record_id, control, treated in pairs(seed, "replay"):
            if collapsed(control) and not collapsed(treated):
                variant.append(gain_member(seed, record_id, control, treated, threshold))
    summary(f"G with threshold {threshold} in k_cap (reported)", variant, "gamma")


# Reported: set W, the mirror of the M1 statistic.
def mirror_member(seed, record_id, control, treated):
    head = {"seed": seed, "record": record_id[:8], "attempt": control["index"],
            "d_control": control["shortest"], "d_treatment": treated["shortest"]}
    found = tracks(control, treated)
    if isinstance(found, str):
        return {**head, "missing": found}
    name, end, c_track, t_track = found
    lam = [
        None if a is None or b is None or a <= 0.0 or b <= 0.0 else math.log(a / b)
        for a, b in zip(t_track["r"], c_track["r"], strict=True)
    ]
    described = {**head, "class": name, "mates": end["closest"]["mates"], "wyckoff": end["closest"]["wyckoff"],
                 "rho_0": defined_rho(c_track, 0)}
    if lam[LAST] is None:
        exact = c_track["r"][LAST] is not None and c_track["r"][LAST] <= 0.0
        return {**described, "missing": "Lambda_end undefined" + (" (control pair exactly coincident)" if exact else "")}
    if lam[LAST] <= 0.0:
        return {**described, "missing": f"Lambda_end {lam[LAST]:.4g} not positive"}
    found_k = None
    for k in range(LAST, -1, -1):
        if lam[k] is not None and lam[k] >= lam[LAST] / 2.0:
            found_k = k
        else:
            break
    return {
        **described,
        "lambda_end": lam[LAST],
        "k_half": found_k,
        "rho_treatment_at_half": defined_rho(t_track, found_k),
        "rho_control_at_half": defined_rho(c_track, found_k),
    }


mirror = []
for seed in SEEDS:
    for record_id, control, treated in pairs(seed, "replay"):
        if invalid(control) and not invalid(treated):
            mirror.append(mirror_member(seed, record_id, control, treated))
print(f"\nset W (replay, invalid in control, valid in treatment): {len(mirror)}")
for m in mirror:
    show(m)
summary("W, mirror of the M1 statistic (reported)", mirror, "rho_treatment_at_half")

# Reported: G on the oracle draws.
oracle = []
for seed in SEEDS:
    for record_id, control, treated in pairs(seed, "oracle"):
        if collapsed(control) and not collapsed(treated):
            oracle.append(gain_member(seed, record_id, control, treated))
print(f"\noracle draws, collapse in control, none in treatment: {len(oracle)}")
for m in oracle:
    show(m)
oracle_median, oracle_rows = summary("G on the oracle draws (reported)", oracle, "gamma")
if oracle_rows:
    print(f"   at most the bound {REGISTERED_BOUND}: {sum(m['gamma'] <= REGISTERED_BOUND for m in oracle_rows)} "
          f"of {len(oracle_rows)}")
