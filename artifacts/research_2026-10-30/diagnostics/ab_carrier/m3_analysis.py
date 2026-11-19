"""Registered analysis of measurement M3 (R1_LOG.md, "Measurement M3", 2026-11-15).

usage: m3_analysis.py M3_FILE M2_FILE STORED_REPLAY SPGLIB_DATA   (Python 3.10 or newer; 2>&1)
M3_FILE is the redraw file of the carrier control seed 20261030, M2_FILE the M2 starts file
of the same field, STORED_REPLAY its stored replay (replay_control_s20261030.json), and
SPGLIB_DATA the snapshot's src/crystalatlas/_data/spglib_v2_7_0, read only for the reported
split by shared Wyckoff axes.  Written before the M3 run; standard library only.
"""

import hashlib
import json
import math
import random
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

KINDS = ("replay", "oracle")
VALID = 0.5
NEAR = 0.1
MAX_DRAWS = 1000
FIELD = "control_s20261030"
M3_SCHEMA = "crystalatlas.research-v7-m3-redraw.v1"
M2_SCHEMA = "crystalatlas.research-v7-m2-starts.v1"
M1_SHA256 = "1f8332a87fc8c707c71ad35a745862665bc2b4c7cd19ebd71465950e0d3ca432"
M2_SHA256 = "5adfad80bc2d570fa08b53a5434549e8d5fe4c87218e7082aed48c4120220bd9"
M2_RESULT_SHA256 = "60ccf1fefafa78929b64c9b7a18cfb0bc40ef6aeb405d0430668d07d6614f844"
STORED_SHA256 = "d2d6c00cba6d07bc71b483ae0c2ad48932726c01d5ec5f775a5a1fb021719075"
M3_SHA256 = "d81edefdf1ff29095eed63b0153c1d6ba0f299f1a7bf3f9556e85ee1feee12c7"  # m3_redraw.py, registered
RATIO_START = 0.5
RATIO_COVER = 0.8
P_START = 0.01
D0_EDGES = (0.3, 0.5, 1.0, 4.0)  # bins of the used start's d0 above NEAR; then above 4
BOOTSTRAP = 2000
BOOTSTRAP_SEED = 20261115


def stop(message):
    sys.exit(f"STOP: {message}")


def invalid(value):
    return value is None or value <= VALID


def sign_test(wins, losses):
    """Exact two-sided binomial test of wins against losses at p = 1/2 (McNemar, sign test)."""

    n = wins + losses
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, k) for k in range(min(wins, losses) + 1)) / 2**n
    return min(1.0, 2.0 * tail)


def d0(start):
    """d0 of a resolved start block (inf when no distinct pair lies within 4 angstrom)."""

    closest = start["closest"]
    return math.inf if closest is None else float(closest["r"])


def share(count, total):
    return f"{count} / {total} = {count / total:.3f}" if total else f"{count} / 0"


def label(key):
    return f"{key[0][:8]} {key[1]} {key[2]}"


def shown(start):
    """d0 of a start block for a message, or 'unresolved'."""

    return d0(start) if start["resolved"] else "unresolved"


m3_path, m2_path, stored_path, spglib = (Path(value) for value in sys.argv[1:5])
m3 = json.loads(m3_path.read_text())
m2_bytes = m2_path.read_bytes()
m2 = json.loads(m2_bytes)
stored_bytes = stored_path.read_bytes()
stored = json.loads(stored_bytes)

# Validity of the inputs.
if m3.get("schema") != M3_SCHEMA or m2.get("schema") != M2_SCHEMA:
    stop(f"schemas {m3.get('schema')} and {m2.get('schema')}")
if hashlib.sha256(m2_bytes).hexdigest() != M2_RESULT_SHA256 or m3["m2_result_sha256"] != M2_RESULT_SHA256:
    stop("the M2 file is not the registered M2 result")
if hashlib.sha256(stored_bytes).hexdigest() != STORED_SHA256:
    stop("the stored replay is not the registered file")
if M3_SHA256 is None or m3["m3_script_sha256"] != M3_SHA256:
    stop(f"M3 script {m3['m3_script_sha256']} is not the registered one")
if m3["m2_script_sha256"] != M2_SHA256 or m3["m1_script_sha256"] != M1_SHA256:
    stop("the M1 or M2 script differs from the registered one")
if m3["cases"] != 50 or len(m3["records"]) != 50 or len(m2["records"]) != 50:
    stop(f"M3 holds {m3['cases']} cases and {len(m3['records'])} records; M2 {len(m2['records'])}")
if Path(m3["field_run"]).name != FIELD or Path(m2["field_run"]).name != FIELD:
    stop(f"field runs {m3['field_run']} and {m2['field_run']}")
if m3["threshold"] != NEAR or m3["max_draws"] != MAX_DRAWS or m3["radius"] != 4.0 or m3["max_reach"] != 8:
    stop("threshold, draw cap or image search differ from the registration")
for name in ("protocol_seed", "record_ids", "cover_checkpoint_sha256", "field_checkpoint_sha256",
             "candidates", "oracle_draws", "code"):
    if m3[name] != m2[name] or m3[name] != stored[name]:
        stop(f"M3, M2 and the stored replay differ in {name}")
if m3["replay_module_sha256"] != m2["replay_module_sha256"]:
    stop("the replay module (oracle path) differs from M2's")
if any(attempt["traced"] for case in m3["records"] for kind in KINDS for attempt in case[kind]):
    stop("M3 traced an attempt")
print(f"M3: field {FIELD}, m3 script {m3['m3_script_sha256'][:12]}, m2 {m3['m2_script_sha256'][:12]}, "
      f"m1 {m3['m1_script_sha256'][:12]}; single-member fallback solves {m3['bucket_solves']}; "
      f"failed padded solves {len(m3['failed_padded_solves'])}; start blocks changed by the final "
      f"recomputation {m3['block_changes']}")
print(f"script checks: {m3['checks']}")
for item in m3["failed_padded_solves"]:
    print(f"   failed padded solve: {item}")

stored_replay = {case["record_id"]: case for case in stored["replay_cases"]}
stored_oracle = {case["record_id"]: case for case in stored["oracle_cases"]}


def outcome(kind, case, index):
    """(shortest distance, matched) of one attempt in a replay or oracle case."""

    if kind == "replay":
        return case["shortest_distances"][index], index in case["matches"]
    return case["shortest_distances"][index], bool(case["matches"][index])


# Per attempt: the M2 start, the M3 record and both outcomes, for every record.
rows = {}
case_near = {}
for new, old in zip(m3["records"], m2["records"], strict=True):
    record_id = new["record_id"]
    if record_id != old["record_id"]:
        stop("record order differs from M2")
    treated = {"replay": new["replay_case"], "oracle": new["oracle_case"]}
    original = {"replay": stored_replay[record_id], "oracle": stored_oracle[record_id]}
    seeds = {new["seed"], old["seed"], treated["replay"]["seed"], treated["oracle"]["seed"],
             original["replay"]["seed"], original["oracle"]["seed"]}
    if len(seeds) != 1:
        stop(f"{record_id[:8]}: request seeds differ {sorted(seeds)}")
    case_near[record_id] = False
    for kind in KINDS:
        expected = m3["candidates"] if kind == "replay" else m3["oracle_draws"]
        if len(new[kind]) != expected or len(old[kind]) != expected:
            stop(f"{record_id[:8]} {kind}: attempt count")
        for position, (a, b) in enumerate(zip(new[kind], old[kind], strict=True)):
            key = (record_id, kind, position)
            if a["index"] != position or b["index"] != position:
                stop(f"{label(key)}: index")
            if not b["solved"] or not b["start"]["resolved"]:
                stop(f"{label(key)}: M2 holds no defined start")
            old_outcome = outcome(kind, original[kind], position)
            if b["shortest"] != old_outcome[0]:
                stop(f"{label(key)}: M2 and the stored replay differ")
            near = d0(b["start"]) <= NEAR
            case_near[record_id] |= near
            rows[key] = {"m2": b, "m3": a, "near": near, "old": old_outcome,
                         "new": outcome(kind, treated[kind], position)}

# Exact checks of the treatment (registration), over every record: each is a stop.
unsolved_near, unsolved_far, capped = [], [], []
for key, row in rows.items():
    a, b = row["m3"], row["m2"]
    if not a["solved"]:
        (unsolved_near if row["near"] else unsolved_far).append(key)
        continue
    if any(a[name] != b[name] for name in ("space_group", "nu", "orbits")):
        stop(f"{label(key)}: the cover differs from M2's")
    redraw = a["redraw"]
    redrawn = redraw["draws"] > 1
    if redrawn != row["near"] or redraw["near_first"] != row["near"]:
        stop(f"{label(key)}: redrawn {redrawn}, M2 near {row['near']}")
    if not redraw["start_equals_accepted"]:
        stop(f"{label(key)}: step 0 of the trajectory is not the accepted start")
    if not redrawn:
        if redraw["near_used"]:
            stop(f"{label(key)}: an attempt that was not redrawn is marked near")
        if a["start"] != b["start"]:
            stop(f"{label(key)}: a start that was not redrawn differs from M2's")
        continue
    if a["first_start"] != b["start"]:
        stop(f"{label(key)}: the first draw differs from M2's start")
    if a["start"]["resolved"]:
        used_near = d0(a["start"]) <= NEAR
    else:
        # An unresolved block was decided by the exact search at 0.1 angstrom.
        search = redraw.get("threshold_search_d0")
        if not isinstance(search, list) or len(search) != 2:
            stop(f"{label(key)}: an unresolved used start has no exact-search d0 pair")
        used_near = search[1] is not None and search[1] <= NEAR
    if redraw["near_used"] != used_near:
        found = shown(a["start"]) if a["start"]["resolved"] else f"unresolved, exact search {search[1]}"
        stop(f"{label(key)}: near_used {redraw['near_used']} but the used start has d0 {found}")
    if redraw["near_used"]:
        if redraw["draws"] != MAX_DRAWS:
            stop(f"{label(key)}: marked near after {redraw['draws']} draws, not {MAX_DRAWS}")
        capped.append(key)
    elif redraw["draws"] > MAX_DRAWS:
        stop(f"{label(key)}: the draw cap was exceeded")
print("every redrawn attempt is an M2 near start and every solved M2 near start is redrawn; "
      "every first draw equals M2's start; every start not redrawn equals M2's; covers equal M2's")
print(f"near starts unsolved in M3 (their redrawn start failed its padded solve; the fallback "
      f"integrated the first draw): {[label(k) for k in unsolved_near]}")
print(f"other attempts unsolved in M3: {[label(k) for k in unsolved_far]}")
print(f"redrawn but still near at the draw cap: {[label(k) for k in capped]}")
if (unsolved_near or unsolved_far) and not m3["failed_padded_solves"]:
    stop("attempts are unsolved in M3 although no padded solve failed")
redrawn_failures = {tuple(item["case"]) for item in m3["failed_padded_solves"]
                    if any(count > 1 for count in item["draws"])}
for key in unsolved_near:
    if (key[0], key[1]) not in redrawn_failures:
        stop(f"{label(key)}: unsolved, but no padded solve of its case with a redrawn row failed")
print(f"unsolved attempts {len(unsolved_near) + len(unsolved_far)}, single-member fallback solves "
      f"{m3['bucket_solves']}")
for item in m3["failed_padded_solves"]:
    if all(count == 1 for count in item["draws"]):
        print(f"   a failed padded solve holds no redrawn row: {item['case']}, space groups {item['space_groups']}")

# Cases used: a case without any M2 near start must reproduce (registration).
left_out = sorted(record_id for record_id, near in case_near.items()
                  if not near and not next(c for c in m3["records"] if c["record_id"] == record_id)["reproduced"])
print(f"cases used {50 - len(left_out)} of 50; left out (no near start and not reproduced): "
      f"{[record_id[:8] for record_id in left_out]}")
rows = {key: row for key, row in rows.items() if key[0] not in left_out}
# The treated outcome of a near start whose redraw failed its padded solve is a failure.
for key in unsolved_near:
    rows[key]["new"] = (None, False)

# Attempts that were not redrawn: their outcomes against the stored replay (reported).
kept = [row for key, row in rows.items() if row["m3"]["solved"] and not row["near"]]
exact = sum(row["new"] != row["old"] for row in kept)
classes = sum((invalid(row["new"][0]), row["new"][1]) != (invalid(row["old"][0]), row["old"][1]) for row in kept)
gaps = [abs(row["new"][0] - row["old"][0]) for row in kept
        if row["new"][0] is not None and row["old"][0] is not None]
print(f"attempts not redrawn: {len(kept)}; outcome differs from the stored replay: {exact}; "
      f"validity or match differs: {classes}; largest |shortest difference|: {max(gaps, default=0.0):.3g} angstrom")

# Registered decision: the near replay states, paired.
near = {key: row for key, row in rows.items() if key[1] == "replay" and row["near"]}
paired = {key: row for key, row in near.items() if key not in capped}
both = sum(invalid(r["old"][0]) and invalid(r["new"][0]) for r in paired.values())
fixed = sum(invalid(r["old"][0]) and not invalid(r["new"][0]) for r in paired.values())
broken = sum(not invalid(r["old"][0]) and invalid(r["new"][0]) for r in paired.values())
neither = len(paired) - both - fixed - broken
n_old, n_new = both + fixed, both + broken
p_value = sign_test(fixed, broken)
failed = [key for key in unsolved_near if key in paired]
print(f"\nnear replay states: {len(near)}; paired {len(paired)}; left out at the draw cap "
      f"{len(near) - len(paired)}; redraw failed its padded solve (counted invalid) {len(failed)}")
print(f"   invalid first draw and redraw: {both}; first draw only: {fixed}; redraw only: {broken}; "
      f"neither: {neither}")
print(f"   invalid: first draw {share(n_old, len(paired))}; redraw {share(n_new, len(paired))}")
capped_near = [key for key in near if key in capped]
capped_old_invalid = sum(invalid(near[key]["old"][0]) for key in capped_near)
if failed and n_old:
    # Counted valid, a failed state with an invalid first draw becomes discordant (fixed) and
    # one with a valid first draw stops being discordant (broken).
    failed_old_invalid = sum(invalid(near[key]["old"][0]) for key in failed)
    p_valid = sign_test(fixed + failed_old_invalid, broken - (len(failed) - failed_old_invalid))
    print(f"   R with the failed solves counted valid: {n_new - len(failed)} / {n_old} = "
          f"{(n_new - len(failed)) / n_old:.4f}; McNemar p {p_valid:.3g}")
if capped_near:
    old_with = n_old + capped_old_invalid
    new_with = n_new + len(capped_near)
    p_with = sign_test(fixed, broken + len(capped_near) - capped_old_invalid)
    print(f"   capped states ({len(capped_near)}, {capped_old_invalid} with an invalid first draw), counted "
          f"invalid redraws (failed solves stay invalid): R = {new_with} / {old_with}"
          + (f" = {new_with / old_with:.4f}" if old_with else "") + f"; McNemar p {p_with:.3g}")
    p_valid_capped = sign_test(fixed + capped_old_invalid, broken)
    print(f"   capped states counted valid redraws (failed solves stay invalid): R = {n_new} / {old_with}"
          + (f" = {n_new / old_with:.4f}" if old_with else "") + f"; McNemar p {p_valid_capped:.3g}")
ratio = None
if n_old == 0:
    print("REGISTERED DECISION: undefined (no invalid first draw among the paired near states)")
else:
    ratio = n_new / n_old
    print(f"   R = {n_new} / {n_old} = {ratio:.4f}; exact McNemar two-sided p = {p_value:.3g} "
          f"(discordant {fixed} against {broken})")
    if ratio <= RATIO_START and p_value <= P_START:
        decision = "(S) the near start causes most of its invalid outcomes"
    elif ratio >= RATIO_COVER:
        decision = "(C) redrawing the start does not remove the invalid outcomes"
    else:
        decision = "(I) intermediate"
    print(f"REGISTERED DECISION: {decision}")

# Reported, not part of the decision.
print("\n== reported")
per_record = defaultdict(lambda: [0, 0])
for key, r in paired.items():
    per_record[key[0]][0] += invalid(r["old"][0])
    per_record[key[0]][1] += invalid(r["new"][0])
if ratio is not None:
    generator = random.Random(BOOTSTRAP_SEED)
    names = sorted(per_record)
    ratios = []
    for _ in range(BOOTSTRAP):
        draw = generator.choices(names, k=len(names))
        old_total = sum(per_record[name][0] for name in draw)
        if old_total:
            ratios.append(sum(per_record[name][1] for name in draw) / old_total)
    ratios.sort()
    print(f"R, record bootstrap ({BOOTSTRAP} resamples of the {len(names)} records, random.Random("
          f"{BOOTSTRAP_SEED}).choices, {len(ratios)} with an invalid first draw): 95 % "
          f"{ratios[int(0.025 * len(ratios))]:.3f} to {ratios[int(0.975 * len(ratios)) - 1]:.3f}")
matched_old = sum(r["old"][1] for r in paired.values())
matched_new = sum(r["new"][1] for r in paired.values())
print(f"near replay states matched: first draw {matched_old}, redraw {matched_new} of {len(paired)}")
better = sum(o > n for o, n in per_record.values())
worse = sum(o < n for o, n in per_record.values())
print(f"records holding paired near states: {len(per_record)}; fewer invalid after the redraw in {better}, "
      f"more in {worse}, equal in {len(per_record) - better - worse}; sign test p = {sign_test(better, worse):.3g}")
far_same = [r for k, r in rows.items() if k[1] == "replay" and not r["near"] and k[0] in per_record]
print(f"reference: far replay starts of the same records, first draw invalid "
      f"{share(sum(invalid(r['old'][0]) for r in far_same), len(far_same))}")


def run_totals(kind, side):
    values = [r[side] for k, r in rows.items() if k[1] == kind]
    by_record = defaultdict(int)
    for k, r in rows.items():
        if k[1] == kind:
            by_record[k[0]] += r[side][1]
    return {
        "attempts": len(values),
        "without structure": sum(v[0] is None for v in values),
        "structurally valid": sum(not invalid(v[0]) for v in values),
        "matched": sum(v[1] for v in values),
        "records with a match": sum(count > 0 for count in by_record.values()),
    }


for kind in KINDS:
    print(f"whole {kind} arm, stored replay against M3: {run_totals(kind, 'old')} against {run_totals(kind, 'new')}")
first_old = sum(r["old"][1] for k, r in rows.items() if k[1] == "oracle" and k[2] == 0)
first_new = sum(r["new"][1] for k, r in rows.items() if k[1] == "oracle" and k[2] == 0)
print(f"oracle first-draw matches: {first_old} against {first_new}")
used_records = {k[0] for k in rows}
for kind, source, name in (("replay", stored_replay, "replay_case"), ("oracle", stored_oracle, "oracle_case")):
    stages_old, stages_new = Counter(), Counter()
    for record_id in used_records:
        stages_old.update(source[record_id]["failed_stages"])
    for case in m3["records"]:
        if case["record_id"] in used_records:
            stages_new.update(case[name]["failed_stages"])
    print(f"{kind} failed stages: {dict(sorted(stages_old.items()))} against {dict(sorted(stages_new.items()))}"
          + (" (a failed redrawn solve appears here with its fallback outcome)" if unsolved_near else ""))
oracle_near = {k: r for k, r in rows.items() if k[1] == "oracle" and r["near"] and k not in capped}
o_fixed = sum(invalid(r["old"][0]) and not invalid(r["new"][0]) for r in oracle_near.values())
o_broken = sum(not invalid(r["old"][0]) and invalid(r["new"][0]) for r in oracle_near.values())
m_gain = sum(r["new"][1] and not r["old"][1] for r in oracle_near.values())
m_loss = sum(r["old"][1] and not r["new"][1] for r in oracle_near.values())
print(f"near oracle draws: {len(oracle_near)}; invalid {sum(invalid(r['old'][0]) for r in oracle_near.values())} "
      f"against {sum(invalid(r['new'][0]) for r in oracle_near.values())} (McNemar p {sign_test(o_fixed, o_broken):.3g}); "
      f"matched {sum(r['old'][1] for r in oracle_near.values())} against {sum(r['new'][1] for r in oracle_near.values())} "
      f"(gained {m_gain}, lost {m_loss}, McNemar p {sign_test(m_gain, m_loss):.3g})")
# Prediction from M2 alone: every oracle draw of a record shares the target cover, so a near
# draw's redraw is predicted by the far draws of the same record (first-draw outcomes).
predicted_invalid = predicted_match = 0.0
for record_id in {k[0] for k in oracle_near}:
    far = [r for k, r in rows.items() if k[0] == record_id and k[1] == "oracle" and not r["near"]]
    count = sum(1 for k in oracle_near if k[0] == record_id)
    if far:
        predicted_invalid += count * sum(invalid(r["old"][0]) for r in far) / len(far)
        predicted_match += count * sum(r["old"][1] for r in far) / len(far)
print(f"   prediction from the far draws of the same records: invalid {predicted_invalid:.1f}, "
      f"matched {predicted_match:.1f}")

# Shared one-parameter Wyckoff axis in the cover (the M2 skeptic's definition).
symbol_to_number, first_hall = {}, {}
for line in (spglib / "spg.csv").read_text().splitlines():
    fields = line.split(",")
    symbol_to_number[fields[7].split(" = ")[0].replace(" ", "")] = int(fields[4])
    first_hall.setdefault(int(fields[4]), int(fields[0]))
wyckoff = defaultdict(dict)
hall = None
for raw in (spglib / "Wyckoff.csv").read_text().splitlines():
    line = raw.strip()
    if line == "end of data":
        break
    fields = line.split(":")
    if fields[0].isdigit():
        hall = int(fields[0])
    elif len(fields) > 3 and fields[2].isdigit():
        wyckoff[hall][fields[3].strip()] = len({ch for ch in fields[5] if ch in "xyz"})


def shared_axis(attempt):
    labels = Counter(orbit[0] for orbit in attempt["orbits"])
    group = first_hall[symbol_to_number[attempt["space_group"].replace(" ", "")]]
    return any(n > 1 and wyckoff[group][name.lstrip("0123456789")] == 1 for name, n in labels.items())


for flag in (True, False):
    part = [r for r in paired.values() if shared_axis(r["m2"]) == flag]
    print(f"near replay states, shared one-parameter axis {flag}: invalid first draw "
          f"{share(sum(invalid(r['old'][0]) for r in part), len(part))}, redraw "
          f"{share(sum(invalid(r['new'][0]) for r in part), len(part))}")

redrawn = [r for r in rows.values() if r["m3"]["solved"] and r["m3"]["redraw"]["draws"] > 1]
draws = [r["m3"]["redraw"]["draws"] for r in redrawn]
if draws:
    print(f"draws per redrawn attempt ({len(draws)}): median {statistics.median(draws)}, mean "
          f"{statistics.mean(draws):.2f}, max {max(draws)}; 2: {sum(d == 2 for d in draws)}, 3 to 5: "
          f"{sum(3 <= d <= 5 for d in draws)}, 6 to 10: {sum(6 <= d <= 10 for d in draws)}, above 10: "
          f"{sum(d > 10 for d in draws)}")
solved_pairs = [r for k, r in paired.items() if k not in unsolved_near]
unresolved = [r for r in solved_pairs if not r["m3"]["start"]["resolved"]]
print(f"used start unresolved at 4 angstrom (decided by the exact search at 0.1): {len(unresolved)}, "
      f"redraw invalid {sum(invalid(r['new'][0]) for r in unresolved)}")
low = NEAR
for high in (*D0_EDGES, math.inf):
    part = [r for r in solved_pairs if r["m3"]["start"]["resolved"] and low < d0(r["m3"]["start"]) <= high]
    name = f"({low}, {high}]" if math.isfinite(high) else f"above {low}"
    print(f"used start d0 {name}: near replay states {len(part)}, redraw invalid "
          f"{sum(invalid(r['new'][0]) for r in part)}")
    low = high
ratios = [r["m3"]["start"]["volume_per_atom"] / r["m2"]["start"]["volume_per_atom"] for r in solved_pairs]
if ratios:
    print(f"volume per atom, used start over first draw: median {statistics.median(ratios):.3f}")
persistent = [r for r in solved_pairs if invalid(r["new"][0])]
same = sum(
    r["m3"]["end"] is not None and r["m3"]["end"]["closest"] is not None
    and r["m2"]["start"]["closest"] is not None
    and r["m3"]["end"]["closest"]["class"] == r["m2"]["start"]["closest"]["class"]
    for r in persistent
)
print(f"invalid redraws with a solved start: {len(persistent)}; closest class at the end equals the "
      f"closest class of the first draw in {same}")
