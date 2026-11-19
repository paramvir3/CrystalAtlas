"""Per-case comparison of two CSP progress files of one protocol (same seeds).

Usage: compare_csp_samplers.py REFERENCE.jsonl CANDIDATE.jsonl
Only cases present in both files are compared.  Discrete fields must be equal;
real fields are reported by their largest absolute gap.
"""

import json
import sys


def load(path):
    lines = open(path).read().splitlines()
    header = json.loads(lines[0])
    cases = {}
    for line in lines[1:]:
        try:
            case = json.loads(line)
        except json.JSONDecodeError:
            continue
        cases[(case["record_id"], case["track"])] = case
    return header, cases


def gap(a, b):
    if a is None or b is None:
        return None if a is None and b is None else float("inf")
    return abs(a - b)


reference_header, reference = load(sys.argv[1])
candidate_header, candidate = load(sys.argv[2])
for key in ("checkpoint_sha256", "partition", "record_ids", "device", "torch"):
    assert reference_header[key] == candidate_header[key], key
for key in ("candidates", "case_seed", "likelihood_steps", "seed"):
    if reference_header["protocol"].get(key) != candidate_header["protocol"].get(key):
        print("PROTOCOL DIFFERS:", key)
# The matcher's two fingerprints hash marshal.dumps(__code__), which embeds the
# source path, so they differ between two snapshots of identical files.
PATH_BOUND = ("fingerprint", "implementation_sha256")
matchers = [
    {k: v for k, v in header["protocol"]["matcher"].items() if k not in PATH_BOUND}
    for header in (reference_header, candidate_header)
]
print("matcher contract equal apart from the path-bound fingerprints:", matchers[0] == matchers[1])
print("reference sampler:", reference_header.get("geometry_sampler", "absent (bucket-v1 code)"))
print("candidate sampler:", candidate_header.get("geometry_sampler"))
print("reference code:", reference_header["code"]["sha256"])
print("candidate code:", candidate_header["code"]["sha256"])

shared = [key for key in reference if key in candidate]
print("shared cases:", len(shared), "of", len(reference), "reference,", len(candidate), "candidate")
discrete = (
    "seed", "attempts", "valid", "distinct_tables", "match_count", "matches",
    "target_table_valid", "target_table_matches", "nu", "space_group", "request_error",
)
nested = (("first", "attempt"), ("first", "match"), ("map", "attempt"), ("map", "match"),
          ("map", "cover_members"), ("map", "error"), ("consensus", "attempt"),
          ("consensus", "match"), ("consensus", "votes"))
real = (("first", "rms"), ("map", "rms"), ("consensus", "rms"), ("map", "log_cover"))
differences = 0
largest = {"best_rms": 0.0, "map.log_geometry": 0.0}
largest.update({".".join(path): 0.0 for path in real})
sample = [0.0, 0.0]
for key in shared:
    a, b = reference[key], candidate[key]
    notes = []
    for name in discrete:
        if a.get(name) != b.get(name):
            notes.append(f"{name}: {a.get(name)} != {b.get(name)}")
    for outer, inner in nested:
        if a[outer].get(inner) != b[outer].get(inner):
            notes.append(f"{outer}.{inner}: {a[outer].get(inner)} != {b[outer].get(inner)}")
    for outer, inner in real:
        value = gap(a[outer].get(inner), b[outer].get(inner))
        if value is not None:
            largest[f"{outer}.{inner}"] = max(largest[f"{outer}.{inner}"], value)
    value = gap(a.get("best_rms"), b.get("best_rms"))
    if value is not None:
        largest["best_rms"] = max(largest["best_rms"], value)
    la, lb = a["map"].get("log_geometry"), b["map"].get("log_geometry")
    if la is not None and lb is not None and len(la) == len(lb):
        for x, y in zip(la, lb):
            if x is not None and y is not None:
                largest["map.log_geometry"] = max(largest["map.log_geometry"], abs(x - y))
    elif la != lb:
        notes.append("map.log_geometry: different layout")
    sample[0] += a["seconds"]["sample"]
    sample[1] += b["seconds"]["sample"]
    if notes:
        differences += 1
        print(key[0][:12], key[1], "; ".join(notes))
print("cases with a discrete difference:", differences)
for name, value in largest.items():
    print(f"largest gap {name}: {value:.6g}")
if shared:
    print(f"sampling seconds on shared cases: reference {sample[0]:.1f} "
          f"(mean {sample[0] / len(shared):.2f}), candidate {sample[1]:.1f} "
          f"(mean {sample[1] / len(shared):.2f}), ratio {sample[0] / max(sample[1], 1e-9):.2f}")
    stages = {}
    structurally = [0, 0]
    for key in shared:
        for stage, count in (candidate[key].get("failed_stages") or {}).items():
            stages[stage] = stages.get(stage, 0) + count
        if candidate[key].get("structurally_valid") is not None:
            structurally[0] += candidate[key]["structurally_valid"]
            structurally[1] += candidate[key]["attempts"]
    print("candidate failed stages:", dict(sorted(stages.items())))
    print("candidate structurally valid attempts:", structurally)
