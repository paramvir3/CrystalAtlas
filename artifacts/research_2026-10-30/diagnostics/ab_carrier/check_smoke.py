"""Run 2 replayed through its own field must equal the stored fused cases."""
import json
import sys

replay = json.load(open(sys.argv[1]))
stored = {}
for line in open(sys.argv[2]).read().splitlines()[1:]:
    case = json.loads(line)
    stored[case["record_id"]] = case
keys = ("seed", "attempts", "valid", "failed_stages", "matches", "match_count", "first",
        "consensus", "best_rms", "distinct_tables", "target_table_valid",
        "target_table_matches", "shortest_distances", "structurally_valid")
bad = 0
for case in replay["replay_cases"]:
    old = stored[case["record_id"]]
    for key in keys:
        if case[key] != old[key]:
            bad += 1
            print("MISMATCH", case["record_id"][:12], key, old[key], case[key])
print("cases", len(replay["replay_cases"]), "mismatching fields", bad)
print("model fields differing", replay["model_fields_differing"])
sys.exit(1 if bad or replay["model_fields_differing"] else 0)
