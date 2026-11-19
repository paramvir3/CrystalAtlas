"""Verdict of the registered A/B of the periodic element prior (R1_LOG.md, 2026-11-10).

Reads only stored outputs of the queue; no model is evaluated here.
"""
import json
import math
import statistics
import sys
from pathlib import Path

root = Path(sys.argv[1])
seeds = (20261030, 20261031, 20261032)
arms = ("control", "periodic")
factors = ("group", "nu", "row", "total")


def record_total(record):
    return record["group_nll"] + record["nu_nll"] + record["row_nll"]


selection = {}
diagnostic = {}
for arm in arms:
    for seed in seeds:
        selection[arm, seed] = json.loads((root / f"{arm}_s{seed}" / "selection-cover.json").read_text())
        diagnostic[arm, seed] = json.loads((root / f"{arm}_s{seed}_cover_factors.json").read_text())

print("== selection (validation cover NLL of the selected checkpoint)")
for arm in arms:
    for seed in seeds:
        s = selection[arm, seed]
        assert s["metric"] == "cover_nll" and s["partition"] == "validation" and s["restored"] is True
        best = min(s["history"], key=lambda h: h["cover_nll"])
        assert best == s["selected"], (best, s["selected"])
        last = max(h["phase_epoch"] for h in s["history"])
        d = diagnostic[arm, seed]
        assert d["checkpoint_label"] == "after-cover"
        validation_total = statistics.fmean(record_total(r) for r in d["records"]["validation"])
        print(
            f"{arm:8s} {seed} selected {s['selected']['cover_nll']:.6f} "
            f"epoch {s['selected']['phase_epoch']:2d} {s['selected']['weights']:3s} "
            f"evaluations {len(s['history'])} last epoch {last} "
            f"diagnostic validation total {validation_total:.6f} "
            f"gap {validation_total - s['selected']['cover_nll']:+.2e}"
        )

print("== P1")
p1 = {arm: statistics.fmean(selection[arm, seed]["selected"]["cover_nll"] for seed in seeds) for arm in arms}
per_seed = [selection["periodic", seed]["selected"]["cover_nll"] - selection["control", seed]["selected"]["cover_nll"] for seed in seeds]
print("mean control", round(p1["control"], 6), "mean periodic", round(p1["periodic"], 6))
print("treatment minus control", round(p1["periodic"] - p1["control"], 6), "per seed", [round(x, 6) for x in per_seed])
print("P1 holds (at least 0.5 nats lower):", p1["periodic"] - p1["control"] <= -0.5)

for partition in ("test", "validation"):
    print(f"== {partition}: per-record difference treatment minus control, averaged over seeds")
    ids = [r["record_id"] for r in diagnostic["control", seeds[0]]["records"][partition]]
    assert len(ids) == len(set(ids)) == 50
    table = {}
    for key, d in diagnostic.items():
        rows = d["records"][partition]
        assert [r["record_id"] for r in rows] == ids, key
        table[key] = rows
        summary = d["summary"][partition]["nll"]
        for factor in ("group", "nu", "row"):
            assert abs(statistics.fmean(r[f"{factor}_nll"] for r in rows) - summary[factor]) < 1e-9
        assert abs(statistics.fmean(record_total(r) for r in rows) - summary["total"]) < 1e-9
    for factor in factors:
        def value(r):
            return record_total(r) if factor == "total" else r[f"{factor}_nll"]
        differences = [
            statistics.fmean(value(table["periodic", seed][i]) - value(table["control", seed][i]) for seed in seeds)
            for i in range(50)
        ]
        mean = statistics.fmean(differences)
        standard_error = statistics.stdev(differences) / math.sqrt(50)
        arm_means = {
            arm: statistics.fmean(statistics.fmean(value(r) for r in table[arm, seed]) for seed in seeds)
            for arm in arms
        }
        seed_differences = [
            statistics.fmean(value(table["periodic", seed][i]) - value(table["control", seed][i]) for i in range(50))
            for seed in seeds
        ]
        print(
            f"{factor:5s} control {arm_means['control']:.4f} periodic {arm_means['periodic']:.4f} "
            f"difference {mean:+.4f} standard error {standard_error:.4f} "
            f"two standard errors {2 * standard_error:.4f} ratio {mean / standard_error:+.2f} "
            f"per seed {[round(x, 4) for x in seed_differences]}"
        )
        if partition == "test" and factor == "total":
            print("P2 holds (negative, magnitude above two standard errors):", mean < 0 and abs(mean) > 2 * standard_error)

print("== ranks on test (mean over seeds)")
for arm in arms:
    top1 = statistics.fmean(diagnostic[arm, seed]["summary"]["test"]["group_top"]["1"] for seed in seeds)
    top5 = statistics.fmean(diagnostic[arm, seed]["summary"]["test"]["group_top"]["5"] for seed in seeds)
    nu1 = statistics.fmean(diagnostic[arm, seed]["summary"]["test"]["nu_top1"] for seed in seeds)
    rows_ok = statistics.fmean(diagnostic[arm, seed]["summary"]["test"]["rows_all_maximal"] for seed in seeds)
    errors = statistics.fmean(diagnostic[arm, seed]["summary"]["test"]["mean_row_errors"] for seed in seeds)
    print(f"{arm:8s} group top1 {top1:.3f} top5 {top5:.3f} nu top1 {nu1:.3f} rows all maximal {rows_ok:.3f} mean row errors {errors:.3f}")
