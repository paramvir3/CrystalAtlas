"""Print the two samplers' trajectories of one attempt side by side."""
import json
import sys

prefix, attempt = sys.argv[1], int(sys.argv[2])
b = json.load(open(f"{prefix}.bucket-v1.json"))
m = json.load(open(f"{prefix}.mixed-v1.json"))
print("ode_steps", b["ode_steps"], "terminal_time", b["ode_terminal_time"], "solves bucket", b["solves"], "mixed", m["solves"])
gaps = []
for eb, em in zip(b["attempts"], m["attempts"]):
    import numpy as np
    zb, zm = np.array(eb["z"]), np.array(em["z"])
    hb, hm = np.array(eb["eta"]), np.array(em["eta"])
    gz = np.abs(zb - zm).max(axis=1) if zb.size else np.zeros(len(eb["times"]))
    if hb.size:
        d = np.abs(hb - hm)
        d = np.minimum(d, 1 - d) if True else d
        ge = d.max(axis=1)
    else:
        ge = np.zeros(len(eb["times"]))
    first = next((i for i, (x, y) in enumerate(zip(gz, ge)) if max(x, y) > 1e-5), None)
    gaps.append((eb["attempt"], eb["space_group"], eb["failed_stage"], em["failed_stage"], float(gz[0]), float(ge[0]), float(gz.max()), float(ge.max()), first, eb["batch_rows"], em["batch_rows"]))
print("attempt group stage_b stage_m gap0_z gap0_eta max_gap_z max_gap_eta(first step above 1e-5) rows_b rows_m")
for g in gaps:
    print(" ", g)
eb, em = b["attempts"][attempt], m["attempts"][attempt]
print("attempt", attempt, eb["space_group"], "nu", eb["nu"], eb["orbits"])
print("coordinate matrices", eb["coordinate_matrices"])
print("parameter maps", eb["parameter_maps"])
print("layout period basis", eb["layout_period_basis"])
print("stage bucket", eb["failed_stage"], "| error", eb["error"])
print("stage mixed", em["failed_stage"], "| error", em["error"], "| shortest", em.get("shortest_distance"))
print("step time | bucket eta | mixed eta | bucket z | mixed z")
for i, t in enumerate(eb["times"]):
    print(f"{i:3d} {t:.5f} | " + " ".join(f"{v:.9e}" for v in eb["eta"][i]) + " | " + " ".join(f"{v:.9e}" for v in em["eta"][i]) + " | " + " ".join(f"{v:.7f}" for v in eb["z"][i]) + " | " + " ".join(f"{v:.7f}" for v in em["z"][i]))
print("final bucket", eb["final_eta"], eb["final_z"], "terminal velocity", eb["terminal_velocity_eta"])
print("final mixed", em["final_eta"], em["final_z"], "terminal velocity", em["terminal_velocity_eta"])
for label, report in (("bucket", b), ("mixed", m)):
    for scan in report["scans"]:
        print("scan", label, "attempt", scan["attempt"], "coordinate", scan["coordinate"])
        for s in scan["steps"]:
            pts = sorted(s["points"], key=lambda p: (p[0] if p[0] < 0.5 else p[0] - 1.0))
            print(f"  step {s['step']:2d} t {s['time']:.4f} trajectory value {s['trajectory_value']:.9e} velocity {s['trajectory_velocity']:.6e}")
            print("    " + "  ".join(f"{(p[0] if p[0] < 0.5 else p[0] - 1.0):+.1e}:{p[1]:+.3e}" for p in pts))
