"""R1: CFM target-variance reduction by the gauge-aligned minibatch OT coupling.

Runs the A-F10L toys (E1, E2) with the production model options of the held-out
protocol and the same budget, under ``coupling="independent"`` (the protocol's
baseline, re-run on the current code) or ``coupling="gauge-ot-v1"``.  Only the
endpoint gates are reported: the exact oracle's velocity field is that of the
independent coupling, so rho_v is not a valid comparison for other couplings.
This is a research run, not an A-F10L claim: the amendment fixes the model
options and the protocol, not the training coupling.
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "f10_flow_oracle_2026-10-19"))
import evaluate_heldout_f10 as heldout  # noqa: E402

from crystalatlas.flow_oracle_learned import (  # noqa: E402
    LearnedToyField,
    toy_oracle,
    train_on_toy,
)
from crystalatlas.residual_gauge import ResidualActionTable  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("toy", choices=tuple(heldout.TOYS))
    parser.add_argument("output", type=Path)
    parser.add_argument("--coupling", default="gauge-ot-v1")
    parser.add_argument("--candidates", type=int, default=8)
    parser.add_argument("--seeds", default="0")
    parser.add_argument("--steps", type=int, default=heldout.STEPS)
    parser.add_argument("--batch", type=int, default=heldout.BATCH)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=tuple(heldout.DTYPES), default="float32")
    args = parser.parse_args()
    dtype = heldout.DTYPES[args.dtype]
    device = args.device
    report: dict[str, object] = {
        "schema": "crystalatlas.r1-coupling.v1",
        "toy": args.toy,
        "toy_spec": json.loads(json.dumps(heldout.TOYS[args.toy], default=str)),
        "coupling": args.coupling,
        "coupling_candidates": args.candidates,
        "steps": args.steps,
        "batch": args.batch,
        "samples": heldout.SAMPLES,
        "thresholds": heldout.THRESHOLDS,
        "device": device,
        "dtype": str(dtype),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "seeds": [],
    }
    for seed in (int(v) for v in args.seeds.split(",")):
        model = heldout.subject_model(seed, device=device, dtype=dtype, tiny=False)
        toy = heldout.build_toy(model, args.toy)
        oracle = toy_oracle(model, toy, layout_gauge=True)
        field = LearnedToyField(model, toy.cover)
        table = ResidualActionTable(
            model.layout(toy.cover), model.scientific_config.model.geometry_residual_gauge
        )
        run: dict[str, object] = {
            "seed": seed,
            "group_order": len(oracle.lattice_actions),
            "residual_table_rows": table.cardinality,
            # Proposition 2.9d rule (ii): the whole table replaces the i.i.d. candidates.
            "exact_quotient": args.coupling == "gauge-ot-v1"
            and table.cardinality <= args.candidates,
        }
        started = time.perf_counter()
        losses = train_on_toy(
            model,
            toy,
            steps=args.steps,
            batch_size=args.batch,
            learning_rate=1e-3,
            final_learning_rate=1e-5,
            seed=seed * 1000 + 7,
            coupling=args.coupling,
            coupling_candidates=args.candidates,
        )
        run["mean_loss_last_1000"] = float(np.mean(losses[-1000:]))
        run["mean_loss_first_1000"] = float(np.mean(losses[:1000]))
        run["train_seconds"] = round(time.perf_counter() - started, 1)
        model.eval()
        learned_draws = heldout.endpoint(field, oracle, device=device, dtype=dtype)
        learned = heldout.measure(*learned_draws, oracle)
        control_draws = heldout.endpoint(
            oracle.velocity, oracle, device="cpu", dtype=torch.float64
        )
        control = heldout.measure(*control_draws, oracle)
        run["learned"] = learned
        run["oracle_control"] = control
        run["verdict"] = heldout.verdict(learned)
        report["seeds"].append(run)  # type: ignore[union-attr]
        args.output.write_text(json.dumps(report, indent=1) + "\n")
        print(args.toy, args.coupling, seed, run["verdict"], flush=True)
    print(args.toy, args.coupling, "done", flush=True)


if __name__ == "__main__":
    main()
