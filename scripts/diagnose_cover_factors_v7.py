"""Where the cover negative log-likelihood of a ``train_research_v7`` run sits.

The cover selector is the chain

    p(cover | c) = p(nu | c) p(G | c, nu) prod_r p(a_r | c, nu, G, a_<r),

so the teacher-forced cover loss of a structure is the sum of three terms.  For
each requested partition this diagnostic reports their means, the same sums for
the policy that is uniform on every exact legal action set, and how often the
target is a maximum of its decision (``crystalatlas.training_objectives``,
``CoverFactorTerms``).  It reads one labelled checkpoint and writes one new
JSON; nothing is trained or sampled.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch
import train_research_v7 as runner
from evaluate_csp_research_v7 import (
    add_run_arguments,
    configure_torch,
    load_run,
    open_research_store,
    partition_record_ids,
    progress,
)

from crystalatlas.errors import DomainError, InvariantViolation
from crystalatlas.trainer import TrainingPhase
from crystalatlas.training_data import StoredTrainingDataset
from crystalatlas.training_objectives import (
    CrystalAtlasTrainingObjective,
    ExactBatchCrystallographicAudit,
    ObjectiveDataMeasure,
)

SCHEMA = "crystalatlas.research-v7-cover-factors.v1"
PARTITIONS = ("train", "validation", "test")
GROUP_TOP = (1, 5, 10, 20)
FACTORS = ("nu", "group", "row")


def record_rows(objective, audit, selector, batches) -> list[dict[str, object]]:  # type: ignore[no-untyped-def]
    """One row of factor terms per record of the batches, in record-id order."""

    rows: dict[str, dict[str, object]] = {}
    for batch in batches:
        audit(batch, torch.float64).require_passed()
        terms = objective.cover_factor_terms(selector, batch)
        topology = batch.bucket_key.topology
        for index, record_id in enumerate(batch.record_ids):
            if record_id in rows:
                raise InvariantViolation("a record occurs twice in the diagnostic view")
            row: dict[str, object] = {
                "record_id": record_id,
                "space_group": topology.space_group,
                "nu": topology.nu,
                "rows": len(topology.row_labels),
                "occupied_rows": sum(1 for action in topology.selected_actions if any(action)),
            }
            for factor in FACTORS:
                for kind in ("nll", "reference"):
                    value = float(getattr(terms, f"{factor}_{kind}")[index].item())
                    if not math.isfinite(value):
                        raise DomainError(f"nonfinite cover {factor} {kind}")
                    row[f"{factor}_{kind}"] = value
            row["nu_rank"] = int(terms.nu_rank[index].item())
            row["group_rank"] = int(terms.group_rank[index].item())
            row["row_errors"] = int(terms.row_errors[index].item())
            rows[record_id] = row
    return [rows[record_id] for record_id in sorted(rows)]


def summarize(rows: list[dict[str, object]]) -> dict[str, object]:
    """Partition means of the per-record rows (every record has weight one)."""

    count = len(rows)
    if count == 0:
        raise DomainError("a partition has no records")

    def mean(key: str) -> float:
        return math.fsum(float(row[key]) for row in rows) / count  # type: ignore[arg-type]

    def fraction(predicate) -> float:  # type: ignore[no-untyped-def]
        return sum(1 for row in rows if predicate(row)) / count

    nll = {factor: mean(f"{factor}_nll") for factor in FACTORS}
    reference = {factor: mean(f"{factor}_reference") for factor in FACTORS}
    return {
        "records": count,
        "nll": {**nll, "total": math.fsum(nll.values())},
        "uniform_reference": {**reference, "total": math.fsum(reference.values())},
        "nu_top1": fraction(lambda row: row["nu_rank"] == 0),
        "group_top": {
            str(k): fraction(lambda row, k=k: row["group_rank"] < k) for k in GROUP_TOP
        },
        "rows_all_maximal": fraction(lambda row: row["row_errors"] == 0),
        "mean_row_errors": mean("row_errors"),
        "mean_rows": mean("rows"),
        "mean_occupied_rows": mean("occupied_rows"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_run_arguments(parser)
    parser.add_argument("--output", type=Path, required=True, help="result JSON (new file)")
    parser.add_argument("--partitions", default=",".join(PARTITIONS))
    args = parser.parse_args()
    partitions = tuple(args.partitions.split(","))
    if not partitions or any(name not in PARTITIONS for name in partitions):
        raise DomainError(f"--partitions takes a comma list from {PARTITIONS}")
    if len(set(partitions)) != len(partitions):
        raise DomainError("--partitions repeats a partition")

    device = configure_torch(args.device, args.cpu_threads, 0)
    loaded = load_run(
        args.run,
        args.checkpoint_label,
        certification=args.certification,
        certification_fingerprint=args.certification_fingerprint,
        device=device,
    )
    model = loaded.model
    model.eval()
    seed = int(loaded.arguments["seed"])  # type: ignore[call-overload]
    batch_size = int(loaded.arguments["batch_size"])  # type: ignore[call-overload]
    audit = ExactBatchCrystallographicAudit(model, memoize=True)
    records: dict[str, list[dict[str, object]]] = {}
    with open_research_store(args) as store, torch.no_grad():
        for partition in partitions:
            dataset = StoredTrainingDataset(
                store,
                model.composition_codec,
                loaded.catalogue,
                domain=model.scientific_config.domain,
                record_ids=partition_record_ids(loaded, store, partition),
            )
            progress(f"{partition}: dataset view of {len(dataset)} records built")
            objective = CrystalAtlasTrainingObjective(
                model, ObjectiveDataMeasure.from_dataset(dataset)
            )
            records[partition] = record_rows(
                objective,
                audit,
                model.cover_selector,
                runner._sampler(dataset, TrainingPhase.COVER, seed, batch_size),
            )
            progress(f"{partition}: factor terms done")
    result = {
        "schema": SCHEMA,
        "run": str(loaded.directory),
        "checkpoint_label": loaded.label,
        "checkpoint_sha256": loaded.checkpoint_sha256,
        "group_top": list(GROUP_TOP),
        "summary": {partition: summarize(rows) for partition, rows in records.items()},
        "records": records,
    }
    runner._write_new(args.output.absolute(), result)
    for partition, summary in result["summary"].items():  # type: ignore[attr-defined]
        print(partition, json.dumps(summary, sort_keys=True, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
