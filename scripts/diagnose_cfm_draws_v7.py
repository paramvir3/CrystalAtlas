"""Monte Carlo spread of the CFM regression metric of a ``train_research_v7`` run.

The CFM regression of a model state is an expectation over the data, the time
and the base point.  The runner estimates it on a fixed dataset view with fixed
draws: one draw in every phase-end evaluation (``<label>-evaluation.json``) and
``--validation-draws`` draws in the periodic validation that selects the
geometry checkpoint (``selection-geometry.json``).  Two numbers of one state
therefore differ by Monte Carlo error, and only numbers with the same draws
compare two states.

This diagnostic re-evaluates one labelled checkpoint with the runner's own
``evaluate_losses`` and reports, for each requested partition,

* the phase-end estimate (all factor losses, one draw), which reproduces the
  stored evaluation of the checkpoint;
* the geometry-only estimates with ``1..D`` draws.  Draw ``d`` is seeded with
  ``seed + 7 + d`` on one fixed batch order, so the estimate with ``k`` draws is
  the mean of the first ``k`` per-draw values; the entry ``validation_draws``
  reproduces the selected value when the selected weights are in the checkpoint;
* the per-draw values ``v_k = k m_k - (k - 1) m_(k-1)``, their mean, their
  sample standard deviation and the standard error of the mean.

The spread is over time and base draws on fixed records; it is not the sampling
error of the finite partition.  Nothing is trained or sampled; one new JSON is
written.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import train_research_v7 as runner
from evaluate_csp_research_v7 import (
    add_run_arguments,
    configure_torch,
    load_run,
    open_research_store,
    partition_record_ids,
    progress,
)

from crystalatlas.errors import DomainError
from crystalatlas.trainer import TrainingPhase
from crystalatlas.training_data import StoredTrainingDataset
from crystalatlas.training_objectives import (
    CrystalAtlasTrainingObjective,
    ExactBatchCrystallographicAudit,
    ObjectiveDataMeasure,
)

SCHEMA = "crystalatlas.research-v7-cfm-draws.v1"
PARTITIONS = ("train", "validation", "test")
METRIC = "cfm_regression"


def draw_statistics(cumulative_means: list[float]) -> dict[str, object]:
    """Per-draw values and their spread from the means over ``1..D`` draws."""

    if not cumulative_means:
        raise DomainError("at least one draw is required")
    values = [
        (k + 1) * mean - k * (cumulative_means[k - 1] if k else 0.0)
        for k, mean in enumerate(cumulative_means)
    ]
    count = len(values)
    mean = math.fsum(values) / count
    result: dict[str, object] = {"per_draw": values, "draws": count, "mean": mean}
    if count > 1:
        variance = math.fsum((value - mean) ** 2 for value in values) / (count - 1)
        result["draw_standard_deviation"] = math.sqrt(variance)
        result["standard_error_of_mean"] = math.sqrt(variance / count)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_run_arguments(parser)
    parser.add_argument("--output", type=Path, required=True, help="result JSON (new file)")
    parser.add_argument("--partitions", default="validation")
    parser.add_argument("--draws", type=int, default=8)
    args = parser.parse_args()
    partitions = tuple(args.partitions.split(","))
    if not partitions or any(name not in PARTITIONS for name in partitions):
        raise DomainError(f"--partitions takes a comma list from {PARTITIONS}")
    if len(set(partitions)) != len(partitions):
        raise DomainError("--partitions repeats a partition")
    if args.draws < 1:
        raise DomainError("--draws must be >= 1")

    device = configure_torch(args.device, args.cpu_threads, 0)
    loaded = load_run(
        args.run,
        args.checkpoint_label,
        certification=args.certification,
        certification_fingerprint=args.certification_fingerprint,
        device=device,
    )
    model = loaded.model
    seed = int(loaded.arguments["seed"])  # type: ignore[call-overload]
    batch_size = int(loaded.arguments["batch_size"])  # type: ignore[call-overload]
    dequantization = runner.research_dequantization(
        str(loaded.arguments.get("options", "default"))
    )
    audit = ExactBatchCrystallographicAudit(model, memoize=True)
    stored = json.loads((loaded.directory / f"{loaded.label}-evaluation.json").read_text())
    selection_path = loaded.directory / "selection-geometry.json"
    selection = json.loads(selection_path.read_text()) if selection_path.exists() else None
    results: dict[str, object] = {}
    with open_research_store(args) as store:
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
                model,
                ObjectiveDataMeasure.from_dataset(dataset),
                dequantization=dequantization,
            )
            # The same call as the runner's phase-end evaluation.
            phase_end = runner.evaluate_losses(
                model, dataset, objective, audit, seed=seed + 7, batch_size=batch_size
            )
            progress(f"{partition}: phase-end estimate done")
            # The same call as the runner's periodic geometry validation.
            cumulative = [
                float(
                    runner.evaluate_losses(
                        model,
                        dataset,
                        objective,
                        audit,
                        seed=seed + 7,
                        batch_size=batch_size,
                        phases=(TrainingPhase.GEOMETRY,),
                        draws=draws,
                    )[METRIC]
                )
                for draws in range(1, args.draws + 1)
            ]
            progress(f"{partition}: {args.draws} cumulative estimates done")
            entry: dict[str, object] = {
                "records": len(dataset),
                "phase_end": phase_end,
                "stored_phase_end": stored.get(partition),
                "cumulative_means": cumulative,
                **draw_statistics(cumulative),
            }
            if selection is not None and partition == selection["partition"]:
                entry["selection"] = {
                    "draws": selection["draws"],
                    "restored": selection["restored"],
                    "selected": selection["selected"],
                    "reproduced": (
                        cumulative[selection["draws"] - 1]
                        if selection["draws"] <= args.draws
                        else None
                    ),
                }
            results[partition] = entry
            print(partition, json.dumps(entry, sort_keys=True, allow_nan=False), flush=True)
    runner._write_new(
        args.output.absolute(),
        {
            "schema": SCHEMA,
            "run": str(loaded.directory),
            "checkpoint_label": loaded.label,
            "checkpoint_sha256": loaded.checkpoint_sha256,
            "seed": seed + 7,
            "partitions": results,
        },
    )


if __name__ == "__main__":
    main()
