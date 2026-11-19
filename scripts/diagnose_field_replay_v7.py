"""Paired geometry-field comparison on fixed covers (research diagnostic, R3).

Two runs are loaded from their bound checkpoints: the *cover run*, whose
selector draws the covers, and the *field run*, whose geometry field integrates
them.  The field run's generator is given the cover run's cover sampler, so for
one protocol seed every field run sees the same covers and the same base draws
and differs in its field only.  This is a declared hybrid of two checkpoints for
a paired diagnostic; it is not a model and its numbers are not benchmark values.

Two measurements per test record:

* ``replay``: the CSP protocol of ``csp_evaluation_v7`` on the reduced-formula
  track with the hybrid generator.  The likelihood rule is not evaluated (its
  cases carry the error ``not evaluated``), every other field is the protocol's.
* ``oracle``: ``--oracle-draws`` geometries of the record's own cover, decoded
  and matched against the record.  It separates the field from the selector.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import train_research_v7 as runner
from csp_evaluation_v7 import (
    STRUCTURAL_VALIDITY_DISTANCE_ANGSTROM,
    case_seed,
    collect_targets,
    evaluate_case,
    shortest_interatomic_distance,
    summarize,
)
from evaluate_csp_research_v7 import (
    add_run_arguments,
    code_fingerprint,
    configure_torch,
    load_run,
    open_research_store,
    partition_record_ids,
    progress,
)

from crystalatlas.errors import DomainError, GeneratedSampleError, InvariantViolation
from crystalatlas.physical_matching import (
    encode_physical_structure,
    physical_structure_match,
    physical_structure_rms,
)
from crystalatlas.trainer import TrainingPhase
from crystalatlas.training_data import StoredTrainingDataset

SCHEMA = "crystalatlas.research-v7-field-replay.v1"
TRACK = "reduced-formula"
ORACLE_STREAM = 1
"""Second word of the oracle seed sequence ``[case seed, ORACLE_STREAM]``."""


def _no_likelihood(*_args, **_kwargs):  # type: ignore[no-untyped-def]
    raise DomainError("not evaluated")


def oracle_case(generator, target, *, draws: int, seed: int) -> dict[str, object]:  # type: ignore[no-untyped-def]
    """``draws`` geometries of the target's own cover, each decoded and matched."""

    model = generator.model
    numerics = model.scientific_config.numerics
    device, _ = generator._device_dtype()
    seeds = np.random.SeedSequence([seed, ORACLE_STREAM]).generate_state(
        draws, dtype=np.uint64
    )
    outcomes = generator._integrate_members_mixed(
        (target.cover,) * draws,
        tuple(generator._torch_generator(int(value), device) for value in seeds),
    )
    layout = model.layout(target.cover)
    stages: dict[str, int] = {}
    shortest: list[float | None] = []
    matches: list[bool] = []
    rms: list[float | None] = []
    for outcome in outcomes:
        stage = None
        distance = None
        matched = False
        value = None
        if isinstance(outcome, Exception):
            stage = "geometry/ode"
        else:
            try:
                decoded = layout.decode(
                    outcome,
                    collision_tolerance=numerics.collision_atol_angstrom,
                    symmetry_tolerance=numerics.symmetry_atol,
                )
                physical = decoded.full_primitive_reduction(
                    cartesian_tolerance=numerics.symmetry_atol
                )
            except GeneratedSampleError:
                stage = "geometry/decode"
            else:
                distance = float(
                    shortest_interatomic_distance(
                        physical.lattice, physical.fractional_positions
                    )
                )
                try:
                    payload = encode_physical_structure(
                        physical.lattice, physical.fractional_positions, physical.atomic_numbers
                    )
                except (DomainError, InvariantViolation):
                    stage = "evaluation/encode"
                else:
                    matched = bool(physical_structure_match(payload, target.payload))
                    if matched:
                        found = physical_structure_rms(payload, target.payload)
                        if found is None or not math.isfinite(found):
                            raise InvariantViolation("a matched pair has no finite RMS witness")
                        value = float(found)
        if stage is not None:
            stages[stage] = stages.get(stage, 0) + 1
        shortest.append(distance)
        matches.append(matched)
        rms.append(value)
    return {
        "record_id": target.record_id,
        "space_group": target.cover.space_group,
        "nu": target.cover.nu,
        "seed": seed,
        "draws": draws,
        "failed_stages": dict(sorted(stages.items())),
        "shortest_distances": shortest,
        "matches": matches,
        "rms": rms,
    }


def distance_counts(rows, key: str) -> dict[str, int]:  # type: ignore[no-untyped-def]
    """Attempt counts by shortest distance; ``None`` is an attempt without a structure."""

    values = [value for row in rows for value in row[key]]
    decoded = [value for value in values if value is not None]
    return {
        "attempts": len(values),
        "without_structure": len(values) - len(decoded),
        "at_most_0.01": sum(value <= 0.01 for value in decoded),
        "at_most_0.1": sum(value <= 0.1 for value in decoded),
        "at_most_0.5": sum(value <= STRUCTURAL_VALIDITY_DISTANCE_ANGSTROM for value in decoded),
        "structurally_valid": sum(
            value > STRUCTURAL_VALIDITY_DISTANCE_ANGSTROM for value in decoded
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_run_arguments(parser)  # --run is the cover run
    parser.add_argument("--field-run", type=Path, required=True)
    parser.add_argument("--field-label", default="after-geometry")
    parser.add_argument("--output", type=Path, required=True, help="result JSON (new file)")
    parser.add_argument("--candidates", type=int, default=20)
    parser.add_argument("--oracle-draws", type=int, default=20)
    parser.add_argument("--max-cases", type=int, default=None)
    args = parser.parse_args()
    if args.candidates < 2 or args.oracle_draws < 1:
        raise DomainError("--candidates must be at least 2 and --oracle-draws at least 1")

    cover_run = args.run.absolute()
    field_run = args.field_run.absolute()
    arguments = json.loads((cover_run / "training-contract.json").read_text())["arguments"]
    seed = int(arguments["seed"]) + 500
    device = configure_torch(args.device, args.cpu_threads, seed)
    pins = dict(
        certification=args.certification,
        certification_fingerprint=args.certification_fingerprint,
        device=device,
        geometry_sampler="mixed-v1",
    )
    covers = load_run(cover_run, args.checkpoint_label, **pins)
    field = load_run(field_run, args.field_label, catalogue=covers.catalogue, **pins)
    cover_config = covers.model.scientific_config
    field_config = field.model.scientific_config
    if cover_config.domain != field_config.domain:
        raise DomainError("the two runs differ in their domain")
    if cover_config.numerics != field_config.numerics:
        raise DomainError("the two runs differ in their numerics")
    if covers.contract["partitions"] != field.contract["partitions"]:
        raise DomainError("the two runs were trained on different partitions")
    cover_model = cover_config.to_dict()["model"]
    field_model = field_config.to_dict()["model"]
    differing = {
        name: [cover_model.get(name), field_model.get(name)]
        for name in sorted(set(cover_model) | set(field_model))
        if cover_model.get(name) != field_model.get(name)
    }
    progress(f"model fields that differ (cover run, field run): {differing}")

    generator = field.generator
    # The declared hybrid: covers from the cover run's selector, everything after
    # the cover from the field run.  The likelihood rule is switched off.
    generator._cover_sampler = covers.generator._cover_sampler
    generator.select_map_sample = _no_likelihood  # type: ignore[method-assign]

    with open_research_store(args) as store:
        record_ids = partition_record_ids(covers, store, "test")
        selected = tuple(sorted(record_ids))
        if args.max_cases is not None:
            selected = selected[: args.max_cases]
        dataset = StoredTrainingDataset(
            store,
            covers.model.composition_codec,
            covers.catalogue,
            domain=cover_config.domain,
            record_ids=selected,
        )
        targets = collect_targets(
            covers.model,
            runner._sampler(
                dataset,
                TrainingPhase.GEOMETRY,
                int(arguments["seed"]),
                int(arguments["batch_size"]),
            ),
        )
    if tuple(target.record_id for target in targets) != selected:
        raise DomainError("the evaluation targets are not the selected records")

    started = time.perf_counter()
    replay: list[dict[str, object]] = []
    oracle: list[dict[str, object]] = []
    generator.model.eval()
    for target in targets:
        request_seed = case_seed(seed, target.record_id)
        case = evaluate_case(
            generator, target, track=TRACK, candidates=args.candidates, seed=request_seed
        )
        replay.append(case)
        clock = time.perf_counter()
        own = oracle_case(generator, target, draws=args.oracle_draws, seed=request_seed)
        own["seconds"] = round(time.perf_counter() - clock, 2)
        oracle.append(own)
        print(
            json.dumps(
                {
                    "record_id": target.record_id[:12],
                    "replay_valid": case["valid"],
                    "replay_structurally_valid": case["structurally_valid"],
                    "replay_matches": case["match_count"],
                    "replay_failed": case["failed_stages"],
                    "oracle_matches": int(sum(own["matches"])),
                    "oracle_failed": own["failed_stages"],
                    "seconds": [case["seconds"]["sample"], own["seconds"]],
                }
            ),
            flush=True,
        )
    oracle_fractions = [sum(case["matches"]) / args.oracle_draws for case in oracle]
    matched_rms = [value for case in oracle for value in case["rms"] if value is not None]
    result = {
        "schema": SCHEMA,
        "cover_run": str(cover_run),
        "cover_label": covers.label,
        "cover_checkpoint_sha256": covers.checkpoint_sha256,
        "field_run": str(field_run),
        "field_label": field.label,
        "field_checkpoint_sha256": field.checkpoint_sha256,
        "field_options": field.arguments.get("options", "default"),
        "model_fields_differing": differing,
        "protocol_seed": seed,
        "candidates": args.candidates,
        "oracle_draws": args.oracle_draws,
        "record_ids": [target.record_id for target in targets],
        "code": code_fingerprint(),
        "device": str(device),
        "torch": torch.__version__,
        "replay_summary": summarize(replay, track=TRACK, candidates=args.candidates),
        "replay_distances": distance_counts(replay, "shortest_distances"),
        "oracle_summary": {
            "cases": len(oracle),
            "draws": args.oracle_draws,
            "match_per_draw": float(np.mean(oracle_fractions)),
            "match_per_draw_standard_error": (
                float(np.std(oracle_fractions, ddof=1) / math.sqrt(len(oracle_fractions)))
                if len(oracle_fractions) > 1
                else None
            ),
            "cases_first_draw_matches": int(sum(case["matches"][0] for case in oracle)),
            "cases_any_draw_matches": int(sum(any(case["matches"]) for case in oracle)),
            "mean_rms_of_matches": float(np.mean(matched_rms)) if matched_rms else None,
            "distances": distance_counts(oracle, "shortest_distances"),
        },
        "replay_cases": replay,
        "oracle_cases": oracle,
        "wall_seconds": round(time.perf_counter() - started, 1),
    }
    runner._write_new(args.output.absolute(), result)
    brief = {key: result[key] for key in ("replay_distances", "oracle_summary")}
    brief["replay_match_per_attempt"] = result["replay_summary"]["match_per_attempt"]
    brief["replay_match_at_k"] = result["replay_summary"]["match_at_k"]
    brief["replay_failed_stages"] = result["replay_summary"]["failed_stages"]
    print(json.dumps(brief), flush=True)


if __name__ == "__main__":
    main()
