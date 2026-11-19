"""Replay one CSP case under both geometry samplers and compare every attempt.

Reads the composition and the request seed of one case from a result JSONL,
draws the request with ``bucket-v1`` and with ``mixed-v1`` from one loaded
model, and reports per attempt the latent gaps and the decode outcome of each
sampler.  Nothing is written.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

import evaluate_csp_research_v7 as harness
import train_research_v7 as runner
from crystalatlas.composition import Composition
from crystalatlas.errors import DomainError, GeneratedSampleError, InvariantViolation
from crystalatlas.physical_matching import (
    encode_physical_structure,
    physical_structure_match,
    physical_structure_rms,
)
from crystalatlas.pipeline import GenerationMode, GenerationRequest
from crystalatlas.trainer import TrainingPhase
from crystalatlas.training_data import StoredTrainingDataset
from csp_evaluation_v7 import collect_targets, shortest_interatomic_distance


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    harness.add_run_arguments(parser)
    parser.add_argument("--cases", type=Path, required=True, help="result JSONL")
    parser.add_argument("--record", action="append", required=True, help="record-id prefix")
    args = parser.parse_args()

    cases = [json.loads(line) for line in args.cases.read_text().splitlines()[1:]]
    run = args.run.absolute()
    arguments = json.loads((run / "training-contract.json").read_text())["arguments"]
    device = harness.configure_torch(args.device, args.cpu_threads, int(arguments["seed"]) + 500)
    loaded = harness.load_run(
        run,
        args.checkpoint_label,
        certification=args.certification,
        certification_fingerprint=args.certification_fingerprint,
        device=device,
        geometry_sampler="bucket-v1",
    )
    generator = loaded.generator
    model = loaded.model
    numerics = model.scientific_config.numerics
    wanted = []
    for prefix in args.record:
        (case,) = [item for item in cases if item["record_id"].startswith(prefix)]
        wanted.append(case)
    with harness.open_research_store(args) as store:
        dataset = StoredTrainingDataset(
            store,
            model.composition_codec,
            loaded.catalogue,
            domain=model.scientific_config.domain,
            record_ids=tuple(sorted(case["record_id"] for case in wanted)),
        )
        targets = {
            target.record_id: target
            for target in collect_targets(
                model,
                runner._sampler(
                    dataset,
                    TrainingPhase.GEOMETRY,
                    int(arguments["seed"]),
                    int(arguments["batch_size"]),
                ),
            )
        }

    def verdict(trace, target):  # type: ignore[no-untyped-def]
        """Matcher verdict of one attempt, as the harness computes it."""

        if not trace.succeeded:
            return f"failed {trace.failed_stage}"
        physical = trace.sample.physical_primitive
        shortest = shortest_interatomic_distance(physical.lattice, physical.fractional_positions)
        try:
            payload = encode_physical_structure(
                physical.lattice, physical.fractional_positions, physical.atomic_numbers
            )
        except (DomainError, InvariantViolation) as exc:
            return f"atoms {len(physical.atomic_numbers)} shortest {shortest:.6f} encode {type(exc).__name__}"
        matched = bool(physical_structure_match(payload, target.payload))
        rms = physical_structure_rms(payload, target.payload)
        return (
            f"atoms {len(physical.atomic_numbers)} shortest {shortest:.6f} "
            f"match {matched} rms {rms!r}"
        )

    created: list[object] = []
    make = generator._torch_generator

    def recording_generator(seed, device):  # type: ignore[no-untyped-def]
        made = make(seed, device)
        created.append(made)
        return made

    generator._torch_generator = recording_generator  # type: ignore[method-assign]
    solved: dict[int, tuple[object, object]] = {}
    bucket = generator._integrate_members
    mixed = generator._integrate_members_mixed

    def recording(inner):  # type: ignore[no-untyped-def]
        def call(covers, generators):  # type: ignore[no-untyped-def]
            outcomes = inner(covers, generators)
            for cover, item, outcome in zip(covers, generators, outcomes, strict=True):
                solved[id(item)] = (cover, outcome)
            return outcomes

        return call

    generator._integrate_members = recording(bucket)  # type: ignore[method-assign]
    generator._integrate_members_mixed = recording(mixed)  # type: ignore[method-assign]

    def draw(request, sampler):  # type: ignore[no-untyped-def]
        created.clear()
        solved.clear()
        generator.geometry_sampler = sampler
        started = time.perf_counter()
        traces = generator.sample(request)
        seconds = time.perf_counter() - started
        if len(created) != 3 * request.samples:
            raise SystemExit(f"unexpected generator count {len(created)}")
        per_attempt = [solved[id(created[3 * index + 2])] for index in range(request.samples)]
        return traces, per_attempt, seconds

    def decode(cover, geometry):  # type: ignore[no-untyped-def]
        try:
            decoded = model.layout(cover).decode(
                geometry,
                collision_tolerance=numerics.collision_atol_angstrom,
                symmetry_tolerance=numerics.symmetry_atol,
            )
        except GeneratedSampleError as exc:
            return f"decode: {type(exc).__name__}: {exc}"
        try:
            decoded.full_primitive_reduction(cartesian_tolerance=numerics.symmetry_atol)
        except GeneratedSampleError as exc:
            return f"primitive reduction: {type(exc).__name__}: {exc}"
        return "ok"

    print("collision_atol_angstrom", numerics.collision_atol_angstrom)
    print("symmetry_atol", numerics.symmetry_atol)
    for prefix, case in zip(args.record, wanted, strict=True):
        target = targets[case["record_id"]]
        print(
            f"case {prefix} target {target.cover.space_group} nu {target.cover.nu} "
            f"primitive atoms {target.primitive_atoms} recorded matches {case['matches']}"
        )
        request = GenerationRequest(
            mode=GenerationMode.CSP,
            composition=Composition(tuple((int(z), int(n)) for z, n in case["composition"])),
            samples=int(case["attempts"]),
            seed=int(case["seed"]),
            primitive_atom_count=case["primitive_atom_count"],
        )
        results = {}
        for sampler in ("bucket-v1", "mixed-v1", "bucket-v1", "mixed-v1"):
            traces, per_attempt, seconds = draw(request, sampler)
            print(f"case {prefix} sampler {sampler} seconds {seconds:.2f}")
            results.setdefault(sampler, []).append((traces, per_attempt))
        for sampler, repeats in results.items():
            same = all(
                type(a[1]) is type(b[1])
                and (
                    isinstance(a[1], Exception)
                    or (
                        np.array_equal(a[1].lattice_latent, b[1].lattice_latent)
                        and np.array_equal(a[1].intrinsic_coordinates, b[1].intrinsic_coordinates)
                    )
                )
                for a, b in zip(repeats[0][1], repeats[1][1], strict=True)
            )
            print(f"case {prefix} sampler {sampler} repeat is bit-identical: {same}")
        bucket_traces, bucket_attempts = results["bucket-v1"][0]
        mixed_traces, mixed_attempts = results["mixed-v1"][0]
        for index in range(request.samples):
            cover_b, out_b = bucket_attempts[index]
            cover_m, out_m = mixed_attempts[index]
            if cover_b != cover_m:
                raise SystemExit(f"attempt {index}: the samplers drew different covers")
            line = [
                f"attempt {index:2d}",
                f"{cover_b.space_group} nu {cover_b.nu}",
                f"stage bucket {bucket_traces[index].failed_stage} mixed {mixed_traces[index].failed_stage}",
            ]
            verdict_b = verdict(bucket_traces[index], target)
            verdict_m = verdict(mixed_traces[index], target)
            if isinstance(out_b, Exception) or isinstance(out_m, Exception):
                line.append(f"ode bucket {out_b!r} mixed {out_m!r}"[:400])
                print("; ".join(line))
                print(f"  verdict bucket: {verdict_b}")
                print(f"  verdict mixed:  {verdict_m}")
                continue
            gap_z = float(np.max(np.abs(out_b.lattice_latent - out_m.lattice_latent), initial=0.0))
            gap_eta = float(
                np.max(
                    np.abs(out_b.intrinsic_coordinates - out_m.intrinsic_coordinates),
                    initial=0.0,
                )
            )
            line.append(f"gap z {gap_z:.3e} eta {gap_eta:.3e}")
            status_b = decode(cover_b, out_b)
            status_m = decode(cover_m, out_m)
            line.append(f"bucket {status_b}"[:600])
            line.append(f"mixed {status_m}"[:600])
            print("; ".join(line))
            print(f"  verdict bucket: {verdict_b}")
            print(f"  verdict mixed:  {verdict_m}")
            if status_b != status_m or verdict_b.split(" rms ")[0] != verdict_m.split(" rms ")[0]:
                print("  cover", cover_b)
                print("  bucket z", out_b.lattice_latent.tolist())
                print("  mixed  z", out_m.lattice_latent.tolist())
                print("  bucket eta", out_b.intrinsic_coordinates.tolist())
                print("  mixed  eta", out_m.intrinsic_coordinates.tolist())


if __name__ == "__main__":
    main()
