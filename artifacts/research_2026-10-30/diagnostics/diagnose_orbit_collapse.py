"""Record the sampling trajectory of every attempt of one CSP request.

The request is given by composition, seed and optional primitive atom count.
The integrator is replaced by an arithmetic-equal version that keeps the RK4
trajectory and the terminal velocity.  ``--sampler`` chooses the geometry
sampler (``both`` runs the bucket sampler and then the fused one from the same
loaded model).  ``--scan ATTEMPT:COORDINATE`` additionally reads the velocity
field on a grid of one intrinsic coordinate around the wrap point 0 = 1, with
every other coordinate held on the recorded trajectory.  One JSON file is
written per sampler; nothing else is changed.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import evaluate_csp_research_v7 as harness
import numpy as np
import torch

import crystalatlas.learned as learned
from crystalatlas import flow
from crystalatlas.composition import Composition
from crystalatlas.pipeline import GenerationMode, GenerationRequest
from csp_evaluation_v7 import shortest_interatomic_distance

SCAN_STEPS = (0, 8, 16, 24, 32, 40, 44, 48, 52, 56, 58, 60, 61, 62, 63, 64)
SCAN_OFFSETS = (1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 3e-3, 1e-2, 2e-2, 3e-2, 5e-2, 0.1, 0.2, 0.3)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    harness.add_run_arguments(parser)
    parser.add_argument("--composition", required=True, help="Z:count,Z:count,...")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--primitive-atom-count", type=int, default=None)
    parser.add_argument("--attempts", type=int, default=20)
    parser.add_argument("--sampler", choices=("bucket-v1", "mixed-v1", "both"), default="bucket-v1")
    parser.add_argument("--scan", action="append", default=[], help="ATTEMPT:COORDINATE")
    parser.add_argument("--dump", type=Path, required=True, help="prefix; .<sampler>.json is added")
    parser.add_argument(
        "--field-run",
        type=Path,
        default=None,
        help="take the geometry field from this run and the covers from --run (paired hybrid)",
    )
    parser.add_argument("--field-label", default="after-geometry")
    args = parser.parse_args()

    run = args.run.absolute()
    arguments = json.loads((run / "training-contract.json").read_text())["arguments"]
    device = harness.configure_torch(
        args.device, args.cpu_threads, int(arguments["seed"]) + 500
    )
    loaded = harness.load_run(
        run,
        args.checkpoint_label,
        certification=args.certification,
        certification_fingerprint=args.certification_fingerprint,
        device=device,
        geometry_sampler="bucket-v1",
    )
    generator, model = loaded.generator, loaded.model
    if args.field_run is not None:
        field = harness.load_run(
            args.field_run.absolute(),
            args.field_label,
            certification=args.certification,
            certification_fingerprint=args.certification_fingerprint,
            device=device,
            geometry_sampler="bucket-v1",
            catalogue=loaded.catalogue,
        )
        field.generator._cover_sampler = generator._cover_sampler
        generator, model = field.generator, field.model
        print("field run", args.field_run, "options", field.arguments.get("options", "default"))
    numerics = model.scientific_config.numerics
    print("ode_steps", numerics.ode_steps, "ode_terminal_time", numerics.ode_terminal_time)
    print("collision_atol_angstrom", numerics.collision_atol_angstrom)
    print("symmetry_atol", numerics.symmetry_atol)

    records: list[dict[str, object]] = []

    def recording_integrator(field, initial_state, period_basis, *, num_steps, terminal_time=1.0):  # type: ignore[no-untyped-def]
        result = flow.integrate_rk4(
            field,
            initial_state,
            period_basis,
            num_steps=num_steps,
            t_end=terminal_time,
            return_trajectory=True,
        )
        state = result.final_state
        velocity = flow._coerce_velocity(
            field(state.z.new_tensor(terminal_time), state.z, state.eta)
        )
        final = (
            state
            if terminal_time == 1.0
            else flow._wrapped_add(
                state, velocity, state.z.new_tensor(1.0 - terminal_time), period_basis
            )
        )
        records.append(
            {
                "field": field,
                "times_tensor": result.times,
                "z_tensor": result.z_trajectory,
                "eta_tensor": result.eta_trajectory,
                "times": result.times.detach().cpu().numpy().astype(np.float64),
                "z": result.z_trajectory.detach().cpu().numpy().astype(np.float64),
                "eta": result.eta_trajectory.detach().cpu().numpy().astype(np.float64),
                "terminal_velocity_z": velocity.z.detach().cpu().numpy().astype(np.float64),
                "terminal_velocity_eta": velocity.eta.detach().cpu().numpy().astype(
                    np.float64
                ),
                "final_z": final.z.detach().cpu().numpy().astype(np.float64),
                "final_eta": final.eta.detach().cpu().numpy().astype(np.float64),
            }
        )
        return final

    learned.integrate_with_endpoint_estimate = recording_integrator  # type: ignore[assignment]

    created: list[object] = []
    make = generator._torch_generator

    def recording_generator(seed, device):  # type: ignore[no-untyped-def]
        made = make(seed, device)
        created.append(made)
        return made

    generator._torch_generator = recording_generator  # type: ignore[method-assign]
    rows: dict[int, tuple[object, dict[str, object], int]] = {}

    def recording_solver(inner):  # type: ignore[no-untyped-def]
        def call(covers, generators):  # type: ignore[no-untyped-def]
            before = len(records)
            geometries = inner(covers, generators)
            if len(records) == before + 1 and len(records[-1]["final_z"]) == len(covers):
                for row, (cover, item) in enumerate(zip(covers, generators, strict=True)):
                    rows[id(item)] = (cover, records[-1], row)
            return geometries

        return call

    generator._integrate_geometry_batch = recording_solver(  # type: ignore[method-assign]
        generator._integrate_geometry_batch
    )
    generator._integrate_mixed_chunk = recording_solver(  # type: ignore[method-assign]
        generator._integrate_mixed_chunk
    )

    pairs = tuple(tuple(int(v) for v in item.split(":")) for item in args.composition.split(","))
    request = GenerationRequest(
        mode=GenerationMode.CSP,
        composition=Composition(pairs),
        samples=args.attempts,
        seed=args.seed,
        primitive_atom_count=args.primitive_atom_count,
    )
    samplers = ("bucket-v1", "mixed-v1") if args.sampler == "both" else (args.sampler,)
    for sampler in samplers:
        created.clear()
        rows.clear()
        records.clear()
        generator.geometry_sampler = sampler
        traces = generator.sample(request)
        if len(created) != 3 * args.attempts:
            raise SystemExit(f"unexpected generator count {len(created)}")
        report: dict[str, object] = {
            "sampler": sampler,
            "composition": [list(pair) for pair in pairs],
            "seed": args.seed,
            "primitive_atom_count": args.primitive_atom_count,
            "ode_steps": numerics.ode_steps,
            "ode_terminal_time": numerics.ode_terminal_time,
            "solves": len(records),
            "attempts": [],
            "scans": [],
        }
        for index, trace in enumerate(traces):
            cover, record, row = rows[id(created[3 * index + 2])]
            layout = model.layout(cover)
            dz, de = layout.lattice_dimension, layout.intrinsic_dimension
            eta = record["eta"][:, row, :de]
            parameters = [
                [value.tolist() for value in layout.reconstruct_parameters(step)]
                for step in eta
            ]
            final_eta = record["final_eta"][row][:de]
            entry = {
                "attempt": index,
                "failed_stage": trace.failed_stage,
                "error": trace.error,
                "space_group": cover.space_group,
                "nu": cover.nu,
                "orbits": [[orbit.wyckoff_label, orbit.atomic_number] for orbit in cover.orbits],
                "coordinate_matrices": [
                    [[str(value) for value in vector] for vector in position.coordinate_matrix]
                    for position in layout.positions
                ],
                "parameter_maps": [mapping.tolist() for mapping in layout.parameter_maps],
                "layout_period_basis": np.asarray(layout.period_basis).tolist(),
                "batch_rows": int(record["z"].shape[1]),
                "times": record["times"].tolist(),
                "z": record["z"][:, row, :dz].tolist(),
                "eta": eta.tolist(),
                "parameters": parameters,
                "terminal_velocity_z": record["terminal_velocity_z"][row][:dz].tolist(),
                "terminal_velocity_eta": record["terminal_velocity_eta"][row][:de].tolist(),
                "final_z": record["final_z"][row][:dz].tolist(),
                "final_eta": final_eta.tolist(),
                "final_parameters": [
                    value.tolist() for value in layout.reconstruct_parameters(final_eta)
                ],
            }
            if trace.succeeded:
                physical = trace.sample.physical_primitive
                entry["atoms"] = int(len(physical.atomic_numbers))
                entry["shortest_distance"] = float(
                    shortest_interatomic_distance(
                        np.asarray(physical.lattice, dtype=np.float64),
                        np.asarray(physical.fractional_positions, dtype=np.float64),
                    )
                )
            report["attempts"].append(entry)
            print(
                json.dumps(
                    {
                        "sampler": sampler,
                        **{
                            key: entry.get(key)
                            for key in (
                                "attempt",
                                "failed_stage",
                                "space_group",
                                "nu",
                                "orbits",
                                "shortest_distance",
                            )
                        },
                    }
                ),
                flush=True,
            )
        targets: list[tuple[int, int]] = []
        for item in args.scan:
            if item != "auto":
                attempt, coordinate = (int(value) for value in item.split(":"))
                targets.append((attempt, coordinate))
                continue
            # Every coordinate that comes within 1e-6 of the wrap point 0 = 1.
            for entry in report["attempts"]:
                eta = np.asarray(entry["eta"], dtype=np.float64)
                if eta.size == 0:
                    continue
                closest = np.minimum(eta, 1.0 - eta).min(axis=0)
                targets.extend(
                    (entry["attempt"], int(coordinate))
                    for coordinate in np.flatnonzero(closest < 1.0e-6)
                )
        print("scan targets", targets, flush=True)
        for attempt, coordinate in targets:
            cover, record, row = rows[id(created[3 * attempt + 2])]
            field = record["field"]
            values = [0.0, 1.0]
            for offset in SCAN_OFFSETS:
                values.extend((offset, 1.0 - offset))
            scan = {"attempt": attempt, "coordinate": coordinate, "steps": []}
            with torch.no_grad():
                for step in SCAN_STEPS:
                    if step >= len(record["times"]):
                        continue
                    time = record["times_tensor"][step]
                    z = record["z_tensor"][step]
                    base = record["eta_tensor"][step]
                    points = []
                    for value in values:
                        eta = base.clone()
                        eta[row, coordinate] = value
                        velocity = flow._coerce_velocity(field(time, z, eta))
                        points.append(
                            [
                                float(eta[row, coordinate]),
                                float(velocity.eta[row, coordinate]),
                            ]
                        )
                    own = flow._coerce_velocity(field(time, z, base))
                    scan["steps"].append(
                        {
                            "step": step,
                            "time": float(time),
                            "trajectory_value": float(base[row, coordinate]),
                            "trajectory_velocity": float(own.eta[row, coordinate]),
                            "points": points,
                        }
                    )
            report["scans"].append(scan)
        target = args.dump.with_name(f"{args.dump.name}.{sampler}.json")
        target.write_text(json.dumps(report))
        print("wrote", target, flush=True)


if __name__ == "__main__":
    main()
