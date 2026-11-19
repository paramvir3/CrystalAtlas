"""Measurement M1 of the carrier A/B: which atom pairs form the short contacts, and when.

Re-runs the paired field replay of ``diagnose_field_replay_v7`` (covers from the
cover run, geometry from the field run, the same protocol seed, the fused
sampler) for the test records, with an arithmetic-equal recording integrator,
and checks every replayed case and every oracle case against the stored result
of that field.  For each attempt it reads the cell and the atom positions the
field itself sees (``PaddedCoverPlan.expand`` of the solve) and records:

* at t = 1, for every contact class (the unordered pair of orbits of two atoms;
  "mates" when both atoms belong to one orbit) with an image within
  ``--radius`` angstrom, the smallest image distance ``r`` and the smooth
  separation ``rho = |sin(2 pi delta) L| / 2 pi`` of the minimizing atom pair,
  which is the argument of the carrier switch (``rho`` is the smallest and
  ``rho_max`` the largest over pairs tied with the minimum to 1e-6 relative);
* for the attempts listed in ``--targets``, the same quantities at every RK4
  step for the classes that end within ``--keep-end`` or come within
  ``--keep-min`` angstrom, the overall closest distance, the shortest lattice
  vector, the numbers of half-lattice-type pairs (``r - rho`` above 0.25
  angstrom, not on {0, 1/2}^3) whose switch at s = 0.2 angstrom would be at most
  0.9 and at most 1/sqrt(2), and the number of pairs on {0, 1/2}^3, whose
  carriers are zero in both arms.

Distances are exact up to ``--radius`` (integer image box of
``minimum_periodic_pair_distance``); a step whose box would exceed
``--max-reach`` images per axis is marked unresolved.  Research diagnostic: it
reads private attributes of the generator and changes no sampled value.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from pathlib import Path

import diagnose_field_replay_v7 as replay
import numpy as np
import torch
import train_research_v7 as runner
from csp_evaluation_v7 import case_seed, collect_targets, evaluate_case
from evaluate_csp_research_v7 import (
    add_run_arguments,
    code_fingerprint,
    configure_torch,
    load_run,
    open_research_store,
    partition_record_ids,
    progress,
)

import crystalatlas.learned as learned
from crystalatlas import flow
from crystalatlas.errors import DomainError, InvariantViolation
from crystalatlas.nn.bridge import PaddedCoverPlan, TorchCoverGeometry
from crystalatlas.trainer import TrainingPhase
from crystalatlas.training_data import StoredTrainingDataset

SCHEMA = "crystalatlas.research-v7-m1-contacts.v1"
REFERENCE_SMOOTHING = 0.2
"""The registered switch length (angstrom) at which the half-lattice counts are read."""
HALF_LATTICE_GAP = 0.25
"""A pair is half-lattice-type when its distance exceeds its smooth separation by this much."""
TIE_RELATIVE = 1.0e-6
"""Pairs within this relative distance of a class minimum count as tied with it."""
REPLAY_FIELDS = (
    "seed", "attempts", "valid", "failed_stages", "matches", "match_count", "first",
    "consensus", "best_rms", "distinct_tables", "target_table_valid",
    "target_table_matches", "shortest_distances", "structurally_valid",
)
ORACLE_FIELDS = ("seed", "draws", "failed_stages", "shortest_distances", "matches", "rms")


def significant(value: float | None, digits: int = 7) -> float | None:
    if value is None or not math.isfinite(value):
        return None
    return float(f"{value:.{digits}g}")


def switch_threshold(value: float, smoothing: float, epsilon: float, reference: float) -> float:
    """``rho`` (angstrom) at which the switch of length ``smoothing`` equals ``value``.

    The switch is ``sqrt((q + epsilon) / (q + (2 pi s)^2))`` with
    ``q = |sin(2 pi delta) L / reference|^2 = (2 pi rho / reference)^2``.
    """

    floor = (2.0 * math.pi * smoothing) ** 2
    q = (value * value * floor - epsilon) / (1.0 - value * value)
    if q <= 0.0:
        raise DomainError("the switch never falls to this value")
    return reference * math.sqrt(q) / (2.0 * math.pi)


class Recorder:
    """Trajectories of the solves of one case, keyed by the geometry generators."""

    def __init__(self) -> None:
        self.created: list[torch.Generator] = []
        self.records: list[dict[str, object]] = []
        self.rows: dict[int, tuple[dict[str, object], int]] = {}
        self.pending_plan: PaddedCoverPlan | None = None
        self.traces: tuple[object, ...] | None = None

    def clear(self) -> None:
        self.created.clear()
        self.records.clear()
        self.rows.clear()
        self.pending_plan = None
        self.traces = None


def install(generator, recorder: Recorder) -> None:  # type: ignore[no-untyped-def]
    """Arithmetic-neutral hooks: record plans, trajectories, generators and traces."""

    class RecordingPlan(PaddedCoverPlan):
        def __init__(self, *args, **kwargs) -> None:  # type: ignore[no-untyped-def]
            super().__init__(*args, **kwargs)
            recorder.pending_plan = self

    def recording_integrator(  # type: ignore[no-untyped-def]
        field, initial_state, period_basis, *, num_steps, terminal_time=1.0
    ):
        # Same calls in the same order as flow.integrate_with_endpoint_estimate;
        # return_trajectory only keeps references to the states it computes.
        if not (math.isfinite(terminal_time) and 0.0 < terminal_time <= 1.0):
            raise ValueError("terminal_time must lie in (0, 1]")
        plan, recorder.pending_plan = recorder.pending_plan, None
        result = flow.integrate_rk4(
            field,
            initial_state,
            period_basis,
            num_steps=num_steps,
            t_end=terminal_time,
            return_trajectory=True,
        )
        if terminal_time == 1.0:
            final = result.final_state
        else:
            state = result.final_state
            velocity = flow._coerce_velocity(
                field(state.z.new_tensor(terminal_time), state.z, state.eta)
            )
            final = flow._wrapped_add(
                state, velocity, state.z.new_tensor(1.0 - terminal_time), period_basis
            )
        recorder.records.append(
            {
                "plan": plan,
                "times": result.times,
                "z": result.z_trajectory,
                "eta": result.eta_trajectory,
                "final_z": final.z,
                "final_eta": final.eta,
            }
        )
        return final

    learned.PaddedCoverPlan = RecordingPlan  # type: ignore[misc]
    learned.integrate_with_endpoint_estimate = recording_integrator  # type: ignore[assignment]

    make = generator._torch_generator

    def recording_generator(seed, device):  # type: ignore[no-untyped-def]
        made = make(seed, device)
        recorder.created.append(made)
        return made

    generator._torch_generator = recording_generator

    chunk = generator._integrate_mixed_chunk

    def recording_chunk(covers, generators):  # type: ignore[no-untyped-def]
        recorder.pending_plan = None
        before = len(recorder.records)
        geometries = chunk(covers, generators)
        record = recorder.records[-1] if len(recorder.records) == before + 1 else None
        if record is None or record["plan"] is None or record["plan"].batch != len(covers):
            raise InvariantViolation("a mixed chunk did not record exactly one padded solve")
        record["covers"] = tuple(covers)
        for row, item in enumerate(generators):
            recorder.rows[id(item)] = (record, row)
        return geometries

    generator._integrate_mixed_chunk = recording_chunk

    bucket = generator._integrate_geometry_batch

    def bucket_batch(*args, **kwargs):  # type: ignore[no-untyped-def]
        # A single-member fallback solve has no padded plan; it is not recorded.
        recorder.pending_plan = None
        return bucket(*args, **kwargs)

    generator._integrate_geometry_batch = bucket_batch

    sample = generator.sample

    def recording_sample(request):  # type: ignore[no-untyped-def]
        traces = sample(request)
        recorder.traces = tuple(traces)
        return traces

    generator.sample = recording_sample


def cell_atoms(model, cover):  # type: ignore[no-untyped-def]
    """Orbit index and atomic number of every atom the field sees, in the plan's order."""

    layout = model.layout(cover)
    bridge = TorchCoverGeometry(
        layout, expansion_cell=model.scientific_config.model.geometry_expansion_cell
    )
    orbit_of: list[int] = []
    for index, constants in enumerate(bridge._orbit_constants):
        operations = constants["operations"]
        if not isinstance(operations, tuple):
            raise InvariantViolation("invalid immutable operation metadata")
        orbit_of.extend([index] * len(operations))
    numbers = [int(value) for value in bridge.atomic_numbers.tolist()]
    orbits = [
        [orbit.wyckoff_label, int(orbit.atomic_number), orbit_of.count(index)]
        for index, orbit in enumerate(cover.orbits)
    ]
    return np.asarray(orbit_of, dtype=np.int64), numbers, orbits


def pair_geometry(lattice, fractional, radius: float, max_reach: int):  # type: ignore[no-untyped-def]
    """Exact image distances up to ``radius`` and smooth separations of all ordered pairs.

    ``lattice`` has the basis vectors as rows (``cartesian = fractional @ lattice``);
    ``delta[i, j] = x_j - x_i`` as in the transformer.  Returns ``None`` when the
    integer image box exceeds ``max_reach`` per axis.
    """

    delta = fractional[None, :, :] - fractional[:, None, :]
    rho = torch.linalg.vector_norm(torch.sin(2.0 * math.pi * delta) @ lattice, dim=-1) / (
        2.0 * math.pi
    )
    reduced = delta - torch.round(delta)
    gram_inverse = torch.linalg.inv(lattice @ lattice.transpose(0, 1))
    reach = radius * torch.sqrt(torch.diagonal(gram_inverse))
    box = [int(math.ceil(float(value) + 0.5)) for value in reach]
    if max(box) > max_reach:
        return None
    axes = [
        torch.arange(-size, size + 1, dtype=lattice.dtype, device=lattice.device)
        for size in box
    ]
    shifts = torch.cartesian_prod(*axes)
    count = fractional.shape[0]
    squared = torch.full((count, count), math.inf, dtype=lattice.dtype, device=lattice.device)
    zero = (shifts == 0).all(dim=-1)
    eye = torch.eye(count, dtype=torch.bool, device=lattice.device)
    block = max(1, 2_000_000 // max(1, count * shifts.shape[0]))
    for start in range(0, count, block):
        part = reduced[start : start + block]
        cartesian = (part[:, :, None, :] + shifts[None, None, :, :]) @ lattice
        values = (cartesian * cartesian).sum(dim=-1)
        self_zero = eye[start : start + block][:, :, None] & zero[None, None, :]
        values = torch.where(self_zero, torch.full_like(values, math.inf), values)
        squared[start : start + block] = values.min(dim=-1).values
    distance = torch.sqrt(squared)
    distance = torch.where(distance <= radius, distance, torch.full_like(distance, math.inf))
    return (
        distance.cpu().numpy(),
        rho.cpu().numpy(),
        reduced.cpu().numpy(),
    )


def class_minima(distance, rho, orbit_of, orbit_count):  # type: ignore[no-untyped-def]
    """Per unordered orbit pair: smallest distance over its atom pairs, with that pair's rho."""

    first, second = np.triu_indices(len(orbit_of), 1)
    if len(first) == 0:
        return {}
    values = distance[first, second]
    smooth = rho[first, second]
    low = np.minimum(orbit_of[first], orbit_of[second])
    high = np.maximum(orbit_of[first], orbit_of[second])
    key = low * orbit_count + high
    order = np.lexsort((values, key))
    ordered = key[order]
    starts = np.flatnonzero(np.r_[True, ordered[1:] != ordered[:-1]])
    stops = np.r_[starts[1:], len(order)]
    result = {}
    for start, stop in zip(starts, stops, strict=True):
        group = order[start:stop]
        leader = group[0]
        best = float(values[leader])
        # Symmetry-related pairs tie in r but not always in rho (3- and 6-fold
        # axes); both ends of rho over the tied pairs are kept.
        tied = (
            group[values[group] <= best * (1.0 + TIE_RELATIVE) + 1.0e-12]
            if math.isfinite(best)
            else group[:1]
        )
        result[(int(low[leader]), int(high[leader]))] = (
            best,
            float(smooth[tied].min()),
            float(smooth[tied].max()),
            int(first[leader]),
            int(second[leader]),
        )
    return result


def half_lattice_counts(distance, rho, reduced, thresholds):  # type: ignore[no-untyped-def]
    """Half-lattice-type pairs with rho at most each threshold, then pairs on {0, 1/2}^3.

    A pair whose reduced delta lies on {0, 1/2}^3 (to 1e-6) has all carriers zero in
    both arms, so the switch does not act on it; it is counted only in the last entry.
    """

    first, second = np.triu_indices(distance.shape[0], 1)
    values = distance[first, second]
    smooth = rho[first, second]
    doubled = 2.0 * reduced[first, second]
    exact = np.all(np.abs(doubled - np.rint(doubled)) <= 2.0e-6, axis=-1)
    far = (~np.isfinite(values) | (values - smooth > HALF_LATTICE_GAP)) & ~exact
    return [int(np.count_nonzero(far & (smooth <= limit))) for limit in thresholds] + [
        int(np.count_nonzero(exact))
    ]


def describe_row(  # type: ignore[no-untyped-def]
    record, row, model, *, steps, radius, max_reach, thresholds, keep_end, keep_min
):
    """End state of one solved row and, for more than one step, its trajectory."""

    plan = record["plan"]
    cover = record["covers"][row]
    orbit_of, numbers, orbits = cell_atoms(model, cover)
    count = plan.atom_counts[row]
    if len(orbit_of) != count or numbers != plan.atomic_numbers[row, :count].tolist():
        raise InvariantViolation("the plan's atoms are not the cover's orbit expansion")
    per_step = []
    with torch.no_grad():
        for step in steps:
            expanded = plan.expand(record["z"][step], record["eta"][step])
            lattice = expanded.lattice[row].to(dtype=torch.float64)
            fractional = expanded.fractional_coordinates[row, :count].to(dtype=torch.float64)
            per_step.append(pair_geometry(lattice, fractional, radius, max_reach))
    end = per_step[-1]
    entry: dict[str, object] = {
        "space_group": cover.space_group,
        "nu": cover.nu,
        "orbits": orbits,
        "atoms": count,
    }
    if end is None:
        entry["end"] = None
    else:
        distance, rho, reduced = end
        classes = class_minima(distance, rho, orbit_of, len(orbits))
        lattice_min = float(distance[0, 0])
        best = min(classes.items(), key=lambda item: item[1][0], default=None)
        closest = None
        if best is not None and math.isfinite(best[1][0]):
            (low, high), (value, smooth, smooth_max, i, j) = best
            closest = {
                "class": [low, high],
                "mates": low == high,
                "atoms": [i, j],
                "atomic_numbers": [numbers[i], numbers[j]],
                "wyckoff": [orbits[low][0], orbits[high][0]],
                "r": significant(value),
                "rho": significant(smooth),
                "rho_max": significant(smooth_max),
                "delta": [significant(float(v)) for v in reduced[i, j]],
            }
        first, second = np.triu_indices(count, 1)
        entry["end"] = {
            "closest": closest,
            "shortest": significant(
                min(lattice_min, best[1][0] if best is not None else math.inf)
            ),
            "lattice_min": significant(lattice_min),
            "contacts_at_most_0.5": int(np.count_nonzero(distance[first, second] <= 0.5)),
            "classes": [
                [low, high, significant(value), significant(smooth), significant(smooth_max)]
                for (low, high), (value, smooth, smooth_max, _, _) in sorted(classes.items())
                if value <= keep_end
            ],
            "half_lattice_counts": half_lattice_counts(distance, rho, reduced, thresholds),
        }
    if len(steps) == 1:
        return entry
    missing = (math.nan, math.nan, math.nan)
    tracks: dict[tuple[int, int], list[tuple[float, float, float]]] = {}
    closest_r: list[float | None] = []
    lattice_min_r: list[float | None] = []
    halves: list[list[int] | None] = []
    unresolved: list[int] = []
    for step, geometry in zip(steps, per_step, strict=True):
        if geometry is None:
            unresolved.append(step)
            closest_r.append(None)
            lattice_min_r.append(None)
            halves.append(None)
            for key in tracks:
                tracks[key].append(missing)
            continue
        distance, rho, reduced = geometry
        classes = class_minima(distance, rho, orbit_of, len(orbits))
        for key, (value, smooth, smooth_max, _, _) in classes.items():
            tracks.setdefault(key, [missing] * (len(closest_r))).append(
                (value, smooth, smooth_max)
            )
        for key in tracks:
            if key not in classes:
                tracks[key].append(missing)
        lattice_min_r.append(significant(float(distance[0, 0])))
        values = [item[0] for item in classes.values()]
        closest_r.append(significant(min([float(distance[0, 0]), *values])))
        halves.append(half_lattice_counts(distance, rho, reduced, thresholds))
    kept = {}
    for (low, high), series in tracks.items():
        values = np.asarray([item[0] for item in series], dtype=np.float64)
        finite = values[np.isfinite(values)]
        end_value = values[-1]
        if (np.isfinite(end_value) and end_value <= keep_end) or (
            finite.size and finite.min() <= keep_min
        ):
            kept[f"{low}-{high}"] = {
                "r": [significant(item[0]) for item in series],
                "rho": [significant(item[1]) for item in series],
                "rho_max": [significant(item[2]) for item in series],
            }
    entry["trajectory"] = {
        "closest_r": closest_r,
        "lattice_min": lattice_min_r,
        "half_lattice_counts": halves,
        "unresolved_steps": unresolved,
        "classes": kept,
    }
    return entry


def compare(new: dict[str, object], old: dict[str, object], fields) -> list[str]:  # type: ignore[no-untyped-def]
    return [field for field in fields if new.get(field) != old.get(field)]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_run_arguments(parser)  # --run is the cover run
    parser.add_argument("--field-run", type=Path, required=True)
    parser.add_argument("--field-label", default="after-geometry")
    parser.add_argument(
        "--stored", type=Path, required=True, help="this field's stored replay JSON"
    )
    parser.add_argument("--targets", type=Path, required=True, help="attempts to trace (JSON)")
    parser.add_argument("--output", type=Path, required=True, help="result JSON (new file)")
    parser.add_argument("--candidates", type=int, default=20)
    parser.add_argument("--oracle-draws", type=int, default=20)
    parser.add_argument("--max-cases", type=int, default=None)
    parser.add_argument("--radius", type=float, default=4.0)
    parser.add_argument("--max-reach", type=int, default=8)
    parser.add_argument("--keep-end", type=float, default=3.0)
    parser.add_argument("--keep-min", type=float, default=1.5)
    args = parser.parse_args()

    stored = json.loads(args.stored.read_text())
    wanted = json.loads(args.targets.read_text())
    cover_run = args.run.absolute()
    field_run = args.field_run.absolute()
    if Path(stored["field_run"]).resolve() != field_run.resolve():
        raise DomainError("--stored belongs to another field run")
    if stored["candidates"] != args.candidates or stored["oracle_draws"] != args.oracle_draws:
        raise DomainError("--candidates and --oracle-draws must equal the stored run's")
    arguments = json.loads((cover_run / "training-contract.json").read_text())["arguments"]
    seed = int(arguments["seed"]) + 500
    if seed != stored["protocol_seed"]:
        raise DomainError("the protocol seed differs from the stored run's")
    device = configure_torch(args.device, args.cpu_threads, seed)
    pins = dict(
        certification=args.certification,
        certification_fingerprint=args.certification_fingerprint,
        device=device,
        geometry_sampler="mixed-v1",
    )
    covers = load_run(cover_run, args.checkpoint_label, **pins)
    field = load_run(field_run, args.field_label, catalogue=covers.catalogue, **pins)
    if field.checkpoint_sha256 != stored["field_checkpoint_sha256"]:
        raise DomainError("the field checkpoint differs from the stored run's")
    if covers.checkpoint_sha256 != stored["cover_checkpoint_sha256"]:
        raise DomainError("the cover checkpoint differs from the stored run's")
    generator = field.generator
    generator._cover_sampler = covers.generator._cover_sampler
    generator.select_map_sample = replay._no_likelihood  # type: ignore[method-assign]
    model = generator.model
    transformer = model.geometry_field.transformer.config
    epsilon = float(transformer.normalization_epsilon)
    reference = float(transformer.reference_length)
    thresholds = (
        switch_threshold(0.9, REFERENCE_SMOOTHING, epsilon, reference),
        switch_threshold(1.0 / math.sqrt(2.0), REFERENCE_SMOOTHING, epsilon, reference),
    )
    numerics = model.scientific_config.numerics
    steps_all = tuple(range(int(numerics.ode_steps) + 1))
    progress(f"switch thresholds at s = {REFERENCE_SMOOTHING}: rho <= {thresholds}")

    with open_research_store(args) as store:
        record_ids = partition_record_ids(covers, store, "test")
        selected = tuple(sorted(record_ids))
        if selected != tuple(stored["record_ids"]):
            raise DomainError("the test records differ from the stored run's")
        if args.max_cases is not None:
            selected = selected[: args.max_cases]
        dataset = StoredTrainingDataset(
            store,
            covers.model.composition_codec,
            covers.catalogue,
            domain=covers.model.scientific_config.domain,
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

    stored_replay = {case["record_id"]: case for case in stored["replay_cases"]}
    stored_oracle = {case["record_id"]: case for case in stored["oracle_cases"]}
    recorder = Recorder()
    install(generator, recorder)
    describe = dict(
        model=model,
        radius=args.radius,
        max_reach=args.max_reach,
        thresholds=thresholds,
        keep_end=args.keep_end,
        keep_min=args.keep_min,
    )
    started = time.perf_counter()
    cases: list[dict[str, object]] = []
    mismatches: list[list[object]] = []
    worst_gap = 0.0
    times: list[float] | None = None
    generator.model.eval()
    for target in targets:
        clock = time.perf_counter()
        request_seed = case_seed(seed, target.record_id)
        recorder.clear()
        case = evaluate_case(
            generator, target, track=replay.TRACK, candidates=args.candidates, seed=request_seed
        )
        replay_generators = list(recorder.created)
        traces = recorder.traces
        own = replay.oracle_case(generator, target, draws=args.oracle_draws, seed=request_seed)
        oracle_generators = recorder.created[len(replay_generators) :]
        if times is None and recorder.records:
            times = [float(value) for value in recorder.records[0]["times"].cpu()]
        if traces is None or len(replay_generators) != 3 * args.candidates:
            raise InvariantViolation("unexpected replay generator count")
        if len(oracle_generators) != args.oracle_draws:
            raise InvariantViolation("unexpected oracle generator count")
        differing = compare(case, stored_replay[target.record_id], REPLAY_FIELDS)
        differing += [
            f"oracle/{name}"
            for name in compare(own, stored_oracle[target.record_id], ORACLE_FIELDS)
        ]
        if differing:
            mismatches.append([target.record_id, differing])
        traced_replay = set(wanted["replay"].get(target.record_id, ()))
        traced_oracle = set(wanted["oracle"].get(target.record_id, ()))
        entry: dict[str, object] = {
            "record_id": target.record_id,
            "seed": request_seed,
            "reproduced": not differing,
            "replay": [],
            "oracle": [],
        }
        for kind, items, shortest, traced in (
            (
                "replay",
                [replay_generators[3 * index + 2] for index in range(args.candidates)],
                case["shortest_distances"],
                traced_replay,
            ),
            ("oracle", oracle_generators, own["shortest_distances"], traced_oracle),
        ):
            for index, item in enumerate(items):
                found = recorder.rows.get(id(item))
                attempt: dict[str, object] = {
                    "index": index,
                    "shortest": shortest[index],
                    "traced": index in traced,
                }
                if kind == "replay":
                    trace = traces[index]
                    attempt["failed_stage"] = trace.failed_stage
                    attempt["error"] = trace.error
                if found is None:
                    attempt["solved"] = False
                else:
                    record, row = found
                    steps = steps_all if index in traced else (steps_all[-1],)
                    attempt["solved"] = True
                    attempt.update(describe_row(record, row, steps=steps, **describe))
                    end = attempt.get("end")
                    decoded = shortest[index]
                    if decoded is not None and end is not None and end["shortest"] is not None:
                        gap = abs(float(end["shortest"]) - float(shortest[index]))
                        attempt["plan_minus_decoded"] = significant(
                            float(end["shortest"]) - float(shortest[index])
                        )
                        worst_gap = max(worst_gap, gap)
                entry[kind].append(attempt)
        entry["seconds"] = round(time.perf_counter() - clock, 2)
        cases.append(entry)
        print(
            json.dumps(
                {
                    "record_id": target.record_id[:12],
                    "reproduced": not differing,
                    "differing": differing,
                    "traced": [len(traced_replay), len(traced_oracle)],
                    "seconds": entry["seconds"],
                }
            ),
            flush=True,
        )
    result = {
        "schema": SCHEMA,
        "cover_run": str(cover_run),
        "cover_checkpoint_sha256": covers.checkpoint_sha256,
        "field_run": str(field_run),
        "field_checkpoint_sha256": field.checkpoint_sha256,
        "field_options": field.arguments.get("options", "default"),
        "stored": str(args.stored.absolute()),
        "protocol_seed": seed,
        "candidates": args.candidates,
        "oracle_draws": args.oracle_draws,
        "record_ids": [target.record_id for target in targets],
        "radius": args.radius,
        "max_reach": args.max_reach,
        "keep_end": args.keep_end,
        "keep_min": args.keep_min,
        "reference_smoothing": REFERENCE_SMOOTHING,
        "half_lattice_gap": HALF_LATTICE_GAP,
        "half_lattice_rho_thresholds": list(thresholds),
        "transformer": {
            "reference_length": reference,
            "normalization_epsilon": epsilon,
            "pair_carrier_smoothing_length": transformer.pair_carrier_smoothing_length,
        },
        "expansion_cell": model.scientific_config.model.geometry_expansion_cell,
        "ode_steps": int(numerics.ode_steps),
        "ode_terminal_time": float(numerics.ode_terminal_time),
        "step_times": times,
        "code": code_fingerprint(),
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "replay_module_sha256": hashlib.sha256(Path(replay.__file__).read_bytes()).hexdigest(),
        "targets_sha256": hashlib.sha256(args.targets.read_bytes()).hexdigest(),
        "device": str(device),
        "torch": torch.__version__,
        "cases_reproduced": sum(1 for case in cases if case["reproduced"]),
        "cases": len(cases),
        "mismatches": mismatches,
        "largest_plan_minus_decoded": significant(worst_gap),
        "wall_seconds": round(time.perf_counter() - started, 1),
        "records": cases,
    }
    runner._write_new(args.output.absolute(), result)
    brief = {
        key: result[key]
        for key in ("cases", "cases_reproduced", "mismatches", "largest_plan_minus_decoded")
    }
    print(json.dumps(brief), flush=True)


if __name__ == "__main__":
    main()
