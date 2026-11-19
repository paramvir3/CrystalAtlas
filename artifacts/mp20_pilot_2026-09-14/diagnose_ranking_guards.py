"""Observe unchanged primitive ranking guards on three predeclared rejected rows."""
from __future__ import annotations

import inspect
import json
from pathlib import Path

import numpy as np

import crystalatlas.primitive_canonicalization as primitive
from crystalatlas.benchmark_data import read_benchmark_cif_csv, select_training_pilot
from crystalatlas.errors import AmbiguousPreprocessingError
from crystalatlas.full_catalogue import pinned_hall_numbers
from crystalatlas.ingestion import parse_cif_structure
from crystalatlas.standardization import FloatingStructure, SpglibPolicy, SpglibStandardizer

observed = []
original = primitive._guarded_bins


def traced_guard(values, quantum, uncertainty):
    try:
        return original(values, quantum, uncertainty)
    except AmbiguousPreprocessingError:
        scaled = np.asarray(values, dtype=np.float64).reshape(-1) / quantum
        bins = np.floor(scaled + 0.5)
        frame = inspect.currentframe()
        caller = frame.f_back
        observed.append({
            "values": np.asarray(values).tolist(), "quantum": quantum,
            "uncertainty": uncertainty, "uncertainty_in_bins": uncertainty / quantum,
            "distance_to_bin_wall": (0.5 - np.abs(scaled - bins)).tolist(),
            "caller_line": caller.f_lineno,
            "vector_if_present": caller.f_locals.get("vector"),
            "basis_if_present": caller.f_locals.get("basis"),
        })
        del frame, caller
        raise


def main():
    destination = Path(__file__).with_name("ranking_guard_observations.json")
    if destination.exists():
        raise ValueError("preserve existing diagnostic output")
    rows = read_benchmark_cif_csv(
        "data/benchmarks/mp20-cdvae-f857f598/train.csv", split="train",
        expected_sha256="133bab58dba02d316f9f91174839f17802dce648be4e8659d1a42b32b97fa01d",
        expected_count=27136,
    )
    selected = select_training_pilot(rows, count=512, seed=20260913)[:3]
    standardizer = SpglibStandardizer(pinned_hall_numbers(), SpglibPolicy())
    results = []
    primitive._guarded_bins = traced_guard
    try:
        for row in selected:
            raw = parse_cif_structure(row.cif)
            observed.clear()
            try:
                standardizer.canonicalize(FloatingStructure(
                    raw.lattice, raw.fractional_positions, raw.atomic_numbers,
                ))
                result = {"material_id": row.material_id, "status": "unexpected_pass"}
            except AmbiguousPreprocessingError as exc:
                result = {"material_id": row.material_id, "status": "reproduced_rejection",
                          "reason": str(exc), "observations": observed.copy()}
            results.append(result)
    finally:
        primitive._guarded_bins = original
    with destination.open("x", encoding="utf-8") as stream:
        json.dump({"unchanged_guard_called": True, "records": results}, stream,
                  sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps(results, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
