"""CSP measurement of a finished ``train_research_v7`` run from its checkpoint.

The model is rebuilt from the run's ``training-contract.json`` (profile, domain
profile and data contract), its scientific manifest must equal the stored one,
and the weights are loaded from the labelled checkpoint with the digest written
by the run.  The frozen partition of the contract is then evaluated with the
protocol of ``csp_evaluation_v7``.  Every finished case is appended to a
``.jsonl`` file next to the output, so an interrupted evaluation resumes without
recomputing or changing a case (``--resume``).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import train_research_v7 as runner
from csp_evaluation_v7 import (
    TRACKS,
    collect_targets,
    evaluate_csp,
    protocol,
)

import crystalatlas
from crystalatlas.benchmarking import _require_deterministic_cuda_environment
from crystalatlas.certification import CrystallographyCertification
from crystalatlas.dataset_identity import DeduplicationLevel
from crystalatlas.dataset_provenance import DatasetUseRequest, evaluate_dataset_use
from crystalatlas.dataset_splits import SplitMode
from crystalatlas.dataset_storage import DatasetStore
from crystalatlas.errors import DomainError
from crystalatlas.full_catalogue import build_full_catalogue
from crystalatlas.learned import (
    GEOMETRY_SAMPLERS,
    CrystalAtlasModel,
    LearnedCrystalGenerator,
)
from crystalatlas.trainer import TrainingPhase
from crystalatlas.training_data import StoredTrainingDataset, split_partition_record_ids

_STARTED = time.monotonic()


def progress(message: str) -> None:
    """One stderr line with the seconds since the script was imported."""

    print(f"[{time.monotonic() - _STARTED:8.1f} s] {message}", file=sys.stderr, flush=True)


def code_fingerprint() -> dict[str, object]:
    """SHA-256 over the package sources and the three scripts of this measurement."""

    package = Path(crystalatlas.__file__).resolve().parent
    scripts = Path(__file__).resolve().parent
    files = [(path.relative_to(package.parent), path) for path in package.rglob("*.py")]
    files += [
        (Path("scripts") / name, scripts / name)
        for name in (
            "csp_evaluation_v7.py",
            "evaluate_csp_research_v7.py",
            "train_research_v7.py",
        )
    ]
    digest = hashlib.sha256()
    for relative, path in sorted(files, key=lambda item: item[0].as_posix()):
        digest.update(relative.as_posix().encode() + b"\0")
        digest.update(hashlib.sha256(path.read_bytes()).hexdigest().encode() + b"\n")
    return {
        "sha256": digest.hexdigest(),
        "files": len(files),
        "rule": "sha256 of 'relative-path NUL sha256(file) LF' over sorted paths",
    }


def add_run_arguments(parser: argparse.ArgumentParser) -> None:
    """Arguments that identify a finished run, its store and its certification."""

    parser.add_argument("--run", type=Path, required=True, help="train_research_v7 output")
    parser.add_argument("--store", type=Path, required=True)
    for name in ("source", "build", "storage", "certification"):
        parser.add_argument(f"--{name}-fingerprint", type=runner._digest, required=True)
    parser.add_argument("--certification", type=Path, required=True)
    parser.add_argument("--checkpoint-label", default="after-geometry")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--cpu-threads", type=int, default=2)


def configure_torch(device_name: str, cpu_threads: int, seed: int) -> torch.device:
    """Deterministic float32 execution on the requested device, seeded."""

    device = torch.device(device_name)
    if device.type == "cuda":
        if not torch.cuda.is_available() or device.index is None:
            raise DomainError("request an available explicit CUDA index such as cuda:0")
        _require_deterministic_cuda_environment(device)
        torch.cuda.set_device(device)
    torch.set_num_threads(cpu_threads)
    torch.use_deterministic_algorithms(True, warn_only=False)
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    return device


@dataclass(frozen=True)
class LoadedRun:
    """A finished run rebuilt from its contract with one labelled checkpoint bound."""

    directory: Path
    contract: dict[str, object]
    arguments: dict[str, object]
    label: str
    checkpoint_sha256: str
    catalogue: object
    model: CrystalAtlasModel
    generator: LearnedCrystalGenerator


def load_run(
    run: Path,
    label: str,
    *,
    certification: Path,
    certification_fingerprint: str,
    device: torch.device,
    geometry_sampler: str = "bucket-v1",
    catalogue: object | None = None,
) -> LoadedRun:
    """Rebuild the run's model, require its stored manifest, and bind the checkpoint.

    ``catalogue`` reuses a catalogue another call of this function built; it is
    certified against the receipt again.
    """

    run = run.absolute()
    contract = json.loads((run / "training-contract.json").read_text())
    arguments = contract["arguments"]
    checkpoint_sha256 = json.loads((run / f"{label}-evaluation.json").read_text())[
        "checkpoint_sha256"
    ]
    if catalogue is None:
        progress("building the catalogue")
        catalogue = build_full_catalogue()
        progress("catalogue built")
    receipt = CrystallographyCertification.read_json(certification)
    if receipt.receipt_fingerprint != certification_fingerprint:
        raise DomainError("crystallographic certification disagrees with external pin")
    receipt.require_for(catalogue)
    progress("catalogue certified")
    # Runs older than the option mechanism have no "options" argument.
    config = runner.model_config(
        arguments["profile"], arguments["domain"], arguments.get("options", "default")
    )
    model = CrystalAtlasModel(config, catalogue=catalogue).to(device=device)
    stored_manifest = contract["scientific_manifest"]
    manifest = model.scientific_manifest(data_contract=stored_manifest["data_contract"])
    if manifest.to_dict() != stored_manifest:
        raise DomainError(
            "the rebuilt model's scientific manifest differs from the run's contract"
        )
    generator = LearnedCrystalGenerator.from_checkpoint(
        model,
        run / f"{label}.pt",
        scientific_manifest=manifest,
        expected_sha256=checkpoint_sha256,
        map_location=device,
        geometry_sampler=geometry_sampler,
    )
    progress(f"model rebuilt and checkpoint {label} bound")
    return LoadedRun(
        directory=run,
        contract=contract,
        arguments=arguments,
        label=label,
        checkpoint_sha256=checkpoint_sha256,
        catalogue=catalogue,
        model=model,
        generator=generator,
    )


@contextmanager
def open_research_store(args: argparse.Namespace) -> Iterator[DatasetStore]:
    """The pinned store, after the same use and certification checks as training."""

    with DatasetStore.open(
        args.store,
        expected_source_manifest_fingerprint=args.source_fingerprint,
        expected_build_manifest_fingerprint=args.build_fingerprint,
        expected_storage_fingerprint=args.storage_fingerprint,
    ) as store:
        if store.build_manifest.crystallography_receipt_fingerprint != (
            args.certification_fingerprint
        ):
            raise DomainError("dataset was built under a different crystallography receipt")
        if store.build_manifest.deduplication_level is not DeduplicationLevel.STRUCTURE:
            raise DomainError("research evaluation requires STRUCTURE deduplication")
        if not store.build_manifest.use_request.train_model:
            raise DomainError("compiled dataset was not approved for model training")
        store.build_manifest.policy_decision.require_authorized()
        evaluate_dataset_use(
            store.source_manifest, DatasetUseRequest(train_model=True)
        ).require_authorized()
        yield store


def partition_record_ids(
    loaded: LoadedRun, store: DatasetStore, partition: str
) -> tuple[str, ...]:
    """The run's records of one partition, checked against the store's frozen split."""

    record_ids = tuple(loaded.contract["partitions"][partition])
    frozen = split_partition_record_ids(store, SplitMode(loaded.arguments["split"]), partition)
    # A smoke run trains on a prefix of each partition (--max-records).
    if record_ids != tuple(frozen[: len(record_ids)]):
        raise DomainError("the run's partition is not a prefix of the store's partition")
    return record_ids


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_run_arguments(parser)
    parser.add_argument("--output", type=Path, required=True, help="result JSON (new file)")
    parser.add_argument("--partition", choices=("train", "validation", "test"), default="test")
    parser.add_argument("--candidates", type=int, default=20)
    parser.add_argument(
        "--seed", type=int, default=None, help="default: the run's seed + 500, as the runner"
    )
    parser.add_argument("--tracks", default=",".join(TRACKS))
    parser.add_argument("--likelihood-steps", type=int, default=None)
    parser.add_argument(
        "--max-cases", type=int, default=None, help="first N records in record-id order"
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--geometry-sampler",
        choices=GEOMETRY_SAMPLERS,
        default="bucket-v1",
        help="bucket-v1: one solve per topology bucket; mixed-v1: fused padded solves",
    )
    args = parser.parse_args()

    tracks = tuple(args.tracks.split(","))
    if args.max_cases is not None and args.max_cases < 1:
        raise DomainError("--max-cases must be positive")
    run = args.run.absolute()
    arguments = json.loads((run / "training-contract.json").read_text())["arguments"]
    seed = int(arguments["seed"]) + 500 if args.seed is None else args.seed
    device = configure_torch(args.device, args.cpu_threads, seed)
    loaded = load_run(
        run,
        args.checkpoint_label,
        certification=args.certification,
        certification_fingerprint=args.certification_fingerprint,
        device=device,
        geometry_sampler=args.geometry_sampler,
    )
    label = loaded.label
    checkpoint_sha256 = loaded.checkpoint_sha256
    catalogue = loaded.catalogue
    model = loaded.model
    generator = loaded.generator
    with open_research_store(args) as store:
        record_ids = partition_record_ids(loaded, store, args.partition)
        # Cases are taken in record-id order, so a bounded run is a prefix of the
        # full one and only its records are re-audited.
        selected = tuple(sorted(record_ids))
        if args.max_cases is not None:
            selected = selected[: args.max_cases]
        dataset = StoredTrainingDataset(
            store,
            model.composition_codec,
            catalogue,
            domain=model.scientific_config.domain,
            record_ids=selected,
        )
        targets = collect_targets(
            model,
            runner._sampler(
                dataset,
                TrainingPhase.GEOMETRY,
                int(arguments["seed"]),
                int(arguments["batch_size"]),
            ),
        )
    if tuple(target.record_id for target in targets) != selected:
        raise DomainError("the evaluation targets are not the selected records")

    header = {
        "run": str(run),
        "checkpoint_label": label,
        "checkpoint_sha256": checkpoint_sha256,
        "partition": args.partition,
        "record_ids": [target.record_id for target in targets],
        "protocol": protocol(
            candidates=args.candidates,
            seed=seed,
            likelihood_steps=args.likelihood_steps,
            tracks=tracks,
        ),
        "code": code_fingerprint(),
        "geometry_sampler": generator.geometry_sampler,
        "device": str(device),
        "torch": torch.__version__,
    }
    output = args.output.absolute()
    progress = output.with_suffix(".jsonl")
    done: dict[tuple[str, str], dict[str, object]] = {}
    ignored_lines = 0
    if args.resume:
        text = progress.read_text()
        lines = text.splitlines()
        if not lines or json.loads(lines[0]) != header:
            raise DomainError("the progress file was written under another protocol or code")
        for line in lines[1:]:
            try:
                case = json.loads(line)
            except json.JSONDecodeError:
                # A line torn by an interruption: its case is computed again, and
                # the per-case seed makes the recomputation the same measurement.
                ignored_lines += 1
                continue
            done[(case["record_id"], case["track"])] = case
        if not text.endswith("\n"):
            with progress.open("a") as handle:
                handle.write("\n")
    else:
        with progress.open("x") as handle:
            handle.write(json.dumps(header, sort_keys=True, allow_nan=False) + "\n")

    def on_case(case) -> None:  # type: ignore[no-untyped-def]
        line = json.dumps(case, sort_keys=True, allow_nan=False)
        with progress.open("a") as handle:
            handle.write(line + "\n")
        brief = {
            key: case[key]
            for key in ("record_id", "track", "valid", "match_count", "distinct_tables")
        }
        brief.update(
            first=case["first"]["match"],
            map=case["map"]["match"],
            consensus=case["consensus"]["match"],
            seconds=case["seconds"],
        )
        print(json.dumps(brief, allow_nan=False), flush=True)

    started = time.perf_counter()
    result = evaluate_csp(
        generator,
        targets,
        candidates=args.candidates,
        seed=seed,
        tracks=tracks,
        likelihood_steps=args.likelihood_steps,
        done=done,
        on_case=on_case,
    )
    result.update(header)
    result["resumed_cases"] = len(done)
    result["ignored_progress_lines"] = ignored_lines
    result["wall_seconds_this_invocation"] = round(time.perf_counter() - started, 1)
    runner._write_new(output, result)
    print(json.dumps(result["summary"], allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
