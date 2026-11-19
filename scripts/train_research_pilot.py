"""Bounded, factor-first learning diagnostic on an authenticated real-data store.

This is not a production release gate or a materials-quality benchmark. Geometry
evaluation estimates a fixed-seed CFM risk, with positive endpoint noise; it does
not fabricate an exact-partition overfit or physical-structure matching pass.

The pilot explicitly expands the composition probability support to all 118
package elements: the frozen train-only subset includes ThSe3, outside the
default narrower DNG vocabulary. No selected rows are filtered or reranked.
Geometric domain bounds, physical tolerances, and numerical policies are unchanged.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from dataclasses import asdict, dataclass, replace
from hashlib import sha256
from pathlib import Path

import numpy as np
import torch

from crystalatlas import __version__
from crystalatlas.benchmarking import (
    _require_deterministic_cuda_environment,
    _runtime_manifest,
)
from crystalatlas.certification import CrystallographyCertification
from crystalatlas.config import CrystalAtlasConfig, DomainConfig, ModelConfig
from crystalatlas.dataset_identity import DeduplicationLevel
from crystalatlas.dataset_provenance import DatasetUseRequest, evaluate_dataset_use
from crystalatlas.dataset_splits import SplitMode
from crystalatlas.dataset_storage import DatasetStore
from crystalatlas.errors import DomainError
from crystalatlas.full_catalogue import build_full_catalogue
from crystalatlas.learned import CrystalAtlasModel
from crystalatlas.provenance import ProvenanceRecord, sha256_file
from crystalatlas.trainer import (
    ComponentModels,
    PrecisionMode,
    ScalableTrainer,
    StagedTrainingSchedule,
    TrainerConfig,
    TrainingPhase,
)
from crystalatlas.training_data import (
    DeterministicBucketBatchSampler,
    StoredTrainingDataset,
    TrainingMeasure,
    split_partition_record_ids,
)
from crystalatlas.training_objectives import (
    CrystalAtlasTrainingObjective,
    ExactBatchCrystallographicAudit,
    ObjectiveDataMeasure,
)

PROFILE = "research-small-invariant-global-all118-v2"
PHASES = (TrainingPhase.COMPOSITION, TrainingPhase.COVER, TrainingPhase.GEOMETRY)
METRICS = ("composition_nll", "cover_nll", "cfm_regression")


def _json_bytes(value: object) -> bytes:
    return (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()


def _write_new(path: Path, value: object) -> None:
    with path.open("xb") as handle:
        handle.write(_json_bytes(value))


def _digest(value: str) -> str:
    if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise argparse.ArgumentTypeError("expected lowercase SHA-256 fingerprint")
    return value


@dataclass(frozen=True)
class PilotSettings:
    records: int = 8
    epochs_per_factor: int = 8
    max_optimizer_steps: int = 200
    seed: int = 20261012
    evaluation_seed: int = 87123
    evaluation_replicates: int = 4
    learning_rate: float = 1.0e-4

    def __post_init__(self) -> None:
        for name, maximum in (
            ("records", 8),
            ("epochs_per_factor", 64),
            ("max_optimizer_steps", 200),
            ("evaluation_replicates", 8),
        ):
            value = getattr(self, name)
            if type(value) is not int or not 1 <= value <= maximum:
                raise DomainError(f"{name} must be an integer in [1,{maximum}]")
        for name in ("seed", "evaluation_seed"):
            value = getattr(self, name)
            if type(value) is not int or not 0 <= value < 2**32 - 1000:
                raise DomainError(f"{name} must be a nonnegative bounded 32-bit seed")
        if not math.isfinite(self.learning_rate) or self.learning_rate <= 0:
            raise DomainError("learning_rate must be finite and positive")

    def required_steps(self, unique_compositions: int) -> int:
        if not 1 <= unique_compositions <= self.records:
            raise DomainError("invalid unique-composition count")
        steps = self.epochs_per_factor * (unique_compositions + 2 * self.records)
        if steps > self.max_optimizer_steps:
            raise DomainError(
                f"planned {steps} optimizer steps exceed budget {self.max_optimizer_steps}"
            )
        return steps


def pilot_model_config() -> CrystalAtlasConfig:
    """Reduce capacity; explicitly expand DNG support without changing geometry."""
    domain = DomainConfig(lattice_chart="ball-v1")  # the frozen pilot geometry
    return CrystalAtlasConfig(
        domain=replace(domain, dng_element_vocabulary=domain.element_vocabulary),
        model=ModelConfig(
            composition_width=64,
            composition_layers=2,
            selector_width=64,
            selector_layers=2,
            geometry_scalar_width=64,
            geometry_pair_width=32,
            geometry_vector_width=16,
            geometry_layers=2,
            attention_heads=4,
            radial_features=16,
        ),
    )


def select_record_ids(train_ids: tuple[str, ...], settings: PilotSettings) -> tuple[str, ...]:
    if len(set(train_ids)) != len(train_ids) or len(train_ids) < settings.records:
        raise DomainError("train partition lacks enough unique records for the frozen subset")

    def rank(record_id: str) -> tuple[str, str]:
        payload = ["crystalatlas.research-pilot-subset.v1", settings.seed, record_id]
        return sha256(_json_bytes(payload)).hexdigest(), record_id

    return tuple(sorted(train_ids, key=rank)[: settings.records])


def _sampler(dataset: StoredTrainingDataset, phase: TrainingPhase, seed: int):
    return DeterministicBucketBatchSampler(
        dataset,
        batch_size=1,
        seed=seed,
        measure=(
            TrainingMeasure.UNIQUE_COMPOSITION
            if phase is TrainingPhase.COMPOSITION
            else TrainingMeasure.STRUCTURE
        ),
    )


def evaluate_factors(model, dataset, objective, audit, settings: PilotSettings):
    """Same finite Monte Carlo sample bank each time; preserve training RNG/modes."""
    device = next(model.parameters()).device
    cuda_devices = [device.index] if device.type == "cuda" else []
    modes = tuple((module, module.training) for module in model.modules())
    components = ComponentModels(
        model,
        model.composition_model,
        model.cover_selector,
        model.geometry_field,
    )
    result: dict[str, object] = {}
    try:
        model.eval()
        with torch.no_grad(), torch.random.fork_rng(devices=cuda_devices):
            for phase, metric in zip(PHASES, METRICS, strict=True):
                replicates = (
                    settings.evaluation_replicates if phase is TrainingPhase.GEOMETRY else 1
                )
                replicate_means = []
                for replicate in range(replicates):
                    seed = settings.evaluation_seed + replicate
                    # torch.manual_seed seeds every CUDA device. Seed only the
                    # CPU and selected device whose states are forked here.
                    torch.random.default_generator.manual_seed(seed)
                    if device.type == "cuda":
                        with torch.cuda.device(device):
                            torch.cuda.manual_seed(seed)
                    values = []
                    for batch in _sampler(dataset, phase, settings.evaluation_seed):
                        audit(batch, torch.float64).require_passed()
                        scalar = getattr(objective(components, batch, phase), metric)
                        value = float(scalar.item())
                        if not math.isfinite(value):
                            raise DomainError(f"nonfinite evaluation {metric}")
                        values.append(value)
                    replicate_means.append(math.fsum(values) / len(values))
                result[metric] = math.fsum(replicate_means) / len(replicate_means)
                result[f"{metric}_replicate_means"] = replicate_means
    finally:
        for module, was_training in modes:
            module.training = was_training
    result["scope"] = "training-subset-only; no held-out or physical-quality claim"
    return result


def train_factors(model, dataset, settings: PilotSettings, output: Path, data_contract: dict):
    """Execute the predeclared finite schedule; never extend it based on results."""
    if len(dataset) != settings.records:
        raise DomainError("training view count differs from the frozen experiment")
    required = settings.required_steps(dataset.unique_composition_count)
    objective = CrystalAtlasTrainingObjective(model, ObjectiveDataMeasure.from_dataset(dataset))
    audit = ExactBatchCrystallographicAudit(model)
    schedule = StagedTrainingSchedule(*(settings.epochs_per_factor for _ in PHASES))
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=settings.learning_rate, weight_decay=0.01
    )
    trainer = ScalableTrainer(
        model,
        optimizer,
        schedule,
        config=TrainerConfig(
            precision=PrecisionMode.FLOAT32,
            accumulation_steps=1,
            max_gradient_norm=1.0,
            ema_decay=None,
        ),
    )
    scientific = model.scientific_manifest(
        data_contract={
            **data_contract,
            "objective": objective.contract,
            "pilot_settings": asdict(settings),
            "model_profile": PROFILE,
        }
    )
    provenance = ProvenanceRecord.create(
        config=model.scientific_config,
        random_seed=settings.seed,
        task="bounded-real-data-factor-learning-diagnostic",
        package_version=__version__,
    )
    _write_new(
        output / "training-contract.json",
        {
            "scientific_manifest": scientific.to_dict(),
            "trainer": trainer.config.to_dict(),
            "schedule": schedule.to_dict(),
            "objective": objective.contract,
            "required_optimizer_steps": required,
            "optimizer": {
                "type": "AdamW",
                "learning_rate": settings.learning_rate,
                "weight_decay": 0.01,
                "betas": [0.9, 0.999],
                "eps": 1e-8,
            },
        },
    )
    checkpoint_receipts = []
    evaluations = []
    epoch_reports = []
    sampler = _sampler(dataset, PHASES[0], settings.seed + 101)

    def checkpoint_and_evaluate(label: str, active_sampler):
        path = output / f"{label}.pt"
        if path.exists():
            raise DomainError("refusing to replace an existing checkpoint")
        receipt = trainer.save_checkpoint(
            path,
            scientific_config=model.scientific_config,
            scientific_manifest=scientific,
            provenance=provenance,
            sampler=active_sampler,
            metadata={"label": label},
        )
        assert receipt is not None
        checkpoint_receipts.append({"path": path.name, "sha256": receipt.sha256})
        evaluation = {
            "label": label,
            "optimizer_step": trainer.cursor.optimizer_step,
            "checkpoint_sha256": receipt.sha256,
            "metrics": evaluate_factors(model, dataset, objective, audit, settings),
        }
        evaluations.append(evaluation)
        _write_new(output / f"{label}-evaluation.json", evaluation)
        print(json.dumps(evaluation, allow_nan=False), flush=True)

    checkpoint_and_evaluate("baseline", sampler)
    started = time.perf_counter()
    for index, phase in enumerate(PHASES):
        if trainer.current_phase is not phase:
            raise DomainError("unexpected staged trainer phase")
        # train_epoch advances this SAME sampler inside the phase. A fresh
        # sampler starts only when the trainer changes factor and phase epoch.
        for _ in range(settings.epochs_per_factor):
            report = trainer.train_epoch(sampler, objective, crystallographic_audit=audit)
            if trainer.cursor.optimizer_step > settings.max_optimizer_steps:
                raise DomainError("optimizer step budget exceeded")
            payload = asdict(report)
            payload["phase"] = report.phase.value
            epoch_reports.append(payload)
            _write_new(output / f"epoch-{report.epoch:03d}.json", payload)
            print(json.dumps({"epoch": payload}, allow_nan=False), flush=True)
        sampler = (
            _sampler(dataset, PHASES[index + 1], settings.seed + 102 + index)
            if index + 1 < len(PHASES)
            else None
        )
        checkpoint_and_evaluate(f"after-{phase.value}", sampler)
    if trainer.cursor.optimizer_step != required:
        raise DomainError("observed optimizer count differs from the frozen plan")
    result = {
        "schema": "crystalatlas.bounded-research-pilot-result.v1",
        "status": "completed",
        "optimizer_steps": trainer.cursor.optimizer_step,
        "wall_seconds_after_baseline": time.perf_counter() - started,
        "checkpoints": checkpoint_receipts,
        "evaluations": evaluations,
        "epochs": epoch_reports,
        "tiny_overfit_gate_passed": None,
        "physical_generation_quality_evaluated": False,
        "held_out_generalization_evaluated": False,
        "joint_training_performed": False,
        "limitation": "Loss diagnostics only; a frozen physical matcher remains required.",
    }
    _write_new(output / "result.json", result)
    return result


def _fresh_output(path: Path, store: Path) -> Path:
    absolute = path.absolute()
    if any(part.is_symlink() for part in (absolute, *absolute.parents)):
        raise DomainError("output path must not traverse symlinks")
    resolved = absolute.resolve()
    root = store.resolve(strict=True)
    if root == resolved or root in resolved.parents or resolved in root.parents:
        raise DomainError("output must not overlap the authenticated store")
    resolved.mkdir(parents=True, exist_ok=False)
    return resolved


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--store", type=Path, required=True)
    for name in ("source", "build", "storage", "certification", "software-artifact"):
        parser.add_argument(f"--{name}-fingerprint", type=_digest, required=True)
    parser.add_argument("--certification", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--require-cuda", action="store_true")
    parser.add_argument(
        "--preflight-only",
        action="store_true",
        help="validate authenticated records and freeze the plan without optimizer updates",
    )
    parser.add_argument("--records", type=int, default=8)
    parser.add_argument("--epochs-per-factor", type=int, default=8)
    parser.add_argument("--max-optimizer-steps", type=int, default=200)
    parser.add_argument("--seed", type=int, default=20261012)
    parser.add_argument("--evaluation-seed", type=int, default=87123)
    parser.add_argument("--evaluation-replicates", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--cpu-threads", type=int, default=1)
    args = parser.parse_args()
    settings = PilotSettings(
        **{name: getattr(args, name) for name in PilotSettings.__dataclass_fields__}
    )
    if args.cpu_threads < 1:
        raise DomainError("cpu_threads must be positive")
    device = torch.device(args.device)
    if device.type not in {"cpu", "cuda"} or (args.require_cuda and device.type != "cuda"):
        raise DomainError(
            "pilot requires explicit CPU or CUDA device; CUDA flag cannot fall back"
        )
    if device.type == "cuda":
        if not torch.cuda.is_available() or device.index is None:
            raise DomainError("request an available explicit CUDA index such as cuda:0")
        _require_deterministic_cuda_environment(device)
        torch.cuda.set_device(device)
    torch.set_num_threads(args.cpu_threads)
    torch.use_deterministic_algorithms(True, warn_only=False)
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    random.seed(settings.seed)
    np.random.seed(settings.seed)
    torch.manual_seed(settings.seed)
    catalogue = build_full_catalogue()
    certification = CrystallographyCertification.read_json(args.certification)
    if certification.receipt_fingerprint != args.certification_fingerprint:
        raise DomainError("crystallographic certification disagrees with external pin")
    certification.require_for(catalogue)
    with DatasetStore.open(
        args.store,
        expected_source_manifest_fingerprint=args.source_fingerprint,
        expected_build_manifest_fingerprint=args.build_fingerprint,
        expected_storage_fingerprint=args.storage_fingerprint,
    ) as store:
        if (
            store.build_manifest.crystallography_receipt_fingerprint
            != args.certification_fingerprint
        ):
            raise DomainError("dataset was built under a different crystallography receipt")
        if store.build_manifest.deduplication_level is not DeduplicationLevel.STRUCTURE:
            raise DomainError("real-data pilot requires STRUCTURE deduplication")
        if not store.build_manifest.use_request.train_model:
            raise DomainError("compiled dataset was not approved for model training")
        store.build_manifest.policy_decision.require_authorized()
        evaluate_dataset_use(
            store.source_manifest, DatasetUseRequest(train_model=True)
        ).require_authorized()
        if set(store.split_manifests) != set(SplitMode):
            raise DomainError("all four split views must be frozen before training")
        train_ids = split_partition_record_ids(store, SplitMode.COMPOSITION, "train")
        selected = select_record_ids(train_ids, settings)
        config = pilot_model_config()
        model = CrystalAtlasModel(config, catalogue=catalogue).to(
            device=device, dtype=torch.float32
        )
        dataset = StoredTrainingDataset(
            store, model.composition_codec, catalogue, domain=config.domain, record_ids=selected
        )
        settings.required_steps(dataset.unique_composition_count)
        contract = {
            "source_manifest_fingerprint": args.source_fingerprint,
            "build_manifest_fingerprint": args.build_fingerprint,
            "storage_manifest_fingerprint": args.storage_fingerprint,
            "crystallography_receipt_fingerprint": args.certification_fingerprint,
            "software_artifact_sha256_external_claim": args.software_artifact_fingerprint,
            "runner_sha256": sha256_file(__file__),
            "dataset_fingerprint": dataset.fingerprint,
            "split_mode": SplitMode.COMPOSITION.value,
            "partition": "train",
            "split_manifest_fingerprint": store.split_manifest_for(
                SplitMode.COMPOSITION
            ).manifest_fingerprint,
            "selection_rule": "sha256-rank-v1; no geometry/energy/model-result filtering",
            "record_ids": list(dataset.record_ids),
            "selection_rank_order": list(selected),
            "input_count": store.build_manifest.input_count,
            "preprocessing_rejection_count": store.build_manifest.rejection_count,
            "domain_claim": (
                "accepted preprocessing subset only; not representative MP20 coverage"
            ),
        }
        output = _fresh_output(args.output, args.store)
        _write_new(
            output / "experiment.json",
            {
                "schema": "crystalatlas.bounded-research-pilot.v1",
                "data_contract": contract,
                "settings": asdict(settings),
                "config": config.to_dict(),
                "model_profile": PROFILE,
                "parameter_count": sum(p.numel() for p in model.parameters()),
                "runtime": _runtime_manifest(device, cpu_threads=args.cpu_threads),
                "batch_size": 1,
                "accumulation_steps": 1,
                "master_dtype": "float32",
                "autocast": False,
                "crystallography_audit_dtype": "float64",
                "physical_matcher": None,
                "overfit_thresholds": None,
            },
        )
        if args.preflight_only:
            _write_new(
                output / "preflight.json",
                {
                    "status": "validated-not-trained",
                    "required_optimizer_steps": settings.required_steps(
                        dataset.unique_composition_count
                    ),
                    "training_launched": False,
                    "record_ids": list(dataset.record_ids),
                },
            )
            return
        train_factors(model, dataset, settings, output, contract)


if __name__ == "__main__":
    main()
