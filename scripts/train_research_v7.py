"""Factor-first training on an authenticated v7 research store, then a CSP match test.

Research runner (2026-10-30, program R3): trains the composition, cover and
geometry factors on every training record of the composition-disjoint split,
checkpoints after each phase with validation losses, and then measures
composition-conditioned CSP on the frozen test partition with the certified
physical matcher (protocol and selection rules: ``csp_evaluation_v7``).  No
energy model, no relaxation, no filtering; failed attempts stay in the
denominator.  This is a bounded research measurement on a small corpus, not a
benchmark claim.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import math
import random
import time
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path

import numpy as np
import torch
from csp_evaluation_v7 import TRACKS, collect_targets, evaluate_csp, print_case

from crystalatlas import __version__
from crystalatlas.benchmarking import _require_deterministic_cuda_environment
from crystalatlas.certification import CrystallographyCertification
from crystalatlas.config import CrystalAtlasConfig, DomainConfig, ModelConfig
from crystalatlas.dataset_identity import DeduplicationLevel
from crystalatlas.dataset_provenance import DatasetUseRequest, evaluate_dataset_use
from crystalatlas.dataset_splits import SplitMode
from crystalatlas.dataset_storage import DatasetStore
from crystalatlas.errors import DomainError
from crystalatlas.full_catalogue import build_full_catalogue
from crystalatlas.learned import CrystalAtlasModel, LearnedCrystalGenerator
from crystalatlas.provenance import ProvenanceRecord
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
    MixedTopologyBatchSampler,
    StoredTrainingDataset,
    TrainingMeasure,
    split_partition_record_ids,
)
from crystalatlas.training_objectives import (
    CFMDequantization,
    CrystalAtlasTrainingObjective,
    ExactBatchCrystallographicAudit,
    ObjectiveDataMeasure,
)

PHASES = (TrainingPhase.COMPOSITION, TrainingPhase.COVER, TrainingPhase.GEOMETRY)
METRICS = ("composition_nll", "cover_nll", "cfm_regression")
PROFILES = {
    # Small: the research-pilot capacity (fast; for smoke tests and tiny corpora).
    "research-small": dict(
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
    # Medium: the first real-data capacity for a few-hundred-record corpus.
    "research-medium": dict(
        composition_width=128,
        composition_layers=3,
        selector_width=128,
        selector_layers=3,
        geometry_scalar_width=128,
        geometry_pair_width=64,
        geometry_vector_width=32,
        geometry_layers=4,
        attention_heads=4,
        radial_features=32,
    ),
}
DOMAINS = {
    "v7": lambda: DomainConfig(max_primitive_atoms=20, max_raw_atoms=80),
    "legacy-v6": lambda: DomainConfig(lattice_chart="ball-v1"),
}


def _json_bytes(value: object) -> bytes:
    return (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()


def _write_new(path: Path, value: object) -> None:
    with path.open("xb") as handle:
        handle.write(_json_bytes(value))


def _digest(value: str) -> str:
    if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise argparse.ArgumentTypeError("expected lowercase SHA-256 fingerprint")
    return value


@dataclass(frozen=True, slots=True)
class ResearchOption:
    """One opt-in mechanism a run may declare with ``--options``.

    ``target`` names the object the fields configure (``model``:
    :class:`ModelConfig`; ``dequantization``: :class:`CFMDequantization`).  A
    mechanism with one declared scale names it in ``parameter``; ``name=value``
    then replaces that field with a finite float.
    """

    target: str
    fields: Mapping[str, object] = field(default_factory=dict)
    parameter: str | None = None


# Declaring an option changes one run and its recorded contract.  It changes no
# default of the library; the legacy configuration is the empty set.
OPTIONS: dict[str, ResearchOption] = {
    # Review 2026-10-29, M3: expand orbits to the primitive cell (N / I_c atoms).
    "primitive-cell": ResearchOption("model", {"geometry_expansion_cell": "primitive-v2"}),
    # Review M2: periodic Gaussian radial pair scalars; widths in angstrom (the
    # widths of the F10 budget scans).
    "radial": ResearchOption("model", {"geometry_radial_gaussian_widths": (0.5, 1.0, 2.0)}),
    # Review m1: operation-derived group and site-symmetry selector inputs.
    "selector-symmetry": ResearchOption("model", {"selector_symmetry_descriptors": True}),
    # Review 2026-10-29, M1: exact-chart Wyckoff-row features for the selector.
    "selector-rows": ResearchOption("model", {"selector_row_geometry_features": True}),
    # Learned-F10 finding: the cover's whole residual group, improper elements
    # included, so the target is racemic within each cover.
    "normalizer-v3": ResearchOption("model", {"geometry_residual_gauge": "normalizer-v3"}),
    # Review 2026-10-29, C1: isotropic Cartesian endpoint kernel.  The scale is the
    # RMS Cartesian displacement per free coordinate in angstrom (l_0 = 1 A).
    "cartesian-kernel": ResearchOption(
        "dequantization",
        {"intrinsic_kernel": "cartesian-v2", "sigma_intrinsic": 0.05},
        parameter="sigma_intrinsic",
    ),
    # Review 2026-10-29, C2: elementary-shear lattice-basis augmentation.
    "shear": ResearchOption("dequantization", {"basis_augmentation": "shear-v1"}),
    # Dropout of the composition and selector Transformers (the geometry field has
    # no dropout layer).  Off in every evaluation and in sampling.
    "dropout": ResearchOption("model", {"dropout": 0.1}, parameter="dropout"),
    # Periodic-table parameter sharing of the selector's and the field's element
    # embeddings (research 2026-11-10); not part of the review set.
    "periodic-elements": ResearchOption("model", {"element_embedding": "periodic-v1"}),
    # Pair carriers that vanish linearly over this length (angstrom) at a
    # coincidence (research 2026-11-12, orbit collapse); not part of the review set.
    "carrier-smoothing": ResearchOption(
        "model",
        {"geometry_pair_carrier_smoothing_length": 0.2},
        parameter="geometry_pair_carrier_smoothing_length",
    ),
}
OPTION_SETS: dict[str, tuple[str, ...]] = {
    "default": (),
    # The opt-in corrections of the 2026-10-29 review plus the residual gauge.
    "review": (
        "primitive-cell",
        "radial",
        "selector-symmetry",
        "selector-rows",
        "normalizer-v3",
        "cartesian-kernel",
        "shear",
    ),
}


def resolve_options(text: str) -> dict[str, dict[str, object]]:
    """Fields per target for a comma list of option names, sets and ``name=value``.

    The result does not depend on the order of the list.  A name given twice
    (directly or through a set) and two options that set one field are errors,
    as is a value for an option without a declared scale.
    """

    if not isinstance(text, str) or not text:
        raise DomainError("--options needs a comma list of option names")
    declared: dict[str, float | None] = {}
    for token in text.split(","):
        name, separator, raw = token.partition("=")
        if name in OPTION_SETS and not separator:
            members: tuple[tuple[str, float | None], ...] = tuple(
                (member, None) for member in OPTION_SETS[name]
            )
        elif name in OPTIONS:
            value = None
            if separator:
                if OPTIONS[name].parameter is None:
                    raise DomainError(f"option {name!r} takes no value")
                try:
                    value = float(raw)
                except ValueError:
                    raise DomainError(f"option {name!r} needs a number, not {raw!r}") from None
                if not math.isfinite(value):
                    raise DomainError(f"option {name!r} needs a finite number")
            members = ((name, value),)
        else:
            known = ", ".join((*OPTION_SETS, *OPTIONS))
            raise DomainError(f"unknown option {token!r}; known: {known}")
        for member, value in members:
            if member in declared:
                raise DomainError(f"option {member!r} is declared twice")
            declared[member] = value
    resolved: dict[str, dict[str, object]] = {"model": {}, "dequantization": {}}
    for name in OPTIONS:  # declaration order, so the list order is irrelevant
        if name not in declared:
            continue
        option = OPTIONS[name]
        fields = dict(option.fields)
        if declared[name] is not None:
            assert option.parameter is not None
            fields[option.parameter] = declared[name]
        target = resolved[option.target]
        clash = sorted(set(fields) & set(target))
        if clash:
            raise DomainError(f"option {name!r} sets {clash} again")
        target.update(fields)
    return resolved


def model_config(profile: str, domain: str, options: str = "default") -> CrystalAtlasConfig:
    base = DOMAINS[domain]()
    fields = resolve_options(options)["model"]
    clash = sorted(set(fields) & set(PROFILES[profile]))
    if clash:
        raise DomainError(f"options set profile fields {clash}")
    return CrystalAtlasConfig(
        domain=replace(base, dng_element_vocabulary=base.element_vocabulary),
        model=ModelConfig(**PROFILES[profile], **fields),  # type: ignore[arg-type]
    )


def research_dequantization(options: str = "default") -> CFMDequantization | None:
    """The CFM endpoint kernel the options declare; ``None`` is the legacy kernel."""

    fields = resolve_options(options)["dequantization"]
    return CFMDequantization(**fields) if fields else None  # type: ignore[arg-type]


BATCHINGS = {
    # One topology fibre per batch (the frozen pilot contract).  On a fragmented
    # corpus most fibres hold one record, so most batches have one record.
    "bucket": DeterministicBucketBatchSampler,
    # Hash-ranked batches across fibres; geometry runs in one padded dense call.
    "mixed": MixedTopologyBatchSampler,
}


def _sampler(dataset, phase, seed, batch_size, batching="bucket"):  # type: ignore[no-untyped-def]
    return BATCHINGS[batching](
        dataset,
        batch_size=batch_size,
        seed=seed,
        measure=(
            TrainingMeasure.UNIQUE_COMPOSITION
            if phase is TrainingPhase.COMPOSITION
            else TrainingMeasure.STRUCTURE
        ),
    )


def evaluate_losses(  # type: ignore[no-untyped-def]
    model, dataset, objective, audit, *, seed, batch_size, phases=None, draws=1
):
    """Mean factor losses over one dataset view with fixed draws; modes restored.

    ``draws`` repeats the pass with the seeds ``seed, seed + 1, ...`` and reports
    the mean, which lowers the Monte Carlo error of the CFM regression (its time
    and base draws are random) while every evaluation of one run still uses the
    same draws, so two states of the model are compared on identical targets.
    """

    phases = PHASES if phases is None else tuple(phases)

    device = next(model.parameters()).device
    cuda_devices = [device.index] if device.type == "cuda" else []
    modes = tuple((module, module.training) for module in model.modules())
    components = ComponentModels(
        model, model.composition_model, model.cover_selector, model.geometry_field
    )
    result: dict[str, object] = {}
    try:
        model.eval()
        with torch.no_grad(), torch.random.fork_rng(devices=cuda_devices):
            totals = {phase: 0.0 for phase in phases}
            for draw in range(draws):
                torch.random.default_generator.manual_seed(seed + draw)
                if device.type == "cuda":
                    with torch.cuda.device(device):
                        torch.cuda.manual_seed(seed + draw)
                for phase in phases:
                    metric = METRICS[PHASES.index(phase)]
                    values = []
                    counts = []
                    for batch in _sampler(dataset, phase, seed, batch_size):
                        audit(batch, torch.float64).require_passed()
                        scalar = getattr(objective(components, batch, phase), metric)
                        value = float(scalar.item())
                        if not math.isfinite(value):
                            raise DomainError(f"nonfinite evaluation {metric}")
                        values.append(value)
                        counts.append(len(batch.record_ids))
                    weighted = math.fsum(v * n for v, n in zip(values, counts, strict=True))
                    totals[phase] += weighted / sum(counts)
            for phase in phases:
                result[METRICS[PHASES.index(phase)]] = totals[phase] / draws
    finally:
        for module, was_training in modes:
            module.training = was_training
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--store", type=Path, required=True)
    for name in ("source", "build", "storage", "certification"):
        parser.add_argument(f"--{name}-fingerprint", type=_digest, required=True)
    parser.add_argument("--certification", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--profile", choices=tuple(PROFILES), default="research-medium"
    )
    parser.add_argument("--domain", choices=tuple(DOMAINS), default="v7")
    parser.add_argument(
        "--options",
        default="default",
        help="comma list of opt-in mechanisms (name or name=value) or sets; names: "
        + ", ".join(OPTIONS)
        + "; sets: "
        + ", ".join(OPTION_SETS),
    )
    parser.add_argument(
        "--split", choices=[mode.value for mode in SplitMode], default="composition"
    )
    parser.add_argument("--epochs", default="30,30,120", help="composition,cover,geometry")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument(
        "--batching",
        choices=tuple(BATCHINGS),
        default="mixed",
        help="training batches; evaluation always uses topology buckets",
    )
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--geometry-learning-rate", type=float, default=1e-4)
    parser.add_argument("--max-records", type=int, default=None, help="smoke tests only")
    parser.add_argument("--candidates", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20261030)
    parser.add_argument("--cpu-threads", type=int, default=4)
    parser.add_argument("--skip-csp", action="store_true")
    parser.add_argument(
        "--csp-tracks", default=",".join(TRACKS), help="comma-separated CSP tracks"
    )
    parser.add_argument(
        "--likelihood-steps",
        type=int,
        default=None,
        help="RK4 steps of the backward likelihood integration (default: the sampler's)",
    )
    parser.add_argument(
        "--eval-every",
        default="0",
        help="validation loss of the active factor every N epochs of its phase; one "
        "integer or composition,cover,geometry (0: no periodic evaluation)",
    )
    parser.add_argument(
        "--validation-draws",
        type=int,
        default=1,
        help="fixed draws averaged in every periodic validation of the CFM regression; "
        "the composition and cover losses are exact likelihoods and use one pass",
    )
    parser.add_argument(
        "--ema-decay",
        type=float,
        default=None,
        help="parameter EMA of the active factor; every periodic evaluation then scores "
        "the raw and the averaged weights and --select-best chooses among both",
    )
    parser.add_argument(
        "--select-best",
        action="store_true",
        help="at the end of each phase restore its factor to the evaluated epoch with "
        "the lowest validation loss (validation partition only; needs --eval-every)",
    )
    args = parser.parse_args()
    resolved_options = resolve_options(args.options)
    epochs = tuple(int(value) for value in args.epochs.split(","))
    if len(epochs) != 3 or any(value < 1 for value in epochs):
        raise DomainError("--epochs needs three positive integers")
    eval_every = tuple(int(value) for value in str(args.eval_every).split(","))
    if len(eval_every) == 1:
        eval_every = eval_every * 3
    if len(eval_every) != 3 or any(value < 0 for value in eval_every):
        raise DomainError("--eval-every needs one or three nonnegative integers")
    if args.validation_draws < 1:
        raise DomainError("--validation-draws must be >= 1")
    if args.select_best and any(value < 1 for value in eval_every):
        raise DomainError("--select-best needs --eval-every >= 1 in every phase")
    if args.ema_decay is not None and not args.select_best:
        # Without the selection the raw and the averaged weights would both survive
        # the run and the weights behind the CSP number would be ambiguous.
        raise DomainError("--ema-decay needs --select-best")
    device = torch.device(args.device)
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
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    output = args.output.absolute()
    output.mkdir(parents=True, exist_ok=False)
    catalogue = build_full_catalogue()
    certification = CrystallographyCertification.read_json(args.certification)
    if certification.receipt_fingerprint != args.certification_fingerprint:
        raise DomainError("crystallographic certification disagrees with external pin")
    certification.require_for(catalogue)
    config = model_config(args.profile, args.domain, args.options)
    dequantization = research_dequantization(args.options)
    torch.manual_seed(args.seed)
    model = CrystalAtlasModel(config, catalogue=catalogue).to(device=device)
    mode = SplitMode(args.split)
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
            raise DomainError("research training requires STRUCTURE deduplication")
        if not store.build_manifest.use_request.train_model:
            raise DomainError("compiled dataset was not approved for model training")
        store.build_manifest.policy_decision.require_authorized()
        evaluate_dataset_use(
            store.source_manifest, DatasetUseRequest(train_model=True)
        ).require_authorized()
        partitions = {
            name: split_partition_record_ids(store, mode, name)
            for name in ("train", "validation", "test")
        }
        if args.max_records is not None:
            partitions = {name: ids[: args.max_records] for name, ids in partitions.items()}
        datasets = {
            name: StoredTrainingDataset(
                store,
                model.composition_codec,
                catalogue,
                domain=model.scientific_config.domain,
                record_ids=ids,
            )
            for name, ids in partitions.items()
        }
        objective = CrystalAtlasTrainingObjective(
            model,
            ObjectiveDataMeasure.from_dataset(datasets["train"]),
            dequantization=dequantization,
        )
        # The objective binds its population measure to one dataset view; the
        # validation view needs its own (same dequantization and contract).
        validation_objective = CrystalAtlasTrainingObjective(
            model,
            ObjectiveDataMeasure.from_dataset(datasets["validation"]),
            dequantization=dequantization,
        )
        audit = ExactBatchCrystallographicAudit(model, memoize=True)
        schedule = StagedTrainingSchedule(
            composition_epochs=epochs[0], cover_epochs=epochs[1], geometry_epochs=epochs[2]
        )
        optimizer = torch.optim.AdamW(
            [
                {"params": model.composition_model.parameters(), "lr": args.learning_rate},
                {"params": model.cover_selector.parameters(), "lr": args.learning_rate},
                {
                    "params": model.geometry_field.parameters(),
                    "lr": args.geometry_learning_rate,
                },
            ],
            weight_decay=1e-2,
        )
        trainer = ScalableTrainer(
            model,
            optimizer,
            schedule,
            config=TrainerConfig(
                precision=PrecisionMode.FLOAT32,
                accumulation_steps=1,
                max_gradient_norm=1.0,
                ema_decay=args.ema_decay,
            ),
        )
        scientific = model.scientific_manifest(
            data_contract={
                "store": str(args.store),
                "split_mode": mode.value,
                "partition_sizes": {name: len(ids) for name, ids in partitions.items()},
                "objective": objective.contract,
                "profile": args.profile,
                "domain_profile": args.domain,
                **(
                    {"options": json.loads(json.dumps(resolved_options))}
                    if any(resolved_options.values())
                    else {}
                ),
                "epochs": list(epochs),
                "batch_size": args.batch_size,
                "batching": args.batching,
                "model_selection": {
                    "eval_every": list(eval_every),
                    "validation_draws": args.validation_draws,
                    "select_best": args.select_best,
                    "ema_decay": args.ema_decay,
                },
            }
        )
        provenance = ProvenanceRecord.create(
            config=model.scientific_config,
            random_seed=args.seed,
            task="research-v7-factor-training-and-csp-match",
            package_version=__version__,
        )
        _write_new(
            output / "training-contract.json",
            {
                "scientific_manifest": scientific.to_dict(),
                "trainer": trainer.config.to_dict(),
                "schedule": schedule.to_dict(),
                "objective": objective.contract,
                "partitions": {name: list(ids) for name, ids in partitions.items()},
                "arguments": {
                    key: (str(value) if isinstance(value, Path) else value)
                    for key, value in vars(args).items()
                },
            },
        )
        evaluations = []

        def checkpoint_and_evaluate(label: str, active_sampler) -> None:  # type: ignore[no-untyped-def]
            receipt = trainer.save_checkpoint(
                output / f"{label}.pt",
                scientific_config=model.scientific_config,
                scientific_manifest=scientific,
                provenance=provenance,
                sampler=active_sampler,
                metadata={"label": label},
            )
            assert receipt is not None
            evaluation = {
                "label": label,
                "optimizer_step": trainer.cursor.optimizer_step,
                "checkpoint_sha256": receipt.sha256,
                "train": evaluate_losses(
                    model,
                    datasets["train"],
                    objective,
                    audit,
                    seed=args.seed + 7,
                    batch_size=args.batch_size,
                ),
                "validation": evaluate_losses(
                    model,
                    datasets["validation"],
                    validation_objective,
                    audit,
                    seed=args.seed + 7,
                    batch_size=args.batch_size,
                ),
            }
            evaluations.append(evaluation)
            _write_new(output / f"{label}-evaluation.json", evaluation)
            print(json.dumps(evaluation, allow_nan=False), flush=True)

        sampler = _sampler(
            datasets["train"], PHASES[0], args.seed + 101, args.batch_size, args.batching
        )
        checkpoint_and_evaluate("baseline", sampler)
        started = time.perf_counter()
        factors = {
            TrainingPhase.COMPOSITION: model.composition_model,
            TrainingPhase.COVER: model.cover_selector,
            TrainingPhase.GEOMETRY: model.geometry_field,
        }
        owned = [{id(tensor) for tensor in factor.parameters()} for factor in factors.values()]
        if sum(len(ids) for ids in owned) != len(set().union(*owned)):
            raise DomainError("factor-wise restoration needs disjoint factor parameters")
        for index, phase in enumerate(PHASES):
            if trainer.current_phase is not phase:
                raise DomainError("unexpected staged trainer phase")
            factor = factors[phase]
            metric = METRICS[index]
            history: list[dict[str, object]] = []
            best_value = math.inf
            best_state = None
            # Only the CFM regression has random draws (time and base point).
            draws = args.validation_draws if phase is TrainingPhase.GEOMETRY else 1
            for phase_epoch in range(1, epochs[index] + 1):
                report = trainer.train_epoch(sampler, objective, crystallographic_audit=audit)
                payload = asdict(report)
                payload["phase"] = report.phase.value
                payload["metrics"] = dict(report.metrics)
                line: dict[str, object] = {
                    "epoch": report.epoch,
                    "phase": report.phase.value,
                    "steps": report.optimizer_steps,
                    "metrics": payload["metrics"],
                }
                if eval_every[index] and (
                    phase_epoch % eval_every[index] == 0 or phase_epoch == epochs[index]
                ):
                    payload["validation"] = {}
                    for weights in ("raw",) if trainer.ema is None else ("raw", "ema"):
                        # Same validation draws for every candidate of the run.
                        with (
                            contextlib.nullcontext()
                            if weights == "raw"
                            else trainer.ema.average_parameters(model)
                        ):
                            value = evaluate_losses(
                                model,
                                datasets["validation"],
                                validation_objective,
                                audit,
                                seed=args.seed + 7,
                                batch_size=args.batch_size,
                                phases=(phase,),
                                draws=draws,
                            )[metric]
                            payload["validation"][weights] = {metric: value}
                            history.append(
                                {
                                    "epoch": report.epoch,
                                    "phase_epoch": phase_epoch,
                                    "weights": weights,
                                    metric: value,
                                }
                            )
                            # Strict improvement: the earliest candidate wins a tie.
                            if args.select_best and value < best_value:
                                best_value = value
                                best_state = {
                                    name: tensor.detach().clone()
                                    for name, tensor in factor.state_dict().items()
                                }
                    line["validation"] = payload["validation"]
                _write_new(output / f"epoch-{report.epoch:04d}.json", payload)
                print(json.dumps(line, allow_nan=False), flush=True)
            if history:
                # ``min`` returns the first minimum, the same candidate the loop kept.
                selected = min(history, key=lambda item: item[metric])
                if args.select_best:
                    assert best_state is not None and selected[metric] == best_value
                    # The factor is frozen in every later phase of the factor-first
                    # schedule, so its optimizer moments are never read again.
                    factor.load_state_dict(best_state, strict=True)
                    del best_state
                    if trainer.ema is not None:
                        # From here on the raw and the averaged weights of this
                        # factor are the selected ones, so no later evaluation
                        # can depend on which of the two it reads.
                        owned_ids = {id(tensor) for tensor in factor.parameters()}
                        ema_state = trainer.ema.state_dict()
                        for name, tensor in model.named_parameters():
                            if id(tensor) in owned_ids and name in ema_state["shadow"]:
                                ema_state["shadow"][name] = tensor.detach().clone()
                        trainer.ema.load_state_dict(ema_state)
                _write_new(
                    output / f"selection-{phase.value}.json",
                    {
                        "phase": phase.value,
                        "metric": metric,
                        "partition": "validation",
                        "draws": draws,
                        "restored": bool(args.select_best),
                        "selected": selected if args.select_best else history[-1],
                        "history": history,
                    },
                )
            sampler = (
                _sampler(
                    datasets["train"],
                    PHASES[index + 1],
                    args.seed + 102 + index,
                    args.batch_size,
                    args.batching,
                )
                if index + 1 < len(PHASES)
                else None
            )
            checkpoint_and_evaluate(f"after-{phase.value}", sampler)
        result: dict[str, object] = {
            "schema": "crystalatlas.research-v7-training-result.v1",
            "status": "trained",
            "optimizer_steps": trainer.cursor.optimizer_step,
            "wall_seconds": round(time.perf_counter() - started, 1),
            "evaluations": evaluations,
        }
        if not args.skip_csp:
            # The measurement is bound to the checkpoint just written: the weights
            # are re-read from the file and verified against its digest.
            generator = LearnedCrystalGenerator.from_checkpoint(
                model,
                output / "after-geometry.pt",
                scientific_manifest=scientific,
                expected_sha256=str(evaluations[-1]["checkpoint_sha256"]),
                map_location=device,
            )
            csp = evaluate_csp(
                generator,
                collect_targets(
                    model,
                    _sampler(
                        datasets["test"], TrainingPhase.GEOMETRY, args.seed, args.batch_size
                    ),
                ),
                candidates=args.candidates,
                seed=args.seed + 500,
                tracks=tuple(args.csp_tracks.split(",")),
                likelihood_steps=args.likelihood_steps,
                on_case=print_case,
            )
            csp["checkpoint_sha256"] = generator.checkpoint_id
            result["csp_test"] = csp
            _write_new(output / "csp-test.json", csp)
            print(json.dumps(csp["summary"], allow_nan=False), flush=True)
        _write_new(output / "result.json", result)


if __name__ == "__main__":
    main()
