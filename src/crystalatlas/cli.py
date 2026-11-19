"""Command-line boundary for exact tools, the reference law, and profiling.

The learned model has no bundled checkpoint.  Consequently this CLI exposes
the explicitly labelled reference generator and optional-ML geometry and
full-objective systems benchmarks; it never silently substitutes any of them
for a trained materials predictor.
"""

from __future__ import annotations

import argparse
import json
import platform
import re
import sys
from collections.abc import Sequence
from dataclasses import asdict
from hashlib import sha256
from pathlib import Path

import numpy as np

from . import __version__
from .catalogue import MINIMAL_CATALOGUE
from .config import CrystalAtlasConfig, DomainConfig
from .elements import atomic_number, element_symbol
from .errors import CrystalAtlasError, DomainError
from .full_catalogue import (
    FROZEN_AFFINE_OPERATION_COUNT,
    FROZEN_CATALOGUE_FINGERPRINT,
    FROZEN_SPACE_GROUP_TYPE_COUNT,
    FROZEN_WYCKOFF_ROW_COUNT,
)
from .io import crystal_to_cif, crystal_to_dict, format_formula, parse_formula
from .pipeline import GenerationMode, GenerationRequest
from .provenance import sha256_file
from .reference import ReferenceGenerator


def _atomic_write_text(path: Path, text: str, *, overwrite: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not overwrite:
        raise DomainError(f"refusing to overwrite existing output: {path}")
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def _parse_species(text: str) -> tuple[int, ...]:
    raw = tuple(token for token in re.split(r"[\s,]+", text.strip()) if token)
    if not raw:
        raise DomainError("--species must contain at least one element")
    values: list[int] = []
    for token in raw:
        try:
            value = int(token) if token.isdecimal() else atomic_number(token)
        except (KeyError, ValueError) as exc:
            raise DomainError(f"unknown element token {token!r}") from exc
        if not 1 <= value <= 118:
            raise DomainError(f"atomic number {value} lies outside [1,118]")
        values.append(value)
    if len(values) != len(set(values)):
        raise DomainError("--species contains a duplicate element")
    return tuple(sorted(values))


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="crystalatlas",
        description="Proof-driven crystal generation research package",
    )
    parser.add_argument("--version", action="version", version=__version__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser(
        "status", help="report implemented and deliberately unavailable capabilities"
    )

    verify = subparsers.add_parser(
        "verify-crystallography",
        help="run the exact all-230 catalogue and pinned-standardization proof gate",
    )
    verify.add_argument(
        "--receipt",
        type=Path,
        help="optional path for the JSON certification receipt",
    )
    verify.add_argument("--overwrite", action="store_true")

    formula = subparsers.add_parser("formula", help="parse and reduce a flat formula")
    formula.add_argument("value", help="for example LiFePO4 or Si2O4")

    generate = subparsers.add_parser(
        "reference-generate",
        help="sample the untrained exact P1/P-1 integration-reference law",
    )
    generate.add_argument(
        "--mode",
        choices=tuple(mode.value for mode in GenerationMode),
        required=True,
    )
    generate.add_argument(
        "--species",
        help="comma/space separated all-and-only element set for species-dng",
    )
    generate.add_argument("--formula", help="flat reduced or reducible formula for csp")
    generate.add_argument("--samples", type=int, default=1)
    generate.add_argument("--seed", type=int, default=0)
    generate.add_argument("--output", type=Path, required=True)
    generate.add_argument("--format", choices=("json", "cif", "both"), default="both")
    generate.add_argument("--overwrite", action="store_true")

    benchmark = subparsers.add_parser(
        "benchmark-geometry",
        help="measure the periodic Transformer on one device without a scaling claim",
    )
    benchmark.add_argument("--device", default="auto")
    benchmark.add_argument(
        "--precision",
        choices=("float32", "float64", "bfloat16", "float16"),
        default="float32",
    )
    benchmark.add_argument("--batch-size", type=int, default=2)
    benchmark.add_argument("--atoms-per-structure", type=int, default=8)
    benchmark.add_argument("--input-scalar-dim", type=int, default=16)
    benchmark.add_argument("--scalar-dim", type=int, default=48)
    benchmark.add_argument("--vector-dim", type=int, default=12)
    benchmark.add_argument("--num-heads", type=int, default=3)
    benchmark.add_argument("--num-layers", type=int, default=2)
    benchmark.add_argument("--radial-basis-dim", type=int, default=8)
    benchmark.add_argument("--cutoff", type=float, default=3.0)
    benchmark.add_argument("--cell-length", type=float, default=6.0)
    benchmark.add_argument("--max-candidates-per-pair", type=int, default=32_768)
    benchmark.add_argument("--max-edges", type=int, default=500_000)
    benchmark.add_argument("--warmup-steps", type=int, default=2)
    benchmark.add_argument("--measurement-steps", type=int, default=5)
    benchmark.add_argument("--seed", type=int, default=1729)
    benchmark.add_argument("--learning-rate", type=float, default=1.0e-3)
    benchmark.add_argument("--cpu-threads", type=int)
    benchmark.add_argument("--output", type=Path)
    benchmark.add_argument("--overwrite", action="store_true")

    backend_comparison = subparsers.add_parser(
        "benchmark-backends",
        help=(
            "compare invariant-global and local-equivariant geometry costs on one "
            "constant-density device workload without a predictive claim"
        ),
    )
    backend_comparison.add_argument("--device", default="auto")
    backend_comparison.add_argument(
        "--precision",
        choices=("float32", "float64", "bfloat16", "float16"),
        default="float32",
    )
    backend_comparison.add_argument("--batch-size", type=int, default=2)
    backend_comparison.add_argument("--atoms-per-structure", type=int, default=20)
    backend_comparison.add_argument("--volume-per-atom", type=float, default=16.0)
    backend_comparison.add_argument("--input-scalar-dim", type=int, default=219)
    backend_comparison.add_argument("--scalar-dim", type=int, default=256)
    backend_comparison.add_argument("--pair-dim", type=int, default=96)
    backend_comparison.add_argument("--vector-dim", type=int, default=32)
    backend_comparison.add_argument("--num-heads", type=int, default=8)
    backend_comparison.add_argument("--num-layers", type=int, default=10)
    backend_comparison.add_argument(
        "--invariant-feedforward-multiplier", type=int, default=4
    )
    backend_comparison.add_argument("--radial-basis-dim", type=int, default=32)
    backend_comparison.add_argument("--reciprocal-mode-radius", type=int, default=2)
    backend_comparison.add_argument("--direct-harmonics", type=int, default=2)
    backend_comparison.add_argument(
        "--cutoff",
        type=float,
        default=DomainConfig().interaction_cutoff_angstrom,
    )
    backend_comparison.add_argument(
        "--max-candidates-per-pair", type=int, default=32_768
    )
    backend_comparison.add_argument(
        "--max-edges-per-structure", type=int, default=500_000
    )
    backend_comparison.add_argument("--max-total-edges", type=int, default=1_000_000)
    backend_comparison.add_argument(
        "--max-dense-pair-interactions", type=int, default=500_000
    )
    backend_comparison.add_argument("--warmup-steps", type=int, default=3)
    backend_comparison.add_argument("--measurement-steps", type=int, default=12)
    backend_comparison.add_argument("--seed", type=int, default=1729)
    backend_comparison.add_argument("--learning-rate", type=float, default=1.0e-3)
    backend_comparison.add_argument("--cpu-threads", type=int)
    backend_comparison.add_argument(
        "--local-prebuilt-core",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="also time the local Transformer with its exact interactions prebuilt",
    )
    backend_comparison.add_argument("--output", type=Path)
    backend_comparison.add_argument("--overwrite", action="store_true")

    objective = subparsers.add_parser(
        "benchmark-objective",
        help=(
            "measure one complete joint training window on an exact synthetic "
            "fixture without a predictive or scaling claim"
        ),
    )
    objective.add_argument(
        "--source",
        choices=("exact-fixture",),
        default="exact-fixture",
    )
    objective.add_argument(
        "--model-profile",
        choices=("micro-v1", "default-v1"),
        default="micro-v1",
    )
    objective.add_argument("--device", default="auto")
    objective.add_argument(
        "--precision",
        choices=("float32", "float64", "bfloat16", "float16"),
        default="float32",
    )
    objective.add_argument("--batch-size", type=int, default=1)
    objective.add_argument("--warmup-steps", type=int, default=1)
    objective.add_argument("--measurement-steps", type=int, default=3)
    objective.add_argument("--seed", type=int, default=1729)
    objective.add_argument("--learning-rate", type=float, default=1.0e-4)
    objective.add_argument("--ema-decay", type=float)
    objective.add_argument("--cpu-threads", type=int)
    objective.add_argument("--output", type=Path)
    objective.add_argument("--overwrite", action="store_true")
    return parser


def _status_payload() -> dict[str, object]:
    config = CrystalAtlasConfig()
    return {
        "package": "crystalatlas",
        "version": __version__,
        "implementation_stage": "research-alpha",
        "trained_checkpoint_bundled": False,
        "predictive_superiority_established": False,
        "reference_generator": {
            "available": True,
            "law": "untrained-uniform-reference",
            "physical_prediction": False,
            "modes": [mode.value for mode in GenerationMode],
        },
        "reference_catalogue": {
            "scope": MINIMAL_CATALOGUE.certification_scope,
            "fingerprint": MINIMAL_CATALOGUE.fingerprint,
            "space_group_keys": list(MINIMAL_CATALOGUE.supported_keys),
            "complete_230_space_groups": MINIMAL_CATALOGUE.complete_space_group_types,
        },
        "production_catalogue": {
            "available": True,
            "construction": "lazy-exact-compiler",
            "space_group_types": FROZEN_SPACE_GROUP_TYPE_COUNT,
            "wyckoff_rows": FROZEN_WYCKOFF_ROW_COUNT,
            "affine_operations": FROZEN_AFFINE_OPERATION_COUNT,
            "fingerprint": FROZEN_CATALOGUE_FINGERPRINT,
            "training_requires_certification_receipt": True,
        },
        "dataset_pipeline": {
            "available": True,
            "database_snapshots_bundled": False,
            "requires_archived_terms_artifacts_record_indices_and_scope_evidence": True,
            "default_deduplication": "post-FullPrim finite-resolution structure identity",
            "optional_scale_free_grouping": "polymorph identity (not the default quotient)",
            "authenticated_sharded_storage": True,
            "stored_source_receipts_alias_attribution_and_split_manifests": True,
            "authenticated_local_cif_orchestration": True,
            "split_modes": [
                "composition",
                "prototype",
                "scaffold",
                "element-set",
            ],
            "model_output_filtering": False,
        },
        "training_pipeline": {
            "available": True,
            "topology_homogeneous_atom_count_bucketing": True,
            "mixed_topology_padded_batching_opt_in": True,
            "distributed_data_parallel": True,
            "mixed_precision_neural_path": ["float32", "bfloat16", "float16"],
            "float64_crystallographic_audits": True,
            "exact_unequal_microbatch_accumulation": True,
            "exact_same_world_size_resume": True,
            "rank_and_common_sampler_policy_verified_collectively": True,
            "fp16_overflow_retries_same_rng_replayed_window": True,
            "uniform_fp32_or_fp64_master_parameters_required": True,
            "explicit_checkpointable_objective_contract_required": True,
            "objective_dataset_sampler_identity_bound": True,
            "factor_first_schedule": [
                "composition",
                "cover",
                "geometry",
                "joint",
            ],
            "topology_batched_geometry_ode": True,
            "default_geometry_backbone": config.model.geometry_backbone,
            "default_geometry_backbone_status": "provisional-engineering-candidate",
            "geometry_learned_hidden_channels": "proper-SO(3)-invariant-scalars",
            "geometry_cartesian_output": "proper-SO(3)-covariant-vectors",
            "geometry_reduced_intrinsic_output_external_rotation_action": (
                "trivial-invariant"
            ),
            "geometry_o3_parity_enforced": False,
            "raw_coordinate_nonequivariant_baseline_implemented": False,
            "raw_coordinate_nonequivariant_baseline_is_default": False,
            "dense_global_periodic_attention": True,
            "dense_pair_interaction_budget": (
                config.numerics.max_dense_pair_interactions
            ),
            "sparse_local_se3_equivariant_ablation": True,
            "large_predictive_run_requires_matched_quality_compute_gate": True,
            "all_catalogue_residual_action_field_sweep_complete": False,
            "vectorized_periodic_edge_enumeration_for_local_ablation": True,
            "trained_checkpoint_bundled": False,
        },
        "diagnostics": {
            "available": True,
            "tiny_dataset_overfit_gate": True,
            "composition_nll_and_calibration": True,
            "legal_scaffold_recall_at_k": True,
            "oracle_scaffold_geometry_attribution": True,
            "equivalence_matcher_contract_fingerprinted": True,
            "population_checkpoint_and_generation_contract_fingerprinted": True,
            "ranked_payload_corpus_receipts_verified": True,
            "matcher_globals_and_helpers_fingerprinted_fail_closed": True,
            "fixed_attempt_budget_enforced": True,
            "evaluated_cases_and_match_decisions_fingerprinted": True,
            "tiny_gate_derives_metrics_from_bound_diagnostics": True,
            "bootstrap_bit_generator": "PCG64",
            "empirical_results_bundled": False,
            "strict_canonical_partition_matching": True,
            "canonical_partition_levels": ["structure", "scaffold"],
            "canonical_partition_payload_prevalidation_required": True,
            "canonical_metric_uncertainty_certified": False,
            "canonical_representation_invariant_physical_equivalence_claim": False,
        },
        "profiling": {
            "single_device_legacy_local_geometry_benchmark": True,
            "single_device_full_training_objective_benchmark": True,
            "same_workload_packaged_backend_comparison": True,
            "backend_comparison_constant_density": True,
            "backend_comparison_includes_forward_backward_adamw": True,
            "backend_comparison_parameter_or_compute_matched": False,
            "backend_comparison_allocator_backend_isolated": False,
            "backend_comparison_allocator_comparison_permitted": False,
            "full_objective_source": "exact-synthetic-P1-fixture",
            "full_objective_devices": ["cpu", "cuda"],
            "full_objective_uses_minimal_p1_catalogue": True,
            "full_objective_authenticated_dataset_source": False,
            "full_objective_generation_ode_measured": False,
            "cpu_compatible": True,
            "canonical_json_report": True,
            "gpu_allocator_peak_reported_only_when_backend_exposes_it": True,
            "multi_device_scaling_claim": False,
            "predictive_quality_claim": False,
        },
        "default_domain": config.to_dict()["domain"],
        "known_hard_limits": [
            "the untrained reference generator deliberately remains P1/P-1 only",
            "raw CIF support is a strict full-occupancy audited subset, not general CIF",
            "floating symmetry labels depend on the frozen numerical tolerance policy",
            "learned geometry ODEs batch equal topologies, but no custom fused GPU "
            "attention/ODE kernel is bundled",
            "no energy, relaxation, stability, or synthesizability model is bundled",
            "the learned Transformer/CFM stack requires training and a checkpoint",
            "large dataset tensor-shard writing is not yet a fused distributed job",
            "the full-objective benchmark currently repeats a synthetic P1 fixture and "
            "does not load an authenticated DatasetStore",
        ],
    }


def _run_crystallography_verification(arguments: argparse.Namespace) -> dict[str, object]:
    """Execute and optionally persist the production crystallography receipt."""

    from .certification import certify_crystallography
    from .full_catalogue import build_full_catalogue

    catalogue = build_full_catalogue()
    receipt = certify_crystallography(catalogue)
    payload = receipt.to_dict()
    if arguments.receipt is not None:
        destination = arguments.receipt.resolve()
        _atomic_write_text(
            destination,
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            overwrite=arguments.overwrite,
        )
    return payload


def _request_from_arguments(arguments: argparse.Namespace) -> GenerationRequest:
    mode = GenerationMode(arguments.mode)
    if mode is GenerationMode.DNG:
        if arguments.species is not None or arguments.formula is not None:
            raise DomainError("dng accepts neither --species nor --formula")
        return GenerationRequest(mode=mode, samples=arguments.samples, seed=arguments.seed)
    if mode is GenerationMode.SPECIES_DNG:
        if arguments.species is None or arguments.formula is not None:
            raise DomainError("species-dng requires --species and forbids --formula")
        return GenerationRequest(
            mode=mode,
            species=_parse_species(arguments.species),
            samples=arguments.samples,
            seed=arguments.seed,
        )
    if arguments.formula is None or arguments.species is not None:
        raise DomainError("csp requires --formula and forbids --species")
    return GenerationRequest(
        mode=mode,
        composition=parse_formula(arguments.formula),
        samples=arguments.samples,
        seed=arguments.seed,
    )


def _run_reference(arguments: argparse.Namespace) -> dict[str, object]:
    request = _request_from_arguments(arguments)
    generator = ReferenceGenerator()
    config = generator.config
    traces = generator.sample(request)
    output = arguments.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    if not arguments.overwrite:
        planned = {output / "manifest.json"}
        for index in range(request.samples):
            base = f"sample_{index:06d}"
            planned.update(
                {
                    output / f"{base}.json",
                    output / f"{base}.cif",
                    output / f"{base}.failure.json",
                }
            )
        existing = sorted(str(path) for path in planned if path.exists())
        if existing:
            raise DomainError(
                "refusing a partially overwriting generation run; existing outputs: "
                + ", ".join(existing)
            )
    records: list[dict[str, object]] = []
    written_outputs: list[Path] = []
    for trace in traces:
        base = f"sample_{trace.sample_index:06d}"
        if not trace.succeeded or trace.crystal is None or trace.composition is None:
            record = {
                "sample_index": trace.sample_index,
                "seed": trace.seed,
                "succeeded": False,
                "failed_stage": trace.failed_stage,
                "error": trace.error,
            }
            failure_path = output / f"{base}.failure.json"
            _atomic_write_text(
                failure_path,
                json.dumps(record, indent=2, sort_keys=True) + "\n",
                overwrite=arguments.overwrite,
            )
            written_outputs.append(failure_path)
            records.append(record)
            continue
        crystal = trace.crystal
        payload = crystal_to_dict(crystal)
        payload.update(
            {
                "sample_index": trace.sample_index,
                "seed": trace.seed,
                "mode": request.mode.value,
                "reduced_formula": format_formula(trace.composition),
                "law": "untrained-uniform-reference",
                "physical_prediction": False,
            }
        )
        if arguments.format in ("json", "both"):
            json_path = output / f"{base}.json"
            _atomic_write_text(
                json_path,
                json.dumps(payload, indent=2, sort_keys=True) + "\n",
                overwrite=arguments.overwrite,
            )
            written_outputs.append(json_path)
        if arguments.format in ("cif", "both"):
            cif_path = output / f"{base}.cif"
            _atomic_write_text(
                cif_path,
                crystal_to_cif(
                    crystal,
                    data_name=base,
                    law="untrained-uniform-reference",
                    physical_prediction=False,
                    provenance={
                        "root_seed": request.seed,
                        "sample_index": trace.sample_index,
                        "sample_seed": trace.seed,
                        "config_fingerprint": config.fingerprint,
                        "catalogue_fingerprint": generator.catalogue.fingerprint,
                    },
                ),
                overwrite=arguments.overwrite,
            )
            written_outputs.append(cif_path)
        records.append(
            {
                "sample_index": trace.sample_index,
                "seed": trace.seed,
                "succeeded": True,
                "reduced_formula": format_formula(trace.composition),
                "space_group_constraint": crystal.setting.key,
                "translation_index": crystal.translation_index,
            }
        )
    law_payload = asdict(generator.law)
    law_fingerprint = sha256(
        json.dumps(law_payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    conditioning: dict[str, object]
    if request.mode is GenerationMode.CSP:
        assert request.composition is not None
        conditioning = {
            "reduced_formula": format_formula(request.composition),
            "counts": [list(item) for item in request.composition.counts],
        }
    elif request.mode is GenerationMode.SPECIES_DNG:
        conditioning = {
            "all_and_only_atomic_numbers": list(request.species or ()),
        }
    else:
        conditioning = {"unconditional": True}
    manifest = {
        "schema": "crystalatlas.reference-generation-manifest.v1",
        "package": {"name": "crystalatlas", "version": __version__},
        "law": "untrained-uniform-reference",
        "law_config": law_payload,
        "law_fingerprint": law_fingerprint,
        "physical_prediction": False,
        "mode": request.mode.value,
        "conditioning": conditioning,
        "root_seed": request.seed,
        "random_stream_contract": (
            "numpy.SeedSequence(root_seed).spawn(samples); each child emits four "
            "uint64 stage seeds"
        ),
        "requested_samples": request.samples,
        "successful_samples": sum(bool(item["succeeded"]) for item in records),
        "config": config.to_dict(),
        "config_fingerprint": config.fingerprint,
        "catalogue": generator.catalogue.identity,
        "runtime": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "numpy": np.__version__,
        },
        "output_sha256": {path.name: sha256_file(path) for path in sorted(written_outputs)},
        "records": records,
    }
    _atomic_write_text(
        output / "manifest.json",
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        overwrite=arguments.overwrite,
    )
    return manifest


def _run_geometry_benchmark(arguments: argparse.Namespace) -> dict[str, object]:
    """Run the optional-ML single-device benchmark and persist only success."""

    from .benchmarking import (
        GeometryBenchmarkConfig,
        run_single_device_geometry_benchmark,
    )

    config = GeometryBenchmarkConfig(
        device=arguments.device,
        precision=arguments.precision,
        batch_size=arguments.batch_size,
        atoms_per_structure=arguments.atoms_per_structure,
        input_scalar_dim=arguments.input_scalar_dim,
        scalar_dim=arguments.scalar_dim,
        vector_dim=arguments.vector_dim,
        num_heads=arguments.num_heads,
        num_layers=arguments.num_layers,
        radial_basis_dim=arguments.radial_basis_dim,
        cutoff=arguments.cutoff,
        cell_length=arguments.cell_length,
        max_candidates_per_pair=arguments.max_candidates_per_pair,
        max_edges=arguments.max_edges,
        warmup_steps=arguments.warmup_steps,
        measurement_steps=arguments.measurement_steps,
        seed=arguments.seed,
        learning_rate=arguments.learning_rate,
        cpu_threads=arguments.cpu_threads,
    )
    report = run_single_device_geometry_benchmark(config)
    payload = report.to_manifest()
    if arguments.output is not None:
        _atomic_write_text(
            arguments.output.resolve(),
            report.to_json() + "\n",
            overwrite=arguments.overwrite,
        )
    return payload


def _run_backend_comparison(arguments: argparse.Namespace) -> dict[str, object]:
    """Compare both geometry implementations on one same-workload fixture."""

    from .backend_comparison import (
        BackendComparisonConfig,
        run_single_device_backend_comparison,
    )

    config = BackendComparisonConfig(
        device=arguments.device,
        precision=arguments.precision,
        batch_size=arguments.batch_size,
        atoms_per_structure=arguments.atoms_per_structure,
        volume_per_atom=arguments.volume_per_atom,
        input_scalar_dim=arguments.input_scalar_dim,
        scalar_dim=arguments.scalar_dim,
        pair_dim=arguments.pair_dim,
        vector_dim=arguments.vector_dim,
        num_heads=arguments.num_heads,
        num_layers=arguments.num_layers,
        invariant_feedforward_multiplier=arguments.invariant_feedforward_multiplier,
        radial_basis_dim=arguments.radial_basis_dim,
        reciprocal_mode_radius=arguments.reciprocal_mode_radius,
        direct_harmonics=arguments.direct_harmonics,
        cutoff=arguments.cutoff,
        max_candidates_per_pair=arguments.max_candidates_per_pair,
        max_edges_per_structure=arguments.max_edges_per_structure,
        max_total_edges=arguments.max_total_edges,
        max_dense_pair_interactions=arguments.max_dense_pair_interactions,
        warmup_steps=arguments.warmup_steps,
        measurement_steps=arguments.measurement_steps,
        seed=arguments.seed,
        learning_rate=arguments.learning_rate,
        cpu_threads=arguments.cpu_threads,
        measure_local_prebuilt_core=arguments.local_prebuilt_core,
    )
    report = run_single_device_backend_comparison(config)
    payload = report.to_manifest()
    if arguments.output is not None:
        _atomic_write_text(
            arguments.output.resolve(),
            report.to_json() + "\n",
            overwrite=arguments.overwrite,
        )
    return payload


def _run_training_objective_benchmark(
    arguments: argparse.Namespace,
) -> dict[str, object]:
    """Run the complete objective only on its explicitly synthetic fixture."""

    from .training_benchmarking import (
        ExactObjectiveFixtureSource,
        TrainingObjectiveBenchmarkConfig,
        objective_benchmark_micro_config,
        run_single_device_training_objective_benchmark,
    )

    if arguments.source != "exact-fixture":  # pragma: no cover - argparse enforces this
        raise DomainError("only the exact synthetic objective fixture is supported")
    scientific_config = (
        objective_benchmark_micro_config()
        if arguments.model_profile == "micro-v1"
        else CrystalAtlasConfig()
    )
    config = TrainingObjectiveBenchmarkConfig(
        source=ExactObjectiveFixtureSource(),
        scientific_config=scientific_config,
        device=arguments.device,
        precision=arguments.precision,
        batch_size=arguments.batch_size,
        warmup_steps=arguments.warmup_steps,
        measurement_steps=arguments.measurement_steps,
        seed=arguments.seed,
        learning_rate=arguments.learning_rate,
        ema_decay=arguments.ema_decay,
        cpu_threads=arguments.cpu_threads,
    )
    report = run_single_device_training_objective_benchmark(config)
    payload = report.to_manifest()
    if arguments.output is not None:
        _atomic_write_text(
            arguments.output.resolve(),
            report.to_json() + "\n",
            overwrite=arguments.overwrite,
        )
    return payload


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    arguments = parser.parse_args(argv)
    try:
        if arguments.command == "status":
            payload = _status_payload()
        elif arguments.command == "verify-crystallography":
            payload = _run_crystallography_verification(arguments)
        elif arguments.command == "formula":
            composition = parse_formula(arguments.value)
            payload = {
                "input": arguments.value,
                "canonical_reduced_formula": format_formula(composition),
                "counts": [
                    {
                        "atomic_number": z,
                        "symbol": element_symbol(z),
                        "coefficient": count,
                    }
                    for z, count in composition.counts
                ],
            }
        elif arguments.command == "reference-generate":
            payload = _run_reference(arguments)
        elif arguments.command == "benchmark-geometry":
            payload = _run_geometry_benchmark(arguments)
        elif arguments.command == "benchmark-backends":
            payload = _run_backend_comparison(arguments)
        elif arguments.command == "benchmark-objective":
            payload = _run_training_objective_benchmark(arguments)
        else:  # pragma: no cover - argparse enforces the finite command set
            raise AssertionError(f"unhandled command {arguments.command!r}")
    except (CrystalAtlasError, ValueError, OSError) as exc:
        print(f"crystalatlas: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
