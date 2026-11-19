"""Single-device benchmark for the complete CrystalAtlas training objective.

Unlike :mod:`crystalatlas.benchmarking`, which isolates the periodic
Transformer, this module executes the production
:class:`~crystalatlas.training_objectives.CrystalAtlasTrainingObjective`
through :class:`~crystalatlas.trainer.ScalableTrainer`.  The bundled workload
is an exact, deterministic, synthetic P1 fixture.  It is useful for measuring
whether the full numerical path executes on one CPU or CUDA device; it is not
materials data and cannot measure predictive quality, convergence, generation
quality, corpus-wide throughput, or multi-device scaling.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from fractions import Fraction
from hashlib import sha256
from pathlib import Path
from typing import Any, TypeAlias, cast

import numpy as np

from . import __version__
from ._optional import require_torch
from .benchmarking import (
    DYADIC_SIGNAL_INITIALIZER,
    _allocator_memory,
    _canonical_json,
    _fingerprint,
    _initialize_model,
    _logical_tensor_bytes,
    _model_sha256,
    _require_deterministic_cuda_environment,
    _runtime_manifest,
    _timed,
)
from .catalogue import MINIMAL_CATALOGUE
from .config import CrystalAtlasConfig, DomainConfig, ModelConfig, NumericsConfig
from .decoder import CrystalScaffold, OrbitAssignment
from .errors import CrystalAtlasError, DomainError, InvariantViolation
from .learned import CrystalAtlasModel
from .trainer import (
    PrecisionMode,
    ScalableTrainer,
    StagedTrainingSchedule,
    TrainerConfig,
    TrainingPhase,
)
from .training_data import (
    CompositionSupervision,
    CoverSupervision,
    CoverTopologyKey,
    GeometrySupervision,
    TrainingBatch,
    TrainingBucketKey,
    TrainingMeasure,
)
from .training_objectives import (
    CrystalAtlasTrainingObjective,
    ExactBatchCrystallographicAudit,
    ObjectiveDataMeasure,
)
from .training_targets import ExactFixtureTrainingTarget, lift_exact_fixture_target

torch = require_torch()
Tensor = torch.Tensor

TRAINING_OBJECTIVE_BENCHMARK_SCHEMA = (
    "crystalatlas.single-device-training-objective-benchmark.v2"
)
TRAINING_OBJECTIVE_BENCHMARK_METHOD = (
    "crystalatlas.scalable-trainer-joint-window.v3"
)
EXACT_OBJECTIVE_FIXTURE_ID = "p1-cno-three-site-v1"

_FIXTURE_SCHEMA = "crystalatlas.exact-objective-benchmark-fixture.v1"
_HEX_DIGITS = frozenset("0123456789abcdef")


def objective_benchmark_micro_config() -> CrystalAtlasConfig:
    """Return the pinned small architecture used by the CPU smoke workload."""

    return CrystalAtlasConfig(
        domain=DomainConfig(
            element_vocabulary=(6, 7, 8),
            dng_element_vocabulary=(6, 7, 8),
            max_distinct_elements=3,
            max_raw_atoms=40,
            max_formula_units=16,
            metric_radius=2.0,
            interaction_cutoff_angstrom=2.5,
            # Pinned workload: loewner-v2 became the default after its baselines.
            lattice_chart="ball-v1",
        ),
        numerics=NumericsConfig(ode_steps=2, max_periodic_edges=20_000),
        model=ModelConfig(
            composition_width=24,
            composition_layers=1,
            selector_width=24,
            selector_layers=1,
            geometry_scalar_width=24,
            geometry_pair_width=24,
            geometry_vector_width=12,
            geometry_layers=1,
            attention_heads=3,
            radial_features=6,
        ),
    )


def _positive_integer(name: str, value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise DomainError(f"{name} must be a positive integer")
    return value


def _positive_finite(name: str, value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise DomainError(f"{name} must be a finite positive number")
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise DomainError(f"{name} must be a finite positive number")
    return result


def _digest(value: object, *, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in _HEX_DIGITS for character in value)
    ):
        raise DomainError(f"{name} must be a lowercase SHA-256 digest")
    return value


@dataclass(frozen=True, slots=True)
class ExactObjectiveFixtureSource:
    """Pinned exact synthetic source for a full-objective systems benchmark."""

    fixture_id: str = EXACT_OBJECTIVE_FIXTURE_ID

    def __post_init__(self) -> None:
        if self.fixture_id != EXACT_OBJECTIVE_FIXTURE_ID:
            raise DomainError("the requested exact objective fixture is not supported")

    def to_dict(self) -> dict[str, object]:
        return {
            "kind": "exact-synthetic-fixture",
            "fixture_id": self.fixture_id,
            "synthetic": True,
            "third_party_content": False,
            "physical_prediction_data": False,
        }


ObjectiveBenchmarkSource: TypeAlias = ExactObjectiveFixtureSource


@dataclass(frozen=True, slots=True)
class TrainingObjectiveBenchmarkConfig:
    """Pinned single-device workload and measurement policy.

    ``float64`` selects Float64 master parameters with autocast disabled.
    ``bfloat16`` and ``float16`` use Float32 master parameters and the trainer's
    existing autocast/gradient-scaling contracts.  FP16 is CUDA-only.
    """

    source: ObjectiveBenchmarkSource = field(default_factory=ExactObjectiveFixtureSource)
    scientific_config: CrystalAtlasConfig = field(
        default_factory=objective_benchmark_micro_config
    )
    device: str = "auto"
    precision: str = "float32"
    batch_size: int = 1
    warmup_steps: int = 1
    measurement_steps: int = 3
    seed: int = 1729
    learning_rate: float = 1.0e-4
    ema_decay: float | None = None
    cpu_threads: int | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.source, ExactObjectiveFixtureSource):
            raise TypeError("source must be ExactObjectiveFixtureSource")
        if not isinstance(self.scientific_config, CrystalAtlasConfig):
            raise TypeError("scientific_config must be CrystalAtlasConfig")
        if not isinstance(self.device, str) or not self.device.strip():
            raise DomainError("benchmark device must be nonempty text")
        if self.precision not in {"float32", "float64", "bfloat16", "float16"}:
            raise DomainError("benchmark precision is not supported")
        for name in ("batch_size", "warmup_steps", "measurement_steps"):
            _positive_integer(name, getattr(self, name))
        if isinstance(self.seed, bool) or not isinstance(self.seed, int):
            raise DomainError("benchmark seed must be an unsigned 64-bit integer")
        if not 0 <= self.seed < 2**64:
            raise DomainError("benchmark seed must be an unsigned 64-bit integer")
        _positive_finite("learning_rate", self.learning_rate)
        if self.ema_decay is not None and (
            not math.isfinite(self.ema_decay) or not 0.0 <= self.ema_decay < 1.0
        ):
            raise DomainError("ema_decay must lie in [0,1) or be None")
        if self.cpu_threads is not None:
            _positive_integer("cpu_threads", self.cpu_threads)
        vocabulary = set(self.scientific_config.domain.dng_element_vocabulary)
        if not {6, 7, 8}.issubset(vocabulary):
            raise DomainError("the exact benchmark fixture requires C, N, and O support")

    def to_dict(self) -> dict[str, object]:
        return {
            "source": self.source.to_dict(),
            "scientific_config": self.scientific_config.to_dict(),
            "scientific_config_fingerprint": self.scientific_config.fingerprint,
            "device": self.device,
            "precision": self.precision,
            "batch_size": self.batch_size,
            "warmup_steps": self.warmup_steps,
            "measurement_steps": self.measurement_steps,
            "seed": self.seed,
            "learning_rate": float(self.learning_rate),
            "ema_decay": self.ema_decay,
            "cpu_threads": self.cpu_threads,
        }

    @property
    def fingerprint(self) -> str:
        return _fingerprint(self.to_dict())

    def to_manifest(self) -> dict[str, object]:
        return cast(
            dict[str, object],
            json.loads(
                _canonical_json({**self.to_dict(), "fingerprint": self.fingerprint})
            ),
        )


@dataclass(frozen=True, slots=True)
class TrainingWindowMeasurement:
    """Synchronized timings and trainer reports for full joint windows."""

    durations_seconds: tuple[float, ...]
    structures_per_iteration: int
    atoms_per_iteration: int
    metrics: tuple[Mapping[str, float], ...]
    gradient_l2_norms: tuple[float, ...]
    gradient_maxima: tuple[float, ...]
    gradient_tensor_counts: tuple[int, ...]
    geometry_interaction_kind: str
    geometry_interaction_counts_per_iteration: tuple[tuple[int, ...], ...]

    def __post_init__(self) -> None:
        count = len(self.durations_seconds)
        if count < 1:
            raise DomainError("a training benchmark needs timing samples")
        if any(not math.isfinite(value) or value <= 0.0 for value in self.durations_seconds):
            raise DomainError("training benchmark durations must be finite and positive")
        _positive_integer("structures_per_iteration", self.structures_per_iteration)
        _positive_integer("atoms_per_iteration", self.atoms_per_iteration)
        aligned = (
            self.metrics,
            self.gradient_l2_norms,
            self.gradient_maxima,
            self.gradient_tensor_counts,
            self.geometry_interaction_counts_per_iteration,
        )
        if any(len(values) != count for values in aligned):
            raise DomainError("training measurements do not align by iteration")
        if any(
            not math.isfinite(value) or value < 0.0
            for value in (*self.gradient_l2_norms, *self.gradient_maxima)
        ):
            raise DomainError("gradient statistics must be finite and nonnegative")
        if any(value < 1 for value in self.gradient_tensor_counts):
            raise DomainError("each measured step must produce model gradients")
        if any(
            not counts or any(value < 0 for value in counts)
            for counts in self.geometry_interaction_counts_per_iteration
        ):
            raise DomainError(
                "each measured step must observe nonnegative geometry interactions"
            )
        if self.geometry_interaction_kind not in {
            "dense_atom_pairs",
            "periodic_image_edges",
        }:
            raise DomainError("geometry interaction kind is not supported")
        for metric in self.metrics:
            required = {
                "objective/total",
                "nll/composition",
                "nll/cover",
                "regression/cfm",
            }
            if set(metric) != required or any(
                not math.isfinite(float(value)) for value in metric.values()
            ):
                raise DomainError("a joint training report lacks finite component metrics")

    def to_manifest(self) -> dict[str, object]:
        total = math.fsum(self.durations_seconds)
        iterations = len(self.durations_seconds)
        return {
            "phase": "full_joint_training_window",
            "includes": [
                "float64 exact-decoder and chart audit",
                "host-to-device supervision tensor construction",
                "composition negative log-likelihood",
                "legal arithmetic-cover negative log-likelihood",
                "conditional-flow-matching path and regression loss",
                "exact cover expansion and backend-specific periodic geometry",
                "geometry Transformer forward",
                "exact tangent-space reduction",
                "backward and gradient finite/norm audit",
                "AdamW update",
                "optional EMA update",
                "device synchronization",
            ],
            "iterations": iterations,
            "durations_seconds": list(self.durations_seconds),
            "total_seconds": total,
            "mean_seconds": total / iterations,
            "minimum_seconds": min(self.durations_seconds),
            "maximum_seconds": max(self.durations_seconds),
            "structures_per_second": self.structures_per_iteration * iterations / total,
            "atoms_per_second": self.atoms_per_iteration * iterations / total,
            "structures_per_iteration": self.structures_per_iteration,
            "atoms_per_iteration": self.atoms_per_iteration,
            "metrics": [dict(sorted(metric.items())) for metric in self.metrics],
            "gradient_l2_norms": list(self.gradient_l2_norms),
            "gradient_maxima": list(self.gradient_maxima),
            "gradient_tensor_counts": list(self.gradient_tensor_counts),
            "geometry_interaction_kind": self.geometry_interaction_kind,
            "geometry_interaction_counts_per_iteration": [
                list(values)
                for values in self.geometry_interaction_counts_per_iteration
            ],
        }


@dataclass(frozen=True, slots=True)
class TrainingObjectiveBenchmarkReport:
    """Canonical evidence for one successful full-objective benchmark."""

    config: TrainingObjectiveBenchmarkConfig
    runtime: Mapping[str, object]
    source: Mapping[str, object]
    model: Mapping[str, object]
    workload: Mapping[str, object]
    phase: TrainingWindowMeasurement
    memory: Mapping[str, object]
    numerics: Mapping[str, object]

    def __post_init__(self) -> None:
        # Canonical JSON validation rejects NaN/Infinity and non-JSON objects.
        _canonical_json(self._payload())

    def _payload(self) -> dict[str, object]:
        return {
            "schema": TRAINING_OBJECTIVE_BENCHMARK_SCHEMA,
            "status": "completed",
            "method": TRAINING_OBJECTIVE_BENCHMARK_METHOD,
            "config": self.config.to_manifest(),
            "runtime": dict(self.runtime),
            "source": dict(self.source),
            "model": dict(self.model),
            "workload": dict(self.workload),
            "phases": [self.phase.to_manifest()],
            "memory": dict(self.memory),
            "numerics": dict(self.numerics),
            "interpretation": {
                "full_crystalatlas_training_objective_measured": True,
                "scalable_trainer_single_device_window_measured": True,
                "predictive_quality_measured": False,
                "optimization_convergence_measured": False,
                "generation_quality_measured": False,
                "generation_ode_measured": False,
                "multi_device_scaling_measured": False,
                "multi_device_extrapolation_permitted": False,
                "full_epoch_input_pipeline_measured": False,
                "dataset_authentication_measured": False,
                "workload_representative_of_materials_corpora": False,
                "reason": (
                    "The benchmark repeats one exact synthetic P1 target on one device. "
                    "It measures the current implementation path, not a materials "
                    "distribution, checkpoint quality, DDP communication, or inference."
                ),
                "cfm_metric_scope": (
                    "CFM is a sampled likelihood-free regression objective; its value "
                    "is not comparable across different topology, dequantization, "
                    "data-measure, or loss-weight contracts."
                ),
            },
        }

    @property
    def fingerprint(self) -> str:
        return _fingerprint(self._payload())

    def to_manifest(self) -> dict[str, object]:
        return cast(
            dict[str, object],
            json.loads(
                _canonical_json({**self._payload(), "fingerprint": self.fingerprint})
            ),
        )

    def to_json(self, *, indent: int | None = 2) -> str:
        return json.dumps(
            self.to_manifest(),
            allow_nan=False,
            ensure_ascii=True,
            indent=indent,
            separators=(",", ":") if indent is None else None,
            sort_keys=True,
        )


class TrainingObjectiveBenchmarkExecutionError(CrystalAtlasError):
    """A full-objective benchmark failed before a report was certified."""

    def __init__(self, stage: str, cause: BaseException) -> None:
        self.stage = stage
        self.cause_type = type(cause).__name__
        super().__init__(
            f"training-objective benchmark failed during {stage}: "
            f"{self.cause_type}: {cause}"
        )


@dataclass(frozen=True, slots=True)
class _FixtureBatches:
    composition: TrainingBatch
    structure: TrainingBatch
    data_measure: ObjectiveDataMeasure
    source_manifest: Mapping[str, object]


def _fixture_scaffold() -> CrystalScaffold:
    return CrystalScaffold.from_mapping(
        nu=1,
        space_group="P1",
        reduced_composition={6: 1, 7: 1, 8: 1},
        orbits=(
            OrbitAssignment("1a", 6, (Fraction(0), Fraction(0), Fraction(0))),
            OrbitAssignment(
                "1a", 7, (Fraction(1, 7), Fraction(2, 9), Fraction(3, 11))
            ),
            OrbitAssignment(
                "1a", 8, (Fraction(2, 5), Fraction(3, 8), Fraction(4, 13))
            ),
        ),
    )


def _fixture_contract(
    scientific_config: CrystalAtlasConfig,
    *,
    batch_size: int,
) -> dict[str, object]:
    payload: dict[str, object] = {
        "schema": _FIXTURE_SCHEMA,
        "fixture_id": EXACT_OBJECTIVE_FIXTURE_ID,
        "catalogue_fingerprint": MINIMAL_CATALOGUE.fingerprint,
        "scientific_config_fingerprint": scientific_config.fingerprint,
        "batch_replication": batch_size,
        "nu": 1,
        "space_group": "P1",
        "reduced_composition": [[6, 1], [7, 1], [8, 1]],
        "orbits": [
            ["1a", 6, ["0/1", "0/1", "0/1"]],
            ["1a", 7, ["1/7", "2/9", "3/11"]],
            ["1a", 8, ["2/5", "3/8", "4/13"]],
        ],
        "metric_float64_hex": [
            [16.0.hex(), 0.0.hex(), 0.0.hex()],
            [0.0.hex(), 25.0.hex(), 0.0.hex()],
            [0.0.hex(), 0.0.hex(), 36.0.hex()],
        ],
        "synthetic": True,
        "physical_prediction_data": False,
    }
    payload["fingerprint"] = _fingerprint(payload)
    return payload


def _target_fingerprint(target: ExactFixtureTrainingTarget) -> str:
    scaffold = target.canonical_scaffold
    payload = {
        "schema": "crystalatlas.objective-benchmark-target.v1",
        "nu": scaffold.nu,
        "space_group": scaffold.space_group,
        "composition": [list(value) for value in scaffold.reduced_composition],
        "orbits": [
            [
                orbit.wyckoff_label,
                int(orbit.species),
                [[value.numerator, value.denominator] for value in orbit.parameter],
            ]
            for orbit in scaffold.orbits
        ],
        "metric": [float(value).hex() for value in target.metric.reshape(-1)],
        "lattice_latent": [
            float(value).hex() for value in target.lattice_latent.reshape(-1)
        ],
        "intrinsic": [
            float(value).hex() for value in target.intrinsic_coordinates.reshape(-1)
        ],
        "period_basis": [float(value).hex() for value in target.period_basis.reshape(-1)],
    }
    return _fingerprint(payload)


def _topology_key(target: ExactFixtureTrainingTarget) -> TrainingBucketKey:
    composition = target.composition
    decisions = target.cover_decisions
    support_index = {
        atomic_number: index
        for index, atomic_number in enumerate(composition.support)
    }
    period = np.asarray(target.period_basis, dtype="<f8", order="C")
    topology = CoverTopologyKey(
        target_type="exact-fixture",
        space_group=target.cover.space_group,
        nu=target.cover.nu,
        support_size=len(composition),
        row_labels=tuple(row.label for row in decisions.rows),
        row_multiplicities=tuple(row.multiplicity for row in decisions.rows),
        row_free_dimensions=tuple(row.free_dimension for row in decisions.rows),
        selected_actions=tuple(
            tuple(int(value) for value in decision.selected_action.counts)
            for decision in decisions.row_decisions
        ),
        orbit_topology=tuple(
            (
                orbit.wyckoff_label,
                support_index[cast(int, orbit.species)],
            )
            for orbit in target.canonical_scaffold.orbits
        ),
        lattice_dimension=target.layout.lattice_dimension,
        intrinsic_dimension=target.layout.intrinsic_dimension,
        nu_vocabulary_size=len(decisions.nu_legal_mask),
        group_vocabulary_size=len(decisions.group_legal_mask),
        period_basis_sha256=sha256(period.tobytes(order="C")).hexdigest(),
    )
    return TrainingBucketKey(
        target.cover.nu * target.composition.total_atoms,
        topology,
    )


def _composition_supervision(
    model: CrystalAtlasModel,
    target: ExactFixtureTrainingTarget,
    *,
    batch_size: int,
    measure: TrainingMeasure,
    population_multiplicity: int,
) -> CompositionSupervision:
    compositions = tuple(target.composition for _ in range(batch_size))
    encoded = model.composition_codec.collate(compositions)
    weights = (
        np.ones(batch_size, dtype=np.float64)
        if measure is TrainingMeasure.UNIQUE_COMPOSITION
        else np.full(batch_size, 1.0 / population_multiplicity, dtype=np.float64)
    )
    return CompositionSupervision(
        token_ids=encoded.token_ids,
        token_types=encoded.token_types,
        legal_mask=encoded.legal_mask,
        sequence_mask=encoded.sequence_mask,
        atomic_numbers=np.tile(
            np.asarray(target.composition.support, dtype=np.int64),
            (batch_size, 1),
        ),
        reduced_counts=np.tile(
            np.asarray(target.composition.coefficients, dtype=np.int64),
            (batch_size, 1),
        ),
        composition_multiplicities=np.full(
            batch_size, population_multiplicity, dtype=np.int64
        ),
        sample_weights=weights,
    )


def _cover_supervision(
    target: ExactFixtureTrainingTarget,
    *,
    batch_size: int,
    enabled: bool,
) -> CoverSupervision:
    cover = target.cover
    decisions = target.cover_decisions
    row_decisions = decisions.row_decisions
    rows = len(row_decisions)
    support = len(target.composition)
    max_actions = max(len(decision.legal_actions) for decision in row_decisions)
    candidates = np.zeros((batch_size, rows, max_actions, support), dtype=np.int64)
    legal = np.zeros((batch_size, rows, max_actions), dtype=np.bool_)
    selected_indices = np.empty((batch_size, rows), dtype=np.int64)
    selected = np.empty((batch_size, rows, support), dtype=np.int64)
    before = np.empty_like(selected)
    after = np.empty_like(selected)
    for row_index, decision in enumerate(row_decisions):
        action_values = np.asarray(
            [action.counts for action in decision.legal_actions], dtype=np.int64
        )
        action_count = action_values.shape[0]
        candidates[:, row_index, :action_count, :] = action_values
        legal[:, row_index, :action_count] = True
        selected_indices[:, row_index] = decision.selected_action_index
        selected[:, row_index, :] = decision.selected_action.counts
        before[:, row_index, :] = decision.residual_before
        after[:, row_index, :] = decision.residual_after
    return CoverSupervision(
        nu=np.full(batch_size, cover.nu, dtype=np.int64),
        nu_legal_mask=np.tile(
            np.asarray(decisions.nu_legal_mask, dtype=np.bool_), (batch_size, 1)
        ),
        nu_target_indices=np.full(
            batch_size, decisions.nu_target_index, dtype=np.int64
        ),
        group_legal_mask=np.tile(
            np.asarray(decisions.group_legal_mask, dtype=np.bool_), (batch_size, 1)
        ),
        group_target_indices=np.full(
            batch_size, decisions.group_target_index, dtype=np.int64
        ),
        row_labels=tuple(row.label for row in decisions.rows),
        row_multiplicities=np.asarray(
            [row.multiplicity for row in decisions.rows], dtype=np.int64
        ),
        row_free_dimensions=np.asarray(
            [row.free_dimension for row in decisions.rows], dtype=np.int64
        ),
        selected_action_indices=selected_indices,
        selected_actions=selected,
        candidate_actions=candidates,
        candidate_legal_mask=legal,
        residual_before=before,
        residual_after=after,
        sample_weights=np.full(batch_size, 1.0 if enabled else 0.0, dtype=np.float64),
    )


def _geometry_supervision(
    target: ExactFixtureTrainingTarget,
    *,
    batch_size: int,
    enabled: bool,
) -> GeometrySupervision:
    return GeometrySupervision(
        scaffolds=tuple(target.canonical_scaffold for _ in range(batch_size)),
        metric=np.tile(target.metric[None, :, :], (batch_size, 1, 1)),
        lattice_latent=np.tile(target.lattice_latent[None, :], (batch_size, 1)),
        intrinsic_coordinates=np.tile(
            target.intrinsic_coordinates[None, :], (batch_size, 1)
        ),
        period_basis=np.tile(target.period_basis[None, :, :], (batch_size, 1, 1)),
        sample_weights=np.full(batch_size, 1.0 if enabled else 0.0, dtype=np.float64),
    )


def _training_batch(
    model: CrystalAtlasModel,
    target: ExactFixtureTrainingTarget,
    *,
    batch_size: int,
    population_size: int,
    dataset_fingerprint: str,
    target_fingerprint: str,
    measure: TrainingMeasure,
) -> TrainingBatch:
    enabled = measure is TrainingMeasure.STRUCTURE
    record_ids = tuple(
        f"{EXACT_OBJECTIVE_FIXTURE_ID}-replica-{index:06d}"
        for index in range(batch_size)
    )
    return TrainingBatch(
        record_ids=record_ids,
        target_fingerprints=tuple(target_fingerprint for _ in range(batch_size)),
        dataset_fingerprint=dataset_fingerprint,
        measure=measure,
        bucket_key=_topology_key(target),
        composition=_composition_supervision(
            model,
            target,
            batch_size=batch_size,
            measure=measure,
            population_multiplicity=population_size,
        ),
        cover=_cover_supervision(target, batch_size=batch_size, enabled=enabled),
        geometry=_geometry_supervision(target, batch_size=batch_size, enabled=enabled),
    )


def _build_fixture_batches(
    model: CrystalAtlasModel,
    *,
    batch_size: int,
) -> _FixtureBatches:
    scientific_config = model.scientific_config
    contract = _fixture_contract(scientific_config, batch_size=batch_size)
    dataset_fingerprint = _fingerprint(
        {
            "schema": "crystalatlas.objective-benchmark-dataset.v1",
            "fixture_fingerprint": contract["fingerprint"],
            "batch_size": batch_size,
            "unique_compositions": 1,
        }
    )
    target = lift_exact_fixture_target(
        _fixture_scaffold(),
        np.diag([16.0, 25.0, 36.0]),
        catalogue=MINIMAL_CATALOGUE,
        domain=scientific_config.domain,
        num_space_groups=230,
    )
    target_fingerprint = _target_fingerprint(target)
    structure = _training_batch(
        model,
        target,
        batch_size=batch_size,
        population_size=batch_size,
        dataset_fingerprint=dataset_fingerprint,
        target_fingerprint=target_fingerprint,
        measure=TrainingMeasure.STRUCTURE,
    )
    composition = _training_batch(
        model,
        target,
        batch_size=1,
        population_size=batch_size,
        dataset_fingerprint=dataset_fingerprint,
        target_fingerprint=target_fingerprint,
        measure=TrainingMeasure.UNIQUE_COMPOSITION,
    )
    source = {
        **contract,
        "dataset_fingerprint": dataset_fingerprint,
        "target_fingerprint": target_fingerprint,
        "record_ids_sha256": _fingerprint(
            {"record_ids": list(structure.record_ids)}
        ),
        "actual_structure_count": batch_size,
        "actual_unique_composition_count": 1,
    }
    return _FixtureBatches(
        composition=composition,
        structure=structure,
        data_measure=ObjectiveDataMeasure(
            structure_count=batch_size,
            unique_composition_count=1,
            dataset_fingerprint=dataset_fingerprint,
        ),
        source_manifest=source,
    )


def _resolve_device(requested: str) -> Any:
    normalized = requested.strip().lower()
    if normalized == "auto":
        return torch.device("cuda", 0) if torch.cuda.is_available() else torch.device("cpu")
    try:
        device = torch.device(normalized)
    except (RuntimeError, ValueError) as exc:
        raise DomainError(f"invalid benchmark device {requested!r}") from exc
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise DomainError("CUDA was requested but is unavailable")
        index = 0 if device.index is None else device.index
        if not 0 <= index < torch.cuda.device_count():
            raise DomainError("requested CUDA device lies outside the visible devices")
        return torch.device("cuda", index)
    if device.type != "cpu" or device.index is not None:
        raise DomainError("training-objective benchmark device must be CPU or CUDA")
    return device


def _precision_policy(precision: str, device: Any) -> tuple[Any, PrecisionMode]:
    _require_deterministic_cuda_environment(device)
    if precision == "float64":
        return torch.float64, PrecisionMode.FLOAT32
    if precision == "float32":
        return torch.float32, PrecisionMode.FLOAT32
    if precision == "bfloat16":
        return torch.float32, PrecisionMode.BFLOAT16
    if device.type != "cuda":
        raise DomainError("FP16 training-objective benchmarking requires CUDA")
    return torch.float32, PrecisionMode.FLOAT16


def _batch_numpy_bytes(batch: TrainingBatch) -> int:
    values = (
        batch.composition.token_ids,
        batch.composition.token_types,
        batch.composition.legal_mask,
        batch.composition.sequence_mask,
        batch.composition.atomic_numbers,
        batch.composition.reduced_counts,
        batch.composition.composition_multiplicities,
        batch.composition.sample_weights,
        batch.cover.nu,
        batch.cover.nu_legal_mask,
        batch.cover.nu_target_indices,
        batch.cover.group_legal_mask,
        batch.cover.group_target_indices,
        batch.cover.row_multiplicities,
        batch.cover.row_free_dimensions,
        batch.cover.selected_action_indices,
        batch.cover.selected_actions,
        batch.cover.candidate_actions,
        batch.cover.candidate_legal_mask,
        batch.cover.residual_before,
        batch.cover.residual_after,
        batch.cover.sample_weights,
        batch.geometry.metric,
        batch.geometry.lattice_latent,
        batch.geometry.intrinsic_coordinates,
        batch.geometry.period_basis,
        batch.geometry.sample_weights,
    )
    return sum(int(value.nbytes) for value in values)


def _optimizer_tensor_values(optimizer: Any) -> tuple[Any, ...]:
    return tuple(
        value
        for state in optimizer.state.values()
        for value in state.values()
        if isinstance(value, Tensor)
    )


def _package_artifact_sha256() -> str:
    """Content-address the installed CrystalAtlas package tree used by the run."""

    root = Path(__file__).resolve().parent
    digest = sha256()
    files = tuple(
        sorted(
            (
                path
                for path in root.rglob("*")
                if path.is_file()
                and "__pycache__" not in path.parts
                and path.suffix not in {".pyc", ".pyo"}
            ),
            key=lambda path: path.relative_to(root).as_posix(),
        )
    )
    if not files:
        raise InvariantViolation("the installed CrystalAtlas package tree is empty")
    for path in files:
        relative = path.relative_to(root).as_posix().encode("utf-8")
        content = path.read_bytes()
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        digest.update(len(content).to_bytes(8, "big"))
        digest.update(content)
    return digest.hexdigest()


def run_single_device_training_objective_benchmark(
    config: TrainingObjectiveBenchmarkConfig | None = None,
) -> TrainingObjectiveBenchmarkReport:
    """Measure repeated complete joint training windows on one exact fixture.

    A successful return is a canonical evidence report.  Failures raise and
    never masquerade as a partially completed benchmark.  Three untimed
    factor-specific optimizer steps are executed first so the measured joint
    phase respects the package's mandatory factor-first schedule and has
    initialized Adam state for all three components.
    """

    policy = TrainingObjectiveBenchmarkConfig() if config is None else config
    if not isinstance(policy, TrainingObjectiveBenchmarkConfig):
        raise TypeError("config must be TrainingObjectiveBenchmarkConfig")
    stage = "device resolution"
    previous_threads = torch.get_num_threads()
    previous_deterministic = torch.are_deterministic_algorithms_enabled()
    previous_warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    previous_matmul_precision = torch.get_float32_matmul_precision()
    previous_cuda_tf32 = torch.backends.cuda.matmul.allow_tf32
    previous_cudnn_tf32 = torch.backends.cudnn.allow_tf32
    previous_cudnn_benchmark = torch.backends.cudnn.benchmark
    previous_cudnn_deterministic = torch.backends.cudnn.deterministic
    try:
        device = _resolve_device(policy.device)
        master_dtype, trainer_precision = _precision_policy(policy.precision, device)
        effective_threads = policy.cpu_threads or previous_threads
        if policy.cpu_threads is not None:
            torch.set_num_threads(policy.cpu_threads)
        torch.use_deterministic_algorithms(True, warn_only=False)
        torch.set_float32_matmul_precision("highest")
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        rng_devices = [cast(int, device.index)] if device.type == "cuda" else []
        with torch.random.fork_rng(devices=rng_devices, enabled=True):
            cpu_generator = torch.Generator(device="cpu")
            cpu_generator.manual_seed(policy.seed)
            torch.set_rng_state(cpu_generator.get_state())
            if device.type == "cuda":
                cuda_generator = torch.Generator(device=device)
                cuda_generator.manual_seed(policy.seed)
                torch.cuda.set_rng_state(cuda_generator.get_state(), device=device)

            stage = "model construction"
            model = CrystalAtlasModel(
                policy.scientific_config,
                catalogue=MINIMAL_CATALOGUE,
            ).to(device=device, dtype=master_dtype)
            _initialize_model(model, policy.seed)
            initial_model_sha256 = _model_sha256(model)

            stage = "exact fixture construction"
            fixture = _build_fixture_batches(model, batch_size=policy.batch_size)
            objective = CrystalAtlasTrainingObjective(model, fixture.data_measure)
            audit = ExactBatchCrystallographicAudit(model)
            explicit_audit = audit(fixture.structure, torch.float64)
            explicit_audit.require_passed()
            audit_residuals: Any = explicit_audit.residuals
            maximum_audit_residual = float(audit_residuals.abs().max().item())

            stage = "trainer construction"
            optimizer_contract = {
                "class": "torch.optim.AdamW",
                "learning_rate": float(policy.learning_rate),
                "betas": [0.9, 0.999],
                "eps": 1.0e-8,
                "weight_decay": 0.01,
                "amsgrad": False,
                "maximize": False,
                "foreach": False,
                "capturable": False,
                "differentiable": False,
                "fused": False,
            }
            optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=policy.learning_rate,
                betas=(0.9, 0.999),
                eps=1.0e-8,
                weight_decay=0.01,
                amsgrad=False,
                maximize=False,
                foreach=False,
                capturable=False,
                differentiable=False,
                fused=False,
            )
            schedule = StagedTrainingSchedule(1, 1, 1, 1)
            trainer_config = TrainerConfig(
                accumulation_steps=1,
                precision=trainer_precision,
                ema_decay=policy.ema_decay,
                transactional_optimizer_step=False,
            )
            trainer = ScalableTrainer(
                model,
                optimizer,
                schedule,
                config=trainer_config,
            )
            interaction_observations: list[int] = []

            def observe_interactions(_module: Any, _inputs: Any, output: Any) -> None:
                edges = getattr(output, "edges", None)
                if edges is not None:
                    interaction_observations.append(int(edges.num_edges))
                    return
                count = getattr(output, "interaction_count", None)
                if count is not None:
                    interaction_observations.append(int(count))

            hook = model.geometry_field.register_forward_hook(observe_interactions)
            try:
                stage = "factor-first preconditioning"
                preconditioning = []
                phase_batches = (
                    fixture.composition,
                    fixture.structure,
                    fixture.structure,
                )
                for expected_phase, batch in zip(
                    (
                        TrainingPhase.COMPOSITION,
                        TrainingPhase.COVER,
                        TrainingPhase.GEOMETRY,
                    ),
                    phase_batches,
                    strict=True,
                ):
                    phase_report = trainer.train_epoch(
                        (batch,),
                        objective,
                        crystallographic_audit=audit,
                    )
                    if phase_report.phase is not expected_phase:
                        raise InvariantViolation(
                            "trainer escaped the pinned factor-first benchmark schedule"
                        )
                    preconditioning.append(
                        {
                            "phase": phase_report.phase.value,
                            "optimizer_steps": phase_report.optimizer_steps,
                            "structures": phase_report.global_structures,
                            "metrics": dict(sorted(phase_report.metrics.items())),
                        }
                    )
                if trainer.current_phase is not TrainingPhase.JOINT:
                    raise InvariantViolation("benchmark did not reach the joint phase")

                stage = "joint warmup"
                for _ in range(policy.warmup_steps):
                    trainer.train_window(
                        (fixture.structure,),
                        objective,
                        crystallographic_audit=audit,
                    )
                premeasurement_model_sha256 = _model_sha256(model)
                interaction_observations.clear()
                if device.type == "cuda":
                    torch.cuda.reset_peak_memory_stats(device)

                durations: list[float] = []
                metrics: list[Mapping[str, float]] = []
                gradient_norms: list[float] = []
                gradient_maxima: list[float] = []
                gradient_counts: list[int] = []
                iteration_interaction_counts: list[tuple[int, ...]] = []
                stage = "joint measurement"
                for _ in range(policy.measurement_steps):
                    interaction_start = len(interaction_observations)
                    duration, joint_report = _timed(
                        device,
                        lambda: trainer.train_window(
                            (fixture.structure,),
                            objective,
                            crystallographic_audit=audit,
                        ),
                    )
                    if joint_report.phase is not TrainingPhase.JOINT:
                        raise InvariantViolation("a measured window was not joint training")
                    interactions = tuple(
                        interaction_observations[interaction_start:]
                    )
                    if not interactions or sum(interactions) <= 0:
                        raise InvariantViolation(
                            "the pinned objective workload exposed no positive periodic "
                            "edge or dense geometry interaction"
                        )
                    durations.append(duration)
                    metrics.append(dict(joint_report.metrics))
                    gradient_norms.append(
                        joint_report.gradients_before_clip.global_l2_norm
                    )
                    gradient_maxima.append(
                        joint_report.gradients_before_clip.max_absolute_value
                    )
                    gradient_counts.append(
                        joint_report.gradients_before_clip.tensors_with_grad
                    )
                    iteration_interaction_counts.append(interactions)
            finally:
                hook.remove()

            stage = "report construction"
            expected_updates = 3 + policy.warmup_steps + policy.measurement_steps
            if trainer.cursor.optimizer_step != expected_updates:
                raise InvariantViolation("trainer optimizer-step count changed unexpectedly")
            parameter_tensors = tuple(model.parameters())
            buffer_tensors = tuple(model.buffers())
            optimizer_tensors = _optimizer_tensor_values(optimizer)
            logical = {
                "parameter_bytes": _logical_tensor_bytes(parameter_tensors),
                "buffer_bytes": _logical_tensor_bytes(buffer_tensors),
                "optimizer_state_bytes": _logical_tensor_bytes(optimizer_tensors),
                "ema_shadow_bytes": (
                    _logical_tensor_bytes(parameter_tensors)
                    if trainer.ema is not None
                    else 0
                ),
                "materialized_structure_batch_numpy_bytes": _batch_numpy_bytes(
                    fixture.structure
                ),
                "materialized_composition_batch_numpy_bytes": _batch_numpy_bytes(
                    fixture.composition
                ),
            }
            memory = {
                "logical_tensor_payload": logical,
                "known_live_logical_tensor_bytes": sum(logical.values()),
                "logical_accounting_excludes": [
                    "autograd saved activations",
                    "parameter gradient tensors materialized before the optimizer step",
                    "temporary kernels",
                    "allocator fragmentation",
                    "framework and process memory",
                    "Python crystallographic objects",
                ],
                "backend_allocator": _allocator_memory(device),
            }
            component_counts = {
                "composition": sum(
                    int(value.numel()) for value in model.composition_model.parameters()
                ),
                "cover": sum(
                    int(value.numel()) for value in model.cover_selector.parameters()
                ),
                "geometry": sum(
                    int(value.numel()) for value in model.geometry_field.parameters()
                ),
            }
            model_manifest = {
                "class": f"{type(model).__module__}.{type(model).__qualname__}",
                "scientific_config_fingerprint": policy.scientific_config.fingerprint,
                "catalogue": MINIMAL_CATALOGUE.identity,
                "trainable_parameter_count": sum(component_counts.values()),
                "component_parameter_counts": component_counts,
                "master_dtype": str(master_dtype),
                "trainer_precision": trainer_precision.value,
                "initial_model_sha256": initial_model_sha256,
                "parameter_initializer": DYADIC_SIGNAL_INITIALIZER,
                "crystalatlas_version": __version__,
                "crystalatlas_package_tree_sha256": _package_artifact_sha256(),
            }
            structure_batch = fixture.structure
            topology = structure_batch.bucket_key.topology
            workload = {
                "workload_generator": EXACT_OBJECTIVE_FIXTURE_ID,
                "requested_batch_size": policy.batch_size,
                "actual_batch_size": len(structure_batch.record_ids),
                "atoms_per_structure": structure_batch.bucket_key.atom_count,
                "orbits_per_structure": len(structure_batch.geometry.scaffolds[0].orbits),
                "composition_tokens_per_structure": int(
                    structure_batch.composition.sequence_mask[0].sum()
                ),
                "selector_rows": len(topology.row_labels),
                "selector_candidate_slots": int(
                    structure_batch.cover.candidate_actions.shape[2]
                ),
                "topology": topology.to_dict(),
                "data_measure": fixture.data_measure.to_dict(),
                "objective_contract": objective.contract,
                "objective_contract_fingerprint": objective.contract_fingerprint,
                "optimizer": optimizer_contract,
                "trainer_config": trainer_config.to_dict(),
                "trainer_config_fingerprint": trainer_config.fingerprint,
                "staged_schedule": schedule.to_dict(),
                "staged_schedule_fingerprint": schedule.fingerprint,
                "factor_first_preconditioning": preconditioning,
                "untimed_preconditioning_optimizer_steps": 3,
                "warmup_optimizer_steps": policy.warmup_steps,
                "measurement_optimizer_steps": policy.measurement_steps,
                "data_authentication_and_batch_collation_timed": False,
                "checkpoint_io_timed": False,
            }
            workload["fingerprint"] = _fingerprint(workload)
            phase = TrainingWindowMeasurement(
                durations_seconds=tuple(durations),
                structures_per_iteration=len(structure_batch.record_ids),
                atoms_per_iteration=(
                    len(structure_batch.record_ids) * structure_batch.bucket_key.atom_count
                ),
                metrics=tuple(metrics),
                gradient_l2_norms=tuple(gradient_norms),
                gradient_maxima=tuple(gradient_maxima),
                gradient_tensor_counts=tuple(gradient_counts),
                geometry_interaction_kind=(
                    "dense_atom_pairs"
                    if policy.scientific_config.model.geometry_backbone
                    == "invariant_global"
                    else "periodic_image_edges"
                ),
                geometry_interaction_counts_per_iteration=tuple(
                    iteration_interaction_counts
                ),
            )
            numerics = {
                "deterministic_execution_requested": True,
                "float64_crystallographic_audit_passed": True,
                "float64_crystallographic_audit_tolerance": explicit_audit.tolerance,
                "float64_crystallographic_audit_maximum_residual": (
                    maximum_audit_residual
                ),
                "all_measured_losses_finite": True,
                "all_measured_gradients_finite": True,
                "optimizer_updates_completed": trainer.cursor.optimizer_step,
                "premeasurement_model_sha256": premeasurement_model_sha256,
                "final_model_sha256": _model_sha256(model),
            }
            runtime = _runtime_manifest(device, cpu_threads=effective_threads)
            runtime["numpy"] = np.__version__
            return TrainingObjectiveBenchmarkReport(
                config=policy,
                runtime=runtime,
                source=fixture.source_manifest,
                model=model_manifest,
                workload=workload,
                phase=phase,
                memory=memory,
                numerics=numerics,
            )
    except DomainError:
        raise
    except Exception as exc:
        raise TrainingObjectiveBenchmarkExecutionError(stage, exc) from exc
    finally:
        if policy.cpu_threads is not None and torch.get_num_threads() != previous_threads:
            torch.set_num_threads(previous_threads)
        torch.use_deterministic_algorithms(previous_deterministic, warn_only=previous_warn_only)
        torch.set_float32_matmul_precision(previous_matmul_precision)
        torch.backends.cuda.matmul.allow_tf32 = previous_cuda_tf32
        torch.backends.cudnn.allow_tf32 = previous_cudnn_tf32
        torch.backends.cudnn.benchmark = previous_cudnn_benchmark
        torch.backends.cudnn.deterministic = previous_cudnn_deterministic


__all__ = [
    "EXACT_OBJECTIVE_FIXTURE_ID",
    "TRAINING_OBJECTIVE_BENCHMARK_METHOD",
    "TRAINING_OBJECTIVE_BENCHMARK_SCHEMA",
    "ExactObjectiveFixtureSource",
    "TrainingObjectiveBenchmarkConfig",
    "TrainingObjectiveBenchmarkExecutionError",
    "TrainingObjectiveBenchmarkReport",
    "TrainingWindowMeasurement",
    "objective_benchmark_micro_config",
    "run_single_device_training_objective_benchmark",
]
