"""Same-workload single-device comparison of two periodic geometry backends.

This diagnostic compares implementation cost on one deterministic, constant-
density workload.  It is deliberately not a predictive-quality experiment and
the two models are not parameter- or FLOP-matched.  Dense ordered atom pairs
and local periodic-image edges are reported as different interaction kinds;
the former are never mislabeled as edges.
"""

from __future__ import annotations

import json
import math
import statistics
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from hashlib import sha256
from importlib import metadata
from pathlib import Path
from typing import Any, TypeVar

from ._optional import require_torch
from .benchmarking import (
    DYADIC_SIGNAL_INITIALIZER,
    _activation_edges,
    _canonical_json,
    _dyadic_modular_signal,
    _edge_tensors,
    _fingerprint,
    _initialize_model,
    _logical_tensor_bytes,
    _model_sha256,
    _precision_policy,
    _resolve_device,
    _runtime_manifest,
    _timed,
)
from .config import DomainConfig
from .errors import CrystalAtlasError, DomainError
from .nn.equivariant_transformer import (
    PeriodicEdges,
    PeriodicSE3Transformer,
    PeriodicTransformerConfig,
    enumerate_packed_periodic_edges,
)
from .nn.invariant_transformer import (
    InvariantPeriodicTransformer,
    InvariantPeriodicTransformerConfig,
)

torch = require_torch()
Tensor = torch.Tensor

_T = TypeVar("_T")
_U = TypeVar("_U")

BACKEND_COMPARISON_SCHEMA = "crystalatlas.geometry-backend-comparison.v2"
BACKEND_COMPARISON_METHOD = "crystalatlas.constant-density-backend-comparison.v3"
_INVARIANT = "invariant_global"
_LOCAL = "equivariant_local"
_DEFAULT_INTERACTION_CUTOFF_ANGSTROM = float(
    DomainConfig().interaction_cutoff_angstrom
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


@dataclass(frozen=True)
class BackendComparisonConfig:
    """Pinned workload and model dimensions for a diagnostic comparison."""

    device: str = "auto"
    precision: str = "float32"
    batch_size: int = 2
    atoms_per_structure: int = 20
    volume_per_atom: float = 16.0
    # Current production field input: 4 atom + 64 species + 128 composition
    # + 17 time + 6 bounded-lattice scalar channels.
    input_scalar_dim: int = 219
    scalar_dim: int = 256
    pair_dim: int = 96
    vector_dim: int = 32
    num_heads: int = 8
    num_layers: int = 10
    invariant_feedforward_multiplier: int = 4
    radial_basis_dim: int = 32
    reciprocal_mode_radius: int = 2
    direct_harmonics: int = 2
    cutoff: float = _DEFAULT_INTERACTION_CUTOFF_ANGSTROM
    max_candidates_per_pair: int = 32_768
    max_edges_per_structure: int = 500_000
    max_total_edges: int = 1_000_000
    max_dense_pair_interactions: int = 500_000
    warmup_steps: int = 3
    measurement_steps: int = 12
    seed: int = 1729
    learning_rate: float = 1.0e-3
    cpu_threads: int | None = None
    measure_local_prebuilt_core: bool = True

    def __post_init__(self) -> None:
        if not isinstance(self.device, str) or not self.device.strip():
            raise DomainError("benchmark device must be a nonempty string")
        if self.precision not in {"float32", "float64", "bfloat16", "float16"}:
            raise DomainError("benchmark precision is not supported")
        for name, value in (
            ("batch_size", self.batch_size),
            ("atoms_per_structure", self.atoms_per_structure),
            ("input_scalar_dim", self.input_scalar_dim),
            ("scalar_dim", self.scalar_dim),
            ("pair_dim", self.pair_dim),
            ("vector_dim", self.vector_dim),
            ("num_heads", self.num_heads),
            ("num_layers", self.num_layers),
            (
                "invariant_feedforward_multiplier",
                self.invariant_feedforward_multiplier,
            ),
            ("radial_basis_dim", self.radial_basis_dim),
            ("reciprocal_mode_radius", self.reciprocal_mode_radius),
            ("direct_harmonics", self.direct_harmonics),
            ("max_candidates_per_pair", self.max_candidates_per_pair),
            ("max_edges_per_structure", self.max_edges_per_structure),
            ("max_total_edges", self.max_total_edges),
            ("max_dense_pair_interactions", self.max_dense_pair_interactions),
            ("warmup_steps", self.warmup_steps),
            ("measurement_steps", self.measurement_steps),
        ):
            _positive_integer(name, value)
        if self.atoms_per_structure < 2:
            raise DomainError("atoms_per_structure must be at least two")
        if self.scalar_dim % self.num_heads:
            raise DomainError("scalar_dim must be divisible by num_heads")
        if self.vector_dim % self.num_heads:
            raise DomainError(
                "vector_dim must be divisible by num_heads for the local backend"
            )
        dense_count = self.batch_size * self.atoms_per_structure**2
        if dense_count > self.max_dense_pair_interactions:
            raise DomainError(
                "constant-density workload exceeds max_dense_pair_interactions"
            )
        if self.max_total_edges < self.batch_size:
            raise DomainError("max_total_edges must permit at least one edge per structure")
        if isinstance(self.seed, bool) or not isinstance(self.seed, int) or self.seed < 0:
            raise DomainError("benchmark seed must be a nonnegative integer")
        if self.cpu_threads is not None:
            _positive_integer("cpu_threads", self.cpu_threads)
        if type(self.measure_local_prebuilt_core) is not bool:
            raise DomainError("measure_local_prebuilt_core must be boolean")
        _positive_finite("volume_per_atom", self.volume_per_atom)
        _positive_finite("cutoff", self.cutoff)
        _positive_finite("learning_rate", self.learning_rate)

    def to_dict(self) -> dict[str, object]:
        return {
            "device": self.device,
            "precision": self.precision,
            "batch_size": self.batch_size,
            "atoms_per_structure": self.atoms_per_structure,
            "volume_per_atom": float(self.volume_per_atom),
            "input_scalar_dim": self.input_scalar_dim,
            "scalar_dim": self.scalar_dim,
            "pair_dim": self.pair_dim,
            "vector_dim": self.vector_dim,
            "num_heads": self.num_heads,
            "num_layers": self.num_layers,
            "invariant_feedforward_multiplier": self.invariant_feedforward_multiplier,
            "radial_basis_dim": self.radial_basis_dim,
            "reciprocal_mode_radius": self.reciprocal_mode_radius,
            "direct_harmonics": self.direct_harmonics,
            "cutoff": float(self.cutoff),
            "max_candidates_per_pair": self.max_candidates_per_pair,
            "max_edges_per_structure": self.max_edges_per_structure,
            "max_total_edges": self.max_total_edges,
            "max_dense_pair_interactions": self.max_dense_pair_interactions,
            "warmup_steps_per_phase": self.warmup_steps,
            "measurement_steps_per_phase": self.measurement_steps,
            "seed": self.seed,
            "learning_rate": float(self.learning_rate),
            "cpu_threads": self.cpu_threads,
            "measure_local_prebuilt_core": self.measure_local_prebuilt_core,
        }

    @property
    def fingerprint(self) -> str:
        return _fingerprint(self.to_dict())

    def to_manifest(self) -> dict[str, object]:
        payload = self.to_dict()
        payload["fingerprint"] = self.fingerprint
        return payload


@dataclass(frozen=True)
class BackendPhaseMeasurement:
    """Raw synchronized timings for one backend and one declared phase."""

    phase: str
    includes: tuple[str, ...]
    durations_seconds: tuple[float, ...]
    structures_per_iteration: int
    atoms_per_iteration: int
    interaction_kind: str
    interactions_per_iteration: int

    def __post_init__(self) -> None:
        if not self.phase or not self.includes or not self.interaction_kind:
            raise DomainError("a comparison phase must have a name and declared scope")
        if not self.durations_seconds or any(
            not math.isfinite(value) or value <= 0.0
            for value in self.durations_seconds
        ):
            raise DomainError("comparison phase timings must be finite and positive")
        for name, value in (
            ("structures_per_iteration", self.structures_per_iteration),
            ("atoms_per_iteration", self.atoms_per_iteration),
            ("interactions_per_iteration", self.interactions_per_iteration),
        ):
            _positive_integer(name, value)

    def to_manifest(self) -> dict[str, object]:
        total = math.fsum(self.durations_seconds)
        iterations = len(self.durations_seconds)
        ordered = sorted(self.durations_seconds)

        def quantile(probability: float) -> float:
            location = probability * (iterations - 1)
            lower = math.floor(location)
            upper = math.ceil(location)
            if lower == upper:
                return ordered[lower]
            weight = location - lower
            return (1.0 - weight) * ordered[lower] + weight * ordered[upper]

        return {
            "phase": self.phase,
            "includes": list(self.includes),
            "iterations": iterations,
            "durations_seconds": list(self.durations_seconds),
            "total_seconds": total,
            "mean_seconds": total / iterations,
            "median_seconds": statistics.median(self.durations_seconds),
            "quartile_25_seconds": quantile(0.25),
            "quartile_75_seconds": quantile(0.75),
            "minimum_seconds": min(self.durations_seconds),
            "maximum_seconds": max(self.durations_seconds),
            "structures_per_second": self.structures_per_iteration * iterations / total,
            "atoms_per_second": self.atoms_per_iteration * iterations / total,
            "interaction_kind": self.interaction_kind,
            "interactions_per_iteration": self.interactions_per_iteration,
            "interactions_per_second": (
                self.interactions_per_iteration * iterations / total
            ),
        }


@dataclass(frozen=True)
class BackendComparisonReport:
    """Strict, content-addressed evidence from a completed comparison."""

    config: BackendComparisonConfig
    runtime: Mapping[str, object]
    workload: Mapping[str, object]
    backends: Mapping[str, object]
    descriptive_ratios: Mapping[str, object]

    def __post_init__(self) -> None:
        if set(self.backends) != {_INVARIANT, _LOCAL}:
            raise DomainError("comparison report must contain exactly both backends")
        matching = self.workload.get("matching_contract")
        if not isinstance(matching, Mapping):
            raise DomainError("comparison workload is missing its matching contract")
        declared = matching.get("parameter_matched")
        if type(declared) is not bool or declared != self._models_parameter_matched():
            raise DomainError(
                "parameter-matching declaration disagrees with measured model sizes"
            )
        precision_contract = self.workload.get("precision_contract")
        if not isinstance(precision_contract, Mapping) or type(
            precision_contract.get("backend_arithmetic_precision_matched")
        ) is not bool:
            raise DomainError("comparison workload is missing its precision contract")
        _canonical_json(self._payload())

    def _models_parameter_matched(self) -> bool:
        counts: list[int] = []
        for name in (_INVARIANT, _LOCAL):
            backend = self.backends[name]
            if not isinstance(backend, Mapping):
                raise DomainError("comparison backend manifest is malformed")
            count = backend.get("trainable_parameter_count")
            if isinstance(count, bool) or not isinstance(count, int) or count < 1:
                raise DomainError("comparison parameter count is malformed")
            counts.append(count)
        return counts[0] == counts[1]

    def _payload(self) -> dict[str, object]:
        precision_contract = self.workload["precision_contract"]
        if not isinstance(precision_contract, Mapping):  # guarded at construction
            raise AssertionError("comparison precision contract disappeared")
        return {
            "schema": BACKEND_COMPARISON_SCHEMA,
            "status": "completed",
            "method": BACKEND_COMPARISON_METHOD,
            "factor": "geometry-backend",
            "config": self.config.to_manifest(),
            "runtime": dict(self.runtime),
            "workload": dict(self.workload),
            "backends": dict(self.backends),
            "descriptive_ratios": dict(self.descriptive_ratios),
            "interpretation": {
                "predictive_quality_measured": False,
                "predictive_superiority_established": False,
                "models_parameter_matched": self._models_parameter_matched(),
                "models_flop_matched": False,
                "models_weight_matched": False,
                "backend_arithmetic_precision_matched": precision_contract[
                    "backend_arithmetic_precision_matched"
                ],
                "multi_device_scaling_measured": False,
                "allocator_values_backend_isolated": False,
                "allocator_comparison_permitted": False,
                "permitted_conclusion": (
                    "single-device implementation cost on this exact synthetic "
                    "constant-density workload and software stack"
                ),
                "measured_scope": (
                    "the two periodic Transformer implementations, including their "
                    "geometry construction; excludes the IntrinsicGeometryField, exact "
                    "cover expansion/reduction, and its certified-geometry fast path"
                ),
                "forbidden_conclusions": [
                    "predictive superiority",
                    "sample-quality superiority",
                    "asymptotic superiority",
                    "multi-GPU scaling",
                    "equal-capacity architectural superiority",
                    "backend-isolated peak-memory superiority",
                ],
                "interaction_count_warning": (
                    "dense ordered atom pairs and exact periodic-image edges are "
                    "different computational objects; their counts are not a fidelity- "
                    "or compute-equivalence claim"
                ),
            },
        }

    @property
    def fingerprint(self) -> str:
        return _fingerprint(self._payload())

    def to_manifest(self) -> dict[str, object]:
        payload = self._payload()
        payload["fingerprint"] = self.fingerprint
        return payload

    def to_json(self, *, indent: int | None = 2) -> str:
        return json.dumps(
            self.to_manifest(),
            allow_nan=False,
            ensure_ascii=True,
            indent=indent,
            separators=(",", ":") if indent is None else None,
            sort_keys=True,
        )


class BackendComparisonExecutionError(CrystalAtlasError):
    """A comparison failed before a complete report could be emitted."""

    def __init__(self, stage: str, cause: BaseException) -> None:
        self.stage = stage
        self.cause_type = type(cause).__name__
        super().__init__(
            f"geometry backend comparison failed during {stage}: "
            f"{self.cause_type}: {cause}"
        )


@dataclass(frozen=True)
class _Workload:
    features: Tensor
    fractional: Tensor
    lattices: Tensor
    graph_index: Tensor
    scalar_target: Tensor
    vector_target: Tensor


def _tensor_content_manifest(value: Tensor) -> dict[str, object]:
    """Describe and hash the exact represented tensor bytes.

    The digest is intentionally computed after the workload has been converted
    to its measurement master dtype.  A benchmark report therefore identifies
    the bytes actually consumed by the models, not merely the constructive
    formula that was intended to produce them.
    """

    represented = value.detach().contiguous().cpu()
    raw_bytes = represented.view(torch.uint8).numpy().tobytes(order="C")
    return {
        "shape": list(represented.shape),
        "dtype": str(represented.dtype),
        "numel": int(represented.numel()),
        "logical_bytes": int(represented.numel()) * int(represented.element_size()),
        "sha256": sha256(raw_bytes).hexdigest(),
    }


def _source_file_manifest(path: Path) -> dict[str, object]:
    if not path.is_file():
        raise RuntimeError(f"comparison source module is unavailable: {path.name}")
    payload = path.read_bytes()
    return {
        "module_file": path.name,
        "bytes": len(payload),
        "sha256": sha256(payload).hexdigest(),
    }


def _implementation_manifest(
    invariant_config: InvariantPeriodicTransformerConfig,
    local_config: PeriodicTransformerConfig,
) -> dict[str, object]:
    """Bind evidence to equations-as-code and fully resolved model configs."""

    package_root = Path(__file__).resolve().parent
    source_paths = {
        "backend_comparison": package_root / "backend_comparison.py",
        "benchmarking": package_root / "benchmarking.py",
        "invariant_transformer": package_root / "nn" / "invariant_transformer.py",
        "equivariant_transformer": package_root / "nn" / "equivariant_transformer.py",
    }
    try:
        package_version = metadata.version("crystalatlas")
    except metadata.PackageNotFoundError:
        package_version = "source-tree-uninstalled"
    resolved_configs = json.loads(
        _canonical_json(
            {
                _INVARIANT: asdict(invariant_config),
                _LOCAL: asdict(local_config),
            }
        )
    )
    payload: dict[str, object] = {
        "crystalatlas_version": package_version,
        "source_modules": {
            name: _source_file_manifest(path) for name, path in source_paths.items()
        },
        "resolved_model_configs": resolved_configs,
    }
    payload["fingerprint"] = _fingerprint(payload)
    return payload


def _make_constant_density_workload(
    config: BackendComparisonConfig,
    *,
    device: Any,
    dtype: Any,
) -> _Workload:
    batch = config.batch_size
    atoms = config.atoms_per_structure
    atom_index = torch.arange(atoms, dtype=torch.float64, device="cpu")
    graph_index_float = torch.arange(batch, dtype=torch.float64, device="cpu")
    base = torch.stack(
        (
            torch.remainder(19.0 * atom_index + 11.0, 251.0) / 251.0,
            torch.remainder(23.0 * atom_index + 17.0, 257.0) / 257.0,
            torch.remainder(29.0 * atom_index + 19.0, 263.0) / 263.0,
        ),
        dim=-1,
    )
    offsets = torch.stack(
        (
            0.031 * graph_index_float,
            0.047 * graph_index_float,
            0.059 * graph_index_float,
        ),
        dim=-1,
    )
    fractional = torch.remainder(base.unsqueeze(0) + offsets.unsqueeze(1), 1.0)

    # A unit-determinant lower-triangular shape gives a mildly triclinic cell.
    # Scaling by (N * volume_per_atom)^(1/3) pins the physical number density.
    shape = torch.tensor(
        ((1.0, 0.0, 0.0), (0.13, 1.0, 0.0), (0.07, 0.11, 1.0)),
        dtype=torch.float64,
    )
    cell_scale = (atoms * float(config.volume_per_atom)) ** (1.0 / 3.0)
    lattices = (cell_scale * shape).unsqueeze(0).repeat(batch, 1, 1)

    features = _dyadic_modular_signal(
        batch * atoms * config.input_scalar_dim,
        multiplier=11_406_733,
        offset=config.seed * 104_729 + 8_191,
        scale_exponent=-25,
    )
    features = features.reshape(batch, atoms, config.input_scalar_dim)
    scalar_target = _dyadic_modular_signal(
        batch * atoms * config.scalar_dim,
        multiplier=12_820_163,
        offset=1_234_567,
        scale_exponent=-26,
    )
    scalar_target = scalar_target.reshape(
        batch, atoms, config.scalar_dim
    )
    vector_target = _dyadic_modular_signal(
        batch * atoms * config.vector_dim * 3,
        multiplier=1_103_515,
        offset=7_654_321,
        scale_exponent=-27,
    )
    vector_target = vector_target.reshape(
        batch, atoms, config.vector_dim, 3
    )
    graph_index = torch.arange(batch, dtype=torch.long).repeat_interleave(atoms)

    def move(value: Tensor) -> Tensor:
        if value.is_floating_point():
            return value.to(device=device, dtype=dtype)
        return value.to(device=device)

    return _Workload(
        features=move(features),
        fractional=move(fractional),
        lattices=move(lattices),
        graph_index=move(graph_index),
        scalar_target=move(scalar_target),
        vector_target=move(vector_target),
    )


def _parameter_count(model: Any) -> int:
    return sum(int(parameter.numel()) for parameter in model.parameters())


def _finite_output(output: Any) -> bool:
    return bool(
        torch.isfinite(output.scalar_channels).all().item()
        and torch.isfinite(output.vector_channels).all().item()
        and torch.isfinite(output.pooled_scalars).all().item()
    )


def _final_measured_gradient_summary(model: Any) -> dict[str, object]:
    parameters = tuple(parameter for parameter in model.parameters() if parameter.requires_grad)
    gradients = tuple(parameter.grad for parameter in parameters)
    present = sum(gradient is not None for gradient in gradients)
    finite = all(
        gradient is not None and bool(torch.isfinite(gradient).all().item())
        for gradient in gradients
    )
    if not finite:
        raise RuntimeError("a trainable parameter has a missing or nonfinite gradient")
    return {
        "trainable_parameter_tensors": len(parameters),
        "final_measured_gradient_tensors_present": present,
        "final_measured_trainable_parameter_gradients_finite": True,
    }


def _optimizer_step_count(optimizer: Any) -> int:
    values = {
        int(value.item()) if isinstance(value, Tensor) else int(value)
        for state in optimizer.state.values()
        for name, value in state.items()
        if name == "step"
    }
    if len(values) != 1:
        raise RuntimeError("AdamW parameter states do not share one exact step count")
    return values.pop()


def _allocator_start(device: Any) -> dict[str, object]:
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        return {
            "allocator": "torch.cuda",
            "baseline_allocated_bytes": int(torch.cuda.memory_allocated(device)),
            "baseline_reserved_bytes": int(torch.cuda.memory_reserved(device)),
        }
    if device.type == "mps" and hasattr(torch.mps, "current_allocated_memory"):
        return {
            "allocator": "torch.mps",
            "baseline_allocated_bytes": int(torch.mps.current_allocated_memory()),
            "baseline_reserved_bytes": None,
        }
    return {
        "allocator": None,
        "baseline_allocated_bytes": None,
        "baseline_reserved_bytes": None,
    }


def _allocator_finish(device: Any, baseline: Mapping[str, object]) -> dict[str, object]:
    payload = dict(baseline)
    if device.type == "cuda":
        allocated = int(torch.cuda.max_memory_allocated(device))
        reserved = int(torch.cuda.max_memory_reserved(device))
        baseline_allocated = int(payload["baseline_allocated_bytes"])
        baseline_reserved = int(payload["baseline_reserved_bytes"])
        payload.update(
            {
                "peak_allocated_bytes": allocated,
                "peak_reserved_bytes": reserved,
                "peak_allocated_increment_bytes": max(0, allocated - baseline_allocated),
                "peak_reserved_increment_bytes": max(0, reserved - baseline_reserved),
                "scope": (
                    "absolute allocator peak and increment from the immediately preceding "
                    "training-phase baseline; both models remain resident"
                ),
            }
        )
    elif device.type == "mps" and hasattr(torch.mps, "current_allocated_memory"):
        payload.update(
            {
                "peak_allocated_bytes": None,
                "peak_reserved_bytes": None,
                "peak_allocated_increment_bytes": None,
                "peak_reserved_increment_bytes": None,
                "current_allocated_bytes_after_phase": int(
                    torch.mps.current_allocated_memory()
                ),
                "scope": "MPS exposes current memory but no resettable phase-local peak",
            }
        )
    else:
        payload.update(
            {
                "peak_allocated_bytes": None,
                "peak_reserved_bytes": None,
                "peak_allocated_increment_bytes": None,
                "peak_reserved_increment_bytes": None,
                "scope": "allocator telemetry is unavailable on CPU",
            }
        )
    return payload


def _measure(
    operation: Callable[[], _T],
    *,
    device: Any,
    steps: int,
) -> tuple[tuple[float, ...], _T]:
    durations: list[float] = []
    result: _T | None = None
    for _ in range(steps):
        duration, result = _timed(device, operation)
        durations.append(duration)
    if result is None:  # steps is validated positive
        raise AssertionError("measurement unexpectedly produced no result")
    return tuple(durations), result


def _measure_interleaved(
    first: Callable[[], _T],
    second: Callable[[], _U],
    *,
    device: Any,
    steps: int,
    capture_allocator: bool = False,
) -> tuple[
    tuple[float, ...],
    _T,
    tuple[float, ...],
    _U,
    tuple[Mapping[str, object], ...],
    tuple[Mapping[str, object], ...],
]:
    """Measure paired operations while reversing their order every iteration.

    Sequentially timing every sample from one backend and then every sample
    from the other confounds the backend factor with process drift, thermal
    state, and allocator history.  Each operation is still independently
    synchronized by :func:`_timed`, but the order alternates AB, BA, AB, ... .
    When requested, resettable allocator telemetry is scoped to each individual
    synchronized operation rather than to a mixed pair.
    """

    first_durations: list[float] = []
    second_durations: list[float] = []
    first_allocator: list[Mapping[str, object]] = []
    second_allocator: list[Mapping[str, object]] = []
    first_result: _T | None = None
    second_result: _U | None = None
    for index in range(steps):
        operations: tuple[tuple[str, Callable[[], object]], ...] = (
            (("first", first), ("second", second))
            if index % 2 == 0
            else (("second", second), ("first", first))
        )
        for label, operation in operations:
            allocator_start = _allocator_start(device) if capture_allocator else None
            duration, result = _timed(device, operation)
            allocator_result = (
                _allocator_finish(device, allocator_start)
                if allocator_start is not None
                else None
            )
            if label == "first":
                first_durations.append(duration)
                first_result = result  # type: ignore[assignment]
                if allocator_result is not None:
                    first_allocator.append(allocator_result)
            else:
                second_durations.append(duration)
                second_result = result  # type: ignore[assignment]
                if allocator_result is not None:
                    second_allocator.append(allocator_result)
    if first_result is None or second_result is None:  # steps is validated positive
        raise AssertionError("interleaved measurement unexpectedly produced no result")
    return (
        tuple(first_durations),
        first_result,
        tuple(second_durations),
        second_result,
        tuple(first_allocator),
        tuple(second_allocator),
    )


def _summarize_allocator_measurements(
    measurements: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    """Preserve raw per-iteration allocator scopes and report only maxima."""

    if not measurements:
        raise AssertionError("allocator measurement sequence cannot be empty")
    allocators = {measurement.get("allocator") for measurement in measurements}
    if len(allocators) != 1:
        raise RuntimeError("allocator identity changed during one benchmark phase")

    def maximum(name: str) -> int | None:
        values = [measurement.get(name) for measurement in measurements]
        integers = [int(value) for value in values if isinstance(value, int)]
        return max(integers) if integers else None

    return {
        "allocator": measurements[0].get("allocator"),
        "measurement_count": len(measurements),
        "peak_allocated_bytes": maximum("peak_allocated_bytes"),
        "peak_reserved_bytes": maximum("peak_reserved_bytes"),
        "peak_allocated_increment_bytes": maximum(
            "peak_allocated_increment_bytes"
        ),
        "peak_reserved_increment_bytes": maximum("peak_reserved_increment_bytes"),
        "scope": (
            "one resettable allocator scope per synchronized measured training "
            "iteration; both models remain resident"
        ),
        "per_iteration": [dict(measurement) for measurement in measurements],
    }


def _logical_backend_memory(
    model: Any,
    optimizer: Any,
    output: Any,
    *,
    interaction_tensors: Sequence[Tensor] = (),
) -> dict[str, object]:
    parameters = tuple(model.parameters())
    buffers = tuple(model.buffers())
    gradients = tuple(
        parameter.grad for parameter in parameters if parameter.grad is not None
    )
    optimizer_tensors = tuple(
        value
        for state in optimizer.state.values()
        for value in state.values()
        if isinstance(value, Tensor)
    )
    output_tensors = (
        output.scalar_channels,
        output.vector_channels,
        output.pooled_scalars,
    )

    def unique_storage_bytes(values: Sequence[Tensor]) -> int:
        """Count each underlying storage once, including mixed-precision edges."""

        seen: set[tuple[str, int]] = set()
        total = 0
        for value in values:
            storage = value.untyped_storage()
            key = (str(value.device), int(storage.data_ptr()))
            if key in seen:
                continue
            seen.add(key)
            total += int(storage.nbytes())
        return total

    payload = {
        "parameter_bytes": _logical_tensor_bytes(parameters),
        "buffer_bytes": _logical_tensor_bytes(buffers),
        "gradient_bytes": _logical_tensor_bytes(gradients),
        "optimizer_state_bytes": _logical_tensor_bytes(optimizer_tensors),
        "retained_output_bytes": _logical_tensor_bytes(output_tensors),
        "retained_interaction_tensor_bytes": unique_storage_bytes(
            interaction_tensors
        ),
    }
    return {
        "logical_tensor_payload": payload,
        "known_live_logical_tensor_bytes": sum(payload.values()),
        "logical_accounting_excludes": [
            "autograd-saved activations",
            "temporary kernel workspaces",
            "allocator fragmentation",
            "framework and process memory",
        ],
        "interaction_storage_deduplicated": True,
    }


def _backend_manifest(
    *,
    name: str,
    interaction_kind: str,
    interaction_count: int,
    parameter_count: int,
    initial_sha256: str,
    final_sha256: str,
    phases: Sequence[BackendPhaseMeasurement],
    losses: Sequence[float],
    optimizer_steps: int,
    gradient_summary: Mapping[str, object],
    logical_memory: Mapping[str, object],
    allocator_memory: Mapping[str, object],
) -> dict[str, object]:
    return {
        "backend": name,
        "interaction_kind": interaction_kind,
        "interaction_count_per_iteration": interaction_count,
        "trainable_parameter_count": parameter_count,
        "initial_model_sha256": initial_sha256,
        "final_model_sha256": final_sha256,
        "phases": [phase.to_manifest() for phase in phases],
        "numerics": {
            "final_measured_forward_output_finite": True,
            "final_measured_training_output_finite": True,
            "all_measured_losses_finite": True,
            "measured_losses": list(losses),
            "optimizer_updates_completed": optimizer_steps,
            **dict(gradient_summary),
        },
        "memory": {
            **dict(logical_memory),
            "backend_allocator_training_phase": dict(allocator_memory),
        },
    }


def _phase_median(backend: Mapping[str, object], phase_name: str) -> float:
    phases = backend["phases"]
    if not isinstance(phases, list):
        raise AssertionError("backend phase manifest is malformed")
    for phase in phases:
        if isinstance(phase, Mapping) and phase.get("phase") == phase_name:
            value = phase["median_seconds"]
            if not isinstance(value, (int, float)):
                raise AssertionError("phase median is malformed")
            return float(value)
    raise AssertionError(f"missing phase {phase_name}")


def _paired_ratio_manifest(
    numerator: Sequence[float], denominator: Sequence[float]
) -> dict[str, object]:
    if len(numerator) != len(denominator) or not numerator:
        raise RuntimeError("paired timing sequences must have equal positive length")
    values = tuple(
        float(numerator_value) / float(denominator_value)
        for numerator_value, denominator_value in zip(
            numerator, denominator, strict=True
        )
    )
    if any(not math.isfinite(value) or value <= 0.0 for value in values):
        raise RuntimeError("paired timing ratio is nonfinite or nonpositive")
    ordered = sorted(values)

    def quantile(probability: float) -> float:
        location = probability * (len(ordered) - 1)
        lower = math.floor(location)
        upper = math.ceil(location)
        if lower == upper:
            return ordered[lower]
        weight = location - lower
        return (1.0 - weight) * ordered[lower] + weight * ordered[upper]

    return {
        "unit": "dimensionless",
        "iterations": len(values),
        "values": list(values),
        "median": statistics.median(values),
        "quartile_25": quantile(0.25),
        "quartile_75": quantile(0.75),
        "paired_by_measurement_iteration": True,
    }


def _run_comparison(
    policy: BackendComparisonConfig,
    *,
    device: Any,
    master_dtype: Any,
    autocast_dtype: Any | None,
    effective_threads: int,
) -> BackendComparisonReport:
    workload = _make_constant_density_workload(
        policy, device=device, dtype=master_dtype
    )
    invariant_config = InvariantPeriodicTransformerConfig(
        input_scalar_dim=policy.input_scalar_dim,
        model_dim=policy.scalar_dim,
        pair_dim=policy.pair_dim,
        num_heads=policy.num_heads,
        num_layers=policy.num_layers,
        feedforward_multiplier=policy.invariant_feedforward_multiplier,
        reciprocal_mode_radius=policy.reciprocal_mode_radius,
        direct_harmonics=policy.direct_harmonics,
        output_vector_dim=policy.vector_dim,
        reference_length=1.0,
        max_pair_interactions=policy.max_dense_pair_interactions,
    )
    local_config = PeriodicTransformerConfig(
        input_scalar_dim=policy.input_scalar_dim,
        scalar_dim=policy.scalar_dim,
        vector_dim=policy.vector_dim,
        num_heads=policy.num_heads,
        num_layers=policy.num_layers,
        radial_basis_dim=policy.radial_basis_dim,
        cutoff=policy.cutoff,
        reference_length=1.0,
        max_candidates_per_pair=policy.max_candidates_per_pair,
        max_edges=policy.max_edges_per_structure,
    )
    invariant_model = InvariantPeriodicTransformer(invariant_config).to(
        device=device, dtype=master_dtype
    )
    local_model = PeriodicSE3Transformer(local_config).to(
        device=device, dtype=master_dtype
    )
    _initialize_model(invariant_model, policy.seed)
    _initialize_model(local_model, policy.seed)
    initial_hashes = {
        _INVARIANT: _model_sha256(invariant_model),
        _LOCAL: _model_sha256(local_model),
    }

    batch = policy.batch_size
    atoms = policy.atoms_per_structure
    flat_features = workload.features.reshape(batch * atoms, policy.input_scalar_dim)
    flat_fractional = workload.fractional.reshape(batch * atoms, 3)
    coordinate_parts = tuple(workload.fractional[index] for index in range(batch))
    lattice_parts = tuple(workload.lattices[index] for index in range(batch))

    def build_local_edges() -> PeriodicEdges:
        return enumerate_packed_periodic_edges(
            coordinate_parts,
            lattice_parts,
            policy.cutoff,
            max_candidates_per_pair=policy.max_candidates_per_pair,
            max_edges=policy.max_edges_per_structure,
            max_total_edges=policy.max_total_edges,
        )

    reference_edges = build_local_edges()
    if reference_edges.num_edges < 1:
        raise DomainError("constant-density workload unexpectedly has no local edges")
    activation_reference_edges = _activation_edges(reference_edges, autocast_dtype)
    dense_interactions = batch * atoms * atoms
    local_interactions = reference_edges.num_edges
    total_atoms = batch * atoms

    def invariant_forward() -> Any:
        with (
            torch.no_grad(),
            torch.autocast(
                device_type=device.type,
                dtype=autocast_dtype,
                enabled=autocast_dtype is not None,
            ),
        ):
            return invariant_model(
                workload.features, workload.fractional, workload.lattices
            )

    def local_forward_core() -> Any:
        with (
            torch.no_grad(),
            torch.autocast(
                device_type=device.type,
                dtype=autocast_dtype,
                enabled=autocast_dtype is not None,
            ),
        ):
            return local_model(
                flat_features,
                flat_fractional,
                workload.lattices[0],
                edges=activation_reference_edges,
                batch_index=workload.graph_index,
                num_graphs=batch,
            )

    def local_forward_end_to_end() -> tuple[Any, PeriodicEdges]:
        exact_edges = build_local_edges()
        activation_edges = _activation_edges(exact_edges, autocast_dtype)
        with (
            torch.no_grad(),
            torch.autocast(
                device_type=device.type,
                dtype=autocast_dtype,
                enabled=autocast_dtype is not None,
            ),
        ):
            output = local_model(
                flat_features,
                flat_fractional,
                workload.lattices[0],
                edges=activation_edges,
                batch_index=workload.graph_index,
                num_graphs=batch,
            )
        return output, exact_edges

    invariant_model.eval()
    local_model.eval()
    for index in range(policy.warmup_steps):
        if index % 2 == 0:
            invariant_forward()
            local_forward_end_to_end()
        else:
            local_forward_end_to_end()
            invariant_forward()
        if policy.measure_local_prebuilt_core:
            local_forward_core()

    (
        invariant_forward_times,
        invariant_forward_output,
        local_forward_times,
        local_forward_result,
        _,
        _,
    ) = _measure_interleaved(
        invariant_forward,
        local_forward_end_to_end,
        device=device,
        steps=policy.measurement_steps,
    )
    local_forward_output, last_edges = local_forward_result
    if last_edges.num_edges != local_interactions:
        raise RuntimeError("fixed workload changed its local interaction count")
    local_core_times: tuple[float, ...] | None = None
    if policy.measure_local_prebuilt_core:
        local_core_times, local_core_output = _measure(
            local_forward_core, device=device, steps=policy.measurement_steps
        )
        if not _finite_output(local_core_output):
            raise RuntimeError("local prebuilt-core forward produced a nonfinite output")
    if not _finite_output(invariant_forward_output):
        raise RuntimeError("invariant forward produced a nonfinite output")
    if not _finite_output(local_forward_output):
        raise RuntimeError("local end-to-end forward produced a nonfinite output")

    invariant_model.train()
    local_model.train()
    invariant_optimizer = torch.optim.AdamW(
        invariant_model.parameters(), lr=policy.learning_rate, foreach=False, fused=False
    )
    local_optimizer = torch.optim.AdamW(
        local_model.parameters(), lr=policy.learning_rate, foreach=False, fused=False
    )
    invariant_scaler = (
        torch.amp.GradScaler("cuda", enabled=True)
        if autocast_dtype is torch.float16
        else None
    )
    local_scaler = (
        torch.amp.GradScaler("cuda", enabled=True)
        if autocast_dtype is torch.float16
        else None
    )

    def invariant_train_step() -> tuple[Tensor, Any]:
        invariant_optimizer.zero_grad(set_to_none=True)
        with torch.autocast(
            device_type=device.type,
            dtype=autocast_dtype,
            enabled=autocast_dtype is not None,
        ):
            output = invariant_model(
                workload.features, workload.fractional, workload.lattices
            )
            loss = torch.nn.functional.mse_loss(
                output.scalar_channels,
                workload.scalar_target.to(dtype=output.scalar_channels.dtype),
            ) + torch.nn.functional.mse_loss(
                output.vector_channels,
                workload.vector_target.to(dtype=output.vector_channels.dtype),
            )
        if invariant_scaler is None:
            loss.backward()
            invariant_optimizer.step()
        else:
            invariant_scaler.scale(loss).backward()
            invariant_scaler.step(invariant_optimizer)
            invariant_scaler.update()
        # Keep device-to-host transfer and scalar materialization outside the
        # timed operation.  _timed supplies the one declared trailing device
        # synchronization for the phase.
        return loss.detach(), output

    def local_train_step() -> tuple[Tensor, Any, PeriodicEdges]:
        local_optimizer.zero_grad(set_to_none=True)
        exact_edges = build_local_edges()
        activation_edges = _activation_edges(exact_edges, autocast_dtype)
        with torch.autocast(
            device_type=device.type,
            dtype=autocast_dtype,
            enabled=autocast_dtype is not None,
        ):
            output = local_model(
                flat_features,
                flat_fractional,
                workload.lattices[0],
                edges=activation_edges,
                batch_index=workload.graph_index,
                num_graphs=batch,
            )
            loss = torch.nn.functional.mse_loss(
                output.scalar_channels,
                workload.scalar_target.reshape(
                    total_atoms, policy.scalar_dim
                ).to(dtype=output.scalar_channels.dtype),
            ) + torch.nn.functional.mse_loss(
                output.vector_channels,
                workload.vector_target.reshape(
                    total_atoms, policy.vector_dim, 3
                ).to(dtype=output.vector_channels.dtype),
            )
        if local_scaler is None:
            loss.backward()
            local_optimizer.step()
        else:
            local_scaler.scale(loss).backward()
            local_scaler.step(local_optimizer)
            local_scaler.update()
        return loss.detach(), output, exact_edges

    # Alternate warmups so both optimizers allocate their states before either
    # measured allocator scope starts.
    for index in range(policy.warmup_steps):
        if index % 2 == 0:
            invariant_loss, invariant_training_output = invariant_train_step()
            local_loss, local_training_output, training_edges = local_train_step()
        else:
            local_loss, local_training_output, training_edges = local_train_step()
            invariant_loss, invariant_training_output = invariant_train_step()
        if not bool(torch.isfinite(invariant_loss).item()) or not bool(
            torch.isfinite(local_loss).item()
        ):
            raise RuntimeError("training warmup produced a nonfinite loss")

    invariant_loss_tensors: list[Tensor] = []

    def measured_invariant_train_step() -> tuple[Tensor, Any]:
        result = invariant_train_step()
        invariant_loss_tensors.append(result[0])
        return result

    local_loss_tensors: list[Tensor] = []

    def measured_local_train_step() -> tuple[Tensor, Any, PeriodicEdges]:
        result = local_train_step()
        local_loss_tensors.append(result[0])
        return result

    (
        invariant_training_times,
        invariant_last,
        local_training_times,
        local_last,
        invariant_allocator_measurements,
        local_allocator_measurements,
    ) = _measure_interleaved(
        measured_invariant_train_step,
        measured_local_train_step,
        device=device,
        steps=policy.measurement_steps,
        capture_allocator=True,
    )
    invariant_allocator = _summarize_allocator_measurements(
        invariant_allocator_measurements
    )
    local_allocator = _summarize_allocator_measurements(
        local_allocator_measurements
    )
    local_training_output = local_last[1]
    training_edges = local_last[2]
    invariant_training_output = invariant_last[1]
    if not _finite_output(invariant_training_output) or not _finite_output(
        local_training_output
    ):
        raise RuntimeError("a measured training step produced a nonfinite output")

    invariant_losses = [
        float(value.to(torch.float64).cpu().item())
        for value in invariant_loss_tensors
    ]
    local_losses = [
        float(value.to(torch.float64).cpu().item()) for value in local_loss_tensors
    ]

    expected_steps = policy.warmup_steps + policy.measurement_steps
    invariant_steps = _optimizer_step_count(invariant_optimizer)
    local_steps = _optimizer_step_count(local_optimizer)
    if invariant_steps != expected_steps or local_steps != expected_steps:
        raise RuntimeError("AdamW did not commit every warmup and measured update")
    if not all(math.isfinite(value) for value in (*invariant_losses, *local_losses)):
        raise RuntimeError("training measurement produced a nonfinite loss")
    if training_edges.num_edges != local_interactions:
        raise RuntimeError("training changed the fixed local interaction count")
    invariant_gradients = _final_measured_gradient_summary(invariant_model)
    local_gradients = _final_measured_gradient_summary(local_model)

    invariant_phases = (
        BackendPhaseMeasurement(
            phase="forward_end_to_end",
            includes=(
                "dense periodic geometry construction",
                "invariant-scalar Transformer and covariant readout forward",
                "device synchronization",
            ),
            durations_seconds=invariant_forward_times,
            structures_per_iteration=batch,
            atoms_per_iteration=total_atoms,
            interaction_kind="dense_ordered_atom_pairs",
            interactions_per_iteration=dense_interactions,
        ),
        BackendPhaseMeasurement(
            phase="training_step_end_to_end",
            includes=(
                "optimizer zero_grad",
                "dense periodic geometry construction",
                "invariant-scalar Transformer and covariant readout forward",
                "loss",
                "backward",
                "AdamW update",
                "device synchronization",
            ),
            durations_seconds=invariant_training_times,
            structures_per_iteration=batch,
            atoms_per_iteration=total_atoms,
            interaction_kind="dense_ordered_atom_pairs",
            interactions_per_iteration=dense_interactions,
        ),
    )
    local_phase_list = [
        BackendPhaseMeasurement(
            phase="forward_end_to_end",
            includes=(
                "exact periodic-image edge enumeration",
                "local proper-SE(3) Transformer forward",
                "device synchronization",
            ),
            durations_seconds=local_forward_times,
            structures_per_iteration=batch,
            atoms_per_iteration=total_atoms,
            interaction_kind="exact_periodic_image_edges",
            interactions_per_iteration=local_interactions,
        )
    ]
    if local_core_times is not None:
        local_phase_list.append(
            BackendPhaseMeasurement(
                phase="forward_prebuilt_interactions_core",
                includes=(
                    "local proper-SE(3) Transformer forward with prebuilt exact edges",
                    "device synchronization",
                ),
                durations_seconds=local_core_times,
                structures_per_iteration=batch,
                atoms_per_iteration=total_atoms,
                interaction_kind="exact_periodic_image_edges",
                interactions_per_iteration=local_interactions,
            )
        )
    local_phase_list.append(
        BackendPhaseMeasurement(
            phase="training_step_end_to_end",
            includes=(
                "optimizer zero_grad",
                "exact periodic-image edge enumeration",
                "local proper-SE(3) Transformer forward",
                "loss",
                "backward",
                "AdamW update",
                "device synchronization",
            ),
            durations_seconds=local_training_times,
            structures_per_iteration=batch,
            atoms_per_iteration=total_atoms,
            interaction_kind="exact_periodic_image_edges",
            interactions_per_iteration=local_interactions,
        )
    )

    invariant_logical = _logical_backend_memory(
        invariant_model, invariant_optimizer, invariant_training_output
    )
    local_logical = _logical_backend_memory(
        local_model,
        local_optimizer,
        local_training_output,
        interaction_tensors=(
            *_edge_tensors(training_edges),
            *_edge_tensors(local_training_output.edges),
        ),
    )
    backends = {
        _INVARIANT: _backend_manifest(
            name=_INVARIANT,
            interaction_kind="dense_ordered_atom_pairs",
            interaction_count=dense_interactions,
            parameter_count=_parameter_count(invariant_model),
            initial_sha256=initial_hashes[_INVARIANT],
            final_sha256=_model_sha256(invariant_model),
            phases=invariant_phases,
            losses=invariant_losses,
            optimizer_steps=invariant_steps,
            gradient_summary=invariant_gradients,
            logical_memory=invariant_logical,
            allocator_memory=invariant_allocator,
        ),
        _LOCAL: _backend_manifest(
            name=_LOCAL,
            interaction_kind="exact_periodic_image_edges",
            interaction_count=local_interactions,
            parameter_count=_parameter_count(local_model),
            initial_sha256=initial_hashes[_LOCAL],
            final_sha256=_model_sha256(local_model),
            phases=tuple(local_phase_list),
            losses=local_losses,
            optimizer_steps=local_steps,
            gradient_summary=local_gradients,
            logical_memory=local_logical,
            allocator_memory=local_allocator,
        ),
    }
    invariant_parameters = _parameter_count(invariant_model)
    local_parameters = _parameter_count(local_model)
    invariant_forward_median = _phase_median(backends[_INVARIANT], "forward_end_to_end")
    local_forward_median = _phase_median(backends[_LOCAL], "forward_end_to_end")
    invariant_training_median = _phase_median(
        backends[_INVARIANT], "training_step_end_to_end"
    )
    local_training_median = _phase_median(
        backends[_LOCAL], "training_step_end_to_end"
    )
    ratios = {
        "local_over_invariant_forward_median_time": (
            local_forward_median / invariant_forward_median
        ),
        "local_over_invariant_training_median_time": (
            local_training_median / invariant_training_median
        ),
        "paired_local_over_invariant_forward_time": _paired_ratio_manifest(
            local_forward_times, invariant_forward_times
        ),
        "paired_local_over_invariant_training_time": _paired_ratio_manifest(
            local_training_times, invariant_training_times
        ),
        "invariant_over_local_trainable_parameters": (
            invariant_parameters / local_parameters
        ),
        "ratios_are_descriptive_not_significance_tests": True,
    }
    shared_tensor_mapping = {
        "features": workload.features,
        "fractional": workload.fractional,
        "lattices": workload.lattices,
        "graph_index": workload.graph_index,
        "scalar_target": workload.scalar_target,
        "vector_target": workload.vector_target,
    }
    shared_tensors = tuple(shared_tensor_mapping.values())
    tensor_manifests = {
        name: _tensor_content_manifest(value)
        for name, value in shared_tensor_mapping.items()
    }
    reference_edge_mapping = dict(
        zip(
            (
                "destination",
                "source",
                "shifts",
                "displacement",
                "squared_distance",
            ),
            _edge_tensors(reference_edges),
            strict=True,
        )
    )
    reference_edge_manifests = {
        name: _tensor_content_manifest(value)
        for name, value in reference_edge_mapping.items()
    }
    cell_volumes = torch.linalg.det(workload.lattices.to(torch.float64)).detach().cpu()
    expected_cell_volume = atoms * float(policy.volume_per_atom)
    relative_volume_errors = [
        abs(float(value.item()) - expected_cell_volume) / expected_cell_volume
        for value in cell_volumes
    ]
    density_tolerance = 1.0e-10 if master_dtype == torch.float64 else 5.0e-6
    if any(error > density_tolerance for error in relative_volume_errors):
        raise RuntimeError(
            "represented lattice determinants violate the constant-density contract"
        )
    actual_number_densities = [
        atoms / float(value.item()) for value in cell_volumes
    ]
    measurement_order = [
        (
            f"{_INVARIANT}_then_{_LOCAL}"
            if index % 2 == 0
            else f"{_LOCAL}_then_{_INVARIANT}"
        )
        for index in range(policy.measurement_steps)
    ]
    workload_manifest = {
        "generator": "constructive-constant-density-triclinic.v2",
        "parameter_initializer": DYADIC_SIGNAL_INITIALIZER,
        "synthetic_signal_initializer": DYADIC_SIGNAL_INITIALIZER,
        "seed": policy.seed,
        "structures_per_iteration": batch,
        "atoms_per_structure": atoms,
        "atoms_per_iteration": total_atoms,
        "volume_per_atom": float(policy.volume_per_atom),
        "length_unit": "angstrom",
        "volume_unit": "angstrom^3",
        "cell_volumes": [float(value.item()) for value in cell_volumes],
        "expected_cell_volume": expected_cell_volume,
        "relative_cell_volume_errors": relative_volume_errors,
        "constant_density_relative_tolerance": density_tolerance,
        "number_density_atoms_per_volume": 1.0 / float(policy.volume_per_atom),
        "actual_number_density_atoms_per_volume": actual_number_densities,
        "density_derived_from_represented_lattice_determinants": True,
        "batch_geometries_are_fractional_translations": True,
        "batch_geometries_are_independent_structural_samples": False,
        "batch_relationship": (
            "each later structure is a global fractional translation of the first "
            "under periodic boundary conditions"
        ),
        "shared_input_and_target_logical_bytes": _logical_tensor_bytes(shared_tensors),
        "tensor_manifests": tensor_manifests,
        "tensor_sha256": {
            name: manifest["sha256"] for name, manifest in tensor_manifests.items()
        },
        "local_reference_interaction_tensor_manifests": reference_edge_manifests,
        "local_reference_interaction_sha256": {
            name: manifest["sha256"]
            for name, manifest in reference_edge_manifests.items()
        },
        "phase_execution_order": [
            "both_backend_forward_warmups_alternating_AB_BA_starting_AB",
            "paired_forward_measurements_alternating_AB_BA_starting_AB",
            *(
                ["equivariant_local_forward_prebuilt_interactions_core"]
                if policy.measure_local_prebuilt_core
                else []
            ),
            "both_backend_training_warmups_alternating_AB_BA_starting_AB",
            "paired_training_measurements_alternating_AB_BA_starting_AB",
        ],
        "paired_forward_backend_order_by_iteration": measurement_order,
        "paired_training_backend_order_by_iteration": measurement_order,
        "optimizer": {
            "class": "torch.optim.AdamW",
            "learning_rate": float(policy.learning_rate),
            "foreach": False,
            "fused": False,
        },
        "master_dtype": str(master_dtype),
        "autocast_dtype": None if autocast_dtype is None else str(autocast_dtype),
        "precision_contract": {
            "comparison_kind": (
                "common-master-precision"
                if autocast_dtype is None
                else "backend-native-mixed-precision"
            ),
            "backend_arithmetic_precision_matched": autocast_dtype is None,
            "invariant_deterministic_geometry_dtype": str(master_dtype),
            "invariant_learned_activation_dtype": (
                str(master_dtype)
                if autocast_dtype is None
                else str(autocast_dtype)
            ),
            "local_edge_topology_arithmetic_dtype": "torch.float64",
            "local_represented_edge_dtype": str(master_dtype),
            "local_geometry_activation_dtype": (
                str(master_dtype)
                if autocast_dtype is None
                else str(autocast_dtype)
            ),
        },
        "local_edge_membership_precision": (
            "float64 topology arithmetic after conversion from the represented "
            f"{master_dtype} inputs"
        ),
        "matching_contract": {
            "same_inputs_targets_density_scalar_width_vector_width_heads_layers": True,
            "parameter_matched": invariant_parameters == local_parameters,
            "compute_matched": False,
            "weights_matched": False,
            "reason": (
                "pair projections, Fourier geometry, radial geometry, and vector-state "
                "parameterizations differ by architecture"
            ),
        },
    }
    workload_manifest["fingerprint"] = _fingerprint(workload_manifest)
    runtime_manifest = _runtime_manifest(device, cpu_threads=effective_threads)
    runtime_manifest["implementation"] = _implementation_manifest(
        invariant_config, local_config
    )
    runtime_manifest["fingerprint"] = _fingerprint(runtime_manifest)
    return BackendComparisonReport(
        config=policy,
        runtime=runtime_manifest,
        workload=workload_manifest,
        backends=backends,
        descriptive_ratios=ratios,
    )


def run_single_device_backend_comparison(
    config: BackendComparisonConfig | None = None,
) -> BackendComparisonReport:
    """Run the pinned comparison and restore RNG and backend-global state."""

    policy = BackendComparisonConfig() if config is None else config
    stage = "device resolution"
    previous_threads = torch.get_num_threads()
    previous_deterministic = torch.are_deterministic_algorithms_enabled()
    previous_warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    previous_matmul_precision = torch.get_float32_matmul_precision()
    previous_cuda_tf32 = torch.backends.cuda.matmul.allow_tf32
    previous_cudnn_tf32 = torch.backends.cudnn.allow_tf32
    previous_cudnn_benchmark = torch.backends.cudnn.benchmark
    previous_cudnn_deterministic = torch.backends.cudnn.deterministic
    previous_mps_rng: Tensor | None = None
    try:
        device = _resolve_device(policy.device)
        master_dtype, autocast_dtype = _precision_policy(policy, device)
        effective_threads = policy.cpu_threads or previous_threads
        if policy.cpu_threads is not None:
            torch.set_num_threads(policy.cpu_threads)
        torch.use_deterministic_algorithms(True, warn_only=False)
        torch.set_float32_matmul_precision("highest")
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        if device.type == "mps" and hasattr(torch.mps, "get_rng_state"):
            previous_mps_rng = torch.mps.get_rng_state().clone()
        cuda_devices = [device.index or 0] if device.type == "cuda" else []
        stage = "comparison execution"
        with torch.random.fork_rng(devices=cuda_devices, enabled=True):
            return _run_comparison(
                policy,
                device=device,
                master_dtype=master_dtype,
                autocast_dtype=autocast_dtype,
                effective_threads=effective_threads,
            )
    except DomainError:
        raise
    except Exception as exc:
        raise BackendComparisonExecutionError(stage, exc) from exc
    finally:
        if previous_mps_rng is not None:
            torch.mps.set_rng_state(previous_mps_rng)
        if policy.cpu_threads is not None and torch.get_num_threads() != previous_threads:
            torch.set_num_threads(previous_threads)
        torch.use_deterministic_algorithms(
            previous_deterministic, warn_only=previous_warn_only
        )
        torch.set_float32_matmul_precision(previous_matmul_precision)
        torch.backends.cuda.matmul.allow_tf32 = previous_cuda_tf32
        torch.backends.cudnn.allow_tf32 = previous_cudnn_tf32
        torch.backends.cudnn.benchmark = previous_cudnn_benchmark
        torch.backends.cudnn.deterministic = previous_cudnn_deterministic


__all__ = [
    "BACKEND_COMPARISON_METHOD",
    "BACKEND_COMPARISON_SCHEMA",
    "BackendComparisonConfig",
    "BackendComparisonExecutionError",
    "BackendComparisonReport",
    "BackendPhaseMeasurement",
    "run_single_device_backend_comparison",
]
