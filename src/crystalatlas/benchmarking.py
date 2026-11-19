"""Reproducible single-device microbenchmark for the local geometry ablation.

This module deliberately reports measurements rather than performance claims.
The benchmark fixes a synthetic periodic workload, deterministic parameter
initialization, phase order, synchronization points, optimizer, and timing
clock.  It can run on CPU, so validating the reporting contract never requires
CUDA.  A CPU result is not evidence about GPU memory or multi-GPU scaling.
"""

from __future__ import annotations

import json
import math
import os
import platform
import statistics
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from hashlib import sha256
from typing import Any, TypeVar, cast

from ._optional import require_torch
from .errors import CrystalAtlasError, DomainError
from .nn.equivariant_transformer import (
    PeriodicEdges,
    PeriodicSE3Transformer,
    PeriodicTransformerConfig,
    enumerate_packed_periodic_edges,
)

torch = require_torch()
Tensor = torch.Tensor

_T = TypeVar("_T")

BENCHMARK_SCHEMA = "crystalatlas.single-device-geometry-benchmark.v1"
BENCHMARK_METHOD = "crystalatlas.periodic-transformer-microbenchmark.v2"
BENCHMARK_FACTOR = "geometry"
DYADIC_SIGNAL_INITIALIZER = "integer-affine-modulo-2^24-dyadic-no-rng.v1"
_HEX_DIGITS = frozenset("0123456789abcdef")
_DETERMINISTIC_CUBLAS_WORKSPACE_CONFIGS = frozenset({":16:8", ":4096:8"})
_MODULAR_SIGNAL_BITS = 24
_MODULAR_SIGNAL_MODULUS = 1 << _MODULAR_SIGNAL_BITS
_MODULAR_SIGNAL_MASK = _MODULAR_SIGNAL_MODULUS - 1
_MODULAR_SIGNAL_CENTER = _MODULAR_SIGNAL_MODULUS // 2
_FLOAT32_EXACT_DYADIC_EXPONENT_MIN = -126
_FLOAT32_EXACT_DYADIC_EXPONENT_MAX = 104


def _strict_json_value(value: object, *, path: str = "$") -> object:
    """Normalize only RFC-8259 data, rejecting Python's JSON extensions."""

    if value is None or type(value) in {bool, str, int}:
        return value
    if type(value) is float:
        if not math.isfinite(value):
            raise DomainError(f"{path} contains a nonfinite float")
        return value
    if isinstance(value, Mapping):
        normalized: dict[str, object] = {}
        for key, item in value.items():
            if type(key) is not str:
                raise DomainError(f"{path} contains a non-string object key")
            normalized[key] = _strict_json_value(item, path=f"{path}.{key}")
        return normalized
    if isinstance(value, (list, tuple)):
        return [
            _strict_json_value(item, path=f"{path}[{index}]")
            for index, item in enumerate(value)
        ]
    raise DomainError(f"{path} contains a non-JSON value of type {type(value).__name__}")


def _canonical_json(payload: Mapping[str, object]) -> str:
    normalized = _strict_json_value(payload)
    return json.dumps(
        normalized,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    )


def _fingerprint(payload: Mapping[str, object]) -> str:
    return sha256(_canonical_json(payload).encode("utf-8")).hexdigest()


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


def _dyadic_modular_signal(
    length: int,
    *,
    multiplier: int,
    offset: int,
    scale_exponent: int,
) -> Tensor:
    """Construct a platform-stable, RNG-free floating-point signal.

    Odd affine multiplication modulo ``2**24`` permutes the integer residues.
    Every residue converts exactly to Float64, and the accepted scale range
    keeps every nonzero result normal, finite, and exactly representable in
    Float32 as well.
    Unlike ``sin``/``cos`` fixture construction, the resulting bytes do not
    depend on vector-libm dispatch, allocation alignment, or CPU thread
    partitioning.
    """

    _positive_integer("signal length", length)
    if (
        isinstance(multiplier, bool)
        or not isinstance(multiplier, int)
        or multiplier <= 0
        or multiplier > _MODULAR_SIGNAL_MASK
        or multiplier % 2 == 0
    ):
        raise DomainError(
            "signal multiplier must be a positive odd integer below 2**24"
        )
    if isinstance(offset, bool) or not isinstance(offset, int):
        raise DomainError("signal offset must be an integer")
    if isinstance(scale_exponent, bool) or not isinstance(scale_exponent, int):
        raise DomainError("signal scale_exponent must be an integer")
    maximum_int64 = torch.iinfo(torch.int64).max
    if not (
        _FLOAT32_EXACT_DYADIC_EXPONENT_MIN
        <= scale_exponent
        <= _FLOAT32_EXACT_DYADIC_EXPONENT_MAX
    ):
        raise DomainError(
            "signal scale_exponent must preserve normal finite exact Float32 values"
        )
    if length > maximum_int64:
        raise DomainError("signal length exceeds the Int64 arange domain")
    maximum_index = length - 1
    normalized_offset = offset & _MODULAR_SIGNAL_MASK
    if maximum_index > (maximum_int64 - normalized_offset) // multiplier:
        raise DomainError("signal length would overflow exact Int64 construction")

    index = torch.arange(length, dtype=torch.int64, device="cpu")
    residue = torch.bitwise_and(
        index * multiplier + normalized_offset,
        _MODULAR_SIGNAL_MASK,
    )
    centered = residue.to(torch.float64) - float(_MODULAR_SIGNAL_CENTER)
    return centered * math.ldexp(1.0, scale_exponent)


@dataclass(frozen=True)
class GeometryBenchmarkConfig:
    """Pinned local proper-SE(3) workload and policy for one device.

    ``precision`` follows the training contract: Float32 and Float64 are master
    dtypes, BF16 is autocast on CPU/CUDA, and FP16 is autocast on CUDA only.
    The default dimensions are intentionally a microbenchmark, not a claim to
    represent the eventual production checkpoint.
    """

    device: str = "auto"
    precision: str = "float32"
    batch_size: int = 2
    atoms_per_structure: int = 8
    input_scalar_dim: int = 16
    scalar_dim: int = 48
    vector_dim: int = 12
    num_heads: int = 3
    num_layers: int = 2
    radial_basis_dim: int = 8
    cutoff: float = 3.0
    cell_length: float = 6.0
    max_candidates_per_pair: int = 32_768
    max_edges: int = 500_000
    warmup_steps: int = 2
    measurement_steps: int = 5
    seed: int = 1729
    learning_rate: float = 1.0e-3
    cpu_threads: int | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.device, str) or not self.device.strip():
            raise DomainError("benchmark device must be a nonempty string")
        if self.precision not in {"float32", "float64", "bfloat16", "float16"}:
            raise DomainError("benchmark precision is not supported")
        integer_fields = {
            "batch_size": self.batch_size,
            "atoms_per_structure": self.atoms_per_structure,
            "input_scalar_dim": self.input_scalar_dim,
            "scalar_dim": self.scalar_dim,
            "vector_dim": self.vector_dim,
            "num_heads": self.num_heads,
            "num_layers": self.num_layers,
            "radial_basis_dim": self.radial_basis_dim,
            "max_candidates_per_pair": self.max_candidates_per_pair,
            "max_edges": self.max_edges,
            "warmup_steps": self.warmup_steps,
            "measurement_steps": self.measurement_steps,
        }
        for name, value in integer_fields.items():
            _positive_integer(name, value)
        if self.atoms_per_structure < 2:
            raise DomainError("atoms_per_structure must be at least two")
        if self.scalar_dim % self.num_heads or self.vector_dim % self.num_heads:
            raise DomainError("scalar_dim and vector_dim must be divisible by num_heads")
        if isinstance(self.seed, bool) or not isinstance(self.seed, int) or self.seed < 0:
            raise DomainError("benchmark seed must be a nonnegative integer")
        _positive_finite("cutoff", self.cutoff)
        _positive_finite("cell_length", self.cell_length)
        _positive_finite("learning_rate", self.learning_rate)
        if self.cpu_threads is not None:
            _positive_integer("cpu_threads", self.cpu_threads)

    def to_dict(self) -> dict[str, object]:
        return {
            "factor": BENCHMARK_FACTOR,
            "device": self.device,
            "precision": self.precision,
            "batch_size": self.batch_size,
            "atoms_per_structure": self.atoms_per_structure,
            "input_scalar_dim": self.input_scalar_dim,
            "scalar_dim": self.scalar_dim,
            "vector_dim": self.vector_dim,
            "num_heads": self.num_heads,
            "num_layers": self.num_layers,
            "radial_basis_dim": self.radial_basis_dim,
            "cutoff": float(self.cutoff),
            "cell_length": float(self.cell_length),
            "max_candidates_per_pair": self.max_candidates_per_pair,
            "max_edges": self.max_edges,
            "warmup_steps_per_phase": self.warmup_steps,
            "measurement_steps_per_phase": self.measurement_steps,
            "seed": self.seed,
            "learning_rate": float(self.learning_rate),
            "cpu_threads": self.cpu_threads,
        }

    @property
    def fingerprint(self) -> str:
        return _fingerprint(self.to_dict())

    def to_manifest(self) -> dict[str, object]:
        payload = self.to_dict()
        payload["fingerprint"] = self.fingerprint
        return payload


@dataclass(frozen=True)
class PhaseMeasurement:
    """Synchronized wall-clock samples for one pinned phase."""

    phase: str
    includes: tuple[str, ...]
    durations_seconds: tuple[float, ...]
    structures_per_iteration: int
    atoms_per_iteration: int
    edges_per_iteration: int

    def __post_init__(self) -> None:
        if not self.phase or not self.includes:
            raise DomainError("a benchmark phase must be named and scoped")
        if not self.durations_seconds:
            raise DomainError("a benchmark phase requires timing samples")
        if any(not math.isfinite(value) or value <= 0.0 for value in self.durations_seconds):
            raise DomainError("benchmark durations must be finite and positive")
        for name, value in (
            ("structures_per_iteration", self.structures_per_iteration),
            ("atoms_per_iteration", self.atoms_per_iteration),
            ("edges_per_iteration", self.edges_per_iteration),
        ):
            _positive_integer(name, value)

    @property
    def total_seconds(self) -> float:
        return math.fsum(self.durations_seconds)

    def to_manifest(self) -> dict[str, object]:
        total = self.total_seconds
        iterations = len(self.durations_seconds)
        return {
            "phase": self.phase,
            "includes": list(self.includes),
            "iterations": iterations,
            "durations_seconds": list(self.durations_seconds),
            "total_seconds": total,
            "mean_seconds": total / iterations,
            "median_seconds": statistics.median(self.durations_seconds),
            "minimum_seconds": min(self.durations_seconds),
            "maximum_seconds": max(self.durations_seconds),
            "structures_per_second": (self.structures_per_iteration * iterations / total),
            "atoms_per_second": self.atoms_per_iteration * iterations / total,
            "edges_per_second": self.edges_per_iteration * iterations / total,
        }


@dataclass(frozen=True)
class GeometryBenchmarkReport:
    """Canonical, content-addressed evidence from a successful benchmark."""

    config: GeometryBenchmarkConfig
    runtime: Mapping[str, object]
    workload: Mapping[str, object]
    phases: tuple[PhaseMeasurement, ...]
    memory: Mapping[str, object]
    numerics: Mapping[str, object]

    def __post_init__(self) -> None:
        if tuple(phase.phase for phase in self.phases) != (
            "periodic_edge_enumeration",
            "transformer_forward_precomputed_edges",
            "training_step_end_to_end",
        ):
            raise DomainError("benchmark phases do not match the pinned method")
        # This also rejects NaN/Infinity and non-JSON extension values.
        _canonical_json(self._payload())

    def _payload(self) -> dict[str, object]:
        return {
            "schema": BENCHMARK_SCHEMA,
            "status": "completed",
            "method": BENCHMARK_METHOD,
            "factor": BENCHMARK_FACTOR,
            "config": self.config.to_manifest(),
            "runtime": dict(self.runtime),
            "workload": dict(self.workload),
            "phases": [phase.to_manifest() for phase in self.phases],
            "memory": dict(self.memory),
            "numerics": dict(self.numerics),
            "interpretation": {
                "single_device_projection": (
                    "A matched workload may project single-device wall time as "
                    "target_iterations * measured_mean_seconds."
                ),
                "multi_device_scaling_measured": False,
                "multi_device_extrapolation_permitted": False,
                "reason": (
                    "One device does not measure DDP communication, input-pipeline "
                    "contention, interconnect bandwidth, imbalance, or scaling efficiency."
                ),
                "predictive_quality_measured": False,
                "full_crystalatlas_training_objective_measured": False,
                "measured_scope": (
                    "legacy local periodic proper-SE(3) Transformer microbenchmark "
                    "only; excludes the invariant_global candidate, "
                    "composition/cover losses, exact cover expansion/reduction, and CFM"
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


class BenchmarkExecutionError(CrystalAtlasError):
    """A benchmark failed before a successful report could be certified."""

    def __init__(self, stage: str, cause: BaseException) -> None:
        self.stage = stage
        self.cause_type = type(cause).__name__
        super().__init__(
            f"geometry benchmark failed during {stage}: {self.cause_type}: {cause}"
        )


def _resolve_device(requested: str) -> Any:
    normalized = requested.strip().lower()
    if normalized == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda", 0)
        if (
            hasattr(torch.backends, "mps")
            and torch.backends.mps.is_built()
            and torch.backends.mps.is_available()
        ):
            return torch.device("mps")
        return torch.device("cpu")
    try:
        device = torch.device(normalized)
    except (RuntimeError, ValueError) as exc:
        raise DomainError(f"invalid benchmark device {requested!r}") from exc
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise DomainError("CUDA was requested but is unavailable")
        index = 0 if device.index is None else device.index
        if index < 0 or index >= torch.cuda.device_count():
            raise DomainError("requested CUDA device lies outside the visible devices")
        return torch.device("cuda", index)
    if device.type == "mps":
        if not (
            hasattr(torch.backends, "mps")
            and torch.backends.mps.is_built()
            and torch.backends.mps.is_available()
        ):
            raise DomainError("MPS was requested but is unavailable")
        return torch.device("mps")
    if device.type != "cpu" or device.index is not None:
        raise DomainError("benchmark device must be cpu, cuda[:index], mps, or auto")
    return device


def _require_deterministic_cuda_environment(device: Any) -> None:
    if device.type == "cuda":
        workspace_config = os.environ.get("CUBLAS_WORKSPACE_CONFIG")
        if workspace_config not in _DETERMINISTIC_CUBLAS_WORKSPACE_CONFIGS:
            raise DomainError(
                "deterministic CUDA benchmarking requires "
                "CUBLAS_WORKSPACE_CONFIG=:4096:8 (or :16:8) to be set before "
                "the Python process starts"
            )


def _precision_policy(config: GeometryBenchmarkConfig, device: Any) -> tuple[Any, Any | None]:
    _require_deterministic_cuda_environment(device)
    if config.precision == "float64":
        if device.type == "mps":
            raise DomainError("Float64 is unavailable on MPS")
        return torch.float64, None
    if config.precision == "float32":
        return torch.float32, None
    if config.precision == "bfloat16":
        if device.type not in {"cpu", "cuda"}:
            raise DomainError("BF16 autocast requires CPU or CUDA")
        return torch.float32, torch.bfloat16
    if device.type != "cuda":
        raise DomainError("FP16 autocast is supported only on CUDA")
    return torch.float32, torch.float16


def _synchronize(device: Any) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()


def _timed(device: Any, operation: Callable[[], _T]) -> tuple[float, _T]:
    _synchronize(device)
    start = time.perf_counter_ns()
    result = operation()
    _synchronize(device)
    elapsed_ns = time.perf_counter_ns() - start
    return max(elapsed_ns, 1) / 1_000_000_000.0, result


def _logical_tensor_bytes(values: Sequence[Any]) -> int:
    return sum(int(value.numel()) * int(value.element_size()) for value in values)


def _edge_tensors(edges: PeriodicEdges) -> tuple[Any, ...]:
    return (
        edges.destination,
        edges.source,
        edges.shifts,
        edges.displacement,
        edges.squared_distance,
    )


def _activation_edges(edges: PeriodicEdges, autocast_dtype: Any | None) -> PeriodicEdges:
    if autocast_dtype is None:
        return edges
    displacement = cast(Any, edges.displacement)
    squared_distance = cast(Any, edges.squared_distance)
    return PeriodicEdges(
        destination=edges.destination,
        source=edges.source,
        shifts=edges.shifts,
        displacement=displacement.to(dtype=autocast_dtype),
        squared_distance=squared_distance.to(dtype=autocast_dtype),
    )


def _initialize_model(model: Any, seed: int) -> None:
    """Initialize parameters from exact dyadic signals without ambient RNG use."""

    with torch.no_grad():
        for parameter_index, parameter in enumerate(model.parameters()):
            offset = (
                (seed % _MODULAR_SIGNAL_MODULUS) * 104_729
                + 8_191
                + parameter_index * 13_007
            ) & _MODULAR_SIGNAL_MASK
            values = _dyadic_modular_signal(
                parameter.numel(),
                multiplier=11_406_733,
                offset=offset,
                scale_exponent=-29,
            )
            parameter.copy_(values.reshape(parameter.shape).to(parameter))


def _model_sha256(model: Any) -> str:
    digest = sha256()
    for name, value in sorted(model.state_dict().items()):
        tensor = value.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(tuple(tensor.shape)).encode("ascii"))
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(tensor.numpy().tobytes(order="C"))
    return digest.hexdigest()


def _runtime_manifest(device: Any, *, cpu_threads: int) -> dict[str, object]:
    if device.type == "cuda":
        properties = torch.cuda.get_device_properties(device)
        device_name = properties.name
        total_memory = int(properties.total_memory)
        capability: list[int] | None = list(torch.cuda.get_device_capability(device))
    elif device.type == "mps":
        device_name = "Apple Metal Performance Shaders"
        total_memory = None
        capability = None
    else:
        device_name = platform.processor() or platform.machine() or "unknown-cpu"
        total_memory = None
        capability = None
    cuda_version = torch.version.cuda
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "torch": str(torch.__version__),
        "device": str(device),
        "device_type": device.type,
        "device_index": device.index,
        "device_name": device_name,
        "device_total_memory_bytes": total_memory,
        "cuda_runtime": cuda_version,
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        "deterministic_cublas_workspace_config_valid": (
            device.type != "cuda"
            or os.environ.get("CUBLAS_WORKSPACE_CONFIG")
            in _DETERMINISTIC_CUBLAS_WORKSPACE_CONFIGS
        ),
        "cuda_capability": capability,
        "cudnn": torch.backends.cudnn.version() if device.type == "cuda" else None,
        "mps_built": bool(hasattr(torch.backends, "mps") and torch.backends.mps.is_built()),
        "mps_available": bool(
            hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
        ),
        "cpu_threads": cpu_threads,
        "interop_threads": torch.get_num_interop_threads(),
        "deterministic_algorithms": bool(torch.are_deterministic_algorithms_enabled()),
        "deterministic_warn_only": bool(torch.is_deterministic_algorithms_warn_only_enabled()),
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
        "cuda_matmul_tf32": bool(torch.backends.cuda.matmul.allow_tf32),
        "cudnn_tf32": bool(torch.backends.cudnn.allow_tf32),
        "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
        "cudnn_deterministic": bool(torch.backends.cudnn.deterministic),
        "timing_clock": "time.perf_counter_ns",
        "synchronization": (
            "torch.cuda.synchronize"
            if device.type == "cuda"
            else "torch.mps.synchronize"
            if device.type == "mps"
            else "synchronous-cpu-dispatch"
        ),
    }


def _make_workload(
    config: GeometryBenchmarkConfig,
    *,
    device: Any,
    dtype: Any,
) -> tuple[Any, Any, Any, Any, Any, Any]:
    batch = config.batch_size
    atoms = config.atoms_per_structure
    atom_index = torch.arange(atoms, dtype=torch.float64, device="cpu")
    batch_index_cpu = torch.arange(batch, dtype=torch.float64, device="cpu")
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
            0.031 * batch_index_cpu,
            0.047 * batch_index_cpu,
            0.059 * batch_index_cpu,
        ),
        dim=-1,
    )
    fractional = torch.remainder(base.unsqueeze(0) + offsets.unsqueeze(1), 1.0)
    lattices = (
        torch.eye(3, dtype=torch.float64)
        .mul(config.cell_length)
        .unsqueeze(0)
        .repeat(batch, 1, 1)
    )
    feature_offset = (
        (config.seed % _MODULAR_SIGNAL_MODULUS) * 104_729 + 8_191
    ) & _MODULAR_SIGNAL_MASK
    features = _dyadic_modular_signal(
        batch * atoms * config.input_scalar_dim,
        multiplier=11_406_733,
        offset=feature_offset,
        scale_exponent=-25,
    )
    features = features.reshape(batch * atoms, config.input_scalar_dim)
    graph_index = torch.arange(batch, dtype=torch.long).repeat_interleave(atoms)
    scalar_target = _dyadic_modular_signal(
        batch * atoms * config.scalar_dim,
        multiplier=12_820_163,
        offset=1_234_567,
        scale_exponent=-26,
    )
    scalar_target = scalar_target.reshape(batch * atoms, config.scalar_dim)
    vector_target = _dyadic_modular_signal(
        batch * atoms * config.vector_dim * 3,
        multiplier=1_103_515,
        offset=7_654_321,
        scale_exponent=-27,
    )
    vector_target = vector_target.reshape(batch * atoms, config.vector_dim, 3)
    return tuple(
        value.to(device=device, dtype=dtype)
        if value.is_floating_point()
        else value.to(device=device)
        for value in (
            features,
            fractional,
            lattices,
            graph_index,
            scalar_target,
            vector_target,
        )
    )


def _allocator_memory(device: Any) -> dict[str, object]:
    if device.type == "cuda":
        return {
            "allocator": "torch.cuda",
            "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
            "peak_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
            "current_allocated_bytes": int(torch.cuda.memory_allocated(device)),
            "peak_scope": "measured training phase after warmup; absolute live bytes",
        }
    if device.type == "mps" and hasattr(torch.mps, "current_allocated_memory"):
        return {
            "allocator": "torch.mps",
            "peak_allocated_bytes": None,
            "peak_reserved_bytes": None,
            "current_allocated_bytes": int(torch.mps.current_allocated_memory()),
            "peak_scope": "MPS exposes current, not resettable phase-local peak memory",
        }
    return {
        "allocator": None,
        "peak_allocated_bytes": None,
        "peak_reserved_bytes": None,
        "current_allocated_bytes": None,
        "peak_scope": (
            "unavailable; logical tensor bytes exclude activations and allocator overhead"
        ),
    }


def run_single_device_geometry_benchmark(
    config: GeometryBenchmarkConfig | None = None,
) -> GeometryBenchmarkReport:
    """Measure one deterministic periodic-Transformer workload on one device.

    A successful return is a canonical evidence report.  Any failed phase raises
    :class:`BenchmarkExecutionError`; it never emits partial timings under a
    ``completed`` status.  Ambient model RNG state is not consumed because
    initialization and synthetic inputs are constructive rather than random.
    """

    policy = GeometryBenchmarkConfig() if config is None else config
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

        stage = "workload construction"
        transformer_config = PeriodicTransformerConfig(
            input_scalar_dim=policy.input_scalar_dim,
            scalar_dim=policy.scalar_dim,
            vector_dim=policy.vector_dim,
            num_heads=policy.num_heads,
            num_layers=policy.num_layers,
            radial_basis_dim=policy.radial_basis_dim,
            cutoff=policy.cutoff,
            reference_length=1.0,
            max_candidates_per_pair=policy.max_candidates_per_pair,
            max_edges=policy.max_edges,
        )
        # Module constructors use the global CPU generator even though every
        # parameter is overwritten below.  Forking prevents this diagnostic
        # utility from perturbing the caller's ambient training RNG stream.
        with torch.random.fork_rng(devices=[], enabled=True):
            model = PeriodicSE3Transformer(transformer_config).to(
                device=device, dtype=master_dtype
            )
        _initialize_model(model, policy.seed)
        initial_model_sha256 = _model_sha256(model)
        (
            scalar_features,
            fractional,
            lattices,
            graph_index,
            scalar_target,
            vector_target,
        ) = _make_workload(policy, device=device, dtype=master_dtype)
        flat_fractional = fractional.reshape(policy.batch_size * policy.atoms_per_structure, 3)
        fractional_parts = tuple(fractional[index] for index in range(policy.batch_size))
        lattice_parts = tuple(lattices[index] for index in range(policy.batch_size))

        def build_edges() -> PeriodicEdges:
            return enumerate_packed_periodic_edges(
                fractional_parts,
                lattice_parts,
                policy.cutoff,
                max_candidates_per_pair=policy.max_candidates_per_pair,
                max_edges=policy.max_edges,
            )

        reference_edges = build_edges()
        if reference_edges.num_edges < 1:
            raise DomainError("synthetic geometry workload unexpectedly has no periodic edges")
        activation_reference_edges = _activation_edges(reference_edges, autocast_dtype)
        atoms_per_iteration = policy.batch_size * policy.atoms_per_structure

        stage = "periodic-edge warmup"
        for _ in range(policy.warmup_steps):
            build_edges()
        edge_times: list[float] = []
        last_edges = reference_edges
        stage = "periodic-edge measurement"
        for _ in range(policy.measurement_steps):
            duration, last_edges = _timed(device, build_edges)
            if last_edges.num_edges != reference_edges.num_edges:
                raise RuntimeError("fixed workload changed its exact periodic edge count")
            edge_times.append(duration)

        def forward_precomputed() -> Any:
            with (
                torch.no_grad(),
                torch.autocast(
                    device_type=device.type,
                    dtype=autocast_dtype,
                    enabled=autocast_dtype is not None,
                ),
            ):
                return model(
                    scalar_features,
                    flat_fractional,
                    lattices[0],
                    edges=activation_reference_edges,
                    batch_index=graph_index,
                    num_graphs=policy.batch_size,
                )

        stage = "forward warmup"
        model.eval()
        for _ in range(policy.warmup_steps):
            forward_precomputed()
        forward_times: list[float] = []
        forward_output = forward_precomputed()
        stage = "forward measurement"
        for _ in range(policy.measurement_steps):
            duration, forward_output = _timed(device, forward_precomputed)
            forward_times.append(duration)
        if not bool(
            torch.isfinite(forward_output.scalar_channels).all().item()
            and torch.isfinite(forward_output.vector_channels).all().item()
        ):
            raise RuntimeError("forward phase produced a nonfinite tensor")

        stage = "training setup"
        model.train()
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=policy.learning_rate,
            foreach=False,
            fused=False,
        )
        scaler = (
            torch.amp.GradScaler("cuda", enabled=True)
            if autocast_dtype is torch.float16
            else None
        )
        observed_losses: list[float] = []
        training_edges = reference_edges
        training_output = forward_output

        def training_step() -> tuple[float, PeriodicEdges, Any]:
            optimizer.zero_grad(set_to_none=True)
            exact_edges = build_edges()
            learned_edges = _activation_edges(exact_edges, autocast_dtype)
            with torch.autocast(
                device_type=device.type,
                dtype=autocast_dtype,
                enabled=autocast_dtype is not None,
            ):
                output = model(
                    scalar_features,
                    flat_fractional,
                    lattices[0],
                    edges=learned_edges,
                    batch_index=graph_index,
                    num_graphs=policy.batch_size,
                )
                loss = torch.nn.functional.mse_loss(
                    output.scalar_channels,
                    scalar_target.to(dtype=output.scalar_channels.dtype),
                ) + torch.nn.functional.mse_loss(
                    output.vector_channels,
                    vector_target.to(dtype=output.vector_channels.dtype),
                )
            if scaler is None:
                loss.backward()
                optimizer.step()
            else:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            return float(loss.detach().to(torch.float64).cpu().item()), exact_edges, output

        stage = "training warmup"
        for _ in range(policy.warmup_steps):
            loss_value, training_edges, training_output = training_step()
            if not math.isfinite(loss_value):
                raise RuntimeError("training warmup produced a nonfinite loss")
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        training_times: list[float] = []
        stage = "training measurement"
        for _ in range(policy.measurement_steps):
            duration, result = _timed(device, training_step)
            loss_value, training_edges, training_output = result
            if not math.isfinite(loss_value):
                raise RuntimeError("training measurement produced a nonfinite loss")
            if training_edges.num_edges != reference_edges.num_edges:
                raise RuntimeError("training step changed its exact periodic edge count")
            observed_losses.append(loss_value)
            training_times.append(duration)
        expected_optimizer_steps = policy.warmup_steps + policy.measurement_steps
        optimizer_step_values = {
            int(value.item())
            for state in optimizer.state.values()
            for name, value in state.items()
            if name == "step" and isinstance(value, Tensor)
        }
        if optimizer_step_values != {expected_optimizer_steps}:
            raise RuntimeError(
                "optimizer did not commit every warmup and measured training step"
            )

        stage = "report construction"
        parameter_tensors = tuple(model.parameters())
        buffer_tensors = tuple(model.buffers())
        gradient_tensors = tuple(
            parameter.grad for parameter in parameter_tensors if parameter.grad is not None
        )
        optimizer_tensors = tuple(
            value
            for state in optimizer.state.values()
            for value in state.values()
            if isinstance(value, Tensor)
        )
        input_tensors = (
            scalar_features,
            fractional,
            lattices,
            graph_index,
            scalar_target,
            vector_target,
        )
        output_tensors = (
            training_output.scalar_channels,
            training_output.vector_channels,
            training_output.pooled_scalars,
        )
        logical = {
            "parameter_bytes": _logical_tensor_bytes(parameter_tensors),
            "buffer_bytes": _logical_tensor_bytes(buffer_tensors),
            "gradient_bytes": _logical_tensor_bytes(gradient_tensors),
            "optimizer_state_bytes": _logical_tensor_bytes(optimizer_tensors),
            "synthetic_input_and_target_bytes": _logical_tensor_bytes(input_tensors),
            "exact_edge_tensor_bytes": _logical_tensor_bytes(_edge_tensors(training_edges)),
            "retained_output_bytes": _logical_tensor_bytes(output_tensors),
        }
        memory: dict[str, object] = {
            "logical_tensor_payload": logical,
            "known_live_logical_tensor_bytes": sum(logical.values()),
            "logical_accounting_excludes": [
                "autograd saved activations",
                "temporary kernels",
                "allocator fragmentation",
                "framework and process memory",
            ],
            "backend_allocator": _allocator_memory(device),
        }
        runtime = _runtime_manifest(device, cpu_threads=effective_threads)
        workload_payload: dict[str, object] = {
            "workload_generator": "constructive-modular-fractional-grid.v2",
            "parameter_initializer": DYADIC_SIGNAL_INITIALIZER,
            "synthetic_signal_initializer": DYADIC_SIGNAL_INITIALIZER,
            "optimizer": {
                "class": "torch.optim.AdamW",
                "learning_rate": float(policy.learning_rate),
                "foreach": False,
                "fused": False,
            },
            "structures_per_iteration": policy.batch_size,
            "atoms_per_iteration": atoms_per_iteration,
            "exact_periodic_edges_per_iteration": reference_edges.num_edges,
            "trainable_parameter_count": sum(
                int(parameter.numel()) for parameter in parameter_tensors
            ),
            "master_dtype": str(master_dtype),
            "autocast_dtype": None if autocast_dtype is None else str(autocast_dtype),
            "edge_membership_precision": str(master_dtype),
            "initial_model_sha256": initial_model_sha256,
        }
        workload_payload["fingerprint"] = _fingerprint(workload_payload)
        phases = (
            PhaseMeasurement(
                phase="periodic_edge_enumeration",
                includes=("exact periodic image enumeration",),
                durations_seconds=tuple(edge_times),
                structures_per_iteration=policy.batch_size,
                atoms_per_iteration=atoms_per_iteration,
                edges_per_iteration=reference_edges.num_edges,
            ),
            PhaseMeasurement(
                phase="transformer_forward_precomputed_edges",
                includes=("proper-SE(3) Transformer forward", "device synchronization"),
                durations_seconds=tuple(forward_times),
                structures_per_iteration=policy.batch_size,
                atoms_per_iteration=atoms_per_iteration,
                edges_per_iteration=reference_edges.num_edges,
            ),
            PhaseMeasurement(
                phase="training_step_end_to_end",
                includes=(
                    "optimizer zero_grad",
                    "exact periodic image enumeration",
                    "proper-SE(3) Transformer forward",
                    "loss",
                    "backward",
                    "AdamW update",
                    "device synchronization",
                ),
                durations_seconds=tuple(training_times),
                structures_per_iteration=policy.batch_size,
                atoms_per_iteration=atoms_per_iteration,
                edges_per_iteration=reference_edges.num_edges,
            ),
        )
        numerics = {
            "all_measured_losses_finite": True,
            "optimizer_updates_completed": expected_optimizer_steps,
            "first_measured_loss": observed_losses[0],
            "last_measured_loss": observed_losses[-1],
            "measured_losses": observed_losses,
            "final_model_sha256": _model_sha256(model),
        }
        return GeometryBenchmarkReport(
            config=policy,
            runtime=runtime,
            workload=workload_payload,
            phases=phases,
            memory=memory,
            numerics=numerics,
        )
    except DomainError:
        raise
    except Exception as exc:
        raise BenchmarkExecutionError(stage, exc) from exc
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
    "BENCHMARK_FACTOR",
    "BENCHMARK_METHOD",
    "BENCHMARK_SCHEMA",
    "DYADIC_SIGNAL_INITIALIZER",
    "BenchmarkExecutionError",
    "GeometryBenchmarkConfig",
    "GeometryBenchmarkReport",
    "PhaseMeasurement",
    "run_single_device_geometry_benchmark",
]
