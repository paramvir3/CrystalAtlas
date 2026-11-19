r"""Scalable, staged optimization for the three CrystalAtlas factors.

The trainer preserves the probabilistic factorization instead of treating all
terms as interchangeable losses.  Composition, arithmetic-cover selection,
and intrinsic conditional-flow matching are trained in separate phases before
an explicitly weighted joint phase.  Each callback returns a *structure mean*
and the number of structures represented by that mean.  For an accumulation
window :math:`W`, the differentiated objective is therefore exactly

.. math::

   \frac{1}{\sum_{b\in W} n_b}
   \sum_{b\in W} n_b\,\mathcal L_b.

This is the explicitly materialized diagnostic-batch convention. Production
bucket samplers additionally declare ``mu=N_epoch/B_epoch`` in every batch.
For these batches the gradient denominator is the fixed nominal
``world_size * accumulation_steps * mu``, including a short final window.
Consequently each retained record has the same epoch gradient coefficient:
small topology buckets and partial windows are not promoted to full batches.
Loss metrics remain observed per-record means, and reports expose the separate
gradient denominator. The nominal denominator also fixes the learning-rate
convention; a short final window has a smaller gradient contribution. Adaptive
optimizers, momentum, clipping and weight decay need not scale their updates
proportionately with that contribution.
Under DDP, PyTorch first averages the accumulated gradient across ranks; the
trainer subsequently multiplies by ``world_size / gradient_denominator``.

PyTorch is optional for CrystalAtlas as a whole.  Importing this module, like
the neural modules, requires the ``ml`` extra.
"""

from __future__ import annotations

import copy
import inspect
import json
import marshal
import math
import pickle
from collections.abc import Callable, Iterable, Mapping, Sequence
from contextlib import ExitStack, contextmanager, nullcontext
from dataclasses import dataclass, replace
from enum import Enum
from hashlib import sha256
from itertools import islice
from pathlib import Path
from typing import Any, Protocol

from ._optional import require_torch
from .config import CrystalAtlasConfig
from .errors import DomainError
from .provenance import ProvenanceRecord
from .training import (
    CheckpointError,
    CheckpointInfo,
    GradientReport,
    LoadedCheckpoint,
    LossWeights,
    ScientificManifest,
    TrainingContext,
    TrainingNumericsError,
    capture_global_rng_state,
    compose_training_loss,
    inspect_gradients,
    load_checkpoint,
    restore_global_rng_state,
    save_checkpoint,
)

torch = require_torch()
Tensor = torch.Tensor
nn = torch.nn

__all__ = [
    "ComponentLosses",
    "ComponentModelNames",
    "ComponentModels",
    "CrystallographicAuditResult",
    "DistributedConfig",
    "EpochReport",
    "ExponentialMovingAverage",
    "ObjectiveFunction",
    "PrecisionMode",
    "ScalableTrainer",
    "SchedulerInterval",
    "StagedTrainingSchedule",
    "TrainStepReport",
    "TrainerConfig",
    "TrainerCursor",
    "TrainingPhase",
]

_TRAINER_STATE_SCHEMA = "crystalatlas.scalable-trainer"
_TRAINER_STATE_VERSION = 6


class _AMPGradientOverflowError(TrainingNumericsError):
    """Internal signal: retry the unchanged window after scaler backoff."""


class TrainingPhase(str, Enum):  # noqa: UP042 - Python 3.10 ML smoke runtime
    """The three factor-wise phases followed by optional joint fine-tuning."""

    COMPOSITION = "composition"
    COVER = "cover"
    GEOMETRY = "geometry"
    JOINT = "joint"


class PrecisionMode(str, Enum):  # noqa: UP042 - Python 3.10 ML smoke runtime
    """Neural forward/backward precision; crystallographic audits stay float64."""

    FLOAT32 = "float32"
    BFLOAT16 = "bfloat16"
    FLOAT16 = "float16"


class SchedulerInterval(str, Enum):  # noqa: UP042 - Python 3.10 ML smoke runtime
    """When ``scheduler.step()`` is invoked."""

    OPTIMIZER_STEP = "optimizer_step"
    EPOCH = "epoch"


@dataclass(frozen=True, slots=True)
class StagedTrainingSchedule:
    """An immutable factor-first training schedule.

    A nonzero joint phase is rejected unless each factor has a nonzero
    pretraining phase.  This makes "separate first, joint second" executable
    policy rather than documentation.
    """

    composition_epochs: int
    cover_epochs: int
    geometry_epochs: int
    joint_epochs: int = 0

    def __post_init__(self) -> None:
        values = (
            self.composition_epochs,
            self.cover_epochs,
            self.geometry_epochs,
            self.joint_epochs,
        )
        if any(
            isinstance(value, bool) or not isinstance(value, int) or value < 0
            for value in values
        ):
            raise DomainError("all phase epoch counts must be nonnegative integers")
        if sum(values) == 0:
            raise DomainError("a staged schedule must contain at least one epoch")
        if self.joint_epochs and any(value == 0 for value in values[:3]):
            raise DomainError(
                "joint fine-tuning requires nonzero composition, cover, and geometry phases"
            )

    @property
    def total_epochs(self) -> int:
        return (
            self.composition_epochs
            + self.cover_epochs
            + self.geometry_epochs
            + self.joint_epochs
        )

    @property
    def fingerprint(self) -> str:
        payload = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))
        return sha256(payload.encode("utf-8")).hexdigest()

    def to_dict(self) -> dict[str, int]:
        return {
            "composition_epochs": self.composition_epochs,
            "cover_epochs": self.cover_epochs,
            "geometry_epochs": self.geometry_epochs,
            "joint_epochs": self.joint_epochs,
        }

    def phase_at(self, epoch: int) -> TrainingPhase:
        if isinstance(epoch, bool) or not isinstance(epoch, int):
            raise TypeError("epoch must be an integer")
        if not 0 <= epoch < self.total_epochs:
            raise DomainError("epoch lies outside the staged schedule")
        boundary = self.composition_epochs
        if epoch < boundary:
            return TrainingPhase.COMPOSITION
        boundary += self.cover_epochs
        if epoch < boundary:
            return TrainingPhase.COVER
        boundary += self.geometry_epochs
        if epoch < boundary:
            return TrainingPhase.GEOMETRY
        return TrainingPhase.JOINT

    def phase_epoch_at(self, epoch: int) -> int:
        phase = self.phase_at(epoch)
        starts = {
            TrainingPhase.COMPOSITION: 0,
            TrainingPhase.COVER: self.composition_epochs,
            TrainingPhase.GEOMETRY: self.composition_epochs + self.cover_epochs,
            TrainingPhase.JOINT: (
                self.composition_epochs + self.cover_epochs + self.geometry_epochs
            ),
        }
        return epoch - starts[phase]


@dataclass(frozen=True, slots=True)
class ComponentModelNames:
    """Dotted attributes locating the three non-overlapping model factors."""

    composition: str = "composition_model"
    cover: str = "cover_selector"
    geometry: str = "geometry_field"

    def __post_init__(self) -> None:
        values = (self.composition, self.cover, self.geometry)
        if any(not isinstance(value, str) or not value.strip() for value in values):
            raise DomainError("component module names must be nonempty text")
        if len(set(values)) != 3:
            raise DomainError("component module names must be distinct")

    def to_dict(self) -> dict[str, str]:
        return {
            "composition": self.composition,
            "cover": self.cover,
            "geometry": self.geometry,
        }


@dataclass(frozen=True, slots=True)
class DistributedConfig:
    """One-process-per-device DDP configuration managed by the trainer."""

    enabled: bool = False
    local_rank: int | None = None
    broadcast_buffers: bool = False
    # Exact fixed-site or zero-edge topologies can make legal field branches
    # inactive.  This can differ across rank-local topology buckets, so the
    # safe crystallographic default must participate in DDP's unused-parameter
    # discovery rather than assuming one dense computational graph.
    find_unused_parameters: bool = True

    def __post_init__(self) -> None:
        if not isinstance(self.enabled, bool):
            raise TypeError("enabled must be boolean")
        if self.local_rank is not None and (
            isinstance(self.local_rank, bool)
            or not isinstance(self.local_rank, int)
            or self.local_rank < 0
        ):
            raise DomainError("local_rank must be a nonnegative integer or None")
        if not isinstance(self.broadcast_buffers, bool):
            raise TypeError("broadcast_buffers must be boolean")
        if not isinstance(self.find_unused_parameters, bool):
            raise TypeError("find_unused_parameters must be boolean")


@dataclass(frozen=True, slots=True)
class TrainerConfig:
    """Numerical and systems policy whose identity is pinned in checkpoints."""

    accumulation_steps: int = 1
    precision: PrecisionMode = PrecisionMode.FLOAT32
    max_amp_overflow_retries: int = 16
    max_gradient_norm: float | None = None
    ema_decay: float | None = 0.999
    scheduler_interval: SchedulerInterval = SchedulerInterval.OPTIMIZER_STEP
    # Full model+Adam+EMA copies on every step are prohibitive at scale.
    # Production relies on atomic checkpoints; opt in only for small diagnostic
    # runs where per-step rollback is worth the memory cost.
    transactional_optimizer_step: bool = False
    joint_loss_weights: LossWeights = LossWeights()
    components: ComponentModelNames = ComponentModelNames()
    distributed: DistributedConfig = DistributedConfig()

    def __post_init__(self) -> None:
        if (
            isinstance(self.accumulation_steps, bool)
            or not isinstance(self.accumulation_steps, int)
            or self.accumulation_steps < 1
        ):
            raise DomainError("accumulation_steps must be a positive integer")
        if not isinstance(self.precision, PrecisionMode):
            raise TypeError("precision must be a PrecisionMode")
        if (
            isinstance(self.max_amp_overflow_retries, bool)
            or not isinstance(self.max_amp_overflow_retries, int)
            or self.max_amp_overflow_retries < 0
        ):
            raise DomainError("max_amp_overflow_retries must be a nonnegative integer")
        if self.max_gradient_norm is not None and (
            not math.isfinite(self.max_gradient_norm) or self.max_gradient_norm <= 0.0
        ):
            raise DomainError("max_gradient_norm must be finite and positive")
        if self.ema_decay is not None and (
            not math.isfinite(self.ema_decay) or not 0.0 <= self.ema_decay < 1.0
        ):
            raise DomainError("ema_decay must lie in [0, 1) or be None")
        if not isinstance(self.scheduler_interval, SchedulerInterval):
            raise TypeError("scheduler_interval must be a SchedulerInterval")
        if not isinstance(self.transactional_optimizer_step, bool):
            raise TypeError("transactional_optimizer_step must be boolean")
        if not isinstance(self.joint_loss_weights, LossWeights):
            raise TypeError("joint_loss_weights must be LossWeights")
        if not isinstance(self.components, ComponentModelNames):
            raise TypeError("components must be ComponentModelNames")
        if not isinstance(self.distributed, DistributedConfig):
            raise TypeError("distributed must be DistributedConfig")

    def to_dict(self) -> dict[str, Any]:
        weights = self.joint_loss_weights
        return {
            "accumulation_steps": self.accumulation_steps,
            "gradient_normalization": "sampler-population-fixed-nominal-window.v1",
            "precision": self.precision.value,
            "max_amp_overflow_retries": self.max_amp_overflow_retries,
            "max_gradient_norm": self.max_gradient_norm,
            "ema_decay": self.ema_decay,
            "scheduler_interval": self.scheduler_interval.value,
            "transactional_optimizer_step": self.transactional_optimizer_step,
            "joint_loss_weights": {
                "composition_nll": weights.composition_nll,
                "cover_nll": weights.cover_nll,
                "cfm_regression": weights.cfm_regression,
            },
            "components": self.components.to_dict(),
            "distributed": {
                "enabled": self.distributed.enabled,
                "broadcast_buffers": self.distributed.broadcast_buffers,
                "find_unused_parameters": self.distributed.find_unused_parameters,
            },
        }

    @property
    def fingerprint(self) -> str:
        payload = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))
        return sha256(payload.encode("utf-8")).hexdigest()


@dataclass(frozen=True, slots=True)
class ComponentLosses:
    """Per-structure means returned by one objective callback.

    ``batch_size`` is the number of structures, not the number of tokens,
    atoms, Wyckoff actions, or intrinsic coordinates.  Each component must be
    reduced to a mean per structure before construction; this common measure is
    what permits exact joint gradient accumulation.
    """

    batch_size: int
    composition_nll: Tensor | None = None
    cover_nll: Tensor | None = None
    cfm_regression: Tensor | None = None

    def __post_init__(self) -> None:
        if (
            isinstance(self.batch_size, bool)
            or not isinstance(self.batch_size, int)
            or self.batch_size < 1
        ):
            raise DomainError("batch_size must be a positive integer")
        devices: set[Any] = set()
        present = 0
        for name in ("composition_nll", "cover_nll", "cfm_regression"):
            value = getattr(self, name)
            if value is None:
                continue
            present += 1
            if not isinstance(value, Tensor) or value.numel() != 1:
                raise TypeError(f"{name} must be a scalar PyTorch tensor or None")
            if not torch.is_floating_point(value):
                raise TypeError(f"{name} must be floating point")
            devices.add(value.device)
        if present == 0:
            raise DomainError("at least one component loss must be present")
        if len(devices) != 1:
            raise DomainError("all component losses must be on the same device")

    def objective(self, phase: TrainingPhase, weights: LossWeights) -> Tensor:
        if phase is TrainingPhase.COMPOSITION:
            if self.composition_nll is None:
                raise DomainError("composition phase requires composition_nll")
            return self.composition_nll
        if phase is TrainingPhase.COVER:
            if self.cover_nll is None:
                raise DomainError("cover phase requires cover_nll")
            return self.cover_nll
        if phase is TrainingPhase.GEOMETRY:
            if self.cfm_regression is None:
                raise DomainError("geometry phase requires cfm_regression")
            return self.cfm_regression
        if (
            self.composition_nll is None
            or self.cover_nll is None
            or self.cfm_regression is None
        ):
            raise DomainError("joint phase requires all three component losses")
        return compose_training_loss(
            self.composition_nll,
            self.cover_nll,
            self.cfm_regression,
            weights=weights,
        ).total_objective

    def present_values(self) -> dict[str, Tensor]:
        values = {
            "nll/composition": self.composition_nll,
            "nll/cover": self.cover_nll,
            "regression/cfm": self.cfm_regression,
        }
        return {name: value for name, value in values.items() if value is not None}


@dataclass(frozen=True, slots=True)
class CrystallographicAuditResult:
    """Float64 residuals from an exact-decoder or symmetry audit."""

    residuals: Tensor
    tolerance: float

    def __post_init__(self) -> None:
        if not isinstance(self.residuals, Tensor):
            raise TypeError("audit residuals must be a PyTorch tensor")
        if self.residuals.dtype != torch.float64:
            raise TypeError("crystallographic audit residuals must have dtype float64")
        if not bool(torch.isfinite(self.residuals).all().item()):
            raise TrainingNumericsError("crystallographic audit residuals are non-finite")
        if not math.isfinite(self.tolerance) or self.tolerance < 0.0:
            raise DomainError("audit tolerance must be finite and nonnegative")

    @property
    def passed(self) -> bool:
        return bool((self.residuals.abs() <= self.tolerance).all().item())

    def require_passed(self) -> None:
        if not self.passed:
            maximum = float(self.residuals.abs().max().item())
            raise DomainError(
                "float64 crystallographic audit failed: "
                f"max residual {maximum:.17g} exceeds {self.tolerance:.17g}"
            )


@dataclass(frozen=True, slots=True)
class ComponentModels:
    """Forward-callable factor modules supplied to the objective callback."""

    root: Any
    composition: Any
    cover: Any
    geometry: Any


class ObjectiveFunction(Protocol):
    def __call__(
        self,
        models: ComponentModels,
        batch: Any,
        phase: TrainingPhase,
    ) -> ComponentLosses: ...


AuditFunction = Callable[[Any, Any], CrystallographicAuditResult]


@dataclass(frozen=True, slots=True)
class TrainerCursor:
    """Committed position; no partially accumulated graph is checkpointed."""

    epoch: int = 0
    microbatch: int = 0
    optimizer_step: int = 0

    def __post_init__(self) -> None:
        for name in ("epoch", "microbatch", "optimizer_step"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise DomainError(f"cursor {name} must be a nonnegative integer")

    def to_dict(self) -> dict[str, int]:
        return {
            "epoch": self.epoch,
            "microbatch": self.microbatch,
            "optimizer_step": self.optimizer_step,
        }


@dataclass(frozen=True, slots=True)
class TrainStepReport:
    phase: TrainingPhase
    optimizer_step: int
    microbatches: int
    global_structures: int
    metrics: Mapping[str, float]
    gradients_before_clip: GradientReport
    gradients_after_clip: GradientReport
    clipped: bool
    learning_rates: tuple[float, ...]
    gradient_normalization_count: float
    gradient_normalization_policy: str


@dataclass(frozen=True, slots=True)
class EpochReport:
    phase: TrainingPhase
    epoch: int
    phase_epoch: int
    optimizer_steps: int
    global_structures: int
    metrics: Mapping[str, float]
    learning_rates: tuple[float, ...]


class ExponentialMovingAverage:
    """Parameter EMA with a strict name/shape/dtype resume contract."""

    def __init__(self, model: Any, decay: float) -> None:
        if not math.isfinite(decay) or not 0.0 <= decay < 1.0:
            raise DomainError("EMA decay must lie in [0, 1)")
        self.decay = float(decay)
        self._shadow = {
            name: parameter.detach().clone()
            for name, parameter in model.named_parameters()
            if parameter.requires_grad
            and (torch.is_floating_point(parameter) or torch.is_complex(parameter))
        }
        if not self._shadow:
            raise DomainError("EMA requires at least one trainable floating-point parameter")
        self._updates = {name: 0 for name in self._shadow}

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(sorted(self._shadow))

    @property
    def update_counts(self) -> Mapping[str, int]:
        return dict(self._updates)

    def update(
        self,
        model: Any,
        *,
        parameter_names: Sequence[str] | None = None,
    ) -> None:
        parameters = dict(model.named_parameters())
        if set(parameters) < set(self._shadow):
            raise DomainError("model parameters no longer contain the EMA parameter set")
        selected = self.names if parameter_names is None else tuple(parameter_names)
        if len(set(selected)) != len(selected) or not set(selected).issubset(self._shadow):
            raise DomainError("EMA update names must be a unique subset of its parameters")
        with torch.no_grad():
            for name in selected:
                average = self._shadow[name]
                value = parameters[name].detach()
                if value.shape != average.shape or value.dtype != average.dtype:
                    raise DomainError(f"EMA parameter contract changed for {name!r}")
                average.mul_(self.decay).add_(value, alpha=1.0 - self.decay)
                self._updates[name] += 1

    def state_dict(self) -> dict[str, Any]:
        return {
            "decay": self.decay,
            "updates": dict(self._updates),
            "shadow": {name: value.detach().clone() for name, value in self._shadow.items()},
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        if set(state) != {"decay", "updates", "shadow"}:
            raise CheckpointError("EMA state has missing or unexpected fields")
        decay = state["decay"]
        updates = state["updates"]
        shadows = state["shadow"]
        if decay != self.decay:
            raise CheckpointError("EMA decay differs from the checkpoint")
        if not isinstance(updates, Mapping) or set(updates) != set(self._shadow):
            raise CheckpointError("EMA per-parameter update counts are invalid")
        if any(
            isinstance(value, bool) or not isinstance(value, int) or value < 0
            for value in updates.values()
        ):
            raise CheckpointError("EMA per-parameter update count is invalid")
        if not isinstance(shadows, Mapping) or set(shadows) != set(self._shadow):
            raise CheckpointError("EMA parameter names differ from the checkpoint")
        validated: dict[str, Tensor] = {}
        for name, current in self._shadow.items():
            saved = shadows[name]
            if not isinstance(saved, Tensor):
                raise CheckpointError(f"EMA value {name!r} is not a tensor")
            if saved.shape != current.shape or saved.dtype != current.dtype:
                raise CheckpointError(f"EMA tensor contract differs for {name!r}")
            if not bool(torch.isfinite(saved).all().item()):
                raise CheckpointError(f"EMA tensor {name!r} is non-finite")
            validated[name] = saved.detach().to(device=current.device).clone()
        self._shadow = validated
        self._updates = {name: int(updates[name]) for name in self._shadow}

    def copy_to(self, model: Any) -> None:
        parameters = dict(model.named_parameters())
        with torch.no_grad():
            for name, average in self._shadow.items():
                parameters[name].copy_(average)

    @contextmanager
    def average_parameters(self, model: Any) -> Any:
        parameters = dict(model.named_parameters())
        originals = {name: parameters[name].detach().clone() for name in self._shadow}
        self.copy_to(model)
        try:
            yield model
        finally:
            with torch.no_grad():
                for name, value in originals.items():
                    parameters[name].copy_(value)


def _resolve_module(root: Any, dotted_name: str) -> Any:
    value = root
    for part in dotted_name.split("."):
        if not hasattr(value, part):
            raise DomainError(f"model has no component module {dotted_name!r}")
        value = getattr(value, part)
    if not isinstance(value, nn.Module):
        raise DomainError(f"model component {dotted_name!r} is not a torch module")
    return value


def _collect_floating_tensors(value: Any, groups: dict[Any, list[Tensor]]) -> bool:
    """Queue every floating or complex tensor, flattened; True on a bad Python float."""

    if isinstance(value, Tensor):
        if torch.is_floating_point(value) or torch.is_complex(value):
            stored = value.detach()
            if stored.is_sparse:
                # The implicit zeros of a sparse tensor are finite.
                stored = stored.coalesce().values()
            groups.setdefault((stored.device, stored.dtype), []).append(stored.reshape(-1))
        return False
    if isinstance(value, Mapping):
        return any(_collect_floating_tensors(child, groups) for child in value.values())
    if isinstance(value, (tuple, list)):
        return any(_collect_floating_tensors(child, groups) for child in value)
    return isinstance(value, float) and not math.isfinite(value)


def _state_has_nonfinite(value: Any) -> bool:
    """Whether any floating or complex tensor, or Python float, in a nested state is non-finite.

    The tensors of one device and dtype are concatenated and tested by a single
    ``isfinite().all()``, so checking a model, optimizer or EMA state costs one
    host synchronization per (device, dtype) instead of one per tensor.  A
    concatenation is finite exactly when every part is, so the verdict is the
    same as that of the tensor-by-tensor scan.
    """

    groups: dict[Any, list[Tensor]] = {}
    if _collect_floating_tensors(value, groups):
        return True
    return any(
        not bool(torch.isfinite(torch.cat(flats)).all().item()) for flats in groups.values()
    )


def _clone_state(value: Any) -> Any:
    return copy.deepcopy(value)


def _qualified_type_name(value: Any) -> str:
    value_type = type(value)
    return f"{value_type.__module__}.{value_type.__qualname__}"


def _contract_value(value: Any) -> Any:
    """Convert scientific construction state into canonical checkpoint data."""

    if isinstance(value, Tensor):
        tensor = value.detach().contiguous().cpu()
        raw = tensor.reshape(-1).view(torch.uint8).numpy().tobytes()
        return {
            "kind": "tensor",
            "dtype": str(tensor.dtype),
            "shape": list(tensor.shape),
            "sha256": sha256(raw).hexdigest(),
        }
    if isinstance(value, Mapping):
        if any(not isinstance(key, (str, int)) or isinstance(key, bool) for key in value):
            raise DomainError("contract state contains an unsupported mapping key")
        normalized_keys = tuple(str(key) for key in value)
        if len(set(normalized_keys)) != len(normalized_keys):
            raise DomainError(
                "contract state mapping keys collide after canonical string encoding"
            )
        return {
            str(key): _contract_value(child)
            for key, child in sorted(value.items(), key=lambda item: str(item[0]))
        }
    if isinstance(value, (tuple, list)):
        return [_contract_value(child) for child in value]
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise DomainError("contract state contains a non-finite float")
        return value
    raise DomainError(
        f"contract state contains an unsupported value of type {_qualified_type_name(value)}"
    )


def _scheduler_contract(scheduler: Any | None) -> dict[str, Any] | None:
    if scheduler is None:
        return None
    state = scheduler.state_dict()
    if not isinstance(state, Mapping):
        raise TypeError("scheduler.state_dict() must return a mapping")
    return {
        "class": _qualified_type_name(scheduler),
        "initial_state": _contract_value(state),
    }


def _callable_name(value: Any) -> str:
    module = getattr(value, "__module__", type(value).__module__)
    qualname = getattr(value, "__qualname__", type(value).__qualname__)
    return f"{module}.{qualname}"


def _objective_contract(value: Any) -> dict[str, Any]:
    explicit = getattr(value, "contract", None)
    if explicit is None:
        if not inspect.isfunction(value) or value.__closure__ is not None:
            raise DomainError(
                "stateful, bound, or closure-based objectives require an explicit "
                "canonical contract and contract_fingerprint"
            )
        payload: Any = {
            "schema": "crystalatlas.generic-objective.v2",
            "callable": _callable_name(value),
            "code_sha256": sha256(marshal.dumps(value.__code__)).hexdigest(),
            "defaults": _contract_value(value.__defaults__ or ()),
            "keyword_defaults": _contract_value(value.__kwdefaults__ or {}),
            "checkpointable": False,
        }
    else:
        if not isinstance(explicit, Mapping):
            raise DomainError("objective contract must be a mapping")
        payload = _contract_value(explicit)
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    fingerprint = sha256(encoded.encode("utf-8")).hexdigest()
    declared = getattr(value, "contract_fingerprint", fingerprint)
    if declared != fingerprint:
        raise DomainError("objective contract fingerprint is internally inconsistent")
    return {"payload": payload, "fingerprint": fingerprint}


def _validate_objective_contract(value: Any) -> dict[str, Any] | None:
    if value is None:
        return None
    if not isinstance(value, Mapping) or set(value) != {"payload", "fingerprint"}:
        raise CheckpointError("saved objective contract is malformed")
    payload = value["payload"]
    normalized = _contract_value(payload)
    encoded = json.dumps(normalized, sort_keys=True, separators=(",", ":"))
    fingerprint = sha256(encoded.encode("utf-8")).hexdigest()
    if value["fingerprint"] != fingerprint:
        raise CheckpointError("saved objective contract fingerprint is inconsistent")
    return {"payload": normalized, "fingerprint": fingerprint}


def _objective_is_checkpointable(contract: Mapping[str, Any] | None) -> bool:
    if contract is None:
        return False
    payload = contract.get("payload")
    return isinstance(payload, Mapping) and payload.get("checkpointable") is True


def _objective_dataset_fingerprint(
    contract: Mapping[str, Any] | None,
) -> str | None:
    """Extract the production population-measure identity when declared."""

    if contract is None:
        return None
    payload = contract.get("payload")
    if not isinstance(payload, Mapping):
        return None
    data_measure = payload.get("data_measure")
    if not isinstance(data_measure, Mapping):
        return None
    fingerprint = data_measure.get("dataset_fingerprint")
    if fingerprint is None:
        return None
    if (
        not isinstance(fingerprint, str)
        or len(fingerprint) != 64
        or any(character not in "0123456789abcdef" for character in fingerprint)
    ):
        raise DomainError("objective dataset_fingerprint must be a lowercase SHA-256 digest")
    return fingerprint


class ScalableTrainer:
    """Staged, accumulation-correct trainer for ``CrystalAtlasModel``-like roots.

    In distributed mode the three factor modules are wrapped independently.
    This is intentional: ``CrystalAtlasModel`` is a parameter collection and
    has no monolithic forward method.  Rebuilding the active factor wrappers at
    phase boundaries also avoids changing ``requires_grad`` after a DDP reducer
    has been constructed.
    """

    def __init__(
        self,
        model: Any,
        optimizer: Any,
        schedule: StagedTrainingSchedule,
        *,
        config: TrainerConfig | None = None,
        scheduler: Any | None = None,
        training_context: TrainingContext | None = None,
    ) -> None:
        if not isinstance(model, nn.Module):
            raise TypeError("model must be a torch.nn.Module")
        if isinstance(model, nn.parallel.DistributedDataParallel):
            raise DomainError(
                "pass the unwrapped root model; ScalableTrainer manages factor-wise DDP"
            )
        if not isinstance(schedule, StagedTrainingSchedule):
            raise TypeError("schedule must be a StagedTrainingSchedule")
        self.model = model
        self.optimizer = optimizer
        self.schedule = schedule
        self.config = config or TrainerConfig()
        self.scheduler = scheduler
        if scheduler is not None and schedule.joint_epochs == 0:
            raise DomainError(
                "the shared scheduler is reserved for the all-factor joint phase; "
                "factor pretraining uses fixed per-group learning rates"
            )
        self._scheduler_contract = _scheduler_contract(scheduler)
        self._objective_contract: dict[str, Any] | None = None
        self.training_context = training_context
        self.cursor = TrainerCursor()
        self._faulted = False
        self._component_modules = {
            TrainingPhase.COMPOSITION: _resolve_module(
                model, self.config.components.composition
            ),
            TrainingPhase.COVER: _resolve_module(model, self.config.components.cover),
            TrainingPhase.GEOMETRY: _resolve_module(model, self.config.components.geometry),
        }
        self._eligible_parameters = {
            id(parameter): bool(parameter.requires_grad) for parameter in model.parameters()
        }
        self._eligible_parameter_names = tuple(
            name for name, parameter in model.named_parameters() if parameter.requires_grad
        )
        self._validate_component_partition()
        self._validate_optimizer_ownership()
        self._device = self._infer_device()
        self._master_dtype = self._infer_master_dtype()
        self._validate_distributed_runtime()
        self._validate_precision()
        self._synchronize_root_model_parameters()
        self._scaler = self._make_grad_scaler()
        self.ema = (
            ExponentialMovingAverage(model, self.config.ema_decay)
            if self.config.ema_decay is not None
            else None
        )
        self._wrapped_components: dict[TrainingPhase, Any] = {}
        self._configure_phase(self.current_phase)
        self.optimizer.zero_grad(set_to_none=True)

    @property
    def current_phase(self) -> TrainingPhase:
        if self.cursor.epoch >= self.schedule.total_epochs:
            return self.schedule.phase_at(self.schedule.total_epochs - 1)
        return self.schedule.phase_at(self.cursor.epoch)

    @property
    def phase_epoch(self) -> int:
        if self.cursor.epoch >= self.schedule.total_epochs:
            last = self.schedule.total_epochs - 1
            return self.schedule.phase_epoch_at(last) + 1
        return self.schedule.phase_epoch_at(self.cursor.epoch)

    @property
    def world_size(self) -> int:
        if not self.config.distributed.enabled:
            return 1
        return int(torch.distributed.get_world_size())

    @property
    def rank(self) -> int:
        if not self.config.distributed.enabled:
            return 0
        return int(torch.distributed.get_rank())

    @property
    def models(self) -> ComponentModels:
        return ComponentModels(
            root=self.model,
            composition=self._wrapped_components.get(
                TrainingPhase.COMPOSITION,
                self._component_modules[TrainingPhase.COMPOSITION],
            ),
            cover=self._wrapped_components.get(
                TrainingPhase.COVER,
                self._component_modules[TrainingPhase.COVER],
            ),
            geometry=self._wrapped_components.get(
                TrainingPhase.GEOMETRY,
                self._component_modules[TrainingPhase.GEOMETRY],
            ),
        )

    def _validate_component_partition(self) -> None:
        memberships: dict[int, list[TrainingPhase]] = {}
        for phase, module in self._component_modules.items():
            for parameter in module.parameters():
                memberships.setdefault(id(parameter), []).append(phase)
        overlap = {
            parameter_id: phases
            for parameter_id, phases in memberships.items()
            if len(phases) != 1
        }
        if overlap:
            raise DomainError(
                "component modules must have non-overlapping parameter sets for factor-wise DDP"
            )
        unassigned = [
            name
            for name, parameter in self.model.named_parameters()
            if id(parameter) not in memberships
        ]
        if unassigned:
            raise DomainError(
                "every root-model parameter must belong to exactly one factor; "
                f"unassigned={tuple(unassigned)}"
            )

    def _validate_optimizer_ownership(self) -> None:
        model_ids = set(self._eligible_parameters)
        optimizer_parameters = tuple(
            parameter for group in self.optimizer.param_groups for parameter in group["params"]
        )
        optimizer_ids = tuple(id(parameter) for parameter in optimizer_parameters)
        if len(set(optimizer_ids)) != len(optimizer_ids):
            raise DomainError("optimizer parameter groups must not repeat a model parameter")
        foreign = set(optimizer_ids) - model_ids
        if foreign:
            raise DomainError("optimizer must not own parameters outside the root model")
        required_ids = {
            parameter_id
            for parameter_id, eligible in self._eligible_parameters.items()
            if eligible
        }
        missing = required_ids - set(optimizer_ids)
        if missing:
            raise DomainError(
                "optimizer must own every initially trainable component parameter"
            )

    def _infer_device(self) -> Any:
        devices = {parameter.device for parameter in self.model.parameters()}
        if not devices:
            raise DomainError("trainer model has no parameters")
        if len(devices) != 1:
            raise DomainError("all model parameters must initially reside on one device")
        return next(iter(devices))

    def _infer_master_dtype(self) -> Any:
        dtypes = {parameter.dtype for parameter in self.model.parameters()}
        if len(dtypes) != 1:
            raise DomainError("all model parameters must share one floating master dtype")
        dtype = next(iter(dtypes))
        if dtype not in {torch.float32, torch.float64}:
            raise DomainError("model master dtype must be float32 or float64")
        return dtype

    def _validate_distributed_runtime(self) -> None:
        distributed = self.config.distributed
        if not distributed.enabled:
            return
        if self._device.type not in {"cpu", "cuda"}:
            raise DomainError("factor-wise DDP is supported only on CPU or CUDA devices")
        if not torch.distributed.is_available() or not torch.distributed.is_initialized():
            raise DomainError("distributed training requires an initialized process group")
        if self._device.type == "cuda":
            local_rank = distributed.local_rank
            if local_rank is None:
                raise DomainError("CUDA DDP requires an explicit local_rank")
            if self._device.index != local_rank:
                raise DomainError("model CUDA device must equal distributed local_rank")

    def _synchronize_root_model_parameters(self) -> None:
        """Broadcast every factor before EMA snapshots or phase-specific DDP."""

        if not self.config.distributed.enabled:
            return
        local_schema = (
            tuple(
                (
                    name,
                    tuple(parameter.shape),
                    str(parameter.dtype),
                    bool(parameter.requires_grad),
                )
                for name, parameter in self.model.named_parameters()
            ),
            tuple(
                (name, tuple(buffer.shape), str(buffer.dtype))
                for name, buffer in self.model.named_buffers()
            ),
        )
        schemas = self._all_gather_objects(local_schema)
        if any(schema != schemas[0] for schema in schemas[1:]):
            raise DomainError("DDP ranks supplied different root-model tensor schemas")
        with torch.no_grad():
            for parameter in self.model.parameters():
                torch.distributed.broadcast(parameter.data, src=0)
            for buffer in self.model.buffers():
                torch.distributed.broadcast(buffer.data, src=0)

    def _validate_precision(self) -> None:
        if self._device.type not in {"cpu", "cuda", "mps"}:
            raise DomainError("exact-resume training supports only CPU, CUDA, or MPS devices")
        if self.config.precision is PrecisionMode.FLOAT16 and self._device.type != "cuda":
            raise DomainError("float16 training is supported only on CUDA")
        if self.config.precision is PrecisionMode.BFLOAT16 and self._device.type not in {
            "cpu",
            "cuda",
        }:
            raise DomainError("bfloat16 autocast requires CPU or CUDA")
        if (
            self.config.precision is not PrecisionMode.FLOAT32
            and self._master_dtype != torch.float32
        ):
            raise DomainError("mixed precision requires float32 master parameters")

    def _make_grad_scaler(self) -> Any | None:
        if self.config.precision is not PrecisionMode.FLOAT16:
            return None
        # torch.cuda.amp.GradScaler remains available throughout the supported
        # torch>=2.4 range and also permits equation tests on slightly older CI.
        return torch.cuda.amp.GradScaler(enabled=True)

    def _active_phases(self, phase: TrainingPhase) -> tuple[TrainingPhase, ...]:
        if phase is TrainingPhase.JOINT:
            return (
                TrainingPhase.COMPOSITION,
                TrainingPhase.COVER,
                TrainingPhase.GEOMETRY,
            )
        return (phase,)

    def _configure_phase(self, phase: TrainingPhase) -> None:
        active = set(self._active_phases(phase))
        active_ids = {
            id(parameter)
            for active_phase in active
            for parameter in self._component_modules[active_phase].parameters()
        }
        for parameter in self.model.parameters():
            parameter.requires_grad_(
                self._eligible_parameters[id(parameter)] and id(parameter) in active_ids
            )
        self._wrapped_components = {}
        if self.config.distributed.enabled:
            torch.distributed.barrier()
            ddp_type = nn.parallel.DistributedDataParallel
            device_ids = (
                [self.config.distributed.local_rank] if self._device.type == "cuda" else None
            )
            for active_phase in self._active_phases(phase):
                self._wrapped_components[active_phase] = ddp_type(
                    self._component_modules[active_phase],
                    device_ids=device_ids,
                    output_device=self.config.distributed.local_rank
                    if self._device.type == "cuda"
                    else None,
                    broadcast_buffers=self.config.distributed.broadcast_buffers,
                    find_unused_parameters=(self.config.distributed.find_unused_parameters),
                )
            torch.distributed.barrier()
        self.model.train()

    def _autocast_context(self) -> Any:
        if self.config.precision is PrecisionMode.FLOAT32:
            return nullcontext()
        dtype = (
            torch.bfloat16 if self.config.precision is PrecisionMode.BFLOAT16 else torch.float16
        )
        return torch.autocast(device_type=self._device.type, dtype=dtype, enabled=True)

    def _audit_context(self) -> Any:
        if self._device.type in {"cpu", "cuda"}:
            return torch.autocast(device_type=self._device.type, enabled=False)
        return nullcontext()

    def _no_sync_context(self, synchronize: bool) -> Any:
        stack = ExitStack()
        if self.config.distributed.enabled and not synchronize:
            for wrapped in self._wrapped_components.values():
                stack.enter_context(wrapped.no_sync())
        return stack

    def _all_gather_objects(self, value: Any) -> list[Any]:
        if not self.config.distributed.enabled:
            return [value]
        gathered: list[Any] = [None for _ in range(self.world_size)]
        torch.distributed.all_gather_object(gathered, value)
        return gathered

    def _synchronized_errors(self, local_error: str | None) -> tuple[str, ...]:
        gathered = self._all_gather_objects(local_error)
        return tuple(
            f"rank {rank}: {message}"
            for rank, message in enumerate(gathered)
            if message is not None
        )

    def _run_audit(self, batch: Any, audit: AuditFunction | None) -> None:
        if audit is None:
            return
        error: str | None = None
        try:
            with self._audit_context():
                result = audit(batch, torch.float64)
            if not isinstance(result, CrystallographicAuditResult):
                raise TypeError("audit callback must return CrystallographicAuditResult")
            result.require_passed()
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
        errors = self._synchronized_errors(error)
        if errors:
            raise DomainError(f"crystallographic audit failed across ranks: {errors}")

    def _bind_objective(self, objective: ObjectiveFunction) -> None:
        contract: dict[str, Any] | None = None
        local_error: str | None = None
        try:
            contract = _objective_contract(objective)
            if self._objective_contract is not None and contract != self._objective_contract:
                raise DomainError(
                    "training objective semantics differ from the bound trainer contract"
                )
        except Exception as exc:
            local_error = f"{type(exc).__name__}: {exc}"
        errors = self._synchronized_errors(local_error)
        if errors:
            raise DomainError(f"objective contract failed across ranks: {errors}")
        assert contract is not None
        if self._objective_contract is None:
            contracts = self._all_gather_objects(contract)
            if any(candidate != contracts[0] for candidate in contracts[1:]):
                raise DomainError("DDP ranks supplied different training objective contracts")
            self._objective_contract = contract

    def _synchronize_window_length(self, length: int) -> None:
        lengths = self._all_gather_objects(length)
        if len(set(lengths)) != 1:
            raise DomainError(
                "all DDP ranks must execute the same number of microbatches per window; "
                f"observed={tuple(lengths)}"
            )

    def _global_structure_count(self, local_count: int) -> int:
        if not self.config.distributed.enabled:
            return local_count
        value = torch.tensor(local_count, dtype=torch.int64, device=self._device)
        torch.distributed.all_reduce(value, op=torch.distributed.ReduceOp.SUM)
        return int(value.item())

    def _global_metric_sums(self, local_sums: Mapping[str, float]) -> dict[str, float]:
        if not self.config.distributed.enabled:
            return dict(local_sums)
        names = tuple(sorted(local_sums))
        values = torch.tensor(
            [local_sums[name] for name in names],
            dtype=torch.float64,
            device=self._device,
        )
        torch.distributed.all_reduce(values, op=torch.distributed.ReduceOp.SUM)
        return dict(zip(names, (float(value) for value in values.cpu().tolist()), strict=True))

    def _scale_gradients(self, coefficient: float) -> None:
        with torch.no_grad():
            for parameter in self.model.parameters():
                gradient = parameter.grad
                if gradient is None:
                    continue
                if gradient.is_sparse:
                    coalesced = gradient.coalesce()
                    coalesced._values().mul_(coefficient)
                    parameter.grad = coalesced
                else:
                    gradient.mul_(coefficient)

    def _clip_gradients(self, preclip: GradientReport) -> tuple[GradientReport, bool]:
        maximum = self.config.max_gradient_norm
        clipped = maximum is not None and preclip.global_l2_norm > maximum
        if clipped and maximum is not None:
            self._scale_gradients(maximum / preclip.global_l2_norm)
        postclip = inspect_gradients(self.model)
        if not postclip.all_finite:
            raise TrainingNumericsError("gradient clipping produced non-finite gradients")
        return postclip, clipped

    def _learning_rates(self) -> tuple[float, ...]:
        rates: list[float] = []
        for group in self.optimizer.param_groups:
            rate = float(group["lr"])
            if not math.isfinite(rate) or rate < 0.0:
                raise TrainingNumericsError(
                    "optimizer learning rates must be finite and nonnegative"
                )
            rates.append(rate)
        return tuple(rates)

    def _snapshot_update_state(self) -> dict[str, Any] | None:
        if not self.config.transactional_optimizer_step:
            return None
        return {
            "model": _clone_state(self.model.state_dict()),
            "optimizer": _clone_state(self.optimizer.state_dict()),
            "scaler": _clone_state(self._scaler.state_dict())
            if self._scaler is not None
            else None,
            "scheduler": _clone_state(self.scheduler.state_dict())
            if self.scheduler is not None
            else None,
            "ema": _clone_state(self.ema.state_dict()) if self.ema is not None else None,
        }

    def _restore_update_state(self, snapshot: Mapping[str, Any]) -> tuple[str, ...]:
        """Best-effort restoration after a failed optimizer transaction.

        Every component is attempted independently.  Stopping at the first
        failure could leave a later component (for example EMA or scheduler)
        mutated while making the trainer look usable.  The caller performs a
        rank-wide consensus over the returned failures.
        """

        failures: list[str] = []

        def restore(label: str, action: Callable[[], None]) -> None:
            try:
                action()
            except Exception as exc:  # pragma: no cover - backend-specific corruption
                failures.append(f"{label}: {type(exc).__name__}: {exc}")

        restore(
            "model",
            lambda: self.model.load_state_dict(snapshot["model"], strict=True),
        )
        restore("optimizer", lambda: self.optimizer.load_state_dict(snapshot["optimizer"]))
        scaler = self._scaler
        if scaler is not None:
            restore("scaler", lambda: scaler.load_state_dict(snapshot["scaler"]))
        scheduler = self.scheduler
        if scheduler is not None:
            restore(
                "scheduler",
                lambda: scheduler.load_state_dict(snapshot["scheduler"]),
            )
        ema = self.ema
        if ema is not None:
            restore("ema", lambda: ema.load_state_dict(snapshot["ema"]))
        restore("zero_grad", lambda: self.optimizer.zero_grad(set_to_none=True))
        if not failures:
            restore("validation", self._validate_updated_state)
        return tuple(failures)

    def _validate_updated_state(self) -> None:
        current_devices = {parameter.device for parameter in self.model.parameters()}
        current_dtypes = {parameter.dtype for parameter in self.model.parameters()}
        if current_devices != {self._device}:
            raise TrainingNumericsError(
                "optimizer changed the frozen model master device contract"
            )
        if current_dtypes != {self._master_dtype}:
            raise TrainingNumericsError(
                "optimizer changed the frozen model master dtype contract"
            )
        if _state_has_nonfinite(self.model.state_dict()):
            raise TrainingNumericsError("optimizer produced non-finite model state")
        if _state_has_nonfinite(self.optimizer.state_dict()):
            raise TrainingNumericsError("optimizer produced non-finite optimizer state")
        if self._scaler is not None and _state_has_nonfinite(self._scaler.state_dict()):
            raise TrainingNumericsError("AMP scaler produced non-finite state")
        if self.scheduler is not None and _state_has_nonfinite(self.scheduler.state_dict()):
            raise TrainingNumericsError("scheduler produced non-finite state")
        if self.ema is not None and _state_has_nonfinite(self.ema.state_dict()):
            raise TrainingNumericsError("EMA produced non-finite state")
        self._learning_rates()

    def _commit_optimizer_step(self) -> None:
        snapshot = self._snapshot_update_state()
        error: str | None = None
        try:
            if self._scaler is None:
                self.optimizer.step()
            else:
                self._scaler.step(self.optimizer)
                self._scaler.update()
            if (
                self.scheduler is not None
                and self.current_phase is TrainingPhase.JOINT
                and self.config.scheduler_interval is SchedulerInterval.OPTIMIZER_STEP
            ):
                self.scheduler.step()
            self._validate_updated_state()
            if self.ema is not None:
                active_names = tuple(
                    name
                    for name, parameter in self.model.named_parameters()
                    if parameter.requires_grad
                )
                self.ema.update(self.model, parameter_names=active_names)
                self._validate_updated_state()
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
        errors = self._synchronized_errors(error)
        if errors:
            rollback_errors: tuple[str, ...]
            if snapshot is not None:
                rollback_errors = self._restore_update_state(snapshot)
            else:
                rollback_errors = (
                    "transactional_optimizer_step is disabled; no rollback snapshot exists",
                )
                try:
                    self.optimizer.zero_grad(set_to_none=True)
                except Exception as exc:  # pragma: no cover - backend-specific corruption
                    rollback_errors += (f"zero_grad: {type(exc).__name__}: {exc}",)
            collective_rollback_errors = self._synchronized_errors(
                None if not rollback_errors else "; ".join(rollback_errors)
            )
            # GradScaler's public state_dict omits its per-optimizer stage and
            # found-inf cache.  Once scaler.step/update has raised, even a
            # successful public-state rollback cannot prove the next unscale is
            # legal.  Fault every rank and require an atomic checkpoint restore.
            amp_transaction_unrecoverable = self._scaler is not None
            if collective_rollback_errors or amp_transaction_unrecoverable:
                self._faulted = True
            if amp_transaction_unrecoverable:
                recovery = (
                    "public model/optimizer state rollback attempted, but AMP's hidden "
                    "optimizer stage is not restorable; trainer faulted"
                )
            else:
                recovery = (
                    "transaction rolled back"
                    if snapshot is not None and not collective_rollback_errors
                    else f"rollback failures={collective_rollback_errors}"
                )
            raise TrainingNumericsError(
                f"optimizer transaction failed across ranks: {errors}; {recovery}"
            )

    def _backoff_amp_scaler(self) -> None:
        """Apply one identical public GradScaler backoff on every DDP rank."""

        scaler = self._scaler
        if scaler is None:
            raise TrainingNumericsError("AMP scaler backoff requested without a scaler")
        try:
            state = scaler.state_dict()
            old_scale = float(scaler.get_scale())
            backoff = float(state["backoff_factor"])
            new_scale = old_scale * backoff
            if not math.isfinite(new_scale) or new_scale <= 0.0 or not 0.0 < backoff < 1.0:
                raise ValueError("invalid AMP scale/backoff state")
            # Supplying the scale explicitly is essential: ranks that did not
            # overflow must still take the same backoff as a rank that did.
            scaler.update(new_scale=new_scale)
            reset_state = dict(scaler.state_dict())
            growth_tracker = reset_state.get("_growth_tracker")
            if isinstance(growth_tracker, bool) or not isinstance(growth_tracker, int):
                raise TypeError("AMP scaler state lacks an integer growth tracker")
            # Explicit ``new_scale`` does not reset this tracker, unlike the
            # ordinary overflow branch.  Reset it through the public state API
            # or the successful retry could immediately regrow the bad scale.
            reset_state["_growth_tracker"] = 0
            scaler.load_state_dict(reset_state)
            public_states = self._all_gather_objects(_contract_value(scaler.state_dict()))
            if any(state != public_states[0] for state in public_states[1:]):
                raise RuntimeError(
                    f"AMP public scaler states diverged across ranks: {public_states}"
                )
        except Exception as exc:
            self._faulted = True
            raise TrainingNumericsError(
                "AMP scaler could not complete a synchronized overflow backoff; "
                "restore the last atomic checkpoint"
            ) from exc

    def train_window(
        self,
        batches: Sequence[Any],
        objective: ObjectiveFunction,
        *,
        crystallographic_audit: AuditFunction | None = None,
    ) -> TrainStepReport:
        """Train one window, retrying FP16 overflow without dropping data.

        Every retry restores the rank-local pre-window RNG state.  The same
        materialized structures therefore receive the same stochastic CFM
        coupling while ``GradScaler`` alone retains its reduced scale.
        """

        rng_before = capture_global_rng_state() if self._scaler is not None else None
        for retry in range(self.config.max_amp_overflow_retries + 1):
            if retry and rng_before is not None:
                restore_global_rng_state(rng_before)
            try:
                return self._train_window_attempt(
                    batches,
                    objective,
                    crystallographic_audit=crystallographic_audit,
                )
            except _AMPGradientOverflowError as exc:
                if retry < self.config.max_amp_overflow_retries:
                    continue
                self._faulted = True
                raise TrainingNumericsError(
                    "FP16 gradients remained non-finite after the frozen overflow "
                    "retry budget; no structures were skipped and the trainer is "
                    "faulted. Restore a checkpoint with a lower initial scale or "
                    "use BF16/FP32."
                ) from exc
        raise AssertionError("AMP overflow retry loop did not terminate")

    def _train_window_attempt(
        self,
        batches: Sequence[Any],
        objective: ObjectiveFunction,
        *,
        crystallographic_audit: AuditFunction | None = None,
    ) -> TrainStepReport:
        """Train one complete accumulation window and commit exactly one step.

        A short window is mathematically exact and is intended for the final
        partial window of an epoch.  Checkpoints are allowed only after this
        method returns, never while an autograd graph is partially accumulated.
        """

        if self._faulted:
            raise TrainingNumericsError(
                "trainer is faulted; restore a known-good checkpoint before continuing"
            )
        if self.cursor.epoch >= self.schedule.total_epochs:
            raise DomainError("the staged training schedule is complete")
        if not isinstance(batches, Sequence):
            raise TypeError("batches must be a finite sequence")
        if not 1 <= len(batches) <= self.config.accumulation_steps:
            raise DomainError(
                "an accumulation window must contain between one and "
                "accumulation_steps microbatches"
            )
        self._synchronize_window_length(len(batches))
        self._bind_objective(objective)
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)
        local_count = 0
        sampling_normalizers: list[float | None] = []
        local_metric_sums: dict[str, float] = {"objective/total": 0.0}
        phase = self.current_phase
        scaler_needs_update = False

        try:
            for index, batch in enumerate(batches):
                self._run_audit(batch, crystallographic_audit)
                losses: ComponentLosses | None = None
                error: str | None = None
                synchronize = index == len(batches) - 1
                try:
                    with self._no_sync_context(synchronize), self._autocast_context():
                        candidate = objective(self.models, batch, phase)
                        if not isinstance(candidate, ComponentLosses):
                            raise TypeError("objective callback must return ComponentLosses")
                        mean_objective = candidate.objective(
                            phase, self.config.joint_loss_weights
                        )
                        if not bool(torch.isfinite(mean_objective).all().item()):
                            raise TrainingNumericsError(
                                "per-structure training objective is non-finite"
                            )
                        weighted_objective = mean_objective * candidate.batch_size
                        normalizer = getattr(batch, "optimizer_normalization_count", None)
                        if normalizer is not None and (
                            isinstance(normalizer, bool)
                            or not isinstance(normalizer, (int, float))
                            or not math.isfinite(normalizer)
                            or normalizer <= 0.0
                        ):
                            raise DomainError(
                                "batch optimizer normalization count must be "
                                "finite and positive"
                            )
                        sampling_normalizers.append(
                            None if normalizer is None else float(normalizer)
                        )
                    losses = candidate
                except Exception as exc:
                    error = f"{type(exc).__name__}: {exc}"
                errors = self._synchronized_errors(error)
                if errors:
                    raise TrainingNumericsError(
                        f"objective construction failed across ranks: {errors}"
                    )
                assert losses is not None
                if self._scaler is None:
                    weighted_objective.backward()
                else:
                    self._scaler.scale(weighted_objective).backward()
                local_count += losses.batch_size
                local_metric_sums["objective/total"] += (
                    float(mean_objective.detach().item()) * losses.batch_size
                )
                for name, value in losses.present_values().items():
                    local_metric_sums[name] = local_metric_sums.get(name, 0.0) + (
                        float(value.detach().item()) * losses.batch_size
                    )

            global_count = self._global_structure_count(local_count)
            if global_count < 1:
                raise TrainingNumericsError("global accumulation window is empty")
            rank_normalizers = self._all_gather_objects(tuple(sampling_normalizers))
            normalizers = {value for values in rank_normalizers for value in values}
            if len(normalizers) != 1:
                raise DomainError(
                    "one optimizer window cannot mix different sampling normalizers "
                    "or mix sampler batches with uncontracted diagnostic batches"
                )
            population_normalizer = next(iter(normalizers))
            if population_normalizer is None:
                gradient_denominator = float(global_count)
                gradient_policy = "explicit-window-record-mean"
            else:
                gradient_denominator = (
                    self.world_size * self.config.accumulation_steps * population_normalizer
                )
                gradient_policy = "sampler-population-fixed-nominal-window.v1"
            if not math.isfinite(gradient_denominator) or gradient_denominator <= 0.0:
                raise DomainError("computed gradient denominator must be finite and positive")
            normalization = self.world_size / gradient_denominator
            if not math.isfinite(normalization) or normalization <= 0.0:
                raise DomainError("computed gradient normalization must be finite and positive")
            if self._scaler is not None:
                self._scaler.unscale_(self.optimizer)
                scaler_needs_update = True
            self._scale_gradients(normalization)
            preclip = inspect_gradients(self.model)
            error = None
            issue = None
            if preclip.tensors_with_grad == 0:
                error = "backward produced no model gradients"
                issue = "missing"
            elif not preclip.all_finite:
                error = f"non-finite gradients: {preclip.nonfinite_parameters}"
                issue = "overflow"
            gradient_issues = self._all_gather_objects((issue, error))
            errors = tuple(
                f"rank {rank}: {message}"
                for rank, (_, message) in enumerate(gradient_issues)
                if message is not None
            )
            if errors:
                kinds = {kind for kind, _ in gradient_issues if kind is not None}
                if kinds == {"overflow"} and self._scaler is not None:
                    raise _AMPGradientOverflowError(
                        f"scaled gradient overflow across ranks: {errors}"
                    )
                raise TrainingNumericsError(f"gradient audit failed across ranks: {errors}")
            postclip, clipped = self._clip_gradients(preclip)
            # From this point onward the optimizer transaction owns the scaler
            # stage and its recovery.  Before this point, a skipped update must
            # still call GradScaler.update() exactly once to leave UNSCALED.
            scaler_needs_update = False
            self._commit_optimizer_step()
        except Exception:
            try:
                if scaler_needs_update and self._scaler is not None:
                    self._backoff_amp_scaler()
            finally:
                self.optimizer.zero_grad(set_to_none=True)
            raise

        self.optimizer.zero_grad(set_to_none=True)
        global_metric_sums = self._global_metric_sums(local_metric_sums)
        metrics = {name: value / global_count for name, value in global_metric_sums.items()}
        self.cursor = TrainerCursor(
            epoch=self.cursor.epoch,
            microbatch=self.cursor.microbatch + len(batches),
            optimizer_step=self.cursor.optimizer_step + 1,
        )
        return TrainStepReport(
            phase=phase,
            optimizer_step=self.cursor.optimizer_step,
            microbatches=len(batches),
            global_structures=global_count,
            metrics=metrics,
            gradients_before_clip=preclip,
            gradients_after_clip=postclip,
            clipped=clipped,
            learning_rates=self._learning_rates(),
            gradient_normalization_count=gradient_denominator,
            gradient_normalization_policy=gradient_policy,
        )

    def _step_epoch_scheduler(self) -> None:
        if (
            self.scheduler is None
            or self.current_phase is not TrainingPhase.JOINT
            or self.config.scheduler_interval is not SchedulerInterval.EPOCH
        ):
            return
        optimizer_before = _clone_state(self.optimizer.state_dict())
        scheduler_before = _clone_state(self.scheduler.state_dict())
        error: str | None = None
        try:
            self.scheduler.step()
            self._validate_updated_state()
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
        errors = self._synchronized_errors(error)
        if errors:
            rollback_errors: list[str] = []
            for label, action in (
                (
                    "optimizer",
                    lambda: self.optimizer.load_state_dict(optimizer_before),
                ),
                (
                    "scheduler",
                    lambda: self.scheduler.load_state_dict(scheduler_before),
                ),
                ("zero_grad", lambda: self.optimizer.zero_grad(set_to_none=True)),
            ):
                try:
                    action()
                except Exception as exc:  # pragma: no cover - backend-specific corruption
                    rollback_errors.append(f"{label}: {type(exc).__name__}: {exc}")
            if not rollback_errors:
                try:
                    self._validate_updated_state()
                except Exception as exc:
                    rollback_errors.append(f"validation: {type(exc).__name__}: {exc}")
            collective_rollback_errors = self._synchronized_errors(
                None if not rollback_errors else "; ".join(rollback_errors)
            )
            if collective_rollback_errors:
                self._faulted = True
            recovery = (
                "rolled back"
                if not collective_rollback_errors
                else f"rollback failures={collective_rollback_errors}"
            )
            raise TrainingNumericsError(
                f"epoch scheduler transaction failed across ranks: {errors}; {recovery}"
            )

    def train_epoch(
        self,
        batches: Iterable[Any],
        objective: ObjectiveFunction,
        *,
        crystallographic_audit: AuditFunction | None = None,
    ) -> EpochReport:
        """Consume an epoch, including its exact final partial accumulation."""

        if self.cursor.epoch >= self.schedule.total_epochs:
            raise DomainError("the staged training schedule is complete")
        stateful_batches = hasattr(batches, "state_dict")
        if self.config.distributed.enabled:
            stateful_contracts = self._all_gather_objects(stateful_batches)
            if not all(stateful_contracts):
                raise DomainError(
                    "distributed train_epoch requires a stateful sampler contract on every rank"
                )
        if stateful_batches:
            sampler_state = batches.state_dict()
            if not isinstance(sampler_state, Mapping):
                raise TypeError("stateful epoch iterable must return a mapping state")
            self._validate_sampler_state_collectively(sampler_state)
        epoch = self.cursor.epoch
        phase = self.current_phase
        phase_epoch = self.phase_epoch
        iterator = iter(batches)
        reports: list[TrainStepReport] = []
        while True:
            window = list(islice(iterator, self.config.accumulation_steps))
            self._synchronize_window_length(len(window))
            if not window:
                break
            reports.append(
                self.train_window(
                    window,
                    objective,
                    crystallographic_audit=crystallographic_audit,
                )
            )
        if not reports:
            raise DomainError("an empty training epoch is not defined")
        self._step_epoch_scheduler()
        global_structures = sum(report.global_structures for report in reports)
        metric_names = set().union(*(report.metrics for report in reports))
        metrics = {
            name: sum(
                report.metrics.get(name, 0.0) * report.global_structures for report in reports
            )
            / global_structures
            for name in metric_names
        }
        self.cursor = TrainerCursor(
            epoch=epoch + 1,
            microbatch=0,
            optimizer_step=self.cursor.optimizer_step,
        )
        if self.cursor.epoch < self.schedule.total_epochs:
            next_phase = self.current_phase
            if next_phase is phase and hasattr(batches, "advance_epoch"):
                batches.advance_epoch()
                advanced = batches.state_dict()
                if not isinstance(advanced, Mapping):
                    raise TypeError(
                        "advanced stateful epoch iterable must return a mapping state"
                    )
                self._validate_sampler_state_collectively(advanced)
            if next_phase is not phase:
                self._configure_phase(next_phase)
        return EpochReport(
            phase=phase,
            epoch=epoch,
            phase_epoch=phase_epoch,
            optimizer_steps=len(reports),
            global_structures=global_structures,
            metrics=metrics,
            learning_rates=self._learning_rates(),
        )

    def _distributed_contract(self) -> dict[str, Any]:
        return {
            "world_size": self.world_size,
            "backend": (
                str(torch.distributed.get_backend())
                if self.config.distributed.enabled
                else "none"
            ),
            "device_type": self._device.type,
        }

    def _require_checkpoint_boundary(self) -> None:
        pending = [
            name
            for name, parameter in self.model.named_parameters()
            if parameter.grad is not None
        ]
        if pending:
            raise CheckpointError(
                "checkpointing is permitted only at an optimizer boundary; "
                f"parameters with pending gradients={tuple(pending)}"
            )

    def _expected_sampler_position(
        self, cursor: TrainerCursor | None = None
    ) -> tuple[int, int] | None:
        selected = self.cursor if cursor is None else cursor
        if selected.epoch >= self.schedule.total_epochs:
            return None
        return self.schedule.phase_epoch_at(selected.epoch), selected.microbatch

    def _validate_sampler_state_position(
        self,
        state: Mapping[str, Any],
        *,
        cursor: TrainerCursor | None = None,
        expected_rank: int | None = None,
        require_position: bool = True,
    ) -> None:
        fingerprint = state.get("sampler_fingerprint")
        if (
            not isinstance(fingerprint, str)
            or len(fingerprint) != 64
            or any(character not in "0123456789abcdef" for character in fingerprint)
        ):
            raise CheckpointError(
                "exact resume requires a lowercase SHA-256 sampler_fingerprint"
            )
        common_policy = state.get("common_policy_fingerprint")
        if (
            not isinstance(common_policy, str)
            or len(common_policy) != 64
            or any(character not in "0123456789abcdef" for character in common_policy)
        ):
            raise CheckpointError(
                "exact resume requires a lowercase SHA-256 common_policy_fingerprint"
            )
        partition_rank = state.get("partition_rank")
        sampler_world_size = state.get("world_size")
        required_rank = self.rank if expected_rank is None else expected_rank
        if (
            isinstance(partition_rank, bool)
            or not isinstance(partition_rank, int)
            or partition_rank != required_rank
        ):
            raise CheckpointError("sampler partition rank differs from the owning process rank")
        if (
            isinstance(sampler_world_size, bool)
            or not isinstance(sampler_world_size, int)
            or sampler_world_size != self.world_size
        ):
            raise CheckpointError("sampler world size differs from the distributed trainer")
        if not require_position:
            return
        expected = self._expected_sampler_position(cursor)
        if expected is None:
            raise CheckpointError(
                "a completed training schedule must not retain a live sampler cursor"
            )
        epoch = state.get("epoch")
        position = state.get("cursor")
        if (
            isinstance(epoch, bool)
            or not isinstance(epoch, int)
            or isinstance(position, bool)
            or not isinstance(position, int)
        ):
            raise CheckpointError(
                "exact resume requires sampler state with integer epoch and cursor"
            )
        if (epoch, position) != expected:
            raise CheckpointError(
                "sampler epoch/cursor disagree with the staged trainer cursor; "
                f"sampler={(epoch, position)}, expected={expected}"
            )

    def _validate_sampler_state_collectively(
        self,
        state: Mapping[str, Any],
        *,
        cursor: TrainerCursor | None = None,
    ) -> None:
        """Fail every rank if one partition or the common policy is wrong."""

        local_error: str | None = None
        try:
            self._validate_sampler_state_position(state, cursor=cursor)
        except Exception as exc:
            local_error = f"{type(exc).__name__}: {exc}"
        errors = self._synchronized_errors(local_error)
        if errors:
            raise CheckpointError(
                f"distributed sampler preflight failed across ranks: {errors}"
            )
        policies = self._all_gather_objects(state["common_policy_fingerprint"])
        if any(policy != policies[0] for policy in policies[1:]):
            raise CheckpointError("distributed ranks use different common sampler policies")

    def _validate_rank_sampler_policies(
        self,
        ranks: Mapping[int, Mapping[str, Any]],
        *,
        cursor: TrainerCursor,
    ) -> None:
        policies: set[str] = set()
        for rank, candidate in ranks.items():
            saved_sampler = candidate["sampler"]
            if saved_sampler is None:
                if cursor.epoch < self.schedule.total_epochs:
                    raise CheckpointError(
                        "unfinished checkpoint is missing a rank-local sampler policy"
                    )
                continue
            if not isinstance(saved_sampler, Mapping):
                raise CheckpointError("saved sampler state must be a mapping")
            self._validate_sampler_state_position(
                saved_sampler,
                cursor=cursor,
                expected_rank=rank,
            )
            policies.add(saved_sampler["common_policy_fingerprint"])
        if len(policies) > 1:
            raise CheckpointError(
                "rank-local checkpoint states disagree on the common sampler policy"
            )

    def _local_rank_state(self, sampler: Any | None) -> dict[str, Any]:
        sampler_state: Mapping[str, Any] | None = None
        if sampler is not None:
            if not hasattr(sampler, "state_dict") or not hasattr(sampler, "load_state_dict"):
                raise TypeError("sampler must implement state_dict/load_state_dict")
            candidate = sampler.state_dict()
            if not isinstance(candidate, Mapping):
                raise TypeError("sampler.state_dict() must return a mapping")
            sampler_state = dict(candidate)
            self._validate_sampler_state_position(sampler_state)
            objective_dataset = _objective_dataset_fingerprint(self._objective_contract)
            if (
                objective_dataset is not None
                and sampler_state.get("dataset_fingerprint") != objective_dataset
            ):
                raise CheckpointError(
                    "sampler dataset differs from the bound objective population"
                )
        elif self.cursor.epoch < self.schedule.total_epochs:
            raise CheckpointError(
                "every unfinished exact-resume checkpoint requires an aligned "
                "sampler policy and cursor"
            )
        return {
            "rank": self.rank,
            "global_rng": capture_global_rng_state(),
            "training_context": (
                self.training_context.state_dict()
                if self.training_context is not None
                else None
            ),
            "sampler": sampler_state,
        }

    def state_dict(self, *, sampler: Any | None = None) -> dict[str, Any]:
        """Collect a complete exact-resume state on every rank.

        In DDP this is collective because every rank owns a distinct RNG and
        sampler stream.  A changed world size is intentionally unsupported:
        repartitioning changes both the stochastic trajectory and the gradient
        reduction order, so it cannot be advertised as exact continuation.
        """

        local_state: dict[str, Any] | None = None
        local_error: str | None = None
        try:
            self._require_checkpoint_boundary()
            if self.cursor.optimizer_step > 0 and not _objective_is_checkpointable(
                self._objective_contract
            ):
                raise CheckpointError(
                    "trained state needs an explicit checkpointable objective contract"
                )
            local_state = self._local_rank_state(sampler)
            # ``all_gather_object`` pickles rank states internally.  Probing
            # serialization before the collective prevents one rank from
            # failing in pickle while its peers block inside the gather.
            pickle.dumps(local_state, protocol=pickle.HIGHEST_PROTOCOL)
        except Exception as exc:
            local_error = f"{type(exc).__name__}: {exc}"
        errors = self._synchronized_errors(local_error)
        if errors:
            raise CheckpointError(f"checkpoint preflight failed across ranks: {errors}")
        assert local_state is not None
        rank_states = self._all_gather_objects(local_state)
        gathered_ranks: dict[int, Mapping[str, Any]] = {}
        for candidate in rank_states:
            if not isinstance(candidate, Mapping):
                raise CheckpointError("a gathered rank-local state is malformed")
            rank = candidate.get("rank")
            if isinstance(rank, bool) or not isinstance(rank, int) or rank in gathered_ranks:
                raise CheckpointError("gathered rank-local identifiers are invalid")
            gathered_ranks[rank] = candidate
        if set(gathered_ranks) != set(range(self.world_size)):
            raise CheckpointError("gathered rank-local identifiers are incomplete")
        self._validate_rank_sampler_policies(gathered_ranks, cursor=self.cursor)
        return {
            "schema": _TRAINER_STATE_SCHEMA,
            "version": _TRAINER_STATE_VERSION,
            "trainer_config": self.config.to_dict(),
            "trainer_config_fingerprint": self.config.fingerprint,
            "schedule": self.schedule.to_dict(),
            "schedule_fingerprint": self.schedule.fingerprint,
            "distributed_contract": self._distributed_contract(),
            "cursor": self.cursor.to_dict(),
            "phase": self.current_phase.value,
            "scaler": self._scaler.state_dict() if self._scaler is not None else None,
            "scheduler_contract": self._scheduler_contract,
            "objective_contract": _clone_state(self._objective_contract),
            "eligible_parameter_names": self._eligible_parameter_names,
            "scheduler": self.scheduler.state_dict() if self.scheduler is not None else None,
            "ema": self.ema.state_dict() if self.ema is not None else None,
            "rank_states": rank_states,
        }

    def save_checkpoint(
        self,
        path: str | Path,
        *,
        scientific_config: CrystalAtlasConfig,
        scientific_manifest: ScientificManifest,
        provenance: ProvenanceRecord,
        sampler: Any | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> CheckpointInfo | None:
        """Atomically save model, optimizer, and all rank-local resume streams.

        Every DDP rank must call this method.  Rank zero writes and returns the
        receipt; other ranks return ``None`` after the write result is
        broadcast, so no process can race ahead after a failed publication.
        """

        trainer_state = self.state_dict(sampler=sampler)
        info: CheckpointInfo | None = None
        root_exception: Exception | None = None
        if self.rank == 0:
            try:
                merged_metadata = dict(metadata or {})
                merged_metadata["trainer_state_schema"] = _TRAINER_STATE_SCHEMA
                info = save_checkpoint(
                    path,
                    model=self.model,
                    optimizer=self.optimizer,
                    config=scientific_config,
                    scientific_manifest=scientific_manifest,
                    provenance=provenance,
                    step=self.cursor.optimizer_step,
                    epoch=self.cursor.epoch,
                    training_context=None,
                    metadata=merged_metadata,
                    auxiliary_state={_TRAINER_STATE_SCHEMA: trainer_state},
                )
            except Exception as exc:  # broadcast failure before re-raising on root
                root_exception = exc
        if self.config.distributed.enabled:
            message = [
                None
                if root_exception is None
                else f"{type(root_exception).__name__}: {root_exception}"
            ]
            torch.distributed.broadcast_object_list(message, src=0)
            if message[0] is not None:
                if self.rank == 0 and root_exception is not None:
                    raise root_exception
                raise CheckpointError(f"rank-zero checkpoint save failed: {message[0]}")
        elif root_exception is not None:
            raise root_exception
        return info

    def _validate_resume_state(
        self,
        state: Mapping[str, Any],
        *,
        sampler: Any | None,
    ) -> tuple[
        TrainerCursor,
        Mapping[str, Any],
        dict[str, Any] | None,
    ]:
        required = {
            "schema",
            "version",
            "trainer_config",
            "trainer_config_fingerprint",
            "schedule",
            "schedule_fingerprint",
            "distributed_contract",
            "cursor",
            "phase",
            "scaler",
            "scheduler_contract",
            "objective_contract",
            "eligible_parameter_names",
            "scheduler",
            "ema",
            "rank_states",
        }
        if set(state) != required:
            raise CheckpointError("trainer state has missing or unexpected fields")
        if (
            state["schema"] != _TRAINER_STATE_SCHEMA
            or state["version"] != _TRAINER_STATE_VERSION
        ):
            raise CheckpointError("unsupported trainer-state schema")
        if state["trainer_config_fingerprint"] != self.config.fingerprint:
            raise CheckpointError("trainer configuration differs from the checkpoint")
        if state["trainer_config"] != self.config.to_dict():
            raise CheckpointError("stored trainer configuration is internally inconsistent")
        if state["schedule_fingerprint"] != self.schedule.fingerprint:
            raise CheckpointError("staged schedule differs from the checkpoint")
        if state["schedule"] != self.schedule.to_dict():
            raise CheckpointError("stored staged schedule is internally inconsistent")
        if state["distributed_contract"] != self._distributed_contract():
            saved = state["distributed_contract"]
            current = self._distributed_contract()
            raise CheckpointError(
                "exact resume requires the identical DDP world size, backend, and device "
                f"type; saved={saved}, current={current}. Changed-world-size resume is a "
                "new stochastic run, not deterministic continuation."
            )
        cursor_payload = state["cursor"]
        if not isinstance(cursor_payload, Mapping) or set(cursor_payload) != {
            "epoch",
            "microbatch",
            "optimizer_step",
        }:
            raise CheckpointError("trainer cursor is malformed")
        try:
            cursor = TrainerCursor(**dict(cursor_payload))
        except (TypeError, DomainError) as exc:
            raise CheckpointError("trainer cursor is invalid") from exc
        if cursor.epoch > self.schedule.total_epochs:
            raise CheckpointError("trainer cursor lies beyond the schedule")
        expected_phase = (
            self.schedule.phase_at(cursor.epoch).value
            if cursor.epoch < self.schedule.total_epochs
            else self.schedule.phase_at(self.schedule.total_epochs - 1).value
        )
        if state["phase"] != expected_phase:
            raise CheckpointError("saved phase disagrees with the staged schedule cursor")
        if (state["scaler"] is None) != (self._scaler is None):
            raise CheckpointError("AMP scaler presence differs from the checkpoint")
        if (state["scheduler"] is None) != (self.scheduler is None):
            raise CheckpointError("scheduler presence differs from the checkpoint")
        if state["scheduler_contract"] != self._scheduler_contract:
            raise CheckpointError(
                "scheduler class or initial semantic configuration differs from the checkpoint"
            )
        if tuple(state["eligible_parameter_names"]) != self._eligible_parameter_names:
            raise CheckpointError("initial trainability mask differs from the checkpoint")
        objective_contract = _validate_objective_contract(state["objective_contract"])
        if cursor.optimizer_step > 0 and not _objective_is_checkpointable(objective_contract):
            raise CheckpointError(
                "a trained checkpoint lacks an explicit checkpointable objective contract"
            )
        if (
            self._objective_contract is not None
            and objective_contract != self._objective_contract
        ):
            raise CheckpointError("bound training objective differs from the checkpoint")
        if (state["ema"] is None) != (self.ema is None):
            raise CheckpointError("EMA presence differs from the checkpoint")
        objective_dataset = _objective_dataset_fingerprint(objective_contract)
        rank_states = state["rank_states"]
        if not isinstance(rank_states, (tuple, list)) or len(rank_states) != self.world_size:
            raise CheckpointError(
                "rank-local resume state count differs from the current world size"
            )
        ranks: dict[int, Mapping[str, Any]] = {}
        for candidate in rank_states:
            if not isinstance(candidate, Mapping) or set(candidate) != {
                "rank",
                "global_rng",
                "training_context",
                "sampler",
            }:
                raise CheckpointError("a rank-local resume state is malformed")
            rank = candidate["rank"]
            if isinstance(rank, bool) or not isinstance(rank, int) or rank in ranks:
                raise CheckpointError("rank-local resume identifiers are invalid")
            ranks[rank] = candidate
        if set(ranks) != set(range(self.world_size)):
            raise CheckpointError("rank-local resume identifiers are incomplete")
        local = ranks[self.rank]
        self._validate_rank_sampler_policies(ranks, cursor=cursor)
        for candidate in ranks.values():
            saved_sampler = candidate["sampler"]
            if saved_sampler is not None:
                if (
                    objective_dataset is not None
                    and saved_sampler.get("dataset_fingerprint") != objective_dataset
                ):
                    raise CheckpointError(
                        "saved sampler dataset differs from the objective population"
                    )
        if (local["training_context"] is None) != (self.training_context is None):
            raise CheckpointError("TrainingContext presence differs from the checkpoint")
        if (local["sampler"] is None) != (sampler is None):
            raise CheckpointError("sampler presence differs from the checkpoint")
        if sampler is not None:
            if not hasattr(sampler, "state_dict") or not hasattr(sampler, "load_state_dict"):
                raise TypeError("sampler must implement state_dict/load_state_dict")
            current_sampler = sampler.state_dict()
            if not isinstance(current_sampler, Mapping):
                raise TypeError("sampler.state_dict() must return a mapping")
            current_fingerprint = current_sampler.get("sampler_fingerprint")
            saved_fingerprint = local["sampler"].get("sampler_fingerprint")
            if current_fingerprint != saved_fingerprint:
                raise CheckpointError("sampler policy fingerprint differs from the checkpoint")
            self._validate_sampler_state_position(
                current_sampler,
                cursor=cursor,
                require_position=False,
            )
            if current_sampler.get("common_policy_fingerprint") != local["sampler"].get(
                "common_policy_fingerprint"
            ):
                raise CheckpointError(
                    "current common sampler policy differs from the checkpoint"
                )
            if (
                objective_dataset is not None
                and current_sampler.get("dataset_fingerprint") != objective_dataset
            ):
                raise CheckpointError(
                    "current sampler dataset differs from the objective population"
                )
        return cursor, local, objective_contract

    def _restore_checkpoint_snapshot(
        self,
        before: Mapping[str, Any],
        *,
        sampler: Any | None,
    ) -> tuple[str, ...]:
        """Best-effort restoration of every mutable trainer component."""

        failures = list(self._restore_update_state(before))

        def restore(label: str, action: Callable[[], None]) -> None:
            try:
                action()
            except Exception as exc:  # pragma: no cover - backend-specific corruption
                failures.append(f"{label}: {type(exc).__name__}: {exc}")

        training_context = self.training_context
        if training_context is not None:
            restore(
                "training_context",
                lambda: training_context.load_state_dict(before["context"]),
            )
        if sampler is not None:
            restore("sampler", lambda: sampler.load_state_dict(before["sampler"]))

        def restore_contract_and_cursor() -> None:
            self._objective_contract = _clone_state(before["objective_contract"])
            self.cursor = before["cursor"]

        restore("trainer_contract", restore_contract_and_cursor)
        restore(
            "global_rng",
            lambda: restore_global_rng_state(
                before["rng"],
                strict_cuda=False,
                strict_mps=False,
            ),
        )
        # ``requires_grad`` flags and factor-wise DDP wrappers are not stored
        # in a module state_dict, so reconstruct them from the restored cursor.
        restore("phase_reconstruction", lambda: self._configure_phase(self.current_phase))
        return tuple(failures)

    def load_checkpoint(
        self,
        path: str | Path,
        *,
        scientific_manifest: ScientificManifest,
        scientific_config: CrystalAtlasConfig,
        sampler: Any | None = None,
        expected_sha256: str | None = None,
        map_location: Any = "cpu",
    ) -> LoadedCheckpoint:
        """Transactionally restore a checkpoint at the identical DDP contract."""

        before: dict[str, Any] | None = None
        snapshot_error: str | None = None
        try:
            before = {
                "model": _clone_state(self.model.state_dict()),
                "optimizer": _clone_state(self.optimizer.state_dict()),
                "scaler": _clone_state(self._scaler.state_dict())
                if self._scaler is not None
                else None,
                "scheduler": _clone_state(self.scheduler.state_dict())
                if self.scheduler is not None
                else None,
                "ema": _clone_state(self.ema.state_dict()) if self.ema is not None else None,
                "objective_contract": _clone_state(self._objective_contract),
                "cursor": self.cursor,
                "rng": capture_global_rng_state(),
                "context": _clone_state(self.training_context.state_dict())
                if self.training_context is not None
                else None,
                "sampler": _clone_state(sampler.state_dict()) if sampler is not None else None,
                "faulted": self._faulted,
            }
        except Exception as exc:
            snapshot_error = f"{type(exc).__name__}: {exc}"
        snapshot_errors = self._synchronized_errors(snapshot_error)
        if snapshot_errors:
            raise CheckpointError(
                "trainer restore preflight failed before mutation across ranks: "
                f"{snapshot_errors}"
            )
        assert before is not None
        loaded: LoadedCheckpoint | None = None
        state: Mapping[str, Any] | None = None
        cursor: TrainerCursor | None = None
        local: Mapping[str, Any] | None = None
        objective_contract: dict[str, Any] | None = None
        local_error: Exception | None = None
        try:
            loaded = load_checkpoint(
                path,
                model=self.model,
                expected_scientific_manifest=scientific_manifest,
                optimizer=self.optimizer,
                expected_config=scientific_config,
                training_context=None,
                map_location=map_location,
                strict_model=True,
                restore_rng=False,
                expected_sha256=expected_sha256,
                # This outer method owns the rank-wide transaction and already
                # holds a complete rollback snapshot.
                transactional=False,
            )
            if set(loaded.auxiliary_state) != {_TRAINER_STATE_SCHEMA}:
                raise CheckpointError("checkpoint has no unique scalable-trainer state")
            state = loaded.auxiliary_state[_TRAINER_STATE_SCHEMA]
            if not isinstance(state, Mapping):
                raise CheckpointError("scalable-trainer state must be a mapping")
            cursor, local, objective_contract = self._validate_resume_state(
                state,
                sampler=sampler,
            )
            if loaded.step != cursor.optimizer_step or loaded.epoch != cursor.epoch:
                raise CheckpointError(
                    "base checkpoint step/epoch disagree with the trainer cursor"
                )
            if (
                bool(torch.are_deterministic_algorithms_enabled())
                != loaded.deterministic_algorithms
            ):
                raise CheckpointError(
                    "current PyTorch deterministic-algorithm mode differs from checkpoint"
                )
            # Apply every rank-local state before any DDP reconstruction.  All
            # ranks then vote on success, preventing one rank from committing
            # while another silently rolls back.
            if self._scaler is not None:
                self._scaler.load_state_dict(state["scaler"])
            if self.scheduler is not None:
                self.scheduler.load_state_dict(state["scheduler"])
            if self.ema is not None:
                self.ema.load_state_dict(state["ema"])
            if sampler is not None:
                sampler.load_state_dict(local["sampler"])
            if self.training_context is not None:
                self.training_context.load_state_dict(local["training_context"])
            self._objective_contract = objective_contract
            self.cursor = cursor
            restore_global_rng_state(local["global_rng"], strict_cuda=True)
            self._validate_updated_state()
        except Exception as exc:
            local_error = exc

        errors = self._synchronized_errors(
            None if local_error is None else f"{type(local_error).__name__}: {local_error}"
        )
        if errors:
            rollback_errors = self._restore_checkpoint_snapshot(
                before,
                sampler=sampler,
            )
            collective_rollback_errors = self._synchronized_errors(
                None if not rollback_errors else "; ".join(rollback_errors)
            )
            if collective_rollback_errors:
                self._faulted = True
            else:
                self._faulted = bool(before["faulted"])
            detail = (
                f"; rollback failures={collective_rollback_errors}"
                if collective_rollback_errors
                else "; transaction rolled back"
            )
            raise CheckpointError(
                f"trainer checkpoint restore failed across ranks: {errors}{detail}"
            ) from local_error

        assert loaded is not None and state is not None and cursor is not None
        assert local is not None
        # Every rank has now validated and applied all noncollective state.
        # Rebuilding factor-wise DDP wrappers is therefore a collective commit,
        # but it remains inside the transaction.
        reconstruction_error: Exception | None = None
        try:
            self._configure_phase(self.current_phase)
            self.optimizer.zero_grad(set_to_none=True)
        except Exception as exc:
            reconstruction_error = exc
        reconstruction_errors = self._synchronized_errors(
            None
            if reconstruction_error is None
            else f"{type(reconstruction_error).__name__}: {reconstruction_error}"
        )
        if reconstruction_errors:
            rollback_errors = self._restore_checkpoint_snapshot(
                before,
                sampler=sampler,
            )
            collective_rollback_errors = self._synchronized_errors(
                None if not rollback_errors else "; ".join(rollback_errors)
            )
            if collective_rollback_errors:
                self._faulted = True
            else:
                self._faulted = bool(before["faulted"])
            detail = (
                f"; rollback failures={collective_rollback_errors}"
                if collective_rollback_errors
                else "; transaction rolled back"
            )
            raise CheckpointError(
                "trainer phase reconstruction failed across ranks: "
                f"{reconstruction_errors}{detail}"
            ) from reconstruction_error
        self._faulted = False
        return replace(loaded, rng_restored=True)
