"""Training numerics and reproducible, atomic checkpoints.

This module deliberately keeps three statistically different objectives
separate.  Composition and cover terms are exact finite-action negative log
likelihoods.  Conditional flow matching (CFM) is a velocity-regression
objective; neither it nor their weighted sum is reported as a physical data
negative log likelihood.

PyTorch remains optional for the package as a whole.  Importing this module
requires the ``ml`` extra, just like the neural and flow modules.
"""

from __future__ import annotations

import copy
import json
import math
import os
import random
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from types import MappingProxyType
from typing import Any

import numpy as np

from ._optional import require_torch
from .catalogue import CertifiedCatalogue
from .certification import CrystallographyCertification
from .config import CrystalAtlasConfig
from .errors import CrystalAtlasError, DomainError, NumericalError
from .provenance import ProvenanceRecord, sha256_file
from .runtime import seed_everything

torch = require_torch()
Tensor = torch.Tensor

__all__ = [
    "CheckpointError",
    "CheckpointInfo",
    "CompositeLossTerms",
    "GradientReport",
    "LoadedCheckpoint",
    "LossWeights",
    "OptimizerStepReport",
    "ScientificManifest",
    "TrainingContext",
    "TrainingIdentity",
    "TrainingNumericsError",
    "capture_global_rng_state",
    "compose_training_loss",
    "exact_finite_action_nll",
    "guarded_optimizer_step",
    "initialize_training",
    "inspect_gradients",
    "load_checkpoint",
    "model_state_fingerprint",
    "restore_global_rng_state",
    "save_checkpoint",
]


CHECKPOINT_SCHEMA = "crystalatlas.training.checkpoint"
CHECKPOINT_SCHEMA_VERSION = 6


class TrainingNumericsError(NumericalError):
    """A loss, gradient, or trainable parameter failed a finite-value guard."""


class CheckpointError(CrystalAtlasError, RuntimeError):
    """A checkpoint is malformed, incompatible, or cannot be resumed exactly."""


@dataclass(frozen=True, slots=True)
class ScientificManifest:
    """Immutable semantic identity not fully represented by tensor shapes.

    ``architecture`` includes all neural hyperparameters and conditioning
    routes. ``representation`` includes token codecs, cover ordering, gauges,
    coordinate conventions, cutoffs, and logit bounds. ``catalogue`` pins the
    exact catalogue version/hash. ``data_contract`` pins the training measure,
    preprocessing, and split identity.  Every section is mandatory and
    nonempty so a checkpoint cannot silently rely on Python-object defaults.
    """

    architecture: Mapping[str, Any]
    representation: Mapping[str, Any]
    catalogue: Mapping[str, Any]
    data_contract: Mapping[str, Any]

    def __post_init__(self) -> None:
        for name in ("architecture", "representation", "catalogue", "data_contract"):
            value = getattr(self, name)
            if not isinstance(value, Mapping) or not value:
                raise DomainError(f"scientific manifest section {name!r} must be nonempty")
            # Accept our own recursively frozen sections without relaxing the
            # finite-JSON contract or changing the canonical fingerprint.
            try:
                thawed = _thaw_json(value)
            except RecursionError as exc:
                raise TypeError(
                    f"scientific manifest {name} must contain only "
                    "finite JSON-compatible values"
                ) from exc
            safe = _json_safe(thawed, name=f"scientific manifest {name}")
            object.__setattr__(self, name, _freeze_json(safe))

    def to_dict(self) -> dict[str, Any]:
        return {
            "architecture": _thaw_json(self.architecture),
            "representation": _thaw_json(self.representation),
            "catalogue": _thaw_json(self.catalogue),
            "data_contract": _thaw_json(self.data_contract),
        }

    @property
    def fingerprint(self) -> str:
        return _mapping_fingerprint(self.to_dict())


@dataclass(frozen=True, slots=True)
class TrainingIdentity:
    """The immutable stochastic/configuration identity of a training run."""

    seed: int
    config_fingerprint: str
    deterministic_algorithms: bool

    def __post_init__(self) -> None:
        # NumPy's legacy global RNG, which is seeded for third-party code,
        # accepts only an unsigned 32-bit seed.
        if isinstance(self.seed, bool) or not 0 <= self.seed <= 2**32 - 1:
            raise DomainError("training seed must be an unsigned 32-bit integer")
        if not isinstance(self.config_fingerprint, str) or len(self.config_fingerprint) != 64:
            raise DomainError("config_fingerprint must be a SHA-256 hex digest")
        try:
            int(self.config_fingerprint, 16)
        except ValueError as exc:
            raise DomainError("config_fingerprint must be hexadecimal") from exc
        if not isinstance(self.deterministic_algorithms, bool):
            raise DomainError("deterministic_algorithms must be boolean")


@dataclass(slots=True)
class TrainingContext:
    """Explicit RNG streams paired with the global seeded runtime.

    The NumPy and CPU PyTorch generators should be passed to samplers instead
    of relying on module-global RNGs.  Their states can be checkpointed exactly.
    Device-specific PyTorch generators may be managed by a higher-level
    distributed trainer; all visible global CUDA streams and the visible MPS
    stream are captured below.
    """

    identity: TrainingIdentity
    numpy_generator: np.random.Generator
    torch_generator: Any
    crystallography_certification: CrystallographyCertification

    def state_dict(self) -> dict[str, Any]:
        """Return a weights-only-safe representation of both explicit streams."""

        return {
            "seed": self.identity.seed,
            "config_fingerprint": self.identity.config_fingerprint,
            "deterministic_algorithms": self.identity.deterministic_algorithms,
            "crystallography_certification_fingerprint": (
                self.crystallography_certification.receipt_fingerprint
            ),
            "numpy_generator_state": _json_safe(
                self.numpy_generator.bit_generator.state,
                name="NumPy generator state",
            ),
            "torch_generator_state": self.torch_generator.get_state().cpu().clone(),
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        """Restore explicit streams after validating their run identity."""

        _validate_training_context_state(state, self)
        self.numpy_generator.bit_generator.state = state["numpy_generator_state"]
        self.torch_generator.set_state(state["torch_generator_state"].cpu())


def initialize_training(
    config: CrystalAtlasConfig,
    seed: int,
    *,
    catalogue: CertifiedCatalogue,
    crystallography_certification: CrystallographyCertification,
    deterministic_algorithms: bool = True,
    require_all_230_catalogue: bool = True,
    require_external_operation_database: bool = True,
) -> TrainingContext:
    """Pass the crystallography gate, then initialize deterministic RNG streams.

    The mandatory receipt is bound to the live catalogue.  Production training
    defaults to the complete 230-type catalogue and its independent pinned
    operation-database cross-check.  Equation-level fixture tests may relax
    those two scope requirements explicitly, but may not omit certification.
    """

    if not isinstance(crystallography_certification, CrystallographyCertification):
        raise TypeError("crystallography_certification must be a CrystallographyCertification")
    crystallography_certification.require_for(
        catalogue,
        require_all_230=require_all_230_catalogue,
        require_external_operation_database=require_external_operation_database,
    )

    identity = TrainingIdentity(
        seed=seed,
        config_fingerprint=config.fingerprint,
        deterministic_algorithms=deterministic_algorithms,
    )
    numpy_generator = seed_everything(
        identity.seed,
        deterministic_torch=identity.deterministic_algorithms,
    )
    torch_generator = torch.Generator(device="cpu")
    torch_generator.manual_seed(identity.seed)
    return TrainingContext(
        identity,
        numpy_generator,
        torch_generator,
        crystallography_certification,
    )


def exact_finite_action_nll(
    logits: Tensor,
    targets: Tensor,
    *,
    legal_actions: Tensor | None = None,
    reduction: str = "mean",
) -> Tensor:
    """Exact categorical NLL normalized over each declared legal action set.

    ``logits`` has shape ``[..., A]`` and integer ``targets`` has shape
    ``[...]``.  A false legal-action entry receives exactly zero probability;
    the observed target must be legal.  This routine is suitable for both
    grammar-constrained composition tokens and suffix-DP-constrained cover
    actions.
    """

    if logits.ndim < 1 or logits.shape[-1] < 1:
        raise ValueError("logits must contain a nonempty action dimension")
    if not torch.is_floating_point(logits):
        raise TypeError("logits must have a floating-point dtype")
    if targets.shape != logits.shape[:-1]:
        raise ValueError("targets must match every non-action logits dimension")
    if targets.dtype != torch.long:
        raise TypeError("targets must have dtype torch.long")
    if targets.device != logits.device:
        raise ValueError("targets and logits must be on the same device")
    if targets.numel() == 0:
        raise ValueError("an empty target batch has no defined training objective")
    action_count = logits.shape[-1]
    if bool(((targets < 0) | (targets >= action_count)).any().item()):
        raise ValueError("a target index lies outside the action vocabulary")

    if legal_actions is None:
        legal = torch.ones_like(logits, dtype=torch.bool)
    else:
        if legal_actions.shape != logits.shape:
            raise ValueError("legal_actions must have exactly the logits shape")
        if legal_actions.dtype != torch.bool:
            raise TypeError("legal_actions must be boolean")
        if legal_actions.device != logits.device:
            raise ValueError("legal_actions and logits must be on the same device")
        legal = legal_actions
    if bool((~legal.any(dim=-1)).any().item()):
        raise ValueError("every categorical decision must retain a legal action")
    if not bool(torch.isfinite(logits[legal]).all().item()):
        raise TrainingNumericsError("all legal logits must be finite")

    chosen_is_legal = legal.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
    if not bool(chosen_is_legal.all().item()):
        raise ValueError("an observed target is illegal under its exact action mask")
    # Softmax normalization is numerically fragile in FP16/BF16 and is part of
    # the probabilistic objective, not a throughput-critical equivariant
    # kernel.  Preserve float64 logits exactly; promote low-precision logits to
    # float32 while retaining an exact gradient path through the cast.
    working_logits = (
        logits.float() if logits.dtype in {torch.float16, torch.bfloat16} else logits
    )
    masked_logits = working_logits.masked_fill(~legal, -torch.inf)
    log_normalizer = torch.logsumexp(masked_logits, dim=-1)
    chosen_logits = working_logits.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
    nll = log_normalizer - chosen_logits

    if reduction == "none":
        return nll
    if reduction == "sum":
        return nll.sum()
    if reduction == "mean":
        return nll.mean()
    raise ValueError("reduction must be 'none', 'sum', or 'mean'")


@dataclass(frozen=True, slots=True)
class LossWeights:
    """Nonnegative coefficients for the three separately reported objectives."""

    composition_nll: float = 1.0
    cover_nll: float = 1.0
    cfm_regression: float = 1.0

    def __post_init__(self) -> None:
        values = (self.composition_nll, self.cover_nll, self.cfm_regression)
        if any(not math.isfinite(value) or value < 0.0 for value in values):
            raise DomainError("loss weights must be finite and nonnegative")
        if not any(value > 0.0 for value in values):
            raise DomainError("at least one loss weight must be positive")


@dataclass(frozen=True, slots=True)
class CompositeLossTerms:
    """Bookkeeping for discrete NLLs and the non-likelihood CFM regression."""

    total_objective: Tensor
    composition_nll: Tensor
    cover_nll: Tensor
    cfm_regression: Tensor
    weights: LossWeights

    def detached_metrics(self) -> dict[str, float]:
        """Return unambiguous scalar names for logging and aggregation."""

        return {
            "objective/total": float(self.total_objective.detach().item()),
            "nll/composition": float(self.composition_nll.detach().item()),
            "nll/cover": float(self.cover_nll.detach().item()),
            "regression/cfm": float(self.cfm_regression.detach().item()),
        }


def compose_training_loss(
    composition_nll: Tensor,
    cover_nll: Tensor,
    cfm_regression: Tensor,
    *,
    weights: LossWeights | None = None,
) -> CompositeLossTerms:
    """Form the optimization objective without conflating its semantics."""

    components = {
        "composition_nll": composition_nll,
        "cover_nll": cover_nll,
        "cfm_regression": cfm_regression,
    }
    devices = set()
    for name, value in components.items():
        if not isinstance(value, Tensor):
            raise TypeError(f"{name} must be a PyTorch tensor")
        if value.numel() != 1:
            raise ValueError(f"{name} must already be reduced to one scalar")
        if not torch.is_floating_point(value):
            raise TypeError(f"{name} must have a floating-point dtype")
        if not bool(torch.isfinite(value).all().item()):
            raise TrainingNumericsError(f"{name} is non-finite")
        devices.add(value.device)
    if len(devices) != 1:
        raise ValueError("all loss components must reside on the same device")

    resolved_weights = weights or LossWeights()
    total = (
        resolved_weights.composition_nll * composition_nll
        + resolved_weights.cover_nll * cover_nll
        + resolved_weights.cfm_regression * cfm_regression
    )
    if not bool(torch.isfinite(total).all().item()):
        raise TrainingNumericsError("weighted training objective is non-finite")
    return CompositeLossTerms(
        total_objective=total,
        composition_nll=composition_nll,
        cover_nll=cover_nll,
        cfm_regression=cfm_regression,
        weights=resolved_weights,
    )


@dataclass(frozen=True, slots=True)
class GradientReport:
    """Global L2 and finite-value audit before or after clipping."""

    all_finite: bool
    global_l2_norm: float
    max_absolute_value: float
    tensors_with_grad: int
    scalar_values_with_grad: int
    nonfinite_parameters: tuple[str, ...]


def inspect_gradients(model: Any) -> GradientReport:
    """Inspect gradients with one host synchronization per device when finite."""

    # Each value is [sum(abs(g)^2), max(abs(g)), all_finite] on one device.
    device_statistics: dict[Any, Tensor] = {}
    gradients: list[tuple[str, Tensor]] = []
    tensors = 0
    scalars = 0
    for name, parameter in model.named_parameters():
        gradient = parameter.grad
        if gradient is None:
            continue
        values = gradient.detach()
        if values.is_sparse:
            values = values.coalesce().values()
        gradients.append((name, values))
        tensors += 1
        scalars += values.numel()
        accumulator_dtype = torch.float32 if values.device.type == "mps" else torch.float64
        magnitudes = values.abs().to(dtype=accumulator_dtype)
        squared = magnitudes.square().sum()
        maximum = magnitudes.max() if magnitudes.numel() else squared.new_zeros(())
        finite = torch.isfinite(values).all().to(dtype=accumulator_dtype)
        statistic = torch.stack((squared, maximum, finite))
        previous = device_statistics.get(values.device)
        if previous is None:
            device_statistics[values.device] = statistic
        else:
            device_statistics[values.device] = torch.stack(
                (
                    previous[0] + statistic[0],
                    torch.maximum(previous[1], statistic[1]),
                    previous[2] * statistic[2],
                )
            )

    squared_norm = 0.0
    maximum_value = 0.0
    globally_finite = True
    for statistic in device_statistics.values():
        squared, maximum, finite = statistic.detach().cpu().tolist()
        squared_norm += float(squared)
        maximum_value = max(maximum_value, float(maximum))
        globally_finite = globally_finite and bool(finite)
    nonfinite: list[str] = []
    if not globally_finite:
        # The exceptional path pays for names precise enough to diagnose a run.
        nonfinite = [
            name for name, values in gradients if not bool(torch.isfinite(values).all().item())
        ]
    norm = math.sqrt(squared_norm) if globally_finite else math.inf
    if globally_finite and not math.isfinite(norm):
        nonfinite.append("<global-norm>")
    return GradientReport(
        all_finite=not nonfinite,
        global_l2_norm=norm,
        max_absolute_value=maximum_value if not nonfinite else math.inf,
        tensors_with_grad=tensors,
        scalar_values_with_grad=scalars,
        nonfinite_parameters=tuple(nonfinite),
    )


@dataclass(frozen=True, slots=True)
class OptimizerStepReport:
    """Numerical audit of one completed optimizer step."""

    loss_value: float
    preclip: GradientReport
    postclip: GradientReport
    clipped: bool


def _scale_gradients(model: Any, coefficient: float) -> None:
    with torch.no_grad():
        for parameter in model.parameters():
            gradient = parameter.grad
            if gradient is None:
                continue
            if gradient.is_sparse:
                gradient = gradient.coalesce()
                gradient._values().mul_(coefficient)
                parameter.grad = gradient
            else:
                gradient.mul_(coefficient)


def _nonfinite_trainable_parameters(model: Any) -> tuple[str, ...]:
    parameters: list[tuple[str, Tensor]] = []
    per_device: dict[Any, Tensor] = {}
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        values = parameter.detach()
        parameters.append((name, values))
        finite = torch.isfinite(values).all()
        previous = per_device.get(values.device)
        per_device[values.device] = finite if previous is None else previous & finite
    if all(bool(value.detach().cpu().item()) for value in per_device.values()):
        return ()
    return tuple(
        name for name, values in parameters if not bool(torch.isfinite(values).all().item())
    )


def _optimizer_parameters(optimizer: Any) -> tuple[Tensor, ...]:
    unique: dict[int, Tensor] = {}
    for group in optimizer.param_groups:
        for parameter in group["params"]:
            unique[id(parameter)] = parameter
    return tuple(unique.values())


def _nonfinite_optimizer_parameters(optimizer: Any) -> tuple[str, ...]:
    parameters = _optimizer_parameters(optimizer)
    per_device: dict[Any, Tensor] = {}
    for parameter in parameters:
        finite = torch.isfinite(parameter).all()
        previous = per_device.get(parameter.device)
        per_device[parameter.device] = finite if previous is None else previous & finite
    if all(bool(value.detach().cpu().item()) for value in per_device.values()):
        return ()
    return tuple(
        f"optimizer_parameter[{index}]"
        for index, parameter in enumerate(parameters)
        if not bool(torch.isfinite(parameter).all().item())
    )


def _restore_optimizer_transaction(
    parameter_states: tuple[tuple[Tensor, Tensor], ...],
    optimizer: Any,
    optimizer_state: Mapping[str, Any],
) -> tuple[str, ...]:
    failures: list[str] = []
    try:
        with torch.no_grad():
            for parameter, state in parameter_states:
                parameter.copy_(state)
    except Exception as exc:  # pragma: no cover - device-level failure
        failures.append(f"parameters: {exc}")
    try:
        optimizer.load_state_dict(optimizer_state)
    except Exception as exc:  # pragma: no cover - custom optimizer failure
        failures.append(f"optimizer: {exc}")
    try:
        optimizer.zero_grad(set_to_none=True)
    except Exception as exc:  # pragma: no cover - custom optimizer failure
        failures.append(f"zero_grad: {exc}")
    return tuple(failures)


def guarded_optimizer_step(
    loss: Tensor,
    model: Any,
    optimizer: Any,
    *,
    max_gradient_norm: float | None = None,
    transactional: bool = True,
) -> OptimizerStepReport:
    """Zero, backpropagate, validate, clip, and take exactly one step.

    The reported norm is the global Euclidean norm over all gradient entries.
    A non-finite loss, parameter, gradient, or norm prevents ``optimizer.step``.
    This helper intentionally owns ``zero_grad`` and therefore is not an
    accumulation API.  The default transactional mode snapshots parameters
    and optimizer state, then rolls both back if a generic optimizer creates a
    non-finite update.  Set it false only when the outer trainer will abort and
    restore the last atomic checkpoint; doing so avoids the snapshot overhead.
    """

    if not isinstance(loss, Tensor) or loss.numel() != 1:
        raise ValueError("loss must be a scalar PyTorch tensor")
    if not torch.is_floating_point(loss):
        raise TypeError("loss must have a floating-point dtype")
    if not bool(torch.isfinite(loss).all().item()):
        raise TrainingNumericsError("loss is non-finite; optimizer step was skipped")
    if not loss.requires_grad:
        raise ValueError("loss must retain a gradient graph")
    if max_gradient_norm is not None and (
        not math.isfinite(max_gradient_norm) or max_gradient_norm <= 0.0
    ):
        raise ValueError("max_gradient_norm must be finite and positive")
    if not isinstance(transactional, bool):
        raise TypeError("transactional must be boolean")
    bad_parameters = _nonfinite_trainable_parameters(model)
    if bad_parameters:
        raise TrainingNumericsError(
            f"non-finite trainable parameters before backward: {bad_parameters}"
        )

    trainable_ids = {
        id(parameter) for parameter in model.parameters() if parameter.requires_grad
    }
    optimizer_ids = {
        id(parameter) for group in optimizer.param_groups for parameter in group["params"]
    }
    if not trainable_ids.issubset(optimizer_ids):
        raise ValueError("optimizer does not own every trainable model parameter")
    bad_optimizer_parameters = _nonfinite_optimizer_parameters(optimizer)
    if bad_optimizer_parameters:
        raise TrainingNumericsError(
            f"non-finite optimizer parameters before backward: {bad_optimizer_parameters}"
        )
    bad_optimizer_state = _nonfinite_state_paths(optimizer.state_dict())
    if bad_optimizer_state:
        raise TrainingNumericsError(
            f"non-finite optimizer state before backward: {bad_optimizer_state}"
        )

    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    preclip = inspect_gradients(model)
    if preclip.tensors_with_grad == 0:
        optimizer.zero_grad(set_to_none=True)
        raise TrainingNumericsError("backward produced no model gradients")
    if not preclip.all_finite:
        optimizer.zero_grad(set_to_none=True)
        raise TrainingNumericsError(
            f"non-finite gradients; optimizer step skipped: {preclip.nonfinite_parameters}"
        )

    clipped = False
    if max_gradient_norm is not None and preclip.global_l2_norm > max_gradient_norm:
        coefficient = max_gradient_norm / preclip.global_l2_norm
        _scale_gradients(model, coefficient)
        clipped = True
    postclip = inspect_gradients(model)
    if not postclip.all_finite:
        optimizer.zero_grad(set_to_none=True)
        raise TrainingNumericsError("gradient clipping produced a non-finite value")
    parameter_states = (
        tuple(
            (parameter, parameter.detach().clone())
            for parameter in _optimizer_parameters(optimizer)
        )
        if transactional
        else ()
    )
    optimizer_state = copy.deepcopy(optimizer.state_dict()) if transactional else None
    try:
        optimizer.step()
    except Exception as exc:
        failures = (
            _restore_optimizer_transaction(parameter_states, optimizer, optimizer_state)
            if transactional and optimizer_state is not None
            else ()
        )
        detail = (
            f"; rollback failures={failures}"
            if failures
            else "; transaction rolled back"
            if transactional
            else ""
        )
        raise TrainingNumericsError(f"optimizer step failed{detail}") from exc
    bad_parameters_after = _nonfinite_optimizer_parameters(optimizer)
    bad_state_after = _nonfinite_state_paths(optimizer.state_dict())
    if bad_parameters_after or bad_state_after:
        failures = (
            _restore_optimizer_transaction(parameter_states, optimizer, optimizer_state)
            if transactional and optimizer_state is not None
            else ()
        )
        recovery = (
            f"rollback failures={failures}"
            if failures
            else "transaction rolled back"
            if transactional
            else "restore a prior checkpoint"
        )
        raise TrainingNumericsError(
            "optimizer produced non-finite state; "
            f"parameters={bad_parameters_after}, state={bad_state_after}; {recovery}"
        )
    return OptimizerStepReport(
        loss_value=float(loss.detach().item()),
        preclip=preclip,
        postclip=postclip,
        clipped=clipped,
    )


def _mps_is_available() -> bool:
    return bool(
        hasattr(torch, "mps")
        and hasattr(torch.backends, "mps")
        and torch.backends.mps.is_available()
    )


def _probe_mps_rng_state(candidate: Tensor) -> None:
    """Validate an opaque MPS state while restoring the live generator exactly."""

    try:
        previous = torch.mps.get_rng_state().detach().cpu().clone()
    except Exception as exc:  # pragma: no cover - backend-level failure
        raise CheckpointError("cannot inspect the live Torch MPS RNG state") from exc

    candidate_error: Exception | None = None
    try:
        torch.mps.set_rng_state(candidate.detach().cpu().contiguous())
    except Exception as exc:  # the backend is the authority on its opaque state format
        candidate_error = exc

    try:
        torch.mps.set_rng_state(previous)
        restored = torch.mps.get_rng_state().detach().cpu()
    except Exception as exc:  # pragma: no cover - catastrophic backend failure
        raise CheckpointError(
            "Torch MPS RNG validation could not restore the prior live state"
        ) from exc
    if not torch.equal(restored, previous):
        raise CheckpointError("Torch MPS RNG validation changed the prior live state")
    if candidate_error is not None:
        raise CheckpointError("invalid Torch MPS RNG state") from candidate_error


def capture_global_rng_state() -> dict[str, Any]:
    """Capture Python, NumPy, CPU Torch, CUDA, and the visible MPS RNG."""

    algorithm, keys, position, has_gauss, cached_gaussian = np.random.get_state()
    cuda_states: tuple[Tensor, ...]
    if torch.cuda.is_available():
        cuda_states = tuple(state.cpu().clone() for state in torch.cuda.get_rng_state_all())
    else:
        cuda_states = ()
    mps_available = _mps_is_available()
    mps_state = torch.mps.get_rng_state().cpu().clone() if mps_available else None
    return {
        "python": random.getstate(),
        "numpy_global": {
            "algorithm": algorithm,
            "keys": torch.from_numpy(keys.copy()),
            "position": int(position),
            "has_gauss": int(has_gauss),
            "cached_gaussian": float(cached_gaussian),
        },
        "torch_cpu": torch.get_rng_state().cpu().clone(),
        "torch_cuda": cuda_states,
        "torch_mps": mps_state,
    }


def _as_tuple(value: Any) -> Any:
    if isinstance(value, list):
        return tuple(_as_tuple(item) for item in value)
    if isinstance(value, tuple):
        return tuple(_as_tuple(item) for item in value)
    return value


def _validate_global_rng_state(
    state: Mapping[str, Any],
    *,
    strict_cuda: bool,
    strict_mps: bool = True,
) -> None:
    required = {"python", "numpy_global", "torch_cpu", "torch_cuda", "torch_mps"}
    if set(state) != required:
        raise CheckpointError("global RNG state has missing or unexpected fields")
    try:
        python_probe = random.Random()
        python_probe.setstate(_as_tuple(state["python"]))
    except (TypeError, ValueError) as exc:
        raise CheckpointError("invalid Python RNG state") from exc

    numpy_state = state["numpy_global"]
    if not isinstance(numpy_state, Mapping):
        raise CheckpointError("invalid NumPy global RNG state")
    try:
        keys = numpy_state["keys"]
        if not isinstance(keys, Tensor):
            raise TypeError
        numpy_tuple = (
            str(numpy_state["algorithm"]),
            keys.detach().cpu().numpy().astype(np.uint32, copy=True),
            int(numpy_state["position"]),
            int(numpy_state["has_gauss"]),
            float(numpy_state["cached_gaussian"]),
        )
        numpy_probe = np.random.RandomState()
        numpy_probe.set_state(numpy_tuple)
    except (KeyError, TypeError, ValueError) as exc:
        raise CheckpointError("invalid NumPy global RNG state") from exc

    torch_cpu = state["torch_cpu"]
    if not isinstance(torch_cpu, Tensor):
        raise CheckpointError("invalid Torch CPU RNG state")
    try:
        torch.Generator(device="cpu").set_state(torch_cpu.detach().cpu())
    except RuntimeError as exc:
        raise CheckpointError("invalid Torch CPU RNG state") from exc

    cuda_states = state["torch_cuda"]
    if not isinstance(cuda_states, (tuple, list)) or not all(
        isinstance(value, Tensor) for value in cuda_states
    ):
        raise CheckpointError("invalid Torch CUDA RNG state collection")
    visible_devices = torch.cuda.device_count() if torch.cuda.is_available() else 0
    if strict_cuda and len(cuda_states) != visible_devices:
        raise CheckpointError(
            "saved and visible CUDA device counts differ; exact RNG resume is impossible"
        )
    mps_state = state["torch_mps"]
    if mps_state is not None and not isinstance(mps_state, Tensor):
        raise CheckpointError("invalid Torch MPS RNG state")
    if isinstance(mps_state, Tensor) and (
        mps_state.dtype != torch.uint8
        or mps_state.layout != torch.strided
        or mps_state.ndim != 1
        or mps_state.numel() == 0
    ):
        raise CheckpointError("invalid Torch MPS RNG state")
    mps_available = _mps_is_available()
    if strict_mps and (mps_state is not None) != mps_available:
        raise CheckpointError(
            "saved and visible MPS RNG availability differs; exact RNG resume is impossible"
        )
    if isinstance(mps_state, Tensor) and mps_available:
        _probe_mps_rng_state(mps_state)


def _apply_global_rng_state(
    state: Mapping[str, Any],
    *,
    strict_cuda: bool,
    strict_mps: bool = True,
) -> None:
    random.setstate(_as_tuple(state["python"]))
    numpy_state = state["numpy_global"]
    keys = numpy_state["keys"].detach().cpu().numpy().astype(np.uint32, copy=True)
    np.random.set_state(
        (
            str(numpy_state["algorithm"]),
            keys,
            int(numpy_state["position"]),
            int(numpy_state["has_gauss"]),
            float(numpy_state["cached_gaussian"]),
        )
    )
    torch.set_rng_state(state["torch_cpu"].detach().cpu())
    cuda_states = tuple(value.detach().cpu() for value in state["torch_cuda"])
    visible_devices = torch.cuda.device_count() if torch.cuda.is_available() else 0
    if cuda_states and len(cuda_states) == visible_devices:
        torch.cuda.set_rng_state_all(cuda_states)
    elif cuda_states and strict_cuda:
        raise CheckpointError("cannot restore all saved CUDA RNG streams")
    mps_state = state["torch_mps"]
    mps_available = _mps_is_available()
    if isinstance(mps_state, Tensor) and mps_available:
        torch.mps.set_rng_state(mps_state.detach().cpu())
    elif mps_state is not None and strict_mps:
        raise CheckpointError("cannot restore the saved MPS RNG stream")


def restore_global_rng_state(
    state: Mapping[str, Any],
    *,
    strict_cuda: bool = True,
    strict_mps: bool = True,
) -> None:
    """Transactionally restore all global RNGs visible to this process."""

    _validate_global_rng_state(
        state,
        strict_cuda=strict_cuda,
        strict_mps=strict_mps,
    )
    previous = capture_global_rng_state()
    try:
        _apply_global_rng_state(
            state,
            strict_cuda=strict_cuda,
            strict_mps=strict_mps,
        )
    except Exception:
        _apply_global_rng_state(previous, strict_cuda=False, strict_mps=False)
        raise


@dataclass(frozen=True, slots=True)
class CheckpointInfo:
    """Integrity and identity information returned after an atomic save."""

    path: Path
    sha256: str
    size_bytes: int
    step: int
    epoch: int
    config_fingerprint: str
    manifest_fingerprint: str
    model_state_fingerprint: str


@dataclass(frozen=True, slots=True)
class LoadedCheckpoint:
    """Validated metadata returned after loading state into caller-owned objects."""

    path: Path
    sha256: str
    step: int
    epoch: int
    seed: int
    config_fingerprint: str
    deterministic_algorithms: bool
    manifest_fingerprint: str
    model_state_fingerprint: str
    config: dict[str, Any]
    scientific_manifest: dict[str, Any]
    provenance: dict[str, Any]
    metadata: dict[str, Any]
    auxiliary_state: dict[str, Any]
    missing_model_keys: tuple[str, ...]
    unexpected_model_keys: tuple[str, ...]
    optimizer_restored: bool
    rng_restored: bool


def _freeze_json(value: Any) -> Any:
    if isinstance(value, dict):
        return MappingProxyType({key: _freeze_json(item) for key, item in value.items()})
    if isinstance(value, list):
        return tuple(_freeze_json(item) for item in value)
    return value


def _thaw_json(value: Any) -> Any:
    if isinstance(value, Mapping):
        # Preserve keys until JSON validation: str(...) here would silently
        # legitimize unsupported keys (e.g. tuples) supplied by a caller.
        return {key: _thaw_json(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_thaw_json(item) for item in value]
    return value


def _json_safe(value: Any, *, name: str) -> Any:
    try:
        payload = json.dumps(value, allow_nan=False, sort_keys=True, separators=(",", ":"))
        return json.loads(payload)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{name} must contain only finite JSON-compatible values") from exc


def _mapping_fingerprint(mapping: Mapping[str, Any]) -> str:
    try:
        payload = json.dumps(mapping, allow_nan=False, sort_keys=True, separators=(",", ":"))
    except (TypeError, ValueError) as exc:
        raise CheckpointError("stored configuration is not canonical JSON") from exc
    return sha256(payload.encode("utf-8")).hexdigest()


def _qualified_class_name(value: Any) -> str:
    cls = type(value)
    return f"{cls.__module__}.{cls.__qualname__}"


def _model_state_schema(model: Any) -> dict[str, Any]:
    schema: dict[str, Any] = {}
    for name, value in model.state_dict().items():
        if isinstance(value, Tensor):
            schema[name] = {
                "kind": "tensor",
                "shape": list(value.shape),
                "dtype": str(value.dtype),
            }
        else:
            schema[name] = {
                "kind": "python",
                "type": _qualified_class_name(value),
            }
    return schema


def _optimizer_parameter_names(model: Any, optimizer: Any) -> list[list[str]]:
    """Return the ordered model-parameter names in every optimizer group.

    PyTorch optimizer state uses transient integer parameter identifiers.  A
    group with the same length but a different parameter order will otherwise
    load successfully and attach moment estimates to the wrong tensors.  The
    ordered name contract makes that silent trajectory change impossible.
    """

    by_identity = {id(parameter): name for name, parameter in model.named_parameters()}
    groups: list[list[str]] = []
    seen: set[int] = set()
    for group_index, group in enumerate(optimizer.param_groups):
        parameters = group.get("params")
        if not isinstance(parameters, list):
            raise CheckpointError(
                f"optimizer parameter group {group_index} has no list-valued params"
            )
        names: list[str] = []
        for parameter in parameters:
            parameter_id = id(parameter)
            name = by_identity.get(parameter_id)
            if name is None:
                raise CheckpointError(
                    "optimizer owns a parameter that is not registered on the model"
                )
            if parameter_id in seen:
                raise CheckpointError(
                    "optimizer parameter appears more than once across parameter groups"
                )
            seen.add(parameter_id)
            names.append(name)
        groups.append(names)
    return groups


def _state_mapping_fingerprint(state: Mapping[str, Any]) -> str:
    """Hash tensor values and Python state with an unambiguous canonical frame."""

    digest = sha256()
    for name in sorted(state):
        value = state[name]
        encoded_name = name.encode("utf-8")
        digest.update(len(encoded_name).to_bytes(8, "big"))
        digest.update(encoded_name)
        if isinstance(value, Tensor):
            tensor = value.detach().contiguous().cpu()
            metadata = json.dumps(
                {"dtype": str(tensor.dtype), "shape": list(tensor.shape)},
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
            raw = tensor.reshape(-1).view(torch.uint8).numpy().tobytes()
            digest.update(b"T")
            digest.update(len(metadata).to_bytes(8, "big"))
            digest.update(metadata)
            digest.update(len(raw).to_bytes(8, "big"))
            digest.update(raw)
        else:
            payload = json.dumps(
                _json_safe(value, name=f"model state {name!r}"),
                allow_nan=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
            digest.update(b"P")
            digest.update(len(payload).to_bytes(8, "big"))
            digest.update(payload)
    return digest.hexdigest()


def model_state_fingerprint(model: Any) -> str:
    """Return a SHA-256 identity for the model's current parameters and buffers."""

    state = model.state_dict()
    if not isinstance(state, Mapping):
        raise TypeError("model.state_dict() must return a mapping")
    return _state_mapping_fingerprint(state)


def _compatibility_manifest(
    scientific_manifest: ScientificManifest,
    model: Any,
    optimizer: Any,
) -> dict[str, Any]:
    declared = scientific_manifest.to_dict()
    return {
        "declared": declared,
        "declared_fingerprint": scientific_manifest.fingerprint,
        "model_class": _qualified_class_name(model),
        "model_state_schema": _model_state_schema(model),
        "optimizer_class": _qualified_class_name(optimizer),
        "optimizer_parameter_names": _optimizer_parameter_names(model, optimizer),
    }


def _validate_live_scientific_manifest(model: Any, manifest: ScientificManifest) -> None:
    """Use a model's live semantic validator when it exposes one.

    Generic tensor modules may rely on the caller's explicit manifest. A
    CrystalAtlas model can additionally derive its numerical/decoder identity;
    passing an old manifest back to the loader must not bypass that check.
    """

    validator = getattr(model, "validate_scientific_manifest", None)
    if validator is None:
        return
    if not callable(validator):
        raise CheckpointError("model scientific-manifest validator is not callable")
    try:
        validator(manifest)
    except (TypeError, ValueError, DomainError) as exc:
        raise CheckpointError("scientific manifest does not match the live model") from exc


def _validate_training_context_state(
    state: Mapping[str, Any], context: TrainingContext
) -> None:
    required = {
        "seed",
        "config_fingerprint",
        "deterministic_algorithms",
        "crystallography_certification_fingerprint",
        "numpy_generator_state",
        "torch_generator_state",
    }
    if set(state) != required:
        raise CheckpointError("training-context RNG state has invalid fields")
    if state["seed"] != context.identity.seed:
        raise CheckpointError("training-context seed differs from the checkpoint")
    if state["config_fingerprint"] != context.identity.config_fingerprint:
        raise CheckpointError("training-context configuration differs from the checkpoint")
    if state["deterministic_algorithms"] != context.identity.deterministic_algorithms:
        raise CheckpointError("deterministic-algorithm mode differs from the checkpoint")
    if (
        state["crystallography_certification_fingerprint"]
        != context.crystallography_certification.receipt_fingerprint
    ):
        raise CheckpointError("crystallography certification differs from the checkpoint")
    numpy_state = state["numpy_generator_state"]
    if not isinstance(numpy_state, Mapping):
        raise CheckpointError("invalid explicit NumPy generator state")
    try:
        numpy_probe = np.random.default_rng()
        numpy_probe.bit_generator.state = numpy_state
    except (TypeError, ValueError) as exc:
        raise CheckpointError("invalid explicit NumPy generator state") from exc
    torch_state = state["torch_generator_state"]
    if not isinstance(torch_state, Tensor):
        raise CheckpointError("invalid explicit Torch generator state")
    try:
        torch.Generator(device="cpu").set_state(torch_state.detach().cpu())
    except RuntimeError as exc:
        raise CheckpointError("invalid explicit Torch generator state") from exc


def _validate_nonnegative_integer(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise CheckpointError(f"{name} must be a nonnegative integer")
    return value


def _fsync_directory(directory: Path) -> None:
    flags = os.O_RDONLY
    if hasattr(os, "O_DIRECTORY"):
        flags |= os.O_DIRECTORY
    try:
        descriptor = os.open(directory, flags)
    except OSError:
        return
    try:
        os.fsync(descriptor)
    except OSError:
        pass
    finally:
        os.close(descriptor)


def save_checkpoint(
    path: str | Path,
    *,
    model: Any,
    optimizer: Any,
    config: CrystalAtlasConfig,
    scientific_manifest: ScientificManifest,
    provenance: ProvenanceRecord,
    step: int,
    epoch: int = 0,
    training_context: TrainingContext | None = None,
    metadata: Mapping[str, Any] | None = None,
    auxiliary_state: Mapping[str, Any] | None = None,
) -> CheckpointInfo:
    """Atomically replace ``path`` with a complete, resumable checkpoint.

    ``auxiliary_state`` is a weights-only-safe extension point for higher-level
    trainers.  It is intended for state dictionaries such as AMP scalers,
    schedulers, EMA weights, sampler cursors, and distributed RNG streams.  The
    base checkpoint loader returns it without interpretation so the owning
    trainer can validate and apply the complete transaction.
    """

    if isinstance(step, bool) or not isinstance(step, int) or step < 0:
        raise ValueError("step must be a nonnegative integer")
    if isinstance(epoch, bool) or not isinstance(epoch, int) or epoch < 0:
        raise ValueError("epoch must be a nonnegative integer")
    if not isinstance(scientific_manifest, ScientificManifest):
        raise TypeError("scientific_manifest must be a ScientificManifest")
    _validate_live_scientific_manifest(model, scientific_manifest)
    if provenance.config_fingerprint != config.fingerprint:
        raise CheckpointError("provenance and configuration fingerprints differ")
    checkpoint_identity = TrainingIdentity(
        seed=provenance.random_seed,
        config_fingerprint=config.fingerprint,
        deterministic_algorithms=(
            training_context.identity.deterministic_algorithms
            if training_context is not None
            else bool(torch.are_deterministic_algorithms_enabled())
        ),
    )
    if training_context is not None:
        if training_context.identity.config_fingerprint != config.fingerprint:
            raise CheckpointError("training context and configuration fingerprints differ")
        if training_context.identity.seed != provenance.random_seed:
            raise CheckpointError("training context and provenance seeds differ")
        declared_catalogue_fingerprint = scientific_manifest.catalogue.get("fingerprint")
        if (
            declared_catalogue_fingerprint
            != training_context.crystallography_certification.catalogue_fingerprint
        ):
            raise CheckpointError(
                "training context certification and scientific-manifest catalogue differ"
            )

    safe_config = _json_safe(config.to_dict(), name="configuration")
    safe_provenance = _json_safe(provenance.to_dict(), name="provenance")
    safe_metadata = _json_safe(dict(metadata or {}), name="checkpoint metadata")
    safe_auxiliary_state = dict(auxiliary_state or {})
    _validate_weights_only_tree(safe_auxiliary_state, name="auxiliary_state")
    nonfinite_auxiliary = _nonfinite_state_paths(safe_auxiliary_state)
    if nonfinite_auxiliary:
        raise CheckpointError(
            f"auxiliary checkpoint state contains non-finite values: {nonfinite_auxiliary}"
        )
    compatibility_manifest = _compatibility_manifest(
        scientific_manifest,
        model,
        optimizer,
    )
    manifest_fingerprint = _mapping_fingerprint(compatibility_manifest)
    state_fingerprint = model_state_fingerprint(model)
    payload = {
        "schema": CHECKPOINT_SCHEMA,
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        "step": step,
        "epoch": epoch,
        "identity": {
            "seed": checkpoint_identity.seed,
            "config_fingerprint": checkpoint_identity.config_fingerprint,
            "deterministic_algorithms": checkpoint_identity.deterministic_algorithms,
            "manifest_fingerprint": manifest_fingerprint,
            "model_state_fingerprint": state_fingerprint,
        },
        "config": safe_config,
        "scientific_manifest": compatibility_manifest,
        "provenance": safe_provenance,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "global_rng_state": capture_global_rng_state(),
        "training_context_state": (
            training_context.state_dict() if training_context is not None else None
        ),
        "metadata": safe_metadata,
        "auxiliary_state": safe_auxiliary_state,
    }

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and destination.is_dir():
        raise CheckpointError(f"checkpoint destination is a directory: {destination}")
    descriptor, temporary_name = tempfile.mkstemp(
        dir=destination.parent,
        prefix=f".{destination.name}.",
        suffix=".tmp",
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        torch.save(payload, temporary)
        with temporary.open("rb") as handle:
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
        _fsync_directory(destination.parent)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise

    return CheckpointInfo(
        path=destination.resolve(),
        sha256=sha256_file(destination),
        size_bytes=destination.stat().st_size,
        step=step,
        epoch=epoch,
        config_fingerprint=config.fingerprint,
        manifest_fingerprint=manifest_fingerprint,
        model_state_fingerprint=state_fingerprint,
    )


def _load_weights_only(path: Path, map_location: Any) -> Mapping[str, Any]:
    try:
        payload = torch.load(path, map_location=map_location, weights_only=True)
    except TypeError as exc:  # pragma: no cover - project requires torch >= 2.4
        raise CheckpointError(
            "installed PyTorch lacks safe weights-only checkpoint loading"
        ) from exc
    except Exception as exc:
        raise CheckpointError(f"failed to read checkpoint: {path}") from exc
    if not isinstance(payload, Mapping):
        raise CheckpointError("checkpoint root must be a mapping")
    return payload


def _validate_model_state(model: Any, state: Mapping[str, Any], *, strict: bool) -> None:
    current = model.state_dict()
    current_keys = set(current)
    saved_keys = set(state)
    if strict and current_keys != saved_keys:
        missing = tuple(sorted(current_keys - saved_keys))
        unexpected = tuple(sorted(saved_keys - current_keys))
        raise CheckpointError(
            f"model-state keys differ; missing={missing}, unexpected={unexpected}"
        )
    for key in current_keys & saved_keys:
        expected = current[key]
        saved = state[key]
        if isinstance(expected, Tensor) and isinstance(saved, Tensor):
            if expected.shape != saved.shape:
                raise CheckpointError(
                    f"model-state tensor {key!r} has shape {tuple(saved.shape)}, "
                    f"expected {tuple(expected.shape)}"
                )
            if (torch.is_floating_point(saved) or torch.is_complex(saved)) and not bool(
                torch.isfinite(saved).all().item()
            ):
                raise CheckpointError(f"model-state tensor {key!r} is non-finite")


def _nonfinite_state_paths(value: Any, prefix: str = "") -> tuple[str, ...]:
    tensors: list[tuple[str, Tensor]] = []
    failures: list[str] = []

    def collect(item: Any, path: str) -> None:
        if isinstance(item, Tensor):
            if torch.is_floating_point(item) or torch.is_complex(item):
                tensors.append((path or "<root>", item))
        elif isinstance(item, Mapping):
            for key, child in item.items():
                collect(child, f"{path}.{key}" if path else str(key))
        elif isinstance(item, (tuple, list)):
            for index, child in enumerate(item):
                collect(child, f"{path}[{index}]")
        elif isinstance(item, float) and not math.isfinite(item):
            failures.append(path or "<root>")

    collect(value, prefix)
    per_device: dict[Any, Tensor] = {}
    for _, tensor in tensors:
        finite = torch.isfinite(tensor).all()
        previous = per_device.get(tensor.device)
        per_device[tensor.device] = finite if previous is None else previous & finite
    if not all(bool(flag.detach().cpu().item()) for flag in per_device.values()):
        failures.extend(
            path for path, tensor in tensors if not bool(torch.isfinite(tensor).all().item())
        )
    return tuple(failures)


def _validate_weights_only_tree(value: Any, *, name: str) -> None:
    """Reject objects that PyTorch's safe weights-only loader cannot represent.

    This deliberately accepts only the ordinary state-dict algebra: tensors,
    mappings with text keys, sequences, and scalar primitives.  In particular,
    arbitrary Python objects and NumPy arrays are rejected before a checkpoint
    can be published but later fail safe loading.
    """

    def visit(item: Any, path: str) -> None:
        if isinstance(item, Tensor) or item is None:
            return
        if isinstance(item, (bool, int, float, str, bytes)):
            if isinstance(item, float) and not math.isfinite(item):
                raise CheckpointError(f"{path} contains a non-finite float")
            return
        if isinstance(item, Mapping):
            for key, child in item.items():
                if not isinstance(key, (str, int)) or isinstance(key, bool):
                    raise CheckpointError(f"{path} contains an unsupported mapping-key type")
                visit(child, f"{path}.{key}")
            return
        if isinstance(item, (tuple, list)):
            for index, child in enumerate(item):
                visit(child, f"{path}[{index}]")
            return
        raise CheckpointError(
            f"{path} contains unsupported weights-only type "
            f"{type(item).__module__}.{type(item).__qualname__}"
        )

    visit(value, name)


def _validate_optimizer_state(optimizer: Any, state: Mapping[str, Any]) -> None:
    if set(state) != {"state", "param_groups"}:
        raise CheckpointError("optimizer state has missing or unexpected fields")
    saved_groups = state["param_groups"]
    current_groups = optimizer.state_dict().get("param_groups")
    if not isinstance(saved_groups, list) or not isinstance(current_groups, list):
        raise CheckpointError("optimizer param_groups must be lists")
    if len(saved_groups) != len(current_groups):
        raise CheckpointError("optimizer parameter-group count differs")
    for index, (saved, current) in enumerate(zip(saved_groups, current_groups, strict=True)):
        if not isinstance(saved, Mapping) or not isinstance(current, Mapping):
            raise CheckpointError("optimizer parameter groups must be mappings")
        saved_parameters = saved.get("params")
        current_parameters = current.get("params")
        if not isinstance(saved_parameters, list) or not isinstance(current_parameters, list):
            raise CheckpointError("optimizer group params must be lists")
        if len(saved_parameters) != len(current_parameters):
            raise CheckpointError(f"optimizer parameter count differs in group {index}")
    nonfinite = _nonfinite_state_paths(state)
    if nonfinite:
        raise CheckpointError(f"optimizer state contains non-finite tensors: {nonfinite}")


def load_checkpoint(
    path: str | Path,
    *,
    model: Any,
    expected_scientific_manifest: ScientificManifest,
    optimizer: Any | None = None,
    expected_config: CrystalAtlasConfig | None = None,
    training_context: TrainingContext | None = None,
    map_location: Any = "cpu",
    strict_model: bool = True,
    restore_rng: bool = False,
    strict_cuda_rng: bool = True,
    strict_mps_rng: bool = True,
    expected_sha256: str | None = None,
    transactional: bool = True,
) -> LoadedCheckpoint:
    """Validate then load a checkpoint; optionally restore an exact RNG point.

    ``restore_rng=False`` is appropriate for inference and avoids surprising
    global side effects.  Exact training resume sets it true and passes the
    same-identity ``TrainingContext`` if one was saved.
    """

    source = Path(path)
    if not isinstance(expected_scientific_manifest, ScientificManifest):
        raise TypeError("expected_scientific_manifest must be a ScientificManifest")
    _validate_live_scientific_manifest(model, expected_scientific_manifest)
    if not isinstance(transactional, bool):
        raise TypeError("transactional must be boolean")
    if not source.is_file():
        raise CheckpointError(f"checkpoint is not a regular file: {source}")
    actual_hash = sha256_file(source)
    if expected_sha256 is not None and actual_hash != expected_sha256:
        raise CheckpointError("checkpoint SHA-256 does not match the expected digest")
    payload = _load_weights_only(source, map_location)

    required = {
        "schema",
        "schema_version",
        "step",
        "epoch",
        "identity",
        "config",
        "scientific_manifest",
        "provenance",
        "model_state",
        "optimizer_state",
        "global_rng_state",
        "training_context_state",
        "metadata",
        "auxiliary_state",
    }
    if set(payload) != required:
        raise CheckpointError("checkpoint has missing or unexpected top-level fields")
    if payload["schema"] != CHECKPOINT_SCHEMA:
        raise CheckpointError("unrecognized checkpoint schema")
    if payload["schema_version"] != CHECKPOINT_SCHEMA_VERSION:
        raise CheckpointError("unsupported checkpoint schema version")
    step = _validate_nonnegative_integer(payload["step"], "step")
    epoch = _validate_nonnegative_integer(payload["epoch"], "epoch")

    identity = payload["identity"]
    config_payload = payload["config"]
    manifest_payload = payload["scientific_manifest"]
    provenance_payload = payload["provenance"]
    metadata_payload = payload["metadata"]
    auxiliary_state = payload["auxiliary_state"]
    model_state = payload["model_state"]
    optimizer_state = payload["optimizer_state"]
    global_rng_state = payload["global_rng_state"]
    context_state = payload["training_context_state"]
    mappings = {
        "identity": identity,
        "config": config_payload,
        "scientific_manifest": manifest_payload,
        "provenance": provenance_payload,
        "metadata": metadata_payload,
        "model_state": model_state,
        "optimizer_state": optimizer_state,
        "global_rng_state": global_rng_state,
        "auxiliary_state": auxiliary_state,
    }
    for name, value in mappings.items():
        if not isinstance(value, Mapping):
            raise CheckpointError(f"checkpoint {name} must be a mapping")
    _validate_weights_only_tree(auxiliary_state, name="auxiliary_state")
    nonfinite_auxiliary = _nonfinite_state_paths(auxiliary_state)
    if nonfinite_auxiliary:
        raise CheckpointError(
            f"auxiliary checkpoint state contains non-finite values: {nonfinite_auxiliary}"
        )

    if set(identity) != {
        "seed",
        "config_fingerprint",
        "deterministic_algorithms",
        "manifest_fingerprint",
        "model_state_fingerprint",
    }:
        raise CheckpointError("checkpoint identity has invalid fields")
    seed = _validate_nonnegative_integer(identity["seed"], "seed")
    stored_fingerprint = identity["config_fingerprint"]
    deterministic_algorithms = identity["deterministic_algorithms"]
    manifest_fingerprint = identity["manifest_fingerprint"]
    state_fingerprint = identity["model_state_fingerprint"]
    if not isinstance(stored_fingerprint, str):
        raise CheckpointError("config fingerprint must be text")
    try:
        stored_identity = TrainingIdentity(
            seed=seed,
            config_fingerprint=stored_fingerprint,
            deterministic_algorithms=deterministic_algorithms,
        )
    except DomainError as exc:
        raise CheckpointError("stored training identity is invalid") from exc
    if not isinstance(manifest_fingerprint, str):
        raise CheckpointError("scientific manifest fingerprint must be text")
    if not isinstance(state_fingerprint, str) or len(state_fingerprint) != 64:
        raise CheckpointError("model-state fingerprint must be a SHA-256 digest")
    try:
        int(state_fingerprint, 16)
    except ValueError as exc:
        raise CheckpointError("model-state fingerprint must be hexadecimal") from exc
    if _mapping_fingerprint(config_payload) != stored_fingerprint:
        raise CheckpointError("stored configuration fingerprint is internally inconsistent")
    if _mapping_fingerprint(manifest_payload) != manifest_fingerprint:
        raise CheckpointError("stored scientific manifest fingerprint is inconsistent")
    if _state_mapping_fingerprint(model_state) != state_fingerprint:
        raise CheckpointError("stored model-state fingerprint is inconsistent")
    manifest_fields = {
        "declared",
        "declared_fingerprint",
        "model_class",
        "model_state_schema",
        "optimizer_class",
        "optimizer_parameter_names",
    }
    if set(manifest_payload) != manifest_fields:
        raise CheckpointError("stored scientific manifest has invalid fields")
    declared_manifest = manifest_payload["declared"]
    if not isinstance(declared_manifest, Mapping):
        raise CheckpointError("stored declared scientific manifest must be a mapping")
    if manifest_payload["declared_fingerprint"] != _mapping_fingerprint(declared_manifest):
        raise CheckpointError("stored declared scientific manifest is inconsistent")
    if expected_scientific_manifest.fingerprint != manifest_payload["declared_fingerprint"]:
        raise CheckpointError("checkpoint has a different scientific manifest")
    if manifest_payload["model_class"] != _qualified_class_name(model):
        raise CheckpointError("checkpoint was created for a different model class")
    if manifest_payload["model_state_schema"] != _model_state_schema(model):
        raise CheckpointError("checkpoint model-state schema is semantically incompatible")
    if optimizer is not None and manifest_payload["optimizer_class"] != _qualified_class_name(
        optimizer
    ):
        raise CheckpointError("checkpoint was created for a different optimizer class")
    if optimizer is not None and manifest_payload[
        "optimizer_parameter_names"
    ] != _optimizer_parameter_names(model, optimizer):
        raise CheckpointError(
            "optimizer parameter groups or their ordered model-parameter names "
            "differ from the checkpoint"
        )
    if provenance_payload.get("config_fingerprint") != stored_fingerprint:
        raise CheckpointError("stored provenance and configuration fingerprints differ")
    if provenance_payload.get("random_seed") != seed:
        raise CheckpointError("stored provenance and training seeds differ")
    if expected_config is not None and expected_config.fingerprint != stored_fingerprint:
        raise CheckpointError("checkpoint was created with a different configuration")
    if training_context is not None:
        if training_context.identity.config_fingerprint != stored_fingerprint:
            raise CheckpointError("training context was created for a different configuration")
        if training_context.identity.seed != seed:
            raise CheckpointError("training context was created with a different seed")
        if (
            training_context.identity.deterministic_algorithms
            != stored_identity.deterministic_algorithms
        ):
            raise CheckpointError("training context uses a different determinism mode")
        if (
            expected_scientific_manifest.catalogue.get("fingerprint")
            != training_context.crystallography_certification.catalogue_fingerprint
        ):
            raise CheckpointError(
                "training context certification and expected catalogue differ"
            )

    _validate_model_state(model, model_state, strict=strict_model)
    if optimizer is not None:
        _validate_optimizer_state(optimizer, optimizer_state)
    if restore_rng:
        if (
            bool(torch.are_deterministic_algorithms_enabled())
            != stored_identity.deterministic_algorithms
        ):
            raise CheckpointError(
                "current PyTorch determinism mode differs from the checkpoint"
            )
        _validate_global_rng_state(
            global_rng_state,
            strict_cuda=strict_cuda_rng,
            strict_mps=strict_mps_rng,
        )
        if context_state is None:
            if training_context is not None:
                raise CheckpointError("checkpoint contains no explicit training-context RNGs")
        else:
            if not isinstance(context_state, Mapping):
                raise CheckpointError("training-context RNG state must be a mapping")
            if training_context is None:
                raise CheckpointError("exact RNG resume requires the saved TrainingContext")
            _validate_training_context_state(context_state, training_context)

    optimizer_restored = optimizer is not None
    model_before = copy.deepcopy(model.state_dict()) if transactional else None
    optimizer_before = (
        copy.deepcopy(optimizer.state_dict())
        if transactional and optimizer is not None
        else None
    )
    global_rng_before = capture_global_rng_state() if transactional and restore_rng else None
    context_before = (
        training_context.state_dict()
        if transactional and restore_rng and training_context is not None
        else None
    )
    try:
        load_result = model.load_state_dict(model_state, strict=strict_model)
        if optimizer is not None:
            optimizer.load_state_dict(optimizer_state)
        if restore_rng:
            restore_global_rng_state(
                global_rng_state,
                strict_cuda=strict_cuda_rng,
                strict_mps=strict_mps_rng,
            )
            if context_state is not None and training_context is not None:
                training_context.load_state_dict(context_state)
        if model_state_fingerprint(model) != state_fingerprint:
            raise CheckpointError(
                "live model state differs from the validated checkpoint after loading"
            )
    except Exception as exc:
        rollback_failures: list[str] = []
        if transactional:
            if model_before is not None:
                try:
                    model.load_state_dict(model_before, strict=True)
                except Exception as rollback_exc:  # pragma: no cover - catastrophic hook
                    rollback_failures.append(f"model: {rollback_exc}")
            if optimizer is not None and optimizer_before is not None:
                try:
                    optimizer.load_state_dict(optimizer_before)
                except Exception as rollback_exc:  # pragma: no cover - catastrophic hook
                    rollback_failures.append(f"optimizer: {rollback_exc}")
            if global_rng_before is not None:
                try:
                    restore_global_rng_state(
                        global_rng_before,
                        strict_cuda=False,
                        strict_mps=False,
                    )
                except Exception as rollback_exc:  # pragma: no cover - platform failure
                    rollback_failures.append(f"global RNG: {rollback_exc}")
            if training_context is not None and context_before is not None:
                try:
                    training_context.load_state_dict(context_before)
                except Exception as rollback_exc:  # pragma: no cover - object corruption
                    rollback_failures.append(f"training context: {rollback_exc}")
        detail = f"; rollback failures={rollback_failures}" if rollback_failures else ""
        raise CheckpointError(f"checkpoint state application failed{detail}") from exc

    return LoadedCheckpoint(
        path=source.resolve(),
        sha256=actual_hash,
        step=step,
        epoch=epoch,
        seed=seed,
        config_fingerprint=stored_fingerprint,
        deterministic_algorithms=stored_identity.deterministic_algorithms,
        manifest_fingerprint=manifest_fingerprint,
        model_state_fingerprint=state_fingerprint,
        config=dict(config_payload),
        scientific_manifest=dict(declared_manifest),
        provenance=dict(provenance_payload),
        metadata=dict(metadata_payload),
        auxiliary_state=dict(auxiliary_state),
        missing_model_keys=tuple(load_result.missing_keys),
        unexpected_model_keys=tuple(load_result.unexpected_keys),
        optimizer_restored=optimizer_restored,
        rng_restored=restore_rng,
    )
