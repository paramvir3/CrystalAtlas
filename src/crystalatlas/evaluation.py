"""Architecture-neutral, fixed-budget evaluation primitives.

This module deliberately records failed attempts and dimensionally distinct
resources.  It does not provide a favourable scalar conversion between GPU,
CPU, relaxation, and DFT costs.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum

import numpy as np
import numpy.typing as npt

from .errors import DomainError


@dataclass(frozen=True, slots=True)
class CostVector:
    """Dimensionally typed resource accounting from proof obligation E3."""

    gpu_hours: float = 0.0
    cpu_hours: float = 0.0
    ml_relaxer_calls: int = 0
    dft_submissions: int = 0
    dft_core_hours: float = 0.0

    def __post_init__(self) -> None:
        real_values = (self.gpu_hours, self.cpu_hours, self.dft_core_hours)
        if any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            or value < 0.0
            for value in real_values
        ):
            raise DomainError("continuous costs must be finite and nonnegative")
        if any(
            isinstance(value, bool) or not isinstance(value, int) or value < 0
            for value in (self.ml_relaxer_calls, self.dft_submissions)
        ):
            raise DomainError("call counts must be nonnegative integers")

    def __add__(self, other: CostVector) -> CostVector:
        if not isinstance(other, CostVector):
            return NotImplemented
        return CostVector(
            gpu_hours=self.gpu_hours + other.gpu_hours,
            cpu_hours=self.cpu_hours + other.cpu_hours,
            ml_relaxer_calls=self.ml_relaxer_calls + other.ml_relaxer_calls,
            dft_submissions=self.dft_submissions + other.dft_submissions,
            dft_core_hours=self.dft_core_hours + other.dft_core_hours,
        )

    def fits_within(self, budget: CostVector, *, atol: float = 1.0e-12) -> bool:
        if not isinstance(budget, CostVector):
            raise TypeError("budget must be a CostVector")
        if (
            isinstance(atol, bool)
            or not isinstance(atol, (int, float))
            or not math.isfinite(atol)
            or atol < 0.0
        ):
            raise DomainError("cost comparison tolerance must be finite and nonnegative")
        return (
            self.gpu_hours <= budget.gpu_hours + atol
            and self.cpu_hours <= budget.cpu_hours + atol
            and self.ml_relaxer_calls <= budget.ml_relaxer_calls
            and self.dft_submissions <= budget.dft_submissions
            and self.dft_core_hours <= budget.dft_core_hours + atol
        )


class AttemptStatus(str, Enum):  # noqa: UP042 - ML smoke runtime is Python 3.10
    GENERATED = "generated"
    GENERATION_FAILED = "generation_failed"
    INVALID = "invalid"
    DUPLICATE = "duplicate"
    SURROGATE_FAILED = "surrogate_failed"
    DFT_FAILED = "dft_failed"
    DFT_CONVERGED = "dft_converged"


@dataclass(frozen=True, slots=True)
class ProtocolAction:
    """One auditable action in an end-to-end discovery trace."""

    name: str
    cost: CostVector
    candidate_id: str | None = None
    status: AttemptStatus | None = None
    metadata: Mapping[str, str | int | float | bool | None] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name:
            raise DomainError("protocol action name must be nonempty text")
        if not isinstance(self.cost, CostVector):
            raise TypeError("protocol action cost must be a CostVector")
        if self.candidate_id is not None and (
            not isinstance(self.candidate_id, str) or not self.candidate_id
        ):
            raise DomainError("candidate identifier must be nonempty text or None")
        if self.status is not None and not isinstance(self.status, AttemptStatus):
            raise TypeError("protocol action status must be an AttemptStatus or None")


@dataclass(frozen=True, slots=True, init=False)
class ProtocolTrace:
    """Append-only history with a frozen budget and fully replayed initial cost.

    Public history is a tuple snapshot. Only ``append`` can advance the history
    and its total together; supplying completed actions never resets their cost.
    """

    budget: CostVector
    _actions: tuple[ProtocolAction, ...] = field(repr=False)
    _cost: CostVector = field(repr=False)

    def __init__(self, budget: CostVector, actions: Sequence[ProtocolAction] = ()) -> None:
        if not isinstance(budget, CostVector):
            raise TypeError("budget must be a CostVector")
        history = tuple(actions)
        cost = CostVector()
        for action in history:
            if not isinstance(action, ProtocolAction):
                raise TypeError("trace history must contain ProtocolAction values")
            cost = cost + action.cost
            if not cost.fits_within(budget):
                raise DomainError("initial protocol actions exceed the frozen budget")
        object.__setattr__(self, "budget", budget)
        object.__setattr__(self, "_actions", history)
        object.__setattr__(self, "_cost", cost)

    @property
    def actions(self) -> tuple[ProtocolAction, ...]:
        return self._actions

    @property
    def cost(self) -> CostVector:
        return self._cost

    def can_append(self, action: ProtocolAction) -> bool:
        if not isinstance(action, ProtocolAction):
            raise TypeError("action must be a ProtocolAction")
        return (self._cost + action.cost).fits_within(self.budget)

    def append(self, action: ProtocolAction) -> None:
        if not isinstance(action, ProtocolAction):
            raise TypeError("action must be a ProtocolAction")
        candidate_cost = self._cost + action.cost
        if not candidate_cost.fits_within(self.budget):
            raise DomainError("the next protocol action would exceed the frozen budget")
        history = (*self._actions, action)
        object.__setattr__(self, "_actions", history)
        object.__setattr__(self, "_cost", candidate_cost)

    def status_counts(self) -> dict[AttemptStatus, int]:
        counts = {status: 0 for status in AttemptStatus}
        for action in self.actions:
            if action.status is not None:
                counts[action.status] += 1
        return counts


@dataclass(frozen=True, slots=True)
class PhaseReference:
    """Uniformly recomputed phase used in a zero-K static-enthalpy hull."""

    identifier: str
    fractions: tuple[float, ...]
    enthalpy_per_atom: float

    def __post_init__(self) -> None:
        if not self.fractions:
            raise DomainError("phase fractions must be nonempty")
        if any(not math.isfinite(value) or value < 0.0 for value in self.fractions):
            raise DomainError("phase fractions must be finite and nonnegative")
        if not math.isclose(sum(self.fractions), 1.0, rel_tol=0.0, abs_tol=1.0e-10):
            raise DomainError("phase fractions must sum to one")
        if not math.isfinite(self.enthalpy_per_atom):
            raise DomainError("phase enthalpy must be finite")


def _small_hull_energy(
    query: npt.NDArray[np.float64], references: Sequence[PhaseReference]
) -> float:
    """Exhaustive-subset, floating-point fallback for small reference sets.

    In an ``E``-element simplex, a feasible point needs at most ``E`` phases.
    Enumerating those subsets is appropriate for unit tests and small systems;
    large production reference sets should install SciPy and use linear
    programming through :func:`hull_energy`.
    """

    from itertools import combinations

    element_count = query.size
    best = math.inf
    for subset_size in range(1, min(element_count, len(references)) + 1):
        for subset in combinations(references, subset_size):
            matrix = np.vstack([reference.fractions for reference in subset]).T
            # The sum-to-one equation is redundant with complete fractions but
            # improves validation under finite arithmetic.
            augmented = np.vstack([matrix, np.ones((1, subset_size))])
            target = np.concatenate([query, np.ones(1)])
            weights, *_ = np.linalg.lstsq(augmented, target, rcond=None)
            if np.any(~np.isfinite(weights)):
                raise DomainError("hull fallback returned nonfinite phase weights")
            if np.linalg.norm(augmented @ weights - target) > 1.0e-9:
                continue
            if np.min(weights) < -1.0e-10:
                continue
            weights = np.maximum(weights, 0.0)
            weights /= weights.sum()
            # The repaired weights, not the original least-squares solution,
            # are used in the energy. Recheck that exact vector's feasibility.
            if np.linalg.norm(augmented @ weights - target) > 1.0e-9:
                continue
            terms = [
                float(weight) * reference.enthalpy_per_atom
                for weight, reference in zip(weights, subset, strict=True)
            ]
            try:
                energy = math.fsum(terms)
            except OverflowError as exc:
                raise DomainError("hull fallback energy reconstruction overflowed") from exc
            if not math.isfinite(energy):
                raise DomainError("hull fallback energy is nonfinite")
            best = min(best, energy)
    if not math.isfinite(best):
        raise DomainError("the reference phases do not span the query composition")
    return best


def hull_energy(
    query_fractions: Sequence[float],
    references: Sequence[PhaseReference],
    *,
    exhaustive_limit: int = 128,
    exhaustive_subset_limit: int = 100_000,
) -> float:
    """Evaluate the common zero-K static-enthalpy hull functional.

    SciPy's HiGHS linear program is used when installed.  The deterministic
    finite-subset fallback is deliberately bounded to prevent accidental
    combinatorial runs. ``exhaustive_limit`` caps the reference count and
    ``exhaustive_subset_limit`` caps the actual number of candidate subsets;
    the latter matters especially for multi-element systems. Both limits
    apply only to the fallback, but must always be positive Python integers.
    """

    for limit in (exhaustive_limit, exhaustive_subset_limit):
        if type(limit) is not int or limit <= 0:
            raise DomainError("hull fallback limits must be positive integers")
    query = np.asarray(query_fractions, dtype=np.float64)
    if query.ndim != 1 or query.size == 0:
        raise DomainError("query_fractions must be a nonempty vector")
    if len(references) == 0:
        raise DomainError("at least one phase reference is required")
    if any(len(reference.fractions) != query.size for reference in references):
        raise DomainError("all phase references must use the same element basis")
    # Use the same absolute simplex tolerance as PhaseReference. NumPy's
    # default relative tolerance would admit a composition error near 1e-5,
    # changing the mass-balance problem before either solver sees it.
    if (
        np.any(~np.isfinite(query))
        or np.any(query < 0.0)
        or not math.isclose(math.fsum(query), 1.0, rel_tol=0.0, abs_tol=1.0e-10)
    ):
        raise DomainError("query fractions must be finite, nonnegative and sum to one")
    try:
        from scipy.optimize import linprog  # type: ignore[import-not-found]
    except ImportError:
        if len(references) > exhaustive_limit:
            raise DomainError(
                "large hull evaluation requires SciPy; refusing combinatorial fallback"
            ) from None
        # Counting just phases does not bound the work: even 32 references
        # in an eight-element system require over 15 million subsets.
        subset_count = 0
        for size in range(1, min(query.size, len(references)) + 1):
            subset_count += math.comb(len(references), size)
            if subset_count > exhaustive_subset_limit:
                raise DomainError(
                    "hull subset budget exceeded; install SciPy or explicitly "
                    "increase exhaustive_subset_limit"
                ) from None
        return _small_hull_energy(query, references)

    fractions = np.vstack([reference.fractions for reference in references]).T
    objective = np.asarray([reference.enthalpy_per_atom for reference in references])
    equality_matrix = np.vstack([fractions, np.ones((1, len(references)))])
    equality_target = np.concatenate([query, np.ones(1)])
    result = linprog(
        objective,
        A_eq=equality_matrix,
        b_eq=equality_target,
        bounds=(0.0, None),
        method="highs",
    )
    if not result.success:
        raise DomainError(f"hull linear program failed: {result.message}")
    # A solver success flag is not a mass-balance check. Do not clip or
    # renormalize its weights: either operation changes the solved problem.
    weights = np.asarray(result.x, dtype=np.float64)
    if (
        weights.shape != objective.shape
        or np.any(~np.isfinite(weights))
        or np.any(weights < -1.0e-10)
    ):
        raise DomainError("hull solver returned invalid phase weights")
    residual = equality_matrix @ weights - equality_target
    if np.any(~np.isfinite(residual)) or np.max(np.abs(residual)) > 1.0e-9:
        raise DomainError("hull solver weights violate mass balance or normalization")
    reported_energy = float(result.fun)
    with np.errstate(over="ignore", invalid="ignore"):
        terms = weights * objective
        absolute_scale = float(np.sum(np.abs(terms)))
    if np.any(~np.isfinite(terms)) or not math.isfinite(absolute_scale):
        raise DomainError("hull solver energy reconstruction overflowed")
    reconstructed_energy = math.fsum(terms)
    # Account for summation roundoff using absolute terms, including when
    # large positive and negative reference energies nearly cancel.
    energy_tolerance = 1.0e-10 + 64 * np.finfo(np.float64).eps * absolute_scale
    if (
        not math.isfinite(reported_energy)
        or abs(reported_energy - reconstructed_energy) > energy_tolerance
    ):
        raise DomainError("hull solver objective disagrees with its phase weights")
    return reconstructed_energy


def energy_above_hull(
    query_fractions: Sequence[float],
    enthalpy_per_atom: float,
    references: Sequence[PhaseReference],
) -> float:
    """Return ``h(query) - H_R(query)`` without clipping negative discoveries."""

    if not math.isfinite(enthalpy_per_atom):
        raise DomainError("query enthalpy must be finite")
    return enthalpy_per_atom - hull_energy(query_fractions, references)


def paired_cluster_bootstrap(
    differences_by_system: Sequence[float],
    *,
    samples: int = 10_000,
    confidence: float = 0.95,
    seed: int = 0,
    batch_size: int = 256,
) -> tuple[float, float, float]:
    """Percentile interval for the equally weighted mean paired system effect.

    Supply one paired aggregate difference per independent chemical system,
    not one entry per structure. Resample systems uniformly with replacement.
    Batching bounds index/sample workspace to O(batch_size * system_count),
    while retaining O(samples) means for the percentile calculation.
    """

    values = np.asarray(differences_by_system, dtype=np.float64)
    if values.ndim != 1 or values.size < 2 or np.any(~np.isfinite(values)):
        raise DomainError("at least two finite paired system differences are required")
    if type(samples) is not int or samples < 100:
        raise DomainError("bootstrap sample count must be an integer at least 100")
    if type(batch_size) is not int or batch_size <= 0:
        raise DomainError("bootstrap batch_size must be a positive integer")
    if type(seed) is not int or seed < 0:
        raise DomainError("bootstrap seed must be a nonnegative integer")
    if not 0.0 < confidence < 1.0:
        raise DomainError("confidence must lie strictly between zero and one")
    rng = np.random.default_rng(seed)
    estimates = np.empty(samples, dtype=np.float64)
    for start in range(0, samples, batch_size):
        stop = min(start + batch_size, samples)
        indices = rng.integers(0, values.size, size=(stop - start, values.size))
        with np.errstate(over="ignore", invalid="ignore"):
            estimates[start:stop] = values[indices].mean(axis=1)
    with np.errstate(over="ignore", invalid="ignore"):
        mean = float(values.mean())
    if not math.isfinite(mean) or np.any(~np.isfinite(estimates)):
        raise DomainError("bootstrap mean calculation overflowed")
    tail = (1.0 - confidence) / 2.0
    lower, upper = np.quantile(estimates, [tail, 1.0 - tail])
    if not math.isfinite(lower) or not math.isfinite(upper):
        raise DomainError("bootstrap percentile calculation overflowed")
    return mean, float(lower), float(upper)
