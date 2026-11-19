"""Deterministic fixed-budget candidate controller from equations (3.16b–c)."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from .errors import DomainError, NumericalError


@dataclass(frozen=True, slots=True)
class CandidateScore:
    """Finite surrogate information used only for budget allocation."""

    identifier: str
    hull_mean: float
    hull_std: float
    novelty_distance: float

    def __post_init__(self) -> None:
        if not self.identifier:
            raise DomainError("candidate identifier must be nonempty")
        if any(
            not math.isfinite(value)
            for value in (self.hull_mean, self.hull_std, self.novelty_distance)
        ):
            raise DomainError("candidate scores must be finite")
        if self.hull_std < 0.0 or self.novelty_distance < 0.0:
            raise DomainError("uncertainty and novelty distance must be nonnegative")


@dataclass(frozen=True, slots=True)
class ControllerConfig:
    uncertainty_weight: float = 0.0
    novelty_weight: float = 0.0
    diversity_weight: float = 0.0
    kernel_scale: float = 1.0
    psd_tolerance: float = 1.0e-9

    def __post_init__(self) -> None:
        for name in (
            "uncertainty_weight",
            "novelty_weight",
            "diversity_weight",
            "kernel_scale",
            "psd_tolerance",
        ):
            value = getattr(self, name)
            if not math.isfinite(value) or value < 0.0:
                raise DomainError(f"{name} must be finite and nonnegative")


def _validate_kernel(kernel: npt.ArrayLike, count: int, tolerance: float) -> np.ndarray:
    matrix = np.asarray(kernel, dtype=np.float64)
    if matrix.shape != (count, count) or not np.all(np.isfinite(matrix)):
        raise DomainError("similarity kernel has the wrong shape or nonfinite entries")
    if not np.allclose(matrix, matrix.T, atol=tolerance, rtol=0.0):
        raise DomainError("similarity kernel must be symmetric")
    if not np.allclose(np.diag(matrix), 1.0, atol=tolerance, rtol=0.0):
        raise DomainError("similarity kernel must be normalized by K_ii=1")
    if count == 0:
        # The empty Gram matrix is PSD; no eigenvalue exists to index.
        return matrix
    eigenvalues = np.linalg.eigvalsh(matrix)
    if float(eigenvalues[0]) < -tolerance:
        raise DomainError("similarity kernel must be positive semidefinite")
    return matrix


def _utility(
    indices: Sequence[int],
    candidates: Sequence[CandidateScore],
    kernel: np.ndarray,
    config: ControllerConfig,
) -> float:
    if not indices:
        return 0.0
    point_value = sum(
        -(
            candidates[index].hull_mean
            - config.uncertainty_weight * candidates[index].hull_std
        )
        + config.novelty_weight * candidates[index].novelty_distance
        for index in indices
    )
    subkernel = kernel[np.ix_(indices, indices)]
    sign, logdet = np.linalg.slogdet(
        np.eye(len(indices), dtype=np.float64) + config.kernel_scale * subkernel
    )
    if sign <= 0.0 or not math.isfinite(float(logdet)):
        raise NumericalError("I + eta K_A is not positive definite numerically")
    return point_value + config.diversity_weight * float(logdet)


def greedy_select(
    candidates: Sequence[CandidateScore],
    kernel: npt.ArrayLike,
    requested: int,
    *,
    config: ControllerConfig | None = None,
) -> tuple[CandidateScore, ...]:
    """Apply the frozen greedy approximation with identifier tie-breaking.

    The function returns fewer than ``requested`` entries when fewer candidates
    have finite scores.  Callers must report missing expensive-evaluation slots;
    this function never creates replacements.
    """

    config = ControllerConfig() if config is None else config
    if isinstance(requested, bool) or not isinstance(requested, int) or requested < 0:
        raise DomainError("requested candidate count must be a nonnegative integer")
    if len({candidate.identifier for candidate in candidates}) != len(candidates):
        raise DomainError("candidate identifiers must be unique")
    matrix = _validate_kernel(kernel, len(candidates), config.psd_tolerance)
    target = min(requested, len(candidates))
    selected: list[int] = []
    remaining = set(range(len(candidates)))
    current = 0.0
    for _ in range(target):
        scored: list[tuple[float, str, int, float]] = []
        for index in remaining:
            candidate_utility = _utility(selected + [index], candidates, matrix, config)
            scored.append(
                (
                    candidate_utility - current,
                    candidates[index].identifier,
                    index,
                    candidate_utility,
                )
            )
        # Maximize marginal gain; lexicographically smallest hash wins exact ties.
        best_gain = max(item[0] for item in scored)
        tied = [
            item
            for item in scored
            if math.isclose(item[0], best_gain, rel_tol=0.0, abs_tol=1.0e-14)
        ]
        _, _, chosen, new_utility = min(tied, key=lambda item: item[1])
        selected.append(chosen)
        remaining.remove(chosen)
        current = new_utility
    return tuple(candidates[index] for index in selected)
