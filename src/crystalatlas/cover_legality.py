"""Atom-bound legality of arithmetic covers, shared by sampler and targets.

The cover sampler's masks and the training targets' masks define the support
on which the selector's likelihood is normalized.  Both call the functions
here, so the two supports cannot diverge.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from fractions import Fraction
from typing import TYPE_CHECKING, Protocol

import numpy as np

from .errors import DomainError, InvariantViolation

if TYPE_CHECKING:
    from .catalogue import CertifiedCatalogue, SpaceGroupSetting
    from .composition import Composition
    from .geometry import ContinuousCover


class _Reachability(Protocol):
    def is_reachable(self) -> bool: ...


Oracle = Callable[["Composition", int, "SpaceGroupSetting"], _Reachability]


@dataclass(frozen=True, slots=True)
class AtomBound:
    """Conventional resource cap plus an optional primitive-cell atom bound.

    ``primitive=None`` is the legacy rule ``nu |c| <= conventional``.  Otherwise
    a cover is admitted iff ``nu |c| <= min(conventional, primitive * I_G)``.
    Canonical targets carry no translations beyond the centring of ``G``, so the
    primitive cell holds exactly ``nu |c| / I_G`` atoms.
    """

    conventional: int
    primitive: int | None = None

    def __post_init__(self) -> None:
        for name, value in (("conventional", self.conventional), ("primitive", self.primitive)):
            if value is None and name == "primitive":
                continue
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise DomainError(f"{name} atom bound must be a positive integer")

    def capacity(self, centering_index: int) -> int:
        """Largest admitted conventional atom count for a centring index."""

        if centering_index < 1:
            raise DomainError("centring index must be positive")
        if self.primitive is None:
            return self.conventional
        return min(self.conventional, self.primitive * centering_index)

    def admits(self, nu: int, formula_atoms: int, centering_index: int) -> bool:
        if nu < 1 or formula_atoms < 1:
            raise DomainError("cover multiplicities must be positive")
        return nu * formula_atoms <= self.capacity(centering_index)


@dataclass(frozen=True, slots=True)
class PrimitiveCountCondition:
    """Condition on the primitive atom count ``N_prim`` (review M6).

    A canonical cover has no translations beyond the centring of ``G``, so it
    holds ``nu |c| / I_G`` primitive atoms; tables whose generic hidden index
    exceeds one are excluded separately (see
    :func:`generic_hidden_translation_index`).
    """

    primitive_atoms: int

    def __post_init__(self) -> None:
        value = self.primitive_atoms
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise DomainError("primitive_atoms must be a positive integer")

    def admits(self, nu: int, formula_atoms: int, centering_index: int) -> bool:
        return nu * formula_atoms == self.primitive_atoms * centering_index


def _legal(
    bound: AtomBound,
    condition: PrimitiveCountCondition | None,
    nu: int,
    formula_atoms: int,
    centering_index: int,
) -> bool:
    return bound.admits(nu, formula_atoms, centering_index) and (
        condition is None or condition.admits(nu, formula_atoms, centering_index)
    )


def legal_settings(
    composition: Composition,
    nu: int,
    settings: Sequence[SpaceGroupSetting],
    *,
    bound: AtomBound,
    oracle: Oracle,
    condition: PrimitiveCountCondition | None = None,
) -> tuple[SpaceGroupSetting, ...]:
    """Settings that admit ``nu`` formula units and have an exact Wyckoff cover."""

    return tuple(
        setting
        for setting in settings
        if _legal(bound, condition, nu, composition.total_atoms, setting.centering_index)
        and oracle(composition, nu, setting).is_reachable()
    )


def feasible_formula_units(
    composition: Composition,
    settings: Sequence[SpaceGroupSetting],
    *,
    max_formula_units: int,
    bound: AtomBound,
    oracle: Oracle,
    condition: PrimitiveCountCondition | None = None,
) -> tuple[int, ...]:
    """Multipliers ``nu`` for which at least one setting is legal."""

    return tuple(
        nu
        for nu in range(1, max_formula_units + 1)
        if any(
            _legal(bound, condition, nu, composition.total_atoms, setting.centering_index)
            and oracle(composition, nu, setting).is_reachable()
            for setting in settings
        )
    )


def generic_hidden_translation_index(
    cover: ContinuousCover, catalogue: CertifiedCatalogue, *, attempts: int = 3
) -> int:
    """Hidden translation index of a cover table at generic parameters.

    Free orbits get distinct rational offsets of their row's certified generic
    parameter, and the exact decoder counts translations beyond the centring.
    The index can only grow at special parameters, so the minimum over the
    attempts is an upper bound on the generic index, and it equals the generic
    index whenever it is one.  An attempt whose parameters specialize an orbit
    is skipped.
    """

    from .decoder import CrystalScaffold, ExactCrystalDecoder, OrbitAssignment

    setting = catalogue.get(cover.space_group)
    metric = np.asarray(setting.reference_metric, dtype=np.float64)
    decoder = ExactCrystalDecoder(catalogue)
    best: int | None = None
    for attempt in range(attempts):
        used: dict[str, int] = {}
        assignments = []
        for orbit in cover.orbits:
            row = setting.wyckoff(orbit.wyckoff_label)
            slot = used.get(orbit.wyckoff_label, 0)
            used[orbit.wyckoff_label] = slot + 1
            parameter = tuple(
                Fraction(value) + Fraction(1 + axis + 3 * slot + 11 * attempt, 997)
                for axis, value in enumerate(row.generic_parameter)
            )
            assignments.append(
                OrbitAssignment(orbit.wyckoff_label, orbit.atomic_number, parameter)
            )
        try:
            scaffold = CrystalScaffold.from_mapping(
                nu=cover.nu,
                space_group=cover.space_group,
                reduced_composition=cover.composition.as_dict(),
                orbits=assignments,
            )
            index = decoder.decode(scaffold, metric).hidden_translation_index
        except (DomainError, InvariantViolation):
            continue
        best = index if best is None else min(best, index)
        if best == 1:
            return 1
    if best is None:
        raise InvariantViolation("no generic parameter choice decoded this cover table")
    return best
