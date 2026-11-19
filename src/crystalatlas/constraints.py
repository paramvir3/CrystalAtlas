"""Exact arithmetic-completion constraints for crystallographic covers.

For a reduced composition ``c`` and formula-unit multiplier ``nu``, this
module implements equations (1.16)--(1.18):

``B(R, rho) = 1[rho == 0]``

``B(r, rho) = OR_a B(r + 1, rho - mu_r a)``.

Rows with zero free dimensions have capacity one.  Positive-dimensional rows
are repeatable, but their action sets remain finite because ``mu_r a <= rho``.
The implementation contains no learned feasibility predictor and uses only
nonnegative integer vectors.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from functools import lru_cache
from itertools import product

from crystalatlas.composition import Composition, SupportsRandrange


class ConstraintError(ValueError):
    """Base class for invalid row specifications and occupation states."""


class TerminalRowError(ConstraintError):
    """Raised when a next-row action is requested at the terminal state."""


class NoCompletionError(ConstraintError):
    """Raised when sampling is requested for an impossible formula/row fibre."""


def _require_int(name: str, value: object, *, minimum: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer, got {type(value).__name__}")
    if minimum is not None and value < minimum:
        raise ValueError(f"{name} must be at least {minimum}, got {value}")
    return value


@dataclass(frozen=True, slots=True, order=True)
class WyckoffRowSpec:
    """The discrete information needed by the arithmetic suffix table.

    ``multiplicity`` is the conventional Wyckoff multiplicity ``mu_gr``.
    ``free_dimension == 0`` imposes the capacity-one rule; otherwise the row
    may contain multiple independent orbits with distinct intrinsic
    parameters.
    """

    label: str
    multiplicity: int
    free_dimension: int

    def __post_init__(self) -> None:
        if not isinstance(self.label, str) or not self.label:
            raise ConstraintError("a Wyckoff row label must be a nonempty string")
        _require_int("multiplicity", self.multiplicity, minimum=1)
        dimension = _require_int("free_dimension", self.free_dimension, minimum=0)
        if dimension > 3:
            raise ConstraintError("a three-dimensional Wyckoff row has at most 3 freedoms")

    @property
    def orbit_capacity(self) -> int | None:
        """One for a fixed row and ``None`` for a repeatable free row."""

        return 1 if self.free_dimension == 0 else None


@dataclass(frozen=True, slots=True, order=True)
class RowAction:
    """An unordered species-count vector ``a_r`` aligned with ``c.support``."""

    counts: tuple[int, ...]

    def __post_init__(self) -> None:
        frozen = tuple(self.counts)
        object.__setattr__(self, "counts", frozen)
        for index, count in enumerate(frozen):
            _require_int(f"counts[{index}]", count, minimum=0)

    @property
    def number_of_orbits(self) -> int:
        return sum(self.counts)

    def __len__(self) -> int:
        return len(self.counts)


@dataclass(frozen=True, slots=True)
class OccupationTable:
    """One complete immutable sequence of row occupation actions."""

    actions: tuple[RowAction, ...]

    def __post_init__(self) -> None:
        frozen = tuple(self.actions)
        object.__setattr__(self, "actions", frozen)
        if any(not isinstance(action, RowAction) for action in frozen):
            raise TypeError("every table entry must be a RowAction")

    def __len__(self) -> int:
        return len(self.actions)

    def __iter__(self) -> Iterator[RowAction]:
        return iter(self.actions)


def _subtract_scaled(
    residual: tuple[int, ...],
    multiplicity: int,
    action: tuple[int, ...],
) -> tuple[int, ...]:
    return tuple(
        rho - multiplicity * count
        for rho, count in zip(residual, action, strict=True)
    )


@lru_cache(maxsize=262_144)
def _local_action_counts(
    row: WyckoffRowSpec,
    residual: tuple[int, ...],
) -> tuple[tuple[int, ...], ...]:
    maxima = tuple(rho // row.multiplicity for rho in residual)
    capacity = row.orbit_capacity
    if capacity == 1:
        # Do not enumerate the full product box merely to retain its origin and
        # axes.  A fixed Wyckoff row admits exactly zero or one orbit, so its
        # action set is {0} union {e_Z : rho_Z >= mu}.  The ordering matches
        # filtered ``itertools.product`` lexicographic order exactly.
        zero = (0,) * len(maxima)
        units = []
        for reverse_index, maximum in enumerate(reversed(maxima)):
            if maximum >= 1:
                species_index = len(maxima) - reverse_index - 1
                unit = [0] * len(maxima)
                unit[species_index] = 1
                units.append(tuple(unit))
        return (zero, *units)
    candidates = product(*(range(maximum + 1) for maximum in maxima))
    return tuple(candidates)


@lru_cache(maxsize=262_144)
def _suffix_completion_count(
    rows: tuple[WyckoffRowSpec, ...],
    row_index: int,
    residual: tuple[int, ...],
) -> int:
    if row_index == len(rows):
        return int(all(value == 0 for value in residual))

    row = rows[row_index]
    if row_index == len(rows) - 1:
        quotient: list[int] = []
        for value in residual:
            count, remainder = divmod(value, row.multiplicity)
            if remainder:
                return 0
            quotient.append(count)
        capacity = row.orbit_capacity
        return int(capacity is None or sum(quotient) <= capacity)

    total = 0
    for action in _local_action_counts(row, residual):
        child = _subtract_scaled(residual, row.multiplicity, action)
        total += _suffix_completion_count(rows, row_index + 1, child)
    return total


@lru_cache(maxsize=262_144)
def _scalar_unbounded_reachable(multiplicities: tuple[int, ...], target: int) -> bool:
    """Exact one-species semigroup membership for repeatable suffix rows."""

    reachable = [False] * (target + 1)
    reachable[0] = True
    for value in range(1, target + 1):
        reachable[value] = any(
            value >= multiplicity and reachable[value - multiplicity]
            for multiplicity in multiplicities
        )
    return reachable[target]


@lru_cache(maxsize=262_144)
def _suffix_is_reachable(
    rows: tuple[WyckoffRowSpec, ...],
    row_index: int,
    residual: tuple[int, ...],
) -> bool:
    """Boolean suffix oracle without computing exponentially large counts.

    Once every remaining row is repeatable, species decouple exactly and each
    residual component is a one-dimensional numerical-semigroup problem.  The
    only cross-species coupling comes from capacity-one fixed rows; their local
    action set contains just zero and at most one species unit vector.
    """

    if row_index == len(rows):
        return all(value == 0 for value in residual)
    suffix = rows[row_index:]
    if all(row.orbit_capacity is None for row in suffix):
        multiplicities = tuple(row.multiplicity for row in suffix)
        return all(
            _scalar_unbounded_reachable(multiplicities, value)
            for value in residual
        )
    row = rows[row_index]
    for action in _local_action_counts(row, residual):
        child = _subtract_scaled(residual, row.multiplicity, action)
        if _suffix_is_reachable(rows, row_index + 1, child):
            return True
    return False


@dataclass(frozen=True, slots=True)
class SuffixReachability:
    """Exact suffix-DP oracle ``B`` for one ``(c, nu, g)`` fibre.

    Public row indices are zero-based: ``row_index == len(rows)`` is the
    terminal condition corresponding to ``r = R_g + 1`` in equation (1.16).
    """

    composition: Composition
    multiplier: int
    rows: tuple[WyckoffRowSpec, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.composition, Composition):
            raise TypeError("composition must be a reduced Composition")
        _require_int("multiplier", self.multiplier, minimum=1)
        frozen_rows = tuple(self.rows)
        object.__setattr__(self, "rows", frozen_rows)
        if any(not isinstance(row, WyckoffRowSpec) for row in frozen_rows):
            raise TypeError("rows must contain only WyckoffRowSpec objects")
        labels = tuple(row.label for row in frozen_rows)
        if len(labels) != len(set(labels)):
            raise ConstraintError("explicit Wyckoff row labels must be unique")

    @property
    def species(self) -> tuple[int, ...]:
        """Element order used by every residual and action vector."""

        return self.composition.support

    @property
    def initial_residual(self) -> tuple[int, ...]:
        """The target conventional count vector ``N = nu c``."""

        return tuple(self.multiplier * value for value in self.composition.coefficients)

    def _row_index(self, row_index: int) -> int:
        index = _require_int("row_index", row_index, minimum=0)
        if index > len(self.rows):
            raise IndexError(f"row_index must not exceed {len(self.rows)}")
        return index

    def _residual(self, residual: Sequence[int] | None) -> tuple[int, ...]:
        values = self.initial_residual if residual is None else tuple(residual)
        if len(values) != len(self.species):
            raise ConstraintError(
                f"residual dimension {len(values)} does not match {len(self.species)} species"
            )
        for index, value in enumerate(values):
            _require_int(f"residual[{index}]", value, minimum=0)
        return values

    def completion_count_from(
        self,
        row_index: int = 0,
        residual: Sequence[int] | None = None,
    ) -> int:
        """Return the number of exact legal suffixes from ``(r, rho)``."""

        index = self._row_index(row_index)
        rho = self._residual(residual)
        return _suffix_completion_count(self.rows, index, rho)

    def is_reachable(
        self,
        row_index: int = 0,
        residual: Sequence[int] | None = None,
    ) -> bool:
        """Evaluate the Boolean suffix table ``B(r, rho)`` exactly."""

        index = self._row_index(row_index)
        rho = self._residual(residual)
        return _suffix_is_reachable(self.rows, index, rho)

    def local_actions(
        self,
        row_index: int,
        residual: Sequence[int],
    ) -> tuple[RowAction, ...]:
        """Enumerate ``H_gr`` intersected with ``mu_gr a <= rho``."""

        index = self._row_index(row_index)
        if index == len(self.rows):
            raise TerminalRowError("the terminal suffix state has no row action")
        rho = self._residual(residual)
        return tuple(
            RowAction(action) for action in _local_action_counts(self.rows[index], rho)
        )

    def allowed_actions(
        self,
        row_index: int,
        residual: Sequence[int],
    ) -> tuple[RowAction, ...]:
        """Return exactly the completion-safe action set in equation (1.18)."""

        index = self._row_index(row_index)
        if index == len(self.rows):
            raise TerminalRowError("the terminal suffix state has no row action")
        rho = self._residual(residual)
        row = self.rows[index]
        if index == len(self.rows) - 1:
            counts: list[int] = []
            for value in rho:
                count, remainder = divmod(value, row.multiplicity)
                if remainder:
                    return ()
                counts.append(count)
            action = RowAction(tuple(counts))
            capacity = row.orbit_capacity
            if capacity is not None and action.number_of_orbits > capacity:
                return ()
            return (action,)

        allowed: list[RowAction] = []
        for counts in _local_action_counts(row, rho):
            child = _subtract_scaled(rho, row.multiplicity, counts)
            if _suffix_is_reachable(self.rows, index + 1, child):
                allowed.append(RowAction(counts))
        return tuple(allowed)

    def advance(
        self,
        row_index: int,
        residual: Sequence[int],
        action: RowAction,
        *,
        require_completion_safe: bool = False,
    ) -> tuple[int, ...]:
        """Apply one locally legal action and return ``rho - mu_r a``."""

        index = self._row_index(row_index)
        if index == len(self.rows):
            raise TerminalRowError("cannot advance beyond the final row")
        rho = self._residual(residual)
        if not isinstance(action, RowAction):
            raise TypeError("action must be a RowAction")
        if len(action) != len(self.species):
            raise ConstraintError("row-action dimension does not match the composition")
        row = self.rows[index]
        if row.orbit_capacity is not None and action.number_of_orbits > row.orbit_capacity:
            raise ConstraintError("a zero-dimensional Wyckoff row has capacity one")
        child = _subtract_scaled(rho, row.multiplicity, action.counts)
        if any(value < 0 for value in child):
            raise ConstraintError("row action exceeds the residual formula counts")
        if require_completion_safe and not self.is_reachable(index + 1, child):
            raise ConstraintError("row action has no formula-compatible suffix")
        return child

    @property
    def number_of_completions(self) -> int:
        """Number of legal full occupation tables in this fibre."""

        return self.completion_count_from()

    def iter_suffixes(
        self,
        row_index: int = 0,
        residual: Sequence[int] | None = None,
    ) -> Iterator[tuple[RowAction, ...]]:
        """Enumerate every legal suffix exactly once in lexicographic order."""

        index = self._row_index(row_index)
        rho = self._residual(residual)
        if index == len(self.rows):
            if all(value == 0 for value in rho):
                yield ()
            return

        for action in self.allowed_actions(index, rho):
            child = self.advance(index, rho, action)
            for suffix in self.iter_suffixes(index + 1, child):
                yield (action,) + suffix

    def iter_completions(self) -> Iterator[OccupationTable]:
        """Enumerate every formula-compatible complete row table."""

        for actions in self.iter_suffixes():
            yield OccupationTable(actions)

    def completion_at(self, index: int) -> OccupationTable:
        """Unrank one complete table using exact suffix cardinalities."""

        rank = _require_int("index", index, minimum=0)
        total = self.number_of_completions
        if rank >= total:
            raise IndexError(f"completion index {rank} is outside range({total})")

        row_index = 0
        residual = self.initial_residual
        selected: list[RowAction] = []
        while row_index < len(self.rows):
            row = self.rows[row_index]
            for action in self.allowed_actions(row_index, residual):
                child = _subtract_scaled(residual, row.multiplicity, action.counts)
                block = _suffix_completion_count(self.rows, row_index + 1, child)
                if rank < block:
                    selected.append(action)
                    residual = child
                    row_index += 1
                    break
                rank -= block
            else:  # pragma: no cover - guarded by the positive total above
                raise AssertionError("reachable row state had no completion-safe action")

        if any(residual):  # pragma: no cover - follows from terminal DP condition
            raise AssertionError("completion unranking ended with a nonzero residual")
        return OccupationTable(tuple(selected))

    def sample_uniform(self, rng: SupportsRandrange) -> OccupationTable:
        """Draw exactly uniformly from all formula-compatible row tables."""

        if not hasattr(rng, "randrange"):
            raise TypeError("rng must provide randrange(stop)")
        total = self.number_of_completions
        if total == 0:
            raise NoCompletionError("the formula has no completion on these rows")
        index = rng.randrange(total)
        if isinstance(index, bool) or not isinstance(index, int) or not 0 <= index < total:
            raise ValueError("rng.randrange(stop) returned a value outside range(stop)")
        return self.completion_at(index)

    def accepts(self, table: OccupationTable) -> bool:
        """Return whether a table satisfies every local and global equation."""

        if not isinstance(table, OccupationTable) or len(table) != len(self.rows):
            return False
        residual = self.initial_residual
        try:
            for row_index, action in enumerate(table):
                residual = self.advance(row_index, residual, action)
        except (ConstraintError, TypeError):
            return False
        return all(value == 0 for value in residual)

    def expanded_counts(self, table: OccupationTable) -> tuple[int, ...]:
        """Compute ``sum_r mu_r a_r`` for an accepted full table."""

        if not self.accepts(table):
            raise ConstraintError("occupation table is not formula-compatible")
        totals = [0] * len(self.species)
        for row, action in zip(self.rows, table.actions, strict=True):
            for index, count in enumerate(action.counts):
                totals[index] += row.multiplicity * count
        return tuple(totals)


__all__ = [
    "ConstraintError",
    "NoCompletionError",
    "OccupationTable",
    "RowAction",
    "SuffixReachability",
    "TerminalRowError",
    "WyckoffRowSpec",
]
