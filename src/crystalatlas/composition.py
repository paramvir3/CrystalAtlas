"""Exact reduced-composition objects and the DNG/S-DNG token grammar.

This module implements the finite language in equations (0.1a), (0.1b), and
(1.20a)--(1.20d) of the mathematical specification.  The neural network is
deliberately absent: it may score only the tokens returned by
``CompositionGrammar.allowed_next``.

All support decisions and all sampling decisions in this module use exact
integer arithmetic.  In particular, ``sample_uniform`` samples leaves rather
than choosing uniformly among local children; child probabilities are weighted
by exact subtree cardinalities.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from functools import cache
from math import comb, gcd
from typing import Protocol

Element = int
"""An element identifier.

The package uses the positive atomic number as the canonical identifier.  A
different element embedding is a learned concern and must not alter this exact
discrete key.
"""

MAX_ATOMIC_NUMBER = 118


class CompositionError(ValueError):
    """Base class for invalid compositions and composition domains."""


class InvalidPrefixError(CompositionError):
    """Raised when a token sequence is not a live grammar prefix."""


class TerminalPrefixError(CompositionError):
    """Raised when next-token actions are requested after EOS."""


class EmptyCompositionDomainError(CompositionError):
    """Raised when sampling is requested from an empty formula fibre."""


class SupportsRandrange(Protocol):
    """Minimal deterministic-RNG interface used by exact uniform samplers."""

    def randrange(self, stop: int, /) -> int:
        """Return an integer in ``range(stop)``."""


def _require_int(
    name: str,
    value: object,
    *,
    minimum: int | None = None,
    maximum: int | None = None,
) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer, got {type(value).__name__}")
    if minimum is not None and value < minimum:
        raise CompositionError(f"{name} must be at least {minimum}, got {value}")
    if maximum is not None and value > maximum:
        raise CompositionError(f"{name} must be at most {maximum}, got {value}")
    return value


def _positive_gcd(values: Iterable[int]) -> int:
    result = 0
    for value in values:
        if value > 0:
            result = gcd(result, value)
    return result


@dataclass(frozen=True, slots=True, order=True)
class Composition:
    """A nonzero canonical reduced composition.

    ``counts`` is a strictly atomic-number-ordered tuple of ``(Z, c_Z)``
    pairs.  Every coefficient is positive and their gcd is one.  Direct
    construction is strict; use :meth:`from_counts` to sort, combine duplicate
    entries, remove explicit zeros, and reduce a raw formula.
    """

    counts: tuple[tuple[Element, int], ...]

    def __post_init__(self) -> None:
        frozen_list: list[tuple[object, object]] = []
        try:
            for pair in self.counts:
                element, coefficient = pair
                frozen_list.append((element, coefficient))
        except (TypeError, ValueError) as exc:
            raise TypeError("counts must contain exact (element, coefficient) pairs") from exc
        frozen_counts = tuple(frozen_list)
        object.__setattr__(self, "counts", frozen_counts)

        if not frozen_counts:
            raise CompositionError("a composition must contain at least one element")

        previous = 0
        coefficients: list[int] = []
        for index, (element, coefficient) in enumerate(frozen_counts):
            z = _require_int(
                f"counts[{index}].element",
                element,
                minimum=1,
                maximum=MAX_ATOMIC_NUMBER,
            )
            n = _require_int(f"counts[{index}].coefficient", coefficient, minimum=1)
            if z <= previous:
                raise CompositionError(
                    "elements must be unique and strictly increasing atomic numbers"
                )
            previous = z
            coefficients.append(n)

        divisor = _positive_gcd(coefficients)
        if divisor != 1:
            raise CompositionError(
                f"composition is not reduced: coefficient gcd is {divisor}, not 1"
            )

    @classmethod
    def from_counts(
        cls,
        counts: Mapping[Element, int] | Iterable[tuple[Element, int]],
    ) -> Composition:
        """Canonicalize arbitrary nonnegative integer counts.

        Duplicate entries in an iterable are added, explicit zeros are
        removed, and the remaining vector is divided by its positive gcd.  The
        result is invariant under input permutation and positive common
        scaling: ``red(m c) == red(c)`` for every positive integer ``m``.
        """

        entries = counts.items() if isinstance(counts, Mapping) else counts
        accumulated: dict[Element, int] = {}
        for index, entry in enumerate(entries):
            try:
                element, coefficient = entry
            except (TypeError, ValueError) as exc:
                raise TypeError(
                    f"counts entry {index} must be an (element, coefficient) pair"
                ) from exc
            z = _require_int(
                f"counts[{index}].element",
                element,
                minimum=1,
                maximum=MAX_ATOMIC_NUMBER,
            )
            n = _require_int(f"counts[{index}].coefficient", coefficient, minimum=0)
            accumulated[z] = accumulated.get(z, 0) + n

        positive = tuple(sorted((z, n) for z, n in accumulated.items() if n > 0))
        if not positive:
            raise CompositionError("a raw composition must have a positive count")
        divisor = _positive_gcd(n for _, n in positive)
        return cls(tuple((z, n // divisor) for z, n in positive))

    @property
    def support(self) -> tuple[Element, ...]:
        """The strictly increasing all-and-only element support."""

        return tuple(z for z, _ in self.counts)

    @property
    def coefficients(self) -> tuple[int, ...]:
        """Reduced coefficients aligned with :attr:`support`."""

        return tuple(n for _, n in self.counts)

    @property
    def total_atoms(self) -> int:
        """The atom count in one reduced formula unit."""

        return sum(self.coefficients)

    def coefficient(self, element: Element) -> int:
        """Return ``c_Z`` (zero when ``element`` is outside the support)."""

        z = _require_int("element", element, minimum=1)
        for present, count in self.counts:
            if present == z:
                return count
            if present > z:
                break
        return 0

    def scaled_counts(self, multiplier: int) -> tuple[tuple[Element, int], ...]:
        """Return the unreduced exact vector ``multiplier * c``."""

        m = _require_int("multiplier", multiplier, minimum=1)
        return tuple((z, m * n) for z, n in self.counts)

    def as_dict(self) -> dict[Element, int]:
        """Return a mutable copy suitable for serialization boundaries."""

        return dict(self.counts)

    def __len__(self) -> int:
        return len(self.counts)


@dataclass(frozen=True, slots=True)
class CompositionDomain:
    """Frozen finite support ``C_DNG`` from equation (0.1a)."""

    elements: tuple[Element, ...]
    max_elements: int
    max_atoms: int

    def __post_init__(self) -> None:
        frozen_elements = tuple(self.elements)
        object.__setattr__(self, "elements", frozen_elements)
        _require_int("max_elements", self.max_elements, minimum=1)
        _require_int("max_atoms", self.max_atoms, minimum=1)
        if not frozen_elements:
            raise CompositionError("the DNG element alphabet must be nonempty")

        previous = 0
        for index, element in enumerate(frozen_elements):
            z = _require_int(
                f"elements[{index}]",
                element,
                minimum=1,
                maximum=MAX_ATOMIC_NUMBER,
            )
            if z <= previous:
                raise CompositionError(
                    "the DNG element alphabet must be strictly increasing and unique"
                )
            previous = z

    @property
    def max_support_size(self) -> int:
        """Largest support cardinality that can have positive coefficients."""

        return min(self.max_elements, self.max_atoms, len(self.elements))

    def contains(self, composition: Composition) -> bool:
        """Return whether ``composition`` lies in the frozen DNG domain."""

        if not isinstance(composition, Composition):
            return False
        return (
            1 <= len(composition) <= self.max_elements
            and composition.total_atoms <= self.max_atoms
            and all(z in self.elements for z in composition.support)
        )

    def require_contains(self, composition: Composition) -> None:
        if not isinstance(composition, Composition):
            raise TypeError("composition must be a Composition")
        if not self.contains(composition):
            raise CompositionError(
                f"composition {composition.counts!r} lies outside the frozen DNG domain"
            )


class TokenKind(str, Enum):  # noqa: UP042 - ML smoke runtime is Python 3.10
    """Disjoint roles in the flat support-first serialization."""

    SUPPORT_SIZE = "K"
    ELEMENT = "Z"
    COEFFICIENT = "c"
    EOS = "EOS"


@dataclass(frozen=True, slots=True)
class CompositionToken:
    """A typed formula token; integer values cannot collide across roles."""

    kind: TokenKind
    value: int | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.kind, TokenKind):
            raise TypeError("kind must be a TokenKind")
        if self.kind is TokenKind.EOS:
            if self.value is not None:
                raise CompositionError("EOS must not carry an integer value")
            return
        _require_int(f"{self.kind.value} token value", self.value, minimum=1)

    @classmethod
    def support_size(cls, size: int) -> CompositionToken:
        return cls(TokenKind.SUPPORT_SIZE, size)

    @classmethod
    def element(cls, atomic_number: Element) -> CompositionToken:
        _require_int(
            "atomic_number",
            atomic_number,
            minimum=1,
            maximum=MAX_ATOMIC_NUMBER,
        )
        return cls(TokenKind.ELEMENT, atomic_number)

    @classmethod
    def coefficient(cls, count: int) -> CompositionToken:
        return cls(TokenKind.COEFFICIENT, count)

    @classmethod
    def eos(cls) -> CompositionToken:
        return cls(TokenKind.EOS)


EOS_TOKEN = CompositionToken.eos()


@dataclass(frozen=True, slots=True)
class _PrefixState:
    support_size: int | None
    support: tuple[Element, ...]
    coefficients: tuple[int, ...]
    terminal: bool


@cache
def _coefficient_suffix_count(slots: int, budget: int, prefix_gcd: int) -> int:
    """Count positive suffixes of at most ``budget`` with final gcd one."""

    if slots == 0:
        return int(prefix_gcd == 1)
    if budget < slots:
        return 0

    total = 0
    largest = budget - (slots - 1)
    for count in range(1, largest + 1):
        total += _coefficient_suffix_count(
            slots - 1,
            budget - count,
            gcd(prefix_gcd, count),
        )
    return total


@dataclass(frozen=True, slots=True)
class CompositionGrammar:
    """Exact live-prefix oracle for canonical DNG and S-DNG formulas.

    The empty prefix is unconditional DNG.  :meth:`support_prefix` constructs
    the clamped prefix ``a(S)`` for exact-system S-DNG.  EOS leaves have no
    successor distribution: :meth:`allowed_next` raises
    :class:`TerminalPrefixError` on them.
    """

    domain: CompositionDomain

    def __post_init__(self) -> None:
        if not isinstance(self.domain, CompositionDomain):
            raise TypeError("domain must be a CompositionDomain")

    def _parse_prefix(self, prefix: Sequence[CompositionToken]) -> _PrefixState:
        tokens = tuple(prefix)
        if any(not isinstance(token, CompositionToken) for token in tokens):
            raise InvalidPrefixError("every prefix item must be a CompositionToken")
        if not tokens:
            return _PrefixState(None, (), (), False)

        first = tokens[0]
        if first.kind is not TokenKind.SUPPORT_SIZE or first.value is None:
            raise InvalidPrefixError("the first generated token must have role K")
        k = first.value
        if not 1 <= k <= self.domain.max_support_size:
            raise InvalidPrefixError(f"support size K={k} is outside the live domain")
        if len(tokens) > 2 * k + 2:
            raise InvalidPrefixError("prefix continues beyond its EOS leaf")

        support_count = min(k, max(0, len(tokens) - 1))
        support: list[Element] = []
        alphabet = self.domain.elements
        alphabet_positions = {z: index for index, z in enumerate(alphabet)}
        for offset in range(support_count):
            token = tokens[1 + offset]
            if token.kind is not TokenKind.ELEMENT or token.value is None:
                raise InvalidPrefixError("all K element tokens must precede coefficients")
            z = token.value
            if z not in alphabet_positions:
                raise InvalidPrefixError(f"element Z={z} is outside the DNG alphabet")
            if support and z <= support[-1]:
                raise InvalidPrefixError("element tokens must be strictly increasing")
            remaining_elements = k - (offset + 1)
            available_after = len(alphabet) - alphabet_positions[z] - 1
            if available_after < remaining_elements:
                raise InvalidPrefixError("element prefix cannot be completed to K elements")
            support.append(z)

        if len(tokens) <= 1 + k:
            return _PrefixState(k, tuple(support), (), False)
        if len(support) != k:
            raise InvalidPrefixError("a coefficient occurred before all element tokens")

        coefficient_token_count = min(k, len(tokens) - (1 + k))
        coefficients: list[int] = []
        for offset in range(coefficient_token_count):
            token = tokens[1 + k + offset]
            if token.kind is not TokenKind.COEFFICIENT or token.value is None:
                raise InvalidPrefixError("coefficient positions require positive count tokens")
            count = token.value
            remaining_slots = k - (offset + 1)
            if sum(coefficients) + count + remaining_slots > self.domain.max_atoms:
                raise InvalidPrefixError("coefficient prefix exceeds the raw atom bound")
            coefficients.append(count)

        if len(coefficients) == k and _positive_gcd(coefficients) != 1:
            raise InvalidPrefixError("a complete coefficient vector must have gcd one")

        eos_index = 1 + 2 * k
        if len(tokens) <= eos_index:
            return _PrefixState(k, tuple(support), tuple(coefficients), False)
        if len(tokens) != eos_index + 1 or tokens[eos_index] != EOS_TOKEN:
            raise InvalidPrefixError("the only token after all coefficients is EOS")
        return _PrefixState(k, tuple(support), tuple(coefficients), True)

    def is_live_prefix(self, prefix: Sequence[CompositionToken]) -> bool:
        """Return whether ``prefix`` is in ``P_comp`` (leaves included)."""

        try:
            self._parse_prefix(prefix)
        except (CompositionError, TypeError):
            return False
        return True

    def is_terminal(self, prefix: Sequence[CompositionToken]) -> bool:
        """Return whether ``prefix`` is a complete EOS leaf."""

        return self._parse_prefix(prefix).terminal

    def allowed_next(
        self,
        prefix: Sequence[CompositionToken],
    ) -> tuple[CompositionToken, ...]:
        """Return the exact completion-safe set ``Omega_comp(prefix)``."""

        state = self._parse_prefix(prefix)
        if state.terminal:
            raise TerminalPrefixError("an EOS leaf has no successor softmax")

        if state.support_size is None:
            return tuple(
                CompositionToken.support_size(k)
                for k in range(1, self.domain.max_support_size + 1)
            )

        k = state.support_size
        if len(state.support) < k:
            last = state.support[-1] if state.support else 0
            remaining_after_choice = k - len(state.support) - 1
            alphabet = self.domain.elements
            return tuple(
                CompositionToken.element(z)
                for index, z in enumerate(alphabet)
                if z > last and len(alphabet) - index - 1 >= remaining_after_choice
            )

        if len(state.coefficients) < k:
            remaining_after_choice = k - len(state.coefficients) - 1
            used = sum(state.coefficients)
            largest = self.domain.max_atoms - used - remaining_after_choice
            prefix_gcd = _positive_gcd(state.coefficients)
            actions: list[CompositionToken] = []
            for count in range(1, largest + 1):
                suffixes = _coefficient_suffix_count(
                    remaining_after_choice,
                    self.domain.max_atoms - used - count,
                    gcd(prefix_gcd, count),
                )
                if suffixes:
                    actions.append(CompositionToken.coefficient(count))
            return tuple(actions)

        return (EOS_TOKEN,)

    def completion_count(self, prefix: Sequence[CompositionToken] = ()) -> int:
        """Return the exact number of accepted formula leaves below a prefix."""

        state = self._parse_prefix(prefix)
        if state.terminal:
            return 1
        if state.support_size is None:
            return sum(
                comb(len(self.domain.elements), k)
                * _coefficient_suffix_count(k, self.domain.max_atoms, 0)
                for k in range(1, self.domain.max_support_size + 1)
            )

        k = state.support_size
        if len(state.support) < k:
            if state.support:
                last_index = self.domain.elements.index(state.support[-1])
                available = len(self.domain.elements) - last_index - 1
            else:
                available = len(self.domain.elements)
            supports = comb(available, k - len(state.support))
            return supports * _coefficient_suffix_count(k, self.domain.max_atoms, 0)

        remaining = k - len(state.coefficients)
        return _coefficient_suffix_count(
            remaining,
            self.domain.max_atoms - sum(state.coefficients),
            _positive_gcd(state.coefficients),
        )

    def _canonical_support(self, support: Iterable[Element]) -> tuple[Element, ...]:
        raw = tuple(support)
        for index, element in enumerate(raw):
            _require_int(
                f"support[{index}]",
                element,
                minimum=1,
                maximum=MAX_ATOMIC_NUMBER,
            )
        if len(set(raw)) != len(raw):
            raise CompositionError("an exact chemical system cannot contain duplicates")
        ordered = tuple(sorted(raw))
        if not 1 <= len(ordered) <= self.domain.max_support_size:
            raise CompositionError("chemical-system cardinality lies outside the DNG domain")
        if any(z not in self.domain.elements for z in ordered):
            raise CompositionError("chemical system contains an element outside the alphabet")
        return ordered

    def support_prefix(
        self,
        support: Iterable[Element],
    ) -> tuple[CompositionToken, ...]:
        """Return the live clamped prefix ``a(S)`` for exact-system S-DNG."""

        ordered = self._canonical_support(support)
        return (CompositionToken.support_size(len(ordered)),) + tuple(
            CompositionToken.element(z) for z in ordered
        )

    def serialize(self, composition: Composition) -> tuple[CompositionToken, ...]:
        """Return the unique support-first typed string ``s(c)``."""

        self.domain.require_contains(composition)
        return (
            CompositionToken.support_size(len(composition)),
            *(CompositionToken.element(z) for z in composition.support),
            *(CompositionToken.coefficient(n) for n in composition.coefficients),
            EOS_TOKEN,
        )

    def deserialize(self, tokens: Sequence[CompositionToken]) -> Composition:
        """Invert :meth:`serialize` on accepted EOS leaves only."""

        state = self._parse_prefix(tokens)
        if not state.terminal or state.support_size is None:
            raise InvalidPrefixError("a serialized composition must end at EOS")
        return Composition(tuple(zip(state.support, state.coefficients, strict=True)))

    def iter_serializations(
        self,
        support: Iterable[Element] | None = None,
    ) -> Iterator[tuple[CompositionToken, ...]]:
        """Enumerate every accepted leaf exactly once in token order."""

        root: tuple[CompositionToken, ...] = (
            () if support is None else self.support_prefix(support)
        )

        def visit(
            prefix: tuple[CompositionToken, ...],
        ) -> Iterator[tuple[CompositionToken, ...]]:
            if self.is_terminal(prefix):
                yield prefix
                return
            for token in self.allowed_next(prefix):
                yield from visit(prefix + (token,))

        yield from visit(root)

    def iter_compositions(
        self,
        support: Iterable[Element] | None = None,
    ) -> Iterator[Composition]:
        """Enumerate ``C_DNG`` or one exact support fibre ``C(S)``."""

        for serialization in self.iter_serializations(support):
            yield self.deserialize(serialization)

    def composition_at(
        self,
        index: int,
        support: Iterable[Element] | None = None,
    ) -> Composition:
        """Unrank one formula in exact grammar order without rejection."""

        rank = _require_int("index", index, minimum=0)
        prefix: tuple[CompositionToken, ...] = (
            () if support is None else self.support_prefix(support)
        )
        total = self.completion_count(prefix)
        if rank >= total:
            raise IndexError(f"formula index {rank} is outside range({total})")

        while not self.is_terminal(prefix):
            for token in self.allowed_next(prefix):
                child = prefix + (token,)
                child_count = self.completion_count(child)
                if rank < child_count:
                    prefix = child
                    break
                rank -= child_count
            else:  # pragma: no cover - guarded by exact positive subtree counts
                raise AssertionError("live prefix had no positive-cardinality child")
        return self.deserialize(prefix)

    def sample_uniform(
        self,
        rng: SupportsRandrange,
        support: Iterable[Element] | None = None,
    ) -> Composition:
        """Draw exactly uniformly from all leaves or from ``C(S)``.

        ``rng`` is injected by the caller.  No global pseudorandom state is
        read, which makes data-loader and distributed-worker behavior
        reproducible when their generators are seeded explicitly.
        """

        if not hasattr(rng, "randrange"):
            raise TypeError("rng must provide randrange(stop)")
        ordered_support = None if support is None else self._canonical_support(support)
        prefix: tuple[CompositionToken, ...] = (
            () if ordered_support is None else self.support_prefix(ordered_support)
        )
        total = self.completion_count(prefix)
        if total <= 0:
            raise EmptyCompositionDomainError("the selected formula fibre is empty")
        index = rng.randrange(total)
        if isinstance(index, bool) or not isinstance(index, int) or not 0 <= index < total:
            raise ValueError("rng.randrange(stop) returned a value outside range(stop)")
        return self.composition_at(index, ordered_support)


__all__ = [
    "Composition",
    "CompositionDomain",
    "CompositionError",
    "CompositionGrammar",
    "CompositionToken",
    "EOS_TOKEN",
    "Element",
    "EmptyCompositionDomainError",
    "InvalidPrefixError",
    "MAX_ATOMIC_NUMBER",
    "SupportsRandrange",
    "TerminalPrefixError",
    "TokenKind",
]
