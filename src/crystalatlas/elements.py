"""Exact element identifiers and user-input canonicalization."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

from .errors import DomainError

ELEMENT_SYMBOLS: tuple[str, ...] = (
    "H",
    "He",
    "Li",
    "Be",
    "B",
    "C",
    "N",
    "O",
    "F",
    "Ne",
    "Na",
    "Mg",
    "Al",
    "Si",
    "P",
    "S",
    "Cl",
    "Ar",
    "K",
    "Ca",
    "Sc",
    "Ti",
    "V",
    "Cr",
    "Mn",
    "Fe",
    "Co",
    "Ni",
    "Cu",
    "Zn",
    "Ga",
    "Ge",
    "As",
    "Se",
    "Br",
    "Kr",
    "Rb",
    "Sr",
    "Y",
    "Zr",
    "Nb",
    "Mo",
    "Tc",
    "Ru",
    "Rh",
    "Pd",
    "Ag",
    "Cd",
    "In",
    "Sn",
    "Sb",
    "Te",
    "I",
    "Xe",
    "Cs",
    "Ba",
    "La",
    "Ce",
    "Pr",
    "Nd",
    "Pm",
    "Sm",
    "Eu",
    "Gd",
    "Tb",
    "Dy",
    "Ho",
    "Er",
    "Tm",
    "Yb",
    "Lu",
    "Hf",
    "Ta",
    "W",
    "Re",
    "Os",
    "Ir",
    "Pt",
    "Au",
    "Hg",
    "Tl",
    "Pb",
    "Bi",
    "Po",
    "At",
    "Rn",
    "Fr",
    "Ra",
    "Ac",
    "Th",
    "Pa",
    "U",
    "Np",
    "Pu",
    "Am",
    "Cm",
    "Bk",
    "Cf",
    "Es",
    "Fm",
    "Md",
    "No",
    "Lr",
    "Rf",
    "Db",
    "Sg",
    "Bh",
    "Hs",
    "Mt",
    "Ds",
    "Rg",
    "Cn",
    "Nh",
    "Fl",
    "Mc",
    "Lv",
    "Ts",
    "Og",
)

SYMBOL_TO_ATOMIC_NUMBER = {symbol: index + 1 for index, symbol in enumerate(ELEMENT_SYMBOLS)}


def atomic_number(value: str | int) -> int:
    """Convert a canonical symbol or integer to an atomic number."""

    if isinstance(value, bool):
        raise DomainError("boolean values are not atomic numbers")
    if isinstance(value, int):
        if 1 <= value <= len(ELEMENT_SYMBOLS):
            return value
        raise DomainError(f"atomic number outside [1, {len(ELEMENT_SYMBOLS)}]: {value}")
    stripped = value.strip()
    if not stripped:
        raise DomainError("empty element symbol")
    canonical = stripped[0].upper() + stripped[1:].lower()
    try:
        return SYMBOL_TO_ATOMIC_NUMBER[canonical]
    except KeyError as exc:
        raise DomainError(f"unknown element symbol: {value!r}") from exc


def element_symbol(z: int) -> str:
    """Return the IUPAC symbol for atomic number ``z``."""

    number = atomic_number(z)
    return ELEMENT_SYMBOLS[number - 1]


NOBLE_GAS_ATOMIC_NUMBERS: tuple[int, ...] = (2, 10, 18, 36, 54, 86, 118)
"""Atomic number of the last element of each period, ``N_1 .. N_7``."""

LONG_FORM_COLUMNS = 32
PERIODIC_BLOCKS: tuple[str, ...] = ("s", "p", "d", "f")


@dataclass(frozen=True, slots=True)
class PeriodicCoordinates:
    """Position of an element in the 32-column long-form periodic table."""

    period: int
    """Row ``1 .. 7``."""
    column: int
    """Long-form column ``1 .. 32``."""
    block: str
    """One of :data:`PERIODIC_BLOCKS`."""


def periodic_coordinates(z: int) -> PeriodicCoordinates:
    """Exact long-form table position of atomic number ``z``; integer arithmetic only.

    With ``N_0 = 0`` and ``N_p`` the noble-gas numbers, the period ``p`` is the
    smallest index with ``z <= N_p``, its length is ``L_p = N_p - N_(p-1)``
    (2, 8, 8, 18, 18, 32, 32) and ``k = z - N_(p-1)`` counts along the period.
    The first two elements of a period fill columns 1 and 2 and the remaining
    ``L_p - 2`` are right-aligned, ``column = 32 - L_p + k``; helium, the
    second and last element of period 1, is right-aligned too (column 32).

    This is the long form with lutetium and lawrencium below yttrium: columns
    1-2 are the s block, 3-16 the f block (La-Yb and Ac-No), 17-26 the d block
    (18-column groups 3-12) and 27-32 the p block (groups 13-18).  Helium has
    no p electrons and is assigned to the s block.  The position is a label of
    the table, not a ground-state electron configuration.
    """

    number = atomic_number(z)
    previous = 0
    for period, last in enumerate(NOBLE_GAS_ATOMIC_NUMBERS, start=1):
        if number <= last:
            length = last - previous
            k = number - previous
            if k == 1 or (k == 2 and length > 2):
                column = k
            else:
                column = LONG_FORM_COLUMNS - length + k
            if column <= 2 or number == 2:
                block = "s"
            elif column <= 16:
                block = "f"
            elif column <= 26:
                block = "d"
            else:
                block = "p"
            return PeriodicCoordinates(period=period, column=column, block=block)
        previous = last
    raise DomainError(f"atomic number {number} lies beyond period 7")  # pragma: no cover


def canonical_species_set(values: Iterable[str | int]) -> tuple[int, ...]:
    """Validate an exact all-and-only species condition.

    Duplicate entries are rejected because accepting them silently would hide a
    malformed condition.  Ordering is removed by sorting atomic numbers.
    """

    numbers = tuple(atomic_number(value) for value in values)
    if not numbers:
        raise DomainError("an exact species set must be nonempty")
    if len(set(numbers)) != len(numbers):
        raise DomainError("an exact species set cannot contain duplicates")
    return tuple(sorted(numbers))


def species_label(values: Iterable[int]) -> str:
    """Stable human-readable label such as ``Fe-Li-O-P`` in atomic-number order."""

    return "-".join(element_symbol(value) for value in canonical_species_set(values))

