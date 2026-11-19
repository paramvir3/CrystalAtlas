"""Dependency-light canonical serialization of arithmetic cover rows."""

from __future__ import annotations

from .catalogue import SpaceGroupSetting
from .constraints import WyckoffRowSpec


def canonical_rows(setting: SpaceGroupSetting) -> tuple[WyckoffRowSpec, ...]:
    """Return the row order used by exact targets and neural cover sampling.

    Fixed rows precede repeatable positive-dimensional rows, and labels break
    ties.  The rule is independent of catalogue dictionary insertion order.
    Keeping it in the exact core prevents data preprocessing from importing
    optional neural dependencies merely to reproduce selector targets.
    """

    if not isinstance(setting, SpaceGroupSetting):
        raise TypeError("setting must be a SpaceGroupSetting")
    return tuple(
        WyckoffRowSpec(row.label, row.multiplicity, row.dimension)
        for row in sorted(
            setting.wyckoff_positions.values(),
            key=lambda row: (row.dimension != 0, row.label),
        )
    )


__all__ = ["canonical_rows"]
