"""Small, deterministic text interfaces for formulas and decoded crystals.

The parser deliberately accepts only flat, fully occupied elemental formulas.
Parentheses, charges, isotope labels, vacancies, and partial occupancies require
additional state variables and are rejected instead of being guessed.
"""

from __future__ import annotations

import json
import math
import re
from collections import Counter
from collections.abc import Mapping
from pathlib import Path
from typing import Any, TypeAlias

import numpy as np

from .composition import Composition
from .decoder import DecodedCrystal
from .elements import atomic_number, element_symbol
from .errors import DomainError
from .geometry import ContinuousDecodedCrystal

CrystalLike: TypeAlias = DecodedCrystal | ContinuousDecodedCrystal

_FORMULA_TOKEN = re.compile(r"([A-Z][a-z]?)([1-9][0-9]*)?")


def parse_formula(text: str) -> Composition:
    """Parse and reduce a flat elemental formula such as ``LiFePO4``.

    A common multiplier is removed because CSP conditions on the reduced
    formula.  For example, ``Si2O4`` and ``SiO2`` produce the same object.
    """

    if not isinstance(text, str) or not text:
        raise DomainError("formula must be a nonempty string")
    counts: Counter[int] = Counter()
    position = 0
    for match in _FORMULA_TOKEN.finditer(text):
        if match.start() != position:
            raise DomainError(
                "formula syntax must be a flat sequence of element symbols and "
                "positive integer coefficients"
            )
        try:
            z = atomic_number(match.group(1))
        except (KeyError, ValueError) as exc:
            raise DomainError(f"unknown element symbol {match.group(1)!r}") from exc
        coefficient = int(match.group(2) or "1")
        counts[z] += coefficient
        position = match.end()
    if position != len(text) or not counts:
        raise DomainError(
            "formula syntax must be a flat sequence of element symbols and "
            "positive integer coefficients"
        )
    return Composition.from_counts(counts)


def format_formula(composition: Composition) -> str:
    """Return the package's atomic-number-ordered canonical formula string."""

    if not isinstance(composition, Composition):
        raise TypeError("composition must be a Composition")
    return "".join(
        f"{element_symbol(z)}{'' if count == 1 else count}"
        for z, count in composition.counts
    )


def _cell_parameters(lattice: np.ndarray) -> tuple[float, float, float, float, float, float]:
    cell = np.asarray(lattice, dtype=np.float64)
    if cell.shape != (3, 3) or not np.all(np.isfinite(cell)):
        raise DomainError("lattice must be a finite 3x3 row-lattice matrix")
    lengths = np.linalg.norm(cell, axis=1)
    if np.any(lengths <= 0.0) or float(np.linalg.det(cell)) <= 0.0:
        raise DomainError("lattice must be right handed with positive lengths")

    def angle(left: np.ndarray, right: np.ndarray, scale: float) -> float:
        cosine = float(np.dot(left, right) / scale)
        return math.degrees(math.acos(float(np.clip(cosine, -1.0, 1.0))))

    a, b, c = (float(value) for value in lengths)
    alpha = angle(cell[1], cell[2], b * c)
    beta = angle(cell[0], cell[2], a * c)
    gamma = angle(cell[0], cell[1], a * b)
    return a, b, c, alpha, beta, gamma


def _crystal_arrays(crystal: CrystalLike) -> tuple[np.ndarray, tuple[int, ...]]:
    if isinstance(crystal, DecodedCrystal):
        positions = crystal.fractional_positions_float
        raw_species = crystal.species
    elif isinstance(crystal, ContinuousDecodedCrystal):
        positions = crystal.fractional_positions
        raw_species = tuple(int(value) for value in crystal.atomic_numbers)
    else:
        raise TypeError("crystal must be a DecodedCrystal or ContinuousDecodedCrystal")
    if any(not isinstance(value, int) for value in raw_species):
        raise DomainError("serialization requires integer atomic-number species")
    return np.asarray(positions, dtype=np.float64), tuple(int(value) for value in raw_species)


def _translation_index(crystal: CrystalLike) -> int:
    return (
        crystal.translation_index
        if isinstance(crystal, DecodedCrystal)
        else crystal.audit.translation_index
    )


def _one_line(value: object) -> str:
    return str(value).replace("\r", " ").replace("\n", " ")


def crystal_to_cif(
    crystal: CrystalLike,
    *,
    data_name: str = "crystalatlas",
    law: str | None = None,
    physical_prediction: bool | None = None,
    provenance: Mapping[str, object] | None = None,
) -> str:
    """Serialize a decoded full motif as a minimal, semantically safe CIF 1.1 block.

    Because coordinates contain the *full symmetry-expanded motif*, the CIF is
    deliberately declared in P1.  The selected containment group is retained
    only as an audit comment.  Declaring that non-P1 group alongside the full
    motif would invite CIF readers to expand equivalent atoms a second time.
    """

    positions, species = _crystal_arrays(crystal)
    safe_name = re.sub(r"[^A-Za-z0-9_.-]", "_", data_name) or "crystalatlas"
    a, b, c, alpha, beta, gamma = _cell_parameters(crystal.lattice)
    lines = [
        f"data_{safe_name}",
        "# Generated by CrystalAtlas; the full motif is intentionally declared as P1.",
        f"# crystalatlas_selected_containment_group: {_one_line(crystal.setting.key)}",
        f"# crystalatlas_selected_hall_symbol: {_one_line(crystal.setting.hall_symbol)}",
        f"# crystalatlas_complete_motif_translation_index: {_translation_index(crystal)}",
        f"_cell_length_a {a:.12g}",
        f"_cell_length_b {b:.12g}",
        f"_cell_length_c {c:.12g}",
        f"_cell_angle_alpha {alpha:.12g}",
        f"_cell_angle_beta {beta:.12g}",
        f"_cell_angle_gamma {gamma:.12g}",
        "_space_group_name_Hall 'P 1'",
        "_space_group_name_H-M_alt 'P 1'",
        "loop_",
        "_atom_site_label",
        "_atom_site_type_symbol",
        "_atom_site_fract_x",
        "_atom_site_fract_y",
        "_atom_site_fract_z",
        "_atom_site_occupancy",
    ]
    if law is not None:
        lines.insert(4, f"# crystalatlas_law: {_one_line(law)}")
    if physical_prediction is not None:
        lines.insert(
            5 if law is not None else 4,
            "# crystalatlas_physical_prediction: "
            + ("true" if physical_prediction else "false"),
        )
    for key, value in sorted((provenance or {}).items()):
        lines.insert(
            6 if law is not None and physical_prediction is not None else 4,
            f"# crystalatlas_provenance_{_one_line(key)}: {_one_line(value)}",
        )
    occurrence: Counter[str] = Counter()
    for atom_species, point in zip(species, positions, strict=True):
        atom_symbol = element_symbol(atom_species)
        occurrence[atom_symbol] += 1
        coordinates = " ".join(f"{float(value):.12g}" for value in point)
        lines.append(
            f"{atom_symbol}{occurrence[atom_symbol]} {atom_symbol} {coordinates} 1"
        )
    return "\n".join(lines) + "\n"


def crystal_to_dict(crystal: CrystalLike) -> dict[str, Any]:
    """Return a JSON-compatible, audit-preserving representation."""

    positions, species = _crystal_arrays(crystal)
    if isinstance(crystal, DecodedCrystal):
        nu = crystal.scaffold.nu
        audit = {
            "positive_volume": crystal.audit.positive_volume,
            "formula_matches": crystal.audit.formula_matches,
            "orbit_multiplicities_match": crystal.audit.orbit_multiplicities_match,
            "no_collisions": crystal.audit.no_collisions,
            "metric_is_invariant": crystal.audit.metric_is_invariant,
            "selected_group_is_contained": crystal.audit.selected_group_is_contained,
            "reduced_composition_matches": crystal.audit.reduced_composition_matches,
        }
    else:
        nu = crystal.cover.nu
        audit = {
            "positive_volume": bool(np.linalg.det(crystal.lattice) > 0.0),
            "formula_matches": crystal.audit.formula_matches,
            "orbit_multiplicities_match": True,
            "no_collisions": crystal.audit.no_collisions,
            "metric_is_invariant": crystal.audit.metric_is_invariant,
            "selected_group_is_contained": crystal.audit.selected_group_is_contained,
            "reduced_composition_matches": crystal.audit.formula_matches,
            "minimum_distance": crystal.audit.minimum_distance,
        }
    return {
        "space_group_constraint": crystal.setting.key,
        "hall_symbol": crystal.setting.hall_symbol,
        "formula_units_advertised": nu,
        "translation_index": _translation_index(crystal),
        "lattice": crystal.lattice.tolist(),
        "metric": crystal.metric.tolist(),
        "fractional_positions": positions.tolist(),
        "atomic_numbers": list(species),
        "audit": audit,
    }


def write_crystal_json(crystal: CrystalLike, path: str | Path) -> None:
    """Write one decoded crystal atomically as UTF-8 JSON."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp")
    temporary.write_text(
        json.dumps(crystal_to_dict(crystal), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(destination)


__all__ = [
    "crystal_to_cif",
    "crystal_to_dict",
    "format_formula",
    "parse_formula",
    "write_crystal_json",
]
