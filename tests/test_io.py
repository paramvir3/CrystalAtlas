from __future__ import annotations

import json

import numpy as np
import pytest

from crystalatlas.composition import Composition
from crystalatlas.decoder import CrystalScaffold, ExactCrystalDecoder, OrbitAssignment
from crystalatlas.errors import DomainError
from crystalatlas.geometry import (
    ContinuousCover,
    ContinuousGeometry,
    IntrinsicCoverLayout,
    OrbitType,
)
from crystalatlas.io import (
    crystal_to_cif,
    crystal_to_dict,
    format_formula,
    parse_formula,
    write_crystal_json,
)


def _crystal():
    scaffold = CrystalScaffold.from_mapping(
        nu=1,
        space_group="P-1",
        reduced_composition={14: 1, 8: 1},
        orbits=(
            OrbitAssignment("1a", 8, ()),
            OrbitAssignment("1h", 14, ()),
        ),
    )
    return ExactCrystalDecoder().decode(scaffold, np.eye(3) * 9.0)


def test_flat_formula_parser_reduces_and_roundtrips() -> None:
    composition = parse_formula("Si2O4")
    assert composition.counts == ((8, 2), (14, 1))
    assert format_formula(composition) == "O2Si"
    assert parse_formula(format_formula(composition)) == composition


@pytest.mark.parametrize("text", ["", "Ca(OH)2", "Fe0O", "Xx2", "Na+Cl-"])
def test_formula_parser_rejects_unsupported_semantics(text: str) -> None:
    with pytest.raises(DomainError):
        parse_formula(text)


def test_cif_and_json_preserve_audit(tmp_path) -> None:
    crystal = _crystal()
    cif = crystal_to_cif(
        crystal,
        law="untrained-uniform-reference",
        physical_prediction=False,
    )
    assert "_space_group_name_Hall 'P 1'" in cif
    assert "crystalatlas_selected_containment_group: P-1" in cif
    assert "crystalatlas_law: untrained-uniform-reference" in cif
    assert "crystalatlas_physical_prediction: false" in cif
    assert "O1 O 0 0 0" in cif
    payload = crystal_to_dict(crystal)
    assert payload["atomic_numbers"] == [8, 14]
    assert payload["audit"]["selected_group_is_contained"] is True
    destination = tmp_path / "result.json"
    write_crystal_json(crystal, destination)
    assert json.loads(destination.read_text())["translation_index"] == 1


def test_continuous_crystal_uses_the_same_safe_full_motif_io() -> None:
    layout = IntrinsicCoverLayout(
        ContinuousCover(
            nu=2,
            space_group="P-1",
            composition=Composition(((8, 1),)),
            orbits=(OrbitType("2i", 8),),
        )
    )
    crystal = layout.decode(
        ContinuousGeometry(
            np.zeros(layout.lattice_dimension),
            np.asarray([0.17, 0.29, 0.41]),
        )
    )
    cif = crystal_to_cif(crystal)
    assert "_space_group_name_Hall 'P 1'" in cif
    assert "crystalatlas_selected_containment_group: P-1" in cif
    payload = crystal_to_dict(crystal)
    assert payload["atomic_numbers"] == [8, 8]
    assert payload["audit"]["minimum_distance"] > 0.0
