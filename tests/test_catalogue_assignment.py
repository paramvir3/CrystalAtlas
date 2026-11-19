from __future__ import annotations

from fractions import Fraction

import numpy as np
import pytest

from crystalatlas.dataset_identity import (
    DeduplicationLevel,
    build_canonical_identity,
    deduplicate_canonical_records,
)
from crystalatlas.full_catalogue import (
    SPGLIB_SOURCE_VERSION,
    build_full_catalogue,
    pinned_hall_numbers,
)
from crystalatlas.ingestion import (
    CatalogueTargetAssigner,
    RawPeriodicStructure,
    cif_to_catalogue_training_target,
    standardize_raw_structure,
    structure_to_catalogue_training_target,
)
from crystalatlas.standardization import SpglibPolicy, SpglibStandardizer


@pytest.fixture(scope="module")
def full_pipeline():
    spglib = pytest.importorskip("spglib")
    if str(spglib.__version__) != SPGLIB_SOURCE_VERSION:
        pytest.skip(f"requires spglib {SPGLIB_SOURCE_VERSION}")
    catalogue = build_full_catalogue()
    standardizer = SpglibStandardizer(
        pinned_hall_numbers(),
        SpglibPolicy(required_spglib_version=SPGLIB_SOURCE_VERSION),
    )
    return catalogue, standardizer, CatalogueTargetAssigner(catalogue)


def _raw_exact_orbit(
    catalogue,
    *,
    international_number: int,
    label: str,
    parameter: tuple[Fraction, ...],
    lattice: np.ndarray,
    atomic_number: int = 8,
) -> RawPeriodicStructure:
    row = catalogue.get(international_number).wyckoff(label)
    points = np.asarray(
        [[float(value) for value in point] for point in row.expand_exact(parameter)],
        dtype=np.float64,
    )
    return RawPeriodicStructure.from_arrays(
        lattice,
        points,
        [atomic_number] * len(points),
    )


def test_assigner_inverts_p1_and_p_minus_one_orbits(full_pipeline) -> None:
    _, standardizer, assigner = full_pipeline
    p1 = RawPeriodicStructure.from_arrays(
        np.asarray([[3.1, 0.2, 0.1], [0.0, 2.7, 0.3], [0.1, 0.2, 2.5]]),
        np.asarray(
            [
                [0.13, 0.27, 0.39],
                [0.45, 0.11, 0.73],
                [0.22, 0.61, 0.08],
                [0.79, 0.42, 0.56],
            ]
        ),
        [6, 7, 8, 9],
    )
    assigned_p1 = assigner.assign(standardize_raw_structure(p1, standardizer=standardizer))
    assert assigned_p1.scaffold.space_group == "P1"
    assert [orbit.wyckoff_label for orbit in assigned_p1.scaffold.orbits] == [
        "1a",
        "1a",
        "1a",
        "1a",
    ]
    assert assigned_p1.receipt.passed

    p = np.asarray([0.17, 0.23, 0.31])
    p_minus_one = RawPeriodicStructure.from_arrays(
        p1.lattice,
        np.asarray([p, np.mod(-p, 1.0)]),
        [8, 8],
    )
    assigned_inverse = assigner.assign(
        standardize_raw_structure(p_minus_one, standardizer=standardizer)
    )
    assert assigned_inverse.scaffold.space_group == "P-1"
    assert [orbit.wyckoff_label for orbit in assigned_inverse.scaffold.orbits] == ["2i"]
    assert assigned_inverse.receipt.roundtrip_max_cartesian_error < 1.0e-9


def test_proper_basis_equivalent_raw_records_deduplicate_after_complete_target_audit(
    full_pipeline,
) -> None:
    catalogue, standardizer, assigner = full_pipeline
    lattice = np.diag([2.0, 3.0, 4.0])
    positions = np.array(
        [[0.13, 0.27, 0.39], [0.45, 0.11, 0.73], [0.22, 0.61, 0.08], [0.79, 0.42, 0.56]]
    )
    basis = np.diag([-1, -1, 1])
    records = tuple(
        structure_to_catalogue_training_target(
            RawPeriodicStructure.from_arrays(cell, fractional, (1, 2, 3, 4)),
            standardizer=standardizer,
            target_assigner=assigner,
        )
        for cell, fractional in (
            (lattice, positions),
            (basis @ lattice, np.mod(positions @ basis, 1.0)),
        )
    )
    identities = tuple(
        build_canonical_identity(record, catalogue=catalogue) for record in records
    )
    assert all(record.canonicalized.receipt.passed for record in records)
    assert identities[0].structure_key == identities[1].structure_key
    assert identities[0].polymorph_key == identities[1].polymorph_key
    assert records[0].target.canonical_scaffold == records[1].target.canonical_scaffold
    result = deduplicate_canonical_records(
        records, catalogue=catalogue, level=DeduplicationLevel.STRUCTURE
    )
    assert len(result.groups) == 1
    assert len(result.groups[0].member_indices) == 2


def test_assigner_inverts_nontrivial_free_pnma_row(full_pipeline) -> None:
    catalogue, standardizer, assigner = full_pipeline
    raw = _raw_exact_orbit(
        catalogue,
        international_number=62,
        label="4c",
        parameter=(Fraction(1, 7), Fraction(2, 9)),
        lattice=np.diag([5.0, 6.0, 7.0]),
    )
    standardized = standardize_raw_structure(raw, standardizer=standardizer)
    assert standardized.numerical_result.international_number == 62
    assigned = assigner.assign(standardized)
    assert assigned.scaffold.space_group == "Pnma"
    assert len(assigned.scaffold.orbits) == 1
    assert assigned.scaffold.orbits[0].wyckoff_label == "4c"
    assert len(assigned.scaffold.orbits[0].parameter) == 2
    assert assigned.receipt.passed


def test_assigner_preserves_uppercase_27th_wyckoff_letter(full_pipeline) -> None:
    catalogue, standardizer, assigner = full_pipeline
    raw = _raw_exact_orbit(
        catalogue,
        international_number=47,
        label="8A",
        parameter=(Fraction(1, 7), Fraction(2, 9), Fraction(3, 11)),
        lattice=np.diag([5.0, 6.0, 7.0]),
    )
    standardized = standardize_raw_structure(raw, standardizer=standardizer)
    assert standardized.numerical_result.international_number == 47
    assert set(standardized.numerical_result.wyckoff_letters) == {"A"}
    assigned = assigner.assign(standardized)
    assert [orbit.wyckoff_label for orbit in assigned.scaffold.orbits] == ["8A"]


def test_centered_fm3m_is_end_to_end_canonical_target(full_pipeline) -> None:
    _, standardizer, assigner = full_pipeline
    centering = np.asarray([[0.0, 0.0, 0.0], [0.0, 0.5, 0.5], [0.5, 0.0, 0.5], [0.5, 0.5, 0.0]])
    raw = RawPeriodicStructure.from_arrays(
        np.eye(3) * 4.0,
        centering,
        [8] * 4,
    )
    record = structure_to_catalogue_training_target(
        raw,
        standardizer=standardizer,
        target_assigner=assigner,
    )
    assert record.canonicalized.scaffold.space_group == "Fm-3m"
    assert record.canonicalized.scaffold.nu == 4
    assert [orbit.wyckoff_label for orbit in record.canonicalized.scaffold.orbits] == ["4a"]
    assert record.canonicalized.receipt.translation_index == 4
    assert record.target.canonical_scaffold == record.canonicalized.scaffold
    assert record.target.intrinsic_coordinates.size == 0


def test_strict_cif_is_an_end_to_end_complete_catalogue_target(full_pipeline) -> None:
    _, standardizer, assigner = full_pipeline
    cif = """data_complete_catalogue_p1
_cell_length_a 4.7
_cell_length_b 5.3
_cell_length_c 6.1
_cell_angle_alpha 79
_cell_angle_beta 83
_cell_angle_gamma 71
_space_group_name_Hall 'P 1'
loop_
_atom_site_label
_atom_site_type_symbol
_atom_site_fract_x
_atom_site_fract_y
_atom_site_fract_z
_atom_site_occupancy
C1 C 0.13 0.27 0.39 1
N1 N 0.45 0.11 0.73 1
O1 O 0.22 0.61 0.08 1
F1 F 0.79 0.42 0.56 1
"""
    record = cif_to_catalogue_training_target(
        cif,
        standardizer=standardizer,
        target_assigner=assigner,
        source_id="in-memory:cif-to-canonical-target",
    )
    assert record.raw_structure.source_id == "in-memory:cif-to-canonical-target"
    assert record.canonicalized.scaffold.space_group == "P1"
    assert record.canonicalized.receipt.passed
    assert record.target.canonical_scaffold == record.canonicalized.scaffold
    assert record.target.intrinsic_coordinates.size == 9
