from __future__ import annotations

from dataclasses import replace
from fractions import Fraction

import numpy as np
import pytest

from crystalatlas.decoder import CrystalScaffold, ExactCrystalDecoder, OrbitAssignment
from crystalatlas.errors import DomainError, InvariantViolation
from crystalatlas.ingestion import (
    CanonicalizationReceipt,
    CanonicalizedStructure,
    CatalogueTargetAssigner,
    RawPeriodicStructure,
    parse_cif_structure,
    read_cif_structure,
    standardize_raw_structure,
    structure_to_exact_fixture_target,
)
from crystalatlas.io import crystal_to_cif
from crystalatlas.standardization import (
    CanonicalizedStructure as NumericalCanonicalizedStructure,
)
from crystalatlas.standardization import FloatingStructure, StandardizationProvenance

_P1_CIF = """data_strict_p1
_cell_length_a 5.0(2)
_cell_length_b 6
_cell_length_c 7
_cell_angle_alpha 90
_cell_angle_beta 90
_cell_angle_gamma 90
_space_group_name_Hall 'P 1'
loop_
_atom_site_label
_atom_site_type_symbol
_atom_site_fract_x
_atom_site_fract_y
_atom_site_fract_z
_atom_site_occupancy
Si1 Si -0.1 0.2 1.3 1
O1 O 0.4 0.5 0.6 1.0
"""


def test_raw_structure_validates_integer_chemistry_and_wraps_positions() -> None:
    structure = RawPeriodicStructure.from_arrays(
        np.diag([4.0, 5.0, 6.0]),
        [[-0.25, 1.25, 0.5]],
        [8],
        labels=["O1"],
    )
    np.testing.assert_allclose(structure.fractional_positions, [[0.75, 0.25, 0.5]])
    np.testing.assert_allclose(structure.metric, np.diag([16.0, 25.0, 36.0]))
    assert structure.volume == pytest.approx(120.0)
    assert structure.species_counts == {8: 1}
    assert not structure.lattice.flags.writeable
    assert not structure.fractional_positions.flags.writeable
    assert len(structure.fingerprint) == 64

    numpy_integer = RawPeriodicStructure.from_arrays(
        np.eye(3), [[0.0, 0.0, 0.0]], np.asarray([8], dtype=np.int64)
    )
    assert numpy_integer.atomic_numbers == (8,)


@pytest.mark.parametrize(
    ("lattice", "positions", "numbers", "occupancies"),
    [
        (np.zeros((3, 3)), [[0.0, 0.0, 0.0]], [8], [1.0]),
        (np.eye(3), [[0.0, 0.0, 0.0]], [0], [1.0]),
        (np.eye(3), [[0.0, 0.0, 0.0]], [8], [0.999999999]),
        (np.eye(3), [[np.nan, 0.0, 0.0]], [8], [1.0]),
    ],
)
def test_raw_structure_rejects_invalid_domain(
    lattice: np.ndarray,
    positions: list[list[float]],
    numbers: list[int],
    occupancies: list[float],
) -> None:
    with pytest.raises(DomainError):
        RawPeriodicStructure.from_arrays(lattice, positions, numbers, occupancies=occupancies)


def test_strict_p1_cif_parses_uncertainty_and_full_occupancy() -> None:
    structure = parse_cif_structure(_P1_CIF)
    assert structure.atomic_numbers == (14, 8)
    assert structure.source_id == "strict_p1"
    np.testing.assert_allclose(
        structure.fractional_positions,
        [[0.9, 0.2, 0.3], [0.4, 0.5, 0.6]],
    )
    assert float(np.linalg.det(structure.lattice)) > 0.0
    assert structure.volume == pytest.approx(210.0)


@pytest.mark.parametrize(
    ("length_b", "gamma", "operations", "valid", "atom_count"),
    [
        (4.0, 90.0, ("x,y,z", "-y,x,z", "-x,-y,z", "y,-x,z"), False, 4),
        (3.0, 90.0, ("x,y,z", "-y,x,z", "-x,-y,z", "y,-x,z"), True, 4),
        (3.0, 120.0, ("x,y,z", "-y,x-y,z", "-x+y,-x,z"), True, 3),
        (3.0, 90.0, ("x,y,z", "-y,x-y,z", "-x+y,-x,z"), False, 3),
    ],
)
def test_cif_symmetry_operations_must_be_isometries_of_declared_cell(
    length_b, gamma, operations, valid, atom_count
) -> None:
    operation_rows = "\n".join(f"'{operation}'" for operation in operations)
    cif = f"""data_metric_compatibility
_cell_length_a 3
_cell_length_b {length_b}
_cell_length_c 5
_cell_angle_alpha 90
_cell_angle_beta 90
_cell_angle_gamma {gamma}
loop_
_space_group_symop_operation_xyz
{operation_rows}
loop_
_atom_site_type_symbol
_atom_site_fract_x
_atom_site_fract_y
_atom_site_fract_z
_atom_site_occupancy
C .1 .2 .3 1
"""
    if not valid:
        with pytest.raises(DomainError, match="incompatible with the cell metric"):
            parse_cif_structure(cif)
    else:
        structure = parse_cif_structure(cif)
        assert structure.atomic_numbers == (6,) * atom_count


def test_package_full_motif_cif_round_trips_through_strict_reader() -> None:
    scaffold = CrystalScaffold.from_mapping(
        nu=1,
        space_group="P-1",
        reduced_composition={8: 1, 14: 1},
        orbits=(OrbitAssignment("1a", 8, ()), OrbitAssignment("1h", 14, ())),
    )
    crystal = ExactCrystalDecoder().decode(scaffold, np.eye(3) * 9.0)
    parsed = parse_cif_structure(crystal_to_cif(crystal))
    assert parsed.atomic_numbers == (8, 14)
    np.testing.assert_allclose(parsed.lattice, crystal.lattice, atol=1.0e-12, rtol=0.0)
    np.testing.assert_allclose(
        parsed.fractional_positions,
        crystal.fractional_positions_float,
        atol=1.0e-12,
        rtol=0.0,
    )


def test_cif_expands_explicit_p_minus_one_group_exactly() -> None:
    cif = """data_inverse
_cell_length_a 4
_cell_length_b 4
_cell_length_c 4
_cell_angle_alpha 90
_cell_angle_beta 90
_cell_angle_gamma 90
_space_group_name_H-M_alt 'P -1'
loop_
_space_group_symop_id
_space_group_symop_operation_xyz
1 'x,y,z'
2 '-x,-y,-z'
loop_
_atom_site_label
_atom_site_type_symbol
_atom_site_fract_x
_atom_site_fract_y
_atom_site_fract_z
_atom_site_occupancy
O1 O 0.2 0.25 0.3 1
"""
    structure = parse_cif_structure(cif)
    assert structure.atomic_numbers == (8, 8)
    assert {tuple(value) for value in structure.fractional_positions} == {
        (0.2, 0.25, 0.3),
        (0.8, 0.75, 0.7),
    }


def test_cif_symmetry_expression_columns_follow_row_coordinate_convention() -> None:
    cif = """data_hexagonal_action
_cell_length_a 4
_cell_length_b 4
_cell_length_c 5
_cell_angle_alpha 90
_cell_angle_beta 90
_cell_angle_gamma 120
_space_group_name_H-M_alt 'P 6'
loop_
_space_group_symop_operation_xyz
'x,y,z'
'x-y,x,z'
'-y,x-y,z'
'-x,-y,z'
'-x+y,-x,z'
'y,-x+y,z'
loop_
_atom_site_label
_atom_site_type_symbol
_atom_site_fract_x
_atom_site_fract_y
_atom_site_fract_z
_atom_site_occupancy
C1 C 0.2 0.3 0.4 1
"""
    structure = parse_cif_structure(cif)
    assert len(structure.atomic_numbers) == 6
    assert any(
        np.allclose(point, [0.9, 0.2, 0.4], atol=1.0e-15, rtol=0.0)
        for point in structure.fractional_positions
    )


@pytest.mark.parametrize(
    ("replacement", "message"),
    [
        ("O1 O 0.4 0.5 0.6 .", "unknown"),
        ("O1 O 0.4 0.5 0.6 0.999", "unit-occupancy"),
    ],
)
def test_cif_rejects_unknown_or_partial_occupancy(replacement: str, message: str) -> None:
    malformed = _P1_CIF.replace("O1 O 0.4 0.5 0.6 1.0", replacement)
    with pytest.raises(DomainError, match=message):
        parse_cif_structure(malformed)


def test_cif_rejects_missing_occupancy_tag() -> None:
    malformed = (
        _P1_CIF.replace("_atom_site_occupancy\n", "")
        .replace("Si1 Si -0.1 0.2 1.3 1\n", "Si1 Si -0.1 0.2 1.3\n")
        .replace("O1 O 0.4 0.5 0.6 1.0\n", "O1 O 0.4 0.5 0.6\n")
    )
    with pytest.raises(DomainError, match="explicit occupancies"):
        parse_cif_structure(malformed)


def test_non_p1_cif_without_operations_fails_closed() -> None:
    malformed = _P1_CIF.replace("'P 1'", "'P -1'")
    with pytest.raises(DomainError, match="explicit symmetry-operation"):
        parse_cif_structure(malformed)


def test_incomplete_symmetry_operation_table_fails_closure() -> None:
    malformed = _P1_CIF.replace(
        "_space_group_name_Hall 'P 1'",
        """_space_group_name_H-M_alt 'P 2'
loop_
_space_group_symop_operation_xyz
'x,y,z'
'-x,y+1/3,-z'""",
    )
    with pytest.raises(DomainError, match="exact finite group"):
        parse_cif_structure(malformed)


def test_full_occupancy_sites_may_not_overlap_after_expansion() -> None:
    malformed = _P1_CIF.replace(
        "O1 O 0.4 0.5 0.6 1.0",
        "O1 O -0.1 0.2 1.3 1.0",
    )
    with pytest.raises(DomainError, match="overlap"):
        parse_cif_structure(malformed)


def test_bounded_cif_reader(tmp_path) -> None:
    source = tmp_path / "input.cif"
    source.write_text(_P1_CIF, encoding="utf-8")
    assert read_cif_structure(source).source_id == str(source.resolve())
    with pytest.raises(DomainError, match="safety limit"):
        read_cif_structure(source, max_bytes=8)


class _FixtureCanonicalizer:
    def __init__(self, *, pass_gates: bool = True, wrong_fingerprint: bool = False):
        self.pass_gates = pass_gates
        self.wrong_fingerprint = wrong_fingerprint

    def canonicalize(self, structure: RawPeriodicStructure) -> CanonicalizedStructure:
        primitive = RawPeriodicStructure.from_arrays(
            structure.lattice,
            [[0.0, 0.0, 0.0]],
            [8],
            labels=["O1"],
            source_id="canonical",
        )
        scaffold = CrystalScaffold.from_mapping(
            nu=1,
            space_group="P1",
            reduced_composition={8: 1},
            orbits=(OrbitAssignment("1a", 8, (Fraction(0), Fraction(0), Fraction(0))),),
        )
        fingerprint = "0" * 64 if self.wrong_fingerprint else structure.fingerprint
        receipt = CanonicalizationReceipt(
            source_fingerprint=fingerprint,
            standardization_fingerprint="1" * 64,
            catalogue_fingerprint="2" * 64,
            hall_number=1,
            hall_symbol="P 1",
            source_to_primitive=(
                (Fraction(1, 2), Fraction(0), Fraction(0)),
                (Fraction(0), Fraction(1), Fraction(0)),
                (Fraction(0), Fraction(0), Fraction(1)),
            ),
            origin_shift=(Fraction(0), Fraction(0), Fraction(0)),
            free_origin_shift=(Fraction(0), Fraction(0), Fraction(0)),
            translation_index=2,
            symmetry_tolerance=1.0e-8,
            roundtrip_max_cartesian_error=0.0,
            symmetry_stability_verified=True,
            catalogue_closure_verified=True,
            wyckoff_multiplicity_verified=True,
            symmetry_verified=self.pass_gates,
            canonical_origin_verified=True,
            scaffold_roundtrip_verified=True,
        )
        return CanonicalizedStructure(primitive, primitive, scaffold, receipt)


def _hidden_translation_raw() -> RawPeriodicStructure:
    return RawPeriodicStructure.from_arrays(
        np.eye(3) * 4.0,
        [[0.0, 0.0, 0.0], [0.5, 0.0, 0.0]],
        [8, 8],
        labels=["O1", "O2"],
    )


class _NumericalStandardizer:
    def canonicalize(self, structure: FloatingStructure) -> NumericalCanonicalizedStructure:
        assert len(structure.species) == 2
        lattice = np.diag([2.0, 4.0, 4.0])
        provenance = StandardizationProvenance(
            backend="test",
            backend_version="1",
            hall_map_fingerprint="2" * 64,
            hall_number=1,
            symprec=1.0e-8,
            angle_tolerance=-1.0,
            canonical_decimals=12,
            input_atom_count=2,
            primitive_atom_count=1,
            conventional_atom_count=1,
            stability_symprec_factors=(0.5, 2.0),
            symmetry_stability_verified=True,
        )
        return NumericalCanonicalizedStructure(
            international_number=1,
            hall_number=1,
            hall_symbol="P 1",
            international_symbol="P1",
            setting_choice="",
            conventional_lattice=lattice,
            conventional_fractional_positions=np.zeros((1, 3)),
            conventional_species=(8,),
            wyckoff_letters=("a",),
            site_symmetry_symbols=("1",),
            crystallographic_orbits=(0,),
            primitive_lattice=lattice,
            primitive_fractional_positions=np.zeros((1, 3)),
            primitive_species=(8,),
            raw_translation_subgroup=np.asarray([[0.0, 0.0, 0.0], [0.5, 0.0, 0.0]]),
            input_mapping_to_primitive=(0, 0),
            transformation_matrix=np.eye(3),
            spglib_origin_shift=np.zeros(3),
            free_origin_axes=(0, 1, 2),
            subtracted_free_origin=np.zeros(3),
            provenance=provenance,
        )


def test_raw_adapter_revalidates_numerical_conventional_and_primitive_cells() -> None:
    standardized = standardize_raw_structure(
        _hidden_translation_raw(), standardizer=_NumericalStandardizer()
    )
    assert standardized.numerical_result.raw_translation_index == 2
    assert standardized.primitive_structure.atomic_numbers == (8,)
    assert standardized.conventional_structure.atomic_numbers == (8,)
    assert standardized.raw_structure.volume == pytest.approx(
        2.0 * standardized.primitive_structure.volume
    )


def test_catalogue_assignment_refuses_uncertified_numerical_translations() -> None:
    standardized = standardize_raw_structure(
        _hidden_translation_raw(), standardizer=_NumericalStandardizer()
    )
    assigner = object.__new__(CatalogueTargetAssigner)
    with pytest.raises(InvariantViolation, match="exact translation certificate"):
        assigner._exact_primitive_basis(standardized)


def test_training_boundary_requires_primitive_quotient_and_all_gates() -> None:
    raw = _hidden_translation_raw()
    record = structure_to_exact_fixture_target(raw, canonicalizer=_FixtureCanonicalizer())
    assert record.canonicalized.receipt.translation_index == 2
    assert record.target.composition.counts == ((8, 1),)
    assert record.target.canonical_scaffold.nu == 1

    with pytest.raises(InvariantViolation, match="proof gates"):
        structure_to_exact_fixture_target(
            raw, canonicalizer=_FixtureCanonicalizer(pass_gates=False)
        )
    with pytest.raises(InvariantViolation, match="different raw input"):
        structure_to_exact_fixture_target(
            raw, canonicalizer=_FixtureCanonicalizer(wrong_fingerprint=True)
        )

    valid = _FixtureCanonicalizer().canonicalize(raw).receipt
    with pytest.raises(DomainError, match="determinant"):
        replace(
            valid,
            source_to_primitive=(
                (Fraction(1), Fraction(0), Fraction(0)),
                (Fraction(0), Fraction(1), Fraction(0)),
                (Fraction(0), Fraction(0), Fraction(1)),
            ),
        )


def test_training_boundary_detects_wrong_translation_index() -> None:
    class WrongIndex(_FixtureCanonicalizer):
        def canonicalize(self, structure: RawPeriodicStructure) -> CanonicalizedStructure:
            result = super().canonicalize(structure)
            return replace(
                result,
                receipt=replace(
                    result.receipt,
                    translation_index=1,
                    source_to_primitive=(
                        (Fraction(1), Fraction(0), Fraction(0)),
                        (Fraction(0), Fraction(1), Fraction(0)),
                        (Fraction(0), Fraction(0), Fraction(1)),
                    ),
                ),
            )

    with pytest.raises(InvariantViolation, match="atom counts"):
        structure_to_exact_fixture_target(_hidden_translation_raw(), canonicalizer=WrongIndex())


def test_exact_fixture_wrapper_checks_scaffold_point_roundtrip() -> None:
    class WrongScaffold(_FixtureCanonicalizer):
        def canonicalize(self, structure: RawPeriodicStructure) -> CanonicalizedStructure:
            result = super().canonicalize(structure)
            wrong = RawPeriodicStructure.from_arrays(
                result.conventional_structure.lattice,
                [[0.125, 0.0, 0.0]],
                [8],
            )
            return replace(result, conventional_structure=wrong)

    with pytest.raises(InvariantViolation, match="periodic round-trip"):
        structure_to_exact_fixture_target(
            _hidden_translation_raw(), canonicalizer=WrongScaffold()
        )
