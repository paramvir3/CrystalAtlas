from __future__ import annotations

from collections import Counter
from fractions import Fraction
from itertools import permutations, product

import numpy as np
import pytest

from crystalatlas.algebra import exact_determinant
from crystalatlas.catalogue import MINIMAL_CATALOGUE
from crystalatlas.errors import (
    AmbiguousPreprocessingError,
    CatalogueError,
    DomainError,
    NumericalError,
)
from crystalatlas.full_catalogue import pinned_hall_numbers
from crystalatlas.primitive_canonicalization import canonicalize_primitive_presentation
from crystalatlas.standardization import (
    FloatingStructure,
    SpglibPolicy,
    SpglibStandardizer,
    canonicalize_exact_origin,
    exact_full_primitive_reduction,
    validate_origin_chart_compatibility,
)

pytestmark = pytest.mark.filterwarnings("ignore:Set OLD_ERROR_HANDLING.*:DeprecationWarning")


def test_exact_full_primitive_reduction_finds_indices_two_and_three() -> None:
    half = Fraction(1, 2)
    third = Fraction(1, 3)
    zero = Fraction(0)
    index_two = exact_full_primitive_reduction(
        ((zero, zero, zero), (half, zero, zero)),
        ("X", "X"),
    )
    assert index_two.translation_index == 2
    assert index_two.basis_change == (
        (half, zero, zero),
        (zero, Fraction(1), zero),
        (zero, zero, Fraction(1)),
    )
    assert index_two.inverse_basis_change == ((2, 0, 0), (0, 1, 0), (0, 0, 1))

    index_three = exact_full_primitive_reduction(
        (
            (zero, zero, zero),
            (third, zero, zero),
            (2 * third, zero, zero),
        ),
        ("X", "X", "X"),
    )
    assert index_three.translation_index == 3
    assert index_three.basis_change[0][0] == third
    assert len(index_three.fractional_positions) == 1


def test_exact_oblique_translation_reduction_is_right_handed_and_round_trips() -> None:
    half = Fraction(1, 2)
    zero = Fraction(0)
    positions = (
        (Fraction(1, 7), Fraction(2, 9), Fraction(3, 11)),
        (Fraction(1, 7) + half, Fraction(2, 9) + half, Fraction(3, 11)),
        (Fraction(4, 7), Fraction(1, 9), Fraction(2, 11)),
        (Fraction(4, 7) + half, Fraction(1, 9) + half, Fraction(2, 11)),
    )
    positions = tuple(
        tuple(value - value.numerator // value.denominator for value in point)
        for point in positions
    )
    reduced = exact_full_primitive_reduction(positions, ("A", "A", "B", "B"))
    assert reduced.translation_index == 2
    assert reduced.basis_change == (
        (half, half, zero),
        (zero, Fraction(1), zero),
        (zero, zero, Fraction(1)),
    )
    assert reduced.inverse_basis_change == ((2, -1, 0), (0, 1, 0), (0, 0, 1))
    assert Counter(reduced.expand_to_input_cell()) == Counter(
        zip(positions, ("A", "A", "B", "B"), strict=True)
    )
    input_lattice = np.asarray([[2.1, 0.2, 0.1], [0.0, 1.8, 0.3], [0.1, 0.2, 2.4]])
    primitive_lattice = reduced.lattice_from_input(input_lattice)
    np.testing.assert_allclose(
        np.linalg.det(primitive_lattice),
        np.linalg.det(input_lattice) / reduced.translation_index,
        rtol=1.0e-14,
        atol=1.0e-14,
    )
    assert np.linalg.det(primitive_lattice) > 0.0


def test_exact_origin_section_is_translation_and_permutation_invariant() -> None:
    positions = (
        (Fraction(1, 7), Fraction(1, 5), Fraction(2, 9)),
        (Fraction(4, 7), Fraction(3, 5), Fraction(8, 9)),
        (Fraction(2, 7), Fraction(4, 5), Fraction(1, 9)),
    )
    species = ("B", "A", "A")
    reference = canonicalize_exact_origin(positions, species, free_axes=(0, 2))
    shift = (Fraction(3, 11), Fraction(0), Fraction(5, 13))
    translated = tuple(
        tuple((point[axis] + shift[axis]) % 1 for axis in range(3)) for point in positions
    )
    permutation = (2, 0, 1)
    equivalent = canonicalize_exact_origin(
        tuple(translated[index] for index in permutation),
        tuple(species[index] for index in permutation),
        free_axes=(2, 0),
    )
    assert equivalent.fractional_positions == reference.fractional_positions
    assert equivalent.species == reference.species
    # The constrained y coordinate is never moved by the origin quotient.
    assert sorted(point[1] for point in reference.fractional_positions) == sorted(
        point[1] for point in positions
    )


def test_exact_fixture_origin_charts_pass_the_general_certificate() -> None:
    axes = validate_origin_chart_compatibility(
        MINIMAL_CATALOGUE,
        require_all_230=False,
    )
    assert dict(axes) == {"P1": (0, 1, 2), "P-1": ()}


def test_floating_structure_rejects_left_handed_input_instead_of_mirroring() -> None:
    with pytest.raises(DomainError, match="right handed"):
        FloatingStructure(
            np.diag([-1.0, 1.0, 1.0]),
            np.asarray([[0.0, 0.0, 0.0]]),
            ("X",),
        )


def test_spglib_backend_version_is_a_hard_dataset_identity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    standardizer = SpglibStandardizer(
        {1: 1},
        SpglibPolicy(required_spglib_version="9.9.9"),
    )

    class FakeSpglib:
        __version__ = "9.9.8"

    monkeypatch.setattr(
        "crystalatlas.standardization.require_module",
        lambda *_args, **_kwargs: FakeSpglib(),
    )
    with pytest.raises(NumericalError, match="version mismatch"):
        standardizer._load_spglib()


def test_spglib_p1_standardization_is_proper_basis_and_origin_invariant() -> None:
    spglib = pytest.importorskip("spglib")
    policy = SpglibPolicy(required_spglib_version=str(spglib.__version__))
    standardizer = SpglibStandardizer({1: 1}, policy)
    lattice = np.asarray(
        [[2.1, 0.2, 0.1], [0.0, 1.8, 0.3], [0.1, 0.2, 2.4]],
        dtype=np.float64,
    )
    positions = np.asarray(
        [
            [0.13, 0.27, 0.39],
            [0.45, 0.11, 0.73],
            [0.22, 0.61, 0.08],
            [0.79, 0.42, 0.56],
        ],
        dtype=np.float64,
    )
    species = ("A", "B", "C", "D")
    reference = standardizer.canonicalize(FloatingStructure(lattice, positions, species))
    basis = np.asarray([[1, 1, 0], [0, 1, 0], [0, 0, 1]], dtype=np.int64)
    shifted = np.mod(positions @ np.linalg.inv(basis) + [0.17, 0.29, 0.31], 1.0)
    equivalent = standardizer.canonicalize(
        FloatingStructure(
            basis @ lattice,
            shifted[[2, 0, 3, 1]],
            tuple(species[index] for index in (2, 0, 3, 1)),
        )
    )
    assert reference.hall_number == equivalent.hall_number == 1
    assert reference.free_origin_axes == equivalent.free_origin_axes == (0, 1, 2)
    assert reference.conventional_species == equivalent.conventional_species
    np.testing.assert_allclose(
        reference.conventional_fractional_positions,
        equivalent.conventional_fractional_positions,
        atol=policy.roundtrip_tolerance,
        rtol=0.0,
    )
    np.testing.assert_allclose(
        reference.conventional_lattice,
        equivalent.conventional_lattice,
        atol=policy.roundtrip_tolerance,
        rtol=policy.roundtrip_tolerance,
    )

    # Reflect Cartesian space, then use a determinant-minus-one fractional
    # basis relabelling only to restore a right-handed matrix presentation.
    # The physical reflection must not collapse onto the SO(3) target.
    reflection = np.diag([-1, 1, 1])
    mirrored = standardizer.canonicalize(
        FloatingStructure(
            reflection @ lattice @ reflection,
            np.mod(positions @ reflection, 1.0),
            species,
        )
    )
    assert mirrored.conventional_species == reference.conventional_species
    assert not np.allclose(
        mirrored.conventional_fractional_positions,
        reference.conventional_fractional_positions,
        atol=policy.roundtrip_tolerance,
        rtol=0.0,
    )


def test_spglib_removes_a_generic_hidden_translation_before_standardization() -> None:
    spglib = pytest.importorskip("spglib")
    policy = SpglibPolicy(required_spglib_version=str(spglib.__version__))
    standardizer = SpglibStandardizer({1: 1}, policy)
    primitive_lattice = np.asarray(
        [[2.1, 0.2, 0.1], [0.0, 1.8, 0.3], [0.1, 0.2, 2.4]],
        dtype=np.float64,
    )
    primitive_positions = np.asarray(
        [
            [0.13, 0.27, 0.39],
            [0.45, 0.11, 0.73],
            [0.22, 0.61, 0.08],
            [0.79, 0.42, 0.56],
        ],
        dtype=np.float64,
    )
    primitive_species = ("A", "B", "C", "D")
    primitive_to_raw = np.diag([0.5, 1.0, 1.0])
    raw_lattice = np.diag([2.0, 1.0, 1.0]) @ primitive_lattice
    first_copy = primitive_positions @ primitive_to_raw
    raw_positions = np.concatenate(
        (first_copy, np.mod(first_copy + [0.5, 0.0, 0.0], 1.0)), axis=0
    )
    result = standardizer.canonicalize(
        FloatingStructure(
            raw_lattice,
            raw_positions,
            primitive_species + primitive_species,
        )
    )
    assert result.international_number == 1
    assert result.raw_translation_index == 2
    assert len(result.primitive_species) == 4
    assert len(result.conventional_species) == 4
    assert result.provenance.input_atom_count == 8
    assert result.provenance.primitive_atom_count == 4
    assert result.input_mapping_to_primitive == (0, 1, 2, 3, 0, 1, 2, 3)
    np.testing.assert_allclose(
        sorted(result.raw_translation_subgroup.tolist()),
        [[0.0, 0.0, 0.0], [0.5, 0.0, 0.0]],
        atol=1.0e-12,
        rtol=0.0,
    )


def test_spglib_certifies_one_third_translation_without_decimal_quantization() -> None:
    spglib = pytest.importorskip("spglib")
    policy = SpglibPolicy(required_spglib_version=str(spglib.__version__))
    standardizer = SpglibStandardizer({1: 1}, policy)
    primitive_lattice = np.asarray(
        [[2.1, 0.2, 0.1], [0.0, 1.8, 0.3], [0.1, 0.2, 2.4]],
        dtype=np.float64,
    )
    primitive_positions = np.asarray(
        [
            [0.13, 0.27, 0.39],
            [0.45, 0.11, 0.73],
            [0.22, 0.61, 0.08],
            [0.79, 0.42, 0.56],
        ],
        dtype=np.float64,
    )
    primitive_species = ("A", "B", "C", "D")
    primitive_to_raw = np.diag([1.0 / 3.0, 1.0, 1.0])
    raw_lattice = np.diag([3.0, 1.0, 1.0]) @ primitive_lattice
    first_copy = primitive_positions @ primitive_to_raw
    raw_positions = np.concatenate(
        tuple(np.mod(first_copy + [copy / 3.0, 0.0, 0.0], 1.0) for copy in range(3)),
        axis=0,
    )
    result = standardizer.canonicalize(
        FloatingStructure(
            raw_lattice,
            raw_positions,
            primitive_species * 3,
        )
    )
    zero = Fraction(0)
    third = Fraction(1, 3)
    assert result.raw_translation_index == 3
    assert result.exact_raw_translation_subgroup == (
        (zero, zero, zero),
        (third, zero, zero),
        (2 * third, zero, zero),
    )
    assert result.exact_input_to_raw_frame_primitive_basis is not None
    assert result.exact_inverse_input_to_raw_frame_primitive_basis is not None
    assert exact_determinant(result.exact_input_to_raw_frame_primitive_basis) == third
    assert result.raw_frame_primitive_lattice is not None
    assert result.raw_frame_to_standard_rotation is not None
    exact_basis = np.asarray(
        [
            [float(value) for value in row]
            for row in result.exact_input_to_raw_frame_primitive_basis
        ]
    )
    np.testing.assert_allclose(
        exact_basis @ raw_lattice,
        result.raw_frame_primitive_lattice,
        atol=policy.roundtrip_tolerance,
        rtol=policy.roundtrip_tolerance,
    )
    np.testing.assert_allclose(
        result.raw_frame_primitive_lattice @ result.raw_frame_to_standard_rotation,
        result.primitive_lattice,
        atol=policy.roundtrip_tolerance,
        rtol=policy.roundtrip_tolerance,
    )
    assert result.exact_translation_max_cartesian_error is not None
    assert result.exact_translation_max_cartesian_error <= policy.symprec


def test_spglib_f_centering_has_exact_count_and_volume_index() -> None:
    spglib = pytest.importorskip("spglib")
    standardizer = SpglibStandardizer(
        {225: 523},
        SpglibPolicy(required_spglib_version=str(spglib.__version__)),
    )
    centering = np.asarray([[0.0, 0.0, 0.0], [0.0, 0.5, 0.5], [0.5, 0.0, 0.5], [0.5, 0.5, 0.0]])
    result = standardizer.canonicalize(
        FloatingStructure(np.eye(3) * 4.0, centering, ("X",) * 4)
    )
    assert result.international_number == 225
    assert result.hall_number == 523
    assert result.raw_translation_index == 4
    assert len(result.conventional_species) == 4
    assert len(result.primitive_species) == 1
    assert result.input_mapping_to_primitive == (0, 0, 0, 0)
    np.testing.assert_allclose(
        np.linalg.det(result.primitive_lattice),
        np.linalg.det(result.conventional_lattice) / 4.0,
        atol=1.0e-12,
        rtol=1.0e-12,
    )


def test_spglib_p_minus_one_quotients_all_discrete_pinned_origins() -> None:
    spglib = pytest.importorskip("spglib")
    policy = SpglibPolicy(required_spglib_version=str(spglib.__version__))
    standardizer = SpglibStandardizer({2: 2}, policy)
    lattice = np.asarray(
        [[2.1, 0.2, 0.1], [0.0, 1.8, 0.3], [0.1, 0.2, 2.4]],
        dtype=np.float64,
    )
    positions = np.mod(
        np.asarray(
            [
                [0.13, 0.27, 0.39],
                [-0.13, -0.27, -0.39],
                [0.22, 0.61, 0.08],
                [-0.22, -0.61, -0.08],
            ]
        ),
        1.0,
    )
    species = ("A", "A", "B", "B")
    reference = standardizer.canonicalize(FloatingStructure(lattice, positions, species))
    assert reference.international_number == 2
    assert reference.free_origin_axes == ()
    for shift in ([0.173, 0.291, 0.317], [0.5, 0.0, 0.0]):
        equivalent = standardizer.canonicalize(
            FloatingStructure(lattice, np.mod(positions + shift, 1.0), species)
        )
        assert equivalent.conventional_species == reference.conventional_species
        np.testing.assert_allclose(
            equivalent.conventional_fractional_positions,
            reference.conventional_fractional_positions,
            atol=policy.roundtrip_tolerance,
            rtol=0.0,
        )

    # A finite normalizer shift changes the coordinate label 1h -> 1a.  Site
    # metadata must therefore be recomputed after, not carried through, the
    # canonical origin quotient.
    shifted_special = standardizer.canonicalize(
        FloatingStructure(lattice, np.asarray([[0.5, 0.5, 0.5]]), ("A",))
    )
    origin_special = standardizer.canonicalize(
        FloatingStructure(lattice, np.asarray([[0.0, 0.0, 0.0]]), ("A",))
    )
    np.testing.assert_allclose(
        shifted_special.conventional_fractional_positions,
        [[0.0, 0.0, 0.0]],
        atol=policy.roundtrip_tolerance,
        rtol=0.0,
    )
    assert shifted_special.wyckoff_letters == origin_special.wyckoff_letters == ("a",)
    assert shifted_special.site_symmetry_symbols == ("-1",)


def test_spglib_preflight_covers_all_230_pinned_origin_gauges() -> None:
    spglib = pytest.importorskip("spglib")
    complete = pinned_hall_numbers()
    assert set(complete) == set(range(1, 231))
    standardizer = SpglibStandardizer(
        complete,
        SpglibPolicy(required_spglib_version=str(spglib.__version__)),
    )
    standardizer.validate_catalogue_preflight()

    deleted = dict(complete)
    del deleted[230]
    incomplete = SpglibStandardizer(
        deleted,
        SpglibPolicy(required_spglib_version=str(spglib.__version__)),
    )
    with pytest.raises(CatalogueError, match="not an exact 230-type bijection"):
        incomplete.validate_catalogue_preflight()


@pytest.mark.parametrize("lengths", [(2.0, 2.0, 2.0), (2.0, 2.0, 3.0), (2.0, 3.0, 4.0)])
def test_p1_canonicalization_covers_proper_signed_bases_and_shears(lengths) -> None:
    """The former 23/24 cubic-presentation mismatch must remain impossible."""

    spglib = pytest.importorskip("spglib")
    standardizer = SpglibStandardizer(
        {1: 1}, SpglibPolicy(required_spglib_version=spglib.__version__)
    )
    lattice = np.diag(lengths)
    positions = np.array(
        [
            [0.13, 0.27, 0.39],
            [0.45, 0.11, 0.73],
            [0.22, 0.61, 0.08],
            [0.79, 0.42, 0.56],
        ]
    )
    species = (1, 2, 3, 4)
    reference = standardizer.canonicalize(FloatingStructure(lattice, positions, species))
    reorder = np.array([2, 0, 3, 1])
    proper_rotation = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    shears = (np.eye(3, dtype=int), np.array([[1, 1, 0], [0, 1, 1], [0, 0, 1]]))
    for order in permutations(range(3)):
        for signs in product((-1, 1), repeat=3):
            signed = np.eye(3, dtype=int)[list(order)] * np.array(signs)[:, None]
            if round(float(np.linalg.det(signed))) != 1:
                continue
            for shear in shears:
                basis = shear @ signed
                raw_lattice = basis @ lattice @ proper_rotation
                raw_positions = np.mod(
                    positions @ np.linalg.inv(basis) + [0.17, 0.29, 0.31], 1.0
                )[reorder]
                raw_species = tuple(species[index] for index in reorder)
                result = standardizer.canonicalize(
                    FloatingStructure(raw_lattice, raw_positions, raw_species)
                )
                assert result.conventional_species == reference.conventional_species
                assert result.wyckoff_letters == reference.wyckoff_letters
                np.testing.assert_allclose(
                    result.conventional_lattice,
                    reference.conventional_lattice,
                    atol=2.0e-10,
                    rtol=0.0,
                )
                np.testing.assert_allclose(
                    result.conventional_fractional_positions,
                    reference.conventional_fractional_positions,
                    atol=2.0e-10,
                    rtol=0.0,
                )
                # The effective affine map still starts at the original input,
                # not the temporary canonical primitive structure.
                mapped = np.mod(
                    raw_positions @ result.transformation_matrix.T
                    + result.spglib_origin_shift
                    - result.subtracted_free_origin,
                    1.0,
                )
                for index, atom_species in enumerate(raw_species):
                    target = result.conventional_species.index(atom_species)
                    delta = mapped[index] - result.conventional_fractional_positions[target]
                    delta -= np.rint(delta)
                    np.testing.assert_allclose(delta, 0.0, atol=2.0e-10, rtol=0.0)
                assert result.provenance.input_atom_count == len(raw_species)
                assert (
                    result.provenance.algorithm
                    == "fullprim-minimax-proper-primitive-pinned-hall-v6"
                )


def test_primitive_canonical_section_compares_different_metrics_and_keeps_mirrors() -> None:
    lattice = np.array([[2.0, 0.0, 0.0], [0.6, 2.2, 0.0], [0.3, 0.4, 2.7]])
    positions = np.array([[0.13, 0.27, 0.39], [0.45, 0.11, 0.73], [0.22, 0.61, 0.08]])
    species = (1, 2, 3)
    basis = np.array([[1, 1, 0], [0, 1, 0], [0, 0, 1]])
    transformed_lattice = basis @ lattice
    assert not np.allclose(lattice @ lattice.T, transformed_lattice @ transformed_lattice.T)
    reference = canonicalize_primitive_presentation(lattice, positions, species)
    transformed = canonicalize_primitive_presentation(
        transformed_lattice, np.mod(positions @ np.linalg.inv(basis), 1.0), species
    )
    mirror = canonicalize_primitive_presentation(lattice, np.mod(-positions, 1.0), species)
    np.testing.assert_allclose(reference.lattice, transformed.lattice, atol=1.0e-12, rtol=0.0)
    np.testing.assert_allclose(
        reference.fractional_positions,
        transformed.fractional_positions,
        atol=1.0e-12,
        rtol=0.0,
    )
    assert not np.allclose(reference.fractional_positions, mirror.fractional_positions)
    assert np.linalg.det(reference.cartesian_rotation) == pytest.approx(1.0)


def test_primitive_canonical_section_rejects_unresolved_ties_and_resource_exhaustion() -> None:
    with pytest.raises(AmbiguousPreprocessingError, match="ties between resolved"):
        canonicalize_primitive_presentation(
            np.diag([2.0, 2.0 + 4.0e-12, 2.0]), np.zeros((1, 3)), (1,)
        )
    with pytest.raises(AmbiguousPreprocessingError, match="resource limit"):
        canonicalize_primitive_presentation(
            np.diag([1.0, 1.0, 200.0]), np.zeros((1, 3)), (1,), decimals=6
        )
