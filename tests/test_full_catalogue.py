from __future__ import annotations

import importlib
from dataclasses import replace
from types import MappingProxyType

import pytest

from crystalatlas.catalogue import CertifiedCatalogue
from crystalatlas.certification import certify_crystallography
from crystalatlas.composition import Composition
from crystalatlas.errors import CatalogueError
from crystalatlas.full_catalogue import (
    FROZEN_AFFINE_OPERATION_COUNT,
    FROZEN_CATALOGUE_FINGERPRINT,
    FROZEN_SPACE_GROUP_TYPE_COUNT,
    FROZEN_WYCKOFF_ROW_COUNT,
    SPGLIB_SOURCE_COMMIT,
    SPGLIB_SOURCE_VERSION,
    build_full_catalogue,
    hall_metadata,
    pinned_hall_numbers,
    source_hall_settings,
    source_identity,
    source_rows_for_type,
    validate_against_spglib,
    validate_frozen_compiled_manifest,
)
from crystalatlas.geometry import ContinuousCover, IntrinsicCoverLayout, OrbitType
from crystalatlas.standardization import validate_origin_chart_compatibility

pytestmark = pytest.mark.filterwarnings(
    "ignore:Set OLD_ERROR_HANDLING.*:DeprecationWarning:spglib\\.spg"
)


def test_pinned_source_and_hall_manifests_are_exhaustive() -> None:
    identity = source_identity()
    assert identity["version"] == "2.7.0"
    assert identity["git_commit"] == "12355c77fb7c505a55f52cae36341d73b781a065"
    assert identity["license"] == "BSD-3-Clause"
    assert len(source_hall_settings()) == 530
    assert len(hall_metadata()) == 530
    assert set(pinned_hall_numbers()) == set(range(1, 231))
    assert tuple(pinned_hall_numbers().values()) == tuple(
        sorted(pinned_hall_numbers().values())
    )
    assert hall_metadata()[pinned_hall_numbers()[48]].choice == "1"
    assert source_rows_for_type(2)["1a"].site_symmetry_symbol == "-1"


def test_complete_catalogue_passes_exact_constructor_certification() -> None:
    catalogue = build_full_catalogue()
    assert catalogue.complete_space_group_types
    assert catalogue.supported_international_numbers == tuple(range(1, 231))
    assert len(catalogue.settings) == 230
    row_count = sum(
        len(setting.wyckoff_positions) for setting in catalogue.settings.values()
    )
    assert row_count == 1731
    assert catalogue.get(1).hall_symbol == "P 1"
    assert catalogue.get(2).hall_symbol == "-P 1"
    assert catalogue.get(146).hall_symbol == "R 3"
    assert catalogue.get(230).hall_symbol == "-I 4bd 2c 3"
    assert len(catalogue.settings) == FROZEN_SPACE_GROUP_TYPE_COUNT
    assert row_count == FROZEN_WYCKOFF_ROW_COUNT
    assert (
        sum(len(setting.operations) for setting in catalogue.settings.values())
        == FROZEN_AFFINE_OPERATION_COUNT
    )
    assert catalogue.fingerprint == FROZEN_CATALOGUE_FINGERPRINT
    origin_axes = validate_origin_chart_compatibility(catalogue)
    assert len(origin_axes) == 230
    assert {
        axes: sum(value == axes for value in origin_axes.values())
        for axes in set(origin_axes.values())
    } == {
        (): 162,
        (0, 1, 2): 1,
        (0, 2): 4,
        (1,): 3,
        (2,): 60,
    }

    # These checks are consequences of exact constructor validation, retained
    # here as visible regression assertions for catalogue consumers.
    for setting in catalogue.settings.values():
        group_order = len(setting.operations)
        assert frozenset(setting.wyckoff_positions) == frozenset(
            row.label for row in setting.wyckoff_positions.values()
        )
        for row in setting.wyckoff_positions.values():
            assert row.multiplicity * row.site_symmetry_order == group_order
            assert len(row.expand_exact(row.generic_parameter)) == row.multiplicity


def test_frozen_manifest_rejects_a_deleted_source_row() -> None:
    catalogue = build_full_catalogue()
    original = catalogue.get(1)
    deleted = replace(original, wyckoff_positions={})

    # This is the historical failure mode: when expected rows are derived from
    # the same compiler output, local setting validation cannot know that 1a
    # was deleted.  The independent frozen release manifest must still reject.
    deleted.validate(frozenset())
    mutated_settings = dict(catalogue.settings)
    mutated_settings[original.key] = deleted
    mutated_expected = {
        key: frozenset(setting.wyckoff_positions)
        for key, setting in mutated_settings.items()
    }
    mutated = object.__new__(CertifiedCatalogue)
    object.__setattr__(mutated, "_settings", MappingProxyType(mutated_settings))
    object.__setattr__(mutated, "_expected_rows", MappingProxyType(mutated_expected))
    object.__setattr__(mutated, "certification_scope", catalogue.certification_scope)
    object.__setattr__(mutated, "complete_space_group_types", True)

    with pytest.raises(CatalogueError, match="manifest count mismatch"):
        validate_frozen_compiled_manifest(mutated)


def test_every_wyckoff_chart_builds_the_certified_continuous_origin_quotient() -> None:
    catalogue = build_full_catalogue()
    origin_axes = validate_origin_chart_compatibility(catalogue)
    for setting in catalogue.settings.values():
        for row in setting.wyckoff_positions.values():
            cover = ContinuousCover(
                row.multiplicity,
                setting.key,
                Composition.from_counts({8: 1}),
                (OrbitType(row.label, 8),),
            )
            layout = IntrinsicCoverLayout(cover, catalogue=catalogue)
            assert layout.intrinsic_dimension == row.dimension - len(
                origin_axes[setting.key]
            )
            assert layout.period_basis.shape == (
                layout.intrinsic_dimension,
                layout.intrinsic_dimension,
            )


def test_all_operations_match_the_pinned_spglib_c_database() -> None:
    try:
        spglib = importlib.import_module("spglib")
    except ImportError:
        pytest.skip("requires the pinned crystallography extra")
    if getattr(spglib, "__version__", None) != SPGLIB_SOURCE_VERSION:
        pytest.fail(
            f"expected spglib {SPGLIB_SOURCE_VERSION} for commit {SPGLIB_SOURCE_COMMIT} gate"
        )
    validate_against_spglib(spglib)


def test_complete_catalogue_produces_a_training_certification_receipt() -> None:
    try:
        spglib = importlib.import_module("spglib")
    except ImportError:
        pytest.skip("requires the pinned crystallography extra")
    receipt = certify_crystallography(
        build_full_catalogue(),
        spglib_module=spglib,
    )
    assert receipt.passed
    assert receipt.complete_space_group_types
    assert receipt.external_operation_database_verified
    assert receipt.exact_origin_chart_verified
    assert receipt.pinned_standardization_preflight_verified
    assert receipt.space_group_count == 230
    assert receipt.wyckoff_row_count == 1731
    assert receipt.affine_operation_count == 4425
