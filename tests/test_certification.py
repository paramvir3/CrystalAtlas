from __future__ import annotations

import json

import pytest

from crystalatlas.catalogue import (
    MINIMAL_CATALOGUE,
    CertifiedCatalogue,
)
from crystalatlas.certification import CrystallographyCertification, certify_crystallography
from crystalatlas.errors import CatalogueError, DomainError


def test_exact_fixture_certification_is_bound_to_catalogue() -> None:
    receipt = certify_crystallography(
        MINIMAL_CATALOGUE,
        require_all_230=False,
        crosscheck_external_operations=False,
    )
    assert receipt.passed
    assert receipt.space_group_count == 2
    assert receipt.wyckoff_row_count == 10
    assert receipt.affine_operation_count == 3
    receipt.require_for(
        MINIMAL_CATALOGUE,
        require_all_230=False,
        require_external_operation_database=False,
    )


def test_fixture_receipt_cannot_authorize_production_training() -> None:
    receipt = certify_crystallography(
        MINIMAL_CATALOGUE,
        require_all_230=False,
        crosscheck_external_operations=False,
    )
    with pytest.raises(CatalogueError, match="all-230"):
        receipt.require_for(
            MINIMAL_CATALOGUE,
            require_all_230=True,
            require_external_operation_database=False,
        )
    with pytest.raises(CatalogueError, match="external operation"):
        receipt.require_for(
            MINIMAL_CATALOGUE,
            require_all_230=False,
            require_external_operation_database=True,
        )


def test_receipt_rejects_semantically_different_live_catalogue() -> None:
    receipt = certify_crystallography(
        MINIMAL_CATALOGUE,
        require_all_230=False,
        crosscheck_external_operations=False,
    )
    changed = CertifiedCatalogue(
        MINIMAL_CATALOGUE.settings,
        {
            key: frozenset(setting.wyckoff_positions)
            for key, setting in MINIMAL_CATALOGUE.settings.items()
        },
        certification_scope="different exact unit-test scope",
    )
    with pytest.raises(CatalogueError, match="different catalogue"):
        receipt.require_for(
            changed,
            require_all_230=False,
            require_external_operation_database=False,
        )


def test_receipt_json_round_trip_is_strict_and_live_catalogue_bound(tmp_path) -> None:
    receipt = certify_crystallography(
        MINIMAL_CATALOGUE,
        require_all_230=False,
        crosscheck_external_operations=False,
    )
    restored = CrystallographyCertification.from_dict(receipt.to_dict())
    assert restored == receipt
    path = tmp_path / "crystallography-receipt.json"
    path.write_text(json.dumps(receipt.to_dict()), encoding="utf-8")
    loaded = CrystallographyCertification.read_json(path)
    loaded.require_for(
        MINIMAL_CATALOGUE,
        require_all_230=False,
        require_external_operation_database=False,
    )

    tampered = receipt.to_dict()
    tampered["exact_symmetry_verified"] = False
    with pytest.raises(DomainError, match="internally inconsistent"):
        CrystallographyCertification.from_dict(tampered)

    extra = receipt.to_dict()
    extra["unrecognized"] = True
    with pytest.raises(DomainError, match="fields disagree"):
        CrystallographyCertification.from_dict(extra)
