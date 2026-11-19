"""Origin verification must use the numerical producer's species ordering.

In particular, atomic numbers are ordered by ``(type name, repr)`` before
spglib encoding: sulfur (16) precedes oxygen (8), not the other way around.
These tests exercise the real numerical producer without constructing the
expensive all-230 exact catalogue.
"""

from dataclasses import replace

import numpy as np
import pytest

from crystalatlas import dataset_builder
from crystalatlas.catalogue import MINIMAL_CATALOGUE
from crystalatlas.config import DomainConfig
from crystalatlas.dataset_builder import CatalogueDatasetTargetBuilder
from crystalatlas.errors import InvariantViolation
from crystalatlas.full_catalogue import pinned_hall_numbers
from crystalatlas.ingestion import (
    CANONICAL_ORIGIN_VERIFIER_ID,
    CatalogueTargetAssigner,
    RawPeriodicStructure,
    StandardizedRawStructure,
    _verify_canonical_free_origin,
    standardize_raw_structure,
)
from crystalatlas.standardization import SpglibPolicy, SpglibStandardizer


@pytest.fixture
def standardized(request: pytest.FixtureRequest) -> StandardizedRawStructure:
    spglib = pytest.importorskip("spglib")
    group = int(request.param)
    if group == 99:
        lattice = np.diag([4.0, 4.0, 6.0])
        positions = np.asarray([[0.0, 0.0, 0.173], [0.0, 0.0, 0.417]])
        species = (8, 16)
    else:
        lattice = np.asarray([[3.1, 0.2, 0.1], [0.0, 3.8, 0.3], [0.1, 0.2, 4.4]])
        positions = np.asarray(
            [[0.13, 0.27, 0.39], [0.45, 0.11, 0.73], [0.22, 0.61, 0.08]]
        )
        species = (8, 14, 16)
        if group == 2:
            positions = np.concatenate([positions, np.mod(-positions, 1.0)])
            species *= 2
    standardizer = SpglibStandardizer(
        {group: pinned_hall_numbers()[group]},
        SpglibPolicy(required_spglib_version=str(spglib.__version__)),
    )
    result = standardize_raw_structure(
        RawPeriodicStructure.from_arrays(lattice, positions, species),
        standardizer=standardizer,
    )
    assert result.numerical_result.international_number == group
    return result


@pytest.mark.parametrize("standardized", (1, 2, 99), indirect=True)
def test_real_standardizer_origin_is_accepted(standardized: StandardizedRawStructure) -> None:
    result = standardized.numerical_result
    # The regression is sensitive to a mismatch between numeric and repr order.
    assert result.conventional_species[0] == min(
        result.conventional_species, key=lambda value: (type(value).__name__, repr(value))
    )
    assert result.conventional_species[0] != min(result.conventional_species)
    _verify_canonical_free_origin(standardized, decimals=12)


@pytest.mark.parametrize("standardized", (1, 99), indirect=True)
def test_shifted_free_origin_is_rejected(standardized: StandardizedRawStructure) -> None:
    result = standardized.numerical_result
    assert result.free_origin_axes
    shift = np.zeros(3)
    shift[list(result.free_origin_axes)] = 0.137
    mutated = replace(
        result,
        conventional_fractional_positions=np.mod(
            result.conventional_fractional_positions + shift, 1.0
        ),
    )
    with pytest.raises(InvariantViolation, match="free-origin section is not canonical"):
        _verify_canonical_free_origin(
            replace(standardized, numerical_result=mutated), decimals=12
        )


def test_dataset_builder_fingerprint_binds_origin_verifier(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Fingerprint-only fixture: no target assignment or catalogue certification
    # is performed, so use an uninitialized assigner with small known metadata.
    assigner = object.__new__(CatalogueTargetAssigner)
    assigner.catalogue = MINIMAL_CATALOGUE
    assigner.hall_map_fingerprint = "a" * 64
    assigner.rational_decimals = 12
    assigner.roundtrip_tolerance = 2.0e-8
    # The legacy default; __init__ always sets it (review B1 orbit merging).
    assigner.merge_quantization_equivalent_orbits = False
    builder = CatalogueDatasetTargetBuilder(
        standardizer=SpglibStandardizer({1: 1}, SpglibPolicy()),
        target_assigner=assigner,
        domain=DomainConfig(),
    )
    payloads: list[dict[str, object]] = []
    fingerprint = dataset_builder._fingerprint

    def capture(payload: object) -> str:
        assert isinstance(payload, dict)
        payloads.append(payload)
        return fingerprint(payload)

    monkeypatch.setattr(dataset_builder, "_fingerprint", capture)
    current_digest = builder.builder_fingerprint
    payload = payloads[-1]
    assert payload["schema"] == "crystalatlas.catalogue-dataset-target-builder.v3"
    assert payload["standardizer"]["algorithm_id"] == builder.standardizer.algorithm_id
    assert payload["canonical_origin_verifier"] == CANONICAL_ORIGIN_VERIFIER_ID
    legacy_payload = dict(payload)
    legacy_payload["schema"] = "crystalatlas.catalogue-dataset-target-builder.v1"
    del legacy_payload["canonical_origin_verifier"]
    legacy_payload["standardizer"] = dict(legacy_payload["standardizer"])
    del legacy_payload["standardizer"]["algorithm_id"]
    assert current_digest != fingerprint(legacy_payload)
    previous_payload = dict(payload)
    previous_payload["schema"] = "crystalatlas.catalogue-dataset-target-builder.v2"
    previous_payload["standardizer"] = legacy_payload["standardizer"]
    assert current_digest != fingerprint(previous_payload)
    monkeypatch.setattr(dataset_builder, "CANONICAL_ORIGIN_VERIFIER_ID", "altered-contract")
    assert builder.builder_fingerprint != current_digest


@pytest.mark.parametrize("standardized", (2,), indirect=True)
def test_numeric_order_is_rejected_for_fixed_origin(
    standardized: StandardizedRawStructure,
) -> None:
    result = standardized.numerical_result
    assert not result.free_origin_axes
    order = sorted(
        range(len(result.conventional_species)),
        key=lambda index: (
            result.conventional_species[index],
            tuple(result.conventional_fractional_positions[index]),
        ),
    )
    mutated = replace(
        result,
        conventional_species=tuple(result.conventional_species[index] for index in order),
        conventional_fractional_positions=result.conventional_fractional_positions[order],
    )
    with pytest.raises(InvariantViolation, match="fixed-origin conventional motif"):
        _verify_canonical_free_origin(
            replace(standardized, numerical_result=mutated), decimals=12
        )
