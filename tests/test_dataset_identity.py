from __future__ import annotations

import json
from dataclasses import replace
from fractions import Fraction
from hashlib import sha256

import numpy as np
import pytest

from crystalatlas import dataset_identity
from crystalatlas.catalogue import MINIMAL_CATALOGUE
from crystalatlas.dataset_identity import (
    AmbiguousIdentityError,
    DeduplicationLevel,
    IdentityPolicy,
    build_canonical_identity,
    deduplicate_canonical_records,
)
from crystalatlas.decoder import CrystalScaffold, ExactCrystalDecoder, OrbitAssignment
from crystalatlas.errors import InvariantViolation
from crystalatlas.ingestion import (
    CanonicalizationReceipt,
    CanonicalizedStructure,
    CanonicalTrainingRecord,
    RawPeriodicStructure,
)
from crystalatlas.training_targets import ExactFixtureTrainingTarget, lift_exact_fixture_target


def _record(
    *,
    shift: Fraction = Fraction(0),
    scale: float = 1.0,
    source_id: str = "source",
    raw_supercell: bool = False,
    species_order: tuple[int, int, int] = (6, 7, 8),
) -> CanonicalTrainingRecord[ExactFixtureTrainingTarget]:
    parameters = (
        (Fraction(0), Fraction(0), Fraction(0)),
        (Fraction(1, 7) + shift, Fraction(2, 9), Fraction(3, 11)),
        (Fraction(2, 5), Fraction(3, 8), Fraction(4, 13)),
    )
    scaffold = CrystalScaffold.from_mapping(
        nu=1,
        space_group="P1",
        reduced_composition={species: 1 for species in species_order},
        orbits=tuple(
            OrbitAssignment("1a", species, parameter)
            for species, parameter in zip(species_order, parameters, strict=True)
        ),
    )
    lattice = np.diag([4.0, 5.0, 6.0]) * scale
    metric = lattice @ lattice.T
    target = lift_exact_fixture_target(scaffold, metric, catalogue=MINIMAL_CATALOGUE)
    decoded = ExactCrystalDecoder(MINIMAL_CATALOGUE).decode(
        target.canonical_scaffold, target.metric
    )
    conventional = RawPeriodicStructure.from_arrays(
        lattice,
        decoded.fractional_positions_float,
        tuple(int(value) for value in decoded.species),
        source_id="canonical",
    )
    if raw_supercell:
        raw_positions = np.concatenate(
            (
                conventional.fractional_positions * np.asarray([0.5, 1.0, 1.0]),
                conventional.fractional_positions * np.asarray([0.5, 1.0, 1.0])
                + np.asarray([0.5, 0.0, 0.0]),
            )
        )
        raw = RawPeriodicStructure.from_arrays(
            np.diag([8.0, 5.0, 6.0]) * scale,
            raw_positions,
            conventional.atomic_numbers * 2,
            source_id=source_id,
        )
        source_to_primitive = (
            (Fraction(1, 2), Fraction(0), Fraction(0)),
            (Fraction(0), Fraction(1), Fraction(0)),
            (Fraction(0), Fraction(0), Fraction(1)),
        )
        translation_index = 2
    else:
        raw = RawPeriodicStructure.from_arrays(
            conventional.lattice,
            conventional.fractional_positions,
            conventional.atomic_numbers,
            source_id=source_id,
        )
        source_to_primitive = (
            (Fraction(1), Fraction(0), Fraction(0)),
            (Fraction(0), Fraction(1), Fraction(0)),
            (Fraction(0), Fraction(0), Fraction(1)),
        )
        translation_index = 1
    receipt = CanonicalizationReceipt(
        source_fingerprint=raw.fingerprint,
        standardization_fingerprint="1" * 64,
        catalogue_fingerprint=MINIMAL_CATALOGUE.fingerprint,
        hall_number=1,
        hall_symbol="P 1",
        source_to_primitive=source_to_primitive,
        origin_shift=(Fraction(0), Fraction(0), Fraction(0)),
        free_origin_shift=(Fraction(0), Fraction(0), Fraction(0)),
        translation_index=translation_index,
        symmetry_tolerance=1.0e-8,
        roundtrip_max_cartesian_error=0.0,
        symmetry_stability_verified=True,
        catalogue_closure_verified=True,
        wyckoff_multiplicity_verified=True,
        symmetry_verified=True,
        canonical_origin_verified=True,
        scaffold_roundtrip_verified=True,
    )
    canonicalized = CanonicalizedStructure(
        conventional, conventional, target.canonical_scaffold, receipt
    )
    return CanonicalTrainingRecord(raw, canonicalized, target)


def _policy() -> IdentityPolicy:
    return IdentityPolicy(require_all_230=False)


def test_supercell_and_source_setting_metadata_do_not_change_physical_identity() -> None:
    primitive = build_canonical_identity(
        _record(source_id="primitive"), catalogue=MINIMAL_CATALOGUE, policy=_policy()
    )
    supercell = build_canonical_identity(
        _record(source_id="supercell", raw_supercell=True),
        catalogue=MINIMAL_CATALOGUE,
        policy=_policy(),
    )
    assert primitive.exact_key == supercell.exact_key
    assert primitive.structure_key == supercell.structure_key
    assert primitive.polymorph_key == supercell.polymorph_key


def test_distinct_polymorphs_are_preserved_but_uniform_scale_is_an_explicit_quotient() -> None:
    reference = build_canonical_identity(
        _record(), catalogue=MINIMAL_CATALOGUE, policy=_policy()
    )
    shifted = build_canonical_identity(
        _record(shift=Fraction(1, 17)), catalogue=MINIMAL_CATALOGUE, policy=_policy()
    )
    scaled = build_canonical_identity(
        _record(scale=1.1), catalogue=MINIMAL_CATALOGUE, policy=_policy()
    )
    assert reference.polymorph_key != shifted.polymorph_key
    # A scaffold is the discrete (nu, Hall setting, Wyckoff x species counts)
    # object.  Continuous intrinsic coordinates belong to geometry, not to h.
    assert reference.scaffold_key == shifted.scaffold_key
    # The anonymous prototype is the species-erased Wyckoff-role template;
    # free-coordinate distortions must therefore remain in the same holdout group.
    assert reference.prototype_key == shifted.prototype_key
    assert reference.structure_key != scaled.structure_key
    assert reference.polymorph_key == scaled.polymorph_key


def test_polymorph_dedup_keeps_aliases_and_distinct_phases() -> None:
    records = (
        _record(source_id="a"),
        _record(scale=1.1, source_id="b"),
        _record(shift=Fraction(1, 17), source_id="c"),
    )
    result = deduplicate_canonical_records(
        records,
        catalogue=MINIMAL_CATALOGUE,
        policy=_policy(),
        level=DeduplicationLevel.POLYMORPH,
    )
    assert len(result.groups) == 2
    assert sorted(len(group.member_indices) for group in result.groups) == [1, 2]
    assert len({identity.composition_key for identity in result.identities}) == 1


def test_anonymous_prototype_is_invariant_to_species_label_permutation() -> None:
    original = build_canonical_identity(
        _record(species_order=(6, 7, 8)),
        catalogue=MINIMAL_CATALOGUE,
        policy=_policy(),
    )
    swapped = build_canonical_identity(
        _record(species_order=(7, 6, 8)),
        catalogue=MINIMAL_CATALOGUE,
        policy=_policy(),
    )
    assert original.polymorph_key != swapped.polymorph_key
    assert original.prototype_key == swapped.prototype_key


def test_identity_boundary_and_receipt_tampering_fail_closed() -> None:
    boundary = _record(shift=Fraction(1, 20_000_000) - Fraction(1, 7))
    with pytest.raises(AmbiguousIdentityError, match="torus-cell boundary"):
        build_canonical_identity(
            boundary,
            catalogue=MINIMAL_CATALOGUE,
            policy=_policy(),
        )

    record = _record()
    tampered = replace(
        record,
        canonicalized=replace(
            record.canonicalized,
            receipt=replace(record.canonicalized.receipt, source_fingerprint="0" * 64),
        ),
    )
    with pytest.raises(InvariantViolation, match="different raw structure"):
        build_canonical_identity(
            tampered,
            catalogue=MINIMAL_CATALOGUE,
            policy=_policy(),
        )


def test_mixed_width_atomic_numbers_use_numeric_identity_projection() -> None:
    record = _record(species_order=(8, 14, 16))
    original_composition = record.target.canonical_scaffold.reduced_composition
    assert original_composition == ((14, 1), (16, 1), (8, 1))
    identity = build_canonical_identity(record, catalogue=MINIMAL_CATALOGUE, policy=_policy())
    numeric_composition = [[8, 1], [14, 1], [16, 1]]
    for payload in (
        identity.exact_payload,
        identity.structure_payload,
        identity.polymorph_payload,
    ):
        assert json.loads(payload)["scaffold"]["composition"] == numeric_composition
    assert record.target.canonical_scaffold.reduced_composition == original_composition

    def expected_digest(payload: object) -> str:
        return sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("ascii")
        ).hexdigest()

    assert identity.composition_key == expected_digest(
        {
            "composition": numeric_composition,
            "schema": "crystalatlas.canonical-structure-identity.v2.composition.v1",
        }
    )
    assert identity.element_set_key == expected_digest(
        {
            "elements": [8, 14, 16],
            "schema": "crystalatlas.canonical-structure-identity.v2.element-set.v1",
        }
    )


def test_mixed_width_species_permutation_and_renaming_keep_anonymous_prototype() -> None:
    def identity(species: tuple[int, int, int]):
        return build_canonical_identity(
            _record(species_order=species), catalogue=MINIMAL_CATALOGUE, policy=_policy()
        )

    original = identity((8, 14, 16))
    permuted = identity((14, 8, 16))
    renamed = identity((9, 15, 17))
    assert original.prototype_key == permuted.prototype_key == renamed.prototype_key
    assert original.composition_key == permuted.composition_key != renamed.composition_key
    assert original.element_set_key == permuted.element_set_key != renamed.element_set_key
    assert len({original.exact_key, permuted.exact_key, renamed.exact_key}) == 3


def test_mixed_width_species_count_tampering_still_fails_audit() -> None:
    record = _record(species_order=(8, 14, 16))
    species = list(record.raw_structure.atomic_numbers)
    species[species.index(14)] = 8
    raw = replace(record.raw_structure, atomic_numbers=tuple(species))
    tampered = replace(
        record,
        raw_structure=raw,
        canonicalized=replace(
            record.canonicalized,
            receipt=replace(record.canonicalized.receipt, source_fingerprint=raw.fingerprint),
        ),
    )
    with pytest.raises(InvariantViolation, match="exact coloured count quotient"):
        build_canonical_identity(tampered, catalogue=MINIMAL_CATALOGUE, policy=_policy())


def test_identity_policy_versions_atomic_number_projection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payloads: list[dict[str, object]] = []
    digest = dataset_identity._digest

    def capture(payload: object) -> str:
        assert isinstance(payload, dict)
        payloads.append(payload)
        return digest(payload)

    monkeypatch.setattr(dataset_identity, "_digest", capture)
    fingerprint = _policy().fingerprint
    payload = payloads[-1]
    assert payload["schema"] == "crystalatlas.canonical-structure-identity.v2.policy.v2"
    assert payload["composition_projection"] == "validated-atomic-number-order.v1"
    previous = dict(payload)
    previous["schema"] = "crystalatlas.canonical-structure-identity.v2.policy.v1"
    del previous["composition_projection"]
    assert fingerprint != digest(previous)
