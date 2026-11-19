from __future__ import annotations

import json
from dataclasses import replace
from hashlib import sha256

import pytest

from crystalatlas.canonical_matching import (
    CanonicalPartitionLevel,
    CanonicalPartitionProvenance,
    canonical_scaffold_partition_equal,
    canonical_structure_partition_equal,
    encode_scaffold_partition,
    encode_structure_partition,
    evaluate_canonical_geometry_decomposition,
    evaluate_canonical_scaffold_retrieval,
    scaffold_partition_matcher_contract,
    structure_partition_matcher_contract,
    validate_canonical_partition_payload,
)
from crystalatlas.catalogue import MINIMAL_CATALOGUE
from crystalatlas.dataset_identity import CanonicalStructureIdentity, IdentityPolicy
from crystalatlas.diagnostics import (
    AcceptableTarget,
    BootstrapProtocol,
    DiagnosticEvaluationContract,
    GeometryDecompositionCase,
    RankedAttempt,
    RankedRetrievalCase,
    geometry_corpus_fingerprints,
    matcher_callable_sha256,
    retrieval_corpus_fingerprints,
)
from crystalatlas.errors import DomainError

_CATALOGUE_OBJECT = MINIMAL_CATALOGUE
_IDENTITY_POLICY = IdentityPolicy(require_all_230=False)
_CATALOGUE = _CATALOGUE_OBJECT.fingerprint
_POLICY = _IDENTITY_POLICY.fingerprint


def _encode_structure(
    identity: CanonicalStructureIdentity,
    provenance: CanonicalPartitionProvenance,
) -> bytes:
    return encode_structure_partition(
        identity,
        provenance,
        catalogue=_CATALOGUE_OBJECT,
        identity_policy=_IDENTITY_POLICY,
    )


def _encode_scaffold(
    identity: CanonicalStructureIdentity,
    provenance: CanonicalPartitionProvenance,
) -> bytes:
    return encode_scaffold_partition(
        identity,
        provenance,
        catalogue=_CATALOGUE_OBJECT,
        identity_policy=_IDENTITY_POLICY,
    )


def _validate(
    payload: bytes,
    *,
    provenance: CanonicalPartitionProvenance,
    level: CanonicalPartitionLevel,
) -> bytes:
    return validate_canonical_partition_payload(
        payload,
        provenance=provenance,
        level=level,
        catalogue=_CATALOGUE_OBJECT,
        identity_policy=_IDENTITY_POLICY,
    )


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")


def _resign_inner(envelope: dict[str, object]) -> bytes:
    canonical_payload = envelope["canonical_payload"]
    envelope["canonical_payload_sha256"] = sha256(
        _canonical_bytes(canonical_payload)
    ).hexdigest()
    return _canonical_bytes(envelope)


def _identity(
    *,
    parameter_bins: tuple[int, ...] = (1_000_000, 2_000_000, 3_000_000),
    metric_bins: tuple[int, int, int, int, int, int] = (
        16_000_000,
        25_000_000,
        36_000_000,
        0,
        0,
        0,
    ),
    species: int = 6,
    catalogue_fingerprint: str = _CATALOGUE,
    policy_fingerprint: str = _POLICY,
) -> CanonicalStructureIdentity:
    decorated_scaffold = {
        "composition": [[species, 1]],
        "nu": 1,
        "orbits": [
            {
                "parameter_bins": list(parameter_bins),
                "species": species,
                "wyckoff": "1a",
            }
        ],
        "setting": {
            "hall_symbol": "P 1",
            "international_number": 1,
            "key": "P1",
        },
    }
    structure_object = {
        "catalogue_fingerprint": catalogue_fingerprint,
        "chirality_quotient": "SO(3)",
        "coordinate_convention": "fractional-row-v1",
        "metric_bins_angstrom2": list(metric_bins),
        "policy_fingerprint": policy_fingerprint,
        "scaffold": decorated_scaffold,
        "schema": "crystalatlas.canonical-structure-identity.v2",
    }
    structure_payload = _canonical_bytes(structure_object).decode("ascii")
    discrete_scaffold = {
        "composition": decorated_scaffold["composition"],
        "nu": 1,
        "orbits": [{"species": species, "wyckoff": "1a"}],
        "setting": decorated_scaffold["setting"],
    }
    scaffold_object = {
        "catalogue_fingerprint": catalogue_fingerprint,
        "chirality_quotient": "SO(3)",
        "coordinate_convention": "fractional-row-v1",
        "scaffold": discrete_scaffold,
        "schema": "crystalatlas.canonical-structure-identity.v2",
    }
    return CanonicalStructureIdentity(
        exact_key="8" * 64,
        structure_key=sha256(structure_payload.encode("ascii")).hexdigest(),
        polymorph_key="9" * 64,
        scaffold_key=sha256(_canonical_bytes(scaffold_object)).hexdigest(),
        prototype_key="a" * 64,
        composition_key="b" * 64,
        element_set_key="c" * 64,
        policy_fingerprint=policy_fingerprint,
        catalogue_fingerprint=catalogue_fingerprint,
        metric_bins=metric_bins,
        shape_bins=(1, 1, 1, 0, 0, 0),
        exact_payload="{}",
        structure_payload=structure_payload,
        polymorph_payload="{}",
    )


def _provenance(
    *,
    catalogue_fingerprint: str = _CATALOGUE,
    policy_fingerprint: str = _POLICY,
    environment_sha256: str = "7" * 64,
) -> CanonicalPartitionProvenance:
    return CanonicalPartitionProvenance(
        dataset_manifest_sha256="0" * 64,
        target_builder_fingerprint="1" * 64,
        crystallography_receipt_fingerprint="2" * 64,
        catalogue_fingerprint=catalogue_fingerprint,
        identity_policy_fingerprint=policy_fingerprint,
        crystalatlas_artifact_sha256="5" * 64,
        crystallography_backend_artifact_sha256="6" * 64,
        environment_sha256=environment_sha256,
    )


def test_structure_envelope_binds_full_payload_provenance_and_limitations() -> None:
    identity = _identity()
    provenance = _provenance()
    encoded = _encode_structure(identity, provenance)

    assert isinstance(encoded, bytes)
    envelope = json.loads(encoded)
    assert envelope["canonical_payload"] == json.loads(identity.structure_payload)
    assert envelope["canonical_payload_sha256"] == identity.structure_key
    assert envelope["identity_level"] == "structure"
    assert envelope["metric_uncertainty_certified"] is False
    assert envelope["representation_invariant_physical_equivalence_claim"] is False
    assert envelope["provenance"]["fingerprint"] == provenance.fingerprint
    assert (
        _validate(
            encoded,
            provenance=provenance,
            level=CanonicalPartitionLevel.STRUCTURE,
        )
        == encoded
    )

    contract = structure_partition_matcher_contract(provenance)
    parameters = json.loads(contract.parameters_json)
    assert parameters["metric_uncertainty_certified"] is False
    assert parameters["representation_invariant_physical_equivalence_claim"] is False
    assert contract.implementation_sha256 == matcher_callable_sha256(
        canonical_structure_partition_equal
    )
    assert canonical_structure_partition_equal(encoded, encoded)


def test_scaffold_partition_is_separate_and_safely_quotients_continuous_geometry() -> None:
    provenance = _provenance()
    first = _identity()
    changed_geometry = _identity(
        parameter_bins=(1_000_001, 2_000_000, 3_000_000),
        metric_bins=(16_000_001, 25_000_000, 36_000_000, 0, 0, 0),
    )
    first_structure = _encode_structure(first, provenance)
    changed_structure = _encode_structure(changed_geometry, provenance)
    first_scaffold = _encode_scaffold(first, provenance)
    changed_scaffold = _encode_scaffold(changed_geometry, provenance)

    assert not canonical_structure_partition_equal(first_structure, changed_structure)
    assert canonical_scaffold_partition_equal(first_scaffold, changed_scaffold)
    assert json.loads(first_scaffold)["canonical_payload_sha256"] == first.scaffold_key

    structure_contract = structure_partition_matcher_contract(provenance)
    scaffold_contract = scaffold_partition_matcher_contract(provenance)
    assert structure_contract.fingerprint != scaffold_contract.fingerprint
    assert scaffold_contract.implementation_sha256 == matcher_callable_sha256(
        canonical_scaffold_partition_equal
    )


def test_identity_policy_catalogue_and_payload_key_mismatches_fail_closed() -> None:
    identity = _identity()
    provenance = _provenance()

    with pytest.raises(DomainError, match="policy differs"):
        _encode_structure(identity, _provenance(policy_fingerprint="d" * 64))
    with pytest.raises(DomainError, match="catalogue differs"):
        _encode_structure(identity, _provenance(catalogue_fingerprint="e" * 64))
    with pytest.raises(DomainError, match="structure key disagrees"):
        _encode_structure(
            replace(identity, structure_key="f" * 64),
            provenance,
        )
    with pytest.raises(DomainError, match="scaffold key disagrees"):
        _encode_scaffold(
            replace(identity, scaffold_key="f" * 64),
            provenance,
        )

    noncanonical = identity.structure_payload.replace("{", "{ ", 1)
    with pytest.raises(DomainError, match="not canonically serialized"):
        _encode_structure(
            replace(
                identity,
                structure_payload=noncanonical,
                structure_key=sha256(noncanonical.encode("ascii")).hexdigest(),
            ),
            provenance,
        )


def test_envelope_validation_rejects_environment_claim_and_level_tampering() -> None:
    provenance = _provenance()
    encoded = _encode_structure(_identity(), provenance)

    with pytest.raises(DomainError, match="provenance differs"):
        _validate(
            encoded,
            provenance=_provenance(environment_sha256="d" * 64),
            level=CanonicalPartitionLevel.STRUCTURE,
        )
    with pytest.raises(DomainError, match="another identity level"):
        _validate(
            encoded,
            provenance=provenance,
            level=CanonicalPartitionLevel.SCAFFOLD,
        )

    envelope = json.loads(encoded)
    envelope["representation_invariant_physical_equivalence_claim"] = True
    with pytest.raises(DomainError, match="cannot claim physical structure equivalence"):
        _validate(
            _canonical_bytes(envelope),
            provenance=provenance,
            level=CanonicalPartitionLevel.STRUCTURE,
        )

    envelope = json.loads(encoded)
    envelope["canonical_payload"]["metric_bins_angstrom2"][0] += 1
    with pytest.raises(DomainError, match="disagrees with its fingerprint"):
        _validate(
            _canonical_bytes(envelope),
            provenance=provenance,
            level=CanonicalPartitionLevel.STRUCTURE,
        )


def test_scaffold_payload_changes_when_discrete_chemistry_changes() -> None:
    provenance = _provenance()
    carbon = _encode_scaffold(_identity(species=6), provenance)
    oxygen = _encode_scaffold(_identity(species=8), provenance)
    assert not canonical_scaffold_partition_equal(carbon, oxygen)


def _evaluation_contract(
    *,
    provenance: CanonicalPartitionProvenance,
    target_corpus_sha256: str,
    generated_attempt_corpus_sha256: str,
) -> DiagnosticEvaluationContract:
    return DiagnosticEvaluationContract(
        dataset_manifest_sha256=provenance.dataset_manifest_sha256,
        split_manifest_sha256="d" * 64,
        checkpoint_sha256="e" * 64,
        target_corpus_sha256=target_corpus_sha256,
        generated_attempt_corpus_sha256=generated_attempt_corpus_sha256,
        generation_protocol_sha256="f" * 64,
        generation_seed=17,
        attempt_budget=1,
        score_policy="descending_score_then_attempt_id",
        validity_policy="canonicalization_failure_is_invalid",
    )


def test_recomputed_hash_cannot_hide_semantically_invalid_inner_payload() -> None:
    provenance = _provenance()
    encoded = _encode_structure(_identity(), provenance)
    envelope = json.loads(encoded)
    envelope["canonical_payload"]["chirality_quotient"] = "O(3)"
    envelope["canonical_payload_sha256"] = sha256(
        _canonical_bytes(envelope["canonical_payload"])
    ).hexdigest()
    with pytest.raises(DomainError, match="proper-rotation quotient"):
        _validate(
            _canonical_bytes(envelope),
            provenance=provenance,
            level=CanonicalPartitionLevel.STRUCTURE,
        )

    scaffold = json.loads(_encode_scaffold(_identity(), provenance))
    del scaffold["canonical_payload"]["scaffold"]
    scaffold["canonical_payload_sha256"] = sha256(
        _canonical_bytes(scaffold["canonical_payload"])
    ).hexdigest()
    with pytest.raises(DomainError, match="fields disagree"):
        _validate(
            _canonical_bytes(scaffold),
            provenance=provenance,
            level=CanonicalPartitionLevel.SCAFFOLD,
        )


@pytest.mark.parametrize(
    ("mutation", "message"),
    (
        ("setting", "absent from the certified catalogue"),
        ("row", "Wyckoff row is absent"),
        ("arity", "parameter-bin arity"),
        ("range", "malformed parameter bins"),
        ("metric", "positive-definite metric"),
    ),
)
def test_live_catalogue_and_policy_reject_recomputed_hash_forgeries(
    mutation: str,
    message: str,
) -> None:
    provenance = _provenance()
    envelope = json.loads(_encode_structure(_identity(), provenance))
    payload = envelope["canonical_payload"]
    scaffold = payload["scaffold"]
    orbit = scaffold["orbits"][0]
    if mutation == "setting":
        scaffold["setting"] = {
            "hall_symbol": "NOT A HALL SYMBOL",
            "international_number": 1,
            "key": "NOT-A-SETTING",
        }
    elif mutation == "row":
        orbit["wyckoff"] = "1zz"
    elif mutation == "arity":
        orbit["parameter_bins"] = []
    elif mutation == "range":
        orbit["parameter_bins"][0] = _IDENTITY_POLICY.fractional_bin_count
    else:
        payload["metric_bins_angstrom2"] = [1, 1, -1, 0, 0, 0]
    with pytest.raises(DomainError, match=message):
        _validate(
            _resign_inner(envelope),
            provenance=provenance,
            level=CanonicalPartitionLevel.STRUCTURE,
        )


def test_live_catalogue_rejects_noncanonical_orbit_order_after_rehash() -> None:
    provenance = _provenance()
    envelope = json.loads(_encode_structure(_identity(), provenance))
    scaffold = envelope["canonical_payload"]["scaffold"]
    scaffold["nu"] = 2
    first = scaffold["orbits"][0]
    second = {
        "parameter_bins": [4_000_000, 5_000_000, 6_000_000],
        "species": 6,
        "wyckoff": "1a",
    }
    scaffold["orbits"] = [second, first]
    with pytest.raises(DomainError, match="not in canonical row order"):
        _validate(
            _resign_inner(envelope),
            provenance=provenance,
            level=CanonicalPartitionLevel.STRUCTURE,
        )


def test_canonical_scaffold_evaluator_prevalidates_bytes_and_dataset_binding() -> None:
    provenance = _provenance()
    payload = _encode_scaffold(_identity(), provenance)
    cases = (
        RankedRetrievalCase(
            "query-1",
            (AcceptableTarget("target-1", payload),),
            (RankedAttempt("attempt-1", 1.0, True, payload),),
        ),
    )
    corpus = retrieval_corpus_fingerprints(cases)
    contract = _evaluation_contract(
        provenance=provenance,
        target_corpus_sha256=corpus.target_corpus_sha256,
        generated_attempt_corpus_sha256=corpus.generated_attempt_corpus_sha256,
    )
    result = evaluate_canonical_scaffold_retrieval(
        cases,
        ks=(1,),
        provenance=provenance,
        catalogue=_CATALOGUE_OBJECT,
        identity_policy=_IDENTITY_POLICY,
        evaluation_contract=contract,
        bootstrap=BootstrapProtocol(samples=100, seed=2),
    )
    assert result.at_k[0].recall.value == 1.0

    garbage_cases = (
        RankedRetrievalCase(
            "query-1",
            (AcceptableTarget("target-1", b"not-an-envelope"),),
            (RankedAttempt("attempt-1", 1.0, True, b"not-an-envelope"),),
        ),
    )
    garbage_corpus = retrieval_corpus_fingerprints(garbage_cases)
    garbage_contract = _evaluation_contract(
        provenance=provenance,
        target_corpus_sha256=garbage_corpus.target_corpus_sha256,
        generated_attempt_corpus_sha256=(
            garbage_corpus.generated_attempt_corpus_sha256
        ),
    )
    with pytest.raises(DomainError, match="canonical partition envelope"):
        evaluate_canonical_scaffold_retrieval(
            garbage_cases,
            ks=(1,),
            provenance=provenance,
            catalogue=_CATALOGUE_OBJECT,
            identity_policy=_IDENTITY_POLICY,
            evaluation_contract=garbage_contract,
            bootstrap=BootstrapProtocol(samples=100, seed=2),
        )

    mismatched = replace(contract, dataset_manifest_sha256="1" * 64)
    with pytest.raises(DomainError, match="dataset manifest differs"):
        evaluate_canonical_scaffold_retrieval(
            cases,
            ks=(1,),
            provenance=provenance,
            catalogue=_CATALOGUE_OBJECT,
            identity_policy=_IDENTITY_POLICY,
            evaluation_contract=mismatched,
            bootstrap=BootstrapProtocol(samples=100, seed=2),
        )


def test_canonical_geometry_evaluator_uses_validated_structure_partitions() -> None:
    provenance = _provenance()
    payload = _encode_structure(_identity(), provenance)
    cases = (
        GeometryDecompositionCase(
            "query-1",
            (AcceptableTarget("target-1", payload),),
            (RankedAttempt("oracle-1", 1.0, True, payload),),
            (RankedAttempt("predicted-1", 1.0, True, payload),),
        ),
    )
    corpus = geometry_corpus_fingerprints(cases)
    contract = _evaluation_contract(
        provenance=provenance,
        target_corpus_sha256=corpus.target_corpus_sha256,
        generated_attempt_corpus_sha256=corpus.generated_attempt_corpus_sha256,
    )
    result = evaluate_canonical_geometry_decomposition(
        cases,
        ks=(1,),
        provenance=provenance,
        catalogue=_CATALOGUE_OBJECT,
        identity_policy=_IDENTITY_POLICY,
        evaluation_contract=contract,
        bootstrap=BootstrapProtocol(samples=100, seed=3),
    )
    assert result.at_k[0].oracle_scaffold_match.value == 1.0
    assert result.at_k[0].predicted_scaffold_match.value == 1.0
