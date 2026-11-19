"""Fail-closed compiler from licensed source rows to trainable crystal records.

The compiler has four independent proof boundaries:

1. the legal/source manifest is authorized for the requested use and every
   local artifact matches its recorded byte count and SHA-256 digest;
2. the ordered source-row index exactly matches the pinned snapshot receipt;
3. each row passes strict parsing, FullPrim/pinned-Hall standardization, exact
   Wyckoff inversion, continuous target lifting, and decoder round-trip; and
4. canonical physical identities are partitioned into leakage-free splits.

Only preprocessing failures are rejected here.  No energy, stability,
property, model score, generated sample, or relaxation outcome is inspected.
That distinction prevents a dataset builder from silently becoming a
model-dependent validity filter.
"""

from __future__ import annotations

import json
import math
import os
import tempfile
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, fields, is_dataclass
from enum import Enum
from fractions import Fraction
from hashlib import sha256
from pathlib import Path
from typing import Any, Protocol, TypeAlias, cast

import numpy as np

from .catalogue import CertifiedCatalogue
from .certification import CrystallographyCertification
from .config import DomainConfig
from .data import CrystalRecord
from .dataset_identity import (
    AmbiguousIdentityError,
    CanonicalStructureIdentity,
    DeduplicationLevel,
    IdentityPolicy,
    build_canonical_identity,
)
from .dataset_provenance import (
    DatasetPolicyDecision,
    DatasetSourceManifest,
    DatasetUseRequest,
    SourceRecordScopeEvidence,
    evaluate_dataset_use,
    ordered_record_ids_sha256,
    ordered_record_payloads_sha256,
    ordered_record_scopes_sha256,
)
from .dataset_splits import (
    DatasetSplit,
    FrozenSplitKeys,
    SplitFractions,
    SplitMode,
    SplitRecord,
    audit_dataset_split,
    build_dataset_split,
)
from .errors import (
    AmbiguousPreprocessingError,
    DomainError,
    InvariantViolation,
    NumericalError,
    RecordPreprocessingError,
)
from .historical_mp20 import require_historical_mp20_archive
from .ingestion import (
    CANONICAL_ORIGIN_VERIFIER_ID,
    CanonicalTrainingRecord,
    CatalogueTargetAssigner,
    NumericalStandardizationBackend,
    RawPeriodicStructure,
    parse_cif_structure,
    structure_to_catalogue_training_target,
)
from .training_targets import (
    CanonicalCatalogueTrainingTarget,
    ExactFixtureTrainingTarget,
)

CanonicalTarget: TypeAlias = CanonicalCatalogueTrainingTarget | ExactFixtureTrainingTarget
CanonicalRecord: TypeAlias = CanonicalTrainingRecord[CanonicalTarget]

_MANIFEST_SCHEMA = "crystalatlas.compiled-dataset-manifest.v3"
DEFAULT_SPLIT_FRACTIONS = SplitFractions.from_values("4/5", "1/10", "1/10")


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    )


def _fingerprint(value: object) -> str:
    return sha256(_canonical_json(value).encode("ascii")).hexdigest()


def _object_without_duplicate_keys(
    pairs: list[tuple[str, object]],
) -> dict[str, object]:
    """Build one JSON object while rejecting ambiguous duplicate names."""

    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise DomainError(f"dataset build manifest repeats JSON key {key!r}")
        result[key] = value
    return result


def _reject_nonfinite_json_number(value: str) -> object:
    raise DomainError(f"dataset build manifest contains nonfinite JSON number {value!r}")


def _require_text(value: object, *, name: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise DomainError(f"{name} must be nonempty stripped text")
    if any(ord(character) < 32 for character in value):
        raise DomainError(f"{name} cannot contain control characters")
    return value


def _require_digest(value: object, *, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise DomainError(f"{name} must be a lowercase SHA-256 digest")
    return value


def _scientific_payload(value: object) -> object:
    """Canonical, lossless JSON projection for immutable scientific objects."""

    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise InvariantViolation("scientific manifests cannot contain nonfinite floats")
        return {"float_hex": float(0.0 if value == 0.0 else value).hex()}
    if isinstance(value, Fraction):
        return {"fraction": [value.numerator, value.denominator]}
    if isinstance(value, Enum):
        return {
            "enum": f"{type(value).__module__}.{type(value).__qualname__}",
            "value": value.value,
        }
    if isinstance(value, np.generic):
        return _scientific_payload(value.item())
    if isinstance(value, np.ndarray):
        if value.dtype.kind in "fc" and not np.all(np.isfinite(value)):
            raise InvariantViolation("scientific manifests cannot contain nonfinite arrays")
        return {
            "array_dtype": value.dtype.str,
            "array_shape": list(value.shape),
            "array_values": [_scientific_payload(item) for item in value.reshape(-1)],
        }
    if is_dataclass(value) and not isinstance(value, type):
        return {
            "dataclass": f"{type(value).__module__}.{type(value).__qualname__}",
            "fields": {
                item.name: _scientific_payload(getattr(value, item.name))
                for item in fields(value)
            },
        }
    if isinstance(value, Mapping):
        entries = [
            (_scientific_payload(key), _scientific_payload(item))
            for key, item in value.items()
        ]
        return {
            "mapping": sorted(entries, key=lambda item: _canonical_json(item[0]))
        }
    if isinstance(value, (tuple, list)):
        return [_scientific_payload(item) for item in value]
    raise InvariantViolation(
        f"no canonical scientific serialization exists for {type(value).__qualname__}"
    )


def canonical_training_target_fingerprint(target: CanonicalTarget) -> str:
    """Hash the complete rebuilt target with the persisted v1 exact encoding.

    Both compilation and the consumer's scientific audit use this function.
    Its float64 hexadecimal projection deliberately has no numerical tolerance:
    an internally consistent replacement is still a different training target.
    """

    if not isinstance(target, (ExactFixtureTrainingTarget, CanonicalCatalogueTrainingTarget)):
        raise TypeError("target must be a certified canonical training target")
    layout = target.layout
    payload = {
        "schema": "crystalatlas.canonical-training-target.v1",
        "composition": _scientific_payload(target.composition),
        "canonical_scaffold": _scientific_payload(target.canonical_scaffold),
        "cover_decisions": _scientific_payload(target.cover_decisions),
        "metric": _scientific_payload(target.metric),
        "lattice_latent": _scientific_payload(target.lattice_latent),
        "intrinsic_coordinates": _scientific_payload(target.intrinsic_coordinates),
        "period_basis": _scientific_payload(target.period_basis),
        "layout": {
            "expanded_atom_count": layout.expanded_atom_count,
            "intrinsic_dimension": layout.intrinsic_dimension,
            "lattice_dimension": layout.lattice_dimension,
            "minimum_lattice_singular_value": _scientific_payload(
                layout.minimum_lattice_singular_value
            ),
            "parameter_maps": _scientific_payload(layout.parameter_maps),
            "reference_atomic_volume": _scientific_payload(
                layout.reference_atomic_volume
            ),
            "setting_key": layout.setting.key,
        },
    }
    return _fingerprint(payload)


def _canonical_target_fingerprint(record: CanonicalRecord) -> str:
    """Compatibility wrapper for the compiler's record-level binding."""

    return canonical_training_target_fingerprint(record.target)


def _policy_payload(policy: object) -> dict[str, object]:
    """Standardizer policy identity; unset optional fields keep legacy payloads."""

    payload = asdict(policy)  # type: ignore[call-overload]
    for name in ("relative_metric_quantum", "idealization_tolerance_angstrom"):
        if payload.get(name, 0.0) is None:
            del payload[name]
    return cast(dict[str, object], payload)


_OPTIONAL_RECEIPT_METADATA = (
    "idealization_lattice_displacement_angstrom",
    "idealization_site_displacement_angstrom",
)


def _canonicalization_receipt_fingerprint(record: CanonicalRecord) -> str:
    payload = cast(dict[str, Any], _scientific_payload(record.canonicalized.receipt))
    for name in _OPTIONAL_RECEIPT_METADATA:
        # Absent metadata leaves legacy receipt fingerprints bit-identical.
        if payload["fields"].get(name, 0) is None:
            del payload["fields"][name]
    return _fingerprint(
        {
            "schema": "crystalatlas.canonicalization-receipt-binding.v1",
            "receipt": payload,
        }
    )


@dataclass(frozen=True, slots=True, order=True)
class SourceRecordReference:
    """Globally unique identity of one row in one immutable source snapshot."""

    source_id: str
    snapshot_id: str
    source_record_id: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "source_id", _require_text(self.source_id, name="source_id"))
        object.__setattr__(
            self, "snapshot_id", _require_text(self.snapshot_id, name="snapshot_id")
        )
        object.__setattr__(
            self,
            "source_record_id",
            _require_text(self.source_record_id, name="source_record_id"),
        )

    @property
    def fingerprint(self) -> str:
        return _fingerprint(self.to_dict())

    def to_dict(self) -> dict[str, str]:
        return {
            "source_id": self.source_id,
            "snapshot_id": self.snapshot_id,
            "source_record_id": self.source_record_id,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> SourceRecordReference:
        if set(payload) != {"source_id", "snapshot_id", "source_record_id"}:
            raise DomainError("source-record reference has missing or unexpected fields")
        return cls(
            source_id=payload["source_id"],  # type: ignore[arg-type]
            snapshot_id=payload["snapshot_id"],  # type: ignore[arg-type]
            source_record_id=payload["source_record_id"],  # type: ignore[arg-type]
        )


@dataclass(frozen=True, slots=True)
class DatasetCandidate:
    """One source row before parsing or crystallographic interpretation.

    ``payload`` is either strict CIF text or a pre-parsed ordered, fully occupied
    structure.  The latter is useful for database APIs that already return
    lattice/coordinate arrays; it is still revalidated by ``RawPeriodicStructure``.
    """

    reference: SourceRecordReference
    payload: str | RawPeriodicStructure
    scope_evidence: SourceRecordScopeEvidence

    def __post_init__(self) -> None:
        if not isinstance(self.reference, SourceRecordReference):
            raise TypeError("reference must be a SourceRecordReference")
        if isinstance(self.payload, str):
            if not self.payload.strip():
                raise DomainError("CIF payload cannot be empty")
        elif not isinstance(self.payload, RawPeriodicStructure):
            raise TypeError("payload must be CIF text or RawPeriodicStructure")
        if not isinstance(self.scope_evidence, SourceRecordScopeEvidence):
            raise TypeError("scope_evidence must be SourceRecordScopeEvidence")

    @property
    def payload_sha256(self) -> str:
        if isinstance(self.payload, str):
            return sha256(self.payload.encode("utf-8")).hexdigest()
        return self.payload.fingerprint


class DatasetTargetBuilder(Protocol):
    """Audited raw-structure to canonical-target callable."""

    @property
    def builder_fingerprint(self) -> str: ...

    @property
    def certified_preprocessing_only(self) -> bool: ...

    def __call__(self, structure: RawPeriodicStructure) -> CanonicalRecord: ...


class _AuditedNumericalStandardizationBackend(
    NumericalStandardizationBackend,
    Protocol,
):
    """Numerical backend plus immutable fields bound by a build receipt."""

    @property
    def hall_map_fingerprint(self) -> str: ...

    @property
    def policy(self) -> Any: ...

    @property
    def algorithm_id(self) -> str: ...


@dataclass(frozen=True, slots=True)
class CatalogueDatasetTargetBuilder:
    """Production all-230 target builder with frozen numerical dependencies."""

    standardizer: _AuditedNumericalStandardizationBackend
    target_assigner: CatalogueTargetAssigner
    domain: DomainConfig

    def __post_init__(self) -> None:
        if not isinstance(self.target_assigner, CatalogueTargetAssigner):
            raise TypeError("target_assigner must be a CatalogueTargetAssigner")
        if not isinstance(self.domain, DomainConfig):
            raise TypeError("domain must be a DomainConfig")
        for name in ("policy", "hall_map_fingerprint", "algorithm_id"):
            if not hasattr(self.standardizer, name):
                raise TypeError(f"standardizer lacks certified field {name!r}")
        _require_text(self.standardizer.algorithm_id, name="standardizer algorithm identity")

    @property
    def certified_preprocessing_only(self) -> bool:
        return True

    @property
    def builder_fingerprint(self) -> str:
        return _fingerprint(
            {
                "schema": "crystalatlas.catalogue-dataset-target-builder.v3",
                "canonical_origin_verifier": CANONICAL_ORIGIN_VERIFIER_ID,
                "standardizer": {
                    "class": (
                        f"{type(self.standardizer).__module__}."
                        f"{type(self.standardizer).__qualname__}"
                    ),
                    "hall_map_fingerprint": self.standardizer.hall_map_fingerprint,
                    "algorithm_id": self.standardizer.algorithm_id,
                    "policy": _policy_payload(self.standardizer.policy),
                },
                "assigner": {
                    "class": (
                        f"{type(self.target_assigner).__module__}."
                        f"{type(self.target_assigner).__qualname__}"
                    ),
                    "catalogue_fingerprint": self.target_assigner.catalogue.fingerprint,
                    "hall_map_fingerprint": self.target_assigner.hall_map_fingerprint,
                    "rational_decimals": self.target_assigner.rational_decimals,
                    "roundtrip_tolerance": self.target_assigner.roundtrip_tolerance.hex(),
                    # Present only when set, so legacy builders keep their identity.
                    **(
                        {"merge_quantization_equivalent_orbits": True}
                        if self.target_assigner.merge_quantization_equivalent_orbits
                        else {}
                    ),
                },
                "domain": self.domain.identity_payload(),
            }
        )

    def __call__(
        self, structure: RawPeriodicStructure
    ) -> CanonicalTrainingRecord[CanonicalCatalogueTrainingTarget]:
        try:
            return structure_to_catalogue_training_target(
                structure,
                standardizer=self.standardizer,
                target_assigner=self.target_assigner,
                domain=self.domain,
            )
        except RecordPreprocessingError:
            raise
        except (DomainError, NumericalError) as exc:
            raise AmbiguousPreprocessingError(str(exc)) from exc


class RejectionStage(str, Enum):  # noqa: UP042 - Python-3.10 ML smoke compatibility
    """The only stages at which an input database row may be rejected."""

    PARSE = "strict-parse"
    CRYSTALLOGRAPHY = "certified-crystallographic-preprocessing"
    IDENTITY = "canonical-identity"


@dataclass(frozen=True, slots=True)
class DatasetRejection:
    """Immutable evidence for one rejected *input* row."""

    reference: SourceRecordReference
    payload_sha256: str
    stage: RejectionStage
    exception_type: str
    reason: str

    def __post_init__(self) -> None:
        if not isinstance(self.reference, SourceRecordReference):
            raise TypeError("reference must be SourceRecordReference")
        object.__setattr__(
            self,
            "payload_sha256",
            _require_digest(self.payload_sha256, name="rejection payload hash"),
        )
        if not isinstance(self.stage, RejectionStage):
            raise TypeError("stage must be RejectionStage")
        object.__setattr__(
            self, "exception_type", _require_text(self.exception_type, name="exception_type")
        )
        object.__setattr__(self, "reason", _require_text(self.reason, name="reason"))

    def to_dict(self) -> dict[str, object]:
        return {
            "reference": self.reference.to_dict(),
            "payload_sha256": self.payload_sha256,
            "stage": self.stage.value,
            "exception_type": self.exception_type,
            "reason": self.reason,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> DatasetRejection:
        expected = {
            "reference",
            "payload_sha256",
            "stage",
            "exception_type",
            "reason",
        }
        if set(payload) != expected or not isinstance(payload["reference"], Mapping):
            raise DomainError("dataset rejection has missing or malformed fields")
        try:
            stage = RejectionStage(payload["stage"])
        except (TypeError, ValueError) as exc:
            raise DomainError("dataset rejection has an unknown stage") from exc
        return cls(
            reference=SourceRecordReference.from_dict(payload["reference"]),
            payload_sha256=payload["payload_sha256"],  # type: ignore[arg-type]
            stage=stage,
            exception_type=payload["exception_type"],  # type: ignore[arg-type]
            reason=payload["reason"],  # type: ignore[arg-type]
        )


@dataclass(frozen=True, slots=True)
class DatasetRecordSummary:
    """Immutable manifest projection of one compiled physical record."""

    record_id: str
    aliases: tuple[SourceRecordReference, ...]
    alias_scope_evidence: tuple[SourceRecordScopeEvidence, ...]
    exact_key: str
    composition_key: str
    element_set_key: str
    polymorph_key: str
    prototype_key: str
    scaffold_key: str
    structure_key: str
    representative_payload_sha256: str
    target_fingerprint: str
    canonicalization_receipt_fingerprint: str

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "record_id", _require_digest(self.record_id, name="record_id")
        )
        aliases = tuple(self.aliases)
        evidence = tuple(self.alias_scope_evidence)
        if len(aliases) != len(evidence):
            raise DomainError("dataset aliases and scope evidence must align one-to-one")
        if any(not isinstance(item, SourceRecordScopeEvidence) for item in evidence):
            raise TypeError("alias_scope_evidence must contain SourceRecordScopeEvidence")
        pairs = tuple(sorted(zip(aliases, evidence, strict=True), key=lambda item: item[0]))
        aliases = tuple(reference for reference, _ in pairs)
        evidence = tuple(item for _, item in pairs)
        if not aliases or len(set(aliases)) != len(aliases):
            raise DomainError("dataset record summary aliases must be nonempty and unique")
        object.__setattr__(self, "aliases", aliases)
        object.__setattr__(self, "alias_scope_evidence", evidence)
        for field_name in (
            "composition_key",
            "element_set_key",
            "polymorph_key",
            "prototype_key",
            "scaffold_key",
            "structure_key",
            "exact_key",
            "representative_payload_sha256",
            "target_fingerprint",
            "canonicalization_receipt_fingerprint",
        ):
            value = getattr(self, field_name)
            object.__setattr__(
                self,
                field_name,
                _require_digest(value, name=field_name),
            )

    def to_dict(self) -> dict[str, object]:
        return {
            "record_id": self.record_id,
            "aliases": [alias.to_dict() for alias in self.aliases],
            "alias_scope_evidence": [item.to_dict() for item in self.alias_scope_evidence],
            "representative_payload_sha256": self.representative_payload_sha256,
            "target_fingerprint": self.target_fingerprint,
            "canonicalization_receipt_fingerprint": (
                self.canonicalization_receipt_fingerprint
            ),
            "identity": {
                "exact": self.exact_key,
                "composition": self.composition_key,
                "element_set": self.element_set_key,
                "polymorph": self.polymorph_key,
                "prototype": self.prototype_key,
                "scaffold": self.scaffold_key,
                "structure": self.structure_key,
            },
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> DatasetRecordSummary:
        if set(payload) != {
            "record_id",
            "aliases",
            "alias_scope_evidence",
            "identity",
            "representative_payload_sha256",
            "target_fingerprint",
            "canonicalization_receipt_fingerprint",
        }:
            raise DomainError("dataset record summary has missing or unexpected fields")
        aliases = payload["aliases"]
        alias_scope_evidence = payload["alias_scope_evidence"]
        identity = payload["identity"]
        if (
            not isinstance(aliases, list)
            or not isinstance(alias_scope_evidence, list)
            or not isinstance(identity, Mapping)
        ):
            raise DomainError("dataset record summary aliases/identity are malformed")
        expected_identity = {
            "composition",
            "element_set",
            "polymorph",
            "prototype",
            "scaffold",
            "structure",
            "exact",
        }
        if set(identity) != expected_identity:
            raise DomainError("dataset record identity has missing or unexpected fields")
        if any(not isinstance(alias, Mapping) for alias in aliases):
            raise DomainError("dataset record alias is malformed")
        return cls(
            record_id=payload["record_id"],  # type: ignore[arg-type]
            aliases=tuple(SourceRecordReference.from_dict(alias) for alias in aliases),
            alias_scope_evidence=tuple(
                SourceRecordScopeEvidence.from_dict(item)
                for item in alias_scope_evidence
            ),
            exact_key=identity["exact"],
            composition_key=identity["composition"],
            element_set_key=identity["element_set"],
            polymorph_key=identity["polymorph"],
            prototype_key=identity["prototype"],
            scaffold_key=identity["scaffold"],
            structure_key=identity["structure"],
            representative_payload_sha256=payload[
                "representative_payload_sha256"
            ],  # type: ignore[arg-type]
            target_fingerprint=payload["target_fingerprint"],  # type: ignore[arg-type]
            canonicalization_receipt_fingerprint=payload[
                "canonicalization_receipt_fingerprint"
            ],  # type: ignore[arg-type]
        )


@dataclass(frozen=True, slots=True)
class CompiledDatasetRecord:
    """One deduplicated physical polymorph and all contributing source aliases."""

    record_id: str
    canonical_record: CanonicalRecord
    identity: CanonicalStructureIdentity
    aliases: tuple[SourceRecordReference, ...]
    alias_scope_evidence: tuple[SourceRecordScopeEvidence, ...]
    representative_payload_sha256: str

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "record_id", _require_digest(self.record_id, name="record_id")
        )
        if not isinstance(self.canonical_record, CanonicalTrainingRecord):
            raise TypeError("canonical_record must be CanonicalTrainingRecord")
        if not isinstance(self.identity, CanonicalStructureIdentity):
            raise TypeError("identity must be CanonicalStructureIdentity")
        object.__setattr__(
            self,
            "representative_payload_sha256",
            _require_digest(
                self.representative_payload_sha256,
                name="representative payload fingerprint",
            ),
        )
        aliases = tuple(self.aliases)
        evidence = tuple(self.alias_scope_evidence)
        if len(aliases) != len(evidence):
            raise DomainError("compiled aliases and scope evidence must align one-to-one")
        if any(not isinstance(item, SourceRecordScopeEvidence) for item in evidence):
            raise TypeError("alias_scope_evidence must contain SourceRecordScopeEvidence")
        pairs = tuple(sorted(zip(aliases, evidence, strict=True), key=lambda item: item[0]))
        aliases = tuple(reference for reference, _ in pairs)
        evidence = tuple(item for _, item in pairs)
        if not aliases or len(set(aliases)) != len(aliases):
            raise DomainError("compiled aliases must be nonempty and unique")
        object.__setattr__(self, "aliases", aliases)
        object.__setattr__(self, "alias_scope_evidence", evidence)

    def split_record(self) -> SplitRecord:
        support = tuple(self.canonical_record.target.composition.support)
        return SplitRecord(
            self.record_id,
            FrozenSplitKeys(
                composition_key=self.identity.composition_key,
                polymorph_key=self.identity.polymorph_key,
                prototype_key=self.identity.prototype_key,
                scaffold_key=self.identity.scaffold_key,
                element_set=support,
            ),
        )

    def training_metadata(self, payload_index: int) -> CrystalRecord:
        if isinstance(payload_index, bool) or not isinstance(payload_index, int):
            raise TypeError("payload_index must be an integer")
        return CrystalRecord(
            record_id=self.record_id,
            species=tuple(self.canonical_record.target.composition.support),
            composition_key=self.identity.composition_key,
            structure_key=self.identity.polymorph_key,
            source=",".join(sorted({alias.source_id for alias in self.aliases})),
            payload_index=payload_index,
            symmetry_ambiguous=(
                not self.canonical_record.canonicalized.receipt.symmetry_stability_verified
            ),
        )

    def manifest_summary(self) -> DatasetRecordSummary:
        return DatasetRecordSummary(
            record_id=self.record_id,
            aliases=self.aliases,
            alias_scope_evidence=self.alias_scope_evidence,
            exact_key=self.identity.exact_key,
            composition_key=self.identity.composition_key,
            element_set_key=self.identity.element_set_key,
            polymorph_key=self.identity.polymorph_key,
            prototype_key=self.identity.prototype_key,
            scaffold_key=self.identity.scaffold_key,
            structure_key=self.identity.structure_key,
            representative_payload_sha256=self.representative_payload_sha256,
            target_fingerprint=_canonical_target_fingerprint(self.canonical_record),
            canonicalization_receipt_fingerprint=(
                _canonicalization_receipt_fingerprint(self.canonical_record)
            ),
        )


@dataclass(frozen=True, slots=True)
class DatasetBuildManifest:
    """Self-fingerprinted summary binding sources, proofs, records, and splits."""

    source_manifest_fingerprint: str
    use_request: DatasetUseRequest
    policy_decision: DatasetPolicyDecision
    target_builder_fingerprint: str
    catalogue_fingerprint: str
    crystallography_receipt_fingerprint: str
    identity_policy_fingerprint: str
    deduplication_level: DeduplicationLevel
    input_count: int
    accepted_pre_dedup_count: int
    output_count: int
    rejection_count: int
    records: tuple[DatasetRecordSummary, ...]
    rejections: tuple[DatasetRejection, ...]
    split_manifest_fingerprints: tuple[tuple[str, str], ...]
    manifest_fingerprint: str | None = None

    def __post_init__(self) -> None:
        for field_name in (
            "source_manifest_fingerprint",
            "target_builder_fingerprint",
            "catalogue_fingerprint",
            "crystallography_receipt_fingerprint",
            "identity_policy_fingerprint",
        ):
            object.__setattr__(
                self,
                field_name,
                _require_digest(getattr(self, field_name), name=field_name),
            )
        if not isinstance(self.use_request, DatasetUseRequest):
            raise TypeError("dataset build manifest use_request must be DatasetUseRequest")
        if not isinstance(self.policy_decision, DatasetPolicyDecision):
            raise TypeError(
                "dataset build manifest policy_decision must be DatasetPolicyDecision"
            )
        if self.policy_decision.manifest_fingerprint != self.source_manifest_fingerprint:
            raise DomainError("policy decision belongs to a different source manifest")
        if self.policy_decision.request_fingerprint != self.use_request.fingerprint:
            raise DomainError("policy decision belongs to a different use request")
        if not self.policy_decision.authorized:
            raise DomainError("a dataset build manifest cannot authorize a denied use")
        records = tuple(self.records)
        rejections = tuple(self.rejections)
        split_fingerprints = tuple(self.split_manifest_fingerprints)
        object.__setattr__(self, "records", records)
        object.__setattr__(self, "rejections", rejections)
        object.__setattr__(self, "split_manifest_fingerprints", split_fingerprints)
        if any(not isinstance(record, DatasetRecordSummary) for record in records):
            raise TypeError("dataset manifest records must be DatasetRecordSummary")
        if any(not isinstance(item, DatasetRejection) for item in rejections):
            raise TypeError("dataset manifest rejections must be DatasetRejection")
        if any(
            not isinstance(item, tuple)
            or len(item) != 2
            or not isinstance(item[0], str)
            or not isinstance(item[1], str)
            for item in split_fingerprints
        ):
            raise TypeError("split fingerprints must be (mode, digest) pairs")
        for mode_name, digest in split_fingerprints:
            try:
                SplitMode(mode_name)
            except ValueError as exc:
                raise DomainError(f"unknown split-manifest mode {mode_name!r}") from exc
            _require_digest(digest, name=f"{mode_name} split-manifest fingerprint")
        if len({mode for mode, _ in split_fingerprints}) != len(split_fingerprints):
            raise DomainError("split-manifest modes must be unique")
        if not isinstance(self.deduplication_level, DeduplicationLevel):
            raise TypeError("deduplication_level must be DeduplicationLevel")
        counts = (
            self.input_count,
            self.accepted_pre_dedup_count,
            self.output_count,
            self.rejection_count,
        )
        if any(
            isinstance(value, bool) or not isinstance(value, int) or value < 0
            for value in counts
        ):
            raise DomainError("dataset manifest counts must be nonnegative integers")
        if self.input_count != self.accepted_pre_dedup_count + self.rejection_count:
            raise DomainError("accepted and rejected rows do not partition the input")
        if self.output_count != len(records) or self.rejection_count != len(rejections):
            raise DomainError("dataset manifest counts disagree with serialized rows")
        if self.output_count > self.accepted_pre_dedup_count:
            raise DomainError("deduplication cannot increase the record count")
        record_ids = tuple(record.record_id for record in records)
        if tuple(sorted(record_ids)) != record_ids or len(set(record_ids)) != len(record_ids):
            raise DomainError("dataset manifest records must be canonically ordered")
        selected_identity = {
            DeduplicationLevel.EXACT: "exact_key",
            DeduplicationLevel.STRUCTURE: "structure_key",
            DeduplicationLevel.POLYMORPH: "polymorph_key",
        }[self.deduplication_level]
        if any(
            record.record_id != getattr(record, selected_identity) for record in records
        ):
            raise DomainError(
                "dataset record identifiers disagree with the selected duplicate quotient"
            )
        if tuple(sorted(split_fingerprints)) != split_fingerprints:
            raise DomainError("split fingerprints must be in canonical order")
        rejection_order = tuple(
            sorted(
                rejections,
                key=lambda item: (item.reference, item.stage.value, item.payload_sha256),
            )
        )
        if rejections != rejection_order:
            raise DomainError("dataset manifest rejections must be canonically ordered")
        accepted_references = tuple(alias for record in records for alias in record.aliases)
        rejected_references = tuple(item.reference for item in rejections)
        all_references = accepted_references + rejected_references
        if len(accepted_references) != self.accepted_pre_dedup_count:
            raise DomainError("record aliases do not cover every accepted source row")
        if len(set(all_references)) != len(all_references):
            raise DomainError("a source row appears more than once in the dataset manifest")
        if len(all_references) != self.input_count:
            raise DomainError("manifest source rows do not partition the input count")
        expected = _fingerprint(self._unsigned_payload())
        if self.manifest_fingerprint is None:
            object.__setattr__(self, "manifest_fingerprint", expected)
        else:
            supplied = _require_digest(
                self.manifest_fingerprint,
                name="dataset build manifest fingerprint",
            )
            if supplied != expected:
                raise DomainError("dataset build manifest is internally inconsistent")

    @property
    def fingerprint(self) -> str:
        assert self.manifest_fingerprint is not None
        return self.manifest_fingerprint

    def _unsigned_payload(self) -> dict[str, object]:
        return {
            "schema": _MANIFEST_SCHEMA,
            "source_manifest_fingerprint": self.source_manifest_fingerprint,
            "use_request": self.use_request.to_dict(),
            "policy_decision": self.policy_decision.to_dict(),
            "target_builder_fingerprint": self.target_builder_fingerprint,
            "catalogue_fingerprint": self.catalogue_fingerprint,
            "crystallography_receipt_fingerprint": self.crystallography_receipt_fingerprint,
            "identity_policy_fingerprint": self.identity_policy_fingerprint,
            "deduplication_level": self.deduplication_level.value,
            "input_count": self.input_count,
            "accepted_pre_dedup_count": self.accepted_pre_dedup_count,
            "output_count": self.output_count,
            "rejection_count": self.rejection_count,
            "records": [record.to_dict() for record in self.records],
            "rejections": [rejection.to_dict() for rejection in self.rejections],
            "split_manifest_fingerprints": dict(self.split_manifest_fingerprints),
        }

    def to_dict(self) -> dict[str, object]:
        return {**self._unsigned_payload(), "manifest_fingerprint": self.fingerprint}

    def write_json(self, path: str | Path) -> None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        payload = (json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n").encode(
            "utf-8"
        )
        descriptor, temporary_name = tempfile.mkstemp(
            dir=destination.parent,
            prefix=f".{destination.name}.",
            suffix=".tmp",
        )
        try:
            with os.fdopen(descriptor, "wb") as stream:
                stream.write(payload)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary_name, destination)
        except BaseException:
            try:
                os.unlink(temporary_name)
            except FileNotFoundError:
                pass
            raise

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> DatasetBuildManifest:
        expected = {
            "schema",
            "source_manifest_fingerprint",
            "use_request",
            "policy_decision",
            "target_builder_fingerprint",
            "catalogue_fingerprint",
            "crystallography_receipt_fingerprint",
            "identity_policy_fingerprint",
            "deduplication_level",
            "input_count",
            "accepted_pre_dedup_count",
            "output_count",
            "rejection_count",
            "records",
            "rejections",
            "split_manifest_fingerprints",
            "manifest_fingerprint",
        }
        if set(payload) != expected or payload["schema"] != _MANIFEST_SCHEMA:
            raise DomainError("dataset build manifest has an unsupported schema or fields")
        records = payload["records"]
        rejections = payload["rejections"]
        split_fingerprints = payload["split_manifest_fingerprints"]
        use_request = payload["use_request"]
        policy_decision = payload["policy_decision"]
        if (
            not isinstance(records, list)
            or not isinstance(rejections, list)
            or not isinstance(split_fingerprints, Mapping)
            or any(not isinstance(item, Mapping) for item in records)
            or any(not isinstance(item, Mapping) for item in rejections)
            or not isinstance(use_request, Mapping)
            or not isinstance(policy_decision, Mapping)
        ):
            raise DomainError("dataset build manifest arrays/maps are malformed")
        if any(
            not isinstance(key, str) or not isinstance(value, str)
            for key, value in split_fingerprints.items()
        ):
            raise DomainError("dataset build split fingerprints must map text to text")
        try:
            level = DeduplicationLevel(payload["deduplication_level"])
        except (TypeError, ValueError) as exc:
            raise DomainError(
                "dataset build manifest has an unknown deduplication level"
            ) from exc
        return cls(
            source_manifest_fingerprint=payload["source_manifest_fingerprint"],  # type: ignore[arg-type]
            use_request=DatasetUseRequest.from_dict(use_request),
            policy_decision=DatasetPolicyDecision.from_dict(policy_decision),
            target_builder_fingerprint=payload["target_builder_fingerprint"],  # type: ignore[arg-type]
            catalogue_fingerprint=payload["catalogue_fingerprint"],  # type: ignore[arg-type]
            crystallography_receipt_fingerprint=payload[
                "crystallography_receipt_fingerprint"
            ],  # type: ignore[arg-type]
            identity_policy_fingerprint=payload["identity_policy_fingerprint"],  # type: ignore[arg-type]
            deduplication_level=level,
            input_count=payload["input_count"],  # type: ignore[arg-type]
            accepted_pre_dedup_count=payload["accepted_pre_dedup_count"],  # type: ignore[arg-type]
            output_count=payload["output_count"],  # type: ignore[arg-type]
            rejection_count=payload["rejection_count"],  # type: ignore[arg-type]
            records=tuple(DatasetRecordSummary.from_dict(item) for item in records),
            rejections=tuple(DatasetRejection.from_dict(item) for item in rejections),
            split_manifest_fingerprints=tuple(sorted(split_fingerprints.items())),
            manifest_fingerprint=payload["manifest_fingerprint"],  # type: ignore[arg-type]
        )

    @classmethod
    def read_json(
        cls,
        path: str | Path,
        *,
        expected_fingerprint: str | None = None,
    ) -> DatasetBuildManifest:
        """Read and validate a manifest, optionally against an external trust anchor."""

        if expected_fingerprint is not None:
            expected_fingerprint = _require_digest(
                expected_fingerprint,
                name="expected dataset build manifest fingerprint",
            )
        try:
            text = Path(path).read_text(encoding="utf-8")
        except (OSError, UnicodeError) as exc:
            raise DomainError(f"cannot read dataset build manifest {path}") from exc
        try:
            payload = json.loads(
                text,
                object_pairs_hook=_object_without_duplicate_keys,
                parse_constant=_reject_nonfinite_json_number,
            )
        except json.JSONDecodeError as exc:
            raise DomainError(f"cannot read dataset build manifest {path}") from exc
        if not isinstance(payload, Mapping):
            raise DomainError("dataset build manifest JSON root must be an object")
        manifest = cls.from_dict(payload)
        if (
            expected_fingerprint is not None
            and manifest.fingerprint != expected_fingerprint
        ):
            raise DomainError(
                "dataset build manifest disagrees with the externally expected fingerprint"
            )
        return manifest


@dataclass(frozen=True, slots=True)
class CompiledDataset:
    """In-memory trainable payload plus all immutable audit receipts."""

    records: tuple[CompiledDatasetRecord, ...]
    rejections: tuple[DatasetRejection, ...]
    splits: tuple[tuple[SplitMode, DatasetSplit], ...]
    source_manifest: DatasetSourceManifest
    policy_decision: DatasetPolicyDecision
    manifest: DatasetBuildManifest

    def __post_init__(self) -> None:
        records = tuple(self.records)
        rejections = tuple(self.rejections)
        splits = tuple(self.splits)
        object.__setattr__(self, "records", records)
        object.__setattr__(self, "rejections", rejections)
        object.__setattr__(self, "splits", splits)
        if not isinstance(self.source_manifest, DatasetSourceManifest):
            raise TypeError("source_manifest must be DatasetSourceManifest")
        if not isinstance(self.policy_decision, DatasetPolicyDecision):
            raise TypeError("policy_decision must be DatasetPolicyDecision")
        if not isinstance(self.manifest, DatasetBuildManifest):
            raise TypeError("manifest must be DatasetBuildManifest")
        record_ids = tuple(record.record_id for record in records)
        if tuple(sorted(record_ids)) != record_ids or len(set(record_ids)) != len(record_ids):
            raise DomainError("compiled record identifiers must be sorted and unique")
        if tuple(mode.value for mode, _ in splits) != tuple(
            sorted(mode.value for mode, _ in splits)
        ):
            raise DomainError("compiled split modes must be in canonical order")
        if len({mode for mode, _ in splits}) != len(splits):
            raise DomainError("compiled split modes must be unique")
        if not self.policy_decision.authorized:
            raise DomainError("a compiled dataset cannot carry a denied source policy")
        if self.source_manifest.fingerprint != self.manifest.source_manifest_fingerprint:
            raise DomainError("compiled source receipts disagree with the build manifest")
        if self.manifest.policy_decision != self.policy_decision:
            raise DomainError("compiled policy decision disagrees with the build manifest")
        if self.manifest.records != tuple(record.manifest_summary() for record in records):
            raise DomainError("compiled records disagree with the build manifest")
        if self.manifest.rejections != rejections:
            raise DomainError("compiled rejections disagree with the build manifest")
        expected_split_fingerprints = tuple(
            (mode.value, split.manifest.manifest_fingerprint) for mode, split in splits
        )
        if self.manifest.split_manifest_fingerprints != expected_split_fingerprints:
            raise DomainError("compiled splits disagree with the build manifest")
        split_records = tuple(record.split_record() for record in records)
        for _mode, split in splits:
            audit_dataset_split(split_records, split.manifest)

    @property
    def training_records(self) -> tuple[CrystalRecord, ...]:
        return tuple(
            record.training_metadata(index) for index, record in enumerate(self.records)
        )

    def split_for(self, mode: SplitMode) -> DatasetSplit:
        try:
            return dict(self.splits)[mode]
        except KeyError as exc:
            raise DomainError(f"dataset has no {mode.value} split") from exc


@dataclass(frozen=True, slots=True)
class DatasetCompiler:
    """Compile one exact source manifest into canonical training records."""

    catalogue: CertifiedCatalogue
    crystallography_certification: CrystallographyCertification
    source_manifest: DatasetSourceManifest
    use_request: DatasetUseRequest
    target_builder: DatasetTargetBuilder
    identity_policy: IdentityPolicy = IdentityPolicy()
    # Preserve isotropic-volume-distinct states by default.  The scale-free
    # POLYMORPH key remains available for grouping and explicit aggressive
    # deduplication, but it can merge pressure-driven or isostructural
    # volume-collapse phases when thermodynamic metadata are absent.
    deduplication_level: DeduplicationLevel = DeduplicationLevel.STRUCTURE
    require_all_230: bool = True
    require_external_operation_database: bool = True

    def __post_init__(self) -> None:
        if not isinstance(self.catalogue, CertifiedCatalogue):
            raise TypeError("catalogue must be CertifiedCatalogue")
        if not isinstance(self.crystallography_certification, CrystallographyCertification):
            raise TypeError("crystallography_certification must be a proof receipt")
        if not isinstance(self.source_manifest, DatasetSourceManifest):
            raise TypeError("source_manifest must be DatasetSourceManifest")
        if not isinstance(self.use_request, DatasetUseRequest):
            raise TypeError("use_request must be DatasetUseRequest")
        if not callable(self.target_builder):
            raise TypeError("target_builder must be callable")
        if not bool(getattr(self.target_builder, "certified_preprocessing_only", False)):
            raise DomainError(
                "target_builder must certify that it performs crystallographic "
                "preprocessing only"
            )
        _require_digest(
            getattr(self.target_builder, "builder_fingerprint", None),
            name="target builder fingerprint",
        )
        if not isinstance(self.identity_policy, IdentityPolicy):
            raise TypeError("identity_policy must be IdentityPolicy")
        if not isinstance(self.deduplication_level, DeduplicationLevel):
            raise TypeError("deduplication_level must be DeduplicationLevel")
        self.crystallography_certification.require_for(
            self.catalogue,
            require_all_230=self.require_all_230,
            require_external_operation_database=self.require_external_operation_database,
        )
        evaluate_dataset_use(self.source_manifest, self.use_request).require_authorized()

    def _verify_snapshot_contract(
        self,
        candidates: tuple[DatasetCandidate, ...],
        artifact_roots: Mapping[tuple[str, str], str | Path],
    ) -> None:
        receipts = {
            (receipt.source.source_id, receipt.snapshot.snapshot_id): receipt
            for receipt in self.source_manifest.receipts
        }
        if set(artifact_roots) != set(receipts):
            missing = sorted(set(receipts).difference(artifact_roots))
            extra = sorted(set(artifact_roots).difference(receipts))
            raise DomainError(
                f"snapshot artifact roots disagree with the source manifest; "
                f"missing={missing}, extra={extra}"
            )
        grouped_rows: dict[
            tuple[str, str],
            list[tuple[str, str, SourceRecordScopeEvidence]],
        ] = defaultdict(list)
        for candidate in candidates:
            key = (candidate.reference.source_id, candidate.reference.snapshot_id)
            if key not in receipts:
                raise DomainError(f"candidate references undeclared source snapshot {key!r}")
            candidate.scope_evidence.require_for(receipts[key].source)
            grouped_rows[key].append(
                (
                    candidate.reference.source_record_id,
                    candidate.payload_sha256,
                    candidate.scope_evidence,
                )
            )
        if set(grouped_rows) != set(receipts):
            missing = sorted(set(receipts).difference(grouped_rows))
            raise DomainError(f"source snapshots have no candidate rows: {missing}")
        for key, receipt in receipts.items():
            receipt.snapshot.require_files(artifact_roots[key])
            rows = tuple(grouped_rows[key])
            ids = tuple(record_id for record_id, _, _ in rows)
            payload_rows = tuple(
                (record_id, payload_sha256)
                for record_id, payload_sha256, _ in rows
            )
            scope_rows = tuple(
                (record_id, scope) for record_id, _, scope in rows
            )
            if len(ids) != receipt.snapshot.record_count:
                raise DomainError(f"snapshot {key!r} record count disagrees with candidates")
            if ordered_record_ids_sha256(ids) != receipt.snapshot.ordered_record_ids_sha256:
                raise DomainError(f"snapshot {key!r} ordered record index hash disagrees")
            if (
                ordered_record_payloads_sha256(payload_rows)
                != receipt.snapshot.ordered_record_payloads_sha256
            ):
                raise DomainError(f"snapshot {key!r} ordered record payload hash disagrees")
            if (
                ordered_record_scopes_sha256(scope_rows)
                != receipt.snapshot.ordered_record_scopes_sha256
            ):
                raise DomainError(f"snapshot {key!r} ordered record scope hash disagrees")
            require_historical_mp20_archive(receipt, artifact_roots[key], rows)

    @staticmethod
    def _bind_raw(candidate: DatasetCandidate) -> RawPeriodicStructure:
        source_id = (
            f"{candidate.reference.source_id}/{candidate.reference.snapshot_id}/"
            f"{candidate.reference.source_record_id}"
        )
        if isinstance(candidate.payload, str):
            return parse_cif_structure(candidate.payload, source_id=source_id)
        raw = candidate.payload
        return RawPeriodicStructure.from_arrays(
            raw.lattice,
            raw.fractional_positions,
            raw.atomic_numbers,
            occupancies=raw.occupancies,
            labels=raw.labels,
            source_id=source_id,
        )

    def compile(
        self,
        candidates: Sequence[DatasetCandidate],
        *,
        artifact_roots: Mapping[tuple[str, str], str | Path],
        split_modes: Sequence[SplitMode] = tuple(SplitMode),
        split_fractions: SplitFractions = DEFAULT_SPLIT_FRACTIONS,
        split_seed: int = 0,
        require_nonempty_splits: bool = True,
    ) -> CompiledDataset:
        """Compile, deduplicate, split, and self-audit one complete snapshot set."""

        frozen_candidates = tuple(candidates)
        if not frozen_candidates:
            raise DomainError("dataset compilation needs at least one source row")
        if any(not isinstance(candidate, DatasetCandidate) for candidate in frozen_candidates):
            raise TypeError("every dataset input must be a DatasetCandidate")
        references = tuple(candidate.reference for candidate in frozen_candidates)
        if len(set(references)) != len(references):
            raise DomainError("dataset candidates contain a duplicate source-row reference")
        modes = tuple(sorted(set(split_modes), key=lambda mode: mode.value))
        if any(not isinstance(mode, SplitMode) for mode in modes):
            raise TypeError("split_modes must contain only SplitMode values")
        self._verify_snapshot_contract(frozen_candidates, artifact_roots)
        policy_decision = evaluate_dataset_use(self.source_manifest, self.use_request)
        policy_decision.require_authorized()

        accepted_candidates: list[DatasetCandidate] = []
        canonical_records: list[CanonicalRecord] = []
        identities: list[CanonicalStructureIdentity] = []
        rejections: list[DatasetRejection] = []
        for candidate in frozen_candidates:
            try:
                raw = self._bind_raw(candidate)
            except DomainError as exc:
                rejections.append(
                    DatasetRejection(
                        candidate.reference,
                        candidate.payload_sha256,
                        RejectionStage.PARSE,
                        type(exc).__name__,
                        str(exc),
                    )
                )
                continue
            try:
                record = self.target_builder(raw)
                if not isinstance(record, CanonicalTrainingRecord):
                    raise InvariantViolation("target builder returned the wrong result type")
                if record.raw_structure.fingerprint != raw.fingerprint:
                    raise InvariantViolation(
                        "target builder returned a record for a different raw structure"
                    )
            except RecordPreprocessingError as exc:
                rejections.append(
                    DatasetRejection(
                        candidate.reference,
                        candidate.payload_sha256,
                        RejectionStage.CRYSTALLOGRAPHY,
                        type(exc).__name__,
                        str(exc),
                    )
                )
                continue
            try:
                identity = build_canonical_identity(
                    record,
                    catalogue=self.catalogue,
                    policy=self.identity_policy,
                )
            except AmbiguousIdentityError as exc:
                rejections.append(
                    DatasetRejection(
                        candidate.reference,
                        candidate.payload_sha256,
                        RejectionStage.IDENTITY,
                        type(exc).__name__,
                        str(exc),
                    )
                )
                continue
            accepted_candidates.append(candidate)
            canonical_records.append(record)
            identities.append(identity)

        if not canonical_records:
            raise DomainError("every source row failed audited preprocessing")

        grouped: dict[str, list[int]] = defaultdict(list)
        payloads: dict[str, str] = {}
        for index, identity in enumerate(identities):
            key = identity.key_for(self.deduplication_level)
            payload = identity.payload_for(self.deduplication_level)
            if key in payloads and payloads[key] != payload:
                raise InvariantViolation("canonical identity SHA-256 collision")
            payloads[key] = payload
            grouped[key].append(index)

        compiled_records: list[CompiledDatasetRecord] = []
        for key in sorted(grouped):
            members = tuple(grouped[key])
            representative = min(
                members,
                key=lambda index: (
                    accepted_candidates[index].reference,
                    accepted_candidates[index].payload_sha256,
                ),
            )
            compiled_records.append(
                CompiledDatasetRecord(
                    record_id=key,
                    canonical_record=canonical_records[representative],
                    identity=identities[representative],
                    aliases=tuple(accepted_candidates[index].reference for index in members),
                    alias_scope_evidence=tuple(
                        accepted_candidates[index].scope_evidence for index in members
                    ),
                    representative_payload_sha256=accepted_candidates[
                        representative
                    ].payload_sha256,
                )
            )
        record_tuple = tuple(compiled_records)
        split_inputs = tuple(record.split_record() for record in record_tuple)
        splits = tuple(
            (
                mode,
                build_dataset_split(
                    split_inputs,
                    mode=mode,
                    fractions=split_fractions,
                    seed=split_seed,
                    require_nonempty=require_nonempty_splits,
                ),
            )
            for mode in modes
        )
        rejection_tuple = tuple(
            sorted(
                rejections,
                key=lambda item: (item.reference, item.stage.value, item.payload_sha256),
            )
        )
        manifest = DatasetBuildManifest(
            source_manifest_fingerprint=self.source_manifest.fingerprint,
            use_request=self.use_request,
            policy_decision=policy_decision,
            target_builder_fingerprint=self.target_builder.builder_fingerprint,
            catalogue_fingerprint=self.catalogue.fingerprint,
            crystallography_receipt_fingerprint=(
                self.crystallography_certification.receipt_fingerprint
            ),
            identity_policy_fingerprint=self.identity_policy.fingerprint,
            deduplication_level=self.deduplication_level,
            input_count=len(frozen_candidates),
            accepted_pre_dedup_count=len(canonical_records),
            output_count=len(record_tuple),
            rejection_count=len(rejection_tuple),
            records=tuple(record.manifest_summary() for record in record_tuple),
            rejections=rejection_tuple,
            split_manifest_fingerprints=tuple(
                (mode.value, split.manifest.manifest_fingerprint) for mode, split in splits
            ),
        )
        return CompiledDataset(
            records=record_tuple,
            rejections=rejection_tuple,
            splits=splits,
            source_manifest=self.source_manifest,
            policy_decision=policy_decision,
            manifest=manifest,
        )


__all__ = [
    "CatalogueDatasetTargetBuilder",
    "CompiledDataset",
    "CompiledDatasetRecord",
    "DatasetBuildManifest",
    "DatasetCandidate",
    "DatasetCompiler",
    "DEFAULT_SPLIT_FRACTIONS",
    "DatasetRejection",
    "DatasetRecordSummary",
    "DatasetTargetBuilder",
    "RejectionStage",
    "SourceRecordReference",
    "canonical_training_target_fingerprint",
]
