"""Pinned dataset-source provenance and fail-closed licence policy.

This module deliberately does not download data or legal documents.  A dataset
builder must archive the exact terms it reviewed, record that archive's SHA-256
digest, and separately pin every acquired data artifact.  A URL alone is not an
immutable provenance record because both database contents and terms can change.

The source factories below encode the narrow data scopes selected for the
initial CrystalAtlas pipeline.  They are source declarations, not snapshots:

* Crystallography Open Database (COD) CIF data files under CC0-1.0; and
* NOMAD records explicitly marked open access under CC-BY-4.0;
* Alexandria data under CC-BY-4.0;
* the generated-prototype subset of OQMD under CC-BY-4.0, excluding structures
  whose upstream coordinates came from a restricted third-party database; and
* Materials Project *core* data under CC-BY-4.0, excluding contributed and
  access-controlled batches (including differently licensed batches).

Operational permission flags are conservative engineering assertions, not legal
advice.  Production acquisition must still pin the contemporaneous terms evidence
and database release, and an appropriate reviewer remains responsible for the
chosen use and jurisdiction.

Embedded receipt and manifest fingerprints are integrity checks, not signatures:
an attacker who can replace a JSON document can recompute its self-fingerprint.
Authenticity therefore requires an independently stored fingerprint (or a reviewer
signature).  ``read_json(..., expected_fingerprint=...)`` treats that external pin
as the trust anchor.
"""

from __future__ import annotations

import json
import os
import re
import tempfile
from dataclasses import dataclass
from datetime import date, datetime
from enum import Enum
from hashlib import sha256
from pathlib import Path, PurePosixPath
from typing import Any, ClassVar
from urllib.parse import urlsplit

from .errors import DomainError
from .provenance import sha256_file

_RECEIPT_SCHEMA = "crystalatlas.dataset-source-receipt.v3"
_MANIFEST_SCHEMA = "crystalatlas.dataset-source-manifest.v3"
_POLICY_SCHEMA = "crystalatlas.dataset-use-policy.v1"
_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
_IDENTIFIER_RE = re.compile(r"[a-z0-9](?:[a-z0-9._-]{0,126}[a-z0-9])?\Z")
_SPDX_RE = re.compile(r"(?:[A-Za-z0-9][A-Za-z0-9.+-]*|LicenseRef-[A-Za-z0-9.-]+)\Z")
_UTC_RE = re.compile(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z\Z")
_MUTABLE_SNAPSHOT_NAMES = frozenset(
    {"current", "default", "develop", "head", "latest", "main", "master", "stable"}
)


def _canonical_json(payload: Any) -> str:
    return json.dumps(
        payload,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def _fingerprint(payload: Any) -> str:
    return sha256(_canonical_json(payload).encode("utf-8")).hexdigest()


def _strict_json_loads(payload: str, label: str) -> Any:
    """Decode standards-compliant JSON while rejecting every duplicate key.

    Python's default JSON decoder silently keeps the final value of a repeated
    object key.  That behaviour is unsafe for signed or externally pinned
    provenance because two parsers could assign different meanings to the same
    bytes.  ``object_pairs_hook`` is invoked for every object, so this check also
    covers arbitrarily nested source, licence, snapshot, and artifact objects.
    """

    def reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise DomainError(f"{label} contains duplicate JSON key {key!r}")
            result[key] = value
        return result

    def reject_nonfinite_number(token: str) -> Any:
        raise DomainError(f"{label} contains non-standard JSON number {token!r}")

    try:
        return json.loads(
            payload,
            object_pairs_hook=reject_duplicate_keys,
            parse_constant=reject_nonfinite_number,
        )
    except DomainError:
        raise
    except (TypeError, json.JSONDecodeError) as exc:
        raise DomainError(f"{label} is not valid JSON") from exc


def _atomic_write_text(path: str | Path, payload: str, label: str) -> None:
    """Atomically replace one UTF-8 document and request durable persistence."""

    destination = Path(path)
    temporary: Path | None = None
    try:
        destination.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="\n",
            dir=destination.parent,
            prefix=f".{destination.name}.",
            suffix=".tmp",
            delete=False,
        ) as stream:
            temporary = Path(stream.name)
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, destination)
        temporary = None
        directory_fd = os.open(destination.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    except (OSError, UnicodeError) as exc:
        if temporary is not None:
            try:
                temporary.unlink(missing_ok=True)
            except OSError:
                pass
        raise DomainError(f"cannot write {label} {destination}") from exc


def canonical_record_index_bytes(record_ids: tuple[str, ...] | list[str]) -> bytes:
    """Serialize one exact, ordered, duplicate-free record index.

    Ordering is part of the acquisition contract.  The helper deliberately
    does not sort: silently sorting would make two independently paginated
    exports indistinguishable and could hide missing or repeated pages.
    """

    values = tuple(_require_string(value, "snapshot record identifier") for value in record_ids)
    if not values:
        raise DomainError("a snapshot record index cannot be empty")
    if len(set(values)) != len(values):
        raise DomainError("a snapshot record index cannot contain duplicate identifiers")
    return _canonical_json(
        {
            "schema": "crystalatlas.dataset-record-index.v1",
            "record_ids": list(values),
        }
    ).encode("utf-8")


def ordered_record_ids_sha256(record_ids: tuple[str, ...] | list[str]) -> str:
    """Hash the canonical archived record-index document."""

    return sha256(canonical_record_index_bytes(record_ids)).hexdigest()


def canonical_record_payload_index_bytes(
    records: tuple[tuple[str, str], ...] | list[tuple[str, str]],
) -> bytes:
    """Serialize every ordered identifier with its exact acquired payload.

    The payload digest is the UTF-8 CIF digest or the canonical
    :class:`RawPeriodicStructure` fingerprint used by ``DatasetCandidate``.
    This second index prevents callers from presenting arbitrary payloads with
    the right source identifiers while pointing at an unrelated frozen archive.
    """

    values = tuple(
        (
            _require_string(record_id, "snapshot record identifier"),
            _require_sha256(payload_sha256, "snapshot record payload digest"),
        )
        for record_id, payload_sha256 in records
    )
    if not values:
        raise DomainError("a snapshot record-payload index cannot be empty")
    if len({record_id for record_id, _ in values}) != len(values):
        raise DomainError("a snapshot record-payload index cannot repeat identifiers")
    return _canonical_json(
        {
            "schema": "crystalatlas.dataset-record-payload-index.v1",
            "records": [list(value) for value in values],
        }
    ).encode("utf-8")


def ordered_record_payloads_sha256(
    records: tuple[tuple[str, str], ...] | list[tuple[str, str]],
) -> str:
    """Hash the canonical archived record-payload-index document."""

    return sha256(canonical_record_payload_index_bytes(records)).hexdigest()


def _require_string(value: object, field_name: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise DomainError(f"{field_name} must be a nonempty, stripped string")
    return value


def _require_identifier(value: object, field_name: str) -> str:
    text = _require_string(value, field_name)
    if _IDENTIFIER_RE.fullmatch(text) is None:
        raise DomainError(
            f"{field_name} must use lowercase ASCII letters, digits, '.', '_' or '-'"
        )
    return text


def _require_sha256(value: object, field_name: str) -> str:
    text = _require_string(value, field_name)
    if _SHA256_RE.fullmatch(text) is None:
        raise DomainError(f"{field_name} must be one lowercase SHA-256 digest")
    return text


def _require_https_uri(value: object, field_name: str) -> str:
    text = _require_string(value, field_name)
    try:
        parsed = urlsplit(text)
    except ValueError as exc:
        raise DomainError(f"{field_name} must be an absolute HTTPS URI") from exc
    if parsed.scheme != "https" or not parsed.netloc or parsed.username is not None:
        raise DomainError(f"{field_name} must be an absolute HTTPS URI")
    if parsed.fragment:
        raise DomainError(f"{field_name} cannot contain a fragment")
    return text


def _require_date(value: object, field_name: str) -> str:
    text = _require_string(value, field_name)
    try:
        parsed = date.fromisoformat(text)
    except ValueError as exc:
        raise DomainError(f"{field_name} must have ISO date form YYYY-MM-DD") from exc
    if parsed.isoformat() != text:
        raise DomainError(f"{field_name} must have canonical ISO date form YYYY-MM-DD")
    return text


def _require_utc(value: object, field_name: str) -> str:
    text = _require_string(value, field_name)
    if _UTC_RE.fullmatch(text) is None:
        raise DomainError(f"{field_name} must have canonical UTC form YYYY-MM-DDTHH:MM:SSZ")
    try:
        datetime.fromisoformat(text.removesuffix("Z") + "+00:00")
    except ValueError as exc:
        raise DomainError(f"{field_name} is not a valid UTC timestamp") from exc
    return text


def _require_positive_int(value: object, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise DomainError(f"{field_name} must be a positive integer")
    return value


def _require_bool(value: object, field_name: str) -> bool:
    if not isinstance(value, bool):
        raise DomainError(f"{field_name} must be a boolean")
    return value


def _expect_fields(payload: object, expected: frozenset[str], label: str) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise DomainError(f"{label} must be a JSON object")
    observed = frozenset(payload)
    if observed != expected:
        missing = sorted(expected - observed)
        extra = sorted(repr(item) for item in observed - expected)
        raise DomainError(f"{label} fields disagree; missing={missing}, extra={extra}")
    return payload


class Permission(str, Enum):  # noqa: UP042 - Python-3.10 ML smoke compatibility
    """Tri-state permission; only :attr:`ALLOWED` authorizes a requested use."""

    ALLOWED = "allowed"
    PROHIBITED = "prohibited"
    UNRESOLVED = "unresolved"


class RecordScopePolicy(str, Enum):  # noqa: UP042 - Python-3.10 ML smoke compatibility
    """Machine-checkable acquisition scope attached to every source row."""

    SYNTHETIC_INTERNAL = "synthetic-internal-v1"
    COD_MAIN_CC0 = "cod-main-cc0-v1"
    NOMAD_OPEN_CC_BY = "nomad-open-cc-by-v1"
    ALEXANDRIA_CC_BY = "alexandria-versioned-cc-by-v1"
    OQMD_GENERATED_CC_BY = "oqmd-generated-prototype-cc-by-v1"
    MP_CORE_CC_BY = "materials-project-core-cc-by-v2"
    CDVAE_MP20_HISTORICAL = "cdvae-mp20-historical-f857f598-v1"


class ArtifactRole(str, Enum):  # noqa: UP042 - Python-3.10 ML smoke compatibility
    """Semantic role of one immutable snapshot artifact.

    Roles prevent a digest that merely happens to occur somewhere in the
    archive from satisfying a different provenance obligation.
    """

    DATA = "data"
    TERMS = "terms"
    SELECTION_SPEC = "selection-spec"
    RECORD_INDEX = "record-index"
    PAYLOAD_INDEX = "payload-index"
    SCOPE_INDEX = "scope-index"
    AUXILIARY = "auxiliary"


_SCOPE_REQUIREMENTS: dict[RecordScopePolicy, dict[str, str]] = {
    RecordScopePolicy.CDVAE_MP20_HISTORICAL: {
        "benchmark": "cdvae-mp20",
        "source_revision": "f857f598d6f6cca5dc1ea0582d228f12dcc2c2ea",
        "source_partition": "train",
        "source_container_sha256":
            "133bab58dba02d316f9f91174839f17802dce648be4e8659d1a42b32b97fa01d",
        "scope_assertion_basis": "reviewed-historical-provenance",
        "reviewed_lineage": "icsd-initialized-mp-dft-relaxed",
        "reviewed_upstream_license": "CC-BY-4.0",
        "record_license_metadata": "not-provided",
        "upstream_database_version": "unknown",
        "provider_signed": "false",
        "review_artifact_sha256":
            "db9a1cc15d6dcc47d98ef6b83b48ae9ec1b3b4bd30973c645f063937adc586ed",
        "selected_bindings_sha256":
            "79f627f38e6794d4c1f8038f4f91cfbdfdadcf24b78caa4b28120eae34af6291",
    },
    RecordScopePolicy.SYNTHETIC_INTERNAL: {
        "origin": "synthetic",
        "third_party_content": "false",
    },
    RecordScopePolicy.COD_MAIN_CC0: {
        "collection": "cod-main",
        "record_license": "CC0-1.0",
    },
    RecordScopePolicy.NOMAD_OPEN_CC_BY: {
        "access": "open",
        "record_license": "CC-BY-4.0",
        "reusable_structure": "true",
    },
    RecordScopePolicy.ALEXANDRIA_CC_BY: {
        "collection": "alexandria-versioned",
        "record_license": "CC-BY-4.0",
    },
    RecordScopePolicy.OQMD_GENERATED_CC_BY: {
        "lineage": "generated-prototype",
        "record_license": "CC-BY-4.0",
        "restricted_upstream": "false",
    },
    RecordScopePolicy.MP_CORE_CC_BY: {
        "access_controlled": "false",
        "collection": "core",
        "gnome": "false",
        "mpcontribs": "false",
        "record_license": "CC-BY-4.0",
    },
}


@dataclass(frozen=True, slots=True)
class SourceRecordScopeEvidence:
    """Frozen source metadata proving that one row lies in the declared scope.

    This evidence is an acquisition receipt, not a cryptographic attestation by
    the upstream provider.  Its ordered archive is immutable and replayable;
    production governance must additionally pin the complete source receipt
    fingerprint (or a reviewer signature) outside the dataset itself.
    """

    policy: RecordScopePolicy
    claims: tuple[tuple[str, str], ...]
    source_metadata_sha256: str
    evidence_uri: str
    attribution: tuple[str, ...]

    _FIELDS: ClassVar[frozenset[str]] = frozenset(
        {
            "policy",
            "claims",
            "source_metadata_sha256",
            "evidence_uri",
            "attribution",
        }
    )

    def __post_init__(self) -> None:
        if not isinstance(self.policy, RecordScopePolicy):
            raise DomainError("record scope evidence needs a known policy")
        claims = tuple(
            sorted(
                (
                    _require_identifier(key, "scope claim key"),
                    _require_string(value, "scope claim value"),
                )
                for key, value in self.claims
            )
        )
        if not claims or len({key for key, _ in claims}) != len(claims):
            raise DomainError("record scope claims must be nonempty and uniquely keyed")
        object.__setattr__(self, "claims", claims)
        object.__setattr__(
            self,
            "source_metadata_sha256",
            _require_sha256(self.source_metadata_sha256, "source metadata digest"),
        )
        object.__setattr__(
            self,
            "evidence_uri",
            _require_https_uri(self.evidence_uri, "scope evidence URI"),
        )
        attribution = tuple(
            sorted({_require_string(value, "record attribution") for value in self.attribution})
        )
        if not attribution:
            raise DomainError("record scope evidence needs explicit attribution")
        object.__setattr__(self, "attribution", attribution)

    @property
    def fingerprint(self) -> str:
        return _fingerprint(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return {
            "policy": self.policy.value,
            "claims": dict(self.claims),
            "source_metadata_sha256": self.source_metadata_sha256,
            "evidence_uri": self.evidence_uri,
            "attribution": list(self.attribution),
        }

    @classmethod
    def from_dict(cls, payload: object) -> SourceRecordScopeEvidence:
        value = _expect_fields(payload, cls._FIELDS, "source-record scope evidence")
        if not isinstance(value["claims"], dict) or not isinstance(value["attribution"], list):
            raise DomainError("source-record scope claims/attribution are malformed")
        if any(
            not isinstance(key, str) or not isinstance(claim, str)
            for key, claim in value["claims"].items()
        ) or any(not isinstance(item, str) for item in value["attribution"]):
            raise DomainError("source-record scope evidence contains non-text values")
        try:
            policy = RecordScopePolicy(value["policy"])
        except (TypeError, ValueError) as exc:
            raise DomainError("source-record scope evidence has an unknown policy") from exc
        return cls(
            policy=policy,
            claims=tuple(value["claims"].items()),
            source_metadata_sha256=value["source_metadata_sha256"],
            evidence_uri=value["evidence_uri"],
            attribution=tuple(value["attribution"]),
        )

    def require_for(self, source: DatasetSourceDeclaration) -> None:
        if self.policy is not source.record_scope_policy:
            raise DomainError(
                f"record scope policy {self.policy.value!r} does not match "
                f"source {source.source_id!r}"
            )
        observed = dict(self.claims)
        required = _SCOPE_REQUIREMENTS[self.policy]
        if self.policy is RecordScopePolicy.CDVAE_MP20_HISTORICAL and set(observed) != set(
            required
        ):
            raise DomainError("historical MP20 scope requires exactly the reviewed claim set")
        failures = tuple(
            f"{key}={observed.get(key)!r}, required {expected!r}"
            for key, expected in required.items()
            if observed.get(key) != expected
        )
        if failures:
            raise DomainError(
                "record is outside the declared legal scope: " + "; ".join(failures)
            )


def canonical_record_scope_index_bytes(
    records: tuple[tuple[str, SourceRecordScopeEvidence], ...]
    | list[tuple[str, SourceRecordScopeEvidence]],
) -> bytes:
    """Serialize an ordered, per-record legal-scope evidence index."""

    values: list[tuple[str, SourceRecordScopeEvidence]] = []
    for record_id, evidence in records:
        identifier = _require_string(record_id, "snapshot record identifier")
        if not isinstance(evidence, SourceRecordScopeEvidence):
            raise TypeError("record scope index values must be SourceRecordScopeEvidence")
        values.append((identifier, evidence))
    if not values:
        raise DomainError("a snapshot record-scope index cannot be empty")
    if len({record_id for record_id, _ in values}) != len(values):
        raise DomainError("a snapshot record-scope index cannot repeat identifiers")
    return _canonical_json(
        {
            "schema": "crystalatlas.dataset-record-scope-index.v1",
            "records": [
                {"record_id": record_id, "scope": evidence.to_dict()}
                for record_id, evidence in values
            ],
        }
    ).encode("utf-8")


def ordered_record_scopes_sha256(
    records: tuple[tuple[str, SourceRecordScopeEvidence], ...]
    | list[tuple[str, SourceRecordScopeEvidence]],
) -> str:
    """Hash the canonical archived legal-scope evidence index."""

    return sha256(canonical_record_scope_index_bytes(records)).hexdigest()


def _require_permission(value: object, field_name: str) -> Permission:
    if not isinstance(value, Permission):
        raise DomainError(f"{field_name} must be an explicit Permission value")
    return value


@dataclass(frozen=True, slots=True)
class Citation:
    """One stable scientific or legal citation."""

    key: str
    title: str
    uri: str

    _FIELDS: ClassVar[frozenset[str]] = frozenset({"key", "title", "uri"})

    def __post_init__(self) -> None:
        object.__setattr__(self, "key", _require_identifier(self.key, "citation key"))
        object.__setattr__(self, "title", _require_string(self.title, "citation title"))
        object.__setattr__(self, "uri", _require_https_uri(self.uri, "citation URI"))

    def to_dict(self) -> dict[str, str]:
        return {"key": self.key, "title": self.title, "uri": self.uri}

    @classmethod
    def from_dict(cls, payload: object) -> Citation:
        value = _expect_fields(payload, cls._FIELDS, "citation")
        return cls(key=value["key"], title=value["title"], uri=value["uri"])


@dataclass(frozen=True, slots=True)
class LicenseDeclaration:
    """Pinned legal evidence and explicit machine-use permissions.

    ``terms_evidence_sha256`` is the digest of a locally archived copy of the
    terms page/PDF reviewed on ``reviewed_on``.  It is not the hash of the URI.
    """

    spdx_id: str
    name: str
    license_uri: str
    terms_uri: str
    reviewed_on: str
    terms_evidence_sha256: str
    training_use: Permission = Permission.UNRESOLVED
    raw_redistribution: Permission = Permission.UNRESOLVED
    derived_redistribution: Permission = Permission.UNRESOLVED
    commercial_use: Permission = Permission.UNRESOLVED
    attribution_required: bool = True
    notice: str = "Licence and database terms must be reviewed before use."

    _FIELDS: ClassVar[frozenset[str]] = frozenset(
        {
            "spdx_id",
            "name",
            "license_uri",
            "terms_uri",
            "reviewed_on",
            "terms_evidence_sha256",
            "training_use",
            "raw_redistribution",
            "derived_redistribution",
            "commercial_use",
            "attribution_required",
            "notice",
        }
    )

    def __post_init__(self) -> None:
        spdx = _require_string(self.spdx_id, "SPDX licence identifier")
        if _SPDX_RE.fullmatch(spdx) is None:
            raise DomainError("spdx_id is not a supported SPDX-style identifier")
        object.__setattr__(self, "spdx_id", spdx)
        object.__setattr__(self, "name", _require_string(self.name, "licence name"))
        object.__setattr__(
            self, "license_uri", _require_https_uri(self.license_uri, "licence URI")
        )
        object.__setattr__(self, "terms_uri", _require_https_uri(self.terms_uri, "terms URI"))
        object.__setattr__(
            self, "reviewed_on", _require_date(self.reviewed_on, "licence review date")
        )
        object.__setattr__(
            self,
            "terms_evidence_sha256",
            _require_sha256(self.terms_evidence_sha256, "terms evidence digest"),
        )
        for field_name in (
            "training_use",
            "raw_redistribution",
            "derived_redistribution",
            "commercial_use",
        ):
            object.__setattr__(
                self,
                field_name,
                _require_permission(getattr(self, field_name), field_name),
            )
        _require_bool(self.attribution_required, "attribution_required")
        object.__setattr__(self, "notice", _require_string(self.notice, "licence notice"))

    def to_dict(self) -> dict[str, Any]:
        return {
            "spdx_id": self.spdx_id,
            "name": self.name,
            "license_uri": self.license_uri,
            "terms_uri": self.terms_uri,
            "reviewed_on": self.reviewed_on,
            "terms_evidence_sha256": self.terms_evidence_sha256,
            "training_use": self.training_use.value,
            "raw_redistribution": self.raw_redistribution.value,
            "derived_redistribution": self.derived_redistribution.value,
            "commercial_use": self.commercial_use.value,
            "attribution_required": self.attribution_required,
            "notice": self.notice,
        }

    @classmethod
    def from_dict(cls, payload: object) -> LicenseDeclaration:
        value = _expect_fields(payload, cls._FIELDS, "licence declaration")
        try:
            permissions = {
                key: Permission(value[key])
                for key in (
                    "training_use",
                    "raw_redistribution",
                    "derived_redistribution",
                    "commercial_use",
                )
            }
        except (TypeError, ValueError) as exc:
            raise DomainError("licence declaration contains an unknown permission") from exc
        return cls(
            spdx_id=value["spdx_id"],
            name=value["name"],
            license_uri=value["license_uri"],
            terms_uri=value["terms_uri"],
            reviewed_on=value["reviewed_on"],
            terms_evidence_sha256=value["terms_evidence_sha256"],
            attribution_required=value["attribution_required"],
            notice=value["notice"],
            **permissions,
        )


@dataclass(frozen=True, slots=True)
class DatasetSourceDeclaration:
    """Immutable declaration for one narrowly scoped upstream database."""

    source_id: str
    name: str
    provider: str
    homepage_uri: str
    data_scope: str
    record_scope_policy: RecordScopePolicy
    license: LicenseDeclaration
    citations: tuple[Citation, ...]

    _FIELDS: ClassVar[frozenset[str]] = frozenset(
        {
            "source_id",
            "name",
            "provider",
            "homepage_uri",
            "data_scope",
            "record_scope_policy",
            "license",
            "citations",
        }
    )

    def __post_init__(self) -> None:
        object.__setattr__(self, "source_id", _require_identifier(self.source_id, "source_id"))
        object.__setattr__(self, "name", _require_string(self.name, "source name"))
        object.__setattr__(self, "provider", _require_string(self.provider, "source provider"))
        object.__setattr__(
            self, "homepage_uri", _require_https_uri(self.homepage_uri, "source homepage")
        )
        object.__setattr__(self, "data_scope", _require_string(self.data_scope, "data scope"))
        if not isinstance(self.record_scope_policy, RecordScopePolicy):
            raise DomainError("source declaration needs a record-scope policy")
        if not isinstance(self.license, LicenseDeclaration):
            raise TypeError("license must be a LicenseDeclaration")
        citations = tuple(self.citations)
        if not citations or any(not isinstance(item, Citation) for item in citations):
            raise DomainError("a source needs at least one valid citation")
        citations = tuple(sorted(citations, key=lambda item: item.key))
        if len({item.key for item in citations}) != len(citations):
            raise DomainError("source citation keys must be unique")
        object.__setattr__(self, "citations", citations)

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_id": self.source_id,
            "name": self.name,
            "provider": self.provider,
            "homepage_uri": self.homepage_uri,
            "data_scope": self.data_scope,
            "record_scope_policy": self.record_scope_policy.value,
            "license": self.license.to_dict(),
            "citations": [item.to_dict() for item in self.citations],
        }

    @classmethod
    def from_dict(cls, payload: object) -> DatasetSourceDeclaration:
        value = _expect_fields(payload, cls._FIELDS, "dataset source declaration")
        if not isinstance(value["citations"], list):
            raise DomainError("source citations must be a JSON array")
        try:
            record_scope_policy = RecordScopePolicy(value["record_scope_policy"])
        except (TypeError, ValueError) as exc:
            raise DomainError("source declaration has an unknown scope policy") from exc
        return cls(
            source_id=value["source_id"],
            name=value["name"],
            provider=value["provider"],
            homepage_uri=value["homepage_uri"],
            data_scope=value["data_scope"],
            record_scope_policy=record_scope_policy,
            license=LicenseDeclaration.from_dict(value["license"]),
            citations=tuple(Citation.from_dict(item) for item in value["citations"]),
        )


@dataclass(frozen=True, slots=True)
class ArtifactChecksum:
    """Checksum and length of one immutable local snapshot artifact."""

    role: ArtifactRole
    relative_path: str
    sha256: str
    byte_count: int

    _FIELDS: ClassVar[frozenset[str]] = frozenset(
        {"role", "relative_path", "sha256", "byte_count"}
    )

    def __post_init__(self) -> None:
        if not isinstance(self.role, ArtifactRole):
            raise DomainError("artifact role must be a known ArtifactRole")
        path_text = _require_string(self.relative_path, "artifact relative path")
        path = PurePosixPath(path_text)
        if (
            path.is_absolute()
            or path_text != path.as_posix()
            or any(part in {"", ".", ".."} for part in path.parts)
        ):
            raise DomainError("artifact path must be a normalized relative POSIX path")
        object.__setattr__(self, "relative_path", path_text)
        object.__setattr__(self, "sha256", _require_sha256(self.sha256, "artifact SHA-256"))
        object.__setattr__(
            self, "byte_count", _require_positive_int(self.byte_count, "artifact byte_count")
        )

    @classmethod
    def from_file(
        cls,
        path: str | Path,
        *,
        role: ArtifactRole,
        relative_path: str | None = None,
    ) -> ArtifactChecksum:
        """Hash one existing regular file; no network access is performed."""

        source = Path(path)
        if not source.is_file() or source.is_symlink():
            raise DomainError(f"snapshot artifact must be a non-symlink regular file: {source}")
        return cls(
            role=role,
            relative_path=relative_path if relative_path is not None else source.name,
            sha256=sha256_file(source),
            byte_count=source.stat().st_size,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "role": self.role.value,
            "relative_path": self.relative_path,
            "sha256": self.sha256,
            "byte_count": self.byte_count,
        }

    @classmethod
    def from_dict(cls, payload: object) -> ArtifactChecksum:
        value = _expect_fields(payload, cls._FIELDS, "artifact checksum")
        try:
            role = ArtifactRole(value["role"])
        except (TypeError, ValueError) as exc:
            raise DomainError("artifact checksum has an unknown role") from exc
        return cls(
            role=role,
            relative_path=value["relative_path"],
            sha256=value["sha256"],
            byte_count=value["byte_count"],
        )


@dataclass(frozen=True, slots=True)
class DatasetSnapshot:
    """An exact, query-pinned export of one declared source."""

    source_id: str
    snapshot_id: str
    retrieved_utc: str
    retrieval_uri: str
    selection_spec_sha256: str
    ordered_record_ids_sha256: str
    ordered_record_payloads_sha256: str
    ordered_record_scopes_sha256: str
    record_count: int
    artifacts: tuple[ArtifactChecksum, ...]

    _FIELDS: ClassVar[frozenset[str]] = frozenset(
        {
            "source_id",
            "snapshot_id",
            "retrieved_utc",
            "retrieval_uri",
            "selection_spec_sha256",
            "ordered_record_ids_sha256",
            "ordered_record_payloads_sha256",
            "ordered_record_scopes_sha256",
            "record_count",
            "artifacts",
        }
    )

    def __post_init__(self) -> None:
        object.__setattr__(self, "source_id", _require_identifier(self.source_id, "source_id"))
        snapshot_id = _require_string(self.snapshot_id, "snapshot_id")
        if snapshot_id.casefold() in _MUTABLE_SNAPSHOT_NAMES:
            raise DomainError(
                "snapshot_id must name an immutable release, DOI, version or commit"
            )
        object.__setattr__(self, "snapshot_id", snapshot_id)
        object.__setattr__(
            self, "retrieved_utc", _require_utc(self.retrieved_utc, "retrieval timestamp")
        )
        object.__setattr__(
            self, "retrieval_uri", _require_https_uri(self.retrieval_uri, "retrieval URI")
        )
        object.__setattr__(
            self,
            "selection_spec_sha256",
            _require_sha256(self.selection_spec_sha256, "selection-spec digest"),
        )
        object.__setattr__(
            self,
            "ordered_record_ids_sha256",
            _require_sha256(self.ordered_record_ids_sha256, "ordered-record-index digest"),
        )
        object.__setattr__(
            self,
            "ordered_record_payloads_sha256",
            _require_sha256(
                self.ordered_record_payloads_sha256,
                "ordered-record-payload-index digest",
            ),
        )
        object.__setattr__(
            self,
            "ordered_record_scopes_sha256",
            _require_sha256(
                self.ordered_record_scopes_sha256,
                "ordered-record-scope-index digest",
            ),
        )
        object.__setattr__(
            self, "record_count", _require_positive_int(self.record_count, "record_count")
        )
        artifacts = tuple(self.artifacts)
        if not artifacts or any(not isinstance(item, ArtifactChecksum) for item in artifacts):
            raise DomainError("a snapshot needs at least one artifact checksum")
        artifacts = tuple(
            sorted(artifacts, key=lambda item: (item.role.value, item.relative_path))
        )
        if len({item.relative_path for item in artifacts}) != len(artifacts):
            raise DomainError("snapshot artifact paths must be unique")
        artifacts_by_role = {
            role: tuple(item for item in artifacts if item.role is role)
            for role in ArtifactRole
        }
        if not artifacts_by_role[ArtifactRole.DATA]:
            raise DomainError("a snapshot needs at least one data artifact")
        if len(artifacts_by_role[ArtifactRole.TERMS]) != 1:
            raise DomainError("a snapshot needs exactly one terms artifact")

        required_evidence = (
            (
                ArtifactRole.SELECTION_SPEC,
                "selection-spec",
                self.selection_spec_sha256,
            ),
            (
                ArtifactRole.RECORD_INDEX,
                "ordered-record-index",
                self.ordered_record_ids_sha256,
            ),
            (
                ArtifactRole.PAYLOAD_INDEX,
                "ordered-record-payload-index",
                self.ordered_record_payloads_sha256,
            ),
            (
                ArtifactRole.SCOPE_INDEX,
                "ordered-record-scope-index",
                self.ordered_record_scopes_sha256,
            ),
        )
        evidence_digests = tuple(digest for _, _, digest in required_evidence)
        if len(set(evidence_digests)) != len(evidence_digests):
            raise DomainError("snapshot selection and index digests must be pairwise distinct")
        for role, label, expected_digest in required_evidence:
            matching = artifacts_by_role[role]
            if len(matching) != 1:
                raise DomainError(f"a snapshot needs exactly one {label} artifact")
            if matching[0].sha256 != expected_digest:
                raise DomainError(f"snapshot {label} artifact digest disagrees with its field")
        object.__setattr__(self, "artifacts", artifacts)

    @property
    def fingerprint(self) -> str:
        return _fingerprint(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_id": self.source_id,
            "snapshot_id": self.snapshot_id,
            "retrieved_utc": self.retrieved_utc,
            "retrieval_uri": self.retrieval_uri,
            "selection_spec_sha256": self.selection_spec_sha256,
            "ordered_record_ids_sha256": self.ordered_record_ids_sha256,
            "ordered_record_payloads_sha256": self.ordered_record_payloads_sha256,
            "ordered_record_scopes_sha256": self.ordered_record_scopes_sha256,
            "record_count": self.record_count,
            "artifacts": [item.to_dict() for item in self.artifacts],
        }

    @classmethod
    def from_dict(cls, payload: object) -> DatasetSnapshot:
        value = _expect_fields(payload, cls._FIELDS, "dataset snapshot")
        if not isinstance(value["artifacts"], list):
            raise DomainError("snapshot artifacts must be a JSON array")
        return cls(
            source_id=value["source_id"],
            snapshot_id=value["snapshot_id"],
            retrieved_utc=value["retrieved_utc"],
            retrieval_uri=value["retrieval_uri"],
            selection_spec_sha256=value["selection_spec_sha256"],
            ordered_record_ids_sha256=value["ordered_record_ids_sha256"],
            ordered_record_payloads_sha256=value["ordered_record_payloads_sha256"],
            ordered_record_scopes_sha256=value["ordered_record_scopes_sha256"],
            record_count=value["record_count"],
            artifacts=tuple(ArtifactChecksum.from_dict(item) for item in value["artifacts"]),
        )

    def verify_files(self, root: str | Path) -> tuple[str, ...]:
        """Return deterministic integrity failures for the pinned local artifacts."""

        base = Path(root)
        try:
            resolved_base = base.resolve(strict=True)
        except OSError:
            return (f"snapshot root does not exist: {base}",)
        if not resolved_base.is_dir():
            return (f"snapshot root is not a directory: {base}",)

        failures: list[str] = []
        for artifact in self.artifacts:
            candidate = resolved_base.joinpath(*PurePosixPath(artifact.relative_path).parts)
            try:
                resolved_candidate = candidate.resolve(strict=True)
            except OSError:
                failures.append(f"missing snapshot artifact: {artifact.relative_path}")
                continue
            if resolved_base not in resolved_candidate.parents:
                failures.append(f"snapshot artifact escapes root: {artifact.relative_path}")
                continue
            relative_parts = PurePosixPath(artifact.relative_path).parts
            prefix = resolved_base
            if any((prefix := prefix / part).is_symlink() for part in relative_parts):
                failures.append(
                    f"snapshot artifact uses a symbolic link: {artifact.relative_path}"
                )
                continue
            if not resolved_candidate.is_file():
                failures.append(f"snapshot artifact is not a file: {artifact.relative_path}")
                continue
            if resolved_candidate.stat().st_size != artifact.byte_count:
                failures.append(f"snapshot artifact size mismatch: {artifact.relative_path}")
                continue
            if sha256_file(resolved_candidate) != artifact.sha256:
                failures.append(f"snapshot artifact hash mismatch: {artifact.relative_path}")
        return tuple(failures)

    def require_files(self, root: str | Path) -> None:
        failures = self.verify_files(root)
        if failures:
            raise DomainError("; ".join(failures))


@dataclass(frozen=True, slots=True)
class DatasetSourceReceipt:
    """One strict JSON receipt binding legal evidence to an exact data snapshot."""

    source: DatasetSourceDeclaration
    snapshot: DatasetSnapshot
    receipt_fingerprint: str | None = None

    _FIELDS: ClassVar[frozenset[str]] = frozenset(
        {"schema", "source", "snapshot", "receipt_fingerprint"}
    )

    def __post_init__(self) -> None:
        if not isinstance(self.source, DatasetSourceDeclaration):
            raise TypeError("source must be a DatasetSourceDeclaration")
        if not isinstance(self.snapshot, DatasetSnapshot):
            raise TypeError("snapshot must be a DatasetSnapshot")
        if self.source.source_id != self.snapshot.source_id:
            raise DomainError("source declaration and snapshot source_id disagree")
        terms_artifacts = tuple(
            artifact
            for artifact in self.snapshot.artifacts
            if artifact.role is ArtifactRole.TERMS
        )
        if len(terms_artifacts) != 1:
            raise DomainError("the snapshot must contain exactly one terms artifact")
        if terms_artifacts[0].sha256 != self.source.license.terms_evidence_sha256:
            raise DomainError(
                "the terms artifact digest disagrees with the reviewed licence evidence"
            )
        expected = _fingerprint(self._unsigned_payload())
        if self.receipt_fingerprint is None:
            object.__setattr__(self, "receipt_fingerprint", expected)
        else:
            supplied = _require_sha256(self.receipt_fingerprint, "receipt fingerprint")
            if supplied != expected:
                raise DomainError("dataset source receipt is internally inconsistent")

    @property
    def fingerprint(self) -> str:
        assert self.receipt_fingerprint is not None
        return self.receipt_fingerprint

    def _unsigned_payload(self) -> dict[str, Any]:
        return {
            "schema": _RECEIPT_SCHEMA,
            "source": self.source.to_dict(),
            "snapshot": self.snapshot.to_dict(),
        }

    def to_dict(self) -> dict[str, Any]:
        value = self._unsigned_payload()
        value["receipt_fingerprint"] = self.fingerprint
        return value

    def to_json(self) -> str:
        return _canonical_json(self.to_dict()) + "\n"

    @classmethod
    def from_dict(cls, payload: object) -> DatasetSourceReceipt:
        value = _expect_fields(payload, cls._FIELDS, "dataset source receipt")
        if value["schema"] != _RECEIPT_SCHEMA:
            raise DomainError("unsupported dataset source receipt schema")
        return cls(
            source=DatasetSourceDeclaration.from_dict(value["source"]),
            snapshot=DatasetSnapshot.from_dict(value["snapshot"]),
            receipt_fingerprint=value["receipt_fingerprint"],
        )

    @classmethod
    def from_json(cls, payload: str) -> DatasetSourceReceipt:
        value = _strict_json_loads(payload, "dataset source receipt")
        return cls.from_dict(value)

    @classmethod
    def read_json(
        cls,
        path: str | Path,
        *,
        expected_fingerprint: str | None = None,
    ) -> DatasetSourceReceipt:
        """Read a receipt and optionally verify an independently stored pin."""

        source = Path(path)
        try:
            payload = source.read_text(encoding="utf-8")
        except (OSError, UnicodeError) as exc:
            raise DomainError(f"cannot read dataset source receipt {source}") from exc
        receipt = cls.from_json(payload)
        if expected_fingerprint is not None:
            pinned = _require_sha256(expected_fingerprint, "expected receipt fingerprint")
            if receipt.fingerprint != pinned:
                raise DomainError(
                    "dataset source receipt does not match the externally pinned fingerprint"
                )
        return receipt


@dataclass(frozen=True, slots=True)
class DatasetSourceManifest:
    """Canonical collection of source receipts used to build one corpus."""

    receipts: tuple[DatasetSourceReceipt, ...]
    manifest_fingerprint: str | None = None

    _FIELDS: ClassVar[frozenset[str]] = frozenset(
        {"schema", "receipts", "manifest_fingerprint"}
    )

    def __post_init__(self) -> None:
        receipts = tuple(self.receipts)
        if not receipts or any(not isinstance(item, DatasetSourceReceipt) for item in receipts):
            raise DomainError("a dataset source manifest needs at least one receipt")
        receipts = tuple(
            sorted(
                receipts, key=lambda item: (item.source.source_id, item.snapshot.snapshot_id)
            )
        )
        keys = tuple((item.source.source_id, item.snapshot.snapshot_id) for item in receipts)
        if len(set(keys)) != len(keys):
            raise DomainError("dataset source receipt identities must be unique")
        object.__setattr__(self, "receipts", receipts)
        expected = _fingerprint(self._unsigned_payload())
        if self.manifest_fingerprint is None:
            object.__setattr__(self, "manifest_fingerprint", expected)
        else:
            supplied = _require_sha256(self.manifest_fingerprint, "manifest fingerprint")
            if supplied != expected:
                raise DomainError("dataset source manifest is internally inconsistent")

    @property
    def fingerprint(self) -> str:
        assert self.manifest_fingerprint is not None
        return self.manifest_fingerprint

    def _unsigned_payload(self) -> dict[str, Any]:
        return {
            "schema": _MANIFEST_SCHEMA,
            "receipts": [item.to_dict() for item in self.receipts],
        }

    def to_dict(self) -> dict[str, Any]:
        value = self._unsigned_payload()
        value["manifest_fingerprint"] = self.fingerprint
        return value

    def to_json(self) -> str:
        return _canonical_json(self.to_dict()) + "\n"

    def write_json(self, path: str | Path) -> None:
        _atomic_write_text(path, self.to_json(), "dataset source manifest")

    @classmethod
    def from_dict(cls, payload: object) -> DatasetSourceManifest:
        value = _expect_fields(payload, cls._FIELDS, "dataset source manifest")
        if value["schema"] != _MANIFEST_SCHEMA:
            raise DomainError("unsupported dataset source manifest schema")
        if not isinstance(value["receipts"], list):
            raise DomainError("dataset source receipts must be a JSON array")
        return cls(
            receipts=tuple(DatasetSourceReceipt.from_dict(item) for item in value["receipts"]),
            manifest_fingerprint=value["manifest_fingerprint"],
        )

    @classmethod
    def from_json(cls, payload: str) -> DatasetSourceManifest:
        value = _strict_json_loads(payload, "dataset source manifest")
        return cls.from_dict(value)

    @classmethod
    def read_json(
        cls,
        path: str | Path,
        *,
        expected_fingerprint: str | None = None,
    ) -> DatasetSourceManifest:
        """Read a manifest and optionally verify an independently stored pin."""

        source = Path(path)
        try:
            payload = source.read_text(encoding="utf-8")
        except (OSError, UnicodeError) as exc:
            raise DomainError(f"cannot read dataset source manifest {source}") from exc
        manifest = cls.from_json(payload)
        if expected_fingerprint is not None:
            pinned = _require_sha256(expected_fingerprint, "expected manifest fingerprint")
            if manifest.fingerprint != pinned:
                raise DomainError(
                    "dataset source manifest does not match the externally pinned fingerprint"
                )
        return manifest


@dataclass(frozen=True, slots=True)
class DatasetUseRequest:
    """The exact downstream uses for which source licences are evaluated."""

    train_model: bool = True
    redistribute_raw_data: bool = False
    redistribute_derived_data: bool = False
    commercial_use: bool = False

    def __post_init__(self) -> None:
        for field_name in (
            "train_model",
            "redistribute_raw_data",
            "redistribute_derived_data",
            "commercial_use",
        ):
            _require_bool(getattr(self, field_name), field_name)
        if not any(
            (
                self.train_model,
                self.redistribute_raw_data,
                self.redistribute_derived_data,
                self.commercial_use,
            )
        ):
            raise DomainError("dataset use request must request at least one use")

    @property
    def fingerprint(self) -> str:
        return _fingerprint(self.to_dict())

    def to_dict(self) -> dict[str, bool]:
        return {
            "train_model": self.train_model,
            "redistribute_raw_data": self.redistribute_raw_data,
            "redistribute_derived_data": self.redistribute_derived_data,
            "commercial_use": self.commercial_use,
        }

    @classmethod
    def from_dict(cls, payload: object) -> DatasetUseRequest:
        value = _expect_fields(
            payload,
            frozenset(
                {
                    "train_model",
                    "redistribute_raw_data",
                    "redistribute_derived_data",
                    "commercial_use",
                }
            ),
            "dataset use request",
        )
        return cls(
            train_model=value["train_model"],
            redistribute_raw_data=value["redistribute_raw_data"],
            redistribute_derived_data=value["redistribute_derived_data"],
            commercial_use=value["commercial_use"],
        )


@dataclass(frozen=True, slots=True)
class DatasetPolicyDecision:
    """Deterministic policy evaluation; denied is the default outcome."""

    manifest_fingerprint: str
    request_fingerprint: str
    authorized: bool
    denial_reasons: tuple[str, ...]
    obligations: tuple[str, ...]
    decision_fingerprint: str | None = None

    def __post_init__(self) -> None:
        _require_sha256(self.manifest_fingerprint, "policy manifest fingerprint")
        _require_sha256(self.request_fingerprint, "policy request fingerprint")
        _require_bool(self.authorized, "authorized")
        reasons = tuple(
            sorted({_require_string(item, "denial reason") for item in self.denial_reasons})
        )
        obligations = tuple(
            sorted({_require_string(item, "licence obligation") for item in self.obligations})
        )
        if self.authorized == bool(reasons):
            raise DomainError("authorized policy decisions must have no denial reasons")
        object.__setattr__(self, "denial_reasons", reasons)
        object.__setattr__(self, "obligations", obligations)
        expected = _fingerprint(self._unsigned_payload())
        if self.decision_fingerprint is None:
            object.__setattr__(self, "decision_fingerprint", expected)
        else:
            supplied = _require_sha256(self.decision_fingerprint, "policy decision fingerprint")
            if supplied != expected:
                raise DomainError("dataset policy decision is internally inconsistent")

    @property
    def fingerprint(self) -> str:
        assert self.decision_fingerprint is not None
        return self.decision_fingerprint

    def _unsigned_payload(self) -> dict[str, Any]:
        return {
            "schema": _POLICY_SCHEMA,
            "manifest_fingerprint": self.manifest_fingerprint,
            "request_fingerprint": self.request_fingerprint,
            "authorized": self.authorized,
            "denial_reasons": list(self.denial_reasons),
            "obligations": list(self.obligations),
        }

    def to_dict(self) -> dict[str, Any]:
        value = self._unsigned_payload()
        value["decision_fingerprint"] = self.fingerprint
        return value

    @classmethod
    def from_dict(cls, payload: object) -> DatasetPolicyDecision:
        value = _expect_fields(
            payload,
            frozenset(
                {
                    "schema",
                    "manifest_fingerprint",
                    "request_fingerprint",
                    "authorized",
                    "denial_reasons",
                    "obligations",
                    "decision_fingerprint",
                }
            ),
            "dataset policy decision",
        )
        if value["schema"] != _POLICY_SCHEMA:
            raise DomainError("unsupported dataset policy-decision schema")
        if not isinstance(value["denial_reasons"], list) or not isinstance(
            value["obligations"], list
        ):
            raise DomainError("dataset policy decision arrays are malformed")
        return cls(
            manifest_fingerprint=value["manifest_fingerprint"],
            request_fingerprint=value["request_fingerprint"],
            authorized=value["authorized"],
            denial_reasons=tuple(value["denial_reasons"]),
            obligations=tuple(value["obligations"]),
            decision_fingerprint=value["decision_fingerprint"],
        )

    def require_authorized(self) -> None:
        if not self.authorized:
            raise DomainError("dataset use denied: " + "; ".join(self.denial_reasons))


def evaluate_dataset_use(
    manifest: DatasetSourceManifest, request: DatasetUseRequest
) -> DatasetPolicyDecision:
    """Evaluate every requested use; anything but explicit ``allowed`` denies it."""

    if not isinstance(manifest, DatasetSourceManifest):
        raise TypeError("manifest must be a DatasetSourceManifest")
    if not isinstance(request, DatasetUseRequest):
        raise TypeError("request must be a DatasetUseRequest")

    reasons: list[str] = []
    obligations: list[str] = []
    requested_permissions = (
        (request.train_model, "training_use", "model training"),
        (request.redistribute_raw_data, "raw_redistribution", "raw-data redistribution"),
        (
            request.redistribute_derived_data,
            "derived_redistribution",
            "derived-data redistribution",
        ),
        (request.commercial_use, "commercial_use", "commercial use"),
    )
    for receipt in manifest.receipts:
        source = receipt.source
        licence = source.license
        for requested, field_name, label in requested_permissions:
            if not requested:
                continue
            permission = getattr(licence, field_name)
            if permission is not Permission.ALLOWED:
                reasons.append(
                    f"{source.source_id}: {label} is {permission.value}, not explicitly allowed"
                )
        obligations.append(f"{source.source_id}: {licence.notice}")
        obligations.extend(
            f"{source.source_id}: cite {citation.key} ({citation.uri})"
            for citation in source.citations
        )
        if licence.attribution_required:
            obligations.append(
                f"{source.source_id}: preserve attribution and the {licence.spdx_id} notice"
            )

    return DatasetPolicyDecision(
        manifest_fingerprint=manifest.fingerprint,
        request_fingerprint=request.fingerprint,
        authorized=not reasons,
        denial_reasons=tuple(reasons),
        obligations=tuple(obligations),
    )


SELECTED_SOURCE_IDS = (
    "alexandria",
    "cod",
    "materials-project-core",
    "nomad-open",
    "oqmd-generated-prototypes",
)


def crystallography_open_database_source(
    *, terms_evidence_sha256: str, reviewed_on: str
) -> DatasetSourceDeclaration:
    """Declare the selected COD CIF-file scope after archiving its CC0 evidence."""

    return DatasetSourceDeclaration(
        source_id="cod",
        name="Crystallography Open Database",
        provider="Crystallography Open Database advisory board and maintainers",
        homepage_uri="https://www.crystallography.net/cod/",
        data_scope=(
            "COD crystallographic CIF data files only; excludes COD software, linked "
            "third-party payloads, and the separate predicted COD collection"
        ),
        record_scope_policy=RecordScopePolicy.COD_MAIN_CC0,
        license=LicenseDeclaration(
            spdx_id="CC0-1.0",
            name="Creative Commons Zero v1.0 Universal",
            license_uri="https://creativecommons.org/publicdomain/zero/1.0/",
            terms_uri="https://crystallography.net/cod/new.html",
            reviewed_on=reviewed_on,
            terms_evidence_sha256=terms_evidence_sha256,
            training_use=Permission.ALLOWED,
            raw_redistribution=Permission.ALLOWED,
            derived_redistribution=Permission.ALLOWED,
            commercial_use=Permission.ALLOWED,
            attribution_required=False,
            notice=(
                "Retain snapshot provenance and cite COD even though CC0 does not "
                "legally require attribution."
            ),
        ),
        citations=(
            Citation(
                key="grazulis-2012-cod",
                title=(
                    "Crystallography Open Database (COD): an open-access collection "
                    "of crystal structures and platform for world-wide collaboration"
                ),
                uri="https://doi.org/10.1093/nar/gkr900",
            ),
            Citation(
                key="cod-licence-policy",
                title="COD CC0 data policy",
                uri="https://crystallography.net/cod/new.html",
            ),
        ),
    )


def nomad_open_source(
    *, terms_evidence_sha256: str, reviewed_on: str
) -> DatasetSourceDeclaration:
    """Declare only NOMAD content whose access state is explicitly open."""

    return DatasetSourceDeclaration(
        source_id="nomad-open",
        name="NOMAD open-access materials data",
        provider="NOMAD Laboratory",
        homepage_uri="https://nomad-lab.eu/",
        data_scope=(
            "NOMAD records explicitly marked open access at snapshot time; excludes "
            "restricted records and metadata-only records without reusable structures"
        ),
        record_scope_policy=RecordScopePolicy.NOMAD_OPEN_CC_BY,
        license=LicenseDeclaration(
            spdx_id="CC-BY-4.0",
            name="Creative Commons Attribution 4.0 International",
            license_uri="https://creativecommons.org/licenses/by/4.0/",
            terms_uri="https://cloud.nomad-lab.eu/nomad-lab/terms.html",
            reviewed_on=reviewed_on,
            terms_evidence_sha256=terms_evidence_sha256,
            training_use=Permission.ALLOWED,
            raw_redistribution=Permission.ALLOWED,
            derived_redistribution=Permission.ALLOWED,
            commercial_use=Permission.ALLOWED,
            attribution_required=True,
            notice=(
                "Use only records whose snapshot access state is open; attribute each "
                "submitter/right-holder and NOMAD, and record transformations."
            ),
        ),
        citations=(
            Citation(
                key="nomad-terms",
                title="NOMAD Terms of Use",
                uri="https://cloud.nomad-lab.eu/nomad-lab/terms.html",
            ),
            Citation(
                key="draxl-2019-nomad",
                title="The NOMAD laboratory: from data sharing to artificial intelligence",
                uri="https://doi.org/10.1088/2515-7639/ab13bb",
            ),
        ),
    )


def alexandria_source(
    *, terms_evidence_sha256: str, reviewed_on: str
) -> DatasetSourceDeclaration:
    """Declare the versioned Alexandria structure-data release."""

    return DatasetSourceDeclaration(
        source_id="alexandria",
        name="Alexandria materials database",
        provider="Interdisciplinary Centre for Advanced Materials Simulation",
        homepage_uri="https://alexandria.icams.rub.de/",
        data_scope="Versioned Alexandria crystal-structure data released under CC-BY-4.0",
        record_scope_policy=RecordScopePolicy.ALEXANDRIA_CC_BY,
        license=LicenseDeclaration(
            spdx_id="CC-BY-4.0",
            name="Creative Commons Attribution 4.0 International",
            license_uri="https://creativecommons.org/licenses/by/4.0/",
            terms_uri="https://alexandria.icams.rub.de/about.html",
            reviewed_on=reviewed_on,
            terms_evidence_sha256=terms_evidence_sha256,
            training_use=Permission.ALLOWED,
            raw_redistribution=Permission.ALLOWED,
            derived_redistribution=Permission.ALLOWED,
            commercial_use=Permission.ALLOWED,
            attribution_required=True,
            notice=(
                "Attribute Alexandria and its authors, link CC-BY-4.0, identify the "
                "snapshot, and state transformations."
            ),
        ),
        citations=(
            Citation(
                key="alexandria-about",
                title="Alexandria database licence and citation policy",
                uri="https://alexandria.icams.rub.de/about.html",
            ),
        ),
    )


def oqmd_generated_prototypes_source(
    *, terms_evidence_sha256: str, reviewed_on: str
) -> DatasetSourceDeclaration:
    """Declare only the OQMD generated-prototype lineage.

    OQMD also contains calculations initialized from ICSD structures.  Those
    rows are intentionally outside this declaration even though OQMD labels
    its distributed data CC-BY-4.0; a snapshot selection specification must
    prove the generated-prototype lineage rather than infer it from geometry.
    """

    return DatasetSourceDeclaration(
        source_id="oqmd-generated-prototypes",
        name="OQMD generated-prototype subset",
        provider="Open Quantum Materials Database",
        homepage_uri="https://www.oqmd.org/",
        data_scope=(
            "OQMD generated-prototype/decoration lineage only; excludes every row whose "
            "upstream coordinates or prototype originated in ICSD or another restricted "
            "third-party structure database"
        ),
        record_scope_policy=RecordScopePolicy.OQMD_GENERATED_CC_BY,
        license=LicenseDeclaration(
            spdx_id="CC-BY-4.0",
            name="Creative Commons Attribution 4.0 International",
            license_uri="https://creativecommons.org/licenses/by/4.0/",
            terms_uri="https://www.oqmd.org/download/",
            reviewed_on=reviewed_on,
            terms_evidence_sha256=terms_evidence_sha256,
            training_use=Permission.ALLOWED,
            raw_redistribution=Permission.ALLOWED,
            derived_redistribution=Permission.ALLOWED,
            commercial_use=Permission.ALLOWED,
            attribution_required=True,
            notice=(
                "Attribute OQMD and preserve CC-BY-4.0; the frozen selection must exclude "
                "all restricted third-party upstream structure lineages."
            ),
        ),
        citations=(
            Citation(
                key="oqmd-download-license",
                title="OQMD versioned downloads and data licence",
                uri="https://www.oqmd.org/download/",
            ),
            Citation(
                key="kirklin-2015-oqmd",
                title="The Open Quantum Materials Database: assessing DFT formation energies",
                uri="https://doi.org/10.1038/npjcompumats.2015.10",
            ),
        ),
    )


def historical_cdvae_mp20_scope(*, source_metadata_sha256: str) -> SourceRecordScopeEvidence:
    """Bind a row digest to the exact historical review, not to inferred metadata."""
    policy = RecordScopePolicy.CDVAE_MP20_HISTORICAL
    claims = _SCOPE_REQUIREMENTS[policy]
    return SourceRecordScopeEvidence(
        policy=policy,
        claims=tuple(claims.items()),
        source_metadata_sha256=source_metadata_sha256,
        evidence_uri=(
            "https://raw.githubusercontent.com/txie-93/cdvae/"
            f"{claims['source_revision']}/data/mp_20/train.csv"
        ),
        attribution=(
            "Materials Project: Jain et al. (2013), doi:10.1063/1.4812323",
            "CDVAE benchmark curation: Xie et al., arXiv:2110.06197",
        ),
    )


def historical_cdvae_mp20_source(
    *, reviewed_on: str, training_use: Permission = Permission.UNRESOLVED
) -> DatasetSourceDeclaration:
    """Declare only the reviewed historical pilot, without inferring permission.

    Passing ALLOWED is an explicit reviewer assertion, not a provider signature
    or a consequence of possessing a file. Other uses remain unresolved. The
    compiler separately requires the exact archived review, metadata and source
    bindings; this declaration alone does not authorize arbitrary MP records.
    """
    terms_uri = (
        "https://matsci.org/t/is-it-ok-to-post-snapshots-of-materials-project-"
        "structures-on-e-g-figshare-i-e-for-manuscript-reproducibility/45437/2"
    )
    return DatasetSourceDeclaration(
        source_id="cdvae-mp20-historical",
        name="Historical CDVAE MP20 training pilot",
        provider="Materials Project (upstream); CDVAE (benchmark curator)",
        homepage_uri="https://github.com/txie-93/cdvae",
        data_scope=(
            "Frozen f857f598 CDVAE MP20 train-only pilot; reviewed ICSD-initialized "
            "DFT-relaxed MP lineage, not a raw ICSD export; original MP database "
            "version and record-level licence metadata unknown"
        ),
        record_scope_policy=RecordScopePolicy.CDVAE_MP20_HISTORICAL,
        license=LicenseDeclaration(
            spdx_id="CC-BY-4.0",
            name="Reported historical upstream CC-BY-4.0",
            license_uri="https://creativecommons.org/licenses/by/4.0/",
            terms_uri=terms_uri,
            reviewed_on=reviewed_on,
            terms_evidence_sha256=
                "67da73530a0444b2135dcc8aaae869349644a4567da45411cfbbcbb82acbd4b6",
            training_use=training_use,
            raw_redistribution=Permission.UNRESOLVED,
            derived_redistribution=Permission.UNRESOLVED,
            commercial_use=Permission.UNRESOLVED,
            attribution_required=True,
            notice=(
                "Retain Materials Project and CDVAE attribution, CC-BY-4.0 notices, "
                "ICSD starting lineage, unknown MP version and transformation record. "
                "Evidence is dated provider clarification, not current terms or a signature."
            ),
        ),
        citations=(
            Citation("jain-2013-materials-project", "Materials Project",
                     "https://doi.org/10.1063/1.4812323"),
            Citation("xie-cdvae", "Crystal Diffusion Variational Autoencoder",
                     "https://arxiv.org/abs/2110.06197"),
            Citation("mp-historical-clarification", "MP staff clarification, 2022-11-14",
                     terms_uri),
        ),
    )


def materials_project_core_source(
    *, terms_evidence_sha256: str, reviewed_on: str
) -> DatasetSourceDeclaration:
    """Declare the selected MP core-data scope after archiving its licence evidence.

    Access-controlled batches and MPContribs are excluded because their licence can
    differ from the core database.  Acquisition code must enforce that exclusion and
    pin the returned access-controlled batch identifiers in its selection specification.
    """

    return DatasetSourceDeclaration(
        source_id="materials-project-core",
        name="Materials Project core database",
        provider="Materials Project",
        homepage_uri="https://materialsproject.org/",
        data_scope=(
            "Materials Project core records explicitly covered by CC-BY-4.0 only; "
            "excludes MPContribs and every access-controlled or differently licensed "
            "batch, including BY-NC batches"
        ),
        record_scope_policy=RecordScopePolicy.MP_CORE_CC_BY,
        license=LicenseDeclaration(
            spdx_id="CC-BY-4.0",
            name="Creative Commons Attribution 4.0 International",
            license_uri="https://creativecommons.org/licenses/by/4.0/",
            terms_uri="https://materialsproject.org/about/terms",
            reviewed_on=reviewed_on,
            terms_evidence_sha256=terms_evidence_sha256,
            training_use=Permission.ALLOWED,
            raw_redistribution=Permission.ALLOWED,
            derived_redistribution=Permission.ALLOWED,
            commercial_use=Permission.ALLOWED,
            attribution_required=True,
            notice=(
                "Attribute Materials Project, state the pinned database version, "
                "identify transformations, and preserve the CC-BY-4.0 notice."
            ),
        ),
        citations=(
            Citation(
                key="jain-2013-materials-project",
                title="Commentary: The Materials Project: A materials genome approach",
                uri="https://doi.org/10.1063/1.4812323",
            ),
            Citation(
                key="materials-project-licence",
                title="Materials Project licence and terms",
                uri="https://materialsproject.org/about/terms",
            ),
            Citation(
                key="materials-project-database-versions",
                title="Materials Project database versions",
                uri="https://docs.materialsproject.org/changes/database-versions",
            ),
        ),
    )


__all__ = [
    "SELECTED_SOURCE_IDS",
    "ArtifactChecksum",
    "ArtifactRole",
    "Citation",
    "DatasetPolicyDecision",
    "DatasetSnapshot",
    "DatasetSourceDeclaration",
    "DatasetSourceManifest",
    "DatasetSourceReceipt",
    "DatasetUseRequest",
    "LicenseDeclaration",
    "Permission",
    "RecordScopePolicy",
    "SourceRecordScopeEvidence",
    "alexandria_source",
    "canonical_record_index_bytes",
    "canonical_record_payload_index_bytes",
    "canonical_record_scope_index_bytes",
    "crystallography_open_database_source",
    "evaluate_dataset_use",
    "historical_cdvae_mp20_source",
    "historical_cdvae_mp20_scope",
    "materials_project_core_source",
    "nomad_open_source",
    "oqmd_generated_prototypes_source",
    "ordered_record_ids_sha256",
    "ordered_record_payloads_sha256",
    "ordered_record_scopes_sha256",
]
