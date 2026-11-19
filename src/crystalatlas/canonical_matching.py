"""Strict canonical-partition payloads for reproducible diagnostics.

This module does **not** implement a tolerance-based physical structure
similarity test.  It packages the existing post-standardization canonical
``STRUCTURE`` partition into immutable, provenance-bound bytes and exposes a
pure equality callable suitable for :mod:`crystalatlas.diagnostics`.

Crystallographic standardization and identity construction must happen before
this boundary.  In particular, generated structures must pass the same
FullPrim, pinned-Hall, canonical-origin, exact-Wyckoff, decoder-round-trip, and
``IdentityPolicy`` gates as dataset targets.  A failure is an invalid generated
attempt with ``payload=None``; it must retain its ranked diagnostic slot.  Raw
failure artifacts are kept outside the canonical matcher input.

Every public encoder, validator, and production evaluator requires the live
``CertifiedCatalogue`` and ``IdentityPolicy`` objects whose fingerprints are
bound by provenance.  The inner JSON is resolved against those live objects;
a recomputed self-hash cannot legitimize an invented setting, Wyckoff row,
parameter arity, orbit order, or nonphysical metric.

The current identity implementation propagates certified positional
uncertainty into fractional-coordinate bin guards, but it does not yet
propagate a certified lattice uncertainty into metric-bin guards.  Every
envelope and matcher contract therefore states
``metric_uncertainty_certified=false`` and
``representation_invariant_physical_equivalence_claim=false``.  Exact byte
equality is an equivalence relation on accepted canonical partition payloads;
it is not claimed to prove representation-invariant physical equivalence.
"""

from __future__ import annotations

import json
import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from hashlib import sha256
from typing import NewType, Protocol, cast

import numpy as np
import numpy.typing as npt

from .catalogue import CertifiedCatalogue, SpaceGroupSetting
from .cover import canonical_rows
from .dataset_identity import IdentityPolicy
from .diagnostics import (
    BootstrapProtocol,
    DiagnosticEvaluationContract,
    EquivalenceMatcherContract,
    GeometryDecompositionCase,
    GeometryDecompositionDiagnostics,
    RankedRetrievalCase,
    ScaffoldRetrievalDiagnostics,
    evaluate_geometry_decomposition,
    evaluate_scaffold_retrieval,
)
from .errors import CatalogueError, DomainError

_PROVENANCE_SCHEMA = "crystalatlas.canonical-partition-provenance.v1"
_ENVELOPE_SCHEMA = "crystalatlas.canonical-partition-envelope.v1"
_IDENTITY_SCHEMA = "crystalatlas.canonical-structure-identity.v2"
_SEMANTICS = "strict-canonical-partition-membership-not-physical-similarity"
_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")

ValidatedCanonicalPartitionPayload = NewType("ValidatedCanonicalPartitionPayload", bytes)


class CanonicalPartitionLevel(str, Enum):  # noqa: UP042 - Python-3.10 ML smoke compatibility
    """The only canonical partitions exposed to diagnostic matchers."""

    STRUCTURE = "structure"
    SCAFFOLD = "scaffold"


class CanonicalIdentityPayload(Protocol):
    """Read-only identity fields shared by live and stored records."""

    @property
    def structure_key(self) -> str: ...

    @property
    def scaffold_key(self) -> str: ...

    @property
    def policy_fingerprint(self) -> str: ...

    @property
    def catalogue_fingerprint(self) -> str: ...

    @property
    def structure_payload(self) -> str: ...


def _require_digest(value: object, *, name: str) -> str:
    if not isinstance(value, str) or _SHA256_PATTERN.fullmatch(value) is None:
        raise DomainError(f"{name} must be a lowercase SHA-256 digest")
    return value


def _canonical_json_bytes(value: object) -> bytes:
    try:
        return json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("ascii")
    except (TypeError, ValueError, UnicodeEncodeError) as exc:
        raise DomainError("canonical partition payload is not finite strict JSON") from exc


def _reject_json_constant(value: str) -> object:
    raise ValueError(f"nonfinite JSON constant {value}")


def _parse_canonical_json_bytes(payload: bytes, *, name: str) -> dict[str, object]:
    if not isinstance(payload, bytes):
        raise TypeError(f"{name} must be immutable bytes")
    try:
        text = payload.decode("ascii")
        decoded = json.loads(text, parse_constant=_reject_json_constant)
    except (UnicodeDecodeError, ValueError, json.JSONDecodeError) as exc:
        raise DomainError(f"{name} must contain finite canonical ASCII JSON") from exc
    if not isinstance(decoded, dict):
        raise DomainError(f"{name} must contain one JSON object")
    result = cast(dict[str, object], decoded)
    if _canonical_json_bytes(result) != payload:
        raise DomainError(f"{name} is not canonically serialized")
    return result


def _parse_identity_structure_payload(payload: str) -> dict[str, object]:
    if not isinstance(payload, str):
        raise TypeError("canonical structure payload must be text")
    try:
        encoded = payload.encode("ascii")
    except UnicodeEncodeError as exc:
        raise DomainError("canonical structure payload must be ASCII") from exc
    return _parse_canonical_json_bytes(encoded, name="canonical structure payload")


def _require_exact_fields(
    value: Mapping[str, object], expected: frozenset[str], *, name: str
) -> None:
    observed = frozenset(value)
    if observed != expected:
        raise DomainError(
            f"{name} fields disagree with the frozen schema; "
            f"missing={sorted(expected - observed)}, extra={sorted(observed - expected)}"
        )


def _require_mapping(value: object, *, name: str) -> dict[str, object]:
    if not isinstance(value, dict) or any(not isinstance(key, str) for key in value):
        raise DomainError(f"{name} must be a JSON object with string keys")
    return cast(dict[str, object], value)


def _require_plain_integer(value: object, *, name: str, minimum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise DomainError(f"{name} must be an integer greater than or equal to {minimum}")
    return value


def _validate_composition(value: object) -> tuple[tuple[int, int], ...]:
    if not isinstance(value, list) or not value:
        raise DomainError("canonical scaffold composition must be a nonempty list")
    result: list[tuple[int, int]] = []
    for index, raw_entry in enumerate(value):
        if not isinstance(raw_entry, list) or len(raw_entry) != 2:
            raise DomainError(f"canonical composition entry {index} must be [Z,count]")
        atomic_number = _require_plain_integer(
            raw_entry[0], name=f"canonical composition atomic number {index}", minimum=1
        )
        if atomic_number > 118:
            raise DomainError("canonical composition atomic number exceeds 118")
        count = _require_plain_integer(
            raw_entry[1], name=f"canonical composition coefficient {index}", minimum=1
        )
        result.append((atomic_number, count))
    frozen = tuple(result)
    if frozen != tuple(sorted(frozen)) or len({value[0] for value in frozen}) != len(frozen):
        raise DomainError("canonical composition must be strictly atomic-number ordered")
    if math.gcd(*(count for _, count in frozen)) != 1:
        raise DomainError("canonical composition must be reduced")
    return frozen


def _validate_setting(
    value: object,
    *,
    catalogue: CertifiedCatalogue,
) -> SpaceGroupSetting:
    setting = _require_mapping(value, name="canonical scaffold setting")
    _require_exact_fields(
        setting,
        frozenset({"hall_symbol", "international_number", "key"}),
        name="canonical scaffold setting",
    )
    for field_name in ("hall_symbol", "key"):
        field_value = setting[field_name]
        if not isinstance(field_value, str) or not field_value:
            raise DomainError(f"canonical setting {field_name} must be nonempty text")
    number = _require_plain_integer(
        setting["international_number"],
        name="canonical setting international number",
        minimum=1,
    )
    if number > 230:
        raise DomainError("canonical setting international number exceeds 230")
    try:
        resolved = catalogue.get(cast(str, setting["key"]))
    except CatalogueError as exc:
        raise DomainError("canonical setting is absent from the certified catalogue") from exc
    if (
        resolved.hall_symbol != setting["hall_symbol"]
        or resolved.international_number != number
    ):
        raise DomainError("canonical setting disagrees with the pinned catalogue entry")
    return resolved


def _validate_live_contract(
    provenance: CanonicalPartitionProvenance,
    *,
    catalogue: CertifiedCatalogue,
    identity_policy: IdentityPolicy,
) -> None:
    if not isinstance(catalogue, CertifiedCatalogue):
        raise TypeError("catalogue must be a CertifiedCatalogue")
    if not isinstance(identity_policy, IdentityPolicy):
        raise TypeError("identity_policy must be an IdentityPolicy")
    if catalogue.fingerprint != provenance.catalogue_fingerprint:
        raise DomainError("live catalogue differs from canonical matcher provenance")
    if identity_policy.fingerprint != provenance.identity_policy_fingerprint:
        raise DomainError("live identity policy differs from canonical matcher provenance")
    catalogue.validate(require_all_230=identity_policy.require_all_230)


def _metric_from_bins(
    metric_bins: Sequence[int], *, policy: IdentityPolicy
) -> npt.NDArray[np.float64]:
    values = np.asarray(metric_bins, dtype=np.float64) * policy.metric_quantum_angstrom2
    metric = np.asarray(
        (
            (values[0], values[3], values[4]),
            (values[3], values[1], values[5]),
            (values[4], values[5], values[2]),
        ),
        dtype=np.float64,
    )
    if not np.all(np.isfinite(metric)):
        raise DomainError("canonical metric bins do not decode to a finite metric")
    eigenvalues = np.linalg.eigvalsh(metric)
    if not np.all(np.isfinite(eigenvalues)) or float(eigenvalues[0]) <= 0.0:
        raise DomainError("canonical metric bins do not decode to a positive-definite metric")
    return metric


def _validate_quantized_metric(
    metric_bins: object,
    *,
    setting: SpaceGroupSetting,
    policy: IdentityPolicy,
) -> None:
    if (
        not isinstance(metric_bins, list)
        or len(metric_bins) != 6
        or any(
            isinstance(bin_value, bool)
            or not isinstance(bin_value, int)
            or abs(bin_value) >= 1 << 51
            for bin_value in metric_bins
        )
    ):
        raise DomainError("canonical structure payload has malformed metric bins")
    integer_bins = cast(list[int], metric_bins)
    metric = _metric_from_bins(integer_bins, policy=policy)
    quantum = policy.metric_quantum_angstrom2
    floating_slack = 128.0 * np.finfo(np.float64).eps * max(
        float(np.linalg.norm(metric, ord=np.inf)), quantum
    )
    for rotation_exact in setting.point_group_rotations:
        rotation = np.asarray(rotation_exact, dtype=np.float64)
        residual = rotation @ metric @ rotation.T - metric
        row_norms = np.sum(np.abs(rotation), axis=1)
        for row in range(3):
            for column in range(3):
                # Every independently quantized symmetric component differs
                # from its physical value by strictly less than q/2.  This is
                # therefore a conservative entrywise bound on R E R^T - E.
                quantization_bound = 0.5 * quantum * (
                    float(row_norms[row] * row_norms[column]) + 1.0
                )
                if abs(float(residual[row, column])) > (
                    quantization_bound + floating_slack
                ):
                    raise DomainError(
                        "canonical metric bins violate the selected point-group metric "
                        "constraint beyond quantization error"
                    )


def _validate_scaffold(
    value: object,
    *,
    decorated: bool,
    catalogue: CertifiedCatalogue,
    identity_policy: IdentityPolicy,
) -> SpaceGroupSetting:
    scaffold = _require_mapping(value, name="canonical scaffold")
    _require_exact_fields(
        scaffold,
        frozenset({"composition", "nu", "orbits", "setting"}),
        name="canonical scaffold",
    )
    composition = _validate_composition(scaffold["composition"])
    composition_counts = dict(composition)
    nu = _require_plain_integer(scaffold["nu"], name="canonical scaffold nu", minimum=1)
    setting = _validate_setting(scaffold["setting"], catalogue=catalogue)
    row_rank = {row.label: index for index, row in enumerate(canonical_rows(setting))}
    orbits = scaffold["orbits"]
    if not isinstance(orbits, list) or not orbits:
        raise DomainError("canonical scaffold must contain a nonempty orbit list")
    expanded_counts = {atomic_number: 0 for atomic_number, _ in composition}
    expected_orbit_fields = (
        frozenset({"parameter_bins", "species", "wyckoff"})
        if decorated
        else frozenset({"species", "wyckoff"})
    )
    ordering_keys: list[tuple[object, ...]] = []
    for index, raw_orbit in enumerate(orbits):
        orbit = _require_mapping(raw_orbit, name=f"canonical scaffold orbit {index}")
        _require_exact_fields(
            orbit,
            expected_orbit_fields,
            name=f"canonical scaffold orbit {index}",
        )
        species = _require_plain_integer(
            orbit["species"], name=f"canonical orbit species {index}", minimum=1
        )
        if species not in composition_counts:
            raise DomainError("canonical orbit species is absent from the composition")
        label = orbit["wyckoff"]
        if not isinstance(label, str):
            raise DomainError("canonical orbit Wyckoff label must be text")
        try:
            row = setting.wyckoff(label)
        except CatalogueError as exc:
            raise DomainError(
                "canonical orbit Wyckoff row is absent from the pinned setting"
            ) from exc
        expanded_counts[species] += row.multiplicity
        if decorated:
            parameter_bins = orbit["parameter_bins"]
            if not isinstance(parameter_bins, list) or any(
                isinstance(bin_value, bool)
                or not isinstance(bin_value, int)
                or bin_value < 0
                or bin_value >= identity_policy.fractional_bin_count
                for bin_value in parameter_bins
            ):
                raise DomainError("canonical orbit has malformed parameter bins")
            if len(parameter_bins) != row.dimension:
                raise DomainError(
                    "canonical orbit parameter-bin arity disagrees with its Wyckoff row"
                )
            ordering_keys.append((row_rank[label], species, tuple(parameter_bins)))
        else:
            ordering_keys.append((row_rank[label], species))
    if ordering_keys != sorted(ordering_keys):
        raise DomainError("canonical scaffold orbits are not in canonical row order")
    expected_counts = {
        atomic_number: nu * count for atomic_number, count in composition
    }
    if expanded_counts != expected_counts:
        raise DomainError("canonical orbit multiplicities violate the formula equation")
    return setting


def _validate_inner_partition(
    payload: Mapping[str, object],
    *,
    level: CanonicalPartitionLevel,
    catalogue: CertifiedCatalogue,
    identity_policy: IdentityPolicy,
) -> None:
    expected = (
        frozenset(
            {
                "catalogue_fingerprint",
                "chirality_quotient",
                "coordinate_convention",
                "metric_bins_angstrom2",
                "policy_fingerprint",
                "scaffold",
                "schema",
            }
        )
        if level is CanonicalPartitionLevel.STRUCTURE
        else frozenset(
            {
                "catalogue_fingerprint",
                "chirality_quotient",
                "coordinate_convention",
                "scaffold",
                "schema",
            }
        )
    )
    _require_exact_fields(payload, expected, name=f"canonical {level.value} payload")
    if payload["schema"] != _IDENTITY_SCHEMA:
        raise DomainError("canonical partition payload has an unknown identity schema")
    if payload["chirality_quotient"] != "SO(3)":
        raise DomainError("canonical partition payload changed the proper-rotation quotient")
    if payload["coordinate_convention"] != "fractional-row-v1":
        raise DomainError("canonical partition payload changed its coordinate convention")
    if payload["catalogue_fingerprint"] != catalogue.fingerprint:
        raise DomainError("canonical partition payload uses another catalogue")
    if level is CanonicalPartitionLevel.STRUCTURE:
        if payload["policy_fingerprint"] != identity_policy.fingerprint:
            raise DomainError("canonical structure payload uses another identity policy")
    setting = _validate_scaffold(
        payload["scaffold"],
        decorated=level is CanonicalPartitionLevel.STRUCTURE,
        catalogue=catalogue,
        identity_policy=identity_policy,
    )
    if level is CanonicalPartitionLevel.STRUCTURE:
        _validate_quantized_metric(
            payload["metric_bins_angstrom2"],
            setting=setting,
            policy=identity_policy,
        )


def _validate_structure_payload(
    identity: CanonicalIdentityPayload,
    *,
    catalogue: CertifiedCatalogue,
    identity_policy: IdentityPolicy,
) -> tuple[dict[str, object], dict[str, object]]:
    structure = _parse_identity_structure_payload(identity.structure_payload)
    _validate_inner_partition(
        structure,
        level=CanonicalPartitionLevel.STRUCTURE,
        catalogue=catalogue,
        identity_policy=identity_policy,
    )
    _require_exact_fields(
        structure,
        frozenset(
            {
                "catalogue_fingerprint",
                "chirality_quotient",
                "coordinate_convention",
                "metric_bins_angstrom2",
                "policy_fingerprint",
                "scaffold",
                "schema",
            }
        ),
        name="canonical structure payload",
    )
    if structure["schema"] != _IDENTITY_SCHEMA:
        raise DomainError("canonical structure payload has an unknown identity schema")
    if structure["chirality_quotient"] != "SO(3)":
        raise DomainError("canonical structure payload changed the proper-rotation quotient")
    if structure["coordinate_convention"] != "fractional-row-v1":
        raise DomainError("canonical structure payload changed its coordinate convention")
    if structure["catalogue_fingerprint"] != identity.catalogue_fingerprint:
        raise DomainError("canonical structure payload names a different catalogue")
    if structure["policy_fingerprint"] != identity.policy_fingerprint:
        raise DomainError("canonical structure payload names a different identity policy")

    metric_bins = structure["metric_bins_angstrom2"]
    if (
        not isinstance(metric_bins, list)
        or len(metric_bins) != 6
        or any(isinstance(value, bool) or not isinstance(value, int) for value in metric_bins)
    ):
        raise DomainError("canonical structure payload has malformed metric bins")

    scaffold = _require_mapping(structure["scaffold"], name="decorated scaffold")
    _require_exact_fields(
        scaffold,
        frozenset({"composition", "nu", "orbits", "setting"}),
        name="decorated scaffold",
    )
    orbits = scaffold["orbits"]
    if not isinstance(orbits, list) or not orbits:
        raise DomainError("decorated scaffold must contain a nonempty orbit list")
    discrete_orbits: list[dict[str, object]] = []
    for index, raw_orbit in enumerate(orbits):
        orbit = _require_mapping(raw_orbit, name=f"decorated scaffold orbit {index}")
        _require_exact_fields(
            orbit,
            frozenset({"parameter_bins", "species", "wyckoff"}),
            name=f"decorated scaffold orbit {index}",
        )
        parameter_bins = orbit["parameter_bins"]
        if not isinstance(parameter_bins, list) or any(
            isinstance(value, bool) or not isinstance(value, int) for value in parameter_bins
        ):
            raise DomainError("decorated scaffold has malformed parameter bins")
        discrete_orbits.append({"species": orbit["species"], "wyckoff": orbit["wyckoff"]})

    structure_bytes = identity.structure_payload.encode("ascii")
    if sha256(structure_bytes).hexdigest() != identity.structure_key:
        raise DomainError("canonical structure key disagrees with its full payload")

    discrete_scaffold: dict[str, object] = {
        "composition": scaffold["composition"],
        "nu": scaffold["nu"],
        "orbits": discrete_orbits,
        "setting": scaffold["setting"],
    }
    scaffold_payload: dict[str, object] = {
        "catalogue_fingerprint": identity.catalogue_fingerprint,
        "chirality_quotient": "SO(3)",
        "coordinate_convention": "fractional-row-v1",
        "scaffold": discrete_scaffold,
        "schema": _IDENTITY_SCHEMA,
    }
    _validate_inner_partition(
        scaffold_payload,
        level=CanonicalPartitionLevel.SCAFFOLD,
        catalogue=catalogue,
        identity_policy=identity_policy,
    )
    if sha256(_canonical_json_bytes(scaffold_payload)).hexdigest() != identity.scaffold_key:
        raise DomainError("canonical scaffold key disagrees with its derived full payload")
    return structure, scaffold_payload


@dataclass(frozen=True, slots=True)
class CanonicalPartitionProvenance:
    """Artifact and scientific-policy identity for one diagnostic partition.

    ``dataset_manifest_sha256`` is the authenticated dataset-manifest artifact
    hash, not merely a source label.  The other artifact fields are likewise
    supplied by the experiment harness; this class validates their shape and
    binds them into every envelope, but does not discover or trust files by
    path.
    """

    dataset_manifest_sha256: str
    target_builder_fingerprint: str
    crystallography_receipt_fingerprint: str
    catalogue_fingerprint: str
    identity_policy_fingerprint: str
    crystalatlas_artifact_sha256: str
    crystallography_backend_artifact_sha256: str
    environment_sha256: str

    def __post_init__(self) -> None:
        for name in (
            "dataset_manifest_sha256",
            "target_builder_fingerprint",
            "crystallography_receipt_fingerprint",
            "catalogue_fingerprint",
            "identity_policy_fingerprint",
            "crystalatlas_artifact_sha256",
            "crystallography_backend_artifact_sha256",
            "environment_sha256",
        ):
            object.__setattr__(
                self,
                name,
                _require_digest(getattr(self, name), name=name),
            )

    def _payload(self) -> dict[str, object]:
        return {
            "catalogue_fingerprint": self.catalogue_fingerprint,
            "crystallography_backend_artifact_sha256": (
                self.crystallography_backend_artifact_sha256
            ),
            "crystallography_receipt_fingerprint": (self.crystallography_receipt_fingerprint),
            "crystalatlas_artifact_sha256": self.crystalatlas_artifact_sha256,
            "dataset_manifest_sha256": self.dataset_manifest_sha256,
            "environment_sha256": self.environment_sha256,
            "identity_policy_fingerprint": self.identity_policy_fingerprint,
            "metric_uncertainty_certified": False,
            "representation_invariant_physical_equivalence_claim": False,
            "schema": _PROVENANCE_SCHEMA,
            "semantics": _SEMANTICS,
            "target_builder_fingerprint": self.target_builder_fingerprint,
        }

    @property
    def fingerprint(self) -> str:
        return sha256(_canonical_json_bytes(self._payload())).hexdigest()

    def to_manifest(self) -> dict[str, object]:
        return {**self._payload(), "fingerprint": self.fingerprint}

    def require_identity(self, identity: CanonicalIdentityPayload) -> None:
        """Reject an identity produced under another policy or catalogue."""

        if identity.catalogue_fingerprint != self.catalogue_fingerprint:
            raise DomainError("canonical identity catalogue differs from matcher provenance")
        if identity.policy_fingerprint != self.identity_policy_fingerprint:
            raise DomainError("canonical identity policy differs from matcher provenance")


def _validate_provenance_manifest(value: object) -> dict[str, object]:
    manifest = _require_mapping(value, name="canonical partition provenance")
    expected = frozenset(
        {
            "catalogue_fingerprint",
            "crystallography_backend_artifact_sha256",
            "crystallography_receipt_fingerprint",
            "crystalatlas_artifact_sha256",
            "dataset_manifest_sha256",
            "environment_sha256",
            "fingerprint",
            "identity_policy_fingerprint",
            "metric_uncertainty_certified",
            "representation_invariant_physical_equivalence_claim",
            "schema",
            "semantics",
            "target_builder_fingerprint",
        }
    )
    _require_exact_fields(manifest, expected, name="canonical partition provenance")
    if manifest["schema"] != _PROVENANCE_SCHEMA or manifest["semantics"] != _SEMANTICS:
        raise DomainError("canonical partition provenance has unknown semantics")
    if manifest["metric_uncertainty_certified"] is not False:
        raise DomainError("canonical partition provenance overclaims metric uncertainty")
    if manifest["representation_invariant_physical_equivalence_claim"] is not False:
        raise DomainError("canonical partition provenance overclaims physical equivalence")
    for name in (
        "catalogue_fingerprint",
        "crystallography_backend_artifact_sha256",
        "crystallography_receipt_fingerprint",
        "crystalatlas_artifact_sha256",
        "dataset_manifest_sha256",
        "environment_sha256",
        "identity_policy_fingerprint",
        "target_builder_fingerprint",
    ):
        _require_digest(manifest[name], name=name)
    fingerprint = _require_digest(manifest["fingerprint"], name="provenance fingerprint")
    payload = {key: item for key, item in manifest.items() if key != "fingerprint"}
    if sha256(_canonical_json_bytes(payload)).hexdigest() != fingerprint:
        raise DomainError("canonical partition provenance fingerprint is inconsistent")
    return manifest


def _canonical_partition_payload(
    identity: CanonicalIdentityPayload,
    provenance: CanonicalPartitionProvenance,
    *,
    level: CanonicalPartitionLevel,
    catalogue: CertifiedCatalogue,
    identity_policy: IdentityPolicy,
) -> ValidatedCanonicalPartitionPayload:
    if not isinstance(provenance, CanonicalPartitionProvenance):
        raise TypeError("provenance must be CanonicalPartitionProvenance")
    if not isinstance(level, CanonicalPartitionLevel):
        raise TypeError("level must be CanonicalPartitionLevel")
    _validate_live_contract(
        provenance,
        catalogue=catalogue,
        identity_policy=identity_policy,
    )
    provenance.require_identity(identity)
    structure, scaffold = _validate_structure_payload(
        identity,
        catalogue=catalogue,
        identity_policy=identity_policy,
    )
    if level is CanonicalPartitionLevel.STRUCTURE:
        canonical_payload = structure
        canonical_payload_sha256 = identity.structure_key
    else:
        canonical_payload = scaffold
        canonical_payload_sha256 = identity.scaffold_key
    envelope = {
        "canonical_payload": canonical_payload,
        "canonical_payload_sha256": canonical_payload_sha256,
        "identity_level": level.value,
        "metric_uncertainty_certified": False,
        "provenance": provenance.to_manifest(),
        "representation_invariant_physical_equivalence_claim": False,
        "schema": _ENVELOPE_SCHEMA,
        "semantics": _SEMANTICS,
    }
    encoded = _canonical_json_bytes(envelope)
    return validate_canonical_partition_payload(
        encoded,
        provenance=provenance,
        level=level,
        catalogue=catalogue,
        identity_policy=identity_policy,
    )


def encode_structure_partition(
    identity: CanonicalIdentityPayload,
    provenance: CanonicalPartitionProvenance,
    *,
    catalogue: CertifiedCatalogue,
    identity_policy: IdentityPolicy,
) -> ValidatedCanonicalPartitionPayload:
    """Envelope the complete canonical ``STRUCTURE`` payload as immutable bytes."""

    return _canonical_partition_payload(
        identity,
        provenance,
        level=CanonicalPartitionLevel.STRUCTURE,
        catalogue=catalogue,
        identity_policy=identity_policy,
    )


def encode_scaffold_partition(
    identity: CanonicalIdentityPayload,
    provenance: CanonicalPartitionProvenance,
    *,
    catalogue: CertifiedCatalogue,
    identity_policy: IdentityPolicy,
) -> ValidatedCanonicalPartitionPayload:
    """Derive, verify, and envelope the complete discrete scaffold payload."""

    return _canonical_partition_payload(
        identity,
        provenance,
        level=CanonicalPartitionLevel.SCAFFOLD,
        catalogue=catalogue,
        identity_policy=identity_policy,
    )


def validate_canonical_partition_payload(
    payload: bytes,
    *,
    provenance: CanonicalPartitionProvenance,
    level: CanonicalPartitionLevel,
    catalogue: CertifiedCatalogue,
    identity_policy: IdentityPolicy,
) -> ValidatedCanonicalPartitionPayload:
    """Validate an envelope against the frozen experiment provenance.

    Call this on every archived target payload and every generated-attempt
    payload before constructing diagnostic cases.  A provenance mismatch is a
    protocol error and raises; it is never counted as a model miss.
    """

    if not isinstance(provenance, CanonicalPartitionProvenance):
        raise TypeError("provenance must be CanonicalPartitionProvenance")
    if not isinstance(level, CanonicalPartitionLevel):
        raise TypeError("level must be CanonicalPartitionLevel")
    _validate_live_contract(
        provenance,
        catalogue=catalogue,
        identity_policy=identity_policy,
    )
    envelope = _parse_canonical_json_bytes(payload, name="canonical partition envelope")
    _require_exact_fields(
        envelope,
        frozenset(
            {
                "canonical_payload",
                "canonical_payload_sha256",
                "identity_level",
                "metric_uncertainty_certified",
                "provenance",
                "representation_invariant_physical_equivalence_claim",
                "schema",
                "semantics",
            }
        ),
        name="canonical partition envelope",
    )
    if envelope["schema"] != _ENVELOPE_SCHEMA or envelope["semantics"] != _SEMANTICS:
        raise DomainError("canonical partition envelope has unknown semantics")
    if envelope["identity_level"] != level.value:
        raise DomainError("canonical partition envelope uses another identity level")
    if envelope["metric_uncertainty_certified"] is not False:
        raise DomainError("metric uncertainty is not certified by this schema version")
    if envelope["representation_invariant_physical_equivalence_claim"] is not False:
        raise DomainError("this partition cannot claim physical structure equivalence")
    embedded_provenance = _validate_provenance_manifest(envelope["provenance"])
    if embedded_provenance != provenance.to_manifest():
        raise DomainError("canonical partition envelope provenance differs from the contract")

    canonical_payload = _require_mapping(
        envelope["canonical_payload"], name="enveloped canonical payload"
    )
    declared = _require_digest(
        envelope["canonical_payload_sha256"], name="canonical payload fingerprint"
    )
    if sha256(_canonical_json_bytes(canonical_payload)).hexdigest() != declared:
        raise DomainError("enveloped canonical payload disagrees with its fingerprint")
    _validate_inner_partition(
        canonical_payload,
        level=level,
        catalogue=catalogue,
        identity_policy=identity_policy,
    )
    return ValidatedCanonicalPartitionPayload(payload)


def _validate_retrieval_cases(
    cases: Sequence[RankedRetrievalCase[bytes]],
    *,
    provenance: CanonicalPartitionProvenance,
    level: CanonicalPartitionLevel,
    catalogue: CertifiedCatalogue,
    identity_policy: IdentityPolicy,
) -> tuple[RankedRetrievalCase[bytes], ...]:
    materialized = tuple(cases)
    for case in materialized:
        for target in case.acceptable_targets:
            validate_canonical_partition_payload(
                target.payload,
                provenance=provenance,
                level=level,
                catalogue=catalogue,
                identity_policy=identity_policy,
            )
        for attempt in case.attempts:
            if attempt.payload is not None:
                validate_canonical_partition_payload(
                    attempt.payload,
                    provenance=provenance,
                    level=level,
                    catalogue=catalogue,
                    identity_policy=identity_policy,
                )
    return materialized


def _validate_geometry_cases(
    cases: Sequence[GeometryDecompositionCase[bytes]],
    *,
    provenance: CanonicalPartitionProvenance,
    catalogue: CertifiedCatalogue,
    identity_policy: IdentityPolicy,
) -> tuple[GeometryDecompositionCase[bytes], ...]:
    materialized = tuple(cases)
    for case in materialized:
        for target in case.acceptable_targets:
            validate_canonical_partition_payload(
                target.payload,
                provenance=provenance,
                level=CanonicalPartitionLevel.STRUCTURE,
                catalogue=catalogue,
                identity_policy=identity_policy,
            )
        for attempts in (
            case.oracle_scaffold_attempts,
            case.predicted_scaffold_attempts,
        ):
            for attempt in attempts:
                if attempt.payload is not None:
                    validate_canonical_partition_payload(
                        attempt.payload,
                        provenance=provenance,
                        level=CanonicalPartitionLevel.STRUCTURE,
                        catalogue=catalogue,
                        identity_policy=identity_policy,
                    )
    return materialized


def _require_evaluation_provenance(
    evaluation_contract: DiagnosticEvaluationContract,
    provenance: CanonicalPartitionProvenance,
) -> None:
    if not isinstance(evaluation_contract, DiagnosticEvaluationContract):
        raise TypeError("evaluation_contract must be a DiagnosticEvaluationContract")
    if evaluation_contract.dataset_manifest_sha256 != provenance.dataset_manifest_sha256:
        raise DomainError(
            "diagnostic dataset manifest differs from canonical matcher provenance"
        )


def evaluate_canonical_scaffold_retrieval(
    cases: Sequence[RankedRetrievalCase[bytes]],
    *,
    ks: Sequence[int],
    provenance: CanonicalPartitionProvenance,
    catalogue: CertifiedCatalogue,
    identity_policy: IdentityPolicy,
    evaluation_contract: DiagnosticEvaluationContract,
    bootstrap: BootstrapProtocol | None = None,
) -> ScaffoldRetrievalDiagnostics:
    """Evaluate strict scaffold partitions only after mandatory prevalidation."""

    _require_evaluation_provenance(evaluation_contract, provenance)
    _validate_live_contract(
        provenance,
        catalogue=catalogue,
        identity_policy=identity_policy,
    )
    materialized = _validate_retrieval_cases(
        cases,
        provenance=provenance,
        level=CanonicalPartitionLevel.SCAFFOLD,
        catalogue=catalogue,
        identity_policy=identity_policy,
    )
    return evaluate_scaffold_retrieval(
        materialized,
        ks=ks,
        matcher=canonical_scaffold_partition_equal,
        matcher_contract=scaffold_partition_matcher_contract(provenance),
        evaluation_contract=evaluation_contract,
        bootstrap=bootstrap,
    )


def evaluate_canonical_geometry_decomposition(
    cases: Sequence[GeometryDecompositionCase[bytes]],
    *,
    ks: Sequence[int],
    provenance: CanonicalPartitionProvenance,
    catalogue: CertifiedCatalogue,
    identity_policy: IdentityPolicy,
    evaluation_contract: DiagnosticEvaluationContract,
    bootstrap: BootstrapProtocol | None = None,
) -> GeometryDecompositionDiagnostics:
    """Evaluate strict structure partitions only after mandatory prevalidation."""

    _require_evaluation_provenance(evaluation_contract, provenance)
    _validate_live_contract(
        provenance,
        catalogue=catalogue,
        identity_policy=identity_policy,
    )
    materialized = _validate_geometry_cases(
        cases,
        provenance=provenance,
        catalogue=catalogue,
        identity_policy=identity_policy,
    )
    return evaluate_geometry_decomposition(
        materialized,
        ks=ks,
        matcher=canonical_structure_partition_equal,
        matcher_contract=structure_partition_matcher_contract(provenance),
        evaluation_contract=evaluation_contract,
        bootstrap=bootstrap,
    )


def canonical_structure_partition_equal(candidate: bytes, target: bytes) -> bool:
    """Return exact equality of two *prevalidated* structure envelopes."""

    return candidate == target


def canonical_scaffold_partition_equal(candidate: bytes, target: bytes) -> bool:
    """Return exact equality of two *prevalidated* scaffold envelopes."""

    return candidate == target


def structure_partition_matcher_contract(
    provenance: CanonicalPartitionProvenance,
) -> EquivalenceMatcherContract:
    """Build the diagnostic contract for strict structure-partition equality."""

    return EquivalenceMatcherContract.from_callable(
        canonical_structure_partition_equal,
        algorithm="crystalatlas-strict-canonical-structure-partition-equality",
        version="1",
        cell_policy=(
            "precomputed-fullprim-pinned-hall-canonical-origin-wyckoff-"
            "guarded-structure-partition"
        ),
        parameters={
            "comparison": "exact-immutable-byte-equality",
            "identity_level": CanonicalPartitionLevel.STRUCTURE.value,
            "metric_uncertainty_certified": False,
            "provenance": provenance.to_manifest(),
            "representation_invariant_physical_equivalence_claim": False,
            "semantics": _SEMANTICS,
        },
    )


def scaffold_partition_matcher_contract(
    provenance: CanonicalPartitionProvenance,
) -> EquivalenceMatcherContract:
    """Build the separate contract for strict discrete-scaffold equality."""

    return EquivalenceMatcherContract.from_callable(
        canonical_scaffold_partition_equal,
        algorithm="crystalatlas-strict-canonical-scaffold-partition-equality",
        version="1",
        cell_policy="precomputed-maximal-pinned-hall-discrete-scaffold-partition",
        parameters={
            "comparison": "exact-immutable-byte-equality",
            "identity_level": CanonicalPartitionLevel.SCAFFOLD.value,
            "metric_uncertainty_certified": False,
            "provenance": provenance.to_manifest(),
            "representation_invariant_physical_equivalence_claim": False,
            "semantics": _SEMANTICS,
        },
    )


__all__ = [
    "CanonicalIdentityPayload",
    "CanonicalPartitionLevel",
    "CanonicalPartitionProvenance",
    "ValidatedCanonicalPartitionPayload",
    "canonical_scaffold_partition_equal",
    "canonical_structure_partition_equal",
    "encode_scaffold_partition",
    "encode_structure_partition",
    "evaluate_canonical_geometry_decomposition",
    "evaluate_canonical_scaffold_retrieval",
    "scaffold_partition_matcher_contract",
    "structure_partition_matcher_contract",
    "validate_canonical_partition_payload",
]
