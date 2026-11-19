"""Executable crystallographic proof gate for model training.

Training is allowed only after the exact catalogue used by the model has
passed the same finite-algebra obligations used by decoding.  The receipt in
this module is not a declaration of chemical accuracy: it certifies catalogue
identity, finite-group closure, orbit multiplicities, symmetry containment,
continuous-origin chart compatibility, and exact decoder round trips.
"""

from __future__ import annotations

import importlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from types import MappingProxyType
from typing import Any

import numpy as np

from .algebra import validate_finite_affine_group
from .catalogue import CertifiedCatalogue
from .decoder import CrystalScaffold, ExactCrystalDecoder, OrbitAssignment
from .errors import CatalogueError, DomainError

_SCHEMA = "crystalatlas.crystallography-certification.v3"


def _canonical_fingerprint(value: Mapping[str, Any]) -> str:
    payload = json.dumps(
        value,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return sha256(payload.encode("utf-8")).hexdigest()


@dataclass(frozen=True, slots=True)
class CrystallographyCertification:
    """Immutable receipt produced only after the executable proof gate passes."""

    catalogue_fingerprint: str
    certification_scope: str
    complete_space_group_types: bool
    space_group_count: int
    wyckoff_row_count: int
    affine_operation_count: int
    pinned_hall_manifest_fingerprint: str | None
    source_identity: Mapping[str, Any]
    exact_manifest_bijection_verified: bool
    exact_group_closure_verified: bool
    exact_multiplicity_verified: bool
    exact_symmetry_verified: bool
    exact_origin_chart_verified: bool
    exact_decoder_roundtrip_verified: bool
    pinned_standardization_preflight_verified: bool
    external_operation_database_verified: bool
    receipt_fingerprint: str

    def __post_init__(self) -> None:
        if len(self.catalogue_fingerprint) != 64 or len(self.receipt_fingerprint) != 64:
            raise DomainError("certification fingerprints must be SHA-256 digests")
        try:
            int(self.catalogue_fingerprint, 16)
            int(self.receipt_fingerprint, 16)
        except ValueError as exc:
            raise DomainError("certification fingerprints must be hexadecimal") from exc
        if not self.certification_scope:
            raise DomainError("certification scope cannot be empty")
        counts = (
            self.space_group_count,
            self.wyckoff_row_count,
            self.affine_operation_count,
        )
        if any(
            isinstance(value, bool) or not isinstance(value, int) or value < 1
            for value in counts
        ):
            raise DomainError("certification counts must be positive integers")
        if self.pinned_hall_manifest_fingerprint is not None:
            if len(self.pinned_hall_manifest_fingerprint) != 64:
                raise DomainError("Hall-manifest fingerprint must be SHA-256")
            try:
                int(self.pinned_hall_manifest_fingerprint, 16)
            except ValueError as exc:
                raise DomainError("Hall-manifest fingerprint must be hexadecimal") from exc
        source = dict(self.source_identity)
        if not source:
            raise DomainError("crystallographic source identity cannot be empty")
        object.__setattr__(self, "source_identity", MappingProxyType(source))
        booleans = (
            self.complete_space_group_types,
            self.exact_manifest_bijection_verified,
            self.exact_group_closure_verified,
            self.exact_multiplicity_verified,
            self.exact_symmetry_verified,
            self.exact_origin_chart_verified,
            self.exact_decoder_roundtrip_verified,
            self.pinned_standardization_preflight_verified,
            self.external_operation_database_verified,
        )
        if any(not isinstance(value, bool) for value in booleans):
            raise DomainError("certification proof flags must be booleans")
        if self.receipt_fingerprint != _canonical_fingerprint(self._unsigned_payload()):
            raise DomainError(
                "crystallography certification receipt is internally inconsistent"
            )

    @property
    def passed(self) -> bool:
        """Whether every exact, backend-independent proof obligation passed."""

        return all(
            (
                self.exact_manifest_bijection_verified,
                self.exact_group_closure_verified,
                self.exact_multiplicity_verified,
                self.exact_symmetry_verified,
                self.exact_origin_chart_verified,
                self.exact_decoder_roundtrip_verified,
            )
        )

    def _unsigned_payload(self) -> dict[str, Any]:
        return {
            "schema": _SCHEMA,
            "catalogue_fingerprint": self.catalogue_fingerprint,
            "certification_scope": self.certification_scope,
            "complete_space_group_types": self.complete_space_group_types,
            "space_group_count": self.space_group_count,
            "wyckoff_row_count": self.wyckoff_row_count,
            "affine_operation_count": self.affine_operation_count,
            "pinned_hall_manifest_fingerprint": self.pinned_hall_manifest_fingerprint,
            "source_identity": dict(self.source_identity),
            "exact_manifest_bijection_verified": self.exact_manifest_bijection_verified,
            "exact_group_closure_verified": self.exact_group_closure_verified,
            "exact_multiplicity_verified": self.exact_multiplicity_verified,
            "exact_symmetry_verified": self.exact_symmetry_verified,
            "exact_origin_chart_verified": self.exact_origin_chart_verified,
            "exact_decoder_roundtrip_verified": self.exact_decoder_roundtrip_verified,
            "pinned_standardization_preflight_verified": (
                self.pinned_standardization_preflight_verified
            ),
            "external_operation_database_verified": (
                self.external_operation_database_verified
            ),
        }

    def to_dict(self) -> dict[str, Any]:
        payload = self._unsigned_payload()
        payload["receipt_fingerprint"] = self.receipt_fingerprint
        return payload

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> CrystallographyCertification:
        """Load a receipt without weakening any identity or proof check.

        The receipt fingerprint is an integrity checksum, not a cryptographic
        signature.  ``require_for`` must still be called with the live catalogue
        before training; that binds the serialized evidence to the exact tables
        actually used by the model.
        """

        if not isinstance(payload, Mapping):
            raise TypeError("crystallography receipt payload must be a mapping")
        expected = {
            "schema",
            "catalogue_fingerprint",
            "certification_scope",
            "complete_space_group_types",
            "space_group_count",
            "wyckoff_row_count",
            "affine_operation_count",
            "pinned_hall_manifest_fingerprint",
            "source_identity",
            "exact_manifest_bijection_verified",
            "exact_group_closure_verified",
            "exact_multiplicity_verified",
            "exact_symmetry_verified",
            "exact_origin_chart_verified",
            "exact_decoder_roundtrip_verified",
            "pinned_standardization_preflight_verified",
            "external_operation_database_verified",
            "receipt_fingerprint",
        }
        observed = set(payload)
        if observed != expected:
            missing = sorted(expected.difference(observed))
            extra = sorted(observed.difference(expected))
            raise DomainError(
                f"crystallography receipt fields disagree; missing={missing}, extra={extra}"
            )
        if payload["schema"] != _SCHEMA:
            raise DomainError("unsupported crystallography receipt schema")
        source = payload["source_identity"]
        if not isinstance(source, Mapping):
            raise DomainError("crystallography source identity must be a mapping")
        return cls(
            catalogue_fingerprint=payload["catalogue_fingerprint"],
            certification_scope=payload["certification_scope"],
            complete_space_group_types=payload["complete_space_group_types"],
            space_group_count=payload["space_group_count"],
            wyckoff_row_count=payload["wyckoff_row_count"],
            affine_operation_count=payload["affine_operation_count"],
            pinned_hall_manifest_fingerprint=(
                payload["pinned_hall_manifest_fingerprint"]
            ),
            source_identity=source,
            exact_manifest_bijection_verified=(
                payload["exact_manifest_bijection_verified"]
            ),
            exact_group_closure_verified=payload["exact_group_closure_verified"],
            exact_multiplicity_verified=payload["exact_multiplicity_verified"],
            exact_symmetry_verified=payload["exact_symmetry_verified"],
            exact_origin_chart_verified=payload["exact_origin_chart_verified"],
            exact_decoder_roundtrip_verified=(
                payload["exact_decoder_roundtrip_verified"]
            ),
            pinned_standardization_preflight_verified=(
                payload["pinned_standardization_preflight_verified"]
            ),
            external_operation_database_verified=(
                payload["external_operation_database_verified"]
            ),
            receipt_fingerprint=payload["receipt_fingerprint"],
        )

    @classmethod
    def read_json(cls, path: str | Path) -> CrystallographyCertification:
        """Read one strict JSON receipt from disk and verify its checksum."""

        source = Path(path)
        try:
            payload = json.loads(source.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise DomainError(f"cannot read crystallography receipt {source}") from exc
        if not isinstance(payload, Mapping):
            raise DomainError("crystallography receipt JSON must contain one object")
        return cls.from_dict(payload)

    def require_for(
        self,
        catalogue: CertifiedCatalogue,
        *,
        require_all_230: bool = True,
        require_external_operation_database: bool = True,
    ) -> None:
        """Fail closed unless this receipt certifies the live catalogue."""

        if not isinstance(catalogue, CertifiedCatalogue):
            raise TypeError("catalogue must be a CertifiedCatalogue")
        if not self.passed:
            raise CatalogueError("crystallography certification proof gates did not pass")
        if catalogue.fingerprint != self.catalogue_fingerprint:
            raise CatalogueError("crystallography receipt belongs to a different catalogue")
        if catalogue.certification_scope != self.certification_scope:
            raise CatalogueError("crystallography receipt has a different catalogue scope")
        if require_all_230 and (
            not self.complete_space_group_types or self.space_group_count != 230
        ):
            raise CatalogueError("training requires a certified all-230 catalogue")
        if require_all_230 and not self.pinned_standardization_preflight_verified:
            raise CatalogueError(
                "training requires the all-230 pinned-setting/origin preflight"
            )
        if (
            require_external_operation_database
            and not self.external_operation_database_verified
        ):
            raise CatalogueError(
                "training requires the pinned external operation-database cross-check"
            )


def _full_source_metadata(
    catalogue: CertifiedCatalogue,
) -> tuple[Mapping[str, Any], str | None]:
    """Bind the complete compiler output to its checksummed upstream source."""

    if not catalogue.complete_space_group_types:
        return (
            MappingProxyType(
                {
                    "schema": "crystalatlas.in-memory-exact-fixture.v1",
                    "scope": catalogue.certification_scope,
                }
            ),
            None,
        )
    from .full_catalogue import (
        build_full_catalogue,
        pinned_hall_numbers,
        source_identity,
    )

    authoritative = build_full_catalogue()
    if catalogue.fingerprint != authoritative.fingerprint:
        raise CatalogueError(
            "complete catalogue is not the compiler output of the pinned source corpus"
        )
    hall_payload = {
        str(number): hall for number, hall in sorted(pinned_hall_numbers().items())
    }
    return MappingProxyType(source_identity()), _canonical_fingerprint(hall_payload)


def certify_crystallography(
    catalogue: CertifiedCatalogue,
    *,
    require_all_230: bool = True,
    crosscheck_external_operations: bool = True,
    spglib_module: Any | None = None,
) -> CrystallographyCertification:
    """Execute every crystallographic gate and return a bound receipt.

    For the complete catalogue this also verifies the frozen Hall manifest and,
    by default, independently compares every affine operation table with the
    pinned spglib 2.7.0 C database.  Fixture certification can disable that
    external comparison explicitly for equation-level unit tests.
    """

    if not isinstance(catalogue, CertifiedCatalogue):
        raise TypeError("catalogue must be a CertifiedCatalogue")
    catalogue.validate(require_all_230=require_all_230)
    source, hall_fingerprint = _full_source_metadata(catalogue)
    from .standardization import validate_origin_chart_compatibility

    origin_axes = validate_origin_chart_compatibility(
        catalogue,
        require_all_230=require_all_230,
    )
    from .geometry import ContinuousCover, IntrinsicCoverLayout

    decoder = ExactCrystalDecoder(catalogue)
    group_count = len(catalogue.settings)
    row_count = 0
    operation_count = 0
    for setting in catalogue.settings.values():
        validate_finite_affine_group(setting.operations)
        operation_count += len(setting.operations)
        metric = np.asarray(
            [[float(value) for value in row] for row in setting.reference_metric],
            dtype=np.float64,
        )
        for row in setting.wyckoff_positions.values():
            row_count += 1
            expected = row.expand_exact(
                row.generic_parameter,
                require_declared_multiplicity=True,
            )
            if len(expected) != row.multiplicity:
                raise CatalogueError(
                    f"{setting.key}:{row.label} failed exact multiplicity certification"
                )
            expected_set = set(expected)
            if any(
                {operation.apply_exact(point) for point in expected} != expected_set
                for operation in setting.operations
            ):
                raise CatalogueError(
                    f"{setting.key}:{row.label} failed exact orbit-symmetry certification"
                )
            scaffold = CrystalScaffold.from_mapping(
                nu=row.multiplicity,
                space_group=setting.key,
                reduced_composition={1: 1},
                orbits=(OrbitAssignment(row.label, 1, row.generic_parameter),),
            )
            decoded = decoder.decode(scaffold, metric)
            if (
                tuple(decoded.fractional_positions) != tuple(expected)
                or not decoded.audit.passed
                or decoded.setting.key != setting.key
            ):
                raise CatalogueError(
                    f"{setting.key}:{row.label} failed exact decoder round trip"
                )
            cover = ContinuousCover.from_exact_scaffold(scaffold)
            layout = IntrinsicCoverLayout(cover, catalogue=catalogue)
            if layout.intrinsic_dimension != row.dimension - len(
                origin_axes[setting.key]
            ):
                raise CatalogueError(
                    f"{setting.key}:{row.label} failed continuous-origin quotient "
                    "dimension certification"
                )

    standardization_preflight_verified = False
    external_verified = False
    if crosscheck_external_operations:
        if not catalogue.complete_space_group_types:
            raise CatalogueError(
                "an external all-Hall operation check is undefined for a fixture catalogue"
            )
        if spglib_module is None:
            try:
                spglib_module = importlib.import_module("spglib")
            except ImportError as exc:
                raise CatalogueError(
                    "the crystallography extra is required for the independent operation gate"
                ) from exc
        from .full_catalogue import pinned_hall_numbers, validate_against_spglib
        from .standardization import SpglibPolicy, SpglibStandardizer

        validate_against_spglib(spglib_module)
        standardizer = SpglibStandardizer(
            pinned_hall_numbers(),
            SpglibPolicy(required_spglib_version=str(spglib_module.__version__)),
        )
        standardizer.validate_catalogue_preflight(require_all_types=True)
        standardization_preflight_verified = True
        external_verified = True

    unsigned: dict[str, Any] = {
        "schema": _SCHEMA,
        "catalogue_fingerprint": catalogue.fingerprint,
        "certification_scope": catalogue.certification_scope,
        "complete_space_group_types": catalogue.complete_space_group_types,
        "space_group_count": group_count,
        "wyckoff_row_count": row_count,
        "affine_operation_count": operation_count,
        "pinned_hall_manifest_fingerprint": hall_fingerprint,
        "source_identity": dict(source),
        "exact_manifest_bijection_verified": True,
        "exact_group_closure_verified": True,
        "exact_multiplicity_verified": True,
        "exact_symmetry_verified": True,
        "exact_origin_chart_verified": True,
        "exact_decoder_roundtrip_verified": True,
        "pinned_standardization_preflight_verified": (
            standardization_preflight_verified
        ),
        "external_operation_database_verified": external_verified,
    }
    return CrystallographyCertification(
        catalogue_fingerprint=catalogue.fingerprint,
        certification_scope=catalogue.certification_scope,
        complete_space_group_types=catalogue.complete_space_group_types,
        space_group_count=group_count,
        wyckoff_row_count=row_count,
        affine_operation_count=operation_count,
        pinned_hall_manifest_fingerprint=hall_fingerprint,
        source_identity=source,
        exact_manifest_bijection_verified=True,
        exact_group_closure_verified=True,
        exact_multiplicity_verified=True,
        exact_symmetry_verified=True,
        exact_origin_chart_verified=True,
        exact_decoder_roundtrip_verified=True,
        pinned_standardization_preflight_verified=standardization_preflight_verified,
        external_operation_database_verified=external_verified,
        receipt_fingerprint=_canonical_fingerprint(unsigned),
    )


__all__ = [
    "CrystallographyCertification",
    "certify_crystallography",
]
