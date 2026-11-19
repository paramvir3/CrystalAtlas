"""Pinned, explicitly scoped Hall/Wyckoff catalogue data.

This module defines the exact catalogue types and ships lightweight complete
mathematical fixtures for P1 and P-1.  The production all-230 compiler lives in
``full_catalogue.py`` so importing the dependency-light core does not eagerly
compile 1,731 Wyckoff rows.  Every catalogue object remains explicitly scoped
and every lookup/completeness request fails closed outside its own manifest.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from fractions import Fraction
from hashlib import sha256
from types import MappingProxyType

from .algebra import (
    AffineOperation,
    FractionMatrix,
    FractionVector,
    as_fraction_matrix,
    as_fraction_vector,
    exact_determinant,
    exact_inverse,
    exact_matmul,
    identity_integer_matrix,
    validate_finite_affine_group,
)
from .errors import CatalogueError, DomainError, InvariantViolation
from .wyckoff import WyckoffPosition

_ZERO = Fraction(0)
_HALF = Fraction(1, 2)
_IDENTITY_ROTATION = identity_integer_matrix(3)
_INVERSION_ROTATION = ((-1, 0, 0), (0, -1, 0), (0, 0, -1))
_IDENTITY_OPERATION = AffineOperation.identity()
_INVERSION_OPERATION = AffineOperation(_INVERSION_ROTATION, (_ZERO, _ZERO, _ZERO))


@dataclass(frozen=True, slots=True)
class SpaceGroupSetting:
    """One pinned finite conventional operation table and all its Wyckoff rows."""

    key: str
    international_number: int
    hall_symbol: str
    hm_symbol: str
    operations: tuple[AffineOperation, ...]
    wyckoff_positions: Mapping[str, WyckoffPosition]
    centering_translations: tuple[FractionVector, ...]
    centering_reduction: FractionMatrix
    reference_metric: FractionMatrix

    def __post_init__(self) -> None:
        if not self.key or self.international_number not in range(1, 231):
            raise DomainError("space-group setting needs a valid key and international number")
        operations = tuple(self.operations)
        positions = MappingProxyType(dict(self.wyckoff_positions))
        centering = tuple(
            as_fraction_vector(value, length=3) for value in self.centering_translations
        )
        reduction = as_fraction_matrix(self.centering_reduction, ncols=3)
        metric = as_fraction_matrix(self.reference_metric, ncols=3)
        if len(reduction) != 3 or len(metric) != 3:
            raise DomainError("centering map and reference metric must be 3x3")
        object.__setattr__(self, "operations", operations)
        object.__setattr__(self, "wyckoff_positions", positions)
        object.__setattr__(self, "centering_translations", centering)
        object.__setattr__(self, "centering_reduction", reduction)
        object.__setattr__(self, "reference_metric", metric)

    @property
    def point_group_rotations(self) -> tuple[tuple[tuple[int, ...], ...], ...]:
        return tuple(dict.fromkeys(operation.rotation for operation in self.operations))

    @property
    def centering_index(self) -> int:
        return len(self.centering_translations)

    def wyckoff(self, label: str) -> WyckoffPosition:
        try:
            return self.wyckoff_positions[label]
        except KeyError as error:
            raise CatalogueError(f"Wyckoff row {self.key}:{label} is not certified") from error

    def validate(self, expected_rows: frozenset[str]) -> None:
        """Validate the stored setting with exact finite algebra."""

        try:
            validate_finite_affine_group(self.operations)
            if frozenset(self.wyckoff_positions) != expected_rows:
                missing = sorted(expected_rows.difference(self.wyckoff_positions))
                extra = sorted(set(self.wyckoff_positions).difference(expected_rows))
                raise CatalogueError(
                    f"{self.key} Wyckoff manifest mismatch; missing={missing}, extra={extra}"
                )
            if not self.centering_translations or len(set(self.centering_translations)) != len(
                self.centering_translations
            ):
                raise CatalogueError("centering translations must be a nonempty exact set")
            pure_translations = {
                operation.translation
                for operation in self.operations
                if operation.is_pure_translation
            }
            if pure_translations != set(self.centering_translations):
                raise CatalogueError(
                    "declared centering subgroup disagrees with operation table"
                )

            determinant = exact_determinant(self.centering_reduction)
            expected_determinant = Fraction(1, self.centering_index)
            if determinant != expected_determinant or determinant <= 0:
                raise CatalogueError(
                    "centering-reduction determinant has the wrong exact index"
                )
            reduction_inverse = exact_inverse(self.centering_reduction)
            if any(value.denominator != 1 for row in reduction_inverse for value in row):
                raise CatalogueError("centering-reduction inverse is not integral")
            for translation in self.centering_translations:
                image = tuple(
                    sum(
                        (
                            translation[k] * reduction_inverse[k][j]
                            for k in range(3)
                        ),
                        Fraction(0),
                    )
                    for j in range(3)
                )
                if any(value.denominator != 1 for value in image):
                    raise CatalogueError(
                        "a declared centering translation is not in the map kernel"
                    )

            if exact_determinant(self.reference_metric) <= 0:
                raise CatalogueError("reference metric is not positive definite")
            # Sylvester is exact for this 3x3 symmetric rational fixture.
            if self.reference_metric != tuple(
                tuple(self.reference_metric[j][i] for j in range(3)) for i in range(3)
            ):
                raise CatalogueError("reference metric is not symmetric")
            for leading_size in (1, 2, 3):
                leading = tuple(
                    tuple(self.reference_metric[i][j] for j in range(leading_size))
                    for i in range(leading_size)
                )
                if exact_determinant(leading) <= 0:
                    raise CatalogueError("reference metric is not positive definite")
            for rotation in self.point_group_rotations:
                left = exact_matmul(rotation, self.reference_metric)
                transformed = exact_matmul(
                    left, tuple(tuple(rotation[j][i] for j in range(3)) for i in range(3))
                )
                if transformed != self.reference_metric:
                    raise CatalogueError("reference metric is not point-group invariant")

            group_members = set(self.operations)
            group_order = len(self.operations)
            for label, position in self.wyckoff_positions.items():
                if any(
                    operation not in group_members
                    for operation in position.coset_representatives
                ):
                    raise CatalogueError(
                        f"{self.key}:{label} contains an unknown coset operation"
                    )
                representative = position.representative_exact(position.generic_parameter)
                stabilizer_order = sum(
                    operation.apply_exact(representative) == representative
                    for operation in self.operations
                )
                if stabilizer_order != position.site_symmetry_order:
                    raise CatalogueError(f"{self.key}:{label} has the wrong site stabilizer")
                if position.multiplicity * position.site_symmetry_order != group_order:
                    raise CatalogueError(f"{self.key}:{label} violates orbit-stabilizer")
                orbit = position.expand_exact(position.generic_parameter)
                orbit_set = set(orbit)
                for operation in self.operations:
                    if {operation.apply_exact(point) for point in orbit} != orbit_set:
                        raise CatalogueError(f"{self.key}:{label} is not a group orbit")
        except (DomainError, InvariantViolation) as error:
            raise CatalogueError(f"invalid exact setting {self.key}: {error}") from error


class CertifiedCatalogue:
    """An exact catalogue whose finite scope is part of its runtime contract.

    Parameters
    ----------
    settings:
        Pinned settings keyed by their explicit catalogue keys.
    expected_rows:
        Independent manifest used to detect missing and extra Wyckoff entries.
    certification_scope:
        Human-readable name for the frozen data scope.
    complete_space_group_types:
        Must remain false for the built-in two-group fixture.  A caller asking
        for all 230 groups receives an exception rather than partial data.
    """

    def __init__(
        self,
        settings: Mapping[str, SpaceGroupSetting],
        expected_rows: Mapping[str, frozenset[str]],
        *,
        certification_scope: str,
        complete_space_group_types: bool = False,
    ) -> None:
        self._settings = MappingProxyType(dict(settings))
        self._expected_rows = MappingProxyType(dict(expected_rows))
        self.certification_scope = certification_scope
        self.complete_space_group_types = complete_space_group_types
        # Every reachable catalogue datum is immutable: the outer mappings are
        # read-only and SpaceGroupSetting/WyckoffPosition/AffineOperation are
        # frozen value objects.  Exact finite-group validation is expensive for
        # the 230-type catalogue, so prove it once at construction rather than
        # accidentally repeating the same proof in every decoder and target
        # lift.  The independent pre-training receipt deliberately performs its
        # own exhaustive checks and does not rely on this marker.
        self._construction_validated = False
        self.validate()

    @property
    def settings(self) -> Mapping[str, SpaceGroupSetting]:
        return self._settings

    @property
    def supported_keys(self) -> tuple[str, ...]:
        return tuple(self._settings)

    @property
    def supported_international_numbers(self) -> tuple[int, ...]:
        return tuple(
            sorted(setting.international_number for setting in self._settings.values())
        )

    @staticmethod
    def _fraction(value: Fraction) -> tuple[int, int]:
        return value.numerator, value.denominator

    @classmethod
    def _fraction_vector(cls, values: FractionVector) -> list[list[int]]:
        return [list(cls._fraction(value)) for value in values]

    @classmethod
    def _fraction_matrix(cls, values: FractionMatrix) -> list[list[list[int]]]:
        return [cls._fraction_vector(row) for row in values]

    @classmethod
    def _operation_payload(cls, operation: AffineOperation) -> dict[str, object]:
        return {
            "rotation": [list(row) for row in operation.rotation],
            "translation": cls._fraction_vector(operation.translation),
        }

    def to_dict(self) -> dict[str, object]:
        """Return a complete canonical JSON-compatible catalogue description.

        The payload includes the exact rational operation, metric, centering,
        and Wyckoff tables—not merely names or row counts.  Consequently its
        fingerprint changes whenever any decoder-relevant catalogue datum
        changes.
        """

        settings: list[dict[str, object]] = []
        for key in sorted(self._settings):
            setting = self._settings[key]
            positions: list[dict[str, object]] = []
            for label in sorted(setting.wyckoff_positions):
                position = setting.wyckoff_positions[label]
                positions.append(
                    {
                        "label": position.label,
                        "multiplicity": position.multiplicity,
                        "representative_offset": self._fraction_vector(
                            position.representative_offset
                        ),
                        "coordinate_matrix": self._fraction_matrix(
                            position.coordinate_matrix
                        ),
                        "period_basis": self._fraction_matrix(position.period_basis),
                        "coset_representatives": [
                            self._operation_payload(operation)
                            for operation in position.coset_representatives
                        ],
                        "site_symmetry_order": position.site_symmetry_order,
                        "generic_parameter": self._fraction_vector(
                            position.generic_parameter
                        ),
                    }
                )
            settings.append(
                {
                    "key": setting.key,
                    "international_number": setting.international_number,
                    "hall_symbol": setting.hall_symbol,
                    "hm_symbol": setting.hm_symbol,
                    "operations": [
                        self._operation_payload(operation)
                        for operation in setting.operations
                    ],
                    "wyckoff_positions": positions,
                    "expected_rows": sorted(self._expected_rows[key]),
                    "centering_translations": [
                        self._fraction_vector(translation)
                        for translation in setting.centering_translations
                    ],
                    "centering_reduction": self._fraction_matrix(
                        setting.centering_reduction
                    ),
                    "reference_metric": self._fraction_matrix(
                        setting.reference_metric
                    ),
                }
            )
        return {
            "schema": "crystalatlas.certified-catalogue.v1",
            "certification_scope": self.certification_scope,
            "complete_space_group_types": self.complete_space_group_types,
            "settings": settings,
        }

    @property
    def fingerprint(self) -> str:
        """SHA-256 of the complete canonical payload, memoized on the frozen data.

        Serializing the exact tables of all 230 types costs about 0.4 s, and the
        trainer re-reads the objective contract, which binds this value, at every
        optimizer step.  Every reachable catalogue datum is immutable (see
        ``__init__``), so the memo is keyed by the identity of the two read-only
        mappings and the value of the two public scope attributes: replacing
        either mapping or changing either attribute recomputes the fingerprint.
        """

        memo = getattr(self, "_fingerprint_memo", None)
        if (
            memo is not None
            and memo[0] is self._settings
            and memo[1] is self._expected_rows
            and memo[2] == self.certification_scope
            and memo[3] is self.complete_space_group_types
        ):
            return memo[4]
        payload = json.dumps(
            self.to_dict(), allow_nan=False, sort_keys=True, separators=(",", ":")
        )
        value = sha256(payload.encode("utf-8")).hexdigest()
        self._fingerprint_memo = (
            self._settings,
            self._expected_rows,
            self.certification_scope,
            self.complete_space_group_types,
            value,
        )
        return value

    @property
    def identity(self) -> dict[str, object]:
        """Return the compact identity that must be bound into checkpoints."""

        return {
            "schema": "crystalatlas.certified-catalogue.identity.v1",
            "fingerprint": self.fingerprint,
            "certification_scope": self.certification_scope,
            "complete_space_group_types": self.complete_space_group_types,
            "supported_keys": list(sorted(self.supported_keys)),
        }

    def validate(self, *, require_all_230: bool = False) -> None:
        if getattr(self, "_construction_validated", False):
            if require_all_230 and not self.complete_space_group_types:
                raise CatalogueError(
                    f"catalogue scope '{self.certification_scope}' is not a complete "
                    "230-group catalogue"
                )
            return
        if set(self._settings) != set(self._expected_rows):
            missing = sorted(set(self._expected_rows).difference(self._settings))
            extra = sorted(set(self._settings).difference(self._expected_rows))
            raise CatalogueError(f"setting manifest mismatch; missing={missing}, extra={extra}")
        seen_numbers: set[int] = set()
        for key, setting in self._settings.items():
            if setting.key != key:
                raise CatalogueError(f"catalogue key {key} disagrees with stored setting key")
            if setting.international_number in seen_numbers:
                raise CatalogueError("catalogue repeats an international space-group type")
            seen_numbers.add(setting.international_number)
            setting.validate(self._expected_rows[key])
        if self.complete_space_group_types and seen_numbers != set(range(1, 231)):
            raise CatalogueError(
                "a purported complete catalogue does not contain all 230 types"
            )
        if require_all_230 and not self.complete_space_group_types:
            raise CatalogueError(
                f"catalogue scope '{self.certification_scope}' is not a complete "
                "230-group catalogue"
            )
        self._construction_validated = True

    def get(self, key_or_number: str | int) -> SpaceGroupSetting:
        if isinstance(key_or_number, str):
            try:
                return self._settings[key_or_number]
            except KeyError as error:
                raise CatalogueError(
                    f"space-group setting {key_or_number!r} is outside certified scope "
                    f"{self.supported_keys}"
                ) from error
        matches = [
            setting
            for setting in self._settings.values()
            if setting.international_number == key_or_number
        ]
        if len(matches) != 1:
            raise CatalogueError(
                f"international space-group number {key_or_number!r} is outside certified scope"
            )
        return matches[0]


def _free_position(
    label: str,
    multiplicity: int,
    cosets: tuple[AffineOperation, ...],
    site_symmetry_order: int,
    generic_parameter: FractionVector,
) -> WyckoffPosition:
    return WyckoffPosition(
        label=label,
        multiplicity=multiplicity,
        representative_offset=(_ZERO, _ZERO, _ZERO),
        coordinate_matrix=(
            (Fraction(1), _ZERO, _ZERO),
            (_ZERO, Fraction(1), _ZERO),
            (_ZERO, _ZERO, Fraction(1)),
        ),
        period_basis=(
            (Fraction(1), _ZERO, _ZERO),
            (_ZERO, Fraction(1), _ZERO),
            (_ZERO, _ZERO, Fraction(1)),
        ),
        coset_representatives=cosets,
        site_symmetry_order=site_symmetry_order,
        generic_parameter=generic_parameter,
    )


def _fixed_position(label: str, offset: FractionVector) -> WyckoffPosition:
    return WyckoffPosition(
        label=label,
        multiplicity=1,
        representative_offset=offset,
        coordinate_matrix=(),
        period_basis=(),
        coset_representatives=(_IDENTITY_OPERATION,),
        site_symmetry_order=2,
        generic_parameter=(),
    )


def _p1_setting() -> SpaceGroupSetting:
    positions = {
        "1a": _free_position(
            "1a",
            1,
            (_IDENTITY_OPERATION,),
            1,
            (Fraction(1, 7), Fraction(2, 7), Fraction(3, 7)),
        )
    }
    return SpaceGroupSetting(
        key="P1",
        international_number=1,
        hall_symbol="P 1",
        hm_symbol="P 1",
        operations=(_IDENTITY_OPERATION,),
        wyckoff_positions=positions,
        centering_translations=((_ZERO, _ZERO, _ZERO),),
        centering_reduction=tuple(
            tuple(Fraction(int(i == j)) for j in range(3)) for i in range(3)
        ),
        reference_metric=tuple(
            tuple(Fraction(int(i == j)) for j in range(3)) for i in range(3)
        ),
    )


def _p_minus_1_setting() -> SpaceGroupSetting:
    positions: dict[str, WyckoffPosition] = {
        "2i": _free_position(
            "2i",
            2,
            (_IDENTITY_OPERATION, _INVERSION_OPERATION),
            1,
            (Fraction(1, 7), Fraction(2, 7), Fraction(3, 7)),
        ),
        "1a": _fixed_position("1a", (_ZERO, _ZERO, _ZERO)),
        "1b": _fixed_position("1b", (_ZERO, _ZERO, _HALF)),
        "1c": _fixed_position("1c", (_ZERO, _HALF, _ZERO)),
        "1d": _fixed_position("1d", (_HALF, _ZERO, _ZERO)),
        "1e": _fixed_position("1e", (_HALF, _HALF, _ZERO)),
        "1f": _fixed_position("1f", (_HALF, _ZERO, _HALF)),
        "1g": _fixed_position("1g", (_ZERO, _HALF, _HALF)),
        "1h": _fixed_position("1h", (_HALF, _HALF, _HALF)),
    }
    return SpaceGroupSetting(
        key="P-1",
        international_number=2,
        hall_symbol="-P 1",
        hm_symbol="P -1",
        operations=(_IDENTITY_OPERATION, _INVERSION_OPERATION),
        wyckoff_positions=positions,
        centering_translations=((_ZERO, _ZERO, _ZERO),),
        centering_reduction=tuple(
            tuple(Fraction(int(i == j)) for j in range(3)) for i in range(3)
        ),
        reference_metric=tuple(
            tuple(Fraction(int(i == j)) for j in range(3)) for i in range(3)
        ),
    )


FIXTURE_EXPECTED_ROWS: Mapping[str, frozenset[str]] = MappingProxyType(
    {
        "P1": frozenset({"1a"}),
        "P-1": frozenset({"2i", "1a", "1b", "1c", "1d", "1e", "1f", "1g", "1h"}),
    }
)


def build_minimal_catalogue() -> CertifiedCatalogue:
    """Return complete P1/P-1 fixtures, explicitly incomplete beyond them."""

    settings = {"P1": _p1_setting(), "P-1": _p_minus_1_setting()}
    return CertifiedCatalogue(
        settings,
        FIXTURE_EXPECTED_ROWS,
        certification_scope="exact P1/P-1 mathematical fixtures",
        complete_space_group_types=False,
    )


MINIMAL_CATALOGUE = build_minimal_catalogue()
