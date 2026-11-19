"""Compile the complete exact 230-type catalogue from pinned spglib sources.

The bundled source is spglib 2.7.0 at commit
``12355c77fb7c505a55f52cae36341d73b781a065``.  We pin the smallest Hall
number for each International space-group number as an explicit deterministic
package convention.  No floating-point value is used to construct an exact
affine operation or Wyckoff chart.

The source CSVs remain bundled, checksummed, and covered by their upstream
BSD-3-Clause licence.  This module is a deterministic compiler, not a second
hand-written crystallographic table.
"""

from __future__ import annotations

import csv
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from fractions import Fraction
from functools import cache, lru_cache
from hashlib import sha256
from importlib.resources import files
from types import MappingProxyType
from typing import Any

from .algebra import (
    AffineOperation,
    FractionMatrix,
    FractionVector,
    exact_inverse,
    exact_row_matmul,
    fraction_mod_one,
    validate_finite_affine_group,
)
from .catalogue import CertifiedCatalogue, SpaceGroupSetting
from .errors import CatalogueError
from .wyckoff import WyckoffPosition

SPGLIB_SOURCE_VERSION = "2.7.0"
SPGLIB_SOURCE_COMMIT = "12355c77fb7c505a55f52cae36341d73b781a065"
SPGLIB_SOURCE_LICENSE = "BSD-3-Clause"

# Independent release manifest.  These literals are deliberately not derived
# from the parser output: a source-row deletion, compiler regression, or
# operation loss must make catalogue construction fail closed.
FROZEN_SPACE_GROUP_TYPE_COUNT = 230
FROZEN_WYCKOFF_ROW_COUNT = 1731
FROZEN_AFFINE_OPERATION_COUNT = 4425
FROZEN_CATALOGUE_FINGERPRINT = (
    "707220a51f46101ed6b7d4e2401f8d0dd2e0416dd51864f9abce3913cd84f029"
)
# Recorded only to prevent accidental reintroduction.  This digest preceded
# canonical rendering of the Hall double-prime marker: twelve spg.csv symbols
# still contained the table's ASCII ``=`` encoding instead of public ``\"``.
REJECTED_PRE_CANONICAL_HALL_FINGERPRINT = (
    "b86799690fb8e0fa6683fb64afa55bd87d72006b48f17d8da5f73fddc3e524ba"
)

_SOURCE_HASHES = MappingProxyType(
    {
        "COPYING": "9c4e602ce15bf48206dad105b666ddaeedba747eeda6465807106bc11f4f587f",
        "Wyckoff.csv": "d3d786a1f0187e5c6d69a3ade35648ffab34fd1b977d61ad84d8b0434b8b7ca0",
        "spg.csv": "4457df1042b14a65ea62af0bad7b5b609a4fc33592df245802cfe005b221f95e",
    }
)
_DATA_PARTS = ("_data", "spglib_v2_7_0")
_VARIABLES = "xyz"
_TERM_PATTERN = re.compile(r"([+-]?)([^+-]+)")
_VARIABLE_TERM_PATTERN = re.compile(r"(\d*)([xyz])\Z")

# In the source generator these Hall settings are the hexagonal-axis
# descriptions of rhombohedral groups.  The leading ``R`` in Wyckoff.csv is
# changed to ``H`` before the conventional centring translations are expanded.
_HEXAGONAL_RHOMBOHEDRAL_HALLS = frozenset({433, 436, 444, 450, 452, 458, 460})

_CENTERING_TRANSLATIONS: Mapping[str, tuple[FractionVector, ...]] = MappingProxyType(
    {
        "P": ((Fraction(0), Fraction(0), Fraction(0)),),
        "R": ((Fraction(0), Fraction(0), Fraction(0)),),
        "A": (
            (Fraction(0), Fraction(0), Fraction(0)),
            (Fraction(0), Fraction(1, 2), Fraction(1, 2)),
        ),
        "B": (
            (Fraction(0), Fraction(0), Fraction(0)),
            (Fraction(1, 2), Fraction(0), Fraction(1, 2)),
        ),
        "C": (
            (Fraction(0), Fraction(0), Fraction(0)),
            (Fraction(1, 2), Fraction(1, 2), Fraction(0)),
        ),
        "I": (
            (Fraction(0), Fraction(0), Fraction(0)),
            (Fraction(1, 2), Fraction(1, 2), Fraction(1, 2)),
        ),
        "F": (
            (Fraction(0), Fraction(0), Fraction(0)),
            (Fraction(0), Fraction(1, 2), Fraction(1, 2)),
            (Fraction(1, 2), Fraction(0), Fraction(1, 2)),
            (Fraction(1, 2), Fraction(1, 2), Fraction(0)),
        ),
        "H": (
            (Fraction(0), Fraction(0), Fraction(0)),
            (Fraction(2, 3), Fraction(1, 3), Fraction(1, 3)),
            (Fraction(1, 3), Fraction(2, 3), Fraction(2, 3)),
        ),
    }
)

# Integer maps M=C^{-1}.  det(M) is the conventional-to-primitive index,
# and every declared centring translation t obeys tM in Z^3 exactly.
_CENTERING_INVERSES: Mapping[str, tuple[tuple[int, int, int], ...]] = MappingProxyType(
    {
        "P": ((1, 0, 0), (0, 1, 0), (0, 0, 1)),
        "R": ((1, 0, 0), (0, 1, 0), (0, 0, 1)),
        "A": ((1, 0, 0), (0, 1, 1), (0, -1, 1)),
        "B": ((1, 0, 1), (0, 1, 0), (-1, 0, 1)),
        "C": ((1, 1, 0), (-1, 1, 0), (0, 0, 1)),
        "I": ((1, 0, 1), (1, 1, 0), (0, 1, 1)),
        "F": ((-1, 1, 1), (1, -1, 1), (1, 1, -1)),
        "H": ((-1, -1, 1), (-1, 0, -1), (0, -1, -1)),
    }
)


@dataclass(frozen=True, slots=True)
class SourceWyckoffRow:
    """One exact row as represented in the pinned upstream CSV."""

    multiplicity: int
    letter: str
    site_symmetry_symbol: str
    coordinate_expressions: tuple[str, ...]

    @property
    def label(self) -> str:
        return f"{self.multiplicity}{self.letter}"


@dataclass(frozen=True, slots=True)
class SourceHallSetting:
    """All upstream Wyckoff rows for one Hall serial number."""

    hall_number: int
    coordinate_symbol: str
    rows: tuple[SourceWyckoffRow, ...]


@dataclass(frozen=True, slots=True)
class HallMetadata:
    """Pinned type metadata needed by the exact catalogue compiler."""

    hall_number: int
    international_number: int
    hall_symbol: str
    hm_symbol: str
    choice: str

    @property
    def key(self) -> str:
        return "".join(self.hm_symbol.split())


def source_identity() -> dict[str, object]:
    """Return the immutable third-party source identity."""

    return {
        "schema": "crystalatlas.third-party-crystallographic-source.v1",
        "name": "spglib crystallographic database",
        "version": SPGLIB_SOURCE_VERSION,
        "git_commit": SPGLIB_SOURCE_COMMIT,
        "repository": "https://github.com/spglib/spglib",
        "license": SPGLIB_SOURCE_LICENSE,
        "selection_rule": "smallest Hall number per International number",
        "sha256": dict(_SOURCE_HASHES),
    }


def _source_path(name: str) -> Any:
    path: Any = files("crystalatlas")
    for part in (*_DATA_PARTS, name):
        path = path.joinpath(part)
    return path


@cache
def _verified_source_text(name: str) -> str:
    try:
        expected = _SOURCE_HASHES[name]
    except KeyError as error:
        raise CatalogueError(f"unrecognized catalogue source file {name!r}") from error
    payload: bytes = _source_path(name).read_bytes()
    actual = sha256(payload).hexdigest()
    if actual != expected:
        raise CatalogueError(
            f"pinned spglib source checksum mismatch for {name}: "
            f"expected {expected}, got {actual}"
        )
    return payload.decode("utf-8")


@lru_cache(maxsize=1)
def source_hall_settings() -> Mapping[int, SourceHallSetting]:
    """Parse and exhaustively validate all 530 upstream Hall datasets."""

    parsed: dict[int, SourceHallSetting] = {}
    hall_number: int | None = None
    coordinate_symbol = ""
    rows: list[SourceWyckoffRow] = []
    current: SourceWyckoffRow | None = None

    def flush() -> None:
        nonlocal rows, current
        if hall_number is None:
            return
        if current is not None:
            rows.append(current)
            current = None
        if not rows:
            raise CatalogueError(f"Hall setting {hall_number} has no Wyckoff rows")
        parsed[hall_number] = SourceHallSetting(
            hall_number, coordinate_symbol, tuple(rows)
        )
        rows = []

    for line_number, raw_line in enumerate(
        _verified_source_text("Wyckoff.csv").splitlines(), start=1
    ):
        line = raw_line.strip()
        if line == "end of data":
            break
        fields = line.split(":")
        if fields[0].isdigit():
            flush()
            hall_number = int(fields[0])
            coordinate_symbol = fields[1].strip()
            continue
        if hall_number is None or len(fields) < 9:
            raise CatalogueError(f"malformed Wyckoff.csv line {line_number}")
        expressions = tuple(value.strip() for value in fields[5:9] if value.strip())
        if fields[2].isdigit():
            if current is not None:
                rows.append(current)
            current = SourceWyckoffRow(
                multiplicity=int(fields[2]),
                letter=fields[3].strip(),
                site_symmetry_symbol=fields[4].strip(),
                coordinate_expressions=expressions,
            )
        else:
            if current is None:
                raise CatalogueError(
                    f"orphaned Wyckoff continuation at source line {line_number}"
                )
            current = SourceWyckoffRow(
                current.multiplicity,
                current.letter,
                current.site_symmetry_symbol,
                current.coordinate_expressions + expressions,
            )
    flush()

    if set(parsed) != set(range(1, 531)):
        missing = sorted(set(range(1, 531)).difference(parsed))
        extra = sorted(set(parsed).difference(range(1, 531)))
        raise CatalogueError(
            f"upstream Hall manifest mismatch; missing={missing}, extra={extra}"
        )
    for number, setting in parsed.items():
        labels = [row.label for row in setting.rows]
        if len(labels) != len(set(labels)):
            raise CatalogueError(f"Hall setting {number} repeats a Wyckoff label")
        centering = _centering_translations(setting)
        for row in setting.rows:
            if not row.letter or not row.coordinate_expressions:
                raise CatalogueError(f"Hall setting {number} has an empty Wyckoff row")
            if len(row.coordinate_expressions) * len(centering) != row.multiplicity:
                raise CatalogueError(
                    f"Hall {number}:{row.label} source multiplicity disagrees "
                    "with its coordinate and centring expansion"
                )
    return MappingProxyType(parsed)


@lru_cache(maxsize=1)
def hall_metadata() -> Mapping[int, HallMetadata]:
    """Parse the 530-row pinned Hall/type metadata table."""

    result: dict[int, HallMetadata] = {}
    reader = csv.reader(_verified_source_text("spg.csv").splitlines())
    for line_number, fields in enumerate(reader, start=1):
        if len(fields) < 10:
            raise CatalogueError(f"malformed spg.csv line {line_number}")
        hall_number = int(fields[0])
        international_number = int(fields[4])
        international = fields[7].strip()
        hm_symbol = international.split(" = ", maxsplit=1)[0].strip()
        result[hall_number] = HallMetadata(
            hall_number=hall_number,
            international_number=international_number,
            # spg.csv uses ``=`` as its ASCII-safe spelling of the Hall
            # double-prime direction marker; the public spglib API renders
            # that marker as ``"``.
            hall_symbol=fields[6].strip().replace("=", '"'),
            hm_symbol=hm_symbol,
            choice=fields[2].strip(),
        )
    if set(result) != set(range(1, 531)):
        raise CatalogueError("spg.csv does not contain each Hall number 1..530 exactly once")
    return MappingProxyType(result)


@lru_cache(maxsize=1)
def pinned_hall_numbers() -> Mapping[int, int]:
    """Return the explicit standard Hall serial selected for each type."""

    selected: dict[int, int] = {}
    for hall_number, metadata in hall_metadata().items():
        selected.setdefault(metadata.international_number, hall_number)
    if set(selected) != set(range(1, 231)):
        raise CatalogueError("smallest-Hall selection did not cover all 230 types")
    return MappingProxyType(selected)


def source_rows_for_type(
    international_number: int,
) -> Mapping[str, SourceWyckoffRow]:
    """Return exact source rows, including site-symmetry symbols, by label."""

    if international_number not in range(1, 231):
        raise CatalogueError("International space-group number must be in 1..230")
    hall_number = pinned_hall_numbers()[international_number]
    rows = source_hall_settings()[hall_number].rows
    return MappingProxyType({row.label: row for row in rows})


def _centering_symbol(setting: SourceHallSetting) -> str:
    if setting.hall_number in _HEXAGONAL_RHOMBOHEDRAL_HALLS:
        return "H"
    symbol = setting.coordinate_symbol[:1]
    if symbol not in _CENTERING_TRANSLATIONS:
        raise CatalogueError(
            f"Hall setting {setting.hall_number} has unsupported centring symbol {symbol!r}"
        )
    return symbol


def _centering_translations(setting: SourceHallSetting) -> tuple[FractionVector, ...]:
    return _CENTERING_TRANSLATIONS[_centering_symbol(setting)]


def _parse_linear_expression(expression: str) -> tuple[tuple[int, int, int], Fraction]:
    text = expression.strip()
    coefficients = [0, 0, 0]
    offset = Fraction(0)
    terms = _TERM_PATTERN.findall(text)
    if not terms or "".join(sign + term for sign, term in terms) != text:
        raise CatalogueError(f"unsupported exact Wyckoff expression {expression!r}")
    for sign_text, term in terms:
        sign = -1 if sign_text == "-" else 1
        variable = _VARIABLE_TERM_PATTERN.fullmatch(term)
        if variable is None:
            try:
                offset += sign * Fraction(term)
            except (ValueError, ZeroDivisionError) as error:
                raise CatalogueError(
                    f"unsupported exact Wyckoff term {term!r}"
                ) from error
            continue
        coefficient = int(variable.group(1) or "1")
        coefficients[_VARIABLES.index(variable.group(2))] += sign * coefficient
    return (coefficients[0], coefficients[1], coefficients[2]), fraction_mod_one(offset)


def _parse_coordinate_expression(
    expression: str,
) -> tuple[tuple[tuple[int, ...], ...], FractionVector]:
    text = expression.strip()
    if not text.startswith("(") or not text.endswith(")"):
        raise CatalogueError(f"malformed Wyckoff coordinate {expression!r}")
    components = text[1:-1].split(",")
    if len(components) != 3:
        raise CatalogueError(f"Wyckoff coordinate is not three-dimensional: {expression!r}")
    parsed = tuple(_parse_linear_expression(component) for component in components)
    # Each parsed coefficient vector belongs to one output component.  The
    # transpose below is the matrix W for the row-vector action f -> fW+w.
    rotation = tuple(tuple(parsed[j][0][i] for j in range(3)) for i in range(3))
    translation = tuple(parsed[j][1] for j in range(3))
    return rotation, translation


def _operation_table(setting: SourceHallSetting) -> tuple[AffineOperation, ...]:
    generic = setting.rows[0]
    centering = _centering_translations(setting)
    operations: list[AffineOperation] = []
    for translation in centering:
        for expression in generic.coordinate_expressions:
            rotation, offset = _parse_coordinate_expression(expression)
            operations.append(
                AffineOperation(
                    rotation,
                    tuple(offset[i] + translation[i] for i in range(3)),
                )
            )
    if len(operations) != generic.multiplicity:
        raise CatalogueError(
            f"Hall {setting.hall_number} generic row does not generate its declared order"
        )
    try:
        return validate_finite_affine_group(operations)
    except Exception as error:
        raise CatalogueError(
            f"Hall {setting.hall_number} source operations are not an exact finite group"
        ) from error


def _chart(
    row: SourceWyckoffRow,
) -> tuple[FractionVector, FractionMatrix, tuple[int, ...]]:
    coefficients, offset = _parse_coordinate_expression(row.coordinate_expressions[0])
    active = tuple(i for i in range(3) if any(coefficients[i][j] for j in range(3)))
    coordinate_matrix = tuple(
        tuple(Fraction(coefficients[i][j]) for j in range(3)) for i in active
    )
    return offset, coordinate_matrix, active


def _representative(
    offset: FractionVector, matrix: FractionMatrix, parameter: FractionVector
) -> FractionVector:
    displacement = (
        exact_row_matmul(parameter, matrix)
        if matrix
        else (Fraction(0), Fraction(0), Fraction(0))
    )
    return tuple(fraction_mod_one(offset[i] + displacement[i]) for i in range(3))


def _evaluate_source_coordinate(expression: str, variables: FractionVector) -> FractionVector:
    rotation, translation = _parse_coordinate_expression(expression)
    transformed = exact_row_matmul(variables, rotation)
    return tuple(fraction_mod_one(transformed[i] + translation[i]) for i in range(3))


def _generic_parameter_and_cosets(
    setting: SourceHallSetting,
    row: SourceWyckoffRow,
    operations: tuple[AffineOperation, ...],
    offset: FractionVector,
    matrix: FractionMatrix,
    active: tuple[int, ...],
) -> tuple[FractionVector, tuple[AffineOperation, ...]]:
    # A Wyckoff stratum is the chart minus finitely many specialization
    # hyperplanes.  Testing a deterministic sequence of exact rational points
    # therefore terminates for every valid row, without any tolerance.
    denominators = (101, 103, 107, 109, 113, 127, 131, 137)
    for attempt, denominator in enumerate(denominators):
        numerators = (1 + 7 * attempt, 11 + 13 * attempt, 37 + 17 * attempt)
        full = tuple(Fraction(value % denominator, denominator) for value in numerators)
        parameter = tuple(full[i] for i in active)
        representative = _representative(offset, matrix, parameter)
        orbit: dict[FractionVector, AffineOperation] = {}
        for operation in operations:
            image = operation.apply_exact(representative)
            orbit.setdefault(image, operation)
        if len(orbit) != row.multiplicity:
            continue

        source_orbit = {
            tuple(fraction_mod_one(point[i] + translation[i]) for i in range(3))
            for translation in _centering_translations(setting)
            for point in (
                _evaluate_source_coordinate(expression, full)
                for expression in row.coordinate_expressions
            )
        }
        if source_orbit != set(orbit):
            raise CatalogueError(
                f"Hall {setting.hall_number}:{row.label} chart does not round-trip "
                "to the pinned source coordinate set"
            )
        return parameter, tuple(orbit.values())
    raise CatalogueError(
        f"Hall {setting.hall_number}:{row.label} has no nonspecial exact test point"
    )


def _reference_metric(operations: Sequence[AffineOperation]) -> FractionMatrix:
    rotations = tuple(dict.fromkeys(operation.rotation for operation in operations))
    # Reynolds average of I over the finite point group: G=sum_R R R^T.
    # It is integral, positive definite, and QGQ^T=G for every point-group Q.
    return tuple(
        tuple(
            Fraction(
                sum(
                    rotation[i][k] * rotation[j][k]
                    for rotation in rotations
                    for k in range(3)
                )
            )
            for j in range(3)
        )
        for i in range(3)
    )


@lru_cache(maxsize=230)
def build_space_group_setting(international_number: int) -> SpaceGroupSetting:
    """Compile one pinned standard setting with exact rational arithmetic."""

    if international_number not in range(1, 231):
        raise CatalogueError("International space-group number must be in 1..230")
    hall_number = pinned_hall_numbers()[international_number]
    source = source_hall_settings()[hall_number]
    metadata = hall_metadata()[hall_number]
    operations = _operation_table(source)
    positions: dict[str, WyckoffPosition] = {}
    for source_row in source.rows:
        offset, matrix, active = _chart(source_row)
        parameter, cosets = _generic_parameter_and_cosets(
            source, source_row, operations, offset, matrix, active
        )
        dimension = len(matrix)
        identity_period = tuple(
            tuple(Fraction(int(i == j)) for j in range(dimension))
            for i in range(dimension)
        )
        if len(operations) % source_row.multiplicity:
            raise CatalogueError(
                f"Hall {hall_number}:{source_row.label} violates orbit-stabilizer divisibility"
            )
        positions[source_row.label] = WyckoffPosition(
            label=source_row.label,
            multiplicity=source_row.multiplicity,
            representative_offset=offset,
            coordinate_matrix=matrix,
            period_basis=identity_period,
            coset_representatives=cosets,
            site_symmetry_order=len(operations) // source_row.multiplicity,
            generic_parameter=parameter,
        )
    centering_symbol = _centering_symbol(source)
    centering_reduction = exact_inverse(_CENTERING_INVERSES[centering_symbol])
    return SpaceGroupSetting(
        key=metadata.key,
        international_number=international_number,
        hall_symbol=metadata.hall_symbol,
        hm_symbol=metadata.hm_symbol,
        operations=operations,
        wyckoff_positions=positions,
        centering_translations=_centering_translations(source),
        centering_reduction=centering_reduction,
        reference_metric=_reference_metric(operations),
    )


def validate_frozen_compiled_manifest(catalogue: CertifiedCatalogue) -> None:
    """Fail unless ``catalogue`` equals the independently frozen release manifest."""

    if not isinstance(catalogue, CertifiedCatalogue):
        raise TypeError("catalogue must be a CertifiedCatalogue")
    observed = (
        len(catalogue.settings),
        sum(len(setting.wyckoff_positions) for setting in catalogue.settings.values()),
        sum(len(setting.operations) for setting in catalogue.settings.values()),
    )
    expected = (
        FROZEN_SPACE_GROUP_TYPE_COUNT,
        FROZEN_WYCKOFF_ROW_COUNT,
        FROZEN_AFFINE_OPERATION_COUNT,
    )
    if observed != expected:
        raise CatalogueError(
            "compiled catalogue manifest count mismatch: "
            f"expected types/rows/operations={expected}, got {observed}"
        )
    fingerprint = catalogue.fingerprint
    if fingerprint == REJECTED_PRE_CANONICAL_HALL_FINGERPRINT:
        raise CatalogueError(
            "compiled catalogue uses the rejected pre-canonical Hall-symbol encoding"
        )
    if fingerprint != FROZEN_CATALOGUE_FINGERPRINT:
        raise CatalogueError(
            "compiled catalogue fingerprint mismatch: "
            f"expected {FROZEN_CATALOGUE_FINGERPRINT}, got {fingerprint}"
        )


@lru_cache(maxsize=1)
def build_full_catalogue() -> CertifiedCatalogue:
    """Build and exactly certify one pinned Hall setting for all 230 types."""

    settings: dict[str, SpaceGroupSetting] = {}
    expected_rows: dict[str, frozenset[str]] = {}
    for international_number in range(1, 231):
        setting = build_space_group_setting(international_number)
        if setting.key in settings:
            raise CatalogueError(f"duplicate standard H-M catalogue key {setting.key!r}")
        settings[setting.key] = setting
        expected_rows[setting.key] = frozenset(setting.wyckoff_positions)
    # CertifiedCatalogue validates every finite-group closure, orbit,
    # stabilizer, metric, and manifest invariant in its constructor.
    catalogue = CertifiedCatalogue(
        settings,
        expected_rows,
        certification_scope=(
            "all 230 International space-group types; smallest Hall serial per type; "
            f"spglib {SPGLIB_SOURCE_VERSION} commit {SPGLIB_SOURCE_COMMIT}"
        ),
        complete_space_group_types=True,
    )
    validate_frozen_compiled_manifest(catalogue)
    return catalogue


def validate_against_spglib(spglib_module: Any) -> None:
    """Cross-check all pinned metadata and operations against spglib's C DB.

    This is an independent release/CI gate; catalogue construction itself does
    not import spglib and never converts floating point into exact source data.
    """

    version = getattr(spglib_module, "__version__", None)
    if version != SPGLIB_SOURCE_VERSION:
        raise CatalogueError(
            f"operation cross-check requires spglib {SPGLIB_SOURCE_VERSION}, got {version}"
        )
    for number, hall_number in pinned_hall_numbers().items():
        metadata = hall_metadata()[hall_number]
        source_operations = set(build_space_group_setting(number).operations)
        group_type = spglib_module.get_spacegroup_type(hall_number)
        symmetry = spglib_module.get_symmetry_from_database(hall_number)
        if group_type is None or symmetry is None:
            raise CatalogueError(f"spglib rejected pinned Hall number {hall_number}")
        if (
            int(group_type.number) != number
            or str(group_type.hall_symbol) != metadata.hall_symbol
            or int(group_type.hall_number) != hall_number
            or str(group_type.international_short) != metadata.key
            or str(group_type.choice) != metadata.choice
        ):
            raise CatalogueError(f"spglib metadata mismatch at Hall {hall_number}")
        external: set[AffineOperation] = set()
        for column_rotation, float_translation in zip(
            symmetry["rotations"], symmetry["translations"], strict=True
        ):
            rotation = tuple(
                tuple(int(column_rotation[j][i]) for j in range(3)) for i in range(3)
            )
            translation: list[Fraction] = []
            for value in float_translation:
                rational = Fraction(float(value)).limit_denominator(24)
                if abs(float(rational) - float(value)) > 1.0e-12:
                    raise CatalogueError(
                        "spglib returned a non-crystallographic translation "
                        f"at Hall {hall_number}"
                    )
                translation.append(rational)
            external.add(AffineOperation(rotation, tuple(translation)))
        if external != source_operations:
            raise CatalogueError(f"operation-table mismatch at Hall {hall_number}")


__all__ = [
    "HallMetadata",
    "FROZEN_AFFINE_OPERATION_COUNT",
    "FROZEN_CATALOGUE_FINGERPRINT",
    "FROZEN_SPACE_GROUP_TYPE_COUNT",
    "FROZEN_WYCKOFF_ROW_COUNT",
    "REJECTED_PRE_CANONICAL_HALL_FINGERPRINT",
    "SPGLIB_SOURCE_COMMIT",
    "SPGLIB_SOURCE_LICENSE",
    "SPGLIB_SOURCE_VERSION",
    "SourceHallSetting",
    "SourceWyckoffRow",
    "build_full_catalogue",
    "build_space_group_setting",
    "hall_metadata",
    "pinned_hall_numbers",
    "source_hall_settings",
    "source_identity",
    "source_rows_for_type",
    "validate_against_spglib",
    "validate_frozen_compiled_manifest",
]
