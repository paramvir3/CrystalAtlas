"""Authenticated stored targets -> deterministic, topology-homogeneous batches.

The dataset compiler and :mod:`crystalatlas.dataset_storage` establish the
provenance and byte-level trust boundary.  This module establishes the next
boundary: a training process may consume a stored record only after rebuilding
its exact scaffold and all three supervised targets from the pinned catalogue.

Three empirical measures are deliberately distinct.  ``UNIQUE_COMPOSITION``
selects one canonical representative of every reduced composition and is the
safe default for fitting ``p_omega(c)``.  ``STRUCTURE`` visits every retained
polymorph once; it is the intended measure for cover and geometry training and
exposes the exact inverse composition multiplicity needed by a joint objective.
``SYSTEM_FORMULA_POLYMORPH`` is the equation (3.15a) search measure of the
discovery-oriented S-DNG checkpoint: it visits every record once with exact
system-, formula- and polymorph-balanced weights.
The selected measure is part of every sampler fingerprint and resume state.

All crystallographic continuous values remain NumPy float64.  This module has
no mixed-precision boundary and performs no float32 crystallographic check.
"""

from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, replace
from enum import Enum
from fractions import Fraction
from hashlib import sha256
from types import MappingProxyType
from typing import Final, TypeAlias, cast

import numpy as np
import numpy.typing as npt

from .catalogue import CertifiedCatalogue
from .composition import Composition
from .config import DomainConfig
from .cover import canonical_rows
from .dataset_builder import canonical_training_target_fingerprint
from .dataset_splits import SplitMode
from .dataset_storage import DatasetStore, StoredTrainingRecord
from .decoder import CrystalScaffold, OrbitAssignment
from .errors import DomainError, InvariantViolation
from .tokenization import CompositionTokenCodec
from .training_targets import (
    CanonicalCatalogueTrainingTarget,
    ExactFixtureTrainingTarget,
    lift_canonical_catalogue_target,
    lift_exact_fixture_target,
)

IntArray: TypeAlias = npt.NDArray[np.int64]
FloatArray: TypeAlias = npt.NDArray[np.float64]
BoolArray: TypeAlias = npt.NDArray[np.bool_]
CanonicalTarget: TypeAlias = CanonicalCatalogueTrainingTarget | ExactFixtureTrainingTarget

_STATE_SCHEMA: Final = "crystalatlas.training-batch-sampler-state.v3"
_DATASET_SCHEMA: Final = "crystalatlas.stored-training-dataset.v1"
_SAMPLER_SCHEMA: Final = "crystalatlas.deterministic-bucket-sampler.v3"
_EPOCH_SCHEMA: Final = "crystalatlas.deterministic-bucket-epoch.v2"
_MIXED_SAMPLER_SCHEMA: Final = "crystalatlas.mixed-topology-batch-sampler.v1"
_DIGEST = re.compile(r"[0-9a-f]{64}\Z")
_PARTITIONS = frozenset({"train", "validation", "test"})


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


def _require_digest(value: object, *, name: str) -> str:
    if not isinstance(value, str) or _DIGEST.fullmatch(value) is None:
        raise DomainError(f"{name} must be a lowercase SHA-256 digest")
    return value


def _require_nonnegative_int(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise DomainError(f"{name} must be a nonnegative integer")
    return value


def _readonly_int(value: npt.ArrayLike, *, name: str) -> IntArray:
    array = np.asarray(value, dtype=np.int64).copy(order="C")
    if array.dtype != np.dtype(np.int64):
        raise InvariantViolation(f"{name} did not retain int64 storage")
    array.setflags(write=False)
    return array


def _readonly_float64(value: npt.ArrayLike, *, name: str) -> FloatArray:
    array = np.asarray(value, dtype=np.float64).copy(order="C")
    if array.dtype != np.dtype(np.float64) or not np.all(np.isfinite(array)):
        raise DomainError(f"{name} must contain finite float64 values")
    array.setflags(write=False)
    return array


def _readonly_bool(value: npt.ArrayLike, *, name: str) -> BoolArray:
    array = np.asarray(value, dtype=np.bool_).copy(order="C")
    if array.dtype != np.dtype(np.bool_):
        raise InvariantViolation(f"{name} did not retain Boolean storage")
    array.setflags(write=False)
    return array


def _require_source_array(
    value: npt.NDArray[np.generic],
    *,
    dtype: np.dtype[np.generic],
    ndim: int,
    name: str,
) -> None:
    if value.dtype != dtype or value.ndim != ndim:
        raise DomainError(f"stored {name} must be {dtype} with rank {ndim}")
    if value.flags.writeable:
        raise DomainError(f"stored {name} must be a read-only authenticated view")


def _same_float64(actual: npt.ArrayLike, expected: npt.ArrayLike, *, name: str) -> None:
    left = np.asarray(actual)
    right = np.asarray(expected)
    if (
        left.dtype != np.dtype(np.float64)
        or right.dtype != np.dtype(np.float64)
        or left.shape != right.shape
        or not np.array_equal(left, right)
    ):
        raise InvariantViolation(f"stored {name} disagrees with the rebuilt float64 target")


DERIVED_LATENT_RELATIVE_TOLERANCE: Final = 1.0e-12
"""Semantic agreement bound for the rebuilt invariant lattice latent.

The latent is the only stored target computed through LAPACK eigendecompositions
(``spd_matrix_sqrt``/``spd_matrix_log``).  Different BLAS/LAPACK builds may round
those differently.  Measured on the Mac/numpy-1.26.4-built MP20 pilot store: the
8 portability records differ from a Linux rebuild by at most 4.4e-16, and a
numpy-2.4.6 rebuild of all 193 records differs bitwise in 59 latents by at most
1.9e-15.  Bitwise equality is therefore a property of one numerical stack,
not of the scientific target.  Stored bytes remain authenticated exactly by the
build-manifest target fingerprint; this bound only certifies that the local rebuild
means the same chart point.  It is about 4.5e3 ulp at unit scale, far below any
physical distinction, and a rotated commutant basis or a different chart would
exceed it by many orders of magnitude.
"""


def _same_derived_latent(stored: npt.ArrayLike, rebuilt: npt.ArrayLike) -> None:
    left = np.asarray(stored)
    right = np.asarray(rebuilt)
    if (
        left.dtype != np.dtype(np.float64)
        or right.dtype != np.dtype(np.float64)
        or left.shape != right.shape
    ):
        raise InvariantViolation(
            "stored lattice latent has a different dtype or chart dimension"
        )
    if not right.size:
        return
    scale = max(1.0, float(np.max(np.abs(right))))
    deviation = float(np.max(np.abs(left - right)))
    if not deviation <= DERIVED_LATENT_RELATIVE_TOLERANCE * scale:
        raise InvariantViolation(
            "stored lattice latent disagrees with the rebuilt chart coordinates "
            f"(max deviation {deviation:.3e} > {DERIVED_LATENT_RELATIVE_TOLERANCE:.1e} x "
            f"{scale:.3g}); the chart basis or metric differs, not merely rounding"
        )


class TrainingMeasure(str, Enum):  # noqa: UP042 - stable serialized string enum
    """The probability measure represented by one complete sampler epoch."""

    UNIQUE_COMPOSITION = "unique-composition"
    STRUCTURE = "structure"
    SYSTEM_FORMULA_POLYMORPH = "system-formula-polymorph"
    """Equation (3.15a), the discovery-oriented S-DNG search measure (review m6)."""


def system_formula_polymorph_weights(
    systems: Sequence[str],
    formulas: Sequence[str],
    polymorphs: Sequence[str],
) -> tuple[tuple[Fraction, ...], tuple[Fraction, ...]]:
    """Exact record weights of ``S ~ Unif``, ``c ~ Unif(C(S))``, ``C ~ Unif(P(c))``.

    Returns ``(N p_i, N P(c_i) / n(c_i))`` per record: the structure weight of
    the cover and geometry terms and the composition weight, with ``N`` records,
    ``p_i`` the equation (3.15a) probability, ``P(c)`` the formula mass and
    ``n(c)`` the records of formula ``c``.  Records with one polymorph key share
    that polymorph's mass, which is a reversible grouping, not deduplication.
    A formula must belong to one species system and a polymorph to one formula.
    """

    count = len(systems)
    if count == 0:
        raise DomainError("the search measure needs a nonempty record set")
    if len(formulas) != count or len(polymorphs) != count:
        raise DomainError("system, formula and polymorph keys must align")
    system_of: dict[str, str] = {}
    formula_of: dict[str, str] = {}
    members: dict[str, dict[str, dict[str, int]]] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(int))
    )
    for system, formula, polymorph in zip(systems, formulas, polymorphs, strict=True):
        if system_of.setdefault(formula, system) != system:
            raise DomainError("a reduced formula must belong to one species system")
        if formula_of.setdefault(polymorph, formula) != formula:
            raise DomainError("a polymorph key must belong to one formula")
        members[system][formula][polymorph] += 1
    formula_records = Counter(formulas)
    total = Fraction(count)
    structure: list[Fraction] = []
    composition: list[Fraction] = []
    for system, formula, polymorph in zip(systems, formulas, polymorphs, strict=True):
        formula_mass = Fraction(1, len(members) * len(members[system]))
        polymorph_mass = formula_mass / len(members[system][formula])
        structure.append(total * polymorph_mass / members[system][formula][polymorph])
        composition.append(total * formula_mass / formula_records[formula])
    return tuple(structure), tuple(composition)


@dataclass(frozen=True, slots=True, order=True)
class CoverTopologyKey:
    """Discrete/layout identity required by the current fixed-cover geometry path."""

    target_type: str
    space_group: str
    nu: int
    support_size: int
    row_labels: tuple[str, ...]
    row_multiplicities: tuple[int, ...]
    row_free_dimensions: tuple[int, ...]
    selected_actions: tuple[tuple[int, ...], ...]
    orbit_topology: tuple[tuple[str, int], ...]
    lattice_dimension: int
    intrinsic_dimension: int
    nu_vocabulary_size: int
    group_vocabulary_size: int
    period_basis_sha256: str

    def __post_init__(self) -> None:
        if self.target_type not in {"exact-fixture", "canonical-catalogue"}:
            raise DomainError("cover topology has an unsupported target type")
        if not self.space_group:
            raise DomainError("cover topology needs a pinned space-group key")
        positive = (self.nu, self.support_size, self.nu_vocabulary_size)
        if any(
            isinstance(value, bool) or not isinstance(value, int) or value < 1
            for value in positive
        ):
            raise DomainError("cover topology positive dimensions are malformed")
        nonnegative = (
            self.lattice_dimension,
            self.intrinsic_dimension,
            self.group_vocabulary_size,
        )
        if any(
            isinstance(value, bool) or not isinstance(value, int) or value < 0
            for value in nonnegative
        ):
            raise DomainError("cover topology dimensions are malformed")
        if self.group_vocabulary_size < 1:
            raise DomainError("group selector vocabulary must be nonempty")
        rows = len(self.row_labels)
        if (
            rows == 0
            or len(self.row_multiplicities) != rows
            or len(self.row_free_dimensions) != rows
            or len(self.selected_actions) != rows
        ):
            raise DomainError("cover topology row fields do not align")
        if len(set(self.row_labels)) != rows or any(not value for value in self.row_labels):
            raise DomainError("cover topology row labels must be nonempty and unique")
        if any(value < 1 for value in self.row_multiplicities):
            raise DomainError("cover topology multiplicities must be positive")
        if any(not 0 <= value <= 3 for value in self.row_free_dimensions):
            raise DomainError("cover topology free dimensions must lie in [0,3]")
        if any(
            len(action) != self.support_size or any(value < 0 for value in action)
            for action in self.selected_actions
        ):
            raise DomainError("cover topology occupation actions are malformed")
        if not self.orbit_topology or any(
            not label or not 0 <= species_index < self.support_size
            for label, species_index in self.orbit_topology
        ):
            raise DomainError("cover topology orbit sequence is malformed")
        _require_digest(self.period_basis_sha256, name="period basis fingerprint")

    def to_dict(self) -> dict[str, object]:
        return {
            "target_type": self.target_type,
            "space_group": self.space_group,
            "nu": self.nu,
            "support_size": self.support_size,
            "row_labels": list(self.row_labels),
            "row_multiplicities": list(self.row_multiplicities),
            "row_free_dimensions": list(self.row_free_dimensions),
            "selected_actions": [list(value) for value in self.selected_actions],
            "orbit_topology": [list(value) for value in self.orbit_topology],
            "lattice_dimension": self.lattice_dimension,
            "intrinsic_dimension": self.intrinsic_dimension,
            "nu_vocabulary_size": self.nu_vocabulary_size,
            "group_vocabulary_size": self.group_vocabulary_size,
            "period_basis_sha256": self.period_basis_sha256,
        }


@dataclass(frozen=True, slots=True, order=True)
class TrainingBucketKey:
    """A batching fibre with one atom count and one exact cover topology."""

    atom_count: int
    topology: CoverTopologyKey

    def __post_init__(self) -> None:
        if (
            isinstance(self.atom_count, bool)
            or not isinstance(self.atom_count, int)
            or self.atom_count < 1
        ):
            raise DomainError("training bucket atom_count must be positive")
        if not isinstance(self.topology, CoverTopologyKey):
            raise TypeError("training bucket topology must be CoverTopologyKey")

    def to_dict(self) -> dict[str, object]:
        return {"atom_count": self.atom_count, "topology": self.topology.to_dict()}


@dataclass(frozen=True, slots=True)
class CompositionSupervision:
    """Grammar-exact composition targets and their declared empirical weights."""

    token_ids: IntArray
    token_types: IntArray
    legal_mask: BoolArray
    sequence_mask: BoolArray
    atomic_numbers: IntArray
    reduced_counts: IntArray
    composition_multiplicities: IntArray
    sample_weights: FloatArray

    def __post_init__(self) -> None:
        token_ids = _readonly_int(self.token_ids, name="composition token IDs")
        token_types = _readonly_int(self.token_types, name="composition token types")
        legal = _readonly_bool(self.legal_mask, name="composition legal mask")
        sequence = _readonly_bool(self.sequence_mask, name="composition sequence mask")
        numbers = _readonly_int(self.atomic_numbers, name="composition atomic numbers")
        counts = _readonly_int(self.reduced_counts, name="composition counts")
        multiplicities = _readonly_int(
            self.composition_multiplicities,
            name="composition multiplicities",
        )
        weights = _readonly_float64(self.sample_weights, name="composition weights")
        if token_ids.ndim != 2 or token_types.shape != token_ids.shape:
            raise DomainError("composition tokens must have shape [B,T]")
        batch, length = token_ids.shape
        if legal.ndim != 3 or legal.shape[:2] != (batch, length):
            raise DomainError("composition legal mask must have shape [B,T,V]")
        if sequence.shape != (batch, length):
            raise DomainError("composition sequence mask must have shape [B,T]")
        if numbers.ndim != 2 or counts.shape != numbers.shape or numbers.shape[0] != batch:
            raise DomainError("composition arrays must have aligned shape [B,K]")
        if multiplicities.shape != (batch,) or weights.shape != (batch,):
            raise DomainError("composition weights and multiplicities must have shape [B]")
        if batch < 1 or length < 1 or legal.shape[2] < 1:
            raise DomainError("composition supervision cannot be empty")
        if np.any(numbers < 1) or np.any(numbers > 118) or np.any(counts < 1):
            raise DomainError("composition support/count arrays are outside their domains")
        if np.any(numbers[:, 1:] <= numbers[:, :-1]):
            raise DomainError("composition support must be strictly increasing")
        if np.any(multiplicities < 1) or np.any(weights <= 0.0):
            raise DomainError("composition multiplicities and weights must be positive")
        if np.any(~legal.any(axis=-1)):
            raise DomainError("every composition position must retain a legal token")
        rows = np.arange(batch)[:, None]
        positions = np.arange(length)[None, :]
        if not np.all(legal[rows, positions, token_ids]):
            raise DomainError("a composition target is absent from its exact legal mask")
        for name, value in (
            ("token_ids", token_ids),
            ("token_types", token_types),
            ("legal_mask", legal),
            ("sequence_mask", sequence),
            ("atomic_numbers", numbers),
            ("reduced_counts", counts),
            ("composition_multiplicities", multiplicities),
            ("sample_weights", weights),
        ):
            object.__setattr__(self, name, value)


@dataclass(frozen=True, slots=True)
class CoverSupervision:
    """Dense exact selector trace for one topology-homogeneous batch."""

    nu: IntArray
    nu_legal_mask: BoolArray
    nu_target_indices: IntArray
    group_legal_mask: BoolArray
    group_target_indices: IntArray
    row_labels: tuple[str, ...]
    row_multiplicities: IntArray
    row_free_dimensions: IntArray
    selected_action_indices: IntArray
    selected_actions: IntArray
    candidate_actions: IntArray
    candidate_legal_mask: BoolArray
    residual_before: IntArray
    residual_after: IntArray
    sample_weights: FloatArray

    def __post_init__(self) -> None:
        nu = _readonly_int(self.nu, name="nu")
        nu_legal = _readonly_bool(self.nu_legal_mask, name="nu legal mask")
        nu_targets = _readonly_int(self.nu_target_indices, name="nu target indices")
        group_legal = _readonly_bool(self.group_legal_mask, name="group legal mask")
        group_targets = _readonly_int(self.group_target_indices, name="group target indices")
        multiplicities = _readonly_int(self.row_multiplicities, name="row multiplicities")
        dimensions = _readonly_int(self.row_free_dimensions, name="row free dimensions")
        selected_indices = _readonly_int(
            self.selected_action_indices, name="selected action indices"
        )
        selected = _readonly_int(self.selected_actions, name="selected actions")
        candidates = _readonly_int(self.candidate_actions, name="candidate actions")
        candidate_legal = _readonly_bool(self.candidate_legal_mask, name="candidate legal mask")
        before = _readonly_int(self.residual_before, name="residual before")
        after = _readonly_int(self.residual_after, name="residual after")
        weights = _readonly_float64(self.sample_weights, name="cover weights")
        if nu.ndim != 1 or nu.shape[0] < 1:
            raise DomainError("nu must have nonempty shape [B]")
        batch = nu.shape[0]
        if nu_legal.ndim != 2 or nu_legal.shape[0] != batch:
            raise DomainError("nu legal masks must have shape [B,N_nu]")
        if nu_targets.shape != (batch,):
            raise DomainError("nu targets must have shape [B]")
        if group_legal.ndim != 2 or group_legal.shape[0] != batch:
            raise DomainError("group legal masks must have shape [B,N_g]")
        if group_targets.shape != (batch,):
            raise DomainError("group targets must have shape [B]")
        rows = len(self.row_labels)
        if rows < 1 or len(set(self.row_labels)) != rows:
            raise DomainError("row labels must be nonempty and unique")
        if multiplicities.shape != (rows,) or dimensions.shape != (rows,):
            raise DomainError("row metadata must have shape [R]")
        if selected_indices.shape != (batch, rows):
            raise DomainError("selected action indices must have shape [B,R]")
        if selected.ndim != 3 or selected.shape[:2] != (batch, rows):
            raise DomainError("selected actions must have shape [B,R,K]")
        support = selected.shape[2]
        if candidates.ndim != 4 or candidates.shape[:2] != (batch, rows):
            raise DomainError("candidate actions must have shape [B,R,A,K]")
        if candidates.shape[3] != support or candidate_legal.shape != candidates.shape[:3]:
            raise DomainError("candidate actions and legal masks do not align")
        if before.shape != selected.shape or after.shape != selected.shape:
            raise DomainError("teacher-forced residuals must align with selected actions")
        if weights.shape != (batch,) or np.any(weights < 0.0):
            raise DomainError("cover weights must be finite nonnegative values")
        if np.any(nu < 1) or np.any(multiplicities < 1):
            raise DomainError("nu and row multiplicities must be positive")
        if np.any(dimensions < 0) or np.any(dimensions > 3):
            raise DomainError("row free dimensions must lie in [0,3]")
        if np.any(selected < 0) or np.any(candidates < 0):
            raise DomainError("selector action tensors cannot contain negative counts")
        if np.any(~nu_legal.any(axis=1)) or np.any(~group_legal.any(axis=1)):
            raise DomainError("every selector decision must retain a legal value")
        batch_indices = np.arange(batch)
        if (
            np.any(nu_targets < 0)
            or np.any(nu_targets >= nu_legal.shape[1])
            or not np.all(nu_legal[batch_indices, nu_targets])
            or not np.array_equal(nu_targets, nu - 1)
        ):
            raise DomainError("nu targets are not exactly legal")
        if (
            np.any(group_targets < 0)
            or np.any(group_targets >= group_legal.shape[1])
            or not np.all(group_legal[batch_indices, group_targets])
        ):
            raise DomainError("space-group targets are not exactly legal")
        if np.any(~candidate_legal.any(axis=2)):
            raise DomainError("every row must retain a completion-safe action")
        if np.any(selected_indices < 0) or np.any(selected_indices >= candidates.shape[2]):
            raise DomainError("a selected row-action index is outside the padded action axis")
        b_grid = np.arange(batch)[:, None]
        r_grid = np.arange(rows)[None, :]
        if not np.all(candidate_legal[b_grid, r_grid, selected_indices]):
            raise DomainError("a selected row action is marked illegal")
        chosen = candidates[b_grid, r_grid, selected_indices]
        if not np.array_equal(chosen, selected):
            raise DomainError("selected row actions disagree with their candidate indices")
        expected_after = before - multiplicities[None, :, None] * selected
        if np.any(expected_after < 0) or not np.array_equal(after, expected_after):
            raise DomainError("selector residuals violate rho' = rho - mu*a")
        if rows > 1 and not np.array_equal(after[:, :-1], before[:, 1:]):
            raise DomainError("selector teacher-forcing trace is discontinuous")
        if np.any(after[:, -1]):
            raise DomainError("selector teacher-forcing trace is not terminal")
        for name, value in (
            ("nu", nu),
            ("nu_legal_mask", nu_legal),
            ("nu_target_indices", nu_targets),
            ("group_legal_mask", group_legal),
            ("group_target_indices", group_targets),
            ("row_multiplicities", multiplicities),
            ("row_free_dimensions", dimensions),
            ("selected_action_indices", selected_indices),
            ("selected_actions", selected),
            ("candidate_actions", candidates),
            ("candidate_legal_mask", candidate_legal),
            ("residual_before", before),
            ("residual_after", after),
            ("sample_weights", weights),
        ):
            object.__setattr__(self, name, value)
        object.__setattr__(self, "row_labels", tuple(self.row_labels))


@dataclass(frozen=True, slots=True)
class GeometrySupervision:
    """Exact scaffolds and float64 continuous coordinates for CFM construction."""

    scaffolds: tuple[CrystalScaffold, ...]
    metric: FloatArray
    lattice_latent: FloatArray
    intrinsic_coordinates: FloatArray
    period_basis: FloatArray
    sample_weights: FloatArray

    def __post_init__(self) -> None:
        scaffolds = tuple(self.scaffolds)
        if not scaffolds or any(not isinstance(item, CrystalScaffold) for item in scaffolds):
            raise DomainError("geometry supervision needs exact CrystalScaffold values")
        metric = _readonly_float64(self.metric, name="metric")
        lattice = _readonly_float64(self.lattice_latent, name="lattice latent")
        intrinsic = _readonly_float64(self.intrinsic_coordinates, name="intrinsic coordinates")
        period = _readonly_float64(self.period_basis, name="period basis")
        weights = _readonly_float64(self.sample_weights, name="geometry weights")
        batch = len(scaffolds)
        if metric.shape != (batch, 3, 3):
            raise DomainError("metrics must have shape [B,3,3]")
        if lattice.ndim != 2 or lattice.shape[0] != batch:
            raise DomainError("lattice latent values must have shape [B,L]")
        if intrinsic.ndim != 2 or intrinsic.shape[0] != batch:
            raise DomainError("intrinsic values must have shape [B,D]")
        if period.shape != (batch, intrinsic.shape[1], intrinsic.shape[1]):
            raise DomainError("period bases must have shape [B,D,D]")
        if weights.shape != (batch,) or np.any(weights < 0.0):
            raise DomainError("geometry weights must be finite nonnegative values")
        if not np.array_equal(metric, np.swapaxes(metric, 1, 2)):
            raise DomainError("batch metrics must be exactly symmetric")
        try:
            eigenvalues = np.linalg.eigvalsh(metric)
        except np.linalg.LinAlgError as exc:
            raise DomainError("batch metric eigendecomposition failed") from exc
        if np.any(eigenvalues <= 0.0):
            raise DomainError("batch metrics must be positive definite")
        object.__setattr__(self, "scaffolds", scaffolds)
        object.__setattr__(self, "metric", metric)
        object.__setattr__(self, "lattice_latent", lattice)
        object.__setattr__(self, "intrinsic_coordinates", intrinsic)
        object.__setattr__(self, "period_basis", period)
        object.__setattr__(self, "sample_weights", weights)


@dataclass(frozen=True, slots=True)
class TrainingBatch:
    """One immutable batch lying entirely inside one exact topology fibre."""

    record_ids: tuple[str, ...]
    target_fingerprints: tuple[str, ...]
    dataset_fingerprint: str
    measure: TrainingMeasure
    bucket_key: TrainingBucketKey
    composition: CompositionSupervision
    cover: CoverSupervision
    geometry: GeometrySupervision
    # N_epoch / B_epoch from the complete rank-independent sampler plan.
    # None denotes an explicitly materialized, non-sampler diagnostic batch.
    optimizer_normalization_count: float | None = None

    def __post_init__(self) -> None:
        normalizer = self.optimizer_normalization_count
        if normalizer is not None and (
            isinstance(normalizer, bool)
            or not isinstance(normalizer, (int, float))
            or not np.isfinite(normalizer)
            or normalizer <= 0.0
        ):
            raise DomainError("optimizer normalization count must be finite and positive")
        record_ids = tuple(self.record_ids)
        target_fingerprints = tuple(self.target_fingerprints)
        if not record_ids or len(set(record_ids)) != len(record_ids):
            raise DomainError("training batch record IDs must be nonempty and unique")
        if len(target_fingerprints) != len(record_ids):
            raise DomainError("target fingerprints must align with record IDs")
        for value in target_fingerprints:
            _require_digest(value, name="target fingerprint")
        _require_digest(
            self.dataset_fingerprint,
            name="training batch dataset fingerprint",
        )
        if not isinstance(self.measure, TrainingMeasure):
            raise TypeError("batch measure must be TrainingMeasure")
        if not isinstance(self.bucket_key, TrainingBucketKey):
            raise TypeError("bucket_key must be TrainingBucketKey")
        batch = len(record_ids)
        if (
            self.composition.token_ids.shape[0] != batch
            or self.cover.nu.shape[0] != batch
            or len(self.geometry.scaffolds) != batch
        ):
            raise DomainError("training batch components have different batch dimensions")
        if self.measure is TrainingMeasure.UNIQUE_COMPOSITION:
            if np.any(self.cover.sample_weights) or np.any(self.geometry.sample_weights):
                raise DomainError(
                    "unique-composition batches must disable cover/geometry objectives"
                )
            if not np.all(self.composition.sample_weights == 1.0):
                raise DomainError("unique-composition batches require unit composition weights")
        elif self.measure is TrainingMeasure.STRUCTURE:
            if not np.all(self.cover.sample_weights == 1.0) or not np.all(
                self.geometry.sample_weights == 1.0
            ):
                raise DomainError("structure batches require unit structure weights")
        else:
            for weights in (
                self.composition.sample_weights,
                self.cover.sample_weights,
                self.geometry.sample_weights,
            ):
                if not np.all(np.isfinite(weights)) or not np.all(weights > 0.0):
                    raise DomainError("search-measure weights must be finite and positive")
            if not np.array_equal(self.cover.sample_weights, self.geometry.sample_weights):
                raise DomainError("search-measure cover and geometry weights must agree")
        object.__setattr__(self, "record_ids", record_ids)
        object.__setattr__(self, "target_fingerprints", target_fingerprints)


@dataclass(frozen=True, slots=True)
class TrainingRecordDescriptor:
    """Small immutable index entry produced only after exact target rebuilding."""

    dataset_index: int
    record_id: str
    composition_key: str
    bucket_key: TrainingBucketKey


@dataclass(frozen=True, slots=True)
class _ValidatedRecord:
    descriptor: TrainingRecordDescriptor
    record: StoredTrainingRecord
    composition: Composition
    scaffold: CrystalScaffold


def _target_for_record(
    record: StoredTrainingRecord,
    scaffold: CrystalScaffold,
    *,
    catalogue: CertifiedCatalogue,
    domain: DomainConfig,
) -> CanonicalTarget:
    if record.target_type == "exact-fixture":
        return lift_exact_fixture_target(
            scaffold,
            record.metric,
            catalogue=catalogue,
            domain=domain,
            num_space_groups=len(record.group_legal_mask),
        )
    if record.target_type == "canonical-catalogue":
        return lift_canonical_catalogue_target(
            scaffold,
            record.metric,
            catalogue=catalogue,
            domain=domain,
            num_space_groups=len(record.group_legal_mask),
        )
    raise DomainError("stored record has an unsupported target type")


def _require_rebuilt_target(
    record: StoredTrainingRecord,
    target: CanonicalTarget,
    scaffold: CrystalScaffold,
) -> None:
    if target.canonical_scaffold != scaffold:
        raise InvariantViolation(
            "stored scaffold is not canonical under the pinned target lift"
        )
    if (
        tuple(record.atomic_numbers.tolist()) != target.composition.support
        or tuple(record.composition_coefficients.tolist()) != target.composition.coefficients
    ):
        raise InvariantViolation("stored composition disagrees with rebuilt target")
    if record.nu != target.cover.nu or record.space_group != target.cover.space_group:
        raise InvariantViolation("stored cover identity disagrees with rebuilt target")
    _same_float64(record.metric, target.metric, name="metric")
    _same_derived_latent(record.lattice_latent, target.lattice_latent)
    _same_float64(
        record.intrinsic_coordinates,
        target.intrinsic_coordinates,
        name="intrinsic coordinates",
    )
    _same_float64(record.period_basis, target.period_basis, name="period basis")
    decisions = target.cover_decisions
    if (
        tuple(record.nu_legal_mask.tolist()) != decisions.nu_legal_mask
        or record.nu_target_index != decisions.nu_target_index
        or tuple(record.group_legal_mask.tolist()) != decisions.group_legal_mask
        or record.group_target_index != decisions.group_target_index
        or len(record.row_decisions) != len(decisions.row_decisions)
    ):
        raise InvariantViolation("stored categorical target disagrees with rebuilt target")
    for actual, expected in zip(record.row_decisions, decisions.row_decisions, strict=True):
        if (
            actual.row_label != expected.row.label
            or actual.multiplicity != expected.row.multiplicity
            or actual.free_dimension != expected.row.free_dimension
            or actual.selected_action_index != expected.selected_action_index
            or tuple(actual.selected_action.tolist()) != expected.selected_action.counts
            or tuple(tuple(row) for row in actual.legal_actions.tolist())
            != tuple(action.counts for action in expected.legal_actions)
            or tuple(actual.residual_before.tolist()) != expected.residual_before
            or tuple(actual.residual_after.tolist()) != expected.residual_after
        ):
            raise InvariantViolation(
                "stored teacher-forced selector trace disagrees with rebuilt target"
            )


def _topology_key(
    record: StoredTrainingRecord,
    composition: Composition,
    scaffold: CrystalScaffold,
) -> TrainingBucketKey:
    support_index = {
        atomic_number: index for index, atomic_number in enumerate(composition.support)
    }
    orbit_topology = tuple(
        (orbit.wyckoff_label, support_index[int(orbit.species)]) for orbit in scaffold.orbits
    )
    period = np.asarray(record.period_basis, dtype="<f8", order="C")
    topology = CoverTopologyKey(
        target_type=record.target_type,
        space_group=record.space_group,
        nu=record.nu,
        support_size=len(composition),
        row_labels=tuple(row.row_label for row in record.row_decisions),
        row_multiplicities=tuple(row.multiplicity for row in record.row_decisions),
        row_free_dimensions=tuple(row.free_dimension for row in record.row_decisions),
        selected_actions=tuple(
            tuple(int(value) for value in row.selected_action) for row in record.row_decisions
        ),
        orbit_topology=orbit_topology,
        lattice_dimension=len(record.lattice_latent),
        intrinsic_dimension=len(record.intrinsic_coordinates),
        nu_vocabulary_size=len(record.nu_legal_mask),
        group_vocabulary_size=len(record.group_legal_mask),
        period_basis_sha256=sha256(period.tobytes(order="C")).hexdigest(),
    )
    atom_count = record.nu * composition.total_atoms
    return TrainingBucketKey(atom_count, topology)


def _validate_stored_record(
    record: StoredTrainingRecord,
    *,
    dataset_index: int,
    codec: CompositionTokenCodec,
    catalogue: CertifiedCatalogue,
    domain: DomainConfig,
) -> _ValidatedRecord:
    if not isinstance(record, StoredTrainingRecord):
        raise TypeError("training data source returned a non-StoredTrainingRecord")
    _require_source_array(
        cast(npt.NDArray[np.generic], record.atomic_numbers),
        dtype=np.dtype(np.uint8),
        ndim=1,
        name="atomic numbers",
    )
    for name, value, ndim in (
        ("composition coefficients", record.composition_coefficients, 1),
        ("metric", record.metric, 2),
        ("lattice latent", record.lattice_latent, 1),
        ("intrinsic coordinates", record.intrinsic_coordinates, 1),
        ("period basis", record.period_basis, 2),
        ("nu legal mask", record.nu_legal_mask, 1),
        ("group legal mask", record.group_legal_mask, 1),
    ):
        dtype = (
            np.dtype(np.bool_)
            if "mask" in name
            else (
                np.dtype(np.int64)
                if name == "composition coefficients"
                else np.dtype(np.float64)
            )
        )
        _require_source_array(
            cast(npt.NDArray[np.generic], value), dtype=dtype, ndim=ndim, name=name
        )
    if not record.canonicalization.passed:
        raise DomainError("stored record lacks a passed canonicalization receipt")
    if (
        record.identity.catalogue_fingerprint != catalogue.fingerprint
        or record.canonicalization.catalogue_fingerprint != catalogue.fingerprint
    ):
        raise DomainError("stored record belongs to a different exact catalogue")
    setting = catalogue.get(record.space_group)
    if record.canonicalization.hall_symbol != setting.hall_symbol:
        raise DomainError("stored Hall receipt disagrees with the pinned setting")
    if record.group_target_index != setting.international_number - 1:
        raise DomainError("stored group target disagrees with its international number")
    if len(record.atomic_numbers) != len(record.composition_coefficients):
        raise DomainError("stored composition arrays do not align")
    composition = Composition(
        tuple(
            (int(atomic_number), int(coefficient))
            for atomic_number, coefficient in zip(
                record.atomic_numbers,
                record.composition_coefficients,
                strict=True,
            )
        )
    )
    encoded = codec.encode(composition)
    if codec.decode(encoded.token_ids.tolist(), encoded.token_types.tolist()) != composition:
        raise InvariantViolation("composition codec failed its live-prefix round-trip")
    scaffold = CrystalScaffold.from_mapping(
        nu=record.nu,
        space_group=record.space_group,
        reduced_composition=cast(Mapping[int | str, int], composition.as_dict()),
        orbits=tuple(
            OrbitAssignment(orbit.wyckoff_label, orbit.atomic_number, orbit.parameter)
            for orbit in record.orbits
        ),
    )
    expected_rows = canonical_rows(setting)
    if tuple(row.row_label for row in record.row_decisions) != tuple(
        row.label for row in expected_rows
    ):
        raise DomainError("stored selector rows are not in canonical catalogue order")
    target = _target_for_record(
        record,
        scaffold,
        catalogue=catalogue,
        domain=domain,
    )
    _require_rebuilt_target(record, target, scaffold)
    # Byte authentication: hash the stored latent bytes, not this stack's rebuild.
    # Every other target field was just required to be bitwise identical, and the
    # latent was required to mean the same chart point within the declared bound.
    authenticated = replace(target, lattice_latent=np.asarray(record.lattice_latent))
    if canonical_training_target_fingerprint(authenticated) != record.target_fingerprint:
        raise InvariantViolation(
            "rebuilt scientific target fingerprint disagrees with the trusted build manifest"
        )
    descriptor = TrainingRecordDescriptor(
        dataset_index=dataset_index,
        record_id=record.record_id,
        composition_key=record.identity.composition_key,
        bucket_key=_topology_key(record, composition, scaffold),
    )
    return _ValidatedRecord(descriptor, record, composition, scaffold)


def split_partition_record_ids(
    store: DatasetStore,
    mode: SplitMode,
    partition: str,
) -> tuple[str, ...]:
    """Return the authenticated record IDs assigned to one frozen partition."""

    if not isinstance(store, DatasetStore):
        raise TypeError("store must be a DatasetStore")
    if not isinstance(mode, SplitMode):
        raise TypeError("mode must be SplitMode")
    if partition not in _PARTITIONS:
        raise DomainError("partition must be 'train', 'validation', or 'test'")
    manifest = store.split_manifest_for(mode)
    return tuple(
        record_id
        for record_id, assigned in manifest.record_assignments
        if assigned == partition
    )


class StoredTrainingDataset:
    """Fully re-audited training view over an authenticated :class:`DatasetStore`."""

    def __init__(
        self,
        store: DatasetStore,
        codec: CompositionTokenCodec,
        catalogue: CertifiedCatalogue,
        *,
        domain: DomainConfig | None = None,
        record_ids: Sequence[str] | None = None,
    ) -> None:
        if not isinstance(store, DatasetStore):
            raise TypeError("store must be a DatasetStore")
        if not isinstance(codec, CompositionTokenCodec):
            raise TypeError("codec must be a CompositionTokenCodec")
        if not isinstance(catalogue, CertifiedCatalogue):
            raise TypeError("catalogue must be a CertifiedCatalogue")
        selected_domain = DomainConfig() if domain is None else domain
        if not isinstance(selected_domain, DomainConfig):
            raise TypeError("domain must be a DomainConfig")
        if store.build_manifest.catalogue_fingerprint != catalogue.fingerprint:
            raise DomainError("dataset build manifest belongs to a different catalogue")
        available = frozenset(store.record_ids)
        if record_ids is None:
            selected_ids = tuple(sorted(available))
        else:
            raw_ids = tuple(record_ids)
            if not raw_ids or any(not isinstance(value, str) or not value for value in raw_ids):
                raise DomainError("training record subset must contain nonempty identifiers")
            if len(set(raw_ids)) != len(raw_ids):
                raise DomainError("training record subset contains duplicate identifiers")
            missing = sorted(set(raw_ids).difference(available))
            if missing:
                raise DomainError(f"training subset contains unknown records: {missing[:3]}")
            selected_ids = tuple(sorted(raw_ids))
        if not selected_ids:
            raise DomainError("training dataset view cannot be empty")

        store_locations = {record_id: index for index, record_id in enumerate(store.record_ids)}
        validated = tuple(
            _validate_stored_record(
                store[record_id],
                dataset_index=index,
                codec=codec,
                catalogue=catalogue,
                domain=selected_domain,
            )
            for index, record_id in enumerate(selected_ids)
        )
        # The descriptor index is local to this view, not the physical shard index.
        for local_index, item in enumerate(validated):
            if item.descriptor.dataset_index != local_index:
                raise InvariantViolation("training view index construction is inconsistent")
            if item.record.record_id not in store_locations:
                raise InvariantViolation("validated record vanished from its store index")
        multiplicities = Counter(item.descriptor.composition_key for item in validated)
        domain_payload = selected_domain.identity_payload()
        self.store = store
        self.codec = codec
        self.catalogue = catalogue
        self.domain = selected_domain
        self._validated = validated
        self._composition_multiplicities = MappingProxyType(dict(multiplicities))
        identities = tuple(item.record.identity for item in validated)
        structure_weights, composition_weights = system_formula_polymorph_weights(
            [identity.element_set_key for identity in identities],
            [identity.composition_key for identity in identities],
            [identity.polymorph_key for identity in identities],
        )
        self._search_measure_weights: Mapping[str, tuple[float, float]] = MappingProxyType(
            {
                item.descriptor.record_id: (float(structure), float(composition))
                for item, structure, composition in zip(
                    validated, structure_weights, composition_weights, strict=True
                )
            }
        )
        self._dataset_fingerprint = _fingerprint(
            {
                "schema": _DATASET_SCHEMA,
                "storage_fingerprint": store.manifest.fingerprint,
                "build_manifest_fingerprint": store.build_manifest.fingerprint,
                "record_ids": list(selected_ids),
                "codec_fingerprint": codec.fingerprint,
                "catalogue_fingerprint": catalogue.fingerprint,
                "domain": domain_payload,
            }
        )

    @property
    def fingerprint(self) -> str:
        return self._dataset_fingerprint

    @property
    def descriptors(self) -> tuple[TrainingRecordDescriptor, ...]:
        return tuple(item.descriptor for item in self._validated)

    @property
    def record_ids(self) -> tuple[str, ...]:
        return tuple(item.descriptor.record_id for item in self._validated)

    @property
    def composition_multiplicities(self) -> Mapping[str, int]:
        return self._composition_multiplicities

    @property
    def unique_composition_count(self) -> int:
        return len(self._composition_multiplicities)

    @property
    def search_measure_weights(self) -> Mapping[str, tuple[float, float]]:
        """Record ID -> (structure, composition) weight under equation (3.15a)."""

        return self._search_measure_weights

    def __len__(self) -> int:
        return len(self._validated)

    def __getitem__(self, index: int) -> TrainingRecordDescriptor:
        if isinstance(index, bool) or not isinstance(index, int):
            raise TypeError("training dataset indices must be integers")
        return self._validated[index].descriptor

    def _collate(
        self,
        indices: tuple[int, ...],
        *,
        measure: TrainingMeasure,
    ) -> TrainingBatch:
        if not indices or len(set(indices)) != len(indices):
            raise DomainError("batch indices must be nonempty and unique")
        try:
            items = tuple(self._validated[index] for index in indices)
        except IndexError as exc:
            raise DomainError("batch index lies outside the training view") from exc
        keys = {item.descriptor.bucket_key for item in items}
        if len(keys) != 1:
            raise DomainError("one training batch cannot cross topology buckets")
        bucket_key = next(iter(keys))
        compositions = tuple(item.composition for item in items)
        encoded = self.codec.collate(compositions)
        atomic_numbers = np.asarray(
            [composition.support for composition in compositions], dtype=np.int64
        )
        reduced_counts = np.asarray(
            [composition.coefficients for composition in compositions], dtype=np.int64
        )
        composition_multiplicities = np.asarray(
            [
                self._composition_multiplicities[item.descriptor.composition_key]
                for item in items
            ],
            dtype=np.int64,
        )
        if measure is TrainingMeasure.UNIQUE_COMPOSITION:
            composition_weights: FloatArray = np.ones(len(items), dtype=np.float64)
            structure_weights: FloatArray = np.zeros(len(items), dtype=np.float64)
        elif measure is TrainingMeasure.STRUCTURE:
            composition_weights = 1.0 / composition_multiplicities.astype(np.float64)
            structure_weights = np.ones(len(items), dtype=np.float64)
        elif measure is TrainingMeasure.SYSTEM_FORMULA_POLYMORPH:
            declared = [
                self._search_measure_weights[item.descriptor.record_id] for item in items
            ]
            structure_weights = np.asarray([value[0] for value in declared], dtype=np.float64)
            composition_weights = np.asarray(
                [value[1] for value in declared], dtype=np.float64
            )
        else:  # pragma: no cover - enum exhaustiveness guard
            raise DomainError("unsupported training measure")
        composition_batch = CompositionSupervision(
            token_ids=encoded.token_ids,
            token_types=encoded.token_types,
            legal_mask=encoded.legal_mask,
            sequence_mask=encoded.sequence_mask,
            atomic_numbers=atomic_numbers,
            reduced_counts=reduced_counts,
            composition_multiplicities=composition_multiplicities,
            sample_weights=composition_weights,
        )

        records = tuple(item.record for item in items)
        rows = len(records[0].row_decisions)
        support = len(records[0].atomic_numbers)
        max_actions = max(
            len(row.legal_actions) for record in records for row in record.row_decisions
        )
        candidate_actions: IntArray = np.zeros(
            (len(records), rows, max_actions, support), dtype=np.int64
        )
        candidate_mask: BoolArray = np.zeros((len(records), rows, max_actions), dtype=np.bool_)
        selected_indices: IntArray = np.empty((len(records), rows), dtype=np.int64)
        selected_actions: IntArray = np.empty((len(records), rows, support), dtype=np.int64)
        residual_before = np.empty_like(selected_actions)
        residual_after = np.empty_like(selected_actions)
        for batch_index, record in enumerate(records):
            for row_index, row in enumerate(record.row_decisions):
                action_count = len(row.legal_actions)
                candidate_actions[batch_index, row_index, :action_count] = row.legal_actions
                candidate_mask[batch_index, row_index, :action_count] = True
                selected_indices[batch_index, row_index] = row.selected_action_index
                selected_actions[batch_index, row_index] = row.selected_action
                residual_before[batch_index, row_index] = row.residual_before
                residual_after[batch_index, row_index] = row.residual_after
        cover_batch = CoverSupervision(
            nu=np.asarray([record.nu for record in records], dtype=np.int64),
            nu_legal_mask=np.stack([record.nu_legal_mask for record in records]),
            nu_target_indices=np.asarray(
                [record.nu_target_index for record in records], dtype=np.int64
            ),
            group_legal_mask=np.stack([record.group_legal_mask for record in records]),
            group_target_indices=np.asarray(
                [record.group_target_index for record in records], dtype=np.int64
            ),
            row_labels=tuple(row.row_label for row in records[0].row_decisions),
            row_multiplicities=np.asarray(
                [row.multiplicity for row in records[0].row_decisions], dtype=np.int64
            ),
            row_free_dimensions=np.asarray(
                [row.free_dimension for row in records[0].row_decisions], dtype=np.int64
            ),
            selected_action_indices=selected_indices,
            selected_actions=selected_actions,
            candidate_actions=candidate_actions,
            candidate_legal_mask=candidate_mask,
            residual_before=residual_before,
            residual_after=residual_after,
            sample_weights=structure_weights,
        )
        geometry_batch = GeometrySupervision(
            scaffolds=tuple(item.scaffold for item in items),
            metric=np.stack([record.metric for record in records]),
            lattice_latent=np.stack([record.lattice_latent for record in records]),
            intrinsic_coordinates=np.stack(
                [record.intrinsic_coordinates for record in records]
            ),
            period_basis=np.stack([record.period_basis for record in records]),
            sample_weights=structure_weights,
        )
        return TrainingBatch(
            record_ids=tuple(record.record_id for record in records),
            target_fingerprints=tuple(record.target_fingerprint for record in records),
            dataset_fingerprint=self.fingerprint,
            measure=measure,
            bucket_key=bucket_key,
            composition=composition_batch,
            cover=cover_batch,
            geometry=geometry_batch,
        )


@dataclass(frozen=True, slots=True)
class _PlannedBatch:
    bucket_key: TrainingBucketKey
    indices: tuple[int, ...]
    optimizer_normalization_count: float = 1.0

    def to_dict(self) -> dict[str, object]:
        return {
            "bucket_key": self.bucket_key.to_dict(),
            "indices": list(self.indices),
            "optimizer_normalization_count": self.optimizer_normalization_count,
        }


def _population_normalized_plan(batches: Sequence[_PlannedBatch]) -> tuple[_PlannedBatch, ...]:
    """Attach one fixed record normalizer before distributed rank partitioning.

    A mean per topology batch overweights singleton/short buckets. With
    mu=N/B, one uniformly ranked microbatch contributes sum(loss_i)/mu.
    Every retained record therefore has the same coefficient over an epoch.
    Explicit drop_last changes the retained population; it is not repaired by
    pretending omitted records contributed gradients.
    """

    if not batches:
        return ()
    normalizer = sum(len(batch.indices) for batch in batches) / len(batches)
    return tuple(
        replace(batch, optimizer_normalization_count=normalizer) for batch in batches
    )


@dataclass(frozen=True, slots=True)
class SamplerState:
    """Self-checking cursor sufficient for exact mid-epoch resumption."""

    sampler_fingerprint: str
    common_policy_fingerprint: str
    dataset_fingerprint: str
    partition_rank: int
    world_size: int
    epoch: int
    cursor: int
    epoch_plan_fingerprint: str
    schema: str = _STATE_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != _STATE_SCHEMA:
            raise DomainError("unsupported training sampler state schema")
        _require_digest(self.sampler_fingerprint, name="sampler fingerprint")
        _require_digest(
            self.common_policy_fingerprint,
            name="common sampler policy fingerprint",
        )
        _require_digest(self.dataset_fingerprint, name="training dataset fingerprint")
        _require_digest(self.epoch_plan_fingerprint, name="epoch plan fingerprint")
        _require_nonnegative_int(self.partition_rank, name="sampler partition rank")
        if isinstance(self.world_size, bool) or not isinstance(self.world_size, int):
            raise DomainError("sampler world size must be a positive integer")
        if self.world_size < 1:
            raise DomainError("sampler world size must be a positive integer")
        if self.partition_rank >= self.world_size:
            raise DomainError("sampler partition rank must lie in [0, world_size)")
        _require_nonnegative_int(self.epoch, name="sampler epoch")
        _require_nonnegative_int(self.cursor, name="sampler cursor")

    def to_dict(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "sampler_fingerprint": self.sampler_fingerprint,
            "common_policy_fingerprint": self.common_policy_fingerprint,
            "dataset_fingerprint": self.dataset_fingerprint,
            "partition_rank": self.partition_rank,
            "world_size": self.world_size,
            "epoch": self.epoch,
            "cursor": self.cursor,
            "epoch_plan_fingerprint": self.epoch_plan_fingerprint,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> SamplerState:
        expected = {
            "schema",
            "sampler_fingerprint",
            "common_policy_fingerprint",
            "dataset_fingerprint",
            "partition_rank",
            "world_size",
            "epoch",
            "cursor",
            "epoch_plan_fingerprint",
        }
        if set(payload) != expected:
            raise DomainError("sampler state has missing or unexpected fields")
        return cls(
            schema=cast(str, payload["schema"]),
            sampler_fingerprint=cast(str, payload["sampler_fingerprint"]),
            common_policy_fingerprint=cast(str, payload["common_policy_fingerprint"]),
            dataset_fingerprint=cast(str, payload["dataset_fingerprint"]),
            partition_rank=cast(int, payload["partition_rank"]),
            world_size=cast(int, payload["world_size"]),
            epoch=cast(int, payload["epoch"]),
            cursor=cast(int, payload["cursor"]),
            epoch_plan_fingerprint=cast(str, payload["epoch_plan_fingerprint"]),
        )


class DeterministicBucketBatchSampler(Iterator[TrainingBatch]):
    """Cryptographically ranked epochs with exact, auditable resume cursors.

    Hash ranks, rather than a library PRNG permutation, make epoch ordering
    independent of NumPy/Python random implementation changes.  A batch never
    crosses an atom-count/cover-topology fibre.
    """

    def __init__(
        self,
        dataset: StoredTrainingDataset,
        batch_size: int,
        *,
        seed: int,
        measure: TrainingMeasure = TrainingMeasure.UNIQUE_COMPOSITION,
        drop_last: bool = False,
        rank: int = 0,
        world_size: int = 1,
    ) -> None:
        if not isinstance(dataset, StoredTrainingDataset):
            raise TypeError("dataset must be a StoredTrainingDataset")
        if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size < 1:
            raise DomainError("batch_size must be a positive integer")
        if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 2**64:
            raise DomainError("sampler seed must be an unsigned 64-bit integer")
        if not isinstance(measure, TrainingMeasure):
            raise TypeError("measure must be TrainingMeasure")
        if not isinstance(drop_last, bool):
            raise TypeError("drop_last must be bool")
        if isinstance(world_size, bool) or not isinstance(world_size, int) or world_size < 1:
            raise DomainError("world_size must be a positive integer")
        if isinstance(rank, bool) or not isinstance(rank, int) or not 0 <= rank < world_size:
            raise DomainError("rank must lie in [0, world_size)")
        self.dataset = dataset
        self.batch_size = batch_size
        self.seed = seed
        self.measure = measure
        self.drop_last = drop_last
        self.rank = rank
        self.world_size = world_size
        self._eligible_indices = self._resolve_eligible_indices()
        self._common_policy_fingerprint = _fingerprint(
            {
                "schema": self._sampler_schema,
                "dataset_fingerprint": dataset.fingerprint,
                "batch_size": batch_size,
                "seed": seed,
                "measure": measure.value,
                "drop_last": drop_last,
                "world_size": world_size,
                "eligible_record_ids": [
                    dataset.descriptors[index].record_id for index in self._eligible_indices
                ],
                **self._policy_extras(),
            }
        )
        self._sampler_fingerprint = _fingerprint(
            {
                "schema": self._sampler_schema,
                "common_policy_fingerprint": self._common_policy_fingerprint,
                "rank": rank,
            }
        )
        self._epoch = 0
        self._cursor = 0
        self._plan = self._build_plan(self._epoch)
        if not self._plan:
            raise DomainError("sampler configuration yields no complete training batches")

    _sampler_schema: str = _SAMPLER_SCHEMA

    def _policy_extras(self) -> dict[str, object]:
        """Policy fields beyond the legacy payload; empty for the bucket sampler."""

        return {}

    @property
    def fingerprint(self) -> str:
        return self._sampler_fingerprint

    @property
    def common_policy_fingerprint(self) -> str:
        """Rank-independent identity of the exact distributed data law."""

        return self._common_policy_fingerprint

    @property
    def epoch(self) -> int:
        return self._epoch

    @property
    def cursor(self) -> int:
        return self._cursor

    @property
    def remaining_batches(self) -> int:
        return len(self._plan) - self._cursor

    @property
    def epoch_plan_fingerprint(self) -> str:
        return self._plan_fingerprint(self._epoch, self._plan)

    def __len__(self) -> int:
        return len(self._plan)

    def _resolve_eligible_indices(self) -> tuple[int, ...]:
        descriptors = self.dataset.descriptors
        if self.measure in (
            TrainingMeasure.STRUCTURE,
            TrainingMeasure.SYSTEM_FORMULA_POLYMORPH,
        ):
            return tuple(range(len(descriptors)))
        representatives: dict[str, int] = {}
        for index, descriptor in enumerate(descriptors):
            previous = representatives.get(descriptor.composition_key)
            if previous is None or descriptor.record_id < descriptors[previous].record_id:
                representatives[descriptor.composition_key] = index
        return tuple(representatives[key] for key in sorted(representatives))

    def _rank(self, *, epoch: int, scope: str, parts: Sequence[object]) -> str:
        return _fingerprint(
            {
                "schema": "crystalatlas.training-batch-rank.v1",
                "seed": self.seed,
                "epoch": epoch,
                "scope": scope,
                "parts": list(parts),
            }
        )

    def _build_plan(self, epoch: int) -> tuple[_PlannedBatch, ...]:
        buckets: dict[TrainingBucketKey, list[int]] = defaultdict(list)
        descriptors = self.dataset.descriptors
        for index in self._eligible_indices:
            buckets[descriptors[index].bucket_key].append(index)
        planned: list[_PlannedBatch] = []
        for key in sorted(buckets):
            ordered = sorted(
                buckets[key],
                key=lambda index: (
                    self._rank(
                        epoch=epoch,
                        scope="record",
                        parts=(descriptors[index].record_id,),
                    ),
                    descriptors[index].record_id,
                ),
            )
            for start in range(0, len(ordered), self.batch_size):
                indices = tuple(ordered[start : start + self.batch_size])
                if len(indices) < self.batch_size and self.drop_last:
                    continue
                planned.append(_PlannedBatch(key, indices))
        planned.sort(
            key=lambda batch: (
                self._rank(
                    epoch=epoch,
                    scope="batch",
                    parts=tuple(descriptors[index].record_id for index in batch.indices),
                ),
                tuple(descriptors[index].record_id for index in batch.indices),
            )
        )
        if self.world_size == 1:
            return _population_normalized_plan(planned)
        remainder = len(planned) % self.world_size
        if remainder:
            target = len(planned) + self.world_size - remainder
            eligible_records = sum(len(batch.indices) for batch in planned)
            if target <= eligible_records:
                # Splitting preserves every record exactly and is preferred to
                # duplicating samples or silently changing the empirical law.
                while len(planned) < target:
                    split_index = next(
                        (
                            index
                            for index, batch in enumerate(planned)
                            if len(batch.indices) > 1
                        ),
                        None,
                    )
                    if split_index is None:  # pragma: no cover - target bound guards this
                        raise InvariantViolation(
                            "distributed batch-count balancing lost a splittable batch"
                        )
                    batch = planned.pop(split_index)
                    planned.append(_PlannedBatch(batch.bucket_key, batch.indices[:-1]))
                    planned.append(_PlannedBatch(batch.bucket_key, batch.indices[-1:]))
                planned.sort(
                    key=lambda batch: (
                        self._rank(
                            epoch=epoch,
                            scope="balanced-batch",
                            parts=tuple(
                                descriptors[index].record_id for index in batch.indices
                            ),
                        ),
                        tuple(descriptors[index].record_id for index in batch.indices),
                    )
                )
            elif self.drop_last:
                keep = len(planned) - remainder
                planned = planned[:keep]
            else:
                raise DomainError(
                    "topology fragmentation prevents equal nonempty DDP steps; "
                    "use an explicit drop_last policy, fewer ranks, or a larger dataset"
                )
        if len(planned) % self.world_size:
            raise InvariantViolation("distributed epoch plan is not rank balanced")
        return _population_normalized_plan(planned)[self.rank :: self.world_size]

    def _plan_fingerprint(self, epoch: int, plan: tuple[_PlannedBatch, ...]) -> str:
        return _fingerprint(
            {
                "schema": _EPOCH_SCHEMA,
                "sampler_fingerprint": self.fingerprint,
                "epoch": epoch,
                "batches": [batch.to_dict() for batch in plan],
            }
        )

    def state(self) -> SamplerState:
        return SamplerState(
            sampler_fingerprint=self.fingerprint,
            common_policy_fingerprint=self.common_policy_fingerprint,
            dataset_fingerprint=self.dataset.fingerprint,
            partition_rank=self.rank,
            world_size=self.world_size,
            epoch=self.epoch,
            cursor=self.cursor,
            epoch_plan_fingerprint=self.epoch_plan_fingerprint,
        )

    def state_dict(self) -> dict[str, object]:
        return self.state().to_dict()

    def load_state_dict(self, payload: Mapping[str, object] | SamplerState) -> None:
        state = (
            payload if isinstance(payload, SamplerState) else SamplerState.from_dict(payload)
        )
        if state.sampler_fingerprint != self.fingerprint:
            raise DomainError("resume state belongs to a different sampler policy")
        if state.common_policy_fingerprint != self.common_policy_fingerprint:
            raise DomainError("resume state belongs to a different common sampler policy")
        if state.dataset_fingerprint != self.dataset.fingerprint:
            raise DomainError("resume state belongs to a different training dataset")
        if state.partition_rank != self.rank or state.world_size != self.world_size:
            raise DomainError("resume state belongs to a different distributed partition")
        plan = self._build_plan(state.epoch)
        expected = self._plan_fingerprint(state.epoch, plan)
        if state.epoch_plan_fingerprint != expected:
            raise DomainError("resume state epoch plan is internally inconsistent")
        if state.cursor > len(plan):
            raise DomainError("resume cursor lies beyond the deterministic epoch plan")
        self._epoch = state.epoch
        self._cursor = state.cursor
        self._plan = plan

    def advance_epoch(self) -> None:
        """Advance only after exhaustion, preventing accidental record skipping."""

        if self._cursor != len(self._plan):
            raise DomainError("cannot advance before the current epoch is exhausted")
        self._epoch += 1
        self._cursor = 0
        self._plan = self._build_plan(self._epoch)
        if not self._plan:
            raise InvariantViolation("a previously nonempty sampler produced an empty epoch")

    def __iter__(self) -> DeterministicBucketBatchSampler:
        return self

    def __next__(self) -> TrainingBatch:
        if self._cursor >= len(self._plan):
            raise StopIteration
        planned = self._plan[self._cursor]
        batch = self.dataset._collate(planned.indices, measure=self.measure)
        if batch.bucket_key != planned.bucket_key:
            raise InvariantViolation("collated batch escaped its deterministic topology fibre")
        self._cursor += 1
        return replace(
            batch, optimizer_normalization_count=planned.optimizer_normalization_count
        )


@dataclass(frozen=True, slots=True)
class MixedTrainingBatch:
    """One optimizer batch spanning several topology fibres (review M5).

    Each part is an ordinary :class:`TrainingBatch`, so every exact
    per-topology invariant is still validated.  The objective evaluates the
    composition and cover losses per part and combines them as record-weighted
    means.  It evaluates geometry in one padded dense call over every record.
    """

    parts: tuple[TrainingBatch, ...]
    optimizer_normalization_count: float | None = None

    def __post_init__(self) -> None:
        parts = tuple(self.parts)
        if not parts or any(not isinstance(part, TrainingBatch) for part in parts):
            raise DomainError("a mixed batch needs one or more TrainingBatch parts")
        if len({part.dataset_fingerprint for part in parts}) != 1:
            raise DomainError("mixed batch parts reference different datasets")
        if len({part.measure for part in parts}) != 1:
            raise DomainError("mixed batch parts use different training measures")
        if len({part.bucket_key for part in parts}) != len(parts):
            raise DomainError("mixed batch parts must lie in distinct topology fibres")
        records = [record for part in parts for record in part.record_ids]
        if len(set(records)) != len(records):
            raise DomainError("a record appears in more than one mixed batch part")
        normalizer = self.optimizer_normalization_count
        if normalizer is not None and (
            isinstance(normalizer, bool)
            or not isinstance(normalizer, (int, float))
            or not np.isfinite(normalizer)
            or normalizer <= 0.0
        ):
            raise DomainError("optimizer normalization count must be finite and positive")
        object.__setattr__(self, "parts", parts)

    @property
    def record_ids(self) -> tuple[str, ...]:
        return tuple(record for part in self.parts for record in part.record_ids)

    @property
    def target_fingerprints(self) -> tuple[str, ...]:
        return tuple(value for part in self.parts for value in part.target_fingerprints)

    @property
    def dataset_fingerprint(self) -> str:
        return self.parts[0].dataset_fingerprint

    @property
    def measure(self) -> TrainingMeasure:
        return self.parts[0].measure

    @property
    def batch_size(self) -> int:
        return sum(len(part.record_ids) for part in self.parts)


@dataclass(frozen=True, slots=True)
class _PlannedMixedBatch:
    indices: tuple[int, ...]
    optimizer_normalization_count: float = 1.0

    def to_dict(self) -> dict[str, object]:
        return {
            "indices": list(self.indices),
            "optimizer_normalization_count": self.optimizer_normalization_count,
        }


class MixedTopologyBatchSampler(DeterministicBucketBatchSampler):
    """Hash-ranked epochs whose batches cross topology fibres (review M5).

    Records are ranked by the same cryptographic rule as the bucket sampler,
    but over the whole eligible set, and consecutive ranks fill batches.  A
    batch is closed early only when adding a record would exceed
    ``max_dense_pair_interactions`` for the padded dense call
    (``len(batch) * max_atoms^2``).  Every record appears once per epoch, and
    each one carries the same population normalizer ``N_epoch / B_epoch``.
    Resume, rank partitioning, and epoch advancement are inherited.
    """

    _sampler_schema = _MIXED_SAMPLER_SCHEMA

    def __init__(
        self,
        dataset: StoredTrainingDataset,
        batch_size: int,
        *,
        seed: int,
        measure: TrainingMeasure = TrainingMeasure.UNIQUE_COMPOSITION,
        drop_last: bool = False,
        rank: int = 0,
        world_size: int = 1,
        max_dense_pair_interactions: int = 500_000,
    ) -> None:
        if (
            isinstance(max_dense_pair_interactions, bool)
            or not isinstance(max_dense_pair_interactions, int)
            or max_dense_pair_interactions < 1
        ):
            raise DomainError("max_dense_pair_interactions must be a positive integer")
        self.max_dense_pair_interactions = max_dense_pair_interactions
        super().__init__(
            dataset,
            batch_size,
            seed=seed,
            measure=measure,
            drop_last=drop_last,
            rank=rank,
            world_size=world_size,
        )

    def _policy_extras(self) -> dict[str, object]:
        return {"max_dense_pair_interactions": self.max_dense_pair_interactions}

    def _build_plan(self, epoch: int) -> tuple[_PlannedBatch, ...]:  # type: ignore[override]
        descriptors = self.dataset.descriptors
        ordered = sorted(
            self._eligible_indices,
            key=lambda index: (
                self._rank(
                    epoch=epoch, scope="mixed-record", parts=(descriptors[index].record_id,)
                ),
                descriptors[index].record_id,
            ),
        )
        budget = self.max_dense_pair_interactions
        planned: list[_PlannedMixedBatch] = []
        current: list[int] = []
        width = 0
        for index in ordered:
            atoms = descriptors[index].bucket_key.atom_count
            if atoms * atoms > budget:
                raise DomainError(
                    f"record {descriptors[index].record_id} alone exceeds the dense pair "
                    f"budget ({atoms}^2 > {budget})"
                )
            wider = max(width, atoms)
            if current and (
                len(current) == self.batch_size or (len(current) + 1) * wider * wider > budget
            ):
                planned.append(_PlannedMixedBatch(tuple(current)))
                current, wider = [], atoms
            current.append(index)
            width = wider
        if current and not (self.drop_last and len(current) < self.batch_size):
            planned.append(_PlannedMixedBatch(tuple(current)))
        if self.world_size > 1:
            remainder = len(planned) % self.world_size
            if remainder:
                target = len(planned) + self.world_size - remainder
                if target <= sum(len(batch.indices) for batch in planned):
                    while len(planned) < target:
                        split = next(
                            index
                            for index, batch in enumerate(planned)
                            if len(batch.indices) > 1
                        )
                        batch = planned.pop(split)
                        planned.append(_PlannedMixedBatch(batch.indices[:-1]))
                        planned.append(_PlannedMixedBatch(batch.indices[-1:]))
                elif self.drop_last:
                    planned = planned[: len(planned) - remainder]
                else:
                    raise DomainError(
                        "too few records for equal nonempty DDP steps; use an explicit "
                        "drop_last policy, fewer ranks, or a larger dataset"
                    )
            if len(planned) % self.world_size:
                raise InvariantViolation("distributed mixed epoch plan is not rank balanced")
        normalized = _population_normalized_plan(planned)  # type: ignore[arg-type]
        return normalized[self.rank :: self.world_size]

    def __next__(self) -> MixedTrainingBatch:  # type: ignore[override]
        if self._cursor >= len(self._plan):
            raise StopIteration
        planned = self._plan[self._cursor]
        descriptors = self.dataset.descriptors
        groups: dict[TrainingBucketKey, list[int]] = defaultdict(list)
        for index in planned.indices:
            groups[descriptors[index].bucket_key].append(index)
        parts = []
        for key in sorted(groups):
            part = self.dataset._collate(tuple(groups[key]), measure=self.measure)
            if part.bucket_key != key:
                raise InvariantViolation("collated part escaped its topology fibre")
            parts.append(part)
        self._cursor += 1
        return MixedTrainingBatch(
            tuple(parts), optimizer_normalization_count=planned.optimizer_normalization_count
        )


__all__ = [
    "CompositionSupervision",
    "CoverSupervision",
    "CoverTopologyKey",
    "DeterministicBucketBatchSampler",
    "GeometrySupervision",
    "MixedTopologyBatchSampler",
    "MixedTrainingBatch",
    "SamplerState",
    "StoredTrainingDataset",
    "TrainingBatch",
    "TrainingBucketKey",
    "TrainingMeasure",
    "TrainingRecordDescriptor",
    "split_partition_record_ids",
]
