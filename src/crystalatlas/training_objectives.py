"""Exact empirical objectives connecting stored batches to the three models.

The dataset exposes two different measures.  Composition pretraining samples
one representative per reduced formula.  Cover and geometry pretraining sample
retained structures.  During joint fine-tuning, inverse composition
multiplicity and the fixed population ratio ``N_structure / N_composition``
express the composition expectation as an unbiased contribution under uniform
structure sampling (the bucket sampler and trainer carry a fixed population
normalizer, rather than treating unequal topology-batch means equally):

``E_structure[(N_s/N_c) L_comp(c) / n(c)] = E_composition[L_comp(c)]``.

This avoids silently letting database polymorph counts define the learned
composition prior while retaining one common per-structure mean for exact
gradient accumulation.
"""

from __future__ import annotations

import json
import math
from collections import OrderedDict
from collections.abc import Mapping
from dataclasses import dataclass
from hashlib import sha256
from types import MappingProxyType
from typing import Any

import numpy as np

from ._optional import require_torch
from .cover import canonical_rows
from .decoder import ExactCrystalDecoder
from .errors import DomainError, InvariantViolation
from .geometry import ContinuousCover
from .learned import CrystalAtlasModel, LearnedCrystalGenerator
from .nn.bridge import geometry_bridge_numerical_contract
from .nn.cover_sampling import operation_derived_row_features
from .trainer import (
    ComponentLosses,
    ComponentModels,
    CrystallographicAuditResult,
    TrainingPhase,
)
from .training import exact_finite_action_nll
from .training_data import (
    MixedTrainingBatch,
    StoredTrainingDataset,
    TrainingBatch,
    TrainingMeasure,
)

torch = require_torch()
Tensor = torch.Tensor
_STRUCTURE_MEASURES = frozenset(
    (TrainingMeasure.STRUCTURE, TrainingMeasure.SYSTEM_FORMULA_POLYMORPH)
)
# Upper bound on memoized per-record audits (least recently used entries leave first).
AUDIT_MEMO_SIZE = 65_536


@dataclass(frozen=True, slots=True)
class ObjectiveDataMeasure:
    """Frozen finite-population normalizers for joint optimization."""

    structure_count: int
    unique_composition_count: int
    dataset_fingerprint: str
    search_measure_weights: Mapping[str, tuple[float, float]] | None = None
    """Opt-in: record ID -> (structure, composition) weights of equation (3.15a).

    Only an objective built with them accepts system-formula-polymorph batches,
    and it requires each batch weight to equal the declared one exactly.  When
    absent, the serialized measure is the legacy payload."""

    def __post_init__(self) -> None:
        weights = self.search_measure_weights
        if weights is not None:
            if not isinstance(weights, Mapping) or len(weights) != self.structure_count:
                raise DomainError("search-measure weights must cover every structure once")
            for record_id, value in weights.items():
                if (
                    not isinstance(record_id, str)
                    or not isinstance(value, tuple)
                    or len(value) != 2
                    or not all(
                        isinstance(item, float) and math.isfinite(item) and item > 0.0
                        for item in value
                    )
                ):
                    raise DomainError(
                        "search-measure weights must be two finite positive floats per record"
                    )
            object.__setattr__(self, "search_measure_weights", MappingProxyType(dict(weights)))
        for name in ("structure_count", "unique_composition_count"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise DomainError(f"{name} must be a positive integer")
        if self.unique_composition_count > self.structure_count:
            raise DomainError("unique compositions cannot outnumber structures")
        if (
            not isinstance(self.dataset_fingerprint, str)
            or len(self.dataset_fingerprint) != 64
            or any(
                character not in "0123456789abcdef" for character in self.dataset_fingerprint
            )
        ):
            raise DomainError("dataset_fingerprint must be a lowercase SHA-256 digest")

    @classmethod
    def from_dataset(
        cls, dataset: StoredTrainingDataset, *, search_measure: bool = False
    ) -> ObjectiveDataMeasure:
        if not isinstance(dataset, StoredTrainingDataset):
            raise TypeError("dataset must be a StoredTrainingDataset")
        if not isinstance(search_measure, bool):
            raise TypeError("search_measure must be bool")
        return cls(
            len(dataset),
            dataset.unique_composition_count,
            dataset.fingerprint,
            dataset.search_measure_weights if search_measure else None,
        )

    @property
    def structure_to_composition_ratio(self) -> float:
        return self.structure_count / self.unique_composition_count

    def to_dict(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "dataset_fingerprint": self.dataset_fingerprint,
            "structure_count": self.structure_count,
            "unique_composition_count": self.unique_composition_count,
        }
        if self.search_measure_weights is not None:
            canonical = sorted(
                (record_id, value[0].hex(), value[1].hex())
                for record_id, value in self.search_measure_weights.items()
            )
            payload["search_measure_weights_sha256"] = sha256(
                json.dumps(canonical, separators=(",", ":")).encode("utf-8")
            ).hexdigest()
        return payload

    def require_search_weights(
        self, record_ids: tuple[str, ...], structure: Any, composition: Any
    ) -> None:
        """Refuse batch weights that are not exactly the declared (3.15a) measure."""

        declared = self.search_measure_weights
        if declared is None:
            raise DomainError(
                "this objective was not built with the declared search-measure weights "
                "(ObjectiveDataMeasure.from_dataset(..., search_measure=True))"
            )
        try:
            expected = np.asarray([declared[record_id] for record_id in record_ids])
        except KeyError as exc:
            raise InvariantViolation(
                "batch records lie outside the declared search-measure weights"
            ) from exc
        if not (
            np.array_equal(np.asarray(structure, dtype=np.float64), expected[:, 0])
            and np.array_equal(np.asarray(composition, dtype=np.float64), expected[:, 1])
        ):
            raise InvariantViolation(
                "batch weights differ from the declared search-measure weights"
            )


@dataclass(frozen=True, slots=True)
class CFMDequantization:
    """Strictly positive endpoint-kernel scales required by the CFM law."""

    sigma_lattice: float = 0.05
    sigma_intrinsic: float = 0.03
    intrinsic_kernel: str = "fractional-v1"
    basis_augmentation: str = "none"
    """``none`` (legacy) or ``shear-v1`` (review 2026-10-29, C2): with probability
    ``basis_augmentation_probability`` a record is moved to a neighbouring
    lattice-basis presentation by an admissible elementary shear, so the target
    is smoother across the canonical-domain boundary of P1, P-1 and the
    monoclinic groups.  Images outside the chart are refused."""
    basis_augmentation_probability: float = 0.5
    """``fractional-v1`` (legacy): ``sigma_intrinsic`` scales ``N(0, I)`` noise in the
    stored fractional torus coordinates, so the blur is anisotropic in space and
    grows with the cell (0.9 A along a 30 A axis).  ``cartesian-v2`` (review
    2026-10-29, C1): the noise covariance is ``sigma^2 (J_hat^T P_N J_hat)^-1`` at
    the record, so ``sigma_intrinsic`` is the RMS translation-quotient Cartesian
    displacement per free coordinate in angstrom, the metric of the loss."""

    def __post_init__(self) -> None:
        if (
            not math.isfinite(self.sigma_lattice)
            or not math.isfinite(self.sigma_intrinsic)
            or self.sigma_lattice <= 0.0
            or self.sigma_intrinsic <= 0.0
        ):
            raise DomainError("CFM dequantization scales must be finite and positive")
        if self.intrinsic_kernel not in {"fractional-v1", "cartesian-v2"}:
            raise DomainError("intrinsic_kernel must be fractional-v1 or cartesian-v2")
        if self.basis_augmentation not in {"none", "shear-v1"}:
            raise DomainError("basis_augmentation must be none or shear-v1")
        probability = self.basis_augmentation_probability
        if (
            isinstance(probability, bool)
            or not isinstance(probability, (int, float))
            or not math.isfinite(probability)
            or not 0.0 <= probability <= 1.0
        ):
            raise DomainError("basis_augmentation_probability must lie in [0, 1]")

    def to_dict(self) -> dict[str, str]:
        payload = {
            "sigma_intrinsic_hex": self.sigma_intrinsic.hex(),
            "sigma_lattice_hex": self.sigma_lattice.hex(),
        }
        if self.intrinsic_kernel != "fractional-v1":  # legacy payload unchanged
            payload["intrinsic_kernel"] = self.intrinsic_kernel
        if self.basis_augmentation != "none":
            payload["basis_augmentation"] = self.basis_augmentation
            payload["basis_augmentation_probability_hex"] = float(
                self.basis_augmentation_probability
            ).hex()
        return payload


@dataclass(frozen=True, slots=True)
class CFMBlockWeights:
    """Block weights ``lambda_z, lambda_eta`` of the CFM loss (3.14) (review 2026-10-29, m2).

    The coordinate block is in ``A^2`` per atom (``l_0 = 1 A``) and the lattice
    block is a dimensionless squared strain; their relative weight is a
    hyperparameter with no physical derivation.  These weights make it explicit
    and recorded.  They are current-state constants, so Theorem 2.9 keeps the
    population minimizer.  Unit weights are the legacy loss, bitwise, and leave
    the objective contract unchanged.
    """

    lattice: float = 1.0
    coordinates: float = 1.0

    def __post_init__(self) -> None:
        for value in (self.lattice, self.coordinates):
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or value <= 0.0
            ):
                raise DomainError("CFM block weights must be finite and positive")

    def as_tuple(self) -> tuple[float, float]:
        return (float(self.lattice), float(self.coordinates))

    @property
    def is_unity(self) -> bool:
        return float(self.lattice) == 1.0 and float(self.coordinates) == 1.0

    def to_dict(self) -> dict[str, str]:
        return {
            "lattice_hex": float(self.lattice).hex(),
            "coordinates_hex": float(self.coordinates).hex(),
        }


@dataclass(frozen=True, slots=True)
class CoverFactorTerms:
    """Per-structure terms of the cover factorization (equations (3.4)--(3.6)).

    The teacher-forced cover negative log-likelihood of one structure is the sum
    ``nu_nll + group_nll + row_nll`` with

    ``nu_nll = -log p(nu | c)``, ``group_nll = -log p(G | c, nu)`` and
    ``row_nll = -sum_r log p(a_r | c, nu, G, a_<r)``,

    each normalized over its exact legal action set.  The ``*_reference`` entries
    are the same sums for the policy that is uniform on every legal set,
    ``log |A_legal|``, so ``reference - nll`` is the information the selector
    holds about that decision.  ``nu_rank`` and ``group_rank`` count the legal
    actions whose logit is strictly larger than the target's (0: the target is a
    maximum); ``row_errors`` counts the rows whose target is not a maximum.
    """

    nu_nll: Tensor
    group_nll: Tensor
    row_nll: Tensor
    nu_reference: Tensor
    group_reference: Tensor
    row_reference: Tensor
    nu_rank: Tensor
    group_rank: Tensor
    row_errors: Tensor


@dataclass(frozen=True, slots=True)
class _CoverLogits:
    nu_logits: Tensor
    group_logits: Tensor
    row_logits: Tensor
    nu_targets: Tensor
    group_targets: Tensor
    row_targets: Tensor
    nu_legal: Tensor
    group_legal: Tensor
    row_legal: Tensor


def _strictly_better(logits: Tensor, targets: Tensor, legal: Tensor) -> Tensor:
    """Number of legal actions with a logit strictly above the target's."""

    target = logits.gather(-1, targets.unsqueeze(-1))
    return ((logits > target) & legal).sum(dim=-1)


class ExactBatchCrystallographicAudit:
    """Float64 exact-decoder and chart round-trip audit for every training batch.

    The audit of one record is a deterministic function of its scaffold and of
    its stored metric, lattice latent, intrinsic coordinates and period basis.
    With ``memoize=True`` the residuals of a record are cached under exactly
    those inputs (the scaffold and the bytes and shapes of the arrays), so a
    record that recurs unchanged in a later epoch is not decoded again, while
    any changed input is audited afresh.  Failures are never cached.  The
    default recomputes every record of every batch.
    """

    def __init__(self, model: CrystalAtlasModel, *, memoize: bool = False) -> None:
        if not isinstance(model, CrystalAtlasModel):
            raise TypeError("model must be a CrystalAtlasModel")
        if not isinstance(memoize, bool):
            raise TypeError("memoize must be bool")
        self.model = model
        self.decoder = ExactCrystalDecoder(
            model.catalogue,
            metric_tolerance=model.scientific_config.numerics.symmetry_atol,
        )
        self.memoize = memoize
        self._memo: OrderedDict[tuple[object, ...], tuple[np.ndarray, ...]] = OrderedDict()

    def _record_residuals(
        self,
        scaffold: Any,
        metric: np.ndarray,
        latent: np.ndarray,
        stored_eta: np.ndarray,
        period: np.ndarray | None,
    ) -> tuple[np.ndarray, ...]:
        reference_length = self.model.scientific_config.domain.reference_length_angstrom
        decoded = self.decoder.decode(scaffold, metric)
        if decoded.hidden_translation_index != 1:
            raise DomainError(
                "training batch contains a nonprimitive hidden-translation target"
            )
        layout = self.model.layout(ContinuousCover.from_exact_scaffold(scaffold))
        rebuilt_metric = layout.chart.decode_latent(latent)
        residuals = [((rebuilt_metric - metric) / (reference_length**2)).reshape(-1)]
        encoded_eta = layout.encode_exact_parameters(scaffold)
        if encoded_eta.shape != stored_eta.shape:
            raise InvariantViolation("training intrinsic target has the wrong chart dimension")
        if period is not None:
            coefficients = np.linalg.solve(period.T, encoded_eta - stored_eta)
            coefficients -= np.round(coefficients)
            residuals.append(coefficients @ period)
        return tuple(residuals)

    def __call__(
        self, batch: TrainingBatch | MixedTrainingBatch, dtype: Any
    ) -> CrystallographicAuditResult:
        if isinstance(batch, MixedTrainingBatch):
            # Every part keeps its own exact audit; residuals are concatenated.
            results = [self(part, dtype) for part in batch.parts]
            return CrystallographicAuditResult(
                torch.cat([result.residuals for result in results]),
                tolerance=results[0].tolerance,
            )
        if not isinstance(batch, TrainingBatch):
            raise TypeError("batch must be a TrainingBatch or MixedTrainingBatch")
        if dtype is not torch.float64:
            raise TypeError("crystallographic audits must be requested in float64")
        residuals: list[np.ndarray] = []
        for index, scaffold in enumerate(batch.geometry.scaffolds):
            metric = np.asarray(batch.geometry.metric[index], dtype=np.float64)
            latent = np.asarray(batch.geometry.lattice_latent[index], dtype=np.float64)
            stored_eta = np.asarray(
                batch.geometry.intrinsic_coordinates[index], dtype=np.float64
            )
            period = (
                np.asarray(batch.geometry.period_basis[index], dtype=np.float64)
                if stored_eta.size
                else None
            )
            if not self.memoize:
                residuals.extend(
                    self._record_residuals(scaffold, metric, latent, stored_eta, period)
                )
                continue
            key = (
                scaffold,
                metric.shape,
                metric.tobytes(),
                latent.shape,
                latent.tobytes(),
                stored_eta.shape,
                stored_eta.tobytes(),
                None if period is None else (period.shape, period.tobytes()),
            )
            cached = self._memo.get(key)
            if cached is None:
                cached = self._record_residuals(scaffold, metric, latent, stored_eta, period)
                for value in cached:
                    value.setflags(write=False)
                self._memo[key] = cached
                if len(self._memo) > AUDIT_MEMO_SIZE:
                    self._memo.popitem(last=False)
            else:
                self._memo.move_to_end(key)
            residuals.extend(cached)
        values = np.concatenate(residuals) if residuals else np.zeros(1, dtype=np.float64)
        return CrystallographicAuditResult(
            torch.tensor(values, dtype=torch.float64),
            tolerance=self.model.scientific_config.numerics.roundtrip_atol,
        )


def _as_tensor(value: np.ndarray, *, device: Any, dtype: Any) -> Tensor:
    # Stored batches are deliberately read-only.  ``torch.as_tensor`` aliases
    # NumPy memory and emits an undefined-behaviour warning for such arrays;
    # training tensors must own writable storage because downstream kernels are
    # not required to preserve alias immutability.
    return torch.tensor(value, device=device, dtype=dtype)


class CrystalAtlasTrainingObjective:
    """Trainer callback implementing factor-wise and joint empirical risks.

    CFM randomness deliberately uses PyTorch's per-rank global CPU/CUDA RNG.
    :class:`ScalableTrainer` checkpoints those streams exactly; accepting an
    arbitrary private generator here would create an untracked resume state.
    """

    def __init__(
        self,
        model: CrystalAtlasModel,
        data_measure: ObjectiveDataMeasure,
        *,
        dequantization: CFMDequantization | None = None,
        block_weights: CFMBlockWeights | None = None,
    ) -> None:
        if not isinstance(model, CrystalAtlasModel):
            raise TypeError("model must be a CrystalAtlasModel")
        if not isinstance(data_measure, ObjectiveDataMeasure):
            raise TypeError("data_measure must be ObjectiveDataMeasure")
        if block_weights is not None and not isinstance(block_weights, CFMBlockWeights):
            raise TypeError("block_weights must be CFMBlockWeights")
        self.model = model
        self.data_measure = data_measure
        self.dequantization = dequantization or CFMDequantization()
        self.block_weights = block_weights or CFMBlockWeights()
        self._device_dtype()
        self._geometry = LearnedCrystalGenerator(model, allow_untrained=True)

    @property
    def contract(self) -> dict[str, object]:
        """Exact semantics that must remain fixed across checkpoint resume."""

        contract: dict[str, object] = {
            "schema": "crystalatlas.training-objective.v4",
            "geometry_bridge_numerics": geometry_bridge_numerical_contract(),
            "checkpointable": True,
            "cfm_gauge_policy": "pushforward-complete-noisy-path-and-tangent.v1",
            "sampling_normalization": "sampler-population-fixed-nominal-window.v1",
            "catalogue_fingerprint": self.model.catalogue.fingerprint,
            "data_measure": self.data_measure.to_dict(),
            "dequantization": self.dequantization.to_dict(),
            "scientific_config_fingerprint": self.model.scientific_config.fingerprint,
        }
        if not self.block_weights.is_unity:  # legacy contract unchanged
            contract["block_weights"] = self.block_weights.to_dict()
        return contract

    @property
    def contract_fingerprint(self) -> str:
        payload = json.dumps(
            self.contract,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        return sha256(payload.encode("utf-8")).hexdigest()

    def _device_dtype(self) -> tuple[Any, Any]:
        parameters = tuple(self.model.parameters())
        if not parameters:
            raise DomainError("training objective model has no parameters")
        devices = {parameter.device for parameter in parameters}
        dtypes = {parameter.dtype for parameter in parameters}
        if len(devices) != 1 or len(dtypes) != 1:
            raise DomainError(
                "training objective requires one device and one master parameter dtype"
            )
        dtype = next(iter(dtypes))
        if dtype not in {torch.float32, torch.float64}:
            raise DomainError("training objective master dtype must be float32 or float64")
        return next(iter(devices)), dtype

    def _composition_loss(self, module: Any, batch: TrainingBatch) -> Tensor:
        device, _ = self._device_dtype()
        supervision = batch.composition
        tokens = _as_tensor(supervision.token_ids, device=device, dtype=torch.long)
        token_types = _as_tensor(supervision.token_types, device=device, dtype=torch.long)
        legal = _as_tensor(supervision.legal_mask, device=device, dtype=torch.bool)
        sequence = _as_tensor(supervision.sequence_mask, device=device, dtype=torch.bool)
        per_token = module(
            tokens,
            token_types,
            legal_mask=legal,
            sequence_mask=sequence,
        )
        if per_token.shape != tokens.shape:
            raise InvariantViolation("composition module returned the wrong loss shape")
        per_sample = (per_token * sequence.to(dtype=per_token.dtype)).sum(dim=-1)
        weights = _as_tensor(
            supervision.sample_weights,
            device=device,
            dtype=per_sample.dtype,
        )
        if batch.measure is TrainingMeasure.UNIQUE_COMPOSITION:
            if not bool(torch.all(weights == 1.0).item()):
                raise InvariantViolation("unique-composition weights are not unity")
            return per_sample.mean()
        if batch.measure is TrainingMeasure.SYSTEM_FORMULA_POLYMORPH:
            self._require_search_weights(batch)
            return (weights * per_sample).mean()
        if batch.measure is not TrainingMeasure.STRUCTURE:
            raise DomainError("unknown training measure")
        ratio = self.data_measure.structure_to_composition_ratio
        return (ratio * weights * per_sample).mean()

    def _require_search_weights(self, batch: TrainingBatch) -> None:
        self.data_measure.require_search_weights(
            tuple(batch.record_ids),
            batch.cover.sample_weights,
            batch.composition.sample_weights,
        )

    def _cover_logits(self, module: Any, batch: TrainingBatch) -> _CoverLogits:
        if batch.measure not in _STRUCTURE_MEASURES:
            raise DomainError("cover training requires a structure measure")
        device, dtype = self._device_dtype()
        composition = batch.composition
        cover = batch.cover
        numbers = _as_tensor(composition.atomic_numbers, device=device, dtype=torch.long)
        counts = _as_tensor(composition.reduced_counts, device=device, dtype=dtype)
        nu_targets = _as_tensor(cover.nu_target_indices, device=device, dtype=torch.long)
        group_targets = _as_tensor(cover.group_target_indices, device=device, dtype=torch.long)
        setting = self.model.catalogue.get(batch.bucket_key.topology.space_group)
        row_specs = canonical_rows(setting)
        if tuple(item.label for item in row_specs) != cover.row_labels:
            raise InvariantViolation("batch row labels disagree with the live catalogue")
        reference = counts
        row_features = (
            operation_derived_row_features(
                setting,
                row_specs,
                feature_dim=self.model.cover_selector.config.row_feature_dim,
                reference=reference,
                geometry=self.model.cover_selector.config.row_geometry_features,
            )
            .unsqueeze(0)
            .expand(numbers.shape[0], -1, -1)
        )
        nu_legal = _as_tensor(cover.nu_legal_mask, device=device, dtype=torch.bool)
        group_legal = _as_tensor(cover.group_legal_mask, device=device, dtype=torch.bool)
        candidate_legal = _as_tensor(
            cover.candidate_legal_mask, device=device, dtype=torch.bool
        )
        selected_actions = _as_tensor(cover.selected_actions, device=device, dtype=dtype)
        candidates = _as_tensor(cover.candidate_actions, device=device, dtype=dtype)
        nu_logits, group_logits, row_logits = module(
            numbers,
            counts,
            nu_targets,
            group_targets,
            row_features,
            selected_actions,
            candidates,
            nu_legal_mask=nu_legal,
            group_legal_mask=group_legal,
            candidate_legal_mask=candidate_legal,
        )
        row_targets = _as_tensor(cover.selected_action_indices, device=device, dtype=torch.long)
        return _CoverLogits(
            nu_logits=nu_logits,
            group_logits=group_logits,
            row_logits=row_logits,
            nu_targets=nu_targets,
            group_targets=group_targets,
            row_targets=row_targets,
            nu_legal=nu_legal,
            group_legal=group_legal,
            row_legal=candidate_legal,
        )

    @staticmethod
    def _cover_factor_nll(terms: _CoverLogits) -> tuple[Tensor, Tensor, Tensor]:
        nu_nll = exact_finite_action_nll(
            terms.nu_logits,
            terms.nu_targets,
            legal_actions=terms.nu_legal,
            reduction="none",
        )
        group_nll = exact_finite_action_nll(
            terms.group_logits,
            terms.group_targets,
            legal_actions=terms.group_legal,
            reduction="none",
        )
        row_nll = exact_finite_action_nll(
            terms.row_logits,
            terms.row_targets,
            legal_actions=terms.row_legal,
            reduction="none",
        ).sum(dim=-1)
        return nu_nll, group_nll, row_nll

    def cover_factor_terms(self, module: Any, batch: TrainingBatch) -> CoverFactorTerms:
        """Per-structure decomposition of the cover loss of one topology batch.

        A diagnostic of the same teacher-forced forward pass as the cover loss:
        the three negative log-likelihoods add up to the per-structure loss whose
        mean the cover phase minimizes.
        """

        if not isinstance(batch, TrainingBatch):
            raise TypeError("cover factor terms are defined for one TrainingBatch")
        terms = self._cover_logits(module, batch)
        nu_nll, group_nll, row_nll = self._cover_factor_nll(terms)
        dtype = nu_nll.dtype
        row_best = _strictly_better(terms.row_logits, terms.row_targets, terms.row_legal)
        return CoverFactorTerms(
            nu_nll=nu_nll,
            group_nll=group_nll,
            row_nll=row_nll,
            nu_reference=terms.nu_legal.sum(dim=-1).to(dtype=dtype).log(),
            group_reference=terms.group_legal.sum(dim=-1).to(dtype=dtype).log(),
            row_reference=terms.row_legal.sum(dim=-1).to(dtype=dtype).log().sum(dim=-1),
            nu_rank=_strictly_better(terms.nu_logits, terms.nu_targets, terms.nu_legal),
            group_rank=_strictly_better(
                terms.group_logits, terms.group_targets, terms.group_legal
            ),
            row_errors=(row_best > 0).sum(dim=-1),
        )

    def _cover_loss(self, module: Any, batch: TrainingBatch) -> Tensor:
        nu_nll, group_nll, row_nll = self._cover_factor_nll(self._cover_logits(module, batch))
        per_structure = nu_nll + group_nll + row_nll
        cover = batch.cover
        weights = _as_tensor(
            cover.sample_weights, device=per_structure.device, dtype=per_structure.dtype
        )
        if batch.measure is TrainingMeasure.SYSTEM_FORMULA_POLYMORPH:
            self._require_search_weights(batch)
            return (weights * per_structure).mean()
        if not bool(torch.all(weights == 1.0).item()):
            raise InvariantViolation("structure-cover weights are not unity")
        return per_structure.mean()

    def _geometry_loss(self, module: Any, batch: TrainingBatch) -> Tensor:
        if batch.measure not in _STRUCTURE_MEASURES:
            raise DomainError("geometry training requires a structure measure")
        device, dtype = self._device_dtype()
        weights = _as_tensor(batch.geometry.sample_weights, device=device, dtype=dtype)
        search = batch.measure is TrainingMeasure.SYSTEM_FORMULA_POLYMORPH
        if search:
            self._require_search_weights(batch)
        elif not bool(torch.all(weights == 1.0).item()):
            raise InvariantViolation("structure-geometry weights are not unity")
        batch_size = len(batch.record_ids)
        time = torch.rand(
            batch_size,
            dtype=dtype,
            device=device,
        )
        result = self._geometry.geometry_cfm_objective_batch(
            batch.geometry.scaffolds,
            tuple(batch.geometry.metric[index] for index in range(batch_size)),
            time=time,
            sigma_z=self.dequantization.sigma_lattice,
            sigma_eta=self.dequantization.sigma_intrinsic,
            generator=None,
            geometry_field=module,
            intrinsic_kernel=self.dequantization.intrinsic_kernel,
            block_weights=self.block_weights.as_tuple(),
            basis_augmentation=self.dequantization.basis_augmentation,
            basis_augmentation_probability=self.dequantization.basis_augmentation_probability,
        )
        if result.loss.per_sample.shape != (batch_size,):
            raise InvariantViolation("packed CFM returned the wrong per-sample shape")
        if search:
            return (weights * result.loss.per_sample).mean()
        return result.loss.per_sample.mean()

    def _geometry_loss_mixed(self, module: Any, batch: MixedTrainingBatch) -> Tensor:
        if batch.measure not in _STRUCTURE_MEASURES:
            raise DomainError("geometry training requires a structure measure")
        device, dtype = self._device_dtype()
        scaffolds = []
        metrics = []
        search = batch.measure is TrainingMeasure.SYSTEM_FORMULA_POLYMORPH
        part_weights = []
        for part in batch.parts:
            weights = _as_tensor(part.geometry.sample_weights, device=device, dtype=dtype)
            if search:
                self._require_search_weights(part)
                part_weights.append(weights)
            elif not bool(torch.all(weights == 1.0).item()):
                raise InvariantViolation("structure-geometry weights are not unity")
            scaffolds.extend(part.geometry.scaffolds)
            metrics.extend(part.geometry.metric[index] for index in range(len(part.record_ids)))
        time = torch.rand(len(scaffolds), dtype=dtype, device=device)
        result = self._geometry.geometry_cfm_objective_padded(
            tuple(scaffolds),
            tuple(metrics),
            time=time,
            sigma_z=self.dequantization.sigma_lattice,
            sigma_eta=self.dequantization.sigma_intrinsic,
            generator=None,
            geometry_field=module,
            intrinsic_kernel=self.dequantization.intrinsic_kernel,
            block_weights=self.block_weights.as_tuple(),
            basis_augmentation=self.dequantization.basis_augmentation,
            basis_augmentation_probability=self.dequantization.basis_augmentation_probability,
        )
        if result.loss.per_sample.shape != (len(scaffolds),):
            raise InvariantViolation("padded CFM returned the wrong per-sample shape")
        if search:
            return (torch.cat(part_weights) * result.loss.per_sample).mean()
        return result.loss.per_sample.mean()

    def _mixed_losses(
        self, models: ComponentModels, batch: MixedTrainingBatch, phase: TrainingPhase
    ) -> ComponentLosses:
        """Record-weighted means of per-part losses, and one padded geometry call.

        A single-part batch returns that part's losses unchanged, so it agrees
        bitwise with the corresponding bucket batch.
        """

        def combined(loss: Any, module: Any) -> Tensor:
            values = [loss(module, part) for part in batch.parts]
            if len(values) == 1:
                return values[0]
            sizes = [len(part.record_ids) for part in batch.parts]
            total = sum(value * size for value, size in zip(values, sizes, strict=True))
            return total / batch.batch_size

        composition_nll = None
        cover_nll = None
        cfm = None
        if phase in (TrainingPhase.COMPOSITION, TrainingPhase.JOINT):
            composition_nll = combined(self._composition_loss, models.composition)
        if phase in (TrainingPhase.COVER, TrainingPhase.JOINT):
            cover_nll = combined(self._cover_loss, models.cover)
        if phase in (TrainingPhase.GEOMETRY, TrainingPhase.JOINT):
            cfm = self._geometry_loss_mixed(models.geometry, batch)
        return ComponentLosses(
            batch_size=batch.batch_size,
            composition_nll=composition_nll,
            cover_nll=cover_nll,
            cfm_regression=cfm,
        )

    def __call__(
        self,
        models: ComponentModels,
        batch: TrainingBatch | MixedTrainingBatch,
        phase: TrainingPhase,
    ) -> ComponentLosses:
        if not isinstance(models, ComponentModels):
            raise TypeError("models must be ComponentModels")
        if models.root is not self.model:
            raise DomainError(
                "training objective and trainer must reference the identical root model"
            )
        if not isinstance(batch, (TrainingBatch, MixedTrainingBatch)):
            raise TypeError("batch must be a TrainingBatch or MixedTrainingBatch")
        if batch.dataset_fingerprint != self.data_measure.dataset_fingerprint:
            raise DomainError(
                "training batch and objective population measure reference different datasets"
            )
        if not isinstance(phase, TrainingPhase):
            raise TypeError("phase must be a TrainingPhase")
        if isinstance(batch, MixedTrainingBatch):
            return self._mixed_losses(models, batch, phase)
        composition_nll = None
        cover_nll = None
        cfm = None
        if phase in (TrainingPhase.COMPOSITION, TrainingPhase.JOINT):
            composition_nll = self._composition_loss(models.composition, batch)
        if phase in (TrainingPhase.COVER, TrainingPhase.JOINT):
            cover_nll = self._cover_loss(models.cover, batch)
        if phase in (TrainingPhase.GEOMETRY, TrainingPhase.JOINT):
            cfm = self._geometry_loss(models.geometry, batch)
        return ComponentLosses(
            batch_size=len(batch.record_ids),
            composition_nll=composition_nll,
            cover_nll=cover_nll,
            cfm_regression=cfm,
        )


__all__ = [
    "CFMBlockWeights",
    "CFMDequantization",
    "CoverFactorTerms",
    "CrystalAtlasTrainingObjective",
    "ExactBatchCrystallographicAudit",
    "ObjectiveDataMeasure",
]
