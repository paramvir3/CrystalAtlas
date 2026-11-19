"""Assembly of the learned DNG/S-DNG/CSP probability kernel.

This module connects the exact grammars and catalogue to the neural scorers,
product base, conditional-flow ODE, and deterministic continuous decoder.  It
does not bundle or pretend to have a trained checkpoint: inference requires an
explicit checkpoint identity unless ``allow_untrained=True`` is deliberately
selected for an equation-level smoke test.
"""

from __future__ import annotations

import math
from collections import OrderedDict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from ._optional import require_torch
from .basis_augmentation import BASIS_AUGMENTATIONS, cover_basis_augmentation
from .catalogue import MINIMAL_CATALOGUE, CertifiedCatalogue
from .composition import (
    Composition,
    CompositionDomain,
    CompositionError,
    CompositionGrammar,
)
from .config import CrystalAtlasConfig
from .decoder import CrystalScaffold, ExactCrystalDecoder
from .errors import (
    DomainError,
    GeneratedSampleError,
    InvariantViolation,
    RecoverableGenerationError,
)
from .flow import (
    CFMLossTerms,
    CFMPathSample,
    FlowState,
    FlowVelocity,
    base_log_density,
    canonicalize_torus_validated,
    conditional_flow_matching_loss,
    conditional_flow_path,
    integrate_with_endpoint_estimate,
    log_density_transport,
    sample_dequantized_endpoint,
    sample_product_base,
    shortest_coordinate_lift,
)
from .geometry import (
    ContinuousCover,
    ContinuousDecodedCrystal,
    ContinuousGeometry,
    IntrinsicCoverLayout,
    PrimitivePhysicalCrystal,
)
from .nn.bridge import (
    PaddedCoverPlan,
    TorchCoverGeometry,
    TorchExpandedGeometry,
    expand_padded,
    geometry_bridge_numerical_contract,
    intrinsic_noise_covariance,
    intrinsic_noise_transform,
    lattice_pullback_metrics,
)
from .nn.composition import (
    CanonicalCompositionTransformer,
    CompositionTransformerConfig,
)
from .nn.cover_sampling import NeuralCoverSample, NeuralCoverSampler
from .nn.field import (
    IntrinsicFieldConfig,
    IntrinsicGeometryField,
    TangentRankError,
    translation_quotient_normal_matrix,
)
from .nn.sampling import sample_composition
from .nn.selector import ArithmeticCoverSelector, CoverSelectorConfig
from .pipeline import GenerationMode, GenerationRequest
from .residual_gauge import ResidualActionTable, cover_residual_gauge
from .symmetry_descriptors import (
    GROUP_DESCRIPTOR_DIM,
    ROW_GEOMETRY_DESCRIPTOR_DIM,
    ROW_SITE_DESCRIPTOR_DIM,
    group_descriptor,
)
from .tokenization import CompositionTokenCodec
from .training import (
    LoadedCheckpoint,
    ScientificManifest,
    load_checkpoint,
    model_state_fingerprint,
)

torch = require_torch()
Tensor = torch.Tensor
nn = torch.nn


LAYOUT_CACHE_SIZE = 4096
VALIDATED_TARGET_CACHE_SIZE = 65536
INTRINSIC_KERNELS = frozenset({"fractional-v1", "cartesian-v2"})
GEOMETRY_SAMPLERS: tuple[str, ...] = ("bucket-v1", "mixed-v1")
"""Geometry ODE batching of :meth:`LearnedCrystalGenerator.sample`.

``bucket-v1`` solves one RK4 problem per topology bucket.  ``mixed-v1`` solves
every attempt of a request in one padded problem; it draws the same base
points and equals ``bucket-v1`` to rounding, not bitwise, so a result must
record which one produced it.
"""
"""Torus dequantization kernels of the CFM objective (``CFMDequantization``)."""
START_REJECTION_MAX_DRAWS = 1000
"""Base draws one attempt may use under start rejection (research 2026-11-16).

An attempt whose base draws all have two distinct atoms within the separation
raises :class:`RecoverableGenerationError` before its solve is integrated, and
ends as a ``geometry/ode`` failure without a structure.  The error fails the
whole solve.  ``mixed-v1`` then halves the solve, restoring every generator,
until the attempt is alone; ``bucket-v1`` replays every member of the bucket
alone.  The siblings keep their draws and are integrated in smaller solves, so
they equal, to rounding and not bitwise, the same attempts in a solve where no
member fails: untreated ones if they were not redrawn, treated ones if they were.
"""
START_REJECTION_MAX_IMAGES = 8
"""Largest integer image range per axis of the start-separation search.

The search is exact; a range beyond this needs a lattice plane spacing below
``separation / 7.5`` and fails the attempt with :class:`RecoverableGenerationError`.
"""


def closest_distinct_pair_within(
    lattice: Tensor, fractional: Tensor, radius: float
) -> float | None:
    """Smallest distance between two distinct atoms if one pair is within ``radius``.

    ``lattice`` has the basis vectors as rows (``cartesian = fractional @ lattice``)
    and ``fractional`` holds one row per atom.  Every periodic image is searched
    exactly: a pair whose minimal image distance is at most ``radius`` has its
    minimizing translation inside the integer box ``|n_k| <= ceil(radius
    sqrt((G^-1)_kk) + 1/2)`` around the reduced difference, ``G`` the Gram matrix.
    Images of one atom are not a pair.  Computed in float64; returns ``None`` when
    no pair is within ``radius``.
    """

    if not (math.isfinite(radius) and radius > 0.0):
        raise DomainError("the search radius must be positive and finite")
    count = int(fractional.shape[0])
    if count < 2:
        return None
    with torch.autocast(device_type=lattice.device.type, enabled=False):
        lattice = lattice.to(dtype=torch.float64)
        fractional = fractional.to(dtype=torch.float64)
        gram_inverse = torch.linalg.inv(lattice @ lattice.transpose(0, 1))
        reach = radius * torch.sqrt(torch.diagonal(gram_inverse))
        box = [int(math.ceil(float(value) + 0.5)) for value in reach.cpu()]
        if max(box) > START_REJECTION_MAX_IMAGES:
            raise RecoverableGenerationError(
                "the start-separation image search needs more than "
                f"{START_REJECTION_MAX_IMAGES} images per axis"
            )
        axes = [
            torch.arange(-size, size + 1, dtype=torch.float64, device=lattice.device)
            for size in box
        ]
        shifts = torch.cartesian_prod(*axes)
        first, second = torch.triu_indices(count, count, 1, device=lattice.device)
        delta = fractional[second] - fractional[first]
        reduced = delta - torch.round(delta)
        cartesian = (reduced[:, None, :] + shifts[None, :, :]) @ lattice
        smallest = float(torch.sqrt((cartesian * cartesian).sum(dim=-1).min()))
    return smallest if smallest <= radius else None


def _require_intrinsic_kernel(kind: str) -> None:
    if kind not in INTRINSIC_KERNELS:
        raise DomainError("intrinsic_kernel must be fractional-v1 or cartesian-v2")


COUPLINGS: tuple[str, ...] = ("independent", "gauge-ot-v1")
"""Base-endpoint couplings of the batched CFM objective (research R1, 2026-10-30).

``independent`` is the legacy product coupling.  ``gauge-ot-v1`` pairs base
samples with records by a minibatch assignment on the gauge-quotient cost
``min_gamma d(gamma^-1 q_0, q_1)`` in the reference loss metric, and pushes each
path by the minimizing residual action.  The assignment runs separately over the
records of each cover, the variable the field is conditioned on.  The minimum is
over the whole residual group when its action table has at most
``coupling_candidates`` rows, and over that many i.i.d. uniform actions per
record otherwise.  Both endpoint marginals of every cover are unchanged
(Proposition 2.9d): the base samples of a cover are only permuted, and because
the base law is invariant under the residual group, the chosen action of every
record is uniform over the group, so its pushed endpoint keeps the symmetrized
target law.  The conditional target variance is what changes.
"""


def _require_coupling(name: str, candidates: int) -> None:
    if name not in COUPLINGS:
        raise DomainError("coupling must be one of " + ", ".join(COUPLINGS))
    if isinstance(candidates, bool) or not isinstance(candidates, int) or candidates < 1:
        raise DomainError("coupling_candidates must be a positive integer")


def _require_basis_augmentation(name: str, probability: float) -> None:
    if name not in BASIS_AUGMENTATIONS:
        raise DomainError("basis_augmentation must be one of " + ", ".join(BASIS_AUGMENTATIONS))
    if (
        isinstance(probability, bool)
        or not isinstance(probability, (int, float))
        or not math.isfinite(float(probability))
        or not 0.0 <= float(probability) <= 1.0
    ):
        raise DomainError("basis_augmentation_probability must lie in [0, 1]")
"""Validated CFM targets one generator remembers (exact; failures are not cached)."""
"""Covers whose layouts one model keeps; each layout is exact and read-only."""


class CrystalAtlasModel(nn.Module):
    """One shared parameter collection for all three generation modes."""

    def __init__(
        self,
        config: CrystalAtlasConfig | None = None,
        *,
        catalogue: CertifiedCatalogue | None = None,
    ) -> None:
        super().__init__()
        if catalogue is None:
            raise DomainError(
                "catalogue must be explicit: use CrystalAtlasModel.fixture() only for "
                "equation tests or CrystalAtlasModel.production() for all-230 inference"
            )
        self.scientific_config = CrystalAtlasConfig() if config is None else config
        self.catalogue = catalogue
        catalogue.validate()
        # Layouts are a deterministic function of the cover for this fixed domain
        # and catalogue, so a bounded cache is exact (review M1 profiling).
        self._layout_cache: OrderedDict[object, IntrinsicCoverLayout] = OrderedDict()
        domain = self.scientific_config.domain
        self.composition_domain = CompositionDomain(
            elements=domain.dng_element_vocabulary,
            max_elements=domain.max_distinct_elements,
            max_atoms=domain.composition_atom_bound,
        )
        self.composition_grammar = CompositionGrammar(self.composition_domain)
        self.composition_codec = CompositionTokenCodec(self.composition_grammar)
        model = self.scientific_config.model
        self.composition_model = CanonicalCompositionTransformer(
            CompositionTransformerConfig(
                vocab_size=self.composition_codec.vocab_size,
                num_token_types=self.composition_codec.num_token_types,
                model_dim=model.composition_width,
                num_heads=model.attention_heads,
                num_layers=model.composition_layers,
                feedforward_dim=4 * model.composition_width,
                max_sequence_length=self.composition_codec.max_sequence_length,
                dropout=model.dropout,
            )
        )
        descriptors = None
        if model.selector_symmetry_descriptors:
            # Groups absent from the catalogue are always masked; zeros are inert.
            table = np.zeros((230, GROUP_DESCRIPTOR_DIM), dtype=np.float64)
            for setting in catalogue.settings.values():
                table[setting.international_number - 1] = group_descriptor(setting)
            descriptors = torch.as_tensor(table, dtype=torch.float32)
        self.cover_selector = ArithmeticCoverSelector(
            CoverSelectorConfig(
                num_nu_choices=domain.max_formula_units,
                num_space_groups=230,
                row_feature_dim=(
                    4
                    + (ROW_SITE_DESCRIPTOR_DIM if model.selector_symmetry_descriptors else 0)
                    + (
                        ROW_GEOMETRY_DESCRIPTOR_DIM
                        if model.selector_row_geometry_features
                        else 0
                    )
                ),
                row_geometry_features=model.selector_row_geometry_features,
                model_dim=model.selector_width,
                num_heads=model.attention_heads,
                num_layers=model.selector_layers,
                feedforward_dim=4 * model.selector_width,
                dropout=model.dropout,
                group_descriptor_dim=(
                    GROUP_DESCRIPTOR_DIM if model.selector_symmetry_descriptors else 0
                ),
                element_embedding=model.element_embedding,
            ),
            group_descriptors=descriptors,
        )
        self.geometry_field = IntrinsicGeometryField(
            IntrinsicFieldConfig(
                atom_feature_dim=TorchCoverGeometry.atom_feature_dim,
                composition_dim=max(32, model.geometry_scalar_width // 2),
                species_embedding_dim=max(16, model.geometry_scalar_width // 4),
                max_lattice_dim=6,
                backbone=model.geometry_backbone,
                scalar_dim=model.geometry_scalar_width,
                pair_dim=model.geometry_pair_width,
                vector_dim=model.geometry_vector_width,
                num_heads=model.attention_heads,
                num_layers=model.geometry_layers,
                radial_basis_dim=model.radial_features,
                invariant_feedforward_multiplier=(
                    model.geometry_invariant_feedforward_multiplier
                ),
                reciprocal_mode_radius=model.geometry_reciprocal_mode_radius,
                heat_scales=model.geometry_heat_scales,
                direct_harmonics=model.geometry_direct_harmonics,
                include_lattice_frame_carriers=(model.geometry_include_lattice_frame_carriers),
                normalization_epsilon=model.geometry_normalization_epsilon,
                lattice_head=model.geometry_lattice_head,
                local_vector_normalization=model.geometry_local_vector_normalization,
                radial_gaussian_widths=model.geometry_radial_gaussian_widths,
                pair_carrier_smoothing_length=model.geometry_pair_carrier_smoothing_length,
                element_embedding=model.element_embedding,
                minimum_dimensionless_lattice_singular_value=(
                    model.geometry_minimum_dimensionless_lattice_singular_value
                ),
                maximum_dimensionless_lattice_singular_value=(
                    model.geometry_maximum_dimensionless_lattice_singular_value
                ),
                maximum_lattice_condition_number=(
                    model.geometry_maximum_lattice_condition_number
                ),
                cutoff=domain.interaction_cutoff_angstrom,
                reference_length=domain.reference_length_angstrom,
                max_candidates_per_pair=(
                    self.scientific_config.numerics.max_pair_image_candidates
                ),
                max_periodic_edges=self.scientific_config.numerics.max_periodic_edges,
                max_dense_pair_interactions=(
                    self.scientific_config.numerics.max_dense_pair_interactions
                ),
                max_reciprocal_modes=(self.scientific_config.numerics.max_reciprocal_modes),
                max_readout_coefficient_channels=(
                    self.scientific_config.numerics.max_geometry_readout_coefficient_channels
                ),
                max_geometry_workspace_bytes=(
                    self.scientific_config.numerics.max_geometry_workspace_bytes
                ),
                maximum_tangent_condition_number_float32=(
                    self.scientific_config.numerics.maximum_tangent_condition_number_float32
                ),
                maximum_tangent_condition_number_float64=(
                    self.scientific_config.numerics.maximum_tangent_condition_number_float64
                ),
            )
        )

    @classmethod
    def fixture(cls, config: CrystalAtlasConfig | None = None) -> CrystalAtlasModel:
        """Construct the deliberately incomplete P1/P-1 equation fixture."""

        return cls(config, catalogue=MINIMAL_CATALOGUE)

    @classmethod
    def production(cls, config: CrystalAtlasConfig | None = None) -> CrystalAtlasModel:
        """Construct against the frozen, exactly compiled all-230 catalogue."""

        from .full_catalogue import build_full_catalogue

        return cls(config, catalogue=build_full_catalogue())

    def cover_sampler(self) -> NeuralCoverSampler:
        return NeuralCoverSampler(
            self.cover_selector,
            catalogue=self.catalogue,
            max_raw_atoms=self.scientific_config.domain.max_raw_atoms,
            max_primitive_atoms=self.scientific_config.domain.max_primitive_atoms,
        )

    def scientific_manifest(self, *, data_contract: Mapping[str, Any]) -> ScientificManifest:
        """Derive the checkpoint contract from this live model and catalogue.

        Only the dataset/split contract is caller supplied.  Architecture,
        representation, and exact catalogue identities are derived here so a
        caller cannot accidentally bless a checkpoint under incompatible live
        decoder semantics.
        """

        config = self.scientific_config.to_dict()
        return ScientificManifest(
            architecture={
                "schema": "crystalatlas.model-architecture.v3",
                "class": f"{type(self).__module__}.{type(self).__qualname__}",
                "model": config["model"],
            },
            representation={
                "schema": "crystalatlas.representation.v2",
                "geometry_bridge_numerics": geometry_bridge_numerical_contract(),
                "config_fingerprint": self.scientific_config.fingerprint,
                "coordinate_convention": self.scientific_config.coordinate_convention,
                "chirality_quotient": self.scientific_config.chirality_quotient,
                "domain": config["domain"],
                "numerics": config["numerics"],
                "composition_serialization": "support-first-eos-v1",
                "cover_order": "fixed-first-row-species-v1",
            },
            catalogue=self.catalogue.identity,
            data_contract=data_contract,
        )

    def validate_scientific_manifest(self, manifest: ScientificManifest) -> None:
        if not isinstance(manifest, ScientificManifest):
            raise TypeError("manifest must be a ScientificManifest")
        expected = self.scientific_manifest(data_contract=manifest.data_contract)
        if expected.fingerprint != manifest.fingerprint:
            raise DomainError(
                "scientific manifest does not match the live model, representation, "
                "and certified catalogue"
            )

    def validate_composition(self, composition: Composition) -> None:
        if not isinstance(composition, Composition):
            raise TypeError("composition must be a Composition")
        domain = self.scientific_config.domain
        if not set(composition.support).issubset(domain.element_vocabulary):
            raise DomainError("composition contains an element outside the model vocabulary")
        if len(composition) > domain.max_distinct_elements:
            raise DomainError("composition exceeds the supported number of elements")
        if composition.total_atoms > domain.composition_atom_bound:
            raise DomainError("reduced composition exceeds the raw-cell atom bound")

    def layout(self, cover: object) -> IntrinsicCoverLayout:
        from .geometry import ContinuousCover

        if not isinstance(cover, ContinuousCover):
            raise TypeError("cover must be a ContinuousCover")
        domain = self.scientific_config.domain
        self.validate_composition(cover.composition)
        if cover.nu > domain.max_formula_units:
            raise DomainError("cover formula-unit multiplier exceeds model support")
        if not domain.atom_bound.admits(
            cover.nu,
            cover.composition.total_atoms,
            self.catalogue.get(cover.space_group).centering_index,
        ):
            raise DomainError("cover exceeds the advertised raw-cell atom bound")
        cached = self._layout_cache.get(cover)
        if cached is not None:
            self._layout_cache.move_to_end(cover)
            return cached
        layout = IntrinsicCoverLayout(
            cover, catalogue=self.catalogue, **domain.layout_options()
        )
        self._layout_cache[cover] = layout
        if len(self._layout_cache) > LAYOUT_CACHE_SIZE:
            self._layout_cache.popitem(last=False)
        return layout


@dataclass(frozen=True, slots=True)
class LearnedSample:
    mode: GenerationMode
    composition: Composition
    cover_sample: NeuralCoverSample
    geometry: ContinuousGeometry
    decoded: ContinuousDecodedCrystal
    physical_primitive: PrimitivePhysicalCrystal
    seed: int
    checkpoint_id: str | None


@dataclass(frozen=True, slots=True)
class SampleScore:
    """Exact model log probability of one generated sample (research R2).

    ``log_joint = log_cover + log_geometry``: the selector's exact log probability
    of the cover and the flow's log density of the geometry under that cover's
    law (Proposition 2.9e).  ``dimension`` is ``d_z + d_eta`` of the cover:
    scores of samples with different dimensions are densities with respect to
    different volumes and must not be compared as numbers.
    """

    sample_index: int
    log_cover: float
    log_geometry: float
    dimension: int

    @property
    def log_joint(self) -> float:
        return self.log_cover + self.log_geometry


@dataclass(frozen=True, slots=True)
class MapSelection:
    """Hierarchical maximum a posteriori choice among generated samples (research R2).

    ``sample_index`` is the chosen position in the input sequence.  ``members`` are
    the input positions of every sample of the selected cover and ``log_geometry``
    their exact geometry log densities in the same order; ``log_cover`` is the
    selector log probability of that cover.
    """

    sample_index: int
    members: tuple[int, ...]
    log_cover: float
    log_geometry: tuple[float, ...]


@dataclass(frozen=True, slots=True)
class LearnedGenerationTrace:
    sample_index: int
    seed: int
    sample: LearnedSample | None = None
    failed_stage: str | None = None
    error: str | None = None

    @property
    def succeeded(self) -> bool:
        return self.sample is not None and self.failed_stage is None


_MEMBER_ODE_ERRORS: tuple[type[Exception], ...] = (
    RecoverableGenerationError,
    TangentRankError,
    ValueError,
)
"""Errors a single sampling attempt may fail with (review 2026-10-29, m3).

``ResourceLimitError`` and ``InvariantViolation`` are not in this set: resource
exhaustion must abort rather than inflate invalidity counts, and an invariant
violation is a defect, not an attempt.
"""


@dataclass(frozen=True)
class _PreparedLearnedAttempt:
    sample_index: int
    reported_seed: int
    composition: Composition
    cover_sample: NeuralCoverSample
    geometry_generator: Any


@dataclass(frozen=True)
class GeometryCFMResult:
    loss: CFMLossTerms
    path: CFMPathSample
    prediction: FlowVelocity


@dataclass(frozen=True)
class GeometryCFMBatchResult:
    """One topology-homogeneous packed conditional-flow objective."""

    loss: CFMLossTerms
    path: CFMPathSample
    prediction: FlowVelocity
    coupling_stats: dict[str, float] | None = None
    """``gauge-ot-v1`` only: mean assigned cost against the identity pairing's, the
    number of single-cover assignment blocks and the fraction of records whose
    cost minimized over the whole residual group."""


@dataclass(frozen=True)
class GeometryCFMPaddedResult:
    """One mixed-topology padded conditional-flow objective (review M5).

    ``paths`` keeps each crystal's unpadded sampled path.  ``prediction`` is
    padded: crystal ``b`` occupies the leading ``lattice_dimensions[b]`` and
    ``intrinsic_dimensions[b]`` slots, and the rest are exactly zero.
    """

    loss: CFMLossTerms
    paths: tuple[CFMPathSample, ...]
    prediction: FlowVelocity
    lattice_dimensions: Tensor
    intrinsic_dimensions: Tensor
    atom_counts: Tensor


def _layout_batch_signature(layout: IntrinsicCoverLayout) -> tuple[object, ...]:
    """Exact dense-shape/decoder signature for packed field evaluation."""

    basis = np.stack(layout.chart.basis) if layout.chart.basis else np.empty((0, 3, 3))
    support = layout.cover.composition.support
    support_index = {atomic_number: index for index, atomic_number in enumerate(support)}
    return (
        layout.setting.key,
        len(support),
        layout.cover.composition.coefficients,
        tuple(support_index[orbit.atomic_number] for orbit in layout.cover.orbits),
        layout.lattice_dimension,
        layout.intrinsic_dimension,
        layout.expanded_atom_count,
        tuple(position.label for position in layout.positions),
        tuple(
            np.asarray(mapping, dtype="<f8", order="C").tobytes(order="C")
            for mapping in layout.parameter_maps
        ),
        np.asarray(layout.period_basis, dtype="<f8", order="C").tobytes(order="C"),
        np.asarray(layout.chart.sqrt_reference, dtype="<f8", order="C").tobytes(order="C"),
        np.asarray(basis, dtype="<f8", order="C").tobytes(order="C"),
    )


class LearnedCrystalGenerator:
    """Ancestral learned sampler with no retries, repairs, or hidden ranking.

    The one declared exception is the opt-in start rejection
    (:attr:`start_min_separation`), which redraws base points before integration.
    """

    def __init__(
        self,
        model: CrystalAtlasModel,
        *,
        validated_checkpoint: LoadedCheckpoint | None = None,
        allow_untrained: bool = False,
        geometry_sampler: str = "bucket-v1",
        start_min_separation: float | None = None,
    ) -> None:
        if not isinstance(model, CrystalAtlasModel):
            raise TypeError("model must be a CrystalAtlasModel")
        if geometry_sampler not in GEOMETRY_SAMPLERS:
            raise DomainError(f"geometry_sampler must be one of {GEOMETRY_SAMPLERS}")
        if (
            geometry_sampler == "mixed-v1"
            and model.geometry_field.config.backbone != "invariant_global"
        ):
            raise DomainError(
                "the mixed-v1 geometry sampler requires the dense invariant_global backbone"
            )
        if validated_checkpoint is None and not allow_untrained:
            raise DomainError(
                "learned inference requires a validated checkpoint receipt; "
                "set allow_untrained=True only for a smoke test"
            )
        if validated_checkpoint is not None and allow_untrained:
            raise DomainError(
                "validated checkpoint inference and allow_untrained are mutually exclusive"
            )
        if validated_checkpoint is not None:
            if not isinstance(validated_checkpoint, LoadedCheckpoint):
                raise TypeError("validated_checkpoint must be a LoadedCheckpoint receipt")
            if validated_checkpoint.config_fingerprint != model.scientific_config.fingerprint:
                raise DomainError("checkpoint receipt belongs to a different configuration")
            try:
                receipt_manifest = ScientificManifest(
                    **validated_checkpoint.scientific_manifest
                )
            except (TypeError, DomainError) as exc:
                raise DomainError(
                    "checkpoint receipt has an invalid scientific manifest"
                ) from exc
            model.validate_scientific_manifest(receipt_manifest)
            if validated_checkpoint.missing_model_keys or (
                validated_checkpoint.unexpected_model_keys
            ):
                raise DomainError("checkpoint receipt was not produced by a strict model load")
            if model_state_fingerprint(model) != validated_checkpoint.model_state_fingerprint:
                raise DomainError(
                    "live model parameters do not match the validated checkpoint receipt"
                )
        self.model = model
        self._checkpoint_receipt = validated_checkpoint
        self.checkpoint_id = (
            validated_checkpoint.sha256 if validated_checkpoint is not None else None
        )
        self.allow_untrained = bool(allow_untrained)
        self.geometry_sampler = geometry_sampler
        self.start_min_separation = start_min_separation
        self._cover_sampler = model.cover_sampler()
        # Exact caches (GPU profiling: re-validation 29%, bridge construction ~10%
        # of a batch-64 step; target re-encoding later 13%).  All are deterministic
        # in their keys.
        self._validated_targets: OrderedDict[tuple[object, ...], None] = OrderedDict()
        self._bridges: OrderedDict[tuple[object, ...], TorchCoverGeometry] = OrderedDict()
        self._encoded_targets: OrderedDict[
            tuple[object, ...], tuple[np.ndarray, np.ndarray]
        ] = OrderedDict()
        self._noise_transforms: OrderedDict[tuple[object, ...], np.ndarray] = OrderedDict()
        self._coupling_metric_cache: dict[tuple[object, ...], tuple[Tensor, Tensor]] = {}
        self._residual_tables: OrderedDict[object, ResidualActionTable] = OrderedDict()
        self._exhaustive_actions: OrderedDict[
            tuple[object, ...], tuple[Tensor, Tensor | None]
        ] = OrderedDict()

    def _require_valid_target(self, scaffold: CrystalScaffold, metric: np.ndarray) -> None:
        """Exact decoder validation of one CFM target, cached on success only."""

        array = np.ascontiguousarray(metric, dtype=np.float64)
        key = (scaffold, array.shape, array.tobytes())
        if key in self._validated_targets:
            self._validated_targets.move_to_end(key)
            return
        validated = ExactCrystalDecoder(
            self.model.catalogue,
            metric_tolerance=self.model.scientific_config.numerics.symmetry_atol,
        ).decode(scaffold, array)
        if validated.hidden_translation_index != 1:
            raise DomainError(
                "training target has hidden translations beyond its declared centring; "
                "reduce and relabel its actual primitive cover before CFM"
            )
        self._validated_targets[key] = None
        if len(self._validated_targets) > VALIDATED_TARGET_CACHE_SIZE:
            self._validated_targets.popitem(last=False)

    def _encoded_target(
        self,
        layout: IntrinsicCoverLayout,
        scaffold: CrystalScaffold,
        metric: np.ndarray,
        device: torch.device,
        dtype: torch.dtype,
    ) -> tuple[Tensor, Tensor]:
        """Chart latent ``z`` and exact torus parameters ``eta`` of one target.

        The float64 encodings are cached read-only, and every call returns fresh
        tensors (``torch.tensor`` copies), so no consumer aliases the cache.
        """

        encoded = self._encoded_arrays(layout, scaffold, metric)
        return (
            torch.tensor(encoded[0], dtype=dtype, device=device),
            torch.tensor(encoded[1], dtype=dtype, device=device),
        )

    def _encoded_arrays(
        self, layout: IntrinsicCoverLayout, scaffold: CrystalScaffold, metric: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """The read-only float64 ``(z, eta)`` encodings behind :meth:`_encoded_target`."""

        array = np.ascontiguousarray(metric, dtype=np.float64)
        key = (layout.cover, scaffold, array.shape, array.tobytes())
        encoded = self._encoded_targets.get(key)
        if encoded is None:
            z_data = np.asarray(layout.chart.encode_latent(array), dtype=np.float64)
            eta_data = np.asarray(layout.encode_exact_parameters(scaffold), dtype=np.float64)
            z_data.setflags(write=False)
            eta_data.setflags(write=False)
            encoded = (z_data, eta_data)
            self._encoded_targets[key] = encoded
            if len(self._encoded_targets) > VALIDATED_TARGET_CACHE_SIZE:
                self._encoded_targets.popitem(last=False)
        else:
            self._encoded_targets.move_to_end(key)
        return encoded

    def _endpoint_record(
        self,
        layout: IntrinsicCoverLayout,
        bridge: TorchCoverGeometry,
        scaffold: CrystalScaffold,
        metric: np.ndarray,
        *,
        sigma_eta: float,
        intrinsic_kernel: str,
        basis_augmentation: str,
        basis_augmentation_probability: float,
        generator: torch.Generator | None,
        device: torch.device,
        dtype: torch.dtype,
    ) -> tuple[Tensor, Tensor, Tensor | None]:
        """``(z_data, eta_data, eta noise transform)`` of one record.

        With ``basis_augmentation="shear-v1"`` (review 2026-10-29, C2) the record
        is moved, with the declared probability, to a neighbouring lattice-basis
        presentation drawn uniformly from the cover's admissible elementary
        shears; an image outside the chart is refused and the record is kept.
        ``"none"`` makes no draw and is the legacy path bitwise.  The
        ``cartesian-v2`` transform is evaluated at the moved record.
        """

        z_data, eta_data = self._encoded_target(layout, scaffold, metric, device, dtype)
        if basis_augmentation == "shear-v1":
            augmentation = cover_basis_augmentation(layout)
            if len(augmentation):
                coin = float(torch.rand((), generator=generator, device=device).item())
                if coin < basis_augmentation_probability:
                    index = int(
                        torch.randint(
                            len(augmentation), (), generator=generator, device=device
                        ).item()
                    )
                    z_array, eta_array = self._encoded_arrays(layout, scaffold, metric)
                    moved = augmentation.transform(index, z_array, eta_array)
                    if moved is not None:
                        z_data = torch.tensor(moved[0], dtype=dtype, device=device)
                        eta_data = torch.tensor(moved[1], dtype=dtype, device=device)
                        transform = None
                        if intrinsic_kernel == "cartesian-v2":
                            covariance = intrinsic_noise_covariance(
                                bridge,
                                torch.tensor(moved[0], dtype=torch.float64, device=device),
                                torch.tensor(moved[1], dtype=torch.float64, device=device),
                                float(sigma_eta),
                                reference_length=(
                                    self.model.scientific_config.domain.reference_length_angstrom
                                ),
                            )
                            transform = intrinsic_noise_transform(covariance).to(
                                dtype=dtype, device=device
                            )
                        return z_data, eta_data, transform
        transform = (
            self._intrinsic_noise_transform(
                layout, bridge, scaffold, metric, sigma_eta, device, dtype
            )
            if intrinsic_kernel == "cartesian-v2"
            else None
        )
        return z_data, eta_data, transform

    def _intrinsic_noise_transform(
        self,
        layout: IntrinsicCoverLayout,
        bridge: TorchCoverGeometry,
        scaffold: CrystalScaffold,
        metric: np.ndarray,
        sigma: float,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Tensor:
        """``T`` with ``T^T T = sigma^2 (J_hat^T P_N J_hat)^-1`` at the record.

        This is the ``cartesian-v2`` torus kernel (review C1).  The float64
        transform is cached read-only per (cover, scaffold, metric, sigma); every
        call returns a fresh tensor.
        """

        array = np.ascontiguousarray(metric, dtype=np.float64)
        key = (layout.cover, scaffold, array.shape, array.tobytes(), float(sigma))
        cached = self._noise_transforms.get(key)
        if cached is None:
            z_data, eta_data = self._encoded_target(
                layout, scaffold, metric, device, torch.float64
            )
            covariance = intrinsic_noise_covariance(
                bridge,
                z_data,
                eta_data,
                float(sigma),
                reference_length=self.model.scientific_config.domain.reference_length_angstrom,
            )
            cached = intrinsic_noise_transform(covariance).cpu().numpy()
            cached.setflags(write=False)
            self._noise_transforms[key] = cached
            if len(self._noise_transforms) > VALIDATED_TARGET_CACHE_SIZE:
                self._noise_transforms.popitem(last=False)
        else:
            self._noise_transforms.move_to_end(key)
        return torch.tensor(cached, dtype=dtype, device=device)

    def _bridge(
        self, layout: IntrinsicCoverLayout, device: torch.device, dtype: torch.dtype
    ) -> TorchCoverGeometry:
        """One read-only expansion bridge per (cover, device, dtype)."""

        key = (layout.cover, str(device), dtype)
        bridge = self._bridges.get(key)
        if bridge is None:
            expansion_cell = self.model.scientific_config.model.geometry_expansion_cell
            bridge = TorchCoverGeometry(layout, expansion_cell=expansion_cell).to(
                device=device, dtype=dtype
            )
            self._bridges[key] = bridge
            if len(self._bridges) > LAYOUT_CACHE_SIZE:
                self._bridges.popitem(last=False)
        else:
            self._bridges.move_to_end(key)
        return bridge

    @classmethod
    def from_checkpoint(
        cls,
        model: CrystalAtlasModel,
        path: str | Path,
        *,
        scientific_manifest: ScientificManifest,
        expected_sha256: str,
        map_location: Any = "cpu",
        geometry_sampler: str = "bucket-v1",
        start_min_separation: float | None = None,
    ) -> LearnedCrystalGenerator:
        """Load, validate, bind, and return a trained inference generator."""

        model.validate_scientific_manifest(scientific_manifest)

        receipt = load_checkpoint(
            path,
            model=model,
            expected_scientific_manifest=scientific_manifest,
            expected_config=model.scientific_config,
            map_location=map_location,
            strict_model=True,
            restore_rng=False,
            expected_sha256=expected_sha256,
            transactional=True,
        )
        return cls(
            model,
            validated_checkpoint=receipt,
            geometry_sampler=geometry_sampler,
            start_min_separation=start_min_separation,
        )

    @property
    def is_trained_inference(self) -> bool:
        return self._checkpoint_receipt is not None

    @property
    def start_min_separation(self) -> float | None:
        """Opt-in start rejection (research 2026-11-16); ``None``, the default, is off.

        With a value ``s`` in angstrom, every base draw of the geometry ODE whose
        two closest distinct atoms are at most ``s`` apart is drawn again from the
        attempt's own generator, which continues its stream, until the closest
        pair is farther than ``s``.  The distance is read on the state the field
        sees at ``t = 0`` (the torus-canonical start, expanded on the cover's cell,
        whose lengths are angstrom in every domain) by an exact image search.  An
        attempt fails with :class:`RecoverableGenerationError`, and so without a
        structure, when :data:`START_REJECTION_MAX_DRAWS` draws are all near or
        when the search needs more than :data:`START_REJECTION_MAX_IMAGES` images
        per axis; see :data:`START_REJECTION_MAX_DRAWS` for its siblings.

        The cover law and the trained field are unchanged; for each cover ``c``
        the base law ``B`` is replaced by its conditional law given ``A = {d_0 >
        s}`` (and given that the image search resolves, which every cell of the
        registered charts does).  That is not the joint law conditioned on ``A``,
        because ``B(A | c)`` depends on the cover.  For a geometry ``q`` whose
        start lies in ``A``, the log density of a treated sample is the flow log
        density (:meth:`geometry_log_density`) minus ``log B(A | c)``; for any
        other ``q`` the density is zero.  Given success, the draw cap leaves this
        law unchanged; it only lowers the success probability of a cover to ``1 -
        (1 - B(A | c))^1000``.  Within one cover the order of densities is unchanged, so the
        choice of :meth:`select_map_sample` keeps its meaning, although its
        ``log_geometry`` values are the unconditioned flow densities.
        :meth:`rank_samples`, whose scores are compared across covers, refuses
        while the option is set; samples carry no record of the option, so the
        guard follows the generator's current setting.  Training is unchanged and
        still uses the unconditioned base.
        """

        return self._start_min_separation

    @start_min_separation.setter
    def start_min_separation(self, value: float | None) -> None:
        if value is None:
            self._start_min_separation: float | None = None
            return
        if (
            isinstance(value, bool)
            or not isinstance(value, int | float)
            or not math.isfinite(float(value))
            or float(value) <= 0.0
        ):
            raise DomainError("start_min_separation must be None or a positive finite length")
        self._start_min_separation = float(value)

    def _reject_close_starts(
        self,
        plan: PaddedCoverPlan,
        layouts: Sequence[IntrinsicCoverLayout],
        bridges: Sequence[TorchCoverGeometry],
        generators: Sequence[torch.Generator],
        z_initial: Tensor,
        eta_initial: Tensor,
    ) -> None:
        """Redraw, in place, every row of the initial state that starts too close.

        Row ``b`` is drawn again with the same call as its first draw, from its own
        generator, until no two distinct atoms of its expanded start are within
        :attr:`start_min_separation`.  Rows are redrawn in rounds, every pending
        row once per round, so a row's accepted draw depends on its own generator
        only.  Rows that start far keep their first draw bitwise.
        """

        separation = self._start_min_separation
        if separation is None:
            return
        # The expanded cell is in angstrom in every domain (the field divides by
        # the reference length itself), so the separation is compared directly.
        radius = separation
        device, dtype = z_initial.device, z_initial.dtype

        def too_close(rows: Sequence[int]) -> list[int]:
            canonical = canonicalize_torus_validated(eta_initial, plan.period_basis)
            expanded = plan.expand(z_initial, canonical)
            return [
                row
                for row in rows
                if closest_distinct_pair_within(
                    expanded.lattice[row],
                    expanded.fractional_coordinates[row, : plan.atom_counts[row]],
                    radius,
                )
                is not None
            ]

        draws = [1] * plan.batch
        with torch.no_grad():
            pending = too_close(range(plan.batch))
            while pending:
                if any(draws[row] >= START_REJECTION_MAX_DRAWS for row in pending):
                    raise RecoverableGenerationError(
                        f"every one of {START_REJECTION_MAX_DRAWS} base draws has two "
                        f"distinct atoms within {separation} angstrom"
                    )
                for row in pending:
                    layout = layouts[row]
                    base = sample_product_base(
                        layout.lattice_dimension,
                        bridges[row].period_basis.to(dtype=dtype, device=device),
                        dtype=dtype,
                        device=device,
                        generator=generators[row],
                    )
                    z_initial[row, : layout.lattice_dimension] = base.z
                    eta_initial[row, : layout.intrinsic_dimension] = base.eta
                    draws[row] += 1
                pending = too_close(pending)

    def _assert_checkpoint_unchanged(self) -> None:
        receipt = self._checkpoint_receipt
        if receipt is None:
            return
        if model_state_fingerprint(self.model) != receipt.model_state_fingerprint:
            raise InvariantViolation(
                "model parameters changed after validated checkpoint binding"
            )

    def _device_dtype(self) -> tuple[torch.device, torch.dtype]:
        parameter = next(self.model.parameters())
        return parameter.device, parameter.dtype

    @staticmethod
    def _torch_generator(seed: int, device: torch.device) -> torch.Generator:
        generator = torch.Generator(device=device)
        generator.manual_seed(seed)
        return generator

    @staticmethod
    def _sample_residual_permutation(
        layout: IntrinsicCoverLayout,
        *,
        generator: torch.Generator | None,
        device: torch.device,
    ) -> tuple[int, ...]:
        """Uniformly augment every identical-orbit relabelling group."""

        permutation = list(range(len(layout.cover.orbits)))
        for group in layout.residual_orbit_groups:
            order = (
                torch.randperm(len(group), generator=generator, device=device)
                .detach()
                .cpu()
                .tolist()
            )
            for new_slot, old_offset in zip(group, order, strict=True):
                permutation[new_slot] = group[old_offset]
        return tuple(permutation)

    def _coupling_metrics(
        self, bridge: TorchCoverGeometry, device: torch.device
    ) -> tuple[Tensor, Tensor]:
        """Reference loss metrics ``(H_z(0), W(0)/N)`` of one cover in float64, cached."""

        key = (bridge.layout.cover, str(device), bridge.expansion_cell)
        cached = self._coupling_metric_cache.get(key)
        if cached is not None:
            return cached
        zeros_z = torch.zeros(bridge.lattice_dimension, dtype=torch.float64, device=device)
        zeros_eta = torch.zeros(bridge.intrinsic_dimension, dtype=torch.float64, device=device)
        with torch.no_grad():
            lattice_metric = bridge.lattice_pullback_metric(zeros_z).to(dtype=torch.float64)
            if bridge.intrinsic_dimension:
                expanded = bridge(zeros_z, zeros_eta)
                coordinate_metric = translation_quotient_normal_matrix(
                    expanded.jacobian_cartesian.to(dtype=torch.float64),
                    reference_length=(
                        self.model.scientific_config.domain.reference_length_angstrom
                    ),
                ) / float(bridge.atom_count)
            else:
                coordinate_metric = torch.zeros((0, 0), dtype=torch.float64, device=device)
        cached = (lattice_metric, coordinate_metric)
        self._coupling_metric_cache[key] = cached
        return cached

    def _residual_action_table(self, layout: IntrinsicCoverLayout) -> ResidualActionTable:
        """The cached digit coding of one cover's residual actions."""

        key = layout.cover
        table = self._residual_tables.get(key)
        if table is None:
            table = ResidualActionTable(
                layout, self.model.scientific_config.model.geometry_residual_gauge
            )
            self._residual_tables[key] = table
            if len(self._residual_tables) > LAYOUT_CACHE_SIZE:
                self._residual_tables.popitem(last=False)
        else:
            self._residual_tables.move_to_end(key)
        return table

    def _residual_action_candidates(
        self,
        layout: IntrinsicCoverLayout,
        records: int,
        candidates: int,
        *,
        generator: torch.Generator | None,
        device: torch.device,
    ) -> tuple[Tensor, Tensor | None, bool]:
        """Candidate residual actions of ``records`` records of one cover, float64.

        If the cover's action table has at most ``candidates`` digit rows, every
        row is a candidate of every record: the list covers the whole group, no
        random number is drawn, and the result is ``([1, M, d, d], [1, M, d] or
        None, True)``.  Otherwise each record gets ``candidates`` i.i.d. uniform
        actions, drawn with one ``randint`` per table column, and the result is
        ``([records, candidates, d, d], [records, candidates, d] or None, False)``.
        The shifts are ``None`` for ``relabel-v1``.
        """

        table = self._residual_action_table(layout)
        dimension = table.dimension
        if table.cardinality <= candidates:
            key = (layout.cover, str(device))
            cached = self._exhaustive_actions.get(key)
            if cached is None:
                matrices, shifts = table.actions(table.all_rows())
                cached = (
                    torch.as_tensor(matrices, dtype=torch.float64, device=device)[None],
                    None
                    if shifts is None
                    else torch.as_tensor(shifts, dtype=torch.float64, device=device)[None],
                )
                self._exhaustive_actions[key] = cached
                if len(self._exhaustive_actions) > LAYOUT_CACHE_SIZE:
                    self._exhaustive_actions.popitem(last=False)
            else:
                self._exhaustive_actions.move_to_end(key)
            return cached[0], cached[1], True
        count = records * candidates
        rows = (
            torch.stack(
                [
                    torch.randint(modulus, (count,), generator=generator, device=device)
                    for modulus in table.moduli
                ],
                dim=1,
            )
            .cpu()
            .numpy()
        )
        matrices, shifts = table.actions(rows)
        return (
            torch.as_tensor(matrices, dtype=torch.float64, device=device).reshape(
                records, candidates, dimension, dimension
            ),
            None
            if shifts is None
            else torch.as_tensor(shifts, dtype=torch.float64, device=device).reshape(
                records, candidates, dimension
            ),
            False,
        )

    def _gauge_aligned_coupling(
        self,
        bridge: TorchCoverGeometry,
        base: FlowState,
        endpoint: FlowState,
        actions: Tensor,
        shifts: Tensor | None,
        period: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor | None, tuple[float, float]]:
        """The ``gauge-ot-v1`` coupling of ``n`` records of one cover.

        ``actions`` is ``[n, K, d, d]`` and ``shifts`` ``[n, K, d]`` or ``None``
        (float64); a leading size of one shares one candidate list among all
        records.  For base ``i``, record ``j`` and candidate ``gamma_jk = (A, m)``
        the cost is ``d_z(z0_i, z1_j) + d_eta(gamma_jk^-1 eta0_i, eta1_j)`` with
        ``gamma^-1 eta = (eta - m) A^-1`` and the half-open lift distance in the
        reference loss metric.  The assignment that minimizes the summed cost
        pairs bases with records; each record keeps its minimizing action and the
        path is built from the unpushed base ``gamma^-1 eta0`` and pushed by
        ``gamma``, so the path starts at the drawn base sample and ends at
        ``gamma eta1``.  Returns ``order`` (``order[j]`` is the base paired with
        record ``j``), the unpushed base ``eta`` in record order, the chosen
        actions and shifts in record order, and the sums of the assigned and of
        the identity-pairing costs.  Proposition 2.9d requires all ``n`` records
        to belong to one cover; the caller forms the blocks.
        """

        from scipy.optimize import linear_sum_assignment

        batch, dimension = base.eta.shape
        dtype, device = base.z.dtype, base.z.device
        if actions.ndim != 4 or actions.shape[0] not in (1, batch):
            raise ValueError("candidate actions must be [n or 1, K, d, d]")
        if actions.shape[2:] != (dimension, dimension) or actions.shape[1] < 1:
            raise ValueError("candidate actions disagree with the torus dimension")
        if shifts is not None and shifts.shape != actions.shape[:3]:
            raise ValueError("candidate shifts must be [n or 1, K, d]")
        lattice_metric, coordinate_metric = self._coupling_metrics(bridge, device)
        z0 = base.z.to(dtype=torch.float64)
        z1 = endpoint.z.to(dtype=torch.float64)
        dz = z0[:, None, :] - z1[None, :, :]
        cost = torch.einsum("ijd,de,ije->ij", dz, lattice_metric, dz)
        actions = actions.to(dtype=torch.float64)
        candidate_count = actions.shape[1]
        record_index = torch.arange(batch, device=device)
        if dimension:
            inverse = torch.linalg.inv(actions)
            eta0 = base.eta.to(dtype=torch.float64)
            eta1 = endpoint.eta.to(dtype=torch.float64)
            shifted = eta0[:, None, None, :]
            if shifts is not None:
                shifted = shifted - shifts.to(dtype=torch.float64)[None]
            # [n, 1|n, 1|K, 1, d] @ [1, 1|n, K, d, d] -> [n, n, K, d]
            unpushed = torch.matmul(shifted[..., None, :], inverse[None]).squeeze(-2)
            unpushed = unpushed.expand(batch, batch, candidate_count, dimension)
            period64 = period.to(dtype=torch.float64)
            lift = shortest_coordinate_lift(
                unpushed, eta1[None, :, None, :].expand_as(unpushed), period64
            )
            cost_eta = torch.einsum("ijkd,de,ijke->ijk", lift, coordinate_metric, lift)
            best_cost, best_k = cost_eta.min(dim=-1)
            cost = cost + best_cost
        else:
            best_k = torch.zeros(batch, batch, dtype=torch.long, device=device)
        rows, cols = linear_sum_assignment(cost.detach().cpu().numpy())
        order = np.empty(batch, dtype=np.int64)
        order[cols] = rows  # order[j] = the base paired with record j
        order_t = torch.as_tensor(order, device=device)
        chosen = best_k[order_t, record_index]
        expanded_actions = actions.expand(batch, -1, -1, -1)
        chosen_actions = expanded_actions[record_index, chosen].to(dtype=dtype)
        chosen_shifts = (
            None
            if shifts is None
            else shifts.expand(batch, -1, -1)[record_index, chosen].to(dtype=dtype)
        )
        if dimension:
            eta_base = canonicalize_torus_validated(
                unpushed[order_t, record_index, chosen].to(dtype=dtype), period
            )
        else:
            eta_base = base.eta[order_t]
        sums = (
            float(cost[order_t, record_index].sum().item()),
            float(cost.diagonal().sum().item()),
        )
        return order_t, eta_base, chosen_actions, chosen_shifts, sums

    def _gauge_aligned_batch(
        self,
        blocks: list[tuple[list[int], TorchCoverGeometry, Tensor, Tensor | None, bool]],
        base: FlowState,
        endpoint: FlowState,
        period: Tensor,
    ) -> tuple[FlowState, Tensor, Tensor | None, dict[str, float]]:
        """Couple a batch block by block, one block per cover.

        The geometry field is conditioned on the cover, so Proposition 2.9d
        applies to the records of one cover: they share the conditional law and
        the residual group.  An assignment across covers would keep the base
        marginal of the batch but not that of each cover.  Every block therefore
        permutes only its own base samples.  ``blocks`` holds, per cover, the
        batch positions, the bridge and the candidates of
        :meth:`_residual_action_candidates`.  Returns the coupled base, the
        chosen actions ``[B, d, d]``, the chosen shifts ``[B, d]`` or ``None``,
        and the statistics.
        """

        batch, dimension = base.eta.shape
        dtype, device = base.z.dtype, base.z.device
        positions = sorted(index for indices, *_ in blocks for index in indices)
        if positions != list(range(batch)):
            raise ValueError("coupling blocks must partition the batch")
        order = torch.empty(batch, dtype=torch.long, device=device)
        eta_base = torch.empty_like(base.eta)
        torus_action = torch.empty(batch, dimension, dimension, dtype=dtype, device=device)
        has_shift = any(shifts is not None for _, _, _, shifts, _ in blocks)
        torus_shift = (
            torch.zeros(batch, dimension, dtype=dtype, device=device) if has_shift else None
        )
        assigned = identity = 0.0
        exact_records = 0
        for indices, bridge, actions, shifts, exhaustive in blocks:
            index = torch.as_tensor(indices, dtype=torch.long, device=device)
            local_order, local_eta, chosen_actions, chosen_shifts, sums = (
                self._gauge_aligned_coupling(
                    bridge,
                    FlowState(base.z[index], base.eta[index]),
                    FlowState(endpoint.z[index], endpoint.eta[index]),
                    actions,
                    shifts,
                    period,
                )
            )
            order[index] = index[local_order]
            eta_base[index] = local_eta
            torus_action[index] = chosen_actions
            if torus_shift is not None and chosen_shifts is not None:
                torus_shift[index] = chosen_shifts
            assigned += sums[0]
            identity += sums[1]
            exact_records += len(indices) if exhaustive else 0
        stats = {
            "assigned_mean_cost": assigned / batch,
            "identity_mean_cost": identity / batch,
            "block_count": float(len(blocks)),
            "exact_quotient_fraction": exact_records / batch,
        }
        return FlowState(base.z[order], eta_base), torus_action, torus_shift, stats

    def _sample_residual_torus_action(
        self,
        layout: IntrinsicCoverLayout,
        *,
        generator: torch.Generator | None,
        device: torch.device,
        dtype: torch.dtype,
    ) -> tuple[Tensor, Tensor | None]:
        """One uniformly sampled residual action ``eta -> eta A + m``.

        ``relabel-v1`` makes exactly the legacy draws (permutation, then P-1
        signs) and has no translation.  ``normalizer-v2`` draws the permutation,
        then one per-orbit gauge per orbit, then one normalizer automorphism,
        each uniformly: the uniform law on the certified group.  ``normalizer-v3``
        makes the same draws over the group with the improper normalizer elements.
        """

        permutation = self._sample_residual_permutation(
            layout, generator=generator, device=device
        )
        kind = self.model.scientific_config.model.geometry_residual_gauge
        if kind == "relabel-v1":
            signs = self._sample_residual_signs(layout, generator=generator, device=device)
            action = torch.as_tensor(
                layout.residual_action_matrix(permutation, orbit_signs=signs),
                dtype=dtype,
                device=device,
            )
            return action, None
        gauge = cover_residual_gauge(layout, improper=kind == "normalizer-v3")

        def draw(size: int) -> int:
            return int(torch.randint(size, (), generator=generator, device=device).item())

        orbit_indices = [draw(len(choices)) for choices in gauge.orbit_choices]
        matrix, shift = gauge.eta_action(
            orbit_indices, permutation, draw(len(gauge.normalizer_maps))
        )
        return (
            torch.as_tensor(matrix, dtype=dtype, device=device),
            torch.as_tensor(shift, dtype=dtype, device=device),
        )

    @staticmethod
    def _sample_residual_signs(
        layout: IntrinsicCoverLayout,
        *,
        generator: torch.Generator | None,
        device: torch.device,
    ) -> tuple[int, ...]:
        signs = [1] * len(layout.cover.orbits)
        if layout.setting.key == "P-1":
            for index, row in enumerate(layout.positions):
                if row.dimension:
                    bit = int(
                        torch.randint(
                            0,
                            2,
                            (),
                            generator=generator,
                            device=device,
                        ).item()
                    )
                    signs[index] = 1 if bit == 0 else -1
        return tuple(signs)

    def _composition(
        self,
        request: GenerationRequest[Composition],
        generator: torch.Generator,
        device: torch.device,
    ) -> Composition:
        if request.mode is GenerationMode.CSP:
            if not isinstance(request.composition, Composition):
                raise TypeError("CSP request composition must be a Composition")
            self.model.validate_composition(request.composition)
            return request.composition
        support = request.species if request.mode is GenerationMode.SPECIES_DNG else None
        if support is not None:
            try:
                # Validate user conditioning before entering the neural loop so
                # an invalid request is a typed, recoverable domain failure.
                self.model.composition_grammar.support_prefix(support)
            except (CompositionError, TypeError) as exc:
                raise DomainError(str(exc)) from exc
        return sample_composition(
            self.model.composition_model,
            self.model.composition_codec,
            support=support,
            generator=generator,
            device=device,
        )

    def _sample_geometry(
        self,
        cover_sample: NeuralCoverSample,
        generator: torch.Generator,
    ) -> tuple[ContinuousGeometry, ContinuousDecodedCrystal]:
        geometry = self._integrate_geometry_batch((cover_sample.cover,), (generator,))[0]
        layout = self.model.layout(cover_sample.cover)
        return geometry, layout.decode(
            geometry,
            collision_tolerance=(self.model.scientific_config.numerics.collision_atol_angstrom),
            symmetry_tolerance=self.model.scientific_config.numerics.symmetry_atol,
        )

    def _integrate_geometry_batch(
        self,
        covers: tuple[ContinuousCover, ...],
        generators: tuple[torch.Generator, ...],
    ) -> tuple[ContinuousGeometry, ...]:
        """Integrate one topology-homogeneous candidate batch in one RK4 solve."""

        if not covers or len(covers) != len(generators):
            raise DomainError(
                "geometry batch requires equal nonempty cover and generator tuples"
            )
        if any(not isinstance(cover, ContinuousCover) for cover in covers):
            raise TypeError("geometry batch covers must be ContinuousCover values")
        layouts = tuple(self.model.layout(cover) for cover in covers)
        signature = _layout_batch_signature(layouts[0])
        if any(_layout_batch_signature(item) != signature for item in layouts[1:]):
            raise DomainError("geometry ODE batch mixes incompatible cover topologies")
        device, dtype = self._device_dtype()
        bridges = tuple(self._bridge(layout, device, dtype) for layout in layouts)
        capacity = self._candidate_capacity(bridges[0], dtype)
        if len(covers) > capacity:
            geometries: list[ContinuousGeometry] = []
            for start in range(0, len(covers), capacity):
                stop = start + capacity
                geometries.extend(
                    self._integrate_geometry_batch(covers[start:stop], generators[start:stop])
                )
            return tuple(geometries)
        if self.model.geometry_field.config.backbone == "equivariant_local":
            for layout in layouts:
                certified_candidates = layout.periodic_candidate_box_upper_bound(
                    self.model.geometry_field.config.cutoff
                )
                certified_edges = layout.periodic_edge_upper_bound(
                    self.model.geometry_field.config.cutoff
                )
                if (
                    certified_candidates
                    > self.model.geometry_field.config.max_candidates_per_pair
                ):
                    raise DomainError(
                        "cover exceeds the configured pair-image allocation support"
                    )
                if certified_edges > self.model.geometry_field.config.max_periodic_edges:
                    raise DomainError("cover exceeds the configured periodic-edge support")
        period = bridges[0].period_basis.to(dtype=dtype, device=device)
        base_samples = tuple(
            sample_product_base(
                layout.lattice_dimension,
                period,
                dtype=dtype,
                device=device,
                generator=generator,
            )
            for layout, generator in zip(layouts, generators, strict=True)
        )
        initial = FlowState(
            torch.stack([item.z for item in base_samples]),
            torch.stack([item.eta for item in base_samples]),
        )
        if self._start_min_separation is not None:
            self._reject_close_starts(
                PaddedCoverPlan(bridges, dtype=dtype, device=device),
                layouts,
                bridges,
                generators,
                initial.z,
                initial.eta,
            )
        velocity = self._batched_velocity(covers, bridges, device, dtype)
        final = integrate_with_endpoint_estimate(
            velocity,
            initial,
            period,
            num_steps=self.model.scientific_config.numerics.ode_steps,
            terminal_time=self.model.scientific_config.numerics.ode_terminal_time,
        )
        z_values = final.z.detach().cpu().numpy()
        eta_values = final.eta.detach().cpu().numpy()
        return tuple(
            ContinuousGeometry(z_values[index], eta_values[index])
            for index in range(len(covers))
        )

    def _candidate_capacity(self, bridge: TorchCoverGeometry, dtype: torch.dtype) -> int:
        """Largest number of candidates of one topology that one field call may hold."""

        field = self.model.geometry_field
        if field.config.backbone != "invariant_global":
            # The local ablation certifies resources per crystal.  Until it
            # has a union-workspace bound, candidate pools use one crystal per
            # ODE chunk rather than growing a packed union without a bound.
            return 1
        capacity = int(
            field.transformer.maximum_batch_size(atom_count=bridge.atom_count, dtype=dtype)
        )
        if capacity == 0:
            raise DomainError(
                "one geometry candidate exceeds the configured dense pair/workspace budget"
            )
        return capacity

    def _single_cover_velocity(
        self,
        cover: ContinuousCover,
        bridge: TorchCoverGeometry,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Callable[[Tensor, Tensor, Tensor], object]:
        """The field of one cover for any number of rows ``(t, z[M, d_z], eta[M, d_eta])``.

        One batched bridge expansion per call, equal to the per-row expansions up
        to rounding, and one composition context shared by every row.  A row's
        velocity depends on that row only, which is what the replicated-batch
        trace of :func:`crystalatlas.flow.log_density_transport` relies on
        (``tests/test_flow_log_density.py`` compares it with the coordinate loop).
        """

        with torch.no_grad():
            # Fixed along the trajectory and not a function of the state.
            composition_context = self.model.geometry_field.encode_composition(
                torch.as_tensor(cover.composition.support, dtype=torch.long, device=device)[
                    None
                ],
                torch.as_tensor(cover.composition.coefficients, dtype=dtype, device=device)[
                    None
                ],
            )

        def velocity(time: Tensor, z: Tensor, eta: Tensor) -> object:
            expanded = bridge.forward_batch(z, eta)
            return self.model.geometry_field(
                time,
                z,
                eta,
                composition_context.expand(z.shape[0], -1),
                atomic_numbers=expanded.atomic_numbers,
                atom_features=expanded.atom_features,
                fractional_coordinates=expanded.fractional_coordinates,
                lattice=expanded.lattice,
                bounded_lattice_coordinates=expanded.bounded_lattice_coordinates,
                jacobian_cartesian=expanded.jacobian_cartesian,
                geometry_is_certified=True,
                lattice_commutant_basis=expanded.lattice_commutant_basis,
            )

        return velocity

    def _batched_velocity(
        self,
        covers: tuple[ContinuousCover, ...],
        bridges: tuple[TorchCoverGeometry, ...],
        device: torch.device,
        dtype: torch.dtype,
        *,
        per_member: bool = False,
    ) -> Callable[[Tensor, Tensor, Tensor], object]:
        """The field as a function of ``(t, z[B, d_z], eta[B, d_eta])`` for one bucket.

        Several members of one cover share one batched bridge expansion unless
        ``per_member`` asks for the reference path, which expands every member
        with its own bridge call.  A single member always takes the reference
        path, so single-candidate results do not depend on this choice.
        """

        if (
            not per_member
            and len(covers) > 1
            and all(bridge is bridges[0] for bridge in bridges)
            and all(cover == covers[0] for cover in covers)
        ):
            return self._single_cover_velocity(covers[0], bridges[0], device, dtype)
        composition_numbers = torch.stack(
            [
                torch.as_tensor(cover.composition.support, dtype=torch.long, device=device)
                for cover in covers
            ]
        )
        composition_counts = torch.stack(
            [
                torch.as_tensor(cover.composition.coefficients, dtype=dtype, device=device)
                for cover in covers
            ]
        )
        # The composition context is fixed along the trajectory: encode it once
        # instead of re-validating the raw values at every RK4 stage (review H1).
        composition_context = self.model.geometry_field.encode_composition(
            composition_numbers, composition_counts
        )

        def velocity(time: Tensor, z: Tensor, eta: Tensor) -> object:
            expanded = [bridge(z[index], eta[index]) for index, bridge in enumerate(bridges)]
            return self.model.geometry_field(
                time,
                z,
                eta,
                composition_context,
                atomic_numbers=torch.stack([item.atomic_numbers for item in expanded]),
                atom_features=torch.stack([item.atom_features for item in expanded]),
                fractional_coordinates=torch.stack(
                    [item.fractional_coordinates for item in expanded]
                ),
                lattice=torch.stack([item.lattice for item in expanded]),
                bounded_lattice_coordinates=torch.stack(
                    [item.bounded_lattice_coordinates for item in expanded]
                ),
                jacobian_cartesian=torch.stack([item.jacobian_cartesian for item in expanded]),
                geometry_is_certified=True,
                # One topology bucket shares one chart, hence one commutant basis.
                lattice_commutant_basis=expanded[0].lattice_commutant_basis,
            )

        return velocity

    def geometry_log_density(
        self,
        cover: ContinuousCover,
        geometries: Sequence[ContinuousGeometry],
        *,
        num_steps: int | None = None,
        divergence: str = "exact",
        probes: int = 8,
        seed: int = 0,
    ) -> np.ndarray:
        """Exact model log density of geometries under the flow law of one cover (R2).

        ``log rho_{theta, alpha}(q_1) = log rho_0(q_0) - int_0^1 div v dt`` along the
        backward trajectory of the learned ODE (instantaneous change of variables
        on ``R^{d_z} x`` the flat intrinsic torus).  It is the density of the
        dequantized endpoint law the flow was trained to match, with respect to
        the cover's own product volume: values are comparable between geometries
        of one cover, and combine with the exact selector probability for that
        cover; densities of covers of different dimension are not comparable as
        numbers.  No energy model is involved.

        ``divergence="exact"`` takes the exact trace of the field Jacobian from
        replicated rows: one forward and one reverse pass per group of coordinates
        that fits the dense candidate budget.  ``"exact-loop"`` is the reference
        implementation of the same trace, one reverse pass per coordinate on the
        per-member expansion; the two agree up to rounding.  ``"hutchinson"`` is
        the unbiased stochastic estimator.  More geometries than the candidate
        budget holds are integrated in consecutive chunks.

        The base density is the unconditioned one.  Under start rejection
        (:attr:`start_min_separation`) the log density of a treated sample whose
        start lies in ``A`` is this value minus ``log B(A | c)``, one unknown
        constant per cover.
        """

        if not isinstance(cover, ContinuousCover):
            raise TypeError("cover must be a ContinuousCover")
        if not geometries:
            raise DomainError("geometry_log_density needs at least one geometry")
        if divergence not in {"exact", "exact-loop", "hutchinson"}:
            raise DomainError("divergence must be 'exact', 'exact-loop' or 'hutchinson'")
        layout = self.model.layout(cover)
        device, dtype = self._device_dtype()
        bridge = self._bridge(layout, device, dtype)
        batch = len(geometries)
        for geometry in geometries:
            if geometry.lattice_latent.shape != (layout.lattice_dimension,):
                raise DomainError("lattice latent has the wrong cover dimension")
            if geometry.intrinsic_coordinates.shape != (layout.intrinsic_dimension,):
                raise DomainError("intrinsic coordinates have the wrong cover dimension")
        z1 = torch.tensor(
            np.stack([g.lattice_latent for g in geometries]), dtype=dtype, device=device
        )
        eta1 = torch.tensor(
            np.stack([g.intrinsic_coordinates for g in geometries]).reshape(
                batch, layout.intrinsic_dimension
            ),
            dtype=dtype,
            device=device,
        )
        period = bridge.period_basis.to(dtype=dtype, device=device)
        steps = (
            self.model.scientific_config.numerics.ode_steps if num_steps is None else num_steps
        )
        capacity = self._candidate_capacity(bridge, dtype)
        members = min(batch, capacity)
        # Replicated rows of one chunk: members x row_chunk <= the candidate budget.
        row_chunk = max(1, capacity // members)
        generator = self._torch_generator(seed, device)
        was_training = self.model.training
        self.model.eval()
        try:
            values: list[Tensor] = []
            for begin in range(0, batch, members):
                count = min(members, batch - begin)
                velocity = (
                    self._batched_velocity(
                        (cover,) * count, (bridge,) * count, device, dtype, per_member=True
                    )
                    if divergence == "exact-loop"
                    else self._single_cover_velocity(cover, bridge, device, dtype)
                )
                with torch.no_grad():
                    start, integral = log_density_transport(
                        velocity,
                        FlowState(z1[begin : begin + count], eta1[begin : begin + count]),
                        period,
                        num_steps=steps,
                        divergence={"exact": "exact-batched", "exact-loop": "exact"}.get(
                            divergence, divergence
                        ),
                        probes=probes,
                        generator=generator,
                        row_chunk=row_chunk if divergence == "exact" else None,
                    )
                    values.append(base_log_density(start.z, period) - integral)
        finally:
            self.model.train(was_training)
        return torch.cat(values).detach().cpu().numpy().astype(np.float64)

    def rank_samples(
        self,
        samples: Sequence[LearnedSample],
        *,
        num_steps: int | None = None,
        divergence: str = "exact",
        probes: int = 8,
        seed: int = 0,
    ) -> tuple[SampleScore, ...]:
        """Exact model log probabilities of generated samples, in input order (R2).

        Samples of one cover are scored in one batched backward integration.  The
        caller ranks by ``log_joint`` among samples of equal ``dimension``; see
        :class:`SampleScore` for why cross-dimension comparison is not defined.
        """

        if self._start_min_separation is not None:
            raise DomainError(
                "rank_samples compares scores across covers; under start rejection each "
                "cover's geometry density lacks the unknown factor 1 / B(A | c)"
            )
        if not samples:
            return ()
        groups: dict[ContinuousCover, list[int]] = {}
        for index, sample in enumerate(samples):
            if not isinstance(sample, LearnedSample):
                raise TypeError("rank_samples expects LearnedSample values")
            groups.setdefault(sample.cover_sample.cover, []).append(index)
        scores: list[SampleScore | None] = [None] * len(samples)
        for cover, indices in groups.items():
            layout = self.model.layout(cover)
            densities = self.geometry_log_density(
                cover,
                [samples[index].geometry for index in indices],
                num_steps=num_steps,
                divergence=divergence,
                probes=probes,
                seed=seed,
            )
            for index, value in zip(indices, densities, strict=True):
                scores[index] = SampleScore(
                    sample_index=index,
                    log_cover=float(samples[index].cover_sample.log_probability),
                    log_geometry=float(value),
                    dimension=layout.lattice_dimension + layout.intrinsic_dimension,
                )
        return tuple(score for score in scores if score is not None)

    def select_map_sample(
        self,
        samples: Sequence[LearnedSample],
        *,
        num_steps: int | None = None,
        divergence: str = "exact",
        probes: int = 8,
        seed: int = 0,
    ) -> MapSelection:
        """Hierarchical maximum a posteriori choice among generated samples (R2).

        The model law is ``p(alpha | c) rho_alpha(q) dq``: a probability mass over
        covers times a density on the chart of the cover.  Masses of different
        covers are comparable numbers; densities of covers of different dimension
        are not (Theorem 2.6, :class:`SampleScore`).  The rule therefore takes the
        cover of largest selector mass among the samples first, and inside that
        cover the sample of largest exact geometry log density.  The first
        occurrence wins a tie at both levels.  Only the samples of the selected
        cover are integrated.  Under a primitive-count condition the selector mass
        lacks the acceptance normalizer, which is one constant for the request and
        does not change the order.
        """

        if not samples:
            raise DomainError("select_map_sample needs at least one sample")
        groups: dict[ContinuousCover, list[int]] = {}
        for index, sample in enumerate(samples):
            if not isinstance(sample, LearnedSample):
                raise TypeError("select_map_sample expects LearnedSample values")
            groups.setdefault(sample.cover_sample.cover, []).append(index)
        masses: dict[ContinuousCover, float] = {}
        for cover, indices in groups.items():
            values = [float(samples[index].cover_sample.log_probability) for index in indices]
            if not all(math.isfinite(value) for value in values):
                raise InvariantViolation("a sampled cover has a nonfinite log probability")
            if max(values) - min(values) > 1.0e-6:
                # One cover is one table of the masked law, so its mass is one number.
                raise InvariantViolation(
                    "samples of one cover disagree on its selector log probability"
                )
            masses[cover] = values[0]
        # ``max`` returns the first maximum; dict order is first occurrence.
        chosen = max(groups, key=lambda cover: masses[cover])
        members = tuple(groups[chosen])
        densities = self.geometry_log_density(
            chosen,
            [samples[index].geometry for index in members],
            num_steps=num_steps,
            divergence=divergence,
            probes=probes,
            seed=seed,
        )
        if not np.isfinite(densities).all():
            raise InvariantViolation("the geometry log density of a sample is not finite")
        best = int(np.argmax(densities))  # the first maximum
        return MapSelection(
            sample_index=members[best],
            members=members,
            log_cover=masses[chosen],
            log_geometry=tuple(float(value) for value in densities),
        )

    def sample_geometry_given_cover(
        self,
        cover: ContinuousCover,
        *,
        seed: int,
    ) -> tuple[ContinuousGeometry, ContinuousDecodedCrystal]:
        """Sample geometry conditional on an externally supplied exact cover.

        This is the oracle-scaffold arm required to distinguish geometry-field
        error from cover-selection error.  It performs one declared attempt,
        with no retries, repair, filtering, or relaxation, except the redraw of
        a near start when :attr:`start_min_separation` is set.
        """

        if not isinstance(cover, ContinuousCover):
            raise TypeError("cover must be a ContinuousCover")
        if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 2**64:
            raise DomainError("geometry seed must be an unsigned 64-bit integer")
        device, _ = self._device_dtype()
        self._assert_checkpoint_unchanged()
        was_training = self.model.training
        self.model.eval()
        try:
            with torch.no_grad():
                geometry = self._integrate_geometry_batch(
                    (cover,),
                    (self._torch_generator(seed, device),),
                )[0]
            decoded = self.model.layout(cover).decode(
                geometry,
                collision_tolerance=(
                    self.model.scientific_config.numerics.collision_atol_angstrom
                ),
                symmetry_tolerance=self.model.scientific_config.numerics.symmetry_atol,
            )
            return geometry, decoded
        finally:
            self.model.train(was_training)

    def sample(
        self, request: GenerationRequest[Composition]
    ) -> tuple[LearnedGenerationTrace, ...]:
        """Sample every requested attempt once and preserve any failed stage."""

        device, _ = self._device_dtype()
        self._assert_checkpoint_unchanged()
        was_training = self.model.training
        self.model.eval()
        try:
            return self._sample_in_eval_mode(request, device)
        finally:
            self.model.train(was_training)

    def _integrate_members(
        self,
        covers: tuple[ContinuousCover, ...],
        generators: tuple[torch.Generator, ...],
    ) -> tuple[ContinuousGeometry | Exception, ...]:
        """Integrate one topology bucket; a member's own failure stays its own.

        Review 2026-10-29 (m3): a ``TangentRankError`` or ``ValueError`` raised
        by the field for one member used to abort the whole ``sample`` call.
        When the batched solve fails, every member is replayed alone from the
        generator state it started with, so the survivors keep exactly their
        draws, and the failing member's error is returned in its slot for the
        ``geometry/ode`` trace.  Resource exhaustion and invariant violations
        are not attempt failures and still propagate.
        """

        states = tuple(generator.get_state() for generator in generators)
        try:
            with torch.no_grad():
                return tuple(self._integrate_geometry_batch(covers, generators))
        except _MEMBER_ODE_ERRORS as exc:
            if len(covers) == 1:
                return (exc,)
        outcomes: list[ContinuousGeometry | Exception] = []
        for cover, generator, state in zip(covers, generators, states, strict=True):
            generator.set_state(state)
            try:
                with torch.no_grad():
                    outcomes.append(self._integrate_geometry_batch((cover,), (generator,))[0])
            except _MEMBER_ODE_ERRORS as exc:
                outcomes.append(exc)
        return tuple(outcomes)

    def _mixed_capacity(self, atom_count: int, dtype: torch.dtype) -> int:
        """Largest padded batch of ``atom_count`` atom slots one field call may hold."""

        return int(
            self.model.geometry_field.transformer.maximum_batch_size(
                atom_count=atom_count, dtype=dtype
            )
        )

    def _mixed_chunks(
        self, bridges: tuple[TorchCoverGeometry, ...], dtype: torch.dtype
    ) -> tuple[tuple[int, ...], ...]:
        """Member indices of each padded solve of the ``mixed-v1`` sampler.

        Members are taken in ascending atom count, so a chunk is exactly as
        wide as its newest member and holds at most :meth:`_mixed_capacity`
        crystals of that width.  A member that exceeds the budget alone is its
        own chunk; :meth:`_integrate_mixed_chunk` refuses it.
        """

        order = sorted(
            range(len(bridges)), key=lambda index: (bridges[index].atom_count, index)
        )
        chunks: list[tuple[int, ...]] = []
        current: list[int] = []
        for index in order:
            capacity = max(1, self._mixed_capacity(bridges[index].atom_count, dtype))
            if len(current) >= capacity:
                chunks.append(tuple(current))
                current = []
            current.append(index)
        if current:
            chunks.append(tuple(current))
        return tuple(chunks)

    def _integrate_mixed_chunk(
        self,
        covers: tuple[ContinuousCover, ...],
        generators: tuple[torch.Generator, ...],
    ) -> tuple[ContinuousGeometry, ...]:
        """Integrate candidates of any covers in one padded RK4 solve.

        Every member draws its base point from its own generator exactly as
        :meth:`_integrate_geometry_batch` does, flows under its own lattice
        chart and on its own torus (row-wise period bases), and reads the field
        through :meth:`IntrinsicGeometryField.forward_padded`, whose real
        outputs equal the per-topology calls to rounding and whose padded
        velocity slots are exactly zero.  The result therefore equals the
        bucket solve to rounding (``tests/test_mixed_sampler.py``).
        """

        if not covers or len(covers) != len(generators):
            raise DomainError(
                "geometry batch requires equal nonempty cover and generator tuples"
            )
        if any(not isinstance(cover, ContinuousCover) for cover in covers):
            raise TypeError("geometry batch covers must be ContinuousCover values")
        field = self.model.geometry_field
        if field.config.backbone != "invariant_global":
            raise DomainError(
                "the mixed-v1 geometry sampler requires the dense invariant_global backbone"
            )
        device, dtype = self._device_dtype()
        layouts = tuple(self.model.layout(cover) for cover in covers)
        bridges = tuple(self._bridge(layout, device, dtype) for layout in layouts)
        capacity = self._mixed_capacity(max(bridge.atom_count for bridge in bridges), dtype)
        if capacity == 0:
            raise DomainError(
                "one geometry candidate exceeds the configured dense pair/workspace budget"
            )
        if len(covers) > capacity:
            raise InvariantViolation("mixed geometry chunk exceeds its planned capacity")
        plan = PaddedCoverPlan(bridges, dtype=dtype, device=device)
        z_initial = torch.zeros(len(covers), plan.lattice_width, dtype=dtype, device=device)
        eta_initial = torch.zeros(
            len(covers), plan.intrinsic_width, dtype=dtype, device=device
        )
        # The composition context is fixed along the trajectory; one encoding
        # per distinct composition, in the unpadded form of the bucket sampler.
        contexts: dict[tuple[object, ...], Tensor] = {}
        context_rows: list[Tensor] = []
        for row, (cover, layout, bridge, generator) in enumerate(
            zip(covers, layouts, bridges, generators, strict=True)
        ):
            base = sample_product_base(
                layout.lattice_dimension,
                bridge.period_basis.to(dtype=dtype, device=device),
                dtype=dtype,
                device=device,
                generator=generator,
            )
            z_initial[row, : layout.lattice_dimension] = base.z
            eta_initial[row, : layout.intrinsic_dimension] = base.eta
            key = (cover.composition.support, cover.composition.coefficients)
            if key not in contexts:
                contexts[key] = field.encode_composition(
                    torch.as_tensor(
                        cover.composition.support, dtype=torch.long, device=device
                    )[None],
                    torch.as_tensor(
                        cover.composition.coefficients, dtype=dtype, device=device
                    )[None],
                )
            context_rows.append(contexts[key])
        if self._start_min_separation is not None:
            self._reject_close_starts(
                plan, layouts, bridges, generators, z_initial, eta_initial
            )
        composition_context = torch.cat(context_rows)
        commutant = (
            plan.lattice_commutant_basis
            if field.config.lattice_head == "commutant-v2"
            else None
        )

        def velocity(time: Tensor, z: Tensor, eta: Tensor) -> object:
            expanded = plan.expand(z, eta)
            return field.forward_padded(
                time,
                z,
                eta,
                composition_context,
                atomic_numbers=plan.atomic_numbers,
                atom_mask=plan.atom_mask,
                atom_features=plan.atom_features,
                fractional_coordinates=expanded.fractional_coordinates,
                lattice=expanded.lattice,
                bounded_lattice_coordinates=expanded.bounded_lattice_coordinates,
                jacobian_cartesian=expanded.jacobian_cartesian,
                lattice_dimensions=plan.lattice_dimensions,
                intrinsic_dimensions=plan.intrinsic_dimensions,
                lattice_commutant_basis=commutant,
            )

        final = integrate_with_endpoint_estimate(
            velocity,
            FlowState(z_initial, eta_initial),
            plan.period_basis,
            num_steps=self.model.scientific_config.numerics.ode_steps,
            terminal_time=self.model.scientific_config.numerics.ode_terminal_time,
        )
        z_values = final.z.detach().cpu().numpy()
        eta_values = final.eta.detach().cpu().numpy()
        return tuple(
            ContinuousGeometry(
                z_values[row, : layout.lattice_dimension],
                eta_values[row, : layout.intrinsic_dimension],
            )
            for row, layout in enumerate(layouts)
        )

    def _integrate_mixed_members(
        self,
        covers: tuple[ContinuousCover, ...],
        generators: tuple[torch.Generator, ...],
    ) -> tuple[ContinuousGeometry | Exception, ...]:
        """One padded solve, halved on a member error until the failure is isolated.

        The generators return to the states they started with before any
        replay, so survivors keep exactly their draws.  A member that fails
        alone is decided by the bucket sampler's single-member solve
        (:meth:`_integrate_members`): its reported error is the reference
        path's, and so is its result if only the padded call failed.
        """

        states = tuple(generator.get_state() for generator in generators)
        try:
            with torch.no_grad():
                return tuple(self._integrate_mixed_chunk(covers, generators))
        except _MEMBER_ODE_ERRORS:
            for generator, state in zip(generators, states, strict=True):
                generator.set_state(state)
        if len(covers) == 1:
            return self._integrate_members(covers, generators)
        middle = len(covers) // 2
        return (
            *self._integrate_mixed_members(covers[:middle], generators[:middle]),
            *self._integrate_mixed_members(covers[middle:], generators[middle:]),
        )

    def _integrate_members_mixed(
        self,
        covers: tuple[ContinuousCover, ...],
        generators: tuple[torch.Generator, ...],
    ) -> tuple[ContinuousGeometry | Exception, ...]:
        """Integrate attempts of any topologies; a member's own failure stays its own.

        The ``mixed-v1`` counterpart of :meth:`_integrate_members`, with the
        same contract: one outcome per member in order, a member error in the
        failing member's slot, and resource exhaustion and invariant violations
        propagated.  A request of ``K`` attempts costs one RK4 solve per chunk
        of :meth:`_mixed_chunks` instead of one per topology bucket.
        """

        if not covers or len(covers) != len(generators):
            raise DomainError(
                "geometry batch requires equal nonempty cover and generator tuples"
            )
        device, dtype = self._device_dtype()
        outcomes: list[ContinuousGeometry | Exception | None] = [None] * len(covers)
        pending: list[int] = []
        bridges: list[TorchCoverGeometry] = []
        for index, cover in enumerate(covers):
            try:
                bridge = self._bridge(self.model.layout(cover), device, dtype)
            except _MEMBER_ODE_ERRORS:
                # The reference path owns the report of a cover it cannot expand.
                outcomes[index] = self._integrate_members((cover,), (generators[index],))[0]
                continue
            pending.append(index)
            bridges.append(bridge)
        for chunk in self._mixed_chunks(tuple(bridges), dtype):
            members = tuple(pending[position] for position in chunk)
            solved = self._integrate_mixed_members(
                tuple(covers[index] for index in members),
                tuple(generators[index] for index in members),
            )
            for index, outcome in zip(members, solved, strict=True):
                outcomes[index] = outcome
        if any(item is None for item in outcomes):
            raise InvariantViolation("mixed geometry sampling lost a member outcome")
        return tuple(item for item in outcomes if item is not None)

    def _sample_in_eval_mode(
        self,
        request: GenerationRequest[Composition],
        device: torch.device,
    ) -> tuple[LearnedGenerationTrace, ...]:
        root = np.random.SeedSequence(request.seed)
        traces: list[LearnedGenerationTrace | None] = [None] * request.samples
        groups: dict[tuple[object, ...], list[_PreparedLearnedAttempt]] = {}
        for index, child in enumerate(root.spawn(request.samples)):
            seeds = child.generate_state(4, dtype=np.uint64)
            reported_seed = int(seeds[0])
            try:
                composition = self._composition(
                    request,
                    self._torch_generator(int(seeds[1]), device),
                    device,
                )
            except DomainError as exc:
                traces[index] = LearnedGenerationTrace(
                    index,
                    reported_seed,
                    failed_stage="composition",
                    error=f"{type(exc).__name__}: {exc}",
                )
                continue
            cover_sample = self._cover_sampler.sample(
                composition,
                generator=self._torch_generator(int(seeds[2]), device),
                device=device,
                primitive_atom_count=request.primitive_atom_count,
            )
            signature = _layout_batch_signature(self.model.layout(cover_sample.cover))
            groups.setdefault(signature, []).append(
                _PreparedLearnedAttempt(
                    index,
                    reported_seed,
                    composition,
                    cover_sample,
                    self._torch_generator(int(seeds[3]), device),
                )
            )

        # ``mixed-v1`` integrates every prepared attempt together; ``bucket-v1``
        # integrates one topology bucket at a time.
        mixed = self.geometry_sampler == "mixed-v1"
        pools = (
            [[item for group in groups.values() for item in group]]
            if mixed
            else list(groups.values())
        )
        integrate = self._integrate_members_mixed if mixed else self._integrate_members
        for prepared in pools:
            if not prepared:
                continue
            outcomes = integrate(
                tuple(item.cover_sample.cover for item in prepared),
                tuple(item.geometry_generator for item in prepared),
            )
            for item, outcome in zip(prepared, outcomes, strict=True):
                if isinstance(outcome, Exception):
                    traces[item.sample_index] = LearnedGenerationTrace(
                        item.sample_index,
                        item.reported_seed,
                        failed_stage="geometry/ode",
                        error=f"{type(outcome).__name__}: {outcome}",
                    )
                    continue
                geometry = outcome
                try:
                    decoded = self.model.layout(item.cover_sample.cover).decode(
                        geometry,
                        collision_tolerance=(
                            self.model.scientific_config.numerics.collision_atol_angstrom
                        ),
                        symmetry_tolerance=(
                            self.model.scientific_config.numerics.symmetry_atol
                        ),
                    )
                    physical_primitive = decoded.full_primitive_reduction(
                        cartesian_tolerance=(
                            self.model.scientific_config.numerics.symmetry_atol
                        )
                    )
                except GeneratedSampleError as exc:
                    traces[item.sample_index] = LearnedGenerationTrace(
                        item.sample_index,
                        item.reported_seed,
                        failed_stage="geometry/decode",
                        error=f"{type(exc).__name__}: {exc}",
                    )
                    continue
                traces[item.sample_index] = LearnedGenerationTrace(
                    item.sample_index,
                    item.reported_seed,
                    sample=LearnedSample(
                        request.mode,
                        item.composition,
                        item.cover_sample,
                        geometry,
                        decoded,
                        physical_primitive,
                        item.reported_seed,
                        self.checkpoint_id,
                    ),
                )
        if any(item is None for item in traces):
            raise InvariantViolation("learned batch sampling lost an attempt trace")
        return tuple(item for item in traces if item is not None)

    def require_samples(
        self, request: GenerationRequest[Composition]
    ) -> tuple[LearnedSample, ...]:
        traces = self.sample(request)
        failed = next((trace for trace in traces if not trace.succeeded), None)
        if failed is not None:
            raise InvariantViolation(
                f"learned attempt {failed.sample_index} failed at "
                f"{failed.failed_stage}: {failed.error}"
            )
        return tuple(trace.sample for trace in traces if trace.sample is not None)

    def geometry_cfm_objective(
        self,
        scaffold: CrystalScaffold,
        target_metric: np.ndarray,
        *,
        time: Tensor,
        sigma_z: float,
        sigma_eta: float,
        generator: torch.Generator | None = None,
        intrinsic_kernel: str = "fractional-v1",
        block_weights: tuple[float, float] = (1.0, 1.0),
        basis_augmentation: str = "none",
        basis_augmentation_probability: float = 0.5,
    ) -> GeometryCFMResult:
        """One canonically labelled, fixed-cover geometry CFM objective.

        This is a likelihood-free conditional regression objective.  The caller
        must preprocess each datum into its declared actual-symmetry cover; this
        method does not claim to marginalize all overlapping cover preimages.
        ``intrinsic_kernel`` selects the torus dequantization kernel
        (:class:`~crystalatlas.training_objectives.CFMDequantization`).
        """

        from .geometry import ContinuousCover

        _require_intrinsic_kernel(intrinsic_kernel)
        _require_basis_augmentation(basis_augmentation, basis_augmentation_probability)
        self._require_valid_target(scaffold, target_metric)
        cover = ContinuousCover.from_exact_scaffold(scaffold)
        layout = self.model.layout(cover)
        device, dtype = self._device_dtype()
        bridge = self._bridge(layout, device, dtype)
        z_data, eta_data, transform = self._endpoint_record(
            layout,
            bridge,
            scaffold,
            target_metric,
            sigma_eta=sigma_eta,
            intrinsic_kernel=intrinsic_kernel,
            basis_augmentation=basis_augmentation,
            basis_augmentation_probability=basis_augmentation_probability,
            generator=generator,
            device=device,
            dtype=dtype,
        )
        period = bridge.period_basis.to(dtype=dtype, device=device)
        # A uniformly sampled finite gauge lift acts on the complete sampled
        # path and tangent, not only the endpoint. The Haar base and atom-set
        # equivariant field then make inference invariant without factorial
        # enumeration of repeated orbit instances.
        residual_action, residual_shift = self._sample_residual_torus_action(
            layout, generator=generator, device=device, dtype=dtype
        )
        # Gauge actions, torus lifts, the flow path, and exact cover expansion
        # are crystallographic algebra.  Keep them in the model's master
        # FP32/FP64 dtype even when the trainer encloses this objective in AMP.
        with torch.autocast(device_type=device.type, enabled=False):
            base = sample_product_base(
                layout.lattice_dimension,
                period,
                dtype=dtype,
                device=device,
                generator=generator,
            )
            endpoint = sample_dequantized_endpoint(
                z_data,
                eta_data,
                period,
                sigma_z=sigma_z,
                sigma_eta=sigma_eta,
                generator=generator,
                eta_noise_transform=transform,
            )
            path = conditional_flow_path(
                base.z,
                base.eta,
                endpoint.z,
                endpoint.eta,
                time.to(dtype=dtype, device=device),
                period,
                torus_action=residual_action,
                torus_translation=residual_shift,
            )
            expanded = bridge(path.state.z, path.state.eta)
        composition = cover.composition
        context = self.model.geometry_field.encode_composition(
            torch.as_tensor(composition.support, dtype=torch.long, device=device),
            torch.as_tensor(composition.coefficients, dtype=dtype, device=device),
        )
        output = self.model.geometry_field(
            time.to(dtype=dtype, device=device),
            path.state.z,
            path.state.eta,
            context,
            atomic_numbers=expanded.atomic_numbers,
            atom_features=expanded.atom_features,
            fractional_coordinates=expanded.fractional_coordinates,
            lattice=expanded.lattice,
            bounded_lattice_coordinates=expanded.bounded_lattice_coordinates,
            jacobian_cartesian=expanded.jacobian_cartesian,
            geometry_is_certified=True,
            lattice_commutant_basis=expanded.lattice_commutant_basis,
        )
        with torch.autocast(device_type=device.type, enabled=False):
            prediction = FlowVelocity(
                output.lattice_velocity.to(dtype=dtype),
                output.intrinsic_velocity.to(dtype=dtype),
            )
            lattice_metric = bridge.lattice_pullback_metric(path.state.z).detach()
            loss = conditional_flow_matching_loss(
                prediction,
                path.target_velocity,
                current_lattice_metric=lattice_metric,
                current_coordinate_metric=output.coordinate_metric.to(dtype=dtype),
                atom_count=bridge.atom_count,
                return_terms=True,
                lattice_weight=block_weights[0],
                coordinate_weight=block_weights[1],
            )
        if not isinstance(loss, CFMLossTerms):  # pragma: no cover - static contract
            raise AssertionError("CFM loss did not return its requested terms")
        return GeometryCFMResult(loss, path, prediction)

    def geometry_cfm_objective_batch(
        self,
        scaffolds: tuple[CrystalScaffold, ...],
        target_metrics: tuple[np.ndarray, ...],
        *,
        time: Tensor,
        sigma_z: float,
        sigma_eta: float,
        generator: torch.Generator | None = None,
        geometry_field: Any | None = None,
        intrinsic_kernel: str = "fractional-v1",
        block_weights: tuple[float, float] = (1.0, 1.0),
        basis_augmentation: str = "none",
        basis_augmentation_probability: float = 0.5,
        coupling: str = "independent",
        coupling_candidates: int = 8,
    ) -> GeometryCFMBatchResult:
        """Evaluate a homogeneous geometry batch with one Transformer call.

        Exact decoder checks are run independently for every target before any
        neural computation.  Covers may differ in element identities, but must
        share the same crystallographic topology and tangent-space dimensions.
        Each sample therefore receives its own exact expansion bridge and
        composition condition; only the dense tensor shapes are shared.
        """

        if not scaffolds or len(scaffolds) != len(target_metrics):
            raise ValueError("scaffolds and target_metrics must be equal nonempty tuples")
        _require_intrinsic_kernel(intrinsic_kernel)
        _require_basis_augmentation(basis_augmentation, basis_augmentation_probability)
        _require_coupling(coupling, coupling_candidates)
        covers = []
        for scaffold, metric in zip(scaffolds, target_metrics, strict=True):
            self._require_valid_target(scaffold, metric)
            covers.append(ContinuousCover.from_exact_scaffold(scaffold))
        layouts = tuple(self.model.layout(cover) for cover in covers)
        layout = layouts[0]
        topology_signature = _layout_batch_signature(layout)
        for candidate in layouts[1:]:
            if _layout_batch_signature(candidate) != topology_signature:
                raise DomainError("packed geometry CFM requires one exact cover topology")
        device, dtype = self._device_dtype()
        bridges = tuple(self._bridge(candidate, device, dtype) for candidate in layouts)
        period = bridges[0].period_basis.to(dtype=dtype, device=device)
        encoded = [
            self._endpoint_record(
                candidate,
                bridge,
                scaffold,
                metric,
                sigma_eta=sigma_eta,
                intrinsic_kernel=intrinsic_kernel,
                basis_augmentation=basis_augmentation,
                basis_augmentation_probability=basis_augmentation_probability,
                generator=generator,
                device=device,
                dtype=dtype,
            )
            for candidate, bridge, scaffold, metric in zip(
                layouts, bridges, scaffolds, target_metrics, strict=True
            )
        ]
        z_data = torch.stack([item[0] for item in encoded])
        transform = (
            torch.stack([item[2] for item in encoded])  # type: ignore[misc]
            if intrinsic_kernel == "cartesian-v2"
            else None
        )
        eta_data = torch.stack([item[1] for item in encoded])
        batch = len(scaffolds)
        # Independent coupling: one residual action per record, in the legacy draw
        # order.  Gauge-aligned coupling: candidate actions per block of one cover
        # (Proposition 2.9d holds within a cover), drawn before the base.
        legacy_actions: list[tuple[Tensor, Tensor | None]] = []
        coupling_blocks: list[
            tuple[list[int], TorchCoverGeometry, Tensor, Tensor | None, bool]
        ] = []
        if coupling == "gauge-ot-v1":
            cover_positions: dict[ContinuousCover, list[int]] = {}
            for index, cover in enumerate(covers):
                cover_positions.setdefault(cover, []).append(index)
            for indices in cover_positions.values():
                coupling_blocks.append(
                    (
                        indices,
                        bridges[indices[0]],
                        *self._residual_action_candidates(
                            layouts[indices[0]],
                            len(indices),
                            coupling_candidates,
                            generator=generator,
                            device=device,
                        ),
                    )
                )
        else:
            legacy_actions = [
                self._sample_residual_torus_action(
                    candidate, generator=generator, device=device, dtype=dtype
                )
                for candidate in layouts
            ]
        coupling_stats: dict[str, float] | None = None
        with torch.autocast(device_type=device.type, enabled=False):
            base = sample_product_base(
                layout.lattice_dimension,
                period,
                batch_shape=(batch,),
                dtype=dtype,
                device=device,
                generator=generator,
            )
            endpoint = sample_dequantized_endpoint(
                z_data,
                eta_data,
                period,
                sigma_z=sigma_z,
                sigma_eta=sigma_eta,
                generator=generator,
                eta_noise_transform=transform,
            )
            time_value = time.to(dtype=dtype, device=device)
            if coupling == "gauge-ot-v1":
                base, torus_action, torus_translation, coupling_stats = (
                    self._gauge_aligned_batch(coupling_blocks, base, endpoint, period)
                )
            else:
                torus_action = torch.stack([action for action, _ in legacy_actions])
                torus_translation = (
                    None
                    if all(shift is None for _, shift in legacy_actions)
                    else torch.stack(
                        [
                            torch.zeros(eta_data.shape[-1], dtype=dtype, device=device)
                            if shift is None
                            else shift
                            for _, shift in legacy_actions
                        ]
                    )
                )
            path = conditional_flow_path(
                base.z,
                base.eta,
                endpoint.z,
                endpoint.eta,
                time_value,
                period,
                torus_action=torus_action,
                torus_translation=torus_translation,
            )
            if all(bridge is bridges[0] for bridge in bridges):
                # One cover: one batched expansion, equal to per-record up to rounding.
                expanded = bridges[0].forward_batch(path.state.z, path.state.eta)
                lattice_metric = bridges[0].lattice_pullback_metric_batch(path.state.z)
            else:
                records = [
                    bridge(path.state.z[index], path.state.eta[index])
                    for index, bridge in enumerate(bridges)
                ]
                expanded = TorchExpandedGeometry(
                    **{
                        name: (
                            records[0].lattice_commutant_basis
                            if name == "lattice_commutant_basis"
                            else torch.stack([getattr(item, name) for item in records])
                        )
                        for name in TorchExpandedGeometry.__dataclass_fields__
                    }
                )
                lattice_metric = torch.stack(
                    [
                        bridge.lattice_pullback_metric(path.state.z[index]).detach()
                        for index, bridge in enumerate(bridges)
                    ]
                )
        composition_numbers = torch.stack(
            [
                torch.as_tensor(cover.composition.support, dtype=torch.long, device=device)
                for cover in covers
            ]
        )
        composition_counts = torch.stack(
            [
                torch.as_tensor(cover.composition.coefficients, dtype=dtype, device=device)
                for cover in covers
            ]
        )
        field = self.model.geometry_field if geometry_field is None else geometry_field
        output = field(
            time_value,
            path.state.z,
            path.state.eta,
            None,
            atomic_numbers=expanded.atomic_numbers,
            atom_features=expanded.atom_features,
            fractional_coordinates=expanded.fractional_coordinates,
            lattice=expanded.lattice,
            bounded_lattice_coordinates=expanded.bounded_lattice_coordinates,
            jacobian_cartesian=expanded.jacobian_cartesian,
            composition_atomic_numbers=composition_numbers,
            composition_counts=composition_counts,
            lattice_commutant_basis=expanded.lattice_commutant_basis,
        )
        with torch.autocast(device_type=device.type, enabled=False):
            prediction = FlowVelocity(
                output.lattice_velocity.to(dtype=dtype),
                output.intrinsic_velocity.to(dtype=dtype),
            )
            loss = conditional_flow_matching_loss(
                prediction,
                path.target_velocity,
                current_lattice_metric=lattice_metric,
                current_coordinate_metric=output.coordinate_metric.to(dtype=dtype),
                atom_count=bridges[0].atom_count,
                return_terms=True,
                lattice_weight=block_weights[0],
                coordinate_weight=block_weights[1],
            )
        if not isinstance(loss, CFMLossTerms):  # pragma: no cover - static contract
            raise AssertionError("packed CFM loss did not return its requested terms")
        return GeometryCFMBatchResult(loss, path, prediction, coupling_stats)


    def geometry_cfm_objective_padded(
        self,
        scaffolds: tuple[CrystalScaffold, ...],
        target_metrics: tuple[np.ndarray, ...],
        *,
        time: Tensor,
        sigma_z: float,
        sigma_eta: float,
        generator: torch.Generator | None = None,
        geometry_field: Any | None = None,
        intrinsic_kernel: str = "fractional-v1",
        block_weights: tuple[float, float] = (1.0, 1.0),
        basis_augmentation: str = "none",
        basis_augmentation_probability: float = 0.5,
    ) -> GeometryCFMPaddedResult:
        """Geometry CFM for crystals of different topologies in one dense call.

        Every crystal is decoded, audited, and sampled exactly as by
        :meth:`geometry_cfm_objective`, in the same random-draw order, with its
        own chart, period basis, and residual action.  Only the field call is
        shared, through :meth:`IntrinsicGeometryField.forward_padded`.  Padded
        velocity slots are exactly zero, and the padded metrics are the identity,
        so padding contributes nothing to any per-sample loss (review M5).
        """

        from .geometry import ContinuousCover

        if not scaffolds or len(scaffolds) != len(target_metrics):
            raise ValueError("scaffolds and target_metrics must be equal nonempty tuples")
        _require_intrinsic_kernel(intrinsic_kernel)
        _require_basis_augmentation(basis_augmentation, basis_augmentation_probability)
        batch = len(scaffolds)
        device, dtype = self._device_dtype()
        times = torch.as_tensor(time, dtype=dtype, device=device)
        if times.shape != (batch,):
            raise ValueError("padded geometry CFM needs one time per crystal")
        covers, layouts, paths, bridges = [], [], [], []
        for index, (scaffold, metric) in enumerate(zip(scaffolds, target_metrics, strict=True)):
            self._require_valid_target(scaffold, metric)
            cover = ContinuousCover.from_exact_scaffold(scaffold)
            layout = self.model.layout(cover)
            bridge = self._bridge(layout, device, dtype)
            z_data, eta_data, transform = self._endpoint_record(
                layout,
                bridge,
                scaffold,
                metric,
                sigma_eta=sigma_eta,
                intrinsic_kernel=intrinsic_kernel,
                basis_augmentation=basis_augmentation,
                basis_augmentation_probability=basis_augmentation_probability,
                generator=generator,
                device=device,
                dtype=dtype,
            )
            period = bridge.period_basis.to(dtype=dtype, device=device)
            action, shift = self._sample_residual_torus_action(
                layout, generator=generator, device=device, dtype=dtype
            )
            with torch.autocast(device_type=device.type, enabled=False):
                base = sample_product_base(
                    layout.lattice_dimension,
                    period,
                    dtype=dtype,
                    device=device,
                    generator=generator,
                )
                endpoint = sample_dequantized_endpoint(
                    z_data,
                    eta_data,
                    period,
                    sigma_z=sigma_z,
                    sigma_eta=sigma_eta,
                    generator=generator,
                    eta_noise_transform=transform,
                )
                path = conditional_flow_path(
                    base.z,
                    base.eta,
                    endpoint.z,
                    endpoint.eta,
                    times[index],
                    period,
                    torus_action=action,
                    torus_translation=shift,
                )
            covers.append(cover)
            layouts.append(layout)
            paths.append(path)
            bridges.append(bridge)

        # One batched expansion for every cover in the batch (review 2026-10-29,
        # H2) instead of a per-record loop over orbits and coset operations.
        with torch.autocast(device_type=device.type, enabled=False):
            expanded = expand_padded(
                bridges, [path.state.z for path in paths], [path.state.eta for path in paths]
            )
            lattice_metrics = lattice_pullback_metrics(
                bridges, [path.state.z for path in paths]
            )
        lattice_dims = [layout.lattice_dimension for layout in layouts]
        intrinsic_dims = [layout.intrinsic_dimension for layout in layouts]
        atom_counts = [bridge.atom_count for bridge in bridges]
        species = [len(cover.composition.support) for cover in covers]
        width_z, width_eta = max(lattice_dims), max(intrinsic_dims)
        width_species = max(species)
        if expanded.jacobian_cartesian.shape[-1] != width_eta or (
            expanded.atom_mask.sum(dim=1).tolist() != atom_counts
        ):
            raise InvariantViolation("padded expansion disagrees with the layouts")

        def zeros(*shape: int, kind: torch.dtype = dtype) -> Tensor:
            return torch.zeros(shape, dtype=kind, device=device)

        z = zeros(batch, width_z)
        eta = zeros(batch, width_eta)
        bounded = zeros(batch, width_z)
        commutant = zeros(batch, 6, width_z)
        composition_numbers = zeros(batch, width_species, kind=torch.long)
        composition_counts = zeros(batch, width_species)
        target_z = zeros(batch, width_z)
        target_eta = zeros(batch, width_eta)
        lattice_metric = torch.eye(width_z, dtype=dtype, device=device).repeat(batch, 1, 1)
        for b in range(batch):
            d, e, k = lattice_dims[b], intrinsic_dims[b], species[b]
            z[b, :d] = paths[b].state.z
            eta[b, :e] = paths[b].state.eta
            target_z[b, :d] = paths[b].target_velocity.z
            target_eta[b, :e] = paths[b].target_velocity.eta
            bounded[b, :d] = expanded.bounded_lattice_coordinates[b]
            commutant[b, :, :d] = expanded.lattice_commutant_basis[b].to(dtype=dtype)
            composition_numbers[b, :k] = torch.as_tensor(
                covers[b].composition.support, dtype=torch.long, device=device
            )
            composition_counts[b, :k] = torch.as_tensor(
                covers[b].composition.coefficients, dtype=dtype, device=device
            )
            lattice_metric[b, :d, :d] = lattice_metrics[b]
        lattice_dimensions = torch.tensor(lattice_dims, device=device)
        intrinsic_dimensions = torch.tensor(intrinsic_dims, device=device)
        field = self.model.geometry_field if geometry_field is None else geometry_field
        output = field.forward_padded(
            times,
            z,
            eta,
            None,
            atomic_numbers=expanded.atomic_numbers,
            atom_mask=expanded.atom_mask,
            atom_features=expanded.atom_features,
            fractional_coordinates=expanded.fractional_coordinates,
            lattice=expanded.lattice,
            bounded_lattice_coordinates=bounded,
            jacobian_cartesian=expanded.jacobian_cartesian,
            lattice_dimensions=lattice_dimensions,
            intrinsic_dimensions=intrinsic_dimensions,
            composition_atomic_numbers=composition_numbers,
            composition_counts=composition_counts,
            lattice_commutant_basis=(
                commutant if field.config.lattice_head == "commutant-v2" else None
            ),
        )
        with torch.autocast(device_type=device.type, enabled=False):
            prediction = FlowVelocity(
                output.lattice_velocity.to(dtype=dtype),
                output.intrinsic_velocity.to(dtype=dtype),
            )
            counts = torch.tensor(atom_counts, dtype=dtype, device=device)
            loss = conditional_flow_matching_loss(
                prediction,
                FlowVelocity(target_z, target_eta),
                current_lattice_metric=lattice_metric,
                current_coordinate_metric=output.coordinate_metric.to(dtype=dtype),
                atom_count=counts,
                return_terms=True,
                lattice_weight=block_weights[0],
                coordinate_weight=block_weights[1],
            )
        if not isinstance(loss, CFMLossTerms):  # pragma: no cover - static contract
            raise AssertionError("padded CFM loss did not return its requested terms")
        return GeometryCFMPaddedResult(
            loss=loss,
            paths=tuple(paths),
            prediction=prediction,
            lattice_dimensions=lattice_dimensions,
            intrinsic_dimensions=intrinsic_dimensions,
            atom_counts=counts,
        )


__all__ = [
    "CrystalAtlasModel",
    "GeometryCFMBatchResult",
    "GeometryCFMPaddedResult",
    "GeometryCFMResult",
    "LearnedCrystalGenerator",
    "LearnedGenerationTrace",
    "LearnedSample",
    "MapSelection",
    "SampleScore",
]
