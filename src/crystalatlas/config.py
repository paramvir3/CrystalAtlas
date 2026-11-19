"""Immutable scientific and numerical configuration.

Changing these values changes the declared probability support or numerical
contract.  Configurations are therefore validated once and serialized with
every dataset, checkpoint, and generated batch.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field, fields
from hashlib import sha256
from typing import Any

from .cover_legality import AtomBound
from .errors import DomainError


def _unique_sorted_atomic_numbers(values: tuple[int, ...], name: str) -> tuple[int, ...]:
    if not isinstance(values, tuple) or not values:
        raise DomainError(f"{name} must be nonempty")
    if any(
        isinstance(z, bool) or not isinstance(z, int) or z < 1 or z > 118
        for z in values
    ):
        raise DomainError(f"{name} contains an invalid atomic number")
    if len(set(values)) != len(values):
        raise DomainError(f"{name} contains duplicate atomic numbers")
    return tuple(sorted(values))


@dataclass(frozen=True, slots=True)
class DomainConfig:
    """Finite support from section 0.2 of the specification."""

    element_vocabulary: tuple[int, ...] = tuple(range(1, 119))
    dng_element_vocabulary: tuple[int, ...] = (
        1,
        3,
        4,
        5,
        6,
        7,
        8,
        9,
        11,
        12,
        13,
        14,
        15,
        16,
        17,
        19,
        20,
        21,
        22,
        23,
        24,
        25,
        26,
        27,
        28,
        29,
        30,
        31,
        32,
        33,
        34,
        35,
        37,
        38,
        39,
        40,
        41,
        42,
        43,
        44,
        45,
        46,
        47,
        48,
        49,
        50,
        51,
        52,
        53,
        55,
        56,
        57,
        58,
        59,
        60,
        62,
        63,
        64,
        65,
        66,
        67,
        68,
        69,
        70,
        71,
        72,
        73,
        74,
        75,
        76,
        77,
        78,
        79,
        80,
        81,
        82,
        83,
    )
    max_distinct_elements: int = 6
    max_raw_atoms: int = 40
    max_formula_units: int = 16
    metric_radius: float = 2.0
    reference_atomic_volume_angstrom3: float = 20.0
    minimum_lattice_singular_value_angstrom: float = 0.75
    reference_length_angstrom: float = 1.0
    interaction_cutoff_angstrom: float = 7.0
    lattice_chart: str = "loewner-v2"
    """Lattice support: ``loewner-v2`` (the default since 2026-10-20, review B2) or
    ``ball-v1`` (legacy; it must now be requested explicitly, and keeps its
    historical identity payload).

    ``ball-v1`` is ``||log(G0^-1/2 G G0^-1/2)||_F < metric_radius``.
    ``loewner-v2`` is the open matrix interval ``l(N) G0 < G < u G0``.  The
    lower bound ``l(N) = s_min^2 / lambda_min(G0(N))`` certifies
    ``sigma_min(L) >= minimum_lattice_singular_value_angstrom`` exactly.
    The upper bound is ``u = exp(loewner_upper_log_eigenvalue)``.
    """
    loewner_upper_log_eigenvalue: float = 4.5
    """``ln u`` for ``loewner-v2``.  Fitted on one training sample by the rule
    ``ceil(2 (max ln lambda_max(W) + 0.25)) / 2``.  It covered 496 of 496
    held-out records (`artifacts/b2_chart_support_2026-10-20/`)."""
    loewner_control_log_volume_window: tuple[float, float] = (-1.5, 2.0)
    """``loewner-v2`` only: Hencky volumetric window of the untrained uniform laws.

    The window is ``lo < ln(V/V0) = tr(log W)/2 < hi``.  It applies to the m3
    control and the reference law, not to the learned support.  The uniform
    law over the Loewner box alone puts 20-72% of its mass outside the volumes
    of real crystals (up to V/V0 = e^6.75), which makes a strawman baseline.
    AIRSS likewise draws random cells around a "natural volume" proportional
    to the atom count (Pickard and Needs 2011, sec. 3.1-3.2).  The window was
    fitted on one sample by the rule of ``loewner_upper_log_eigenvalue`` on
    each side, and covered 496 of 496 held-out records
    (`artifacts/b2_chart_support_2026-10-20/volumetric_window_validation.json`).
    """
    max_primitive_atoms: int | None = None
    """Primitive-cell atom bound (MP-20's own definition; review B2).

    ``None`` keeps the legacy conventional bound ``nu |c| <= max_raw_atoms``,
    whose primitive capacity is only ``max_raw_atoms / I_G``.  When set, a cover
    is admitted iff ``nu |c| <= max_primitive_atoms * I_G``; ``max_raw_atoms``
    then remains a conventional resource cap and must be at least
    ``4 * max_primitive_atoms`` (the largest centring index is 4).
    """

    def identity_payload(self) -> dict[str, Any]:
        """Fingerprint payload; the legacy domain keeps its historical payload."""

        payload = asdict(self)
        if payload["max_primitive_atoms"] is None:
            del payload["max_primitive_atoms"]
        if payload["lattice_chart"] == "ball-v1":
            del payload["lattice_chart"]
            del payload["loewner_upper_log_eigenvalue"]
            del payload["loewner_control_log_volume_window"]
        return payload

    def layout_options(self) -> dict[str, Any]:
        """Keyword arguments that bind an ``IntrinsicCoverLayout`` to this domain."""

        return {
            "metric_bound": self.metric_radius,
            "reference_atomic_volume": self.reference_atomic_volume_angstrom3,
            "minimum_lattice_singular_value": self.minimum_lattice_singular_value_angstrom,
            "lattice_chart": self.lattice_chart,
            "loewner_upper_log_eigenvalue": self.loewner_upper_log_eigenvalue,
            "loewner_control_log_volume_window": self.loewner_control_log_volume_window,
        }

    @property
    def composition_atom_bound(self) -> int:
        """Bound on reduced-formula atoms: a primitive cell holds a whole formula."""

        return self.max_raw_atoms if self.max_primitive_atoms is None else (
            self.max_primitive_atoms
        )

    @property
    def atom_bound(self) -> AtomBound:
        return AtomBound(conventional=self.max_raw_atoms, primitive=self.max_primitive_atoms)

    def __post_init__(self) -> None:
        all_elements = _unique_sorted_atomic_numbers(
            self.element_vocabulary, "element_vocabulary"
        )
        dng_elements = _unique_sorted_atomic_numbers(
            self.dng_element_vocabulary, "dng_element_vocabulary"
        )
        if not set(dng_elements).issubset(all_elements):
            raise DomainError("DNG elements must be a subset of the package vocabulary")
        object.__setattr__(self, "element_vocabulary", all_elements)
        object.__setattr__(self, "dng_element_vocabulary", dng_elements)
        for name in (
            "max_distinct_elements",
            "max_raw_atoms",
            "max_formula_units",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise DomainError(f"{name} must be a positive integer")
        if not 1 <= self.max_distinct_elements <= len(dng_elements):
            raise DomainError("max_distinct_elements is inconsistent with DNG vocabulary")
        if self.max_raw_atoms < self.max_distinct_elements:
            raise DomainError("max_raw_atoms must admit the all-one composition witness")
        if self.max_primitive_atoms is not None:
            value = self.max_primitive_atoms
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise DomainError("max_primitive_atoms must be None or a positive integer")
            if value < self.max_distinct_elements:
                raise DomainError("max_primitive_atoms must admit the all-one composition")
            if self.max_raw_atoms < 4 * value:
                raise DomainError(
                    "max_raw_atoms is the conventional resource cap and must be at least "
                    "4 * max_primitive_atoms"
                )
        if self.lattice_chart not in {"ball-v1", "loewner-v2"}:
            raise DomainError("lattice_chart must be ball-v1 or loewner-v2")
        upper = self.loewner_upper_log_eigenvalue
        if (
            isinstance(upper, bool)
            or not isinstance(upper, (int, float))
            or not math.isfinite(float(upper))
            or float(upper) <= 0.0
        ):
            raise DomainError("loewner_upper_log_eigenvalue must be finite and positive")
        window = self.loewner_control_log_volume_window
        if (
            not isinstance(window, (tuple, list))
            or len(window) != 2
            or any(
                isinstance(value, bool) or not isinstance(value, (int, float))
                for value in window
            )
            or not all(math.isfinite(float(value)) for value in window)
            or not float(window[0]) < 0.0 < float(window[1])
        ):
            raise DomainError(
                "loewner_control_log_volume_window must be two finite numbers lo < 0 < hi"
            )
        object.__setattr__(
            self,
            "loewner_control_log_volume_window",
            (float(window[0]), float(window[1])),
        )
        for name in (
            "metric_radius",
            "reference_atomic_volume_angstrom3",
            "minimum_lattice_singular_value_angstrom",
            "reference_length_angstrom",
            "interaction_cutoff_angstrom",
        ):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                or float(value) <= 0.0
            ):
                raise DomainError(f"{name} must be finite and positive")
        # A parameter of the inactive chart has no effect (and under ball-v1 is
        # not fingerprinted), so a non-default value is refused, not ignored.
        defaults = {item.name: item.default for item in fields(DomainConfig)}
        inactive = (
            ("metric_radius",)
            if self.lattice_chart == "loewner-v2"
            else ("loewner_upper_log_eigenvalue", "loewner_control_log_volume_window")
        )
        for name in inactive:
            if getattr(self, name) != defaults[name]:
                raise DomainError(
                    f"{name} has no effect under lattice_chart={self.lattice_chart!r}; "
                    "leave it at its default or select the chart it belongs to"
                )


@dataclass(frozen=True, slots=True)
class NumericsConfig:
    """Numerical tolerances; none of these replaces an exact oracle."""

    symmetry_atol: float = 1.0e-10
    spd_eigenvalue_floor: float = 1.0e-12
    roundtrip_atol: float = 1.0e-9
    collision_atol_angstrom: float = 1.0e-8
    ode_steps: int = 64
    ode_terminal_time: float = 1.0
    """Integrate to this time, then apply ``x_t + (1 - t) v`` (review M4); 1 = plain RK4."""
    max_pair_image_candidates: int = 32_768
    max_periodic_edges: int = 500_000
    max_dense_pair_interactions: int = 500_000
    max_reciprocal_modes: int = 4_096
    max_geometry_readout_coefficient_channels: int = 4_096
    max_geometry_workspace_bytes: int = 2_000_000_000
    maximum_tangent_condition_number_float32: float = 1.0e4
    maximum_tangent_condition_number_float64: float = 1.0e6

    def __post_init__(self) -> None:
        for name in (
            "symmetry_atol",
            "spd_eigenvalue_floor",
            "roundtrip_atol",
            "collision_atol_angstrom",
            "maximum_tangent_condition_number_float32",
            "maximum_tangent_condition_number_float64",
        ):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                or float(value) <= 0.0
            ):
                raise DomainError(f"{name} must be finite and positive")
        if self.maximum_tangent_condition_number_float32 < 1.0 or (
            self.maximum_tangent_condition_number_float64 < 1.0
        ):
            raise DomainError("tangent condition-number bounds must be at least one")
        for name in (
            "ode_steps",
            "max_pair_image_candidates",
            "max_periodic_edges",
            "max_dense_pair_interactions",
            "max_reciprocal_modes",
            "max_geometry_readout_coefficient_channels",
            "max_geometry_workspace_bytes",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise DomainError(f"{name} must be a positive integer")
        terminal = self.ode_terminal_time
        if (
            isinstance(terminal, bool)
            or not isinstance(terminal, (int, float))
            or not math.isfinite(float(terminal))
            or not 0.0 < float(terminal) <= 1.0
        ):
            raise DomainError("ode_terminal_time must lie in (0, 1]")


@dataclass(frozen=True, slots=True)
class ModelConfig:
    """Shared Transformer and flow dimensions.

    ``invariant_global`` means invariant learned scalar channels plus a
    deterministic proper-rotation-covariant output basis.  It must not be
    confused with a raw-coordinate non-equivariant network.
    """

    composition_width: int = 256
    composition_layers: int = 6
    selector_width: int = 256
    selector_layers: int = 6
    geometry_backbone: str = "invariant_global"
    geometry_scalar_width: int = 256
    geometry_pair_width: int = 96
    geometry_vector_width: int = 32
    geometry_layers: int = 10
    attention_heads: int = 8
    radial_features: int = 32
    geometry_invariant_feedforward_multiplier: int = 4
    geometry_reciprocal_mode_radius: int = 2
    geometry_heat_scales: tuple[float, ...] = (0.04, 0.16, 0.64)
    geometry_direct_harmonics: int = 2
    geometry_include_lattice_frame_carriers: bool = True
    geometry_normalization_epsilon: float = 1.0e-4
    geometry_minimum_dimensionless_lattice_singular_value: float = 1.0e-6
    geometry_maximum_dimensionless_lattice_singular_value: float = 1.0e6
    geometry_maximum_lattice_condition_number: float = 1.0e6
    geometry_readout_schema: str = "fourier-pair-lattice-frame-so3-v2"
    dropout: float = 0.0
    geometry_lattice_head: str = "commutant-v2"
    """``commutant-v2`` (the default since 2026-10-29) or ``truncated-v1`` (legacy).

    ``commutant-v2`` reads and writes the lattice state in canonical Sym(3)
    coordinates through the chart's orthonormal commutant basis ``C`` (review
    M3), so one shared head means the same thing in every crystal family.
    ``truncated-v1`` indexes the chart's own basis slots, whose meaning changes
    between families: slot 0 is the c-axis strain for tetragonal settings and
    the a-axis strain for hexagonal ones (review 2026-10-29, C3).  It keeps its
    historical identity payload and must now be requested explicitly.
    """
    geometry_expansion_cell: str = "conventional-v1"
    """``conventional-v1`` (legacy) or ``primitive-v2`` (review 2026-10-29, M3).

    The dense field is ``O(N^2)`` in expanded atoms.  ``primitive-v2`` expands
    every orbit to the primitive cell (one coset representative per class
    modulo the centring translations, lattice ``P L``), the same geometry with
    ``N / I_c`` atoms, so an F-centred crystal costs 1/16 of the pairs.  The
    chart, the decoder and the Cartesian kernel keep the conventional cell;
    the per-atom loss normalization makes the coordinate loss identical.
    Dense ``invariant_global`` backbone only.
    """
    geometry_radial_gaussian_widths: tuple[float, ...] = ()
    """Periodic Gaussian radial pair-scalar widths in A (review M2); () is legacy."""
    geometry_pair_carrier_smoothing_length: float | None = None
    """Coincidence smoothing length of the dense backbone's pair carriers in units
    of ``DomainConfig.reference_length_angstrom`` (angstrom in the registered
    domains, whose reference length is 1 A), research 2026-11-12.  ``None`` is the
    legacy carrier, a unit vector down to 0.0016 reference lengths.  The switch
    also acts near half-lattice separations (see
    ``InvariantPeriodicTransformerConfig.pair_carrier_smoothing_length``).  Dense
    ``invariant_global`` backbone only."""
    selector_symmetry_descriptors: bool = False
    """Operation-derived group and site-symmetry selector inputs (review m1)."""
    selector_row_geometry_features: bool = False
    """Exact-chart Wyckoff-row features for the selector (review 2026-10-29, M1).

    The legacy row features cannot tell Pm-3m 12h ``(x, 1/2, 0)`` from 12i
    ``(0, y, y)``; with this flag each row carries the exact chart of its
    normalizer-canonical representative and its canonical index
    (:func:`crystalatlas.symmetry_descriptors.row_geometry_descriptor`).
    """
    geometry_local_vector_normalization: str = "rms-v1"
    """``equivariant_local`` only: ``rms-v1`` (legacy) or ``soft-v2`` (C1 at the cutoff)."""
    geometry_residual_gauge: str = "relabel-v1"
    """Residual torus gauge sampled by the geometry CFM objective.

    ``relabel-v1`` (legacy) samples identical-orbit relabellings and P-1 sign
    flips only.  ``normalizer-v2`` samples the cover's whole certified residual
    group (:mod:`crystalatlas.residual_gauge`): per-orbit in-group gauges,
    relabellings, and proper Euclidean-normalizer automorphisms.  A physically
    invariant field can then represent the target (learned-F10 finding).
    ``normalizer-v3`` adds the improper normalizer elements, which map a crystal
    onto its enantiomorph, so the target is racemic within each cover (owner
    decision 2026-10-25).  An equivariant field stagnates at achiral
    configurations (fixed-point subspaces are flow-invariant), so it cannot
    move probability between enantiomorphic sectors.  Only the racemic target is
    then representable.  It equals normalizer-v2 for every group with an improper
    operation, and for the 22 enantiomorphic types.
    """
    element_embedding: str = "learned-v1"
    """``learned-v1`` (legacy) or ``periodic-v1`` (research 2026-11-10).

    ``learned-v1`` gives every atomic number a free vector in the selector's
    two composition encoders and in the geometry field's species and
    composition embeddings, so an element is trained only by the records that
    contain it.  ``periodic-v1`` replaces those tables by
    ``e(Z) = P[period] + G[column] + B[block]`` with the exact long-form
    coordinates of :func:`crystalatlas.elements.periodic_coordinates`
    (:class:`crystalatlas.nn.composition.PeriodicElementEmbedding`).  The
    composition model's token embedding is not changed.
    """

    def __post_init__(self) -> None:
        integer_fields = (
            "composition_width",
            "composition_layers",
            "selector_width",
            "selector_layers",
            "geometry_scalar_width",
            "geometry_pair_width",
            "geometry_vector_width",
            "geometry_layers",
            "attention_heads",
            "radial_features",
            "geometry_invariant_feedforward_multiplier",
            "geometry_direct_harmonics",
        )
        if any(
            isinstance(getattr(self, name), bool)
            or not isinstance(getattr(self, name), int)
            or getattr(self, name) < 1
            for name in integer_fields
        ):
            raise DomainError("all model dimensions must be positive integers")
        if (
            isinstance(self.geometry_reciprocal_mode_radius, bool)
            or not isinstance(self.geometry_reciprocal_mode_radius, int)
            or self.geometry_reciprocal_mode_radius < 0
        ):
            raise DomainError(
                "geometry_reciprocal_mode_radius must be a nonnegative integer"
            )
        attention_widths: tuple[str, ...] = (
            "composition_width",
            "selector_width",
            "geometry_scalar_width",
        )
        if self.geometry_backbone == "equivariant_local":
            attention_widths += ("geometry_vector_width",)
        incompatible = tuple(
            name
            for name in attention_widths
            if getattr(self, name) % self.attention_heads != 0
        )
        if incompatible:
            joined = ", ".join(incompatible)
            raise DomainError(
                f"attention widths must be divisible by attention_heads: {joined}"
            )
        if self.geometry_backbone not in {"invariant_global", "equivariant_local"}:
            raise DomainError(
                "geometry_backbone must be 'invariant_global' or 'equivariant_local'"
            )
        if (
            self.geometry_backbone == "invariant_global"
            and self.geometry_reciprocal_mode_radius < 1
        ):
            raise DomainError(
                "invariant_global requires a positive reciprocal-mode radius"
            )
        if not isinstance(self.geometry_heat_scales, tuple) or not (
            self.geometry_heat_scales
        ) or any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or float(value) <= 0.0
            for value in self.geometry_heat_scales
        ):
            raise DomainError(
                "geometry_heat_scales must be a nonempty tuple of positive values"
            )
        if type(self.geometry_include_lattice_frame_carriers) is not bool:
            raise DomainError(
                "geometry_include_lattice_frame_carriers must be boolean"
            )
        for name in (
            "geometry_normalization_epsilon",
            "geometry_minimum_dimensionless_lattice_singular_value",
            "geometry_maximum_dimensionless_lattice_singular_value",
            "geometry_maximum_lattice_condition_number",
        ):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                or float(value) <= 0.0
            ):
                raise DomainError(f"{name} must be finite and positive")
        if (
            self.geometry_maximum_dimensionless_lattice_singular_value
            <= self.geometry_minimum_dimensionless_lattice_singular_value
        ):
            raise DomainError(
                "geometry maximum lattice singular value must exceed its minimum"
            )
        if self.geometry_maximum_lattice_condition_number < 1.0:
            raise DomainError(
                "geometry_maximum_lattice_condition_number must be at least one"
            )
        if self.geometry_readout_schema != "fourier-pair-lattice-frame-so3-v2":
            raise DomainError("unsupported geometry_readout_schema")
        if not isinstance(self.selector_row_geometry_features, bool):
            raise DomainError("selector_row_geometry_features must be boolean")
        if self.geometry_lattice_head not in {"truncated-v1", "commutant-v2"}:
            raise DomainError("geometry_lattice_head must be truncated-v1 or commutant-v2")
        if self.geometry_expansion_cell not in {"conventional-v1", "primitive-v2"}:
            raise DomainError("geometry_expansion_cell must be conventional-v1 or primitive-v2")
        if (
            self.geometry_expansion_cell == "primitive-v2"
            and self.geometry_backbone != "invariant_global"
        ):
            raise DomainError("primitive-v2 expansion requires the invariant_global backbone")
        if self.geometry_residual_gauge not in {"relabel-v1", "normalizer-v2", "normalizer-v3"}:
            raise DomainError(
                "geometry_residual_gauge must be relabel-v1, normalizer-v2 or normalizer-v3"
            )
        if self.element_embedding not in {"learned-v1", "periodic-v1"}:
            raise DomainError("element_embedding must be learned-v1 or periodic-v1")
        if self.geometry_local_vector_normalization not in {"rms-v1", "soft-v2"}:
            raise DomainError(
                "geometry_local_vector_normalization must be rms-v1 or soft-v2"
            )
        widths = tuple(self.geometry_radial_gaussian_widths)
        if any(
            isinstance(width, bool)
            or not isinstance(width, (int, float))
            or not math.isfinite(float(width))
            or float(width) <= 0.0
            for width in widths
        ):
            raise DomainError("geometry_radial_gaussian_widths must be finite and positive")
        object.__setattr__(self, "geometry_radial_gaussian_widths", widths)
        smoothing = self.geometry_pair_carrier_smoothing_length
        if smoothing is not None:
            if (
                isinstance(smoothing, bool)
                or not isinstance(smoothing, (int, float))
                or not math.isfinite(float(smoothing))
                or float(smoothing) <= 0.0
                or (2.0 * math.pi * float(smoothing)) ** 2
                <= float(self.geometry_normalization_epsilon)
            ):
                raise DomainError(
                    "geometry_pair_carrier_smoothing_length must be None or a finite "
                    "positive length with (2 pi s)^2 above geometry_normalization_epsilon"
                )
            if self.geometry_backbone != "invariant_global":
                raise DomainError(
                    "geometry_pair_carrier_smoothing_length requires the "
                    "invariant_global backbone"
                )
        if (
            isinstance(self.dropout, bool)
            or not isinstance(self.dropout, (int, float))
            or not math.isfinite(float(self.dropout))
            or not 0.0 <= float(self.dropout) < 1.0
        ):
            raise DomainError("dropout must be finite and lie in [0, 1)")


@dataclass(frozen=True, slots=True)
class CrystalAtlasConfig:
    """Top-level immutable configuration."""

    domain: DomainConfig = field(default_factory=DomainConfig)
    numerics: NumericsConfig = field(default_factory=NumericsConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    coordinate_convention: str = "fractional-row-v1"
    chirality_quotient: str = "SO(3)"

    def __post_init__(self) -> None:
        if not isinstance(self.domain, DomainConfig):
            raise DomainError("domain must be a DomainConfig")
        if not isinstance(self.numerics, NumericsConfig):
            raise DomainError("numerics must be a NumericsConfig")
        if not isinstance(self.model, ModelConfig):
            raise DomainError("model must be a ModelConfig")
        if self.coordinate_convention != "fractional-row-v1":
            raise DomainError("only the normative fractional-row convention is supported")
        if self.chirality_quotient != "SO(3)":
            raise DomainError("version 1 keeps enantiomers distinct and requires SO(3)")

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible representation."""

        payload = asdict(self)
        payload["domain"] = self.domain.identity_payload()
        # Legacy defaults keep the historical configuration fingerprint.
        if payload["model"]["geometry_lattice_head"] == "truncated-v1":
            del payload["model"]["geometry_lattice_head"]
        if payload["model"]["geometry_expansion_cell"] == "conventional-v1":
            del payload["model"]["geometry_expansion_cell"]
        if payload["numerics"]["ode_terminal_time"] == 1.0:
            del payload["numerics"]["ode_terminal_time"]
        if not payload["model"]["geometry_radial_gaussian_widths"]:
            del payload["model"]["geometry_radial_gaussian_widths"]
        if payload["model"]["geometry_pair_carrier_smoothing_length"] is None:
            del payload["model"]["geometry_pair_carrier_smoothing_length"]
        if not payload["model"]["selector_symmetry_descriptors"]:
            del payload["model"]["selector_symmetry_descriptors"]
        if not payload["model"]["selector_row_geometry_features"]:
            del payload["model"]["selector_row_geometry_features"]
        if payload["model"]["geometry_local_vector_normalization"] == "rms-v1":
            del payload["model"]["geometry_local_vector_normalization"]
        if payload["model"]["geometry_residual_gauge"] == "relabel-v1":
            del payload["model"]["geometry_residual_gauge"]
        if payload["model"]["element_embedding"] == "learned-v1":
            del payload["model"]["element_embedding"]
        return payload

    @property
    def fingerprint(self) -> str:
        """Stable identity used in datasets, checkpoints, and generated batches."""

        payload = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))
        return sha256(payload.encode("utf-8")).hexdigest()
