"""Dense invariant periodic attention with an equivariant vector readout.

This module deliberately contains no neighbour graph, cutoff, sparse indexing,
or segmented aggregation.  Every atom attends to every atom in its own
homogeneous batch member.  Geometry is represented by exact characters of the
fractional three-torus.  For integer modes ``n`` in the complete symmetric
cube ``[-K,K]^3``, the pair phase and reciprocal vector are

``phi_ij,n = 2 pi n . (f_j - f_i)``,
``k_n = 2 pi n L^{-T}``.

The metric-dependent heat channels

``H_ij(tau) = sum_n exp(-tau l0^2 |k_n|^2) cos(phi_ij,n) / sum_n weight_n``

are smooth on nonsingular lattices and exactly periodic in fractional space.
Both an attention bias and a pair-dependent value are learned from these
features, so geometry is not lost when every input token is identical.

All learned hidden channels are SO(3)-invariant scalars.  Cartesian
vectors are produced only by a deterministic readout

``v_i,c = (1/N) sum_j,b a_ij,cb B_ij,b``

where the coefficients ``a`` are invariant and ``B`` contains smooth periodic
direct moments, reciprocal heat-gradient moments, their cross products, and
three smoothly normalized rows of the pinned canonical lattice.  The last
carriers preserve directional reachability when all atoms collide but scalar
tokens differ; identical colliding tokens still reduce to a common
translation.  Polar and axial carriers are mixed by scalar coefficients, so
the construction has a guaranteed SO(3) action but deliberately has no single
O(3) parity assignment.  Row-vector convention is used: ``L -> L Q`` implies
``v -> v Q`` for every ``Q`` in SO(3).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from functools import partial
from numbers import Real

from .._optional import require_torch
from .host_sync import DeviceChecks

torch = require_torch()
Tensor = torch.Tensor
nn = torch.nn


def _smooth_unit_vectors(vectors: Tensor, epsilon: float) -> Tensor:
    """Normalize covariants without a singularity at a collision."""

    return vectors / torch.sqrt(
        vectors.square().sum(dim=-1, keepdim=True) + epsilon
    )


def _reciprocal_mode_cube(radius: int) -> Tensor:
    """Return every integer mode in the symmetric cube ``[-radius,radius]^3``."""

    if isinstance(radius, bool) or not isinstance(radius, int) or radius < 0:
        raise ValueError("reciprocal_mode_radius must be a nonnegative integer")
    axis = torch.arange(-radius, radius + 1, dtype=torch.long)
    mesh = torch.meshgrid(axis, axis, axis, indexing="ij")
    return torch.stack(mesh, dim=-1).reshape(-1, 3)


def _independent_phase_mode_indices(modes: Tensor) -> tuple[Tensor, Tensor]:
    """Return one representative of each ``{n,-n}`` pair and its nonzero mask.

    The complete symmetric cube remains the source for the normalized heat
    sums and reciprocal covariants.  Raw sine/cosine pair features, however,
    contain exact duplicates: ``cos(-theta)=cos(theta)`` and
    ``sin(-theta)=-sin(theta)``.  Removing those linearly dependent columns is
    information preserving in the pinned fractional basis and substantially
    reduces the first pair projection.  The zero-mode sine is omitted too.
    """

    rows = modes.tolist()
    indices: list[int] = []
    nonzero: list[bool] = []
    for index, row in enumerate(rows):
        first_nonzero = next((value for value in row if value != 0), 0)
        if first_nonzero >= 0:
            indices.append(index)
            nonzero.append(first_nonzero > 0)
    return (
        torch.tensor(indices, dtype=torch.long),
        torch.tensor(nonzero, dtype=torch.bool),
    )


@dataclass(frozen=True)
class InvariantPeriodicTransformerConfig:
    """Hyperparameters for :class:`InvariantPeriodicTransformer`."""

    input_scalar_dim: int
    model_dim: int = 192
    pair_dim: int = 96
    num_heads: int = 6
    num_layers: int = 6
    feedforward_multiplier: int = 4
    reciprocal_mode_radius: int = 2
    heat_scales: tuple[float, ...] = (0.04, 0.16, 0.64)
    direct_harmonics: int = 2
    include_lattice_frame_carriers: bool = True
    output_vector_dim: int = 48
    reference_length: float = 1.0
    # Dimensionless smooth-normalization floor.  The map
    # v/sqrt(||v||^2+epsilon) has Jacobian norm at most 1/sqrt(epsilon), so the
    # default explicitly caps this geometric preprocessing factor at 100.
    normalization_epsilon: float = 1.0e-4
    max_pair_interactions: int = 500_000
    max_reciprocal_modes: int = 4_096
    max_readout_coefficient_channels: int = 4_096
    max_geometry_workspace_bytes: int = 2_000_000_000
    minimum_dimensionless_lattice_singular_value: float = 1.0e-6
    maximum_dimensionless_lattice_singular_value: float = 1.0e6
    maximum_lattice_condition_number: float = 1.0e6
    radial_gaussian_widths: tuple[float, ...] = ()
    """Widths ``w`` (in ``reference_length`` units) of metric-aware pair scalars (review M2).

    Each adds ``rho_w(i, j) = sum_n s(|r_ij + n L|) exp(-|r_ij + n L|^2 / 2 w^2)`` over
    every periodic image within ``R = 6 max(w)``, where ``s(r) = (1 - r^2/R^2)^2``
    is a C^1 cutoff.  The image set is selected by distance, so the feature is
    invariant under proper rotations and lattice basis changes; the dropped
    tail is at most ``exp(-18)`` per image.  ``()`` keeps the legacy features.
    """
    max_radial_images: int = 4_096
    """Periodic images per radial chunk (memory bound).  Every image within ``R``
    is summed, chunk by chunk, so small cells (the CFM base reaches the chart's
    shortest-vector floor) are never refused for memory alone."""
    max_radial_image_chunks: int = 64
    """Chunks per evaluation (time bound): at most ``max_radial_images *
    max_radial_image_chunks`` images, else a ``ValueError``."""
    pair_carrier_smoothing_length: float | None = None
    """Length ``s`` (in ``reference_length`` units) over which every pair carrier
    vanishes at a coincidence (research 2026-11-12); ``None`` keeps the legacy carriers.

    The legacy carrier ``m / sqrt(|m|^2 + epsilon)`` is a unit vector down to a
    separation ``sqrt(epsilon) / 2 pi`` (0.0016 for the default epsilon), so the
    field is close to a step function of the separation of two atoms at their
    coincidence, and the sampling ODE collapses symmetry-related atoms onto a
    special position (R1_LOG.md, "Orbit collapse").  With ``s`` set, every direct
    and reciprocal pair carrier is multiplied by

        sqrt((|m_1|^2 + epsilon) / (|m_1|^2 + (2 pi s)^2)),

    where ``m_1 = sin(2 pi delta) L / reference_length`` is the first direct
    moment and ``|m_1| = 2 pi rho``.  ``rho = r (1 + O(|delta|^2))`` near a
    coincidence in every basis; ``rho <= r`` holds in orthogonal cells but not in
    a strongly sheared basis.  The first direct carrier becomes exactly
    ``m_1 / sqrt(|m_1|^2 + (2 pi s)^2)``, that is ``rho / sqrt(rho^2 + s^2)``
    times its direction, and every carrier is unchanged where ``rho`` is far
    above ``s``.

    Scope: ``m_1`` vanishes at all eight ``delta`` in ``{0, 1/2}^3``, so ``rho`` is
    the distance to the nearest half-lattice translation, not the pair
    separation.  The switch therefore also makes every pair carrier vanish
    linearly within about ``s`` of the seven non-zero half-lattice separations,
    in a cell of any size, where the legacy carriers already vanish and change
    sign over ``sqrt(epsilon) / 2 pi``.  The node chirality is computed from the
    switched carriers.  The factor is a function of ``|m_1|^2`` only, so SO(3)
    and permutation equivariance are untouched; it adds no parameter.
    Requires ``(2 pi s)^2 > normalization_epsilon``.
    """


@dataclass(frozen=True)
class InvariantPeriodicTransformerOutput:
    """Invariant atom channels and proper-SO(3)-equivariant row vectors."""

    scalar_channels: Tensor
    vector_channels: Tensor
    pooled_scalars: Tensor


@dataclass(frozen=True)
class _PeriodicGeometry:
    pair_features: Tensor
    covariant_bases: Tensor
    node_chirality: Tensor


class _DenseInvariantAttentionBlock(nn.Module):
    """Pre-norm full attention with scalar pair bias and pair value."""

    def __init__(self, config: InvariantPeriodicTransformerConfig) -> None:
        super().__init__()
        self.num_heads = config.num_heads
        self.head_dim = config.model_dim // config.num_heads
        self.model_dim = config.model_dim

        self.attention_norm = nn.LayerNorm(config.model_dim)
        self.query = nn.Linear(config.model_dim, config.model_dim, bias=False)
        self.key = nn.Linear(config.model_dim, config.model_dim, bias=False)
        self.value = nn.Linear(config.model_dim, config.model_dim, bias=False)
        self.pair_bias = nn.Linear(config.pair_dim, config.num_heads, bias=False)
        self.pair_value = nn.Sequential(
            nn.Linear(config.pair_dim, config.pair_dim),
            nn.SiLU(),
            nn.Linear(config.pair_dim, config.model_dim, bias=False),
        )
        self.attention_output = nn.Linear(config.model_dim, config.model_dim)

        hidden_dim = config.feedforward_multiplier * config.model_dim
        self.feedforward_norm = nn.LayerNorm(config.model_dim)
        self.feedforward = nn.Sequential(
            nn.Linear(config.model_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, config.model_dim),
        )

    def forward(
        self, scalars: Tensor, pair_channels: Tensor, key_mask: Tensor | None = None
    ) -> Tensor:
        batch_size, atom_count, _ = scalars.shape
        normalized = self.attention_norm(scalars)
        query = self.query(normalized).reshape(
            batch_size, atom_count, self.num_heads, self.head_dim
        )
        key = self.key(normalized).reshape(
            batch_size, atom_count, self.num_heads, self.head_dim
        )
        value = self.value(normalized).reshape(
            batch_size, atom_count, self.num_heads, self.head_dim
        )

        pair_bias = self.pair_bias(pair_channels).permute(0, 3, 1, 2)
        pair_value = self.pair_value(pair_channels).reshape(
            batch_size,
            atom_count,
            atom_count,
            self.num_heads,
            self.head_dim,
        )

        # Q/K projections may participate in BF16/FP16 autocast, but logits,
        # normalization, and weighted reduction are accumulated in at least
        # FP32.  This mirrors stable attention kernels and prevents low-
        # precision exponentiation from becoming the numerical contract.
        accumulation_dtype = (
            torch.float64 if query.dtype == torch.float64 else torch.float32
        )
        with torch.autocast(device_type=scalars.device.type, enabled=False):
            logits = torch.einsum(
                "bihd,bjhd->bhij",
                query.to(dtype=accumulation_dtype),
                key.to(dtype=accumulation_dtype),
            )
            logits = logits / math.sqrt(self.head_dim)
            logits = logits + pair_bias.to(dtype=accumulation_dtype)
            if key_mask is not None:
                # Padded atoms are never keys; every crystal keeps one real key.
                logits = logits.masked_fill(~key_mask[:, None, None, :], -torch.inf)
            attention = torch.softmax(logits, dim=-1)
            attended = torch.einsum(
                "bhij,bjhd->bihd",
                attention,
                value.to(dtype=accumulation_dtype),
            )
            attended = attended + torch.einsum(
                "bhij,bijhd->bihd",
                attention,
                pair_value.to(dtype=accumulation_dtype),
            )
        attended = attended.to(dtype=value.dtype)
        scalars = scalars + self.attention_output(
            attended.reshape(batch_size, atom_count, self.model_dim)
        )
        scalars = scalars + self.feedforward(self.feedforward_norm(scalars))
        return scalars


class InvariantPeriodicTransformer(nn.Module):
    """Scalar all-pairs attention with no sparse local graph message passing.

    ``scalar_features`` may be ``[N,D]`` or homogeneous packed ``[B,N,D]``.
    Fractional coordinates and row lattices must respectively be ``[N,3]`` /
    ``[3,3]`` or ``[B,N,3]`` / ``[B,3,3]``.  The returned tensors preserve the
    corresponding unbatched or packed convention.
    """

    def __init__(self, config: InvariantPeriodicTransformerConfig) -> None:
        super().__init__()
        self._validate_config(config)
        self.config = config
        mode_count = (2 * config.reciprocal_mode_radius + 1) ** 3
        if mode_count > config.max_reciprocal_modes:
            raise ValueError(
                "reciprocal-mode cube exceeds max_reciprocal_modes before allocation"
            )
        modes = _reciprocal_mode_cube(config.reciprocal_mode_radius)
        self.register_buffer("reciprocal_modes", modes, persistent=True)
        feature_indices, feature_nonzero = _independent_phase_mode_indices(modes)
        self.register_buffer(
            "phase_feature_mode_indices", feature_indices, persistent=True
        )
        self.register_buffer(
            "phase_feature_nonzero", feature_nonzero, persistent=True
        )
        self._raw_phase_feature_count = int(feature_indices.numel()) + int(
            feature_nonzero.sum().item()
        )

        if any(
            not math.isfinite(float(width)) or float(width) <= 0.0
            for width in config.radial_gaussian_widths
        ):
            raise ValueError("radial_gaussian_widths must be finite and positive")
        for name in ("max_radial_images", "max_radial_image_chunks"):
            value = getattr(config, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        smoothing = config.pair_carrier_smoothing_length
        if smoothing is not None and (
            isinstance(smoothing, bool)
            or not isinstance(smoothing, Real)
            or not math.isfinite(float(smoothing))
            or float(smoothing) <= 0.0
            or (2.0 * math.pi * float(smoothing)) ** 2 <= float(config.normalization_epsilon)
        ):
            raise ValueError(
                "pair_carrier_smoothing_length must be None or a finite positive length "
                "with (2 pi s)^2 above normalization_epsilon"
            )
        raw_pair_dim = (
            self._raw_phase_feature_count
            + len(config.heat_scales)
            + 3
            + len(config.radial_gaussian_widths)
        )
        self.scalar_input = nn.Sequential(
            nn.Linear(config.input_scalar_dim, config.model_dim),
            nn.SiLU(),
            nn.Linear(config.model_dim, config.model_dim),
        )
        self.chirality_input = nn.Linear(1, config.model_dim, bias=False)
        self.pair_input = nn.Sequential(
            nn.Linear(raw_pair_dim, config.pair_dim),
            nn.SiLU(),
            nn.Linear(config.pair_dim, config.pair_dim),
        )
        self.blocks = nn.ModuleList(
            [_DenseInvariantAttentionBlock(config) for _ in range(config.num_layers)]
        )
        self.final_norm = nn.LayerNorm(config.model_dim)

        primary_basis_count = config.direct_harmonics + len(config.heat_scales)
        basis_count = (
            primary_basis_count
            + config.direct_harmonics * len(config.heat_scales)
            + (3 if config.include_lattice_frame_carriers else 0)
        )
        coefficient_channels = config.output_vector_dim * basis_count
        if coefficient_channels > config.max_readout_coefficient_channels:
            raise ValueError(
                "vector readout exceeds max_readout_coefficient_channels before "
                "parameter allocation"
            )
        readout_input_dim = 2 * config.model_dim + config.pair_dim
        self.vector_coefficients = nn.Sequential(
            nn.Linear(readout_input_dim, config.model_dim),
            nn.SiLU(),
            nn.Linear(
                config.model_dim,
                coefficient_channels,
            ),
            nn.Tanh(),
        )
        self._basis_count = basis_count

    @staticmethod
    def _validate_config(config: InvariantPeriodicTransformerConfig) -> None:
        positive_integers = {
            "input_scalar_dim": config.input_scalar_dim,
            "model_dim": config.model_dim,
            "pair_dim": config.pair_dim,
            "num_heads": config.num_heads,
            "num_layers": config.num_layers,
            "feedforward_multiplier": config.feedforward_multiplier,
            "direct_harmonics": config.direct_harmonics,
            "output_vector_dim": config.output_vector_dim,
            "max_pair_interactions": config.max_pair_interactions,
            "max_reciprocal_modes": config.max_reciprocal_modes,
            "max_readout_coefficient_channels": (
                config.max_readout_coefficient_channels
            ),
            "max_geometry_workspace_bytes": config.max_geometry_workspace_bytes,
        }
        for name, value in positive_integers.items():
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if type(config.include_lattice_frame_carriers) is not bool:
            raise ValueError("include_lattice_frame_carriers must be boolean")
        if config.model_dim % config.num_heads:
            raise ValueError("model_dim must be divisible by num_heads")
        if (
            isinstance(config.reciprocal_mode_radius, bool)
            or not isinstance(config.reciprocal_mode_radius, int)
            or config.reciprocal_mode_radius < 1
        ):
            raise ValueError("reciprocal_mode_radius must be a positive integer")
        if not isinstance(config.heat_scales, tuple) or not config.heat_scales:
            raise ValueError(
                "heat_scales must be a nonempty tuple of finite positive values"
            )
        if any(
            isinstance(scale, bool)
            or not isinstance(scale, Real)
            or not math.isfinite(float(scale))
            or float(scale) <= 0.0
            for scale in config.heat_scales
        ):
            raise ValueError(
                "heat_scales must be a nonempty tuple of finite positive values"
            )
        if config.direct_harmonics + len(config.heat_scales) < 3:
            raise ValueError("at least three primary covariant bases are required")
        for name, value in (
            ("reference_length", config.reference_length),
            ("normalization_epsilon", config.normalization_epsilon),
            (
                "minimum_dimensionless_lattice_singular_value",
                config.minimum_dimensionless_lattice_singular_value,
            ),
            (
                "maximum_dimensionless_lattice_singular_value",
                config.maximum_dimensionless_lattice_singular_value,
            ),
            (
                "maximum_lattice_condition_number",
                config.maximum_lattice_condition_number,
            ),
        ):
            if (
                isinstance(value, bool)
                or not isinstance(value, Real)
                or not math.isfinite(float(value))
                or float(value) <= 0.0
            ):
                raise ValueError(f"{name} must be finite and positive")
        if (
            config.maximum_dimensionless_lattice_singular_value
            <= config.minimum_dimensionless_lattice_singular_value
        ):
            raise ValueError(
                "maximum_dimensionless_lattice_singular_value must exceed its minimum"
            )
        if config.maximum_lattice_condition_number < 1.0:
            raise ValueError("maximum_lattice_condition_number must be at least one")

    def _canonicalize_inputs(
        self,
        scalar_features: Tensor,
        fractional_coordinates: Tensor,
        lattice: Tensor,
        *,
        validate_geometry: bool,
        checks: DeviceChecks | None = None,
    ) -> tuple[Tensor, Tensor, Tensor, bool]:
        unbatched = scalar_features.ndim == 2
        if unbatched:
            scalar_features = scalar_features.unsqueeze(0)
            fractional_coordinates = fractional_coordinates.unsqueeze(0)
            lattice = lattice.unsqueeze(0)
        elif scalar_features.ndim != 3:
            raise ValueError("scalar_features must have shape [N,D] or [B,N,D]")

        if scalar_features.shape[-1] != self.config.input_scalar_dim:
            raise ValueError("the final scalar feature dimension is inconsistent")
        if scalar_features.shape[:2] != fractional_coordinates.shape[:2] or (
            fractional_coordinates.ndim != 3
            or fractional_coordinates.shape[-1] != 3
        ):
            raise ValueError("fractional coordinates must have shape [N,3] or [B,N,3]")
        if lattice.shape != (scalar_features.shape[0], 3, 3):
            raise ValueError("lattice must have shape [3,3] or [B,3,3]")
        if scalar_features.shape[1] < 1:
            raise ValueError("every structure must contain at least one atom")
        pair_interactions = (
            scalar_features.shape[0]
            * scalar_features.shape[1]
            * scalar_features.shape[1]
        )
        if pair_interactions > self.config.max_pair_interactions:
            raise ValueError(
                "dense pair-interaction budget exceeded before allocation "
                f"({pair_interactions} > {self.config.max_pair_interactions})"
            )
        if not (
            scalar_features.is_floating_point()
            and fractional_coordinates.is_floating_point()
            and lattice.is_floating_point()
        ):
            raise ValueError("all model inputs must be floating-point tensors")
        if not (
            scalar_features.dtype
            == fractional_coordinates.dtype
            == lattice.dtype
        ):
            raise ValueError("all model inputs must share one dtype")
        if not (
            scalar_features.device
            == fractional_coordinates.device
            == lattice.device
        ):
            raise ValueError("all model inputs must share one device")
        parameter = next(self.parameters())
        if (
            scalar_features.dtype != parameter.dtype
            or scalar_features.device != parameter.device
        ):
            raise ValueError(
                "model inputs and parameters must share one master dtype/device; "
                "use an autocast context rather than low-precision input tensors"
            )
        # Finite values and positive handedness are current-state obligations,
        # never properties that a caller-provided certification flag may waive.
        # Their flags are read in one host synchronization (review 2026-10-29,
        # H1): here, or by the caller that collects ``checks``.
        own = checks is None
        if checks is None:
            checks = DeviceChecks()
        for name, value in (
            ("scalar_features", scalar_features),
            ("fractional_coordinates", fractional_coordinates),
            ("lattice", lattice),
        ):
            checks.add(
                ~torch.isfinite(value).all(),
                partial(ValueError, f"{name} entries must be finite"),
            )
        validation_dtype = (
            torch.float64 if lattice.dtype == torch.float64 else torch.float32
        )
        validation_lattice = lattice.to(dtype=validation_dtype)
        determinant = torch.linalg.det(validation_lattice)
        checks.add(
            (determinant <= 0.0).any(),
            partial(ValueError, "every lattice must be right-handed and nonsingular"),
        )
        if own:
            # Standalone callers read the finiteness flags before any LAPACK call:
            # some backends raise on an SVD of non-finite data, which would hide
            # the input error.  A collecting caller's ``deferred`` block handles it.
            checks.raise_first()
        if validate_geometry:
            scaled_lattice = validation_lattice / float(self.config.reference_length)
            singular_values = torch.linalg.svdvals(scaled_lattice)
            lower = float(self.config.minimum_dimensionless_lattice_singular_value)
            upper = float(self.config.maximum_dimensionless_lattice_singular_value)
            maximum_condition = float(self.config.maximum_lattice_condition_number)
            condition_number = singular_values[..., 0] / singular_values[..., -1]
            checks.add(
                (
                    (singular_values[..., -1] < lower)
                    | (singular_values[..., 0] > upper)
                    | (condition_number > maximum_condition)
                ).any(),
                partial(ValueError, "lattice lies outside the supported nonsingular domain"),
            )
        if own:
            checks.raise_first()
        return scalar_features, fractional_coordinates, lattice, unbatched

    def _estimated_workspace_bytes(
        self,
        *,
        batch_size: int,
        atom_count: int,
        dtype: torch.dtype,
    ) -> int:
        """Conservative differentiable-tensor materialization estimate.

        This is a resource preflight, not a FLOP or allocator-peak model.  It
        intentionally sums intermediates that autograd may retain across the
        Fourier geometry, every attention layer, and the covariant readout.
        Kernel workspaces and framework allocator overhead remain outside this
        count and are measured separately by accelerator pilots.
        """

        pairs = batch_size * atom_count * atom_count
        modes = int(self.reciprocal_modes.shape[0])
        scales = len(self.config.heat_scales)
        harmonics = self.config.direct_harmonics
        basis_count = self._basis_count
        raw_phase = self._raw_phase_feature_count
        raw_pair = raw_phase + scales + 3 + len(self.config.radial_gaussian_widths)
        geometry = pairs * (
            3
            + 3 * modes
            + raw_phase
            + scales
            + 6 * scales
            + 6 * harmonics
            + 3 * basis_count
            + raw_pair
        )
        attention = self.config.num_layers * pairs * (
            2 * self.config.num_heads
            + self.config.pair_dim
            + self.config.model_dim
        )
        learned_pairs = pairs * (
            2 * self.config.pair_dim
            + 3 * self.config.model_dim
            + self.config.output_vector_dim * basis_count
        )
        node_state = (
            batch_size
            * atom_count
            * self.config.model_dim
            * (3 + 2 * self.config.num_layers)
        )
        bytes_per_element = 8 if dtype == torch.float64 else 4
        return int(
            bytes_per_element * (geometry + attention + learned_pairs + node_state)
        )

    def maximum_batch_size(self, *, atom_count: int, dtype: torch.dtype) -> int:
        """Largest homogeneous batch admitted by pair/workspace preflight.

        The workspace estimate is linear in the batch size.  Returning zero
        means even one candidate exceeds the declared resource contract;
        callers may split candidate pools, but must not silently change their
        individual scientific support.  Allocator overhead remains outside
        this deterministic tensor-workspace estimate.
        """

        if isinstance(atom_count, bool) or not isinstance(atom_count, int) or atom_count < 1:
            raise ValueError("atom_count must be a positive integer")
        if dtype not in {torch.float32, torch.float64}:
            raise ValueError("batch capacity requires float32 or float64 master precision")
        per_candidate = self._estimated_workspace_bytes(
            batch_size=1, atom_count=atom_count, dtype=dtype
        )
        return min(
            self.config.max_pair_interactions // (atom_count * atom_count),
            self.config.max_geometry_workspace_bytes // per_candidate,
        )

    def _periodic_geometry(
        self,
        fractional_coordinates: Tensor,
        lattice: Tensor,
        atom_mask: Tensor | None = None,
    ) -> _PeriodicGeometry:
        dtype = fractional_coordinates.dtype
        device = fractional_coordinates.device
        reference = self.config.reference_length
        epsilon = self.config.normalization_epsilon
        modes = self.reciprocal_modes.to(device=device, dtype=dtype)

        # delta[b,i,j] is the source-j minus destination-i torus coordinate.
        delta = (
            fractional_coordinates[:, None, :, :]
            - fractional_coordinates[:, :, None, :]
        )
        phase = 2.0 * math.pi * torch.einsum("bijc,mc->bijm", delta, modes)
        phase_cosine = torch.cos(phase)
        phase_sine = torch.sin(phase)
        feature_phase = phase.index_select(
            -1, self.phase_feature_mode_indices.to(device=device)
        )
        feature_cosine = torch.cos(feature_phase)
        feature_sine = torch.sin(
            feature_phase[..., self.phase_feature_nonzero.to(device=device)]
        )

        inverse_transpose = torch.linalg.inv(lattice).transpose(-1, -2)
        reciprocal = 2.0 * math.pi * torch.einsum(
            "mc,bcd->bmd", modes, inverse_transpose
        )
        reciprocal_squared = (reference * reciprocal).square().sum(dim=-1)
        scales = torch.as_tensor(
            self.config.heat_scales, dtype=dtype, device=device
        )
        heat_weight = torch.exp(
            -scales[None, :, None] * reciprocal_squared[:, None, :]
        )
        heat_normalizer = heat_weight.sum(dim=-1).clamp_min(epsilon)
        heat_features = torch.einsum(
            "btm,bijm->bijt", heat_weight, phase_cosine
        ) / heat_normalizer[:, None, None, :]

        reciprocal_moments = torch.einsum(
            "btm,bijm,bmc->bijtc",
            heat_weight,
            phase_sine,
            reference * reciprocal,
        ) / heat_normalizer[:, None, None, :, None]

        harmonics = torch.arange(
            1,
            self.config.direct_harmonics + 1,
            dtype=dtype,
            device=device,
        )
        direct_phase = 2.0 * math.pi * (
            delta[..., None, :] * harmonics[None, None, None, :, None]
        )
        direct_fractional = torch.sin(direct_phase) / harmonics[
            None, None, None, :, None
        ]
        direct_moments = torch.einsum(
            "bijhc,bcd->bijhd", direct_fractional, lattice / reference
        )

        direct_unit = _smooth_unit_vectors(direct_moments, epsilon)
        reciprocal_unit = _smooth_unit_vectors(reciprocal_moments, epsilon)
        smoothing = self.config.pair_carrier_smoothing_length
        if smoothing is not None:
            # |m_1|^2 = (2 pi rho)^2 for the smooth periodic separation rho.  The
            # argument of the root is at least epsilon / (2 pi s)^2 > 0, so the
            # switch is smooth at a coincidence and on the diagonal.
            first_squared = direct_moments[..., 0, :].square().sum(dim=-1)
            floor = (2.0 * math.pi * float(smoothing)) ** 2
            switch = torch.sqrt((first_squared + epsilon) / (first_squared + floor))
            direct_unit = direct_unit * switch[..., None, None]
            reciprocal_unit = reciprocal_unit * switch[..., None, None]
        primary_bases = torch.cat((direct_unit, reciprocal_unit), dim=-2)

        # A local, permutation-equivariant pseudoscalar gives scalar attention
        # access to handedness without introducing learned vector channels.
        if atom_mask is None:
            node_moments = primary_bases.mean(dim=2)
        else:
            partners = atom_mask.to(dtype=dtype)
            node_moments = (primary_bases * partners[:, None, :, None, None]).sum(
                dim=2
            ) / partners.sum(dim=1)[:, None, None, None]
        node_chirality = torch.tanh(
            (
                node_moments[:, :, 0]
                * torch.cross(
                    node_moments[:, :, 1], node_moments[:, :, 2], dim=-1
                )
            ).sum(dim=-1, keepdim=True)
        )

        cross_bases = torch.cross(
            direct_unit[..., :, None, :].expand(
                *direct_unit.shape[:-2],
                self.config.direct_harmonics,
                len(self.config.heat_scales),
                3,
            ),
            reciprocal_unit[..., None, :, :].expand(
                *reciprocal_unit.shape[:-2],
                self.config.direct_harmonics,
                len(self.config.heat_scales),
                3,
            ),
            dim=-1,
        ).flatten(start_dim=-3, end_dim=-2)
        covariant_parts = [primary_bases, cross_bases]
        if self.config.include_lattice_frame_carriers:
            # At a complete advertised-slot collision every sine-odd pair
            # carrier vanishes.  The three rows of the pinned canonical lattice
            # are bounded, nonzero polar covariants and therefore retain a legal
            # direction source for chemically distinct tokens.  Identical
            # colliding tokens still receive the same vector by permutation
            # equivariance, so the translation quotient removes their common
            # motion as required.
            frame = _smooth_unit_vectors(lattice / reference, epsilon)
            atom_count = fractional_coordinates.shape[1]
            frame = frame[:, None, None, :, :].expand(
                -1, atom_count, atom_count, -1, -1
            )
            covariant_parts.append(frame)
        covariant_bases = torch.cat(covariant_parts, dim=-2)

        atom_count = fractional_coordinates.shape[1]
        chirality_i = node_chirality[:, :, None, :].expand(
            -1, -1, atom_count, -1
        )
        chirality_j = node_chirality[:, None, :, :].expand(
            -1, atom_count, -1, -1
        )
        pair_parts = [
            feature_cosine,
            feature_sine,
            heat_features,
            chirality_i,
            chirality_j,
            chirality_i * chirality_j,
        ]
        if self.config.radial_gaussian_widths:
            pair_parts.append(self._radial_gaussian_features(delta, lattice / reference))
        raw_pair_features = torch.cat(pair_parts, dim=-1)
        return _PeriodicGeometry(
            pair_features=raw_pair_features,
            covariant_bases=covariant_bases,
            node_chirality=node_chirality,
        )

    def _radial_gaussian_features(self, delta: Tensor, lattice: Tensor) -> Tensor:
        """Smoothly truncated periodic Gaussian radial basis, ``[B, N, N, K]``.

        Reduced displacements satisfy ``|d_j| <= 1/2``.  An image ``(d + n) L``
        within ``R`` has ``|d_j + n_j| <= R sqrt((G^-1)_jj)``, so the integer box
        ``|n_j| <= ceil(R sqrt((G^-1)_jj) + 1/2)`` contains every image the
        cutoff keeps.
        """

        widths = torch.as_tensor(
            self.config.radial_gaussian_widths, dtype=delta.dtype, device=delta.device
        )
        radius = 6.0 * float(max(self.config.radial_gaussian_widths))
        reduced = delta - torch.round(delta)
        with torch.no_grad():
            inverse_metric = torch.linalg.inv(lattice @ lattice.transpose(-1, -2))
            reach = radius * torch.sqrt(torch.diagonal(inverse_metric, dim1=-2, dim2=-1))
            bounds = torch.ceil(reach.amax(dim=0) + 0.5).to(torch.long).tolist()
        axes = [
            torch.arange(-bound, bound + 1, device=delta.device, dtype=delta.dtype)
            for bound in bounds
        ]
        shifts = torch.stack(torch.meshgrid(*axes, indexing="ij"), dim=-1).reshape(-1, 3)
        chunk = self.config.max_radial_images
        if shifts.shape[0] > chunk * self.config.max_radial_image_chunks:
            raise ValueError(
                "radial pair features need more periodic images than "
                "max_radial_images * max_radial_image_chunks"
            )
        total: Tensor | None = None
        for start in range(0, shifts.shape[0], chunk):
            images = torch.einsum(
                "bijsc,bcd->bijsd",
                reduced[..., None, :] + shifts[start : start + chunk],
                lattice,
            )
            squared = images.square().sum(dim=-1)
            cutoff = torch.clamp(1.0 - squared / (radius * radius), min=0.0).square()
            gaussian = torch.exp(-squared[..., None] / (2.0 * widths * widths))
            part = (cutoff[..., None] * gaussian).sum(dim=-2)
            total = part if total is None else total + part
        if total is None:  # pragma: no cover - the box always holds the zero shift
            raise RuntimeError("empty radial image box")
        return total

    def forward(
        self,
        scalar_features: Tensor,
        fractional_coordinates: Tensor,
        lattice: Tensor,
        *,
        validate_geometry: bool = True,
        atom_mask: Tensor | None = None,
        checks: DeviceChecks | None = None,
    ) -> InvariantPeriodicTransformerOutput:
        """Evaluate the model, optionally trusting a certified bridge boundary.

        Direct callers should retain ``validate_geometry=True``.  ``False`` may
        skip only the declared singular-value envelope check after a separately
        bound chart proof; finite-value and positive-handedness checks are never
        skipped.  CrystalAtlas field calls currently retain the full check.
        ``checks`` lets the field read every input flag in one host
        synchronization (review H1); the checks themselves still run.

        ``atom_mask`` (``[B, N]`` bool, review M5) marks real atoms in a padded
        batch.  Padded atoms are never attention keys, partner means, vector
        sources, or pooled members, and their output channels are zero.  The
        real atoms therefore receive their unpadded values to rounding.  An
        all-true mask takes the unmasked path.
        """

        if not isinstance(validate_geometry, bool):
            raise ValueError("validate_geometry must be boolean")
        scalars_in, fractional, lattice_in, unbatched = self._canonicalize_inputs(
            scalar_features,
            fractional_coordinates,
            lattice,
            validate_geometry=validate_geometry,
            checks=checks,
        )
        if atom_mask is not None:
            if unbatched:
                atom_mask = atom_mask.unsqueeze(0)
            if atom_mask.dtype != torch.bool or atom_mask.shape != scalars_in.shape[:2]:
                raise ValueError("atom_mask must be a bool tensor of shape [B, N]")
            if atom_mask.device != scalars_in.device:
                raise ValueError("atom_mask must share the model input device")
            if not bool(atom_mask.any(dim=1).all().item()):
                raise ValueError("every padded structure must keep at least one atom")
            if bool(atom_mask.all().item()):
                atom_mask = None
        estimated_workspace_bytes = self._estimated_workspace_bytes(
            batch_size=scalars_in.shape[0],
            atom_count=scalars_in.shape[1],
            dtype=scalars_in.dtype,
        )
        if estimated_workspace_bytes > self.config.max_geometry_workspace_bytes:
            raise ValueError(
                "dense geometry workspace budget exceeded before pair allocation "
                f"({estimated_workspace_bytes} > "
                f"{self.config.max_geometry_workspace_bytes} bytes)"
            )
        # Fourier phases, reciprocal vectors, and handedness features are
        # deterministic crystallographic state.  Keep them in master
        # precision even when the learned linear/attention maps participate in
        # an enclosing AMP context.
        geometry_dtype = (
            torch.float64 if fractional.dtype == torch.float64 else torch.float32
        )
        with torch.autocast(device_type=fractional.device.type, enabled=False):
            geometry = self._periodic_geometry(
                fractional.to(dtype=geometry_dtype),
                lattice_in.to(dtype=geometry_dtype),
                atom_mask,
            )
        pair_channels = self.pair_input(geometry.pair_features)
        scalars = self.scalar_input(scalars_in) + self.chirality_input(
            geometry.node_chirality
        )
        for block in self.blocks:
            scalars = block(scalars, pair_channels, atom_mask)
        scalars = self.final_norm(scalars)

        atom_count = scalars.shape[1]
        scalar_i = scalars[:, :, None, :].expand(-1, -1, atom_count, -1)
        scalar_j = scalars[:, None, :, :].expand(-1, atom_count, -1, -1)
        coefficient_input = torch.cat(
            (scalar_i, scalar_j, pair_channels), dim=-1
        )
        coefficients = self.vector_coefficients(coefficient_input).reshape(
            scalars.shape[0],
            atom_count,
            atom_count,
            self.config.output_vector_dim,
            self._basis_count,
        )
        if atom_mask is None:
            vectors = torch.einsum(
                "bijvk,bijkc->bivc", coefficients, geometry.covariant_bases
            ) / float(atom_count)
            pooled = scalars.mean(dim=1)
        else:
            real = atom_mask.to(dtype=scalars.dtype)
            counts = real.sum(dim=1)
            sources = coefficients * real[:, None, :, None, None]
            vectors = torch.einsum(
                "bijvk,bijkc->bivc", sources, geometry.covariant_bases
            ) / counts[:, None, None, None]
            vectors = vectors * real[:, :, None, None]
            scalars = scalars * real[:, :, None]
            pooled = scalars.sum(dim=1) / counts[:, None]

        if unbatched:
            return InvariantPeriodicTransformerOutput(
                scalar_channels=scalars[0],
                vector_channels=vectors[0],
                pooled_scalars=pooled[0],
            )
        return InvariantPeriodicTransformerOutput(
            scalar_channels=scalars,
            vector_channels=vectors,
            pooled_scalars=pooled,
        )


__all__ = [
    "InvariantPeriodicTransformer",
    "InvariantPeriodicTransformerConfig",
    "InvariantPeriodicTransformerOutput",
]
