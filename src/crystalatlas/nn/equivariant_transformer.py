"""A periodic proper-SE(3)-equivariant atom-set Transformer.

The implementation follows equations (1.39)--(1.40) and (3.11)--(3.13):

* attention is bidirectional and image resolved;
* the complete unnormalised edge weight is multiplied by a C1 cutoff;
* the central self edge is replaced by a persistent positive self-token path;
* directional values are regular polynomials, never ``d / ||d||``;
* scalar and Cartesian-vector channels transform under proper rotations; and
* oriented cross/triple-product paths make the model SO(3), not accidentally
  O(3), equivariant.

The finite network is smooth for finite parameters and the edge topology is C1
across the cutoff.  This alone does not prove the uniform global Lipschitz and
linear-growth assumptions required by theorem F9; those remain a checkpoint-
specific analytic/numerical obligation.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

from .._optional import require_torch

torch = require_torch()
Tensor = torch.Tensor
nn = torch.nn


def smooth_whole_edge_cutoff(squared_distance: Tensor, cutoff_squared: float) -> Tensor:
    """A nonnegative compact C1 cutoff as a function of squared distance.

    ``chi(s) = (1 - s / rc^2)^2`` inside the open cutoff and zero outside.
    Both the value and derivative with respect to ``s`` vanish at ``rc^2``.
    """

    if cutoff_squared <= 0.0:
        raise ValueError("cutoff_squared must be positive")
    x = squared_distance / cutoff_squared
    return torch.where(x < 1.0, torch.square(1.0 - x), torch.zeros_like(x))


def regular_solid_harmonics(displacement: Tensor, max_degree: int = 2) -> Tensor:
    """Real regular solid-harmonic polynomial features through degree two.

    The normalization is conventional rather than orthonormal; covariance and
    collision regularity, not basis normalization, are the relevant contract.
    All positive-degree components vanish at a collision.
    """

    if displacement.shape[-1] != 3:
        raise ValueError("displacement must end in a Cartesian dimension of 3")
    if max_degree not in (0, 1, 2):
        raise ValueError("this implementation supports degrees 0, 1, and 2")
    one = torch.ones_like(displacement[..., :1])
    features = [one]
    if max_degree >= 1:
        features.append(displacement)
    if max_degree >= 2:
        x, y, z = displacement.unbind(dim=-1)
        root_three = math.sqrt(3.0)
        degree_two = torch.stack(
            (
                x * y,
                y * z,
                z * x,
                0.5 * (x.square() - y.square()),
                (2.0 * z.square() - x.square() - y.square()) / (2.0 * root_three),
            ),
            dim=-1,
        )
        features.append(degree_two)
    return torch.cat(features, dim=-1)


def oriented_triple_product(first: Tensor, second: Tensor, third: Tensor) -> Tensor:
    """SO(3)-invariant, reflection-odd scalar ``first . (second x third)``."""

    if first.shape[-1] != 3 or second.shape[-1] != 3 or third.shape[-1] != 3:
        raise ValueError("oriented triple products require Cartesian 3-vectors")
    return (first * torch.cross(second, third, dim=-1)).sum(dim=-1)


@dataclass(frozen=True)
class PeriodicEdges:
    destination: Tensor
    source: Tensor
    shifts: Tensor
    displacement: Tensor
    squared_distance: Tensor

    @property
    def num_edges(self) -> int:
        return int(self.destination.numel())


class PeriodicImageBudgetError(RuntimeError):
    """A certified edge enumeration exceeded its explicit resource contract."""


def concatenate_periodic_edges(
    edge_sets: Sequence[PeriodicEdges],
    atom_counts: Sequence[int],
    *,
    max_edges: int,
) -> PeriodicEdges:
    """Form an exact block-diagonal union of periodic edge lists.

    Atom indices of graph ``b`` are shifted by ``sum(atom_counts[:b])``.
    Displacements and cell shifts retain their original per-crystal meaning;
    no edge between different crystals is introduced.  The explicit total
    budget is checked before concatenation and is never used as a truncation.
    """

    if len(edge_sets) != len(atom_counts) or not edge_sets:
        raise ValueError("edge_sets and atom_counts must be equal nonempty sequences")
    if max_edges < 1:
        raise ValueError("max_edges must be positive")
    if any(
        isinstance(count, bool) or not isinstance(count, int) or count < 1
        for count in atom_counts
    ):
        raise ValueError("every packed graph must contain a positive integer atom count")
    first = edge_sets[0]
    device = first.destination.device
    displacement_dtype = first.displacement.dtype
    total_edges = sum(edges.num_edges for edges in edge_sets)
    if total_edges > max_edges:
        raise PeriodicImageBudgetError(
            f"packed edge count {total_edges} exceeds max_edges={max_edges}"
        )

    destinations: list[Tensor] = []
    sources: list[Tensor] = []
    shifts: list[Tensor] = []
    displacements: list[Tensor] = []
    squared: list[Tensor] = []
    offset = 0
    for edges, atom_count in zip(edge_sets, atom_counts, strict=True):
        if (
            edges.destination.device != device
            or edges.source.device != device
            or edges.shifts.device != device
            or edges.displacement.device != device
            or edges.squared_distance.device != device
        ):
            raise ValueError("all packed edge tensors must share one device")
        if edges.displacement.dtype != displacement_dtype:
            raise ValueError("all packed edge displacements must share one dtype")
        if edges.num_edges:
            if bool(
                (
                    (edges.destination < 0)
                    | (edges.destination >= atom_count)
                    | (edges.source < 0)
                    | (edges.source >= atom_count)
                ).any().item()
            ):
                raise ValueError("a component edge index lies outside its atom block")
            destinations.append(edges.destination + offset)
            sources.append(edges.source + offset)
            shifts.append(edges.shifts)
            displacements.append(edges.displacement)
            squared.append(edges.squared_distance)
        offset += atom_count
    if total_edges == 0:
        empty_index = torch.empty(0, dtype=torch.long, device=device)
        return PeriodicEdges(
            destination=empty_index,
            source=empty_index.clone(),
            shifts=torch.empty(0, 3, dtype=torch.long, device=device),
            displacement=torch.empty(
                0, 3, dtype=displacement_dtype, device=device
            ),
            squared_distance=torch.empty(
                0, dtype=first.squared_distance.dtype, device=device
            ),
        )
    return PeriodicEdges(
        destination=torch.cat(destinations),
        source=torch.cat(sources),
        shifts=torch.cat(shifts),
        displacement=torch.cat(displacements),
        squared_distance=torch.cat(squared),
    )


def enumerate_packed_periodic_edges(
    fractional_coordinates: Sequence[Tensor],
    lattices: Sequence[Tensor],
    cutoff: float,
    *,
    max_candidates_per_pair: int = 32_768,
    max_edges: int = 500_000,
    max_total_edges: int | None = None,
) -> PeriodicEdges:
    """Enumerate a block-diagonal periodic edge list for many crystals.

    Each component is certified independently by
    :func:`enumerate_periodic_edges`; the final union additionally obeys one
    batch-wide edge budget when ``max_total_edges`` is supplied.  ``max_edges``
    is deliberately a *per-crystal* support contract: packing a crystal with
    otherwise unrelated crystals cannot make that same crystal invalid.  If
    no total budget is supplied, the exact union is allowed to contain at most
    ``len(lattices) * max_edges`` edges.  A resource excess raises rather than
    silently dropping images or records.
    """

    if len(fractional_coordinates) != len(lattices) or not lattices:
        raise ValueError("coordinate and lattice sequences must be equal and nonempty")
    if max_total_edges is not None and max_total_edges < 1:
        raise ValueError("max_total_edges must be positive when supplied")
    edge_sets: list[PeriodicEdges] = []
    atom_counts: list[int] = []
    for coordinates, lattice in zip(fractional_coordinates, lattices, strict=True):
        atom_counts.append(int(coordinates.shape[0]))
        edge_sets.append(
            enumerate_periodic_edges(
                coordinates,
                lattice,
                cutoff,
                max_candidates_per_pair=max_candidates_per_pair,
                max_edges=max_edges,
            )
        )
    packed_budget = (
        len(edge_sets) * max_edges if max_total_edges is None else max_total_edges
    )
    return concatenate_periodic_edges(
        edge_sets,
        atom_counts,
        max_edges=packed_budget,
    )


def _integer_mesh(lower: Tensor, upper: Tensor, *, device: torch.device) -> Tensor:
    axes = [
        torch.arange(
            int(lower[index].item()),
            int(upper[index].item()) + 1,
            device=device,
            dtype=torch.long,
        )
        for index in range(3)
    ]
    if any(axis.numel() == 0 for axis in axes):
        return torch.empty(0, 3, dtype=torch.long, device=device)
    grid = torch.meshgrid(*axes, indexing="ij")
    return torch.stack(grid, dim=-1).reshape(-1, 3)


def _integer_mesh_slice(
    lower: Tensor,
    side_lengths: tuple[int, int, int],
    start: int,
    stop: int,
    *,
    device: torch.device,
) -> Tensor:
    """Return a lexicographic slice without materializing the whole mesh."""

    if not 0 <= start <= stop <= math.prod(side_lengths):
        raise ValueError("integer-mesh slice lies outside the mesh")
    linear = torch.arange(start, stop, dtype=torch.long, device=device)
    second_third = side_lengths[1] * side_lengths[2]
    first = torch.div(linear, second_third, rounding_mode="floor")
    remainder = torch.remainder(linear, second_third)
    second = torch.div(remainder, side_lengths[2], rounding_mode="floor")
    third = torch.remainder(remainder, side_lengths[2])
    return torch.stack((first, second, third), dim=-1) + lower


def enumerate_periodic_edges(
    fractional_coordinates: Tensor,
    lattice: Tensor,
    cutoff: float,
    *,
    max_candidates_per_pair: int = 32_768,
    max_edges: int = 500_000,
) -> PeriodicEdges:
    """Enumerate every directed periodic image strictly inside ``cutoff``.

    Coordinates are row vectors and lattice vectors are lattice rows.  For a
    displacement ``d = x L``, Cauchy--Schwarz gives

    ``|x_j| <= cutoff * ||column_j(L^{-1})||``.

    These three certified component bounds produce a finite pair-specific
    integer box.  Exact distances then filter that box.  Only ``a=b,n=0`` is
    excluded, so noncentral periodic self-images are retained.

    Edge-list membership is a discrete operation and is not differentiated.
    Returned displacements are recomputed from the original tensors, so values
    and forces inside a fixed list remain differentiable.
    """

    if fractional_coordinates.ndim != 2 or fractional_coordinates.shape[-1] != 3:
        raise ValueError("fractional_coordinates must have shape [N,3]")
    if lattice.shape != (3, 3):
        raise ValueError("lattice must have shape [3,3]")
    if cutoff <= 0.0:
        raise ValueError("cutoff must be positive")
    if max_candidates_per_pair < 1 or max_edges < 1:
        raise ValueError("periodic-image resource budgets must be positive")
    if fractional_coordinates.shape[0] == 0:
        raise ValueError("at least one atom is required")
    if not torch.isfinite(fractional_coordinates).all() or not torch.isfinite(lattice).all():
        raise ValueError("coordinates and lattice must be finite")
    determinant = torch.linalg.det(lattice)
    if not bool((determinant > 0).item()):
        raise ValueError("lattice must be right handed and nonsingular")

    # Detaching the integer bounds makes the nondifferentiable topology choice
    # explicit; selected displacements are recomputed from the original input
    # tensors below and therefore retain autograd.  Integer translations first
    # canonicalize each atom into one fractional cell.  A pair's canonical
    # image m and public/original image n are related exactly by
    #
    #     m = n + floor(x_source) - floor(x_destination).
    #
    # This prevents atoms represented many cells apart from inflating the
    # vectorized union box while preserving both physical displacements and
    # the shifts returned by the historical public API.
    detached_coordinates = fractional_coordinates.detach()
    detached_lattice = lattice.detach()

    # Topology decisions must not inherit float16/float32 roundoff from the
    # neural path.  In particular, rounding an image-box endpoint inward can
    # omit a physically in-cutoff image.  Convert the represented inputs to
    # float64, construct a one-integer-cell conservative halo around every
    # analytically derived component box, and perform the final membership
    # test in float64.  The halo is not a scientific truncation: extra
    # candidates are removed by the distance test below.  It also makes the
    # topology robust to the last-bit error of the 3x3 inverse.  Pathological
    # near-singular cells still fail the explicit candidate budget rather than
    # losing images silently.
    topology_coordinates = detached_coordinates.to(dtype=torch.float64)
    topology_lattice = detached_lattice.to(dtype=torch.float64)
    inverse = torch.linalg.inv(topology_lattice)
    component_bound = cutoff * torch.linalg.vector_norm(inverse, dim=0)
    if not bool(torch.isfinite(component_bound).all().item()):
        raise ValueError("the inverse-lattice image bound must be finite")

    # IEEE floating point cannot encode useful unit-cell translations near the
    # int64 limit.  Keeping a wide guard below 2**63 also makes every offset
    # difference and mapped image shift provably representable as torch.long.
    maximum_safe_offset = float(2**60)
    floored_coordinates = torch.floor(topology_coordinates)
    if bool((floored_coordinates.abs() > maximum_safe_offset).any().item()):
        raise ValueError("fractional-coordinate cell offsets exceed int64 support")
    cell_offsets = floored_coordinates.to(torch.long)

    atom_count = int(fractional_coordinates.shape[0])
    total_pairs = atom_count * atom_count
    # This is an implementation-only materialization cap, not a scientific
    # truncation.  All pair/shift blocks are eventually visited.
    materialized_candidates = 262_144

    global_lower: Tensor | None = None
    global_upper: Tensor | None = None
    preflight_pairs = materialized_candidates
    for pair_start in range(0, total_pairs, preflight_pairs):
        pair_stop = min(pair_start + preflight_pairs, total_pairs)
        pair_index = torch.arange(
            pair_start, pair_stop, dtype=torch.long, device=lattice.device
        )
        destination = torch.div(pair_index, atom_count, rounding_mode="floor")
        source = torch.remainder(pair_index, atom_count)
        detached_delta = (
            topology_coordinates[source] - topology_coordinates[destination]
        )
        lower_float = torch.ceil(-detached_delta - component_bound) - 1.0
        upper_float = torch.floor(-detached_delta + component_bound) + 1.0
        if bool(
            (
                (lower_float.abs() > maximum_safe_offset)
                | (upper_float.abs() > maximum_safe_offset)
            )
            .any()
            .item()
        ):
            raise PeriodicImageBudgetError(
                "certified periodic-image bounds exceed int64 support"
            )
        side_lengths = torch.clamp(upper_float - lower_float + 1.0, min=0.0)
        candidate_count = torch.prod(side_lengths, dim=-1)
        if bool((candidate_count > max_candidates_per_pair).any().item()):
            raise PeriodicImageBudgetError(
                "certified periodic-image box exceeds max_candidates_per_pair; "
                "the lattice is outside this Transformer's declared HPC support"
            )

        nonempty = candidate_count > 0.0
        if not bool(nonempty.any().item()):
            continue
        lower = lower_float.to(torch.long)
        upper = upper_float.to(torch.long)
        offset_delta = cell_offsets[source] - cell_offsets[destination]
        mapped_lower = (lower + offset_delta)[nonempty]
        mapped_upper = (upper + offset_delta)[nonempty]
        chunk_lower = torch.amin(mapped_lower, dim=0)
        chunk_upper = torch.amax(mapped_upper, dim=0)
        global_lower = (
            chunk_lower
            if global_lower is None
            else torch.minimum(global_lower, chunk_lower)
        )
        global_upper = (
            chunk_upper
            if global_upper is None
            else torch.maximum(global_upper, chunk_upper)
        )

    if global_lower is None or global_upper is None:
        empty_index = torch.empty(0, dtype=torch.long, device=lattice.device)
        return PeriodicEdges(
            destination=empty_index,
            source=empty_index.clone(),
            shifts=torch.empty(0, 3, dtype=torch.long, device=lattice.device),
            displacement=lattice.new_empty(0, 3),
            squared_distance=lattice.new_empty(0),
        )

    canonical_side_lengths = tuple(
        int(value)
        for value in (global_upper - global_lower + 1).detach().cpu().tolist()
    )
    canonical_shift_count = math.prod(canonical_side_lengths)
    if canonical_shift_count <= materialized_candidates:
        canonical_shifts: Tensor | None = _integer_mesh(
            global_lower, global_upper, device=lattice.device
        )
        pairs_per_chunk = max(1, materialized_candidates // canonical_shift_count)
        shifts_per_chunk = canonical_shift_count
    else:
        canonical_shifts = None
        # One pair at a time preserves destination/source/shift lexicographic
        # order when a union box itself must be split into shift chunks.
        pairs_per_chunk = 1
        shifts_per_chunk = materialized_candidates

    destinations: list[Tensor] = []
    sources: list[Tensor] = []
    shifts: list[Tensor] = []
    displacements: list[Tensor] = []
    squared: list[Tensor] = []
    cutoff_squared = cutoff * cutoff
    accepted_edge_count = 0

    for pair_start in range(0, total_pairs, pairs_per_chunk):
        pair_stop = min(pair_start + pairs_per_chunk, total_pairs)
        pair_index = torch.arange(
            pair_start, pair_stop, dtype=torch.long, device=lattice.device
        )
        pair_destination = torch.div(
            pair_index, atom_count, rounding_mode="floor"
        )
        pair_source = torch.remainder(pair_index, atom_count)
        fractional_delta = (
            fractional_coordinates[pair_source]
            - fractional_coordinates[pair_destination]
        )
        detached_delta = (
            topology_coordinates[pair_source]
            - topology_coordinates[pair_destination]
        )
        mapped_lower = (
            (torch.ceil(-detached_delta - component_bound) - 1.0).to(torch.long)
            + cell_offsets[pair_source]
            - cell_offsets[pair_destination]
        )
        mapped_upper = (
            (torch.floor(-detached_delta + component_bound) + 1.0).to(torch.long)
            + cell_offsets[pair_source]
            - cell_offsets[pair_destination]
        )
        offset_delta = cell_offsets[pair_source] - cell_offsets[pair_destination]

        for shift_start in range(0, canonical_shift_count, shifts_per_chunk):
            shift_stop = min(
                shift_start + shifts_per_chunk, canonical_shift_count
            )
            shift_block = (
                canonical_shifts[shift_start:shift_stop]
                if canonical_shifts is not None
                else _integer_mesh_slice(
                    global_lower,
                    canonical_side_lengths,
                    shift_start,
                    shift_stop,
                    device=lattice.device,
                )
            )
            inside = (
                (shift_block[None, :, :] >= mapped_lower[:, None, :])
                & (shift_block[None, :, :] <= mapped_upper[:, None, :])
            ).all(dim=-1)
            candidate_position = torch.nonzero(inside, as_tuple=False)
            if candidate_position.numel() == 0:
                continue
            pair_row = candidate_position[:, 0]
            shift_column = candidate_position[:, 1]
            accepted_destination = pair_destination[pair_row]
            accepted_source = pair_source[pair_row]
            accepted_shifts = (
                shift_block[shift_column] - offset_delta[pair_row]
            )
            candidate_displacements = (
                fractional_delta[pair_row]
                + accepted_shifts.to(fractional_delta.dtype)
            ) @ lattice
            candidate_squared = candidate_displacements.square().sum(dim=-1)
            topology_displacements = (
                detached_delta[pair_row]
                + accepted_shifts.to(dtype=torch.float64)
            ) @ topology_lattice
            topology_squared = topology_displacements.square().sum(dim=-1)
            accepted = topology_squared < cutoff_squared
            accepted = accepted & ~(
                (accepted_destination == accepted_source)
                & (accepted_shifts == 0).all(dim=-1)
            )
            if not bool(accepted.any().item()):
                continue
            count = int(torch.count_nonzero(accepted).item())
            accepted_edge_count += count
            if accepted_edge_count > max_edges:
                raise PeriodicImageBudgetError(
                    "periodic edge list exceeds max_edges; no edge was silently truncated"
                )
            destinations.append(accepted_destination[accepted])
            sources.append(accepted_source[accepted])
            shifts.append(accepted_shifts[accepted])
            displacements.append(candidate_displacements[accepted])
            squared.append(candidate_squared[accepted])

    if not destinations:
        empty_index = torch.empty(0, dtype=torch.long, device=lattice.device)
        return PeriodicEdges(
            destination=empty_index,
            source=empty_index.clone(),
            shifts=torch.empty(0, 3, dtype=torch.long, device=lattice.device),
            displacement=lattice.new_empty(0, 3),
            squared_distance=lattice.new_empty(0),
        )
    return PeriodicEdges(
        destination=torch.cat(destinations),
        source=torch.cat(sources),
        shifts=torch.cat(shifts),
        displacement=torch.cat(displacements),
        squared_distance=torch.cat(squared),
    )


class VectorLinear(nn.Module):
    """Bias-free channel mixing that commutes with Cartesian rotations."""

    def __init__(self, input_channels: int, output_channels: int) -> None:
        super().__init__()
        if input_channels <= 0 or output_channels <= 0:
            raise ValueError("vector channel counts must be positive")
        self.weight = nn.Parameter(torch.empty(output_channels, input_channels))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, vectors: Tensor) -> Tensor:
        if vectors.shape[-2] != self.weight.shape[1] or vectors.shape[-1] != 3:
            raise ValueError("vectors must have shape [..., input_channels, 3]")
        return torch.einsum("...ci,oc->...oi", vectors, self.weight)


def _vector_rms_normalize(vectors: Tensor, epsilon: float) -> Tensor:
    scale = torch.sqrt(vectors.square().mean(dim=(-2, -1), keepdim=True) + epsilon)
    return vectors / scale


def _vector_soft_normalize(vectors: Tensor) -> Tensor:
    """``N(v) = v / sqrt(1 + <v^2>)``: a non-amplifying, 1-Lipschitz vector norm.

    The legacy ``v / sqrt(<v^2> + eps)`` has Jacobian ``I / sqrt(eps)`` at
    ``v = 0``.  Vector channels start at zero, so that map renormalizes a
    cutoff-gated vanishing message (proportional to ``chi(r) -> 0``) to O(1)
    weight, and the network becomes numerically discontinuous at r_c (review
    F6).  Here, with ``r = ||v||`` over ``n`` components, the Jacobian eigenvalues
    are ``(1 + r^2/n)^-3/2`` (radial) and ``(1 + r^2/n)^-1/2`` (tangential).  Both
    are at most one, and the Jacobian is exactly ``I`` at zero.  Large inputs
    still reach unit RMS.  The map scales every vector by one invariant scalar,
    so it is O(3)-equivariant.
    """

    return vectors / torch.sqrt(1.0 + vectors.square().mean(dim=(-2, -1), keepdim=True))


_VECTOR_NORMALIZATIONS = frozenset({"rms-v1", "soft-v2"})


class _BoundedMLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, output_dim),
            nn.Tanh(),
        )

    def forward(self, inputs: Tensor) -> Tensor:
        return self.net(inputs)


@dataclass(frozen=True)
class PeriodicTransformerConfig:
    input_scalar_dim: int
    scalar_dim: int = 192
    vector_dim: int = 48
    num_heads: int = 6
    num_layers: int = 6
    radial_basis_dim: int = 24
    cutoff: float = 6.0
    reference_length: float = 1.0
    self_weight_epsilon: float = 1.0e-4
    normalization_epsilon: float = 1.0e-8
    logit_bound: float = 12.0
    max_candidates_per_pair: int = 32_768
    max_edges: int = 500_000
    vector_normalization: str = "rms-v1"
    """``rms-v1`` (legacy, amplifies vanishing vectors) or ``soft-v2`` (1-Lipschitz)."""


class _PeriodicAttentionBlock(nn.Module):
    def __init__(self, config: PeriodicTransformerConfig) -> None:
        super().__init__()
        if config.scalar_dim % config.num_heads or config.vector_dim % config.num_heads:
            raise ValueError("scalar_dim and vector_dim must be divisible by num_heads")
        self.config = config
        self.scalar_head_dim = config.scalar_dim // config.num_heads
        self.vector_head_dim = config.vector_dim // config.num_heads
        self.scalar_norm = nn.LayerNorm(config.scalar_dim)
        self.query = nn.Linear(config.scalar_dim, config.scalar_dim, bias=False)
        self.key = nn.Linear(config.scalar_dim, config.scalar_dim, bias=False)
        self.scalar_value = nn.Linear(config.scalar_dim, config.scalar_dim, bias=False)
        self.scalar_self = nn.Linear(config.scalar_dim, config.scalar_dim, bias=False)
        self.vector_query = VectorLinear(config.vector_dim, config.num_heads)
        self.vector_key = VectorLinear(config.vector_dim, config.num_heads)
        self.vector_value = VectorLinear(config.vector_dim, config.vector_dim)
        self.vector_self = VectorLinear(config.vector_dim, config.vector_dim)
        self.vector_chiral = VectorLinear(config.vector_dim, config.vector_dim)
        self.vector_quadrupole = VectorLinear(config.vector_dim, config.vector_dim)

        radial = config.radial_basis_dim
        self.edge_bias = _BoundedMLP(radial, config.scalar_dim, config.num_heads)
        self.edge_scalar_value = _BoundedMLP(
            radial, config.scalar_dim, config.scalar_dim
        )
        self.direction_gate = _BoundedMLP(
            radial + config.scalar_dim, config.scalar_dim, config.vector_dim
        )
        self.chiral_logit_scale = nn.Parameter(
            torch.full((config.num_heads,), 0.1)
        )
        self.chiral_scalar_scale = nn.Parameter(
            torch.empty(config.num_heads, self.scalar_head_dim)
        )
        self.chiral_vector_scale = nn.Parameter(torch.empty(config.vector_dim))
        self.quadrupole_scale = nn.Parameter(torch.empty(config.vector_dim))
        nn.init.normal_(self.chiral_scalar_scale, std=0.03)
        nn.init.normal_(self.chiral_vector_scale, std=0.05)
        nn.init.normal_(self.quadrupole_scale, std=0.05)
        self.self_log_weight = nn.Parameter(torch.zeros(config.num_heads))

        self.scalar_output = nn.Linear(config.scalar_dim, config.scalar_dim)
        self.vector_output = VectorLinear(config.vector_dim, config.vector_dim)
        self.post_scalar_norm = nn.LayerNorm(config.scalar_dim)
        self.ff_vector_a = VectorLinear(config.vector_dim, config.vector_dim)
        self.ff_vector_b = VectorLinear(config.vector_dim, config.vector_dim)
        self.ff_vector_base = VectorLinear(config.vector_dim, config.vector_dim)
        self.ff_triple_a = VectorLinear(config.vector_dim, 1)
        self.ff_triple_b = VectorLinear(config.vector_dim, 1)
        self.ff_triple_c = VectorLinear(config.vector_dim, 1)
        self.scalar_ff = nn.Sequential(
            nn.Linear(config.scalar_dim + config.vector_dim + 1, 2 * config.scalar_dim),
            nn.SiLU(),
            nn.Linear(2 * config.scalar_dim, config.scalar_dim),
            nn.Tanh(),
        )
        self.vector_gate = nn.Linear(config.scalar_dim, config.vector_dim)
        self.vector_cross_gate = nn.Linear(config.scalar_dim, config.vector_dim)

    def _radial_features(self, squared_distance: Tensor) -> Tensor:
        normalized = squared_distance / (
            self.config.cutoff / self.config.reference_length
        ) ** 2
        centers = torch.linspace(
            0.0,
            1.0,
            self.config.radial_basis_dim,
            dtype=squared_distance.dtype,
            device=squared_distance.device,
        )
        width = float(max(self.config.radial_basis_dim - 1, 1))
        return torch.exp(-width * width * (normalized.unsqueeze(-1) - centers).square())

    def _normalize_vectors(self, vectors: Tensor) -> Tensor:
        if self.config.vector_normalization == "soft-v2":
            return _vector_soft_normalize(vectors)
        return _vector_rms_normalize(vectors, self.config.normalization_epsilon)

    @staticmethod
    def _scatter_edges(values: Tensor, destination: Tensor, atom_count: int) -> Tensor:
        output = values.new_zeros((atom_count, *values.shape[1:]))
        if destination.numel():
            output.index_add_(0, destination, values)
        return output

    def forward(
        self,
        scalars: Tensor,
        vectors: Tensor,
        edges: PeriodicEdges,
    ) -> tuple[Tensor, Tensor]:
        atom_count = scalars.shape[0]
        heads = self.config.num_heads
        scalar_head = self.scalar_head_dim
        vector_head = self.vector_head_dim
        scalar_input = self.scalar_norm(scalars)
        vector_input = self._normalize_vectors(vectors)

        query = self.query(scalar_input).reshape(atom_count, heads, scalar_head)
        key = self.key(scalar_input).reshape(atom_count, heads, scalar_head)
        scalar_value = self.scalar_value(scalar_input).reshape(
            atom_count, heads, scalar_head
        )
        self_scalar = self.scalar_self(scalar_input).reshape(
            atom_count, heads, scalar_head
        )
        vector_value = self.vector_value(vector_input).reshape(
            atom_count, heads, vector_head, 3
        )
        self_vector = self.vector_self(vector_input).reshape(
            atom_count, heads, vector_head, 3
        )

        destination = edges.destination
        source = edges.source
        displacement = edges.displacement / self.config.reference_length
        regular_features = regular_solid_harmonics(displacement, max_degree=2)
        regular_direction = regular_features[:, 1:4]
        distance_squared = edges.squared_distance / self.config.reference_length**2
        radial = self._radial_features(distance_squared)
        cutoff_squared = (self.config.cutoff / self.config.reference_length) ** 2
        cutoff = smooth_whole_edge_cutoff(distance_squared, cutoff_squared)

        if edges.num_edges:
            scalar_logit = (
                query[destination] * key[source]
            ).sum(dim=-1) / math.sqrt(scalar_head)
            vector_query = self.vector_query(vector_input)
            vector_key = self.vector_key(vector_input)
            oriented = oriented_triple_product(
                vector_query[destination],
                vector_key[source],
                regular_direction.unsqueeze(1),
            )
            raw_logit = (
                scalar_logit
                + self.edge_bias(radial)
                + torch.tanh(oriented) * self.chiral_logit_scale
            )
            bounded_logit = self.config.logit_bound * torch.tanh(
                raw_logit / self.config.logit_bound
            )
            edge_weight = cutoff.unsqueeze(-1) * torch.exp(bounded_logit)
        else:
            oriented = scalars.new_empty(0, heads)
            edge_weight = scalars.new_empty(0, heads)

        self_weight = (
            torch.nn.functional.softplus(self.self_log_weight)
            + self.config.self_weight_epsilon
        )
        denominator = self._scatter_edges(edge_weight, destination, atom_count)
        denominator = denominator + self_weight.unsqueeze(0)

        if edges.num_edges:
            edge_scalar = self.edge_scalar_value(radial).reshape(
                edges.num_edges, heads, scalar_head
            )
            chirality_scalar = (
                torch.tanh(oriented).unsqueeze(-1)
                * self.chiral_scalar_scale.unsqueeze(0)
            )
            scalar_message = scalar_value[source] + edge_scalar + chirality_scalar
            scalar_sum = self._scatter_edges(
                edge_weight.unsqueeze(-1) * scalar_message,
                destination,
                atom_count,
            )

            source_vector = vector_value[source]
            direction_input = torch.cat((radial, scalar_input[source]), dim=-1)
            direction_gate = self.direction_gate(direction_input).reshape(
                edges.num_edges, heads, vector_head, 1
            )
            direction = regular_direction[:, None, None, :]
            chiral_source = self.vector_chiral(vector_input).reshape(
                atom_count, heads, vector_head, 3
            )[source]
            chiral_value = torch.cross(
                chiral_source, direction.expand_as(chiral_source), dim=-1
            )
            quad_source = self.vector_quadrupole(vector_input).reshape(
                atom_count, heads, vector_head, 3
            )[source]
            dot = (quad_source * direction).sum(dim=-1, keepdim=True)
            quadrupole_value = dot * direction - (
                distance_squared[:, None, None, None] / 3.0
            ) * quad_source
            vector_message = (
                source_vector
                + direction_gate * direction
                + self.chiral_vector_scale.reshape(1, heads, vector_head, 1)
                * chiral_value
                + self.quadrupole_scale.reshape(1, heads, vector_head, 1)
                * quadrupole_value
            )
            vector_sum = self._scatter_edges(
                edge_weight[:, :, None, None] * vector_message,
                destination,
                atom_count,
            )
        else:
            scalar_sum = scalars.new_zeros(atom_count, heads, scalar_head)
            vector_sum = vectors.new_zeros(atom_count, heads, vector_head, 3)

        attended_scalar = (
            scalar_sum + self_weight[None, :, None] * self_scalar
        ) / denominator[:, :, None]
        attended_vector = (
            vector_sum + self_weight[None, :, None, None] * self_vector
        ) / denominator[:, :, None, None]
        scalars = scalars + self.scalar_output(
            attended_scalar.reshape(atom_count, self.config.scalar_dim)
        )
        vectors = vectors + self.vector_output(
            attended_vector.reshape(atom_count, self.config.vector_dim, 3)
        )

        normalized_scalar = self.post_scalar_norm(scalars)
        normalized_vector = self._normalize_vectors(vectors)
        vector_norms = torch.sqrt(
            normalized_vector.square().sum(dim=-1)
            + self.config.normalization_epsilon
        )
        triple = oriented_triple_product(
            self.ff_triple_a(normalized_vector).squeeze(-2),
            self.ff_triple_b(normalized_vector).squeeze(-2),
            self.ff_triple_c(normalized_vector).squeeze(-2),
        ).unsqueeze(-1)
        scalars = scalars + self.scalar_ff(
            torch.cat((normalized_scalar, vector_norms, torch.tanh(triple)), dim=-1)
        )
        base_vector = self.ff_vector_base(normalized_vector)
        cross_vector = torch.cross(
            self.ff_vector_a(normalized_vector),
            self.ff_vector_b(normalized_vector),
            dim=-1,
        )
        base_gate = torch.sigmoid(self.vector_gate(normalized_scalar)).unsqueeze(-1)
        cross_gate = torch.tanh(self.vector_cross_gate(normalized_scalar)).unsqueeze(-1)
        vectors = vectors + base_gate * base_vector + cross_gate * cross_vector
        return scalars, vectors


@dataclass(frozen=True)
class PeriodicTransformerOutput:
    scalar_channels: Tensor
    vector_channels: Tensor
    pooled_scalars: Tensor
    edges: PeriodicEdges


class PeriodicSE3Transformer(nn.Module):
    """Full noncausal periodic proper-SE(3)-equivariant Transformer.

    Inputs contain no atom-index embedding.  ``scalar_features`` must itself be
    invariant under Cartesian rotations, while optional ``vector_features``
    must be polar/SO(3) vector channels.  The returned vectors use row-vector
    convention and transform as ``v -> v Q`` for every ``Q in SO(3)``.
    """

    def __init__(
        self,
        config: PeriodicTransformerConfig,
        *,
        input_vector_dim: int = 0,
    ) -> None:
        super().__init__()
        if config.input_scalar_dim <= 0:
            raise ValueError("input_scalar_dim must be positive")
        if config.reference_length <= 0.0 or config.cutoff <= 0.0:
            raise ValueError("length scales must be positive")
        if config.max_candidates_per_pair < 1 or config.max_edges < 1:
            raise ValueError("periodic-image resource budgets must be positive")
        if config.num_layers < 1:
            raise ValueError("num_layers must be positive")
        if config.vector_normalization not in _VECTOR_NORMALIZATIONS:
            raise ValueError("vector_normalization must be 'rms-v1' or 'soft-v2'")
        self.config = config
        self.input_vector_dim = int(input_vector_dim)
        self.scalar_input = nn.Sequential(
            nn.Linear(config.input_scalar_dim, config.scalar_dim),
            nn.SiLU(),
            nn.Linear(config.scalar_dim, config.scalar_dim),
        )
        self.vector_input = (
            VectorLinear(input_vector_dim, config.vector_dim)
            if input_vector_dim > 0
            else None
        )
        self.blocks = nn.ModuleList(
            [_PeriodicAttentionBlock(config) for _ in range(config.num_layers)]
        )
        self.final_scalar_norm = nn.LayerNorm(config.scalar_dim)

    def forward(
        self,
        scalar_features: Tensor,
        fractional_coordinates: Tensor,
        lattice: Tensor,
        *,
        vector_features: Tensor | None = None,
        edges: PeriodicEdges | None = None,
        batch_index: Tensor | None = None,
        num_graphs: int | None = None,
    ) -> PeriodicTransformerOutput:
        if (
            scalar_features.ndim != 2
            or scalar_features.shape[-1] != self.config.input_scalar_dim
        ):
            raise ValueError("scalar_features must have shape [N,input_scalar_dim]")
        if fractional_coordinates.shape != (scalar_features.shape[0], 3):
            raise ValueError("fractional coordinates must have shape [N,3]")
        if batch_index is None:
            if num_graphs not in (None, 1):
                raise ValueError("num_graphs requires a packed batch_index")
        else:
            if edges is None:
                raise ValueError("packed Transformer evaluation requires explicit edges")
            if batch_index.shape != (scalar_features.shape[0],):
                raise ValueError("batch_index must have shape [N]")
            if batch_index.dtype != torch.long or batch_index.device != scalar_features.device:
                raise ValueError("batch_index must be a long tensor on the feature device")
            if isinstance(num_graphs, bool) or not isinstance(num_graphs, int):
                raise ValueError("packed evaluation requires an integer num_graphs")
            if num_graphs < 1 or bool(
                ((batch_index < 0) | (batch_index >= num_graphs)).any().item()
            ):
                raise ValueError("batch_index lies outside num_graphs")
            if edges.num_edges and bool(
                (batch_index[edges.destination] != batch_index[edges.source]).any().item()
            ):
                raise ValueError("packed periodic edges cannot connect different crystals")
        if edges is None:
            edges = enumerate_periodic_edges(
                fractional_coordinates,
                lattice,
                self.config.cutoff,
                max_candidates_per_pair=self.config.max_candidates_per_pair,
                max_edges=self.config.max_edges,
            )
        scalars = self.scalar_input(scalar_features)
        if self.vector_input is None:
            if vector_features is not None:
                raise ValueError("model was constructed without input vector channels")
            vectors = scalar_features.new_zeros(
                scalar_features.shape[0], self.config.vector_dim, 3
            )
        else:
            if vector_features is None or vector_features.shape != (
                scalar_features.shape[0], self.input_vector_dim, 3
            ):
                raise ValueError("vector_features have the wrong channel shape")
            vectors = self.vector_input(vector_features)
        for block in self.blocks:
            scalars, vectors = block(scalars, vectors, edges)
        scalars = self.final_scalar_norm(scalars)
        if batch_index is None:
            pooled = scalars.mean(dim=0)
        else:
            if num_graphs is None:  # guarded above; keeps the type checker precise
                raise AssertionError("packed graph count disappeared")
            pooled = scalars.new_zeros(num_graphs, scalars.shape[-1])
            pooled.index_add_(0, batch_index, scalars)
            counts = torch.bincount(batch_index, minlength=num_graphs).to(
                dtype=scalars.dtype
            )
            if bool((counts == 0).any().item()):
                raise ValueError("every packed crystal must contain at least one atom")
            pooled = pooled / counts.unsqueeze(-1)
        return PeriodicTransformerOutput(
            scalar_channels=scalars,
            vector_channels=vectors,
            pooled_scalars=pooled,
            edges=edges,
        )
