"""Certified periodic-image geometry in the normative row-vector convention."""

from __future__ import annotations

import math
from collections.abc import Iterator
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from .errors import DomainError, NumericalError, ResourceLimitError

FloatArray = npt.NDArray[np.float64]
DEFAULT_MAX_SEARCH_POINTS = 1_000_000


def _as_vector3(value: npt.ArrayLike, *, name: str) -> FloatArray:
    array = np.asarray(value, dtype=np.float64)
    if array.shape != (3,) or not np.all(np.isfinite(array)):
        raise DomainError(f"{name} must be a finite vector of shape (3,)")
    return array


def _as_lattice(value: npt.ArrayLike) -> FloatArray:
    lattice = np.asarray(value, dtype=np.float64)
    if lattice.shape != (3, 3) or not np.all(np.isfinite(lattice)):
        raise DomainError("lattice must be a finite matrix of shape (3, 3)")
    with np.errstate(over="ignore", invalid="ignore", under="ignore"):
        determinant = float(np.linalg.det(lattice))
    if not math.isfinite(determinant):
        raise NumericalError("lattice determinant is not representable in float64")
    if determinant <= 0.0:
        raise DomainError("lattice must be right-handed with positive determinant")
    return lattice


def _squared_cutoff(cutoff: float) -> float:
    if not math.isfinite(cutoff) or cutoff <= 0.0:
        raise DomainError("cutoff must be finite and positive")
    squared = float(cutoff) * float(cutoff)
    if not math.isfinite(squared) or squared == 0.0:
        raise NumericalError("squared cutoff must be positive and finite in float64")
    return squared


def minimum_singular_value(lattice: npt.ArrayLike) -> float:
    """Return the smallest singular value of a right-handed lattice matrix."""

    values = np.linalg.svd(_as_lattice(lattice), compute_uv=False)
    sigma_min = float(values[-1])
    if not math.isfinite(sigma_min) or sigma_min <= 0.0:
        raise NumericalError("lattice is numerically singular")
    return sigma_min


def _integer_interval(center: float, radius: float) -> range:
    """Integers strictly inside ``(center-radius, center+radius)`` plus endpoints.

    The inclusive integer box is deliberately conservative; the exact Cartesian
    norm test removes boundary and false-positive entries.
    """

    lower = math.floor(center - radius)
    upper = math.ceil(center + radius)
    return range(lower, upper + 1)


@dataclass(frozen=True, slots=True)
class PeriodicImage:
    """One lattice image satisfying a Cartesian cutoff."""

    shift: tuple[int, int, int]
    fractional_displacement: FloatArray
    cartesian_displacement: FloatArray
    squared_distance: float


def images_within_cutoff(
    delta_fractional: npt.ArrayLike,
    lattice: npt.ArrayLike,
    cutoff: float,
    *,
    include_boundary: bool = False,
    max_search_points: int = DEFAULT_MAX_SEARCH_POINTS,
) -> tuple[PeriodicImage, ...]:
    """Enumerate every ``n`` with ``||(delta+n)L||`` inside the cutoff.

    Completeness follows from

    ``sigma_min(L) ||delta+n|| <= ||(delta+n)L||``.

    Consequently any accepted image lies inside a finite fractional Euclidean
    ball of radius ``cutoff / sigma_min(L)``; enumerating its enclosing integer
    box cannot miss an image, including for strongly skew cells.

    Integral translations are removed before forming that box, then composed
    into the returned Python-integer shifts. This avoids loss of neighboring
    images when float64 cannot distinguish adjacent integers at a large center.
    The budget caps candidate box points, not accepted images. Exceeding it
    raises ResourceLimitError before enumeration; no truncated list is returned.
    """

    if type(max_search_points) is not int or max_search_points <= 0:
        raise DomainError("max_search_points must be a positive integer")
    delta = _as_vector3(delta_fractional, name="delta_fractional")
    cell = _as_lattice(lattice)
    cutoff2 = _squared_cutoff(cutoff)
    translation = tuple(-int(np.rint(value)) for value in delta)
    reduced = delta + np.asarray(translation, dtype=np.float64)
    radius = cutoff / minimum_singular_value(cell)
    if not math.isfinite(radius):
        raise NumericalError("periodic enumeration radius is not finite")
    comparator = (
        (lambda value: value <= cutoff2)
        if include_boundary
        else (lambda value: value < cutoff2)
    )
    result: list[PeriodicImage] = []
    ranges = tuple(_integer_interval(-float(reduced[j]), radius) for j in range(3))
    # Do not use len(range): enormous valid Python ranges can exceed sys.maxsize.
    search_points = math.prod(interval.stop - interval.start for interval in ranges)
    if search_points > max_search_points:
        raise ResourceLimitError(
            f"periodic search requires {search_points} box points, exceeding "
            f"max_search_points={max_search_points}"
        )
    for n0 in ranges[0]:
        for n1 in ranges[1]:
            for n2 in ranges[2]:
                local_shift = (n0, n1, n2)
                shift = tuple(a + b for a, b in zip(translation, local_shift, strict=True))
                fractional = reduced + np.asarray(local_shift, dtype=np.float64)
                cartesian = fractional @ cell
                squared = float(cartesian @ cartesian)
                if comparator(squared):
                    result.append(
                        PeriodicImage(
                            shift=shift,
                            fractional_displacement=fractional,
                            cartesian_displacement=cartesian,
                            squared_distance=squared,
                        )
                    )
    result.sort(key=lambda image: (image.squared_distance, image.shift))
    return tuple(result)


def closest_periodic_image(
    delta_fractional: npt.ArrayLike,
    lattice: npt.ArrayLike,
    *,
    max_search_points: int = DEFAULT_MAX_SEARCH_POINTS,
) -> PeriodicImage:
    """Solve the three-dimensional closest-vector problem exhaustively.

    Componentwise wrapping is used only to obtain an initial upper bound; it is
    not assumed to be the closest image in a skew metric.
    """

    if type(max_search_points) is not int or max_search_points <= 0:
        raise DomainError("max_search_points must be a positive integer")
    delta = _as_vector3(delta_fractional, name="delta_fractional")
    cell = _as_lattice(lattice)
    # Lattice translations are mathematical integers, not bounded int64s.
    # Search around the reduced displacement to avoid enormous integer-box
    # centers and cancellation when the supplied coordinates are unwrapped.
    initial_shift = tuple(-int(np.rint(value)) for value in delta)
    initial_fractional = delta + np.asarray(initial_shift, dtype=np.float64)
    initial_cartesian = initial_fractional @ cell
    initial_distance = float(np.linalg.norm(initial_cartesian))
    # A zero-distance collision has already solved the CVP.
    if initial_distance == 0.0:
        return PeriodicImage(initial_shift, initial_fractional, initial_cartesian, 0.0)
    candidates = images_within_cutoff(
        initial_fractional,
        cell,
        math.nextafter(initial_distance, math.inf),
        include_boundary=True,
        max_search_points=max_search_points,
    )
    if not candidates:  # pragma: no cover - initial candidate must be present
        raise NumericalError("closest-image enumeration returned an empty set")
    closest = candidates[0]
    shift = tuple(a + b for a, b in zip(initial_shift, closest.shift, strict=True))
    return PeriodicImage(
        shift,
        closest.fractional_displacement,
        closest.cartesian_displacement,
        closest.squared_distance,
    )


def periodic_pairs_within(
    source: npt.ArrayLike,
    target: npt.ArrayLike,
    lattice: npt.ArrayLike,
    tolerance: float,
) -> tuple[npt.NDArray[np.bool_], FloatArray]:
    """Which ``(s, t)`` pairs have a periodic image within ``tolerance``, and how close.

    Returns ``within[i, j]`` and ``distance[i, j]``.  ``within`` is exactly the
    exhaustive answer ``|closest image of s_i - t_j| <= tolerance``, and
    ``distance`` is the closest-image distance wherever ``within`` holds (up to
    floating-point rounding of the image coordinates).  Elsewhere ``distance``
    is only an upper bound.

    Proof: if ``|(f + n) L| <= tol`` then ``|f_j + n_j| <= tol sqrt((G^-1)_jj)``,
    so when that bound is below one half componentwise rounding returns this
    ``n``.  Every other image differs by a nonzero lattice vector ``m L`` with
    ``|m L| >= sigma_min(L) |m| >= sigma_min(L)``; if ``sigma_min(L) > 2 tol`` it
    lies farther than ``tol``, so the rounded image is the closest one.  When
    either condition fails the exhaustive closest-vector search is used.
    """

    cell = _as_lattice(lattice)
    first = np.asarray(source, dtype=np.float64).reshape(-1, 3)
    second = np.asarray(target, dtype=np.float64).reshape(-1, 3)
    limit = float(tolerance)
    if not math.isfinite(limit) or limit < 0.0:
        raise DomainError("tolerance must be finite and nonnegative")
    reach = limit * float(np.sqrt(np.max(np.diag(np.linalg.inv(cell @ cell.T)))))
    if reach < 0.5 and minimum_singular_value(cell) > 2.0 * limit:
        delta = first[:, None, :] - second[None, :, :]
        delta -= np.rint(delta)
        cartesian = delta @ cell
        distance = np.sqrt(np.einsum("ijk,ijk->ij", cartesian, cartesian))
    else:
        distance = np.empty((len(first), len(second)), dtype=np.float64)
        for i, point in enumerate(first):
            for j, other in enumerate(second):
                image = closest_periodic_image(point - other, cell)
                distance[i, j] = math.sqrt(image.squared_distance)
    return distance <= limit, distance


def minimum_periodic_pair_distance(
    fractional: np.ndarray, lattice: np.ndarray, *, upper_bound: float | None = None
) -> float:
    """Exact ``min_{i<j}`` closest-periodic-image distance, vectorized.

    ``U`` is any upper bound on the answer: the supplied bound or the best
    componentwise-rounded image.  An improving image ``(d + n) L`` with reduced
    ``|d_j| <= 1/2`` has ``|d_j + n_j| <= U sqrt((G^-1)_jj)``, so every ``n``
    with ``|n_j| <= U sqrt((G^-1)_jj) + 1/2`` is enumerated: the minimum over
    that integer box is the minimum over all pairs and images, exact up to
    floating-point rounding of the image coordinates.
    """

    cell = np.asarray(lattice, dtype=np.float64)
    points = np.asarray(fractional, dtype=np.float64)
    left, right = np.tril_indices(len(points), -1)
    delta = points[left] - points[right]
    delta -= np.rint(delta)
    rounded = delta @ cell
    bound = float(np.sqrt(np.min(np.einsum("ij,ij->i", rounded, rounded))))
    if upper_bound is not None:
        bound = min(bound, float(upper_bound))
    reach = bound * np.sqrt(np.diag(np.linalg.inv(cell @ cell.T)))
    ranges = [np.arange(-int(np.ceil(r + 0.5)), int(np.ceil(r + 0.5)) + 1) for r in reach]
    shifts = np.stack(np.meshgrid(*ranges, indexing="ij"), axis=-1).reshape(-1, 3)
    best = np.inf
    for start in range(0, len(delta), 4096):
        block = delta[start : start + 4096]
        cartesian = (block[:, None, :] + shifts[None, :, :]) @ cell
        best = min(best, float(np.min(np.einsum("ijk,ijk->ij", cartesian, cartesian))))
    return float(np.sqrt(best))


def shortest_nonzero_lattice_image(
    lattice: npt.ArrayLike, *, max_search_points: int = DEFAULT_MAX_SEARCH_POINTS
) -> PeriodicImage:
    """Return the shortest nonzero periodic self image of a lattice site.

    The shortest input lattice-row norm is a finite upper bound.  The same
    certified image enumerator used for pair geometry then solves the lattice
    shortest-vector problem in three dimensions; the zero shift is discarded.
    """

    cell = _as_lattice(lattice)
    upper = float(np.min(np.linalg.norm(cell, axis=1)))
    candidates = images_within_cutoff(
        np.zeros(3, dtype=np.float64),
        cell,
        math.nextafter(upper, math.inf),
        include_boundary=True,
        max_search_points=max_search_points,
    )
    for image in candidates:
        if image.shift != (0, 0, 0):
            return image
    raise NumericalError("shortest-vector enumeration returned no nonzero image")


def smooth_cutoff(squared_distance: npt.ArrayLike, cutoff: float) -> npt.NDArray[np.float64]:
    """Compact nonnegative ``C²`` cutoff used to gate whole attention edges.

    ``chi(s) = (1 - s/r_c²)^3`` for ``0 <= s < r_c²`` and zero otherwise.
    Both the value and first derivative vanish at the dynamic-list boundary.
    """

    cutoff2 = _squared_cutoff(cutoff)
    squared = np.asarray(squared_distance, dtype=np.float64)
    if np.any(~np.isfinite(squared)) or np.any(squared < 0.0):
        raise DomainError("squared_distance must be finite and nonnegative")
    # Distances outside support need no division, which could overflow for a
    # small but representable cutoff even though the correct result is zero.
    normalized = np.ones_like(squared)
    np.divide(squared, cutoff2, out=normalized, where=squared < cutoff2)
    return np.where(normalized < 1.0, np.maximum(1.0 - normalized, 0.0) ** 3, 0.0)


def enumerate_pair_images(
    fractional_coordinates: npt.ArrayLike,
    lattice: npt.ArrayLike,
    cutoff: float,
    *,
    max_search_points: int = DEFAULT_MAX_SEARCH_POINTS,
) -> Iterator[tuple[int, int, PeriodicImage]]:
    """Yield all ordered atom-pair images, excluding only ``(a=a,n=0)``.

    Coordinates and cell are copied when iteration first starts. Later caller
    mutations cannot mix different geometries within this neighbor sequence.
    This is not synchronization against concurrent writes during the copies.
    """

    if type(max_search_points) is not int or max_search_points <= 0:
        raise DomainError("max_search_points must be a positive integer")
    coordinates = np.array(fractional_coordinates, dtype=np.float64, copy=True)
    if coordinates.ndim != 2 or coordinates.shape[1] != 3:
        raise DomainError("fractional_coordinates must have shape (N, 3)")
    if np.any(~np.isfinite(coordinates)):
        raise DomainError("fractional_coordinates must be finite")
    # Validate the full input before yielding any edge, including N=0. This
    # prevents malformed later atoms from producing a plausible partial list.
    cell = _as_lattice(np.array(lattice, dtype=np.float64, copy=True))
    _squared_cutoff(cutoff)
    for target, f_target in enumerate(coordinates):
        for source, f_source in enumerate(coordinates):
            for image in images_within_cutoff(
                f_source - f_target, cell, cutoff, max_search_points=max_search_points
            ):
                if target == source and image.shift == (0, 0, 0):
                    continue
                yield target, source, image


def exhaustive_cube_images(
    delta_fractional: npt.ArrayLike,
    lattice: npt.ArrayLike,
    cutoff: float,
    half_width: int,
) -> set[tuple[int, int, int]]:
    """Independent bounded oracle for tests, never used by production sampling."""

    if half_width < 0:
        raise DomainError("half_width must be nonnegative")
    delta = _as_vector3(delta_fractional, name="delta_fractional")
    cell = _as_lattice(lattice)
    cutoff2 = cutoff * cutoff
    accepted: set[tuple[int, int, int]] = set()
    for n0 in range(-half_width, half_width + 1):
        for n1 in range(-half_width, half_width + 1):
            for n2 in range(-half_width, half_width + 1):
                shift = (n0, n1, n2)
                cartesian = (delta + np.asarray(shift, dtype=np.float64)) @ cell
                if float(cartesian @ cartesian) < cutoff2:
                    accepted.add(shift)
    return accepted
