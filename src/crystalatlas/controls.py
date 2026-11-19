"""Untrained control: uniform geometry inside the true cover (review m3).

Given the true discrete cover of a target, the control draws

* lattice coefficients ``b`` uniformly in the open chart ball of radius ``B``
  (radius ``B U^{1/d}`` times a uniform direction), mapped to the latent by the
  chart's exact inverse of ``kappa``.  Under ``loewner-v2`` instead, ``b`` is
  uniform on the Loewner support intersected with the domain's volumetric
  window ``lo < ln(V/V0) < hi``; and
* intrinsic coordinates ``eta = u P`` with ``u ~ U[0, 1)^d`` and ``P`` the
  period basis, which is the Haar-uniform law on the parameter torus,

then decodes through the production layout.  Every reported Match@K must beat
this control on the same targets and attempt budget, because it uses the
oracle cover with no learned geometry at all.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .errors import CrystalAtlasError, DomainError
from .geometry import ContinuousDecodedCrystal, ContinuousGeometry, IntrinsicCoverLayout
from .lattice import LoewnerSPDChart


def uniform_geometry_in_cover(
    layout: IntrinsicCoverLayout, rng: np.random.Generator
) -> ContinuousGeometry:
    """One draw of the uniform control on ``layout``'s chart ball and torus."""

    dimension = layout.lattice_dimension
    if isinstance(layout.chart, LoewnerSPDChart):
        # Uniform in log-metric coefficients over the Loewner support (review B2),
        # within the domain's physical volumetric window.
        latent = layout.chart.unbounded_latent(
            layout.chart.uniform_coefficients(
                rng, log_volume_window=layout.control_log_volume_window
            )
        )
        unit = rng.random(layout.intrinsic_dimension)
        eta = unit @ layout.period_basis if layout.intrinsic_dimension else unit
        return ContinuousGeometry(latent, eta)
    bound = layout.chart.bound
    if bound is None:
        raise DomainError("the uniform control needs a bounded lattice chart")
    direction = rng.standard_normal(dimension)
    norm = float(np.linalg.norm(direction))
    if not np.isfinite(norm) or norm == 0.0:
        raise DomainError("could not draw a finite chart direction")
    radius = min(bound * float(rng.random()) ** (1.0 / dimension), np.nextafter(bound, 0.0))
    coefficients = radius * direction / norm
    latent = layout.chart.unbounded_latent(coefficients)
    unit = rng.random(layout.intrinsic_dimension)
    eta = unit @ layout.period_basis if layout.intrinsic_dimension else unit
    return ContinuousGeometry(latent, eta)


@dataclass(frozen=True, slots=True)
class ControlAttempt:
    geometry: ContinuousGeometry
    decoded: ContinuousDecodedCrystal | None
    error: str | None


def uniform_cover_control_attempts(
    layout: IntrinsicCoverLayout,
    *,
    samples: int,
    seed: int,
    collision_tolerance: float = 1.0e-8,
    symmetry_tolerance: float = 1.0e-8,
) -> tuple[ControlAttempt, ...]:
    """``samples`` control attempts; decode failures keep their ranked slot."""

    if isinstance(samples, bool) or not isinstance(samples, int) or samples < 1:
        raise DomainError("samples must be a positive integer")
    rng = np.random.default_rng(seed)
    attempts = []
    for _ in range(samples):
        geometry = uniform_geometry_in_cover(layout, rng)
        try:
            decoded = layout.decode(
                geometry,
                collision_tolerance=collision_tolerance,
                symmetry_tolerance=symmetry_tolerance,
            )
            attempts.append(ControlAttempt(geometry, decoded, None))
        except CrystalAtlasError as exc:
            attempts.append(ControlAttempt(geometry, None, f"{type(exc).__name__}: {exc}"))
    return tuple(attempts)


__all__ = ["ControlAttempt", "uniform_cover_control_attempts", "uniform_geometry_in_cover"]
