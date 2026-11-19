"""``AffineOperation.apply_float`` returns half-open ``[0, 1)`` coordinates (review minor).

``np.mod`` of a tiny negative float rounds to exactly ``1.0``; only that endpoint
is equivalent to ``0`` on the half-open torus, so it is mapped to ``0.0``
exactly as ``FloatingStructure`` does.  Every representable remainder below one
is preserved.
"""

from __future__ import annotations

from fractions import Fraction

import numpy as np

from crystalatlas.algebra import AffineOperation


def test_apply_float_maps_the_rounded_endpoint_to_zero() -> None:
    identity = AffineOperation(
        ((1, 0, 0), (0, 1, 0), (0, 0, 1)), (Fraction(0), Fraction(0), Fraction(0))
    )
    result = identity.apply_float([-1e-20, -1e-17, 0.25])
    assert np.all(result >= 0.0) and np.all(result < 1.0)
    assert result[0] == 0.0 and result[1] == 0.0 and result[2] == 0.25
    largest_below_one = np.nextafter(1.0, 0.0)
    assert identity.apply_float([largest_below_one, 0.0, 0.0])[0] == largest_below_one
