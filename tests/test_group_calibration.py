"""Per-group calibration of the cover selector's group head (review m1)."""

from __future__ import annotations

import math

import numpy as np
import pytest

from crystalatlas.diagnostics import per_group_calibration
from crystalatlas.errors import DomainError


def test_per_group_calibration_matches_hand_computation() -> None:
    probabilities = np.array(
        [[0.7, 0.2, 0.1], [0.6, 0.3, 0.1], [0.1, 0.8, 0.1], [0.5, 0.4, 0.1]]
    )
    observed = [0, 0, 1, 1]
    report = per_group_calibration(probabilities, observed, group_numbers=(225, 194, 1))
    assert report[225]["observed"] == 2
    assert report[225]["observed_frequency"] == pytest.approx(0.5)
    assert report[225]["mean_predicted"] == pytest.approx((0.7 + 0.6 + 0.1 + 0.5) / 4)
    expected_nll = -(math.log(0.7) + math.log(0.6)) / 2
    assert report[225]["nll_when_observed"] == pytest.approx(expected_nll)
    assert report[1]["observed"] == 0 and report[1]["nll_when_observed"] is None
    assert report[1]["mean_predicted"] == pytest.approx(0.1)


def test_per_group_calibration_rejects_improper_forecasts() -> None:
    with pytest.raises(DomainError):
        per_group_calibration(np.array([[0.5, 0.6]]), [0], group_numbers=(1, 2))
    with pytest.raises(DomainError):
        per_group_calibration(np.array([[0.5, 0.5]]), [2], group_numbers=(1, 2))
