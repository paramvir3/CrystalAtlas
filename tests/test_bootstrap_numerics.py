"""Deterministic checks of paired system-level percentile resampling."""

import numpy as np
import pytest

from crystalatlas.errors import DomainError
from crystalatlas.evaluation import paired_cluster_bootstrap


@pytest.mark.parametrize("batch_size", [1, 7, 256, 2000])
def test_matches_dense_reference_including_partial_batch(batch_size):
    values = np.array([-3.0, -1.0, 0.0, 2.0, 4.0, 5.0, 9.0])
    indices = np.random.default_rng(42).integers(0, len(values), size=(1001, len(values)))
    estimates = values[indices].mean(axis=1)
    expected = (float(values.mean()), *np.quantile(estimates, [0.05, 0.95]))
    actual = paired_cluster_bootstrap(
        values, samples=1001, confidence=0.9, seed=42, batch_size=batch_size
    )
    assert actual == pytest.approx(expected, abs=1e-14, rel=0)


@pytest.mark.parametrize(
    "parameter,value",
    [
        ("samples", True),
        ("samples", 100.5),
        ("samples", 99),
        ("batch_size", 0),
        ("batch_size", True),
        ("batch_size", 1.5),
        ("seed", -1),
        ("seed", True),
        ("seed", 1.5),
        ("confidence", float("nan")),
        ("confidence", 1.0),
    ],
)
def test_invalid_configuration(parameter, value):
    with pytest.raises(DomainError):
        paired_cluster_bootstrap([0.0, 1.0], **{parameter: value})


def test_constant_system_effect_and_equal_system_weighting():
    assert paired_cluster_bootstrap([2.0, 2.0, 2.0], samples=100) == (2.0, 2.0, 2.0)
    assert paired_cluster_bootstrap([-1.0, 3.0], samples=100)[0] == 1.0


def test_overflow_is_not_reported_as_confidence_interval():
    with pytest.raises(DomainError, match="overflowed"):
        paired_cluster_bootstrap([1e308, 1e308], samples=100)


def test_resampling_workspace_is_batched(monkeypatch):
    generator = np.random.default_rng(9)
    requested_shapes = []

    class RecordingGenerator:
        def integers(self, low, high, *, size):
            requested_shapes.append(size)
            assert size[0] <= 32
            return generator.integers(low, high, size=size)

    monkeypatch.setattr(np.random, "default_rng", lambda seed: RecordingGenerator())
    paired_cluster_bootstrap([-1.0, 0.0, 2.0], samples=101, batch_size=32, seed=9)
    assert requested_shapes == [(32, 3), (32, 3), (32, 3), (5, 3)]


def test_reversing_comparison_reverses_interval_and_preserves_inputs():
    values = np.array([-7.0, -1.0, 2.0, 3.0, 8.0])
    original = values.copy()
    mean, lower, upper = paired_cluster_bootstrap(values, samples=1001, seed=91)
    reversed_result = paired_cluster_bootstrap(-values, samples=1001, seed=91)
    assert reversed_result == pytest.approx((-mean, -upper, -lower), abs=1e-13, rel=0)
    np.testing.assert_array_equal(values, original)


def test_translation_and_positive_unit_scaling():
    values = np.array([-2.0, 0.0, 1.0, 4.0])
    baseline = np.array(paired_cluster_bootstrap(values, samples=1001, seed=31))
    transformed = paired_cluster_bootstrap(2.0 * values + 3.0, samples=1001, seed=31)
    np.testing.assert_allclose(transformed, 2.0 * baseline + 3.0, atol=1e-13, rtol=0)


@pytest.mark.parametrize("values", [[], [1.0], [[1.0, 2.0]], [0.0, np.nan], [0.0, np.inf]])
def test_invalid_system_vector(values):
    with pytest.raises(DomainError, match="finite paired system differences"):
        paired_cluster_bootstrap(values)
