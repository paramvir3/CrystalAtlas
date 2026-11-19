"""Metric-aware periodic radial pair scalars (review finding M2).

The legacy pair scalars are lattice-independent Fourier modes, so attention never
sees a Cartesian distance directly.  ``radial_gaussian_widths`` adds
``rho_w(i, j) = sum_n s(|r_ij + n L|) exp(-|r_ij + n L|^2 / 2 w^2)``.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from crystalatlas.nn.invariant_transformer import (  # noqa: E402
    InvariantPeriodicTransformer,
    InvariantPeriodicTransformerConfig,
)
from crystalatlas.periodic import images_within_cutoff  # noqa: E402

pytestmark = pytest.mark.ml

_WIDTHS = (0.5, 1.0)


def _transformer(widths=_WIDTHS, max_images=4096, max_chunks=64):  # type: ignore[no-untyped-def]
    config = InvariantPeriodicTransformerConfig(
        input_scalar_dim=4,
        model_dim=8,
        pair_dim=4,
        num_heads=1,
        num_layers=1,
        output_vector_dim=2,
        radial_gaussian_widths=widths,
        max_radial_images=max_images,
        max_radial_image_chunks=max_chunks,
    )
    return InvariantPeriodicTransformer(config).double()


def _features(transformer, fractional, lattice):  # type: ignore[no-untyped-def]
    f = torch.as_tensor(fractional, dtype=torch.float64)[None]
    lat = torch.as_tensor(lattice, dtype=torch.float64)[None]
    delta = f[:, None, :, :] - f[:, :, None, :]
    return transformer._radial_gaussian_features(delta, lat)[0].numpy()


def _brute_force(fractional, lattice, widths):  # type: ignore[no-untyped-def]
    radius = 6.0 * max(widths)
    count = len(fractional)
    result = np.zeros((count, count, len(widths)))
    for i in range(count):
        for j in range(count):
            for image in images_within_cutoff(fractional[j] - fractional[i], lattice, radius):
                squared = image.squared_distance
                cutoff = (1.0 - squared / radius**2) ** 2
                for k, width in enumerate(widths):
                    result[i, j, k] += cutoff * math.exp(-squared / (2.0 * width * width))
    return result


def test_radial_features_equal_the_certified_image_sum_in_a_skew_cell() -> None:
    lattice = np.array([[3.2, 0.0, 0.0], [2.6, 2.1, 0.0], [0.7, 0.9, 3.4]])
    fractional = np.random.default_rng(2).random((4, 3))
    np.testing.assert_allclose(
        _features(_transformer(), fractional, lattice),
        _brute_force(fractional, lattice, _WIDTHS),
        rtol=1e-11,
        atol=1e-13,
    )


def test_radial_features_are_rotation_and_basis_invariant() -> None:
    transformer = _transformer()
    lattice = np.array([[3.2, 0.0, 0.0], [0.6, 3.1, 0.0], [0.7, 0.9, 3.4]])
    fractional = np.random.default_rng(3).random((5, 3))
    reference = _features(transformer, fractional, lattice)
    q, r = np.linalg.qr(np.random.default_rng(4).standard_normal((3, 3)))
    q = q @ np.diag(np.sign(np.diag(r)))
    if np.linalg.det(q) < 0:
        q[:, 0] *= -1
    np.testing.assert_allclose(
        _features(transformer, fractional, lattice @ q), reference, rtol=1e-11, atol=1e-13
    )
    unimodular = np.array([[1, 1, 0], [0, 1, 0], [1, 0, 1]], dtype=float)
    assert round(np.linalg.det(unimodular)) == 1
    changed = fractional @ np.linalg.inv(unimodular)
    np.testing.assert_allclose(
        _features(transformer, changed, unimodular @ lattice), reference, rtol=1e-10, atol=1e-12
    )


def test_radial_features_resolve_short_contacts_in_a_fifteen_angstrom_cell() -> None:
    transformer = _transformer(widths=(0.5,))
    lattice = 15.0 * np.eye(3)
    close = np.array([[0.0, 0.0, 0.0], [0.8 / 15.0, 0.0, 0.0]])
    far = np.array([[0.0, 0.0, 0.0], [2.5 / 15.0, 0.0, 0.0]])
    near_value = _features(transformer, close, lattice)[0, 1, 0]
    far_value = _features(transformer, far, lattice)[0, 1, 0]
    cutoff = lambda d: (1.0 - d * d / 9.0) ** 2  # noqa: E731
    assert near_value == pytest.approx(cutoff(0.8) * math.exp(-0.8**2 / 0.5), rel=1e-12)
    assert far_value == pytest.approx(cutoff(2.5) * math.exp(-2.5**2 / 0.5), rel=1e-12)
    assert near_value > 1e4 * far_value


def test_radial_features_are_resource_bounded_and_widths_validated() -> None:
    with pytest.raises(ValueError, match="max_radial_images"):
        _features(_transformer(max_images=8, max_chunks=1), np.zeros((2, 3)), np.eye(3))
    with pytest.raises(ValueError, match="radial_gaussian_widths"):
        _transformer(widths=(0.0,))
    for bad in (0, -1, 1.5, True):
        with pytest.raises(ValueError, match="max_radial_image"):
            _transformer(max_images=bad)
        with pytest.raises(ValueError, match="max_radial_image_chunks"):
            _transformer(max_chunks=bad)


def test_small_cells_are_summed_in_bounded_chunks_exactly() -> None:
    """A cell far below the cutoff needs thousands of images; none may be refused.

    The CFM base reaches cells near the Loewner floor (0.75 A), where a 2 A width
    (cutoff 12 A) needs about 19^3 images.  Chunks of 128 images give the
    certified image sum, and so does a single chunk.
    """

    rng = np.random.default_rng(3)
    fractional = rng.random((3, 3))
    lattice = np.array([[1.3, 0.0, 0.0], [0.2, 1.4, 0.0], [0.1, -0.3, 1.2]])
    widths = (0.5, 1.0, 2.0)
    expected = _brute_force(fractional, lattice, widths)
    chunked = _features(
        _transformer(widths, max_images=128, max_chunks=4096), fractional, lattice
    )
    whole = _features(
        _transformer(widths, max_images=60_000, max_chunks=1), fractional, lattice
    )
    np.testing.assert_allclose(chunked, expected, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(whole, expected, rtol=1e-12, atol=1e-12)
    assert np.isfinite(_features(_transformer(widths), fractional, lattice)).all()


def test_radial_features_train_end_to_end_and_leave_the_legacy_config_unchanged() -> None:
    from dataclasses import replace
    from fractions import Fraction

    from test_learned import _tiny_config

    from crystalatlas.catalogue import MINIMAL_CATALOGUE
    from crystalatlas.decoder import CrystalScaffold, OrbitAssignment
    from crystalatlas.geometry import ContinuousCover
    from crystalatlas.learned import CrystalAtlasModel, LearnedCrystalGenerator

    base = _tiny_config()
    assert "geometry_radial_gaussian_widths" not in base.to_dict()["model"]
    config = replace(base, model=replace(base.model, geometry_radial_gaussian_widths=(0.5,)))
    assert config.fingerprint != base.fingerprint
    torch.manual_seed(31)
    model = CrystalAtlasModel(config, catalogue=MINIMAL_CATALOGUE).double()
    scaffold = CrystalScaffold.from_mapping(
        nu=2,
        space_group="P-1",
        reduced_composition={8: 1},
        orbits=(OrbitAssignment("2i", 8, (Fraction(1, 7), Fraction(2, 7), Fraction(3, 7))),),
    )
    layout = model.layout(ContinuousCover.from_exact_scaffold(scaffold))
    result = LearnedCrystalGenerator(model, allow_untrained=True).geometry_cfm_objective(
        scaffold,
        layout.chart.decode_latent(np.zeros(layout.lattice_dimension)),
        time=torch.tensor(0.37, dtype=torch.float64),
        sigma_z=0.05,
        sigma_eta=0.03,
        generator=torch.Generator().manual_seed(101),
    )
    result.loss.total.backward()
    first = model.geometry_field.transformer.pair_input[0].weight.grad
    assert first is not None and bool(torch.isfinite(first).all().item())
