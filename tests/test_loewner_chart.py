"""Loewner-interval lattice chart (review B2 support redesign).

The support is the open matrix interval  l G0 < G < u G0  inside the
point-group-invariant SPD cone: every lattice direction's length lies within
[sqrt(l), sqrt(u)] of the reference.  The latent map is

    W(Z) = (u e^Z + c l I)(e^Z + c I)^-1,  c = (u - 1)/(1 - l),  G = S W S,

with Z in the commutant.  W is a monotone scalar function of the symmetric
matrix Z, so it maps eigenvalues of e^Z in (0, inf) onto (l, u), preserves the
commutant (exact point-group invariance), fixes W(0) = I, and certifies
sigma_min(L) >= sqrt(lambda_min(G0) l), independently of u.
"""

from __future__ import annotations

import numpy as np
import pytest

from crystalatlas.errors import DomainError
from crystalatlas.full_catalogue import build_space_group_setting
from crystalatlas.lattice import InvariantSPDChart, LoewnerSPDChart, scale_metric_to_volume

_SETTINGS = {
    name: build_space_group_setting(number)
    for name, number in (
        ("triclinic", 1),
        ("monoclinic", 12),
        ("hexagonal", 194),
        ("trigonal", 166),
        ("cubic", 225),
    )
}
LOWER, UPPER = float(np.exp(-2.0)), float(np.exp(4.5))


def _chart(name: str, lower: float = LOWER, upper: float = UPPER) -> LoewnerSPDChart:
    setting = _SETTINGS[name]
    reference = scale_metric_to_volume(np.asarray(setting.reference_metric, dtype=float), 80.0)
    rotations = [np.asarray(r, dtype=float) for r in setting.point_group_rotations]
    return LoewnerSPDChart(rotations, reference_metric=reference, lower=lower, upper=upper)


@pytest.mark.parametrize("name", list(_SETTINGS))
def test_origin_is_the_reference_and_latents_round_trip(name: str) -> None:
    chart = _chart(name)
    np.testing.assert_allclose(
        chart.decode_latent(np.zeros(chart.dimension)), chart.reference_metric, rtol=1e-14
    )
    generator = np.random.default_rng(3)
    for scale in (0.1, 1.0, 4.0):
        z = scale * generator.standard_normal(chart.dimension)
        metric = chart.decode_latent(z)
        assert chart.is_invariant(metric)
        np.testing.assert_allclose(chart.encode_latent(metric), z, rtol=0.0, atol=1e-9)
        b = chart.bounded_coefficients(z)
        scale = float(np.abs(metric).max())
        np.testing.assert_allclose(chart.decode(b), metric, rtol=1e-12, atol=1e-13 * scale)
        np.testing.assert_allclose(chart.unbounded_latent(b), z, rtol=0.0, atol=1e-9)


@pytest.mark.parametrize("name", list(_SETTINGS))
def test_every_latent_lies_strictly_inside_the_loewner_interval(name: str) -> None:
    chart = _chart(name)
    generator = np.random.default_rng(5)
    floor = chart.minimum_lattice_singular_value_bound
    ceiling = chart.maximum_lattice_singular_value_bound
    for scale in (0.5, 5.0, 50.0, 500.0):
        for _ in range(20):
            metric = chart.decode_latent(scale * generator.standard_normal(chart.dimension))
            whitened = chart.inverse_sqrt_reference @ metric @ chart.inverse_sqrt_reference
            eigenvalues = np.linalg.eigvalsh((whitened + whitened.T) / 2.0)
            assert eigenvalues[0] >= LOWER * (1.0 - 1e-12)
            assert eigenvalues[-1] <= UPPER * (1.0 + 1e-12)
            singular = np.sqrt(np.linalg.eigvalsh(metric))
            assert singular[0] >= floor * (1.0 - 1e-12)
            assert singular[-1] <= ceiling * (1.0 + 1e-12)


def test_floor_certificate_is_independent_of_the_upper_bound() -> None:
    small, large = _chart("hexagonal", upper=float(np.exp(1.0))), _chart("hexagonal")
    assert (
        small.minimum_lattice_singular_value_bound == large.minimum_lattice_singular_value_bound
    )
    reference_floor = float(np.sqrt(np.linalg.eigvalsh(large.reference_metric)[0]))
    assert large.minimum_lattice_singular_value_bound == pytest.approx(
        reference_floor * np.sqrt(LOWER), rel=1e-15
    )


def test_metrics_outside_the_open_interval_are_refused() -> None:
    chart = _chart("cubic")
    with pytest.raises(DomainError, match="Loewner interval"):
        chart.encode_latent(chart.reference_metric * UPPER * 1.01)
    with pytest.raises(DomainError, match="Loewner interval"):
        chart.encode_latent(chart.reference_metric * LOWER * 0.99)
    with pytest.raises(DomainError, match="Loewner interval"):
        chart.unbounded_latent(np.array([np.sqrt(3.0) * 5.0]))  # A = 5 I exceeds log(u)


def test_the_legacy_ball_is_contained_in_a_box_of_equal_half_widths() -> None:
    """||A||_2 <= ||A||_F < B implies -B < lambda(A) < B: the box extends the ball."""

    setting = _SETTINGS["monoclinic"]
    reference = scale_metric_to_volume(np.asarray(setting.reference_metric, dtype=float), 80.0)
    rotations = [np.asarray(r, dtype=float) for r in setting.point_group_rotations]
    ball = InvariantSPDChart(rotations, reference_metric=reference, bound=2.0)
    box = LoewnerSPDChart(
        rotations,
        reference_metric=reference,
        lower=float(np.exp(-2.0)),
        upper=float(np.exp(2.0)),
    )
    generator = np.random.default_rng(7)
    for _ in range(50):
        metric = ball.decode_latent(3.0 * generator.standard_normal(ball.dimension))
        box.encode_latent(metric)  # never raises


def test_interval_parameters_are_validated() -> None:
    for lower, upper in ((1.0, 2.0), (0.5, 1.0), (0.0, 2.0), (np.nan, 2.0), (0.5, np.inf)):
        with pytest.raises(DomainError):
            _chart("cubic", lower=lower, upper=upper)


# --- domain option and cover layouts -----------------------------------------------


def _catalogue():  # type: ignore[no-untyped-def]
    from crystalatlas.catalogue import (
        FIXTURE_EXPECTED_ROWS,
        MINIMAL_CATALOGUE,
        CertifiedCatalogue,
    )

    settings = {key: MINIMAL_CATALOGUE.get(key) for key in ("P1", "P-1")}
    rows = dict(FIXTURE_EXPECTED_ROWS)
    for number in (166, 194, 225):
        setting = build_space_group_setting(number)
        settings[setting.key] = setting
        rows[setting.key] = frozenset(setting.wyckoff_positions)
    return CertifiedCatalogue(settings, rows, certification_scope="Loewner layout test subset")


def _layout(cover, domain):  # type: ignore[no-untyped-def]
    from crystalatlas.geometry import IntrinsicCoverLayout

    return IntrinsicCoverLayout(cover, catalogue=_catalogue(), **domain.layout_options())


def _covers():  # type: ignore[no-untyped-def]
    from crystalatlas.composition import Composition
    from crystalatlas.geometry import ContinuousCover, OrbitType

    carbon = Composition.from_counts({6: 1})
    return {
        "P1": ContinuousCover(1, "P1", carbon, (OrbitType("1a", 6),)),
        "P6_3/mmc": ContinuousCover(2, "P6_3/mmc", carbon, (OrbitType("2b", 6),)),
        "R-3m": ContinuousCover(3, "R-3m", carbon, (OrbitType("3a", 6),)),
        "Fm-3m": ContinuousCover(4, "Fm-3m", carbon, (OrbitType("4a", 6),)),
    }


def test_lattice_chart_is_a_versioned_domain_option() -> None:
    from crystalatlas.config import DomainConfig

    legacy = DomainConfig(lattice_chart="ball-v1")
    assert "lattice_chart" not in legacy.identity_payload()
    assert "loewner_upper_log_eigenvalue" not in legacy.identity_payload()
    loewner = DomainConfig(lattice_chart="loewner-v2")
    assert DomainConfig() == loewner  # the default since 2026-10-20
    payload = loewner.identity_payload()
    assert payload["lattice_chart"] == "loewner-v2"
    assert payload["loewner_upper_log_eigenvalue"] == 4.5
    for bad in (dict(lattice_chart="ellipsoid"), dict(loewner_upper_log_eigenvalue=0.0)):
        with pytest.raises(DomainError):
            DomainConfig(**bad)


def test_parameters_of_the_inactive_chart_must_stay_at_their_defaults() -> None:
    """A parameter that has no effect under the selected chart is refused.

    Under loewner-v2 (the default) ``metric_radius`` is inert, and under ball-v1
    the Loewner bounds are inert and left out of the identity payload.  Setting
    either silently would change nothing while looking like a setting.
    """

    from crystalatlas.config import DomainConfig
    from crystalatlas.geometry import IntrinsicCoverLayout

    with pytest.raises(DomainError, match="metric_radius"):
        DomainConfig(metric_radius=1.0)
    with pytest.raises(DomainError, match="loewner_upper_log_eigenvalue"):
        DomainConfig(lattice_chart="ball-v1", loewner_upper_log_eigenvalue=3.0)
    with pytest.raises(DomainError, match="loewner_control_log_volume_window"):
        DomainConfig(lattice_chart="ball-v1", loewner_control_log_volume_window=(-1.0, 1.0))
    assert DomainConfig(lattice_chart="ball-v1", metric_radius=1.0).metric_radius == 1.0
    assert DomainConfig(loewner_upper_log_eigenvalue=4.0).loewner_upper_log_eigenvalue == 4.0
    from dataclasses import fields

    from crystalatlas.geometry import INACTIVE_CHART_PARAMETER_DEFAULTS

    domain_defaults = {item.name: item.default for item in fields(DomainConfig)}
    assert INACTIVE_CHART_PARAMETER_DEFAULTS["loewner-v2"] == {
        "metric_bound": domain_defaults["metric_radius"]
    }
    assert INACTIVE_CHART_PARAMETER_DEFAULTS["ball-v1"] == {
        name: domain_defaults[name]
        for name in ("loewner_upper_log_eigenvalue", "loewner_control_log_volume_window")
    }
    cover = _covers()["P1"]
    with pytest.raises(DomainError, match="metric_bound"):
        IntrinsicCoverLayout(cover, catalogue=_catalogue(), metric_bound=1.0)
    with pytest.raises(DomainError, match="loewner_upper_log_eigenvalue"):
        IntrinsicCoverLayout(
            cover,
            catalogue=_catalogue(),
            lattice_chart="ball-v1",
            loewner_upper_log_eigenvalue=3.0,
        )
    with pytest.raises(DomainError, match="loewner_control_log_volume_window"):
        IntrinsicCoverLayout(
            cover,
            catalogue=_catalogue(),
            lattice_chart="ball-v1",
            loewner_control_log_volume_window=(-1.0, 1.0),
        )


def test_loewner_chart_island_ignores_enclosing_neural_autocast(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    """The Loewner spectral map runs in float64 even under a bfloat16 autocast."""

    from crystalatlas.config import DomainConfig
    from crystalatlas.nn.bridge import TorchCoverGeometry

    bridge = TorchCoverGeometry(_layout(_covers()["P1"], DomainConfig())).float()
    z = torch.tensor([0.1] * bridge.lattice_dimension, dtype=torch.float32, requires_grad=True)
    eta = torch.zeros(bridge.intrinsic_dimension, dtype=torch.float32)
    observed = []
    real_eigh = torch.linalg.eigh

    def checked_eigh(value):  # type: ignore[no-untyped-def]
        observed.append(value.dtype)
        return real_eigh(value)

    monkeypatch.setattr(torch.linalg, "eigh", checked_eigh)
    baseline = bridge(z, eta)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        actual = bridge(z, eta)
    assert observed == [torch.float64, torch.float64]
    for name in ("bounded_lattice_coordinates", "metric", "lattice"):
        assert getattr(actual, name).dtype == torch.float32
        torch.testing.assert_close(
            getattr(actual, name), getattr(baseline, name), rtol=0, atol=0
        )


@pytest.mark.parametrize("key", ["P1", "P6_3/mmc", "R-3m", "Fm-3m"])
def test_loewner_layout_certifies_exactly_the_declared_shortest_vector_floor(key: str) -> None:
    from crystalatlas.config import DomainConfig

    domain = DomainConfig(lattice_chart="loewner-v2")
    chart = _layout(_covers()[key], domain).chart
    assert isinstance(chart, LoewnerSPDChart)
    floor = domain.minimum_lattice_singular_value_angstrom
    assert floor <= chart.minimum_lattice_singular_value_bound <= floor * (1.0 + 1e-12)
    assert chart.upper == pytest.approx(np.exp(4.5), rel=1e-15)


def test_layered_lattice_refused_by_the_ball_is_inside_the_loewner_support() -> None:
    """A graphite-like 2b cell at c/a = 5: anisotropic, yet no short axis."""

    from crystalatlas.config import DomainConfig

    cover = _covers()["P6_3/mmc"]
    a, c = 2.46, 5.0 * 2.46
    lattice = np.array([[a, 0.0, 0.0], [-a / 2.0, a * np.sqrt(3.0) / 2.0, 0.0], [0.0, 0.0, c]])
    metric = lattice @ lattice.T
    with pytest.raises(DomainError, match="ball"):
        _layout(cover, DomainConfig(lattice_chart="ball-v1")).chart.encode_latent(metric)
    chart = _layout(cover, DomainConfig(lattice_chart="loewner-v2")).chart
    latent = chart.encode_latent(metric)
    np.testing.assert_allclose(
        chart.decode_latent(latent), metric, rtol=1e-12, atol=1e-12 * c * c
    )


# --- torch twin (flow bridge) ---------------------------------------------------------

torch = pytest.importorskip("torch")


@pytest.mark.parametrize("name", list(_SETTINGS))
def test_torch_loewner_map_matches_the_exact_chart(name: str) -> None:
    from crystalatlas.nn.bridge import loewner_whitened_metric

    chart = _chart(name)
    basis = torch.tensor(np.stack(chart.basis), dtype=torch.float64)
    generator = np.random.default_rng(11)
    for scale in (0.0, 0.7, 3.0):
        z = scale * generator.standard_normal(chart.dimension)
        algebra = torch.einsum("i,ijk->jk", torch.tensor(z), basis)
        whitened, logarithm = loewner_whitened_metric(algebra, chart.lower, chart.upper)
        sqrt_reference = torch.tensor(chart.sqrt_reference)
        np.testing.assert_allclose(
            (sqrt_reference @ whitened @ sqrt_reference).numpy(),
            chart.decode_latent(z),
            rtol=1e-12,
            atol=1e-12 * float(np.abs(chart.reference_metric).max()),
        )
        coefficients = torch.einsum("jk,ijk->i", logarithm, basis).numpy()
        np.testing.assert_allclose(coefficients, chart.bounded_coefficients(z), atol=1e-12)


@pytest.mark.parametrize(
    "spectrum",
    [(0.3, -1.1, 2.0), (0.4, 0.4, 0.4), (-0.7, -0.7, 1.5), (40.0, -35.0, 0.0)],
    ids=["distinct", "cubic-degenerate", "uniaxial-degenerate", "saturated"],
)
def test_daleckii_krein_backward_passes_gradcheck(spectrum) -> None:  # type: ignore[no-untyped-def]
    from crystalatlas.nn.bridge import loewner_whitened_metric

    rotation = torch.linalg.qr(
        torch.randn(3, 3, generator=torch.Generator().manual_seed(2), dtype=torch.float64)
    ).Q
    algebra = rotation @ torch.diag(torch.tensor(spectrum, dtype=torch.float64)) @ rotation.T
    algebra.requires_grad_(True)
    assert torch.autograd.gradcheck(
        lambda value: loewner_whitened_metric(value, LOWER, UPPER),
        (algebra,),
        eps=1e-6,
        atol=1e-7,
        rtol=1e-6,
    )


@pytest.mark.parametrize("key", ["P1", "P6_3/mmc", "Fm-3m"])
def test_bridge_with_loewner_chart_matches_numpy_and_pulls_back_the_metric(key: str) -> None:
    from crystalatlas.config import DomainConfig
    from crystalatlas.nn.bridge import TorchCoverGeometry

    layout = _layout(_covers()[key], DomainConfig(lattice_chart="loewner-v2"))
    bridge = TorchCoverGeometry(layout).double()
    chart = layout.chart
    z = np.linspace(-0.8, 1.3, chart.dimension)
    eta = np.zeros(layout.intrinsic_dimension)
    expanded = bridge(torch.tensor(z), torch.tensor(eta))
    np.testing.assert_allclose(
        expanded.metric.numpy(),
        chart.decode_latent(z),
        rtol=1e-12,
        atol=1e-12 * float(np.abs(chart.reference_metric).max()),
    )
    np.testing.assert_allclose(
        expanded.bounded_lattice_coordinates.numpy(), chart.bounded_coefficients(z), atol=1e-12
    )
    pullback = bridge.lattice_pullback_metric(torch.tensor(z)).detach().numpy()
    step = 1e-6
    metric = chart.decode_latent(z)
    inverse = np.linalg.inv(metric)
    derivatives = []
    for index in range(chart.dimension):
        shift = np.zeros(chart.dimension)
        shift[index] = step
        derivatives.append(
            (chart.decode_latent(z + shift) - chart.decode_latent(z - shift)) / (2.0 * step)
        )
    expected = np.array(
        [[np.trace(inverse @ a @ inverse @ b) for b in derivatives] for a in derivatives]
    )
    np.testing.assert_allclose(pullback, expected, rtol=1e-6, atol=1e-9)
    assert np.all(np.linalg.eigvalsh(pullback) > 0.0)


def test_loewner_bridge_refuses_second_derivatives_instead_of_dropping_them() -> None:
    """The Daleckii--Krein adjoint is first-order only.

    torch's ``once_differentiable`` raises only when the incoming gradient
    requires grad.  A ``create_graph`` Jacobian (``lattice_pullback_metric(...,
    create_graph=True)``) instead differentiates through the saved spectrum,
    and that dependence would be dropped silently, giving zero second
    derivatives.  The map must refuse.  First-order paths, which are the only
    ones production uses (the pullback weight is detached), stay exact.
    """

    from crystalatlas.config import DomainConfig
    from crystalatlas.nn.bridge import TorchCoverGeometry

    layout = _layout(_covers()["P1"], DomainConfig(lattice_chart="loewner-v2"))
    bridge = TorchCoverGeometry(layout).double()
    z = torch.linspace(-0.4, 0.7, layout.chart.dimension, dtype=torch.float64)
    with pytest.raises(RuntimeError, match="first derivatives only"):
        bridge.lattice_pullback_metric(z.clone().requires_grad_(True), create_graph=True)
    probe = z.clone().requires_grad_(True)
    metric = bridge._metric_from_latent(probe)[1]
    (gradient,) = torch.autograd.grad(metric.sum(), probe, create_graph=False)
    assert torch.all(torch.isfinite(gradient))
    assert torch.autograd.gradcheck(
        lambda value: bridge._metric_from_latent(value)[1]
        / float(np.linalg.norm(layout.chart.reference_metric)),
        (probe,),
        eps=1e-6,
        atol=2e-7,
        rtol=1e-5,
    )


# --- uniform law over the support (controls and reference law) ----------------------


def test_uniform_coefficients_fill_the_support_exactly() -> None:
    """Batched rejection from the cube |b_j| <= sqrt(3) max(|ln l|, ln u) is exact.

    For the cubic chart, A = b I/sqrt(3), so b is uniform on
    (sqrt(3) ln l, sqrt(3) ln u): a one-sample Kolmogorov--Smirnov check.
    """

    chart = _chart("cubic")
    rng = np.random.default_rng(13)
    samples = np.array([chart.uniform_coefficients(rng)[0] for _ in range(4000)])
    low, high = np.sqrt(3.0) * np.log(LOWER), np.sqrt(3.0) * np.log(UPPER)
    assert samples.min() > low and samples.max() < high
    ordered = np.sort((samples - low) / (high - low))
    statistic = np.max(np.abs(ordered - (np.arange(1, 4001) - 0.5) / 4000))
    assert statistic < 1.63 / np.sqrt(4000)  # KS critical value at alpha = 0.01
    triclinic = _chart("triclinic")
    for _ in range(20):
        b = triclinic.uniform_coefficients(rng)
        spectrum = np.linalg.eigvalsh(triclinic.algebra_matrix(b))
        assert np.log(LOWER) < spectrum[0] and spectrum[-1] < np.log(UPPER)


def test_log_volume_window_restricts_the_uniform_law_exactly() -> None:
    """The window intersects the support with the slab lo < tr(A)/2 < hi.

    tr(A)/2 = ln(V/V0) is the Hencky volumetric strain against the reference.
    The intersection of the support and the slab is still inside the proposal
    cube, so the first accepted proposal is exactly uniform on it.  For the
    cubic chart A = b I/sqrt(3), so the volumetric strain is uniform on
    (max(lo, 1.5 ln l), min(hi, 1.5 ln u)).  A window wider than the support
    leaves every draw unchanged.
    """

    window = (-1.5, 2.0)
    chart = _chart("cubic")
    rng = np.random.default_rng(29)
    volumetric = np.array(
        [
            np.trace(
                chart.algebra_matrix(chart.uniform_coefficients(rng, log_volume_window=window))
            )
            / 2.0
            for _ in range(4000)
        ]
    )
    low = max(window[0], 1.5 * np.log(LOWER))
    high = min(window[1], 1.5 * np.log(UPPER))
    assert volumetric.min() > low and volumetric.max() < high
    ordered = np.sort((volumetric - low) / (high - low))
    statistic = np.max(np.abs(ordered - (np.arange(1, 4001) - 0.5) / 4000))
    assert statistic < 1.63 / np.sqrt(4000)  # KS critical value at alpha = 0.01
    triclinic = _chart("triclinic")
    for _ in range(50):
        a = triclinic.algebra_matrix(
            triclinic.uniform_coefficients(rng, log_volume_window=window)
        )
        spectrum = np.linalg.eigvalsh(a)
        assert np.log(LOWER) < spectrum[0] and spectrum[-1] < np.log(UPPER)
        assert window[0] < np.trace(a) / 2.0 < window[1]
    wide = (1.5 * np.log(LOWER) - 1.0, 1.5 * np.log(UPPER) + 1.0)
    for name in ("cubic", "triclinic"):
        unrestricted, restricted = np.random.default_rng(31), np.random.default_rng(31)
        for _ in range(20):
            np.testing.assert_array_equal(
                _chart(name).uniform_coefficients(unrestricted),
                _chart(name).uniform_coefficients(restricted, log_volume_window=wide),
            )


@pytest.mark.parametrize(
    "window", [(0.0, 2.0), (-1.5, 0.0), (1.0, 2.0), (-1.5, float("nan")), (-1.5, float("inf"))]
)
def test_log_volume_window_must_contain_the_reference_volume(
    window: tuple[float, float],
) -> None:
    with pytest.raises(DomainError, match="log_volume_window"):
        _chart("cubic").uniform_coefficients(np.random.default_rng(0), log_volume_window=window)


def test_control_log_volume_window_is_a_validated_loewner_domain_option() -> None:
    from crystalatlas.config import DomainConfig

    legacy = DomainConfig(lattice_chart="ball-v1")
    assert "loewner_control_log_volume_window" not in legacy.identity_payload()
    loewner = DomainConfig(lattice_chart="loewner-v2")
    assert loewner.loewner_control_log_volume_window == (-1.5, 2.0)
    assert loewner.identity_payload()["loewner_control_log_volume_window"] == (-1.5, 2.0)
    assert loewner.layout_options()["loewner_control_log_volume_window"] == (-1.5, 2.0)
    from_json = DomainConfig(
        lattice_chart="loewner-v2", loewner_control_log_volume_window=[-1, 3]  # type: ignore[arg-type]
    )
    assert from_json.loewner_control_log_volume_window == (-1.0, 3.0)
    for bad in ((0.0, 2.0), (-1.5, 0.0), (-1.5, float("inf")), (-1.5,), "ab", (True, 2.0)):
        with pytest.raises(DomainError, match="loewner_control_log_volume_window"):
            DomainConfig(lattice_chart="loewner-v2", loewner_control_log_volume_window=bad)


def test_uniform_control_and_reference_law_respect_the_log_volume_window() -> None:
    """Under loewner-v2 the untrained laws keep physical densities (AIRSS-style).

    Without the window about 55% of the uniform Loewner law for P6_3/mmc lies
    outside (-1.5, 2.0) (`volumetric_window_validation.json`), so 200 draws
    discriminate.
    """

    from crystalatlas.config import CrystalAtlasConfig, DomainConfig
    from crystalatlas.controls import uniform_geometry_in_cover
    from crystalatlas.reference import ReferenceGenerator, ReferenceLawConfig

    domain = DomainConfig(lattice_chart="loewner-v2")
    lo, hi = domain.loewner_control_log_volume_window
    layout = _layout(_covers()["P6_3/mmc"], domain)
    reference_volume = np.sqrt(np.linalg.det(layout.chart.reference_metric))
    rng = np.random.default_rng(37)
    for _ in range(200):
        latent = uniform_geometry_in_cover(layout, rng).lattice_latent
        metric = layout.chart.decode_latent(latent)
        assert lo < np.log(np.sqrt(np.linalg.det(metric)) / reference_volume) < hi
    config = CrystalAtlasConfig(
        domain=DomainConfig(
            element_vocabulary=(1,),
            dng_element_vocabulary=(1,),
            max_distinct_elements=1,
            lattice_chart="loewner-v2",
        )
    )
    generator = ReferenceGenerator(config, ReferenceLawConfig(space_groups=("P1",)))
    setting = generator.catalogue.get("P1")
    reference = scale_metric_to_volume(np.asarray(setting.reference_metric, float), 40.0)
    rng = np.random.default_rng(41)
    for _ in range(100):
        _, metric = generator.geometry_sampler._sample_metric(setting, 2, rng)
        ratio = np.sqrt(np.linalg.det(metric) / np.linalg.det(reference))
        assert lo < np.log(ratio) < hi


def test_uniform_control_and_reference_law_respect_the_loewner_support() -> None:
    from crystalatlas.config import CrystalAtlasConfig, DomainConfig
    from crystalatlas.controls import uniform_geometry_in_cover
    from crystalatlas.reference import ReferenceGenerator, ReferenceLawConfig

    domain = DomainConfig(lattice_chart="loewner-v2")
    layout = _layout(_covers()["P6_3/mmc"], domain)
    rng = np.random.default_rng(17)
    for _ in range(10):
        geometry = uniform_geometry_in_cover(layout, rng)
        layout.chart.decode_latent(geometry.lattice_latent)  # inside the support
    config = CrystalAtlasConfig(
        domain=DomainConfig(
            element_vocabulary=(1,),
            dng_element_vocabulary=(1,),
            max_distinct_elements=1,
            lattice_chart="loewner-v2",
        )
    )
    generator = ReferenceGenerator(config, ReferenceLawConfig(space_groups=("P1",)))
    setting = generator.catalogue.get("P1")
    coefficients, metric = generator.geometry_sampler._sample_metric(
        setting, 2, np.random.default_rng(19)
    )
    reference = scale_metric_to_volume(np.asarray(setting.reference_metric, float), 40.0)
    whitened = (
        np.linalg.inv(np.linalg.cholesky(reference))
        @ metric
        @ np.linalg.inv(np.linalg.cholesky(reference)).T
    )
    singular = np.sqrt(np.linalg.eigvalsh(metric))
    assert singular[0] >= 0.75
    assert np.all(np.isfinite(coefficients)) and np.all(np.isfinite(whitened))


def test_model_trains_and_generates_on_the_loewner_support() -> None:
    from dataclasses import replace
    from fractions import Fraction

    from test_learned import _tiny_config

    from crystalatlas.catalogue import MINIMAL_CATALOGUE
    from crystalatlas.composition import Composition
    from crystalatlas.decoder import CrystalScaffold, OrbitAssignment
    from crystalatlas.geometry import ContinuousCover
    from crystalatlas.learned import CrystalAtlasModel, LearnedCrystalGenerator
    from crystalatlas.pipeline import GenerationMode, GenerationRequest

    base = _tiny_config()
    # The fixture is the radius-1 ball; its radius is inert under loewner-v2.
    config = replace(
        base, domain=replace(base.domain, lattice_chart="loewner-v2", metric_radius=2.0)
    )
    assert config.fingerprint != base.fingerprint
    torch.manual_seed(43)
    model = CrystalAtlasModel(config, catalogue=MINIMAL_CATALOGUE).double()
    generator = LearnedCrystalGenerator(model, allow_untrained=True)
    traces = generator.sample(
        GenerationRequest(
            mode=GenerationMode.CSP,
            composition=Composition.from_counts({6: 1, 8: 1}),
            samples=3,
            seed=7,
        )
    )
    assert all(trace.succeeded for trace in traces)
    for trace in traces:
        chart = model.layout(trace.sample.cover_sample.cover).chart
        assert isinstance(chart, LoewnerSPDChart)
        singular = np.sqrt(np.linalg.eigvalsh(trace.sample.decoded.metric))
        assert singular[0] >= config.domain.minimum_lattice_singular_value_angstrom
    scaffold = CrystalScaffold.from_mapping(
        nu=2,
        space_group="P-1",
        reduced_composition={8: 1},
        orbits=(OrbitAssignment("2i", 8, (Fraction(1, 7), Fraction(2, 7), Fraction(3, 7))),),
    )
    layout = model.layout(ContinuousCover.from_exact_scaffold(scaffold))
    metric = layout.chart.decode_latent(np.full(layout.lattice_dimension, 0.2))
    single = generator.geometry_cfm_objective(
        scaffold,
        metric,
        time=torch.tensor(0.4, dtype=torch.float64),
        sigma_z=0.05,
        sigma_eta=0.03,
        generator=torch.Generator().manual_seed(3),
    )
    padded = generator.geometry_cfm_objective_padded(
        (scaffold, scaffold),
        (metric, metric),
        time=torch.tensor([0.3, 0.8], dtype=torch.float64),
        sigma_z=0.05,
        sigma_eta=0.03,
        generator=torch.Generator().manual_seed(5),
    )
    total = single.loss.total + padded.loss.total
    assert torch.isfinite(total)
    total.backward()
    gradients = [p.grad for p in model.geometry_field.parameters() if p.grad is not None]
    assert gradients and all(bool(torch.isfinite(g).all()) for g in gradients)


@pytest.mark.parametrize("key", ["P1", "P6_3/mmc", "R-3m", "Fm-3m"])
def test_analytic_loewner_pullback_equals_the_autograd_pullback(key: str) -> None:
    """H_ij = sum_ab Gamma_ab^2 M_i,ab M_j,ab / (w_a w_b), M_i = Q^T B_i Q.

    Gamma is the Daleckii--Krein divided-difference matrix of the spectral map,
    and the trace is computed in whitened space (G = S W S).  The closed form
    replaces one reverse-mode pass per metric entry.  The origin (all
    eigenvalues equal) and a saturating latent check the repeated-eigenvalue
    and log-cosh branches.
    """

    from crystalatlas.config import DomainConfig
    from crystalatlas.nn.bridge import TorchCoverGeometry

    layout = _layout(_covers()[key], DomainConfig(lattice_chart="loewner-v2"))
    bridge = TorchCoverGeometry(layout).double()
    dimension = layout.chart.dimension
    for z in (
        np.zeros(dimension),
        np.linspace(-0.8, 1.3, dimension),
        np.full(dimension, 3.0),
        np.linspace(4.0, -6.0, dimension),
    ):
        latent = torch.tensor(z, dtype=torch.float64)
        analytic = bridge.lattice_pullback_metric(latent)
        reference = bridge._lattice_pullback_metric_autograd(latent)
        scale = float(reference.abs().max())
        torch.testing.assert_close(analytic, reference, rtol=1e-9, atol=1e-12 * scale)
    single = bridge.lattice_pullback_metric(torch.zeros(dimension, dtype=torch.float32))
    assert single.dtype == torch.float32
