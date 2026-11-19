"""Fused mixed-topology geometry sampling (``geometry_sampler="mixed-v1"``).

The bucket sampler integrates one topology at a time, so a CSP request whose
attempts land on ``M`` distinct covers costs ``M`` Runge--Kutta solves of small
batch.  The fused sampler pads every attempt to one state ``(z[B, Z],
eta[B, E])``, gives row ``b`` its own lattice chart and its own torus
``R^k / (Z^k B_b)`` (the period basis completed by the identity), and reads the
field once per stage through ``forward_padded``.  Every member draws its base
point from its own generator exactly as before, so the two samplers are the
same map and agree to floating-point rounding.  Torus coordinates are compared
modulo the period lattice, because a wrap at a cell face may round to either
side.
"""

from __future__ import annotations

import math
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from test_learned import _manifest, _provenance, _tiny_config  # noqa: E402
from test_padded_batching import (  # noqa: E402
    _mixed_scaffolds,
    mixed_model,  # noqa: F401 - pytest fixture
)
from test_primitive_expansion import _catalogue as _centred_catalogue  # noqa: E402
from test_primitive_expansion import _scaffolds as _centred_scaffolds  # noqa: E402

from crystalatlas.catalogue import MINIMAL_CATALOGUE  # noqa: E402
from crystalatlas.composition import Composition  # noqa: E402
from crystalatlas.errors import (  # noqa: E402
    DomainError,
    InvariantViolation,
    ResourceLimitError,
)
from crystalatlas.flow import (  # noqa: E402
    FlowState,
    canonicalize_torus,
    canonicalize_torus_validated,
    integrate_rk4,
    integrate_with_endpoint_estimate,
    shortest_coordinate_lift,
    validate_period_basis_rows,
)
from crystalatlas.geometry import ContinuousCover, IntrinsicCoverLayout  # noqa: E402
from crystalatlas.learned import (  # noqa: E402
    GEOMETRY_SAMPLERS,
    CrystalAtlasModel,
    LearnedCrystalGenerator,
    _layout_batch_signature,
)
from crystalatlas.nn.bridge import PaddedCoverPlan, TorchCoverGeometry  # noqa: E402
from crystalatlas.nn.field import TangentRankError  # noqa: E402
from crystalatlas.pipeline import GenerationMode, GenerationRequest  # noqa: E402
from crystalatlas.training import save_checkpoint  # noqa: E402

pytestmark = pytest.mark.ml

F64 = torch.float64
F32 = torch.float32
CPU = torch.device("cpu")
SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
# ``torch.matrix_exp`` evaluates one matrix of 1-norm in about (3.4e-4, 5.0e-2) with
# a degree-8 polynomial whose relative error is about 8e-4 * norm**5, up to 2.4e-10
# (measured at float64, torch 2.12.0).  A batch of two or more matrices takes the
# degree-18 branch, which is accurate to rounding.  The ``ball-v1`` chart therefore
# agrees between an unbatched and a batched evaluation to this bound, not to
# rounding.  ``loewner-v2`` uses a symmetric eigendecomposition and is unaffected.
BALL_CHART_RELATIVE_GAP = 2.5e-10


def _torus_gap(actual: torch.Tensor, expected: torch.Tensor, period: torch.Tensor) -> float:
    """Largest coordinate of ``expected - actual`` modulo the period lattice."""

    if actual.numel() == 0:
        return 0.0
    return float(shortest_coordinate_lift(actual, expected, period).abs().max())


# --- row-wise tori ------------------------------------------------------------

_ROWS = ((2, 3), (1, 1), (3, 0), (1, 2))  # (d_z, d_eta) of each row


def _row_bases() -> list[torch.Tensor]:
    generator = torch.Generator().manual_seed(3)
    return [
        torch.eye(k, dtype=F64) + 0.3 * torch.randn(k, k, generator=generator, dtype=F64)
        for _, k in _ROWS
    ]


def _identity_padded(bases: list[torch.Tensor]) -> torch.Tensor:
    width = max(len(basis) for basis in bases)
    padded = torch.eye(width, dtype=F64).repeat(len(bases), 1, 1)
    for row, basis in enumerate(bases):
        padded[row, : len(basis), : len(basis)] = basis
    return padded


def _padded_rows(values: list[torch.Tensor], width: int) -> torch.Tensor:
    padded = torch.zeros(len(values), width, dtype=F64)
    for row, value in enumerate(values):
        padded[row, : value.numel()] = value
    return padded


def test_row_wise_torus_wrap_equals_the_wrap_of_each_row() -> None:
    bases = _row_bases()
    padded_bases = _identity_padded(bases)
    generator = torch.Generator().manual_seed(4)
    values = [5.0 * torch.randn(len(b), generator=generator, dtype=F64) for b in bases]
    validate_period_basis_rows(padded_bases, len(bases), 3)
    wrapped = canonicalize_torus_validated(_padded_rows(values, 3), padded_bases)
    for row, (basis, value) in enumerate(zip(bases, values, strict=True)):
        k = len(basis)
        torch.testing.assert_close(
            wrapped[row, :k], canonicalize_torus(value, basis), rtol=0, atol=1e-12
        )
        # The identity completion keeps a padded coordinate at exactly zero.
        assert torch.equal(wrapped[row, k:], torch.zeros(3 - k, dtype=F64))
        if k:
            coefficients = torch.linalg.solve(basis.T, wrapped[row, :k])
            assert bool(((coefficients > -1e-12) & (coefficients < 1.0 + 1e-12)).all())


def test_row_wise_period_bases_are_validated() -> None:
    bases = _identity_padded(_row_bases())
    assert validate_period_basis_rows(torch.zeros(4, 0, 0, dtype=F64), 4, 0) is None
    with pytest.raises(ValueError, match=r"shape \[3,3,3\]"):
        validate_period_basis_rows(bases, 3, 3)
    broken = bases.clone()
    broken[1, 0, 0] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        validate_period_basis_rows(broken, 4, 3)
    singular = bases.clone()
    singular[2] = 0.0
    with pytest.raises(ValueError, match="nonsingular"):
        validate_period_basis_rows(singular, 4, 3)

    def rest(time, z, eta):  # type: ignore[no-untyped-def]
        return torch.zeros_like(z), torch.zeros_like(eta)

    state = FlowState(torch.zeros(4, 2, dtype=F64), torch.zeros(4, 3, dtype=F64))
    with pytest.raises(ValueError, match="nonsingular"):
        integrate_rk4(rest, state, singular, num_steps=1)
    with pytest.raises(ValueError, match=r"state of shape \[B,d\]"):
        integrate_rk4(
            rest,
            FlowState(torch.zeros(2, dtype=F64), torch.zeros(3, dtype=F64)),
            bases,
            num_steps=1,
        )
    with pytest.raises(ValueError, match="row-wise period bases need values"):
        canonicalize_torus_validated(torch.zeros(2, 3, dtype=F64), bases)


def test_row_wise_integration_of_a_constant_field_is_the_exact_wrapped_translate() -> None:
    """RK4 is exact for a constant field, so the answer is known in closed form."""

    bases = _row_bases()
    generator = torch.Generator().manual_seed(5)
    starts = [torch.randn(len(b), generator=generator, dtype=F64) for b in bases]
    # Several periods along every axis: each row wraps many times on its own torus.
    speeds = [7.3 * torch.randn(len(b), generator=generator, dtype=F64) @ b for b in bases]
    padded_speed = _padded_rows(speeds, 3)

    def field(time, z, eta):  # type: ignore[no-untyped-def]
        return torch.zeros_like(z), padded_speed

    final = integrate_rk4(
        field,
        FlowState(torch.zeros(4, 1, dtype=F64), _padded_rows(starts, 3)),
        _identity_padded(bases),
        num_steps=5,
    ).final_state
    for row, (basis, start, speed) in enumerate(zip(bases, starts, speeds, strict=True)):
        k = len(basis)
        exact = canonicalize_torus(start + speed, basis)
        assert _torus_gap(final.eta[row, :k], exact, basis) <= 1e-12
        assert torch.equal(final.eta[row, k:], torch.zeros(3 - k, dtype=F64))
        if k:
            assert float((start + speed - exact).abs().max()) > 1.0  # it did wrap


def _row_field(basis: torch.Tensor, drift: torch.Tensor):  # type: ignore[no-untyped-def]
    """A smooth field of one row that is periodic on the row's own torus."""

    def field(time, z, eta):  # type: ignore[no-untyped-def]
        wave = (
            torch.sin(2.0 * math.pi * torch.linalg.solve(basis.T, eta))
            if eta.numel()
            else eta
        )
        return (
            -0.5 * z + time + wave.sum(),
            drift * torch.cos(z).sum() + 0.4 * wave @ basis,
        )

    return field


@pytest.mark.parametrize("terminal_time", [1.0, 0.9])
def test_row_wise_integration_equals_integrating_each_row(terminal_time: float) -> None:
    bases = _row_bases()
    generator = torch.Generator().manual_seed(6)
    drifts = [3.0 * torch.randn(k, generator=generator, dtype=F64) for _, k in _ROWS]
    starts = [
        FlowState(
            torch.randn(d, generator=generator, dtype=F64),
            2.0 * torch.randn(k, generator=generator, dtype=F64),
        )
        for d, k in _ROWS
    ]
    fields = [_row_field(basis, drift) for basis, drift in zip(bases, drifts, strict=True)]

    def padded_field(time, z, eta):  # type: ignore[no-untyped-def]
        z_velocity, eta_velocity = torch.zeros_like(z), torch.zeros_like(eta)
        for row, ((d, k), single) in enumerate(zip(_ROWS, fields, strict=True)):
            z_velocity[row, :d], eta_velocity[row, :k] = single(
                time, z[row, :d], eta[row, :k]
            )
        return z_velocity, eta_velocity

    final = integrate_with_endpoint_estimate(
        padded_field,
        FlowState(
            _padded_rows([start.z for start in starts], 3),
            _padded_rows([start.eta for start in starts], 3),
        ),
        _identity_padded(bases),
        num_steps=8,
        terminal_time=terminal_time,
    )
    for row, ((d, k), basis, single, start) in enumerate(
        zip(_ROWS, bases, fields, starts, strict=True)
    ):
        alone = integrate_with_endpoint_estimate(
            single, start, basis, num_steps=8, terminal_time=terminal_time
        )
        torch.testing.assert_close(final.z[row, :d], alone.z, rtol=0, atol=1e-11)
        assert _torus_gap(final.eta[row, :k], alone.eta, basis) <= 1e-11
        assert torch.equal(final.z[row, d:], torch.zeros(3 - d, dtype=F64))
        assert torch.equal(final.eta[row, k:], torch.zeros(3 - k, dtype=F64))


# --- padded cover plan --------------------------------------------------------


def _ball_bridges(model, dtype=F64):  # type: ignore[no-untyped-def]
    covers = [ContinuousCover.from_exact_scaffold(s) for s in _mixed_scaffolds()]
    return [TorchCoverGeometry(model.layout(cover)).to(dtype=dtype) for cover in covers]


def _loewner_bridges(model, dtype=F64):  # type: ignore[no-untyped-def]
    covers = [ContinuousCover.from_exact_scaffold(s) for s in _mixed_scaffolds()]
    return [
        TorchCoverGeometry(
            IntrinsicCoverLayout(cover, catalogue=model.catalogue, lattice_chart="loewner-v2")
        ).to(dtype=dtype)
        for cover in covers
    ]


def _centred_bridges(cells: tuple[str, ...]):  # type: ignore[no-untyped-def]
    bridges = []
    for scaffold, cell in zip(_centred_scaffolds(), cells, strict=True):
        layout = IntrinsicCoverLayout(
            ContinuousCover.from_exact_scaffold(scaffold), catalogue=_centred_catalogue()
        )
        bridges.append(TorchCoverGeometry(layout, expansion_cell=cell).double())
    return bridges


def _states(bridges, dtype, seed: int = 9):  # type: ignore[no-untyped-def]
    generator = torch.Generator().manual_seed(seed)
    z_values = [
        (0.6 * torch.randn(b.lattice_dimension, generator=generator, dtype=F64)).to(dtype)
        for b in bridges
    ]
    eta_values = [
        (2.0 * torch.randn(b.intrinsic_dimension, generator=generator, dtype=F64)).to(dtype)
        for b in bridges
    ]
    return z_values, eta_values


def _pad_states(plan, z_values, eta_values, fill: float = 0.0):  # type: ignore[no-untyped-def]
    z = torch.full((plan.batch, plan.lattice_width), fill, dtype=plan.dtype)
    eta = torch.full((plan.batch, plan.intrinsic_width), -fill, dtype=plan.dtype)
    for row, (z_row, eta_row) in enumerate(zip(z_values, eta_values, strict=True)):
        z[row, : z_row.numel()] = z_row
        eta[row, : eta_row.numel()] = eta_row
    return z, eta


def _assert_plan_equals_per_record(bridges, dtype, tolerance: float) -> None:  # type: ignore[no-untyped-def]
    plan = PaddedCoverPlan(bridges, dtype=dtype, device=CPU)
    z_values, eta_values = _states(bridges, dtype)
    expansion = plan.expand(*_pad_states(plan, z_values, eta_values))
    widths = (plan.lattice_width, plan.intrinsic_width, plan.atom_width)
    assert widths == (
        max(b.lattice_dimension for b in bridges),
        max(b.intrinsic_dimension for b in bridges),
        max(b.atom_count for b in bridges),
    )
    assert plan.atom_counts == tuple(b.atom_count for b in bridges)
    assert plan.lattice_dimensions.tolist() == [b.lattice_dimension for b in bridges]
    assert plan.intrinsic_dimensions.tolist() == [b.intrinsic_dimension for b in bridges]
    for row, (bridge, z, eta) in enumerate(zip(bridges, z_values, eta_values, strict=True)):
        single = bridge(z, eta)
        n, d, k = bridge.atom_count, bridge.lattice_dimension, bridge.intrinsic_dimension
        period = bridge.period_basis.to(dtype=dtype)
        lattice_tolerance = tolerance
        if bridge.chart_kind == "ball-v1":
            scale = float(single.metric.abs().max())
            lattice_tolerance = max(tolerance, BALL_CHART_RELATIVE_GAP * scale)

        def close(actual, expected, atol=tolerance) -> None:  # type: ignore[no-untyped-def]
            torch.testing.assert_close(actual, expected, rtol=0, atol=atol)

        # State-independent constants are copied, not recomputed.
        assert plan.atom_mask[row].tolist() == [True] * n + [False] * (widths[2] - n)
        assert torch.equal(plan.atomic_numbers[row, :n], single.atomic_numbers)
        assert bool((plan.atomic_numbers[row, n:] == 0).all())
        assert torch.equal(plan.atom_features[row, :n], single.atom_features)
        assert bool((plan.atom_features[row, n:] == 0).all())
        assert torch.equal(
            plan.lattice_commutant_basis[row, :, :d], single.lattice_commutant_basis
        )
        assert bool((plan.lattice_commutant_basis[row, :, d:] == 0).all())
        completed = torch.eye(widths[1], dtype=dtype)
        completed[:k, :k] = period
        assert torch.equal(plan.period_basis[row], completed)
        # The state-dependent expansion equals the per-record bridge to rounding.
        bounded = expansion.bounded_lattice_coordinates[row]
        close(bounded[:d], single.bounded_lattice_coordinates)
        assert bool((bounded[d:] == 0).all())
        close(expansion.metric[row], single.metric, lattice_tolerance)
        close(expansion.lattice[row], single.lattice, lattice_tolerance)
        assert (
            _torus_gap(
                expansion.intrinsic_coordinates[row, :k], single.intrinsic_coordinates, period
            )
            <= tolerance
        )
        assert bool((expansion.intrinsic_coordinates[row, k:] == 0).all())
        gap = expansion.fractional_coordinates[row, :n] - single.fractional_coordinates
        assert float((torch.remainder(gap + 0.5, 1.0) - 0.5).abs().max()) <= tolerance
        assert torch.equal(
            expansion.fractional_coordinates[row, n:],
            expansion.fractional_coordinates[row, :1].expand(widths[2] - n, 3),
        )
        jacobian = expansion.jacobian_cartesian[row]
        close(jacobian[:n, :, :k], single.jacobian_cartesian, lattice_tolerance)
        assert bool((jacobian[n:] == 0).all()) and bool((jacobian[:, :, k:] == 0).all())


@pytest.mark.parametrize("chart", ["ball-v1", "loewner-v2", "both"])
def test_padded_plan_equals_the_per_record_bridge(mixed_model, chart: str) -> None:  # type: ignore[no-untyped-def]  # noqa: F811,E501
    ball, loewner = _ball_bridges(mixed_model), _loewner_bridges(mixed_model)
    # (d_z, d_eta, atoms) = (1, 1, 7), (6, 3, 2), (2, 1, 3), (1, 0, 4).
    bridges = {
        "ball-v1": ball,
        "loewner-v2": loewner,
        # Two charts in one plan: rows are grouped by chart and written back in place.
        "both": [ball[0], loewner[1], ball[2], loewner[3], loewner[0], ball[1]],
    }[chart]
    assert {b.chart_kind for b in bridges} == (
        {"ball-v1", "loewner-v2"} if chart == "both" else {chart}
    )
    _assert_plan_equals_per_record(bridges, F64, 1e-12)


def test_ball_chart_plan_equals_the_batched_bridge_to_rounding(mixed_model) -> None:  # type: ignore[no-untyped-def]  # noqa: F811,E501
    """Against a batched reference the ball chart agrees to rounding as well."""

    bridges = _ball_bridges(mixed_model)
    plan = PaddedCoverPlan(bridges, dtype=F64, device=CPU)
    z_values, eta_values = _states(bridges, F64)
    expansion = plan.expand(*_pad_states(plan, z_values, eta_values))
    for row, (bridge, z, eta) in enumerate(zip(bridges, z_values, eta_values, strict=True)):
        # Two copies of the state: the bridge's own batched chart, on the same branch.
        twice = bridge.forward_batch(torch.stack([z, z]), torch.stack([eta, eta]))
        n, k = bridge.atom_count, bridge.intrinsic_dimension
        for actual, expected in (
            (expansion.metric[row], twice.metric[0]),
            (expansion.lattice[row], twice.lattice[0]),
            (expansion.jacobian_cartesian[row, :n, :, :k], twice.jacobian_cartesian[0]),
        ):
            torch.testing.assert_close(actual, expected, rtol=0, atol=1e-12)


@pytest.mark.parametrize("chart", ["ball-v1", "loewner-v2"])
def test_padded_plan_keeps_the_float64_chart_under_a_float32_master(mixed_model, chart: str) -> None:  # type: ignore[no-untyped-def]  # noqa: F811,E501
    build = _ball_bridges if chart == "ball-v1" else _loewner_bridges
    _assert_plan_equals_per_record(build(mixed_model, F32), F32, 2e-5)


@pytest.mark.parametrize(
    "cells",
    [("primitive-v2",) * 4, ("primitive-v2", "conventional-v1") * 2],
)
def test_padded_plan_expands_centred_groups_on_the_primitive_cell(cells) -> None:  # type: ignore[no-untyped-def]
    bridges = _centred_bridges(cells)
    assert [b.expansion_cell for b in bridges] == list(cells)
    _assert_plan_equals_per_record(bridges, F64, 1e-12)


def test_a_plan_of_parameter_free_covers_has_no_intrinsic_slot(mixed_model) -> None:  # type: ignore[no-untyped-def]  # noqa: F811,E501
    bridge = _ball_bridges(mixed_model)[3]
    assert bridge.intrinsic_dimension == 0
    plan = PaddedCoverPlan([bridge, bridge], dtype=F64, device=CPU)
    assert plan.intrinsic_width == 0 and plan.period_basis.shape == (2, 0, 0)
    _assert_plan_equals_per_record([bridge, bridge], F64, 1e-12)


def test_padded_state_slots_cannot_leak_into_a_crystal(mixed_model) -> None:  # type: ignore[no-untyped-def]  # noqa: F811,E501
    bridges = _loewner_bridges(mixed_model)
    plan = PaddedCoverPlan(bridges, dtype=F64, device=CPU)
    z_values, eta_values = _states(bridges, F64)
    clean = plan.expand(*_pad_states(plan, z_values, eta_values))
    dirty = plan.expand(*_pad_states(plan, z_values, eta_values, fill=7.25))
    for name in (
        "bounded_lattice_coordinates",
        "metric",
        "lattice",
        "intrinsic_coordinates",
        "fractional_coordinates",
        "jacobian_cartesian",
    ):
        assert torch.equal(getattr(clean, name), getattr(dirty, name)), name


def test_padded_plan_refuses_malformed_input(mixed_model) -> None:  # type: ignore[no-untyped-def]  # noqa: F811,E501
    bridges = _ball_bridges(mixed_model)
    with pytest.raises(ValueError, match="at least one bridge"):
        PaddedCoverPlan([], dtype=F64, device=CPU)
    with pytest.raises(TypeError, match="TorchCoverGeometry"):
        PaddedCoverPlan([bridges[0], object()], dtype=F64, device=CPU)
    with pytest.raises(ValueError, match="float32 or float64"):
        PaddedCoverPlan(bridges, dtype=torch.float16, device=CPU)
    plan = PaddedCoverPlan(bridges, dtype=F64, device=CPU)
    z, eta = _pad_states(plan, *_states(bridges, F64))
    with pytest.raises(ValueError, match="z must have shape"):
        plan.expand(z[:, :-1], eta)
    with pytest.raises(ValueError, match="eta must have shape"):
        plan.expand(z, eta[:-1])
    with pytest.raises(ValueError, match="master dtype"):
        plan.expand(z.float(), eta.float())


# --- sampler ------------------------------------------------------------------


@pytest.fixture(scope="module")
def centred_model():  # type: ignore[no-untyped-def]
    """Loewner chart, primitive-cell expansion and an endpoint estimate from t = 0.9."""

    base = _tiny_config()
    config = replace(
        base,
        domain=replace(
            base.domain,
            max_raw_atoms=8,
            max_formula_units=4,
            lattice_chart="loewner-v2",
            metric_radius=2.0,  # the default: it has no effect under loewner-v2
        ),
        numerics=replace(base.numerics, ode_terminal_time=0.9),
        model=replace(base.model, geometry_expansion_cell="primitive-v2"),
    )
    torch.manual_seed(37)
    return CrystalAtlasModel(config, catalogue=_centred_catalogue()).double()


def _request(samples: int = 12, seed: int = 29) -> GenerationRequest:
    return GenerationRequest(
        mode=GenerationMode.CSP,
        composition=Composition.from_counts({6: 1, 8: 1}),
        samples=samples,
        seed=seed,
    )


def _signatures(model, traces) -> list[tuple[object, ...]]:  # type: ignore[no-untyped-def]
    return [
        _layout_batch_signature(model.layout(trace.sample.cover_sample.cover))
        for trace in traces
        if trace.succeeded
    ]


def _assert_same_traces(model, actual, expected, tolerance: float) -> None:  # type: ignore[no-untyped-def]
    assert len(actual) == len(expected)
    for trace, reference in zip(actual, expected, strict=True):
        assert trace.sample_index == reference.sample_index
        assert trace.seed == reference.seed
        assert trace.failed_stage == reference.failed_stage
        if reference.failed_stage == "geometry/ode":
            assert trace.error == reference.error
        if not reference.succeeded:
            continue
        assert trace.succeeded
        assert trace.sample.cover_sample == reference.sample.cover_sample
        np.testing.assert_allclose(
            trace.sample.geometry.lattice_latent,
            reference.sample.geometry.lattice_latent,
            rtol=tolerance,
            atol=tolerance,
        )
        layout = model.layout(reference.sample.cover_sample.cover)
        gap = _torus_gap(
            torch.tensor(np.array(trace.sample.geometry.intrinsic_coordinates, dtype=float)),
            torch.tensor(
                np.array(reference.sample.geometry.intrinsic_coordinates, dtype=float)
            ),
            torch.tensor(np.array(layout.period_basis, dtype=float)),
        )
        assert gap <= tolerance


def _spy_on_chunks(monkeypatch, fail=None):  # type: ignore[no-untyped-def]
    """Record every padded solve; ``fail(covers)`` may return an error to raise after it."""

    original = LearnedCrystalGenerator._integrate_mixed_chunk
    chunks: list[tuple[int, int]] = []

    def spy(self, covers, generators):  # type: ignore[no-untyped-def]
        chunks.append(
            (len(covers), max(self.model.layout(c).expanded_atom_count for c in covers))
        )
        # The solve runs first, so the generators have advanced when it fails.
        result = original(self, covers, generators)
        error = None if fail is None else fail(covers)
        if error is not None:
            raise error
        return result

    monkeypatch.setattr(LearnedCrystalGenerator, "_integrate_mixed_chunk", spy)
    return chunks


def _forbid_bucket_solves(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    def forbidden(self, covers, generators):  # type: ignore[no-untyped-def]
        raise AssertionError("the fused sampler fell back to a bucket solve")

    monkeypatch.setattr(LearnedCrystalGenerator, "_integrate_geometry_batch", forbidden)


@pytest.mark.parametrize("fixture", ["mixed_model", "centred_model"])
def test_mixed_sampler_equals_the_bucket_sampler_in_one_solve(  # type: ignore[no-untyped-def]
    fixture: str, request, monkeypatch
) -> None:
    model = request.getfixturevalue(fixture)
    expected = LearnedCrystalGenerator(model, allow_untrained=True).sample(_request())
    signatures = _signatures(model, expected)
    # The comparison is between many topologies, not copies of one.
    assert len(set(signatures)) >= 4
    fused = LearnedCrystalGenerator(model, allow_untrained=True, geometry_sampler="mixed-v1")
    assert fused.geometry_sampler == "mixed-v1"
    chunks = _spy_on_chunks(monkeypatch)
    _forbid_bucket_solves(monkeypatch)
    actual = fused.sample(_request())
    assert [size for size, _ in chunks] == [12]
    _assert_same_traces(model, actual, expected, 1e-10)
    assert model.training  # the caller's mode is restored


def test_mixed_sampler_equals_the_bucket_sampler_at_float32(mixed_model, monkeypatch) -> None:  # type: ignore[no-untyped-def]  # noqa: F811,E501
    model = CrystalAtlasModel(mixed_model.scientific_config, catalogue=mixed_model.catalogue)
    model.load_state_dict(mixed_model.state_dict())
    assert next(model.parameters()).dtype == F32
    expected = LearnedCrystalGenerator(model, allow_untrained=True).sample(_request())
    fused = LearnedCrystalGenerator(model, allow_untrained=True, geometry_sampler="mixed-v1")
    _forbid_bucket_solves(monkeypatch)
    _assert_same_traces(model, fused.sample(_request()), expected, 1e-4)


def test_mixed_chunks_are_ascending_in_width_and_respect_each_capacity(mixed_model) -> None:  # type: ignore[no-untyped-def]  # noqa: F811,E501
    base = mixed_model.scientific_config
    bounded = CrystalAtlasModel(
        replace(base, numerics=replace(base.numerics, max_dense_pair_interactions=64)),
        catalogue=mixed_model.catalogue,
    ).double()
    generator = LearnedCrystalGenerator(
        bounded, allow_untrained=True, geometry_sampler="mixed-v1"
    )
    assert [generator._mixed_capacity(n, F64) for n in (2, 4, 8, 9)] == [16, 4, 1, 0]
    atoms = [4, 2, 9, 2, 4, 4, 2, 4, 4, 9, 8]
    # Only the atom count of a bridge enters the plan of chunks.
    stand_ins = tuple(SimpleNamespace(atom_count=count) for count in atoms)
    chunks = generator._mixed_chunks(stand_ins, F64)
    # Three 2-atom members and one 4-atom member fill the width-4 capacity of 4;
    # an 8-atom member is alone, and so is each over-budget 9-atom member.
    assert chunks == ((1, 3, 6, 0), (4, 5, 7, 8), (10,), (2,), (9,))
    for chunk in chunks:
        width = max(atoms[index] for index in chunk)
        assert len(chunk) <= max(1, generator._mixed_capacity(width, F64))


def test_mixed_sampler_chunks_a_request_without_changing_attempts(mixed_model, monkeypatch) -> None:  # type: ignore[no-untyped-def]  # noqa: F811,E501
    base = mixed_model.scientific_config
    bounded = CrystalAtlasModel(
        replace(base, numerics=replace(base.numerics, max_dense_pair_interactions=32)),
        catalogue=mixed_model.catalogue,
    ).double()
    bounded.load_state_dict(mixed_model.state_dict())
    expected = LearnedCrystalGenerator(mixed_model, allow_untrained=True).sample(_request())
    fused = LearnedCrystalGenerator(bounded, allow_untrained=True, geometry_sampler="mixed-v1")
    capacity = {n: fused._mixed_capacity(n, F64) for n in (2, 4)}
    assert capacity == {2: 8, 4: 2}
    chunks = _spy_on_chunks(monkeypatch)
    _forbid_bucket_solves(monkeypatch)
    actual = fused.sample(_request())
    assert len(chunks) > 1 and sum(size for size, _ in chunks) == 12
    assert all(size <= capacity[width] for size, width in chunks)
    assert [width for _, width in chunks] == sorted(width for _, width in chunks)
    _assert_same_traces(bounded, actual, expected, 1e-10)


def test_a_failing_member_is_isolated_and_the_others_keep_their_draws(mixed_model, monkeypatch) -> None:  # type: ignore[no-untyped-def]  # noqa: F811,E501
    baseline = LearnedCrystalGenerator(mixed_model, allow_untrained=True).sample(_request())
    victim = baseline[5].sample.cover_sample.cover
    doomed = [
        trace.sample_index
        for trace in baseline
        if trace.succeeded and trace.sample.cover_sample.cover == victim
    ]
    chunks = _spy_on_chunks(
        monkeypatch,
        fail=lambda covers: (
            TangentRankError("injected padded failure") if victim in covers else None
        ),
    )
    original = LearnedCrystalGenerator._integrate_geometry_batch
    alone: list[int] = []

    def injected(self, covers, generators):  # type: ignore[no-untyped-def]
        alone.append(len(covers))
        if covers[0] == victim:
            raise TangentRankError("injected member failure")
        return original(self, covers, generators)

    monkeypatch.setattr(LearnedCrystalGenerator, "_integrate_geometry_batch", injected)
    fused = LearnedCrystalGenerator(
        mixed_model, allow_untrained=True, geometry_sampler="mixed-v1"
    )
    traces = fused.sample(_request())
    failed = [trace for trace in traces if trace.failed_stage == "geometry/ode"]
    assert [trace.sample_index for trace in failed] == doomed
    assert {trace.error for trace in failed} == {"TangentRankError: injected member failure"}
    # The request is halved until the victim is alone; only it reaches the bucket solve.
    assert chunks[0][0] == 12 and min(size for size, _ in chunks) == 1
    assert alone == [1] * len(doomed)
    survivors = [trace for trace in traces if trace.sample_index not in doomed]
    references = [trace for trace in baseline if trace.sample_index not in doomed]
    _assert_same_traces(mixed_model, survivors, references, 1e-10)


def test_a_failure_of_the_padded_call_alone_is_decided_by_the_bucket_solve(mixed_model, monkeypatch) -> None:  # type: ignore[no-untyped-def]  # noqa: F811,E501
    baseline = LearnedCrystalGenerator(mixed_model, allow_untrained=True).sample(_request())
    # Every padded solve consumes its members' draws and then fails: the result
    # equals the baseline only if every generator returned to its initial state.
    chunks = _spy_on_chunks(monkeypatch, fail=lambda covers: ValueError("injected"))
    fused = LearnedCrystalGenerator(
        mixed_model, allow_untrained=True, geometry_sampler="mixed-v1"
    )
    traces = fused.sample(_request())
    # Halving 12 members down to single ones: 12, 6 + 6, four 3s, each 3 as 1 + 2.
    assert sorted(size for size, _ in chunks) == [1] * 12 + [2] * 4 + [3] * 4 + [6] * 2 + [12]
    _assert_same_traces(mixed_model, traces, baseline, 1e-10)


@pytest.mark.parametrize(
    "error", [ResourceLimitError("test work budget"), InvariantViolation("test defect")]
)
def test_resource_and_invariant_failures_still_abort(mixed_model, monkeypatch, error) -> None:  # type: ignore[no-untyped-def]  # noqa: F811,E501
    def failing(self, covers, generators):  # type: ignore[no-untyped-def]
        raise error

    monkeypatch.setattr(LearnedCrystalGenerator, "_integrate_mixed_chunk", failing)
    fused = LearnedCrystalGenerator(
        mixed_model, allow_untrained=True, geometry_sampler="mixed-v1"
    )
    with pytest.raises(type(error)) as caught:
        fused.sample(_request())
    assert caught.value is error


def test_an_over_budget_member_fails_alone_with_the_bucket_samplers_report(mixed_model) -> None:  # type: ignore[no-untyped-def]  # noqa: F811,E501
    base = mixed_model.scientific_config
    bounded = CrystalAtlasModel(
        replace(base, numerics=replace(base.numerics, max_dense_pair_interactions=4)),
        catalogue=mixed_model.catalogue,
    ).double()
    bounded.load_state_dict(mixed_model.state_dict())
    expected = LearnedCrystalGenerator(bounded, allow_untrained=True).sample(_request())
    refused = [trace for trace in expected if trace.failed_stage == "geometry/ode"]
    assert refused and any(trace.succeeded for trace in expected)
    assert {trace.error for trace in refused} == {
        "DomainError: one geometry candidate exceeds the configured dense "
        "pair/workspace budget"
    }
    fused = LearnedCrystalGenerator(bounded, allow_untrained=True, geometry_sampler="mixed-v1")
    _assert_same_traces(bounded, fused.sample(_request()), expected, 1e-10)


def test_the_bucket_sampler_stays_the_default(mixed_model, monkeypatch) -> None:  # type: ignore[no-untyped-def]  # noqa: F811,E501
    assert GEOMETRY_SAMPLERS == ("bucket-v1", "mixed-v1")
    generator = LearnedCrystalGenerator(mixed_model, allow_untrained=True)
    assert generator.geometry_sampler == "bucket-v1"

    def forbidden(self, covers, generators):  # type: ignore[no-untyped-def]
        raise AssertionError("the default sampler entered the fused path")

    monkeypatch.setattr(LearnedCrystalGenerator, "_integrate_members_mixed", forbidden)
    monkeypatch.setattr(LearnedCrystalGenerator, "_integrate_mixed_chunk", forbidden)
    assert len(generator.sample(_request(samples=3))) == 3


def test_sampler_names_and_backbones_are_validated() -> None:
    config = _tiny_config()
    model = CrystalAtlasModel(config, catalogue=MINIMAL_CATALOGUE)
    with pytest.raises(DomainError, match="geometry_sampler must be one of"):
        LearnedCrystalGenerator(model, allow_untrained=True, geometry_sampler="fused")
    local = CrystalAtlasModel(
        replace(config, model=replace(config.model, geometry_backbone="equivariant_local")),
        catalogue=MINIMAL_CATALOGUE,
    )
    with pytest.raises(DomainError, match="requires the dense invariant_global backbone"):
        LearnedCrystalGenerator(local, allow_untrained=True, geometry_sampler="mixed-v1")
    assert (
        LearnedCrystalGenerator(local, allow_untrained=True).geometry_sampler == "bucket-v1"
    )


def test_a_checkpoint_bound_generator_takes_the_sampler(tmp_path) -> None:  # type: ignore[no-untyped-def]
    config = _tiny_config()
    model = CrystalAtlasModel(config, catalogue=MINIMAL_CATALOGUE).double()
    info = save_checkpoint(
        tmp_path / "model.pt",
        model=model,
        optimizer=torch.optim.Adam(model.parameters(), lr=1.0e-3),
        config=config,
        scientific_manifest=_manifest(model),
        provenance=_provenance(config),
        step=0,
    )

    def bound(**options):  # type: ignore[no-untyped-def]
        return LearnedCrystalGenerator.from_checkpoint(
            model,
            tmp_path / "model.pt",
            scientific_manifest=_manifest(model),
            expected_sha256=info.sha256,
            **options,
        )

    assert bound().geometry_sampler == "bucket-v1"
    fused = bound(geometry_sampler="mixed-v1")
    assert fused.geometry_sampler == "mixed-v1" and fused.is_trained_inference
    _assert_same_traces(
        model, fused.sample(_request(samples=5)), bound().sample(_request(samples=5)), 1e-10
    )


def test_the_csp_evaluator_declares_the_sampler(monkeypatch, capsys) -> None:  # type: ignore[no-untyped-def]
    import inspect

    monkeypatch.syspath_prepend(str(SCRIPTS))
    import evaluate_csp_research_v7 as evaluator

    assert inspect.signature(evaluator.load_run).parameters["geometry_sampler"].default == (
        "bucket-v1"
    )
    monkeypatch.setattr("sys.argv", ["evaluate_csp_research_v7.py", "--help"])
    with pytest.raises(SystemExit) as stopped:
        evaluator.main()
    assert stopped.value.code == 0
    assert "--geometry-sampler {bucket-v1,mixed-v1}" in capsys.readouterr().out
