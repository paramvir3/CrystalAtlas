"""Fused host synchronizations in the geometry field (review 2026-10-29, H1).

Every finiteness, handedness and rank check used to end in its own ``.item()``:
25 host reads per packed field call on a four-atom P1 cover.  The checks still
run; their flags are now read once per call, and a failing check raises the
same error it always did.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from test_learned import _tiny_config  # noqa: E402

from crystalatlas.catalogue import MINIMAL_CATALOGUE  # noqa: E402
from crystalatlas.composition import Composition  # noqa: E402
from crystalatlas.learned import CrystalAtlasModel, LearnedCrystalGenerator  # noqa: E402
from crystalatlas.nn.bridge import TorchCoverGeometry  # noqa: E402
from crystalatlas.nn.field import (  # noqa: E402
    TangentRankError,
    reduce_cartesian_to_intrinsic_packed,
)
from crystalatlas.nn.host_sync import DeviceChecks, check  # noqa: E402


class _HostReads:
    """Count every host read of a tensor (``.item()`` and ``bool(tensor)``)."""

    def __init__(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self.items = 0
        self.bools = 0
        item, as_bool = torch.Tensor.item, torch.Tensor.__bool__

        def counting_item(tensor):  # type: ignore[no-untyped-def]
            self.items += 1
            return item(tensor)

        def counting_bool(tensor):  # type: ignore[no-untyped-def]
            self.bools += 1
            return as_bool(tensor)

        monkeypatch.setattr(torch.Tensor, "item", counting_item)
        monkeypatch.setattr(torch.Tensor, "__bool__", counting_bool)


def test_device_checks_raise_the_first_failure_after_one_read(monkeypatch) -> None:
    checks = DeviceChecks()
    checks.add(torch.tensor(False), lambda: ValueError("first"))
    checks.add(torch.tensor(True), lambda: TangentRankError("second"))
    checks.add(torch.tensor(True), lambda: ValueError("third"))
    reads = _HostReads(monkeypatch)
    with pytest.raises(TangentRankError, match="second"):
        checks.raise_first()
    assert reads.items == 3  # the fused read, then flags one and two
    clean = DeviceChecks()
    clean.add(torch.tensor(False), lambda: ValueError("never"))
    clean.add(torch.zeros(1, dtype=torch.bool), lambda: ValueError("never"))
    reads = _HostReads(monkeypatch)
    clean.raise_first()
    DeviceChecks().raise_first()
    assert reads.items == 1 and reads.bools == 0
    with pytest.raises(ValueError, match="single boolean"):
        checks.add(torch.tensor([True, False]), lambda: ValueError("shape"))
    with pytest.raises(ValueError, match="single boolean"):
        checks.add(torch.tensor(1.0), lambda: ValueError("dtype"))


def test_check_without_a_collector_reads_immediately() -> None:
    check(None, torch.tensor(False), lambda: ValueError("never"))
    with pytest.raises(ValueError, match="now"):
        check(None, torch.tensor(True), lambda: ValueError("now"))
    checks = DeviceChecks()
    check(checks, torch.tensor(True), lambda: ValueError("later"))
    assert len(checks) == 1
    with pytest.raises(ValueError, match="later"):
        checks.raise_first()


def test_deferred_block_prefers_a_failed_check_over_a_later_runtime_error() -> None:
    checks = DeviceChecks()
    with pytest.raises(ValueError, match="input"):
        with checks.deferred():
            checks.add(torch.tensor(True), lambda: ValueError("input"))
            torch.linalg.inv(torch.zeros(2, 2))  # the operation the bad input breaks
    checks = DeviceChecks()
    with pytest.raises(RuntimeError, match="singular"):
        with checks.deferred():
            checks.add(torch.tensor(False), lambda: ValueError("input"))
            torch.linalg.inv(torch.zeros(2, 2))
    checks = DeviceChecks()
    with pytest.raises(ValueError, match="at exit"):
        with checks.deferred():
            checks.add(torch.tensor(True), lambda: ValueError("at exit"))


def _cover_with_free_coordinates(generator: LearnedCrystalGenerator, model: CrystalAtlasModel):  # type: ignore[no-untyped-def]
    for counts in ({6: 2, 8: 2}, {6: 3, 8: 3}, {6: 4, 8: 4}):
        for seed in range(1, 40):
            sample = generator._cover_sampler.sample(
                Composition.from_counts(counts), generator=torch.Generator().manual_seed(seed)
            )
            if model.layout(sample.cover).intrinsic_dimension > 0:
                return sample.cover
    raise AssertionError("no cover with free coordinates was sampled")


def test_packed_field_call_reads_the_host_once(monkeypatch) -> None:
    torch.manual_seed(0)
    model = CrystalAtlasModel(_tiny_config(), catalogue=MINIMAL_CATALOGUE).double()
    generator = LearnedCrystalGenerator(model, allow_untrained=True)
    cover = _cover_with_free_coordinates(generator, model)
    layout = model.layout(cover)
    bridge = TorchCoverGeometry(layout).double()
    field = model.geometry_field
    batch = 3
    z = 0.1 * torch.randn(batch, layout.lattice_dimension, dtype=torch.float64)
    eta = torch.rand(batch, layout.intrinsic_dimension, dtype=torch.float64)
    expanded = bridge.forward_batch(z, eta)
    numbers = torch.as_tensor(cover.composition.support, dtype=torch.long).expand(batch, -1)
    counts = torch.as_tensor(cover.composition.coefficients, dtype=torch.float64)
    counts = counts.expand(batch, -1)
    context = field.encode_composition(numbers, counts)

    def call(z_value: torch.Tensor, lattice: torch.Tensor):  # type: ignore[no-untyped-def]
        return field.forward_packed(
            torch.tensor(0.5, dtype=torch.float64),
            z_value,
            eta,
            context,
            atomic_numbers=expanded.atomic_numbers,
            atom_features=expanded.atom_features,
            fractional_coordinates=expanded.fractional_coordinates,
            lattice=lattice,
            bounded_lattice_coordinates=expanded.bounded_lattice_coordinates,
            jacobian_cartesian=expanded.jacobian_cartesian,
            geometry_is_certified=True,
            lattice_commutant_basis=expanded.lattice_commutant_basis,
        )

    reads = _HostReads(monkeypatch)
    with torch.no_grad():
        output = call(z, expanded.lattice)
    assert reads.items == 1 and reads.bools == 0
    single = bridge(z[0], eta[0])
    reads = _HostReads(monkeypatch)
    with torch.no_grad():
        single_output = field(
            torch.tensor(0.5, dtype=torch.float64),
            z[0],
            eta[0],
            context[0],
            atomic_numbers=single.atomic_numbers,
            atom_features=single.atom_features,
            fractional_coordinates=single.fractional_coordinates,
            lattice=single.lattice,
            bounded_lattice_coordinates=single.bounded_lattice_coordinates,
            jacobian_cartesian=single.jacobian_cartesian,
            geometry_is_certified=True,
            lattice_commutant_basis=single.lattice_commutant_basis,
        )
    assert reads.items == 1 and reads.bools == 0
    torch.testing.assert_close(
        single_output.intrinsic_velocity, output.intrinsic_velocity[0], rtol=1e-9, atol=1e-9
    )
    assert output.intrinsic_velocity.shape == (batch, layout.intrinsic_dimension)
    assert bool(torch.isfinite(output.intrinsic_velocity).all())
    assert bool(torch.isfinite(output.lattice_velocity).all())
    # The checks still run and keep their messages.
    bad_z = z.clone()
    bad_z[1, 0] = float("nan")
    with pytest.raises(ValueError, match="packed z and eta entries must be finite"):
        call(bad_z, expanded.lattice)
    bad_lattice = expanded.lattice.clone()
    bad_lattice[2] = 0.0
    with pytest.raises(ValueError, match="right-handed and nonsingular"):
        call(z, bad_lattice)


def test_standalone_packed_reduction_keeps_its_certificate_and_messages() -> None:
    torch.manual_seed(1)
    jacobian = torch.randn(2, 4, 3, 5, dtype=torch.float64)
    vectors = torch.randn(2, 4, 3, dtype=torch.float64)
    reduced = reduce_cartesian_to_intrinsic_packed(vectors, jacobian)
    assert reduced.shape == (2, 5)
    jacobian[1, :, :, 4] = 0.0  # a dead coordinate: rank deficient
    with pytest.raises(TangentRankError, match="rank deficient"):
        reduce_cartesian_to_intrinsic_packed(vectors, jacobian)
    vectors[0, 0, 0] = float("inf")
    fresh = torch.randn(2, 4, 3, 5, dtype=torch.float64)
    with pytest.raises(ValueError, match="finite"):
        reduce_cartesian_to_intrinsic_packed(vectors, fresh)
