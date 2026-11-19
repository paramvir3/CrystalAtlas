"""Exact caches in the geometry CFM objective (GPU-throughput profiling).

Profiling one batch-64 step on the RTX 5060 Ti showed the network at 8% of the
time.  Exact target re-validation took 29% and per-record bridge construction
about 10% (with the pullback Jacobian 36%, tested in test_loewner_chart).  Both
are deterministic functions of their keys, so caching them is exact.  A warm
objective must reproduce a cold one bit for bit, and an invalid target must
still be refused every time.
"""

from __future__ import annotations

from fractions import Fraction

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from crystalatlas.decoder import CrystalScaffold, OrbitAssignment  # noqa: E402
from crystalatlas.errors import DomainError  # noqa: E402
from crystalatlas.learned import LearnedCrystalGenerator  # noqa: E402

pytestmark = pytest.mark.ml


def _toy():  # type: ignore[no-untyped-def]
    import test_flow_oracle_learned as toy_module

    model = toy_module._model()
    return model, toy_module._toy(model)


def _run(generator, model, toy, seed: int):  # type: ignore[no-untyped-def]
    g = torch.Generator().manual_seed(seed)
    indices = torch.randint(len(toy.scaffolds), (16,), generator=g).tolist()
    return generator.geometry_cfm_objective_batch(
        tuple(toy.scaffolds[i] for i in indices),
        tuple(toy.metrics[i] for i in indices),
        time=torch.rand(16, generator=g, dtype=torch.float64),
        sigma_z=toy.sigma_z,
        sigma_eta=toy.sigma_eta,
        generator=g,
        geometry_field=model.geometry_field,
    )


def test_warm_caches_reproduce_the_cold_objective_bit_for_bit() -> None:
    model, toy = _toy()
    cold = LearnedCrystalGenerator(model, allow_untrained=True)
    first = _run(cold, model, toy, 3)
    assert cold._validated_targets and cold._bridges  # the caches were filled
    second = _run(cold, model, toy, 3)  # warm
    fresh = _run(LearnedCrystalGenerator(model, allow_untrained=True), model, toy, 3)
    for result in (second, fresh):
        torch.testing.assert_close(
            result.loss.per_sample, first.loss.per_sample, rtol=0, atol=0
        )
        torch.testing.assert_close(result.path.state.eta, first.path.state.eta, rtol=0, atol=0)


def test_an_invalid_target_is_refused_every_time() -> None:
    model, _ = _toy()
    generator = LearnedCrystalGenerator(model, allow_untrained=True)
    # Two carbon atoms related by (1/2, 1/2, 1/2): a hidden translation in P1.
    scaffold = CrystalScaffold.from_mapping(
        nu=2,
        space_group="P1",
        reduced_composition={6: 1},
        orbits=(
            OrbitAssignment("1a", 6, (Fraction(0), Fraction(0), Fraction(0))),
            OrbitAssignment("1a", 6, (Fraction(1, 2), Fraction(1, 2), Fraction(1, 2))),
        ),
    )
    metric = np.diag([16.0, 16.0, 16.0])
    for _ in range(2):
        with pytest.raises(DomainError, match="hidden translations"):
            generator.geometry_cfm_objective_batch(
                (scaffold,),
                (metric,),
                time=torch.tensor([0.5], dtype=torch.float64),
                sigma_z=0.25,
                sigma_eta=0.125,
                geometry_field=model.geometry_field,
            )


def test_encoded_targets_are_cached_read_only() -> None:
    """The chart latent and torus parameters of a target are cached exactly.

    The cached arrays are read-only, so no consumer can alias and corrupt them.
    """

    model, toy = _toy()
    generator = LearnedCrystalGenerator(model, allow_untrained=True)
    first = _run(generator, model, toy, 5)
    assert generator._encoded_targets
    for arrays in generator._encoded_targets.values():
        assert len(arrays) == 2 and not any(array.flags.writeable for array in arrays)
    second = _run(generator, model, toy, 5)
    torch.testing.assert_close(second.loss.per_sample, first.loss.per_sample, rtol=0, atol=0)


def test_single_cover_batch_uses_one_batched_expansion_equal_to_per_record(
    monkeypatch,
) -> None:  # type: ignore[no-untyped-def]
    """One cover: ``forward_batch`` and the batched pullback, equal to per-record calls."""

    from crystalatlas.nn.bridge import TorchCoverGeometry

    model, toy = _toy()
    calls = {"forward_batch": 0, "pullback_batch": 0}
    original_forward = TorchCoverGeometry.forward_batch
    original_pullback = TorchCoverGeometry.lattice_pullback_metric_batch

    def counted_forward(self, z, eta):  # type: ignore[no-untyped-def]
        calls["forward_batch"] += 1
        return original_forward(self, z, eta)

    def counted_pullback(self, z):  # type: ignore[no-untyped-def]
        calls["pullback_batch"] += 1
        return original_pullback(self, z)

    monkeypatch.setattr(TorchCoverGeometry, "forward_batch", counted_forward)
    monkeypatch.setattr(
        TorchCoverGeometry, "lattice_pullback_metric_batch", counted_pullback
    )
    batched = _run(LearnedCrystalGenerator(model, allow_untrained=True), model, toy, 7)
    assert calls == {"forward_batch": 1, "pullback_batch": 1}

    def per_record_forward(self, z, eta):  # type: ignore[no-untyped-def]
        from crystalatlas.nn.bridge import TorchExpandedGeometry

        items = [self(z[index], eta[index]) for index in range(z.shape[0])]
        return TorchExpandedGeometry(
            **{
                name: (
                    items[0].lattice_commutant_basis
                    if name == "lattice_commutant_basis"
                    else torch.stack([getattr(item, name) for item in items])
                )
                for name in TorchExpandedGeometry.__dataclass_fields__
            }
        )

    def per_record_pullback(self, z):  # type: ignore[no-untyped-def]
        return torch.stack([self.lattice_pullback_metric(item) for item in z])

    monkeypatch.setattr(TorchCoverGeometry, "forward_batch", per_record_forward)
    monkeypatch.setattr(
        TorchCoverGeometry, "lattice_pullback_metric_batch", per_record_pullback
    )
    reference = _run(LearnedCrystalGenerator(model, allow_untrained=True), model, toy, 7)
    torch.testing.assert_close(batched.path.state.z, reference.path.state.z, rtol=0, atol=0)
    torch.testing.assert_close(batched.path.state.eta, reference.path.state.eta, rtol=0, atol=0)
    torch.testing.assert_close(
        batched.loss.per_sample, reference.loss.per_sample, rtol=1e-12, atol=1e-14
    )
