"""One failed member no longer aborts a sampling batch (review 2026-10-29, m3).

``_sample_in_eval_mode`` integrates each topology bucket in one batched solve.
A ``TangentRankError`` or ``ValueError`` raised for one member used to
propagate and fail the whole ``sample`` call.  Now the bucket is replayed one
member at a time from the recorded generator states: the failing member gets a
``geometry/ode`` trace and every other member keeps exactly its draws.
Resource exhaustion and invariant violations still propagate.
"""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from test_learned import _tiny_config  # noqa: E402

from crystalatlas.catalogue import MINIMAL_CATALOGUE  # noqa: E402
from crystalatlas.composition import Composition  # noqa: E402
from crystalatlas.errors import InvariantViolation, ResourceLimitError  # noqa: E402
from crystalatlas.learned import CrystalAtlasModel, LearnedCrystalGenerator  # noqa: E402
from crystalatlas.nn.field import TangentRankError  # noqa: E402
from crystalatlas.pipeline import GenerationMode, GenerationRequest  # noqa: E402


def _generator_and_request():  # type: ignore[no-untyped-def]
    torch.manual_seed(13)
    model = CrystalAtlasModel(_tiny_config(), catalogue=MINIMAL_CATALOGUE).double()
    generator = LearnedCrystalGenerator(model, allow_untrained=True)
    request = GenerationRequest(
        mode=GenerationMode.CSP,
        composition=Composition.from_counts({6: 1, 8: 1}),
        samples=4,
        seed=29,
    )
    return generator, request


def test_a_failing_member_is_recorded_and_the_others_keep_their_draws(monkeypatch) -> None:
    generator, request = _generator_and_request()
    baseline = generator.sample(request)
    assert all(trace.succeeded for trace in baseline)
    victim = baseline[1].sample.cover_sample.cover
    original = LearnedCrystalGenerator._integrate_geometry_batch
    calls: list[int] = []

    def injected(self, covers, generators):  # type: ignore[no-untyped-def]
        calls.append(len(covers))
        if len(covers) > 1:
            raise TangentRankError("injected batch certificate failure")
        if covers[0] == victim:
            raise TangentRankError("injected member failure")
        return original(self, covers, generators)

    monkeypatch.setattr(LearnedCrystalGenerator, "_integrate_geometry_batch", injected)
    traces = generator.sample(request)
    assert [trace.sample_index for trace in traces] == [0, 1, 2, 3]
    assert [trace.seed for trace in traces] == [trace.seed for trace in baseline]
    failed = [trace for trace in traces if not trace.succeeded]
    assert [trace.sample_index for trace in failed] == [1]
    assert failed[0].failed_stage == "geometry/ode"
    assert "TangentRankError: injected member failure" == failed[0].error
    assert max(calls) >= 1 and calls.count(1) >= 3  # every survivor was replayed alone
    for trace, reference in zip(traces, baseline, strict=True):
        if trace.succeeded:
            assert trace.sample.cover_sample == reference.sample.cover_sample
            np.testing.assert_allclose(
                trace.sample.geometry.lattice_latent,
                reference.sample.geometry.lattice_latent,
                rtol=1e-10,
                atol=1e-10,
            )
            np.testing.assert_allclose(
                trace.sample.geometry.intrinsic_coordinates,
                reference.sample.geometry.intrinsic_coordinates,
                rtol=1e-10,
                atol=1e-10,
            )


@pytest.mark.parametrize(
    "error", [ResourceLimitError("test work budget"), InvariantViolation("test defect")]
)
def test_resource_and_invariant_failures_still_abort(monkeypatch, error) -> None:
    generator, request = _generator_and_request()

    def failing(self, covers, generators):  # type: ignore[no-untyped-def]
        raise error

    monkeypatch.setattr(LearnedCrystalGenerator, "_integrate_geometry_batch", failing)
    with pytest.raises(type(error)) as caught:
        generator.sample(request)
    assert caught.value is error


def test_members_of_one_cover_share_one_batched_expansion(monkeypatch) -> None:
    """Several candidates of one cover take one batched bridge call per field
    evaluation; their geometries equal the one-at-a-time reference to rounding."""

    generator, request = _generator_and_request()
    cover = generator.sample(request)[0].sample.cover_sample.cover
    device = torch.device("cpu")
    seeds = (3, 4, 5)
    built: list[int] = []
    original = LearnedCrystalGenerator._single_cover_velocity

    def spy(self, *arguments):  # type: ignore[no-untyped-def]
        built.append(1)
        return original(self, *arguments)

    monkeypatch.setattr(LearnedCrystalGenerator, "_single_cover_velocity", spy)
    generator.model.eval()
    with torch.no_grad():
        together = generator._integrate_geometry_batch(
            (cover,) * 3, tuple(generator._torch_generator(seed, device) for seed in seeds)
        )
        assert built == [1]
        alone = [
            generator._integrate_geometry_batch(
                (cover,), (generator._torch_generator(seed, device),)
            )[0]
            for seed in seeds
        ]
    assert built == [1]  # a single member keeps the per-member reference path
    for batched, single in zip(together, alone, strict=True):
        np.testing.assert_allclose(
            batched.lattice_latent, single.lattice_latent, rtol=1e-10, atol=1e-10
        )
        np.testing.assert_allclose(
            batched.intrinsic_coordinates,
            single.intrinsic_coordinates,
            rtol=1e-10,
            atol=1e-10,
        )
    # Different base draws: the agreement above is not between identical members.
    assert not np.allclose(together[0].lattice_latent, together[1].lattice_latent)
