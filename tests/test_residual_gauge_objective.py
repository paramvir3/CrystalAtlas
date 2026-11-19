"""The production CFM objective under the certified residual gauge (option (a)).

``ModelConfig.geometry_residual_gauge = "normalizer-v2"`` samples each path's
residual action uniformly from the cover's certified group: per-orbit in-group
gauges, identical-orbit relabellings, and proper normalizer automorphisms.  The
production target is then the gauge-symmetrized law that a physically invariant
field can represent.  "relabel-v1" (the default) is the legacy behaviour.

On the P2_13 4a toy the full gauge is {x, x + 1/2, 1/4 - x, 3/4 - x}.  Under
normalizer-v2 the objective's own path samples must satisfy
E[(u - v*(q_t)) phi(q_t)] = 0 against the exact oracle with that gauge.  Under
relabel-v1 the check fails, which is what makes it discriminating.
"""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import replace

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from crystalatlas.config import CrystalAtlasConfig, ModelConfig  # noqa: E402
from crystalatlas.errors import DomainError  # noqa: E402
from crystalatlas.flow_oracle_learned import toy_oracle  # noqa: E402
from crystalatlas.learned import CrystalAtlasModel, LearnedCrystalGenerator  # noqa: E402
from crystalatlas.residual_gauge import cover_residual_gauge  # noqa: E402

pytestmark = pytest.mark.ml


def test_residual_gauge_is_a_versioned_model_option() -> None:
    legacy = CrystalAtlasConfig()
    assert legacy.model.geometry_residual_gauge == "relabel-v1"
    assert "geometry_residual_gauge" not in legacy.to_dict()["model"]
    v2 = replace(legacy, model=replace(legacy.model, geometry_residual_gauge="normalizer-v2"))
    assert v2.to_dict()["model"]["geometry_residual_gauge"] == "normalizer-v2"
    assert v2.fingerprint != legacy.fingerprint
    v3 = replace(legacy, model=replace(legacy.model, geometry_residual_gauge="normalizer-v3"))
    assert v3.to_dict()["model"]["geometry_residual_gauge"] == "normalizer-v3"
    assert len({legacy.fingerprint, v2.fingerprint, v3.fingerprint}) == 3
    with pytest.raises(DomainError, match="geometry_residual_gauge"):
        ModelConfig(geometry_residual_gauge="normalizer")


def _models(gauge: str):  # type: ignore[no-untyped-def]
    import test_flow_oracle_learned as toy_module

    model = toy_module._model()
    config = replace(
        model.scientific_config,
        model=replace(model.scientific_config.model, geometry_residual_gauge=gauge),
    )
    torch.manual_seed(0)
    model = CrystalAtlasModel(config, catalogue=model.catalogue).double()
    return model, toy_module._toy(model)


def _residual_statistics(model, toy, oracle, time: float, batches: int = 24):  # type: ignore[no-untyped-def]
    generator = LearnedCrystalGenerator(model, allow_untrained=True)
    g = torch.Generator().manual_seed(17)
    rows = []
    for _ in range(batches):
        indices = torch.randint(len(toy.scaffolds), (64,), generator=g).tolist()
        result = generator.geometry_cfm_objective_batch(
            tuple(toy.scaffolds[i] for i in indices),
            tuple(toy.metrics[i] for i in indices),
            time=torch.full((64,), time, dtype=torch.float64),
            sigma_z=toy.sigma_z,
            sigma_eta=toy.sigma_eta,
            generator=g,
            geometry_field=model.geometry_field,
        )
        rows.append(
            (
                result.path.state.eta.detach(),
                result.path.target_velocity.eta.detach(),
                result.path.state.z.detach(),
            )
        )
    eta = torch.cat([r[0] for r in rows])
    target = torch.cat([r[1] for r in rows])
    z = torch.cat([r[2] for r in rows])
    exact = oracle.velocity(torch.tensor(time, dtype=torch.float64), z, eta)
    residual = (target - exact.eta)[:, 0]
    scores = []
    for phi in (
        torch.sin(2 * math.pi * eta[:, 0]),
        torch.cos(2 * math.pi * eta[:, 0]),
        torch.sin(4 * math.pi * eta[:, 0]),
        torch.cos(4 * math.pi * eta[:, 0]),
    ):
        values = residual * phi
        scores.append(float(values.mean() / (values.std() / math.sqrt(len(values)))))
    return scores


def test_normalizer_gauge_objective_samples_the_symmetrized_oracle_law() -> None:
    model, toy = _models("normalizer-v2")
    oracle = toy_oracle(model, toy, layout_gauge=True)
    assert len(oracle.lattice_actions) == 4
    for time in (0.4, 0.8):
        scores = _residual_statistics(model, toy, oracle, time)
        assert max(abs(s) for s in scores) < 4.0, scores


def test_legacy_objective_is_detectably_not_symmetrized() -> None:
    model, toy = _models("relabel-v1")
    oracle = toy_oracle(model, toy, layout_gauge=True)
    # 4096 samples: the legacy deviation is about z = 5 at 1536 samples, so z = 8 expected.
    scores = _residual_statistics(model, toy, oracle, 0.8, batches=64)
    assert max(abs(s) for s in scores) > 6.0, scores


@pytest.mark.parametrize(
    ("gauge", "improper", "order"), [("normalizer-v2", False, 4), ("normalizer-v3", True, 8)]
)
def test_sampled_residual_actions_are_uniform_over_the_certified_group(
    gauge: str, improper: bool, order: int
) -> None:
    model, toy = _models(gauge)
    generator = LearnedCrystalGenerator(model, allow_untrained=True)
    layout = model.layout(toy.cover)
    group = cover_residual_gauge(layout, improper=improper).all_eta_actions()
    assert len(group) == order
    keys = {
        tuple(np.round(np.concatenate([m.ravel(), s]), 9)): i for i, (m, s) in enumerate(group)
    }
    g = torch.Generator().manual_seed(5)
    counts: Counter[int] = Counter()
    for _ in range(800):
        action, shift = generator._sample_residual_torus_action(
            layout, generator=g, device=torch.device("cpu"), dtype=torch.float64
        )
        key = tuple(
            np.round(np.concatenate([action.numpy().ravel(), np.mod(shift.numpy(), 1.0)]), 9)
        )
        counts[keys[key]] += 1
    assert set(counts) == set(range(len(group)))
    expected = 800 / len(group)
    chi_square = sum((value - expected) ** 2 / expected for value in counts.values())
    # 99.9% quantiles of chi^2 with 3 and 7 degrees of freedom.
    assert chi_square < {4: 16.27, 8: 24.32}[order]


def _matched_racemic_score(objective_gauge: str, time: float, batches: int) -> float:
    """Score of E[(u - v3*) g] with the matched function g = v2* - v3* (eta part).

    Under normalizer-v3 the expectation is zero.  Under normalizer-v2 it is
    E[g^2] > 0.  The chirality signal on this toy is small (sector masses 0.469
    against 0.5), so generic Fourier test functions are too weak to detect it.
    """

    model, toy = _models(objective_gauge)
    v2_model, _ = _models("normalizer-v2")
    v3_model, _ = _models("normalizer-v3")
    v2_oracle = toy_oracle(v2_model, toy, layout_gauge=True)
    v3_oracle = toy_oracle(v3_model, toy, layout_gauge=True)
    generator = LearnedCrystalGenerator(model, allow_untrained=True)
    g = torch.Generator().manual_seed(23)
    values = []
    for _ in range(batches):
        indices = torch.randint(len(toy.scaffolds), (64,), generator=g).tolist()
        result = generator.geometry_cfm_objective_batch(
            tuple(toy.scaffolds[i] for i in indices),
            tuple(toy.metrics[i] for i in indices),
            time=torch.full((64,), time, dtype=torch.float64),
            sigma_z=toy.sigma_z,
            sigma_eta=toy.sigma_eta,
            generator=g,
            geometry_field=model.geometry_field,
        )
        state, target = result.path.state, result.path.target_velocity
        clock = torch.tensor(time, dtype=torch.float64)
        racemic = v3_oracle.velocity(clock, state.z.detach(), state.eta.detach()).eta
        proper = v2_oracle.velocity(clock, state.z.detach(), state.eta.detach()).eta
        values.append(((target.eta.detach() - racemic) * (proper - racemic))[:, 0])
    joined = torch.cat(values)
    return float(joined.mean() / (joined.std() / math.sqrt(len(joined))))


def test_racemic_gauge_objective_samples_the_racemic_oracle_law() -> None:
    """normalizer-v3: the objective's paths match the order-8 racemic oracle."""

    model, toy = _models("normalizer-v3")
    oracle = toy_oracle(model, toy, layout_gauge=True)
    assert len(oracle.lattice_actions) == 8
    for time in (0.4, 0.8):
        scores = _residual_statistics(model, toy, oracle, time)
        assert max(abs(s) for s in scores) < 4.0, scores
    assert abs(_matched_racemic_score("normalizer-v3", 0.6, batches=32)) < 4.0


def test_proper_gauge_objective_is_detectably_not_racemic() -> None:
    assert _matched_racemic_score("normalizer-v2", 0.6, batches=32) > 6.0
