from __future__ import annotations

import json
import math
import random
from hashlib import sha256

import numpy as np
import pytest

torch = pytest.importorskip("torch")

import crystalatlas.training as training_module  # noqa: E402
from crystalatlas.catalogue import MINIMAL_CATALOGUE  # noqa: E402
from crystalatlas.certification import certify_crystallography  # noqa: E402
from crystalatlas.config import CrystalAtlasConfig, DomainConfig  # noqa: E402
from crystalatlas.errors import CatalogueError  # noqa: E402
from crystalatlas.provenance import ProvenanceRecord  # noqa: E402
from crystalatlas.training import (  # noqa: E402
    CheckpointError,
    LossWeights,
    ScientificManifest,
    TrainingNumericsError,
    capture_global_rng_state,
    compose_training_loss,
    exact_finite_action_nll,
    guarded_optimizer_step,
    initialize_training,
    load_checkpoint,
    model_state_fingerprint,
    restore_global_rng_state,
    save_checkpoint,
)

pytestmark = pytest.mark.ml


def test_global_rng_contract_includes_the_optional_mps_stream() -> None:
    state = capture_global_rng_state()
    assert set(state) == {
        "python",
        "numpy_global",
        "torch_cpu",
        "torch_cuda",
        "torch_mps",
    }
    mps_available = bool(
        hasattr(torch, "mps")
        and hasattr(torch.backends, "mps")
        and torch.backends.mps.is_available()
    )
    assert (state["torch_mps"] is not None) == mps_available
    restore_global_rng_state(state)


@pytest.mark.parametrize(
    "corrupt_state",
    (
        torch.tensor([1.0], dtype=torch.float32),
        torch.empty(0, dtype=torch.uint8),
        torch.ones((1, 1), dtype=torch.uint8),
        torch.sparse_coo_tensor(
            torch.tensor([[0]]),
            torch.tensor([1], dtype=torch.uint8),
            size=(1,),
        ),
    ),
    ids=("wrong-dtype", "empty", "wrong-rank", "sparse-layout"),
)
def test_mps_rng_state_rejects_structurally_invalid_tensors(corrupt_state) -> None:
    state = capture_global_rng_state()
    state["torch_mps"] = corrupt_state
    cpu_before = torch.get_rng_state().clone()

    with pytest.raises(CheckpointError, match="invalid Torch MPS RNG state"):
        restore_global_rng_state(state, strict_cuda=False, strict_mps=False)

    torch.testing.assert_close(torch.get_rng_state(), cpu_before)


def test_mps_rng_validation_probe_restores_the_live_stream(monkeypatch) -> None:
    original = torch.tensor([3, 4, 5], dtype=torch.uint8)
    candidate = torch.tensor([9, 8, 7], dtype=torch.uint8)
    live = {"state": original.clone()}
    set_history: list[torch.Tensor] = []

    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: True)
    monkeypatch.setattr(torch.mps, "get_rng_state", lambda: live["state"].clone())

    def set_rng_state(value) -> None:  # type: ignore[no-untyped-def]
        copied = value.detach().cpu().clone()
        set_history.append(copied)
        live["state"] = copied

    monkeypatch.setattr(torch.mps, "set_rng_state", set_rng_state)
    state = capture_global_rng_state()
    state["torch_mps"] = candidate

    training_module._validate_global_rng_state(  # noqa: SLF001
        state,
        strict_cuda=False,
        strict_mps=True,
    )

    torch.testing.assert_close(live["state"], original)
    assert len(set_history) == 2
    torch.testing.assert_close(set_history[0], candidate)
    torch.testing.assert_close(set_history[1], original)


def test_mps_rng_validation_rolls_back_a_backend_rejected_state(monkeypatch) -> None:
    original = torch.tensor([11, 12, 13], dtype=torch.uint8)
    rejected = torch.tensor([99, 98, 97], dtype=torch.uint8)
    live = {"state": original.clone()}
    set_history: list[torch.Tensor] = []

    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: True)
    monkeypatch.setattr(torch.mps, "get_rng_state", lambda: live["state"].clone())

    def set_rng_state(value) -> None:  # type: ignore[no-untyped-def]
        copied = value.detach().cpu().clone()
        set_history.append(copied)
        if torch.equal(copied, rejected):
            raise RuntimeError("opaque MPS state rejected")
        live["state"] = copied

    monkeypatch.setattr(torch.mps, "set_rng_state", set_rng_state)
    state = capture_global_rng_state()
    state["torch_mps"] = rejected
    cpu_before = torch.get_rng_state().clone()

    with pytest.raises(CheckpointError, match="invalid Torch MPS RNG state"):
        restore_global_rng_state(
            state,
            strict_cuda=False,
            strict_mps=True,
        )

    torch.testing.assert_close(live["state"], original)
    torch.testing.assert_close(torch.get_rng_state(), cpu_before)
    assert len(set_history) == 2
    torch.testing.assert_close(set_history[0], rejected)
    torch.testing.assert_close(set_history[1], original)


def _manifest() -> ScientificManifest:
    return ScientificManifest(
        architecture={"name": "linear-unit-test", "version": 1},
        representation={
            "token_codec": "test-v1",
            "gauge": "fractional-row-v1",
            "cover_order": "test-v1",
        },
        catalogue=MINIMAL_CATALOGUE.identity,
        data_contract={"measure": "record", "split_sha256": "1" * 64},
    )


def test_scientific_manifest_reconstructs_nested_frozen_sections_exactly() -> None:
    plain = {
        name: {
            "nested": {"records": [{"active": True, "weight": 0.125}, None]},
            "version": 1,
        }
        for name in ("architecture", "representation", "catalogue", "data_contract")
    }
    expected = json.loads(json.dumps(plain))
    expected_fingerprint = sha256(
        json.dumps(plain, allow_nan=False, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    manifest = ScientificManifest(**plain)
    reconstructed = ScientificManifest(**{name: getattr(manifest, name) for name in plain})
    assert reconstructed.to_dict() == expected == manifest.to_dict()
    assert reconstructed.fingerprint == expected_fingerprint == manifest.fingerprint
    assert ScientificManifest(**manifest.to_dict()).fingerprint == expected_fingerprint

    # Neither the caller's dictionaries nor exported JSON may mutate identity.
    plain["architecture"]["nested"]["records"][0]["weight"] = 0.5
    exported = reconstructed.to_dict()
    exported["data_contract"]["nested"]["records"][0]["weight"] = 0.75
    assert reconstructed.to_dict() == expected == manifest.to_dict()
    with pytest.raises(TypeError):
        reconstructed.data_contract["nested"]["records"][0]["weight"] = 1.0


def test_live_model_validates_nested_frozen_scientific_manifest() -> None:
    from crystalatlas.learned import CrystalAtlasModel
    from crystalatlas.training_benchmarking import objective_benchmark_micro_config

    model = CrystalAtlasModel.fixture(objective_benchmark_micro_config())
    contract = {"measure": {"kind": "record", "sources": [{"synthetic": True}]}}
    manifest = model.scientific_manifest(data_contract=contract)
    model.validate_scientific_manifest(manifest)
    model.validate_scientific_manifest(ScientificManifest(**manifest.to_dict()))
    assert model.scientific_manifest(data_contract=manifest.data_contract).fingerprint == (
        manifest.fingerprint
    )


@pytest.mark.parametrize(
    "invalid",
    (
        float("nan"),
        float("inf"),
        float("-inf"),
        {"not-json"},
        object(),
        {("tuple", "key"): "invalid"},
        {object(): "invalid"},
        {1: "integer-key", "one": "incomparable-sort-key"},
    ),
    ids=(
        "nan",
        "inf",
        "negative-inf",
        "set",
        "object",
        "tuple-key",
        "object-key",
        "mixed-keys",
    ),
)
def test_scientific_manifest_thaw_does_not_coerce_non_json_values(invalid) -> None:
    sections = _manifest().to_dict()
    sections["data_contract"] = {"nested": [{"value": invalid}]}
    with pytest.raises(TypeError, match="only finite JSON-compatible values"):
        ScientificManifest(**sections)


def test_scientific_manifest_rejects_cyclic_data_contract() -> None:
    cyclic = {}
    cyclic["cycle"] = cyclic
    sections = _manifest().to_dict()
    sections["data_contract"] = cyclic
    with pytest.raises(TypeError, match="only finite JSON-compatible values"):
        ScientificManifest(**sections)


def _initialize_training(config: CrystalAtlasConfig, seed: int):
    receipt = certify_crystallography(
        MINIMAL_CATALOGUE,
        require_all_230=False,
        crosscheck_external_operations=False,
    )
    return initialize_training(
        config,
        seed,
        catalogue=MINIMAL_CATALOGUE,
        crystallography_certification=receipt,
        require_all_230_catalogue=False,
        require_external_operation_database=False,
    )


def _provenance(config: CrystalAtlasConfig, seed: int) -> ProvenanceRecord:
    return ProvenanceRecord(
        created_utc="2026-08-25T00:00:00+00:00",
        config_fingerprint=config.fingerprint,
        dataset_hashes=(),
        catalogue_hash=None,
        package_version="0.1.0a1",
        python_version="test",
        platform="test",
        git_revision=None,
        random_seed=seed,
        task="unit-test",
    )


def test_production_training_rejects_a_fixture_only_crystallography_receipt() -> None:
    config = CrystalAtlasConfig()
    receipt = certify_crystallography(
        MINIMAL_CATALOGUE,
        require_all_230=False,
        crosscheck_external_operations=False,
    )
    with pytest.raises(CatalogueError, match="all-230"):
        initialize_training(
            config,
            7,
            catalogue=MINIMAL_CATALOGUE,
            crystallography_certification=receipt,
        )


def test_exact_finite_action_nll_normalizes_only_legal_actions() -> None:
    logits = torch.tensor([[0.2, 1000.0, -0.4], [2.0, 0.0, -900.0]], dtype=torch.float64)
    legal = torch.tensor([[True, False, True], [True, True, False]])
    targets = torch.tensor([2, 0])
    nll = exact_finite_action_nll(
        logits,
        targets,
        legal_actions=legal,
        reduction="none",
    )
    expected = torch.stack(
        (
            torch.logsumexp(logits[0, [0, 2]], dim=0) - logits[0, 2],
            torch.logsumexp(logits[1, [0, 1]], dim=0) - logits[1, 0],
        )
    )
    torch.testing.assert_close(nll, expected)
    with pytest.raises(ValueError, match="observed target is illegal"):
        exact_finite_action_nll(logits, torch.tensor([1, 0]), legal_actions=legal)


@pytest.mark.parametrize("dtype", (torch.float16, torch.bfloat16))
def test_exact_finite_action_nll_promotes_low_precision_normalization(dtype) -> None:
    logits32 = torch.tensor(
        [[12.0, 11.875, -4.0], [-8.0, -8.125, 3.0]],
        dtype=torch.float32,
    )
    targets = torch.tensor([1, 0], dtype=torch.long)
    legal = torch.tensor([[True, True, False], [True, True, False]])
    expected = exact_finite_action_nll(
        logits32,
        targets,
        legal_actions=legal,
        reduction="none",
    )
    low_precision = logits32.to(dtype=dtype).requires_grad_(True)
    actual = exact_finite_action_nll(
        low_precision,
        targets,
        legal_actions=legal,
        reduction="none",
    )
    assert actual.dtype == torch.float32
    torch.testing.assert_close(actual, expected, rtol=2.0e-3, atol=2.0e-3)
    actual.sum().backward()
    assert low_precision.grad is not None
    assert torch.isfinite(low_precision.grad).all()


def test_composite_loss_keeps_nll_and_cfm_semantics_separate() -> None:
    composition = torch.tensor(2.0, requires_grad=True)
    cover = torch.tensor(3.0, requires_grad=True)
    cfm = torch.tensor(5.0, requires_grad=True)
    terms = compose_training_loss(
        composition,
        cover,
        cfm,
        weights=LossWeights(composition_nll=1.0, cover_nll=0.5, cfm_regression=0.25),
    )
    torch.testing.assert_close(terms.total_objective, torch.tensor(4.75))
    assert set(terms.detached_metrics()) == {
        "objective/total",
        "nll/composition",
        "nll/cover",
        "regression/cfm",
    }
    assert all("physical" not in key for key in terms.detached_metrics())


def test_guarded_optimizer_step_reports_and_clips_global_gradient() -> None:
    model = torch.nn.Linear(2, 1, bias=False).double()
    with torch.no_grad():
        model.weight.copy_(torch.tensor([[2.0, -1.0]], dtype=torch.float64))
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    prediction = model(torch.tensor([[3.0, 4.0]], dtype=torch.float64)).sum()
    loss = prediction.square()
    report = guarded_optimizer_step(loss, model, optimizer, max_gradient_norm=0.5)
    assert report.clipped
    assert report.preclip.global_l2_norm > 0.5
    assert math.isclose(report.postclip.global_l2_norm, 0.5, rel_tol=1.0e-12)
    assert report.preclip.all_finite and report.postclip.all_finite


def test_nonfinite_gradient_skips_optimizer_step_and_clears_gradients() -> None:
    model = torch.nn.Linear(1, 1, bias=False).double()
    with torch.no_grad():
        model.weight.zero_()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    before = model.weight.detach().clone()
    # sqrt(0) is finite in the forward pass but has an infinite derivative.
    loss = torch.sqrt(model(torch.ones(1, 1, dtype=torch.float64)).sum())
    with pytest.raises(TrainingNumericsError, match="non-finite gradients"):
        guarded_optimizer_step(loss, model, optimizer)
    torch.testing.assert_close(model.weight, before)
    assert model.weight.grad is None


def test_nonfinite_optimizer_update_is_rolled_back() -> None:
    class InfOptimizer(torch.optim.Optimizer):
        def __init__(self, parameters):
            super().__init__(parameters, {"lr": 1.0})

        def step(self, closure=None):
            del closure
            with torch.no_grad():
                for group in self.param_groups:
                    for parameter in group["params"]:
                        parameter.fill_(math.inf)

    model = torch.nn.Linear(1, 1, bias=False).double()
    optimizer = InfOptimizer(model.parameters())
    before = model.weight.detach().clone()
    loss = model(torch.ones(1, 1, dtype=torch.float64)).square().sum()
    with pytest.raises(TrainingNumericsError, match="transaction rolled back"):
        guarded_optimizer_step(loss, model, optimizer)
    torch.testing.assert_close(model.weight, before)
    assert model.weight.grad is None


def test_atomic_checkpoint_roundtrip_restores_states_and_rngs(tmp_path) -> None:
    config = CrystalAtlasConfig()
    seed = 179
    context = _initialize_training(config, seed)
    model = torch.nn.Linear(2, 1).double()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    loss = model(torch.tensor([[1.0, -2.0]], dtype=torch.float64)).square().sum()
    guarded_optimizer_step(loss, model, optimizer, max_gradient_norm=10.0)
    expected_parameters = {
        name: value.detach().clone() for name, value in model.state_dict().items()
    }

    destination = tmp_path / "nested" / "state.pt"
    info = save_checkpoint(
        destination,
        model=model,
        optimizer=optimizer,
        config=config,
        scientific_manifest=_manifest(),
        provenance=_provenance(config, seed),
        step=17,
        epoch=2,
        training_context=context,
        metadata={"objective": "three-part"},
    )
    assert info.path == destination.resolve()
    assert info.size_bytes > 0 and len(info.sha256) == 64
    assert info.model_state_fingerprint == model_state_fingerprint(model)
    assert list(destination.parent.glob(f".{destination.name}.*.tmp")) == []

    expected_python = random.random()
    expected_numpy_global = float(np.random.random())
    expected_torch_global = torch.rand(3)
    expected_numpy_explicit = context.numpy_generator.random(3)
    expected_torch_explicit = torch.rand(3, generator=context.torch_generator)
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.add_(100.0)
    random.seed(999)
    np.random.seed(999)
    torch.manual_seed(999)
    context.numpy_generator.random(7)
    torch.rand(7, generator=context.torch_generator)

    loaded = load_checkpoint(
        destination,
        model=model,
        expected_scientific_manifest=_manifest(),
        optimizer=optimizer,
        expected_config=config,
        training_context=context,
        restore_rng=True,
        expected_sha256=info.sha256,
    )
    assert loaded.step == 17 and loaded.epoch == 2
    assert loaded.model_state_fingerprint == model_state_fingerprint(model)
    assert loaded.optimizer_restored and loaded.rng_restored
    assert loaded.metadata == {"objective": "three-part"}
    for name, value in model.state_dict().items():
        torch.testing.assert_close(value, expected_parameters[name])
    assert random.random() == expected_python
    assert float(np.random.random()) == expected_numpy_global
    torch.testing.assert_close(torch.rand(3), expected_torch_global)
    np.testing.assert_array_equal(context.numpy_generator.random(3), expected_numpy_explicit)
    torch.testing.assert_close(
        torch.rand(3, generator=context.torch_generator), expected_torch_explicit
    )


def test_configuration_mismatch_fails_before_model_mutation(tmp_path) -> None:
    config = CrystalAtlasConfig()
    seed = 5
    context = _initialize_training(config, seed)
    model = torch.nn.Linear(1, 1).double()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    path = tmp_path / "state.pt"
    save_checkpoint(
        path,
        model=model,
        optimizer=optimizer,
        config=config,
        scientific_manifest=_manifest(),
        provenance=_provenance(config, seed),
        step=0,
        training_context=context,
    )
    before = {name: value.detach().clone() for name, value in model.state_dict().items()}
    other = CrystalAtlasConfig(domain=DomainConfig(max_raw_atoms=39))
    with pytest.raises(CheckpointError, match="different configuration"):
        load_checkpoint(
            path,
            model=model,
            expected_scientific_manifest=_manifest(),
            expected_config=other,
        )
    incompatible_manifest = ScientificManifest(
        architecture={"name": "linear-unit-test", "version": 2},
        representation={"token_codec": "test-v1"},
        catalogue={"name": "unit-test"},
        data_contract={"measure": "record", "split_sha256": "1" * 64},
    )
    with pytest.raises(CheckpointError, match="different scientific manifest"):
        load_checkpoint(
            path,
            model=model,
            expected_scientific_manifest=incompatible_manifest,
            expected_config=config,
        )
    for name, value in model.state_dict().items():
        torch.testing.assert_close(value, before[name])


def test_checkpoint_rejects_optimizer_parameter_order_swaps(tmp_path) -> None:
    class TwoParameterModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.a = torch.nn.Parameter(torch.tensor(0.5, dtype=torch.float64))
            self.b = torch.nn.Parameter(torch.tensor(-2.0, dtype=torch.float64))

    config = CrystalAtlasConfig()
    seed = 43
    source = TwoParameterModel()
    source_optimizer = torch.optim.Adam([source.a, source.b], lr=0.01)
    loss = 0.1 * source.a.square() + source.b.square()
    guarded_optimizer_step(loss, source, source_optimizer)
    path = tmp_path / "ordered-optimizer.pt"
    info = save_checkpoint(
        path,
        model=source,
        optimizer=source_optimizer,
        config=config,
        scientific_manifest=_manifest(),
        provenance=_provenance(config, seed),
        step=1,
    )

    destination = TwoParameterModel()
    reversed_optimizer = torch.optim.Adam([destination.b, destination.a], lr=0.01)
    before = {name: value.detach().clone() for name, value in destination.state_dict().items()}
    with pytest.raises(CheckpointError, match="ordered model-parameter names"):
        load_checkpoint(
            path,
            model=destination,
            optimizer=reversed_optimizer,
            expected_scientific_manifest=_manifest(),
            expected_config=config,
            expected_sha256=info.sha256,
        )
    for name, value in destination.state_dict().items():
        torch.testing.assert_close(value, before[name])
