"""Regression checks for the bounded real-data pilot runner (synthetic fixtures)."""

from __future__ import annotations

import importlib.util
import json
import sys
from dataclasses import replace
from pathlib import Path

import pytest
from test_training_data import _store

torch = pytest.importorskip("torch")

from crystalatlas.catalogue import MINIMAL_CATALOGUE  # noqa: E402
from crystalatlas.composition import Composition  # noqa: E402
from crystalatlas.config import DomainConfig, NumericsConfig  # noqa: E402
from crystalatlas.errors import DomainError  # noqa: E402
from crystalatlas.learned import CrystalAtlasModel  # noqa: E402
from crystalatlas.provenance import sha256_file  # noqa: E402
from crystalatlas.training_data import StoredTrainingDataset  # noqa: E402
from crystalatlas.training_objectives import (  # noqa: E402
    CrystalAtlasTrainingObjective,
    ExactBatchCrystallographicAudit,
    ObjectiveDataMeasure,
)

pytestmark = pytest.mark.ml


@pytest.fixture(scope="module")
def runner():
    path = Path(__file__).resolve().parents[1] / "scripts" / "train_research_pilot.py"
    spec = importlib.util.spec_from_file_location("research_pilot_runner_under_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize(
    "override",
    [
        {"records": 0},
        {"records": 9},
        {"records": True},
        {"max_optimizer_steps": 201},
        {"evaluation_replicates": 0},
        {"learning_rate": float("nan")},
        {"seed": -1},
    ],
)
def test_pilot_settings_reject_invalid_or_unbounded_work(runner, override):
    with pytest.raises(DomainError):
        runner.PilotSettings(**override)


def test_planned_steps_are_hard_bounded(runner):
    assert runner.PilotSettings().required_steps(8) == 192
    assert runner.PilotSettings().required_steps(1) == 136
    with pytest.raises(DomainError, match="exceed budget"):
        runner.PilotSettings(epochs_per_factor=9).required_steps(8)
    with pytest.raises(DomainError, match="unique-composition"):
        runner.PilotSettings().required_steps(9)


def test_reduced_profile_preserves_physical_domain_and_numerics(runner):
    config = runner.pilot_model_config()
    # The frozen pilot keeps the ball-v1 chart; loewner-v2 became the default later.
    domain = DomainConfig(lattice_chart="ball-v1")
    assert config.domain == replace(
        domain,
        dng_element_vocabulary=domain.element_vocabulary,
    )
    assert config.domain.dng_element_vocabulary == tuple(range(1, 119))
    assert config.numerics == NumericsConfig()
    assert config.model.geometry_backbone == "invariant_global"
    assert config.model.composition_width == 64


def test_pilot_composition_support_includes_frozen_thorium_selenide_row(runner):
    model = CrystalAtlasModel(runner.pilot_model_config(), catalogue=MINIMAL_CATALOGUE)
    composition = Composition.from_counts({34: 3, 90: 1})
    encoded = model.composition_codec.encode(composition)
    assert encoded.token_ids.size > 0


def test_subset_selection_is_order_independent_and_never_refills(runner):
    settings = runner.PilotSettings(records=2)
    first = runner.select_record_ids(("a", "b", "c"), settings)
    assert first == runner.select_record_ids(("c", "b", "a"), settings)
    assert len(first) == 2 and set(first).issubset({"a", "b", "c"})
    with pytest.raises(DomainError):
        runner.select_record_ids(("a",), settings)
    with pytest.raises(DomainError):
        runner.select_record_ids(("a", "a"), settings)


def test_outputs_are_exclusive_and_cannot_overlap_store(runner, tmp_path):
    store = tmp_path / "store"
    store.mkdir()
    with pytest.raises(DomainError, match="overlap"):
        runner._fresh_output(store / "run", store)
    output = runner._fresh_output(tmp_path / "output", store)
    with pytest.raises(FileExistsError):
        runner._fresh_output(output, store)
    runner._write_new(output / "report.json", {"status": "test"})
    with pytest.raises(FileExistsError):
        runner._write_new(output / "report.json", {"status": "replacement"})
    assert json.loads((output / "report.json").read_text()) == {"status": "test"}


@pytest.mark.parametrize(
    "device",
    [
        "cpu",
        pytest.param(
            "cuda:0",
            marks=pytest.mark.skipif(
                not torch.cuda.is_available(),
                reason="requires CUDA RNG state",
            ),
        ),
    ],
)
def test_fixed_evaluation_preserves_modes_and_rng(runner, tmp_path, device):
    config = runner.pilot_model_config()
    settings = runner.PilotSettings(records=2, evaluation_replicates=2)
    with _store(tmp_path, config.domain) as store:
        model = CrystalAtlasModel(config, catalogue=MINIMAL_CATALOGUE).to(device)
        dataset = StoredTrainingDataset(
            store,
            model.composition_codec,
            MINIMAL_CATALOGUE,
            domain=config.domain,
        )
        objective = CrystalAtlasTrainingObjective(
            model,
            ObjectiveDataMeasure.from_dataset(dataset),
        )
        audit = ExactBatchCrystallographicAudit(model)
        model.train()
        model.cover_selector.eval()
        modes = tuple(module.training for module in model.modules())
        cpu_rng = torch.random.get_rng_state().clone()
        cuda_rng = torch.cuda.get_rng_state(0).clone() if device != "cpu" else None
        first = runner.evaluate_factors(model, dataset, objective, audit, settings)
        second = runner.evaluate_factors(model, dataset, objective, audit, settings)
        assert first == second
        assert torch.equal(cpu_rng, torch.random.get_rng_state())
        if cuda_rng is not None:
            assert torch.equal(cuda_rng, torch.cuda.get_rng_state(0))
        assert modes == tuple(module.training for module in model.modules())


def test_stored_fixture_trains_all_three_factors_across_multiple_epochs(runner, tmp_path):
    """Two epochs/factor catches accidental sampler resets at every epoch."""
    config = runner.pilot_model_config()
    settings = runner.PilotSettings(records=2, epochs_per_factor=2, evaluation_replicates=1)
    with _store(tmp_path, config.domain) as store:
        model = CrystalAtlasModel(config, catalogue=MINIMAL_CATALOGUE)
        dataset = StoredTrainingDataset(
            store,
            model.composition_codec,
            MINIMAL_CATALOGUE,
            domain=config.domain,
        )
        output = tmp_path / "run"
        output.mkdir()
        result = runner.train_factors(
            model,
            dataset,
            settings,
            output,
            {"test_fixture": True, "dataset_fingerprint": dataset.fingerprint},
        )
    assert result["optimizer_steps"] == 10
    assert [row["phase"] for row in result["epochs"]] == [
        "composition",
        "composition",
        "cover",
        "cover",
        "geometry",
        "geometry",
    ]
    assert [row["optimizer_step"] for row in result["evaluations"]] == [0, 2, 6, 10]
    assert result["tiny_overfit_gate_passed"] is None
    assert result["physical_generation_quality_evaluated"] is False
    assert result["held_out_generalization_evaluated"] is False
    assert result["joint_training_performed"] is False
    for checkpoint in result["checkpoints"]:
        assert sha256_file(output / checkpoint["path"]) == checkpoint["sha256"]
    objective = json.loads((output / "training-contract.json").read_text())["objective"]
    assert float.fromhex(objective["dequantization"]["sigma_lattice_hex"]) > 0
    assert float.fromhex(objective["dequantization"]["sigma_intrinsic_hex"]) > 0
