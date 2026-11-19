from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

import crystalatlas.training_benchmarking as training_benchmarking_module  # noqa: E402
from crystalatlas.cli import main  # noqa: E402
from crystalatlas.config import (  # noqa: E402
    CrystalAtlasConfig,
    DomainConfig,
    ModelConfig,
    NumericsConfig,
)
from crystalatlas.errors import DomainError  # noqa: E402
from crystalatlas.training_benchmarking import (  # noqa: E402
    TRAINING_OBJECTIVE_BENCHMARK_METHOD,
    TRAINING_OBJECTIVE_BENCHMARK_SCHEMA,
    ExactObjectiveFixtureSource,
    TrainingObjectiveBenchmarkConfig,
    TrainingObjectiveBenchmarkExecutionError,
    run_single_device_training_objective_benchmark,
)

pytestmark = pytest.mark.ml


def test_objective_cuda_policy_requires_deterministic_cublas_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    device = SimpleNamespace(type="cuda")
    monkeypatch.delenv("CUBLAS_WORKSPACE_CONFIG", raising=False)
    with pytest.raises(DomainError, match="CUBLAS_WORKSPACE_CONFIG"):
        training_benchmarking_module._precision_policy("float32", device)

    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    master_dtype, precision = training_benchmarking_module._precision_policy("float32", device)
    assert master_dtype is torch.float32
    assert precision.value == "float32"


def _tiny_scientific_config(
    *,
    cutoff: float = 2.5,
    geometry_backbone: str = "invariant_global",
) -> CrystalAtlasConfig:
    return CrystalAtlasConfig(
        domain=DomainConfig(
            element_vocabulary=(6, 7, 8),
            dng_element_vocabulary=(6, 7, 8),
            max_distinct_elements=3,
            max_raw_atoms=40,
            max_formula_units=16,
            metric_radius=2.0,
            interaction_cutoff_angstrom=cutoff,
        ),
        numerics=NumericsConfig(ode_steps=1, max_periodic_edges=20_000),
        model=ModelConfig(
            composition_width=8,
            composition_layers=1,
            selector_width=8,
            selector_layers=1,
            geometry_backbone=geometry_backbone,
            geometry_scalar_width=8,
            geometry_pair_width=8,
            geometry_vector_width=4,
            geometry_layers=1,
            attention_heads=1,
            radial_features=2,
        ),
    )


def _tiny_cpu_config() -> TrainingObjectiveBenchmarkConfig:
    return TrainingObjectiveBenchmarkConfig(
        source=ExactObjectiveFixtureSource(),
        scientific_config=_tiny_scientific_config(),
        device="cpu",
        precision="float32",
        batch_size=1,
        warmup_steps=1,
        measurement_steps=1,
        seed=31,
        learning_rate=1.0e-4,
        cpu_threads=1,
    )


@pytest.mark.parametrize(
    ("updates", "message"),
    [
        ({"batch_size": True}, "batch_size"),
        ({"warmup_steps": 0}, "warmup_steps"),
        ({"measurement_steps": 0}, "measurement_steps"),
        ({"precision": "tf32"}, "precision"),
        ({"seed": -1}, "seed"),
        ({"learning_rate": float("nan")}, "learning_rate"),
        ({"ema_decay": 1.0}, "ema_decay"),
        ({"cpu_threads": False}, "cpu_threads"),
    ],
)
def test_objective_benchmark_config_fails_closed(
    updates: dict[str, object], message: str
) -> None:
    values: dict[str, object] = {
        "scientific_config": _tiny_scientific_config(),
        "device": "cpu",
        "batch_size": 1,
        "warmup_steps": 1,
        "measurement_steps": 1,
    }
    values.update(updates)
    with pytest.raises(DomainError, match=message):
        TrainingObjectiveBenchmarkConfig(**values)  # type: ignore[arg-type]


def test_exact_fixture_source_has_one_pinned_identity() -> None:
    with pytest.raises(DomainError, match="not supported"):
        ExactObjectiveFixtureSource("different-fixture")


def test_cpu_full_objective_report_is_scoped_finite_and_restores_globals() -> None:
    config = _tiny_cpu_config()
    prior_rng = torch.get_rng_state().clone()
    prior_threads = torch.get_num_threads()
    prior_deterministic = torch.are_deterministic_algorithms_enabled()
    prior_precision = torch.get_float32_matmul_precision()

    report = run_single_device_training_objective_benchmark(config)

    assert torch.equal(torch.get_rng_state(), prior_rng)
    assert torch.get_num_threads() == prior_threads
    assert torch.are_deterministic_algorithms_enabled() == prior_deterministic
    assert torch.get_float32_matmul_precision() == prior_precision

    manifest = report.to_manifest()
    assert manifest["schema"] == TRAINING_OBJECTIVE_BENCHMARK_SCHEMA
    assert manifest["method"] == TRAINING_OBJECTIVE_BENCHMARK_METHOD
    assert manifest["method"].endswith(".v3")
    assert manifest["status"] == "completed"
    assert manifest["config"]["fingerprint"] == config.fingerprint  # type: ignore[index]
    assert manifest["source"]["synthetic"] is True  # type: ignore[index]
    assert manifest["source"]["physical_prediction_data"] is False  # type: ignore[index]
    assert manifest["runtime"]["device_type"] == "cpu"  # type: ignore[index]

    interpretation = manifest["interpretation"]
    assert interpretation["full_crystalatlas_training_objective_measured"] is True
    assert interpretation["scalable_trainer_single_device_window_measured"] is True
    assert interpretation["predictive_quality_measured"] is False
    assert interpretation["optimization_convergence_measured"] is False
    assert interpretation["generation_quality_measured"] is False
    assert interpretation["multi_device_scaling_measured"] is False

    phases = manifest["phases"]
    assert isinstance(phases, list) and len(phases) == 1
    phase = phases[0]
    assert phase["phase"] == "full_joint_training_window"
    assert phase["iterations"] == 1
    assert phase["structures_per_iteration"] == 1
    assert phase["atoms_per_iteration"] == 3
    assert set(phase["metrics"][0]) == {
        "objective/total",
        "nll/composition",
        "nll/cover",
        "regression/cfm",
    }
    assert all(value > 0.0 for value in phase["metrics"][0].values())
    assert phase["gradient_tensor_counts"][0] > 0
    assert phase["geometry_interaction_kind"] == "dense_atom_pairs"
    assert len(phase["geometry_interaction_counts_per_iteration"][0]) == 1
    assert "edge_counts_per_iteration" not in phase
    assert phase["structures_per_second"] > 0.0

    numerics = manifest["numerics"]
    assert numerics["float64_crystallographic_audit_passed"] is True
    assert numerics["all_measured_losses_finite"] is True
    assert numerics["all_measured_gradients_finite"] is True
    # Composition, cover, geometry, one joint warmup, one measured joint step.
    assert numerics["optimizer_updates_completed"] == 5
    assert len(numerics["final_model_sha256"]) == 64
    assert manifest["model"]["parameter_initializer"] == (  # type: ignore[index]
        "integer-affine-modulo-2^24-dyadic-no-rng.v1"
    )

    workload = manifest["workload"]
    assert [item["phase"] for item in workload["factor_first_preconditioning"]] == [
        "composition",
        "cover",
        "geometry",
    ]
    assert workload["objective_contract"]["schema"] == ("crystalatlas.training-objective.v4")
    assert workload["data_authentication_and_batch_collation_timed"] is False
    assert workload["trainer_config_fingerprint"]
    assert workload["staged_schedule"] == {
        "composition_epochs": 1,
        "cover_epochs": 1,
        "geometry_epochs": 1,
        "joint_epochs": 1,
    }
    assert workload["optimizer"] == {
        "amsgrad": False,
        "betas": [0.9, 0.999],
        "capturable": False,
        "class": "torch.optim.AdamW",
        "differentiable": False,
        "eps": 1.0e-8,
        "foreach": False,
        "fused": False,
        "learning_rate": 1.0e-4,
        "maximize": False,
        "weight_decay": 0.01,
    }
    logical = manifest["memory"]["logical_tensor_payload"]  # type: ignore[index]
    assert logical["materialized_structure_batch_numpy_bytes"] > 0
    assert logical["materialized_composition_batch_numpy_bytes"] > 0
    assert manifest["memory"]["known_live_logical_tensor_bytes"] == sum(  # type: ignore[index]
        logical.values()
    )
    assert manifest["memory"]["backend_allocator"]["allocator"] is None  # type: ignore[index]
    assert len(manifest["model"]["crystalatlas_package_tree_sha256"]) == 64  # type: ignore[index]

    encoded = report.to_json(indent=None)
    assert json.loads(encoded) == manifest
    assert "NaN" not in encoded and "Infinity" not in encoded


def test_repeat_run_preserves_workload_and_numerical_identity() -> None:
    config = _tiny_cpu_config()
    first = run_single_device_training_objective_benchmark(config).to_manifest()
    second = run_single_device_training_objective_benchmark(config).to_manifest()

    assert first["config"] == second["config"]
    assert first["source"] == second["source"]
    assert first["model"] == second["model"]
    assert first["workload"] == second["workload"]
    assert first["numerics"] == second["numerics"]
    first_phase = first["phases"][0]  # type: ignore[index]
    second_phase = second["phases"][0]  # type: ignore[index]
    assert first_phase["metrics"] == second_phase["metrics"]
    assert first_phase["gradient_l2_norms"] == second_phase["gradient_l2_norms"]
    assert (
        first_phase["geometry_interaction_counts_per_iteration"]
        == (second_phase["geometry_interaction_counts_per_iteration"])
    )


def test_fp16_cpu_is_rejected_without_accelerator_fallback() -> None:
    config = TrainingObjectiveBenchmarkConfig(
        scientific_config=_tiny_scientific_config(),
        device="cpu",
        precision="float16",
        warmup_steps=1,
        measurement_steps=1,
    )
    with pytest.raises(DomainError, match="requires CUDA"):
        run_single_device_training_objective_benchmark(config)


def test_pinned_objective_workload_requires_positive_periodic_edges() -> None:
    config = TrainingObjectiveBenchmarkConfig(
        scientific_config=_tiny_scientific_config(
            cutoff=1.0e-6,
            geometry_backbone="equivariant_local",
        ),
        device="cpu",
        precision="float32",
        warmup_steps=1,
        measurement_steps=1,
        cpu_threads=1,
    )
    with pytest.raises(
        TrainingObjectiveBenchmarkExecutionError,
        match="no positive periodic edge",
    ):
        run_single_device_training_objective_benchmark(config)


def test_local_objective_report_labels_periodic_image_edges_not_dense_pairs() -> None:
    config = TrainingObjectiveBenchmarkConfig(
        scientific_config=_tiny_scientific_config(geometry_backbone="equivariant_local"),
        device="cpu",
        precision="float32",
        warmup_steps=1,
        measurement_steps=1,
        cpu_threads=1,
    )
    manifest = run_single_device_training_objective_benchmark(config).to_manifest()
    phase = manifest["phases"][0]  # type: ignore[index]
    assert phase["geometry_interaction_kind"] == "periodic_image_edges"
    assert phase["geometry_interaction_counts_per_iteration"][0][0] > 0
    assert "edge_counts_per_iteration" not in phase


def test_cli_persists_only_successful_objective_report(tmp_path, capsys) -> None:
    destination = tmp_path / "objective-benchmark.json"
    exit_code = main(
        (
            "benchmark-objective",
            "--source",
            "exact-fixture",
            "--model-profile",
            "micro-v1",
            "--device",
            "cpu",
            "--batch-size",
            "1",
            "--warmup-steps",
            "1",
            "--measurement-steps",
            "1",
            "--cpu-threads",
            "1",
            "--output",
            str(destination),
        )
    )
    captured = capsys.readouterr()
    assert exit_code == 0, captured.err
    printed = json.loads(captured.out)
    persisted = json.loads(destination.read_text(encoding="utf-8"))
    assert printed == persisted
    assert persisted["status"] == "completed"
    assert persisted["interpretation"]["predictive_quality_measured"] is False

    failed_destination = tmp_path / "must-not-exist.json"
    exit_code = main(
        (
            "benchmark-objective",
            "--device",
            "cpu",
            "--precision",
            "float16",
            "--warmup-steps",
            "1",
            "--measurement-steps",
            "1",
            "--output",
            str(failed_destination),
        )
    )
    captured = capsys.readouterr()
    assert exit_code == 2
    assert "requires CUDA" in captured.err
    assert not failed_destination.exists()
