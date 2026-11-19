from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

import crystalatlas.backend_comparison as backend_comparison_module  # noqa: E402
from crystalatlas.backend_comparison import (  # noqa: E402
    BACKEND_COMPARISON_METHOD,
    BACKEND_COMPARISON_SCHEMA,
    BackendComparisonConfig,
    BackendComparisonExecutionError,
    BackendComparisonReport,
    BackendPhaseMeasurement,
    run_single_device_backend_comparison,
)
from crystalatlas.cli import _build_parser, main  # noqa: E402
from crystalatlas.config import DomainConfig  # noqa: E402
from crystalatlas.errors import DomainError  # noqa: E402

pytestmark = pytest.mark.ml


def _tiny_config(**updates: object) -> BackendComparisonConfig:
    values: dict[str, object] = {
        "device": "cpu",
        "precision": "float32",
        "batch_size": 1,
        "atoms_per_structure": 2,
        "volume_per_atom": 16.0,
        "input_scalar_dim": 4,
        "scalar_dim": 6,
        "pair_dim": 4,
        "vector_dim": 3,
        "num_heads": 1,
        "num_layers": 1,
        "invariant_feedforward_multiplier": 2,
        "radial_basis_dim": 2,
        "reciprocal_mode_radius": 1,
        "direct_harmonics": 1,
        "cutoff": 3.0,
        "warmup_steps": 1,
        "measurement_steps": 2,
        "seed": 37,
        "cpu_threads": 1,
    }
    values.update(updates)
    return BackendComparisonConfig(**values)  # type: ignore[arg-type]


def _assert_sha256(value: object) -> None:
    assert isinstance(value, str)
    assert len(value) == 64
    int(value, 16)


def _process_state() -> dict[str, object]:
    return {
        "threads": torch.get_num_threads(),
        "deterministic": torch.are_deterministic_algorithms_enabled(),
        "warn_only": torch.is_deterministic_algorithms_warn_only_enabled(),
        "matmul_precision": torch.get_float32_matmul_precision(),
        "cuda_tf32": torch.backends.cuda.matmul.allow_tf32,
        "cudnn_tf32": torch.backends.cudnn.allow_tf32,
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
        "cpu_rng": torch.get_rng_state().clone(),
    }


def _assert_process_state_equal(
    observed: dict[str, object], expected: dict[str, object]
) -> None:
    assert observed.keys() == expected.keys()
    for name in observed:
        if name == "cpu_rng":
            assert torch.equal(observed[name], expected[name])
        else:
            assert observed[name] == expected[name]


@pytest.mark.parametrize(
    ("updates", "message"),
    [
        ({"batch_size": True}, "batch_size"),
        ({"atoms_per_structure": 1}, "at least two"),
        ({"precision": "tf32"}, "precision"),
        ({"scalar_dim": 7, "vector_dim": 4, "num_heads": 2}, "scalar_dim"),
        ({"vector_dim": 5, "num_heads": 2, "scalar_dim": 6}, "vector_dim"),
        ({"volume_per_atom": float("nan")}, "volume_per_atom"),
        ({"max_dense_pair_interactions": 3}, "max_dense_pair_interactions"),
        ({"measure_local_prebuilt_core": 1}, "boolean"),
    ],
)
def test_backend_comparison_config_fails_closed(
    updates: dict[str, object], message: str
) -> None:
    with pytest.raises(DomainError, match=message):
        _tiny_config(**updates)


def test_backend_comparison_defaults_track_the_production_domain_and_cli() -> None:
    config = BackendComparisonConfig()
    domain = DomainConfig()
    arguments = _build_parser().parse_args(("benchmark-backends",))

    assert config.cutoff == domain.interaction_cutoff_angstrom
    assert arguments.cutoff == config.cutoff
    assert arguments.warmup_steps == config.warmup_steps
    assert arguments.measurement_steps == config.measurement_steps


def test_production_width_fixture_features_have_frozen_exact_bytes() -> None:
    config = _tiny_config(
        atoms_per_structure=16,
        input_scalar_dim=219,
        scalar_dim=24,
        pair_dim=16,
        vector_dim=12,
        num_heads=3,
        radial_basis_dim=6,
        seed=1_729,
    )
    workload = backend_comparison_module._make_constant_density_workload(
        config,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )

    assert hashlib.sha256(workload.features.numpy().tobytes()).hexdigest() == (
        "c4e2ec5753d2f76b6f93dce434855223d9cd8d0d368e83f3c4aae0ce604cb6f8"
    )


def test_interleaved_measurement_counterbalances_ab_ba_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    execution_order: list[str] = []
    timer_calls: list[int] = []

    def first() -> str:
        execution_order.append("A")
        return f"A{len(execution_order)}"

    def second() -> str:
        execution_order.append("B")
        return f"B{len(execution_order)}"

    def timed(_device: object, operation: object) -> tuple[float, object]:
        timer_calls.append(len(timer_calls) + 1)
        assert callable(operation)
        return float(timer_calls[-1]), operation()

    monkeypatch.setattr(backend_comparison_module, "_timed", timed)
    (
        first_durations,
        first_result,
        second_durations,
        second_result,
        first_allocator,
        second_allocator,
    ) = backend_comparison_module._measure_interleaved(
        first,
        second,
        device=object(),
        steps=4,
    )

    assert execution_order == ["A", "B", "B", "A", "A", "B", "B", "A"]
    assert first_durations == (1.0, 4.0, 5.0, 8.0)
    assert second_durations == (2.0, 3.0, 6.0, 7.0)
    assert first_result == "A8"
    assert second_result == "B7"
    assert first_allocator == ()
    assert second_allocator == ()


def test_phase_quantiles_use_declared_linear_interpolation() -> None:
    measurement = BackendPhaseMeasurement(
        phase="witness",
        includes=("deterministic operation",),
        durations_seconds=(4.0, 1.0, 3.0, 2.0),
        structures_per_iteration=1,
        atoms_per_iteration=2,
        interaction_kind="witness_interactions",
        interactions_per_iteration=4,
    ).to_manifest()

    assert measurement["minimum_seconds"] == 1.0
    assert measurement["quartile_25_seconds"] == pytest.approx(1.75)
    assert measurement["median_seconds"] == pytest.approx(2.5)
    assert measurement["quartile_75_seconds"] == pytest.approx(3.25)
    assert measurement["maximum_seconds"] == 4.0


def test_constant_density_workload_closes_and_uses_translated_batch_copies() -> None:
    config = _tiny_config(
        batch_size=3,
        atoms_per_structure=4,
        volume_per_atom=17.5,
        max_dense_pair_interactions=48,
    )
    workload = backend_comparison_module._make_constant_density_workload(
        config,
        device=torch.device("cpu"),
        dtype=torch.float64,
    )

    expected_volume = config.atoms_per_structure * config.volume_per_atom
    volumes = torch.linalg.det(workload.lattices)
    torch.testing.assert_close(
        volumes,
        torch.full_like(volumes, expected_volume),
        rtol=1.0e-12,
        atol=1.0e-12,
    )
    torch.testing.assert_close(
        config.atoms_per_structure / volumes,
        torch.full_like(volumes, 1.0 / config.volume_per_atom),
        rtol=1.0e-12,
        atol=1.0e-12,
    )
    for graph in range(1, config.batch_size):
        translation = torch.remainder(
            workload.fractional[graph] - workload.fractional[0], 1.0
        )
        torch.testing.assert_close(
            translation,
            translation[0].expand_as(translation),
            rtol=0.0,
            atol=1.0e-14,
        )


@pytest.mark.parametrize(
    ("invariant_parameters", "local_parameters", "expected"),
    [(101, 101, True), (101, 103, False)],
)
def test_report_derives_parameter_matching_from_actual_counts(
    invariant_parameters: int, local_parameters: int, expected: bool
) -> None:
    report = BackendComparisonReport(
        config=_tiny_config(),
        runtime={},
        workload={
            "matching_contract": {
                "parameter_matched": expected,
                "compute_matched": False,
                "weights_matched": False,
            },
            "precision_contract": {
                "backend_arithmetic_precision_matched": True,
            },
        },
        backends={
            "invariant_global": {
                "trainable_parameter_count": invariant_parameters,
            },
            "equivariant_local": {
                "trainable_parameter_count": local_parameters,
            },
        },
        descriptive_ratios={},
    )

    manifest = report.to_manifest()
    assert manifest["interpretation"]["models_parameter_matched"] is expected


def test_failure_path_restores_rng_threads_and_backend_globals(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prior = _process_state()
    reached_failure = False

    def fail_after_global_mutation(*_args: object, **_kwargs: object) -> object:
        nonlocal reached_failure
        reached_failure = True
        assert torch.are_deterministic_algorithms_enabled()
        assert not torch.is_deterministic_algorithms_warn_only_enabled()
        assert torch.get_float32_matmul_precision() == "highest"
        assert torch.get_num_threads() == 1
        torch.rand(7)
        raise RuntimeError("injected comparator failure")

    monkeypatch.setattr(
        backend_comparison_module, "_run_comparison", fail_after_global_mutation
    )
    with pytest.raises(
        BackendComparisonExecutionError, match="injected comparator failure"
    ):
        run_single_device_backend_comparison(_tiny_config(cpu_threads=1))

    assert reached_failure
    _assert_process_state_equal(_process_state(), prior)


def test_cpu_comparison_is_strict_scoped_and_restores_process_globals() -> None:
    config = _tiny_config()
    prior_threads = torch.get_num_threads()
    prior_deterministic = torch.are_deterministic_algorithms_enabled()
    prior_warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    prior_precision = torch.get_float32_matmul_precision()
    prior_cuda_tf32 = torch.backends.cuda.matmul.allow_tf32
    prior_cudnn_tf32 = torch.backends.cudnn.allow_tf32
    prior_cudnn_benchmark = torch.backends.cudnn.benchmark
    prior_cudnn_deterministic = torch.backends.cudnn.deterministic
    prior_rng = torch.get_rng_state().clone()

    report = run_single_device_backend_comparison(config)

    assert torch.get_num_threads() == prior_threads
    assert torch.are_deterministic_algorithms_enabled() == prior_deterministic
    assert torch.is_deterministic_algorithms_warn_only_enabled() == prior_warn_only
    assert torch.get_float32_matmul_precision() == prior_precision
    assert torch.backends.cuda.matmul.allow_tf32 == prior_cuda_tf32
    assert torch.backends.cudnn.allow_tf32 == prior_cudnn_tf32
    assert torch.backends.cudnn.benchmark == prior_cudnn_benchmark
    assert torch.backends.cudnn.deterministic == prior_cudnn_deterministic
    assert torch.equal(torch.get_rng_state(), prior_rng)

    manifest = report.to_manifest()
    assert manifest["schema"] == BACKEND_COMPARISON_SCHEMA
    assert manifest["method"] == BACKEND_COMPARISON_METHOD
    assert manifest["method"].endswith(".v3")
    assert manifest["status"] == "completed"
    assert manifest["config"]["fingerprint"] == config.fingerprint  # type: ignore[index]
    runtime = manifest["runtime"]
    assert runtime["device_type"] == "cpu"
    _assert_sha256(runtime["fingerprint"])
    implementation = runtime["implementation"]
    _assert_sha256(implementation["fingerprint"])
    assert set(implementation["source_modules"]) == {
        "backend_comparison",
        "benchmarking",
        "invariant_transformer",
        "equivariant_transformer",
    }
    for source in implementation["source_modules"].values():
        assert source["bytes"] > 0
        _assert_sha256(source["sha256"])
    assert set(implementation["resolved_model_configs"]) == {
        "invariant_global",
        "equivariant_local",
    }
    workload = manifest["workload"]
    assert workload["generator"].endswith(".v2")
    assert workload["parameter_initializer"] == (
        "integer-affine-modulo-2^24-dyadic-no-rng.v1"
    )
    assert workload["synthetic_signal_initializer"] == (
        "integer-affine-modulo-2^24-dyadic-no-rng.v1"
    )
    assert workload["volume_per_atom"] == 16.0
    assert workload["length_unit"] == "angstrom"
    assert workload["volume_unit"] == "angstrom^3"
    assert workload["seed"] == config.seed
    assert workload["batch_geometries_are_fractional_translations"] is True
    assert workload["cell_volumes"] == pytest.approx(
        [config.atoms_per_structure * config.volume_per_atom] * config.batch_size,
        rel=1.0e-6,
        abs=1.0e-6,
    )
    assert workload["number_density_atoms_per_volume"] == pytest.approx(
        1.0 / config.volume_per_atom
    )
    assert workload["actual_number_density_atoms_per_volume"] == pytest.approx(
        [1.0 / config.volume_per_atom] * config.batch_size
    )
    assert workload["density_derived_from_represented_lattice_determinants"] is True
    assert max(workload["relative_cell_volume_errors"]) <= workload[
        "constant_density_relative_tolerance"
    ]
    assert workload["batch_geometries_are_independent_structural_samples"] is False
    assert workload["phase_execution_order"] == [
        "both_backend_forward_warmups_alternating_AB_BA_starting_AB",
        "paired_forward_measurements_alternating_AB_BA_starting_AB",
        "equivariant_local_forward_prebuilt_interactions_core",
        "both_backend_training_warmups_alternating_AB_BA_starting_AB",
        "paired_training_measurements_alternating_AB_BA_starting_AB",
    ]
    expected_order = [
        "invariant_global_then_equivariant_local",
        "equivariant_local_then_invariant_global",
    ]
    assert workload["paired_forward_backend_order_by_iteration"] == expected_order
    assert workload["paired_training_backend_order_by_iteration"] == expected_order
    tensor_sha256 = workload["tensor_sha256"]
    assert set(tensor_sha256) == {
        "features",
        "fractional",
        "lattices",
        "graph_index",
        "scalar_target",
        "vector_target",
    }
    for digest in tensor_sha256.values():
        _assert_sha256(digest)
    assert set(workload["tensor_manifests"]) == set(tensor_sha256)
    assert set(workload["local_reference_interaction_sha256"]) == {
        "destination",
        "source",
        "shifts",
        "displacement",
        "squared_distance",
    }
    for digest in workload["local_reference_interaction_sha256"].values():
        _assert_sha256(digest)
    _assert_sha256(workload["fingerprint"])

    matching = workload["matching_contract"]
    assert matching["compute_matched"] is False
    assert matching["weights_matched"] is False
    precision_contract = workload["precision_contract"]
    assert precision_contract["comparison_kind"] == "common-master-precision"
    assert precision_contract["backend_arithmetic_precision_matched"] is True

    backends = manifest["backends"]
    invariant = backends["invariant_global"]
    local = backends["equivariant_local"]
    assert invariant["interaction_kind"] == "dense_ordered_atom_pairs"
    assert invariant["interaction_count_per_iteration"] == 4
    assert local["interaction_kind"] == "exact_periodic_image_edges"
    assert local["interaction_count_per_iteration"] > 0
    assert invariant["trainable_parameter_count"] != local["trainable_parameter_count"]

    for backend in (invariant, local):
        phases = backend["phases"]
        assert phases[0]["phase"] == "forward_end_to_end"
        assert phases[-1]["phase"] == "training_step_end_to_end"
        for phase in phases:
            assert len(phase["durations_seconds"]) == config.measurement_steps
            assert all(
                math.isfinite(value) and value > 0.0
                for value in phase["durations_seconds"]
            )
            assert (
                phase["minimum_seconds"]
                <= phase["quartile_25_seconds"]
                <= phase["median_seconds"]
                <= phase["quartile_75_seconds"]
                <= phase["maximum_seconds"]
            )
            assert phase["interactions_per_iteration"] > 0
            assert phase["interactions_per_second"] > 0.0
        numerics = backend["numerics"]
        assert numerics["final_measured_forward_output_finite"] is True
        assert numerics["final_measured_training_output_finite"] is True
        assert numerics[
            "final_measured_trainable_parameter_gradients_finite"
        ] is True
        assert numerics["all_measured_losses_finite"] is True
        assert len(numerics["measured_losses"]) == config.measurement_steps
        memory = backend["memory"]
        assert memory["known_live_logical_tensor_bytes"] > 0
        assert memory["interaction_storage_deduplicated"] is True
        allocator = memory["backend_allocator_training_phase"]
        assert allocator["allocator"] is None
        assert allocator["measurement_count"] == config.measurement_steps
        assert allocator["peak_allocated_bytes"] is None

    assert [phase["phase"] for phase in local["phases"]] == [
        "forward_end_to_end",
        "forward_prebuilt_interactions_core",
        "training_step_end_to_end",
    ]
    interpretation = manifest["interpretation"]
    assert interpretation["predictive_quality_measured"] is False
    assert interpretation["predictive_superiority_established"] is False
    assert matching["parameter_matched"] is (
        invariant["trainable_parameter_count"] == local["trainable_parameter_count"]
    )
    assert interpretation["models_parameter_matched"] is matching["parameter_matched"]
    assert interpretation["models_flop_matched"] is False
    assert interpretation["backend_arithmetic_precision_matched"] is True
    assert interpretation["allocator_values_backend_isolated"] is False
    assert interpretation["allocator_comparison_permitted"] is False
    assert "dense ordered atom pairs" in interpretation["interaction_count_warning"]

    ratios = manifest["descriptive_ratios"]
    for name in (
        "paired_local_over_invariant_forward_time",
        "paired_local_over_invariant_training_time",
    ):
        paired = ratios[name]
        assert paired["iterations"] == config.measurement_steps
        assert paired["unit"] == "dimensionless"
        assert paired["paired_by_measurement_iteration"] is True
        assert len(paired["values"]) == config.measurement_steps
        assert all(math.isfinite(value) and value > 0.0 for value in paired["values"])
        assert paired["quartile_25"] <= paired["median"] <= paired["quartile_75"]

    encoded = report.to_json(indent=None)
    assert json.loads(encoded) == manifest
    assert len(manifest["fingerprint"]) == 64
    assert "NaN" not in encoded and "Infinity" not in encoded


def test_workload_hashes_and_fingerprint_change_with_seed() -> None:
    first_config = _tiny_config(
        measurement_steps=1,
        measure_local_prebuilt_core=False,
        seed=101,
    )
    second_config = _tiny_config(
        measurement_steps=1,
        measure_local_prebuilt_core=False,
        seed=102,
    )

    first = run_single_device_backend_comparison(first_config).to_manifest()["workload"]
    second = run_single_device_backend_comparison(second_config).to_manifest()["workload"]

    assert first["seed"] == 101
    assert second["seed"] == 102
    assert first["tensor_sha256"]["features"] != second["tensor_sha256"]["features"]
    for name in (
        "fractional",
        "lattices",
        "graph_index",
        "scalar_target",
        "vector_target",
    ):
        assert first["tensor_sha256"][name] == second["tensor_sha256"][name]
    assert first["fingerprint"] != second["fingerprint"]


def test_bfloat16_report_declares_backend_native_unmatched_arithmetic() -> None:
    report = run_single_device_backend_comparison(
        _tiny_config(
            precision="bfloat16",
            measurement_steps=1,
            measure_local_prebuilt_core=False,
        )
    )
    manifest = report.to_manifest()
    precision_contract = manifest["workload"]["precision_contract"]

    assert precision_contract["comparison_kind"] == "backend-native-mixed-precision"
    assert precision_contract["backend_arithmetic_precision_matched"] is False
    assert manifest["interpretation"][
        "backend_arithmetic_precision_matched"
    ] is False


def test_cli_writes_backend_comparison_report(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    destination = tmp_path / "comparison.json"
    exit_code = main(
        (
            "benchmark-backends",
            "--device",
            "cpu",
            "--batch-size",
            "1",
            "--atoms-per-structure",
            "2",
            "--input-scalar-dim",
            "4",
            "--scalar-dim",
            "6",
            "--pair-dim",
            "4",
            "--vector-dim",
            "3",
            "--num-heads",
            "1",
            "--num-layers",
            "1",
            "--invariant-feedforward-multiplier",
            "2",
            "--radial-basis-dim",
            "2",
            "--reciprocal-mode-radius",
            "1",
            "--direct-harmonics",
            "1",
            "--cutoff",
            "3.0",
            "--warmup-steps",
            "1",
            "--measurement-steps",
            "1",
            "--cpu-threads",
            "1",
            "--no-local-prebuilt-core",
            "--output",
            str(destination),
        )
    )
    captured = capsys.readouterr()
    assert exit_code == 0, captured.err
    printed = json.loads(captured.out)
    persisted = json.loads(destination.read_text(encoding="utf-8"))
    assert printed == persisted
    assert persisted["schema"] == BACKEND_COMPARISON_SCHEMA
    local_phases = persisted["backends"]["equivariant_local"]["phases"]
    assert [phase["phase"] for phase in local_phases] == [
        "forward_end_to_end",
        "training_step_end_to_end",
    ]
