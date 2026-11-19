from __future__ import annotations

import hashlib
import json
import math
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

import crystalatlas.benchmarking as benchmarking_module  # noqa: E402
from crystalatlas.benchmarking import (  # noqa: E402
    BENCHMARK_METHOD,
    BENCHMARK_SCHEMA,
    GeometryBenchmarkConfig,
    run_single_device_geometry_benchmark,
)
from crystalatlas.cli import main  # noqa: E402
from crystalatlas.errors import DomainError  # noqa: E402

pytestmark = pytest.mark.ml


def _tiny_cpu_config() -> GeometryBenchmarkConfig:
    return GeometryBenchmarkConfig(
        device="cpu",
        precision="float32",
        batch_size=1,
        atoms_per_structure=2,
        input_scalar_dim=4,
        scalar_dim=6,
        vector_dim=3,
        num_heads=1,
        num_layers=1,
        radial_basis_dim=2,
        cutoff=3.0,
        cell_length=6.0,
        warmup_steps=1,
        measurement_steps=2,
        seed=31,
        cpu_threads=1,
    )


@pytest.mark.parametrize(
    ("updates", "message"),
    [
        ({"batch_size": True}, "batch_size"),
        ({"atoms_per_structure": 1}, "at least two"),
        ({"precision": "tf32"}, "precision"),
        ({"scalar_dim": 7}, "divisible"),
        ({"warmup_steps": 0}, "warmup_steps"),
        ({"cutoff": float("nan")}, "cutoff"),
        ({"cpu_threads": False}, "cpu_threads"),
    ],
)
def test_benchmark_config_fails_closed(updates: dict[str, object], message: str) -> None:
    values: dict[str, object] = {
        "batch_size": 2,
        "atoms_per_structure": 4,
        "scalar_dim": 12,
        "vector_dim": 6,
        "num_heads": 3,
    }
    values.update(updates)
    with pytest.raises(DomainError, match=message):
        GeometryBenchmarkConfig(**values)  # type: ignore[arg-type]


def test_cuda_policy_requires_a_deterministic_cublas_workspace_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    device = SimpleNamespace(type="cuda")
    monkeypatch.delenv("CUBLAS_WORKSPACE_CONFIG", raising=False)
    with pytest.raises(DomainError, match="CUBLAS_WORKSPACE_CONFIG"):
        benchmarking_module._precision_policy(_tiny_cpu_config(), device)

    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    master_dtype, autocast_dtype = benchmarking_module._precision_policy(
        _tiny_cpu_config(), device
    )
    assert master_dtype is torch.float32
    assert autocast_dtype is None


def test_dyadic_modular_signal_has_frozen_platform_stable_bytes() -> None:
    specifications = (
        (
            3_504,
            11_406_733,
            1_729 * 104_729 + 8_191,
            -25,
            "d323bd5df0d12bb387f80c37bb746320981ca0cd78f826c04afb19247fb77f84",
        ),
        (
            4_096,
            12_820_163,
            1_234_567,
            -26,
            "f5711393be6d52f83457d43acf14995e89072a59bbeba3c880ce0a62c6ee7f39",
        ),
        (
            1_536,
            1_103_515,
            7_654_321,
            -27,
            "bfd35a482cfc560240842e8309331d68fdc2e267ba4499604cdf879a833eba8b",
        ),
    )

    for length, multiplier, offset, exponent, expected_sha256 in specifications:
        signal = benchmarking_module._dyadic_modular_signal(
            length,
            multiplier=multiplier,
            offset=offset,
            scale_exponent=exponent,
        )
        assert signal.dtype is torch.float64
        assert signal.device.type == "cpu"
        assert hashlib.sha256(signal.numpy().tobytes()).hexdigest() == expected_sha256


@pytest.mark.parametrize(
    ("updates", "message"),
    [
        ({"length": 0}, "length"),
        ({"length": torch.iinfo(torch.int64).max + 1}, "arange"),
        ({"multiplier": 2}, "positive odd"),
        ({"offset": False}, "offset"),
        ({"scale_exponent": -127}, "normal finite exact Float32"),
        ({"scale_exponent": 105}, "normal finite exact Float32"),
    ],
)
def test_dyadic_modular_signal_fails_closed(
    updates: dict[str, object], message: str
) -> None:
    arguments: dict[str, object] = {
        "length": 8,
        "multiplier": 11_406_733,
        "offset": 37,
        "scale_exponent": -25,
    }
    arguments.update(updates)
    with pytest.raises(DomainError, match=message):
        benchmarking_module._dyadic_modular_signal(**arguments)  # type: ignore[arg-type]


def test_model_initializer_has_frozen_bytes_and_does_not_consume_rng() -> None:
    model = torch.nn.ParameterList(
        (
            torch.nn.Parameter(torch.empty(7, dtype=torch.float32)),
            torch.nn.Parameter(torch.empty((3, 5), dtype=torch.float64)),
        )
    )
    prior_rng = torch.get_rng_state().clone()

    benchmarking_module._initialize_model(model, seed=37)

    assert torch.equal(torch.get_rng_state(), prior_rng)
    assert benchmarking_module._model_sha256(model) == (
        "828422f8ea47a01760c0f3311f89a883a19e1950cf0afe594156b4f86ade3cda"
    )


def test_cpu_benchmark_emits_canonical_scoped_evidence_and_restores_globals() -> None:
    config = _tiny_cpu_config()
    prior_threads = torch.get_num_threads()
    prior_deterministic = torch.are_deterministic_algorithms_enabled()
    prior_precision = torch.get_float32_matmul_precision()
    prior_rng = torch.get_rng_state().clone()

    report = run_single_device_geometry_benchmark(config)

    assert torch.get_num_threads() == prior_threads
    assert torch.are_deterministic_algorithms_enabled() == prior_deterministic
    assert torch.get_float32_matmul_precision() == prior_precision
    assert torch.equal(torch.get_rng_state(), prior_rng)

    manifest = report.to_manifest()
    assert manifest["schema"] == BENCHMARK_SCHEMA
    assert manifest["method"] == BENCHMARK_METHOD
    assert manifest["method"].endswith(".v2")
    assert manifest["status"] == "completed"
    assert manifest["factor"] == "geometry"
    assert manifest["config"]["fingerprint"] == config.fingerprint  # type: ignore[index]
    assert manifest["runtime"]["device_type"] == "cpu"  # type: ignore[index]
    assert manifest["runtime"]["deterministic_algorithms"] is True  # type: ignore[index]
    assert (  # type: ignore[index]
        manifest["runtime"]["deterministic_cublas_workspace_config_valid"] is True
    )
    assert manifest["workload"]["exact_periodic_edges_per_iteration"] > 0  # type: ignore[index,operator]
    assert manifest["workload"]["workload_generator"].endswith(".v2")  # type: ignore[index]
    assert manifest["interpretation"]["multi_device_scaling_measured"] is False  # type: ignore[index]
    assert (  # type: ignore[index]
        manifest["interpretation"]["multi_device_extrapolation_permitted"] is False
    )
    assert (  # type: ignore[index]
        manifest["interpretation"]["full_crystalatlas_training_objective_measured"]
        is False
    )

    phases = manifest["phases"]
    assert isinstance(phases, list)
    assert [phase["phase"] for phase in phases] == [
        "periodic_edge_enumeration",
        "transformer_forward_precomputed_edges",
        "training_step_end_to_end",
    ]
    for phase in phases:
        assert phase["iterations"] == config.measurement_steps
        assert len(phase["durations_seconds"]) == config.measurement_steps
        assert all(math.isfinite(value) and value > 0.0 for value in phase["durations_seconds"])
        assert phase["structures_per_second"] > 0.0

    memory = manifest["memory"]
    assert memory["known_live_logical_tensor_bytes"] > 0  # type: ignore[index,operator]
    allocator = memory["backend_allocator"]  # type: ignore[index]
    assert allocator["allocator"] is None
    assert allocator["peak_allocated_bytes"] is None
    assert "exclude activations" in allocator["peak_scope"]
    encoded = report.to_json(indent=None)
    assert json.loads(encoded) == manifest
    assert "NaN" not in encoded and "Infinity" not in encoded


def test_constructive_cpu_workload_has_repeatable_numerical_identity() -> None:
    config = _tiny_cpu_config()
    first = run_single_device_geometry_benchmark(config).to_manifest()
    second = run_single_device_geometry_benchmark(config).to_manifest()

    assert first["config"] == second["config"]
    assert first["workload"] == second["workload"]
    assert first["numerics"] == second["numerics"]
    # Wall-clock samples and the report fingerprint are correctly not expected
    # to be equal across two otherwise identical runs.
    assert first["phases"] != []


def test_unavailable_accelerator_is_rejected_instead_of_falling_back() -> None:
    if torch.cuda.is_available():
        pytest.skip("the test host exposes CUDA")
    config = GeometryBenchmarkConfig(device="cuda", warmup_steps=1, measurement_steps=1)
    with pytest.raises(DomainError, match="CUDA was requested but is unavailable"):
        run_single_device_geometry_benchmark(config)


def test_cli_persists_only_a_successful_canonical_report(tmp_path, capsys) -> None:
    destination = tmp_path / "benchmark.json"
    exit_code = main(
        (
            "benchmark-geometry",
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
            "--vector-dim",
            "3",
            "--num-heads",
            "1",
            "--num-layers",
            "1",
            "--radial-basis-dim",
            "2",
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
    assert len(persisted["fingerprint"]) == 64
