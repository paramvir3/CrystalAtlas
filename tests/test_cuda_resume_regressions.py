"""Explicit CUDA trainer regressions; these are not crystal-quality benchmarks.

Run in a fresh process with CUBLAS_WORKSPACE_CONFIG=:4096:8.  These tests use
synthetic factor modules to isolate optimizer, stochastic-stream and checkpoint
semantics.  They do not replace production-objective or crystallographic tests.
"""

from __future__ import annotations

import copy
import json
import os
import random
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from test_training import _manifest, _provenance

torch = pytest.importorskip("torch")

from crystalatlas.config import CrystalAtlasConfig  # noqa: E402
from crystalatlas.trainer import (  # noqa: E402
    ComponentLosses,
    PrecisionMode,
    ScalableTrainer,
    StagedTrainingSchedule,
    TrainerConfig,
    TrainingPhase,
)
from crystalatlas.training import (  # noqa: E402
    capture_global_rng_state,
    restore_global_rng_state,
)
from crystalatlas.training_data import DeterministicBucketBatchSampler  # noqa: E402

pytestmark = [
    pytest.mark.ml,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="real CUDA device required"),
]


@pytest.fixture(autouse=True)
def _strict_cuda_runtime():  # type: ignore[no-untyped-def]
    # Do not silently set this after CUDA initialization: the process launcher
    # must establish the deterministic cuBLAS policy before importing PyTorch.
    assert os.environ.get("CUBLAS_WORKSPACE_CONFIG") in {":4096:8", ":16:8"}
    original_rng = capture_global_rng_state()
    original_device = torch.cuda.current_device()
    original_deterministic = torch.are_deterministic_algorithms_enabled()
    original_warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    original_precision = torch.get_float32_matmul_precision()
    original_cuda_tf32 = torch.backends.cuda.matmul.allow_tf32
    original_cudnn_tf32 = torch.backends.cudnn.allow_tf32
    original_cudnn_benchmark = torch.backends.cudnn.benchmark
    try:
        torch.cuda.set_device(0)
        torch.use_deterministic_algorithms(True)
        torch.set_float32_matmul_precision("highest")
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
        yield
        torch.cuda.synchronize()
    finally:
        restore_global_rng_state(original_rng)
        torch.cuda.set_device(original_device)
        torch.use_deterministic_algorithms(original_deterministic, warn_only=original_warn_only)
        torch.set_float32_matmul_precision(original_precision)
        torch.backends.cuda.matmul.allow_tf32 = original_cuda_tf32
        torch.backends.cudnn.allow_tf32 = original_cudnn_tf32
        torch.backends.cudnn.benchmark = original_cudnn_benchmark


@dataclass(frozen=True)
class _Batch:
    x: torch.Tensor
    y: torch.Tensor
    optimizer_normalization_count: float

    @property
    def size(self) -> int:
        return int(self.x.shape[0])


@dataclass
class _SamplerPosition:
    """Synthetic aligned cursor, not a substitute for a stored-data sampler."""

    epoch: int
    cursor: int

    def state_dict(self):  # type: ignore[no-untyped-def]
        return {
            "sampler_fingerprint": "a" * 64,
            "common_policy_fingerprint": "c" * 64,
            "partition_rank": 0,
            "world_size": 1,
            "epoch": self.epoch,
            "cursor": self.cursor,
        }

    def load_state_dict(self, state):  # type: ignore[no-untyped-def]
        assert state["sampler_fingerprint"] == "a" * 64
        self.epoch = int(state["epoch"])
        self.cursor = int(state["cursor"])


class _StochasticFactors(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()

        def factor():  # type: ignore[no-untyped-def]
            return torch.nn.Sequential(
                torch.nn.Linear(4, 8),
                torch.nn.Tanh(),
                torch.nn.Dropout(0.25),
                torch.nn.Linear(8, 1),
            )

        self.composition_model = factor()
        self.cover_selector = factor()
        self.geometry_field = factor()
        self.to(device="cuda:0", dtype=torch.float32)


def _stochastic_objective(models, batch, phase):  # type: ignore[no-untyped-def]
    assert batch.x.device.type == batch.y.device.type == "cuda"
    # Exercise all global streams restored by the checkpoint, including CUDA
    # dropout kernels. All returned losses are reduced in float32 under AMP.
    jitter = 0.01 * (random.random() + np.random.random() + torch.rand(()).item())
    target = batch.y + jitter + 0.01 * torch.rand_like(batch.y)

    def loss(module):  # type: ignore[no-untyped-def]
        output = module(batch.x)
        assert output.device.type == "cuda"
        expected_dtype = (
            torch.get_autocast_dtype("cuda")
            if torch.is_autocast_enabled("cuda")
            else torch.float32
        )
        assert output.dtype is expected_dtype
        return (output.float() - target).square().mean()

    return ComponentLosses(
        batch.size,
        loss(models.composition)
        if phase in {TrainingPhase.COMPOSITION, TrainingPhase.JOINT}
        else None,
        loss(models.cover) if phase in {TrainingPhase.COVER, TrainingPhase.JOINT} else None,
        loss(models.geometry)
        if phase in {TrainingPhase.GEOMETRY, TrainingPhase.JOINT}
        else None,
    )


_stochastic_objective.contract = {  # type: ignore[attr-defined]
    "schema": "crystalatlas.cuda-stochastic-trainer-regression.v1",
    "checkpointable": True,
    "population": "synthetic-structures",
    "loss": "float32-mse-with-cuda-dropout-and-four-global-rng-streams",
}


def _exact_state_equal(expected, actual):  # type: ignore[no-untyped-def]
    if isinstance(expected, torch.Tensor):
        assert isinstance(actual, torch.Tensor)
        assert actual.device == expected.device
        assert actual.dtype == expected.dtype
        assert torch.equal(actual, expected)
    elif isinstance(expected, np.ndarray):
        np.testing.assert_array_equal(actual, expected)
    elif isinstance(expected, dict):
        assert actual.keys() == expected.keys()
        for key in expected:
            _exact_state_equal(expected[key], actual[key])
    elif isinstance(expected, (tuple, list)):
        assert type(actual) is type(expected)
        assert len(actual) == len(expected)
        for lhs, rhs in zip(expected, actual, strict=True):
            _exact_state_equal(lhs, rhs)
    else:
        assert actual == expected


@pytest.mark.parametrize("precision", tuple(PrecisionMode))
def test_cuda_mid_joint_resume_restores_exact_optimizer_rng_and_amp_trajectory(
    tmp_path: Path, precision: PrecisionMode
) -> None:
    if precision is PrecisionMode.BFLOAT16 and not torch.cuda.is_bf16_supported():
        pytest.skip("this CUDA device lacks bfloat16 support")
    random.seed(541)
    np.random.seed(541)
    torch.manual_seed(541)
    torch.cuda.manual_seed_all(541)
    scientific_config = CrystalAtlasConfig()
    schedule = StagedTrainingSchedule(1, 1, 1, 2)
    config = TrainerConfig(accumulation_steps=2, precision=precision, ema_decay=0.9)

    def construct():  # type: ignore[no-untyped-def]
        model = _StochasticFactors()
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, foreach=False, fused=False)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.8)
        trainer = ScalableTrainer(
            model, optimizer, schedule, config=config, scheduler=scheduler
        )
        assert all(parameter.is_cuda for parameter in model.parameters())
        return model, optimizer, scheduler, trainer

    # Three unequal microbatches represent N=6 records and B=3; mu=2. The
    # nominal denominator remains A*mu=4 for the one-microbatch final window.
    batches = tuple(
        _Batch(
            torch.arange(n * 4, device="cuda:0", dtype=torch.float32).reshape(n, 4) / 64.0,
            torch.full((n, 1), 0.125, device="cuda:0"),
            2.0,
        )
        for n in (4, 1, 1)
    )
    source, optimizer, scheduler, trainer = construct()
    for phase in (TrainingPhase.COMPOSITION, TrainingPhase.COVER, TrainingPhase.GEOMETRY):
        assert trainer.current_phase is phase and trainer.phase_epoch == 0
        trainer.train_epoch(batches, _stochastic_objective)
    first_joint = trainer.train_window(batches[:2], _stochastic_objective)
    assert first_joint.gradient_normalization_count == 4.0
    assert trainer.cursor.epoch == 3 and trainer.cursor.microbatch == 2
    assert trainer.current_phase is TrainingPhase.JOINT and trainer.phase_epoch == 0
    # Each factor owns a phase-local epoch sequence: global epoch 3 is the
    # first joint epoch, so its sampler epoch is 0, not 3.
    sampler = _SamplerPosition(0, 2)
    path = tmp_path / f"cuda-{precision.value}-exact-resume.pt"
    info = trainer.save_checkpoint(
        path,
        scientific_config=scientific_config,
        scientific_manifest=_manifest(),
        provenance=_provenance(scientific_config, 541),
        sampler=sampler,
    )
    assert info is not None
    checkpoint_rng = capture_global_rng_state()
    reference_reports = (
        trainer.train_epoch(batches[2:], _stochastic_objective),
        trainer.train_epoch(batches, _stochastic_objective),
    )
    expected_rng = capture_global_rng_state()
    expected_model = copy.deepcopy(source.state_dict())
    expected_optimizer = copy.deepcopy(optimizer.state_dict())
    expected_trainer = copy.deepcopy(trainer.state_dict())
    expected_scheduler = copy.deepcopy(scheduler.state_dict())

    # Construction and explicit draws deliberately perturb every live stream.
    resumed, resumed_optimizer, resumed_scheduler, resumed_trainer = construct()
    random.random()
    np.random.random(7)
    torch.rand(11)
    torch.rand(13, device="cuda:0")
    resumed_sampler = _SamplerPosition(0, 0)
    receipt = resumed_trainer.load_checkpoint(
        path,
        scientific_manifest=_manifest(),
        scientific_config=scientific_config,
        expected_sha256=info.sha256,
        sampler=resumed_sampler,
    )
    assert receipt.optimizer_restored and receipt.rng_restored
    assert resumed_sampler == sampler
    assert resumed_trainer.cursor.epoch == 3 and resumed_trainer.cursor.microbatch == 2
    assert resumed_trainer.current_phase is TrainingPhase.JOINT
    assert resumed_trainer.phase_epoch == resumed_sampler.epoch == 0
    _exact_state_equal(checkpoint_rng, capture_global_rng_state())
    actual_reports = (
        resumed_trainer.train_epoch(batches[2:], _stochastic_objective),
        resumed_trainer.train_epoch(batches, _stochastic_objective),
    )
    _exact_state_equal(expected_rng, capture_global_rng_state())
    _exact_state_equal(expected_model, resumed.state_dict())
    _exact_state_equal(expected_optimizer, resumed_optimizer.state_dict())
    _exact_state_equal(expected_scheduler, resumed_scheduler.state_dict())
    _exact_state_equal(expected_trainer, resumed_trainer.state_dict())
    assert reference_reports == actual_reports
    assert all(value["exp_avg"].is_cuda for value in resumed_optimizer.state.values())
    assert (resumed_trainer._scaler is not None) is (precision is PrecisionMode.FLOAT16)
    if resumed_trainer._scaler is not None:
        assert resumed_trainer._scaler.is_enabled()


@pytest.mark.parametrize("dtype", (torch.float32, torch.float64))
def test_cuda_unequal_topology_batches_keep_uniform_record_coefficients(dtype) -> None:  # type: ignore[no-untyped-def]
    model = torch.nn.Module()
    for name in ("composition_model", "cover_selector", "geometry_field"):
        layer = torch.nn.Linear(1, 1, bias=False, device="cuda:0", dtype=dtype)
        with torch.no_grad():
            layer.weight.zero_()
        setattr(model, name, layer)
    trainer = ScalableTrainer(
        model,
        torch.optim.SGD(model.parameters(), lr=0.0),
        StagedTrainingSchedule(1, 0, 0),
        config=TrainerConfig(ema_decay=None),
    )
    planner = object.__new__(DeterministicBucketBatchSampler)
    planner.dataset = SimpleNamespace(
        descriptors=tuple(
            SimpleNamespace(record_id=str(i), bucket_key=0 if i < 4 else 1) for i in range(5)
        )
    )
    planner._eligible_indices = tuple(range(5))
    planner.batch_size = 4
    planner.drop_last = False
    planner.world_size = 1
    planner.rank = 0
    planner.seed = 31
    coefficients = torch.tensor(
        [[1.0], [1.0], [1.0], [1.0], [10.0]], device="cuda:0", dtype=dtype
    )

    def objective(models, batch, phase):  # type: ignore[no-untyped-def]
        assert phase is TrainingPhase.COMPOSITION
        output = models.composition(batch.x)
        assert output.is_cuda
        return ComponentLosses(batch.size, (output - batch.y).square().mean(), None, None)

    reports = []
    for planned in planner._build_plan(0):
        selected = coefficients[list(planned.indices)]
        reports.append(
            trainer.train_window(
                (
                    _Batch(
                        torch.ones_like(selected),
                        -selected / 2.0,
                        planned.optimizer_normalization_count,
                    ),
                ),
                objective,
            )
        )
    assert sorted(report.global_structures for report in reports) == [1, 4]
    assert all(report.gradient_normalization_count == 2.5 for report in reports)
    gradient = sum(report.gradients_before_clip.global_l2_norm for report in reports) / 2.0
    assert gradient == pytest.approx(2.8, rel=0, abs=2e-7 if dtype is torch.float32 else 1e-14)


def test_cuda_real_grad_scaler_overflow_retries_without_consuming_rng() -> None:
    model = torch.nn.Module()
    for name in ("composition_model", "cover_selector", "geometry_field"):
        layer = torch.nn.Linear(1, 1, bias=False, device="cuda:0")
        with torch.no_grad():
            layer.weight.zero_()
        setattr(model, name, layer)
    trainer = ScalableTrainer(
        model,
        torch.optim.SGD(model.parameters(), lr=0.1),
        StagedTrainingSchedule(1, 0, 0),
        config=TrainerConfig(precision=PrecisionMode.FLOAT16, ema_decay=None),
    )
    scaler = trainer._scaler
    assert scaler is not None and scaler.is_enabled()
    initial_scale = scaler.get_scale()
    draws = []

    def objective(models, batch, phase):  # type: ignore[no-untyped-def]
        assert phase is TrainingPhase.COMPOSITION
        draws.append(torch.rand(8, device="cuda:0"))
        # sqrt(0) has finite forward value but infinite derivative. The next
        # call is a controlled recovery witness, not a production objective.
        loss = (
            torch.sqrt(models.composition.weight).sum()
            if len(draws) == 1
            else (models.composition.weight - 1.0).square().sum()
        )
        return ComponentLosses(batch.size, loss, None, None)

    # This closure deliberately changes behavior after its first invocation.
    # Declare that stateful diagnostic policy explicitly; it cannot back an
    # exact-resume checkpoint or a scientific training objective.
    objective.contract = {  # type: ignore[attr-defined]
        "schema": "crystalatlas.cuda-real-scaler-overflow-diagnostic.v1",
        "checkpointable": False,
        "population": "synthetic-single-record",
        "stateful_policy": "first-call-infinite-derivative-then-finite-quadratic",
        "rng_policy": "record-cuda-draws-to-test-replayed-overflow-retry",
    }
    objective.contract_fingerprint = sha256(  # type: ignore[attr-defined]
        json.dumps(objective.contract, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    batch = _Batch(torch.ones(1, 1, device="cuda:0"), torch.ones(1, 1, device="cuda:0"), 1.0)
    report = trainer.train_window((batch,), objective)
    assert report.optimizer_step == 1
    assert len(draws) == 2
    assert torch.equal(draws[0], draws[1])
    assert scaler.get_scale() == initial_scale * scaler.get_backoff_factor()
    assert torch.isfinite(model.composition_model.weight).all()
    assert model.composition_model.weight.item() == pytest.approx(0.2)
