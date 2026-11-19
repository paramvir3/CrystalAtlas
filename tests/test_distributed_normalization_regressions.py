"""Two-process CPU/Gloo numerical tests, not multi-GPU scaling evidence.

The production planner supplies unequal topology batches and its global
population normalizer. The production trainer executes real DDP collectives.
For l_i(w)=(w+c_i/2)^2, the exact per-record gradient is 2w+c_i;
this independent scalar formula is the reference for every optimizer window.
"""

from __future__ import annotations

import multiprocessing
import os
import queue
import socket
import sys
import time
import traceback
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from crystalatlas.trainer import (  # noqa: E402
    ComponentLosses,
    DistributedConfig,
    ScalableTrainer,
    StagedTrainingSchedule,
    TrainerConfig,
    TrainingPhase,
)
from crystalatlas.training_data import DeterministicBucketBatchSampler  # noqa: E402

pytestmark = pytest.mark.ml

_WORLD_SIZE = 2
_LEARNING_RATE = 0.005


def _plan(groups: tuple[int, ...], rank: int):  # type: ignore[no-untyped-def]
    # The planner's arithmetic is independent of CIF/store authentication,
    # which has its own integration tests. Exercise its actual implementation.
    planner = object.__new__(DeterministicBucketBatchSampler)
    planner.dataset = SimpleNamespace(
        descriptors=tuple(
            SimpleNamespace(record_id=str(i), bucket_key=group)
            for i, group in enumerate(groups)
        )
    )
    planner._eligible_indices = tuple(range(len(groups)))
    planner.batch_size = 4
    planner.drop_last = False
    planner.world_size = _WORLD_SIZE
    planner.rank = rank
    planner.seed = 41
    return planner._build_plan(0)


@dataclass(frozen=True)
class _Batch:
    x: torch.Tensor
    y: torch.Tensor
    optimizer_normalization_count: float

    @property
    def size(self) -> int:
        return int(self.x.shape[0])


def _objective(models, batch, phase):  # type: ignore[no-untyped-def]
    assert phase is TrainingPhase.COMPOSITION
    output = models.composition(batch.x)
    assert output.device.type == "cpu" and output.dtype == torch.float64
    return ComponentLosses(batch.size, (output - batch.y).square().mean(), None, None)


def _worker(
    rank: int,
    rendezvous_uri: str,
    groups: tuple[int, ...],
    coefficients: tuple[float, ...],
    accumulation_steps: int,
    results,
) -> None:  # type: ignore[no-untyped-def]
    outcome: dict[str, object]
    try:
        torch.set_num_threads(1)
        torch.use_deterministic_algorithms(True)
        torch.distributed.init_process_group(
            "gloo",
            init_method=rendezvous_uri,
            rank=rank,
            world_size=_WORLD_SIZE,
            timeout=timedelta(seconds=20),
        )
        model = torch.nn.Module()
        for name in ("composition_model", "cover_selector", "geometry_field"):
            layer = torch.nn.Linear(1, 1, bias=False, dtype=torch.float64, device="cpu")
            with torch.no_grad():
                layer.weight.zero_()
            setattr(model, name, layer)
        trainer = ScalableTrainer(
            model,
            torch.optim.SGD(model.parameters(), lr=_LEARNING_RATE),
            StagedTrainingSchedule(1, 0, 0),
            config=TrainerConfig(
                accumulation_steps=accumulation_steps,
                ema_decay=None,
                distributed=DistributedConfig(enabled=True),
            ),
        )
        assert trainer.world_size == _WORLD_SIZE and trainer.rank == rank
        assert isinstance(
            models := trainer.models.composition, torch.nn.parallel.DistributedDataParallel
        )
        assert next(models.parameters()).device.type == "cpu"
        planned = _plan(groups, rank)
        records = []
        for start in range(0, len(planned), accumulation_steps):
            window = planned[start : start + accumulation_steps]
            batches = []
            for item in window:
                coefficient = torch.tensor(
                    [coefficients[index] for index in item.indices], dtype=torch.float64
                ).unsqueeze(-1)
                batches.append(
                    _Batch(
                        torch.ones_like(coefficient),
                        -coefficient / 2.0,
                        item.optimizer_normalization_count,
                    )
                )
            report = trainer.train_window(batches, _objective)
            records.append(
                {
                    "indices": [list(item.indices) for item in window],
                    "normalizer": report.gradient_normalization_count,
                    "normalization_policy": report.gradient_normalization_policy,
                    "gradient_norm": report.gradients_before_clip.global_l2_norm,
                    "global_structures": report.global_structures,
                    "local_structures": sum(batch.size for batch in batches),
                    "microbatches": report.microbatches,
                    "weight": model.composition_model.weight.item(),
                    "optimizer_step": report.optimizer_step,
                }
            )
        outcome = {
            "rank": rank,
            "pid": os.getpid(),
            "backend": torch.distributed.get_backend(),
            "device": next(model.parameters()).device.type,
            "world_size": trainer.world_size,
            "records": records,
            "error": None,
        }
    except BaseException:
        # Report assertions and backend exceptions to pytest. Parent also
        # checks exit codes, so a native crash cannot be mistaken for a pass.
        outcome = {"rank": rank, "error": traceback.format_exc()}
    finally:
        if torch.distributed.is_initialized():
            try:
                torch.distributed.destroy_process_group()
            except BaseException:
                outcome = {"rank": rank, "error": traceback.format_exc()}
    results.put(outcome)


def _run_workers(
    tmp_path: Path,
    groups: tuple[int, ...],
    coefficients: tuple[float, ...],
    accumulation_steps: int,
):  # type: ignore[no-untyped-def]
    context = multiprocessing.get_context("spawn")
    results = context.Queue()
    rendezvous = tmp_path / "two-process-gloo-store"
    assert not rendezvous.exists()
    processes = [
        context.Process(
            target=_worker,
            args=(
                rank,
                rendezvous.resolve().as_uri(),
                groups,
                coefficients,
                accumulation_steps,
                results,
            ),
            name=f"crystalatlas-cpu-gloo-{rank}",
        )
        for rank in range(_WORLD_SIZE)
    ]
    outcomes = []
    deadline = time.monotonic() + 55.0
    try:
        for process in processes:
            process.start()
        while len(outcomes) < _WORLD_SIZE:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                pytest.fail("two-process CPU/Gloo regression exceeded 55-second deadline")
            try:
                outcome = results.get(timeout=min(remaining, 1.0))
            except queue.Empty:
                failures = [
                    (process.name, process.exitcode)
                    for process in processes
                    if process.exitcode not in {None, 0}
                ]
                if failures:
                    pytest.fail(
                        f"CPU/Gloo worker crashed without a passing receipt: {failures}"
                    )
                continue
            outcomes.append(outcome)
            if outcome["error"] is not None:
                pytest.fail(f"CPU/Gloo rank {outcome['rank']} failed:\n{outcome['error']}")
        for process in processes:
            process.join(timeout=max(0.0, deadline - time.monotonic()))
            if process.is_alive():
                pytest.fail(f"CPU/Gloo worker {process.name} failed to exit by the deadline")
            assert process.exitcode == 0
    finally:
        # These exact Process objects were created by this test. Never leave a
        # peer blocked in a collective after a test failure or interrupted run.
        for process in processes:
            if process.pid is not None and process.is_alive():
                process.terminate()
        leaked_workers = []
        for process in processes:
            if process.pid is not None:
                process.join(timeout=3.0)
                if process.is_alive():
                    process.kill()
                    process.join(timeout=3.0)
                if process.is_alive():
                    leaked_workers.append(process.pid)
        results.close()
        results.join_thread()
        assert not leaked_workers, f"could not stop test workers {leaked_workers}"
    return sorted(outcomes, key=lambda value: value["rank"])


@pytest.mark.parametrize(
    ("groups", "coefficients", "accumulation_steps"),
    [
        ((0, 0, 0, 0, 1), (1.0, 1.0, 1.0, 1.0, 10.0), 1),
        ((0, 0, 0, 0, 1, 1, 1, 1, 2, 3, 4), tuple(float(i) for i in range(1, 12)), 2),
    ],
    ids=("rank-unequal-single-window", "accumulation-and-short-final-window"),
)
def test_two_cpu_gloo_processes_match_uniform_record_gradient_reference(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    groups: tuple[int, ...],
    coefficients: tuple[float, ...],
    accumulation_steps: int,
) -> None:
    if sys.platform == "darwin":
        pytest.skip("the established macOS PyTorch build aborts inside ProcessGroupGloo")
    if not torch.distributed.is_available() or not torch.distributed.is_gloo_available():
        pytest.skip("PyTorch CPU/Gloo distributed backend is unavailable")
    if not hasattr(socket, "if_nameindex"):
        pytest.skip("named loopback enumeration is unavailable on this platform")
    interfaces = {name for _, name in socket.if_nameindex()}
    loopback = "lo" if "lo" in interfaces else "lo0" if "lo0" in interfaces else None
    if loopback is None:
        pytest.skip("no named loopback interface is available for Gloo")
    monkeypatch.setenv("GLOO_SOCKET_IFNAME", loopback)
    plans = [_plan(groups, rank) for rank in range(_WORLD_SIZE)]
    assert len({len(plan) for plan in plans}) == 1
    global_batches = [batch for plan in plans for batch in plan]
    all_indices = [index for batch in global_batches for index in batch.indices]
    assert sorted(all_indices) == list(range(len(coefficients)))
    assert len(all_indices) == len(set(all_indices))
    population_normalizer = len(coefficients) / len(global_batches)
    assert all(
        batch.optimizer_normalization_count == population_normalizer for batch in global_batches
    )
    denominator = _WORLD_SIZE * accumulation_steps * population_normalizer
    outcomes = _run_workers(tmp_path, groups, coefficients, accumulation_steps)
    assert [outcome["rank"] for outcome in outcomes] == [0, 1]
    assert len({outcome["pid"] for outcome in outcomes}) == _WORLD_SIZE
    assert all(outcome["pid"] != os.getpid() for outcome in outcomes)
    assert all(
        outcome["backend"] == "gloo"
        and outcome["device"] == "cpu"
        and outcome["world_size"] == _WORLD_SIZE
        for outcome in outcomes
    )
    expected_windows = (len(plans[0]) + accumulation_steps - 1) // accumulation_steps
    assert all(len(outcome["records"]) == expected_windows for outcome in outcomes)
    expected_weight = 0.0
    frozen_gradient_sum = 0.0
    rank_unequal_window_observed = False
    for window_index, start in enumerate(range(0, len(plans[0]), accumulation_steps)):
        windows = [plan[start : start + accumulation_steps] for plan in plans]
        selected = [index for window in windows for batch in window for index in batch.indices]
        coefficient_sum = sum(coefficients[index] for index in selected)
        expected_gradient = (
            2.0 * expected_weight * len(selected) + coefficient_sum
        ) / denominator
        assert expected_gradient > 0.0  # gradient norm equals the signed scalar here
        previous_weight = expected_weight
        expected_weight -= _LEARNING_RATE * expected_gradient
        receipts = [outcome["records"][window_index] for outcome in outcomes]
        rank_unequal_window_observed |= (
            receipts[0]["local_structures"] != receipts[1]["local_structures"]
        )
        for rank, receipt in enumerate(receipts):
            assert receipt["indices"] == [list(batch.indices) for batch in windows[rank]]
            assert receipt["normalizer"] == denominator
            assert (
                receipt["normalization_policy"] == "sampler-population-fixed-nominal-window.v1"
            )
            assert receipt["global_structures"] == len(selected)
            assert receipt["local_structures"] == sum(
                len(batch.indices) for batch in windows[rank]
            )
            assert receipt["microbatches"] == len(windows[rank])
            assert receipt["optimizer_step"] == window_index + 1
            assert receipt["gradient_norm"] == pytest.approx(
                expected_gradient, rel=0.0, abs=1e-12
            )
            assert receipt["weight"] == pytest.approx(expected_weight, rel=0.0, abs=1e-12)
        # Remove the known change of evaluation point between SGD updates.
        # What remains must assign every retained record coefficient 1/D.
        frozen_gradient_sum += receipts[0]["gradient_norm"] - (
            2.0 * previous_weight * len(selected) / denominator
        )
    assert rank_unequal_window_observed
    uniform_record_gradient = sum(coefficients) / len(coefficients)
    assert frozen_gradient_sum * denominator / len(coefficients) == pytest.approx(
        uniform_record_gradient, rel=0.0, abs=1e-12
    )
    if accumulation_steps == 1:
        assert outcomes[0]["records"][0]["gradient_norm"] == pytest.approx(
            2.8, rel=0.0, abs=1e-12
        )
    else:
        assert len(plans[0]) % accumulation_steps == 1
        assert outcomes[0]["records"][-1]["microbatches"] == 1
        assert outcomes[0]["records"][-1]["normalizer"] == denominator
