from __future__ import annotations

import socket
import sys
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import pytest
from test_training import _manifest, _provenance

torch = pytest.importorskip("torch")

from crystalatlas.config import CrystalAtlasConfig  # noqa: E402
from crystalatlas.errors import DomainError  # noqa: E402
from crystalatlas.trainer import (  # noqa: E402
    ComponentLosses,
    CrystallographicAuditResult,
    DistributedConfig,
    PrecisionMode,
    ScalableTrainer,
    StagedTrainingSchedule,
    TrainerConfig,
    TrainerCursor,
    TrainingPhase,
)
from crystalatlas.training import CheckpointError, TrainingNumericsError  # noqa: E402
from crystalatlas.training_data import DeterministicBucketBatchSampler  # noqa: E402

pytestmark = pytest.mark.ml


class TinyFactorModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.composition_model = torch.nn.Linear(1, 1, bias=False).double()
        self.cover_selector = torch.nn.Linear(1, 1, bias=False).double()
        self.geometry_field = torch.nn.Linear(1, 1, bias=False).double()
        with torch.no_grad():
            for module in (
                self.composition_model,
                self.cover_selector,
                self.geometry_field,
            ):
                module.weight.zero_()


@dataclass(frozen=True)
class TinyBatch:
    x: torch.Tensor
    y: torch.Tensor
    optimizer_normalization_count: float | None = None

    @property
    def size(self) -> int:
        return int(self.x.shape[0])


@dataclass
class CursorSampler:
    epoch: int
    cursor: int
    sampler_fingerprint: str = "a" * 64
    common_policy_fingerprint: str = "c" * 64
    partition_rank: int = 0
    world_size: int = 1

    def state_dict(self):  # type: ignore[no-untyped-def]
        return {
            "sampler_fingerprint": self.sampler_fingerprint,
            "common_policy_fingerprint": self.common_policy_fingerprint,
            "partition_rank": self.partition_rank,
            "world_size": self.world_size,
            "epoch": self.epoch,
            "cursor": self.cursor,
        }

    def load_state_dict(self, state):  # type: ignore[no-untyped-def]
        if state["sampler_fingerprint"] != self.sampler_fingerprint:
            raise RuntimeError("sampler policy changed")
        self.epoch = int(state["epoch"])
        self.cursor = int(state["cursor"])


def _objective(models, batch: TinyBatch, phase: TrainingPhase):  # type: ignore[no-untyped-def]
    def loss(module):  # type: ignore[no-untyped-def]
        return (module(batch.x) - batch.y).square().mean()

    composition = (
        loss(models.composition)
        if phase
        in {
            TrainingPhase.COMPOSITION,
            TrainingPhase.JOINT,
        }
        else None
    )
    cover = (
        loss(models.cover)
        if phase
        in {
            TrainingPhase.COVER,
            TrainingPhase.JOINT,
        }
        else None
    )
    geometry = (
        loss(models.geometry)
        if phase
        in {
            TrainingPhase.GEOMETRY,
            TrainingPhase.JOINT,
        }
        else None
    )
    return ComponentLosses(batch.size, composition, cover, geometry)


_objective.contract = {  # type: ignore[attr-defined]
    "schema": "crystalatlas.test-objective.v1",
    "checkpointable": True,
    "loss": "mean-squared-error",
    "population": "structures",
}


def _batch(values: tuple[float, ...]) -> TinyBatch:
    x = torch.tensor(values, dtype=torch.float64).unsqueeze(-1)
    return TinyBatch(x, torch.ones_like(x))


def _optimizer(model: TinyFactorModel, *, lr: float = 0.1):  # type: ignore[no-untyped-def]
    return torch.optim.SGD(model.parameters(), lr=lr)


def _fragmented_plan(*, world_size=1, rank=0):  # type: ignore[no-untyped-def]
    # Isolate the actual sampler's planning rule from the independently tested
    # CIF/store boundary. Integer bucket keys suffice for _build_plan itself.
    planner = object.__new__(DeterministicBucketBatchSampler)
    planner.dataset = SimpleNamespace(  # type: ignore[assignment]
        descriptors=tuple(
            SimpleNamespace(record_id=str(i), bucket_key=0 if i < 4 else 1)
            for i in range(5)
        )
    )
    planner._eligible_indices = tuple(range(5))
    planner.batch_size = 4
    planner.drop_last = False
    planner.world_size = world_size
    planner.rank = rank
    planner.seed = 31
    return planner._build_plan(0)


def test_topology_population_normalizer_restores_uniform_record_gradients() -> None:
    model = TinyFactorModel()
    trainer = ScalableTrainer(
        model, _optimizer(model, lr=0.0), StagedTrainingSchedule(1, 0, 0),
        config=TrainerConfig(ema_decay=None),
    )
    coefficients = torch.tensor([[1.0], [1.0], [1.0], [1.0], [10.0]]).double()
    reports = []
    for planned in _fragmented_plan():
        values = coefficients[list(planned.indices)]
        # d/dw mean((w-y)^2) at w=0 is the chosen positive coefficient.
        batch = TinyBatch(
            torch.ones_like(values), -values / 2.0, planned.optimizer_normalization_count
        )
        reports.append(trainer.train_window((batch,), _objective))
    assert sorted(report.global_structures for report in reports) == [1, 4]
    assert [report.gradient_normalization_count for report in reports] == [2.5, 2.5]
    mean_gradient = sum(report.gradients_before_clip.global_l2_norm for report in reports) / 2
    assert mean_gradient == pytest.approx(2.8, abs=1e-14)
    assert all(report.gradient_normalization_policy ==
               "sampler-population-fixed-nominal-window.v1" for report in reports)


def test_population_normalizer_does_not_promote_short_final_windows() -> None:
    model = TinyFactorModel()
    trainer = ScalableTrainer(
        model, _optimizer(model, lr=0.0), StagedTrainingSchedule(1, 0, 0),
        config=TrainerConfig(accumulation_steps=2, ema_decay=None),
    )
    def batch(values):  # type: ignore[no-untyped-def]
        coefficient = torch.tensor(values, dtype=torch.float64).unsqueeze(-1)
        return TinyBatch(torch.ones_like(coefficient), -coefficient / 2.0, 2.0)

    first = trainer.train_window((batch([1.0] * 4), batch([10.0])), _objective)
    last = trainer.train_window((batch([2.0]),), _objective)
    assert first.gradient_normalization_count == last.gradient_normalization_count == 4.0
    assert first.gradients_before_clip.global_l2_norm == pytest.approx(3.5)
    assert last.gradients_before_clip.global_l2_norm == pytest.approx(0.5)


def test_distributed_plan_attaches_one_global_not_rank_local_normalizer() -> None:
    plans = [_fragmented_plan(world_size=3, rank=rank) for rank in range(3)]
    assert [len(plan) for plan in plans] == [1, 1, 1]
    batches = [batch for plan in plans for batch in plan]
    assert sorted(index for batch in batches for index in batch.indices) == list(range(5))
    assert all(batch.optimizer_normalization_count == pytest.approx(5 / 3) for batch in batches)


def test_mixed_sampling_measures_fail_before_optimizer_commit() -> None:
    model = TinyFactorModel()
    trainer = ScalableTrainer(
        model, _optimizer(model), StagedTrainingSchedule(1, 0, 0),
        config=TrainerConfig(accumulation_steps=2, ema_decay=None),
    )
    plain = _batch((1.0,))
    contracted = TinyBatch(plain.x, plain.y, 2.5)
    with pytest.raises(DomainError, match="cannot mix"):
        trainer.train_window((plain, contracted), _objective)
    assert trainer.cursor.optimizer_step == 0
    assert float(model.composition_model.weight.item()) == 0.0
    assert all(parameter.grad is None for parameter in model.parameters())


@pytest.mark.parametrize("normalizer", (1e308, 5e-324))
def test_overflowed_gradient_normalization_fails_before_commit(normalizer: float) -> None:
    model = TinyFactorModel()
    trainer = ScalableTrainer(
        model, _optimizer(model), StagedTrainingSchedule(1, 0, 0),
        config=TrainerConfig(accumulation_steps=2, ema_decay=None),
    )
    plain = _batch((1.0,))
    batch = TinyBatch(plain.x, plain.y, normalizer)
    with pytest.raises(DomainError, match="computed gradient"):
        trainer.train_window((batch, batch), _objective)
    assert trainer.cursor.optimizer_step == 0
    assert float(model.composition_model.weight.item()) == 0.0
    assert all(parameter.grad is None for parameter in model.parameters())


def test_trainer_rejects_optimizer_state_not_owned_by_the_root_model() -> None:
    model = TinyFactorModel()
    foreign = torch.nn.Parameter(torch.zeros(1, dtype=torch.float64))
    optimizer = torch.optim.SGD((*model.parameters(), foreign), lr=0.1)
    with pytest.raises(DomainError, match="outside the root model"):
        ScalableTrainer(
            model,
            optimizer,
            StagedTrainingSchedule(1, 0, 0),
            config=TrainerConfig(ema_decay=None),
        )


def test_cpu_bfloat16_autocast_keeps_finite_master_gradients() -> None:
    model = TinyFactorModel().float()
    trainer = ScalableTrainer(
        model,
        torch.optim.SGD(model.parameters(), lr=0.1),
        StagedTrainingSchedule(1, 0, 0),
        config=TrainerConfig(
            precision=PrecisionMode.BFLOAT16,
            ema_decay=None,
        ),
    )
    values = torch.tensor([[1.0], [2.0]], dtype=torch.float32)
    report = trainer.train_window(
        (TinyBatch(values, torch.ones_like(values)),),
        _objective,
    )
    assert report.gradients_before_clip.all_finite
    assert all(torch.isfinite(parameter).all() for parameter in model.parameters())


def test_cpu_float16_policy_is_rejected_before_training() -> None:
    model = TinyFactorModel().float()
    with pytest.raises(DomainError, match="only on CUDA"):
        ScalableTrainer(
            model,
            torch.optim.SGD(model.parameters(), lr=0.1),
            StagedTrainingSchedule(1, 0, 0),
            config=TrainerConfig(
                precision=PrecisionMode.FLOAT16,
                ema_decay=None,
            ),
        )


def test_trainer_rejects_low_or_mixed_master_parameter_dtypes() -> None:
    half_model = TinyFactorModel().half()
    with pytest.raises(DomainError, match="master dtype must be float32 or float64"):
        ScalableTrainer(
            half_model,
            torch.optim.SGD(half_model.parameters(), lr=0.1),
            StagedTrainingSchedule(1, 0, 0),
            config=TrainerConfig(ema_decay=None),
        )

    mixed_model = TinyFactorModel()
    mixed_model.geometry_field.float()
    with pytest.raises(DomainError, match="share one floating master dtype"):
        ScalableTrainer(
            mixed_model,
            torch.optim.SGD(mixed_model.parameters(), lr=0.1),
            StagedTrainingSchedule(1, 0, 0),
            config=TrainerConfig(ema_decay=None),
        )


def test_amp_requires_float32_master_parameters() -> None:
    model = TinyFactorModel()
    with pytest.raises(DomainError, match="mixed precision requires float32"):
        ScalableTrainer(
            model,
            _optimizer(model),
            StagedTrainingSchedule(1, 0, 0),
            config=TrainerConfig(
                precision=PrecisionMode.BFLOAT16,
                ema_decay=None,
            ),
        )


def test_amp_overflow_retries_same_window_with_replayed_rng() -> None:
    class StageCheckingScaler:
        def __init__(self) -> None:
            self.stage = "ready"
            self.updates = 0
            self.current_scale = 8.0
            self.growth_tracker = 7

        def scale(self, value):  # type: ignore[no-untyped-def]
            if self.stage != "ready":
                raise RuntimeError("scale called before the prior update completed")
            return value

        def unscale_(self, optimizer):  # type: ignore[no-untyped-def]
            del optimizer
            if self.stage != "ready":
                raise RuntimeError("unscale_ called twice")
            self.stage = "unscaled"

        def update(self, new_scale=None):  # type: ignore[no-untyped-def]
            if self.stage != "unscaled":
                raise RuntimeError("update called in the wrong scaler stage")
            if new_scale is not None:
                self.current_scale = float(new_scale)
            else:
                self.growth_tracker += 1
            self.stage = "ready"
            self.updates += 1

        def step(self, optimizer):  # type: ignore[no-untyped-def]
            if self.stage != "unscaled":
                raise RuntimeError("step called before unscale")
            optimizer.step()

        def state_dict(self):  # type: ignore[no-untyped-def]
            return {
                "scale": self.current_scale,
                "backoff_factor": 0.5,
                "_growth_tracker": self.growth_tracker,
                "updates": self.updates,
            }

        def load_state_dict(self, state):  # type: ignore[no-untyped-def]
            self.current_scale = float(state["scale"])
            self.growth_tracker = int(state["_growth_tracker"])

        def get_scale(self):  # type: ignore[no-untyped-def]
            return self.current_scale

    class InfiniteGradientObjective:
        contract = {
            "schema": "crystalatlas.test-infinite-gradient.v1",
            "checkpointable": False,
        }

        def __init__(self) -> None:
            self.calls = 0
            self.random_draws: list[torch.Tensor] = []

        def __call__(self, models, batch, phase):  # type: ignore[no-untyped-def]
            assert phase is TrainingPhase.COMPOSITION
            self.calls += 1
            self.random_draws.append(torch.rand(4))
            # First attempt: sqrt(0) is finite while its derivative is +infinity.
            # The retry uses an ordinary finite objective.
            loss = (
                torch.sqrt(models.composition.weight).mean()
                if self.calls == 1
                else (models.composition.weight - 1.0).square().mean()
            )
            return ComponentLosses(batch.size, loss, None, None)

    model = TinyFactorModel()
    trainer = ScalableTrainer(
        model,
        _optimizer(model),
        StagedTrainingSchedule(1, 0, 0),
        config=TrainerConfig(ema_decay=None, max_amp_overflow_retries=2),
    )
    scaler = StageCheckingScaler()
    trainer._scaler = scaler
    objective = InfiniteGradientObjective()

    report = trainer.train_window((_batch((1.0,)),), objective)
    assert report.optimizer_step == 1
    assert objective.calls == 2
    assert torch.equal(objective.random_draws[0], objective.random_draws[1])
    assert scaler.stage == "ready"
    assert scaler.updates == 2
    assert scaler.current_scale == 4.0
    assert scaler.growth_tracker == 1


def test_amp_commit_failure_faults_even_after_public_state_rollback() -> None:
    class BrokenCommitScaler:
        def state_dict(self):  # type: ignore[no-untyped-def]
            # Deliberately omits the hidden per-optimizer stage, as the public
            # PyTorch GradScaler state does.
            return {"scale": 8.0, "backoff_factor": 0.5}

        def load_state_dict(self, state):  # type: ignore[no-untyped-def]
            del state

        def step(self, optimizer):  # type: ignore[no-untyped-def]
            del optimizer
            raise RuntimeError("synthetic scaler step failure after stage mutation")

        def update(self, new_scale=None):  # type: ignore[no-untyped-def]
            del new_scale

    model = TinyFactorModel()
    trainer = ScalableTrainer(
        model,
        _optimizer(model),
        StagedTrainingSchedule(1, 0, 0),
        config=TrainerConfig(
            ema_decay=None,
            transactional_optimizer_step=True,
        ),
    )
    trainer._scaler = BrokenCommitScaler()
    with pytest.raises(TrainingNumericsError, match="trainer faulted"):
        trainer._commit_optimizer_step()
    assert trainer._faulted
    with pytest.raises(TrainingNumericsError, match="trainer is faulted"):
        trainer.train_window((_batch((1.0,)),), _objective)


def test_ddp_defaults_to_unused_parameter_discovery_for_sparse_topologies() -> None:
    policy = DistributedConfig(enabled=True)
    assert policy.find_unused_parameters


def test_unequal_microbatch_accumulation_equals_the_full_population_gradient() -> None:
    model = TinyFactorModel()
    trainer = ScalableTrainer(
        model,
        _optimizer(model, lr=0.1),
        StagedTrainingSchedule(1, 0, 0),
        config=TrainerConfig(
            accumulation_steps=2,
            ema_decay=None,
            transactional_optimizer_step=False,
        ),
    )
    first = _batch((1.0,))
    second = _batch((2.0, 3.0, 4.0))
    full_x = torch.cat((first.x, second.x))
    expected_gradient = float((2.0 * full_x * -1.0).mean().item())

    report = trainer.train_window((first, second), _objective)
    expected_weight = -0.1 * expected_gradient
    assert float(model.composition_model.weight.item()) == pytest.approx(
        expected_weight, abs=1.0e-14
    )
    assert report.global_structures == 4
    assert report.microbatches == 2
    assert report.gradients_before_clip.global_l2_norm == pytest.approx(
        abs(expected_gradient), abs=1.0e-14
    )


def test_failed_optimizer_rollback_faults_the_trainer() -> None:
    class RollbackRejectingSGD(torch.optim.SGD):
        def step(self, closure=None):  # type: ignore[no-untyped-def]
            del closure
            with torch.no_grad():
                self.param_groups[0]["params"][0].fill_(float("nan"))
            raise RuntimeError("synthetic update failure")

        def load_state_dict(self, state_dict):  # type: ignore[no-untyped-def]
            del state_dict
            raise RuntimeError("synthetic rollback failure")

    model = TinyFactorModel()
    trainer = ScalableTrainer(
        model,
        RollbackRejectingSGD(model.parameters(), lr=0.1),
        StagedTrainingSchedule(1, 0, 0),
        config=TrainerConfig(ema_decay=None),
    )
    with pytest.raises(TrainingNumericsError, match="rollback failures"):
        trainer.train_window((_batch((1.0,)),), _objective)
    with pytest.raises(TrainingNumericsError, match="trainer is faulted"):
        trainer.train_window((_batch((1.0,)),), _objective)


def test_staged_training_freezes_inactive_factors_and_ema_tracks_only_active() -> None:
    model = TinyFactorModel()
    trainer = ScalableTrainer(
        model,
        _optimizer(model),
        StagedTrainingSchedule(1, 1, 1, 1),
        config=TrainerConfig(ema_decay=0.5),
    )
    batch = _batch((1.0, 2.0))
    names = {
        phase: tuple(
            name for name, _ in getattr(model, attribute).named_parameters(prefix=attribute)
        )
        for phase, attribute in (
            (TrainingPhase.COMPOSITION, "composition_model"),
            (TrainingPhase.COVER, "cover_selector"),
            (TrainingPhase.GEOMETRY, "geometry_field"),
        )
    }
    previous = {name: value.detach().clone() for name, value in model.named_parameters()}
    for phase in (
        TrainingPhase.COMPOSITION,
        TrainingPhase.COVER,
        TrainingPhase.GEOMETRY,
    ):
        report = trainer.train_epoch((batch,), _objective)
        assert report.phase is phase
        current = dict(model.named_parameters())
        changed = {
            name
            for name, value in current.items()
            if not torch.equal(value.detach(), previous[name])
        }
        assert changed == set(names[phase])
        assert trainer.ema is not None
        counts = trainer.ema.update_counts
        for active_phase, active_names in names.items():
            # Phase strings are not chronological; use explicit membership.
            completed = {
                TrainingPhase.COMPOSITION: {TrainingPhase.COMPOSITION},
                TrainingPhase.COVER: {
                    TrainingPhase.COMPOSITION,
                    TrainingPhase.COVER,
                },
                TrainingPhase.GEOMETRY: {
                    TrainingPhase.COMPOSITION,
                    TrainingPhase.COVER,
                    TrainingPhase.GEOMETRY,
                },
            }[phase]
            expected = int(active_phase in completed)
            assert all(counts[name] == expected for name in active_names)
        previous = {name: value.detach().clone() for name, value in current.items()}


def test_state_finite_check_agrees_with_an_elementwise_scan() -> None:
    from crystalatlas.trainer import _state_has_nonfinite

    finite = {
        "weight": torch.arange(6, dtype=torch.float64).reshape(2, 3),
        "steps": torch.tensor([3, 4], dtype=torch.int64),
        "flags": torch.tensor([True, False]),
        "nested": [{"momentum": torch.zeros(4)}, (torch.ones(2, dtype=torch.float32), 0.5)],
        "empty": torch.zeros(0),
        "label": "adamw",
        "count": 7,
        "none": None,
    }
    assert not _state_has_nonfinite(finite)
    assert not _state_has_nonfinite({})
    for bad in (float("nan"), float("inf"), -float("inf")):
        assert _state_has_nonfinite(bad)
        assert _state_has_nonfinite({"a": [finite, (1.0, bad)]})
        tensor = torch.zeros(5, dtype=torch.float32)
        tensor[3] = bad
        assert _state_has_nonfinite(tensor)
        # One bad entry anywhere in the tree is enough, whatever precedes or follows it.
        assert _state_has_nonfinite({"first": finite, "second": {"deep": [tensor]}})
        assert _state_has_nonfinite({"second": {"deep": [tensor]}, "last": finite})
    complex_bad = torch.zeros(2, dtype=torch.complex64)
    complex_bad[1] = complex(0.0, float("nan"))
    assert _state_has_nonfinite([complex_bad])
    assert not _state_has_nonfinite([torch.zeros(2, dtype=torch.complex64)])
    # Mixed dtypes and a sparse tensor are grouped, never promoted or densified wrongly.
    sparse = torch.sparse_coo_tensor([[0, 2]], [1.0, 2.0], (4,))
    assert not _state_has_nonfinite([sparse, torch.ones(3, dtype=torch.float16)])
    sparse_bad = torch.sparse_coo_tensor([[0, 2]], [1.0, float("inf")], (4,))
    assert _state_has_nonfinite([torch.ones(3, dtype=torch.float64), sparse_bad])
    # A float32 sum would overflow here; the check must not confuse that with a bad entry.
    assert not _state_has_nonfinite([torch.full((4,), 3.0e38, dtype=torch.float32)])


def test_float64_crystallographic_audit_runs_outside_neural_precision() -> None:
    model = TinyFactorModel()
    trainer = ScalableTrainer(
        model,
        _optimizer(model),
        StagedTrainingSchedule(1, 0, 0),
        config=TrainerConfig(ema_decay=None),
    )
    observed: list[object] = []

    def audit(batch: TinyBatch, dtype):  # type: ignore[no-untyped-def]
        observed.append(dtype)
        return CrystallographicAuditResult(
            torch.zeros(batch.size, dtype=torch.float64),
            tolerance=0.0,
        )

    trainer.train_window((_batch((1.0,)),), _objective, crystallographic_audit=audit)
    assert observed == [torch.float64]


def test_mid_epoch_checkpoint_requires_an_aligned_sampler_cursor() -> None:
    model = TinyFactorModel()
    trainer = ScalableTrainer(
        model,
        _optimizer(model),
        StagedTrainingSchedule(2, 0, 0),
        config=TrainerConfig(ema_decay=None),
    )
    trainer.train_window((_batch((1.0,)),), _objective)
    assert trainer.cursor == TrainerCursor(epoch=0, microbatch=1, optimizer_step=1)
    with pytest.raises(CheckpointError, match="aligned sampler"):
        trainer.state_dict()
    with pytest.raises(CheckpointError, match="disagree"):
        trainer.state_dict(sampler=CursorSampler(0, 0))
    state = trainer.state_dict(sampler=CursorSampler(0, 1))
    assert state["rank_states"][0]["sampler"] == {
        "sampler_fingerprint": "a" * 64,
        "common_policy_fingerprint": "c" * 64,
        "partition_rank": 0,
        "world_size": 1,
        "epoch": 0,
        "cursor": 1,
    }


def test_checkpoint_rejects_sampler_partition_or_world_size_mismatch() -> None:
    model = TinyFactorModel()
    trainer = ScalableTrainer(
        model,
        _optimizer(model),
        StagedTrainingSchedule(2, 0, 0),
        config=TrainerConfig(ema_decay=None),
    )
    trainer.train_window((_batch((1.0,)),), _objective)
    with pytest.raises(CheckpointError, match="partition rank"):
        trainer.state_dict(sampler=CursorSampler(0, 1, partition_rank=1, world_size=2))
    with pytest.raises(CheckpointError, match="world size"):
        trainer.state_dict(sampler=CursorSampler(0, 1, world_size=2))


def test_resume_rejects_a_changed_initial_trainability_mask() -> None:
    source = TinyFactorModel()
    source_trainer = ScalableTrainer(
        source,
        _optimizer(source),
        StagedTrainingSchedule(1, 0, 0),
        config=TrainerConfig(ema_decay=None),
    )
    state = source_trainer.state_dict(sampler=CursorSampler(0, 0))

    destination = TinyFactorModel()
    destination.geometry_field.weight.requires_grad_(False)
    destination_trainer = ScalableTrainer(
        destination,
        _optimizer(destination),
        StagedTrainingSchedule(1, 0, 0),
        config=TrainerConfig(ema_decay=None),
    )
    with pytest.raises(CheckpointError, match="trainability mask"):
        destination_trainer._validate_resume_state(
            state,
            sampler=CursorSampler(0, 0),
        )


def test_ddp_root_sync_precedes_ema_snapshot(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeDDP(torch.nn.Module):
        def __init__(self, module, **kwargs):  # type: ignore[no-untyped-def]
            super().__init__()
            del kwargs
            self.module = module

        def forward(self, *args, **kwargs):  # type: ignore[no-untyped-def]
            return self.module(*args, **kwargs)

        def no_sync(self):  # type: ignore[no-untyped-def]
            from contextlib import nullcontext

            return nullcontext()

    monkeypatch.setattr(torch.distributed, "is_available", lambda: True)
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda: 2)
    monkeypatch.setattr(torch.distributed, "get_rank", lambda: 1)
    monkeypatch.setattr(torch.distributed, "barrier", lambda: None)

    def gather(output, value):  # type: ignore[no-untyped-def]
        output[:] = [value, value]

    def broadcast(tensor, src):  # type: ignore[no-untyped-def]
        assert src == 0
        tensor.fill_(2.5)

    monkeypatch.setattr(torch.distributed, "all_gather_object", gather)
    monkeypatch.setattr(torch.distributed, "broadcast", broadcast)
    monkeypatch.setattr(torch.nn.parallel, "DistributedDataParallel", FakeDDP)

    model = TinyFactorModel()
    trainer = ScalableTrainer(
        model,
        _optimizer(model),
        StagedTrainingSchedule(1, 0, 0),
        config=TrainerConfig(
            ema_decay=0.9,
            distributed=DistributedConfig(enabled=True),
        ),
    )
    assert all(torch.all(parameter == 2.5) for parameter in model.parameters())
    assert trainer.ema is not None
    assert all(torch.all(value == 2.5) for value in trainer.ema.state_dict()["shadow"].values())


def test_trained_checkpoint_requires_the_bound_objective_contract() -> None:
    model = TinyFactorModel()
    trainer = ScalableTrainer(
        model,
        _optimizer(model),
        StagedTrainingSchedule(2, 0, 0),
        config=TrainerConfig(ema_decay=None),
    )
    trainer.train_window((_batch((1.0,)),), _objective)
    trainer._objective_contract = None
    with pytest.raises(CheckpointError, match="explicit checkpointable objective"):
        trainer.state_dict(sampler=CursorSampler(0, 1))


def test_stateful_objective_requires_an_explicit_semantic_contract() -> None:
    class StatefulObjective:
        def __init__(self, scale: float) -> None:
            self.scale = scale

        def __call__(self, models, batch, phase):  # type: ignore[no-untyped-def]
            result = _objective(models, batch, phase)
            return ComponentLosses(
                result.structure_count,
                None if result.composition is None else self.scale * result.composition,
                result.cover,
                result.geometry,
            )

    model = TinyFactorModel()
    trainer = ScalableTrainer(
        model,
        _optimizer(model),
        StagedTrainingSchedule(1, 0, 0),
        config=TrainerConfig(ema_decay=None),
    )
    with pytest.raises(DomainError, match="require an explicit canonical contract"):
        trainer.train_window((_batch((1.0,)),), StatefulObjective(1.0))


def test_generic_function_objective_is_trainable_but_not_exactly_resumable() -> None:
    def generic(models, batch, phase):  # type: ignore[no-untyped-def]
        return _objective(models, batch, phase)

    model = TinyFactorModel()
    trainer = ScalableTrainer(
        model,
        _optimizer(model),
        StagedTrainingSchedule(2, 0, 0),
        config=TrainerConfig(ema_decay=None),
    )
    trainer.train_window((_batch((1.0,)),), generic)
    with pytest.raises(CheckpointError, match="explicit checkpointable objective"):
        trainer.state_dict(sampler=CursorSampler(0, 1))


def test_unfinished_epoch_boundary_checkpoint_requires_sampler_policy() -> None:
    model = TinyFactorModel()
    trainer = ScalableTrainer(
        model,
        _optimizer(model),
        StagedTrainingSchedule(1, 0, 0),
        config=TrainerConfig(ema_decay=None),
    )
    with pytest.raises(CheckpointError, match="every unfinished"):
        trainer.state_dict()


def test_checkpoint_boundary_failures_pass_through_collective_preflight() -> None:
    model = TinyFactorModel()
    trainer = ScalableTrainer(
        model,
        _optimizer(model),
        StagedTrainingSchedule(1, 0, 0),
        config=TrainerConfig(ema_decay=None),
    )
    next(model.parameters()).grad = torch.ones(1, 1, dtype=torch.float64)
    with pytest.raises(CheckpointError, match="preflight failed across ranks"):
        trainer.state_dict()


def test_unserializable_rank_state_fails_before_collective_gather() -> None:
    class UnserializableSampler(CursorSampler):
        def state_dict(self):  # type: ignore[no-untyped-def]
            state = super().state_dict()
            state["unpicklable"] = lambda value: value
            return state

    model = TinyFactorModel()
    trainer = ScalableTrainer(
        model,
        _optimizer(model),
        StagedTrainingSchedule(1, 0, 0),
        config=TrainerConfig(ema_decay=None),
    )
    with pytest.raises(CheckpointError, match="preflight failed across ranks"):
        trainer.state_dict(sampler=UnserializableSampler(0, 0))


def test_restore_snapshot_failures_pass_through_collective_preflight(
    tmp_path: Path,
) -> None:
    class BrokenSampler:
        def state_dict(self):  # type: ignore[no-untyped-def]
            raise RuntimeError("rank-local sampler snapshot failed")

        def load_state_dict(self, state):  # type: ignore[no-untyped-def]
            del state

    model = TinyFactorModel()
    trainer = ScalableTrainer(
        model,
        _optimizer(model),
        StagedTrainingSchedule(1, 0, 0),
        config=TrainerConfig(ema_decay=None),
    )
    with pytest.raises(CheckpointError, match="before mutation across ranks"):
        trainer.load_checkpoint(
            tmp_path / "never-opened.pt",
            scientific_manifest=_manifest(),
            scientific_config=CrystalAtlasConfig(),
            sampler=BrokenSampler(),
        )


def test_scheduler_class_and_initial_semantics_are_checkpoint_bound(
    tmp_path: Path,
) -> None:
    config = CrystalAtlasConfig()
    schedule = StagedTrainingSchedule(1, 1, 1, 1)
    source = TinyFactorModel()
    source_optimizer = _optimizer(source)
    source_scheduler = torch.optim.lr_scheduler.StepLR(source_optimizer, step_size=3, gamma=0.5)
    source_trainer = ScalableTrainer(
        source,
        source_optimizer,
        schedule,
        scheduler=source_scheduler,
        config=TrainerConfig(ema_decay=None),
    )
    path = tmp_path / "trainer.pt"
    info = source_trainer.save_checkpoint(
        path,
        scientific_config=config,
        scientific_manifest=_manifest(),
        provenance=_provenance(config, 59),
        sampler=CursorSampler(0, 0),
    )
    assert info is not None

    destination = TinyFactorModel()
    destination_optimizer = _optimizer(destination)
    incompatible = torch.optim.lr_scheduler.ExponentialLR(destination_optimizer, gamma=0.5)
    destination_trainer = ScalableTrainer(
        destination,
        destination_optimizer,
        schedule,
        scheduler=incompatible,
        config=TrainerConfig(ema_decay=None),
    )
    with pytest.raises(CheckpointError, match="scheduler class"):
        destination_trainer.load_checkpoint(
            path,
            scientific_manifest=_manifest(),
            scientific_config=config,
            expected_sha256=info.sha256,
            sampler=CursorSampler(0, 0),
        )


def test_epoch_boundary_resume_reproduces_the_uninterrupted_trajectory(
    tmp_path: Path,
) -> None:
    scientific_config = CrystalAtlasConfig()
    schedule = StagedTrainingSchedule(2, 0, 0)
    batch = _batch((1.0, 2.0, 3.0))

    source = TinyFactorModel()
    source_optimizer = torch.optim.AdamW(source.parameters(), lr=0.01)
    source_trainer = ScalableTrainer(
        source,
        source_optimizer,
        schedule,
        config=TrainerConfig(ema_decay=0.9),
    )
    source_trainer.train_epoch((batch,), _objective)
    path = tmp_path / "exact-resume.pt"
    info = source_trainer.save_checkpoint(
        path,
        scientific_config=scientific_config,
        scientific_manifest=_manifest(),
        provenance=_provenance(scientific_config, 71),
        sampler=CursorSampler(1, 0),
    )
    assert info is not None
    source_trainer.train_epoch((batch,), _objective)

    resumed = TinyFactorModel()
    resumed_optimizer = torch.optim.AdamW(resumed.parameters(), lr=0.01)
    resumed_trainer = ScalableTrainer(
        resumed,
        resumed_optimizer,
        schedule,
        config=TrainerConfig(ema_decay=0.9),
    )
    with pytest.raises(CheckpointError, match="sampler policy fingerprint"):
        resumed_trainer.load_checkpoint(
            path,
            scientific_manifest=_manifest(),
            scientific_config=scientific_config,
            expected_sha256=info.sha256,
            sampler=CursorSampler(1, 0, sampler_fingerprint="b" * 64),
        )
    receipt = resumed_trainer.load_checkpoint(
        path,
        scientific_manifest=_manifest(),
        scientific_config=scientific_config,
        expected_sha256=info.sha256,
        sampler=CursorSampler(1, 0),
    )
    assert receipt.rng_restored
    resumed_trainer.train_epoch((batch,), _objective)

    for expected, actual in zip(
        source.parameters(),
        resumed.parameters(),
        strict=True,
    ):
        assert torch.equal(expected, actual)
    assert source_trainer.cursor == resumed_trainer.cursor
    assert source_trainer.ema is not None and resumed_trainer.ema is not None
    source_ema = source_trainer.ema.state_dict()
    resumed_ema = resumed_trainer.ema.state_dict()
    assert source_ema["updates"] == resumed_ema["updates"]
    for name in source_ema["shadow"]:
        assert torch.equal(source_ema["shadow"][name], resumed_ema["shadow"][name])


def test_phase_reconstruction_failure_rolls_back_the_complete_load(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scientific_config = CrystalAtlasConfig()
    schedule = StagedTrainingSchedule(2, 0, 0)

    source = TinyFactorModel()
    source_trainer = ScalableTrainer(
        source,
        torch.optim.AdamW(source.parameters(), lr=0.01),
        schedule,
        config=TrainerConfig(ema_decay=None),
    )
    source_trainer.train_window((_batch((1.0,)),), _objective)
    path = tmp_path / "phase-reconstruction.pt"
    info = source_trainer.save_checkpoint(
        path,
        scientific_config=scientific_config,
        scientific_manifest=_manifest(),
        provenance=_provenance(scientific_config, 73),
        sampler=CursorSampler(0, 1),
    )
    assert info is not None

    destination = TinyFactorModel()
    destination_trainer = ScalableTrainer(
        destination,
        torch.optim.AdamW(destination.parameters(), lr=0.01),
        schedule,
        config=TrainerConfig(ema_decay=None),
    )
    sampler = CursorSampler(0, 0)
    before_model = {
        name: value.detach().clone() for name, value in destination.state_dict().items()
    }
    before_rng = torch.get_rng_state().clone()
    original_configure = destination_trainer._configure_phase
    calls = 0

    def fail_once(phase):  # type: ignore[no-untyped-def]
        nonlocal calls
        calls += 1
        if calls == 1:
            raise RuntimeError("synthetic phase reconstruction failure")
        original_configure(phase)

    monkeypatch.setattr(destination_trainer, "_configure_phase", fail_once)
    with pytest.raises(CheckpointError, match="phase reconstruction failed"):
        destination_trainer.load_checkpoint(
            path,
            scientific_manifest=_manifest(),
            scientific_config=scientific_config,
            expected_sha256=info.sha256,
            sampler=sampler,
        )

    assert destination_trainer.cursor == TrainerCursor()
    assert destination_trainer._objective_contract is None
    assert sampler == CursorSampler(0, 0)
    assert torch.equal(torch.get_rng_state(), before_rng)
    for name, value in destination.state_dict().items():
        assert torch.equal(value, before_model[name])


def test_factorwise_ddp_path_executes_with_the_same_accumulation_contract(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if sys.platform == "darwin":
        pytest.skip("this PyTorch build aborts inside ProcessGroupGloo on macOS")
    if not torch.distributed.is_available():
        pytest.skip("PyTorch distributed support is unavailable")
    interfaces = {name for _, name in socket.if_nameindex()}
    loopback = "lo0" if "lo0" in interfaces else "lo" if "lo" in interfaces else None
    if loopback is None:
        pytest.skip("no named loopback interface is available for Gloo")
    monkeypatch.setenv("GLOO_SOCKET_IFNAME", loopback)
    rendezvous = (tmp_path / "gloo-rendezvous").resolve()
    torch.distributed.init_process_group(
        "gloo",
        init_method=f"file://{rendezvous}",
        rank=0,
        world_size=1,
    )
    try:
        model = TinyFactorModel()
        trainer = ScalableTrainer(
            model,
            _optimizer(model),
            StagedTrainingSchedule(1, 0, 0),
            config=TrainerConfig(
                accumulation_steps=2,
                ema_decay=None,
                distributed=DistributedConfig(enabled=True),
            ),
        )
        report = trainer.train_window(
            (_batch((1.0,)), _batch((2.0, 3.0))),
            _objective,
        )
        assert report.global_structures == 3
        assert report.optimizer_step == 1
        assert torch.isfinite(model.composition_model.weight).all()
    finally:
        torch.distributed.destroy_process_group()
