"""Mixed-topology batch sampler (review finding M5).

Planning is tested against a planning-only dataset double, because a real
store with dozens of topologies would be expensive to compile here.  The
integration test collates real batches from the shared fixture store.
"""

from __future__ import annotations

import statistics
from pathlib import Path

import pytest
from test_training_data import _codec, _store

from crystalatlas.catalogue import MINIMAL_CATALOGUE
from crystalatlas.errors import DomainError
from crystalatlas.training_data import (
    CoverTopologyKey,
    DeterministicBucketBatchSampler,
    MixedTopologyBatchSampler,
    MixedTrainingBatch,
    StoredTrainingDataset,
    TrainingBucketKey,
    TrainingMeasure,
    TrainingRecordDescriptor,
    _fingerprint,
)


def _topology(index: int) -> CoverTopologyKey:
    return CoverTopologyKey(
        target_type="canonical-catalogue",
        space_group=f"G{index}",
        nu=1,
        support_size=1,
        row_labels=("1a",),
        row_multiplicities=(1,),
        row_free_dimensions=(0,),
        selected_actions=((1,),),
        orbit_topology=(("1a", 0),),
        lattice_dimension=1,
        intrinsic_dimension=0,
        nu_vocabulary_size=4,
        group_vocabulary_size=230,
        period_basis_sha256="0" * 64,
    )


class _PlanningDataset(StoredTrainingDataset):
    """Descriptors only: enough for epoch planning, never for collation."""

    def __init__(self, atom_counts: list[int], topologies: list[int]) -> None:
        self._planning_descriptors = tuple(
            TrainingRecordDescriptor(
                dataset_index=index,
                record_id=f"record-{index:04d}",
                composition_key=f"composition-{index:04d}",
                bucket_key=TrainingBucketKey(atom_count, _topology(topology)),
            )
            for index, (atom_count, topology) in enumerate(
                zip(atom_counts, topologies, strict=True)
            )
        )

    @property
    def descriptors(self) -> tuple[TrainingRecordDescriptor, ...]:
        return self._planning_descriptors

    @property
    def fingerprint(self) -> str:
        return "a" * 64

    def _collate(self, indices, *, measure):  # type: ignore[no-untyped-def]
        raise AssertionError("planning-only dataset")


def _fragmented(records: int = 100, topologies: int = 70) -> _PlanningDataset:
    # Like the MP-20 pilot: most topologies are singletons.
    return _PlanningDataset(
        [2 + (index * 7) % 13 for index in range(records)],
        [index % topologies for index in range(records)],
    )


def _plan_sizes(sampler) -> list[int]:  # type: ignore[no-untyped-def]
    return [len(batch.indices) for batch in sampler._plan]


def test_mixed_batches_cover_every_record_once_and_reach_full_occupancy() -> None:
    dataset = _fragmented()
    mixed = MixedTopologyBatchSampler(dataset, 32, seed=3, measure=TrainingMeasure.STRUCTURE)
    legacy = DeterministicBucketBatchSampler(
        dataset, 32, seed=3, measure=TrainingMeasure.STRUCTURE
    )
    for sampler in (mixed, legacy):
        indices = [index for batch in sampler._plan for index in batch.indices]
        assert sorted(indices) == list(range(len(dataset.descriptors)))
    assert _plan_sizes(mixed) == [32, 32, 32, 4]
    assert statistics.median(_plan_sizes(mixed)) >= 16
    assert statistics.median(_plan_sizes(legacy)) <= 2
    crossing = {
        len({dataset.descriptors[i].bucket_key for i in batch.indices}) for batch in mixed._plan
    }
    assert max(crossing) > 1
    expected = len(dataset.descriptors) / len(mixed._plan)
    assert all(batch.optimizer_normalization_count == expected for batch in mixed._plan)


def test_mixed_plan_is_deterministic_seeded_and_epoch_dependent() -> None:
    dataset = _fragmented()
    first = MixedTopologyBatchSampler(dataset, 32, seed=3, measure=TrainingMeasure.STRUCTURE)
    again = MixedTopologyBatchSampler(dataset, 32, seed=3, measure=TrainingMeasure.STRUCTURE)
    other = MixedTopologyBatchSampler(dataset, 32, seed=4, measure=TrainingMeasure.STRUCTURE)
    assert first.epoch_plan_fingerprint == again.epoch_plan_fingerprint
    assert first.fingerprint == again.fingerprint
    assert first._plan != other._plan
    assert first._build_plan(0) != first._build_plan(1)


def test_dense_pair_budget_splits_batches_and_refuses_oversized_records() -> None:
    dataset = _PlanningDataset([10] * 40, list(range(40)))
    sampler = MixedTopologyBatchSampler(
        dataset,
        32,
        seed=1,
        measure=TrainingMeasure.STRUCTURE,
        max_dense_pair_interactions=1_000,
    )
    sizes = _plan_sizes(sampler)
    assert sum(sizes) == 40 and max(sizes) == 10  # 10 * 10^2 <= 1000 < 11 * 10^2
    with pytest.raises(DomainError, match="dense pair budget"):
        MixedTopologyBatchSampler(
            dataset,
            32,
            seed=1,
            measure=TrainingMeasure.STRUCTURE,
            max_dense_pair_interactions=99,
        )


def test_mixed_sampler_has_its_own_identity_and_leaves_the_legacy_one_unchanged() -> None:
    dataset = _fragmented()
    legacy = DeterministicBucketBatchSampler(dataset, 32, seed=3)
    mixed = MixedTopologyBatchSampler(dataset, 32, seed=3)
    assert mixed.common_policy_fingerprint != legacy.common_policy_fingerprint
    # The legacy payload is exactly the pre-M5 payload.
    expected = _fingerprint(
        {
            "schema": "crystalatlas.deterministic-bucket-sampler.v3",
            "dataset_fingerprint": dataset.fingerprint,
            "batch_size": 32,
            "seed": 3,
            "measure": TrainingMeasure.UNIQUE_COMPOSITION.value,
            "drop_last": False,
            "world_size": 1,
            "eligible_record_ids": [
                dataset.descriptors[index].record_id for index in legacy._eligible_indices
            ],
        }
    )
    assert legacy.common_policy_fingerprint == expected


def test_distributed_mixed_plans_are_rank_balanced_and_disjoint() -> None:
    dataset = _fragmented(records=97)
    plans = [
        MixedTopologyBatchSampler(
            dataset, 32, seed=5, measure=TrainingMeasure.STRUCTURE, rank=rank, world_size=3
        )._plan
        for rank in range(3)
    ]
    assert len({len(plan) for plan in plans}) == 1
    indices = [index for plan in plans for batch in plan for index in batch.indices]
    assert sorted(indices) == list(range(97))


def test_mixed_sampler_collates_validated_parts_from_a_real_store(tmp_path: Path) -> None:
    with _store(tmp_path) as store:
        dataset = StoredTrainingDataset(store, _codec(), MINIMAL_CATALOGUE)
        sampler = MixedTopologyBatchSampler(
            dataset, 2, seed=17, measure=TrainingMeasure.STRUCTURE
        )
        batch = next(sampler)
        assert isinstance(batch, MixedTrainingBatch)
        assert batch.batch_size == 2 and len(batch.parts) == 1
        assert batch.optimizer_normalization_count == 2.0
        assert set(batch.record_ids) == set(store.record_ids)
        assert batch.measure is TrainingMeasure.STRUCTURE
        assert batch.dataset_fingerprint == dataset.fingerprint
        state = sampler.state_dict()
        resumed = MixedTopologyBatchSampler(
            dataset, 2, seed=17, measure=TrainingMeasure.STRUCTURE
        )
        resumed.load_state_dict(state)
        assert resumed.cursor == 1 and resumed.remaining_batches == 0
        with pytest.raises(DomainError, match="different sampler policy"):
            DeterministicBucketBatchSampler(
                dataset, 2, seed=17, measure=TrainingMeasure.STRUCTURE
            ).load_state_dict(state)
