from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import replace
from hashlib import sha256
from pathlib import Path
from types import MappingProxyType

import pytest
from test_dataset_storage import _compiled_dataset

from crystalatlas.catalogue import MINIMAL_CATALOGUE
from crystalatlas.composition import CompositionDomain, CompositionGrammar
from crystalatlas.dataset_storage import DatasetStore, write_compiled_dataset
from crystalatlas.errors import DomainError
from crystalatlas.tokenization import CompositionTokenCodec
from crystalatlas.training_data import (
    DeterministicBucketBatchSampler,
    StoredTrainingDataset,
    TrainingMeasure,
)


def _codec() -> CompositionTokenCodec:
    domain = CompositionDomain((6, 7, 8), 3, 8)
    return CompositionTokenCodec(CompositionGrammar(domain))


@contextmanager
def _expanded_dataset(
    tmp_path: Path,
    *,
    record_count: int,
    topology_groups: tuple[int, ...] | None = None,
) -> Iterator[StoredTrainingDataset]:
    """Clone an exactly audited fixture without weakening the production boundary."""

    groups = (0,) * record_count if topology_groups is None else topology_groups
    if len(groups) != record_count:
        raise ValueError("topology_groups must align with record_count")
    compiled = _compiled_dataset(tmp_path)
    root = tmp_path / "distributed-training-store"
    write_compiled_dataset(compiled, root, records_per_shard=1)
    with DatasetStore.open(
        root,
        expected_build_manifest_fingerprint=compiled.manifest.fingerprint,
    ) as store:
        dataset = StoredTrainingDataset(store, _codec(), MINIMAL_CATALOGUE)
        template = dataset._validated[0]
        validated = []
        for index, group in enumerate(groups):
            record_id = f"distributed-{index:03d}"
            bucket_key = replace(
                template.descriptor.bucket_key,
                atom_count=template.descriptor.bucket_key.atom_count + group,
            )
            descriptor = replace(
                template.descriptor,
                dataset_index=index,
                record_id=record_id,
                bucket_key=bucket_key,
            )
            validated.append(
                replace(
                    template,
                    descriptor=descriptor,
                    record=replace(template.record, record_id=record_id),
                )
            )
        dataset._validated = tuple(validated)
        dataset._composition_multiplicities = MappingProxyType(
            {template.descriptor.composition_key: record_count}
        )
        identity = "|".join(
            f"{item.descriptor.record_id}:{item.descriptor.bucket_key.atom_count}"
            for item in validated
        )
        dataset._dataset_fingerprint = sha256(identity.encode("ascii")).hexdigest()
        yield dataset


def _record_ids(
    sampler: DeterministicBucketBatchSampler,
) -> tuple[tuple[str, ...], ...]:
    return tuple(batch.record_ids for batch in sampler)


def _assert_exact_rank_partition(
    dataset: StoredTrainingDataset,
    rank_batches: tuple[tuple[tuple[str, ...], ...], ...],
) -> None:
    assert len({len(batches) for batches in rank_batches}) == 1
    rank_records = tuple(
        tuple(record_id for batch in batches for record_id in batch) for batches in rank_batches
    )
    assert all(records for records in rank_records)
    assert all(
        set(left).isdisjoint(right)
        for index, left in enumerate(rank_records)
        for right in rank_records[index + 1 :]
    )
    combined = tuple(record_id for records in rank_records for record_id in records)
    assert len(combined) == len(set(combined))
    assert sorted(combined) == sorted(dataset.record_ids)


def test_one_large_topology_batch_is_split_losslessly_across_ranks(
    tmp_path: Path,
) -> None:
    with _expanded_dataset(tmp_path, record_count=4) as dataset:
        samplers = tuple(
            DeterministicBucketBatchSampler(
                dataset,
                batch_size=8,
                seed=41,
                measure=TrainingMeasure.STRUCTURE,
                rank=rank,
                world_size=2,
            )
            for rank in range(2)
        )
        assert tuple(len(sampler) for sampler in samplers) == (1, 1)
        _assert_exact_rank_partition(dataset, tuple(_record_ids(item) for item in samplers))


def test_rank_plans_have_equal_steps_without_duplication_or_drop(
    tmp_path: Path,
) -> None:
    with _expanded_dataset(tmp_path, record_count=10) as dataset:
        samplers = tuple(
            DeterministicBucketBatchSampler(
                dataset,
                batch_size=4,
                seed=20260827,
                measure=TrainingMeasure.STRUCTURE,
                rank=rank,
                world_size=2,
            )
            for rank in range(2)
        )
        assert tuple(len(sampler) for sampler in samplers) == (2, 2)
        _assert_exact_rank_partition(dataset, tuple(_record_ids(item) for item in samplers))

        for sampler in samplers:
            sampler.advance_epoch()
        assert tuple(len(sampler) for sampler in samplers) == (2, 2)
        _assert_exact_rank_partition(dataset, tuple(_record_ids(item) for item in samplers))


def test_distributed_resume_and_fingerprints_are_rank_exact(tmp_path: Path) -> None:
    with _expanded_dataset(tmp_path, record_count=12) as dataset:
        original = DeterministicBucketBatchSampler(
            dataset,
            batch_size=3,
            seed=911,
            measure=TrainingMeasure.STRUCTURE,
            rank=1,
            world_size=2,
        )
        first = next(original).record_ids
        state = original.state_dict()
        expected_next = next(original).record_ids
        assert first != expected_next

        resumed = DeterministicBucketBatchSampler(
            dataset,
            batch_size=3,
            seed=911,
            measure=TrainingMeasure.STRUCTURE,
            rank=1,
            world_size=2,
        )
        assert resumed.fingerprint == state["sampler_fingerprint"]
        resumed.load_state_dict(state)
        assert resumed.cursor == 1
        assert next(resumed).record_ids == expected_next

        other_rank = DeterministicBucketBatchSampler(
            dataset,
            batch_size=3,
            seed=911,
            measure=TrainingMeasure.STRUCTURE,
            rank=0,
            world_size=2,
        )
        assert other_rank.fingerprint != resumed.fingerprint
        assert other_rank.common_policy_fingerprint == resumed.common_policy_fingerprint
        assert state["partition_rank"] == 1
        assert state["world_size"] == 2
        assert state["common_policy_fingerprint"] == resumed.common_policy_fingerprint
        with pytest.raises(DomainError, match="different sampler policy"):
            other_rank.load_state_dict(state)

        changed_seed = DeterministicBucketBatchSampler(
            dataset,
            batch_size=3,
            seed=912,
            measure=TrainingMeasure.STRUCTURE,
            rank=1,
            world_size=2,
        )
        assert changed_seed.common_policy_fingerprint != resumed.common_policy_fingerprint


def test_topology_fragmentation_fails_instead_of_duplicating_records(
    tmp_path: Path,
) -> None:
    with _expanded_dataset(
        tmp_path,
        record_count=3,
        topology_groups=(0, 1, 2),
    ) as dataset:
        with pytest.raises(
            DomainError,
            match="topology fragmentation prevents equal nonempty DDP steps",
        ):
            DeterministicBucketBatchSampler(
                dataset,
                batch_size=8,
                seed=7,
                measure=TrainingMeasure.STRUCTURE,
                rank=0,
                world_size=2,
            )
