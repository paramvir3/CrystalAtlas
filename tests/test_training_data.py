from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from test_dataset_storage import _compiled_dataset

import crystalatlas.dataset_storage as storage_module
from crystalatlas.catalogue import MINIMAL_CATALOGUE
from crystalatlas.composition import CompositionDomain, CompositionGrammar
from crystalatlas.config import DomainConfig
from crystalatlas.dataset_builder import canonical_training_target_fingerprint
from crystalatlas.dataset_splits import SplitMode
from crystalatlas.dataset_storage import DatasetStore, StoredFile, write_compiled_dataset
from crystalatlas.errors import DomainError, InvariantViolation
from crystalatlas.tokenization import CompositionTokenCodec
from crystalatlas.training_data import (
    DeterministicBucketBatchSampler,
    StoredTrainingDataset,
    TrainingMeasure,
    split_partition_record_ids,
)
from crystalatlas.training_targets import lift_exact_fixture_target


def _codec() -> CompositionTokenCodec:
    return CompositionTokenCodec(CompositionGrammar(CompositionDomain((6, 7, 8), 3, 8)))


def _store(tmp_path: Path, domain: DomainConfig | None = None) -> DatasetStore:
    compiled = _compiled_dataset(tmp_path, domain)
    root = tmp_path / "training-store"
    write_compiled_dataset(compiled, root, records_per_shard=1)
    return DatasetStore.open(
        root,
        expected_build_manifest_fingerprint=compiled.manifest.fingerprint,
    )


def test_structure_batch_is_exact_immutable_and_measure_correct(tmp_path: Path) -> None:
    with _store(tmp_path) as store:
        dataset = StoredTrainingDataset(store, _codec(), MINIMAL_CATALOGUE)
        assert len(dataset) == 2
        assert dataset.unique_composition_count == 1
        assert split_partition_record_ids(store, SplitMode.COMPOSITION, "train") == tuple(
            sorted(store.record_ids)
        )

        sampler = DeterministicBucketBatchSampler(
            dataset,
            2,
            seed=17,
            measure=TrainingMeasure.STRUCTURE,
        )
        batch = next(sampler)
        assert sampler.remaining_batches == 0
        assert batch.dataset_fingerprint == dataset.fingerprint
        assert batch.measure is TrainingMeasure.STRUCTURE
        assert batch.optimizer_normalization_count == 2.0
        assert batch.bucket_key.atom_count == 3
        assert batch.bucket_key.topology.support_size == 3
        assert batch.composition.token_ids.shape == (2, 8)
        assert batch.composition.atomic_numbers.shape == (2, 3)
        np.testing.assert_array_equal(batch.composition.composition_multiplicities, [2, 2])
        np.testing.assert_array_equal(batch.composition.sample_weights, [0.5, 0.5])
        np.testing.assert_array_equal(batch.cover.sample_weights, [1.0, 1.0])
        np.testing.assert_array_equal(batch.geometry.sample_weights, [1.0, 1.0])
        assert batch.geometry.metric.dtype == np.float64
        assert batch.geometry.lattice_latent.dtype == np.float64
        assert batch.geometry.intrinsic_coordinates.dtype == np.float64
        assert batch.geometry.period_basis.dtype == np.float64
        assert all(
            not value.flags.writeable
            for value in (
                batch.composition.token_ids,
                batch.composition.legal_mask,
                batch.cover.candidate_actions,
                batch.cover.residual_before,
                batch.geometry.metric,
            )
        )
        with pytest.raises(ValueError, match="read-only"):
            batch.geometry.metric[0, 0, 0] = 0.0

        chosen = batch.cover.candidate_actions[
            np.arange(2)[:, None],
            np.arange(batch.cover.selected_actions.shape[1])[None, :],
            batch.cover.selected_action_indices,
        ]
        np.testing.assert_array_equal(chosen, batch.cover.selected_actions)
        expected_after = (
            batch.cover.residual_before
            - batch.cover.row_multiplicities[None, :, None] * batch.cover.selected_actions
        )
        np.testing.assert_array_equal(expected_after, batch.cover.residual_after)


def test_composition_measure_uses_one_representative_and_disables_other_losses(
    tmp_path: Path,
) -> None:
    with _store(tmp_path) as store:
        dataset = StoredTrainingDataset(store, _codec(), MINIMAL_CATALOGUE)
        sampler = DeterministicBucketBatchSampler(dataset, 8, seed=8)
        assert sampler.measure is TrainingMeasure.UNIQUE_COMPOSITION
        batch = next(sampler)
        assert len(batch.record_ids) == 1
        assert batch.optimizer_normalization_count == 1.0
        np.testing.assert_array_equal(batch.composition.composition_multiplicities, [2])
        np.testing.assert_array_equal(batch.composition.sample_weights, [1.0])
        np.testing.assert_array_equal(batch.cover.sample_weights, [0.0])
        np.testing.assert_array_equal(batch.geometry.sample_weights, [0.0])


def test_sampler_resume_reproduces_the_exact_next_batch(tmp_path: Path) -> None:
    with _store(tmp_path) as store:
        dataset = StoredTrainingDataset(store, _codec(), MINIMAL_CATALOGUE)
        first = DeterministicBucketBatchSampler(
            dataset,
            1,
            seed=1234,
            measure=TrainingMeasure.STRUCTURE,
        )
        first_record = next(first).record_ids
        state = first.state_dict()
        expected_next = next(first).record_ids
        with pytest.raises(StopIteration):
            next(first)

        resumed = DeterministicBucketBatchSampler(
            dataset,
            1,
            seed=1234,
            measure=TrainingMeasure.STRUCTURE,
        )
        assert (
            next(
                DeterministicBucketBatchSampler(
                    dataset,
                    1,
                    seed=1234,
                    measure=TrainingMeasure.STRUCTURE,
                )
            ).record_ids
            == first_record
        )
        resumed.load_state_dict(state)
        assert resumed.cursor == 1
        assert next(resumed).record_ids == expected_next
        resumed.advance_epoch()
        assert resumed.epoch == 1
        assert resumed.cursor == 0

        wrong_measure = DeterministicBucketBatchSampler(dataset, 1, seed=1234)
        with pytest.raises(DomainError, match="different sampler policy"):
            wrong_measure.load_state_dict(state)


def test_sampler_refuses_epoch_advance_that_would_skip_records(tmp_path: Path) -> None:
    with _store(tmp_path) as store:
        dataset = StoredTrainingDataset(store, _codec(), MINIMAL_CATALOGUE)
        sampler = DeterministicBucketBatchSampler(
            dataset,
            1,
            seed=9,
            measure=TrainingMeasure.STRUCTURE,
        )
        with pytest.raises(DomainError, match="before the current epoch is exhausted"):
            sampler.advance_epoch()


def test_training_boundary_rejects_float32_crystallography(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with _store(tmp_path) as store:
        original = DatasetStore.__getitem__

        def corrupted(self: DatasetStore, key: int | str):  # type: ignore[no-untyped-def]
            record = original(self, key)
            metric = np.asarray(record.metric, dtype=np.float32)
            metric.setflags(write=False)
            return replace(record, metric=metric)

        monkeypatch.setattr(DatasetStore, "__getitem__", corrupted)
        with pytest.raises(DomainError, match="stored metric must be float64"):
            StoredTrainingDataset(store, _codec(), MINIMAL_CATALOGUE)


def test_training_boundary_rejects_consistent_target_substitution_under_build_pin(
    tmp_path: Path,
) -> None:
    compiled = _compiled_dataset(tmp_path)
    root = tmp_path / "target-substitution-store"
    manifest = write_compiled_dataset(compiled, root, records_per_shard=2)
    build_pin = compiled.manifest.fingerprint
    source_pin = compiled.source_manifest.fingerprint
    original = compiled.records[0].canonical_record.target
    changed = lift_exact_fixture_target(
        original.canonical_scaffold,
        original.metric * 1.21,
        catalogue=MINIMAL_CATALOGUE,
    )
    assert canonical_training_target_fingerprint(original) == (
        compiled.manifest.records[0].target_fingerprint
    )
    assert canonical_training_target_fingerprint(changed) != (
        compiled.manifest.records[0].target_fingerprint
    )
    with DatasetStore.open(root, expected_storage_fingerprint=manifest.fingerprint) as store:
        assert len(StoredTrainingDataset(store, _codec(), MINIMAL_CATALOGUE)) == 2

    shard_path = root / "shard-00000"
    metrics = np.load(shard_path / "metric.npy", allow_pickle=False)
    metrics[0] = changed.metric
    np.save(shard_path / "metric.npy", metrics, allow_pickle=False)
    offsets = np.load(shard_path / "lattice_offsets.npy", allow_pickle=False)
    latents = np.load(shard_path / "lattice_latent.npy", allow_pickle=False)
    latents[offsets[0] : offsets[1]] = changed.lattice_latent
    np.save(shard_path / "lattice_latent.npy", latents, allow_pickle=False)
    shard = manifest.shards[0]
    files = tuple(
        StoredFile.from_path(root / item.relative_path, relative_path=item.relative_path)
        for item in shard.files
    )
    updated = replace(
        manifest,
        shards=(replace(shard, files=files),),
        storage_fingerprint=None,
    )
    (root / "storage-manifest.json").write_bytes(
        storage_module._canonical_json_bytes(updated.to_dict())
    )

    with pytest.raises(DomainError, match="externally pinned manifest"):
        DatasetStore.open(root, expected_storage_fingerprint=manifest.fingerprint)
    # Storage byte/schema verification alone cannot prove target equivalence.
    # The consumer must authenticate the rebuilt target against the unchanged
    # trusted build fingerprint, even when the new tensors agree with each other.
    with DatasetStore.open(
        root,
        expected_build_manifest_fingerprint=build_pin,
        expected_source_manifest_fingerprint=source_pin,
    ) as store:
        assert store[0].target_fingerprint == compiled.manifest.records[0].target_fingerprint
        np.testing.assert_array_equal(store[0].metric, changed.metric)
        with pytest.raises(InvariantViolation, match="scientific target fingerprint"):
            StoredTrainingDataset(store, _codec(), MINIMAL_CATALOGUE)


def test_target_fingerprint_retains_exact_float64_distinctions(tmp_path: Path) -> None:
    target = _compiled_dataset(tmp_path).records[0].canonical_record.target
    metric = target.metric.copy()
    metric[0, 0] = np.nextafter(metric[0, 0], np.inf)
    changed = replace(target, metric=metric)
    assert canonical_training_target_fingerprint(changed) != (
        canonical_training_target_fingerprint(target)
    )


def _perturb_rebuilt_latent(monkeypatch: pytest.MonkeyPatch, delta: float) -> None:
    """Simulate rebuilding targets on a different BLAS/LAPACK numerical stack."""

    import crystalatlas.training_data as training_data_module

    original = training_data_module._target_for_record

    def rebuilt_elsewhere(*args, **kwargs):  # type: ignore[no-untyped-def]
        target = original(*args, **kwargs)
        latent = target.lattice_latent.copy()
        latent = latent + delta if delta else latent
        return replace(target, lattice_latent=latent)

    monkeypatch.setattr(training_data_module, "_target_for_record", rebuilt_elsewhere)


def test_cross_stack_rounding_of_derived_latent_is_accepted_and_bytes_stay_authenticated(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The Mac-built pilot latents differ from a Linux rebuild by at most 4.4e-16.
    _perturb_rebuilt_latent(monkeypatch, 4.0e-16)
    with _store(tmp_path) as store:
        stored = [store[index].lattice_latent.copy() for index in range(len(store))]
        dataset = StoredTrainingDataset(store, _codec(), MINIMAL_CATALOGUE)
        sampler = DeterministicBucketBatchSampler(
            dataset, 1, seed=3, measure=TrainingMeasure.STRUCTURE
        )
        served = {batch.record_ids[0]: batch.geometry.lattice_latent[0] for batch in sampler}
        for index, record_id in enumerate(store.record_ids):
            # Byte-authenticated stored values are served, never the local rebuild.
            np.testing.assert_array_equal(served[record_id], stored[index])


def test_rebuilt_latent_beyond_declared_tolerance_fails_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _perturb_rebuilt_latent(monkeypatch, 1.0e-9)
    with _store(tmp_path) as store, pytest.raises(InvariantViolation, match="lattice latent"):
        StoredTrainingDataset(store, _codec(), MINIMAL_CATALOGUE)
