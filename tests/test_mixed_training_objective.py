"""Training objective and audit on mixed-topology batches (review finding M5)."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest
from test_training_objectives import _setup

torch = pytest.importorskip("torch")

from crystalatlas.trainer import TrainingPhase  # noqa: E402
from crystalatlas.training_data import (  # noqa: E402
    DeterministicBucketBatchSampler,
    MixedTrainingBatch,
    TrainingBucketKey,
    TrainingMeasure,
)
from crystalatlas.training_objectives import ExactBatchCrystallographicAudit  # noqa: E402

pytestmark = pytest.mark.ml


def _structure_batch(dataset):  # type: ignore[no-untyped-def]
    return next(
        DeterministicBucketBatchSampler(dataset, 2, seed=11, measure=TrainingMeasure.STRUCTURE)
    )


def test_single_part_mixed_batch_reproduces_the_bucket_batch_losses(tmp_path: Path) -> None:
    store, model, dataset, objective, models = _setup(tmp_path)
    try:
        legacy = _structure_batch(dataset)
        mixed = MixedTrainingBatch(
            (legacy,), optimizer_normalization_count=legacy.optimizer_normalization_count
        )
        for phase in (TrainingPhase.COMPOSITION, TrainingPhase.COVER):
            expected = objective(models, legacy, phase)
            actual = objective(models, mixed, phase)
            assert actual.batch_size == expected.batch_size == 2
            for name in ("composition_nll", "cover_nll"):
                left, right = getattr(actual, name), getattr(expected, name)
                assert (left is None) == (right is None)
                if left is not None:
                    assert torch.equal(left, right), name
        torch.manual_seed(23)
        geometry = objective(models, mixed, TrainingPhase.GEOMETRY).cfm_regression
        torch.manual_seed(23)
        time = torch.rand(2, dtype=torch.float64)
        reference = objective._geometry.geometry_cfm_objective_padded(
            legacy.geometry.scaffolds,
            tuple(legacy.geometry.metric[index] for index in range(2)),
            time=time,
            sigma_z=objective.dequantization.sigma_lattice,
            sigma_eta=objective.dequantization.sigma_intrinsic,
            geometry_field=models.geometry,
        )
        assert torch.equal(geometry, reference.loss.per_sample.mean())
        audit = ExactBatchCrystallographicAudit(model)
        assert torch.equal(
            audit(mixed, torch.float64).residuals, audit(legacy, torch.float64).residuals
        )
    finally:
        store.close()


def test_mixed_losses_are_record_weighted_means_of_the_parts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store, model, dataset, objective, models = _setup(tmp_path)
    try:
        one = dataset._collate((0,), measure=TrainingMeasure.STRUCTURE)
        # A second, two-record fibre for the arithmetic only (losses are patched
        # below): relabelled records of the store's other batch.
        other = TrainingBucketKey(one.bucket_key.atom_count + 1, one.bucket_key.topology)
        pair = replace(
            dataset._collate((0, 1), measure=TrainingMeasure.STRUCTURE),
            record_ids=("synthetic-c", "synthetic-d"),
            bucket_key=other,
        )
        values = {one.record_ids: 1.0, pair.record_ids: 4.0}

        def fake(module, batch):  # type: ignore[no-untyped-def]
            return torch.tensor(values[batch.record_ids], dtype=torch.float64)

        monkeypatch.setattr(objective, "_cover_loss", fake)
        monkeypatch.setattr(objective, "_composition_loss", fake)
        losses = objective(models, MixedTrainingBatch((one, pair)), TrainingPhase.COVER)
        assert losses.batch_size == 3
        # (1 * 1.0 + 2 * 4.0) / 3, not the per-part mean 2.5.
        assert float(losses.cover_nll) == 3.0
    finally:
        store.close()


def test_mixed_batches_are_validated() -> None:
    with pytest.raises(Exception, match="one or more TrainingBatch parts"):
        MixedTrainingBatch(())
