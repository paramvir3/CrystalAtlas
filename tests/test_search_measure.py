"""Equation (3.15a) search measure on the stored training path (review m6).

    S ~ Unif(S_train),  c ~ Unif(C_obs(S)),  C ~ Unif(P_obs(c))

The discovery-oriented S-DNG checkpoint uses this system-, formula- and
polymorph-balanced measure (theory S4).  Every retained record is visited once.
Record i carries the structure weight w_i = N p_i and the composition weight
N P(c_i) / n(c_i), with N the view size, p_i its 3.15a probability, P(c) the
formula mass and n(c) the records of formula c.  An epoch mean of w_i l_i is then
exactly sum_i p_i l_i, and the composition term is exactly sum_c P(c) l_c.
Records with one identity polymorph key share that polymorph's mass; this
grouping is reversible and discards nothing.
"""

from __future__ import annotations

from fractions import Fraction
from pathlib import Path

import numpy as np
import pytest

from crystalatlas.data import CrystalRecord, record_probabilities
from crystalatlas.data import TrainingMeasure as InMemoryMeasure
from crystalatlas.errors import DomainError
from crystalatlas.training_data import system_formula_polymorph_weights


def test_weights_follow_equation_3_15a_exactly() -> None:
    systems = ("A", "A", "A", "A", "B")
    formulas = ("a1", "a1", "a1", "a2", "b1")
    polymorphs = ("p1", "p2", "p2", "p3", "p4")
    structure, composition = system_formula_polymorph_weights(systems, formulas, polymorphs)
    # |S| = 2; A has formulas {a1, a2}; a1 has polymorphs {p1, p2}; p2 has two records.
    probability = (
        Fraction(1, 8), Fraction(1, 16), Fraction(1, 16), Fraction(1, 4), Fraction(1, 2)
    )
    assert structure == tuple(5 * value for value in probability)
    assert composition == (
        Fraction(5, 12), Fraction(5, 12), Fraction(5, 12), Fraction(5, 4), Fraction(5, 2)
    )
    assert sum(structure) == 5 and sum(composition) == 5
    # Composition weights depend on c only and integrate to N P(c).
    assert composition[0] * 3 == 5 * (probability[0] + probability[1] + probability[2])


def test_weights_agree_with_the_in_memory_reference_implementation() -> None:
    rng = np.random.default_rng(3)
    count = 200
    system_ids = rng.integers(0, 6, size=count)
    formula_ids = system_ids * 10 + rng.integers(0, 4, size=count)
    polymorph_ids = formula_ids * 10 + rng.integers(0, 3, size=count)
    records = [
        CrystalRecord(
            record_id=f"r{index}",
            species=(int(system), int(system) + 100),
            composition_key=f"c{formula}",
            structure_key=f"s{polymorph}",
            source="fixture",
            payload_index=index,
        )
        for index, (system, formula, polymorph) in enumerate(
            zip(system_ids, formula_ids, polymorph_ids, strict=True)
        )
    ]
    reference = record_probabilities(records, InMemoryMeasure.SYSTEM_FORMULA_POLYMORPH)
    structure, _ = system_formula_polymorph_weights(
        [str(record.species) for record in records],
        [record.composition_key for record in records],
        [record.structure_key for record in records],
    )
    np.testing.assert_allclose(
        np.asarray([float(value) for value in structure]) / count, reference, rtol=1e-15
    )


@pytest.mark.parametrize(
    ("systems", "formulas", "polymorphs", "message"),
    [
        ((), (), (), "nonempty"),
        (("A",), ("a", "b"), ("p",), "align"),
        (("A", "B"), ("a", "a"), ("p", "q"), "one species system"),
        (("A", "A"), ("a", "b"), ("p", "p"), "one formula"),
    ],
)
def test_inconsistent_identity_keys_are_refused(systems, formulas, polymorphs, message) -> None:  # type: ignore[no-untyped-def]
    with pytest.raises(DomainError, match=message):
        system_formula_polymorph_weights(systems, formulas, polymorphs)


def _dataset(tmp_path: Path):  # type: ignore[no-untyped-def]
    from test_training_data import _codec, _store

    from crystalatlas.catalogue import MINIMAL_CATALOGUE
    from crystalatlas.training_data import StoredTrainingDataset

    store = _store(tmp_path)
    return store, StoredTrainingDataset(store, _codec(), MINIMAL_CATALOGUE)


def test_dataset_batches_carry_the_declared_search_weights(tmp_path: Path) -> None:
    from crystalatlas.training_data import DeterministicBucketBatchSampler, TrainingMeasure

    store, dataset = _dataset(tmp_path)
    try:
        identities = [store[record_id].identity for record_id in dataset.record_ids]
        expected = system_formula_polymorph_weights(
            [item.element_set_key for item in identities],
            [item.composition_key for item in identities],
            [item.polymorph_key for item in identities],
        )
        declared = dataset.search_measure_weights
        assert tuple(declared[record_id][0] for record_id in dataset.record_ids) == tuple(
            float(value) for value in expected[0]
        )
        sampler = DeterministicBucketBatchSampler(
            dataset, 8, seed=5, measure=TrainingMeasure.SYSTEM_FORMULA_POLYMORPH
        )
        batch = next(sampler)
        assert batch.measure is TrainingMeasure.SYSTEM_FORMULA_POLYMORPH
        assert set(batch.record_ids) == set(dataset.record_ids)  # every record, once
        for index, record_id in enumerate(batch.record_ids):
            assert batch.cover.sample_weights[index] == declared[record_id][0]
            assert batch.geometry.sample_weights[index] == declared[record_id][0]
            assert batch.composition.sample_weights[index] == declared[record_id][1]
    finally:
        store.close()


torch = pytest.importorskip("torch")


def _objective_setup(tmp_path: Path):  # type: ignore[no-untyped-def]
    from test_training_objectives import _setup

    return _setup(tmp_path)


def test_search_measure_reduces_to_structure_when_polymorphs_are_balanced(
    tmp_path: Path,
) -> None:
    """Two polymorphs of one formula: 3.15a weights are all 1, so losses agree."""

    from crystalatlas.trainer import TrainingPhase
    from crystalatlas.training_data import DeterministicBucketBatchSampler, TrainingMeasure
    from crystalatlas.training_objectives import (
        CrystalAtlasTrainingObjective,
        ObjectiveDataMeasure,
    )

    store, model, dataset, legacy, models = _objective_setup(tmp_path)
    try:
        measure = ObjectiveDataMeasure.from_dataset(dataset, search_measure=True)
        # Legacy objectives keep their contract; the opt-in adds only a digest.
        assert set(measure.to_dict()) - set(legacy.data_measure.to_dict()) == {
            "search_measure_weights_sha256"
        }
        objective = CrystalAtlasTrainingObjective(model, measure)
        batches = {
            measure: next(DeterministicBucketBatchSampler(dataset, 2, seed=17, measure=measure))
            for measure in (TrainingMeasure.STRUCTURE, TrainingMeasure.SYSTEM_FORMULA_POLYMORPH)
        }
        assert all(value == (1.0, 1.0) for value in dataset.search_measure_weights.values())
        results = {}
        for measure, batch in batches.items():
            torch.manual_seed(23)
            results[measure] = objective(models, batch, TrainingPhase.JOINT)
        left = results[TrainingMeasure.STRUCTURE]
        right = results[TrainingMeasure.SYSTEM_FORMULA_POLYMORPH]
        for name in ("composition_nll", "cover_nll", "cfm_regression"):
            torch.testing.assert_close(
                getattr(left, name), getattr(right, name), rtol=0, atol=0
            )
    finally:
        store.close()


def test_objective_applies_non_uniform_search_weights_exactly(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from dataclasses import replace
    from types import MappingProxyType, SimpleNamespace

    from crystalatlas.trainer import TrainingPhase
    from crystalatlas.training_data import DeterministicBucketBatchSampler, TrainingMeasure
    from crystalatlas.training_objectives import (
        CrystalAtlasTrainingObjective,
        ObjectiveDataMeasure,
    )

    store, model, dataset, _, models = _objective_setup(tmp_path)
    try:
        batch = next(
            DeterministicBucketBatchSampler(
                dataset, 2, seed=17, measure=TrainingMeasure.SYSTEM_FORMULA_POLYMORPH
            )
        )
        weights = {batch.record_ids[0]: (1.5, 0.25), batch.record_ids[1]: (0.5, 1.75)}
        structure = np.asarray([weights[r][0] for r in batch.record_ids])
        composition = np.asarray([weights[r][1] for r in batch.record_ids])
        weighted = replace(
            batch,
            composition=replace(batch.composition, sample_weights=composition),
            cover=replace(batch.cover, sample_weights=structure),
            geometry=replace(batch.geometry, sample_weights=structure),
        )
        base = ObjectiveDataMeasure.from_dataset(dataset)
        objective = CrystalAtlasTrainingObjective(
            model, replace(base, search_measure_weights=MappingProxyType(weights))
        )
        unit = CrystalAtlasTrainingObjective(model, base)
        per_record = {}
        for record_id in batch.record_ids:
            single = next(
                b
                for b in DeterministicBucketBatchSampler(
                    dataset, 1, seed=17, measure=TrainingMeasure.STRUCTURE
                )
                if b.record_ids == (record_id,)
            )
            losses = unit(models, single, TrainingPhase.COVER)
            per_record[record_id] = losses.cover_nll
        cover = objective(models, weighted, TrainingPhase.COVER).cover_nll
        expected = sum(
            weights[r][0] * per_record[r] for r in batch.record_ids
        ) / len(batch.record_ids)
        torch.testing.assert_close(cover, expected, rtol=1e-12, atol=1e-12)
        fake = torch.tensor([2.0, 3.0], dtype=torch.float64)
        monkeypatch.setattr(
            objective._geometry,
            "geometry_cfm_objective_batch",
            lambda *args, **kwargs: SimpleNamespace(loss=SimpleNamespace(per_sample=fake)),
        )
        cfm = objective(models, weighted, TrainingPhase.GEOMETRY).cfm_regression
        torch.testing.assert_close(
            cfm, (torch.as_tensor(structure) * fake).mean(), rtol=0, atol=0
        )
        # Weights that differ from the declared measure are refused.
        with pytest.raises(Exception, match="declared search-measure weights"):
            unit(models, weighted, TrainingPhase.COVER)
    finally:
        store.close()


def test_mixed_topology_batches_apply_the_same_search_weights(tmp_path: Path) -> None:
    from crystalatlas.trainer import TrainingPhase
    from crystalatlas.training_data import (
        DeterministicBucketBatchSampler,
        MixedTopologyBatchSampler,
        TrainingMeasure,
    )
    from crystalatlas.training_objectives import (
        CrystalAtlasTrainingObjective,
        ObjectiveDataMeasure,
    )

    store, model, dataset, _, models = _objective_setup(tmp_path)
    try:
        objective = CrystalAtlasTrainingObjective(
            model, ObjectiveDataMeasure.from_dataset(dataset, search_measure=True)
        )
        measure = TrainingMeasure.SYSTEM_FORMULA_POLYMORPH
        mixed = next(MixedTopologyBatchSampler(dataset, 2, seed=29, measure=measure))
        bucket = next(DeterministicBucketBatchSampler(dataset, 2, seed=29, measure=measure))
        assert mixed.measure is measure
        assert sorted(r for part in mixed.parts for r in part.record_ids) == sorted(
            bucket.record_ids
        )
        for phase, name in (
            (TrainingPhase.COMPOSITION, "composition_nll"),
            (TrainingPhase.COVER, "cover_nll"),
        ):
            torch.testing.assert_close(
                getattr(objective(models, mixed, phase), name),
                getattr(objective(models, bucket, phase), name),
                rtol=1e-12,
                atol=1e-12,
            )
        torch.manual_seed(31)
        cfm = objective(models, mixed, TrainingPhase.GEOMETRY).cfm_regression
        assert cfm is not None and torch.isfinite(cfm)
    finally:
        store.close()
