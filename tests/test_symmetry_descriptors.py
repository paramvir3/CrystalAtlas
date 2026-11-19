"""Operation-derived group and site-symmetry descriptors (review finding m1)."""

from __future__ import annotations

from collections import defaultdict

import numpy as np
import pytest

from crystalatlas.full_catalogue import build_full_catalogue, pinned_hall_numbers
from crystalatlas.symmetry_descriptors import (
    GROUP_DESCRIPTOR_DIM,
    group_descriptor,
    rotation_type_histogram,
    site_stabilizer_rotations,
)

spglib = pytest.importorskip("spglib")


@pytest.fixture(scope="module")
def catalogue():  # type: ignore[no-untyped-def]
    return build_full_catalogue()


def test_site_stabilizers_have_the_catalogued_order_for_every_row(catalogue) -> None:  # type: ignore[no-untyped-def]
    checked = 0
    for setting in catalogue.settings.values():
        for row in setting.wyckoff_positions.values():
            assert len(site_stabilizer_rotations(setting, row)) == row.site_symmetry_order
            checked += 1
    assert checked > 1700  # every Wyckoff row of all 230 pinned settings


def test_rotation_histograms_separate_exactly_the_32_point_group_classes(catalogue) -> None:  # type: ignore[no-untyped-def]
    halls = pinned_hall_numbers()
    by_class: dict[str, set[tuple[float, ...]]] = defaultdict(set)
    for setting in catalogue.settings.values():
        spacegroup = spglib.get_spacegroup_type(halls[setting.international_number])
        symbol = spacegroup.pointgroup_international
        rotations = [np.asarray(r) for r in setting.point_group_rotations]
        by_class[symbol].add(tuple(np.round(rotation_type_histogram(rotations), 12)))
    assert len(by_class) == 32
    assert all(len(histograms) == 1 for histograms in by_class.values())
    assert len({next(iter(h)) for h in by_class.values()}) == 32


def test_group_descriptor_flags_and_shape(catalogue) -> None:  # type: ignore[no-untyped-def]
    numbers = (1, 2, 4, 19, 225, 229, 166)
    descriptor = {number: group_descriptor(catalogue.get(number)) for number in numbers}
    for value in descriptor.values():
        assert value.shape == (GROUP_DESCRIPTOR_DIM,)
    polar, chiral = -2, -1
    assert descriptor[1][polar] == 1.0 and descriptor[1][chiral] == 1.0  # P1
    assert descriptor[2][polar] == 0.0 and descriptor[2][chiral] == 0.0  # P-1
    assert descriptor[4][polar] == 1.0 and descriptor[4][chiral] == 1.0  # P2_1
    assert descriptor[19][polar] == 0.0 and descriptor[19][chiral] == 1.0  # P2_12_12_1
    assert descriptor[225][chiral] == 0.0 and descriptor[225][polar] == 0.0  # Fm-3m
    centring = slice(11, 16)  # P, A/B/C, I, F, R
    assert descriptor[225][centring].tolist() == [0.0, 0.0, 0.0, 1.0, 0.0]
    assert descriptor[229][centring].tolist() == [0.0, 0.0, 1.0, 0.0, 0.0]
    assert descriptor[166][centring].tolist() == [0.0, 0.0, 0.0, 0.0, 1.0]


def test_selector_with_descriptors_scores_groups_and_rows_end_to_end() -> None:
    torch = pytest.importorskip("torch")
    from dataclasses import replace

    from test_learned import _tiny_config

    from crystalatlas.catalogue import MINIMAL_CATALOGUE
    from crystalatlas.composition import Composition
    from crystalatlas.learned import CrystalAtlasModel

    base = _tiny_config()
    assert "selector_symmetry_descriptors" not in base.to_dict()["model"]
    config = replace(base, model=replace(base.model, selector_symmetry_descriptors=True))
    model = CrystalAtlasModel(config, catalogue=MINIMAL_CATALOGUE).double()
    selector = model.cover_selector
    assert selector.config.group_descriptor_dim == 18
    assert selector.config.row_feature_dim == 14
    assert "group_descriptors" not in selector.state_dict()
    sample = model.cover_sampler().sample(
        Composition.from_counts({8: 1}), generator=torch.Generator().manual_seed(3)
    )
    assert sample.cover.space_group in {"P1", "P-1"}
    legacy = CrystalAtlasModel(base, catalogue=MINIMAL_CATALOGUE)
    assert set(legacy.cover_selector.state_dict()) < set(selector.state_dict())
