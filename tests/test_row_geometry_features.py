"""Geometric Wyckoff-row features for the cover selector (review 2026-10-29, M1).

The legacy row features (multiplicity, free dimension, site order, fixed flag)
and the opt-in site-symmetry histogram cannot tell Pm-3m 12h (x, 1/2, 0) from
12i (0, y, y), nor I4/mmm 8g from 8h, which are different crystal chemistry.
The geometry descriptor is the exact chart (offset and coordinate matrix) of the
row's canonical representative under the full Euclidean normalizer, plus the
canonical row index: equivalent rows share the class part, inequivalent rows
differ, and the index breaks the gauge tie the way the data canonicalization
does.
"""

from __future__ import annotations

import numpy as np
import pytest

from crystalatlas.cover import canonical_rows
from crystalatlas.full_catalogue import build_space_group_setting
from crystalatlas.symmetry_descriptors import (
    ROW_GEOMETRY_DESCRIPTOR_DIM,
    ROW_SITE_DESCRIPTOR_DIM,
    row_geometry_descriptor,
    row_normalizer_classes,
)


def _descriptor(number: int, label: str) -> np.ndarray:
    setting = build_space_group_setting(number)
    return row_geometry_descriptor(setting, setting.wyckoff(label))


@pytest.mark.parametrize(
    ("number", "equivalent", "distinct"),
    [
        # Pm-3m: N_E = Im-3m, the extra coset is t = (1/2, 1/2, 1/2); it swaps
        # 1a/1b, 3c/3d, 6e/6f, 12i/12j and 24k/24l and fixes 8g, 12h, 24m, 48n.
        (
            221,
            [("1a", "1b"), ("3c", "3d"), ("6e", "6f"), ("12i", "12j"), ("24k", "24l")],
            [("12h", "12i"), ("12h", "12j"), ("24k", "24m"), ("8g", "12h")],
        ),
        # I4/mmm: N_E = P4/mmm ((a-b)/2, (a+b)/2, c/2), extra coset t = (0, 0, 1/2).
        (139, [("8i", "8j"), ("2a", "2b")], [("8g", "8h"), ("8g", "8i"), ("8h", "8i")]),
        # Fm-3m: N_E = Pm-3m (a/2, b/2, c/2), extra coset t = (1/2, 1/2, 1/2).
        (225, [("4a", "4b"), ("48h", "48i")], [("48g", "48h"), ("48g", "48i"), ("96j", "96k")]),
        # P6_3/mmc: N_E = P6/mmm (a, b, c/2), extra coset t = (0, 0, 1/2).
        (194, [("2c", "2d")], [("2a", "2b"), ("2b", "2c"), ("4e", "4f"), ("12j", "12k")]),
    ],
)
def test_class_part_is_normalizer_invariant_and_separates_inequivalent_rows(  # type: ignore[no-untyped-def]
    number, equivalent, distinct
) -> None:
    class_part = slice(0, ROW_GEOMETRY_DESCRIPTOR_DIM - 1)
    for left, right in equivalent:
        a, b = _descriptor(number, left), _descriptor(number, right)
        assert a.shape == (ROW_GEOMETRY_DESCRIPTOR_DIM,)
        np.testing.assert_array_equal(a[class_part], b[class_part])
        assert a[-1] != b[-1]  # the own canonical index differs
    for left, right in distinct:
        a, b = _descriptor(number, left), _descriptor(number, right)
        assert not np.array_equal(a[class_part], b[class_part]), (left, right)


def test_classes_partition_the_rows_and_use_the_first_canonical_row() -> None:
    setting = build_space_group_setting(221)
    classes = row_normalizer_classes(setting)
    rows = canonical_rows(setting)
    labels = [row.label for row in rows]
    assert sorted(label for group in classes.values() for label in group) == sorted(labels)
    for representative, members in classes.items():
        assert representative == min(members, key=labels.index)
    assert classes["1a"] == ("1a", "1b") and classes["3c"] == ("3c", "3d")
    assert classes["12i"] == ("12i", "12j") and classes["12h"] == ("12h",)
    assert classes["24k"] == ("24k", "24l") and classes["24m"] == ("24m",)
    assert len(rows) <= 32  # the index entries are divided by 32 (largest table: 27)


def test_descriptor_is_the_exact_chart_of_the_canonical_row() -> None:
    setting = build_space_group_setting(221)
    value = _descriptor(221, "12i")  # (0, y, y): canonical in its own class
    row = setting.wyckoff("12i")
    assert value[:3].tolist() == [float(v) for v in row.representative_offset]
    assert value[3:6].tolist() == [float(v) for v in row.coordinate_matrix[0]]
    assert value[6:12].tolist() == [0.0] * 6  # one free coordinate: rows 2-3 padded
    assert value[-2] == value[-1] == canonical_rows(setting).index(
        next(r for r in canonical_rows(setting) if r.label == "12i")
    ) / 32.0
    moved = _descriptor(221, "1b")  # canonical representative is 1a
    assert moved[:3].tolist() == [0.0, 0.0, 0.0] and moved[-2] == 0.0 and moved[-1] != 0.0


def test_row_features_append_the_geometry_block_only_when_asked() -> None:
    torch = pytest.importorskip("torch")
    from crystalatlas.errors import DomainError
    from crystalatlas.nn.cover_sampling import operation_derived_row_features

    setting = build_space_group_setting(139)
    rows = canonical_rows(setting)
    reference = torch.zeros(1, dtype=torch.float64)
    legacy = operation_derived_row_features(setting, rows, feature_dim=4, reference=reference)
    site = operation_derived_row_features(
        setting, rows, feature_dim=4 + ROW_SITE_DESCRIPTOR_DIM, reference=reference
    )
    geometry = operation_derived_row_features(
        setting, rows, feature_dim=4 + ROW_GEOMETRY_DESCRIPTOR_DIM, reference=reference,
        geometry=True,
    )
    both = operation_derived_row_features(
        setting, rows,
        feature_dim=4 + ROW_SITE_DESCRIPTOR_DIM + ROW_GEOMETRY_DESCRIPTOR_DIM,
        reference=reference, geometry=True,
    )
    torch.testing.assert_close(site[:, :4], legacy, rtol=0, atol=0)
    torch.testing.assert_close(geometry[:, :4], legacy, rtol=0, atol=0)
    torch.testing.assert_close(both[:, : 4 + ROW_SITE_DESCRIPTOR_DIM], site, rtol=0, atol=0)
    torch.testing.assert_close(
        both[:, 4 + ROW_SITE_DESCRIPTOR_DIM :], geometry[:, 4:], rtol=0, atol=0
    )
    with pytest.raises(DomainError, match="geometry block"):
        operation_derived_row_features(
            setting, rows, feature_dim=4 + ROW_GEOMETRY_DESCRIPTOR_DIM - 1,
            reference=reference, geometry=True,
        )
    index = {row.label: i for i, row in enumerate(rows)}
    # 8g, 8h and 8i collide on every legacy feature and separate on the geometry block.
    assert torch.equal(site[index["8g"]], site[index["8h"]])
    assert not torch.equal(geometry[index["8g"], 4:], geometry[index["8h"], 4:])
    assert not torch.equal(geometry[index["8h"], 4:], geometry[index["8i"], 4:])


def test_model_option_is_versioned_and_sizes_the_selector() -> None:
    torch = pytest.importorskip("torch")
    from dataclasses import replace

    from test_learned import _tiny_config

    from crystalatlas.catalogue import MINIMAL_CATALOGUE
    from crystalatlas.composition import Composition
    from crystalatlas.errors import DomainError
    from crystalatlas.learned import CrystalAtlasModel, LearnedCrystalGenerator

    base = _tiny_config()
    assert base.model.selector_row_geometry_features is False
    assert "selector_row_geometry_features" not in base.to_dict()["model"]
    config = replace(base, model=replace(base.model, selector_row_geometry_features=True))
    assert config.fingerprint != base.fingerprint
    assert config.to_dict()["model"]["selector_row_geometry_features"] is True
    with pytest.raises(DomainError, match="selector_row_geometry_features"):
        replace(base.model, selector_row_geometry_features=1)
    torch.manual_seed(3)
    model = CrystalAtlasModel(config, catalogue=MINIMAL_CATALOGUE).double()
    assert model.cover_selector.config.row_geometry_features is True
    assert model.cover_selector.config.row_feature_dim == 4 + ROW_GEOMETRY_DESCRIPTOR_DIM
    generator = LearnedCrystalGenerator(model, allow_untrained=True)
    sample = generator._cover_sampler.sample(
        Composition(((8, 1),)), generator=torch.Generator().manual_seed(1)
    )
    assert sample.cover.space_group in {"P1", "P-1"}
