from __future__ import annotations

from collections import Counter

import pytest

torch = pytest.importorskip("torch")

from crystalatlas.catalogue import MINIMAL_CATALOGUE  # noqa: E402
from crystalatlas.composition import Composition  # noqa: E402
from crystalatlas.nn.cover_sampling import NeuralCoverSampler  # noqa: E402
from crystalatlas.nn.selector import (  # noqa: E402
    ArithmeticCoverSelector,
    CoverSelectorConfig,
)

pytestmark = pytest.mark.ml


def test_neural_cover_sampling_is_rejection_free_and_formula_exact() -> None:
    torch.manual_seed(59)
    model = ArithmeticCoverSelector(
        CoverSelectorConfig(
            num_nu_choices=4,
            num_space_groups=230,
            row_feature_dim=6,
            model_dim=24,
            num_heads=4,
            num_layers=2,
            feedforward_dim=48,
        )
    ).eval()
    sampler = NeuralCoverSampler(
        model,
        catalogue=MINIMAL_CATALOGUE,
        max_raw_atoms=12,
    )
    generator = torch.Generator().manual_seed(101)
    compositions = (
        Composition(((8, 1),)),
        Composition(((8, 2), (14, 1))),
        Composition(((3, 1), (8, 1), (14, 1))),
    )
    for composition in compositions:
        for _ in range(8):
            result = sampler.sample(composition, generator=generator)
            actual = Counter()
            setting_rows = {row.label: row for row in result.rows}
            for orbit in result.cover.orbits:
                actual[orbit.atomic_number] += setting_rows[orbit.wyckoff_label].multiplicity
            expected = Counter(
                {z: result.cover.nu * count for z, count in composition.counts}
            )
            assert actual == expected
            assert result.log_probability <= 1.0e-6


def test_cover_sampler_rejects_an_absent_composition() -> None:
    """D0b null-composition rejection (theory 4.9, mutation 22)."""

    model = ArithmeticCoverSelector(
        CoverSelectorConfig(
            num_nu_choices=4,
            num_space_groups=230,
            row_feature_dim=6,
            model_dim=24,
            num_heads=4,
            num_layers=2,
            feedforward_dim=48,
        )
    ).eval()
    sampler = NeuralCoverSampler(model, catalogue=MINIMAL_CATALOGUE, max_raw_atoms=12)
    with pytest.raises(TypeError, match="composition must be a Composition"):
        sampler.sample(None, generator=torch.Generator().manual_seed(1))  # type: ignore[arg-type]
