"""Primitive-atom-count conditioning track (review finding M6).

At generic parameters the translation index of a cover is a function of its
discrete table alone, ``J_gen = I_G * h_gen`` with a hidden index ``h_gen``.
Conditioning on ``N_prim`` is then the exact discrete constraint
``nu |c| = I_G * N_prim`` on (nu, G), shared by sampler and targets, plus the rare
exclusion of tables with ``h_gen > 1``.  A table attains ``h_gen = 1`` iff some
parameter choice decodes without hidden translations (``h >= h_gen >= 1``).
"""

from __future__ import annotations

import pytest

from crystalatlas.catalogue import MINIMAL_CATALOGUE
from crystalatlas.composition import Composition
from crystalatlas.constraints import SuffixReachability
from crystalatlas.cover import canonical_rows
from crystalatlas.cover_legality import (
    AtomBound,
    PrimitiveCountCondition,
    feasible_formula_units,
    generic_hidden_translation_index,
    legal_settings,
)
from crystalatlas.errors import DomainError
from crystalatlas.geometry import ContinuousCover, OrbitType


def _oracle(composition, nu, setting):  # type: ignore[no-untyped-def]
    return SuffixReachability(composition, nu, canonical_rows(setting))


def test_condition_admits_exactly_the_primitive_count() -> None:
    condition = PrimitiveCountCondition(6)
    assert condition.admits(nu=2, formula_atoms=3, centering_index=1)
    assert not condition.admits(nu=1, formula_atoms=3, centering_index=1)
    assert condition.admits(nu=8, formula_atoms=3, centering_index=4)
    with pytest.raises(DomainError):
        PrimitiveCountCondition(0)


def test_conditioned_masks_select_only_matching_formula_units() -> None:
    composition = Composition.from_counts({8: 1})
    settings = tuple(MINIMAL_CATALOGUE.settings.values())
    bound = AtomBound(conventional=40)
    condition = PrimitiveCountCondition(4)
    nus = feasible_formula_units(
        composition, settings, max_formula_units=16, bound=bound, oracle=_oracle,
        condition=condition,
    )
    assert nus == (4,)
    for nu in range(1, 9):
        expected = nu == 4
        assert bool(
            legal_settings(
                composition, nu, settings, bound=bound, oracle=_oracle, condition=condition
            )
        ) is expected


def test_generic_hidden_index_detects_a_body_centred_fixed_point_table() -> None:
    oxygen = Composition.from_counts({8: 1})
    body_centred = ContinuousCover(
        2, "P-1", oxygen, (OrbitType("1a", 8), OrbitType("1h", 8))
    )
    general = ContinuousCover(4, "P-1", oxygen, (OrbitType("2i", 8), OrbitType("2i", 8)))
    assert generic_hidden_translation_index(body_centred, MINIMAL_CATALOGUE) == 2
    assert generic_hidden_translation_index(general, MINIMAL_CATALOGUE) == 1


def test_learned_csp_samples_honour_the_primitive_atom_count() -> None:
    torch = pytest.importorskip("torch")
    from test_learned import _tiny_config

    from crystalatlas.learned import CrystalAtlasModel, LearnedCrystalGenerator
    from crystalatlas.pipeline import GenerationMode, GenerationRequest

    torch.manual_seed(37)
    model = CrystalAtlasModel(_tiny_config(), catalogue=MINIMAL_CATALOGUE).double()
    generator = LearnedCrystalGenerator(model, allow_untrained=True)
    composition = Composition.from_counts({8: 1})
    for primitive_atoms in (1, 2):
        traces = generator.sample(
            GenerationRequest(
                mode=GenerationMode.CSP,
                composition=composition,
                samples=6,
                seed=11,
                primitive_atom_count=primitive_atoms,
            )
        )
        for trace in traces:
            assert trace.sample is not None, trace.error
            cover = trace.sample.cover_sample.cover
            centering = MINIMAL_CATALOGUE.get(cover.space_group).centering_index
            assert cover.nu * composition.total_atoms == primitive_atoms * centering
            assert generic_hidden_translation_index(cover, MINIMAL_CATALOGUE) == 1
    with pytest.raises(DomainError, match="CSP only"):
        GenerationRequest(mode=GenerationMode.DNG, primitive_atom_count=4)


def test_primitive_count_enters_the_masked_law_not_a_post_filter(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    """S3 contract: the condition shapes the selector's (nu, G) masks.

    Post-filtering unconditioned draws would also return samples with the
    right count, but from a different law.  The selector must see only the
    conditioned formula units (theory 4.9, mutation 7).
    """

    torch = pytest.importorskip("torch")
    from test_learned import _tiny_config

    from crystalatlas.learned import CrystalAtlasModel

    torch.manual_seed(41)
    model = CrystalAtlasModel(_tiny_config(), catalogue=MINIMAL_CATALOGUE).double()
    selector = model.cover_selector
    original = selector.nu_logits
    masks = []

    def recording(context, *, legal_mask):  # type: ignore[no-untyped-def]
        masks.append(legal_mask.detach().clone())
        return original(context, legal_mask=legal_mask)

    monkeypatch.setattr(selector, "nu_logits", recording)
    sampler = model.cover_sampler()
    composition = Composition.from_counts({8: 1})
    for seed in range(4):
        sample = sampler.sample(
            composition, generator=torch.Generator().manual_seed(seed), primitive_atom_count=2
        )
        assert sample.cover.nu == 2
    assert masks
    for mask in masks:
        assert [int(index) + 1 for index in torch.nonzero(mask[0]).flatten()] == [2]
