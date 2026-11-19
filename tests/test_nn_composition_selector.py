from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from crystalatlas.nn.composition import (  # noqa: E402
    CanonicalCompositionTransformer,
    CompositionEncoder,
    CompositionTransformerConfig,
    apply_legal_mask,
)
from crystalatlas.nn.selector import (  # noqa: E402
    ArithmeticCoverSelector,
    CoverSelectorConfig,
)

pytestmark = pytest.mark.ml


def test_composition_encoder_is_permutation_invariant() -> None:
    torch.manual_seed(4)
    encoder = CompositionEncoder(16, element_dim=8, hidden_dim=24).double().eval()
    numbers = torch.tensor([8, 14, 26, 0])
    counts = torch.tensor([4.0, 1.0, 2.0, 0.0], dtype=torch.float64)
    permutation = torch.tensor([2, 0, 3, 1])
    first = encoder(numbers, counts)
    second = encoder(numbers[permutation], counts[permutation])
    torch.testing.assert_close(first, second, rtol=1.0e-12, atol=1.0e-12)


def test_composition_transformer_has_no_target_leakage_and_exact_mask() -> None:
    torch.manual_seed(7)
    config = CompositionTransformerConfig(
        vocab_size=13,
        num_token_types=4,
        model_dim=24,
        num_heads=4,
        num_layers=2,
        feedforward_dim=48,
        max_sequence_length=8,
    )
    model = CanonicalCompositionTransformer(config).double().eval()
    types = torch.tensor([[0, 1, 1, 2]])
    first = torch.tensor([[3, 5, 8, 2]])
    second = torch.tensor([[3, 11, 4, 9]])
    first_logits = model.teacher_forced_logits(first, types)
    second_logits = model.teacher_forced_logits(second, types)
    # Position one sees BOS and token zero only; its own changed target and all
    # later changes cannot affect the logits.
    torch.testing.assert_close(
        first_logits[:, :2], second_logits[:, :2], rtol=0.0, atol=0.0
    )

    raw = torch.tensor([[1.0, 2.0, 3.0]])
    masked = apply_legal_mask(raw, torch.tensor([[True, False, True]]))
    assert torch.isneginf(masked[0, 1])
    assert torch.softmax(masked, dim=-1)[0, 1].item() == 0.0
    with pytest.raises(ValueError, match="retain a legal action"):
        apply_legal_mask(raw, torch.zeros_like(raw, dtype=torch.bool))


def test_cover_row_score_sees_only_previous_actions() -> None:
    torch.manual_seed(12)
    config = CoverSelectorConfig(
        num_nu_choices=3,
        num_space_groups=5,
        row_feature_dim=6,
        model_dim=24,
        num_heads=4,
        num_layers=2,
        feedforward_dim=48,
    )
    model = ArithmeticCoverSelector(config).double().eval()
    composition = torch.randn(1, 24, dtype=torch.float64)
    nu = torch.tensor([1])
    group = torch.tensor([3])
    numbers = torch.tensor([[8, 14]])
    rows = torch.randn(1, 3, 6, dtype=torch.float64)
    previous = torch.tensor([[[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]], dtype=torch.float64)
    changed = previous.clone()
    changed[:, 2] = torch.tensor([7.0, 4.0], dtype=torch.float64)
    candidates = torch.tensor(
        [[[[0.0, 0.0], [1.0, 0.0]], [[0.0, 1.0], [1.0, 1.0]], [[1.0, 0.0], [0.0, 1.0]]]],
        dtype=torch.float64,
    )
    legal = torch.ones(1, 3, 2, dtype=torch.bool)
    logits = model.row_logits(
        composition,
        nu,
        group,
        numbers,
        rows,
        previous,
        candidates,
        legal_mask=legal,
    )
    changed_logits = model.row_logits(
        composition,
        nu,
        group,
        numbers,
        rows,
        changed,
        candidates,
        legal_mask=legal,
    )
    # The last realized action would first condition a nonexistent fourth row.
    torch.testing.assert_close(logits, changed_logits, rtol=0.0, atol=0.0)

