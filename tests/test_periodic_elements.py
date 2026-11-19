"""Exact periodic-table coordinates and the ``periodic-v1`` element embedding."""

from __future__ import annotations

import dataclasses
from collections import Counter

import pytest

from crystalatlas.config import CrystalAtlasConfig, ModelConfig
from crystalatlas.elements import (
    ELEMENT_SYMBOLS,
    LONG_FORM_COLUMNS,
    NOBLE_GAS_ATOMIC_NUMBERS,
    PERIODIC_BLOCKS,
    PeriodicCoordinates,
    atomic_number,
    periodic_coordinates,
)
from crystalatlas.errors import DomainError

ALL = {symbol: periodic_coordinates(atomic_number(symbol)) for symbol in ELEMENT_SYMBOLS}


def _cell(symbol: str) -> tuple[int, int, str]:
    item = ALL[symbol]
    return item.period, item.column, item.block


def test_every_element_has_its_own_cell_of_the_long_form() -> None:
    assert len(ALL) == 118
    cells = {(item.period, item.column) for item in ALL.values()}
    assert len(cells) == 118
    assert all(1 <= item.period <= 7 for item in ALL.values())
    assert all(1 <= item.column <= LONG_FORM_COLUMNS for item in ALL.values())
    assert all(item.block in PERIODIC_BLOCKS for item in ALL.values())
    lengths = Counter(item.period for item in ALL.values())
    assert [lengths[period] for period in range(1, 8)] == [2, 8, 8, 18, 18, 32, 32]
    # Columns increase with atomic number inside a period.
    for symbol, later in zip(ELEMENT_SYMBOLS, ELEMENT_SYMBOLS[1:], strict=False):
        if ALL[symbol].period == ALL[later].period:
            assert ALL[symbol].column < ALL[later].column


def test_known_cells() -> None:
    expected = {
        "H": (1, 1, "s"),
        "He": (1, 32, "s"),
        "Li": (2, 1, "s"),
        "Be": (2, 2, "s"),
        "B": (2, 27, "p"),
        "O": (2, 30, "p"),
        "Ne": (2, 32, "p"),
        "Si": (3, 28, "p"),
        "Ca": (4, 2, "s"),
        "Sc": (4, 17, "d"),
        "Fe": (4, 22, "d"),
        "Zn": (4, 26, "d"),
        "Ga": (4, 27, "p"),
        "Y": (5, 17, "d"),
        "Xe": (5, 32, "p"),
        "Ba": (6, 2, "s"),
        "La": (6, 3, "f"),
        "Gd": (6, 10, "f"),
        "Yb": (6, 16, "f"),
        "Lu": (6, 17, "d"),
        "Hf": (6, 18, "d"),
        "Hg": (6, 26, "d"),
        "Tl": (6, 27, "p"),
        "Ac": (7, 3, "f"),
        "U": (7, 6, "f"),
        "No": (7, 16, "f"),
        "Lr": (7, 17, "d"),
        "Og": (7, 32, "p"),
    }
    assert {symbol: _cell(symbol) for symbol in expected} == expected


def test_chemical_families_share_a_column() -> None:
    families = {
        1: ("H", "Li", "Na", "K", "Rb", "Cs", "Fr"),
        2: ("Be", "Mg", "Ca", "Sr", "Ba", "Ra"),
        17: ("Sc", "Y", "Lu", "Lr"),
        18: ("Ti", "Zr", "Hf", "Rf"),
        25: ("Cu", "Ag", "Au", "Rg"),
        28: ("C", "Si", "Ge", "Sn", "Pb", "Fl"),
        30: ("O", "S", "Se", "Te", "Po", "Lv"),
        31: ("F", "Cl", "Br", "I", "At", "Ts"),
        32: ("He", "Ne", "Ar", "Kr", "Xe", "Rn", "Og"),
    }
    for column, symbols in families.items():
        assert {ALL[symbol].column for symbol in symbols} == {column}
        members = {symbol for symbol, item in ALL.items() if item.column == column}
        assert members == set(symbols)


def test_blocks_and_eighteen_column_groups() -> None:
    blocks = Counter(item.block for item in ALL.values())
    assert blocks == {"s": 14, "f": 28, "d": 40, "p": 36}
    for symbol, item in ALL.items():
        if symbol == "He":
            assert (item.column, item.block) == (32, "s")
        elif item.column <= 2:
            assert item.block == "s"
        elif item.column <= 16:
            assert item.block == "f" and item.period in (6, 7)
        elif item.column <= 26:
            assert item.block == "d" and item.period >= 4
        else:
            assert item.block == "p" and item.period >= 2
    # The f block is exactly La-Yb and Ac-No.
    f_block = [symbol for symbol in ELEMENT_SYMBOLS if ALL[symbol].block == "f"]
    assert f_block == list(ELEMENT_SYMBOLS[56:70]) + list(ELEMENT_SYMBOLS[88:102])
    # IUPAC groups 1-18 of the elements outside the f block.
    groups = {"Na": 1, "Mg": 2, "Sc": 3, "Ti": 4, "Cr": 6, "Fe": 8, "Ni": 10, "Cu": 11}
    groups |= {"Zn": 12, "Al": 13, "C": 14, "N": 15, "O": 16, "Cl": 17, "Ar": 18, "He": 18}
    for symbol, group in groups.items():
        column = ALL[symbol].column
        assert (column if column <= 2 else column - 14) == group
    assert NOBLE_GAS_ATOMIC_NUMBERS == tuple(
        atomic_number(symbol) for symbol in ("He", "Ne", "Ar", "Kr", "Xe", "Rn", "Og")
    )


def test_coordinates_refuse_non_elements() -> None:
    for value in (0, 119, -1, True):
        with pytest.raises(DomainError):
            periodic_coordinates(value)  # type: ignore[arg-type]
    assert periodic_coordinates(26) == PeriodicCoordinates(period=4, column=22, block="d")


def test_configuration_identity() -> None:
    legacy = CrystalAtlasConfig()
    assert legacy.model.element_embedding == "learned-v1"
    assert "element_embedding" not in legacy.to_dict()["model"]
    periodic = dataclasses.replace(
        legacy, model=dataclasses.replace(legacy.model, element_embedding="periodic-v1")
    )
    assert periodic.to_dict()["model"]["element_embedding"] == "periodic-v1"
    assert periodic.fingerprint != legacy.fingerprint
    without = dict(periodic.to_dict()["model"])
    del without["element_embedding"]
    assert without == legacy.to_dict()["model"]
    with pytest.raises(DomainError, match="element_embedding"):
        ModelConfig(element_embedding="periodic-v2")


torch = pytest.importorskip("torch")

from crystalatlas.nn.composition import (  # noqa: E402
    ELEMENT_EMBEDDINGS,
    CompositionEncoder,
    PeriodicElementEmbedding,
    make_element_embedding,
)
from crystalatlas.nn.selector import CountVectorEncoder  # noqa: E402


@pytest.mark.ml
def test_embedding_is_the_sum_of_its_three_tables() -> None:
    torch.manual_seed(3)
    module = PeriodicElementEmbedding(119, 12).double()
    numbers = torch.arange(119)
    values = module(numbers)
    assert values.shape == (119, 12)
    assert torch.equal(values[0], torch.zeros(12, dtype=torch.float64))
    for symbol, item in ALL.items():
        z = atomic_number(symbol)
        expected = (
            module.period.weight[item.period]
            + module.column.weight[item.column]
            + module.block.weight[1 + PERIODIC_BLOCKS.index(item.block)]
        )
        assert torch.equal(values[z], expected)
    # Any index shape, and distinct elements stay distinct.
    assert torch.equal(module(numbers.view(7, 17)), values.view(7, 17, 12))
    distances = torch.cdist(values[1:].detach(), values[1:].detach())
    assert float((distances + torch.eye(118)).min()) > 1.0e-3
    # A column difference is the same in every period pair: the sharing itself.
    na, k, mg, ca = (atomic_number(symbol) for symbol in ("Na", "K", "Mg", "Ca"))
    torch.testing.assert_close(
        values[na] - values[k], values[mg] - values[ca], rtol=0.0, atol=1.0e-12
    )
    assert set(module.state_dict()) == {"period.weight", "column.weight", "block.weight"}
    assert sum(parameter.numel() for parameter in module.parameters()) == (8 + 33 + 5) * 12


@pytest.mark.ml
def test_padding_stays_zero_and_initial_scale_matches_the_free_table() -> None:
    torch.manual_seed(11)
    module = PeriodicElementEmbedding(119, 2048)
    for table in (module.period, module.column, module.block):
        assert torch.equal(table.weight[0], torch.zeros(2048))
        assert abs(float(table.weight[1:].std()) - 3.0**-0.5) < 0.01
    values = module(torch.arange(1, 119))
    assert abs(float(values.var()) - 1.0) < 0.05
    loss = module(torch.tensor([[0, 26, 8, 0]])).square().sum()
    loss.backward()
    for table in (module.period, module.column, module.block):
        assert torch.equal(table.weight.grad[0], torch.zeros(2048))
    assert float(module.column.weight.grad[ALL["Fe"].column].abs().sum()) > 0.0
    assert float(module.column.weight.grad[ALL["Na"].column].abs().sum()) == 0.0
    for size in (1, 120):
        with pytest.raises(ValueError, match="num_embeddings"):
            PeriodicElementEmbedding(size, 4)
    with pytest.raises(ValueError, match="embedding_dim"):
        PeriodicElementEmbedding(119, 0)


@pytest.mark.ml
def test_the_legacy_table_is_unchanged() -> None:
    assert ELEMENT_EMBEDDINGS == ("learned-v1", "periodic-v1")
    torch.manual_seed(5)
    reference = torch.nn.Embedding(119, 6, padding_idx=0)
    following = torch.rand(3)
    torch.manual_seed(5)
    made = make_element_embedding("learned-v1", 119, 6)
    assert type(made) is torch.nn.Embedding and made.padding_idx == 0
    assert torch.equal(made.weight, reference.weight)
    # Same use of the random stream, so every later initializer is unchanged.
    assert torch.equal(torch.rand(3), following)
    with pytest.raises(ValueError, match="element embedding"):
        make_element_embedding("periodic-v2", 119, 6)

    torch.manual_seed(9)
    default = CompositionEncoder(16, element_dim=8, hidden_dim=24)
    torch.manual_seed(9)
    explicit = CompositionEncoder(
        16, element_dim=8, hidden_dim=24, element_embedding="learned-v1"
    )
    assert list(default.state_dict()) == list(explicit.state_dict())
    assert "element_embedding.weight" in default.state_dict()
    for key, value in default.state_dict().items():
        assert torch.equal(value, explicit.state_dict()[key])


@pytest.mark.ml
def test_encoders_with_the_periodic_embedding() -> None:
    torch.manual_seed(4)
    encoder = (
        CompositionEncoder(16, element_dim=8, hidden_dim=24, element_embedding="periodic-v1")
        .double()
        .eval()
    )
    assert isinstance(encoder.element_embedding, PeriodicElementEmbedding)
    numbers = torch.tensor([8, 14, 26, 0])
    counts = torch.tensor([4.0, 1.0, 2.0, 0.0], dtype=torch.float64)
    permutation = torch.tensor([2, 0, 3, 1])
    first = encoder(numbers, counts)
    torch.testing.assert_close(
        first, encoder(numbers[permutation], counts[permutation]), rtol=1.0e-12, atol=1.0e-12
    )
    # Padding is inert and a different element changes the code.
    torch.testing.assert_close(
        first, encoder(numbers[:3], counts[:3]), rtol=1.0e-12, atol=1.0e-12
    )
    assert not torch.allclose(first, encoder(torch.tensor([8, 14, 27, 0]), counts))

    actions = CountVectorEncoder(
        max_atomic_number=118, model_dim=12, hidden_dim=12, element_embedding="periodic-v1"
    ).double()
    assert isinstance(actions.element_embedding, PeriodicElementEmbedding)
    species = torch.tensor([[8, 26, 0]])
    table = torch.tensor([[[1.0, 2.0, 0.0], [0.0, 1.0, 0.0]]], dtype=torch.float64)
    encoded = actions(species, table)
    assert encoded.shape == (1, 2, 12) and bool(torch.isfinite(encoded).all())


@pytest.mark.ml
def test_model_uses_the_declared_embedding_everywhere() -> None:
    from test_learned import _tiny_config

    from crystalatlas.catalogue import MINIMAL_CATALOGUE
    from crystalatlas.learned import CrystalAtlasModel

    def sites(model: CrystalAtlasModel) -> list[object]:
        return [
            model.cover_selector.composition_encoder.element_embedding,
            model.cover_selector.action_encoder.element_embedding,
            model.geometry_field.species_embedding,
            model.geometry_field.flow_composition_encoder.element_embedding,
        ]

    config = _tiny_config()
    torch.manual_seed(2)
    legacy = CrystalAtlasModel(config, catalogue=MINIMAL_CATALOGUE)
    assert all(type(site) is torch.nn.Embedding for site in sites(legacy))
    periodic_config = dataclasses.replace(
        config, model=dataclasses.replace(config.model, element_embedding="periodic-v1")
    )
    torch.manual_seed(2)
    periodic = CrystalAtlasModel(periodic_config, catalogue=MINIMAL_CATALOGUE)
    assert all(isinstance(site, PeriodicElementEmbedding) for site in sites(periodic))
    legacy_keys, periodic_keys = set(legacy.state_dict()), set(periodic.state_dict())
    removed = {
        "cover_selector.composition_encoder.element_embedding.weight",
        "cover_selector.action_encoder.element_embedding.weight",
        "geometry_field.species_embedding.weight",
        "geometry_field.flow_composition_encoder.element_embedding.weight",
    }
    assert legacy_keys - periodic_keys == removed
    added = {
        key[: -len("weight")] + f"{table}.weight"
        for key in removed
        for table in ("period", "column", "block")
    }
    assert periodic_keys - legacy_keys == added
    assert legacy.scientific_config.fingerprint != periodic.scientific_config.fingerprint
