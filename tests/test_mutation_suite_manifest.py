"""The theory 4.9 mutation manifest cannot silently rot (review m5).

The expensive kill run is ``scripts/run_mutation_suite.py``.  This fast test
checks that every entry is still applicable: numbering is exactly 1..24, each
mutation snippet occurs exactly once, every mutant compiles, every killing test
resolves to a real test function, and every gap carries its reason.
"""

from __future__ import annotations

import ast
from pathlib import Path

from mutation_suite import MUTATIONS

ROOT = Path(__file__).resolve().parents[1]
STATUSES = {"executable", "obligation-open", "blocked-by-defect"}


def _defined_tests(path: Path) -> set[str]:
    tree = ast.parse(path.read_text())
    names: set[str] = set()
    for node in tree.body:
        if isinstance(node, ast.FunctionDef):
            names.add(node.name)
        elif isinstance(node, ast.ClassDef):
            for item in node.body:
                if isinstance(item, ast.FunctionDef):
                    names.add(f"{node.name}::{item.name}")
    return names


def test_manifest_numbers_the_24_theory_mutations_exactly_once() -> None:
    assert [mutation.number for mutation in MUTATIONS] == list(range(1, 25))
    assert all(mutation.status in STATUSES for mutation in MUTATIONS)
    assert all(mutation.defect and mutation.obligations for mutation in MUTATIONS)


def test_executable_mutations_apply_once_compile_and_name_real_killers() -> None:
    for mutation in MUTATIONS:
        if mutation.status != "executable":
            assert mutation.note, mutation.number
            assert not mutation.find and not mutation.killers, mutation.number
            continue
        target = ROOT / mutation.path
        source = target.read_text()
        assert source.count(mutation.find) == 1, mutation.number
        assert mutation.replace != mutation.find, mutation.number
        compile(source.replace(mutation.find, mutation.replace), mutation.path, "exec")
        assert mutation.killers, mutation.number
        for killer in mutation.killers:
            file_part, _, test_part = killer.partition("::")
            test_file = ROOT / file_part
            assert test_file.is_file(), killer
            assert test_part in _defined_tests(test_file), killer
