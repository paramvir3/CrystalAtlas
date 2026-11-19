#!/usr/bin/env python3
"""Dependency-light exact-core test entry point.

Pytest-style and optional-PyTorch modules are intentionally omitted here.  The
complete development gate remains ``pytest -q`` with the ``dev,ml`` extras.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root))
    sys.path.insert(0, str(root / "src"))
    modules = (
        "tests.test_algebra",
        "tests.test_catalogue",
        "tests.test_cli",
        "tests.test_composition",
        "tests.test_config",
        "tests.test_constraints",
        "tests.test_controller",
        "tests.test_data",
        "tests.test_elements",
        "tests.test_evaluation",
        "tests.test_lattice",
        "tests.test_periodic",
        "tests.test_pipeline",
        "tests.test_reference",
        "tests.test_tokenization",
        "tests.test_wyckoff_decoder",
    )
    suite = unittest.defaultTestLoader.loadTestsFromNames(modules)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    raise SystemExit(main())
