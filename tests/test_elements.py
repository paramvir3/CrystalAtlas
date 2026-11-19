from __future__ import annotations

import unittest

from crystalatlas.elements import (
    atomic_number,
    canonical_species_set,
    element_symbol,
    species_label,
)
from crystalatlas.errors import DomainError


class ElementTests(unittest.TestCase):
    def test_symbol_roundtrip(self) -> None:
        self.assertEqual(atomic_number("fe"), 26)
        self.assertEqual(element_symbol(26), "Fe")

    def test_species_are_canonicalized_as_a_set(self) -> None:
        species = canonical_species_set(["O", "Li", "P", "Fe"])
        self.assertEqual(species, (3, 8, 15, 26))
        self.assertEqual(species_label(species), "Li-O-P-Fe")

    def test_duplicate_species_fail(self) -> None:
        with self.assertRaises(DomainError):
            canonical_species_set(["O", 8])


if __name__ == "__main__":
    unittest.main()
