from __future__ import annotations

import json
import unittest
from fractions import Fraction
from hashlib import sha256
from types import MappingProxyType

from crystalatlas.catalogue import (
    FIXTURE_EXPECTED_ROWS,
    MINIMAL_CATALOGUE,
    CertifiedCatalogue,
)
from crystalatlas.errors import CatalogueError


class CertifiedCatalogueTests(unittest.TestCase):
    def test_fixture_scope_and_complete_p_minus_one_rows(self) -> None:
        MINIMAL_CATALOGUE.validate()
        self.assertEqual(MINIMAL_CATALOGUE.supported_international_numbers, (1, 2))
        p_minus_one = MINIMAL_CATALOGUE.get("P-1")
        self.assertEqual(frozenset(p_minus_one.wyckoff_positions), FIXTURE_EXPECTED_ROWS["P-1"])
        orbit = p_minus_one.wyckoff("2i").expand_exact(
            (Fraction(1, 7), Fraction(2, 7), Fraction(3, 7))
        )
        self.assertEqual(
            orbit,
            (
                (Fraction(1, 7), Fraction(2, 7), Fraction(3, 7)),
                (Fraction(6, 7), Fraction(5, 7), Fraction(4, 7)),
            ),
        )

    def test_unknown_group_and_all_230_claim_fail_closed(self) -> None:
        with self.assertRaises(CatalogueError):
            MINIMAL_CATALOGUE.get("P2")
        with self.assertRaises(CatalogueError):
            MINIMAL_CATALOGUE.validate(require_all_230=True)

    def test_manifest_detects_a_deleted_setting(self) -> None:
        with self.assertRaises(CatalogueError):
            CertifiedCatalogue(
                {"P1": MINIMAL_CATALOGUE.get("P1")},
                FIXTURE_EXPECTED_ROWS,
                certification_scope="deliberately incomplete mutation",
            )

    def test_fingerprint_covers_exact_catalogue_and_is_stable(self) -> None:
        self.assertEqual(len(MINIMAL_CATALOGUE.fingerprint), 64)
        self.assertEqual(MINIMAL_CATALOGUE.fingerprint, MINIMAL_CATALOGUE.fingerprint)
        self.assertEqual(
            MINIMAL_CATALOGUE.identity["fingerprint"],
            MINIMAL_CATALOGUE.fingerprint,
        )
        changed_scope = CertifiedCatalogue(
            MINIMAL_CATALOGUE.settings,
            FIXTURE_EXPECTED_ROWS,
            certification_scope="same tables, intentionally different identity",
        )
        self.assertNotEqual(changed_scope.fingerprint, MINIMAL_CATALOGUE.fingerprint)

    def test_fingerprint_memo_equals_a_fresh_serialization_of_the_current_data(self) -> None:
        catalogue = CertifiedCatalogue(
            MINIMAL_CATALOGUE.settings,
            FIXTURE_EXPECTED_ROWS,
            certification_scope="fingerprint memo fixture",
        )

        def fresh() -> str:
            payload = json.dumps(
                catalogue.to_dict(), allow_nan=False, sort_keys=True, separators=(",", ":")
            )
            return sha256(payload.encode("utf-8")).hexdigest()

        first = catalogue.fingerprint
        self.assertEqual(first, fresh())
        self.assertEqual(catalogue.fingerprint, first)
        seen = {first}
        # Each change of a datum the payload reads must defeat the memo.
        catalogue.certification_scope = "fingerprint memo fixture, renamed"
        self.assertEqual(catalogue.fingerprint, fresh())
        seen.add(catalogue.fingerprint)
        catalogue.complete_space_group_types = True
        self.assertEqual(catalogue.fingerprint, fresh())
        seen.add(catalogue.fingerprint)
        catalogue._settings = MappingProxyType({"P1": MINIMAL_CATALOGUE.get("P1")})
        self.assertEqual(catalogue.fingerprint, fresh())
        seen.add(catalogue.fingerprint)
        catalogue._expected_rows = MappingProxyType(
            {"P1": frozenset(FIXTURE_EXPECTED_ROWS["P1"]) | {"memo-sentinel-row"}}
        )
        self.assertEqual(catalogue.fingerprint, fresh())
        seen.add(catalogue.fingerprint)
        self.assertEqual(len(seen), 5)


if __name__ == "__main__":
    unittest.main()
