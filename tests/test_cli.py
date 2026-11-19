from __future__ import annotations

import json
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
from pathlib import Path

from crystalatlas.cli import main


class CommandLineTests(unittest.TestCase):
    def _run(self, *arguments: str) -> tuple[int, str, str]:
        stdout = StringIO()
        stderr = StringIO()
        with redirect_stdout(stdout), redirect_stderr(stderr):
            code = main(arguments)
        return code, stdout.getvalue(), stderr.getvalue()

    def test_status_is_honest_about_untrained_and_catalogue_scope(self) -> None:
        code, stdout, stderr = self._run("status")
        self.assertEqual(code, 0, stderr)
        payload = json.loads(stdout)
        self.assertFalse(payload["trained_checkpoint_bundled"])
        self.assertFalse(payload["predictive_superiority_established"])
        self.assertEqual(
            payload["reference_catalogue"]["space_group_keys"], ["P1", "P-1"]
        )
        self.assertFalse(payload["reference_catalogue"]["complete_230_space_groups"])
        self.assertTrue(payload["production_catalogue"]["available"])
        self.assertEqual(payload["production_catalogue"]["space_group_types"], 230)

    def test_formula_parser_reports_reduced_canonical_counts(self) -> None:
        code, stdout, stderr = self._run("formula", "Si2O4")
        self.assertEqual(code, 0, stderr)
        payload = json.loads(stdout)
        self.assertEqual(payload["canonical_reduced_formula"], "O2Si")

    def test_reference_generation_writes_auditable_untrained_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "generated"
            code, stdout, stderr = self._run(
                "reference-generate",
                "--mode",
                "csp",
                "--formula",
                "CO2",
                "--samples",
                "1",
                "--seed",
                "42",
                "--format",
                "both",
                "--output",
                str(output),
            )
            self.assertEqual(code, 0, stderr)
            summary = json.loads(stdout)
            self.assertEqual(summary["successful_samples"], 1)
            self.assertFalse(summary["physical_prediction"])
            self.assertTrue((output / "sample_000000.cif").is_file())
            sample = json.loads((output / "sample_000000.json").read_text())
            self.assertEqual(sample["law"], "untrained-uniform-reference")
            self.assertFalse(sample["physical_prediction"])
            manifest = json.loads((output / "manifest.json").read_text())
            self.assertEqual(manifest["root_seed"], 42)
            self.assertEqual(manifest["conditioning"]["reduced_formula"], "CO2")
            self.assertEqual(len(manifest["config_fingerprint"]), 64)
            self.assertEqual(len(manifest["catalogue"]["fingerprint"]), 64)
            self.assertIn("sample_000000.cif", manifest["output_sha256"])
            cif = (output / "sample_000000.cif").read_text()
            self.assertIn("crystalatlas_law: untrained-uniform-reference", cif)
            self.assertIn("crystalatlas_physical_prediction: false", cif)
            self.assertIn("crystalatlas_provenance_root_seed: 42", cif)

    def test_mode_contract_errors_return_nonzero_without_a_traceback(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            code, _, stderr = self._run(
                "reference-generate",
                "--mode",
                "dng",
                "--formula",
                "SiO2",
                "--output",
                temporary,
            )
        self.assertEqual(code, 2)
        self.assertIn("accepts neither", stderr)


if __name__ == "__main__":
    unittest.main()
