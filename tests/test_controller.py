from __future__ import annotations

import unittest

import numpy as np

from crystalatlas.controller import (
    CandidateScore,
    ControllerConfig,
    greedy_select,
)
from crystalatlas.errors import DomainError


class ControllerTests(unittest.TestCase):
    def test_empty_candidate_pool_preserves_missing_slots(self) -> None:
        for requested in (0, 3):
            with self.subTest(requested=requested):
                self.assertEqual(greedy_select((), np.empty((0, 0)), requested), ())
        with self.assertRaises(DomainError):
            greedy_select((), np.empty((1, 1)), 3)
        for invalid in (True, -1, 0.5):
            with self.subTest(invalid=invalid), self.assertRaises(DomainError):
                greedy_select((), np.empty((0, 0)), invalid)  # type: ignore[arg-type]

    def test_low_hull_candidate_is_selected_without_other_terms(self) -> None:
        candidates = (
            CandidateScore("a", 0.2, 0.0, 0.0),
            CandidateScore("b", -0.1, 0.0, 0.0),
        )
        selected = greedy_select(candidates, np.eye(2), 1)
        self.assertEqual(selected[0].identifier, "b")

    def test_diversity_avoids_a_duplicate(self) -> None:
        candidates = (
            CandidateScore("a", 0.0, 0.0, 0.0),
            CandidateScore("b", 0.0, 0.0, 0.0),
            CandidateScore("c", 0.0, 0.0, 0.0),
        )
        kernel = np.asarray(
            [[1.0, 0.999, 0.0], [0.999, 1.0, 0.0], [0.0, 0.0, 1.0]]
        )
        selected = greedy_select(
            candidates,
            kernel,
            2,
            config=ControllerConfig(diversity_weight=1.0),
        )
        self.assertEqual({candidate.identifier for candidate in selected}, {"a", "c"})

    def test_non_psd_kernel_fails_closed(self) -> None:
        candidates = (
            CandidateScore("a", 0.0, 0.0, 0.0),
            CandidateScore("b", 0.0, 0.0, 0.0),
        )
        with self.assertRaises(DomainError):
            greedy_select(candidates, [[1.0, 2.0], [2.0, 1.0]], 1)


if __name__ == "__main__":
    unittest.main()
