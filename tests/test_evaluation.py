from __future__ import annotations

import unittest
from dataclasses import FrozenInstanceError

from crystalatlas.errors import DomainError
from crystalatlas.evaluation import (
    AttemptStatus,
    CostVector,
    PhaseReference,
    ProtocolAction,
    ProtocolTrace,
    energy_above_hull,
)


class EvaluationTests(unittest.TestCase):
    def test_restored_trace_counts_all_initial_actions_and_freezes_history(self) -> None:
        initial = [ProtocolAction("completed", CostVector(dft_submissions=1))]
        trace = ProtocolTrace(CostVector(dft_submissions=1), actions=initial)
        self.assertEqual(trace.cost.dft_submissions, 1)
        initial.clear()
        self.assertEqual(len(trace.actions), 1)
        extra = ProtocolAction("extra", CostVector(dft_submissions=1))
        self.assertFalse(trace.can_append(extra))
        with self.assertRaises(DomainError):
            trace.append(extra)
        self.assertEqual(
            trace.actions, (ProtocolAction("completed", CostVector(dft_submissions=1)),)
        )
        self.assertIsInstance(trace.actions, tuple)
        with self.assertRaises(FrozenInstanceError):
            trace.budget = CostVector(dft_submissions=10)  # type: ignore[misc]
        with self.assertRaises((FrozenInstanceError, TypeError, AttributeError)):
            trace.actions = ()  # type: ignore[misc]

    def test_initial_history_cannot_exceed_any_budget_component(self) -> None:
        with self.assertRaises(DomainError):
            ProtocolTrace(
                CostVector(dft_submissions=1),
                actions=[ProtocolAction("completed", CostVector(dft_submissions=2))],
            )
        trace = ProtocolTrace(
            CostVector(cpu_hours=1.0),
            actions=[ProtocolAction("completed", CostVector(cpu_hours=0.4))],
        )
        snapshot = trace.actions
        trace.append(ProtocolAction("next", CostVector(cpu_hours=0.5)))
        self.assertAlmostEqual(trace.cost.cpu_hours, 0.9)
        self.assertEqual(len(snapshot), 1)
        self.assertEqual(len(trace.actions), 2)

    def test_discrete_resource_counts_are_finite_nonnegative_integers(self) -> None:
        for invalid in (True, 0.5, float("nan"), float("inf"), -1):
            with self.subTest(invalid=invalid), self.assertRaises(DomainError):
                CostVector(dft_submissions=invalid)  # type: ignore[arg-type]
        with self.assertRaises(DomainError):
            CostVector().fits_within(CostVector(), atol=float("inf"))

    def test_vector_budget_cannot_be_scalar_traded(self) -> None:
        trace = ProtocolTrace(CostVector(gpu_hours=1.0, dft_submissions=1))
        trace.append(
            ProtocolAction(
                "generate", CostVector(gpu_hours=0.5), status=AttemptStatus.GENERATED
            )
        )
        with self.assertRaises(DomainError):
            trace.append(ProtocolAction("dft", CostVector(dft_submissions=2)))
        self.assertEqual(len(trace.actions), 1)

    def test_failed_attempts_remain_in_status_counts(self) -> None:
        trace = ProtocolTrace(CostVector(cpu_hours=2.0))
        trace.append(
            ProtocolAction(
                "decode", CostVector(cpu_hours=0.1), status=AttemptStatus.INVALID
            )
        )
        self.assertEqual(trace.status_counts()[AttemptStatus.INVALID], 1)

    def test_binary_hull(self) -> None:
        references = [
            PhaseReference("A", (1.0, 0.0), 0.0),
            PhaseReference("B", (0.0, 1.0), 0.0),
            PhaseReference("AB", (0.5, 0.5), -1.0),
        ]
        self.assertAlmostEqual(energy_above_hull((0.5, 0.5), -0.9, references), 0.1)
        self.assertAlmostEqual(energy_above_hull((0.25, 0.75), -0.4, references), 0.1)


if __name__ == "__main__":
    unittest.main()
