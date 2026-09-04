"""The leftover probe must retain terminal diagnostics in its only report."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from types import SimpleNamespace
import unittest

from retro_harness import TaskStatus

from harvest.scripts.d2_leftover_probe import _terminal_payload


class _Outcome(Enum):
    WORK_REMAINING = "work_remaining"


@dataclass(frozen=True)
class _Stamina:
    current: int
    maximum: int


@dataclass(frozen=True)
class _FarmStatus:
    outcome: _Outcome
    stamina: _Stamina
    stumps: int
    stumps_by_chunk: tuple[int, ...]
    hands_clear: bool
    reason: str


class D2LeftoverProbeReportTests(unittest.TestCase):
    def test_terminal_payload_retains_failure_reason_and_full_farm_observation(self) -> None:
        result = SimpleNamespace(
            status=TaskStatus.FAILURE,
            reason="motion stalled while clearing se stumps",
        )
        farm = _FarmStatus(
            outcome=_Outcome.WORK_REMAINING,
            stamina=_Stamina(current=80, maximum=100),
            stumps=5,
            stumps_by_chunk=(0, 0, 0, 5),
            hands_clear=True,
            reason="work_remaining",
        )

        payload = _terminal_payload(result, farm)

        self.assertEqual(payload["terminal_status"], "failure")
        self.assertEqual(payload["terminal_reason"], result.reason)
        self.assertEqual(
            payload["farm_status"],
            {
                "outcome": "work_remaining",
                "stamina": {"current": 80, "maximum": 100},
                "stumps": 5,
                "stumps_by_chunk": [0, 0, 0, 5],
                "hands_clear": True,
                "reason": "work_remaining",
            },
        )

    def test_terminal_payload_labels_missing_task_result(self) -> None:
        self.assertEqual(
            _terminal_payload(None, None),
            {"terminal_status": "none", "terminal_reason": "", "farm_status": {}},
        )


if __name__ == "__main__":
    unittest.main()
