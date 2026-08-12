"""Classify forage vs talk from RAM deltas and existing tape traces."""

from __future__ import annotations

from pathlib import Path
import json
import unittest

from harvest.core.interact import classify_interact, first_held_change
from harvest.core.npc_catalog import search_text_records, text_record_for_id
from harvest.paths import TASKS_DIR


class InteractClassifyTests(unittest.TestCase):
    def test_keep_menu_is_not_talk(self) -> None:
        self.assertEqual(
            classify_interact(
                held_before=0,
                held_after=0x03,
                lock_after=2,
                text_choices=("Eat", "Don't eat"),
            ),
            "forage_keep_menu",
        )

    def test_empty_hands_lock_near_npc_is_talk(self) -> None:
        self.assertEqual(
            classify_interact(
                held_before=0,
                held_after=0,
                lock_after=2,
                npc_in_face=True,
            ),
            "npc_talk",
        )

    def test_grape_tape_first_held_is_on_stand(self) -> None:
        path = Path(TASKS_DIR) / "mountain_grape_stand.json"
        if not path.exists():
            self.skipTest("mountain_grape_stand tape missing")
        data = json.loads(path.read_text(encoding="utf-8"))
        trace = data.get("trace") or []
        if not trace:
            self.skipTest("tape has no RAM trace")
        change = first_held_change(trace)
        self.assertIsNotNone(change)
        assert change is not None
        self.assertEqual(change["held_after"], 3)
        self.assertEqual((change.get("x"), change.get("y")), (326, 409))
        self.assertEqual(
            classify_interact(
                held_before=change["held_before"],
                held_after=change["held_after"],
                lock_after=int(change.get("input_lock") or 1),
                text_choices=("Don't eat",),
            ),
            "forage_keep_menu",
        )

    def test_decomp_grape_box_is_eat_dont_eat(self) -> None:
        hits = search_text_records("wild grape")
        self.assertTrue(hits)
        rec = next((h for h in hits if "Don't eat" in h.text or "Don't eat" in "".join(h.choices)), hits[0])
        self.assertTrue(rec.choices or "Eat" in rec.text)
        by_id = text_record_for_id(rec.text_id)
        self.assertIsNotNone(by_id)
        self.assertEqual(by_id.text_id, rec.text_id)


if __name__ == "__main__":
    unittest.main()
