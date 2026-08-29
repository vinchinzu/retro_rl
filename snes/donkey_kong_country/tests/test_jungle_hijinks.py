"""No-ROM gates for isolated Jungle Hijinks (real 0x72 exit, not bonus)."""

from __future__ import annotations

import json
from pathlib import Path

from retro_harness.platformer.level_config import get_level_config

import donkey_kong_country.platformer_levels  # noqa: F401

GAME_DIR = Path(__file__).resolve().parents[1]
EVIDENCE = GAME_DIR / "recordings" / "jungle_hijinks_isolated_clear.json"
ACTIONS = GAME_DIR / "recordings" / "jungle_hijinks_isolated_actions.json"
BONUS_IDS = {"0x06", "0x25"}
REAL_EXIT = "0x72"
NEXT_LEVEL = "0x0C"


def test_jungle_hijinks_config_rejects_bonus_rooms() -> None:
    cfg = get_level_config("jungle_hijinks")
    assert cfg.start_state == "JungleHijinks"
    assert cfg.completion_signal == "level_id_change"
    assert cfg.completion_level_ids == [0x72]
    assert set(cfg.completion_exclude_ids) >= {0x06, 0x25}
    assert cfg.completion_min_progress >= 4000.0
    assert get_level_config("dkc_jungle") is cfg


def test_jungle_hijinks_isolated_evidence_is_real_exit() -> None:
    data = json.loads(EVIDENCE.read_text(encoding="utf-8"))
    assert data["start_state"] == "JungleHijinks"
    assert data["natural_entry"] is False
    assert data["completed"] is True
    assert data["died"] is False
    assert data["real_level_complete"] is True
    assert data["false_bonus_green"] is False
    assert data["level_id_leftover"] == REAL_EXIT
    assert data["level_id_after_full_sequence"] == NEXT_LEVEL
    assert data["bonus_ids_visited"] == []
    assert [t["to"] for t in data["level_id_transitions"]] == [REAL_EXIT, NEXT_LEVEL]
    assert not any(t["to"] in BONUS_IDS for t in data["level_id_transitions"])
    assert data["frames"] == 1682
    assert data["max_progress"] >= 5200
    assert data["trial_count"] == 3
    assert data["success_rate"] == 1.0
    assert data["start"]["level_id_hex"] == "0x16"
    tape = json.loads(ACTIONS.read_text(encoding="utf-8"))
    assert tape["start_state"] == "JungleHijinks"
    assert tape["num_frames"] == len(tape["actions"]) == 1781
