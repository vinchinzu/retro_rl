"""Unit tests for stitchless 8-3 skills + rich handoff fingerprints (no emu)."""

from __future__ import annotations

import numpy as np

from smb.ram import (
    ADDR_ENEMY_FLAG,
    ADDR_ENEMY_STATE,
    ADDR_ENEMY_TYPE,
    ADDR_ENEMY_X,
    ADDR_ENEMY_X_PAGE,
    ADDR_ENEMY_Y,
    ADDR_FRAME_COUNTER,
    ADDR_PLAYER_X_FRAC,
    ADDR_PLAYER_Y_FRAC,
    ADDR_X_SPEED,
    ENEMY_TYPE_HAMMER_BRO,
    read_enemy_slots,
    read_snapshot,
    rich_handoff_fingerprint,
)
from smb.reactive_route import snapshot_fingerprint
from smb.tas.skills_8_3 import (
    FLAGPOLE_STYLES,
    RUN,
    RUN_JUMP,
    fpg_fireworks_hold,
    flagpole_macro,
    hop_pattern,
    open_skill_catalog,
    score_trial,
)
from smb.tas.slice import is_8_3_control, is_8_4_control


def test_rich_handoff_fingerprint_fields() -> None:
    ram = np.zeros(0x800, dtype=np.uint8)
    ram[0x006D] = 1  # x page
    ram[0x0086] = 40  # x off → 296
    ram[0x00CE] = 176
    ram[ADDR_X_SPEED] = 40
    ram[ADDR_PLAYER_X_FRAC] = 12
    ram[ADDR_PLAYER_Y_FRAC] = 34
    ram[ADDR_FRAME_COUNTER] = 163
    ram[0x075F] = 7
    ram[0x0760] = 2
    ram[0x0770] = 1
    ram[0x000E] = 0x08
    ram[0x07F8] = 3
    ram[0x07F9] = 0
    ram[0x07FA] = 1  # timer 301
    # one hammer bro
    ram[ADDR_ENEMY_FLAG + 0] = 1
    ram[ADDR_ENEMY_TYPE + 0] = ENEMY_TYPE_HAMMER_BRO
    ram[ADDR_ENEMY_STATE + 0] = 2
    ram[ADDR_ENEMY_X_PAGE + 0] = 1
    ram[ADDR_ENEMY_X + 0] = 10
    ram[ADDR_ENEMY_Y + 0] = 160

    snap = read_snapshot(ram)
    fp = rich_handoff_fingerprint(ram, snap=snap)
    assert fp["player_x"] == 296
    assert fp["x_frac"] == 12
    assert fp["y_frac"] == 34
    assert fp["x_speed"] == 40
    assert fp["frame_counter"] == 163
    assert fp["timer"] == 301
    assert fp["timer_mod21"] == 301 % 21
    assert fp["n_hammer_bro"] == 1
    assert ENEMY_TYPE_HAMMER_BRO in fp["enemy_types"]
    enemies = read_enemy_slots(ram)
    assert len(enemies) == 1
    assert enemies[0]["state"] == 2


def test_snapshot_fingerprint_includes_grounded_and_mod21() -> None:
    ram = np.zeros(0x800, dtype=np.uint8)
    ram[0x000E] = 0x08
    ram[0x009F] = 0  # y speed 0 → grounded candidate
    ram[0x07F8] = 3
    ram[0x07F9] = 0
    ram[0x07FA] = 1
    snap = read_snapshot(ram)
    fp = snapshot_fingerprint(snap)
    assert "grounded" in fp
    assert fp["timer_mod21"] == 301 % 21
    assert "screen_x" in fp


def test_flagpole_macro_styles_short() -> None:
    for style in FLAGPOLE_STYLES:
        body = flagpole_macro(style=style)
        assert len(body) >= 40
        assert all(len(f) == 9 for f in body)
        # uses RIGHT and usually B
        assert any(f[7] == 1 for f in body)
    hop = hop_pattern(run0=2, jhold=10, gap=4, hops=2, run_tail=5)
    assert len(hop) == 2 + 10 + 4 + 10 + 4 + 5


def test_fpg_fireworks_hold_has_a_edge() -> None:
    body = fpg_fireworks_hold(b_hold=20, jump_at=5, jump_hold=8, total=30)
    assert len(body) == 30
    assert body[5][8] == 1  # A on at jump_at
    assert body[12][8] == 1  # still in [jump_at, jump_at+jump_hold)
    assert body[13][8] == 0  # after jump window
    assert body[0][0] == 1  # B
    assert body[25][0] == 0  # B released


def test_score_trial_prefers_leave_over_raw_max_x() -> None:
    leave = score_trial({"leave": 2000, "max_x": 3000, "death": None, "timer_mod21": 0})
    noleave = score_trial({"leave": None, "max_x": 3500, "death": 1800, "timer_mod21": 0})
    assert leave > noleave
    # closer timer_mod21 better among leaves
    a = score_trial({"leave": 2100, "max_x": 3400, "death": None, "timer_mod21": 0})
    b = score_trial({"leave": 2100, "max_x": 3400, "death": None, "timer_mod21": 10})
    assert a > b


def test_skill_catalog_names() -> None:
    cat = open_skill_catalog()
    assert "hammer_bro_absorber" in cat
    assert "flagpole_macro" in cat
    assert "fpg_fireworks_hold" in cat
    assert set(cat["flagpole_macro"]["styles"]) == set(FLAGPOLE_STYLES)


def test_8_3_8_4_control_gates() -> None:
    class _S:
        def __init__(self, **kw: int) -> None:
            self.world = kw.get("world", 7)
            self.level = kw.get("level", 2)
            self.oper_mode = kw.get("oper_mode", 1)
            self.player_state = kw.get("player_state", 7)
            self.dying = bool(kw.get("dying", 0))
            self.timer = kw.get("timer", 301)
            self.player_x = kw.get("player_x", 40)

    assert is_8_3_control(_S())
    assert not is_8_3_control(_S(level=3))
    assert is_8_4_control(_S(level=3, player_x=40))
    assert not is_8_4_control(_S(level=3, player_x=500))


def test_run_jump_button_layout() -> None:
    """Preserve L+R semantics: RUN is B+RIGHT, RUN_JUMP adds A — never strip LR."""
    assert RUN[0] == 1 and RUN[7] == 1 and RUN[8] == 0
    assert RUN_JUMP[0] == 1 and RUN_JUMP[7] == 1 and RUN_JUMP[8] == 1


def test_stitchless_skills_leave_metadata() -> None:
    """Exported leave seed is control-relative, not a natural_82 mid-splice."""
    from smb.paths import MODELS_DIR
    from smb.policy import expand_nes9_rle, load_nes9_rle_seed
    from smb.tas.slice import HL_8_3_LEAVE_FRAMES, HL_8_3_SKILLS_LEAVE

    path = HL_8_3_SKILLS_LEAVE if HL_8_3_SKILLS_LEAVE.exists() else (
        MODELS_DIR / "smb_8_3_stitchless_skills_leave.json"
    )
    if not path.exists():
        return  # optional artifact until first leave export
    data = load_nes9_rle_seed(path)
    assert data.get("stitchless") is True
    assert data.get("no_natural_82_splice") is True
    assert data.get("verified_leave_8_4") is True
    assert data.get("preserve_lr") is True
    assert data.get("leave_frames") == HL_8_3_LEAVE_FRAMES == 2374
    frames = expand_nes9_rle(data)
    assert len(frames) == 2374
    # Must not claim natural_82 as primary body source
    src = str(data.get("source") or "")
    assert "natural_82" not in src or "no natural" in src.lower()
    # Trials recorded
    trials = data.get("verify_trials") or []
    assert len(trials) >= 2
    assert all(t.get("ok") for t in trials)
