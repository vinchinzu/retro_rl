"""ROM-free tests for Endurance 1 capture helpers."""

from __future__ import annotations

import numpy as np

from unittest.mock import patch

from mortal_kombat.ram import ADDR_MATCH_COUNTER, make_test_ram
from mortal_kombat.roster import KIND_RAM_V3, KIND_SCRIPT, StageSlot
from mortal_kombat.scripted import B, DOWN, LEFT, RIGHT, UP, X, Y
from mortal_kombat.scripts.capture_natural_endurance1 import (
    ADDR_KNIFE_X,
    CourtyardKanoPolicy,
    NoJumpFireballPolicy,
    RelabelMatchPolicy,
    RoundMixPolicy,
    apply_oracle,
    format_rle,
    knife_incoming,
    make_policy_loader,
    mask_from_buttons,
    rle_encode,
)


def test_rle_encode_and_format() -> None:
    encoded = rle_encode([0, 0, 8, 8, 8, 144])
    assert encoded == [(0, 2), (8, 3), (144, 1)]
    text = format_rle(encoded)
    assert text.startswith("NATURAL_ENDURANCE1_RLE:")
    assert "(0, 2)" in text and "(8, 3)" in text


def test_mask_from_buttons_twelve_snes_bits() -> None:
    buttons = np.zeros(12, dtype=np.int8)
    buttons[0] = 1
    buttons[7] = 1
    buttons[11] = 1
    assert mask_from_buttons(buttons) == (1 << 0) | (1 << 7) | (1 << 11)


def test_relabel_match_pokes_copy_only() -> None:
    class Inner:
        kind = KIND_RAM_V3
        seen = -1

        def reset(self) -> None:
            return None

        def act(self, ram, rgb, *, deterministic: bool = False):
            del rgb, deterministic
            self.seen = int(ram[ADDR_MATCH_COUNTER])
            return np.zeros(12, dtype=np.int8)

    inner = Inner()
    policy = RelabelMatchPolicy(inner, 4)
    ram = make_test_ram(match_counter=7)
    policy.act(ram, None, deterministic=True)
    assert inner.seen == 4
    assert int(ram[ADDR_MATCH_COUNTER]) == 7


def test_apply_oracle_forces_every_slot_and_drops_backups() -> None:
    class Runner:
        slots = [
            StageSlot("Endurance1", "E1", 7, "old.zip", KIND_RAM_V3, backups=["pix.zip"]),
            StageSlot("Endurance1B", "E1B", 8, "old.zip", KIND_RAM_V3, backups=["pix.zip"]),
        ]

    runner = Runner()
    apply_oracle(runner, ladder_model="mk1_v3_Match5_ppo_final.zip", pixel_model=None)
    assert {slot.model for slot in runner.slots} == {"mk1_v3_Match5_ppo_final.zip"}
    assert all(slot.kind == KIND_RAM_V3 and slot.backups == [] for slot in runner.slots)
    assert {slot.match_id for slot in runner.slots} == {7, 8}


def _kano(intro: int = 0) -> NoJumpFireballPolicy:
    return NoJumpFireballPolicy(intro_frames=intro)


def _knife_ram(*, p1_x: int, p2_x: int, knife_x: int, **fields):
    ram = make_test_ram(p1_x=p1_x, p2_x=p2_x, **fields)
    ram[ADDR_KNIFE_X] = knife_x
    return ram


def test_knife_incoming_when_sprite_leaves_kano() -> None:
    ram = _knife_ram(p1_x=68, p2_x=180, knife_x=139)
    assert knife_incoming(ram, 68, 180)
    attached = _knife_ram(p1_x=68, p2_x=180, knife_x=180)
    assert not knife_incoming(attached, 68, 180)


def test_knife_incoming_ignores_stale_sprite_when_kano_walks() -> None:
    ram = _knife_ram(p1_x=68, p2_x=167, knife_x=180)
    assert not knife_incoming(ram, 68, 167)


def test_kano_keepaway_ducks_knives_at_range() -> None:
    policy = _kano()
    frame = policy.act(_knife_ram(p1_x=40, p2_x=140, knife_x=90), None)
    assert frame[DOWN] == 1
    assert frame[X] == 0
    assert frame[Y] == 0


def test_kano_keepaway_blocks_close() -> None:
    policy = _kano()
    frame = policy.act(make_test_ram(p1_x=100, p2_x=130, p2_state=1), None)
    assert frame[X] == 1
    assert frame[DOWN] == 1
    assert frame[Y] == 0


def test_kano_keepaway_fireball_only_from_far() -> None:
    policy = _kano()
    start = policy.act(make_test_ram(p1_x=40, p2_x=200), None)
    assert start[RIGHT] == 1
    assert start[Y] == 0
    policy.reset()
    close = policy.act(make_test_ram(p1_x=80, p2_x=140), None)
    assert close[LEFT] == 1
    assert close[Y] == 0
    assert close[RIGHT] == 0


def test_kano_keepaway_holds_duck_after_knife_sprite_reattaches() -> None:
    policy = _kano()
    first = policy.act(_knife_ram(p1_x=40, p2_x=140, knife_x=90), None)
    assert first[DOWN] == 1
    held = policy.act(_knife_ram(p1_x=40, p2_x=140, knife_x=140), None)
    assert held[DOWN] == 1
    assert held[Y] == 0
    assert held[RIGHT] == 0


def test_kano_keepaway_walks_back_on_cooldown_not_idle() -> None:
    policy = _kano()
    far = make_test_ram(p1_x=40, p2_x=200)
    for _ in range(15):
        policy.act(far, None)
    cooldown = policy.act(far, None)
    assert cooldown[LEFT] == 1
    assert int(cooldown.sum()) == 1


def test_kano_keepaway_cancels_fireball_into_duck() -> None:
    policy = _kano()
    start = policy.act(make_test_ram(p1_x=40, p2_x=200), None)
    assert start[RIGHT] == 1
    ducked = policy.act(_knife_ram(p1_x=40, p2_x=200, knife_x=120), None)
    assert ducked[DOWN] == 1
    assert ducked[RIGHT] == 0
    assert ducked[Y] == 0


def test_round_mix_switches_after_ko_refill() -> None:
    class Inner:
        kind = KIND_RAM_V3

        def __init__(self, label: str):
            self.name = label
            self.acts = 0
            self.resets = 0

        def reset(self) -> None:
            self.resets += 1

        def act(self, ram, rgb, *, deterministic: bool = False):
            del ram, rgb, deterministic
            self.acts += 1
            out = np.zeros(12, dtype=np.int8)
            out[0] = 1 if self.name == "first" else 0
            out[1] = 1 if self.name == "rest" else 0
            return out

    first = Inner("first")
    rest = Inner("rest")
    mix = RoundMixPolicy(first, rest)
    leftover = mix.act(
        make_test_ram(
            p1_health=59,
            p2_health=0,
            p2_character=3,
            match_counter=7,
            timer=102,
            p1_rounds=2,
        ),
        None,
    )
    assert leftover[0] == 1 and leftover[1] == 0
    opening = mix.act(make_test_ram(p1_health=161, p2_health=80), None)
    assert opening[0] == 1 and opening[1] == 0
    mix.act(make_test_ram(p1_health=80, p2_health=0), None)
    refill = mix.act(make_test_ram(p1_health=161, p2_health=161), None)
    assert refill[0] == 0 and refill[1] == 1
    assert rest.resets == 1
    mix.reset()
    after_fight_reset = mix.act(make_test_ram(p1_health=161, p2_health=161), None)
    assert after_fight_reset[1] == 1
    damaged = mix.act(make_test_ram(p1_health=120, p2_health=160), None)
    assert damaged[1] == 1


def test_round2_kano_loader_wraps_ram_oracle() -> None:
    class Dummy:
        kind = KIND_RAM_V3
        name = "dummy"

        def reset(self) -> None:
            return None

        def act(self, ram, rgb, *, deterministic: bool = False):
            del ram, rgb, deterministic
            return np.zeros(12, dtype=np.int8)

    loader = make_policy_loader(None, round2_kano=True)
    with patch("mortal_kombat.policy.load_policy", return_value=Dummy()):
        wrapped = loader("mk1_v3_Match5_ppo_final.zip", KIND_RAM_V3)
    assert isinstance(wrapped, RoundMixPolicy)
    assert isinstance(wrapped.rest, NoJumpFireballPolicy)
    assert wrapped.rest.intro_frames == 0
    assert wrapped.kind == KIND_RAM_V3
    assert wrapped.first.name == "dummy"


def test_courtyard_loader_returns_jump_specialist() -> None:
    loader = make_policy_loader(None, courtyard=True)
    policy = loader("ignored.zip", KIND_SCRIPT)
    assert isinstance(policy, CourtyardKanoPolicy)
    assert policy.name == "scripted-courtyard"
    assert policy.jump_at == 296


def test_courtyard_idles_until_jump_then_air_hk() -> None:
    policy = CourtyardKanoPolicy(round1_jump_at=5, later_jump_at=5)
    idle = make_test_ram(p1_x=68, p2_x=180, p1_y=144)
    for _ in range(4):
        frame = policy.act(idle, None)
        assert int(frame.sum()) == 0
        assert frame[UP] == 0
    jump = policy.act(idle, None)
    assert jump[UP] == 1
    assert jump[RIGHT] == 1
    for _ in range(9):
        held = policy.act(idle, None)
        assert held[UP] == 1
    # Still grounded: startup wait, not a miss. Do not HK yet.
    wait = policy.act(idle, None)
    assert wait[UP] == 0
    assert wait[B] == 0
    air = make_test_ram(p1_x=90, p2_x=180, p1_y=70)
    kick = policy.act(air, None)
    assert kick[B] == 1
    assert kick[UP] == 0
    assert kick[DOWN] == 0
    for _ in range(5):
        held = policy.act(air, None)
        assert held[B] == 1
    almost = policy.act(make_test_ram(p1_x=150, p2_x=180, p1_y=143), None)
    assert almost[B] == 1
    assert almost[RIGHT] == 0
    # Land far: opener done, do not land-HK (that crosses).
    land = policy.act(make_test_ram(p1_x=151, p2_x=202, p1_y=144), None)
    assert land[B] == 0
    assert land[UP] == 0


def test_courtyard_air_hk_ignores_knife_sprite() -> None:
    policy = CourtyardKanoPolicy(round1_jump_at=2, later_jump_at=2)
    idle = make_test_ram(p1_x=68, p2_x=180, p1_y=144)
    policy.act(idle, None)
    policy.act(idle, None)
    for _ in range(10):
        policy.act(idle, None)
    air = _knife_ram(p1_x=90, p2_x=180, knife_x=139, p1_y=70)
    kick = policy.act(air, None)
    assert kick[B] == 1
    assert kick[DOWN] == 0


def test_courtyard_standing_hk_when_kano_walks_in() -> None:
    policy = CourtyardKanoPolicy(round1_jump_at=1, later_jump_at=1)
    idle = make_test_ram(p1_x=68, p2_x=180, p1_y=144)
    policy.act(idle, None)
    for _ in range(10):
        policy.act(idle, None)
    policy.act(make_test_ram(p1_x=90, p2_x=180, p1_y=70), None)
    policy.act(make_test_ram(p1_x=151, p2_x=202, p1_y=144), None)
    close = policy.act(make_test_ram(p1_x=182, p2_x=214, p1_y=144), None)
    assert close[B] == 1
    assert close[UP] == 0


def test_courtyard_does_not_chase_kano_off_right_edge() -> None:
    policy = CourtyardKanoPolicy(round1_jump_at=1, later_jump_at=1)
    idle = make_test_ram(p1_x=68, p2_x=180, p1_y=144)
    policy.act(idle, None)
    for _ in range(10):
        policy.act(idle, None)
    policy.act(make_test_ram(p1_x=90, p2_x=180, p1_y=70), None)
    policy.act(make_test_ram(p1_x=151, p2_x=202, p1_y=144), None)
    close = make_test_ram(p1_x=182, p2_x=214, p1_y=144)
    policy.act(close, None)
    for _ in range(4):
        policy.act(close, None)
    chase = policy.act(make_test_ram(p1_x=170, p2_x=214, p1_y=144), None)
    assert chase[RIGHT] == 1
    assert chase[B] == 0
    rim = policy.act(make_test_ram(p1_x=192, p2_x=231, p1_y=144), None)
    assert rim[LEFT] == 1
    assert rim[RIGHT] == 0
    assert rim[B] == 0


def test_courtyard_later_round_uses_shorter_jump_clock() -> None:
    policy = CourtyardKanoPolicy(round1_jump_at=8, later_jump_at=3)
    ram = make_test_ram(p1_x=68, p2_x=180, p1_y=144)
    policy.act(ram, None)
    assert policy.jump_at == 8
    policy.reset()
    policy.act(ram, None)
    policy.act(ram, None)
    jump = policy.act(ram, None)
    assert policy.jump_at == 3
    assert jump[UP] == 1


def test_courtyard_idles_on_leftover_ko_hud() -> None:
    policy = CourtyardKanoPolicy(round1_jump_at=5)
    leftover = _knife_ram(
        p1_x=68,
        p2_x=180,
        knife_x=90,
        p1_health=59,
        p2_health=0,
        p1_rounds=2,
        p1_y=144,
    )
    frame = policy.act(leftover, None)
    assert int(frame.sum()) == 0


def test_courtyard_ducks_real_knife_after_opener() -> None:
    policy = CourtyardKanoPolicy(round1_jump_at=1, later_jump_at=1)
    idle = make_test_ram(p1_x=68, p2_x=180, p1_y=144)
    policy.act(idle, None)
    for _ in range(10):
        policy.act(idle, None)
    policy.act(make_test_ram(p1_x=90, p2_x=180, p1_y=70), None)
    policy.act(make_test_ram(p1_x=151, p2_x=202, p1_y=144), None)
    duck = policy.act(_knife_ram(p1_x=151, p2_x=202, knife_x=170, p1_y=144), None)
    assert duck[DOWN] == 1
    walk = policy.act(_knife_ram(p1_x=151, p2_x=167, knife_x=180, p1_y=144), None)
    assert walk[DOWN] == 0
    assert walk[UP] == 0
