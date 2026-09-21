"""One enemy model: KIND_POLICY names the type, ObjectTracker owns the motion.

Card rr-ttyu.4 (C4): "re-use the same enemy code every time we encounter one".
Two halves, and the file is split into them:

* **behavioural** — what a controller now reads for a given RAM frame. These
  are the ones that go red if a defect comes back.
* **structural** — the ratchet that keeps a module from quietly growing its own
  copy of an enemy sensor again. Per project memory this suite detects renames
  far better than behaviour, so the structural tests are deliberately few and
  each carries an allowlist that has to be argued down, not edited up.
"""

from __future__ import annotations

import pathlib
import re

import numpy as np
import pytest

from zelda_i import combat
from zelda_i.dungeon import gohma as shared_gohma
from zelda_i.dungeon import species as species_mod
from zelda_i.dungeon.behaviors import (
    GEL_SPLIT_TYPE,
    KEESE_BLACK_TYPE,
    KEESE_TYPE,
    VIRE_SPLIT_KEESE_TYPE,
    EnemyKind,
    KIND_POLICY,
    kind_for_type,
    live_among,
    liveness,
)
from zelda_i.dungeon.ids import AliveRule
from zelda_i.dungeon.species import Contact, species_of
from zelda_i.dungeon.tracking import HazardClass, ObjectTracker
from zelda_i.ram import (
    ADDR_HEALTH,
    ADDR_HEART_PARTIAL,
    ADDR_LEVEL,
    ADDR_LINK_FACING,
    ADDR_LINK_X,
    ADDR_LINK_Y,
    ADDR_MODE,
    ADDR_OBJ_HP,
    ADDR_OBJ_STATE,
    ADDR_OBJ_TYPE,
    ADDR_SCREEN,
    PLAY_MODE,
    ZeldaObject,
    read_snapshot,
)

ZORA_TYPE = 0x11
LEEVER_BLUE = 0x0F
LEEVER_RED = 0x10
SRC = pathlib.Path(__file__).resolve().parents[1]


def _snap(link: tuple[int, int], objects: tuple = (), *, screen: int = 0x7B):
    ram = np.zeros(0x1000, dtype=np.uint8)
    ram[ADDR_MODE] = PLAY_MODE
    ram[ADDR_LEVEL] = 0
    ram[ADDR_SCREEN] = screen
    ram[ADDR_LINK_X] = link[0]
    ram[ADDR_LINK_Y] = link[1]
    ram[ADDR_HEALTH] = 0x22
    ram[ADDR_HEART_PARTIAL] = 0xFF
    for slot, type_id, x, y, hp, state in objects:
        ram[ADDR_OBJ_TYPE + slot] = type_id
        ram[ADDR_LINK_X + slot] = x
        ram[ADDR_LINK_Y + slot] = y
        ram[ADDR_OBJ_HP + slot] = hp
        ram[ADDR_OBJ_STATE + slot] = state
        ram[ADDR_LINK_FACING + slot] = 0
    return read_snapshot(ram)


def _obj(type_id: int, *, state: int = 0, hp: int = 0, x: int = 120, y: int = 120):
    return ZeldaObject(
        slot=1, type_id=type_id, x=x, y=y, facing=0, hp=hp, state=state
    )


# =====================================================================
# BEHAVIOURAL — defect 1: $1D is a third Keese
# =====================================================================


def test_black_keese_is_alive_at_zero_hp() -> None:
    """A Keese keeps HP 0 while alive, so hp>0 reads it dead on arrival.

    $1D fell through `_TYPE_TO_KIND` to EnemyKind.UNKNOWN, whose alive_rule is
    TYPE_AND_HP — the exact failure the KEESE note exists to prevent.
    """
    black = _obj(KEESE_BLACK_TYPE, hp=0)
    assert liveness(black, AliveRule.TYPE_AND_HP)
    assert liveness(black, AliveRule.TYPE)
    assert live_among((black,), AliveRule.TYPE_AND_HP) == (black,)


def test_all_three_keese_types_are_one_kind() -> None:
    for type_id in (KEESE_TYPE, VIRE_SPLIT_KEESE_TYPE, KEESE_BLACK_TYPE):
        assert kind_for_type(type_id) is EnemyKind.KEESE
        assert species_of(type_id).hp == 0


def test_a_moving_black_keese_tracks_as_a_body_not_a_shot() -> None:
    """The liveness hole leaked into the hazard class: an HP-0 UNKNOWN slot
    moving at flyer speed was classed PROJECTILE — unkillable, dodge-only."""
    tracker = ObjectTracker()
    tracked = ()
    for step in range(6):
        tracked = tracker.observe(
            _snap((120, 120), ((1, KEESE_BLACK_TYPE, 40 + step * 4, 100, 0, 1),))
        )
    assert tracked[0].kind is EnemyKind.KEESE
    assert tracked[0].hazard is HazardClass.BODY
    assert tracked[0].speed >= 1.6  # fast enough that the old rule said "shot"


def test_zol_split_gel_residual_is_a_gel() -> None:
    """$14 is the same HP-0 hole as $1D: the Zol-split residual of $15."""
    assert kind_for_type(GEL_SPLIT_TYPE) is EnemyKind.GEL
    assert species_of(GEL_SPLIT_TYPE).hp == 0


# =====================================================================
# BEHAVIOURAL — defect 2: the Zora shares the burrower dormant state
# =====================================================================


def test_all_three_burrowers_run_no_collision_check_in_state_zero() -> None:
    """`UpdateZora` runs `Burrower_AnimateDrawAndCheckCollisions` too, which
    returns before any collision check while ObjState is 0."""
    for type_id in (LEEVER_BLUE, LEEVER_RED, ZORA_TYPE):
        assert combat.no_contact_frame(_obj(type_id, state=0, hp=16))
        assert not combat.no_contact_frame(_obj(type_id, state=2, hp=16))
    assert combat.DORMANT_BURROWER_TYPES == frozenset(
        {LEEVER_BLUE, LEEVER_RED, ZORA_TYPE}
    )


def test_a_state_zero_zora_is_still_a_threat_and_still_never_engaged() -> None:
    """The ROM fact is real; filtering on it here is not.

    `ZORA_CYCLE` measures ObjState 0 as **2 frames of surfacing**, not the
    96-frame submerged phase (that is ObjState 5), and `zora_shot_eta` counts
    *from* state 0. Dropping a state-0 Zora out of `overworld_threat_objects`
    blinds the clock rung that has to see it.
    """
    zora = _obj(ZORA_TYPE, state=0, hp=16, x=200, y=141)
    assert combat.no_contact_frame(zora)
    assert not combat.dormant_body(zora)
    snap = _snap((120, 141), ((1, ZORA_TYPE, 200, 141, 16, 0),))
    assert [o.type_id for o in combat.overworld_threat_objects(snap)] == [ZORA_TYPE]


def test_dormant_body_is_still_only_the_two_leevers() -> None:
    assert combat.dormant_body(_obj(LEEVER_BLUE, state=0, hp=16))
    assert combat.dormant_body(_obj(LEEVER_RED, state=0, hp=16))
    assert not combat.dormant_body(_obj(LEEVER_RED, state=2, hp=16))
    assert not combat.dormant_body(_obj(ZORA_TYPE, state=0, hp=16))


# =====================================================================
# BEHAVIOURAL — defect 3: dormant is not motionless
# =====================================================================


def test_a_buried_blue_leever_creeps_and_is_still_dormant() -> None:
    """"It never moved" is a *red*-leever fact. `BlueLeeverStateQSpeeds[0]` is
    $08 = 0.125 px/frame, so ObjectTracker sees a buried blue leever move while
    `dormant_body` still says do not cut it and do not dodge it."""
    assert species_of(LEEVER_BLUE).px_per_frame(0) == pytest.approx(0.125)
    assert species_of(LEEVER_RED).px_per_frame(0) == 0.0

    tracker = ObjectTracker()
    tracked = ()
    for step in range(9):  # 0.125 px/f -> 1 px in 8 frames
        tracked = tracker.observe(
            _snap((120, 120), ((1, LEEVER_BLUE, 60 + step // 8, 100, 16, 0),))
        )
    assert tracked[0].moving, "a dormant blue leever is not motionless"
    assert combat.dormant_body(
        _obj(LEEVER_BLUE, state=0, hp=16, x=tracked[0].x, y=tracked[0].y)
    )


# =====================================================================
# BEHAVIOURAL — the L6/L8 Gohma collapse is arithmetic-identical
# =====================================================================


def _inline_aim(hist: list[int], gx: int, by: int, ly: int, bounds) -> int:
    """The code `level6/gohma.py` and `level8/magic_key.py` each carried."""
    hist.append(gx)
    del hist[:-8]
    gvx = (hist[-1] - hist[0]) / (len(hist) - 1) if len(hist) >= 4 else 0.0
    flight = max(1.0, (ly - by) / shared_gohma.ARROW_SPEED)
    lead = int(round(gvx * flight))
    lead = max(-shared_gohma.LEAD_CLAMP, min(shared_gohma.LEAD_CLAMP, lead))
    lo, hi = bounds
    return max(lo, min(hi, gx + lead))


def test_shared_gohma_aim_reproduces_the_inline_arithmetic() -> None:
    """Both chains are frame-perfect: the extract must not move one pixel."""
    bounds = (40, 216)
    old_hist: list[int] = []
    new_hist: list[int] = []
    gx = 128
    for frame in range(200):
        gx = max(96, min(200, gx + (3 if (frame // 17) % 2 else -2)))
        by = 112 + (frame % 33)
        ly = 162 + (frame % 5)
        want = _inline_aim(old_hist, gx, by, ly, bounds)
        got = shared_gohma.aim_column(
            gx, shared_gohma.strafe_vx(new_hist, gx), by, ly, bounds
        )
        assert got == want, f"frame {frame}: {got} != {want}"
        assert new_hist == old_hist


def test_arrow_aim_x_is_strafe_then_column() -> None:
    hist_a: list[int] = []
    hist_b: list[int] = []
    body = _obj(0x33, x=150, y=120)
    for _ in range(6):
        want = shared_gohma.aim_column(
            150, shared_gohma.strafe_vx(hist_a, 150), 120, 170, (40, 216)
        )
        assert shared_gohma.arrow_aim_x(hist_b, body, 170, (40, 216)) == want


def test_strafe_read_is_zero_below_the_minimum_sample_count() -> None:
    hist: list[int] = []
    reads = [shared_gohma.strafe_vx(hist, 100 + 4 * n) for n in range(6)]
    assert reads[: shared_gohma.GX_MIN_SAMPLES - 1] == [0.0, 0.0, 0.0]
    assert reads[-1] == pytest.approx(4.0)
    assert len(hist) == 6


def test_the_eye_clock_only_reads_fresh_on_the_rising_edge() -> None:
    """Firing on a fixed cadence aliases onto the closed blink (20+ arrows,
    0 connects). `advance_eye` resets to -1 whenever the eye shuts."""
    since = -1
    fresh = []
    for byte in [shared_gohma.EYE_SHUT] * 3 + [0x60] * 20 + [shared_gohma.EYE_SHUT]:
        since = shared_gohma.advance_eye(since, byte)
        fresh.append(shared_gohma.eye_fresh_open(since))
    assert fresh[:3] == [False, False, False]
    assert fresh[3 : 3 + shared_gohma.EYE_EDGE_WINDOW + 1] == [True] * (
        shared_gohma.EYE_EDGE_WINDOW + 1
    )
    assert fresh[3 + shared_gohma.EYE_EDGE_WINDOW + 1] is False
    assert fresh[-1] is False


def test_an_unreadable_eye_never_reads_open() -> None:
    assert shared_gohma.advance_eye(5, None) == -1
    assert not shared_gohma.eye_fresh_open(-1)


def test_l6_and_l8_read_the_same_gohma_body_table() -> None:
    from zelda_i.level6 import gohma as l6
    from zelda_i.level8 import magic_key as l8

    assert l6.GOHMA_TYPES is shared_gohma.GOHMA_TYPES
    assert l8.GOHMA_BODY_TYPES_1E is shared_gohma.GOHMA_TYPES
    assert l6.gohma_live is shared_gohma.gohma_live


# =====================================================================
# STRUCTURAL — the ratchet
# =====================================================================

# Every production module that keeps a frame-to-frame history of an *enemy's*
# position. `dungeon/tracking.py` is the canonical implementation and
# `dungeon/gohma.py` is the one deliberate exception (see its docstring: the
# L6/L8 arrow history is only advanced on aim frames, not on every observed
# frame, and both chains are frame-perfect against that cadence).
#
# NOT on this list, and not a violation: `prev_xy` / `last_xy` over
# **Link's own** position. Seventeen modules keep one as a did-not-move stall
# counter; that is walk physics, not an enemy model, and ObjectTracker's
# six-sample mean is the wrong instrument for it.
ENEMY_MOTION_OWNERS = {
    "dungeon/tracking.py",
    "dungeon/gohma.py",
}

# What is banned is the *derivation*, not the buffer: a controller may still
# own the list it hands to `dungeon.gohma`, but the difference quotient over it
# belongs to one module. Two shapes catch every case this card measured:
#   * `hist[-1] ... hist[0]`  -- the mean-velocity divisor, spelled out locally;
#   * a per-slot dict of an enemy's previous position.
_ENEMY_HISTORY = re.compile(
    r"(?P<quotient>hist\[-1\][^\n]*hist\[0\])|\b(?P<dict>_prev_shot_xy|prev_enemy_xy)\b"
)

# Known, argued exception: the L6 wizzrobe waist-beam dodge reads a **one
# frame** difference off a per-slot dict (`_prev_shot_xy`). ObjectTracker's
# estimator is a six-sample mean with `(slot, type_id)` identity and a respawn
# guard — strictly better, and a different number on the frame it matters. The
# L6 chain is frame-perfect, so this stays until it can be measured on the ROM.
ENEMY_MOTION_EXCEPTIONS = {"level6/wizzrobe.py": "_prev_shot_xy"}


def _production_modules() -> list[pathlib.Path]:
    skip = {"tests", "scratch", "scripts", "__pycache__", "recordings"}
    return [
        path
        for path in sorted(SRC.rglob("*.py"))
        if not (set(path.relative_to(SRC).parts) & skip)
    ]


def test_no_module_grows_its_own_enemy_motion_history() -> None:
    """21 -> 2. A new `gx_hist` in a level module is the regression."""
    offenders: dict[str, set[str]] = {}
    for path in _production_modules():
        rel = path.relative_to(SRC).as_posix()
        if rel in ENEMY_MOTION_OWNERS:
            continue
        hits = {
            match.lastgroup if match.group("dict") is None else match.group("dict")
            for match in _ENEMY_HISTORY.finditer(path.read_text())
        }
        hits.discard(ENEMY_MOTION_EXCEPTIONS.get(rel))
        if hits:
            offenders[rel] = hits
    assert not offenders, (
        "enemy motion re-derived outside dungeon.tracking / dungeon.gohma: "
        f"{offenders}. Use ObjectTracker, or add an argued exception."
    )


def test_the_gohma_eye_clock_lives_in_exactly_one_module() -> None:
    """`$03C7` and its `0xC0` blink were transcribed into two level modules.

    Prose may still name the address; only one module may *define* it, and only
    one may compare a byte against the blink.
    """
    defines = [
        path.relative_to(SRC).as_posix()
        for path in _production_modules()
        if "EYE_ADDR = 0x03C7" in path.read_text()
    ]
    assert defines == ["dungeon/gohma.py"]
    blinks = [
        path.relative_to(SRC).as_posix()
        for path in _production_modules()
        if "EYE_SHUT = " in path.read_text() or "== EYE_SHUT" in path.read_text()
    ]
    assert blinks == ["dungeon/gohma.py"]


def test_every_harmful_rom_type_with_zero_hp_has_type_liveness() -> None:
    """The guard the $1D hole slipped through.

    The standing `test_rom_hp_zero_types_never_get_an_hp_liveness_rule` walks
    `_TYPE_TO_KIND`, so a type that is *missing* from it is invisible to it.
    This one walks the ROM. Harmful only: an HP-0 type that deals no damage
    ($4B) is not a combatant, and giving it TYPE liveness would make it read
    alive forever and hang a room-clear predicate.
    """
    offenders = []
    for type_id in range(1, 0x60):
        row = species_of(type_id)
        if row.hp != 0 or row.contact is not Contact.HARM:
            continue
        policy = KIND_POLICY[kind_for_type(type_id)]
        if policy.alive_rule is not AliveRule.TYPE:
            offenders.append(f"0x{type_id:02X} ({row.name})")
    assert not offenders, (
        "ROM HP-0 combatant judged by hp>0 — dead on arrival: "
        + ", ".join(offenders)
    )


def test_harmless_zero_hp_types_are_deliberately_left_out() -> None:
    """Pins the exclusion above so it cannot be widened by accident."""
    row = species_of(0x4B)
    assert row.hp == 0 and row.contact is Contact.HARMLESS
    assert kind_for_type(0x4B) is EnemyKind.UNKNOWN


def test_species_is_the_only_place_the_dormant_state_is_named() -> None:
    """`combat` reads the burrower state off the ROM table, never a local 0."""
    assert combat.LEEVER_DORMANT_STATE is species_mod.BURROWER_DORMANT_STATE
    assert combat.DORMANT_BURROWER_TYPES is species_mod.BURROWER_TYPES
